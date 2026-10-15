"""
Particle hot paths as Taichi kernels operating directly on PyTorch tensors.

Only used when Taichi can share memory with the torch device (torch CUDA + Taichi CUDA, or torch CPU + Taichi CPU).
On any other pairing (e.g. torch XPU + Taichi Vulkan) Taichi copies every tensor argument to the host and back
on every call, so kernels.py selects kernels_torch instead.

Grid layout: physics grids are (H, W) row-major, indexed [j, i].
"""
import taichi as ti


@ti.func
def _cic_corners(px, py, cell_size, grid_w, grid_h, periodic):
    x = px / cell_size - 0.5
    y = py / cell_size - 0.5
    if periodic == 0:
        # Clamped CIC (reflective walls): identical to the torch backend's edge-replicated halo
        x = ti.max(0.0, ti.min(x, grid_w - 1.0))
        y = ti.max(0.0, ti.min(y, grid_h - 1.0))
    fx = ti.floor(x)
    fy = ti.floor(y)
    wx = x - fx
    wy = y - fy
    i0 = ti.cast(fx, ti.i32)
    j0 = ti.cast(fy, ti.i32)
    i1 = i0 + 1
    j1 = j0 + 1
    if periodic == 0:
        i1 = ti.min(i1, grid_w - 1)
        j1 = ti.min(j1, grid_h - 1)
    else:
        # Taichi's % follows Python semantics, so the result is non-negative
        i0 = i0 % grid_w
        j0 = j0 % grid_h
        i1 = i1 % grid_w
        j1 = j1 % grid_h
    return i0, j0, i1, j1, wx, wy


@ti.kernel
def deposit_kernel(
    pos: ti.types.ndarray(), mass: ti.types.ndarray(), density: ti.types.ndarray(),
    num_active: ti.i32, grid_w: ti.i32, grid_h: ti.i32, cell_size: ti.f32, periodic: ti.i32
):
    for p in range(num_active):
        i0, j0, i1, j1, wx, wy = _cic_corners(pos[p, 0], pos[p, 1], cell_size, grid_w, grid_h, periodic)
        m = ti.cast(mass[p], ti.f32)
        ti.atomic_add(density[j0, i0], m * (1.0 - wx) * (1.0 - wy))
        ti.atomic_add(density[j0, i1], m * wx * (1.0 - wy))
        ti.atomic_add(density[j1, i0], m * (1.0 - wx) * wy)
        ti.atomic_add(density[j1, i1], m * wx * wy)


@ti.kernel
def gather_integrate_kernel(
    pos: ti.types.ndarray(), vel: ti.types.ndarray(), mass: ti.types.ndarray(),
    force_x: ti.types.ndarray(), force_y: ti.types.ndarray(),
    accel_x: ti.types.ndarray(), accel_y: ti.types.ndarray(),
    num_active: ti.i32, grid_w: ti.i32, grid_h: ti.i32, cell_size: ti.f32, dt: ti.f32, damping: ti.f32,
    use_charge: ti.i32, periodic: ti.i32, universe_w: ti.f32, universe_h: ti.f32, has_accel: ti.i32,
    uniform_x: ti.f32, uniform_y: ti.f32
):
    for p in range(num_active):
        i0, j0, i1, j1, wx, wy = _cic_corners(pos[p, 0], pos[p, 1], cell_size, grid_w, grid_h, periodic)
        w00 = (1.0 - wx) * (1.0 - wy)
        w10 = wx * (1.0 - wy)
        w01 = (1.0 - wx) * wy
        w11 = wx * wy
        fx = force_x[j0, i0] * w00 + force_x[j0, i1] * w10 + force_x[j1, i0] * w01 + force_x[j1, i1] * w11
        fy = force_y[j0, i0] * w00 + force_y[j0, i1] * w10 + force_y[j1, i0] * w01 + force_y[j1, i1] * w11

        # Coulomb: the field is a charge potential, so the force scales with the particle's own charge (+/-1)
        q = 1.0
        if use_charge != 0:
            q = ti.cast(mass[p], ti.f32)
        fx *= q
        fy *= q
        # Charge-independent extra acceleration (gas pressure in Coulomb mode)
        if has_accel != 0:
            fx += accel_x[j0, i0] * w00 + accel_x[j0, i1] * w10 + accel_x[j1, i0] * w01 + accel_x[j1, i1] * w11
            fy += accel_y[j0, i0] * w00 + accel_y[j0, i1] * w10 + accel_y[j1, i0] * w01 + accel_y[j1, i1] * w11
        # Uniform gravity, folded into the kick
        fx += uniform_x
        fy += uniform_y

        vel[p, 0] = (vel[p, 0] + fx * dt) * damping
        vel[p, 1] = (vel[p, 1] + fy * dt) * damping
        x = pos[p, 0] + vel[p, 0] * dt
        y = pos[p, 1] + vel[p, 1] * dt
        if periodic != 0:
            x -= universe_w * ti.floor(x / universe_w)
            y -= universe_h * ti.floor(y / universe_h)
        pos[p, 0] = x
        pos[p, 1] = y


@ti.kernel
def splat_view_kernel(
    pos: ti.types.ndarray(), out: ti.types.ndarray(), num_active: ti.i32,
    cam_x: ti.f32, cam_y: ti.f32, scale: ti.f32, view_w: ti.i32, view_h: ti.i32
):
    for p in range(num_active):
        gx = ti.floor((pos[p, 0] - cam_x) * scale)
        gy = ti.floor((pos[p, 1] - cam_y) * scale)
        if 0 <= gx < view_w and 0 <= gy < view_h:
            # out is (view_w, view_h) int32, x-major (GGUI [x, y] layout), like kernels_torch.splat_view
            ti.atomic_add(out[ti.cast(gx, ti.i32), ti.cast(gy, ti.i32)], 1)
