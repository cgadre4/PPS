"""
Particle-kernel backend selection.

Both backends expose the same methods, used by the PM solver, the engine and the renderers:
    prepare(pos, n, cell_size, grid_w, grid_h, periodic)        -> stencil shared by deposit + gather (or None)
    deposit(stencil, pos, mass, n, density, cell_size, periodic)
    gather_integrate(stencil, pos, vel, mass, n, fx, fy, cell_size, dt, damping, use_charge,
                     periodic, universe_w, universe_h, accel_x=None, accel_y=None)
    wrap(pos, n, universe_w, universe_h)
    splat_view(pos, n, cam_x, cam_y, zoom, view_w, view_h, render_cell_size) -> float (view_h, view_w) tensor

"taichi": Taichi kernels on shared memory (torch CUDA + Taichi CUDA, or torch CPU + Taichi CPU). Fastest there.
"torch":  pure PyTorch on the torch device. Used for Intel XPU, where Taichi (Vulkan) cannot see torch memory.
"""
import taichi as ti
import torch

import kernels_torch as KT
import kernels_taichi as KTI


def taichi_shares_memory(device):
    """True when Taichi kernels can read/write tensors on `device` without host round-trips."""
    arch = ti.lang.impl.current_cfg().arch
    if device.type == 'cuda':
        return arch == ti.cuda
    if device.type == 'cpu':
        return arch in (ti.x64, ti.arm64)
    return False


def to_taichi(tensor):
    """Tensor to hand to field.from_torch(): as-is when Taichi shares its memory, else one explicit host copy
    (passing a device tensor would make Taichi copy it to the host AND back again after the kernel)."""
    return tensor if taichi_shares_memory(tensor.device) else tensor.cpu()


def _with_eager_fallback(fn, compiled):
    """Call the torch.compile'd version; if compilation fails (e.g. no C compiler for Triton), fall back to eager."""
    state = {"fn": compiled}

    def call(*args, **kwargs):
        try:
            return state["fn"](*args, **kwargs)
        except Exception as e:
            if state["fn"] is fn:
                raise
            print(f"[kernels] torch.compile unavailable ({type(e).__name__}: {str(e).splitlines()[0][:120]}); "
                  "using eager PyTorch")
            state["fn"] = fn
            return fn(*args, **kwargs)
    return call


class TorchBackend:
    name = "torch"

    def __init__(self, use_compile=False):
        self._stencil = KT.cic_stencil
        self._deposit = KT.deposit
        self._gather = KT.gather_integrate
        self._splat = KT.splat_view
        if use_compile:
            self._stencil = _with_eager_fallback(KT.cic_stencil, torch.compile(KT.cic_stencil, dynamic=True))
            self._gather = _with_eager_fallback(KT.gather_integrate, torch.compile(KT.gather_integrate, dynamic=True))
            self._splat = _with_eager_fallback(KT.splat_view, torch.compile(KT.splat_view, dynamic=True))
            self.name = "torch+compile"

    def prepare(self, pos, n, cell_size, grid_w, grid_h, periodic):
        return self._stencil(pos, n, cell_size, grid_w, grid_h, periodic)

    def deposit(self, stencil, pos, mass, n, density, cell_size, periodic):
        self._deposit(stencil, mass, n, density)

    def gather_integrate(self, stencil, pos, vel, mass, n, fx, fy, cell_size, dt, damping, use_charge,
                         periodic, universe_w, universe_h, accel_x=None, accel_y=None):
        self._gather(stencil, pos, vel, mass, n, fx, fy, dt, damping, use_charge, periodic, universe_w, universe_h,
                     accel_x, accel_y)

    def wrap(self, pos, n, universe_w, universe_h):
        KT.wrap_positions(pos, n, universe_w, universe_h)

    def splat_view(self, pos, n, cam_x, cam_y, zoom, view_w, view_h, render_cell_size):
        return self._splat(pos, n, cam_x, cam_y, zoom, view_w, view_h, render_cell_size)


class TaichiBackend:
    name = "taichi"

    def prepare(self, pos, n, cell_size, grid_w, grid_h, periodic):
        return (grid_w, grid_h)

    def deposit(self, stencil, pos, mass, n, density, cell_size, periodic):
        grid_w, grid_h = stencil
        KTI.deposit_kernel(pos, mass, density, n, grid_w, grid_h, float(cell_size), int(periodic))

    def gather_integrate(self, stencil, pos, vel, mass, n, fx, fy, cell_size, dt, damping, use_charge,
                         periodic, universe_w, universe_h, accel_x=None, accel_y=None):
        grid_w, grid_h = stencil
        has_accel = accel_x is not None
        if not has_accel:
            # Taichi needs real arrays for every ndarray argument; a 1x1 dummy is never read (has_accel = 0)
            accel_x = accel_y = fx[:1, :1]
        KTI.gather_integrate_kernel(pos, vel, mass, fx, fy, accel_x, accel_y, n, grid_w, grid_h, float(cell_size),
                                    float(dt), float(damping), int(use_charge), int(periodic),
                                    float(universe_w), float(universe_h), int(has_accel))

    def wrap(self, pos, n, universe_w, universe_h):
        KT.wrap_positions(pos, n, universe_w, universe_h)

    def splat_view(self, pos, n, cam_x, cam_y, zoom, view_w, view_h, render_cell_size):
        out = torch.zeros((view_h, view_w), device=pos.device, dtype=torch.float32)
        if n > 0:
            KTI.splat_view_kernel(pos, out, n, float(cam_x), float(cam_y), float(zoom / render_cell_size),
                                  view_w, view_h)
        return out


def make_backend(device, choice="auto", use_compile=False):
    """choice: "auto" | "torch" | "taichi" (from simulation.KERNEL_BACKEND)."""
    choice = (choice or "auto").lower()
    if choice == "auto":
        choice = "taichi" if taichi_shares_memory(device) else "torch"
    if choice == "taichi":
        if not taichi_shares_memory(device):
            print(f"[kernels] Warning: Taichi cannot share memory with torch device '{device}'; every kernel call "
                  "will copy through the host. Set KERNEL_BACKEND to \"auto\" or \"torch\".")
        return TaichiBackend()
    return TorchBackend(use_compile=use_compile)
