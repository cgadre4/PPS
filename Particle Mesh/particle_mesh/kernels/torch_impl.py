"""
Particle hot paths in pure PyTorch.

Runs on any torch device (Intel XPU, CUDA, CPU) with no host copies, so it is the backend used whenever Taichi
cannot share memory with the torch device (e.g. torch on XPU + Taichi on Vulkan). Every function is a plain
tensor-in / tensor-out op so it can be wrapped with torch.compile or replaced by a Triton kernel later.

Cloud-in-cell (CIC) assignment and interpolation use PyTorch's fused bilinear sampler (AUDIT §8 F02):
  - gather  = torch.nn.functional.grid_sample (bilinear, align_corners=False), which on a cell-centred grid is
              exactly CIC interpolation;
  - deposit = aten::grid_sampler_2d_backward with the particle masses as the incoming gradient, which scatters
              each mass to the 4 surrounding cells with the same bilinear weights, i.e. exactly CIC assignment.
Both work on a one-cell-padded grid (H+2, W+2): for periodic boundaries the halo wraps (deposit folds it back,
gather sees wrapped copies); for reflective walls it replicates the edge, which equals the clamped CIC.
Measured on the Arc B390: stencil + deposit + gather 6.5 -> 1.9 ms at 1M particles, 24.6 -> 6.3 ms at 4M.

Grid layout: physics grids are (H, W) row-major, indexed [j, i]; flat index = j * W + i.
"""
import torch

from particle_mesh.runtime.device import const_tensor as _const

_BUFFERS = {}


def _buffer(name, shape, device, dtype=torch.float32):
    """Persistent scratch buffer (zero-initialised once), reused every step."""
    key = (name, tuple(shape), str(device), dtype)
    t = _BUFFERS.get(key)
    if t is None:
        t = _BUFFERS[key] = torch.zeros(shape, device=device, dtype=dtype)
    return t


class CICStencil:
    """grid_sample coordinates (1, 1, N, 2) of the particles on the padded physics grid. Built once per step and
    shared by the deposit and the force gather of that step."""
    __slots__ = ("grid", "grid_w", "grid_h", "periodic")

    def __init__(self, grid, grid_w, grid_h, periodic):
        self.grid = grid
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.periodic = periodic


def cic_stencil(pos, num_active, cell_size, grid_w, grid_h, periodic):
    """Normalised sampler coordinates on the (H+2, W+2) padded grid: padded cell index xp = x / cell_size + 0.5,
    and grid_sample's align_corners=False convention u = (2 * xp + 1) / (W + 2) - 1. One fused multiply-add."""
    wp, hp = grid_w + 2, grid_h + 2
    a = _const((2.0 / (cell_size * wp), 2.0 / (cell_size * hp)), pos.device, pos.dtype)
    b = _const((2.0 / wp - 1.0, 2.0 / hp - 1.0), pos.device, pos.dtype)
    grid = torch.addcmul(b, pos[:num_active], a).view(1, 1, num_active, 2)
    return CICStencil(grid, grid_w, grid_h, periodic)


def _fold_halo(padded, periodic):
    """Fold the one-cell halo of a deposited (H+2, W+2) grid back onto the interior: wrapped (periodic) or onto
    the edge cells (reflective walls, i.e. clamped CIC). Rows first (full width), then columns, so corners land
    correctly. Returns the (H, W) interior view."""
    h, w = padded.shape[0] - 2, padded.shape[1] - 2
    if periodic:
        padded[1, :] += padded[h + 1, :]
        padded[h, :] += padded[0, :]
        padded[:, 1] += padded[:, w + 1]
        padded[:, w] += padded[:, 0]
    else:
        padded[1, :] += padded[0, :]
        padded[h, :] += padded[h + 1, :]
        padded[:, 1] += padded[:, 0]
        padded[:, w] += padded[:, w + 1]
    return padded[1:h + 1, 1:w + 1]


def deposit(stencil, mass, num_active, density):
    """CIC-scatter particle masses onto `density` (H, W), overwriting it."""
    h, w = stencil.grid_h, stencil.grid_w
    weights = mass[:num_active].to(torch.float32).view(1, 1, 1, num_active)
    zeros = _buffer("deposit_input", (1, 1, h + 2, w + 2), density.device)   # shape carrier; never written
    padded, _ = torch.ops.aten.grid_sampler_2d_backward(
        weights, zeros, stencil.grid, 0, 1, False, [True, False])          # bilinear, border, align_corners=False
    density.copy_(_fold_halo(padded[0, 0], stencil.periodic))


def force_buffers(grid_h, grid_w, device):
    """(fx, fy) tensors the field solver writes forces into: the interiors of one persistent (2, H+2, W+2)
    padded buffer, so the gather only has to refresh the thin halo (no 32 MB copy per step)."""
    buf = _buffer("forces", (1, 2, grid_h + 2, grid_w + 2), device)
    return buf[0, 0, 1:-1, 1:-1], buf[0, 1, 1:-1, 1:-1]


def _fill_halo(buf, periodic):
    """Refresh the halo of a (1, C, H+2, W+2) buffer from its interior: wrapped or edge-replicated."""
    h, w = buf.shape[2] - 2, buf.shape[3] - 2
    if periodic:
        buf[:, :, 0, 1:-1] = buf[:, :, h, 1:-1]
        buf[:, :, h + 1, 1:-1] = buf[:, :, 1, 1:-1]
        buf[:, :, :, 0] = buf[:, :, :, w]
        buf[:, :, :, w + 1] = buf[:, :, :, 1]
    else:
        buf[:, :, 0, 1:-1] = buf[:, :, 1, 1:-1]
        buf[:, :, h + 1, 1:-1] = buf[:, :, h, 1:-1]
        buf[:, :, :, 0] = buf[:, :, :, 1]
        buf[:, :, :, w + 1] = buf[:, :, :, w]


def _padded_pair(name, gx, gy, periodic):
    """(1, 2, H+2, W+2) padded buffer holding gx/gy with a refreshed halo. Zero-copy when gx/gy already are the
    interiors returned by force_buffers()."""
    h, w = gx.shape
    buf = _buffer(name, (1, 2, h + 2, w + 2), gx.device)
    if gx.data_ptr() != buf[0, 0, 1:-1, 1:-1].data_ptr():
        buf[0, 0, 1:-1, 1:-1] = gx
        buf[0, 1, 1:-1, 1:-1] = gy
    _fill_halo(buf, periodic)
    return buf


def _gather(buf, stencil):
    """Bilinear (CIC) interpolation of a (1, 2, H+2, W+2) buffer at the particles -> (2, N)."""
    return torch.nn.functional.grid_sample(buf, stencil.grid, mode='bilinear', padding_mode='border',
                                           align_corners=False)[0, :, 0, :]


def gather_integrate(stencil, pos, vel, mass, num_active, force_x, force_y, dt, damping, use_charge,
                     periodic, universe_w, universe_h, accel_x=None, accel_y=None, uniform_accel=(0.0, 0.0)):
    """Interpolate forces at the particles, kick, drift (semi-implicit Euler) and wrap if periodic. In place.
    use_charge scales force_x/force_y by each particle's +/-1 charge; accel_x/accel_y (optional grids, e.g. gas
    pressure in Coulomb mode) and uniform_accel (uniform gravity) are added without that factor."""
    p = pos[:num_active]
    v = vel[:num_active]
    acc = _gather(_padded_pair("forces", force_x, force_y, periodic), stencil)
    if use_charge:
        acc = acc * mass[:num_active].to(acc.dtype)
    if accel_x is not None:
        acc = acc + _gather(_padded_pair("accel", accel_x, accel_y, periodic), stencil)
    v.add_(acc.t(), alpha=dt)
    if uniform_accel[0] != 0.0 or uniform_accel[1] != 0.0:
        v.add_(_const((uniform_accel[0] * dt, uniform_accel[1] * dt), v.device, v.dtype))
    v.mul_(damping)
    p.add_(v, alpha=dt)
    if periodic:
        wrap_positions(pos, num_active, universe_w, universe_h)


def wrap_positions(pos, num_active, universe_w, universe_h):
    """Wrap positions into [0, universe) in place (no temporary copy)."""
    pos[:num_active].remainder_(_const((float(universe_w), float(universe_h)), pos.device, pos.dtype))


def splat_view(pos, num_active, cam_x, cam_y, zoom, view_w, view_h, render_cell_size):
    """
    Count particles per viewport render cell. Returns integer counts, shape (view_w, view_h): x-major, i.e. GGUI's
    [x, y] image layout, so the renderer needs no transpose.

    Cell indices are shifted by one and clamped into a one-cell border ring around the view; off-screen particles
    land on the ring, which is dropped. That needs no bool masks or torch.where, and spreads off-screen hits over
    the ring's ~2*(W+H) cells instead of one hot bin.
    """
    p = pos[:num_active]
    scale = zoom / render_cell_size
    gx = (p[:, 0] - cam_x).mul_(scale).floor_().add_(1.0).clamp_(0.0, view_w + 1.0)
    gy = (p[:, 1] - cam_y).mul_(scale).floor_().add_(1.0).clamp_(0.0, view_h + 1.0)
    flat = gx.mul_(view_h + 2).add_(gy).to(torch.int64)        # exact in float32 below 2^24 cells
    counts = torch.bincount(flat, minlength=(view_w + 2) * (view_h + 2))
    return counts.view(view_w + 2, view_h + 2)[1:-1, 1:-1]
