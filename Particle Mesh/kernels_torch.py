"""
Particle hot paths in pure PyTorch.

Runs on any torch device (Intel XPU, CUDA, CPU) with no host copies, so it is the backend used whenever Taichi
cannot share memory with the torch device (e.g. torch on XPU + Taichi on Vulkan). Every function is a plain
tensor-in / tensor-out op so it can be wrapped with torch.compile or replaced by a Triton kernel later.

Grid layout: physics grids are (H, W) row-major, indexed [j, i]; flat index = j * W + i.
"""
import torch


class CICStencil:
    """Cloud-in-cell corner indices (4, N) and weights (4, N) for one set of positions.
    The deposit and the force gather of a PM step use the same positions, so this is built once per step."""
    __slots__ = ("idx", "w")

    def __init__(self, idx, w):
        self.idx = idx
        self.w = w


def cic_stencil(pos, num_active, cell_size, grid_w, grid_h, periodic):
    p = pos[:num_active]
    xy = p * (1.0 / cell_size) - 0.5                     # (N, 2) grid coordinates of the cell-centred lattice
    if not periodic:
        xy[:, 0].clamp_(0.0, grid_w - 1.001)
        xy[:, 1].clamp_(0.0, grid_h - 1.001)
    f = torch.floor(xy)
    w1 = xy - f                                          # weight of the +1 neighbour, per axis
    w0 = 1.0 - w1
    c0 = f.to(torch.int64)
    if periodic:
        dims = torch.tensor([grid_w, grid_h], device=pos.device, dtype=torch.int64)
        c0.remainder_(dims)
        c1 = c0 + 1
        c1 = torch.where(c1 == dims, 0, c1)
    else:
        c1 = c0 + 1
    r0 = c0[:, 1] * grid_w
    r1 = c1[:, 1] * grid_w
    idx = torch.stack((r0 + c0[:, 0], r0 + c1[:, 0], r1 + c0[:, 0], r1 + c1[:, 0]))
    w = torch.stack((w0[:, 0] * w0[:, 1], w1[:, 0] * w0[:, 1], w0[:, 0] * w1[:, 1], w1[:, 0] * w1[:, 1]))
    return CICStencil(idx, w)


def deposit(stencil, mass, num_active, density):
    """Scatter particle masses onto `density` (H, W), which the caller has zeroed."""
    m = mass[:num_active].to(stencil.w.dtype)
    density.view(-1).index_add_(0, stencil.idx.view(-1), (stencil.w * m).view(-1))


def gather_integrate(stencil, pos, vel, mass, num_active, force_x, force_y, dt, damping, use_charge,
                     periodic, universe_w, universe_h):
    """Interpolate forces at the particles, kick, drift (semi-implicit Euler) and wrap if periodic. In place."""
    p = pos[:num_active]
    v = vel[:num_active]
    ax = (force_x.view(-1)[stencil.idx] * stencil.w).sum(0)
    ay = (force_y.view(-1)[stencil.idx] * stencil.w).sum(0)
    if use_charge:
        q = mass[:num_active].to(ax.dtype)
        ax.mul_(q)
        ay.mul_(q)
    v[:, 0].add_(ax, alpha=dt)
    v[:, 1].add_(ay, alpha=dt)
    v.mul_(damping)
    p.add_(v, alpha=dt)
    if periodic:
        wrap_positions(pos, num_active, universe_w, universe_h)


def wrap_positions(pos, num_active, universe_w, universe_h):
    p = pos[:num_active]
    size = torch.tensor([universe_w, universe_h], device=pos.device, dtype=pos.dtype)
    p.copy_(torch.remainder(p, size))


def splat_view(pos, num_active, cam_x, cam_y, zoom, view_w, view_h, render_cell_size):
    """Count particles per viewport render cell. Returns a float (view_h, view_w) tensor."""
    p = pos[:num_active]
    scale = zoom / render_cell_size
    gx = torch.floor((p[:, 0] - cam_x) * scale)
    gy = torch.floor((p[:, 1] - cam_y) * scale)
    inside = (gx >= 0) & (gx < view_w) & (gy >= 0) & (gy < view_h)
    # Off-screen particles go to one extra trailing bin that is dropped (no compaction, no host sync)
    flat = torch.where(inside, gy * view_w + gx, float(view_w * view_h)).to(torch.int64)
    counts = torch.bincount(flat, minlength=view_w * view_h + 1)
    return counts[:view_w * view_h].view(view_h, view_w).to(torch.float32)
