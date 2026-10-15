import torch

from particle_mesh.kernels import nbody_accel

# Elements per (tile_rows x N) pairwise tensor. Memory is O(N * tile) instead of O(N^2) (AUDIT 3.5).
TILE_ELEMENTS = 1 << 22
_KINDS = {'gravity': 0, 'coulomb': 1, 'harmonic': 2, 'lennard_jones': 3}


class DirectNBodySolver:
    def __init__(self, device, tile_elements=TILE_ELEMENTS, use_taichi=True):
        self.device = device
        self.tile_elements = tile_elements
        # Fused Taichi kernel (AUDIT §8 F36) unless disabled; the tiled torch path is the fallback/reference
        self.use_taichi = use_taichi

    def solve(self, pos, masses, potential_type, params, periodic=False, universe_w=None, universe_h=None):
        """
        Computes exact O(N^2) pairwise accelerations. Returns (ax, ay), each (N,), on pos.device.
        periodic: use minimum-image displacements, matching the wrapped (toroidal) universe.
        """
        N = len(pos)
        if N <= 1:
            return torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
        if potential_type not in _KINDS:
            raise ValueError(f"Unknown potential type: {potential_type}")
        if self.use_taichi:
            return self._solve_taichi(pos, masses, potential_type, params, periodic, universe_w, universe_h)
        return self._solve_torch(pos, masses, potential_type, params, periodic, universe_w, universe_h)

    def _solve_taichi(self, pos, masses, potential_type, params, periodic, universe_w, universe_h):
        """One fused Taichi kernel (one thread per target, no N x N temporaries), launched via kernels.nbody_accel."""
        N = len(pos)
        return nbody_accel(
            pos, masses, _KINDS[potential_type],
            float(params.get('G', 1.0)), float(params.get('spring_k', 0.1)) / N,
            float(params.get('epsilon', 1000.0)), float(params.get('sigma', 40.0)),
            float(params.get('softening_length', 1.5)) ** 2, 1.0 / float(params.get('force_range', 8192.0)),
            int(bool(periodic)), float(universe_w or 1.0), float(universe_h or 1.0)
        )

    def _solve_torch(self, pos, masses, potential_type, params, periodic, universe_w, universe_h):
        """
        Pure PyTorch reference (any torch device): target particles in row tiles against all sources.
        Self-interaction needs no masking: its displacement is exactly 0 and every force is softened/clamped,
        so the self term contributes factor * 0 = 0.
        """
        N = len(pos)

        px = pos[:, 0]
        py = pos[:, 1]
        m = masses.to(pos.dtype)
        eps2 = params.get('softening_length', 1.5) ** 2          # Plummer softening
        inv_range = 1.0 / params.get('force_range', 8192.0)      # exponential force-range screening
        G = params.get('G', 1.0)
        spring_k = params.get('spring_k', 0.1)
        epsilon = params.get('epsilon', 1000.0)
        sigma = params.get('sigma', 40.0)

        ax = torch.empty(N, device=pos.device, dtype=pos.dtype)
        ay = torch.empty(N, device=pos.device, dtype=pos.dtype)
        block = max(1, self.tile_elements // N)
        for i0 in range(0, N, block):
            i1 = min(N, i0 + block)
            # Displacements diff[i, j] = pos[j] - pos[i] for this tile of targets i
            dx = px[None, :] - px[i0:i1, None]
            dy = py[None, :] - py[i0:i1, None]
            if periodic:
                dx -= universe_w * torch.round(dx / universe_w)
                dy -= universe_h * torch.round(dy / universe_h)

            r_soft_sq = dx * dx + dy * dy + eps2
            r = torch.sqrt(r_soft_sq)
            factor = torch.exp(-r * inv_range)                    # screening

            if potential_type == 'gravity':
                # a_i = sum_j G * m_j * diff_ij / (r_ij^2 + softening^2)   ->  |a| = G*m/r (2D law)
                factor.div_(r_soft_sq).mul_(G * m[None, :])
            elif potential_type == 'coulomb':
                # Like charges repel: opposite sign to gravity, scaled by the target's own charge below
                factor.div_(r_soft_sq).mul_(-G * m[None, :])
            elif potential_type == 'harmonic':
                # Springs with mean-field (1/N) scaling: a_i = (k/N) * sum_j m_j * diff_ij
                factor.mul_((spring_k / N) * m[None, :])
            else:  # lennard_jones
                s6 = (sigma / r) ** 6
                # Force magnitude F(r) = 24 * eps / r * (2 (sigma/r)^12 - (sigma/r)^6), clamped against spikes
                force_mag = ((24.0 * epsilon) / r * (2.0 * s6 * s6 - s6)).clamp_(-500000.0, 500000.0)
                factor.mul_(force_mag).div_(r).mul_(m[None, :])

            ax[i0:i1] = (factor * dx).sum(1)
            ay[i0:i1] = (factor * dy).sum(1)

        if potential_type == 'coulomb':
            ax.mul_(m)
            ay.mul_(m)
        return ax, ay
