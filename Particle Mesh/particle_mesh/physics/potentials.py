import torch
import math

from particle_mesh.physics.grid_ops import neg_gradient


LJ_FORCE_CLAMP = 500000.0
# Radial potential tables are integrated on this many samples per physics cell
_TABLE_SAMPLES_PER_CELL = 16
# Elements per chunk when sampling a radial table onto a grid (bounds the int64 index temporaries)
_SAMPLE_CHUNK = 1 << 22


def attraction(potential_type, params, d):
    """
    Radial acceleration a(d) of a target towards a unit source at distance d (positive = pulled towards it): the
    direct N-body force law (nbody_solver, taichi_impl.nbody_accel_kernel), so the PM kernels built from it match
    N-body by construction. Plummer softening r = sqrt(d^2 + eps^2) and exponential force-range screening.
    Coulomb is gravity with the opposite sign (the gather multiplies by the target's own charge).
    """
    eps2 = params.get('softening_length', 1.5) ** 2
    inv_range = 1.0 / params.get('force_range', math.inf)
    r2 = d * d + eps2
    r = torch.sqrt(r2)
    screen = torch.exp(-r * inv_range)
    if potential_type in ('gravity', 'coulomb'):
        sign = 1.0 if potential_type == 'gravity' else -1.0
        return sign * params.get('G', 1.0) * screen * d / r2          # |a| = G * m / r: the 2D law
    if potential_type == 'harmonic':
        return params.get('spring_k', 0.1) * screen * d                # mean-field 1/N applied by the solver
    if potential_type == 'lennard_jones':
        s6 = (params.get('sigma', 40.0) / r) ** 6
        epsilon = params.get('epsilon', 1000.0)
        repulsion = (24.0 * epsilon / r * (2.0 * s6 * s6 - s6)).clamp(-LJ_FORCE_CLAMP, LJ_FORCE_CLAMP)
        return -repulsion * screen * d / r
    raise ValueError(f"Unknown potential type: {potential_type}")


def uses_analytic_kernel(potential_type, isolated):
    """Periodic gravity/Coulomb use the exact Fourier-space kernel; every other case a sampled real-space one."""
    return not isolated and potential_type in ('gravity', 'coulomb')


class PotentialsManager:
    def __init__(self, grid_w, grid_h, cell_size, device):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.device = device
        # Isolated (open / walled) solves zero-pad the grid to one square S x S FFT, S = 2 * max(W, H): every
        # pair offset inside the mesh then maps to a unique kernel cell (no periodic images), and the kernel
        # reaches equally far along both axes.
        self.isolated_size = 2 * max(grid_w, grid_h)

        # |k|^2 in rfft2 layout (H, W//2 + 1) of the unpadded grid: periodic gravity/Coulomb kernels, Gaussian
        # softening, pressure smoothing and the band-pass view. Kernels are built directly in half-spectrum layout.
        kx = torch.fft.rfftfreq(grid_w, d=cell_size, device=device) * 2 * math.pi
        ky = torch.fft.fftfreq(grid_h, d=cell_size, device=device) * 2 * math.pi
        self.k_sq = ky[:, None] ** 2 + kx[None, :] ** 2

    def _offsets(self, n):
        """Minimum-image offsets of the cells of a periodic axis of n cells from cell 0, in world units."""
        i = torch.arange(n, device=self.device, dtype=torch.float32)
        return torch.where(i > n / 2.0, i - n, i) * self.cell_size

    def _radial_potential(self, potential_type, params, r_max):
        """
        V(d) = -integral_d^r_max a(s) ds on a fine radial table (float64 on the CPU, trapezoid rule), so that
        a = -dV/dd reproduces the N-body force law. V is zero at r_max: large constant offsets would cost float32
        precision in phi. Returns (float32 table on the device, sample spacing).
        """
        h = self.cell_size / _TABLE_SAMPLES_PER_CELL
        d = torch.arange(int(math.ceil(r_max / h)) + 2, dtype=torch.float64) * h
        a = attraction(potential_type, params, d)
        cum = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum((a[1:] + a[:-1]) * (0.5 * h), 0)])
        return (cum - cum[-1]).to(device=self.device, dtype=torch.float32), h

    def _sample_radial(self, potential_type, params, size_h, size_w):
        """The radial potential at the minimum-image distance of every cell of a (size_h, size_w) grid from cell
        (0, 0), linearly interpolated from the table. Built in row chunks to bound memory."""
        dx, dy = self._offsets(size_w), self._offsets(size_h)
        r_max = math.hypot(float(dx.abs().max()), float(dy.abs().max()))
        table, h = self._radial_potential(potential_type, params, r_max)
        out = torch.empty((size_h, size_w), device=self.device)
        rows = max(1, _SAMPLE_CHUNK // size_w)
        for r0 in range(0, size_h, rows):
            u = torch.sqrt(dy[r0:r0 + rows, None] ** 2 + dx[None, :] ** 2).div_(h)
            i0 = u.floor()
            frac = u.sub_(i0)
            i0 = i0.long()
            v0 = table[i0]
            out[r0:r0 + rows] = v0.add_(table[i0 + 1].sub_(v0).mul_(frac))
        return out

    def get_fourier_kernel(self, potential_type, params, isolated=False):
        """
        The 2D Fourier-space interaction kernel K_k for the selected potential, per unit source mass per cell.
        Periodic: rfft2 layout of the (H, W) grid. Isolated: rfft2 layout of the zero-padded square
        (S, S) grid (S = isolated_size).

        potential_type: 'gravity', 'lennard_jones', 'harmonic', 'coulomb'
        params: dict containing sliders (G, epsilon, sigma, spring_k, force_range, softening_length);
                force_range = inf means no screening.

        Periodic gravity/Coulomb use the exact Fourier kernel (real float32; the solver adds Gaussian softening).
        Every other case samples the real-space potential of the N-body force law (attraction()) and FFTs it.
        """
        if potential_type not in ('gravity', 'coulomb', 'harmonic', 'lennard_jones'):
            raise ValueError(f"Unknown potential type: {potential_type}")

        if isolated:
            # Free-space convolution (Hockney & Eastwood): no k = 0 removal, the full sum is the physics
            s = self.isolated_size
            return torch.fft.rfft2(self._sample_radial(potential_type, params, s, s))

        if uses_analytic_kernel(potential_type, isolated):
            # Units (AUDIT 2.11): the 2D law |a| = G * m / r, i.e. lap(phi) = 2*pi*G*rho_area and
            # phi_k = -2*pi*G * rho_area_k / k^2, screened as Yukawa 1/(k^2 + 1/range^2). Mass is deposited per
            # CELL, so divide by the cell area (rho_area = rho_cell / cell_size^2).
            area_scale = 2.0 * math.pi / (self.cell_size ** 2)
            lambda_sq = (1.0 / params.get('force_range', math.inf)) ** 2
            denom = self.k_sq + lambda_sq
            denom[0, 0] = 1.0   # k = 0 is zeroed below; avoid dividing by ~0 there
            # Gravity attracts (negative); Coulomb repels like charges (positive)
            sign = -1.0 if potential_type == 'gravity' else 1.0
            K_k = (sign * params.get('G', 1.0) * area_scale) / denom
        else:
            # Harmonic / Lennard-Jones: minimum-image pair potential, the same convention as periodic N-body
            K_k = torch.fft.rfft2(self._sample_radial(potential_type, params, self.grid_h, self.grid_w))

        # Zero the k = 0 (mean) mode. It only adds a constant to phi, which exerts no force, but for gravity/Coulomb
        # it equals -/+G * M_total / lambda^2: large enough to destroy float32 precision in phi (and therefore in
        # the finite-difference forces) and to wash out the potential view.
        K_k[0, 0] = 0.0
        return K_k

    def compute_pressure_forces(self, density, pressure_strength, gamma, periodic, density_floor, out_x, out_y):
        """
        Polytropic gas pressure acceleration a = -grad(P) / rho with P = K * rho^gamma (AUDIT 2.9), written into
        out_x / out_y. For gamma = 1 (isothermal) this is -K * grad(ln rho): it depends on the density *profile*,
        not its amplitude, so dense clumps no longer get explosive kicks.

        density: smoothed number density per unit area (H, W), >= 0
        pressure_strength: K
        gamma: polytropic index (1.0 = isothermal, 1.4 = air, 1.67 = monatomic gas)
        density_floor: rho used in the denominator is at least this (keeps near-empty cells finite)
        """
        if pressure_strength <= 1e-5:
            out_x.zero_()
            out_y.zero_()
            return out_x, out_y

        p = density if gamma == 1.0 else density ** gamma          # no pow pass when isothermal (AUDIT §8 F22)
        neg_gradient(p, out_x, out_y, self.cell_size, periodic)    # -grad(P / K), slices (no roll copies)
        scale = density.clamp(min=density_floor).reciprocal_().mul_(pressure_strength)
        return out_x.mul_(scale), out_y.mul_(scale)

    def get_uniform_gravity_forces(self, magnitude, angle_deg):
        """
        Computes uniform background gravity vector.

        magnitude: acceleration rate
        angle_deg: direction in degrees (0 = right, 90 = up, 180 = left, 270 = down)
        """
        angle_rad = math.radians(angle_deg)
        gx = magnitude * math.cos(angle_rad)
        gy = magnitude * math.sin(angle_rad)
        return gx, gy
