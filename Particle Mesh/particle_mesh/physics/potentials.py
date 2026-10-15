import torch
import math

from particle_mesh.physics.grid_ops import neg_gradient


class PotentialsManager:
    def __init__(self, grid_w, grid_h, cell_size, device):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.device = device

        # |k|^2 in rfft2 layout (H, W//2 + 1): the only k-space grid the solver needs (AUDIT 3.1). Kernels are
        # built directly in this half-spectrum layout, so nothing full-size or full-complex is kept.
        kx = torch.fft.rfftfreq(grid_w, d=cell_size, device=device) * 2 * math.pi
        ky = torch.fft.fftfreq(grid_h, d=cell_size, device=device) * 2 * math.pi
        self.k_sq = ky[:, None] ** 2 + kx[None, :] ** 2

    def _distance_grid(self):
        """Periodic distance from cell (0, 0) for every cell, (H, W). Built on demand (harmonic/LJ kernels only)."""
        dx = torch.arange(self.grid_w, device=self.device, dtype=torch.float32)
        dy = torch.arange(self.grid_h, device=self.device, dtype=torch.float32)
        # Periodic distance mapping centered at index (0,0) to prevent spatial shifting
        dx = torch.where(dx > self.grid_w / 2.0, dx - self.grid_w, dx) * self.cell_size
        dy = torch.where(dy > self.grid_h / 2.0, dy - self.grid_h, dy) * self.cell_size
        return torch.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)

    def get_fourier_kernel(self, potential_type, params):
        """
        Generates the 2D Fourier-space interaction kernel K_k (rfft2 layout) for the selected potential.
        Gravity/Coulomb kernels are real float32; harmonic/LJ kernels are the complex rfft2 of a sampled V(r).

        potential_type: 'gravity', 'lennard_jones', 'harmonic', 'coulomb'
        params: dict containing sliders (G, epsilon, sigma, spring_k, force_range)
        """
        force_range = params.get('force_range', 1000.0)

        # Units (AUDIT 2.11). Gravity and Coulomb use the 2D law |a| = G * m / r, the same law as the direct
        # N-body solver: the Poisson equation is lap(phi) = 2*pi*G*rho_area, i.e. phi_k = -2*pi*G * rho_area_k / k^2.
        # The solver deposits mass per CELL, so divide by the cell area (rho_area = rho_cell / cell_size^2).
        # Harmonic and Lennard-Jones are FFTs of a sampled pair potential V(r): a discrete convolution with
        # mass-per-cell already sums V over particles, so they need no area factor.
        area_scale = 2.0 * math.pi / (self.cell_size ** 2)

        if potential_type in ('gravity', 'coulomb'):
            strength = params.get('G', 1.0)
            # Screened Yukawa interaction in Fourier space: lambda_sq = (1 / force_range)^2
            lambda_sq = (1.0 / force_range) ** 2
            denom = self.k_sq + lambda_sq
            denom[0, 0] = 1.0   # k = 0 is zeroed below; avoid dividing by ~0 there
            # Gravity attracts (negative); Coulomb repels like charges (positive)
            sign = -1.0 if potential_type == 'gravity' else 1.0
            K_k = (sign * strength * area_scale) / denom

        elif potential_type == 'harmonic':
            spring_k = params.get('spring_k', 1.0)
            r = self._distance_grid()
            # Mutual spring potential V(r) = 0.5 * spring_k * r^2, with force-range screening against periodic jumps
            V = 0.5 * spring_k * (r ** 2) * torch.exp(-r / force_range)
            K_k = torch.fft.rfft2(V)

        elif potential_type == 'lennard_jones':
            epsilon = params.get('epsilon', 100.0)
            sigma = params.get('sigma', 10.0)
            r = self._distance_grid()
            # Soften grid center to avoid unphysical infinite repulsion at r=0
            r_soft = torch.clamp(r, min=self.cell_size * 0.5)
            # V(r) = 4 * eps * ((sigma/r)^12 - (sigma/r)^6)
            V = 4.0 * epsilon * ((sigma / r_soft) ** 12 - (sigma / r_soft) ** 6)
            V = torch.clamp(V, min=-500000.0, max=500000.0)
            # Universal force-range screening smooths outer cutoff
            V = V * torch.exp(-r / force_range)
            K_k = torch.fft.rfft2(V)

        else:
            raise ValueError(f"Unknown potential type: {potential_type}")

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
