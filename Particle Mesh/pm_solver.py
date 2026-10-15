import torch
import math

from grid_ops import neg_gradient


class ParticleMeshSolver:
    def __init__(self, grid_w, grid_h, cell_size, device, potentials_manager, kernels):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.device = device
        self.pot_manager = potentials_manager
        self.kernels = kernels  # particle deposit/gather backend (kernels.make_backend)

        # Grid allocations. fx/fy come from the backend: for the torch backend they are the interiors of a padded
        # buffer the fused CIC gather reads directly; for Taichi they are plain contiguous grids.
        self.density_grid = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.phi = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.fx, self.fy = kernels.force_buffers(self.grid_h, self.grid_w, self.device)
        self._px = torch.zeros((self.grid_h, self.grid_w), device=self.device)       # pressure acceleration
        self._py = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self._number = None                                                          # Coulomb: |mass| deposit
        self._pressure_accel = None    # (px, py) gathered without the charge factor (Coulomb + pressure)
        self.rho_k = None  # rfft2 of the last deposited density; reused by the FFT minimap and bandpass view
        self.field_valid = False

        # |k|^2 in rfft2 layout (H, W//2 + 1), shared with the potentials manager
        self.k_sq = potentials_manager.k_sq
        # Inverse FFTs use norm='forward' (no separate 1/N pass, measured 1.36 -> 1.09 ms at 2048^2); the 1/N is
        # folded into the cached kernels instead (AUDIT §8 F24).
        self._inv_cells = 1.0 / (self.grid_w * self.grid_h)

        self.K_k = None
        self.active_potential_type = None
        self._kernel_version = 0   # bumped on every rebuild; part of the K_eff cache key (id() can be reused)
        # K_k * Gaussian softening / N, in rfft2 layout; rebuilt only when the kernel or the softening changes
        self._K_eff = None
        self._K_eff_key = None
        # Gaussian (/ N) used to smooth the number density for pressure; rebuilt when the smoothing length changes
        self._smooth = None
        self._smooth_key = None

    def update_kernel(self, potential_type, params):
        """
        Retrieves the updated Fourier kernel from potentials.py.
        """
        self.active_potential_type = potential_type
        self.K_k = self.pot_manager.get_fourier_kernel(potential_type, params)
        self._kernel_version += 1

    def _compute_field(self, stencil, pos, masses, num_active, potential_type, params, periodic):
        """
        Deposit -> rfft2 -> kernel (with Gaussian softening) -> irfft2 -> central differences.
        Updates self.density_grid, self.rho_k, self.phi, self.fx, self.fy (the force per unit mass/charge).
        """
        # Re-fetch kernel if the potential type switched
        if self.K_k is None or self.active_potential_type != potential_type:
            self.update_kernel(potential_type, params)

        # Scatter masses (the backend overwrites density_grid)
        self.kernels.deposit(stencil, pos, masses, num_active, self.density_grid, self.cell_size, periodic)

        # Forward Real-FFT (rfft2)
        rho_k = torch.fft.rfft2(self.density_grid)
        self.rho_k = rho_k

        # Convolution with the kernel, Gaussian force softening and the inverse-FFT 1/N (cached product)
        softening_length = params.get('softening_length', 1.5)
        key = (self._kernel_version, softening_length)
        if self._K_eff_key != key:
            gaussian = torch.exp(-self.k_sq * (softening_length**2) / 2.0).mul_(self._inv_cells)
            self._K_eff = self.K_k * gaussian
            self._K_eff_key = key
        phi_k = rho_k * self._K_eff
        if potential_type == 'harmonic':
            # Mean-field scaling: spring force per unit total mass, as in the N-body solver (independent of N)
            phi_k.mul_(1.0 / num_active)

        # Inverse FFT to get potential
        self.phi = torch.fft.irfft2(phi_k, s=self.density_grid.shape, norm='forward')

        # Forces F = -grad(phi), central differences written straight into the force buffers
        neg_gradient(self.phi, self.fx, self.fy, self.cell_size, periodic)
        self.field_valid = True

    def _apply_pressure(self, stencil, pos, masses, num_active, potential_type, params, pressure_model,
                        pressure_strength, gamma, periodic):
        """
        Gas pressure: a = -grad(P)/rho on a smoothed NUMBER density (AUDIT 2.9). Added into fx/fy, except in
        Coulomb mode, where it is kept separate (self._pressure_accel) and gathered without the charge factor.
        """
        gamma_val = gamma if pressure_model == 'polytropic' else 1.0
        if potential_type == 'coulomb':
            # density_grid holds charge here; pressure acts on particles, so deposit |mass| separately
            if self._number is None:
                self._number = torch.zeros_like(self.density_grid)
            self.kernels.deposit(stencil, pos, masses[:num_active].abs(), num_active, self._number, self.cell_size, periodic)
            number_k = torch.fft.rfft2(self._number)
        else:
            number_k = self.rho_k
        smooth_len = max(params.get('softening_length', 1.5), self.cell_size)
        if self._smooth_key != smooth_len:
            self._smooth = torch.exp(-self.k_sq * (smooth_len**2) / 2.0).mul_(self._inv_cells)
            self._smooth_key = smooth_len
        area = self.cell_size ** 2
        rho_smooth = torch.fft.irfft2(number_k * self._smooth, s=self.density_grid.shape, norm='forward')
        rho_smooth.clamp_(min=0.0).div_(area)
        self.pot_manager.compute_pressure_forces(
            rho_smooth, pressure_strength, gamma_val, periodic, 0.05 / area, self._px, self._py
        )
        if potential_type == 'coulomb':
            self._pressure_accel = (self._px, self._py)   # charge-independent
        else:
            self._pressure_accel = None
            self.fx.add_(self._px)
            self.fy.add_(self._py)

    def sample_forces(self, positions, universe_w, universe_h):
        """
        Field acceleration (bilinear = CIC interpolation of fx/fy) at the given (M, 2) positions.
        """
        size = torch.tensor([universe_w, universe_h], device=positions.device, dtype=positions.dtype)
        coords = (positions / size * 2.0 - 1.0).view(1, 1, -1, 2)
        ax = torch.nn.functional.grid_sample(self.fx[None, None], coords, mode='bilinear', padding_mode='border',
                                             align_corners=False)[0, 0, 0]
        ay = torch.nn.functional.grid_sample(self.fy[None, None], coords, mode='bilinear', padding_mode='border',
                                             align_corners=False)[0, 0, 0]
        return ax, ay

    def accelerations_at(self, pos, masses, num_active, start, end, potential_type, params, periodic,
                         universe_w, universe_h):
        """
        Field acceleration (no pressure, no integration) felt by particles start:end, from all num_active
        particles, via a field-only solve. Overwrites the solver's field state (the next solve() recomputes it).
        The normal spawn path avoids this second solve (engine applies orbit speeds inside step(); AUDIT §8 F04);
        this remains for the case where particles must be erased before the next step.
        """
        stencil = self.kernels.prepare(pos, num_active, self.cell_size, self.grid_w, self.grid_h, periodic)
        self._compute_field(stencil, pos, masses, num_active, potential_type, params, periodic)
        return self.sample_forces(pos[start:end], universe_w, universe_h)

    def solve(self, pos, vel, masses, num_active, potential_type, params, dt, friction, pressure_model, pressure_strength, gamma,
              periodic, universe_w, universe_h, uniform_accel=(0.0, 0.0), recompute_field=True, before_gather=None):
        """
        Runs Particle-Mesh solver:
        1. Deposits mass to grid (CIC) and solves the Poisson equation with FFTs (skipped when recompute_field is
           False and a field for this potential exists: opt-in multi-rate updates, AUDIT §8 F32).
        2. Calls before_gather(solver), if given, with the fresh field-only forces (used for spawn orbit speeds).
        3. Adds gas pressure, gathers forces back to the particles and integrates them (wraps them if periodic).
        """
        if num_active == 0:
            return

        # The deposit and the gather both use this step's positions, so the CIC stencil is built once
        stencil = self.kernels.prepare(pos, num_active, self.cell_size, self.grid_w, self.grid_h, periodic)

        pressure_on = pressure_model != 'none' and pressure_strength > 1e-5
        if recompute_field or not self.field_valid or potential_type != self.active_potential_type:
            self._compute_field(stencil, pos, masses, num_active, potential_type, params, periodic)
            if before_gather is not None:
                before_gather(self)
            if pressure_on:
                self._apply_pressure(stencil, pos, masses, num_active, potential_type, params, pressure_model,
                                     pressure_strength, gamma, periodic)
            else:
                self._pressure_accel = None
        accel_x, accel_y = self._pressure_accel if (pressure_on and self._pressure_accel is not None) else (None, None)

        # Exponential friction decay: stays in (0, 1] for any friction * dt, unlike (1 - friction * dt)
        damping = math.exp(-friction * dt)

        # Force gathering, kick (incl. uniform gravity), drift (and periodic wrap)
        self.kernels.gather_integrate(
            stencil, pos, vel, masses, num_active,
            self.fx, self.fy, self.cell_size, dt, damping,
            potential_type == 'coulomb', periodic, universe_w, universe_h,
            accel_x, accel_y, uniform_accel
        )
