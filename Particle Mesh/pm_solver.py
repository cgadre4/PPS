import torch
import math


class ParticleMeshSolver:
    def __init__(self, grid_w, grid_h, cell_size, device, potentials_manager, kernels):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.device = device
        self.pot_manager = potentials_manager
        self.kernels = kernels  # particle deposit/gather backend (kernels.make_backend)

        # Grid allocations
        self.density_grid = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.phi = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.fx = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.fy = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.rho_k = None  # rfft2 of the last deposited density; reused by the FFT minimap and bandpass view

        # |k|^2 in rfft2 layout (H, W//2 + 1), shared with the potentials manager
        self.k_sq = potentials_manager.k_sq

        self.K_k = None
        self.active_potential_type = None
        # K_k * Gaussian softening, in rfft2 layout; rebuilt only when the kernel or the softening length changes
        self._K_eff = None
        self._K_eff_key = None
        # Gaussian used to smooth the number density for pressure; rebuilt when the smoothing length changes
        self._smooth = None
        self._smooth_key = None

    def update_kernel(self, potential_type, params):
        """
        Retrieves the updated Fourier kernel from potentials.py.
        """
        self.active_potential_type = potential_type
        self.K_k = self.pot_manager.get_fourier_kernel(potential_type, params)

    def _compute_field(self, stencil, pos, masses, num_active, potential_type, params, periodic):
        """
        Deposit -> rfft2 -> kernel (with Gaussian softening) -> irfft2 -> central differences.
        Updates self.density_grid, self.rho_k, self.phi, self.fx, self.fy (the force per unit mass/charge).
        """
        # Re-fetch kernel if the potential type switched
        if self.K_k is None or self.active_potential_type != potential_type:
            self.update_kernel(potential_type, params)

        # Clear grid and scatter masses
        self.density_grid.zero_()
        self.kernels.deposit(stencil, pos, masses, num_active, self.density_grid, self.cell_size, periodic)

        # Forward Real-FFT (rfft2)
        rho_k = torch.fft.rfft2(self.density_grid)
        self.rho_k = rho_k

        # Convolution with the kernel and Gaussian force softening (cached product)
        softening_length = params.get('softening_length', 1.5)
        key = (id(self.K_k), softening_length)
        if self._K_eff_key != key:
            gaussian = torch.exp(-self.k_sq * (softening_length**2) / 2.0)
            self._K_eff = self.K_k * gaussian
            self._K_eff_key = key
        phi_k = rho_k * self._K_eff
        if potential_type == 'harmonic':
            # Mean-field scaling: spring force per unit total mass, as in the N-body solver (independent of N)
            phi_k.mul_(1.0 / num_active)

        # Inverse FFT to get potential
        self.phi = torch.fft.irfft2(phi_k, s=self.density_grid.shape)

        # Potential gradients: F = -grad(phi), central differences (wrapping at the edges when periodic)
        if periodic:
            inv_2dx = 0.5 / self.cell_size
            self.fx = (torch.roll(self.phi, 1, dims=1) - torch.roll(self.phi, -1, dims=1)).mul_(inv_2dx)
            self.fy = (torch.roll(self.phi, 1, dims=0) - torch.roll(self.phi, -1, dims=0)).mul_(inv_2dx)
        else:
            self.fy, self.fx = torch.gradient(self.phi)
            self.fx.neg_().div_(self.cell_size)
            self.fy.neg_().div_(self.cell_size)

    def accelerations_at(self, pos, masses, num_active, start, end, potential_type, params, periodic,
                         universe_w, universe_h):
        """
        Field acceleration (no pressure, no integration) felt by particles start:end, from all num_active
        particles. CIC interpolation on the cell-centred grid is exactly bilinear grid_sample with
        align_corners=False. Used to give freshly spawned galaxies their true circular speed.
        Overwrites the solver's field state (the next solve() recomputes it anyway).
        """
        stencil = self.kernels.prepare(pos, num_active, self.cell_size, self.grid_w, self.grid_h, periodic)
        self._compute_field(stencil, pos, masses, num_active, potential_type, params, periodic)
        p = pos[start:end]
        size = torch.tensor([universe_w, universe_h], device=pos.device, dtype=pos.dtype)
        coords = (p / size * 2.0 - 1.0).view(1, 1, -1, 2)
        forces = torch.nn.functional.grid_sample(torch.stack((self.fx, self.fy))[None], coords,
                                                 mode='bilinear', padding_mode='border', align_corners=False)
        return forces[0, 0, 0], forces[0, 1, 0]

    def solve(self, pos, vel, masses, num_active, potential_type, params, dt, friction, pressure_model, pressure_strength, gamma,
              periodic, universe_w, universe_h):
        """
        Runs Particle-Mesh solver:
        1. Deposits mass to grid (CIC).
        2. Solves the Poisson equation with FFTs on the torch device.
        3. Gathers forces back to the particles and integrates them (and wraps them if periodic).
        """
        if num_active == 0:
            return

        # The deposit and the gather both use this step's positions, so the CIC stencil is built once
        stencil = self.kernels.prepare(pos, num_active, self.cell_size, self.grid_w, self.grid_h, periodic)

        self._compute_field(stencil, pos, masses, num_active, potential_type, params, periodic)
        rho_k = self.rho_k
        softening_length = params.get('softening_length', 1.5)

        # Gas pressure: a = -grad(P)/rho on a smoothed NUMBER density (AUDIT 2.9)
        accel_x = accel_y = None
        if pressure_model != 'none' and pressure_strength > 1e-5:
            gamma_val = gamma if pressure_model == 'polytropic' else 1.0
            if potential_type == 'coulomb':
                # density_grid holds charge here; pressure acts on particles, so deposit |mass| separately
                number = torch.zeros_like(self.density_grid)
                self.kernels.deposit(stencil, pos, masses[:num_active].abs(), num_active, number, self.cell_size, periodic)
                number_k = torch.fft.rfft2(number)
            else:
                number_k = rho_k
            smooth_len = max(softening_length, self.cell_size)
            if self._smooth_key != smooth_len:
                self._smooth = torch.exp(-self.k_sq * (smooth_len**2) / 2.0)
                self._smooth_key = smooth_len
            area = self.cell_size ** 2
            rho_smooth = torch.fft.irfft2(number_k * self._smooth, s=self.density_grid.shape).clamp_(min=0.0).div_(area)
            px, py = self.pot_manager.compute_pressure_forces(
                rho_smooth, pressure_strength, gamma_val, periodic, density_floor=0.05 / area
            )
            if potential_type == 'coulomb':
                accel_x, accel_y = px, py   # charge-independent: gathered without the +/-1 charge factor
            else:
                self.fx += px
                self.fy += py

        # Exponential friction decay: stays in (0, 1] for any friction * dt, unlike (1 - friction * dt)
        damping = math.exp(-friction * dt)

        # Force gathering, kick, drift (and periodic wrap)
        self.kernels.gather_integrate(
            stencil, pos, vel, masses, num_active,
            self.fx, self.fy, self.cell_size, dt, damping,
            potential_type == 'coulomb', periodic, universe_w, universe_h,
            accel_x, accel_y
        )
