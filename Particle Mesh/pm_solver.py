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

        # Precompute k_sq for FFT
        kx = torch.fft.fftfreq(self.grid_w, d=self.cell_size, device=self.device) * 2 * math.pi
        ky = torch.fft.fftfreq(self.grid_h, d=self.cell_size, device=self.device) * 2 * math.pi
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='xy')

        self.k_sq = kx_grid**2 + ky_grid**2
        self.k_sq[0, 0] = 1e-8

        self.K_k = None
        self.active_potential_type = None
        # K_k * Gaussian softening, in rfft2 layout; rebuilt only when the kernel or the softening length changes
        self._K_eff = None
        self._K_eff_key = None

    def update_kernel(self, potential_type, params):
        """
        Retrieves the updated Fourier kernel from potentials.py.
        """
        self.active_potential_type = potential_type
        self.K_k = self.pot_manager.get_fourier_kernel(potential_type, params)

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

        # Re-fetch kernel if model parameters changed or type switched
        if self.K_k is None or self.active_potential_type != potential_type:
            self.update_kernel(potential_type, params)

        # The deposit and the gather both use this step's positions, so the CIC stencil is built once
        stencil = self.kernels.prepare(pos, num_active, self.cell_size, self.grid_w, self.grid_h, periodic)

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
            cols = rho_k.shape[1]
            gaussian = torch.exp(-self.k_sq[:, :cols] * (softening_length**2) / 2.0)
            self._K_eff = self.K_k[:, :cols] * gaussian
            self._K_eff_key = key
        phi_k = rho_k * self._K_eff

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

        # Add real-space gas pressure if active
        if pressure_model != 'none':
            gamma_val = gamma if pressure_model == 'polytropic' else 1.0
            force_px, force_py = self.pot_manager.compute_pressure_forces(self.density_grid, pressure_strength, gamma_val)
            self.fx += force_px
            self.fy += force_py

        # Exponential friction decay: stays in (0, 1] for any friction * dt, unlike (1 - friction * dt)
        damping = math.exp(-friction * dt)

        # Force gathering, kick, drift (and periodic wrap)
        self.kernels.gather_integrate(
            stencil, pos, vel, masses, num_active,
            self.fx, self.fy, self.cell_size, dt, damping,
            potential_type == 'coulomb', periodic, universe_w, universe_h
        )
