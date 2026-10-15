import taichi as ti
import torch
import math

@ti.kernel
def scatter_mass_kernel(
    pos: ti.types.ndarray(), mass: ti.types.ndarray(), density: ti.types.ndarray(),
    num_active: ti.i32, grid_w: ti.i32, grid_h: ti.i32, cell_size: ti.f32
):
    for p in range(num_active):
        x = pos[p, 0] / cell_size - 0.5
        y = pos[p, 1] / cell_size - 0.5

        x = ti.max(0.0, ti.min(x, grid_w - 1.001))
        y = ti.max(0.0, ti.min(y, grid_h - 1.001))

        i, j = ti.cast(x, ti.i32), ti.cast(y, ti.i32)
        wx, wy = x - ti.cast(i, ti.f32), y - ti.cast(j, ti.f32)

        w00, w10 = (1.0 - wx) * (1.0 - wy), wx * (1.0 - wy)
        w01, w11 = (1.0 - wx) * wy, wx * wy

        m = ti.cast(mass[p], ti.f32) 
        ti.atomic_add(density[j, i], m * w00)
        ti.atomic_add(density[j, i + 1], m * w10)
        ti.atomic_add(density[j + 1, i], m * w01)
        ti.atomic_add(density[j + 1, i + 1], m * w11)

@ti.kernel
def gather_and_integrate_kernel(
    pos: ti.types.ndarray(), vel: ti.types.ndarray(), force_x: ti.types.ndarray(), force_y: ti.types.ndarray(),
    num_active: ti.i32, grid_w: ti.i32, grid_h: ti.i32, cell_size: ti.f32, dt: ti.f32, friction: ti.f32
):
    for p in range(num_active):
        x = pos[p, 0] / cell_size - 0.5
        y = pos[p, 1] / cell_size - 0.5

        x = ti.max(0.0, ti.min(x, grid_w - 1.001))
        y = ti.max(0.0, ti.min(y, grid_h - 1.001))

        i, j = ti.cast(x, ti.i32), ti.cast(y, ti.i32)
        wx, wy = x - ti.cast(i, ti.f32), y - ti.cast(j, ti.f32)

        w00, w10 = (1.0 - wx) * (1.0 - wy), wx * (1.0 - wy)
        w01, w11 = (1.0 - wx) * wy, wx * wy

        fx = force_x[j, i]*w00 + force_x[j, i+1]*w10 + force_x[j+1, i]*w01 + force_x[j+1, i+1]*w11
        fy = force_y[j, i]*w00 + force_y[j, i+1]*w10 + force_y[j+1, i]*w01 + force_y[j+1, i+1]*w11

        vel[p, 0] = (vel[p, 0] + fx * dt) * (1.0 - friction * dt)
        vel[p, 1] = (vel[p, 1] + fy * dt) * (1.0 - friction * dt)
        
        pos[p, 0] += vel[p, 0] * dt
        pos[p, 1] += vel[p, 1] * dt


@ti.kernel
def poisson_rbgs_kernel(
    phi: ti.types.ndarray(),
    density: ti.types.ndarray(),
    grid_w: ti.i32,
    grid_h: ti.i32,
    cell_size: ti.f32,
    red_black: ti.i32
):
    dx2 = cell_size * cell_size
    for j, i in ti.ndrange((1, grid_h - 1), (1, grid_w - 1)):
        if (i + j) % 2 == red_black:
            phi[j, i] = 0.25 * (
                phi[j, i + 1] + phi[j, i - 1] +
                phi[j + 1, i] + phi[j - 1, i] -
                dx2 * density[j, i]
            )

@ti.kernel
def compute_forces_kernel(
    phi: ti.types.ndarray(),
    force_x: ti.types.ndarray(),
    force_y: ti.types.ndarray(),
    grid_w: ti.i32,
    grid_h: ti.i32,
    cell_size: ti.f32
):
    inv_2dx = 0.5 / cell_size
    for j, i in ti.ndrange((1, grid_h - 1), (1, grid_w - 1)):
        force_x[j, i] = -(phi[j, i + 1] - phi[j, i - 1]) * inv_2dx
        force_y[j, i] = -(phi[j + 1, i] - phi[j - 1, i]) * inv_2dx


class ParticleMeshSolver:
    def __init__(self, grid_w, grid_h, cell_size, device, potentials_manager):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.device = device
        self.pot_manager = potentials_manager
        
        # Grid allocations
        self.density_grid = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.phi = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.fx = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        self.fy = torch.zeros((self.grid_h, self.grid_w), device=self.device)
        
        # Precompute k_sq for FFT
        kx = torch.fft.fftfreq(self.grid_w, d=self.cell_size, device=self.device) * 2 * math.pi
        ky = torch.fft.fftfreq(self.grid_h, d=self.cell_size, device=self.device) * 2 * math.pi
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='xy')
        
        self.k_sq = kx_grid**2 + ky_grid**2
        self.k_sq[0, 0] = 1e-8
        
        self.K_k = None
        self.active_potential_type = None

    def update_kernel(self, potential_type, params):
        """
        Retrieves the updated Fourier kernel from potentials.py.
        """
        self.active_potential_type = potential_type
        self.K_k = self.pot_manager.get_fourier_kernel(potential_type, params)

    def solve(self, pos, vel, masses, num_active, potential_type, params, dt, friction, pressure_model, pressure_strength, gamma):
        """
        Runs Particle-Mesh solver:
        1. Deposits mass to grid.
        2. Solves Poisson equation (PyTorch GPU FFT or Taichi Vulkan GPU solver).
        3. Integrates forces and updates particles via Taichi kernels.
        """
        if num_active == 0:
            return
            
        # Re-fetch kernel if model parameters changed or type switched
        if self.K_k is None or self.active_potential_type != potential_type:
            self.update_kernel(potential_type, params)
            
        # Clear grid and scatter masses
        self.density_grid.zero_()
        scatter_mass_kernel(pos, masses, self.density_grid, num_active, self.grid_w, self.grid_h, float(self.cell_size))
        
        if self.device.type in ('cuda', 'xpu'):
            # PyTorch GPU FFT pipeline
            rho_k = torch.fft.rfft2(self.density_grid)
            K_k_sliced = self.K_k[:, :rho_k.shape[1]]
            phi_k = rho_k * K_k_sliced
            
            softening_length = params.get('softening_length', 1.5)
            k_sq_real = self.k_sq[:, :rho_k.shape[1]]
            gaussian = torch.exp(-k_sq_real * (softening_length**2) / 2.0)
            phi_k.mul_(gaussian)
            
            self.phi = torch.fft.irfft2(phi_k, s=self.density_grid.shape).real
            self.fy, self.fx = torch.gradient(self.phi)
            self.fx.neg_().div_(self.cell_size)
            self.fy.neg_().div_(self.cell_size)
        else:
            # Pure Taichi Vulkan GPU Poisson solver (Runs 100% on Intel GPU via Vulkan)
            self.phi.zero_()
            for _ in range(12):
                poisson_rbgs_kernel(self.phi, self.density_grid, self.grid_w, self.grid_h, float(self.cell_size), 0)
                poisson_rbgs_kernel(self.phi, self.density_grid, self.grid_w, self.grid_h, float(self.cell_size), 1)
            compute_forces_kernel(self.phi, self.fx, self.fy, self.grid_w, self.grid_h, float(self.cell_size))
        
        # Add real-space gas pressure if active
        if pressure_model != 'none':
            gamma_val = gamma if pressure_model == 'polytropic' else 1.0
            force_px, force_py = self.pot_manager.compute_pressure_forces(self.density_grid, pressure_strength, gamma_val)
            self.fx += force_px
            self.fy += force_py
            
        # Taichi force gathering and integration
        gather_and_integrate_kernel(
            pos, 
            vel, 
            self.fx, 
            self.fy, 
            num_active, 
            self.grid_w, 
            self.grid_h, 
            float(self.cell_size), 
            float(dt), 
            float(friction)
        )
