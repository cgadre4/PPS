import torch
import math

class PotentialsManager:
    def __init__(self, grid_w, grid_h, cell_size, device):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell_size = cell_size
        self.device = device
        
        # Precompute kx, ky grids for Fourier-space gravity Poisson solver
        kx = torch.fft.fftfreq(grid_w, d=cell_size, device=device) * 2 * math.pi
        ky = torch.fft.fftfreq(grid_h, d=cell_size, device=device) * 2 * math.pi
        self.kx_grid, self.ky_grid = torch.meshgrid(kx, ky, indexing='xy')
        self.k_sq = self.kx_grid**2 + self.ky_grid**2
        
        # Precompute real-space distance grid self.r for real-space potentials (LJ, Harmonic)
        dx = torch.arange(grid_w, device=device, dtype=torch.float32)
        dy = torch.arange(grid_h, device=device, dtype=torch.float32)
        
        # Periodic distance mapping centered at index (0,0) to prevent spatial shifting
        dx = torch.where(dx > grid_w / 2.0, dx - grid_w, dx) * cell_size
        dy = torch.where(dy > grid_h / 2.0, dy - grid_h, dy) * cell_size
        yy, xx = torch.meshgrid(dy, dx, indexing='ij')
        self.r = torch.sqrt(xx**2 + yy**2)

    def get_fourier_kernel(self, potential_type, params):
        """
        Generates the 2D Fourier-space interaction kernel K_k for the selected potential.
        
        potential_type: 'gravity', 'lennard_jones', 'harmonic'
        params: dict containing sliders (G, epsilon, sigma, spring_k, force_range)
        """
        force_range = params.get('force_range', 1000.0)
        
        if potential_type == 'gravity':
            G = params.get('G', 1.0)
            # Screened Yukawa gravity in Fourier space: lambda_sq = (1 / force_range)^2
            lambda_sq = (1.0 / force_range)**2
            denom = self.k_sq + lambda_sq
            denom_safe = denom.clone()
            
            # Prevent division by zero at k = (0,0) if there's no screening
            if lambda_sq < 1e-8:
                denom_safe[0, 0] = 1e-8
                
            K_k = -G / denom_safe
            
        elif potential_type == 'harmonic':
            spring_k = params.get('spring_k', 1.0)
            # Mutual spring potential: V(r) = 0.5 * spring_k * r^2
            V = 0.5 * spring_k * (self.r ** 2)
            
            # Universal force-range screening prevents boundary periodic jumps
            V = V * torch.exp(-self.r / force_range)
            K_k = torch.fft.fft2(V)
            
        elif potential_type == 'lennard_jones':
            epsilon = params.get('epsilon', 100.0)
            sigma = params.get('sigma', 10.0)
            
            # Soften grid center to avoid unphysical infinite repulsion at r=0
            r_soft = torch.clamp(self.r, min=self.cell_size * 0.5)
            
            # V(r) = 4 * eps * ((sigma/r)^12 - (sigma/r)^6)
            V = 4.0 * epsilon * ((sigma / r_soft)**12 - (sigma / r_soft)**6)
            V = torch.clamp(V, min=-500000.0, max=500000.0)
            
            # Universal force-range screening smooths outer cutoff
            V = V * torch.exp(-self.r / force_range)
            K_k = torch.fft.fft2(V)
            
        elif potential_type == 'coulomb':
            ke = params.get('G', 1.0)
            # Screened Yukawa-like Coulomb in Fourier space: lambda_sq = (1 / force_range)^2
            lambda_sq = (1.0 / force_range)**2
            denom = self.k_sq + lambda_sq
            denom_safe = denom.clone()
            
            # Prevent division by zero at k = (0,0) if there's no screening
            if lambda_sq < 1e-8:
                denom_safe[0, 0] = 1e-8
                
            # Repelling force for like charges: positive sign
            K_k = ke / denom_safe

        else:
            raise ValueError(f"Unknown potential type: {potential_type}")

        # Zero the k = 0 (mean) mode. It only adds a constant to phi, which exerts no force, but for gravity/Coulomb
        # it equals -/+G * M_total / lambda^2: large enough to destroy float32 precision in phi (and therefore in
        # the finite-difference forces) and to wash out the potential view.
        K_k[0, 0] = 0.0
        return K_k

    def compute_pressure_forces(self, density, pressure_strength, gamma):
        """
        Computes polytropic gas pressure forces in real space: F = -P * grad(rho^gamma).
        
        density: 2D tensor of mass density on grid
        pressure_strength: scaling constant P
        gamma: polytropic index (1.0 = isothermal, 1.4 = air, 1.67 = monatomic gas)
        """
        if pressure_strength <= 1e-5:
            return torch.zeros_like(density), torch.zeros_like(density)
            
        # Polytropic density relation
        rho_poly = density ** gamma
        
        # Calculate real-space spatial gradients
        grad_py, grad_px = torch.gradient(rho_poly)
        
        # Convert grid step gradient to physical scale
        force_px = -pressure_strength * grad_px / self.cell_size
        force_py = -pressure_strength * grad_py / self.cell_size
        
        return force_px, force_py

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
