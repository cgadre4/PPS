import torch
import taichi as ti

@ti.kernel
def nbody_taichi_solve_kernel(
    pos: ti.types.ndarray(),
    mass: ti.types.ndarray(),
    ax: ti.types.ndarray(),
    ay: ti.types.ndarray(),
    num_active: ti.i32,
    G: ti.f32,
    softening: ti.f32
):
    eps2 = softening * softening
    for p in range(num_active):
        px, py = pos[p, 0], pos[p, 1]
        accum_x, accum_y = 0.0, 0.0
        for o in range(num_active):
            if p != o:
                dx = pos[o, 0] - px
                dy = pos[o, 1] - py
                r2 = dx * dx + dy * dy + eps2
                inv_r3 = 1.0 / (r2 * ti.sqrt(r2))
                m = ti.cast(mass[o], ti.f32)
                accum_x += G * m * dx * inv_r3
                accum_y += G * m * dy * inv_r3
        ax[p] = accum_x
        ay[p] = accum_y


class DirectNBodySolver:
    def __init__(self, device):
        self.device = device

    def solve(self, pos, masses, potential_type, params):
        """
        Computes exact O(N^2) pairwise accelerations.
        """
        N = len(pos)
        if N <= 1:
            return torch.zeros(N, device=self.device), torch.zeros(N, device=self.device)
            
        if self.device.type not in ('cuda', 'xpu') and potential_type == 'gravity':
            # Pure Taichi Vulkan GPU N-body solver (Runs 100% on Intel GPU via Vulkan)
            ax = torch.zeros(N, device=self.device)
            ay = torch.zeros(N, device=self.device)
            G = float(params.get('G', 1.0))
            softening = float(params.get('softening_length', 1.5))
            nbody_taichi_solve_kernel(pos, masses, ax, ay, N, G, softening)
            return ax, ay
            
        # Compute pairwise displacement vectors diff[i, j] = pos[j] - pos[i]
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dx = diff[:, :, 0]
        dy = diff[:, :, 1]
        
        # Compute distance matrix with force softening (Plummer)
        r_sq = dx**2 + dy**2
        epsilon_soft = params.get('softening_length', 1.5)
        r_soft_sq = r_sq + epsilon_soft**2
        r = torch.sqrt(r_soft_sq)
        
        # Exponential Force range screening
        force_range = params.get('force_range', 8192.0)
        screening = torch.exp(-r / force_range)
        
        # Set diagonal to infinity to ignore self-interactions
        r.fill_diagonal_(float('inf'))
        
        ax = torch.zeros(N, device=self.device)
        ay = torch.zeros(N, device=self.device)
        
        if potential_type == 'gravity':
            G = params.get('G', 1.0)
            inv_r_sq = 1.0 / r_soft_sq
            inv_r_sq.fill_diagonal_(0.0)
            
            # a_i = sum_j G * m_j * diff_ij / (r_ij^2 + softening^2)
            factor = G * masses.unsqueeze(0) * inv_r_sq * screening
            ax = torch.sum(factor * dx, dim=1)
            ay = torch.sum(factor * dy, dim=1)
            
        elif potential_type == 'harmonic':
            spring_k = params.get('spring_k', 0.1)
            # F_ij = -k * (pos_i - pos_j) / N (normalized by total particles for scale safety)
            factor = (spring_k / N) * masses.unsqueeze(0) * screening
            ax = torch.sum(factor * dx, dim=1)
            ay = torch.sum(factor * dy, dim=1)
            
        elif potential_type == 'lennard_jones':
            epsilon = params.get('epsilon', 1000.0)
            sigma = params.get('sigma', 40.0)
            
            sigma_over_r = sigma / r
            # Force magnitude: F(r) = 24 * eps / r * (2 * (sigma/r)^12 - (sigma/r)^6)
            force_mag = (24.0 * epsilon / r) * (2.0 * (sigma_over_r ** 12) - (sigma_over_r ** 6))
            force_mag.fill_diagonal_(0.0)
            
            # Clamp force magnitude to prevent unphysical velocity spikes
            force_mag = torch.clamp(force_mag, min=-500000.0, max=500000.0)
            
            inv_r = 1.0 / r
            inv_r.fill_diagonal_(0.0)
            
            factor = force_mag * inv_r * masses.unsqueeze(0) * screening
            ax = torch.sum(factor * dx, dim=1)
            ay = torch.sum(factor * dy, dim=1)
            
        elif potential_type == 'coulomb':
            G = params.get('G', 1.0)
            inv_r_sq = 1.0 / r_soft_sq
            inv_r_sq.fill_diagonal_(0.0)
            # Coulomb repulsive for like charges, attractive for opposite.
            # Put a negative sign relative to attractive gravity.
            factor = -G * masses.unsqueeze(0) * inv_r_sq * screening
            ax = masses * torch.sum(factor * dx, dim=1)
            ay = masses * torch.sum(factor * dy, dim=1)
            
        return ax, ay
