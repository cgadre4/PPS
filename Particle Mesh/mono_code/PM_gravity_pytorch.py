import torch
import pygame
import sys
import math
import numpy as np

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Screen & Camera limits
WIDTH, HEIGHT = 2560, 1440
TARGET_FPS = 0

# --- Global User Variables ---
SIM_SIZE = 2.0          # 1.0 = exact screen size, 2.0 = universe is twice as wide/tall
MIN_GRAVITY = 0.01
MAX_GRAVITY = 100.0

# --- Universe & Grid Calculations ---
CELL_SIZE = 1.0 
UNIVERSE_W = int(WIDTH * SIM_SIZE)
UNIVERSE_H = int(HEIGHT * SIM_SIZE)

GRID_W = int(UNIVERSE_W / CELL_SIZE)
GRID_H = int(UNIVERSE_H / CELL_SIZE)

# Camera offsets to keep the screen dead-center in the expanded universe
CAM_X = (UNIVERSE_W - WIDTH) / 2.0
CAM_Y = (UNIVERSE_H - HEIGHT) / 2.0

# Visual Density Limits (Log10 scale)
MIN_LOG_DENSITY = 0.0  
MAX_LOG_DENSITY = 1.0 / CELL_SIZE 

class FFTGravitySystem:
    def __init__(self, num_particles):
        self.device = DEVICE
        self.G = MAX_GRAVITY / 10 # Start at max slider value
        
        self.pos = torch.rand((num_particles, 2), device=self.device)
        # Spawn initial particles randomly across the ENTIRE universe, not just the screen
        self.pos[:, 0] *= UNIVERSE_W
        self.pos[:, 1] *= UNIVERSE_H
        
        self.vel = (torch.rand((num_particles, 2), device=self.device) - 0.5) * 10.0
        self.masses = torch.ones(num_particles, device=self.device)
        
        self.colors = torch.full((num_particles, 3), 255, dtype=torch.float32, device=self.device)
        
        # User Custom 7-Color Thermal Map
        self.color_map = torch.tensor([
            [0, 0, 128],     # Dark Blue
            [0, 0, 255],     # Blue
            [255, 0, 128],   # Pink/Magenta
            [255, 0, 0],     # Red
            [255, 128, 0],   # Orange
            [255, 255, 0],   # Yellow
            [255, 255, 255]  # White
        ], dtype=torch.float32, device=self.device)
        
        # Precompute the Fourier Space grid
        kx = torch.fft.fftfreq(GRID_W, d=CELL_SIZE, device=self.device) * 2 * math.pi
        ky = torch.fft.fftfreq(GRID_H, d=CELL_SIZE, device=self.device) * 2 * math.pi
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='xy')
        
        # Poisson's Equation denominator
        self.k_sq = kx_grid**2 + ky_grid**2
        self.k_sq[0, 0] = 1e-8 # Prevent divide-by-zero

    def add_galaxy(self, mouse_x, mouse_y, radius, count=500):
        # Convert screen click to true universe coordinates
        center_x = mouse_x + CAM_X
        center_y = mouse_y + CAM_Y
        
        count = int(count)
        r = torch.rand(count, device=self.device) * radius
        theta = torch.rand(count, device=self.device) * 2 * math.pi
        
        new_pos = torch.zeros((count, 2), device=self.device)
        new_pos[:, 0] = center_x + r * torch.cos(theta)
        new_pos[:, 1] = center_y + r * torch.sin(theta)
        
        # Orbital velocity math
        orbital_speed = torch.sqrt(abs(self.G) * 100.0 / (r + 10.0))
        new_vel = torch.zeros((count, 2), device=self.device)
        new_vel[:, 0] = -torch.sin(theta) * orbital_speed
        new_vel[:, 1] = torch.cos(theta) * orbital_speed
        
        new_masses = torch.ones(count, device=self.device)
        new_colors = torch.full((count, 3), 255, dtype=torch.float32, device=self.device)
        
        self.pos = torch.cat([self.pos, new_pos])
        self.vel = torch.cat([self.vel, new_vel])
        self.masses = torch.cat([self.masses, new_masses])
        self.colors = torch.cat([self.colors, new_colors])

    def remove_particles(self, mouse_x, mouse_y, radius):
        if len(self.pos) == 0: return
        # Convert screen click to true universe coordinates
        uni_pos = torch.tensor([mouse_x + CAM_X, mouse_y + CAM_Y], dtype=torch.float32, device=self.device)
        
        distances = torch.norm(self.pos - uni_pos, dim=1)
        keep_mask = distances > radius
        self.pos = self.pos[keep_mask]
        self.vel = self.vel[keep_mask]
        self.masses = self.masses[keep_mask]
        self.colors = self.colors[keep_mask]

    def step(self, dt):
        if len(self.pos) == 0: return

        # 1. Update positions
        self.pos += self.vel * dt

        # 2. Delete Out-of-Universe Particles
        in_bounds = (self.pos[:, 0] >= 0) & (self.pos[:, 0] < UNIVERSE_W) & \
                    (self.pos[:, 1] >= 0) & (self.pos[:, 1] < UNIVERSE_H)
        
        self.pos = self.pos[in_bounds]
        self.vel = self.vel[in_bounds]
        self.masses = self.masses[in_bounds]
        self.colors = self.colors[in_bounds]

        if len(self.pos) == 0: return

        # --- PM FFT ALGORITHM ---
        grid_pos = (self.pos / CELL_SIZE).long()
        indices = grid_pos[:, 1] * GRID_W + grid_pos[:, 0]
        
        density = torch.bincount(indices, weights=self.masses, minlength=GRID_W*GRID_H)
        density = density.view(GRID_H, GRID_W)
        
        # --- VECTORIZED PARTICLE COLORING ---
        particle_densities = density[grid_pos[:, 1], grid_pos[:, 0]]
        log_dens = torch.log10(particle_densities + 0.1)
        norm_dens = torch.clamp((log_dens - MIN_LOG_DENSITY) / (MAX_LOG_DENSITY - MIN_LOG_DENSITY), 0.0, 1.0)
        
        # Multiply by 6.0 because there are 6 gaps between the 7 colors
        idx = norm_dens * 6.0
        idx_lower = torch.floor(idx).long()
        idx_upper = torch.ceil(idx).long()
        idx_upper = torch.clamp(idx_upper, 0, 6) 
        
        weight = (idx - idx_lower).unsqueeze(1)
        self.colors = self.color_map[idx_lower] * (1.0 - weight) + self.color_map[idx_upper] * weight
        
        # --- GRAVITY MATH ---
        rho_k = torch.fft.fft2(density)
        phi_k = -self.G * rho_k / self.k_sq
        phi = torch.fft.ifft2(phi_k).real
        
        grad_y, grad_x = torch.gradient(phi)
        
        force_x = -grad_x[grid_pos[:, 1], grid_pos[:, 0]]
        force_y = -grad_y[grid_pos[:, 1], grid_pos[:, 0]]
        
        self.vel[:, 0] += force_x * dt
        self.vel[:, 1] += force_y * dt


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PyTorch Particle-Mesh Gravity Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    system = FFTGravitySystem(num_particles=20000)
    interaction_radius = 100.0
    
    # Slider Setup
    slider_rect = pygame.Rect(10, 45, 200, 20)
    slider_dragging = False
    
    # Pre-calculate log ranges for the slider math
    log_min = math.log10(MIN_GRAVITY)
    log_max = math.log10(MAX_GRAVITY)
    log_range = log_max - log_min
    
    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0
        if dt > 0.05: dt = 0.05 

        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4: # Scroll Up
                    interaction_radius += 20
                elif event.button == 5: # Scroll Down
                    interaction_radius = max(20, interaction_radius - 20)
                elif event.button == 1: # Left Click
                    if slider_rect.collidepoint(event.pos):
                        slider_dragging = True
                    else:
                        system.add_galaxy(mouse_x, mouse_y, interaction_radius, count=interaction_radius*5)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    slider_dragging = False

        # --- LOG SLIDER MATH ---
        if slider_dragging:
            t = (mouse_x - slider_rect.x) / slider_rect.width
            t = max(0.0, min(1.0, t))
            system.G = 10 ** (log_min + (log_range * t))
        
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[2]: # Right Click
            system.remove_particles(mouse_x, mouse_y, interaction_radius)

        system.step(dt)

        # --- RENDERING ---
        screen.fill((5, 5, 8)) # Very dark background to make colors pop
            
        if len(system.pos) > 0:
            positions_cpu = system.pos.cpu().numpy()
            colors_cpu = system.colors.cpu().numpy().astype(int)
            
            # Translate true universe coordinates to screen space for the camera
            screen_pos = positions_cpu - np.array([CAM_X, CAM_Y])
            
            # Filter out particles that are off-screen to save Pygame CPU cycles
            mask = (screen_pos[:, 0] >= 0) & (screen_pos[:, 0] < WIDTH) & \
                   (screen_pos[:, 1] >= 0) & (screen_pos[:, 1] < HEIGHT)
            
            visible_pos = screen_pos[mask].astype(int)
            visible_colors = colors_cpu[mask]

            for i in range(len(visible_pos)):
                pygame.draw.circle(screen, visible_colors[i], visible_pos[i], 1)

        # Draw Interaction Brushes
        if mouse_buttons[2]:
            pygame.draw.circle(screen, (255, 50, 50), (mouse_x, mouse_y), int(interaction_radius), 2)
        elif not slider_dragging:
            pygame.draw.circle(screen, (50, 200, 150), (mouse_x, mouse_y), int(interaction_radius), 2)

        # HUD Text
        fps = clock.get_fps()
        count = len(system.pos)
        
        if len(system.pos) > 0:
            visible_count = len(visible_pos)
        else:
            visible_count = 0
            
        hud_text = font.render(f"FPS: {fps:.1f} | Total Stars: {count} | Visible: {visible_count}", True, (255, 255, 255))
        screen.blit(hud_text, (10, 10))

        # Slider rendering
        pygame.draw.rect(screen, (80, 80, 100), slider_rect, border_radius=10)
        current_t = (math.log10(system.G) - log_min) / log_range
        handle_x = slider_rect.x + int(current_t * slider_rect.width)
        
        pygame.draw.circle(screen, (200, 220, 255), (handle_x, slider_rect.centery), 12)
        g_text_surf = font.render(f"G: {system.G:.4f}", True, (255, 255, 255))
        screen.blit(g_text_surf, (slider_rect.right + 15, slider_rect.y - 3))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()