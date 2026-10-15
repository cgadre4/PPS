import torch
import pygame
import sys

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH, HEIGHT = 2880, 1500
TARGET_FPS = 120
PARTICLE_RADIUS = 2.0
MASS = 1.0
GRAVITY = 0
MAX_VELOCITY = 1E6
RESTITUTION = 1 # 1.0 = perfectly elastic
SUB_STEPS = 1


class ParticleSystem:
    def __init__(self, num_particles):
        self.device = DEVICE
        
        # Initialize random positions and velocities
        self.pos = torch.rand((num_particles, 2), device=self.device)
        self.pos[:, 0] *= WIDTH
        self.pos[:, 1] *= HEIGHT
        
        self.vel = (torch.rand((num_particles, 2), device=self.device) - 0.5) * 200.0 # Pixels per second
        
        # Variable properties
        self.radii = torch.full((num_particles,), PARTICLE_RADIUS, device=self.device)
        self.masses = torch.full((num_particles,), MASS, device=self.device)
        
    def add_particles(self, x, y, count=10, radius=PARTICLE_RADIUS, mass=MASS, initial_vel=None):
        """Add new particles at the specified mouse coordinates."""
        count = int(count)
        new_pos = torch.zeros((count, 2), device=self.device)
        new_pos[:, 0] = x
        new_pos[:, 1] = y
        # Add slight jitter so they don't spawn exactly on top of each other
        new_pos += (torch.rand((count, 2), device=self.device) - 0.5) * (radius * 0.5)
        
        if initial_vel is None:
            new_vel = (torch.rand((count, 2), device=self.device) - 0.5) * 200.0
        else:
            new_vel = torch.tensor(initial_vel, dtype=torch.float32, device=self.device).repeat(count, 1)
            
        new_radii = torch.full((count,), radius, device=self.device)
        new_masses = torch.full((count,), mass, device=self.device)
        
        self.pos = torch.cat([self.pos, new_pos])
        self.vel = torch.cat([self.vel, new_vel])
        self.radii = torch.cat([self.radii, new_radii])
        self.masses = torch.cat([self.masses, new_masses])

    def remove_particles(self, x, y, radius):
        """Remove particles within a certain radius of the mouse."""
        if len(self.pos) == 0:
            return
            
        mouse_pos = torch.tensor([x, y], dtype=torch.float32, device=self.device)
        distances = torch.norm(self.pos - mouse_pos, dim=1)
        
        # Keep particles that are OUTSIDE the deletion radius
        keep_mask = distances > radius
        self.pos = self.pos[keep_mask]
        self.vel = self.vel[keep_mask]
        self.radii = self.radii[keep_mask]
        self.masses = self.masses[keep_mask]

    def step(self, dt):
        if len(self.pos) == 0:
            return

        # 0. Apply Gravity (Add this line)
        self.vel[:, 1] += GRAVITY * dt

        # 1. Update positions based on velocity and delta time
        self.pos += self.vel * dt

        # 2. Wall Collisions
        hit_left = self.pos[:, 0] < self.radii
        hit_right = self.pos[:, 0] > WIDTH - self.radii
        self.pos[:, 0] = torch.where(hit_left, self.radii, self.pos[:, 0])
        self.pos[:, 0] = torch.where(hit_right, WIDTH - self.radii, self.pos[:, 0])
        self.vel[:, 0] = torch.where(hit_left | hit_right, -self.vel[:, 0] * RESTITUTION, self.vel[:, 0])

        hit_top = self.pos[:, 1] < self.radii
        hit_bottom = self.pos[:, 1] > HEIGHT - self.radii
        self.pos[:, 1] = torch.where(hit_top, self.radii, self.pos[:, 1])
        self.pos[:, 1] = torch.where(hit_bottom, HEIGHT - self.radii, self.pos[:, 1])
        self.vel[:, 1] = torch.where(hit_top | hit_bottom, -self.vel[:, 1] * RESTITUTION, self.vel[:, 1])

        # 3. Particle-Particle Collisions (O(N^2) Vectorized)
        N = len(self.pos)
        if N > 1:
            dist_matrix = torch.cdist(self.pos, self.pos)
            dist_matrix.fill_diagonal_(float('inf'))
            
            r_matrix = self.radii.unsqueeze(0) + self.radii.unsqueeze(1)
            
            mask = torch.triu(dist_matrix < r_matrix, diagonal=1)
            idx1, idx2 = torch.where(mask)
            
            if len(idx1) > 0:
                p1, p2 = self.pos[idx1], self.pos[idx2]
                v1, v2 = self.vel[idx1], self.vel[idx2]
                m1, m2 = self.masses[idx1].unsqueeze(1), self.masses[idx2].unsqueeze(1)
                
                # Normal vector
                n = p1 - p2
                n_norm = torch.norm(n, dim=1, keepdim=True)
                n = n / (n_norm + 1e-8) 
                
                # Relative velocity
                v_rel = v1 - v2
                v_n = torch.sum(v_rel * n, dim=1, keepdim=True)
                
                resolve_mask = (v_n < 0).squeeze()
                
                if resolve_mask.dim() == 0:
                    resolve_mask = resolve_mask.unsqueeze(0)
                
                if resolve_mask.any():
                    idx1 = idx1[resolve_mask]
                    idx2 = idx2[resolve_mask]
                    n = n[resolve_mask]
                    v_n = v_n[resolve_mask]
                    n_norm = n_norm[resolve_mask]
                    m1 = m1[resolve_mask]
                    m2 = m2[resolve_mask]
                    
                    # Impulse calculation using variable mass
                    j = -(1 + RESTITUTION) * v_n / (1/m1 + 1/m2)
                    
                    self.vel[idx1] += (j / m1) * n
                    self.vel[idx2] -= (j / m2) * n
                    
                    # Mass-weighted position correction
                    overlap = r_matrix[idx1, idx2].unsqueeze(1) - n_norm
                    total_mass = m1 + m2
                    
                    correction1 = overlap * (m2 / total_mass) * n
                    correction2 = overlap * (m1 / total_mass) * n
                    
                    self.pos[idx1] += correction1
                    self.pos[idx2] -= correction2

    def step_uniformgrid(self, dt):
        # 3. Particle-Particle Collisions (1D Vectorized Sweep & Prune)
        N = len(self.pos)
        if N > 1:
            # 1. Sort all arrays purely by the X coordinate
            sort_idx = torch.argsort(self.pos[:, 0])
            self.pos = self.pos[sort_idx]
            self.vel = self.vel[sort_idx]
            self.radii = self.radii[sort_idx]
            self.masses = self.masses[sort_idx]
            
            # 2. Check the next K neighbors to the right
            # K = 32 assumes we will rarely have more than 32 particles stacked 
            # perfectly vertically at the exact same X coordinate.
            K = 64 
            MAX_CHECK = min(K, N - 1)
            
            # This loop runs a fixed maximum of 32 times, regardless of particle count.
            for offset in range(1, MAX_CHECK + 1):
                p1 = self.pos[:-offset]
                p2 = self.pos[offset:]
                
                # --- Broad Phase (X-Axis only) ---
                x_dist = p2[:, 0] - p1[:, 0]
                r_sum = self.radii[:-offset] + self.radii[offset:]
                
                # If they are too far apart horizontally, skip heavy math
                mask = x_dist < r_sum
                
                if mask.any():
                    # Filter down to pairs that passed the X-check
                    idx1 = torch.where(mask)[0]
                    idx2 = idx1 + offset
                    
                    pos1, pos2 = self.pos[idx1], self.pos[idx2]
                    
                    # --- Narrow Phase (True 2D Distance) ---
                    n = pos1 - pos2
                    dist_sq = torch.sum(n**2, dim=1)
                    r_sum_sq = r_sum[mask]**2
                    
                    # True collision check (using squared distance to avoid early sqrt)
                    hit_mask = dist_sq < r_sum_sq
                    
                    if hit_mask.any():
                        # Isolate the exact indices of particles actively colliding
                        hit_idx1 = idx1[hit_mask]
                        hit_idx2 = idx2[hit_mask]
                        n_hit = n[hit_mask]
                        dist_sq_hit = dist_sq[hit_mask]
                        
                        # Calculate exact normal vector
                        n_norm = torch.sqrt(dist_sq_hit).unsqueeze(1)
                        n_dir = n_hit / (n_norm + 1e-8)
                        
                        v1, v2 = self.vel[hit_idx1], self.vel[hit_idx2]
                        m1, m2 = self.masses[hit_idx1].unsqueeze(1), self.masses[hit_idx2].unsqueeze(1)
                        
                        v_rel = v1 - v2
                        v_n = torch.sum(v_rel * n_dir, dim=1, keepdim=True)
                        
                        # Only resolve if moving towards each other
                        resolve_mask = (v_n < 0).squeeze()
                        if resolve_mask.dim() == 0: resolve_mask = resolve_mask.unsqueeze(0)
                        
                        if resolve_mask.any():
                            # Final resolution targets
                            f_idx1 = hit_idx1[resolve_mask]
                            f_idx2 = hit_idx2[resolve_mask]
                            n_f = n_dir[resolve_mask]
                            v_n_f = v_n[resolve_mask]
                            n_norm_f = n_norm[resolve_mask]
                            m1_f = m1[resolve_mask]
                            m2_f = m2[resolve_mask]
                            
                            # Impulse Calculation
                            j = -(1 + RESTITUTION) * v_n_f / (1/m1_f + 1/m2_f)
                            
                            self.vel[f_idx1] += (j / m1_f) * n_f
                            self.vel[f_idx2] -= (j / m2_f) * n_f
                            
                            # Position Correction
                            overlap = (self.radii[f_idx1].unsqueeze(1) + self.radii[f_idx2].unsqueeze(1)) - n_norm_f
                            total_mass = m1_f + m2_f
                            
                            self.pos[f_idx1] += overlap * (m2_f / total_mass) * n_f
                            self.pos[f_idx2] -= overlap * (m1_f / total_mass) * n_f

        # Clamp velocity
        self.vel = torch.clamp(self.vel, min=-MAX_VELOCITY, max=MAX_VELOCITY)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PyTorch GPU Particle Physics")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    ui_font = pygame.font.SysFont("Arial", 18)

    system = ParticleSystem(num_particles=10800)
    interaction_radius = 50.0
    
    # UI Setup
    btn_default = pygame.Rect(10, 50, 140, 30)
    btn_giant = pygame.Rect(160, 50, 140, 30)
    spawn_mode = "DEFAULT" # Can be "DEFAULT" or "GIANT"
    
    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 500.0
        if dt > 0.1: dt = 0.1 

        mouse_x, mouse_y = pygame.mouse.get_pos()
        over_ui = btn_default.collidepoint(mouse_x, mouse_y) or btn_giant.collidepoint(mouse_x, mouse_y)

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4: # Scroll Up
                    interaction_radius += 10
                elif event.button == 5: # Scroll Down
                    interaction_radius = max(10, interaction_radius - 10)
                elif event.button == 1: # Left click (once)
                    if btn_default.collidepoint(mouse_x, mouse_y):
                        spawn_mode = "DEFAULT"
                    elif btn_giant.collidepoint(mouse_x, mouse_y):
                        spawn_mode = "GIANT"
                    elif spawn_mode == "GIANT":
                        # Spawn 1 giant particle. Calculate mass to maintain constant density.
                        # Density = Mass / Area. (Mass_default / R_default^2) = (Mass_giant / R_giant^2)
                        area_ratio = (interaction_radius**2) / (PARTICLE_RADIUS**2)
                        giant_mass = MASS * area_ratio
                        system.add_particles(
                            mouse_x, mouse_y, 
                            count=1, 
                            radius=interaction_radius, 
                            mass=giant_mass, 
                            initial_vel=[0.0, 0.0]
                        )
        
        # Continuous mouse presses
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] and spawn_mode == "DEFAULT" and not over_ui: 
            system.add_particles(mouse_x, mouse_y, count=interaction_radius/10)
        if mouse_buttons[2]: # Right Click - Delete
            system.remove_particles(mouse_x, mouse_y, interaction_radius)

        # --- Physics Update ---
        sub_dt = dt / SUB_STEPS
        
        for _ in range(SUB_STEPS):
            system.step(sub_dt)

        # --- Rendering ---
        screen.fill((15, 15, 20))
        
        if mouse_buttons[2]:
            pygame.draw.circle(screen, (100, 30, 30), (mouse_x, mouse_y), int(interaction_radius), 5)
        elif mouse_buttons[0] and not over_ui:
            pygame.draw.circle(screen, (30, 100, 30), (mouse_x, mouse_y), int(interaction_radius), 5)
        elif not over_ui:
            pygame.draw.circle(screen, (50, 50, 50), (mouse_x, mouse_y), int(interaction_radius), 5)

        if len(system.pos) > 0:
            positions_cpu = system.pos.cpu().numpy()
            radii_cpu = system.radii.cpu().numpy()
            for i in range(len(positions_cpu)):
                x, y = positions_cpu[i]
                r = radii_cpu[i]
                pygame.draw.circle(screen, (100, 200, 255), (int(x), int(y)), int(r))

        # --- HUD & UI ---
        fps = clock.get_fps()
        count = len(system.pos)
        
        hud_text = font.render(f"FPS: {fps:.1f} | Particles: {count}", True, (255, 255, 255))
        screen.blit(hud_text, (10, 10))

        # Draw Buttons
        pygame.draw.rect(screen, (60, 120, 60) if spawn_mode == "DEFAULT" else (60, 60, 60), btn_default)
        pygame.draw.rect(screen, (60, 120, 60) if spawn_mode == "GIANT" else (60, 60, 60), btn_giant)
        pygame.draw.rect(screen, (200, 200, 200), btn_default, 2)
        pygame.draw.rect(screen, (200, 200, 200), btn_giant, 2)
        
        txt_default = ui_font.render("Default Spawner", True, (255, 255, 255))
        txt_giant = ui_font.render("Giant Spawner", True, (255, 255, 255))
        screen.blit(txt_default, (btn_default.x + 10, btn_default.y + 5))
        screen.blit(txt_giant, (btn_giant.x + 20, btn_giant.y + 5))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()