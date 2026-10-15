import taichi as ti
import torch
import math
import time

# Disable VSYNC so we can manually control the framerate without VSYNC cliffs
ti.init(arch=ti.cuda)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration ---
WIDTH, HEIGHT = 3840, 2160
FPS_TARGET = 120.0

# --- Global User Variables ---
SIM_SIZE = 1.0          # Confined to screen size for fluid-in-a-box physics
RESTITUTION = 0.5       # Bounciness of the walls

# --- THE DECOUPLED GRIDS ---
# PHYSICS: Keeps the FFT incredibly fast.
PHYSICS_CELL_SIZE = 1.0

# RENDER: Determines pixel blockiness. 1.0 = Sharp 4K pixels. 4.0 = Retro blocky pixels.
RENDER_CELL_SIZE = 1.0

# --- Physics Grid Calculations ---
UNIVERSE_W = int(WIDTH * SIM_SIZE)
UNIVERSE_H = int(HEIGHT * SIM_SIZE)
GRID_W = int(UNIVERSE_W / PHYSICS_CELL_SIZE)
GRID_H = int(UNIVERSE_H / PHYSICS_CELL_SIZE)

# --- Render Grid Calculations ---
VIEW_W = int(WIDTH / RENDER_CELL_SIZE)
VIEW_H = int(HEIGHT / RENDER_CELL_SIZE)

# Camera offsets to keep the screen dead-center
CAM_X = (UNIVERSE_W - WIDTH) / 2.0
CAM_Y = (UNIVERSE_H - HEIGHT) / 2.0

# Visual Density Limits
MIN_LOG_DENSITY = 0.0  
MAX_LOG_DENSITY = 1.0 / PHYSICS_CELL_SIZE

# --- UI Lock Coordinates (Top-Left Origin) ---
UI_X, UI_Y = 0.01, 0.02
UI_W, UI_H = 0.25, 0.30

# Convert UI bounds to Bottom-Left Origin for precise mouse checking
MOUSE_SAFE_X_MIN = UI_X
MOUSE_SAFE_X_MAX = UI_X + UI_W
MOUSE_SAFE_Y_MIN = 1.0 - (UI_Y + UI_H)
MOUSE_SAFE_Y_MAX = 1.0 - UI_Y

# --- TAICHI PIXEL RENDER FIELDS ---
# Internal float buffer for safe atomic accumulation (prevents 8-bit overflow/wrapping)
render_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(VIEW_W, VIEW_H))

# 8-bit presentation buffer to completely eliminate the Vulkan bandwidth bottleneck
display_pixels = ti.Vector.field(3, dtype=ti.u8, shape=(VIEW_W, VIEW_H))
ring_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=64)

@ti.kernel
def render_to_pixels(
    pos_arr: ti.types.ndarray(),
    color_arr: ti.types.ndarray(),
    num_particles: ti.i32,
    cam_x: ti.f32,
    cam_y: ti.f32,
    render_scale: ti.f32
):
    # Clear the internal float screen instantly
    for i, j in render_pixels:
        render_pixels[i, j] = [0.0, 0.0, 0.0]

    # Compute Shader mapping particles to the decoupled render grid
    for i in range(num_particles):
        sx = (pos_arr[i, 0] - cam_x) / render_scale
        sy = (pos_arr[i, 1] - cam_y) / render_scale
        
        px = ti.cast(sx, ti.i32)
        py = ti.cast(sy, ti.i32)
        
        if 0 <= px < VIEW_W and 0 <= py < VIEW_H:
            c = ti.Vector([color_arr[i, 0] / 255.0,
                           color_arr[i, 1] / 255.0,
                           color_arr[i, 2] / 255.0])
            
            # Additive blending for gorgeous dense thermodynamic clusters
            render_pixels[px, py] += c * 0.8

@ti.kernel
def prepare_display():
    # Convert the internal float accumulation to an 8-bit display payload
    for i, j in render_pixels:
        # Scale back to 0-255, clamp to prevent wrapping, and cast to 8-bit integer
        r = ti.cast(ti.min(render_pixels[i, j][0] * 255.0, 255.0), ti.u8)
        g = ti.cast(ti.min(render_pixels[i, j][1] * 255.0, 255.0), ti.u8)
        b = ti.cast(ti.min(render_pixels[i, j][2] * 255.0, 255.0), ti.u8)
        display_pixels[i, j] = [r, g, b]

@ti.kernel
def update_brush_ring(center_x: ti.f32, center_y: ti.f32, radius_x: ti.f32, radius_y: ti.f32):
    for i in range(64):
        angle = float(i) / 64.0 * 2.0 * 3.14159265
        ring_pos_field[i] = [center_x + ti.cos(angle) * radius_x, center_y + ti.sin(angle) * radius_y]

class LennardJonesSystem:
    def __init__(self, num_particles):
        self.device = DEVICE
        
        # Lennard-Jones Parameters
        self.epsilon = 1000.0  # Depth of the potential well (Attraction strength)
        self.sigma = 40.0      # Distance at which intermolecular potential is zero (Size)
        self.global_gravity = 500.0 # Downward pull
        
        self.pos = torch.rand((num_particles, 2), device=self.device)
        self.pos[:, 0] = (self.pos[:, 0] * WIDTH) + CAM_X
        self.pos[:, 1] = (self.pos[:, 1] * HEIGHT) + CAM_Y
        
        self.vel = (torch.rand((num_particles, 2), device=self.device) - 0.5) * 10.0
        self.masses = torch.ones(num_particles, device=self.device)
        self.colors = torch.full((num_particles, 3), 255, dtype=torch.float32, device=self.device)
        
        # User Custom 7-Color Thermal Map
        self.color_map = torch.tensor([
            [0, 0, 128],    # Dark Blue
            [0, 0, 255],    # Blue
            [255, 0, 128],  # Pink/Magenta
            [255, 0, 0],    # Red
            [255, 128, 0],  # Orange
            [255, 255, 0],  # Yellow
            [255, 255, 255] # White
        ], dtype=torch.float32, device=self.device)
        
        self.precompute_LJ()

    def precompute_LJ(self):
        """
        Creates the real-space Lennard-Jones potential on a grid and runs 
        an FFT on it so we can multiply it instantly during the simulation loop.
        """
        dx = torch.arange(GRID_W, device=self.device, dtype=torch.float32)
        dy = torch.arange(GRID_H, device=self.device, dtype=torch.float32)
        
        dx = torch.where(dx > GRID_W/2, dx - GRID_W, dx) * PHYSICS_CELL_SIZE
        dy = torch.where(dy > GRID_H/2, dy - GRID_H, dy) * PHYSICS_CELL_SIZE
        yy, xx = torch.meshgrid(dy, dx, indexing='ij')
        
        r = torch.sqrt(xx**2 + yy**2)
        # Soften the center to avoid mathematically infinite repulsion
        r = torch.clamp(r, min=PHYSICS_CELL_SIZE * 0.5) 
        
        # V(r) = 4 * eps * ((sigma/r)^12 - (sigma/r)^6)
        V = 4.0 * self.epsilon * ((self.sigma / r)**12 - (self.sigma / r)**6)
        V = torch.clamp(V, min=-500000.0, max=500000.0)
        
        self.LJ_k = torch.fft.fft2(V)

    def add_cluster(self, mouse_x, mouse_y, radius, count=5000):
        center_x = mouse_x + CAM_X
        center_y = mouse_y + CAM_Y
        
        count = int(count)
        if count <= 0: return
        
        r = torch.rand(count, device=self.device) * radius
        theta = torch.rand(count, device=self.device) * 2 * math.pi
        
        new_pos = torch.zeros((count, 2), device=self.device)
        new_pos[:, 0] = center_x + r * torch.cos(theta)
        new_pos[:, 1] = center_y + r * torch.sin(theta)
        
        new_vel = (torch.rand((count, 2), device=self.device) - 0.5) * 50.0
        
        new_masses = torch.ones(count, device=self.device)
        new_colors = torch.full((count, 3), 255, dtype=torch.float32, device=self.device)
        
        self.pos = torch.cat([self.pos, new_pos])
        self.vel = torch.cat([self.vel, new_vel])
        self.masses = torch.cat([self.masses, new_masses])
        self.colors = torch.cat([self.colors, new_colors])

    def remove_particles(self, mouse_x, mouse_y, radius):
        if len(self.pos) == 0: return
        uni_pos = torch.tensor([mouse_x + CAM_X, mouse_y + CAM_Y], dtype=torch.float32, device=self.device)
        distances = torch.norm(self.pos - uni_pos, dim=1)
        keep_mask = distances > radius
        
        self.pos = self.pos[keep_mask]
        self.vel = self.vel[keep_mask]
        self.masses = self.masses[keep_mask]
        self.colors = self.colors[keep_mask]

    def step(self, dt):
        if len(self.pos) == 0: return

        # 1. Update positions & apply Global Gravity
        self.vel[:, 1] -= self.global_gravity * dt
        self.pos += self.vel * dt

        # 2. Wall Collisions (Screen Boundaries)
        hit_left = self.pos[:, 0] < 0
        hit_right = self.pos[:, 0] > UNIVERSE_W
        hit_top = self.pos[:, 1] < 0
        hit_bottom = self.pos[:, 1] > UNIVERSE_H
        
        self.pos[:, 0] = torch.where(hit_left, 0.0, self.pos[:, 0])
        self.pos[:, 0] = torch.where(hit_right, float(UNIVERSE_W), self.pos[:, 0])
        self.vel[:, 0] = torch.where(hit_left | hit_right, -self.vel[:, 0] * RESTITUTION, self.vel[:, 0])

        self.pos[:, 1] = torch.where(hit_top, 0.0, self.pos[:, 1])
        self.pos[:, 1] = torch.where(hit_bottom, float(UNIVERSE_H), self.pos[:, 1])
        self.vel[:, 1] = torch.where(hit_top | hit_bottom, -self.vel[:, 1] * RESTITUTION, self.vel[:, 1])

        # --- PM FFT ALGORITHM ---
        grid_pos = (self.pos / PHYSICS_CELL_SIZE).long()
        grid_pos[:, 0] = torch.clamp(grid_pos[:, 0], 0, GRID_W - 1)
        grid_pos[:, 1] = torch.clamp(grid_pos[:, 1], 0, GRID_H - 1)
        
        indices = grid_pos[:, 1] * GRID_W + grid_pos[:, 0]
        
        density = torch.bincount(indices, weights=self.masses, minlength=GRID_W*GRID_H)
        density = density.view(GRID_H, GRID_W)
        
        # --- VECTORIZED PARTICLE COLORING ---
        particle_densities = density[grid_pos[:, 1], grid_pos[:, 0]]
        log_dens = torch.log10(particle_densities + 0.1)
        norm_dens = torch.clamp((log_dens - MIN_LOG_DENSITY) / (MAX_LOG_DENSITY - MIN_LOG_DENSITY), 0.0, 1.0)
        
        idx = norm_dens * 6.0
        idx_lower = torch.floor(idx).long()
        idx_upper = torch.ceil(idx).long()
        idx_upper = torch.clamp(idx_upper, 0, 6)
        
        weight = (idx - idx_lower).unsqueeze(1)
        self.colors = self.color_map[idx_lower] * (1.0 - weight) + self.color_map[idx_upper] * weight
        
        # --- LENNARD-JONES FFT ---
        rho_k = torch.fft.fft2(density)
        phi_k = rho_k * self.LJ_k
        phi = torch.fft.ifft2(phi_k).real
        
        grad_y, grad_x = torch.gradient(phi)
        
        force_x = -(grad_x[grid_pos[:, 1], grid_pos[:, 0]] / PHYSICS_CELL_SIZE)
        force_y = -(grad_y[grid_pos[:, 1], grid_pos[:, 0]] / PHYSICS_CELL_SIZE)
        
        self.vel[:, 0] += force_x * dt
        self.vel[:, 1] += force_y * dt
        
        # Friction/Damping for thermodynamic pooling
        self.vel *= 0.99  
        speed = torch.norm(self.vel, dim=1, keepdim=True)
        self.vel = torch.where(speed > 1500.0, (self.vel / speed) * 1500.0, self.vel)


def main():
    # Adding pos=(0, 0) pins the window to the top-left for the borderless fullscreen effect
    window = ti.ui.Window("Lennard-Jones Thermodynamics Engine", (WIDTH, HEIGHT), vsync=False, pos=(0, 0))
    canvas = window.get_canvas()
    gui = window.get_gui()

    system = LennardJonesSystem(num_particles=100000)
    
    interaction_radius = 150.0
    spawn_density = 0.5
    
    last_time = time.time()
    frames = 0
    fps_display = 0.0
    target_frame_time = 1.0 / FPS_TARGET

    while window.running:
        loop_start = time.time()
        dt = target_frame_time

        # --- ESCAPE HATCH FOR BORDERLESS WINDOW ---
        if window.is_pressed(ti.ui.ESCAPE):
            break

        # --- FPS TRACKER ---
        frames += 1
        current_time = time.time()
        if current_time - last_time >= 0.5:
            fps_display = frames / (current_time - last_time)
            frames = 0
            last_time = current_time

        window.get_events()
        mouse_norm = window.get_cursor_pos()
        mouse_x = mouse_norm[0] * WIDTH
        mouse_y = mouse_norm[1] * HEIGHT
        
        is_hovering_ui = (MOUSE_SAFE_X_MIN <= mouse_norm[0] <= MOUSE_SAFE_X_MAX) and \
                         (MOUSE_SAFE_Y_MIN <= mouse_norm[1] <= MOUSE_SAFE_Y_MAX)
        
        if window.is_pressed(ti.ui.LMB) and not is_hovering_ui:
            spawn_count = math.pi * (interaction_radius**2) * spawn_density * 0.1
            system.add_cluster(mouse_x, mouse_y, interaction_radius, count=spawn_count)
            
        if window.is_pressed(ti.ui.RMB) and not is_hovering_ui:
            system.remove_particles(mouse_x, mouse_y, interaction_radius)

        # --- PHYSICS ---
        system.step(dt)

        # --- RENDERING ---
        canvas.set_background_color((0.02, 0.02, 0.03))
        
        if len(system.pos) > 0:
            # We now pass the raw PyTorch tensors directly to the GPU!
            pos_tensor = system.pos.contiguous()
            color_tensor = system.colors.contiguous()
            
            # Fire the Compute Shader into the internal float32 buffer
            render_to_pixels(pos_tensor, color_tensor, len(pos_tensor), CAM_X, CAM_Y, float(RENDER_CELL_SIZE))
            
            # Prepare the 8-bit presentation buffer
            prepare_display()
            
            # Instantly blit the low-bandwidth uint8 map to the screen
            canvas.set_image(display_pixels)

        # --- BRUSH OVERLAY ---
        brush_r_norm_x = interaction_radius / float(WIDTH)
        brush_r_norm_y = interaction_radius / float(HEIGHT)
        update_brush_ring(mouse_norm[0], mouse_norm[1], brush_r_norm_x, brush_r_norm_y)
        
        if window.is_pressed(ti.ui.RMB):
            canvas.circles(ring_pos_field, radius=0.001, color=(1.0, 0.2, 0.2))
        else:
            canvas.circles(ring_pos_field, radius=0.001, color=(0.2, 0.8, 0.6))

        # --- UI OVERLAY ---
        gui.begin("Fluid Controls", UI_X, UI_Y, UI_W, UI_H)
        
        gui.text(f"FPS: {fps_display:.1f} / {FPS_TARGET:.0f}")
        gui.text(f"Total Particles: {len(system.pos)}")
        gui.text(f"Physics Grid: {PHYSICS_CELL_SIZE}px  |  Render Grid: {RENDER_CELL_SIZE}px")
        
        old_epsilon = system.epsilon
        old_sigma = system.sigma
        
        system.global_gravity = gui.slider_float("Global Gravity", system.global_gravity, -500.0, 5000.0)
        system.epsilon = gui.slider_float("LJ Epsilon (Attraction)", system.epsilon, 10.0, 5000.0)
        system.sigma = gui.slider_float("LJ Sigma (Repulsion/Size)", system.sigma, 1.0, 200.0)
        
        if old_epsilon != system.epsilon or old_sigma != system.sigma:
            system.precompute_LJ()
            
        interaction_radius = gui.slider_float("Brush Radius", interaction_radius, 20.0, 1000.0)
        spawn_density = gui.slider_float("Spawn Density", spawn_density, 0.001, 2.0)
        
        gui.end()
        window.show()
        
        # --- FPS THROTTLE ---
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_frame_time:
            time.sleep(target_frame_time - loop_elapsed)

if __name__ == "__main__":
    main()