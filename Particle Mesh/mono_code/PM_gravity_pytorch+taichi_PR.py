import os
# Premium hardware memory allocation configuration (prevents VRAM fragmentation on Windows CUDA)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import taichi as ti
import torch
import math
import time
import shutil
import json
from potentials import PotentialsManager
from colors import THEMES

# Disable VSYNC so we can manually control the framerate without VSYNC cliffs
# Initialize Taichi with locked-down memory preallocation fraction (plays exceptionally nice with PyTorch VRAM caching)
ti.init(arch=ti.cuda, device_memory_fraction=0.15)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    # Enable hardware-level cuDNN benchmark to find mathematically optimal kernels for this exact hardware
    torch.backends.cudnn.benchmark = True

# --- Load Configuration ---
try:
    _dir = os.path.dirname(os.path.abspath(__file__))
    _config_candidates = [
        os.path.join(_dir, 'config.json'),
        os.path.join(_dir, '..', 'config.json')
    ]
    CONFIG_PATH = next((p for p in _config_candidates if os.path.exists(p)), 'config.json')
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = {}

# --- Configuration & Canvas Setup ---
display_cfg = CONFIG.get("display", {})
WIDTH = display_cfg.get("WIDTH", 3840)
HEIGHT = display_cfg.get("HEIGHT", 2160)
FPS_TARGET = display_cfg.get("FPS_TARGET", 120.0)

sim_cfg = CONFIG.get("simulation", {})
MAX_PARTICLES = sim_cfg.get("MAX_PARTICLES", 100000000)
CELL_SIZE = sim_cfg.get("PHYSICS_CELL_SIZE", 1.0)
PARTICLE_RADIUS = 0.0005 # Visual radius of stars

# Default 8k x 8k square universe physics grid if CELL_SIZE is 1.0
UNIVERSE_W = 8192
UNIVERSE_H = 8192
GRID_W = int(UNIVERSE_W / CELL_SIZE)
GRID_H = int(UNIVERSE_H / CELL_SIZE)

# Camera offsets to keep the screen dead-center
CAM_X = (UNIVERSE_W - WIDTH) / 2.0
CAM_Y = (UNIVERSE_H - HEIGHT) / 2.0

# Visual Density Limits
MIN_LOG_DENSITY = sim_cfg.get("MIN_LOG_DENSITY", 0.0)
MAX_LOG_DENSITY = sim_cfg.get("MAX_LOG_DENSITY", 1.0 / CELL_SIZE)

# --- UI Layout Constraints (Protected Bounds) ---
UI_X, UI_Y = 0.01, 0.02
UI_W, UI_H = 0.25, 0.85

# Convert UI bounds to Bottom-Left Origin for precise mouse hover protection
MOUSE_SAFE_X_MIN = UI_X
MOUSE_SAFE_X_MAX = UI_X + UI_W
MOUSE_SAFE_Y_MIN = 1.0 - (UI_Y + UI_H)
MOUSE_SAFE_Y_MAX = 1.0 - UI_Y

# --- TAICHI RENDER FIELDS (Zero-Copy Bridge) ---
MAX_RENDER_PARTICLES = 290000  
render_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_RENDER_PARTICLES)
render_color_field = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER_PARTICLES)
ring_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=64)

@ti.kernel
def copy_to_taichi(
    pos_arr: ti.types.ndarray(), 
    color_arr: ti.types.ndarray(), 
    num_particles: ti.i32,
    cam_x: ti.f32,
    cam_y: ti.f32,
    screen_w: ti.f32,
    screen_h: ti.f32
):
    for i in render_pos_field:
        if i < num_particles:
            # Mathematical translation & normalization directly on the GPU!
            render_pos_field[i] = [(pos_arr[i, 0] - cam_x) / screen_w, 
                                   (pos_arr[i, 1] - cam_y) / screen_h]
            # Normalize colors instantly without CPU copying
            render_color_field[i] = [color_arr[i, 0] / 255.0, 
                                     color_arr[i, 1] / 255.0, 
                                     color_arr[i, 2] / 255.0]
        else:
            # Teleport inactive particles off-screen
            render_pos_field[i] = [-10.0, -10.0]

@ti.kernel
def update_brush_ring(center_x: ti.f32, center_y: ti.f32, radius_x: ti.f32, radius_y: ti.f32):
    for i in range(64):
        angle = float(i) / 64.0 * 2.0 * 3.14159265
        ring_pos_field[i] = [center_x + ti.cos(angle) * radius_x, center_y + ti.sin(angle) * radius_y]


class FFTGravitySystem:
    def __init__(self, num_particles):
        self.device = DEVICE
        
        # Load custom sliders from config
        sliders = CONFIG.get("sliders", {})
        
        # Active Model Parameters
        self.potential_type = 'gravity' # 'gravity', 'lennard_jones', 'harmonic'
        self.pressure_model = 'none' # 'none', 'isothermal', 'polytropic'
        self.boundary_mode = 'wrap_delete' # 'wrap_delete', 'reflective_walls'
        
        # Physics Parameters
        self.G = sliders.get("gravity", [0.01, 100.0, 0.01])[2]
        self.force_range = sliders.get("gravity_range", [1.0, 8192.0, 8192.0])[2]
        self.pressure_strength = sliders.get("pressure", [0.01, 10000.0, 0.0])[2] # Starts at 0
        self.gamma = sliders.get("polytropic_gamma", [0.0, 3.0, 1.0])[2]
        
        self.spring_k = sliders.get("spring_k", [0.001, 10.0, 0.1])[2]
        self.epsilon = sliders.get("epsilon", [1.0, 5000.0, 1000.0])[2]
        self.sigma = sliders.get("sigma", [1.0, 200.0, 40.0])[2]
        
        self.uniform_gravity_mag = sliders.get("uniform_gravity_mag", [0.0, 2000.0, 0.0])[2]
        self.uniform_gravity_angle = sliders.get("uniform_gravity_angle", [0.0, 360.0, 270.0])[2]
        
        # Preallocate static arrays to MAX_PARTICLES (100 million) to eliminate any VRAM fluctuation
        self.num_active = num_particles
        self.pos = torch.zeros((MAX_PARTICLES, 2), dtype=torch.float32, device=self.device)
        self.vel = torch.zeros((MAX_PARTICLES, 2), dtype=torch.float32, device=self.device)
        self.masses = torch.zeros(MAX_PARTICLES, dtype=torch.float32, device=self.device)
        self.colors = torch.zeros((MAX_PARTICLES, 3), dtype=torch.float32, device=self.device)
        
        # Initialize the active particles
        self.pos[:self.num_active, 0] = (torch.rand(self.num_active, device=self.device) * WIDTH) + CAM_X
        self.pos[:self.num_active, 1] = (torch.rand(self.num_active, device=self.device) * HEIGHT) + CAM_Y
        self.vel[:self.num_active] = (torch.rand((self.num_active, 2), device=self.device) - 0.5) * 10.0
        self.masses[:self.num_active] = 1.0
        self.colors[:self.num_active] = 255.0
        
        self.color_theme = 0 
        self.theme_names = list(THEMES.keys())
        self.color_maps = [
            torch.tensor(THEMES[name], dtype=torch.float32, device=self.device) 
            for name in self.theme_names
        ]
        
        # Initialize the potentials manager
        self.pot_manager = PotentialsManager(GRID_W, GRID_H, CELL_SIZE, self.device)
        self.precompute_kernel()
 
    @torch.no_grad()
    def precompute_kernel(self):
        """
        Creates the Fourier-space interaction kernel from potentials.py dynamically.
        """
        params = {
            'G': self.G,
            'force_range': self.force_range,
            'spring_k': self.spring_k,
            'epsilon': self.epsilon,
            'sigma': self.sigma
        }
        self.K_k = self.pot_manager.get_fourier_kernel(self.potential_type, params)

    @torch.no_grad()
    def set_potential_type(self, p_type):
        """
        Switches the potential type and assigns random +1/-1 charges for Coulomb or restores masses to 1.
        """
        self.potential_type = p_type
        if p_type == 'coulomb':
            self.masses[:self.num_active] = torch.randint(0, 2, (self.num_active,), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        else:
            self.masses[:self.num_active] = 1.0
        self.precompute_kernel()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def add_galaxy(self, world_x, world_y, world_radius, count=5000):
        count = int(count)
        if count <= 0 or self.num_active >= MAX_PARTICLES: return
        
        count = min(count, MAX_PARTICLES - self.num_active)
        start, end = self.num_active, self.num_active + count
        
        # Perform all spawner coordinate generation entirely on GPU to eliminate Host-to-Device transfer stalls
        r = torch.rand(count, device=self.device) * world_radius
        theta = torch.rand(count, device=self.device) * (2.0 * math.pi)
        
        # Write directly to the preallocated GPU slices in-place
        self.pos[start:end, 0] = world_x + r * torch.cos(theta)
        self.pos[start:end, 1] = world_y + r * torch.sin(theta)
        
        # Generate initial velocities matching the physical mode
        if self.potential_type == 'gravity':
            orbital_speed = torch.sqrt(abs(self.G) * 100.0 / (r + 10.0))
            self.vel[start:end, 0] = -torch.sin(theta) * orbital_speed
            self.vel[start:end, 1] = torch.cos(theta) * orbital_speed
        else:
            # Thermal random velocities directly in-place
            self.vel[start:end].uniform_(-0.5, 0.5)
            self.vel[start:end] *= 50.0
            
        if self.potential_type == 'coulomb':
            self.masses[start:end] = (torch.randint(0, 2, (count,), device=self.device, dtype=torch.float32) * 2.0 - 1.0)
        else:
            self.masses[start:end] = 1.0
            
        self.colors[start:end] = 255.0
        self.num_active += count

    @torch.no_grad()
    def remove_particles(self, world_x, world_y, world_radius):
        if self.num_active == 0: return
        uni_pos = torch.tensor([world_x, world_y], dtype=torch.float32, device=self.device)
        
        distances = torch.norm(self.pos[:self.num_active] - uni_pos, dim=1)
        keep_mask = distances > world_radius
        new_count = keep_mask.sum().item()
        
        self.pos[:new_count] = self.pos[:self.num_active][keep_mask]
        self.vel[:new_count] = self.vel[:self.num_active][keep_mask]
        self.masses[:new_count] = self.masses[:self.num_active][keep_mask]
        self.colors[:new_count] = self.colors[:self.num_active][keep_mask]
        self.num_active = new_count

    @torch.no_grad()
    def step(self, dt):
        if self.num_active == 0: return

        # 1. Update positions
        self.pos[:self.num_active] += self.vel[:self.num_active] * dt

        # 2. Handle Boundary Collisions/Filtering
        if self.boundary_mode == 'reflective_walls':
            hit_left = self.pos[:self.num_active, 0] < 0
            hit_right = self.pos[:self.num_active, 0] > UNIVERSE_W
            hit_top = self.pos[:self.num_active, 1] < 0
            hit_bottom = self.pos[:self.num_active, 1] > UNIVERSE_H
            
            self.pos[:self.num_active, 0] = torch.where(hit_left, 0.0, self.pos[:self.num_active, 0])
            self.pos[:self.num_active, 0] = torch.where(hit_right, float(UNIVERSE_W), self.pos[:self.num_active, 0])
            self.vel[:self.num_active, 0] = torch.where(hit_left | hit_right, -self.vel[:self.num_active, 0] * 0.5, self.vel[:self.num_active, 0])

            self.pos[:self.num_active, 1] = torch.where(hit_top, 0.0, self.pos[:self.num_active, 1])
            self.pos[:self.num_active, 1] = torch.where(hit_bottom, float(UNIVERSE_H), self.pos[:self.num_active, 1])
            self.vel[:self.num_active, 1] = torch.where(hit_top | hit_bottom, -self.vel[:self.num_active, 1] * 0.5, self.vel[:self.num_active, 1])
        else:
            in_bounds = (self.pos[:self.num_active, 0] >= 0) & (self.pos[:self.num_active, 0] < UNIVERSE_W) & \
                        (self.pos[:self.num_active, 1] >= 0) & (self.pos[:self.num_active, 1] < UNIVERSE_H)
            if not in_bounds.all().item():
                new_count = in_bounds.sum().item()
                self.pos[:new_count] = self.pos[:self.num_active][in_bounds]
                self.vel[:new_count] = self.vel[:self.num_active][in_bounds]
                self.masses[:new_count] = self.masses[:self.num_active][in_bounds]
                self.colors[:new_count] = self.colors[:self.num_active][in_bounds]
                self.num_active = new_count

        if self.num_active == 0: return

        # --- PM FFT DENSITY MAPPING ---
        grid_pos = (self.pos[:self.num_active] / CELL_SIZE).long()
        grid_pos[:, 0] = torch.clamp(grid_pos[:, 0], 0, GRID_W - 1)
        grid_pos[:, 1] = torch.clamp(grid_pos[:, 1], 0, GRID_H - 1)
        
        indices = grid_pos[:, 1] * GRID_W + grid_pos[:, 0]
        density = torch.bincount(indices, weights=self.masses[:self.num_active], minlength=GRID_W*GRID_H)
        density = density.view(GRID_H, GRID_W)
        
        # --- VECTORIZED PARTICLE COLORING ---
        particle_densities = density[grid_pos[:, 1], grid_pos[:, 0]]
        log_dens = torch.log10(particle_densities + 0.1)
        norm_dens = torch.clamp((log_dens - MIN_LOG_DENSITY) / (MAX_LOG_DENSITY - MIN_LOG_DENSITY), 0.0, 1.0)
        
        # 9-color themes have 8 gaps
        idx = norm_dens * 8.0
        idx_lower = torch.floor(idx).long()
        idx_upper = torch.ceil(idx).long()
        idx_upper = torch.clamp(idx_upper, 0, 8) 
        
        weight = (idx - idx_lower).unsqueeze(1)
        active_map = self.color_maps[self.color_theme]
        self.colors[:self.num_active] = active_map[idx_lower] * (1.0 - weight) + active_map[idx_upper] * weight
        
        # --- MUTUAL POTENTIALS (FFT Convolutions) ---
        rho_k = torch.fft.fft2(density)
        phi_k = rho_k * self.K_k
        phi = torch.fft.ifft2(phi_k).real
        
        # --- CALCULATE GRID FIELD FORCES ---
        grad_y, grad_x = torch.gradient(phi)
        force_x = -(grad_x / CELL_SIZE)
        force_y = -(grad_y / CELL_SIZE)
        
        # --- POLYTROPIC PRESSURE ACCELERATION ---
        if self.pressure_model != 'none':
            gamma_val = self.gamma if self.pressure_model == 'polytropic' else 1.0
            force_px, force_py = self.pot_manager.compute_pressure_forces(density, self.pressure_strength, gamma_val)
            force_x += force_px
            force_y += force_py
            
        # Sample net forces at particle coordinates
        part_fx = force_x[grid_pos[:, 1], grid_pos[:, 0]]
        part_fy = force_y[grid_pos[:, 1], grid_pos[:, 0]]
        
        # --- UNIFORM GRAVITY ACCELERATION ---
        if self.uniform_gravity_mag > 0:
            ugx, ugy = self.pot_manager.get_uniform_gravity_forces(self.uniform_gravity_mag, self.uniform_gravity_angle)
            part_fx += ugx
            part_fy += ugy
            
        # Update velocities with charge (masses) multiplication
        self.vel[:self.num_active, 0] += part_fx * self.masses[:self.num_active] * dt
        self.vel[:self.num_active, 1] += part_fy * self.masses[:self.num_active] * dt
        
        # Apply fluid dampening cooling for Lennard-Jones to allow thermodynamic condensation
        if self.potential_type == 'lennard_jones':
            self.vel[:self.num_active] *= 0.99


def compile_video(frame_count, output_name="simulation.mp4", fps=60):
    """
    Offline video compiler that loads captured simulation images from temp_frames
    and encodes them into a perfectly smooth 60 FPS MP4 video file.
    """
    import cv2
    print(f"Post-processing simulation: Compiling {frame_count} frames into '{output_name}' at {fps} FPS...")
    
    first_frame_path = "temp_frames/frame_00000.png"
    if not os.path.exists(first_frame_path):
        print("Error: No captured frames found for compiling!")
        return False
        
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print("Error: Failed to load captured frames with cv2!")
        return False
    height, width, layers = first_frame.shape
    
    # Universal Windows-compatible MP4 container codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    
    for i in range(frame_count):
        frame_path = f"temp_frames/frame_{i:05d}.png"
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            video.write(frame)
            
    video.release()
    print(f"Recording successfully completed: '{output_name}' is ready to watch!")
    
    # Auto clean-up temporary screenshots
    try:
        shutil.rmtree('temp_frames')
        print("Cleaned up temporary directories.")
    except Exception as e:
        print(f"Warning cleaning temp frames: {e}")
        
    return True


def main():
    window = ti.ui.Window("PyTorch + Taichi Flexible Physical Sandbox", (WIDTH, HEIGHT), vsync=False)
    canvas = window.get_canvas()
    gui = window.get_gui()

    system = FFTGravitySystem(num_particles=200000)
    interaction_radius = 100.0
    
    # Slider setup loaded from configurations
    sliders_cfg = CONFIG.get("sliders", {})
    s_grav = sliders_cfg.get("gravity", [0.01, 100.0, 0.01])
    s_fr = sliders_cfg.get("gravity_range", [1.0, 8192.0, 8192.0])
    s_pres = sliders_cfg.get("pressure", [0.01, 10000.0, 0.0])
    s_rad = sliders_cfg.get("brush_radius", [20.0, 1000.0, 100.0])
    s_gamma = sliders_cfg.get("polytropic_gamma", [0.0, 3.0, 1.0])
    s_spring = sliders_cfg.get("spring_k", [0.001, 10.0, 0.1])
    s_eps = sliders_cfg.get("epsilon", [1.0, 5000.0, 1000.0])
    s_sig = sliders_cfg.get("sigma", [1.0, 200.0, 40.0])
    s_ug_mag = sliders_cfg.get("uniform_gravity_mag", [0.0, 2000.0, 0.0])
    s_ug_ang = sliders_cfg.get("uniform_gravity_angle", [0.0, 360.0, 270.0])

    cam_x = (UNIVERSE_W - WIDTH) / 2.0
    cam_y = (UNIVERSE_H - HEIGHT) / 2.0
    zoom = 1.0
    ui_open = True
    
    prev_r = False
    prev_m = False
    
    prev_pan_active = False
    prev_mouse_norm = (0.0, 0.0)
    prev_z_lmb = False
    prev_z_rmb = False

    last_time = time.time()
    frames = 0
    fps_display = 0.0
    
    target_frame_time = 1.0 / FPS_TARGET
    
    # Video Recording State Variables
    recording = False
    record_frames = 0

    while window.running:
        loop_start = time.time()
        dt = target_frame_time

        # --- FPS CALCULATION ---
        frames += 1
        current_time = time.time()
        if current_time - last_time >= 0.5: 
            fps_display = frames / (current_time - last_time)
            frames = 0
            last_time = current_time

        z_pressed = window.is_pressed('z')
        space_pressed = window.is_pressed(ti.ui.SPACE)
        
        r_pressed = window.is_pressed('r')
        m_pressed = window.is_pressed('m')
        
        # Edge-triggered shortcuts
        if r_pressed and not prev_r:
            cam_x = (UNIVERSE_W - WIDTH) / 2.0
            cam_y = (UNIVERSE_H - HEIGHT) / 2.0
            zoom = 1.0
        if m_pressed and not prev_m:
            ui_open = not ui_open
            
        prev_r = r_pressed
        prev_m = m_pressed

        mouse_norm = window.get_cursor_pos()
        
        # Keyboard Zooming
        if window.is_pressed(ti.ui.UP):
            mouse_x, mouse_y = mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT
            w_x = (mouse_x / zoom) + cam_x
            w_y = (mouse_y / zoom) + cam_y
            zoom *= 1.02
            zoom = max(0.0001, min(10000.0, zoom))
            cam_x = w_x - (mouse_x / zoom)
            cam_y = w_y - (mouse_y / zoom)
            
        if window.is_pressed(ti.ui.DOWN):
            mouse_x, mouse_y = mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT
            w_x = (mouse_x / zoom) + cam_x
            w_y = (mouse_y / zoom) + cam_y
            zoom *= (1.0 / 1.02)
            zoom = max(0.0001, min(10000.0, zoom))
            cam_x = w_x - (mouse_x / zoom)
            cam_y = w_y - (mouse_y / zoom)

        # Mouse Wheel Event Dispatching
        for e in window.get_events():
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == getattr(ti.ui, 'WHEEL', 'Wheel') or e.key == getattr(ti.GUI, 'WHEEL', 'Wheel'):
                zoom_delta = e.delta[1]
                if z_pressed:
                    mouse_x, mouse_y = mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT
                    w_x = (mouse_x / zoom) + cam_x
                    w_y = (mouse_y / zoom) + cam_y
                    
                    zoom *= 1.1 if zoom_delta > 0 else (1.0 / 1.1)
                    zoom = max(0.0001, min(10000.0, zoom)) 
                    
                    cam_x = w_x - (mouse_x / zoom)
                    cam_y = w_y - (mouse_y / zoom)
                else:
                    interaction_radius *= 1.1 if zoom_delta > 0 else (1.0 / 1.1)
                    interaction_radius = max(s_rad[0], min(s_rad[1], interaction_radius))

        mouse_x = mouse_norm[0] * WIDTH
        mouse_y = mouse_norm[1] * HEIGHT
        
        # Drag Panning
        lmb_pressed = window.is_pressed(ti.ui.LMB)
        rmb_pressed = window.is_pressed(ti.ui.RMB)
        mmb_pressed = window.is_pressed(ti.ui.MMB)
        
        pan_active = (lmb_pressed and space_pressed) or mmb_pressed
        if pan_active:
            if prev_pan_active:
                dx = (mouse_norm[0] - prev_mouse_norm[0]) * WIDTH / zoom
                dy = (mouse_norm[1] - prev_mouse_norm[1]) * HEIGHT / zoom
                cam_x -= dx
                cam_y -= dy
        prev_pan_active = pan_active
        
        # Z+Drag Actions
        z_lmb_active = lmb_pressed and z_pressed
        if z_lmb_active:
            if prev_z_lmb:
                dy = mouse_norm[1] - prev_mouse_norm[1]
                zoom_factor = math.exp(dy * 5.0) 
                
                center_x, center_y = WIDTH / 2.0, HEIGHT / 2.0
                world_cx = (center_x / zoom) + cam_x
                world_cy = (center_y / zoom) + cam_y
                
                zoom *= zoom_factor
                zoom = max(0.0001, min(10000.0, zoom))
                
                cam_x = world_cx - (center_x / zoom)
                cam_y = world_cy - (center_y / zoom)
        prev_z_lmb = z_lmb_active

        z_rmb_active = rmb_pressed and z_pressed
        if z_rmb_active:
            if prev_z_rmb:
                dy = mouse_norm[1] - prev_mouse_norm[1]
                interaction_radius *= math.exp(dy * 5.0)
                interaction_radius = max(s_rad[0], min(s_rad[1], interaction_radius))
        prev_z_rmb = z_rmb_active

        prev_mouse_norm = mouse_norm
        
        # Translate to World Coordinates
        world_x = (mouse_x / zoom) + cam_x
        world_y = (mouse_y / zoom) + cam_y
        world_radius = interaction_radius / zoom

        # Perfect UI Hover Checking
        is_hovering_ui = False
        if ui_open:
            is_hovering_ui = (MOUSE_SAFE_X_MIN <= mouse_norm[0] <= MOUSE_SAFE_X_MAX) and \
                             (MOUSE_SAFE_Y_MIN <= mouse_norm[1] <= MOUSE_SAFE_Y_MAX)
        
        is_modifying_view = z_pressed or space_pressed or pan_active
        
        if lmb_pressed and not is_hovering_ui and not is_modifying_view:
            system.add_galaxy(world_x, world_y, world_radius, count=interaction_radius)
            
        if rmb_pressed and not is_hovering_ui and not is_modifying_view:
            system.remove_particles(world_x, world_y, world_radius)

        # --- PHYSICS STEP ---
        system.step(dt)

        # --- RENDERING ---
        canvas.set_background_color((0.02, 0.02, 0.03)) 
            
        if system.num_active > 0:
            # Pass the fully preallocated contiguous tensors directly to avoid any per-frame slice allocations!
            pos_tensor = system.pos
            color_tensor = system.colors
            
            num_render = system.num_active
            if num_render > MAX_RENDER_PARTICLES:
                num_render = MAX_RENDER_PARTICLES

            copy_to_taichi(pos_tensor, color_tensor, num_render, cam_x, cam_y, float(WIDTH) / zoom, float(HEIGHT) / zoom)

            canvas.circles(
                render_pos_field, 
                radius=PARTICLE_RADIUS, 
                per_vertex_color=render_color_field
            )

        # --- BRUSH RING RENDERING ---
        brush_r_norm_x = interaction_radius / float(WIDTH)
        brush_r_norm_y = interaction_radius / float(HEIGHT)
        
        update_brush_ring(mouse_norm[0], mouse_norm[1], brush_r_norm_x, brush_r_norm_y)
        
        if window.is_pressed(ti.ui.RMB) and not is_hovering_ui and not is_modifying_view:
            canvas.circles(ring_pos_field, radius=0.001, color=(1.0, 0.2, 0.2))
        else:
            canvas.circles(ring_pos_field, radius=0.001, color=(0.2, 0.8, 0.6))

        # --- RECORD FRAME CAPTURE ---
        if recording:
            filename = f"temp_frames/frame_{record_frames:05d}.png"
            window.save_image(filename)
            record_frames += 1

        # --- UI OVERLAY ---
        if ui_open:
            gui.begin("Simulation Controls", UI_X, UI_Y, UI_W, UI_H)
            
            gui.text(f"FPS: {fps_display:.1f} / {FPS_TARGET:.0f}")
            gui.text(f"Total Stars: {system.num_active}")
            gui.text(f"Render Limit: {MAX_RENDER_PARTICLES}")
            
            # Movie Recording Controller
            if recording:
                gui.text(f"REC ACTIVE: {record_frames} frames captured")
                if gui.button("STOP & SAVE MOVIE"):
                    recording = False
                    compile_video(record_frames, output_name="simulation.mp4", fps=60)
                    record_frames = 0
            else:
                if gui.button("RECORD MOVIE"):
                    if os.path.exists('temp_frames'):
                        try:
                            shutil.rmtree('temp_frames')
                        except Exception:
                            pass
                    os.makedirs('temp_frames', exist_ok=True)
                    recording = True
                    record_frames = 0
                    
            gui.text("")
            gui.text("--- ACTIVE POTENTIAL ---")
            gui.text(f"Selected: {system.potential_type.upper()}")
            if gui.button("Switch to Gravity"):
                system.set_potential_type('gravity')
            if gui.button("Switch to Lennard-Jones"):
                system.set_potential_type('lennard_jones')
            if gui.button("Switch to Mutual Springs"):
                system.set_potential_type('harmonic')
            if gui.button("Switch to Coulomb"):
                system.set_potential_type('coulomb')
                
            gui.text("")
            gui.text("--- PRESSURE WIDGETS ---")
            gui.text(f"Active Model: {system.pressure_model.upper()}")
            if gui.button("Gas Mode: Cold Dust (None)"):
                system.pressure_model = 'none'
            if gui.button("Gas Mode: Isothermal (Linear)"):
                system.pressure_model = 'isothermal'
            if gui.button("Gas Mode: Polytropic (Adiabatic)"):
                system.pressure_model = 'polytropic'
                
            gui.text("")
            gui.text("--- VISUAL THEMES ---")
            gui.text(f"Theme: {system.theme_names[system.color_theme]}")
            if gui.button("Next Color Theme"):
                system.color_theme = (system.color_theme + 1) % len(system.theme_names)
                
            gui.text("")
            gui.text("--- UNIVERSAL PARAMETERS ---")
            
            # Screen parameter Force Range
            old_fr = system.force_range
            system.force_range = gui.slider_float("Force Range", system.force_range, s_fr[0], s_fr[1])
            if old_fr != system.force_range:
                system.precompute_kernel()
                
            # Gas pressure slider
            system.pressure_strength = gui.slider_float("Gas Pressure P", system.pressure_strength, s_pres[0], s_pres[1])
            
            # Polytropic index gamma
            if system.pressure_model == 'polytropic':
                system.gamma = gui.slider_float("Poly Index (Gamma)", system.gamma, s_gamma[0], s_gamma[1])
                
            # Boundary Toggle (Walls vs Wrap)
            bound_text = "SOLID WALL BOUNCE" if system.boundary_mode == 'reflective_walls' else "UNIVERSE WRAPPED"
            if gui.button(f"Boundary: {bound_text}"):
                if system.boundary_mode == 'reflective_walls':
                    system.boundary_mode = 'wrap_delete'
                else:
                    system.boundary_mode = 'reflective_walls'
                    
            gui.text("")
            gui.text("--- POTENTIAL COEFFICIENTS ---")
            
            if system.potential_type in ('gravity', 'coulomb'):
                log_min = math.log10(s_grav[0])
                log_max = math.log10(s_grav[1])
                log_G_val = math.log10(system.G)
                
                label = "Coulomb Const (Log)" if system.potential_type == 'coulomb' else "Gravity G (Log)"
                log_G_val = gui.slider_float(label, log_G_val, log_min, log_max)
                new_G = 10 ** log_G_val
                if new_G != system.G:
                    system.G = new_G
                    system.precompute_kernel()
                val_label = "Active ke" if system.potential_type == 'coulomb' else "Active G"
                gui.text(f"{val_label}: {system.G:.4f}")
                
            elif system.potential_type == 'harmonic':
                old_k = system.spring_k
                system.spring_k = gui.slider_float("Spring K", system.spring_k, s_spring[0], s_spring[1])
                if old_k != system.spring_k:
                    system.precompute_kernel()
                    
            elif system.potential_type == 'lennard_jones':
                old_eps = system.epsilon
                old_sig = system.sigma
                system.epsilon = gui.slider_float("LJ Epsilon", system.epsilon, s_eps[0], s_eps[1])
                system.sigma = gui.slider_float("LJ Sigma", system.sigma, s_sig[0], s_sig[1])
                if old_eps != system.epsilon or old_sig != system.sigma:
                    system.precompute_kernel()
                    
            gui.text("")
            gui.text("--- UNIFORM FIELD ---")
            system.uniform_gravity_mag = gui.slider_float("Gravity Mag", system.uniform_gravity_mag, s_ug_mag[0], s_ug_mag[1])
            system.uniform_gravity_angle = gui.slider_float("Gravity Angle (Deg)", system.uniform_gravity_angle, s_ug_ang[0], s_ug_ang[1])
            
            gui.text("")
            gui.text("--- SPAWNER BRUSH ---")
            interaction_radius = gui.slider_float("Brush Radius", interaction_radius, s_rad[0], s_rad[1])
            
            gui.text("")
            gui.text("--- CRASH-PROOF SHORTCUTS ---")
            gui.text("LMB: Spawn | RMB: Delete")
            gui.text("Space+LMB Drag: Pan View")
            gui.text("Z+LMB Drag (or Scroll): Zoom")
            gui.text("Z+RMB Drag (or Scroll): Brush Size")
            gui.text("R: Reset View | M: Hide Panel")
            
            gui.end()

        # Dispatches the frame
        window.show()
        
        # --- FPS THROTTLE ---
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_frame_time:
            time.sleep(target_frame_time - loop_elapsed)

if __name__ == "__main__":
    main()
