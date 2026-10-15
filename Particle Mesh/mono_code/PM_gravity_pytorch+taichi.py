import taichi as ti
import torch
import math
import time
import os
import shutil
import json
from potentials import PotentialsManager

# Disable VSYNC so we can manually control the framerate without VSYNC cliffs
ti.init(arch=ti.cuda)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        # Spawner setup
        self.pos = torch.rand((num_particles, 2), device=self.device)
        self.pos[:, 0] = (self.pos[:, 0] * WIDTH) + CAM_X
        self.pos[:, 1] = (self.pos[:, 1] * HEIGHT) + CAM_Y
        
        self.vel = (torch.rand((num_particles, 2), device=self.device) - 0.5) * 10.0
        self.masses = torch.ones(num_particles, device=self.device)
        self.colors = torch.full((num_particles, 3), 255, dtype=torch.float32, device=self.device)
        
        # 7-Color Thermal Density Map
        self.color_map = torch.tensor([
            [0, 0, 128],     # Dark Blue
            [0, 0, 255],     # Blue
            [255, 0, 128],   # Pink/Magenta
            [255, 0, 0],     # Red
            [255, 128, 0],   # Orange
            [255, 255, 0],   # Yellow
            [255, 255, 255]  # White
        ], dtype=torch.float32, device=self.device)
        
        # Initialize the potentials manager
        self.pot_manager = PotentialsManager(GRID_W, GRID_H, CELL_SIZE, self.device)
        self.precompute_kernel()

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

    def add_galaxy(self, mouse_x, mouse_y, radius, count=5000):
        center_x = mouse_x + CAM_X
        center_y = mouse_y + CAM_Y
        
        count = int(count)
        if count <= 0: return
        
        r = torch.rand(count, device=self.device) * radius
        theta = torch.rand(count, device=self.device) * 2 * math.pi
        
        new_pos = torch.zeros((count, 2), device=self.device)
        new_pos[:, 0] = center_x + r * torch.cos(theta)
        new_pos[:, 1] = center_y + r * torch.sin(theta)
        
        # Generate initial velocities matching the physical mode
        if self.potential_type == 'gravity':
            orbital_speed = torch.sqrt(abs(self.G) * 100.0 / (r + 10.0))
            new_vel = torch.zeros((count, 2), device=self.device)
            new_vel[:, 0] = -torch.sin(theta) * orbital_speed
            new_vel[:, 1] = torch.cos(theta) * orbital_speed
        else:
            # Thermal random velocities for LJ fluid drops or spring structures
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

        # 1. Update positions
        self.pos += self.vel * dt

        # 2. Handle Boundary Collisions/Filtering
        if self.boundary_mode == 'reflective_walls':
            hit_left = self.pos[:, 0] < 0
            hit_right = self.pos[:, 0] > UNIVERSE_W
            hit_top = self.pos[:, 1] < 0
            hit_bottom = self.pos[:, 1] > UNIVERSE_H
            
            self.pos[:, 0] = torch.where(hit_left, 0.0, self.pos[:, 0])
            self.pos[:, 0] = torch.where(hit_right, float(UNIVERSE_W), self.pos[:, 0])
            self.vel[:, 0] = torch.where(hit_left | hit_right, -self.vel[:, 0] * 0.5, self.vel[:, 0])

            self.pos[:, 1] = torch.where(hit_top, 0.0, self.pos[:, 1])
            self.pos[:, 1] = torch.where(hit_bottom, float(UNIVERSE_H), self.pos[:, 1])
            self.vel[:, 1] = torch.where(hit_top | hit_bottom, -self.vel[:, 1] * 0.5, self.vel[:, 1])
        else:
            in_bounds = (self.pos[:, 0] >= 0) & (self.pos[:, 0] < UNIVERSE_W) & \
                        (self.pos[:, 1] >= 0) & (self.pos[:, 1] < UNIVERSE_H)
            if not in_bounds.all().item():
                self.pos = self.pos[in_bounds]
                self.vel = self.vel[in_bounds]
                self.masses = self.masses[in_bounds]
                self.colors = self.colors[in_bounds]

        if len(self.pos) == 0: return

        # --- PM FFT DENSITY MAPPING ---
        grid_pos = (self.pos / CELL_SIZE).long()
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
            
        # Update velocities
        self.vel[:, 0] += part_fx * dt
        self.vel[:, 1] += part_fy * dt
        
        # Apply fluid dampening cooling for Lennard-Jones to allow thermodynamic condensation
        if self.potential_type == 'lennard_jones':
            self.vel *= 0.99


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

        window.get_events() 

        mouse_norm = window.get_cursor_pos()
        mouse_x = mouse_norm[0] * WIDTH
        mouse_y = mouse_norm[1] * HEIGHT
        
        # Perfect UI hover checking
        is_hovering_ui = (MOUSE_SAFE_X_MIN <= mouse_norm[0] <= MOUSE_SAFE_X_MAX) and \
                         (MOUSE_SAFE_Y_MIN <= mouse_norm[1] <= MOUSE_SAFE_Y_MAX)
        
        if window.is_pressed(ti.ui.LMB) and not is_hovering_ui:
            system.add_galaxy(mouse_x, mouse_y, interaction_radius, count=interaction_radius)
            
        if window.is_pressed(ti.ui.RMB) and not is_hovering_ui:
            system.remove_particles(mouse_x, mouse_y, interaction_radius)

        # --- PHYSICS STEP ---
        system.step(dt)

        # --- RENDERING ---
        canvas.set_background_color((0.02, 0.02, 0.03)) 
            
        if len(system.pos) > 0:
            pos_tensor = system.pos.contiguous()
            color_tensor = system.colors.contiguous()
            
            num_active = len(pos_tensor)
            if num_active > MAX_RENDER_PARTICLES:
                num_active = MAX_RENDER_PARTICLES

            copy_to_taichi(pos_tensor, color_tensor, num_active, CAM_X, CAM_Y, float(WIDTH), float(HEIGHT))

            canvas.circles(
                render_pos_field, 
                radius=PARTICLE_RADIUS, 
                per_vertex_color=render_color_field
            )

        # --- BRUSH RING RENDERING ---
        brush_r_norm_x = interaction_radius / float(WIDTH)
        brush_r_norm_y = interaction_radius / float(HEIGHT)
        
        update_brush_ring(mouse_norm[0], mouse_norm[1], brush_r_norm_x, brush_r_norm_y)
        
        if window.is_pressed(ti.ui.RMB) and not is_hovering_ui:
            canvas.circles(ring_pos_field, radius=0.001, color=(1.0, 0.2, 0.2))
        else:
            canvas.circles(ring_pos_field, radius=0.001, color=(0.2, 0.8, 0.6))

        # --- RECORD FRAME CAPTURE ---
        if recording:
            filename = f"temp_frames/frame_{record_frames:05d}.png"
            window.save_image(filename)
            record_frames += 1

        # --- UI OVERLAY ---
        gui.begin("Simulation Controls", UI_X, UI_Y, UI_W, UI_H)
        
        gui.text(f"FPS: {fps_display:.1f} / {FPS_TARGET:.0f}")
        gui.text(f"Total Stars: {len(system.pos)}")
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
            system.potential_type = 'gravity'
            system.precompute_kernel()
        if gui.button("Switch to Lennard-Jones"):
            system.potential_type = 'lennard_jones'
            system.precompute_kernel()
        if gui.button("Switch to Mutual Springs"):
            system.potential_type = 'harmonic'
            system.precompute_kernel()
            
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
        
        if system.potential_type == 'gravity':
            log_min = math.log10(s_grav[0])
            log_max = math.log10(s_grav[1])
            log_G_val = math.log10(system.G)
            
            log_G_val = gui.slider_float("Gravity G (Log)", log_G_val, log_min, log_max)
            new_G = 10 ** log_G_val
            if new_G != system.G:
                system.G = new_G
                system.precompute_kernel()
            gui.text(f"Active G: {system.G:.4f}")
            
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
        
        gui.end()

        # Dispatches the frame
        window.show()
        
        # --- FPS THROTTLE ---
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_frame_time:
            time.sleep(target_frame_time - loop_elapsed)

if __name__ == "__main__":
    main()