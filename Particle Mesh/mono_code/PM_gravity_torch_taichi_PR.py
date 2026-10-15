import os
# Premium hardware memory allocation configuration (prevents VRAM fragmentation on Windows CUDA)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import taichi as ti
import torch
import math
import time
import json
import shutil
from colors import THEMES
from potentials import PotentialsManager

# --- Load Configuration ---
_dir = os.path.dirname(os.path.abspath(__file__))
_config_candidates = [
    os.path.join(_dir, 'config.json'),
    os.path.join(_dir, '..', 'config.json')
]
CONFIG_PATH = next((p for p in _config_candidates if os.path.exists(p)), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Global Display Settings
WIDTH = CONFIG["display"]["WIDTH"]
HEIGHT = CONFIG["display"]["HEIGHT"]
FPS_TARGET = CONFIG["display"]["FPS_TARGET"]

# --- Vector Display Settings ---
ARROW_WIDTH = CONFIG["display"].get("arrow_width", 0.001)
ARROW_LENGTH_SCALE = CONFIG["display"].get("arrow_length_scale", 1.5)
ARROW_HEAD_SIZE = CONFIG["display"].get("arrow_head_size", 0.25)
ARROW_HEAD_ANGLE = CONFIG["display"].get("arrow_head_angle", 0.5)

# Global Simulation Settings (Dynamic Fallback for backwards compatibility)
MAX_PARTICLES = CONFIG["simulation"]["MAX_PARTICLES"]
MIN_LOG_DENSITY = CONFIG["simulation"]["MIN_LOG_DENSITY"]
MAX_LOG_DENSITY = CONFIG["simulation"]["MAX_LOG_DENSITY"]
RENDER_CELL_SIZE = CONFIG["simulation"]["RENDER_CELL_SIZE"]

# Scale visual intensity based on zoom level to keep particles crisp
ZOOM_INTENSITY_SCALE = 0.5 

# Default 8k x 8k square universe physics grid if PHYSICS_CELL_SIZE is 1.0
UNIVERSE_W = 8192
UNIVERSE_H = 8192
PHYSICS_CELL_SIZE = CONFIG["simulation"].get("PHYSICS_CELL_SIZE", 1.0)
GRID_W = int(UNIVERSE_W / PHYSICS_CELL_SIZE)
GRID_H = int(UNIVERSE_H / PHYSICS_CELL_SIZE)

# Initialize Taichi with locked-down memory preallocation fraction (plays exceptionally nice with PyTorch VRAM caching)
ti.init(arch=ti.cuda, device_memory_fraction=0.15)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    # Enable hardware-level cuDNN benchmark to find mathematically optimal kernels for this exact hardware
    torch.backends.cudnn.benchmark = True

# --- Grid Calculations ---
VIEW_W = int(WIDTH / RENDER_CELL_SIZE)
VIEW_H = int(HEIGHT / RENDER_CELL_SIZE)

CAM_X = (UNIVERSE_W - WIDTH) / 2.0
CAM_Y = (UNIVERSE_H - HEIGHT) / 2.0

# --- Vector Field Pre-Allocation ---
PIXELS_PER_ARROW = CONFIG["display"].get("pixels_per_arrow", 20)
NUM_ARROWS_X = VIEW_W // PIXELS_PER_ARROW
NUM_ARROWS_Y = VIEW_H // PIXELS_PER_ARROW
TOTAL_ARROWS = NUM_ARROWS_X * NUM_ARROWS_Y

if TOTAL_ARROWS > 0:
    # 6 vertices per arrow (2 for shaft, 2 for left wing, 2 for right wing)
    vector_verts = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_ARROWS * 6)
    vector_colors = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_ARROWS * 6)

# --- UI Layout Constraints ---
UI_X, UI_Y = 0.01, 0.02
UI_W = 0.15 
MINIMAP_W = int(WIDTH * UI_W)
MINIMAP_H = int(WIDTH * UI_W) 

# --- TAICHI NATIVE FIELDS ---
display_pixels = ti.Vector.field(3, dtype=ti.u8, shape=(VIEW_W, VIEW_H))
ring_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=64)
arrow_verts = ti.Vector.field(2, dtype=ti.f32, shape=6)
visual_density = ti.field(dtype=ti.f32, shape=(VIEW_W, VIEW_H))

@ti.kernel
def overlay_fft_pip(thumb: ti.types.ndarray(), offset_x: ti.i32, offset_y: ti.i32, thumb_w: ti.i32, thumb_h: ti.i32):
    for i, j in ti.ndrange(thumb_w, thumb_h):
        val = ti.cast(thumb[j, i], ti.u8)
        px = offset_x + i
        py = offset_y + j
        if 0 <= px < VIEW_W and 0 <= py < VIEW_H:
            display_pixels[px, py] = [val, val, val]

@ti.kernel
def init_arrow():
    arrow_verts[0] = [0.016, 0.985]
    arrow_verts[1] = [0.016, 0.965]
    arrow_verts[2] = [0.006, 0.975]
    arrow_verts[3] = [0.028, 0.985]
    arrow_verts[4] = [0.028, 0.965]
    arrow_verts[5] = [0.018, 0.975]

@ti.kernel
def update_brush_ring(center_x: ti.f32, center_y: ti.f32, radius_x: ti.f32, radius_y: ti.f32):
    for i in range(64):
        angle = float(i) / 64.0 * 2.0 * 3.14159265
        ring_pos_field[i] = [center_x + ti.cos(angle) * radius_x, center_y + ti.sin(angle) * radius_y]

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
    pos: ti.types.ndarray(), vel: ti.types.ndarray(), force_x: ti.types.ndarray(), force_y: ti.types.ndarray(), mass: ti.types.ndarray(),
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

        q = ti.cast(mass[p], ti.f32)
        vel[p, 0] = (vel[p, 0] + q * fx * dt) * (1.0 - friction * dt)
        vel[p, 1] = (vel[p, 1] + q * fy * dt) * (1.0 - friction * dt)
        
        pos[p, 0] += vel[p, 0] * dt
        pos[p, 1] += vel[p, 1] * dt

@ti.kernel
def clear_visual_density():
    for i, j in visual_density:
        visual_density[i, j] = 0.0

@ti.kernel
def render_scatter_kernel(
    pos: ti.types.ndarray(), num_active: ti.i32, cam_x: ti.f32, cam_y: ti.f32, zoom: ti.f32,
    view_w: ti.i32, view_h: ti.i32, render_cell_size: ti.f32
):
    for p in range(num_active):
        px, py = (pos[p, 0] - cam_x) * zoom, (pos[p, 1] - cam_y) * zoom
        if 0 <= px < view_w * render_cell_size and 0 <= py < view_h * render_cell_size:
            gx, gy = ti.cast(px / render_cell_size, ti.i32), ti.cast(py / render_cell_size, ti.i32)
            if 0 <= gx < view_w and 0 <= gy < view_h:
                ti.atomic_add(visual_density[gx, gy], 1.0)
                
@ti.kernel
def render_color_kernel(color_map: ti.types.ndarray(), min_log: ti.f32, max_log: ti.f32, scale: ti.f32):
    for i, j in visual_density:
        dens = visual_density[i, j]
        if dens == 0.0:
            display_pixels[i, j] = [ti.cast(0, ti.u8), ti.cast(0, ti.u8), ti.cast(0, ti.u8)]
        else:
            log_dens = ti.log(dens * scale + 0.1) / 2.30258509299
            norm_dens = ti.max(0.0, ti.min((log_dens - min_log) / (max_log - min_log), 1.0))
            
            idx_f = norm_dens * 8.0
            idx_lower = ti.cast(ti.floor(idx_f), ti.i32)
            idx_upper = ti.max(0, ti.min(ti.cast(ti.ceil(idx_f), ti.i32), 8))
            
            weight_c = idx_f - ti.cast(idx_lower, ti.f32)
            
            r = color_map[idx_lower, 0] * (1.0 - weight_c) + color_map[idx_upper, 0] * weight_c
            g = color_map[idx_lower, 1] * (1.0 - weight_c) + color_map[idx_upper, 1] * weight_c
            b = color_map[idx_lower, 2] * (1.0 - weight_c) + color_map[idx_upper, 2] * weight_c
            
            display_pixels[i, j] = [ti.cast(r, ti.u8), ti.cast(g, ti.u8), ti.cast(b, ti.u8)]

@ti.kernel
def update_vector_field(
    fx_view: ti.types.ndarray(), fy_view: ti.types.ndarray(),
    color_map: ti.types.ndarray(),
    min_log: ti.f32, max_log: ti.f32,
    spacing: ti.i32, view_w: ti.i32, view_h: ti.i32,
    length_scale: ti.f32, head_size: ti.f32, head_angle: ti.f32
):
    num_x = view_w // spacing
    num_y = view_h // spacing
    for i, j in ti.ndrange(num_x, num_y):
        idx = i * num_y + j
        px = i * spacing + spacing // 2
        py = j * spacing + spacing // 2
        
        # Grid sample tensor outputs mapped smoothly across the screen shape (VIEW_H, VIEW_W)
        f_x = fx_view[py, px]
        f_y = fy_view[py, px]
        f_mag = ti.sqrt(f_x**2 + f_y**2)
        
        # Exact colormap match from the pixel render kernel
        log_dens = ti.log(f_mag + 0.1) / 2.30258509299
        norm_dens = ti.max(0.0, ti.min((log_dens - min_log) / (max_log - min_log), 1.0))
        
        idx_f = norm_dens * 8.0
        idx_lower = ti.cast(ti.floor(idx_f), ti.i32)
        idx_upper = ti.max(0, ti.min(ti.cast(ti.ceil(idx_f), ti.i32), 8))
        weight_c = idx_f - ti.cast(idx_lower, ti.f32)
        
        r = color_map[idx_lower, 0] * (1.0 - weight_c) + color_map[idx_upper, 0] * weight_c
        g = color_map[idx_lower, 1] * (1.0 - weight_c) + color_map[idx_upper, 1] * weight_c
        b = color_map[idx_lower, 2] * (1.0 - weight_c) + color_map[idx_upper, 2] * weight_c
        c = [r / 255.0, g / 255.0, b / 255.0]
        
        # Direction Vectors
        dir_x, dir_y = 0.0, 0.0
        if f_mag > 1e-5:
            dir_x = f_x / f_mag
            dir_y = f_y / f_mag
            
        # Linear normalized magnitude mapping for arrow length
        max_val = ti.exp(max_log * 2.30258509299)
        min_val = ti.exp(min_log * 2.30258509299)
        linear_frac = ti.max(0.0, ti.min((f_mag - min_val) / ti.max(1e-5, max_val - min_val), 1.0))
        length = linear_frac * length_scale * spacing
        
        # Canvas space is [0, 1] mapped coordinate geometry
        sx = ti.cast(px, ti.f32) / view_w
        sy = ti.cast(py, ti.f32) / view_h
        
        ex = sx + (dir_x * length) / view_w
        ey = sy + (dir_y * length) / view_h
        
        # Trigonometry for the arrow head (wings)
        wing_len = length * head_size
        cos_a = ti.cos(head_angle)
        sin_a = ti.sin(head_angle)
        
        # Rotate direction vector backward by +head_angle and -head_angle
        wx1 = ex + (-dir_x * cos_a + dir_y * sin_a) * wing_len / view_w
        wy1 = ey + (-dir_x * sin_a - dir_y * cos_a) * wing_len / view_h
        wx2 = ex + (-dir_x * cos_a - dir_y * sin_a) * wing_len / view_w
        wy2 = ey + ( dir_x * sin_a - dir_y * cos_a) * wing_len / view_h
        
        base_idx = idx * 6
        
        # Line 1: Main Shaft
        vector_verts[base_idx] = [sx, sy]
        vector_verts[base_idx + 1] = [ex, ey]
        
        # Line 2: Left Wing
        vector_verts[base_idx + 2] = [ex, ey]
        vector_verts[base_idx + 3] = [wx1, wy1]
        
        # Line 3: Right Wing
        vector_verts[base_idx + 4] = [ex, ey]
        vector_verts[base_idx + 5] = [wx2, wy2]
        
        for k in ti.static(range(6)):
            vector_colors[base_idx + k] = c


class FFTGravitySystem:
    def __init__(self, initial_particles):
        self.device = DEVICE
        
        # Decouple potentials selection
        self.potential_type = 'gravity' # 'gravity', 'lennard_jones', 'harmonic'
        self.pressure_model = 'none' # 'none', 'isothermal', 'polytropic'
        self.boundary_mode = 'wrap_delete' # 'wrap_delete', 'reflective_walls'
        
        # Base sliders
        self.G = CONFIG["sliders"]["gravity"][2]
        self.gravity_range = CONFIG["sliders"]["gravity_range"][2] # Screens all mutual kernels
        self.friction = CONFIG["sliders"]["friction"][2]
        self.pressure = CONFIG["sliders"]["pressure"][2] # Fourier pressure sound speed
        self.softening_length = CONFIG["sliders"]["softening_length"][2]
        self.uniform_spawn = False 
        
        # Custom sliders
        sliders = CONFIG.get("sliders", {})
        self.spring_k = sliders.get("spring_k", [0.001, 10.0, 0.1])[2]
        self.epsilon = sliders.get("epsilon", [1.0, 5000.0, 1000.0])[2]
        self.sigma = sliders.get("sigma", [1.0, 200.0, 40.0])[2]
        self.pressure_strength = sliders.get("pressure", [0.01, 10000.0, 0.0])[2] # Default to 0 for real-space pressure
        self.gamma = sliders.get("polytropic_gamma", [0.0, 3.0, 1.0])[2]
        self.uniform_gravity_mag = sliders.get("uniform_gravity_mag", [0.0, 2000.0, 0.0])[2]
        self.uniform_gravity_angle = sliders.get("uniform_gravity_angle", [0.0, 360.0, 270.0])[2]
        
        self.show_fft = True
        self.use_bandpass_render = False
        
        self.show_potential = False 
        self.pot_min_log = CONFIG.get("sliders", {}).get("potential_min_log", [-5.0, 10.0, 2.0])[2]
        self.pot_max_log = CONFIG.get("sliders", {}).get("potential_max_log", [-5.0, 15.0, 6.0])[2]
        
        self.show_field = False
        self.field_min_log = CONFIG.get("sliders", {}).get("field_min_log", [-5.0, 10.0, 0.0])[2]
        self.field_max_log = CONFIG.get("sliders", {}).get("field_max_log", [-5.0, 10.0, 3.0])[2]

        self.bp_inner = 0.0
        self.bp_outer = 5.0
        self.bp_transition = 0.1
        
        self.custom_image_path = None
        self.custom_image_name = None
        
        self.num_active = initial_particles
        self.pos = torch.zeros((MAX_PARTICLES, 2), device=self.device)
        self.vel = torch.zeros((MAX_PARTICLES, 2), device=self.device)
        self.masses = torch.zeros(MAX_PARTICLES, dtype=torch.int8, device=self.device) 
        
        self.density_grid = torch.zeros((GRID_H, GRID_W), device=self.device)
        self.screen_y, self.screen_x = torch.meshgrid(
            torch.arange(VIEW_H, device=self.device, dtype=torch.float32), 
            torch.arange(VIEW_W, device=self.device, dtype=torch.float32), 
            indexing='ij'
        )
        
        init_cam_x = (UNIVERSE_W - WIDTH) / 2.0
        init_cam_y = (UNIVERSE_H - HEIGHT) / 2.0
        self.pos[:self.num_active, 0] = (torch.rand(self.num_active, device=self.device) * WIDTH) + init_cam_x
        self.pos[:self.num_active, 1] = (torch.rand(self.num_active, device=self.device) * HEIGHT) + init_cam_y
        self.vel[:self.num_active] = (torch.rand((self.num_active, 2), device=self.device) - 0.5) * 10.0
        self.masses[:self.num_active] = 1 
        
        self.color_theme = 0 
        self.theme_names = list(THEMES.keys())
        self.color_maps = [
            torch.tensor(THEMES[name], dtype=torch.float32, device=self.device) 
            for name in self.theme_names
        ]
        
        # Instantiate potentials.py manager
        self.pot_manager = PotentialsManager(GRID_W, GRID_H, PHYSICS_CELL_SIZE, self.device)
        
        kx = torch.fft.fftfreq(GRID_W, d=PHYSICS_CELL_SIZE, device=self.device) * 2 * math.pi
        ky = torch.fft.fftfreq(GRID_H, d=PHYSICS_CELL_SIZE, device=self.device) * 2 * math.pi
        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='xy')
        
        self.k_sq = kx_grid**2 + ky_grid**2
        self.k_sq[0, 0] = 1e-8 
        self.precompute_kernel()

    @torch.no_grad()
    def precompute_kernel(self):
        """
        Calculates the active interaction Fourier kernel from potentials.py dynamically.
        """
        params = {
            'G': self.G,
            'force_range': self.gravity_range,
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
            self.masses[:self.num_active] = torch.randint(0, 2, (self.num_active,), device=self.device, dtype=torch.int8) * 2 - 1
        else:
            self.masses[:self.num_active] = 1
        self.precompute_kernel()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def add_galaxy(self, world_x, world_y, world_radius, count=5000, vel_mult=0.0):
        count = int(count)
        if count <= 0 or self.num_active >= MAX_PARTICLES: return
        
        count = min(count, MAX_PARTICLES - self.num_active)
        start, end = self.num_active, self.num_active + count
        
        # Perform all spawner coordinate generation entirely on GPU to eliminate Host-to-Device transfer stalls
        r_rand = torch.rand(count, device=self.device)
        if self.uniform_spawn:
            r = torch.sqrt(r_rand) * world_radius
        else:
            r = r_rand * world_radius
            
        theta = torch.rand(count, device=self.device) * (2.0 * math.pi)
        
        # Write directly to the preallocated GPU slices in-place
        self.pos[start:end, 0] = world_x + r * torch.cos(theta)
        self.pos[start:end, 1] = world_y + r * torch.sin(theta)
        
        if self.potential_type == 'gravity':
            orbital_speed = torch.sqrt(abs(self.G) * 100.0 / (r + 10.0))
            self.vel[start:end, 0] = torch.sin(theta) * orbital_speed * vel_mult
            self.vel[start:end, 1] = -torch.cos(theta) * orbital_speed * vel_mult
        else:
            # Thermal random velocity directly in-place
            self.vel[start:end].uniform_(-0.5, 0.5)
            self.vel[start:end] *= (50.0 * vel_mult)
            
        if self.potential_type == 'coulomb':
            self.masses[start:end] = (torch.randint(0, 2, (count,), device=self.device, dtype=torch.int8) * 2 - 1)
        else:
            self.masses[start:end] = 1 
            
        self.num_active += count

    @torch.no_grad()
    def load_image_particles(self, file_path, cam_x, cam_y, zoom, spawn_density):
        try:
            from PIL import Image, ImageOps
            import numpy as np
            
            # Load image, apply EXIF transpose to maintain physical orientation
            img_raw = Image.open(file_path)
            img_oriented = ImageOps.exif_transpose(img_raw)
            img_rgb = img_oriented.convert('RGB')
            
            w_raw, h_raw = img_rgb.size
            
            # Calculate resized dimensions to fit within VIEW_W x VIEW_H while maintaining aspect ratio
            aspect_img = float(w_raw) / float(h_raw)
            aspect_screen = float(VIEW_W) / float(VIEW_H)
            
            if aspect_img > aspect_screen:
                target_w = VIEW_W
                target_h = int(VIEW_W / aspect_img)
            else:
                target_h = VIEW_H
                target_w = int(VIEW_H * aspect_img)
                
            target_w = max(1, target_w)
            target_h = max(1, target_h)
            
            bicubic_filter = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC
            img_resized = img_rgb.resize((target_w, target_h), bicubic_filter)
            img_data = np.array(img_resized, dtype=np.float32)
            intensities = np.mean(img_data, axis=2)
            
            col_start = (VIEW_W - target_w) // 2
            row_start = (VIEW_H - target_h) // 2
            
            cell_w = (WIDTH / zoom) / float(VIEW_W)
            cell_h = (HEIGHT / zoom) / float(VIEW_H)
            
            cols_mesh, rows_mesh = np.meshgrid(np.arange(target_w), np.arange(target_h))
            screen_cols_all = col_start + cols_mesh
            screen_rows_all = row_start + rows_mesh
            
            x_world_all = cam_x + (screen_cols_all + 0.5) * cell_w
            y_world_all = cam_y + (VIEW_H - screen_rows_all - 0.5) * cell_h
            
            in_grid_mask = (x_world_all >= 0.0) & (x_world_all < float(UNIVERSE_W)) & \
                           (y_world_all >= 0.0) & (y_world_all < float(UNIVERSE_H))
            
            expected_counts = (intensities / 255.0) * spawn_density * (RENDER_CELL_SIZE ** 2)
            expected_counts[~in_grid_mask] = 0.0
            
            floor_counts = np.floor(expected_counts).astype(np.int32)
            fractional_parts = expected_counts - floor_counts
            rand_vals = np.random.rand(*expected_counts.shape)
            cell_counts = floor_counts + (rand_vals < fractional_parts).astype(np.int32)
            
            r_indices, c_indices = np.where(cell_counts > 0)
            counts_to_spawn = cell_counts[r_indices, c_indices]
            
            total_spawn = int(np.sum(counts_to_spawn))
            if total_spawn == 0 or self.num_active >= MAX_PARTICLES:
                return
                
            available_slots = MAX_PARTICLES - self.num_active
            if total_spawn > available_slots:
                scale_factor = float(available_slots) / total_spawn
                counts_to_spawn = np.floor(counts_to_spawn * scale_factor).astype(np.int32)
                total_spawn = int(np.sum(counts_to_spawn))
                if total_spawn == 0:
                    return
            
            screen_cols = col_start + np.repeat(c_indices, counts_to_spawn)
            screen_rows = row_start + np.repeat(r_indices, counts_to_spawn)
            
            new_pos = np.zeros((total_spawn, 2), dtype=np.float32)
            new_pos[:, 0] = cam_x + (screen_cols + np.random.rand(total_spawn)) * cell_w
            new_pos[:, 1] = cam_y + (VIEW_H - screen_rows - 1 + np.random.rand(total_spawn)) * cell_h
            
            new_pos[:, 0] = np.clip(new_pos[:, 0], 0.0, float(UNIVERSE_W - 0.001))
            new_pos[:, 1] = np.clip(new_pos[:, 1], 0.0, float(UNIVERSE_H - 0.001))
            
            if self.potential_type == 'coulomb':
                new_masses = np.random.choice([-1, 1], size=total_spawn).astype(np.int8)
            else:
                new_masses = np.ones(total_spawn, dtype=np.int8)
            
            start, end = self.num_active, self.num_active + total_spawn
            self.pos[start:end] = torch.from_numpy(new_pos).to(self.device)
            self.vel[start:end].zero_()
            self.masses[start:end] = torch.from_numpy(new_masses).to(self.device)
            self.num_active += total_spawn
            print(f"Spawned {total_spawn} particles from image {file_path}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")

    def browse_custom_image(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            file_path = filedialog.askopenfilename(
                title="Select Astro-Image",
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp")]
            )
            root.destroy()
            
            if file_path:
                self.custom_image_path = file_path
                self.custom_image_name = os.path.basename(file_path)
                print(f"Selected custom image: {self.custom_image_name}")
        except Exception as e:
            print(f"Error browsing image: {e}")

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
        self.num_active = new_count

    @torch.no_grad()
    def step(self, dt, cam_x, cam_y, zoom):
        if self.num_active == 0: return

        # Handle Boundaries
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
                self.num_active = new_count

        if self.num_active == 0: return

        self.density_grid.zero_() 
        scatter_mass_kernel(self.pos, self.masses, self.density_grid, self.num_active, GRID_W, GRID_H, PHYSICS_CELL_SIZE)
        
        rho_k = torch.fft.rfft2(self.density_grid)
        k_sq_real = self.k_sq[:, :rho_k.shape[1]]
        k_mag = torch.sqrt(k_sq_real)
        
        bp_trans = max(self.bp_transition, 0.001)
        bandpass = torch.ones_like(k_mag)
        
        if self.bp_inner > 0.01:
            bandpass *= torch.sigmoid((k_mag - self.bp_inner) / bp_trans * 4.0)
            
        if self.bp_outer < 4.99:
            bandpass *= torch.sigmoid((self.bp_outer - k_mag) / bp_trans * 4.0)
            
        if self.show_fft:
            mag = torch.abs(rho_k) * bandpass
            mag_4d = mag.unsqueeze(0).unsqueeze(0)
            
            thumb_half = torch.nn.functional.interpolate(
                mag_4d, size=(MINIMAP_H, MINIMAP_W // 2), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            
            thumb_half = torch.log1p(thumb_half)
            thumb_half = torch.fft.fftshift(thumb_half, dim=0)
            thumb_full = torch.cat([torch.flip(thumb_half, dims=[1]), thumb_half], dim=1)
            
            thumb_max = thumb_full.max() + 1e-5
            self.fft_thumb_tensor = (thumb_full / thumb_max * 255.0).to(torch.uint8)
            
        # Calculate transform grid ONCE per frame if ANY advanced view is on to save massive performance
        if self.show_potential or self.use_bandpass_render or self.show_field:
            world_x = (self.screen_x / zoom) + cam_x
            world_y = (self.screen_y / zoom) + cam_y
            grid_x_norm = (world_x / UNIVERSE_W) * 2.0 - 1.0
            grid_y_norm = (world_y / UNIVERSE_H) * 2.0 - 1.0
            grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1).unsqueeze(0)

        # --- Mutual Potentials ---
        K_k_sliced = self.K_k[:, :rho_k.shape[1]]
        phi_k = rho_k * K_k_sliced
        
        # Traditional Fourier sound pressure (if present in gravity mode)
        if self.potential_type == 'gravity' and self.pressure > 0.01:
            phi_k = phi_k + rho_k * self.pressure
            
        # Force Softening (Plummer Gaussian)
        gaussian = torch.exp(-k_sq_real * (self.softening_length**2) / 2.0)
        phi_k.mul_(gaussian)
        
        phi = torch.fft.irfft2(phi_k, s=self.density_grid.shape).real
        fy, fx = torch.gradient(phi)
        
        fx.neg_().div_(PHYSICS_CELL_SIZE)
        fy.neg_().div_(PHYSICS_CELL_SIZE)
        
        # --- Real-Space Polytropic Pressure ---
        if self.pressure_model != 'none':
            gamma_val = self.gamma if self.pressure_model == 'polytropic' else 1.0
            force_px, force_py = self.pot_manager.compute_pressure_forces(self.density_grid, self.pressure_strength, gamma_val)
            fx += force_px
            fy += force_py
        
        # Taichi native particle gathering and integration
        gather_and_integrate_kernel(self.pos, self.vel, fx, fy, self.masses, self.num_active, GRID_W, GRID_H, PHYSICS_CELL_SIZE, dt, self.friction)

        # --- Uniform Gravity acceleration (applied directly on CUDA) ---
        if self.uniform_gravity_mag > 0:
            ugx, ugy = self.pot_manager.get_uniform_gravity_forces(self.uniform_gravity_mag, self.uniform_gravity_angle)
            self.vel[:self.num_active, 0] += ugx * dt
            self.vel[:self.num_active, 1] += ugy * dt

        # Dampen particles for LJ fluid pooling cooling
        if self.potential_type == 'lennard_jones':
            self.vel[:self.num_active] *= 0.99

        # --- Hardware Accelerated Density/Potential Renderer ---
        if self.show_potential:
            phi_view = torch.abs(phi).unsqueeze(0).unsqueeze(0)
            view_dens = torch.nn.functional.grid_sample(
                phi_view, grid, align_corners=False, padding_mode='zeros'
            ).squeeze(0).squeeze(0)
            view_dens = torch.clamp(view_dens, min=0.0)
            visual_density.from_torch(view_dens.transpose(0, 1).contiguous())
            
        elif self.use_bandpass_render:
            filtered_rho_k = rho_k * bandpass
            filtered_density = torch.fft.irfft2(filtered_rho_k, s=self.density_grid.shape).real
            view_dens = torch.nn.functional.grid_sample(
                filtered_density.unsqueeze(0).unsqueeze(0), 
                grid, align_corners=False, padding_mode='zeros'
            ).squeeze(0).squeeze(0)
            view_dens = torch.clamp(view_dens, min=0.0)
            visual_density.from_torch(view_dens.transpose(0, 1).contiguous())
        else:
            clear_visual_density()
            render_scatter_kernel(self.pos, self.num_active, cam_x, cam_y, zoom, VIEW_W, VIEW_H, RENDER_CELL_SIZE)

        # --- Hardware Accelerated Vector Field Sub-Renderer ---
        if self.show_field and TOTAL_ARROWS > 0:
            force_view = torch.stack((fx, fy), dim=0).unsqueeze(0)
            view_forces = torch.nn.functional.grid_sample(
                force_view, grid, align_corners=False, padding_mode='zeros'
            ).squeeze(0)
            self.view_fx = view_forces[0].contiguous()
            self.view_fy = view_forces[1].contiguous()

        active_map = self.color_maps[self.color_theme]
        if self.show_potential:
            render_color_kernel(active_map, self.pot_min_log, self.pot_max_log, 1.0)
        else:
            scale = 1.0
            if not self.use_bandpass_render:
                scale = float(max(1.0, zoom) ** ZOOM_INTENSITY_SCALE)
            render_color_kernel(active_map, MIN_LOG_DENSITY, MAX_LOG_DENSITY, scale)


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
    window = ti.ui.Window("PyTorch + Taichi Gravity", (WIDTH, HEIGHT), vsync=False, pos=(0, 0))
    canvas = window.get_canvas()
    gui = window.get_gui()

    system = FFTGravitySystem(initial_particles=1000000) # Start with 1,000,000 active masses
    init_arrow() 
    
    # --- Dynamic Camera & Interaction States ---
    ui_open = False      
    prev_lmb = False
    
    # Track drag states
    prev_z_lmb = False
    prev_z_rmb = False
    prev_pan_active = False
    prev_mouse_norm = (0.0, 0.0)
    
    # Track edge-triggered key toggles
    prev_right = False
    prev_left = False
    prev_p = False
    prev_f = False
    prev_r = False
    prev_m = False
    
    cam_x = (UNIVERSE_W - WIDTH) / 2.0
    cam_y = (UNIVERSE_H - HEIGHT) / 2.0
    zoom = 1.0

    s_grav = CONFIG["sliders"]["gravity"]
    s_grng = CONFIG["sliders"]["gravity_range"]
    s_fric = CONFIG["sliders"]["friction"]
    s_pres = CONFIG["sliders"]["pressure"]
    s_soft = CONFIG["sliders"]["softening_length"]
    s_time = CONFIG["sliders"]["time_scale_log"]
    s_rad  = CONFIG["sliders"]["brush_radius"]
    s_vel  = CONFIG["sliders"]["spawn_velocity"]
    s_dens = CONFIG["sliders"]["spawn_density"]
    s_bp_in  = CONFIG["sliders"]["bp_inner"]
    s_bp_out = CONFIG["sliders"]["bp_outer"]
    s_bp_w   = CONFIG["sliders"]["bp_transition"]
    
    s_gamma = CONFIG["sliders"].get("polytropic_gamma", [0.0, 3.0, 1.0])
    s_spring = CONFIG["sliders"].get("spring_k", [0.001, 10.0, 0.1])
    s_eps = CONFIG["sliders"].get("epsilon", [1.0, 5000.0, 1000.0])
    s_sig = CONFIG["sliders"].get("sigma", [1.0, 200.0, 40.0])
    s_ug_mag = CONFIG["sliders"].get("uniform_gravity_mag", [0.0, 2000.0, 0.0])
    s_ug_ang = CONFIG["sliders"].get("uniform_gravity_angle", [0.0, 360.0, 270.0])

    log_G_val = math.log10(s_grav[2])
    log_gravity_range_val = math.log10(s_grng[2])
    friction_val = s_fric[2]
    log_pressure_val = math.log10(s_pres[2])
    softening_val = s_soft[2]
    log_time_val = s_time[2]
    interaction_radius = s_rad[2]
    spawn_vel = s_vel[2]
    spawn_density = s_dens[2]

    last_time = time.time()
    frames = 0
    fps_display = 0.0
    target_frame_time = 1.0 / FPS_TARGET
    
    # Video Recording State Variables
    recording = False
    record_frames = 0

    while window.running:
        loop_start = time.time()
        dt = target_frame_time * (10 ** log_time_val)

        frames += 1
        current_time = time.time()
        if current_time - last_time >= 0.5: 
            fps_display = frames / (current_time - last_time)
            frames = 0
            last_time = current_time

        z_pressed = window.is_pressed('z')
        space_pressed = window.is_pressed(ti.ui.SPACE)
        
        right_pressed = window.is_pressed(ti.ui.RIGHT)
        left_pressed = window.is_pressed(ti.ui.LEFT)
        p_pressed = window.is_pressed('p')
        f_pressed = window.is_pressed('f')
        r_pressed = window.is_pressed('r')
        m_pressed = window.is_pressed('m')
        
        # Edge-Triggered Shortcuts
        if right_pressed and not prev_right:
            system.color_theme = (system.color_theme + 1) % len(system.theme_names)
        if left_pressed and not prev_left:
            system.color_theme = (system.color_theme - 1) % len(system.theme_names)
        if p_pressed and not prev_p:
            system.show_potential = not system.show_potential
        if f_pressed and not prev_f:
            system.show_field = not system.show_field
        if r_pressed and not prev_r:
            cam_x = (UNIVERSE_W - WIDTH) / 2.0
            cam_y = (UNIVERSE_H - HEIGHT) / 2.0
            zoom = 1.0
        if m_pressed and not prev_m:
            ui_open = not ui_open
            
        prev_right = right_pressed
        prev_left = left_pressed
        prev_p = p_pressed
        prev_f = f_pressed
        prev_r = r_pressed
        prev_m = m_pressed

        for e in window.get_events():
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == getattr(ti.ui, 'WHEEL', 'Wheel') or e.key == getattr(ti.GUI, 'WHEEL', 'Wheel'):
                zoom_delta = e.delta[1]
                if z_pressed:
                    mouse_norm = window.get_cursor_pos()
                    mouse_x, mouse_y = mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT
                    world_x = (mouse_x / zoom) + cam_x
                    world_y = (mouse_y / zoom) + cam_y
                    
                    zoom *= 1.1 if zoom_delta > 0 else (1.0 / 1.1)
                    zoom = max(0.0001, min(10000.0, zoom)) 
                    
                    cam_x = world_x - (mouse_x / zoom)
                    cam_y = world_y - (mouse_y / zoom)
                else:
                    interaction_radius *= 1.1 if zoom_delta > 0 else (1.0 / 1.1)
                    interaction_radius = max(s_rad[0], min(s_rad[1], interaction_radius))

        if window.is_pressed(ti.ui.UP):
            mouse_norm = window.get_cursor_pos()
            mouse_x, mouse_y = mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT
            world_x = (mouse_x / zoom) + cam_x
            world_y = (mouse_y / zoom) + cam_y
            zoom *= 1.02
            zoom = max(0.0001, min(10000.0, zoom))
            cam_x = world_x - (mouse_x / zoom)
            cam_y = world_y - (mouse_y / zoom)
            
        if window.is_pressed(ti.ui.DOWN):
            mouse_norm = window.get_cursor_pos()
            mouse_x, mouse_y = mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT
            world_x = (mouse_x / zoom) + cam_x
            world_y = (mouse_y / zoom) + cam_y
            zoom *= (1.0 / 1.02)
            zoom = max(0.0001, min(10000.0, zoom))
            cam_x = world_x - (mouse_x / zoom)
            cam_y = world_y - (mouse_y / zoom)

        mouse_norm = window.get_cursor_pos()
        mouse_x, mouse_y = mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT
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
        
        world_x = (mouse_x / zoom) + cam_x
        world_y = (mouse_y / zoom) + cam_y
        world_radius = interaction_radius / zoom

        if lmb_pressed and not prev_lmb:
            if not ui_open and mouse_norm[0] <= 0.04 and mouse_norm[1] >= 0.95:
                ui_open = True
        
        is_hovering_ui = False
        if ui_open:
            if mouse_norm[0] <= UI_W: is_hovering_ui = True
        else:
            if mouse_norm[0] <= 0.04 and mouse_norm[1] >= 0.95: is_hovering_ui = True
            
        is_modifying_view = z_pressed or space_pressed or pan_active
        
        if lmb_pressed and not is_modifying_view and not is_hovering_ui:
            brush_area = math.pi * (interaction_radius ** 2)
            spawn_count = max(1, int(brush_area * spawn_density))
            system.add_galaxy(world_x, world_y, world_radius, count=spawn_count, vel_mult=spawn_vel)
            
        if rmb_pressed and not is_modifying_view and not is_hovering_ui:
            system.remove_particles(world_x, world_y, world_radius)

        prev_lmb = lmb_pressed

        system.step(dt, cam_x, cam_y, zoom)

        canvas.set_background_color((0.02, 0.02, 0.03)) 
        
        if ui_open and system.show_fft and hasattr(system, 'fft_thumb_tensor'):
            overlay_fft_pip(system.fft_thumb_tensor.contiguous(), 0, 0, MINIMAP_W, MINIMAP_H)
            
        if system.num_active > 0:
            canvas.set_image(display_pixels)
            
        # Draw dynamic vector field exactly over the image (if activated)
        if system.show_field and TOTAL_ARROWS > 0:
            update_vector_field(
                system.view_fx, system.view_fy,
                system.color_maps[system.color_theme],
                system.field_min_log, system.field_max_log,
                PIXELS_PER_ARROW, VIEW_W, VIEW_H,
                ARROW_LENGTH_SCALE, ARROW_HEAD_SIZE, ARROW_HEAD_ANGLE
            )
            canvas.lines(vector_verts, width=ARROW_WIDTH, per_vertex_color=vector_colors)

        update_brush_ring(mouse_norm[0], mouse_norm[1], interaction_radius / float(WIDTH), interaction_radius / float(HEIGHT))
        if rmb_pressed and not is_hovering_ui and not is_modifying_view:
            canvas.circles(ring_pos_field, radius=0.001, color=(1.0, 0.2, 0.2))
        else:
            canvas.circles(ring_pos_field, radius=0.001, color=(0.2, 0.8, 0.6))

        # --- Movie Recording Frame Capture ---
        if recording:
            filename = f"temp_frames/frame_{record_frames:05d}.png"
            window.save_image(filename)
            record_frames += 1

        if not ui_open:
            canvas.triangles(arrow_verts, color=(0.8, 0.8, 0.8))
        else:
            panel_height = 1.0 - (MINIMAP_H / float(HEIGHT)) if system.show_fft else 1.0
            gui.begin("Simulation Controls", 0.0, 0.0, UI_W, panel_height)
            
            if gui.button("<< Close Panel"): ui_open = False
                
            gui.text("") 
            gui.text("--- SYSTEM STATS ---")
            gui.text(f"FPS: {fps_display:.1f} / {FPS_TARGET:.0f}")
            gui.text(f"Total Stars: {system.num_active} / {MAX_PARTICLES}")
            
            # Movie recording widgets
            if recording:
                gui.text(f"REC ACTIVE: {record_frames} frames")
                if gui.button("STOP & COMPRESS"):
                    recording = False
                    compile_video(record_frames, output_name="simulation.mp4", fps=60)
                    record_frames = 0
            else:
                if gui.button("RECORD 60 FPS MOVIE"):
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
            if gui.button("Force: Gravity"):
                system.set_potential_type('gravity')
            if gui.button("Force: Lennard-Jones"):
                system.set_potential_type('lennard_jones')
            if gui.button("Force: Mutual Springs"):
                system.set_potential_type('harmonic')
            if gui.button("Force: Coulomb"):
                system.set_potential_type('coulomb')
                
            gui.text("")
            gui.text("--- GAS PRESSURE ---")
            gui.text(f"Selected: {system.pressure_model.upper()}")
            if gui.button("Gas: Cold Dust (None)"):
                system.pressure_model = 'none'
            if gui.button("Gas: Isothermal (Linear)"):
                system.pressure_model = 'isothermal'
            if gui.button("Gas: Polytropic (Adiabatic)"):
                system.pressure_model = 'polytropic'
                
            gui.text("")
            gui.text("--- VISUALS ---")
            gui.text(f"Theme: {system.theme_names[system.color_theme]}")
            if gui.button("Next Color Theme"): system.color_theme = (system.color_theme + 1) % len(system.theme_names)
            
            gui.text("")
            gui.text("--- VIEW SETTINGS ---")
            system.pot_min_log = gui.slider_float("Potential Min (Log)", system.pot_min_log, -5.0, 15.0)
            system.pot_max_log = gui.slider_float("Potential Max (Log)", system.pot_max_log, -5.0, 15.0)
            system.field_min_log = gui.slider_float("Field Min (Log)", system.field_min_log, -5.0, 15.0)
            system.field_max_log = gui.slider_float("Field Max (Log)", system.field_max_log, -5.0, 15.0)

            gui.text("")
            gui.text("--- ENGINE MODIFIERS ---")
            
            # Universal Force Range (Screening length)
            old_fr = system.gravity_range
            system.gravity_range = gui.slider_float("Force Range", system.gravity_range, s_grng[0], s_grng[1])
            if old_fr != system.gravity_range:
                system.precompute_kernel()
                
            # Gas pressure sliders
            system.pressure_strength = gui.slider_float("Gas Pressure P", system.pressure_strength, s_pres[0], s_pres[1])
            
            # Polytropic index gamma
            if system.pressure_model == 'polytropic':
                system.gamma = gui.slider_float("Poly Index (Gamma)", system.gamma, s_gamma[0], s_gamma[1])
                
            # Boundary selector
            bound_text = "SOLID WALL BOUNCE" if system.boundary_mode == 'reflective_walls' else "UNIVERSE WRAPPED"
            if gui.button(f"Boundary: {bound_text}"):
                if system.boundary_mode == 'reflective_walls':
                    system.boundary_mode = 'wrap_delete'
                else:
                    system.boundary_mode = 'reflective_walls'
            
            # Dampening cooling
            friction_val = gui.slider_float("Friction (Gas Cooling)", friction_val, s_fric[0], s_fric[1])
            system.friction = friction_val
            
            # Force Softening
            softening_val = gui.slider_float("Force Softening (Plummer)", softening_val, s_soft[0], s_soft[1])
            system.softening_length = softening_val

            log_time_val = gui.slider_float("Time Scale (Log)", log_time_val, s_time[0], s_time[1])
            
            gui.text("")
            gui.text("--- POTENTIAL COEFFICIENTS ---")
            if system.potential_type in ('gravity', 'coulomb'):
                label = "Coulomb Const (Log)" if system.potential_type == 'coulomb' else "Gravity (Log)"
                log_G_val = gui.slider_float(label, log_G_val, math.log10(s_grav[0]), math.log10(s_grav[1]))
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
            spawn_vel = gui.slider_float("Spawn Velocity (CW)", spawn_vel, s_vel[0], s_vel[1])
            spawn_density = gui.slider_float("Density (stars/px^2)", spawn_density, s_dens[0], s_dens[1])
            
            spawn_type_text = "Uniform Disk" if system.uniform_spawn else "Dense Core"
            gui.text(f"Spawn Mode: {spawn_type_text}")
            if gui.button("Toggle Spawn Mode"): system.uniform_spawn = not system.uniform_spawn

            gui.text("")
            gui.text("--- LOAD ASTRO-IMAGE ---")
            if gui.button("Browse Image (Explorer)..."):
                system.browse_custom_image()
                
            if system.custom_image_path and system.custom_image_name:
                if gui.button(f"Spawn {system.custom_image_name}"):
                    system.load_image_particles(system.custom_image_path, cam_x, cam_y, zoom, spawn_density)
                    
            import glob
            img_list = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
                img_list.extend(glob.glob(ext))
                
            img_list = [name for name in img_list if name != system.custom_image_name]
            
            for img_name in img_list:
                if gui.button(f"Spawn {img_name}"):
                    system.load_image_particles(img_name, cam_x, cam_y, zoom, spawn_density)

            gui.text("")
            gui.text("--- BANDPASS & FFT RADAR ---")
            
            show_fft_text = "Compute FFT Minimap: ON" if system.show_fft else "Compute FFT Minimap: OFF"
            if gui.button(show_fft_text): system.show_fft = not system.show_fft
                
            bp_render_text = "Apply Filter to Main Render: ON" if system.use_bandpass_render else "Apply Filter to Main Render: OFF"
            if gui.button(bp_render_text): system.use_bandpass_render = not system.use_bandpass_render
            
            system.bp_inner = gui.slider_float("High-Pass (Inner)", system.bp_inner, 0.0, 5.0)
            system.bp_outer = gui.slider_float("Low-Pass (Outer)", system.bp_outer, 0.0, 5.0)
            system.bp_transition = gui.slider_float("Filter Softness", system.bp_transition, 0.001, 1.0)
            
            gui.text("")
            gui.text("--- CRASH-PROOF CONTROLS ---")
            gui.text("LMB: Spawn | RMB: Delete")
            gui.text("Space+LMB Drag: Pan View")
            gui.text("Z+LMB Drag (or Z+Scroll): Zoom")
            gui.text("Z+RMB Drag (or Scroll): Brush Size")
            gui.text("P: Toggle Potential View")
            gui.text("F: Toggle Vector Field")
            gui.text("R: Reset View")
            gui.text("M: Toggle Controls Panel")
            gui.text("Left/Right Arrow: Color Theme")
            
            gui.text("")
            gui.text("--- DEBUG READOUT ---")
            gui.text(f"Zoom Level: {zoom:.2f}x")
            
            if gui.button("Reset View"):
                cam_x = (UNIVERSE_W - WIDTH) / 2.0
                cam_y = (UNIVERSE_H - HEIGHT) / 2.0
                zoom = 1.0
            
            gui.end()

        window.show()
        
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_frame_time:
            time.sleep(target_frame_time - loop_elapsed)

if __name__ == "__main__":
    main()