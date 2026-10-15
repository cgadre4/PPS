import os
# Premium hardware memory allocation configuration (prevents VRAM fragmentation on Windows CUDA)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import taichi as ti
import torch
import math
import time
import json

# --- Load Configuration ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Global Display Settings
WIDTH = CONFIG["display"]["WIDTH"]
HEIGHT = CONFIG["display"]["HEIGHT"]
FPS_TARGET = CONFIG["display"]["FPS_TARGET"]
SOLVER_MODE = CONFIG["simulation"].get("SOLVER", "pm").lower()

# Initialize Taichi BEFORE importing renderers or GUI (since they declare fields at module-import time)
def _init_taichi():
    preferred = CONFIG.get("simulation", {}).get("TAICHI_BACKEND", None)
    arch_map = {
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "opengl": ti.opengl,
        "cpu": ti.cpu
    }
    candidates = []
    if preferred and str(preferred).lower() in arch_map:
        candidates.append(arch_map[str(preferred).lower()])
    
    # Priority hierarchy:
    # If CUDA is detected, try CUDA first.
    # Otherwise (Intel Arc / Panther Lake B390 / AMD), try Vulkan GPU first!
    if torch.cuda.is_available():
        gpu_order = [ti.cuda, ti.vulkan, ti.opengl, ti.cpu]
    else:
        gpu_order = [ti.vulkan, ti.opengl, ti.cuda, ti.cpu]
        
    for arch in gpu_order:
        if arch not in candidates:
            candidates.append(arch)
            
    for arch in candidates:
        try:
            if arch == ti.cuda:
                ti.init(arch=arch, device_memory_fraction=0.15)
            else:
                ti.init(arch=arch)
            
            actual_arch = ti.lang.impl.current_cfg().arch
            # Taichi internally falls back to CPU (x64) without raising an exception if CUDA driver is missing
            if arch != ti.cpu and actual_arch == ti.cpu:
                print(f"[Taichi] Requested {arch} but driver fell back to CPU. Trying next backend...")
                ti.reset()
                continue
                
            print(f"[Taichi] Successfully initialized GPU backend: {actual_arch}")
            return actual_arch
        except Exception as e:
            print(f"[Taichi] Failed to initialize {arch}: {e}")
            continue
            
    raise RuntimeError("[Taichi] Failed to initialize any backend")

_init_taichi()

from colors import THEMES
from engine import SimulationEngine
from camera import SimulationCamera
from circle_renderer import CircleRenderer
from mesh_renderer import MeshRenderer
from gui import SimulationGUI

# Setup GGUI Fields for brush rings and arrow signs
ring_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=64)
arrow_verts = ti.Vector.field(2, dtype=ti.f32, shape=6)

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
        angle = float(i) / 64.0 * 2.0 * math.pi
        ring_pos_field[i] = [center_x + ti.cos(angle) * radius_x, center_y + ti.sin(angle) * radius_y]


def main():
    # Setup Taichi GGUI Window
    window = ti.ui.Window("PyTorch + Taichi Flexible Physical Sandbox", (WIDTH, HEIGHT), vsync=False, pos=(0, 0))
    canvas = window.get_canvas()
    gui = window.get_gui()

    init_arrow()

    # Determine safe startup active particle count from configuration
    initial_particles = CONFIG["simulation"].get("initial_particles", 1000000)
    if SOLVER_MODE == 'nbody' and initial_particles > 20000:
        print(f"[Warning] initial_particles ({initial_particles}) is extremely high for O(N^2) Direct N-Body mode.")
        print("          Automatically capping to 10,000 to prevent system hang. Lower it in config.json if desired.")
        initial_particles = 10000
    
    # Initialize state engine and override solver mode from configuration
    engine = SimulationEngine(initial_particles=initial_particles)
    engine.solver_type = SOLVER_MODE

    print("=" * 65)
    print("                    HARDWARE & COMPUTE STATUS                     ")
    print("=" * 65)
    print(f"  Taichi Backend:   {ti.lang.impl.current_cfg().arch} (GPU Accelerated)")
    print(f"  PyTorch Device:   {engine.device}")
    print(f"  PyTorch Version:  {torch.__version__}")
    xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"  Intel XPU Active: {xpu_ok}")
    if not xpu_ok and engine.device.type == "cpu":
        print("  [NOTE] PyTorch is performing FFTs on CPU cores.")
        print(f"         Grid size: {CONFIG['simulation'].get('PHYSICS_GRID_W', 7680)}x{CONFIG['simulation'].get('PHYSICS_GRID_H', 7680)} cells.")
        print("         Consider reducing PHYSICS_GRID_W/H to 1024 or 2048 if CPU usage is too high.")
    print("=" * 65)
    
    # Pre-calculate active potentials
    engine.precompute_kernel()
    
    # Setup themes directly on engine
    engine.color_theme = 0
    engine.theme_names = list(THEMES.keys())
    engine.color_maps = [
        torch.tensor(THEMES[name], dtype=torch.float32, device=engine.device) 
        for name in engine.theme_names
    ]

    # Initialize Camera, Renderers and GUI controllers
    camera = SimulationCamera(WIDTH, HEIGHT, engine.universe_w, engine.universe_h)
    circle_renderer = CircleRenderer()
    mesh_renderer = MeshRenderer()
    sim_gui = SimulationGUI(engine, camera, CONFIG)

    # Spawner Brush settings
    interaction_radius = CONFIG["sliders"]["brush_radius"][2]
    spawn_vel = CONFIG["sliders"]["spawn_velocity"][2]
    spawn_density = CONFIG["sliders"]["spawn_density"][2]

    # Key Edge-trigger tracking
    prev_right = False
    prev_left = False
    prev_p = False
    prev_f = False
    prev_r = False
    prev_m = False
    prev_lmb = False

    # Drag tracking states
    prev_pan_active = False
    prev_z_lmb = False
    prev_z_rmb = False
    prev_mouse_norm = (0.0, 0.0)

    # Panel visibilities
    ui_open = True

    # Performance Stats
    last_time = time.time()
    frames = 0
    fps_display = 0.0
    target_frame_time = 1.0 / FPS_TARGET

    while window.running:
        loop_start = time.time()
        
        # Calculate active time delta based on time scale log slider
        dt = target_frame_time * (10 ** sim_gui.log_time_val)

        # FPS calculation
        frames += 1
        current_time = time.time()
        if current_time - last_time >= 0.5:
            fps_display = frames / (current_time - last_time)
            frames = 0
            last_time = current_time

        # Capture key states
        z_pressed = window.is_pressed('z')
        space_pressed = window.is_pressed(ti.ui.SPACE)
        right_pressed = window.is_pressed(ti.ui.RIGHT)
        left_pressed = window.is_pressed(ti.ui.LEFT)
        p_pressed = window.is_pressed('p')
        f_pressed = window.is_pressed('f')
        r_pressed = window.is_pressed('r')
        m_pressed = window.is_pressed('m')

        # Keyboard Toggles (Edge-Triggered)
        if right_pressed and not prev_right:
            engine.color_theme = (engine.color_theme + 1) % len(engine.theme_names)
        if left_pressed and not prev_left:
            engine.color_theme = (engine.color_theme - 1) % len(engine.theme_names)
        if p_pressed and not prev_p and engine.solver_type == 'pm':
            engine.show_potential = not engine.show_potential
        if f_pressed and not prev_f and engine.solver_type == 'pm':
            engine.show_field = not engine.show_field
        if r_pressed and not prev_r:
            camera.reset()
        if m_pressed and not prev_m:
            ui_open = not ui_open

        prev_right = right_pressed
        prev_left = left_pressed
        prev_p = p_pressed
        prev_f = f_pressed
        prev_r = r_pressed
        prev_m = m_pressed

        # Handle mouse scroll events
        mouse_norm = window.get_cursor_pos()
        for e in window.get_events():
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == getattr(ti.ui, 'WHEEL', 'Wheel') or e.key == getattr(ti.GUI, 'WHEEL', 'Wheel'):
                zoom_delta = e.delta[1]
                if z_pressed:
                    camera.zoom_around_mouse(mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT, 1.1 if zoom_delta > 0 else (1.0 / 1.1))
                else:
                    interaction_radius *= 1.1 if zoom_delta > 0 else (1.0 / 1.1)
                    interaction_radius = max(CONFIG["sliders"]["brush_radius"][0], min(CONFIG["sliders"]["brush_radius"][1], interaction_radius))

        # Keyboard Pan/Zoom fallbacks
        if window.is_pressed(ti.ui.UP):
            camera.zoom_around_mouse(mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT, 1.02)
        if window.is_pressed(ti.ui.DOWN):
            camera.zoom_around_mouse(mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT, 1.0 / 1.02)

        # Mouse Click/Drag events
        lmb_pressed = window.is_pressed(ti.ui.LMB)
        rmb_pressed = window.is_pressed(ti.ui.RMB)
        mmb_pressed = window.is_pressed(ti.ui.MMB)

        # Panning Action
        pan_active = (lmb_pressed and space_pressed) or mmb_pressed
        if pan_active:
            if prev_pan_active:
                dx = (mouse_norm[0] - prev_mouse_norm[0]) * WIDTH
                dy = (mouse_norm[1] - prev_mouse_norm[1]) * HEIGHT
                camera.pan(dx, dy)
        prev_pan_active = pan_active

        # Zoom Dragting Action
        z_lmb_active = lmb_pressed and z_pressed
        if z_lmb_active:
            if prev_z_lmb:
                dy = mouse_norm[1] - prev_mouse_norm[1]
                zoom_factor = math.exp(dy * 5.0)
                # Zoom relative to viewport center
                camera.zoom_around_mouse(WIDTH / 2.0, HEIGHT / 2.0, zoom_factor)
        prev_z_lmb = z_lmb_active

        # Brush Resize Action
        z_rmb_active = rmb_pressed and z_pressed
        if z_rmb_active:
            if prev_z_rmb:
                dy = mouse_norm[1] - prev_mouse_norm[1]
                interaction_radius *= math.exp(dy * 5.0)
                interaction_radius = max(CONFIG["sliders"]["brush_radius"][0], min(CONFIG["sliders"]["brush_radius"][1], interaction_radius))
        prev_z_rmb = z_rmb_active

        prev_mouse_norm = mouse_norm

        # Protect GUI panel area from clicks
        is_hovering_ui = False
        if ui_open:
            if mouse_norm[0] <= sim_gui.ui_w:
                is_hovering_ui = True
        else:
            if mouse_norm[0] <= 0.04 and mouse_norm[1] >= 0.95:
                is_hovering_ui = True

        if lmb_pressed and not prev_lmb:
            if not ui_open and mouse_norm[0] <= 0.04 and mouse_norm[1] >= 0.95:
                ui_open = True

        # Translate cursor to true simulation coordinates
        world_x, world_y = camera.screen_to_world(mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT)
        world_radius = interaction_radius / camera.zoom

        # Spawning and Deleting Particles
        is_modifying_view = z_pressed or space_pressed or pan_active
        if lmb_pressed and not is_modifying_view and not is_hovering_ui:
            brush_area = math.pi * (interaction_radius ** 2)
            spawn_count = max(1, int(brush_area * spawn_density))
            engine.add_galaxy(world_x, world_y, world_radius, count=spawn_count, vel_mult=spawn_vel)
        if rmb_pressed and not is_modifying_view and not is_hovering_ui:
            engine.remove_particles(world_x, world_y, world_radius)

        prev_lmb = lmb_pressed

        # Step Simulation
        engine.step(dt)

        # Render background
        canvas.set_background_color((0.02, 0.02, 0.03))

        # Render frame
        active_map = engine.color_maps[engine.color_theme]
        
        if engine.num_active > 0:
            if engine.solver_type == 'nbody':
                # Dynamically color circles based on speed/charges in PyTorch VRAM
                if engine.potential_type == 'coulomb':
                    charges = engine.masses[:engine.num_active]
                    color_tensor = torch.zeros((engine.num_active, 3), dtype=torch.float32, device=engine.device)
                    color_tensor[charges > 0] = torch.tensor([255.0, 50.0, 50.0], device=engine.device)
                    color_tensor[charges < 0] = torch.tensor([50.0, 150.0, 255.0], device=engine.device)
                else:
                    speed = torch.norm(engine.vel[:engine.num_active], dim=1)
                    max_speed = 50.0
                    norm_speed = torch.clamp(speed / max_speed, 0.0, 1.0)
                    idx_f = norm_speed * 8.0
                    idx_lower = torch.clamp(idx_f.long(), 0, 7)
                    idx_upper = torch.clamp(idx_lower + 1, 0, 8)
                    weight = (idx_f - idx_lower.float()).unsqueeze(1)
                    color_tensor = active_map[idx_lower] * (1.0 - weight) + active_map[idx_upper] * weight

                circle_renderer.render(
                    canvas,
                    engine.pos[:engine.num_active],
                    color_tensor,
                    engine.num_active,
                    camera.cam_x, camera.cam_y,
                    camera.width, camera.height,
                    0.0005 # Visual radius
                )
                
            elif engine.solver_type == 'pm':
                # Perform grid sampling on GPU before rendering advanced views
                if engine.show_potential or engine.use_bandpass_render or engine.show_field:
                    grid_w = int(camera.width / CONFIG["simulation"]["RENDER_CELL_SIZE"])
                    grid_h = int(camera.height / CONFIG["simulation"]["RENDER_CELL_SIZE"])
                    
                    screen_y, screen_x = torch.meshgrid(
                        torch.arange(grid_h, device=engine.device, dtype=torch.float32), 
                        torch.arange(grid_w, device=engine.device, dtype=torch.float32), 
                        indexing='ij'
                    )
                    world_gx = (screen_x / camera.zoom) + camera.cam_x
                    world_gy = (screen_y / camera.zoom) + camera.cam_y
                    
                    grid_x_norm = (world_gx / engine.universe_w) * 2.0 - 1.0
                    grid_y_norm = (world_gy / engine.universe_h) * 2.0 - 1.0
                    grid_transform = torch.stack((grid_x_norm, grid_y_norm), dim=-1).unsqueeze(0)

                if engine.show_potential:
                    phi_view = torch.abs(engine.pm_solver.phi).unsqueeze(0).unsqueeze(0)
                    view_dens = torch.nn.functional.grid_sample(
                        phi_view, grid_transform, align_corners=False, padding_mode='zeros'
                    ).squeeze(0).squeeze(0)
                    view_dens = torch.clamp(view_dens, min=0.0)
                    
                    mesh_renderer.render_grid(
                        canvas, view_dens, active_map, 
                        engine.pot_min_log, engine.pot_max_log, 1.0
                    )
                elif engine.use_bandpass_render:
                    # Filter densities on the fly
                    rho_k = torch.fft.rfft2(engine.pm_solver.density_grid)
                    k_sq_real = engine.pm_solver.k_sq[:, :rho_k.shape[1]]
                    k_mag = torch.sqrt(k_sq_real)
                    
                    bp_trans = max(engine.bp_transition, 0.001)
                    bandpass = torch.ones_like(k_mag)
                    if engine.bp_inner > 0.01:
                        bandpass *= torch.sigmoid((k_mag - engine.bp_inner) / bp_trans * 4.0)
                    if engine.bp_outer < 4.99:
                        bandpass *= torch.sigmoid((engine.bp_outer - k_mag) / bp_trans * 4.0)
                        
                    filtered_rho_k = rho_k * bandpass
                    filtered_density = torch.fft.irfft2(filtered_rho_k, s=engine.pm_solver.density_grid.shape).real
                    
                    view_dens = torch.nn.functional.grid_sample(
                        filtered_density.unsqueeze(0).unsqueeze(0), 
                        grid_transform, align_corners=False, padding_mode='zeros'
                    ).squeeze(0).squeeze(0)
                    view_dens = torch.clamp(view_dens, min=0.0)
                    
                    mesh_renderer.render_grid(
                        canvas, view_dens, active_map,
                        CONFIG["simulation"]["MIN_LOG_DENSITY"], CONFIG["simulation"]["MAX_LOG_DENSITY"], 1.0
                    )
                else:
                    mesh_renderer.render(
                        canvas,
                        engine.pos,
                        engine.num_active,
                        camera.cam_x, camera.cam_y,
                        camera.zoom,
                        active_map,
                        CONFIG["simulation"]["MIN_LOG_DENSITY"],
                        CONFIG["simulation"]["MAX_LOG_DENSITY"],
                        CONFIG["simulation"]["ZOOM_INTENSITY_SCALE"]
                    )

                # Overlay vector field
                if engine.show_field:
                    force_view = torch.stack((engine.pm_solver.fx, engine.pm_solver.fy), dim=0).unsqueeze(0)
                    view_forces = torch.nn.functional.grid_sample(
                        force_view, grid_transform, align_corners=False, padding_mode='zeros'
                    ).squeeze(0)
                    
                    mesh_renderer.render_vectors(
                        canvas, view_forces[0].contiguous(), view_forces[1].contiguous(), active_map,
                        engine.field_min_log, engine.field_max_log,
                        CONFIG["display"].get("arrow_length_scale", 1.5),
                        CONFIG["display"].get("arrow_head_size", 0.25),
                        CONFIG["display"].get("arrow_head_angle", 0.5),
                        CONFIG["display"].get("arrow_width", 0.001)
                    )

                # Overlay FFT PIP radar
                if engine.show_fft:
                    rho_k = torch.fft.rfft2(engine.pm_solver.density_grid)
                    k_sq_real = engine.pm_solver.k_sq[:, :rho_k.shape[1]]
                    k_mag = torch.sqrt(k_sq_real)
                    
                    bp_trans = max(engine.bp_transition, 0.001)
                    bandpass = torch.ones_like(k_mag)
                    if engine.bp_inner > 0.01:
                        bandpass *= torch.sigmoid((k_mag - engine.bp_inner) / bp_trans * 4.0)
                    if engine.bp_outer < 4.99:
                        bandpass *= torch.sigmoid((engine.bp_outer - k_mag) / bp_trans * 4.0)
                        
                    mag = torch.abs(rho_k) * bandpass
                    mag_4d = mag.unsqueeze(0).unsqueeze(0)
                    
                    minimap_w = int(camera.width * sim_gui.ui_w)
                    minimap_h = int(camera.width * sim_gui.ui_w)
                    
                    thumb_half = torch.nn.functional.interpolate(
                        mag_4d, size=(minimap_h, minimap_w // 2), mode='bilinear', align_corners=False
                    ).squeeze(0).squeeze(0)
                    
                    thumb_half = torch.log1p(thumb_half)
                    thumb_half = torch.fft.fftshift(thumb_half, dim=0)
                    thumb_full = torch.cat([torch.flip(thumb_half, dims=[1]), thumb_half], dim=1)
                    
                    thumb_max = thumb_full.max() + 1e-5
                    fft_thumb = (thumb_full / thumb_max * 255.0).to(torch.uint8)
                    
                    mesh_renderer.overlay_fft(fft_thumb.contiguous(), 0, 0, minimap_w, minimap_h)

        # Draw brush ring overlay
        update_brush_ring(mouse_norm[0], mouse_norm[1], interaction_radius / float(WIDTH), interaction_radius / float(HEIGHT))
        if rmb_pressed and not is_hovering_ui and not is_modifying_view:
            canvas.circles(ring_pos_field, radius=0.001, color=(1.0, 0.2, 0.2))
        else:
            canvas.circles(ring_pos_field, radius=0.001, color=(0.2, 0.8, 0.6))

        # Draw Escape panel indicator if hidden
        if not ui_open:
            canvas.triangles(arrow_verts, color=(0.8, 0.8, 0.8))
        else:
            # Render imgui panels
            interaction_radius, spawn_vel, spawn_density = sim_gui.render(
                gui, fps_display, interaction_radius, spawn_vel, spawn_density
            )

        window.show()

        # Throttle FPS
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_frame_time:
            time.sleep(target_frame_time - loop_elapsed)


if __name__ == "__main__":
    main()
