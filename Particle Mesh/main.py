import taichi as ti
import torch

import math
import time

from particle_mesh.settings import CONFIG, SETTINGS

# Global Display Settings
WIDTH = SETTINGS.width
HEIGHT = SETTINGS.height
FPS_TARGET = SETTINGS.fps_target
SOLVER_MODE = SETTINGS.solver

from particle_mesh.runtime import bootstrap

bootstrap.configure_torch_env()
# Taichi is initialised when this module is imported (tests and tools rely on `import main` doing it).
_init_taichi = bootstrap.init_taichi
_init_taichi()

from particle_mesh.render.themes import THEMES
from particle_mesh.render.colormap import Palettes
from particle_mesh.physics.engine import SimulationEngine
from particle_mesh.render.camera import SimulationCamera
from particle_mesh.render.circle_renderer import CircleRenderer
from particle_mesh.render.mesh_renderer import MeshRenderer
from particle_mesh.ui.panel import SimulationGUI

# Brush ring and "show panel" arrow overlays. The fields are created in main() (after ti.init) and passed in.
@ti.kernel
def init_arrow(arrow_verts: ti.template()):
    arrow_verts[0] = [0.016, 0.985]
    arrow_verts[1] = [0.016, 0.965]
    arrow_verts[2] = [0.006, 0.975]
    arrow_verts[3] = [0.028, 0.985]
    arrow_verts[4] = [0.028, 0.965]
    arrow_verts[5] = [0.018, 0.975]

@ti.kernel
def update_brush_ring(ring_pos_field: ti.template(), center_x: ti.f32, center_y: ti.f32, radius_x: ti.f32, radius_y: ti.f32):
    for i in range(ring_pos_field.shape[0]):
        angle = float(i) / ring_pos_field.shape[0] * 2.0 * math.pi
        ring_pos_field[i] = [center_x + ti.cos(angle) * radius_x, center_y + ti.sin(angle) * radius_y]


class KeyEdges:
    """Per-frame key/button state with rising-edge detection (replaces hand-rolled prev_* flags)."""

    def __init__(self, window, keys):
        self.window = window
        self.keys = tuple(keys)
        self.now = dict.fromkeys(self.keys, False)
        self.before = dict(self.now)

    def update(self):
        self.before = self.now
        self.now = {k: bool(self.window.is_pressed(k)) for k in self.keys}

    def held(self, key):
        return self.now[key]

    def pressed(self, key):
        """True only on the frame the key goes down."""
        return self.now[key] and not self.before[key]



def main():
    # Setup Taichi GGUI Window
    window = ti.ui.Window("PyTorch + Taichi Flexible Physical Sandbox", (WIDTH, HEIGHT), vsync=False, pos=(0, 0))
    canvas = window.get_canvas()
    gui = window.get_gui()

    ring_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=64)
    arrow_verts = ti.Vector.field(2, dtype=ti.f32, shape=6)
    init_arrow(arrow_verts)

    # Startup count and solver from the configuration (the engine caps N-body startup counts)
    engine = SimulationEngine(solver=SOLVER_MODE)

    bootstrap.print_status(engine)
    
    # Pre-calculate active potentials
    engine.precompute_kernel()
    
    # Colour themes on the sim device; the active index is engine.color_theme (engine.view)
    palettes = Palettes(THEMES, engine.device)

    # Initialize Camera, Renderers and GUI controllers
    camera = SimulationCamera(WIDTH, HEIGHT, engine.universe_w, engine.universe_h)
    circle_renderer = CircleRenderer()
    mesh_renderer = MeshRenderer(engine.kernels)
    sim_gui = SimulationGUI(engine, camera, CONFIG, palettes)

    # N-body Coulomb circle colours (created once, not every frame)
    positive_color = torch.tensor([255.0, 50.0, 50.0], device=engine.device)
    negative_color = torch.tensor([50.0, 150.0, 255.0], device=engine.device)

    # Spawner Brush settings
    interaction_radius = SETTINGS.slider("brush_radius")[2]
    spawn_vel = SETTINGS.slider("spawn_velocity")[2]
    spawn_density = SETTINGS.slider("spawn_density")[2]

    # Keyboard / mouse button state with edge detection
    keys = KeyEdges(window, ('z', 'p', 'f', 'r', 'm', ti.ui.SPACE, ti.ui.RIGHT, ti.ui.LEFT, ti.ui.UP, ti.ui.DOWN,
                             ti.ui.LMB, ti.ui.RMB, ti.ui.MMB))

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
        keys.update()
        z_pressed = keys.held('z')
        space_pressed = keys.held(ti.ui.SPACE)

        # Keyboard Toggles (Edge-Triggered)
        if keys.pressed(ti.ui.RIGHT):
            engine.color_theme = palettes.cycle(engine.color_theme, 1)
        if keys.pressed(ti.ui.LEFT):
            engine.color_theme = palettes.cycle(engine.color_theme, -1)
        if keys.pressed('p') and engine.solver_type == 'pm':
            engine.show_potential = not engine.show_potential
        if keys.pressed('f') and engine.solver_type == 'pm':
            engine.show_field = not engine.show_field
        if keys.pressed('r'):
            camera.reset()
        if keys.pressed('m'):
            ui_open = not ui_open

        # Key events. (Taichi 1.7 GGUI reports no mouse-wheel events, so zoom and brush size use drags/keys.)
        mouse_norm = window.get_cursor_pos()
        for e in window.get_events():
            if e.key == ti.ui.ESCAPE:
                window.running = False

        # Keyboard Pan/Zoom fallbacks
        if keys.held(ti.ui.UP):
            camera.zoom_around_mouse(mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT, 1.02)
        if keys.held(ti.ui.DOWN):
            camera.zoom_around_mouse(mouse_norm[0] * WIDTH, mouse_norm[1] * HEIGHT, 1.0 / 1.02)

        # Mouse Click/Drag events
        lmb_pressed = keys.held(ti.ui.LMB)
        rmb_pressed = keys.held(ti.ui.RMB)
        mmb_pressed = keys.held(ti.ui.MMB)

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
                interaction_radius = max(SETTINGS.slider("brush_radius")[0], min(SETTINGS.slider("brush_radius")[1], interaction_radius))
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

        if keys.pressed(ti.ui.LMB):
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

        # Step Simulation
        engine.step(dt)

        # Render background
        canvas.set_background_color((0.02, 0.02, 0.03))

        # Render frame
        active_map = palettes.get(engine.color_theme)
        
        if engine.num_active > 0:
            if engine.solver_type == 'nbody':
                # Dynamically color circles based on speed/charges in PyTorch VRAM
                if engine.potential_type == 'coulomb':
                    positive = (engine.masses[:engine.num_active] > 0).unsqueeze(1)
                    color_tensor = torch.where(positive, positive_color, negative_color)
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
                    camera.width / camera.zoom, camera.height / camera.zoom, # Visible world extent
                    0.0005 # Visual radius
                )
                
            elif engine.solver_type == 'pm':
                # Density / potential / band-pass view, FFT minimap and force arrows (see MeshRenderer.render_pm)
                mesh_renderer.render_pm(canvas, engine, camera, active_map, sim_gui.ui_w)

        # Draw brush ring overlay
        update_brush_ring(ring_pos_field, mouse_norm[0], mouse_norm[1], interaction_radius / float(WIDTH), interaction_radius / float(HEIGHT))
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
