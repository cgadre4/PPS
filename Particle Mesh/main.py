import taichi as ti

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
from particle_mesh.render.overlays import BrushOverlay
from particle_mesh.ui.input import InputController, KeyEdges  # noqa: F401  (KeyEdges re-exported)
from particle_mesh.ui.panel import SimulationGUI
from particle_mesh.ui.state import Controls

def main():
    # Setup Taichi GGUI Window
    window = ti.ui.Window("PyTorch + Taichi Flexible Physical Sandbox", (WIDTH, HEIGHT), vsync=False, pos=(0, 0))
    canvas = window.get_canvas()
    imgui = window.get_gui()

    # Brush ring and "show panel" arrow overlays
    brush = BrushOverlay()

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
    # Brush, spawn, time-scale and panel state shared by input handling and the panel
    controls = Controls.from_settings()
    sim_gui = SimulationGUI(engine, camera, CONFIG, palettes, controls)
    controller = InputController(window, camera, WIDTH, HEIGHT, SETTINGS.slider("brush_radius")[:2])

    # Performance Stats
    last_time = time.time()
    frames = 0
    fps_display = 0.0
    target_frame_time = 1.0 / FPS_TARGET

    while window.running:
        loop_start = time.time()
        
        # Calculate active time delta based on time scale log slider
        dt = target_frame_time * (10 ** controls.log_time)

        # FPS calculation
        frames += 1
        current_time = time.time()
        if current_time - last_time >= 0.5:
            fps_display = frames / (current_time - last_time)
            frames = 0
            last_time = current_time

        # Keys, camera, panel toggles and this frame's brush action
        intent = controller.process(engine, palettes, controls, sim_gui.ui_w)
        if intent.spawn:
            engine.add_galaxy(intent.world_x, intent.world_y, intent.world_radius, count=intent.spawn_count,
                              vel_mult=controls.spawn_vel)
        if intent.erase:
            engine.remove_particles(intent.world_x, intent.world_y, intent.world_radius)

        # Step Simulation
        engine.step(dt)

        # Render background
        canvas.set_background_color((0.02, 0.02, 0.03))

        # Render frame
        active_map = palettes.get(engine.color_theme)
        
        if engine.num_active > 0:
            if engine.solver_type == 'nbody':
                # Circles coloured by speed (or charge in Coulomb mode), computed on the sim device
                circle_renderer.render_nbody(canvas, engine, camera, active_map)
            elif engine.solver_type == 'pm':
                # Density / potential / band-pass view, FFT minimap and force arrows (see MeshRenderer.render_pm)
                mesh_renderer.render_pm(canvas, engine, camera, active_map, sim_gui.ui_w, view=engine.view)

        # Draw brush ring overlay
        brush.draw_ring(canvas, intent.mouse_norm, controls.radius, WIDTH, HEIGHT, erasing=intent.erase)

        # Draw Escape panel indicator if hidden
        if not controls.ui_open:
            brush.draw_arrow(canvas)
        else:
            # Render imgui panels
            sim_gui.render(imgui, fps_display)

        window.show()

        # Throttle FPS
        loop_elapsed = time.time() - loop_start
        if loop_elapsed < target_frame_time:
            time.sleep(target_frame_time - loop_elapsed)


if __name__ == "__main__":
    main()
