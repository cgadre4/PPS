"""
Composition root: builds the window, engine, renderers and UI, and runs the frame loop.

Frame: input -> brush spawn/erase -> engine.step(dt) -> render (PM mesh or N-body circles) -> overlays -> ImGui
panel -> window.show() -> FPS throttle (run() only). dt = (1 / FPS_TARGET) * 10**controls.log_time.
"""
import time

import taichi as ti

from particle_mesh.physics.engine import SimulationEngine
from particle_mesh.render.camera import SimulationCamera
from particle_mesh.render.circle_renderer import CircleRenderer
from particle_mesh.render.colormap import Palettes
from particle_mesh.render.mesh_renderer import MeshRenderer
from particle_mesh.render.overlays import BrushOverlay
from particle_mesh.render.themes import THEMES
from particle_mesh.runtime import bootstrap
from particle_mesh.settings import CONFIG, SETTINGS
from particle_mesh.ui.input import InputController
from particle_mesh.ui.panel import SimulationGUI
from particle_mesh.ui.state import Controls
from particle_mesh.ui.wheel import attach_wheel

WINDOW_TITLE = "PyTorch + Taichi Flexible Physical Sandbox"
BACKGROUND = (0.02, 0.02, 0.03)


class App:
    def __init__(self, window, engine, camera, palettes, controls, controller, panel, brush, circle_renderer,
                 mesh_renderer, settings=SETTINGS):
        self.window = window
        self.canvas = window.get_canvas()
        self.imgui = window.get_gui()
        self.engine = engine
        self.camera = camera
        self.palettes = palettes
        self.controls = controls
        self.controller = controller
        self.panel = panel
        self.brush = brush
        self.circle_renderer = circle_renderer
        self.mesh_renderer = mesh_renderer
        self.width = settings.width
        self.height = settings.height
        self.target_frame_time = 1.0 / settings.fps_target
        # FPS readout, refreshed every 0.5 s
        self.fps_display = 0.0
        self._fps_frames = 0
        self._fps_since = time.time()

    def frame(self):
        """One frame, without the FPS throttle."""
        engine, canvas, controls = self.engine, self.canvas, self.controls

        # Calculate active time delta based on time scale log slider
        dt = self.target_frame_time * (10 ** controls.log_time)

        # FPS calculation
        self._fps_frames += 1
        now = time.time()
        if now - self._fps_since >= 0.5:
            self.fps_display = self._fps_frames / (now - self._fps_since)
            self._fps_frames = 0
            self._fps_since = now

        # Keys, camera, panel toggles and this frame's brush action
        intent = self.controller.process(engine, self.palettes, controls, self.panel.ui_w)
        if intent.spawn:
            engine.add_galaxy(intent.world_x, intent.world_y, intent.world_radius, count=intent.spawn_count,
                              vel_mult=controls.spawn_vel)
        if intent.erase:
            engine.remove_particles(intent.world_x, intent.world_y, intent.world_radius)

        engine.step(dt)

        canvas.set_background_color(BACKGROUND)
        active_map = self.palettes.get(engine.color_theme)
        if engine.num_active > 0:
            if engine.solver_type == 'nbody':
                # Circles coloured by speed (or charge in Coulomb mode), computed on the sim device
                self.circle_renderer.render_nbody(canvas, engine, self.camera, active_map)
            elif engine.solver_type == 'pm':
                # Density / potential / band-pass view, FFT minimap and force arrows (see MeshRenderer.render_pm)
                self.mesh_renderer.render_pm(canvas, engine, self.camera, active_map, self.panel.ui_w, view=engine.view)

        # Overlays, then the panel (or the "show panel" arrow while it is hidden)
        self.brush.draw_ring(canvas, intent.mouse_norm, controls.radius, self.width, self.height, erasing=intent.erase)
        if not controls.ui_open:
            self.brush.draw_arrow(canvas)
        else:
            self.panel.render(self.imgui, self.fps_display)

        self.window.show()

    def run(self):
        while self.window.running:
            loop_start = time.time()
            self.frame()
            # Throttle FPS
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < self.target_frame_time:
                time.sleep(self.target_frame_time - loop_elapsed)


def build_app(solver=None, window_factory=None, engine_factory=None, settings=SETTINGS):
    """
    Wires the app together. Requires bootstrap.init_taichi() first (the renderers and overlays create Taichi fields).
    window_factory defaults to ti.ui.Window, looked up at call time so test harnesses can patch it.
    """
    window_factory = window_factory or ti.ui.Window
    engine_factory = engine_factory or SimulationEngine
    window = window_factory(WINDOW_TITLE, (settings.width, settings.height), vsync=False, pos=(0, 0))
    # Mouse wheel from the Windows message loop (GGUI has none); None elsewhere or if the window isn't found
    wheel = attach_wheel(WINDOW_TITLE)
    brush = BrushOverlay()

    # Startup count and solver from the configuration (the engine caps N-body startup counts)
    engine = engine_factory(solver=solver or settings.solver)
    bootstrap.print_status(engine)
    engine.precompute_kernel()

    # Colour themes on the sim device; the active index is engine.color_theme (engine.view)
    palettes = Palettes(THEMES, engine.device)
    camera = SimulationCamera(settings.width, settings.height, engine.universe_w, engine.universe_h)
    circle_renderer = CircleRenderer()
    mesh_renderer = MeshRenderer(engine.kernels)
    # Brush, spawn, time-scale and panel state shared by input handling and the panel
    controls = Controls.from_settings(settings)
    panel = SimulationGUI(engine, camera, CONFIG, palettes, controls)
    controller = InputController(window, camera, settings.width, settings.height, settings.slider("brush_radius")[:2],
                                 wheel=wheel.take if wheel is not None else None)
    app = App(window, engine, camera, palettes, controls, controller, panel, brush, circle_renderer, mesh_renderer,
              settings)
    app.wheel = wheel
    return app


def main(solver=None, window_factory=None, engine_factory=None):
    bootstrap.configure_torch_env()
    bootstrap.init_taichi()
    build_app(solver, window_factory, engine_factory).run()
