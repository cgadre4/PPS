"""InputController / KeyEdges / Controls / panel API, driven by a scripted fake window (no GGUI)."""
import math
from types import SimpleNamespace

import pytest
import taichi as ti

from particle_mesh.render.camera import SimulationCamera
from particle_mesh.render.colormap import Palettes
from particle_mesh.render.themes import THEMES
from particle_mesh.settings import SETTINGS
from particle_mesh.ui.input import KEY_ZOOM_RATE, MAX_KEY_DT, WHEEL_ZOOM_STEP, InputController, KeyEdges
from particle_mesh.ui.state import Controls

W, H = 1000, 500
LIMITS = SETTINGS.slider("brush_radius")[:2]


class FakeWindow:
    def __init__(self):
        self.held = set()
        self.cursor = (0.5, 0.5)
        self.events = []
        self.running = True

    def is_pressed(self, key):
        return key in self.held

    def get_cursor_pos(self):
        return self.cursor

    def get_events(self):
        events, self.events = self.events, []
        return events


class FakeEngine:
    def __init__(self, solver="pm"):
        self.solver_type = solver
        self.color_theme = 0
        self.show_potential = False
        self.show_field = False


class FakeClock:
    """Scripted frame clock: every call advances by `step` seconds (120 FPS by default)."""

    def __init__(self, step=1.0 / 120.0):
        self.t = 0.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


@pytest.fixture
def rig():
    window = FakeWindow()
    camera = SimulationCamera(W, H, 8192, 8192)
    wheel = SimpleNamespace(notches=0.0)

    def take():
        n, wheel.notches = wheel.notches, 0.0
        return n

    clock = FakeClock()
    controller = InputController(window, camera, W, H, LIMITS, wheel=take, clock=clock)
    controls = Controls.from_settings()
    engine = FakeEngine()
    palettes = SimpleNamespace(cycle=lambda i, step=1: (i + step) % 5)

    def frame(held=(), cursor=None, ui_w=0.15, notches=0.0):
        window.held = set(held)
        if cursor is not None:
            window.cursor = cursor
        wheel.notches += notches
        return controller.process(engine, palettes, controls, ui_w)

    return SimpleNamespace(window=window, camera=camera, controls=controls, engine=engine, frame=frame,
                           clock=clock, wheel=wheel)


def test_key_edges_rising_edge_only():
    w = FakeWindow()
    k = KeyEdges(w, ('a',))
    w.held = {'a'}
    k.update()
    assert k.pressed('a') and k.held('a')
    k.update()
    assert not k.pressed('a') and k.held('a')
    w.held = set()
    k.update()
    assert not k.pressed('a') and not k.held('a')


def test_toggles_fire_once_per_press(rig):
    rig.frame(held={'p', 'f', ti.ui.RIGHT})
    rig.frame(held={'p', 'f', ti.ui.RIGHT})
    assert rig.engine.show_potential and rig.engine.show_field and rig.engine.color_theme == 1
    rig.frame(held={ti.ui.LEFT})
    assert rig.engine.color_theme == 0


def test_pm_only_toggles_ignored_in_nbody(rig):
    rig.engine.solver_type = "nbody"
    rig.frame(held={'p', 'f'})
    assert not rig.engine.show_potential and not rig.engine.show_field


def test_m_toggles_panel_and_arrow_click_reopens(rig):
    rig.frame(held={'m'})
    assert not rig.controls.ui_open
    intent = rig.frame(held=(), cursor=(0.02, 0.97))
    assert intent.hovering_ui                                   # the arrow is a UI element while the panel is hidden
    rig.frame(held={ti.ui.LMB}, cursor=(0.02, 0.97))
    assert rig.controls.ui_open


def test_escape_stops_window(rig):
    rig.window.events = [SimpleNamespace(key=ti.ui.ESCAPE)]
    intent = rig.frame()
    assert rig.window.running is False and intent is not None   # the frame still completes


@pytest.mark.parametrize("pan_keys", [{ti.ui.SPACE, ti.ui.LMB}, {ti.ui.MMB}])
def test_pan_drag(rig, pan_keys):
    ref = SimulationCamera(W, H, 8192, 8192)
    rig.frame(held=pan_keys, cursor=(0.5, 0.5))                # first frame only latches
    assert (rig.camera.cam_x, rig.camera.cam_y) == (ref.cam_x, ref.cam_y)
    intent = rig.frame(held=pan_keys, cursor=(0.6, 0.4))
    ref.pan((0.6 - 0.5) * W, (0.4 - 0.5) * H)
    assert (rig.camera.cam_x, rig.camera.cam_y) == pytest.approx((ref.cam_x, ref.cam_y))
    assert intent.modifying_view and not intent.spawn


def test_ctrl_lmb_zoom_about_drag_start(rig):
    ref = SimulationCamera(W, H, 8192, 8192)
    rig.frame(held={ti.ui.CTRL, ti.ui.LMB}, cursor=(0.3, 0.5))       # press latches the anchor
    intent = rig.frame(held={ti.ui.CTRL, ti.ui.LMB}, cursor=(0.3, 0.6))
    rig.frame(held={ti.ui.CTRL, ti.ui.LMB}, cursor=(0.35, 0.7))     # anchor stays where the drag started
    ref.zoom_around_mouse(0.3 * W, 0.5 * H, math.exp((0.6 - 0.5) * 5.0))
    ref.zoom_around_mouse(0.3 * W, 0.5 * H, math.exp((0.7 - 0.6) * 5.0))
    assert rig.camera.zoom == pytest.approx(ref.zoom)
    assert (rig.camera.cam_x, rig.camera.cam_y) == pytest.approx((ref.cam_x, ref.cam_y))
    assert intent.modifying_view and not intent.spawn                   # Ctrl+LMB never spawns


def test_ctrl_rmb_resizes_brush_with_clamp(rig):
    r0 = rig.controls.radius
    z0 = rig.camera.zoom
    rig.frame(held={ti.ui.CTRL, ti.ui.RMB}, cursor=(0.5, 0.5))
    intent = rig.frame(held={ti.ui.CTRL, ti.ui.RMB}, cursor=(0.5, 0.55))
    assert rig.controls.radius == pytest.approx(min(LIMITS[1], r0 * math.exp(0.05 * 5.0)))
    assert not intent.erase and rig.camera.zoom == z0                   # Ctrl+RMB never erases or zooms
    rig.frame(held={ti.ui.CTRL, ti.ui.RMB}, cursor=(0.5, 5.0))
    assert rig.controls.radius == LIMITS[1]
    rig.frame(held={ti.ui.CTRL, ti.ui.RMB}, cursor=(0.5, -5.0))
    assert rig.controls.radius == LIMITS[0]


def test_z_no_longer_steers_the_view(rig):
    z0, r0 = rig.camera.zoom, rig.controls.radius
    rig.frame(held={'z', ti.ui.LMB}, cursor=(0.5, 0.5))
    intent = rig.frame(held={'z', ti.ui.LMB}, cursor=(0.5, 0.6))
    assert rig.camera.zoom == z0 and rig.controls.radius == r0 and intent.spawn


def test_spawn_and_erase_respect_panel(rig):
    assert not rig.frame(held={ti.ui.LMB}, cursor=(0.1, 0.5), ui_w=0.15).spawn      # over the panel
    intent = rig.frame(held={ti.ui.LMB}, cursor=(0.5, 0.5), ui_w=0.15)
    assert intent.spawn and not intent.erase
    c = rig.controls
    assert intent.spawn_count == max(1, int(math.pi * c.radius ** 2 * c.spawn_density))
    assert intent.world_radius == pytest.approx(c.radius / rig.camera.zoom)
    assert (intent.world_x, intent.world_y) == pytest.approx(rig.camera.screen_to_world(0.5 * W, 0.5 * H))
    intent = rig.frame(held={ti.ui.RMB}, cursor=(0.5, 0.5))
    assert intent.erase and not intent.spawn
    assert not rig.frame(held={ti.ui.RMB, ti.ui.CTRL}, cursor=(0.5, 0.5)).erase      # Ctrl steers the brush size


def test_zoom_out_stops_at_the_mesh_size(rig):
    """The view can never span more than the physics mesh along either axis (drag or arrow keys)."""
    cam = rig.camera
    assert cam.min_zoom == pytest.approx(max(W / 8192, H / 8192))
    for _ in range(5):
        rig.frame(held={ti.ui.CTRL, ti.ui.LMB}, cursor=(0.5, 1.0))
        rig.frame(held={ti.ui.CTRL, ti.ui.LMB}, cursor=(0.5, 0.0))
        rig.frame()
    assert cam.zoom == pytest.approx(cam.min_zoom)
    assert W / cam.zoom <= 8192 + 1e-6 and H / cam.zoom <= 8192 + 1e-6
    for _ in range(50):
        rig.frame(held={ti.ui.DOWN})
    assert cam.zoom == pytest.approx(cam.min_zoom)
    for _ in range(5):
        rig.frame(held={ti.ui.CTRL, ti.ui.LMB}, cursor=(0.5, 0.0))
        rig.frame(held={ti.ui.CTRL, ti.ui.LMB}, cursor=(0.5, 1.0))
        rig.frame()
    assert cam.zoom == cam.max_zoom


def test_camera_ignores_bad_zoom_factors():
    cam = SimulationCamera(W, H, 8192, 8192)
    before = (cam.cam_x, cam.cam_y, cam.zoom)
    for factor in (0.0, -2.0, float("nan"), float("inf")):
        cam.zoom_around_mouse(10.0, 10.0, factor)
    assert (cam.cam_x, cam.cam_y, cam.zoom) == before


def test_reset_respects_the_zoom_limit():
    cam = SimulationCamera(W, H, 400, 400)                 # mesh smaller than the window
    assert cam.zoom == cam.min_zoom == pytest.approx(W / 400)
    assert cam.cam_x == pytest.approx(0.0) and cam.cam_y == pytest.approx((400 - H / cam.zoom) / 2)


def test_keyboard_zoom(rig):
    z0 = rig.camera.zoom
    rig.frame(held={ti.ui.UP})
    assert rig.camera.zoom == pytest.approx(z0 * 1.02)            # 1.02x per frame at 120 FPS, as before
    rig.frame(held={ti.ui.DOWN})
    assert rig.camera.zoom == pytest.approx(z0)


@pytest.mark.parametrize("fps", [30.0, 120.0, 240.0])
def test_keyboard_zoom_is_frame_rate_independent(rig, fps):
    """One second of holding Up zooms by the same factor at any frame rate (it was 1.02x per frame)."""
    rig.clock.step = 1.0 / fps
    rig.frame()                                                    # latch the clock
    z0 = rig.camera.zoom
    for _ in range(int(fps)):
        rig.frame(held={ti.ui.UP})
    assert rig.camera.zoom == pytest.approx(z0 * math.exp(KEY_ZOOM_RATE))


def test_keyboard_zoom_ignores_long_stalls(rig):
    rig.frame()
    z0 = rig.camera.zoom
    rig.clock.step = 5.0                                           # a 5 s hitch must not jump the zoom
    rig.frame(held={ti.ui.UP})
    assert rig.camera.zoom == pytest.approx(z0 * math.exp(KEY_ZOOM_RATE * MAX_KEY_DT))


def test_wheel_zooms_towards_the_cursor(rig):
    ref = SimulationCamera(W, H, 8192, 8192)
    intent = rig.frame(cursor=(0.7, 0.3), notches=2.0)
    ref.zoom_around_mouse(0.7 * W, 0.3 * H, WHEEL_ZOOM_STEP ** 2)
    assert rig.camera.zoom == pytest.approx(ref.zoom)
    assert (rig.camera.cam_x, rig.camera.cam_y) == pytest.approx((ref.cam_x, ref.cam_y))
    # the world point under the cursor stays put, and the brush maps through the new zoom
    assert (intent.world_x, intent.world_y) == pytest.approx(rig.camera.screen_to_world(0.7 * W, 0.3 * H))
    rig.frame(cursor=(0.7, 0.3), notches=-2.0)
    assert rig.camera.zoom == pytest.approx(SimulationCamera(W, H, 8192, 8192).zoom)
    assert not intent.spawn and not intent.erase


def test_wheel_over_the_panel_does_not_zoom(rig):
    z0 = rig.camera.zoom
    rig.frame(cursor=(0.05, 0.5), notches=3.0)                     # over the panel: ImGui scrolls instead
    rig.frame(cursor=(0.6, 0.5))                                   # the notches were consumed, not deferred
    assert rig.camera.zoom == z0


def test_wheel_respects_zoom_limits(rig):
    rig.frame(cursor=(0.6, 0.5), notches=-100.0)
    assert rig.camera.zoom == pytest.approx(rig.camera.min_zoom)
    rig.frame(cursor=(0.6, 0.5), notches=1000.0)
    assert rig.camera.zoom == rig.camera.max_zoom


def test_r_resets_camera(rig):
    rig.camera.pan(100.0, 50.0)
    rig.frame(held={'r'})
    ref = SimulationCamera(W, H, 8192, 8192)
    assert (rig.camera.cam_x, rig.camera.cam_y, rig.camera.zoom) == (ref.cam_x, ref.cam_y, ref.zoom)


class FakeImGui:
    """Records slider labels and returns scripted values; buttons in `press` report a click."""

    def __init__(self, values, press=()):
        self.values = values
        self.press = set(press)
        self.sliders = []

    def begin(self, *a):
        pass

    def end(self):
        pass

    def text(self, *a):
        pass

    def button(self, label):
        return label in self.press

    def slider_float(self, label, value, lo, hi):
        self.sliders.append(label)
        return self.values.get(label, value)


def test_panel_writes_controls_in_place(make_engine):
    from particle_mesh.render.camera import SimulationCamera as Cam
    from particle_mesh.settings import CONFIG
    from particle_mesh.ui.panel import SimulationGUI
    eng = make_engine(0)
    controls = Controls.from_settings()
    gui = SimulationGUI(eng, Cam(W, H, eng.universe_w, eng.universe_h), CONFIG, Palettes(THEMES, eng.device),
                        controls, scan=lambda: [])
    imgui = FakeImGui({"Brush Radius": 123.0, "Spawn Velocity (x orbit)": -1.0, "Density (stars/px^2)": 0.5,
                       "Time Scale (Log)": 1.5})
    assert gui.render(imgui, 60.0) is None
    assert (controls.radius, controls.spawn_vel, controls.spawn_density, controls.log_time) == (123.0, -1.0, 0.5, 1.5)
    assert gui.image_files == []


def _panel(make_engine):
    from particle_mesh.settings import CONFIG
    from particle_mesh.ui.panel import SimulationGUI
    eng = make_engine(0)
    cam = SimulationCamera(W, H, eng.universe_w, eng.universe_h)
    gui = SimulationGUI(eng, cam, CONFIG, Palettes(THEMES, eng.device), Controls.from_settings(), scan=lambda: [])
    return gui, cam


def test_panel_zoom_slider_and_buttons(make_engine):
    from particle_mesh.ui.panel import PANEL_ZOOM_STEP
    gui, cam = _panel(make_engine)
    centre = gui._scene_centre()
    assert centre == pytest.approx(((gui.ui_w + (1 - gui.ui_w) / 2) * W, H / 2))
    world_before = cam.screen_to_world(*centre)
    gui.render(FakeImGui({"Zoom (log2)": 3.0}), 60.0)
    assert cam.zoom == pytest.approx(8.0)
    assert cam.screen_to_world(*centre) == pytest.approx(world_before)     # zooms about the scene centre
    gui.render(FakeImGui({}, press={"Zoom In"}), 60.0)
    assert cam.zoom == pytest.approx(8.0 * PANEL_ZOOM_STEP)
    gui.render(FakeImGui({}, press={"Zoom Out"}), 60.0)
    assert cam.zoom == pytest.approx(8.0)
    gui.render(FakeImGui({"Zoom (log2)": -100.0}), 60.0)
    assert cam.zoom == pytest.approx(cam.min_zoom)


def test_untouched_panel_zoom_slider_never_drifts(make_engine):
    """ImGui returns the slider value as a float32; an untouched slider must leave the camera alone."""
    import numpy as np
    gui, cam = _panel(make_engine)
    cam.zoom_around_mouse(123.0, 45.0, 1.2345678)
    before = (cam.cam_x, cam.cam_y, cam.zoom)

    class Float32Gui(FakeImGui):
        def slider_float(self, label, value, lo, hi):
            return float(np.float32(value))

    for _ in range(20):
        gui.render(Float32Gui({}), 60.0)
    assert (cam.cam_x, cam.cam_y, cam.zoom) == before
