"""InputController / KeyEdges / Controls / panel API, driven by a scripted fake window (no GGUI)."""
import math
from types import SimpleNamespace

import pytest
import taichi as ti

from particle_mesh.render.camera import SimulationCamera
from particle_mesh.render.colormap import Palettes
from particle_mesh.render.themes import THEMES
from particle_mesh.settings import SETTINGS
from particle_mesh.ui.input import InputController, KeyEdges
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


@pytest.fixture
def rig():
    window = FakeWindow()
    camera = SimulationCamera(W, H, 8192, 8192)
    controller = InputController(window, camera, W, H, LIMITS)
    controls = Controls.from_settings()
    engine = FakeEngine()
    palettes = SimpleNamespace(cycle=lambda i, step=1: (i + step) % 5)

    def frame(held=(), cursor=None, ui_w=0.15):
        window.held = set(held)
        if cursor is not None:
            window.cursor = cursor
        return controller.process(engine, palettes, controls, ui_w)

    return SimpleNamespace(window=window, camera=camera, controls=controls, engine=engine, frame=frame)


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
    assert rig.camera.zoom == pytest.approx(z0 * 1.02)
    rig.frame(held={ti.ui.DOWN})
    assert rig.camera.zoom == pytest.approx(z0)


def test_r_resets_camera(rig):
    rig.camera.pan(100.0, 50.0)
    rig.frame(held={'r'})
    ref = SimulationCamera(W, H, 8192, 8192)
    assert (rig.camera.cam_x, rig.camera.cam_y, rig.camera.zoom) == (ref.cam_x, ref.cam_y, ref.zoom)


class FakeImGui:
    """Records slider labels and returns scripted values; buttons are never pressed."""

    def __init__(self, values):
        self.values = values
        self.sliders = []

    def begin(self, *a):
        pass

    def end(self):
        pass

    def text(self, *a):
        pass

    def button(self, label):
        return False

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
