"""
Scripted end-to-end action test: drives the real App.frame (input handling, engine, renderers, ImGui panel) through
every user action and the extremes of every slider, checking after each frame that nothing crashed and the state is
sane. Used by test_actions.py (one subprocess per scenario: GGUI allows one window per process).

    uv run python tests/headless_actions.py <pm|nbody> [--particles N] [--visible] [--quick]

--visible opens a real window and presents every frame (window.show()); otherwise the window is offscreen.
--quick runs fewer frames per step (the pytest default). Prints one JSON line with the result; exit code 1 on any
crash or broken invariant.

Input is scripted per step: keys/buttons held, the cursor (moved linearly over the step for drags), panel buttons
pressed and panel sliders set (by label; "min"/"max" pick the slider's own bounds). The panel runs on a proxy of the
real ImGui object, so its widgets are drawn and its own code paths run.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from types import SimpleNamespace

PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PM_DIR)
os.chdir(PM_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("solver", choices=["pm", "nbody"])
parser.add_argument("--particles", type=int, default=None)
parser.add_argument("--visible", action="store_true")
parser.add_argument("--quick", action="store_true")
ARGS = parser.parse_args()

from particle_mesh.runtime import bootstrap  # noqa: E402

bootstrap.configure_torch_env()
bootstrap.init_taichi()

import taichi as ti  # noqa: E402
import torch  # noqa: E402

from particle_mesh.app import build_app  # noqa: E402
from particle_mesh.physics import engine as E  # noqa: E402

LMB, RMB, MMB, CTRL, SPACE = ti.ui.LMB, ti.ui.RMB, ti.ui.MMB, ti.ui.CTRL, ti.ui.SPACE
UP, DOWN, LEFT, RIGHT = ti.ui.UP, ti.ui.DOWN, ti.ui.LEFT, ti.ui.RIGHT
SCENE = (0.6, 0.5)            # cursor over the scene, right of the panel
FRAMES = 2 if ARGS.quick else 4   # frames per "hold" step


class ScriptedWindow:
    """Real GGUI window (offscreen unless --visible) with scripted input."""

    def __init__(self, name, res, vsync=False, pos=(0, 0)):
        self._w = ti.ui.Window(name, res, vsync=vsync, show_window=ARGS.visible, pos=pos)
        self.running = True
        self.held = set()
        self.cursor = SCENE
        self.events = []

    def __getattr__(self, k):
        return getattr(self._w, k)

    def is_pressed(self, *keys):
        return any(k in self.held for k in keys)

    def get_cursor_pos(self):
        return self.cursor

    def get_events(self, *args):
        events, self.events = self.events, []
        return events

    def show(self):
        """Visible: present the frame. Offscreen: the real show() crashes without a window, so draw the frame
        into the image buffer instead; never drawing lets GGUI work pile up until the GPU runs out of resources
        (UR_RESULT_ERROR_OUT_OF_RESOURCES), which the real app never does."""
        if ARGS.visible:
            self._w.show()
        else:
            self.last_image = self._w.get_image_buffer_as_numpy()


class ScriptedGui:
    """Proxy of the real ImGui object: draws every widget, but button presses and slider values come from the
    script. Records every label it saw, to check coverage."""

    def __init__(self, gui):
        self._gui = gui
        self.press = set()
        self.sliders = {}
        self.seen_buttons, self.seen_sliders = set(), set()
        self.used_buttons, self.used_sliders = set(), set()

    def __getattr__(self, k):
        return getattr(self._gui, k)

    def button(self, label):
        self._gui.button(label)
        self.seen_buttons.add(label)
        if label.startswith("Boundary:"):
            self.boundary_label = label        # its text changes with the mode
        hit = label in self.press
        if hit:
            self.press.discard(label)          # one click
            self.used_buttons.add(label)
        return hit

    def slider_float(self, label, value, lo, hi):
        value = self._gui.slider_float(label, value, lo, hi)
        self.seen_sliders.add(label)
        if label in self.sliders:
            want = self.sliders.pop(label)
            value = lo if want == "min" else hi if want == "max" else float(want)
            self.used_sliders.add(label)
        return value


def build():
    factory = None
    if ARGS.particles is not None:
        factory = lambda solver: E.SimulationEngine(initial_particles=ARGS.particles, solver=solver)  # noqa: E731
    app = build_app(ARGS.solver, window_factory=ScriptedWindow, engine_factory=factory)
    app.imgui = ScriptedGui(app.imgui)
    return app


APP = build()
ENG, CAM, WIN, GUI = APP.engine, APP.camera, APP.window, APP.imgui


def _sync_sections():
    """ACTIONS_SYNC=1: synchronise and time each frame section, so an asynchronous GPU failure is attributed to
    the section that caused it, and slow sections are reported."""
    def wrap(obj, name):
        fn = getattr(obj, name)

        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                try:
                    torch.xpu.synchronize()
                except Exception:
                    print(f"SYNC FAILED after {name}", file=sys.stderr, flush=True)
                    raise
                ms = (time.perf_counter() - t0) * 1e3
                if ms > 200:
                    print(f"SLOW {name} {ms:.0f} ms n={APP.engine.num_active} zoom={CAM.zoom:.4g}",
                          file=sys.stderr, flush=True)
        setattr(obj, name, timed)
    for obj, name in ((APP.engine, "step"), (APP.engine, "add_galaxy"), (APP.engine, "remove_particles"),
                      (APP.mesh_renderer, "render_pm"), (APP.circle_renderer, "render_nbody"),
                      (APP.brush, "draw_ring"), (APP.panel, "render")):
        wrap(obj, name)


if os.environ.get("ACTIONS_SYNC") and torch.xpu.is_available():
    _sync_sections()
FAILURES = []
STATS = {"frames": 0, "max_active": 0, "min_zoom": CAM.zoom, "max_zoom": CAM.zoom, "steps": 0}
T0 = time.perf_counter()


def check(label):
    """Invariants after every frame."""
    eng = APP.engine
    n = eng.num_active
    problems = []
    if not 0 <= n <= eng.capacity():
        problems.append(f"num_active {n} outside [0, {eng.capacity()}]")
    if n:
        if not bool(torch.isfinite(eng.pos[:n]).all()):
            problems.append("non-finite positions")
        if not bool(torch.isfinite(eng.vel[:n]).all()):
            problems.append("non-finite velocities")
        if eng.boundary_mode == 'open':
            p = eng.pos[:n]
            if bool(((p < 0) | (p[:, 0:1] >= eng.universe_w) | (p[:, 1:2] >= eng.universe_h)).any()):
                problems.append("particle outside the mesh in open mode")
    if not all(math.isfinite(v) for v in (CAM.cam_x, CAM.cam_y, CAM.zoom)):
        problems.append("non-finite camera")
    min_zoom = getattr(CAM, "min_zoom", 0.0)
    if CAM.zoom < min_zoom * (1 - 1e-9):
        problems.append(f"zoom {CAM.zoom} below the limit {min_zoom}")
    for p in problems:
        FAILURES.append(f"{label}: {p}")
    STATS["max_active"] = max(STATS["max_active"], n)
    STATS["min_zoom"] = min(STATS["min_zoom"], CAM.zoom)
    STATS["max_zoom"] = max(STATS["max_zoom"], CAM.zoom)
    return not problems


def step(label, frames=1, held=(), cursor=None, to=None, press=(), sliders=None, events=(), call=None):
    """Run `frames` frames with this input. cursor -> to moves linearly (a drag; the first frame latches it)."""
    if os.environ.get("ACTIONS_TRACE"):
        mem = torch.xpu.memory_allocated() / 2**30 if torch.xpu.is_available() else 0.0
        print(f"STEP {label} n={APP.engine.num_active} zoom={CAM.zoom:.4g} mem={mem:.2f}GiB", file=sys.stderr, flush=True)
    STATS["steps"] += 1
    if call is not None:
        call()
    GUI.press = set(press)
    GUI.sliders = dict(sliders or {})
    WIN.events = [SimpleNamespace(key=k, type=ti.ui.PRESS) for k in events]
    start = cursor or WIN.cursor
    end = to or start
    for i in range(frames):
        t = i / (frames - 1) if frames > 1 else 1.0
        WIN.held = set(held)
        WIN.cursor = (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
        try:
            APP.frame()
            ok = check(f"{label} [frame {i}]")
        except Exception:
            FAILURES.append(f"{label} [frame {i}]: CRASH\n{traceback.format_exc()}")
            finish()
        STATS["frames"] += 1
        if not ok:
            return
    WIN.held = set()


def frame_ok(label):
    """The last drawn frame has content. Offscreen only: reading the image buffer of a visible window outside
    show() breaks ImGui's frame scope (an assertion abort), and there the frame is on screen anyway."""
    if ARGS.visible:
        return None
    img = WIN.last_image
    lit = float((img[..., :3].max(axis=-1) > 0.05).mean())
    if not lit > 0.0:
        FAILURES.append(f"{label}: empty frame")
    return lit


def release():
    step("release", 1)


def drag(label, keys, start, end, frames=3):
    step(label, frames, held=keys, cursor=start, to=end)
    release()


def finish():
    unused_b = sorted(GUI.seen_buttons - GUI.used_buttons)
    unused_s = sorted(GUI.seen_sliders - GUI.used_sliders)
    print(json.dumps({
        "solver": ARGS.solver, "visible": ARGS.visible, "ok": not FAILURES, "failures": FAILURES[:20],
        "seconds": round(time.perf_counter() - T0, 1), **STATS,
        "buttons_pressed": sorted(GUI.used_buttons), "sliders_set": sorted(GUI.used_sliders),
        "buttons_never_pressed": unused_b, "sliders_never_set": unused_s,
    }), flush=True)
    os._exit(0 if not FAILURES else 1)


# ====================================================================== the script

PM = ARGS.solver == 'pm'
step("warm-up", 3)

# --- brush: spawn, erase, both brush modes
step("spawn", FRAMES, held={LMB}, cursor=SCENE)
step("erase", FRAMES, held={RMB}, cursor=SCENE)
step("toggle spawn mode", 1, press={"Toggle Spawn Mode"})
step("spawn uniform disk", FRAMES, held={LMB})
step("toggle spawn mode back", 1, press={"Toggle Spawn Mode"})
step("click on panel does not spawn", 1, held={LMB}, cursor=(0.05, 0.5))
release()

# --- brush size: Ctrl+RMB drag to both limits, spawn/erase at each
drag("brush max", {CTRL, RMB}, (0.6, 0.0), (0.6, 1.0))
drag("brush max more", {CTRL, RMB}, (0.6, 0.0), (0.6, 1.0))
step("spawn at max brush", FRAMES, held={LMB}, cursor=SCENE)
step("erase at max brush", FRAMES, held={RMB}, cursor=SCENE)
drag("brush min", {CTRL, RMB}, (0.6, 1.0), (0.6, 0.0))
drag("brush min more", {CTRL, RMB}, (0.6, 1.0), (0.6, 0.0))
step("spawn at min brush", FRAMES, held={LMB}, cursor=SCENE)
step("erase at min brush", FRAMES, held={RMB}, cursor=SCENE)
step("brush default", 1, sliders={"Brush Radius": 500.0})

# --- zoom out to the limit (Ctrl+LMB drag down, then the Down arrow), act there
for k in range(4):
    drag(f"zoom out drag {k}", {CTRL, LMB}, (0.6, 1.0), (0.6, 0.0))
step("zoom out arrow", 3 * FRAMES, held={DOWN})
step("spawn at min zoom", FRAMES, held={LMB}, cursor=SCENE)
step("erase at min zoom", FRAMES, held={RMB}, cursor=SCENE)
drag("brush max at min zoom", {CTRL, RMB}, (0.6, 0.0), (0.6, 1.0))
step("spawn max brush at min zoom", FRAMES, held={LMB}, cursor=SCENE)
step("erase max brush at min zoom", FRAMES, held={RMB}, cursor=SCENE)
step("brush default", 1, sliders={"Brush Radius": 500.0})
if PM:
    step("potential view at min zoom", FRAMES, events=(), held={'p'})
    release()
    step("field view at min zoom", FRAMES, held={'f'})
    release()
    step("views off", 1, held={'p'})
    release()
    step("views off 2", 1, held={'f'})
    release()
step("pan at min zoom", 3, held={SPACE, LMB}, cursor=(0.3, 0.3), to=(0.9, 0.9))
release()
frame_ok("min zoom frame")

# --- reset, then zoom in to the limit (Ctrl+LMB drag up, then the Up arrow), act there
step("reset R", 1, held={'r'})
release()
for k in range(4):
    drag(f"zoom in drag {k}", {CTRL, LMB}, (0.6, 0.0), (0.6, 1.0))
step("zoom in arrow", 3 * FRAMES, held={UP})
step("spawn at max zoom", FRAMES, held={LMB}, cursor=SCENE)
step("erase at max zoom", FRAMES, held={RMB}, cursor=SCENE)
if PM:
    step("potential view at max zoom", 1, held={'p'})
    release()
    step("field view at max zoom", 1, held={'f'})
    release()
    step("both views at max zoom", FRAMES)
    step("views off", 1, held={'p'})
    release()
    step("views off 2", 1, held={'f'})
    release()
frame_ok("max zoom frame")
step("reset button", 1, press={"Reset View"})

# --- panning far away (Space+LMB and MMB), act off the mesh, come back
for k in range(6):
    drag(f"pan space {k}", {SPACE, LMB}, (0.9, 0.9), (0.2, 0.1))
step("spawn off the mesh", FRAMES, held={LMB}, cursor=SCENE)
step("erase off the mesh", FRAMES, held={RMB}, cursor=SCENE)
for k in range(6):
    drag(f"pan mmb {k}", {MMB}, (0.2, 0.1), (0.9, 0.9))
step("reset R", 1, held={'r'})
release()

# --- keyboard toggles, panel hide / arrow click, themes
for key in ('p', 'f', 'p', 'f'):
    step(f"key {key}", 1, held={key})
    release()
step("theme right", 1, held={RIGHT})
release()
step("theme left", 1, held={LEFT})
release()
step("next theme button", 1, press={"Next Color Theme"})
step("hide panel (M)", 1, held={'m'})
release()
step("spawn with panel hidden", FRAMES, held={LMB}, cursor=(0.1, 0.5))
release()
step("reopen panel via arrow", 1, held={LMB}, cursor=(0.02, 0.97))
if not APP.controls.ui_open:
    FAILURES.append("panel arrow click did not reopen the panel")
release()

# --- image spawning (folder scan + the real loader)
step("rescan images", 1, press={"Rescan Images"})
for name in sorted(APP.panel.image_files):
    step(f"spawn image {name}", 2, press={f"Spawn {name}"})

# --- PM views: FFT minimap, band-pass render with its sliders at the extremes
if PM:
    step("fft minimap on", FRAMES, press={"Compute FFT Minimap: OFF"})
    step("bandpass render on", FRAMES, press={"Apply Filter to Main Render: OFF"})
    for lab in ("High-Pass (Inner)", "Low-Pass (Outer)", "Filter Softness"):
        step(f"{lab} min", 2, sliders={lab: "min"})
        step(f"{lab} max", 2, sliders={lab: "max"})
    step("bandpass defaults", 2, sliders={"High-Pass (Inner)": -0.5, "Low-Pass (Outer)": 5.0, "Filter Softness": 0.1})
    for lab in ("Potential Min (Log)", "Potential Max (Log)", "Field Min (Log)", "Field Max (Log)"):
        step(f"{lab} min", 1, sliders={lab: "min"}, held={'p', 'f'} if lab.startswith("Potential") else ())
        step(f"{lab} max", 1, sliders={lab: "max"})
    release()
    step("bandpass render off", 1, press={"Apply Filter to Main Render: ON"})
    step("fft minimap off", 1, press={"Compute FFT Minimap: ON"})

# --- every potential x every boundary, with each potential's own sliders at both extremes
POTENTIAL_BUTTONS = {"gravity": "Force: Gravity", "lennard_jones": "Force: Lennard-Jones",
                     "harmonic": "Force: Mutual Springs", "coulomb": "Force: Coulomb"}
OWN_SLIDERS = {"gravity": ["Gravity (Log)"], "coulomb": ["Coulomb Const (Log)"], "harmonic": ["Spring K"],
               "lennard_jones": ["LJ Epsilon", "LJ Sigma"]}
SHARED = ["Force Range", "Friction (Gas Cooling)", "Force Softening Length", "Time Scale (Log)",
          "Gravity Mag", "Gravity Angle (Deg)", "Spawn Velocity (x orbit)", "Density (stars/px^2)"]
REBUILD = int(math.ceil(E.KERNEL_REBUILD_DELAY * E.FPS_TARGET)) + 2   # frames until a debounced rebuild runs


def wait_rebuild(label):
    time.sleep(E.KERNEL_REBUILD_DELAY + 0.02)
    step(label, 2)


for kind, button in POTENTIAL_BUTTONS.items():
    step(f"potential {kind}", FRAMES, press={button})
    for b in range(3):
        step(f"{kind}: boundary {ENG.boundary_mode}", FRAMES, held={LMB}, cursor=SCENE)
        step(f"{kind}: erase in {ENG.boundary_mode}", 1, held={RMB}, cursor=SCENE)
        step(f"{kind}: next boundary", 1, press={GUI.boundary_label})
    for lab in OWN_SLIDERS[kind] + SHARED:
        for end in ("min", "max"):
            step(f"{kind}: {lab} {end}", 1, sliders={lab: end})
            wait_rebuild(f"{kind}: {lab} {end} (rebuilt)")
            step(f"{kind}: {lab} {end} + spawn", FRAMES, held={LMB}, cursor=SCENE)
    # back to the defaults
    step(f"{kind}: defaults", 1, sliders={
        "Force Range": "max", "Friction (Gas Cooling)": 0.1, "Force Softening Length": 1.5, "Time Scale (Log)": 0.0,
        "Gravity Mag": 0.0, "Gravity Angle (Deg)": 270.0, "Spawn Velocity (x orbit)": 1.0,
        "Density (stars/px^2)": 1.0, "Gravity (Log)": math.log10(0.64), "Coulomb Const (Log)": math.log10(0.64),
        "Spring K": 0.1, "LJ Epsilon": 1000.0, "LJ Sigma": 40.0})
    wait_rebuild(f"{kind}: defaults (rebuilt)")

# --- gas pressure models with their sliders at the extremes (PM)
if PM:
    for button in ("Gas: Isothermal (Linear)", "Gas: Polytropic (Adiabatic)"):
        step(button, FRAMES, press={button})
        for lab in ("Gas Pressure P", "Poly Index (Gamma)"):
            for end in ("min", "max"):
                step(f"{button}: {lab} {end}", FRAMES, sliders={lab: end}, held={LMB}, cursor=SCENE)
    step("gas pressure P default", 1, sliders={"Gas Pressure P": 1.0, "Poly Index (Gamma)": 1.0})
    step("Gas: Cold Dust (None)", FRAMES, press={"Gas: Cold Dust (None)"})

# --- worst case together: strongest gravity, longest time step, no friction, smallest softening and range
step("worst case", 1, press={"Force: Gravity"}, sliders={
    "Gravity (Log)": "max", "Time Scale (Log)": "max", "Friction (Gas Cooling)": "min",
    "Force Softening Length": "min", "Force Range": "min"})
wait_rebuild("worst case (rebuilt)")
step("worst case + spawn", 3 * FRAMES, held={LMB}, cursor=SCENE)
for b in range(3):
    step(f"worst case boundary {ENG.boundary_mode}", 2 * FRAMES)
    step("worst case next boundary", 1, press={GUI.boundary_label})
step("worst case at min zoom", 1, call=lambda: CAM.zoom_around_mouse(0, 0, 1e-9))
step("worst case at min zoom", 2 * FRAMES, held={LMB}, cursor=SCENE)
frame_ok("worst case frame")

# --- fill to capacity, then clear everything, then empty-universe frames
step("defaults", 1, sliders={"Gravity (Log)": math.log10(0.64), "Time Scale (Log)": 0.0, "Force Range": "max",
                             "Force Softening Length": 1.5, "Friction (Gas Cooling)": 0.1,
                             "Density (stars/px^2)": "max", "Brush Radius": "max"})
step("reset R", 1, held={'r'})
release()
fill_frames = 0
while ENG.num_active < ENG.capacity() and fill_frames < 40:
    step("fill to capacity", 1, held={LMB}, cursor=SCENE)
    fill_frames += 1
step("at capacity", FRAMES, held={LMB}, cursor=SCENE)
step("zoom out", 1, call=lambda: CAM.zoom_around_mouse(0, 0, 1e-9))
step("erase everything", 6, held={RMB}, cursor=SCENE)
step("empty universe", FRAMES)
if PM:
    step("empty universe views", FRAMES, held={'p'})
    release()
    step("empty universe fft", FRAMES, press={"Compute FFT Minimap: OFF"})
step("spawn into empty universe", FRAMES, held={LMB}, cursor=SCENE)
frame_ok("final frame")

# --- Escape ends the loop
step("escape", 1, events=(ti.ui.ESCAPE,))
if WIN.running:
    FAILURES.append("escape: window still running")
finish()
