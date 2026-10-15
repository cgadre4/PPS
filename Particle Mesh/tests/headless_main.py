"""
Runs main.main() for a few frames with an offscreen GGUI window (used by test_smoke.py in a subprocess).

    python tests/headless_main.py <pm|nbody|potential|bandpass>

Input is stubbed: the cursor sits right of the UI panel, LMB is held on frames 1-2 (spawns a galaxy), and on
frame 3 'f' (vector field) and, in 'potential' mode, 'p' are pressed. Prints one JSON line with the result.
"""
import json
import os
import sys

PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PM_DIR)
os.chdir(PM_DIR)

MODE = sys.argv[1] if len(sys.argv) > 1 else "pm"
FRAMES = 8

import taichi as ti  # noqa: E402

_RealWindow = ti.ui.Window
captured = {}
state = {}


class HeadlessWindow:
    def __init__(self, name, res, vsync=False, pos=(100, 100)):
        self._w = _RealWindow(name, res, vsync=vsync, show_window=False, pos=pos)
        self.running = True
        self.frames = 0

    def __getattr__(self, k):
        return getattr(self._w, k)

    def get_cursor_pos(self):
        return (0.6, 0.5)

    def get_events(self, *args):
        return []

    def is_pressed(self, *keys):
        if self.frames in (1, 2) and ti.ui.LMB in keys:
            return True
        if self.frames == 3 and MODE in ("potential", "bandpass"):
            if 'f' in keys:
                return True
            if MODE == "potential" and 'p' in keys:
                return True
        return False

    def show(self):  # the real show() crashes without a visible window; capture the frame instead
        self.frames += 1
        if self.frames == FRAMES:
            captured["img"] = self._w.get_image_buffer_as_numpy()
            self.running = False


import main  # noqa: E402

ti.ui.Window = HeadlessWindow
main.SOLVER_MODE = "nbody" if MODE == "nbody" else "pm"
_Engine = main.SimulationEngine


class Engine(_Engine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_bandpass_render = MODE == "bandpass"
        state["engine"] = self


main.SimulationEngine = Engine
main.main()

eng = state["engine"]
img = captured["img"]
ui_px = int(main.WIDTH * main.CONFIG["display"].get("ui_width", 0.15))
corner = img[:ui_px, :ui_px, :3]
grey = (abs(corner[..., 0] - corner[..., 1]) < 1e-3) & (abs(corner[..., 1] - corner[..., 2]) < 1e-3) & (corner[..., 0] > 0.05)
print(json.dumps({
    "mode": MODE,
    "frames": FRAMES,
    "solver": eng.solver_type,
    "num_active": int(eng.num_active),
    "show_potential": bool(eng.show_potential),
    "show_field": bool(eng.show_field),
    "bandpass": bool(eng.use_bandpass_render),
    "lit_fraction": float((img[..., :3].max(axis=-1) > 0.05).mean()),
    "minimap_grey_fraction": float(grey.mean()),
}))
