"""
Frame-time benchmark: engine.step + PM render/present at several particle counts, offscreen.

    uv run python tools/bench_frame.py [counts...]      e.g.  uv run python tools/bench_frame.py 1e6 4e6

Uses the same Taichi/torch/kernel backend selection as the app (imports main). Prints ms per stage.
"""
import os
import sys
import time

PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PM_DIR)
os.chdir(PM_DIR)

import main  # noqa: E402  (initialises Taichi like the app)
import taichi as ti  # noqa: E402
import torch  # noqa: E402
import engine as E  # noqa: E402
import mesh_renderer as MR  # noqa: E402
from colors import THEMES  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def bench(n, frames, canvas):
    eng = E.SimulationEngine(initial_particles=n)
    eng.precompute_kernel()
    renderer = MR.MeshRenderer(eng.kernels)
    cmap = torch.tensor(THEMES["Standard Theme"], dtype=torch.float32, device=eng.device)
    cam_x, cam_y = (eng.universe_w - MR.WIDTH) / 2.0, (eng.universe_h - MR.HEIGHT) / 2.0
    dt = 1.0 / E.FPS_TARGET

    def frame():
        t0 = time.perf_counter()
        eng.step(dt)
        _sync(eng.device)
        t1 = time.perf_counter()
        renderer.render(canvas, eng.pos, eng.num_active, cam_x, cam_y, 1.0, cmap, 0.0, 3.0, 0.01)
        _sync(eng.device)
        # No ti.sync() here: blocking on the Vulkan queue every frame idles the shared GPU and slowed the next
        # XPU step by ~25% in measurements; the app's own loop doesn't block like that either.
        return t1 - t0, time.perf_counter() - t1

    for _ in range(3):
        frame()
    step = render = 0.0
    for _ in range(frames):
        a, b = frame()
        step += a
        render += b
    return step / frames * 1e3, render / frames * 1e3, eng


def main_():
    counts = [int(float(c)) for c in sys.argv[1:]] or [1_000_000, 2_000_000, 4_000_000]
    window = ti.ui.Window("bench", (MR.WIDTH, MR.HEIGHT), show_window=False)
    canvas = window.get_canvas()
    print(f"device={E.DEVICE} grid={E.GRID_W}^2 view={MR.VIEW_W}x{MR.VIEW_H} sort_interval={E.CONFIG['simulation'].get('SORT_INTERVAL')}")
    for n in counts:
        n = min(n, E.MAX_PARTICLES)
        step, render, eng = bench(n, 100 if n <= 2_000_000 else 60, canvas)
        total = step + render
        print(f"{n/1e6:5.1f}M  kernels={eng.kernels.name:6s}  step {step:6.2f} ms + render {render:5.2f} ms "
              f"= {total:6.2f} ms (~{1000/total:5.1f} FPS)")
        del eng


if __name__ == "__main__":
    main_()
