"""
Frame-time benchmark: engine.step + the real PM frame (MeshRenderer.render_pm: density view, FFT minimap,
packed present), offscreen.

    uv run python tools/bench_frame.py [counts...] [--zoom Z] [--field] [--frames N]
    e.g.  uv run python tools/bench_frame.py 1e6 4e6
          uv run python tools/bench_frame.py 1e6 --zoom 4      (zoomed-in camera: most particles off screen)

Uses the same Taichi/torch/kernel backend selection as the app (imports main). Prints ms per stage.
Always A/B performance changes back to back (e.g. against a git worktree of master), never against stored numbers.
"""
import argparse
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
from camera import SimulationCamera  # noqa: E402
from colors import THEMES  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def bench(n, frames, canvas, zoom=1.0, field=False):
    eng = E.SimulationEngine(initial_particles=n)
    eng.precompute_kernel()
    eng.show_field = field
    renderer = MR.MeshRenderer(eng.kernels)
    camera = SimulationCamera(MR.WIDTH, MR.HEIGHT, eng.universe_w, eng.universe_h)
    if zoom != 1.0:
        camera.zoom_around_mouse(MR.WIDTH / 2.0, MR.HEIGHT / 2.0, zoom)
    cmap = torch.tensor(THEMES["Standard Theme"], dtype=torch.float32, device=eng.device)
    ui_w = E.CONFIG["display"].get("ui_width", 0.15)
    dt = 1.0 / E.FPS_TARGET

    def frame():
        t0 = time.perf_counter()
        eng.step(dt)
        _sync(eng.device)
        t1 = time.perf_counter()
        renderer.render_pm(canvas, eng, camera, cmap, ui_w)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("counts", nargs="*", default=["1e6", "2e6", "4e6"], help="particle counts")
    parser.add_argument("--zoom", type=float, default=1.0, help="camera zoom (1 = default view)")
    parser.add_argument("--field", action="store_true", help="also draw the force-arrow overlay")
    parser.add_argument("--frames", type=int, default=0, help="frames to time (default 100, or 60 above 2M)")
    args = parser.parse_args()

    window = ti.ui.Window("bench", (MR.WIDTH, MR.HEIGHT), show_window=False)
    canvas = window.get_canvas()
    print(f"device={E.DEVICE} grid={E.GRID_W}x{E.GRID_H} view={MR.VIEW_W}x{MR.VIEW_H} zoom={args.zoom} "
          f"field={args.field} sort_interval={E.CONFIG['simulation'].get('SORT_INTERVAL')}")
    for c in args.counts:
        n = min(int(float(c)), E.MAX_PARTICLES)
        frames = args.frames or (100 if n <= 2_000_000 else 60)
        step, render, eng = bench(n, frames, canvas, args.zoom, args.field)
        total = step + render
        print(f"{n/1e6:5.1f}M  kernels={eng.kernels.name:6s}  step {step:6.2f} ms + render {render:5.2f} ms "
              f"= {total:6.2f} ms (~{1000/total:5.1f} FPS)")
        del eng


if __name__ == "__main__":
    main_()
