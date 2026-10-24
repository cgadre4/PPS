"""
Frame-time benchmark: engine.step + the real PM frame (MeshRenderer.render_pm: density view, FFT minimap,
packed present), offscreen.

    uv run python tools/bench_frame.py [counts...] [--zoom Z] [--field] [--frames N] [--solver pm|nbody] [--app]
                                       [--boundary open|periodic] [--potential P] [--range R]
    e.g.  uv run python tools/bench_frame.py 1e6 4e6
          uv run python tools/bench_frame.py 1e6 --zoom 4      (zoomed-in camera: most particles off screen)
          uv run python tools/bench_frame.py 1e4 2e4 --solver nbody   (direct N-body step + circle render)
          uv run python tools/bench_frame.py 1e6 --app                (whole App.frame: input, step, render, panel)
  --app numbers are for A/B only: offscreen, GGUI work is never presented and the shared GPU stalls (~75 ms/frame at
  1M on the B390), an artefact of never calling window.show(). The real windowed App.frame measured ~11 ms at 1M.

Uses the same Taichi/torch/kernel backend selection as the app (runtime.bootstrap). Prints ms per stage.
Always A/B performance changes back to back (e.g. against a git worktree of master), never against stored numbers.
"""
import argparse
import os
import sys
import time

PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PM_DIR)
os.chdir(PM_DIR)

from particle_mesh.runtime import bootstrap  # noqa: E402

bootstrap.configure_torch_env()
bootstrap.init_taichi()

import taichi as ti  # noqa: E402
import torch  # noqa: E402
from particle_mesh.physics import engine as E  # noqa: E402
from particle_mesh.render import mesh_renderer as MR  # noqa: E402
from particle_mesh.render.camera import SimulationCamera  # noqa: E402
from particle_mesh.render.themes import THEMES  # noqa: E402
from particle_mesh.render.circle_renderer import CircleRenderer  # noqa: E402


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def bench(n, frames, canvas, zoom=1.0, field=False, solver="pm", boundary=None, potential="gravity",
          force_range=None):
    eng = E.SimulationEngine(initial_particles=n, solver=solver)
    if boundary is not None:
        eng.boundary_mode = boundary
    if force_range is not None:
        eng.gravity_range = force_range
    eng.set_potential(potential)          # also builds the kernel
    eng.show_field = field
    renderer = MR.MeshRenderer(eng.kernels)
    camera = SimulationCamera(MR.WIDTH, MR.HEIGHT, eng.universe_w, eng.universe_h)
    if zoom != 1.0:
        camera.zoom_around_mouse(MR.WIDTH / 2.0, MR.HEIGHT / 2.0, zoom)
    cmap = torch.tensor(THEMES["Standard Theme"], dtype=torch.float32, device=eng.device)
    ui_w = E.CONFIG["display"].get("ui_width", 0.15)
    dt = 1.0 / E.FPS_TARGET

    circles = CircleRenderer() if solver == "nbody" else None

    def draw():
        if circles is None:
            renderer.render_pm(canvas, eng, camera, cmap, ui_w)
            return
        n_act = eng.num_active
        colors = cmap[4].expand(n_act, 3)
        circles.render(canvas, eng.pos[:n_act], colors, n_act, camera.cam_x, camera.cam_y,
                       camera.width / camera.zoom, camera.height / camera.zoom, 0.0005)

    def frame():
        t0 = time.perf_counter()
        eng.step(dt)
        _sync(eng.device)
        t1 = time.perf_counter()
        draw()
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


def bench_app(n, frames, solver):
    """Whole App.frame() (input, step, render, overlays, ImGui panel) on an offscreen window; ms per frame (A/B only)."""
    from particle_mesh.app import build_app

    class OffscreenWindow:
        def __init__(self, name, res, vsync=False, pos=(0, 0)):
            self._w = ti.ui.Window(name, res, vsync=vsync, show_window=False, pos=pos)
            self.running = True

        def __getattr__(self, k):
            return getattr(self._w, k)

        def get_cursor_pos(self):
            return (0.6, 0.5)

        def get_events(self, *args):
            return []

        def is_pressed(self, *keys):
            return False

        def show(self):  # the real show() needs a visible window; the frame is still built and uploaded
            pass

    app = build_app(solver, window_factory=OffscreenWindow,
                    engine_factory=lambda solver: E.SimulationEngine(initial_particles=n, solver=solver))
    for _ in range(3):
        app.frame()
    _sync(app.engine.device)
    t0 = time.perf_counter()
    for _ in range(frames):
        app.frame()
    _sync(app.engine.device)
    return (time.perf_counter() - t0) / frames * 1e3, app.engine


def main_():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("counts", nargs="*", default=["1e6", "2e6", "4e6"], help="particle counts")
    parser.add_argument("--zoom", type=float, default=1.0, help="camera zoom (1 = default view)")
    parser.add_argument("--field", action="store_true", help="also draw the force-arrow overlay")
    parser.add_argument("--frames", type=int, default=0, help="frames to time (default 100, or 60 above 2M)")
    parser.add_argument("--solver", choices=("pm", "nbody"), default="pm", help="solver to time (default pm)")
    parser.add_argument("--app", action="store_true", help="time the whole App.frame (one count only; GGUI allows one window)")
    parser.add_argument("--boundary", choices=("open", "periodic"), default=None,
                        help="boundary mode (default: the engine's, open)")
    parser.add_argument("--potential", choices=("gravity", "coulomb", "harmonic", "lennard_jones"), default="gravity")
    parser.add_argument("--range", type=float, default=None, dest="force_range",
                        help="force range (default: the slider default, i.e. no screening)")
    args = parser.parse_args()

    if args.app:
        cap = min(E.MAX_PARTICLES, E.NBODY_MAX_PARTICLES) if args.solver == "nbody" else E.MAX_PARTICLES
        n = min(int(float(args.counts[0])), cap)
        ms, eng = bench_app(n, args.frames or 100, args.solver)
        print(f"{n:>9,d}  kernels={eng.kernels.name:6s}  App.frame {ms:6.2f} ms (~{1000 / ms:5.1f} FPS)")
        return
    window = ti.ui.Window("bench", (MR.WIDTH, MR.HEIGHT), show_window=False)
    canvas = window.get_canvas()
    print(f"device={E.DEVICE} grid={E.GRID_W}x{E.GRID_H} view={MR.VIEW_W}x{MR.VIEW_H} zoom={args.zoom} "
          f"field={args.field} solver={args.solver} sort_interval={E.CONFIG['simulation'].get('SORT_INTERVAL')} "
          f"boundary={args.boundary or 'default'} potential={args.potential} range={args.force_range or 'default'}")
    for c in args.counts:
        cap = min(E.MAX_PARTICLES, E.NBODY_MAX_PARTICLES) if args.solver == "nbody" else E.MAX_PARTICLES
        n = min(int(float(c)), cap)
        frames = args.frames or (100 if n <= 2_000_000 else 60)
        step, render, eng = bench(n, frames, canvas, args.zoom, args.field, args.solver, args.boundary,
                                  args.potential, args.force_range)
        total = step + render
        print(f"{n:>9,d}  kernels={eng.kernels.name:6s}  step {step:6.2f} ms + render {render:5.2f} ms "
              f"= {total:6.2f} ms (~{1000/total:5.1f} FPS)")
        del eng


if __name__ == "__main__":
    main_()
