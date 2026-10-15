"""
Shared test setup. Run from the Particle Mesh folder:  uv run pytest

Importing `main` initialises Taichi exactly as the app does (CUDA, else Vulkan, ...) and imports the renderers,
whose Taichi fields are allocated at import time. Everything runs headless; tests that need a full app loop
(test_smoke.py) start it in a subprocess because GGUI supports one window per process.
"""
import os
import sys

import pytest

PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PM_DIR)
os.chdir(PM_DIR)

import main  # noqa: E402,F401  (side effect: ti.init + renderer field allocation)
import torch  # noqa: E402
import engine as E  # noqa: E402


@pytest.fixture
def make_engine():
    """Factory for a deterministic engine: no uniform gravity, no sorting unless a test enables it."""
    def _make(n=0, solver='pm', boundary='periodic'):
        torch.manual_seed(0)
        eng = E.SimulationEngine(initial_particles=n)
        eng.solver_type = solver
        eng.boundary_mode = boundary
        eng.uniform_gravity_mag = 0.0
        eng.sort_interval = 0
        return eng
    return _make


@pytest.fixture(scope="session")
def headless_canvas():
    """One offscreen GGUI window sized to the render-cell viewport (never call window.show() on it)."""
    import taichi as ti
    import mesh_renderer as MR
    window = ti.ui.Window("tests", (MR.VIEW_W, MR.VIEW_H), show_window=False)
    return window, window.get_canvas()


@pytest.fixture
def centre():
    """World coordinates of a physics-cell centre near the middle of the universe."""
    cs = E.PHYSICS_CELL_SIZE
    return E.UNIVERSE_W / 2.0 + cs / 2.0, E.UNIVERSE_H / 2.0 + cs / 2.0
