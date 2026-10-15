"""
Particle Mesh sandbox entry point:  uv run python main.py   (or  uv run python -m particle_mesh)

The app itself lives in particle_mesh.app (which bootstraps torch and Taichi when it starts). Importing this module
has no side effects; the names below stay importable for tests and tools.
"""
from particle_mesh.settings import CONFIG, SETTINGS  # noqa: F401  (re-exported)

# Global Display Settings
WIDTH = SETTINGS.width
HEIGHT = SETTINGS.height
FPS_TARGET = SETTINGS.fps_target
SOLVER_MODE = SETTINGS.solver

from particle_mesh import app
from particle_mesh.physics.engine import SimulationEngine
from particle_mesh.ui.input import KeyEdges  # noqa: F401  (re-exported)


def main():
    # SOLVER_MODE and SimulationEngine are read at call time, so a harness can patch them on this module
    return app.main(solver=SOLVER_MODE, engine_factory=SimulationEngine)


if __name__ == "__main__":
    main()
