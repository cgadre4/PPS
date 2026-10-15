"""
Particle Mesh sandbox entry point:  uv run python main.py   (or  uv run python -m particle_mesh)

The app itself lives in particle_mesh.app. The names below stay importable for tests and tools.
"""
from particle_mesh.runtime import bootstrap
from particle_mesh.settings import CONFIG, SETTINGS  # noqa: F401  (re-exported)

# Global Display Settings
WIDTH = SETTINGS.width
HEIGHT = SETTINGS.height
FPS_TARGET = SETTINGS.fps_target
SOLVER_MODE = SETTINGS.solver

bootstrap.configure_torch_env()
# Taichi is initialised when this module is imported (tests and tools rely on `import main` doing it).
_init_taichi = bootstrap.init_taichi
_init_taichi()

from particle_mesh import app  # noqa: E402
from particle_mesh.physics.engine import SimulationEngine  # noqa: E402
from particle_mesh.ui.input import KeyEdges  # noqa: E402,F401  (re-exported)


def main():
    # SOLVER_MODE and SimulationEngine are read at call time, so a harness can patch them on this module
    return app.main(solver=SOLVER_MODE, engine_factory=SimulationEngine)


if __name__ == "__main__":
    main()
