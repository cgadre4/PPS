"""
Single place that loads config.json (resolved next to main.py, so the working directory doesn't matter).

SETTINGS is the typed, read-only view the package uses. CONFIG (the raw dict) stays available for the GUI panel and
older callers. Every key has exactly one fallback here, equal to the shipped config.json value.
"""
import json
import os
from dataclasses import dataclass
from types import MappingProxyType

# config.json lives next to main.py, one level above this package
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# Largest N-body vertex buffer GGUI is given, whatever NBODY_MAX_PARTICLES says
MAX_CIRCLE_BUFFER = 290000

# Shipped [min, max, default] for every slider, used when config.json lacks one
_SLIDER_DEFAULTS = {
    "gravity": (0.01, 100.0, 0.64),
    "gravity_range": (1.0, 8192.0, 8192.0),     # max = no screening (engine.FORCE_RANGE_OFF)
    "friction": (0.0, 5.0, 0.1),
    "pressure": (0.01, 10000.0, 1.0),
    "softening_length": (0.1, 10.0, 1.5),
    "time_scale_log": (-2.0, 2.0, 0.0),
    "brush_radius": (20.0, 1000.0, 500.0),
    "spawn_velocity": (-2.0, 2.0, 1.0),
    "spawn_density": (0.001, 2.0, 1.0),
    "bp_inner": (-0.5, 5.0, -0.5),
    "bp_outer": (-0.5, 5.0, 5.0),
    "bp_transition": (0.001, 1.0, 0.1),
    "potential_min_log": (-2.0, 10.0, 2.0),
    "potential_max_log": (-2.0, 15.0, 6.0),
    "field_min_log": (-2.0, 10.0, 0.0),
    "field_max_log": (-2.0, 10.0, 3.0),
    "spring_k": (0.001, 10.0, 0.1),
    "epsilon": (1.0, 5000.0, 1000.0),
    "sigma": (1.0, 200.0, 40.0),
    "polytropic_gamma": (0.0, 3.0, 1.0),
    "uniform_gravity_mag": (0.0, 10000.0, 0.0),
    "uniform_gravity_angle": (0.0, 360.0, 270.0),
}


@dataclass(frozen=True)
class Settings:
    # display
    width: int = 2880
    height: int = 1800
    fps_target: float = 120.0
    ui_width: float = 0.15
    pixels_per_arrow: int = 32
    minimap_interval: int = 4
    arrow_width: float = 0.0002
    arrow_length_scale: float = 1
    arrow_head_size: float = 0.25
    arrow_head_angle: float = 0.5
    # simulation
    solver: str = "pm"
    max_particles: int = 4000000
    nbody_max_particles: int = 25000
    initial_particles: int = 1000000
    grid_w: int = 2048
    grid_h: int = 2048
    physics_cell_size: float = 4.0
    render_cell_size: float = 1.0
    zoom_intensity_scale: float = 0.01
    min_log_density: float = 0.0
    max_log_density: float = 3.0
    taichi_backend: str = None
    kernel_backend: str = "auto"
    torch_compile: bool = False
    nbody_kernel: str = "auto"
    sort_interval: int = 50
    field_interval: int = 1
    show_fft: bool = True
    sliders: MappingProxyType = MappingProxyType(_SLIDER_DEFAULTS)

    @classmethod
    def from_config(cls, config):
        d = config.get("display", {})
        s = config.get("simulation", {})
        base = cls()
        sliders = dict(_SLIDER_DEFAULTS)
        sliders.update({k: tuple(v) for k, v in config.get("sliders", {}).items()})

        def pick(section, key, field):
            return section.get(key, getattr(base, field))

        return cls(
            width=pick(d, "WIDTH", "width"),
            height=pick(d, "HEIGHT", "height"),
            fps_target=pick(d, "FPS_TARGET", "fps_target"),
            ui_width=pick(d, "ui_width", "ui_width"),
            pixels_per_arrow=pick(d, "pixels_per_arrow", "pixels_per_arrow"),
            minimap_interval=max(1, int(pick(d, "minimap_interval", "minimap_interval"))),
            arrow_width=pick(d, "arrow_width", "arrow_width"),
            arrow_length_scale=pick(d, "arrow_length_scale", "arrow_length_scale"),
            arrow_head_size=pick(d, "arrow_head_size", "arrow_head_size"),
            arrow_head_angle=pick(d, "arrow_head_angle", "arrow_head_angle"),
            solver=str(pick(s, "SOLVER", "solver")).lower(),
            max_particles=pick(s, "MAX_PARTICLES", "max_particles"),
            nbody_max_particles=pick(s, "NBODY_MAX_PARTICLES", "nbody_max_particles"),
            initial_particles=pick(s, "initial_particles", "initial_particles"),
            grid_w=pick(s, "PHYSICS_GRID_W", "grid_w"),
            grid_h=pick(s, "PHYSICS_GRID_H", "grid_h"),
            physics_cell_size=pick(s, "PHYSICS_CELL_SIZE", "physics_cell_size"),
            render_cell_size=pick(s, "RENDER_CELL_SIZE", "render_cell_size"),
            zoom_intensity_scale=pick(s, "ZOOM_INTENSITY_SCALE", "zoom_intensity_scale"),
            min_log_density=pick(s, "MIN_LOG_DENSITY", "min_log_density"),
            max_log_density=pick(s, "MAX_LOG_DENSITY", "max_log_density"),
            taichi_backend=pick(s, "TAICHI_BACKEND", "taichi_backend"),
            kernel_backend=pick(s, "KERNEL_BACKEND", "kernel_backend"),
            torch_compile=pick(s, "TORCH_COMPILE", "torch_compile"),
            nbody_kernel=str(pick(s, "NBODY_KERNEL", "nbody_kernel")).lower(),
            sort_interval=pick(s, "SORT_INTERVAL", "sort_interval"),
            field_interval=max(1, int(pick(s, "FIELD_INTERVAL", "field_interval"))),
            show_fft=pick(s, "show_fft", "show_fft"),
            sliders=MappingProxyType(sliders),
        )

    # ---- derived values
    @property
    def universe_w(self):
        return int(self.grid_w * self.physics_cell_size)

    @property
    def universe_h(self):
        return int(self.grid_h * self.physics_cell_size)

    @property
    def view_w(self):
        """Viewport width in render cells."""
        return int(self.width / self.render_cell_size)

    @property
    def view_h(self):
        return int(self.height / self.render_cell_size)

    @property
    def nbody_capacity(self):
        return min(self.max_particles, self.nbody_max_particles)

    @property
    def circle_buffer_size(self):
        """N-body mode is capped at nbody_max_particles, so the circle vertex buffers are sized to that."""
        return min(MAX_CIRCLE_BUFFER, self.nbody_max_particles)

    def slider(self, name):
        """(min, max, default) for a slider."""
        return self.sliders[name]


SETTINGS = Settings.from_config(CONFIG)
