"""
User-controlled values shared by the input handler, the panel and the frame loop.
"""
from dataclasses import dataclass

from particle_mesh.settings import SETTINGS


@dataclass
class Controls:
    radius: float           # brush radius, screen pixels
    spawn_vel: float        # spawn velocity multiplier (x circular orbit speed for gravity)
    spawn_density: float    # spawned particles per screen pixel^2
    log_time: float         # log10 of the time-scale multiplier
    ui_open: bool = True    # panel visible

    @classmethod
    def from_settings(cls, settings=SETTINGS):
        s = settings.slider
        return cls(radius=s("brush_radius")[2], spawn_vel=s("spawn_velocity")[2],
                   spawn_density=s("spawn_density")[2], log_time=s("time_scale_log")[2])
