"""
What the user has chosen to look at: view toggles, colour ranges, band-pass settings and the colour theme.
Pure Python (bottom layer), so the engine can hold one; the renderers and the panel read and write it.
"""
from dataclasses import dataclass

from particle_mesh.settings import SETTINGS


@dataclass
class ViewState:
    show_fft: bool = True
    use_bandpass_render: bool = False
    show_potential: bool = False
    show_field: bool = False
    pot_min_log: float = 2.0
    pot_max_log: float = 6.0
    field_min_log: float = 0.0
    field_max_log: float = 3.0
    bp_inner: float = -0.5
    bp_outer: float = 5.0
    bp_transition: float = 0.1
    color_theme: int = 0

    @classmethod
    def from_settings(cls, settings=SETTINGS):
        s = settings.slider
        return cls(
            show_fft=settings.show_fft,
            pot_min_log=s("potential_min_log")[2], pot_max_log=s("potential_max_log")[2],
            field_min_log=s("field_min_log")[2], field_max_log=s("field_max_log")[2],
            bp_inner=s("bp_inner")[2], bp_outer=s("bp_outer")[2], bp_transition=s("bp_transition")[2],
        )


# Names the engine forwards to its ViewState (engine.show_fft is engine.view.show_fft)
VIEW_FIELDS = tuple(ViewState.__dataclass_fields__)
