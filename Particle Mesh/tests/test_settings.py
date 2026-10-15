"""Settings: the typed view must agree with config.json and with the module constants derived from it."""
import pytest

from particle_mesh.settings import CONFIG, SETTINGS, Settings, _SLIDER_DEFAULTS

D, S = CONFIG["display"], CONFIG["simulation"]

FIELDS = [
    ("width", D, "WIDTH"), ("height", D, "HEIGHT"), ("fps_target", D, "FPS_TARGET"), ("ui_width", D, "ui_width"),
    ("pixels_per_arrow", D, "pixels_per_arrow"), ("minimap_interval", D, "minimap_interval"),
    ("arrow_width", D, "arrow_width"), ("arrow_length_scale", D, "arrow_length_scale"),
    ("arrow_head_size", D, "arrow_head_size"), ("arrow_head_angle", D, "arrow_head_angle"),
    ("max_particles", S, "MAX_PARTICLES"), ("nbody_max_particles", S, "NBODY_MAX_PARTICLES"),
    ("initial_particles", S, "initial_particles"), ("grid_w", S, "PHYSICS_GRID_W"), ("grid_h", S, "PHYSICS_GRID_H"),
    ("physics_cell_size", S, "PHYSICS_CELL_SIZE"), ("render_cell_size", S, "RENDER_CELL_SIZE"),
    ("zoom_intensity_scale", S, "ZOOM_INTENSITY_SCALE"), ("min_log_density", S, "MIN_LOG_DENSITY"),
    ("max_log_density", S, "MAX_LOG_DENSITY"), ("kernel_backend", S, "KERNEL_BACKEND"),
    ("torch_compile", S, "TORCH_COMPILE"), ("sort_interval", S, "SORT_INTERVAL"),
    ("field_interval", S, "FIELD_INTERVAL"),
]


@pytest.mark.parametrize("field,section,key", FIELDS, ids=[f for f, _, _ in FIELDS])
def test_fields_match_config(field, section, key):
    if key in section:
        assert getattr(SETTINGS, field) == section[key]


def test_sliders_match_config():
    for name, triple in CONFIG["sliders"].items():
        assert SETTINGS.slider(name) == tuple(triple)


def test_fallbacks_are_the_shipped_values():
    """An empty config gives the shipped config.json values (one fallback per key)."""
    empty = Settings.from_config({})
    shipped = Settings.from_config(CONFIG)
    for field, _, _ in FIELDS:
        assert getattr(empty, field) == getattr(shipped, field), field
    assert dict(empty.sliders) == _SLIDER_DEFAULTS
    assert {k: tuple(v) for k, v in CONFIG["sliders"].items()} == _SLIDER_DEFAULTS


def test_settings_is_frozen():
    with pytest.raises(Exception):
        SETTINGS.width = 1


def test_module_constants_follow_settings():
    from particle_mesh.physics import engine as E
    from particle_mesh.render import circle_renderer as CR
    from particle_mesh.render import mesh_renderer as MR
    assert (E.UNIVERSE_W, E.UNIVERSE_H) == (int(S["PHYSICS_GRID_W"] * S["PHYSICS_CELL_SIZE"]),
                                            int(S["PHYSICS_GRID_H"] * S["PHYSICS_CELL_SIZE"]))
    assert (MR.VIEW_W, MR.VIEW_H) == (int(D["WIDTH"] / S["RENDER_CELL_SIZE"]), int(D["HEIGHT"] / S["RENDER_CELL_SIZE"]))
    assert CR.MAX_RENDER_PARTICLES == min(290000, S["NBODY_MAX_PARTICLES"])
    assert E.NBODY_MAX_PARTICLES == S["NBODY_MAX_PARTICLES"] and E.MAX_PARTICLES == S["MAX_PARTICLES"]
