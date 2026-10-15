"""ViewState forwarding on the engine, the colormap module split and Palettes."""
import torch

from particle_mesh.render import colormap as CM
from particle_mesh.render import mesh_renderer as MR
from particle_mesh.render.themes import THEMES
from particle_mesh.settings import SETTINGS
from particle_mesh.view_state import VIEW_FIELDS, ViewState


def test_defaults_come_from_settings(make_engine):
    eng = make_engine(0)
    assert eng.view == ViewState.from_settings()
    assert eng.show_fft == SETTINGS.show_fft and not eng.show_potential and eng.color_theme == 0
    assert eng.bp_outer == SETTINGS.slider("bp_outer")[2]


def test_engine_forwards_every_view_field(make_engine):
    eng = make_engine(0)
    for i, name in enumerate(VIEW_FIELDS):
        value = getattr(eng.view, name)
        new = (not value) if isinstance(value, bool) else value + i + 1
        setattr(eng, name, new)
        assert getattr(eng.view, name) == new and getattr(eng, name) == new
    assert "show_fft" not in vars(eng)          # stored on the ViewState, not the engine


def test_colormap_reexported_by_mesh_renderer():
    for name in ("ALPHA", "_LUT_CACHE", "_color_lut", "_gradient", "_log_norm", "_pack_gray", "_pack_rgb", "colorize"):
        assert getattr(MR, name) is getattr(CM, name)


def test_palettes(make_engine):
    eng = make_engine(0)
    p = CM.Palettes(THEMES, eng.device)
    assert p.names == list(THEMES) and len(p) == len(THEMES)
    for name, m in zip(p.names, p.maps):
        assert m.shape == (9, 3) and m.dtype == torch.float32 and m.device.type == eng.device.type
        assert torch.equal(m.cpu(), torch.tensor(THEMES[name], dtype=torch.float32))
    assert p.get(1) is p.maps[1]
    assert p.cycle(len(p) - 1) == 0 and p.cycle(0, -1) == len(p) - 1
