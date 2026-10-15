"""Render pieces extracted from main.py / MeshRenderer: N-body colouring, viewport re-exports, presenter, overlays."""
import pytest
import torch

from particle_mesh.render import mesh_renderer as MR
from particle_mesh.render import viewport as VP
from particle_mesh.render.circle_renderer import CircleRenderer
from particle_mesh.render.ggui_present import HostFramePresenter
from particle_mesh.render.overlays import BrushOverlay
from particle_mesh.render.themes import THEMES
from test_safety_net import reference_nbody_colors


@pytest.fixture(scope="module")
def circles():
    return CircleRenderer()


@pytest.mark.parametrize("potential", ["gravity", "coulomb"])
def test_nbody_colors_match_reference(make_engine, circles, potential):
    eng = make_engine(3000, solver="nbody")
    eng.vel[:3000] = torch.randn(3000, 2, device=eng.device) * 40.0
    if potential == "coulomb":
        eng.set_potential("coulomb")
    cmap = torch.tensor(THEMES["Standard Theme"], dtype=torch.float32, device=eng.device)
    got = circles._nbody_colors(eng, cmap)
    assert torch.allclose(got, reference_nbody_colors(eng, cmap))


def test_render_nbody_passes_main_arguments(make_engine, circles, monkeypatch):
    eng = make_engine(100, solver="nbody")
    calls = []
    monkeypatch.setattr(circles, "render", lambda *a: calls.append(a))

    class Cam:
        cam_x, cam_y, width, height, zoom = 10.0, 20.0, 800.0, 600.0, 2.0

    cmap = torch.tensor(THEMES["Standard Theme"], dtype=torch.float32, device=eng.device)
    circles.render_nbody("canvas", eng, Cam, cmap)
    (canvas, pos, colors, n, cx, cy, sw, sh, radius), = calls
    assert canvas == "canvas" and n == 100 and pos.shape == (100, 2) and colors.shape == (100, 3)
    assert (cx, cy, sw, sh, radius) == (10.0, 20.0, 400.0, 300.0, 0.0005)
    eng.num_active = 0
    circles.render_nbody("canvas", eng, Cam, cmap)
    assert len(calls) == 1


def test_viewport_reexported():
    assert MR.build_view_grid is VP.build_view_grid and MR._sample is VP._sample


def test_presenter_owns_the_host_buffer(make_engine):
    eng = make_engine(0)
    renderer = MR.MeshRenderer(eng.kernels)
    assert isinstance(renderer._presenter, HostFramePresenter)
    assert renderer._host is renderer._presenter.host and renderer._host.shape == (MR.VIEW_W, MR.VIEW_H)


def test_brush_overlay_draw_calls():
    brush = BrushOverlay()
    calls = []

    class Canvas:
        def circles(self, field, radius, color):
            calls.append(("circles", color))

        def triangles(self, field, color):
            calls.append(("triangles", color))

    brush.draw_ring(Canvas(), (0.5, 0.5), 100.0, 1000, 500, erasing=False)
    brush.draw_ring(Canvas(), (0.5, 0.5), 100.0, 1000, 500, erasing=True)
    brush.draw_arrow(Canvas())
    assert calls == [("circles", (0.2, 0.8, 0.6)), ("circles", (1.0, 0.2, 0.2)), ("triangles", (0.8, 0.8, 0.8))]
    ring = brush.ring_pos_field.to_numpy()
    assert abs(ring[0][0] - 0.6) < 1e-6 and abs(ring[0][1] - 0.5) < 1e-6      # radius 100/1000 in x at angle 0
    arrow = brush.arrow_verts.to_numpy()
    assert abs(arrow[0][0] - 0.016) < 1e-6 and abs(arrow[5][1] - 0.975) < 1e-6
