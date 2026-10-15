"""Rendering checks: splat + colour map + overlay land in the presented frame, and the advanced views
(grid_sample of solver grids) line up with the particle density view when RENDER_CELL_SIZE != 1."""
import pytest
import torch

import mesh_renderer as MR
from camera import SimulationCamera


def test_splat_colormap_and_overlay_reach_the_frame(make_engine, headless_canvas):
    window, canvas = headless_canvas
    eng = make_engine()
    renderer = MR.MeshRenderer(eng.kernels)
    red = torch.tensor([[255, 0, 0]] * 9, dtype=torch.float32, device=eng.device)
    cam_x, cam_y = (eng.universe_w - MR.WIDTH) / 2.0, (eng.universe_h - MR.HEIGHT) / 2.0
    gx, gy = 1000, 600
    wx = cam_x + (gx + 0.5) * MR.RENDER_CELL_SIZE
    wy = cam_y + (gy + 0.5) * MR.RENDER_CELL_SIZE
    eng.pos[:500] = torch.tensor([wx, wy])
    eng.num_active = 500
    thumb = torch.full((50, 60), 200, dtype=torch.uint8, device=eng.device)
    renderer.render(canvas, eng.pos, eng.num_active, cam_x, cam_y, 1.0, red, 0.0, 3.0, 0.01, overlay=(thumb, 60, 50))
    img = window.get_image_buffer_as_numpy()                     # [x, y, rgba], y = 0 at the bottom
    assert img[gx, gy, 0] > 0.9 and img[gx, gy, 1] < 0.1        # clump drawn at its render cell
    assert float(img[1300, 100, :3].max()) == 0.0               # empty cells stay black
    assert float(img[:60, :50, :3].mean()) == pytest.approx(200 / 255, abs=0.02)   # overlay pasted bottom-left


@pytest.mark.parametrize("zoom,frac", [(1.0, (0.70, 0.40)), (3.0, (0.30, 0.65)), (0.5, (0.55, 0.30))])
def test_view_grid_aligns_with_particle_view(make_engine, zoom, frac):
    """AUDIT 2.10: a marker on the physics grid and particles at the same world point land in the same cell."""
    eng = make_engine()
    rcs = MR.RENDER_CELL_SIZE
    cs = eng.pm_solver.cell_size
    gh, gw = eng.pm_solver.density_grid.shape
    cam = SimulationCamera(MR.WIDTH, MR.HEIGHT, eng.universe_w, eng.universe_h)
    if zoom != 1.0:
        cam.zoom_around_mouse(MR.WIDTH * 0.5, MR.HEIGHT * 0.5, zoom)
    wx, wy = cam.screen_to_world(frac[0] * MR.WIDTH, frac[1] * MR.HEIGHT)
    i, j = int(wx // cs), int(wy // cs)
    mx, my = (i + 0.5) * cs, (j + 0.5) * cs

    pos = torch.tensor([[mx, my]], device=eng.device).repeat(500, 1)
    counts = eng.kernels.splat_view(pos, 500, cam.cam_x, cam.cam_y, cam.zoom, MR.VIEW_W, MR.VIEW_H, rcs)
    pidx = int(torch.argmax(counts))
    particle_cell = (pidx % MR.VIEW_W, pidx // MR.VIEW_W)
    expected = (int(((mx - cam.cam_x) * cam.zoom) // rcs), int(((my - cam.cam_y) * cam.zoom) // rcs))
    assert particle_cell == expected

    jj, ii = torch.meshgrid(torch.arange(gh, device=eng.device, dtype=torch.float32),
                            torch.arange(gw, device=eng.device, dtype=torch.float32), indexing='ij')
    blob = torch.exp(-((ii - i) ** 2 + (jj - j) ** 2) / (2 * 3.0 ** 2))
    grid = MR.build_view_grid(cam, eng.universe_w, eng.universe_h, eng.device)
    sampled = torch.nn.functional.grid_sample(blob[None, None], grid, align_corners=False)[0, 0]
    sidx = int(torch.argmax(sampled))
    view_cell = (sidx % sampled.shape[1], sidx // sampled.shape[1])
    assert abs(view_cell[0] - particle_cell[0]) <= 1 and abs(view_cell[1] - particle_cell[1]) <= 1
