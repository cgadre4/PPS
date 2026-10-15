"""Safety-net tests written before the package refactor: they pin behaviour that had no coverage."""
import numpy as np
import pytest
import torch

import engine as E
import mesh_renderer as MR


def reference_nbody_colors(engine, color_map):
    """The N-body circle colouring as main.py computed it before the refactor (speed gradient, or charge colours)."""
    n = engine.num_active
    if engine.potential_type == 'coulomb':
        charges = engine.masses[:n]
        color = torch.zeros((n, 3), dtype=torch.float32, device=engine.device)
        color[charges > 0] = torch.tensor([255.0, 50.0, 50.0], device=engine.device)
        color[charges < 0] = torch.tensor([50.0, 150.0, 255.0], device=engine.device)
        return color
    speed = torch.norm(engine.vel[:n], dim=1)
    norm_speed = torch.clamp(speed / 50.0, 0.0, 1.0)
    idx_f = norm_speed * 8.0
    idx_lower = torch.clamp(idx_f.long(), 0, 7)
    idx_upper = torch.clamp(idx_lower + 1, 0, 8)
    weight = (idx_f - idx_lower.float()).unsqueeze(1)
    return color_map[idx_lower] * (1.0 - weight) + color_map[idx_upper] * weight


def test_remove_particles_keeps_survivors_in_order(make_engine):
    eng = make_engine(0)
    cx, cy = E.UNIVERSE_W / 2, E.UNIVERSE_H / 2
    clumps = [(cx - 300, cy), (cx, cy), (cx + 300, cy)]
    per = 100
    for k, (x, y) in enumerate(clumps):
        sl = slice(k * per, (k + 1) * per)
        eng.pos[sl] = torch.tensor([x, y]) + torch.randn(per, 2) * 5
        eng.vel[sl] = torch.tensor([float(k), -float(k)])
        eng.masses[sl] = 1
    eng.num_active = 3 * per
    expected_pos = torch.cat((eng.pos[:per], eng.pos[2 * per:3 * per])).clone()
    expected_vel = torch.cat((eng.vel[:per], eng.vel[2 * per:3 * per])).clone()
    eng.remove_particles(cx, cy, 60.0)
    assert eng.num_active == 2 * per
    assert torch.equal(eng.pos[:2 * per], expected_pos)
    assert torch.equal(eng.vel[:2 * per], expected_vel)


def test_present_public_fallback_matches_private_bytes(make_engine):
    eng = make_engine()
    renderer = MR.MeshRenderer(eng.kernels)

    class FakeCanvas:            # no .canvas attribute -> the public set_image(numpy) fallback
        def __init__(self):
            self.received = None

        def set_image(self, img):
            self.received = np.array(img)

    g = torch.Generator().manual_seed(2)
    image = torch.randint(-2**31, 2**31 - 1, (MR.VIEW_W, MR.VIEW_H), dtype=torch.int32, generator=g).to(eng.device)
    fake = FakeCanvas()
    renderer._present(fake, image.clone(), None)
    expected = renderer._host_np.view(np.uint8).reshape(MR.VIEW_W, MR.VIEW_H, 4)[..., :3]
    assert fake.received is not None and np.array_equal(fake.received, expected)
    assert np.array_equal(renderer._host_np.view(np.int32), image.cpu().numpy())


def test_count_lut_matches_colorize(make_engine):
    eng = make_engine()
    renderer = MR.MeshRenderer(eng.kernels)
    cmap = torch.tensor([[i * 30, 255 - i * 25, (i * 77) % 256] for i in range(9)], dtype=torch.float32,
                        device=eng.device)
    lut = renderer._counts_lut(cmap, 0.0, 3.0, 1.0)
    counts = torch.arange(lut.shape[0] + 5, device=eng.device)
    via_lut = lut[counts.clamp(max=lut.shape[0] - 1)]
    via_float = MR.colorize(counts.to(torch.float32), cmap, 0.0, 3.0, 1.0)
    assert torch.equal(via_lut, via_float)


@pytest.mark.parametrize("potential", ["gravity", "coulomb"])
def test_reference_nbody_colors_are_well_formed(make_engine, potential):
    eng = make_engine(500, 'nbody')
    eng.potential_type = potential
    if potential == 'coulomb':
        eng.masses[:500] = torch.randint(0, 2, (500,), dtype=torch.int8, device=eng.device) * 2 - 1
    eng.vel[:500] = torch.randn(500, 2, device=eng.device) * 40
    cmap = torch.tensor([[i * 28, i * 20, 255 - i * 28] for i in range(9)], dtype=torch.float32, device=eng.device)
    colors = reference_nbody_colors(eng, cmap)
    assert colors.shape == (500, 3) and float(colors.min()) >= 0 and float(colors.max()) <= 255
