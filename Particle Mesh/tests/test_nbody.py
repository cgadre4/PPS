"""N-body: the fused Taichi kernel (AUDIT §8 F36) must match the tiled torch reference."""
import math

import pytest
import torch

from particle_mesh.physics import engine as E
from particle_mesh.physics.nbody_solver import DirectNBodySolver


@pytest.mark.parametrize("potential", ["gravity", "harmonic", "lennard_jones", "coulomb"])
@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "open"])
@pytest.mark.parametrize("force_range", [300.0, math.inf], ids=["screened", "unscreened"])
def test_taichi_kernel_matches_torch(potential, periodic, force_range):
    dev = E.DEVICE
    g = torch.Generator(device=dev).manual_seed(11)
    N = 1500
    pos = torch.rand(N, 2, device=dev, generator=g) * 800.0
    pos[1] = pos[0] + 0.01                                   # near-coincident pair (softening / LJ clamp)
    masses = torch.randint(0, 2, (N,), device=dev, dtype=torch.int8, generator=g) * 2 - 1
    params = dict(G=0.64, force_range=force_range, spring_k=0.1, epsilon=1000.0, sigma=20.0, softening_length=1.5)
    ref = DirectNBodySolver(dev, use_taichi=False).solve(pos, masses, potential, params, periodic, 800.0, 800.0)
    fused = DirectNBodySolver(dev, use_taichi=True).solve(pos, masses, potential, params, periodic, 800.0, 800.0)
    for a, b in zip(ref, fused):
        assert a.device == b.device
        scale = float(a.abs().max()) + 1e-6
        assert float((a - b).abs().max()) <= 1e-4 * scale + 1e-4


def test_engine_uses_fused_kernel_by_default(make_engine):
    eng = make_engine(100, 'nbody')
    assert eng.nbody_solver.use_taichi
