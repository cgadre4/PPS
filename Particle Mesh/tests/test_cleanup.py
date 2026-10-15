"""Checks for the cleanup/performance batch (AUDIT 3.1, 3.5, 3.6)."""
import time

import pytest
import torch

import engine as E
from nbody_solver import DirectNBodySolver


def test_kernels_are_half_spectrum_and_real_where_possible(make_engine):
    """AUDIT 3.1: kernels live in rfft2 layout; gravity/Coulomb are real (no full complex grid)."""
    eng = make_engine()
    H, W = E.GRID_H, E.GRID_W
    params = dict(G=1.0, force_range=100.0, spring_k=0.1, epsilon=1000.0, sigma=40.0)
    for potential in ('gravity', 'coulomb'):
        K = eng.pot_manager.get_fourier_kernel(potential, params)
        assert K.shape == (H, W // 2 + 1) and not K.is_complex()
    for potential in ('harmonic', 'lennard_jones'):
        K = eng.pot_manager.get_fourier_kernel(potential, params)
        assert K.shape == (H, W // 2 + 1) and K.is_complex()
    assert eng.pm_solver.k_sq is eng.pot_manager.k_sq                  # shared, not duplicated
    assert not hasattr(eng.pot_manager, 'kx_grid') and not hasattr(eng.pot_manager, 'r')


@pytest.mark.parametrize("potential", ["gravity", "harmonic", "lennard_jones", "coulomb"])
@pytest.mark.parametrize("periodic", [True, False])
def test_nbody_tiling_matches_single_tile(potential, periodic):
    """AUDIT 3.5: row tiling must not change the result."""
    dev = E.DEVICE
    g = torch.Generator(device=dev).manual_seed(7)
    N = 1500
    pos = torch.rand(N, 2, device=dev, generator=g) * 800.0
    pos[1] = pos[0] + 0.01                                             # near-coincident pair (clamps/softening)
    masses = torch.randint(0, 2, (N,), device=dev, dtype=torch.int8, generator=g) * 2 - 1
    params = dict(G=0.64, force_range=300.0, spring_k=0.1, epsilon=1000.0, sigma=20.0, softening_length=1.5)
    whole = DirectNBodySolver(dev, tile_elements=N * N).solve(pos, masses, potential, params, periodic, 800.0, 800.0)
    tiled = DirectNBodySolver(dev, tile_elements=N * 37).solve(pos, masses, potential, params, periodic, 800.0, 800.0)
    for a, b in zip(whole, tiled):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-3)


def test_slider_kernel_rebuild_is_debounced(make_engine, centre):
    """AUDIT 3.6: a slider change rebuilds the kernel only after it has been still for KERNEL_REBUILD_DELAY."""
    eng = make_engine(1)
    eng.precompute_kernel()
    eng.pos[0] = torch.tensor(centre)
    before = eng.pm_solver.K_k
    eng.G *= 2.0
    eng.request_kernel_rebuild()
    eng.step(1e-4)
    assert eng.pm_solver.K_k is before                                  # still dragging: no rebuild yet
    time.sleep(E.KERNEL_REBUILD_DELAY + 0.05)
    eng.step(1e-4)
    assert eng.pm_solver.K_k is not before
    ratio = float(eng.pm_solver.K_k[1, 1] / before[1, 1])
    assert ratio == pytest.approx(2.0, rel=1e-5)
