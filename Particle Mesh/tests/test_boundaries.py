"""Open boundary (deletion + isolated, zero-padded square FFT) and the shared PM / N-body force law."""
import math

import pytest
import torch

from particle_mesh.physics import engine as E
from particle_mesh.physics.potentials import attraction

CS = E.PHYSICS_CELL_SIZE
CY = E.UNIVERSE_H / 2.0 + CS / 2
# LJ checks scale with the grid: PM resolves the core only to about the CIC width (sigma = 20 cells here)
LJ_SIGMA = 20 * CS


def _pair_accel(make_engine, solver, potential, x0, x1, boundary='open', force_range=8192.0, **params):
    """x-acceleration of particle 0 (at x0) from particle 1 (at x1), both at rest, after one tiny step."""
    eng = make_engine(2, solver, boundary)
    eng.potential_type = potential
    eng.G = params.get('G', 1.0)
    eng.spring_k = params.get('spring_k', 0.1)
    eng.epsilon = params.get('epsilon', 1000.0)
    eng.sigma = params.get('sigma', 40.0)
    eng.gravity_range = force_range
    eng.friction = 0.0
    eng.precompute_kernel()
    eng.pos[0] = torch.tensor([x0, CY])
    eng.pos[1] = torch.tensor([x1, CY])
    eng.vel[:2] = 0.0
    eng.masses[:2] = 1
    dt = 1e-3
    eng.step(dt)
    assert eng.num_active == 2
    return eng.vel[0, 0].item() / dt


def _law(potential, sep, force_range=math.inf, **params):
    p = {'G': 1.0, 'spring_k': 0.1, 'epsilon': 1000.0, 'sigma': 40.0, 'softening_length': 1.5,
         'force_range': force_range, **params}
    return float(attraction(potential, p, torch.tensor(float(sep), dtype=torch.float64)))


def test_open_boundary_deletes_escaped_particles(make_engine):
    eng = make_engine(3, boundary='open')
    eng.G = 1e-6
    eng.friction = 0.0
    eng.precompute_kernel()
    eng.pos[:3] = torch.tensor([[1.0, CY], [E.UNIVERSE_W / 2, CY], [E.UNIVERSE_W - 1.0, CY]])
    eng.vel[:3] = torch.tensor([[-2000.0, 0.0], [0.0, 0.0], [5.0, 3.0]])
    eng.step(1.0 / 120.0)
    assert eng.num_active == 2                                   # the first one left through x = 0
    assert eng.pos[0, 0].item() == pytest.approx(E.UNIVERSE_W / 2, abs=1e-2)
    assert eng.vel[1, 1].item() == pytest.approx(3.0, abs=1e-3)  # survivors keep their state, compacted in order
    for _ in range(60):
        eng.step(1.0 / 120.0)
    assert eng.num_active == 1                                   # the third crossed x = universe_w


def test_open_spawn_outside_the_mesh_is_dropped(make_engine):
    eng = make_engine(0, boundary='open')
    eng.add_galaxy(0.0, CY, 50.0, count=10000, vel_mult=0.0)     # disk centred on the left edge
    assert 3500 < eng.num_active < 6500
    assert bool((eng.pos[:eng.num_active, 0] >= 0).all())


@pytest.mark.parametrize("boundary", ["open", "reflective_walls"])
def test_isolated_solve_has_no_periodic_images(make_engine, boundary):
    """Two particles near opposite edges attract across the mesh (periodic would pull them through the edge)."""
    x0, x1 = 100.0 + CS / 2, E.UNIVERSE_W - 100.0 + CS / 2
    a = _pair_accel(make_engine, 'pm', 'gravity', x0, x1, boundary)
    assert a > 0
    assert a == pytest.approx(1.0 / (x1 - x0), rel=0.05)         # free-space |a| = G m / r
    a_periodic = _pair_accel(make_engine, 'pm', 'gravity', x0, x1, 'periodic')
    assert a_periodic < 0


@pytest.mark.parametrize("potential,sep,force_range,rel", [
    ("gravity", 20 * CS, 8192.0, 0.05),
    ("gravity", 1500 * CS, 8192.0, 0.05),
    ("gravity", 200 * CS, 300.0, 0.05),       # screened: same exp(-r/range) form in both solvers
    ("coulomb", 20 * CS, 8192.0, 0.05),
    ("harmonic", 300 * CS, 8192.0, 0.05),
    ("harmonic", 300 * CS, 100.0, 0.05),      # beyond 2 x range: still attractive
    ("lennard_jones", 0.75 * LJ_SIGMA, 8192.0, 0.6),   # inside sigma: repulsive core (steep: CIC smooths it, ~1.4x)
    ("lennard_jones", 1.5 * LJ_SIGMA, 8192.0, 0.25),   # beyond the minimum: attractive tail
])
def test_open_pm_matches_nbody_force_law(make_engine, potential, sep, force_range, rel):
    cx = E.UNIVERSE_W / 2.0 + CS / 2
    x0, x1 = cx - sep / 2, cx + sep / 2
    params = {"sigma": LJ_SIGMA} if potential == "lennard_jones" else {}
    a_pm = _pair_accel(make_engine, "pm", potential, x0, x1, "open", force_range, **params)
    a_nb = _pair_accel(make_engine, "nbody", potential, x0, x1, "open", force_range, **params)
    expected = _law(potential, x1 - x0, force_range if force_range < 8192.0 else math.inf, **params)
    if potential == 'harmonic':
        expected *= 0.5                                            # mean-field 1/N with N = 2
    assert a_nb == pytest.approx(expected, rel=1e-3)
    assert math.copysign(1, a_pm) == math.copysign(1, expected)
    assert a_pm == pytest.approx(expected, rel=rel)


def test_lennard_jones_core_repels_in_nbody(make_engine):
    """N-body LJ had the sign flipped (core attracted, tail repelled)."""
    cx = E.UNIVERSE_W / 2.0
    assert _pair_accel(make_engine, 'nbody', 'lennard_jones', cx - 15.0, cx + 15.0, 'periodic') < 0
    assert _pair_accel(make_engine, 'nbody', 'lennard_jones', cx - 30.0, cx + 30.0, 'periodic') > 0


def test_periodic_harmonic_springs_stay_attractive(make_engine):
    """The PM spring potential was screened instead of the force, so springs pushed apart beyond 2 x range."""
    cx = E.UNIVERSE_W / 2.0 + CS / 2
    assert _pair_accel(make_engine, 'pm', 'harmonic', cx - 300.0, cx + 300.0, 'periodic', 100.0) > 0


def test_isolated_kernel_is_square(make_engine):
    eng = make_engine(0, boundary='open')
    eng.precompute_kernel()
    s = 2 * max(E.GRID_W, E.GRID_H)
    assert eng.pm_solver.K_k.shape == (s, s // 2 + 1)
    assert eng.pm_solver.phi.shape == (E.GRID_H, E.GRID_W) or eng.pm_solver.field_valid is False


def test_force_range_at_max_means_no_screening(make_engine):
    eng = make_engine(0)
    eng.gravity_range = E.FORCE_RANGE_OFF
    assert eng._physics_params()['force_range'] == math.inf
    eng.gravity_range = 100.0
    assert eng._physics_params()['force_range'] == 100.0
