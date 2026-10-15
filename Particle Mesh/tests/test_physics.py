"""Physics behaviour checks (each was a verified audit bug or a property of the periodic port)."""
import math

import numpy as np
import pytest
import torch

from particle_mesh.physics import engine as E

UW, UH = E.UNIVERSE_W, E.UNIVERSE_H
CS = E.PHYSICS_CELL_SIZE


@pytest.mark.parametrize("solver", ["pm", "nbody"])
def test_friction_never_flips_velocity(make_engine, centre, solver):
    """AUDIT 2.3: damping is exp(-friction*dt), even at max friction x max time scale."""
    eng = make_engine(1, solver)
    eng.G = 1e-6
    eng.precompute_kernel()
    eng.pos[0] = torch.tensor(centre)
    eng.vel[0] = torch.tensor([1.0, 0.0])
    eng.friction = 5.0
    dt = 100.0 / 120.0
    eng.step(dt)
    assert eng.vel[0, 0].item() == pytest.approx(math.exp(-5.0 * dt), abs=1e-3)


@pytest.mark.parametrize("potential", ["gravity", "harmonic", "lennard_jones", "coulomb"])
def test_kernel_mean_mode_is_zero(make_engine, potential):
    """AUDIT 2.7: K[0,0] = 0 so phi has zero mean."""
    eng = make_engine()
    params = dict(G=1.0, force_range=8192.0, spring_k=0.1, epsilon=1000.0, sigma=40.0)
    K = eng.pot_manager.get_fourier_kernel(potential, params)
    assert abs(complex(K[0, 0].item())) == 0.0


def test_phi_has_zero_mean_at_long_range(make_engine):
    eng = make_engine(200000)
    eng.gravity_range = 8192.0
    eng.G = 1.0
    eng.precompute_kernel()
    eng.step(1e-6)
    phi = eng.pm_solver.phi
    assert abs(phi.mean().item()) < 1e-3 * phi.std().item() + 1e-6


def _pair(make_engine, potential, q_left, q_right, x_left, x_right, y, solver='pm', G=100.0):
    eng = make_engine(2, solver)
    eng.potential_type = potential
    eng.G = G
    eng.gravity_range = 8192.0
    eng.friction = 0.0
    eng.precompute_kernel()
    eng.pos[0] = torch.tensor([x_left, y])
    eng.pos[1] = torch.tensor([x_right, y])
    eng.vel[:2] = 0.0
    eng.masses[0] = q_left
    eng.masses[1] = q_right
    eng.step(0.01)
    return eng.vel[0, 0].item(), eng.vel[1, 0].item()


@pytest.mark.parametrize("solver", ["pm", "nbody"])
@pytest.mark.parametrize("potential,q_left,q_right,attract", [
    ("gravity", 1, 1, True),
    ("coulomb", 1, -1, True),
    ("coulomb", 1, 1, False),
    ("coulomb", -1, -1, False),
])
def test_pair_force_signs(make_engine, centre, solver, potential, q_left, q_right, attract):
    """AUDIT 2.2: Coulomb uses each particle's own charge; gravity attracts."""
    cx, cy = centre
    vl, vr = _pair(make_engine, potential, q_left, q_right, cx - 20, cx + 20, cy, solver)
    if attract:
        assert vl > 0 > vr
    else:
        assert vl < 0 < vr


def test_spawned_coulomb_charges_are_plus_minus_one(make_engine, centre):
    eng = make_engine()
    eng.potential_type = 'coulomb'
    eng.add_galaxy(*centre, 50.0, count=2000)
    q = eng.masses[:eng.num_active].float()
    assert bool(((q == 1) | (q == -1)).all())
    assert (q == 1).any() and (q == -1).any()


def test_nbody_particle_cap(make_engine, centre):
    """AUDIT 2.5: brush and image spawning respect NBODY_MAX_PARTICLES; PM is not capped by it."""
    eng = make_engine(100, 'nbody')
    eng.add_galaxy(*centre, 200.0, count=10**6)
    assert eng.num_active == E.NBODY_MAX_PARTICLES
    eng.load_particles_from_array(np.full((5000, 2), centre[0], dtype=np.float32))
    assert eng.num_active == E.NBODY_MAX_PARTICLES
    pm = make_engine(100, 'pm')
    pm.add_galaxy(*centre, 200.0, count=50000)
    assert pm.num_active == 50100


def test_periodic_particle_reappears_on_opposite_edge(make_engine, centre):
    """AUDIT 2.8: particles wrap instead of being deleted."""
    eng = make_engine(1)
    eng.G = 1e-6
    eng.precompute_kernel()
    eng.friction = 0.0
    eng.pos[0] = torch.tensor([UW - 1.0, centre[1]])
    eng.vel[0] = torch.tensor([300.0, 0.0])
    eng.step(0.01)
    assert eng.num_active == 1
    assert 0.0 <= eng.pos[0, 0].item() < 5.0


@pytest.mark.parametrize("solver", ["pm", "nbody"])
def test_periodic_attraction_across_edge(make_engine, centre, solver):
    vl, vr = _pair(make_engine, 'gravity', 1, 1, 10.0, UW - 10.0, centre[1], solver)
    assert vl < 0 < vr          # pulled toward each other through the x boundary


def test_reflective_walls_bounce(make_engine, centre):
    eng = make_engine(1, boundary='reflective_walls')
    eng.G = 1e-6
    eng.precompute_kernel()
    eng.friction = 0.0
    eng.pos[0] = torch.tensor([UW + 5.0, centre[1]])
    eng.vel[0] = torch.tensor([10.0, 0.0])
    eng.step(0.001)
    assert eng.vel[0, 0].item() < 0
    assert eng.pos[0, 0].item() <= UW


def test_sorting_preserves_physics(make_engine):
    a = make_engine(200000)
    b = make_engine(0)
    n = a.num_active
    b.pos[:n] = a.pos[:n]
    b.vel[:n] = a.vel[:n]
    b.masses[:n] = a.masses[:n]
    b.num_active = n
    for eng in (a, b):
        eng.G = 1.0
        eng.precompute_kernel()
    b.sort_particles()
    keys = b.sort_keys().to(torch.int32)                       # tile keys (int16 on the 2048^2 grid)
    assert bool((keys[1:] >= keys[:-1]).all())
    a.step(0.01)
    b.step(0.01)
    assert (a.pm_solver.density_grid - b.pm_solver.density_grid).abs().max() < 1e-3
    ea = float(torch.sort(a.vel[:n].norm(dim=1))[0].sum())
    eb = float(torch.sort(b.vel[:n].norm(dim=1))[0].sum())
    assert ea == pytest.approx(eb, rel=1e-4)
