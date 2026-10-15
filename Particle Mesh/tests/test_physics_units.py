"""Units and model consistency (AUDIT 2.11, 2.9, 2.12)."""
import math

import pytest
import torch

from particle_mesh.physics import engine as E

CS = E.PHYSICS_CELL_SIZE


def _pair_accel(make_engine, solver, potential, sep, q=(1, 1), G=1.0, spring_k=0.1):
    """Acceleration of the left particle of a pair after one tiny step from rest."""
    eng = make_engine(2, solver)
    eng.potential_type = potential
    eng.G = G
    eng.spring_k = spring_k
    eng.gravity_range = 8192.0
    eng.friction = 0.0
    eng.precompute_kernel()
    cx, cy = E.UNIVERSE_W / 2.0 + CS / 2, E.UNIVERSE_H / 2.0 + CS / 2
    eng.pos[0] = torch.tensor([cx - sep / 2, cy])
    eng.pos[1] = torch.tensor([cx + sep / 2, cy])
    eng.vel[:2] = 0.0
    eng.masses[0], eng.masses[1] = q
    dt = 1e-3
    eng.step(dt)
    return eng.vel[0, 0].item() / dt


@pytest.mark.parametrize("potential,q", [("gravity", (1, 1)), ("coulomb", (1, -1)), ("coulomb", (1, 1))])
def test_pm_matches_nbody_inverse_r_law(make_engine, potential, q):
    """Same G -> same 2D |a| = G*m/r in both solvers (was 2*pi and cell_size^2 apart)."""
    sep = 20 * CS
    a_pm = _pair_accel(make_engine, 'pm', potential, sep, q)
    a_nb = _pair_accel(make_engine, 'nbody', potential, sep, q)
    assert math.copysign(1, a_pm) == math.copysign(1, a_nb)
    assert a_pm == pytest.approx(a_nb, rel=0.15)
    assert abs(a_nb) == pytest.approx(1.0 / sep, rel=0.02)          # |a| = G*m/r with G = m = 1


def test_pm_matches_nbody_harmonic(make_engine):
    """Harmonic springs use the same mean-field (1/N) scaling in both solvers."""
    sep = 20 * CS
    a_pm = _pair_accel(make_engine, 'pm', 'harmonic', sep)
    a_nb = _pair_accel(make_engine, 'nbody', 'harmonic', sep)
    assert a_pm > 0 and a_nb > 0
    assert a_pm == pytest.approx(a_nb, rel=0.2)


def _clump(make_engine, n, radius, potential='gravity', mixed_charge=False):
    eng = make_engine(0)
    eng.potential_type = potential
    eng.G = 1e-6
    eng.friction = 0.0
    eng.pressure_model = 'isothermal'
    eng.pressure_strength = 100.0
    eng.precompute_kernel()
    cx, cy = E.UNIVERSE_W / 2.0, E.UNIVERSE_H / 2.0
    g = torch.Generator(device=eng.device).manual_seed(3)
    eng.pos[:n] = torch.randn(n, 2, device=eng.device, generator=g) * radius + torch.tensor([cx, cy], device=eng.device)
    eng.vel[:n] = 0.0
    eng.masses[:n] = 1
    if mixed_charge:
        eng.masses[:n] = torch.randint(0, 2, (n,), device=eng.device, dtype=torch.int8, generator=g) * 2 - 1
    eng.num_active = n
    dt = 1e-3
    eng.step(dt)
    d = eng.pos[:n] - torch.tensor([cx, cy], device=eng.device)
    radial = (eng.vel[:n] * d).sum(1) / d.norm(dim=1).clamp(min=1e-3) / dt
    return eng, radial


def test_isothermal_pressure_depends_on_profile_not_amplitude(make_engine):
    """a = -K grad(ln rho): a 100x denser clump of the same shape gets ~the same push (old form: ~100x)."""
    _, sparse = _clump(make_engine, 20000, 30.0)
    _, dense = _clump(make_engine, 2000000, 30.0)
    s, d = sparse.median().item(), dense.median().item()
    assert s > 0 and d > 0                                  # pressure pushes outward
    assert 0.5 < d / s < 2.0


def test_pressure_in_coulomb_mode_ignores_charge_sign(make_engine):
    eng, radial = _clump(make_engine, 200000, 30.0, potential='coulomb', mixed_charge=True)
    q = eng.masses[:eng.num_active]
    assert radial[q > 0].median().item() > 0
    assert radial[q < 0].median().item() > 0


@pytest.mark.parametrize("vel_mult,expect_stable", [(1.0, True), (0.0, False)])
@pytest.mark.parametrize("uniform_spawn", [False, True], ids=["dense_core", "uniform_disk"])
def test_spawned_galaxy_equilibrium(make_engine, vel_mult, expect_stable, uniform_spawn):
    """A disk spawned at the measured circular speed keeps its size (within 10% over 2 s, checked every
    0.5 s); one spawned at rest collapses."""
    eng = make_engine(0)
    eng.friction = 0.0
    eng.uniform_spawn = uniform_spawn
    eng.precompute_kernel()
    cx, cy = E.UNIVERSE_W / 2.0, E.UNIVERSE_H / 2.0
    eng.add_galaxy(cx, cy, 300.0, count=200000, vel_mult=vel_mult)
    centre = torch.tensor([cx, cy], device=eng.device)
    r0 = (eng.pos[:eng.num_active] - centre).norm(dim=1).median().item()
    ratios = []
    for i in range(240):
        eng.step(1.0 / 120.0)
        if (i + 1) % 60 == 0:
            ratios.append((eng.pos[:eng.num_active] - centre).norm(dim=1).median().item() / r0)
    if expect_stable:
        assert all(0.9 < q < 1.1 for q in ratios), f"median radius ratios {[round(q, 3) for q in ratios]}"
    else:
        assert min(ratios) < 0.8, f"median radius ratios {[round(q, 3) for q in ratios]}"


def test_nbody_galaxy_equilibrium(make_engine):
    """Same equilibrium in direct N-body mode (velocities from the N-body field)."""
    eng = make_engine(0, 'nbody')
    eng.friction = 0.0
    cx, cy = E.UNIVERSE_W / 2.0, E.UNIVERSE_H / 2.0
    eng.add_galaxy(cx, cy, 300.0, count=5000, vel_mult=1.0)
    centre = torch.tensor([cx, cy], device=eng.device)
    r0 = (eng.pos[:eng.num_active] - centre).norm(dim=1).median().item()
    for _ in range(60):
        eng.step(1.0 / 120.0)
    ratio = (eng.pos[:eng.num_active] - centre).norm(dim=1).median().item() / r0
    assert 0.9 < ratio < 1.1, f"median radius ratio {ratio:.3f}"
