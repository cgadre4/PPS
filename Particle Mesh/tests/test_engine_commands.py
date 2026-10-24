"""Engine commands used by the GUI panel, and the N-body startup cap the engine now applies itself."""
import pytest

from particle_mesh.physics import engine as E


@pytest.mark.parametrize("kind", ["gravity", "lennard_jones", "harmonic"])
def test_set_potential_resets_masses_to_one(make_engine, kind):
    eng = make_engine(500)
    eng.set_potential("coulomb")
    eng.set_potential(kind)
    assert eng.potential_type == kind
    assert bool((eng.masses[:500] == 1).all())


def test_set_potential_coulomb_randomises_charges(make_engine):
    eng = make_engine(2000)
    eng.set_potential("coulomb")
    m = eng.masses[:2000]
    assert bool(((m == 1) | (m == -1)).all())
    assert 0 < int((m == -1).sum()) < 2000
    assert int(eng.masses[2000:].abs().sum()) == 0      # beyond the live prefix untouched


def test_set_potential_rebuilds_kernel(make_engine):
    eng = make_engine(100)
    before = eng.pm_solver.K_k
    eng.set_potential("harmonic")
    assert eng.pm_solver.K_k is not before
    assert eng._field_dirty and eng._kernel_dirty_since is None


def test_set_potential_rejects_unknown(make_engine):
    eng = make_engine(10)
    with pytest.raises(ValueError):
        eng.set_potential("magnetism")
    assert eng.potential_type == "gravity"


def test_pressure_model(make_engine):
    eng = make_engine(10)
    for model in ("isothermal", "polytropic", "none"):
        eng.set_pressure_model(model)
        assert eng.pressure_model == model
    with pytest.raises(ValueError):
        eng.set_pressure_model("steam")


def test_toggle_boundary(make_engine):
    eng = make_engine(10)
    assert eng.boundary_mode == "periodic" and eng.periodic
    eng.toggle_boundary()
    assert eng.boundary_mode == "open" and not eng.periodic
    eng.toggle_boundary()
    assert eng.boundary_mode == "periodic"


def test_open_is_the_default_boundary():
    from particle_mesh.physics import engine as E
    eng = E.SimulationEngine(initial_particles=10)
    assert eng.boundary_mode == "open" and not eng.periodic


def test_nbody_startup_cap(capsys):
    n = E.NBODY_MAX_PARTICLES + 1
    eng = E.SimulationEngine(initial_particles=n, solver="nbody")
    assert eng.num_active == E.NBODY_MAX_PARTICLES and eng.solver_type == "nbody"
    out = capsys.readouterr().out
    assert f"[Warning] initial_particles ({n}) is extremely high for O(N^2) Direct N-Body mode." in out
    assert f"Capping to NBODY_MAX_PARTICLES ({E.NBODY_MAX_PARTICLES})" in out


def test_pm_has_no_cap():
    n = E.NBODY_MAX_PARTICLES + 1
    eng = E.SimulationEngine(initial_particles=n, solver="pm")
    assert eng.num_active == n and eng.solver_type == "pm"
