"""Step-path behaviour from the performance work (AUDIT §8: F02, F03, F04, F27, F32)."""
import pytest
import torch

from particle_mesh.physics import engine as E
from particle_mesh import kernels as K
from particle_mesh.physics.grid_ops import neg_gradient


@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "clamped"])
def test_neg_gradient_matches_reference(periodic):
    """F03: slice-based differences equal the old torch.roll / torch.gradient formulas."""
    g = torch.Generator().manual_seed(5)
    phi = torch.randn(48, 64, generator=g)
    cs = 2.0
    fx, fy = torch.empty_like(phi), torch.empty_like(phi)
    neg_gradient(phi, fx, fy, cs, periodic)
    if periodic:
        rx = (torch.roll(phi, 1, 1) - torch.roll(phi, -1, 1)) * (0.5 / cs)
        ry = (torch.roll(phi, 1, 0) - torch.roll(phi, -1, 0)) * (0.5 / cs)
    else:
        gy, gx = torch.gradient(phi)
        rx, ry = -gx / cs, -gy / cs
    assert torch.allclose(fx, rx, atol=1e-6) and torch.allclose(fy, ry, atol=1e-6)


def test_neg_gradient_writes_into_padded_interiors():
    """The torch backend's force buffers are non-contiguous interiors of a padded buffer."""
    phi = torch.randn(32, 40)
    buf = torch.zeros(2, 34, 42)
    fx, fy = buf[0, 1:-1, 1:-1], buf[1, 1:-1, 1:-1]
    neg_gradient(phi, fx, fy, 1.0, True)
    ref_x = (torch.roll(phi, 1, 1) - torch.roll(phi, -1, 1)) * 0.5
    assert torch.allclose(buf[0, 1:-1, 1:-1], ref_x, atol=1e-6)
    assert float(buf[:, 0, :].abs().max()) == 0.0               # halo untouched


@pytest.mark.parametrize("backend", ["torch", "taichi"])
def test_deposit_overwrites(backend):
    """Backends overwrite the density grid (no accumulation across calls)."""
    b = K.TorchBackend() if backend == "torch" else K.TaichiBackend()
    pos = torch.rand(5000, 2) * torch.tensor([128.0, 96.0])
    mass = torch.ones(5000, dtype=torch.int8)
    dens = torch.full((48, 64), 7.0)
    st = b.prepare(pos, 5000, 2.0, 64, 48, True)
    b.deposit(st, pos, mass, 5000, dens, 2.0, True)
    first = dens.clone()
    b.deposit(st, pos, mass, 5000, dens, 2.0, True)
    assert torch.allclose(dens, first) and abs(float(dens.sum()) - 5000) < 1e-2


def test_field_interval_skips_solves(make_engine, monkeypatch):
    """F32: with FIELD_INTERVAL 2 the field is rebuilt every other step (and immediately after a spawn)."""
    eng = make_engine(20000)
    eng.field_interval = 2
    eng.precompute_kernel()
    calls = []
    real = eng.pm_solver._compute_field
    monkeypatch.setattr(eng.pm_solver, "_compute_field", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    for _ in range(6):
        eng.step(1e-3)
    assert len(calls) == 3              # steps 0 (dirty and due), 2 and 4; steps 1, 3, 5 reuse the field
    eng.add_galaxy(E.UNIVERSE_W / 2, E.UNIVERSE_H / 2, 50.0, count=1000, vel_mult=1.0)
    before = len(calls)
    eng.step(1e-3)                      # step 6 is due anyway; spawn also forces a rebuild
    eng.step(1e-3)                      # step 7: skipped
    assert len(calls) == before + 1


def test_spawn_then_erase_flushes_orbit_speeds(make_engine):
    """F04: if particles are erased before the next step, pending orbit speeds are applied first."""
    eng = make_engine(0)
    eng.precompute_kernel()
    cx, cy = E.UNIVERSE_W / 2, E.UNIVERSE_H / 2
    eng.add_galaxy(cx, cy, 200.0, count=5000, vel_mult=1.0)
    assert eng._pending_orbits
    eng.remove_particles(10.0, 10.0, 5.0)               # erase far away (nothing there)
    assert not eng._pending_orbits
    speed = eng.vel[:eng.num_active].norm(dim=1)
    assert float(speed.median()) > 0.0


@pytest.mark.parametrize("solver", ["pm", "nbody"])
def test_uniform_gravity_folded_into_kick(make_engine, centre, solver):
    """F27: uniform gravity is applied in the kick of both solvers."""
    eng = make_engine(1, solver)
    eng.G = 1e-6
    eng.precompute_kernel()
    eng.friction = 0.0
    eng.pos[0] = torch.tensor(centre)
    eng.vel[0] = 0.0
    eng.uniform_gravity_mag = 100.0
    eng.uniform_gravity_angle = 270.0                   # straight down
    eng.step(0.01)
    assert eng.vel[0, 1].item() == pytest.approx(-1.0, abs=1e-3)
    assert abs(eng.vel[0, 0].item()) < 1e-3


def test_neg_gradient_prescaled_skips_the_scale():
    """AUDIT §9 B1: with prescaled=True the caller has folded 1 / (2 dx) into the source."""
    g = torch.Generator().manual_seed(7)
    phi = torch.randn(40, 56, generator=g)
    cs = 4.0
    fx, fy = torch.empty_like(phi), torch.empty_like(phi)
    neg_gradient(phi, fx, fy, cs, True)
    px, py = torch.empty_like(phi), torch.empty_like(phi)
    neg_gradient(phi * (0.5 / cs), px, py, cs, True, prescaled=True)
    assert torch.allclose(fx, px, atol=1e-6) and torch.allclose(fy, py, atol=1e-6)


@pytest.mark.parametrize("boundary", ["periodic", "open"])
def test_phi_is_the_true_potential(make_engine, boundary):
    """The solver keeps phi / (2 dx); phi (property) undoes that, and the forces are -grad(phi)."""
    eng = make_engine(5000, boundary=boundary)
    eng.precompute_kernel()
    eng.step(1e-6)
    s = eng.pm_solver
    assert torch.allclose(s.phi, s.phi_scaled * (2.0 * E.PHYSICS_CELL_SIZE))
    fx, fy = torch.empty_like(s.phi), torch.empty_like(s.phi)
    neg_gradient(s.phi, fx, fy, E.PHYSICS_CELL_SIZE, boundary == "periodic")
    scale = float(s.fx.abs().max())
    assert (fx[2:-2, 2:-2] - s.fx[2:-2, 2:-2]).abs().max().item() <= 1e-5 * scale + 1e-12


def test_harmonic_mean_field_scale_follows_n(make_engine):
    """AUDIT §9 B2: the harmonic 1/N lives in the cached kernel, so it must be rebuilt when N changes."""
    eng = make_engine(1000)
    eng.set_potential('harmonic')
    eng.step(1e-6)
    k1 = eng.pm_solver._K_eff
    eng.add_galaxy(E.UNIVERSE_W / 2, E.UNIVERSE_H / 2, 50.0, count=1000, vel_mult=0.0)
    eng.step(1e-6)
    k2 = eng.pm_solver._K_eff
    assert eng.pm_solver._K_eff_key[2] == 2000
    assert torch.allclose(k2 * 2.0, k1, rtol=1e-5, atol=0.0)       # twice the particles, half the spring per pair
