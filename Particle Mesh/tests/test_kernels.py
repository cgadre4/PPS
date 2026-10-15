"""The torch and Taichi particle-kernel backends must agree, and torch on the accelerator must match torch on CPU.
Taichi kernels are fed CPU tensors, which every Taichi arch accepts (copied when memory isn't shared)."""
import pytest
import torch

import kernels as K

W, H, CS = 64, 48, 2.0
UW, UH = W * CS, H * CS
N = 20000


def _inputs():
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(N, 2, generator=g) * torch.tensor([UW, UH])
    # Edge cases: exactly 0, just inside the far edge, exactly at it, slightly negative, slightly beyond
    edge = torch.tensor([[0.0, 0.0], [UW - 1e-4, UH - 1e-4], [UW, UH], [-0.3, 5.0], [UW + 0.4, UH * 0.5], [1.0, UH]])
    pos[:len(edge)] = edge
    mass = torch.randint(0, 2, (N,), dtype=torch.int8, generator=g) * 2 - 1
    fx = torch.randn(H, W, generator=g)
    fy = torch.randn(H, W, generator=g)
    return pos, mass, fx, fy


def _run(backend, device, periodic):
    pos0, mass, fx, fy = _inputs()
    pos = pos0.to(device)
    vel = torch.zeros(N, 2, device=device)
    m = mass.to(device)
    dens = torch.zeros(H, W, device=device)
    stencil = backend.prepare(pos, N, CS, W, H, periodic)
    backend.deposit(stencil, pos, m, N, dens, CS, periodic)
    backend.gather_integrate(stencil, pos, vel, m, N, fx.to(device), fy.to(device), CS, 0.37, 0.9, True,
                             periodic, UW, UH)
    view = backend.splat_view(pos, N, 3.0, -2.0, 1.7, 80, 60, 2.0)
    assert view.shape == (80, 60)                               # (view_w, view_h), x-major
    return dens.cpu(), pos.cpu(), vel.cpu(), view.cpu().long(), mass


@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "clamped"])
def test_torch_matches_taichi(periodic):
    d_t, p_t, v_t, s_t, mass = _run(K.TorchBackend(), "cpu", periodic)
    d_a, p_a, v_a, s_a, _ = _run(K.TaichiBackend(), "cpu", periodic)
    assert (d_t - d_a).abs().max() < 1e-4
    assert abs(float(d_t.sum()) - float(mass.float().sum())) < 1e-2      # charge conserved by CIC
    assert (v_t - v_a).abs().max() < 1e-4
    assert (p_t - p_a).abs().max() < 1e-3
    assert torch.equal(s_t, s_a)
    if periodic:
        assert bool(((p_t >= 0) & (p_t < torch.tensor([UW, UH]) + 1e-3)).all())


@pytest.mark.skipif(not (torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())),
                    reason="no GPU torch device")
@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "clamped"])
def test_torch_gpu_matches_cpu(periodic):
    device = "cuda" if torch.cuda.is_available() else "xpu"
    d_c, p_c, v_c, s_c, _ = _run(K.TorchBackend(), "cpu", periodic)
    d_g, p_g, v_g, s_g, _ = _run(K.TorchBackend(), device, periodic)
    assert (d_g - d_c).abs().max() < 1e-4
    assert (v_g - v_c).abs().max() < 1e-4
    assert (p_g - p_c).abs().max() < 1e-3
    assert torch.equal(s_g, s_c)


def test_backend_selection_never_hands_taichi_unshared_tensors():
    import engine as E
    backend = K.make_backend(E.DEVICE, "auto")
    if K.taichi_shares_memory(E.DEVICE):
        assert backend.name == "taichi"
    else:
        assert backend.name.startswith("torch")
