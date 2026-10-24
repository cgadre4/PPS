"""torch_build_hint: the startup warning when the installed torch build can't use this machine's GPU."""
import torch

from particle_mesh.runtime import device

CPU = torch.device("cpu")


def _build(monkeypatch, cuda=None, xpu=None):
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "xpu", xpu, raising=False)


def test_no_hint_on_a_gpu():
    assert device.torch_build_hint(torch.device("cuda"), has_nvidia=True) is None
    assert device.torch_build_hint(torch.device("xpu"), has_nvidia=False) is None


def test_nvidia_gpu_with_non_cuda_build_points_to_setup(monkeypatch):
    _build(monkeypatch, xpu="20260100")
    hint = device.torch_build_hint(CPU, has_nvidia=True)
    assert "not a CUDA build" in hint and "setup_env.py" in hint


def test_cuda_build_without_gpu_points_to_driver(monkeypatch):
    _build(monkeypatch, cuda="13.0")
    hint = device.torch_build_hint(CPU, has_nvidia=True)
    assert "driver" in hint and "setup_env.py" not in hint


def test_xpu_build_without_intel_gpu_points_to_setup(monkeypatch):
    _build(monkeypatch, xpu="20260100")
    assert "setup_env.py" in device.torch_build_hint(CPU, has_nvidia=False)


def test_cpu_build_mentions_setup(monkeypatch):
    _build(monkeypatch)
    assert "setup_env.py" in device.torch_build_hint(CPU, has_nvidia=False)
