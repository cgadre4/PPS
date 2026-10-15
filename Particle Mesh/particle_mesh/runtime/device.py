"""
Torch device selection and torch <-> Taichi interop helpers (no Taichi kernels here).
"""
import functools

import taichi as ti
import torch

_CONSTS = {}


@functools.cache
def get_torch_device():
    """CUDA if available, else Intel XPU, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def taichi_shares_memory(device):
    """True when Taichi kernels can read/write tensors on `device` without host round-trips. Requires ti.init()."""
    arch = ti.lang.impl.current_cfg().arch
    if device.type == 'cuda':
        return arch == ti.cuda
    if device.type == 'cpu':
        return arch in (ti.x64, ti.arm64)
    return False


def to_taichi(tensor):
    """Tensor to hand to field.from_torch(): as-is when Taichi shares its memory, else one explicit host copy
    (passing a device tensor would make Taichi copy it to the host AND back again after the kernel)."""
    return tensor if taichi_shares_memory(tensor.device) else tensor.cpu()


def const_tensor(values, device, dtype):
    """Small constant tensor, created once per (values, device, dtype). Creating it per call costs a host->device
    copy and an implicit queue sync on every step (AUDIT §8 F07)."""
    key = (values, str(device), dtype)
    t = _CONSTS.get(key)
    if t is None:
        t = _CONSTS[key] = torch.tensor(values, device=device, dtype=dtype)
    return t
