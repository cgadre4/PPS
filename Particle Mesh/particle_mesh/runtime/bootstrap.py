"""
Process bootstrap: the torch allocator environment and Taichi initialisation.

This is the only module that calls ti.init / ti.reset. init_taichi() is idempotent and must run before a
SimulationEngine (kernels.make_backend reads the active arch) or any renderer (they create Taichi fields) is built.
"""
import os

import taichi as ti
import torch

from particle_mesh.settings import CONFIG


def configure_torch_env():
    """Expandable segments reduce VRAM fragmentation on CUDA. Only set it for CUDA builds: newer PyTorch applies
    the same setting to Intel XPU, where it measured ~15-20% slower. The allocator reads it at its first allocation,
    so calling this after `import torch` but before any tensor still takes effect."""
    if torch.version.cuda:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def is_initialized():
    return bool(ti.lang.impl.get_runtime().prog)


def init_taichi(preferred=None):
    """Initialise Taichi on the best available arch (CUDA when torch sees CUDA, else Vulkan -> OpenGL -> CUDA -> CPU;
    simulation.TAICHI_BACKEND overrides). Returns the active arch; a no-op if Taichi is already initialised."""
    if is_initialized():
        return ti.lang.impl.current_cfg().arch
    if preferred is None:
        preferred = CONFIG.get("simulation", {}).get("TAICHI_BACKEND", None)
    arch_map = {
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "opengl": ti.opengl,
        "cpu": ti.cpu
    }
    candidates = []
    if preferred and str(preferred).lower() in arch_map:
        candidates.append(arch_map[str(preferred).lower()])

    # Priority hierarchy:
    # If CUDA is detected, try CUDA first.
    # Otherwise (Intel Arc / Panther Lake B390 / AMD), try Vulkan GPU first!
    if torch.cuda.is_available():
        gpu_order = [ti.cuda, ti.vulkan, ti.opengl, ti.cpu]
    else:
        gpu_order = [ti.vulkan, ti.opengl, ti.cuda, ti.cpu]

    for arch in gpu_order:
        if arch not in candidates:
            candidates.append(arch)

    for arch in candidates:
        try:
            if arch == ti.cuda:
                # Taichi only holds small buffers here (particle state lives in torch), so a fixed 0.5 GB pool
                # instead of 15% of VRAM gives torch 1-3 GB back on CUDA cards (AUDIT §8 F41)
                ti.init(arch=arch, device_memory_GB=0.5)
            else:
                ti.init(arch=arch)

            actual_arch = ti.lang.impl.current_cfg().arch
            # Taichi internally falls back to CPU (x64) without raising an exception if CUDA driver is missing
            if arch != ti.cpu and actual_arch == ti.cpu:
                print(f"[Taichi] Requested {arch} but driver fell back to CPU. Trying next backend...")
                ti.reset()
                continue

            print(f"[Taichi] Successfully initialized GPU backend: {actual_arch}")
            return actual_arch
        except Exception as e:
            print(f"[Taichi] Failed to initialize {arch}: {e}")
            continue

    raise RuntimeError("[Taichi] Failed to initialize any backend")


def print_status(engine):
    """Startup banner: Taichi arch, torch device/version and the particle-kernel backend in use."""
    print("=" * 65)
    print("                    HARDWARE & COMPUTE STATUS                     ")
    print("=" * 65)
    print(f"  Taichi Backend:   {ti.lang.impl.current_cfg().arch} (GPU Accelerated)")
    print(f"  PyTorch Device:   {engine.device}")
    print(f"  PyTorch Version:  {torch.__version__}")
    xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"  Intel XPU Active: {xpu_ok}")
    print(f"  Particle Kernels: {engine.kernels.name}")
    if not xpu_ok and engine.device.type == "cpu":
        print("  [NOTE] PyTorch is performing FFTs on CPU cores.")
        print(f"         Grid size: {CONFIG['simulation'].get('PHYSICS_GRID_W', 7680)}x{CONFIG['simulation'].get('PHYSICS_GRID_H', 7680)} cells.")
        print("         Consider reducing PHYSICS_GRID_W/H to 1024 or 2048 if CPU usage is too high.")
    print("=" * 65)
