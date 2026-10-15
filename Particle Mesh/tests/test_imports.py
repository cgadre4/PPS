"""No module allocates Taichi fields at import time (AUDIT §7 item 5): everything must import before ti.init().
Runs in a subprocess because this test session has already initialised Taichi via conftest."""
import os
import subprocess
import sys

PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r"""
import sys, os
sys.path.insert(0, {pm!r}); os.chdir({pm!r})
import config, colors, camera, potentials, kernels_torch, kernels_taichi, kernels, nbody_solver, pm_solver
import engine, mesh_renderer, circle_renderer, gui, image_loader
import taichi as ti
assert not ti.lang.impl.get_runtime().prog, "a module initialised Taichi at import"
print("IMPORT_OK")
"""


def test_modules_import_before_taichi_init():
    proc = subprocess.run([sys.executable, "-c", SCRIPT.format(pm=PM_DIR)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "IMPORT_OK" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-4000:]
