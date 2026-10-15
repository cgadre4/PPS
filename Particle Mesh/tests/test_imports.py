"""Importing never initialises Taichi or allocates Taichi fields (AUDIT §7 item 5): every package module and main.py
import cleanly before ti.init(); bootstrap.init_taichi() is the only initialisation point. Runs in a subprocess
because this test session has already initialised Taichi via conftest."""
import os
import subprocess
import sys

PM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r"""
import importlib, pkgutil, sys, os
sys.path.insert(0, {pm!r}); os.chdir({pm!r})
import particle_mesh
names = [m.name for m in pkgutil.walk_packages(particle_mesh.__path__, "particle_mesh.")
         if m.name != "particle_mesh.__main__"]
for name in names:
    importlib.import_module(name)
import main
import taichi as ti
assert not ti.lang.impl.get_runtime().prog, "a module initialised Taichi at import"
print("IMPORT_OK", len(names))
"""


def test_modules_import_before_taichi_init():
    proc = subprocess.run([sys.executable, "-c", SCRIPT.format(pm=PM_DIR)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "IMPORT_OK" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-4000:]
    assert int(proc.stdout.split("IMPORT_OK")[1].split()[0]) >= 20      # the walk really covered the package


def test_no_flat_module_shims_left():
    flat = ["config", "colors", "camera", "grid_ops", "potentials", "pm_solver", "nbody_solver", "engine", "kernels",
            "kernels_torch", "kernels_taichi", "mesh_renderer", "circle_renderer", "gui", "image_loader"]
    assert [f for f in flat if os.path.exists(os.path.join(PM_DIR, f + ".py"))] == []
