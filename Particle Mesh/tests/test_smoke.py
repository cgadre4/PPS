"""Full app loop smoke tests: main.main() runs offscreen for a few frames in each mode (one subprocess each,
because GGUI supports a single window per process)."""
import json
import os
import subprocess
import sys

import pytest

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headless_main.py")


def _run(mode):
    proc = subprocess.run([sys.executable, HARNESS, mode], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    result = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    assert result, proc.stdout[-2000:]
    return json.loads(result[-1])


def test_pm_loop_with_fft_minimap():
    r = _run("pm")
    assert r["solver"] == "pm" and r["num_active"] > 0
    assert r["minimap_grey_fraction"] > 0.3        # AUDIT 2.1: the minimap actually reaches the frame


def test_nbody_loop_spawn_is_capped():
    from particle_mesh.physics import engine as E
    r = _run("nbody")
    assert r["solver"] == "nbody"
    assert r["num_active"] <= E.NBODY_MAX_PARTICLES


@pytest.mark.parametrize("mode", ["potential", "bandpass"])
def test_advanced_views_with_vector_field(mode):
    r = _run(mode)
    assert r["show_field"]
    assert r["show_potential"] == (mode == "potential")
    assert r["bandpass"] == (mode == "bandpass")
    assert r["lit_fraction"] > 0.01
