"""Every user action and the extremes of every slider, through the real App.frame, in both solvers: no crash, no
broken invariant, full panel coverage. Each scenario runs tests/headless_actions.py in a subprocess (GGUI allows
one window per process). The script also runs in a visible window: python tests/headless_actions.py pm --visible"""
import json
import os
import subprocess
import sys

import pytest

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headless_actions.py")
# Opens a file dialog, so the script cannot press it (the image-spawn path behind it is covered by the folder scan)
NOT_SCRIPTABLE = {"Browse Image (Explorer)..."}

pytestmark = pytest.mark.actions


@pytest.mark.parametrize("solver", ["pm", "nbody"])
def test_every_action_and_extreme(solver):
    proc = subprocess.run([sys.executable, HARNESS, solver, "--quick", "--particles", "200000"],
                          capture_output=True, text=True, timeout=1800)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    r = json.loads(lines[-1])
    assert r["ok"], "\n".join(r["failures"])
    assert proc.returncode == 0
    assert set(r["buttons_never_pressed"]) <= NOT_SCRIPTABLE, r["buttons_never_pressed"]
    assert r["sliders_never_set"] == []
    assert r["min_zoom"] > 0 and r["max_zoom"] >= 1000.0
