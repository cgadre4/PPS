"""
One-time environment setup: detect this machine's GPU and install the matching torch build.

    uv run --no-project setup_env.py              # detect and sync
    uv run --no-project setup_env.py --dry-run    # only report what it would install
    uv run --no-project setup_env.py --backend cpu

torch ships one build per GPU vendor, exposed as the `cuda`, `xpu` and `cpu` extras in pyproject.toml. This picks one
(NVIDIA -> cuda, Intel Arc / Core Ultra graphics -> xpu, anything else -> cpu) and runs `uv sync --extra <name>`.
Afterwards plain `uv run python main.py` keeps that build. Standard library only, so it runs before the venv exists.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKENDS = ("cuda", "xpu", "cpu")
# The `cuda` extra uses the cu130 wheels, which need a driver that reports CUDA 13.0 or newer
MIN_CUDA = (13, 0)


def _run(cmd):
    """stdout of `cmd`, or "" when it is missing or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def nvidia_gpu():
    """(GPU names, driver CUDA version tuple or None), or None without an NVIDIA driver."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    names = [n.strip() for n in _run([smi, "--query-gpu=name", "--format=csv,noheader"]).splitlines() if n.strip()]
    if not names:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", _run([smi]))
    return names, (int(m.group(1)), int(m.group(2))) if m else None


def gpu_names():
    """Display adapter names reported by the OS (best effort)."""
    system = platform.system()
    if system == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        return [n.strip() for n in out.splitlines() if n.strip()]
    if system == "Linux":
        out = _run(["lspci"])
        return [line for line in out.splitlines() if re.search(r"VGA|Display|3D", line)]
    return []


def is_xpu_capable(name):
    """PyTorch XPU supports Intel Arc discrete GPUs and the Arc graphics in Core Ultra CPUs, not older Iris Xe/UHD."""
    return "intel" in name.lower() and "arc" in name.lower()


def detect():
    """(backend, reason) for this machine."""
    nv = nvidia_gpu()
    if nv:
        names, cuda = nv
        if cuda is None or cuda >= MIN_CUDA:
            return "cuda", f"NVIDIA GPU: {', '.join(names)}" + (f" (driver supports CUDA {cuda[0]}.{cuda[1]})" if cuda else "")
        sys.exit(f"NVIDIA GPU found ({', '.join(names)}), but its driver only supports CUDA {cuda[0]}.{cuda[1]}. "
                 f"The cuda build needs CUDA {MIN_CUDA[0]}.{MIN_CUDA[1]}+: update the NVIDIA driver, "
                 f"or pass --backend cpu.")
    names = gpu_names()
    intel = [n for n in names if is_xpu_capable(n)]
    if intel:
        return "xpu", f"Intel GPU: {', '.join(intel)}"
    seen = f" (found: {', '.join(names)})" if names else ""
    return "cpu", f"no NVIDIA or Intel Arc GPU detected{seen}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--backend", choices=BACKENDS, help="skip detection and install this torch build")
    parser.add_argument("--dry-run", action="store_true", help="print the choice and the command without running it")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)     # keep our lines in order with uv's output

    if args.backend:
        backend, reason = args.backend, "chosen with --backend"
    else:
        backend, reason = detect()
    print(f"[setup] torch build: {backend}  ({reason})")

    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        sys.exit("uv not found on PATH: install it from https://docs.astral.sh/uv/")
    cmd = [uv, "sync", "--extra", backend]
    print(f"[setup] {' '.join(cmd[1:])}  (in {ROOT})")
    if args.dry_run:
        return

    if subprocess.run(cmd, cwd=ROOT).returncode:
        sys.exit("[setup] uv sync failed")

    check = ("import torch; "
             "dev = 'cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'; "
             "name = torch.cuda.get_device_name() if dev == 'cuda' else torch.xpu.get_device_name() if dev == 'xpu' else ''; "
             "print(f'[setup] torch {torch.__version__} -> device {dev} {name}'.rstrip())")
    subprocess.run([uv, "run", "--no-sync", "python", "-c", check], cwd=ROOT)
    print("[setup] Done. Run the app with:  cd \"Particle Mesh\"  then  uv run python main.py")


if __name__ == "__main__":
    main()
