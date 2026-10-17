# PPS

Personal particle physics simulation projects.

## Projects

- `Ep1/` — early particle simulation / YouTube tutorial work
- `Particle Mesh/` — particle-mesh gravity simulation

## Setup

Run once per machine from the repo root. It detects the GPU (NVIDIA → CUDA build, Intel Arc → XPU build, otherwise
CPU) and installs the matching PyTorch build with uv:

```bash
uv run --no-project setup_env.py
```

Then run the sandbox from `Particle Mesh/` with `uv run python main.py`.