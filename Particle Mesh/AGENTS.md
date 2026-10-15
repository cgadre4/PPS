# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

An interactive, GPU-accelerated 2D particle physics sandbox (gravity, Lennard-Jones, mutual springs, Coulomb). PyTorch holds particle state and does the FFT/field math; Taichi provides custom kernels (mass scatter, force gather/integrate, rendering) and the GGUI window + ImGui panel. On CUDA both share the same tensors with no copies (Taichi kernels take PyTorch tensors as `ti.types.ndarray()`).

This folder lives inside the `PPS` repo (`../`), which also holds `Ep1/` (unrelated early tutorial work). The uv project (`pyproject.toml`, `uv.lock`, `.venv`) is at the repo root.

## Running

```bash
uv run python main.py
```

- Run from this folder. uv finds the project at the repo root. Python 3.13 (`../.python-version`); `torch` comes from the PyTorch XPU wheel index (`[tool.uv] extra-index-url` in `../pyproject.toml`).
- `config.json` is resolved relative to each module, so the working directory doesn't matter for config. It does matter for `gui.py`, which globs the current directory for spawnable images.
- Backend selection at startup:
  - torch device: CUDA if available, else Intel XPU (`engine._get_torch_device`), else CPU.
  - Taichi arch: CUDA first if torch sees CUDA, otherwise Vulkan → OpenGL → CUDA → CPU (`main._init_taichi`). Override with `simulation.TAICHI_BACKEND` in `config.json`.
  - A hardware/compute status banner is printed at startup.
- There is no build step, test suite or linter. Verify changes by running the app.
- In `nbody` mode `main.py` caps startup particles at 10,000 because the solver is O(N^2).

### Known XPU limitation

On Intel GPUs torch runs on `xpu` while Taichi runs on Vulkan. Taichi 1.7.4 cannot read XPU tensors, so every Taichi kernel call copies each tensor argument to the CPU and back (`taichi/lang/kernel_impl.py`). That makes the particle kernels very slow on XPU. See `AUDIT.md` §5 for measurements and the recommended fix: pure torch/Triton on XPU, with Taichi used only for the window. Don't add new Taichi kernels that receive XPU tensors.

## Configuration (`config.json`)

- `display`: window size, FPS cap, arrow-overlay styling, `ui_width` (panel width as a fraction of the window).
- `simulation`: `SOLVER` (`"pm"` or `"nbody"`; chosen at startup only, there is no runtime switch), optional `TAICHI_BACKEND`, grid size, `PHYSICS_CELL_SIZE`, `RENDER_CELL_SIZE`, particle limits, log-density colour range. The checked-in values are sized for the laptop's Arc iGPU (1440² grid, 1M max particles, `RENDER_CELL_SIZE` 2).
- `sliders`: every entry is `[min, max, default]`. `engine.py` reads index `[2]` for defaults; `gui.py` reads `[0]`/`[1]` for slider bounds. A new tunable parameter should follow this convention.

## Architecture

Frame loop (`main.py`): input handling -> brush spawn/erase -> `engine.step(dt)` -> render (PM density mesh or N-body circles, plus optional overlays) -> ImGui panel -> `window.show()` -> FPS throttle. `dt = (1/FPS_TARGET) * 10**time_scale_log`.

- **`engine.py` `SimulationEngine`** is the single source of state: preallocated `pos`/`vel` (`MAX_PARTICLES x 2`, float32) and `masses` (int8), with `num_active` marking the live prefix. Particles are always kept compacted at the front; spawn appends, and deletion (brush or leaving the universe in `wrap_delete` mode) re-packs survivors via boolean masks. Every per-particle op must slice `[:num_active]`. The GUI and `main.py` mutate engine attributes directly; there is no setter layer.
- **`masses` doubles as charge.** It is 1 for every particle except in Coulomb mode, where the GUI randomises it to +/-1. Switching potential resets it.
- **PM solver (`pm_solver.py`)**, per step: CIC-scatter mass onto `density_grid` (Taichi atomics) -> Poisson solve -> `fx`/`fy` -> optional pressure forces -> Taichi kernel gathers forces back with CIC weights and does a semi-implicit Euler update with friction in place.
  - If the torch device is CUDA or XPU, the Poisson solve uses torch FFT: `rfft2` -> multiply by the cached Fourier kernel `K_k` and a Gaussian softening filter -> `irfft2` -> `torch.gradient`.
  - If the torch device is CPU, it falls back to a Taichi red-black Gauss-Seidel solve (`poisson_rbgs_kernel`, 12 sweeps, gravity only, Dirichlet edges). That is far from converged; see `AUDIT.md` §5.2.
  - `phi`, `fx`, `fy` and `density_grid` are kept on the solver and read by the renderer for the potential/field/bandpass/FFT views.
- **Kernel caching:** `potentials.py` `PotentialsManager.get_fourier_kernel` builds `K_k` (gravity/Coulomb analytically as screened Yukawa `-+G/(k^2 + 1/range^2)`; harmonic/LJ by FFT of a real-space periodic potential on the full grid). The PM solver only rebuilds it when the potential *type* changes. Any change to a kernel parameter (G, force range, spring_k, epsilon, sigma) must call `engine.precompute_kernel()`, as the GUI does for its sliders. Softening is applied per step, not baked into `K_k`.
- **N-body solver (`nbody_solver.py`)** builds full NxN pairwise matrices in PyTorch and returns accelerations; integration happens in `engine.step`. On a CPU torch device with gravity, it uses a Taichi kernel instead (`nbody_taichi_solve_kernel`; note that kernel uses a 3D `1/r^2` force while the torch path uses 2D `1/r`).
- **Adding or changing a potential touches four places:** `potentials.get_fourier_kernel` (PM), `nbody_solver.solve` (N-body), `engine.precompute_kernel`/`step` param dicts, and the buttons/sliders in `gui.py`.

### Taichi import-order constraint

`mesh_renderer.py`, `circle_renderer.py` and `main.py` allocate Taichi fields at module import time, sized from `config.json`. `ti.init()` (via `_init_taichi()`) must run before these modules are imported; this is why `main.py` imports them below it. Keep that ordering when adding modules that declare fields.

### Array layout conventions

- PyTorch grids (`density_grid`, `phi`, `fx`, `fy`, `k_sq`) are `[row=y, col=x]`, indexed `[j, i]` in Taichi kernels.
- Taichi display fields (`display_pixels`, `visual_density`) are `[x, y]`; `MeshRenderer.render_grid` transposes before `from_torch`.
- World coordinates span `[0, PHYSICS_GRID_W * PHYSICS_CELL_SIZE)`. `camera.py` maps screen pixels to world via `cam_x/cam_y` (world-space top-left) and `zoom`. Advanced PM views resample grids into the viewport with `grid_sample`, using normalised `[-1, 1]` coordinates computed in `main.py`.

### Rendering

- PM mode: `MeshRenderer` scatters visible particles into a viewport-sized density field and maps `log10` density through a theme colourmap. The potential, bandpass and vector-field views are fed by grid-sampled solver arrays instead.
- N-body mode: `CircleRenderer` copies up to `MAX_RENDER_PARTICLES` (290,000) on-screen particles into Taichi fields for `canvas.circles`.
- Colour themes (`colors.py`) must have exactly **9** RGB entries in 0-255; kernels interpolate between indices 0-8.

## Known issues

`AUDIT.md` lists verified bugs, performance problems and the XPU plan, in a suggested fix order (§6). Check it before changing physics or rendering code.

## Legacy code

`mono_code/` holds earlier single-file versions (`PM_gravity_pytorch*.py`, `PM_LJ.py`, `PPS_pytorch_final.py`, etc.) that predate the modular refactor. They are not imported by the current app. Treat them as reference only and don't edit them unless asked.
