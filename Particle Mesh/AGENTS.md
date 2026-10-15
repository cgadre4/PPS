# AGENTS.md

Guidance for AI coding agents working in this repository.

## Rule: no actions without "execute"

You may read files and answer questions freely. Anything that changes state needs the user to say **"execute"** first. That includes:
- editing or creating files;
- running scripts, tests or the app;
- installing packages;
- any git or GitHub action (commit, push, branch, PR, merge).

"Execute" approves the most recently proposed plan. If the plan changes, propose it again and wait for another "execute".

## What this is

An interactive, GPU-accelerated 2D particle physics sandbox (gravity, Lennard-Jones, mutual springs, Coulomb). PyTorch holds particle state, does the FFT/field math, and builds every rendered frame on the simulation device. Taichi provides the GGUI window, the ImGui panel and the vertex/image buffers. On machines where Taichi can share memory with PyTorch, it also runs the particle kernels.

This folder lives inside the `PPS` repo (`../`), which also holds `Ep1/` (unrelated early tutorial work). The uv project (`pyproject.toml`, `uv.lock`, `.venv`) is at the repo root.

## Running

```bash
uv run python main.py
```

- Run from this folder. uv finds the project at the repo root. Python 3.13 (`../.python-version`); `torch` comes from the PyTorch XPU wheel index (`[tool.uv] extra-index-url` in `../pyproject.toml`).
- `config.json` is read once by `config.py`, relative to that file, so the working directory doesn't matter for config. It does matter for `gui.py`, which scans the current directory for spawnable images (at startup and on "Rescan Images").
- Device and backend selection at startup (a status banner prints the result):
  - torch device: CUDA if available, else Intel XPU (`engine._get_torch_device`), else CPU.
  - Taichi arch: CUDA first if torch sees CUDA, otherwise Vulkan → OpenGL → CUDA → CPU (`main._init_taichi`). Override with `simulation.TAICHI_BACKEND`.
  - Particle kernels (`kernels.make_backend`):
    - `taichi` when Taichi shares memory with the torch device (torch CUDA + Taichi CUDA, or torch CPU + Taichi CPU);
    - `torch` otherwise. The development laptop (Intel Arc B390) uses torch XPU + Taichi Vulkan, so it gets `torch`.
    - Override with `simulation.KERNEL_BACKEND`.
- **Tests:** `uv run pytest` from this folder (~40 s, all headless). Run the suite before and after any change.
  - `test_kernels.py`: torch vs Taichi backend parity.
  - `test_physics.py`: audit-bug regressions and periodic behaviour.
  - `test_rendering.py`: frame contents and view alignment.
  - `test_smoke.py`: full `main.main()` loop per mode, each in a subprocess via `tests/headless_main.py`.
- **Benchmark:** `uv run python tools/bench_frame.py [1e6 4e6 ...]` reports step and render milliseconds with the app's backend selection. Compare before and after performance changes.
- **Headless technique:** wrap `ti.ui.Window` with `show_window=False`, stub the input methods, and read `get_image_buffer_as_numpy()`. Don't call `window.show()` without a visible window; it crashes. GGUI supports one window per process.
- There is no build step or linter.
- Don't set `PYTORCH_CUDA_ALLOC_CONF`/expandable segments for non-CUDA builds. PyTorch applies it to XPU too, and it measured ~15-20% slower (`main.py` only sets it when `torch.version.cuda`).
- In `nbody` mode, `simulation.NBODY_MAX_PARTICLES` (default 10,000) caps both the startup count and brush/image spawning (`SimulationEngine.capacity()`).
  - The solver is tiled (`TILE_ELEMENTS` per tile), so memory stays ~0.13 GiB at any N.
  - The limit is O(N²) speed: on the B390, 10k ≈ 146 ms/step and 20k ≈ 630 ms. Raise the cap on bigger GPUs.
- `config.py` is the only module that reads `config.json`; import `CONFIG` from it.
- Mouse-wheel input doesn't exist in Taichi 1.7 GGUI. Zoom and brush size use Z+drag and the arrow keys.

### Intel XPU rule: never pass XPU tensors to Taichi kernels

Taichi 1.7.4 has no XPU backend. On Vulkan it copies every non-CPU tensor argument to the host before a kernel and back again after (`taichi/lang/kernel_impl.py`). Measured on the B390, 1M-particle physics took 115 ms per step that way, against 11.5 ms with the torch backend.
- Particle hot paths go through `engine.kernels`.
- Data destined for a Taichi field goes through `kernels.to_taichi(tensor)` then `field.from_torch(...)`. That makes one explicit host copy only when memory isn't shared.

### Measured performance (Arc B390, torch backend, 2048² grid, 2880×1800 window, `RENDER_CELL_SIZE` 2)

| Particles | Step | Render + present | Total |
|---|---|---|---|
| 1M | 10.5–12 ms | 4.7–7.5 ms | ~15–19 ms (~50–65 FPS) |
| 2M | 17.4–22 ms | 6.2–9.6 ms | ~24–31 ms (~32–42 FPS) |
| 4M | 32–38 ms | 9–12 ms | ~42–50 ms (~20–24 FPS) |

The ranges are thermal and power state. The same commit measured ~15 ms at 1M on a cool machine and ~19 ms after sustained load. Always A/B performance changes back to back (for example against a `git worktree` of `master`), never against numbers in this file.

Where the time goes at 1M:
- **Step:** FFT and forces ~6.6 ms; building the CIC stencil ~3.4 ms; gather ~2.3 ms; deposit ~1 ms.
- **Render:** Taichi's `set_image` RGBA8 copy ~2.8 ms; colormap; host upload.

The next speed-up is fused Triton kernels / `torch.compile` (`AUDIT.md` §5.4 option B). That needs MSVC Build Tools; then set `TORCH_COMPILE: true`. Without a compiler it prints one notice and stays eager.

## Configuration (`config.json`)

- `display`: window size, FPS cap, arrow-overlay styling, `ui_width` (panel width as a fraction of the window).
- `simulation`:
  - `SOLVER` (`"pm"` or `"nbody"`; chosen at startup only, there is no runtime switch).
  - `TAICHI_BACKEND`, `KERNEL_BACKEND` (`"auto"`/`"torch"`/`"taichi"`), `TORCH_COMPILE`.
  - `SORT_INTERVAL`: re-sort particles by grid cell every N PM steps; 0 disables.
  - `NBODY_MAX_PARTICLES`, grid size, `PHYSICS_CELL_SIZE`, `RENDER_CELL_SIZE`, particle limits, log-density colour range.
  - The checked-in values are sized for the B390: 2048² grid, 4M max particles, 1M at startup. The 2048² float32 grid is 16 MB, which fits the 16 MB last-level cache; 4096² makes CIC ~9× slower per particle.
- `sliders`: every entry is `[min, max, default]`. `engine.py` reads index `[2]` for defaults; `gui.py` reads `[0]`/`[1]` for slider bounds. A new tunable parameter should follow this convention.

## Architecture

Frame loop (`main.py`): input handling -> brush spawn/erase -> `engine.step(dt)` -> render (PM density mesh or N-body circles, plus optional overlays) -> ImGui panel -> `window.show()` -> FPS throttle. `dt = (1/FPS_TARGET) * 10**time_scale_log`.

- **`engine.py` `SimulationEngine`** is the single source of state:
  - Preallocated `pos`/`vel` (`MAX_PARTICLES x 2`, float32) and `masses` (int8), with `num_active` marking the live prefix. Every per-particle op must slice `[:num_active]`.
  - Particles are always kept compacted at the front. Spawning appends; brush deletion re-packs survivors via boolean masks.
  - The GUI and `main.py` mutate engine attributes directly; there is no setter layer.
- **Boundaries.**
  - `periodic` (default): the universe is a torus, matching the FFT solver. Positions are wrapped right after each drift (inside the gather kernel for PM, `kernels.wrap` for N-body), so particles are never deleted at the edges. CIC deposit/gather wrap their neighbour indices, and N-body uses minimum-image displacements.
  - `reflective_walls`: clamps positions and bounces velocities in `engine.step`; CIC clamps instead of wrapping.
- **Particle kernels (`kernels.py`)** have one interface with two implementations:
  - `kernels_torch.py`: pure PyTorch, any device. It builds a `CICStencil` (corner indices + weights) once per step and shares it between deposit and gather; that sharing is the biggest eager-mode saving.
  - `kernels_taichi.py`: Taichi kernels on shared memory.
  - Methods: `prepare`, `deposit`, `gather_integrate`, `wrap`, `splat_view`.
  - `test_parity`-style checks (same inputs through both backends on Taichi's CPU arch) are how to validate changes to either.
- **Sorting:** `SimulationEngine.sort_particles()` reorders particles by physics cell every `SORT_INTERVAL` PM steps. Memory-coherent scatter/gather is 2-4× faster.
- **`masses` doubles as charge.**
  - It is 1 for every particle except in Coulomb mode. There the GUI randomises it to +/-1, and new spawns get random +/-1 (`SimulationEngine._new_masses`). Switching potential resets it.
  - In Coulomb mode the PM gather scales each particle's force by its own charge (`use_charge`).
- **Friction** is applied as `exp(-friction * dt)`, computed on the host, so it can never flip a velocity's sign.
- **PM solver (`pm_solver.py`)**, per step:
  1. `kernels.prepare` → `kernels.deposit` onto `density_grid`.
  2. `rfft2` (kept as `rho_k`, which the FFT minimap and bandpass view reuse with no extra FFT).
  3. Multiply by the cached `K_eff = K_k * Gaussian softening`, then `irfft2` to `phi`.
  4. Central-difference `fx`/`fy`: `torch.roll` when periodic, `torch.gradient` otherwise.
  5. Optional pressure forces.
  6. `kernels.gather_integrate`: kick, drift, wrap.
  
  `phi`, `fx`, `fy` and `density_grid` are kept on the solver and read by the renderer.
- **Units and force laws (shared by both solvers; `tests/test_physics_units.py` checks PM vs N-body parity):**
  - Gravity and Coulomb follow the 2D law `|a| = G·m/r`. PM solves `lap(phi) = 2π·G·ρ_area`; the deposited mass per cell is divided by `cell_size²` inside the kernel (`area_scale` in `potentials.py`). N-body uses `G·m·dx/(r² + ε²)`.
  - Harmonic springs use mean-field scaling (force per unit total mass, i.e. ÷N) in both solvers.
  - Lennard-Jones and harmonic PM kernels are FFTs of a sampled `V(r)` and need no area factor.
  - Pressure is `a = -∇P/ρ`, `P = K·ρ^γ`, on a Gaussian-smoothed number density per unit area with a density floor. In Coulomb mode, pressure uses a separate `|mass|` deposit and is gathered without the charge factor (`accel_x/accel_y`).
  - Spawned gravity disks get their circular speed from the field they actually feel: `v = vel_mult · sqrt(r · a_inward)`.
    - The acceleration comes from `pm_solver.accelerations_at` (a field-only solve, bilinear-sampled, which equals CIC) or from the N-body solver, so it includes the disk itself, existing particles, softening, screening and periodic images.
    - `vel_mult` is the "Spawn Velocity (x orbit)" slider (1 = circular). Disks hold their radius to within 0.5% over 2 s.
    - Other potentials get a thermal spread of `THERMAL_SPEED·vel_mult`.
  - Default G is 0.64, retuned so the default PM scene keeps its pre-fix strength at `PHYSICS_CELL_SIZE: 2`.
- **Kernel caching:**
  - `potentials.py` `PotentialsManager.get_fourier_kernel` builds `K_k`: gravity/Coulomb analytically as screened Yukawa `-+G/(k^2 + 1/range^2)`, harmonic/LJ by FFT of a real-space periodic potential on the full grid.
  - `K_k[0, 0]` is always zeroed, so phi has zero mean. Don't remove that: the mean term destroys float32 precision in phi.
  - The PM solver only rebuilds `K_k` when the potential *type* changes. Any change to a kernel parameter (G, force range, spring_k, epsilon, sigma) must trigger a rebuild: `engine.request_kernel_rebuild()` (debounced; what the GUI sliders use) or `engine.precompute_kernel()` (immediate).
  - `K_eff` is rebuilt automatically when `K_k` or the softening length changes.
  - Kernels and `k_sq` are stored in **rfft2 half-spectrum layout** `(H, W//2+1)`. Gravity/Coulomb `K_k` is real; harmonic/LJ is the complex `rfft2` of `V(r)`. `pm_solver.k_sq` is the same tensor as `pot_manager.k_sq`. The real-space distance grid is built on demand only.
  - GUI sliders call `engine.request_kernel_rebuild()`, which is debounced: the rebuild runs in `step()` once the value has been still for `KERNEL_REBUILD_DELAY` (0.15 s). Call `precompute_kernel()` directly when an immediate rebuild is needed (e.g. switching potential).
- **N-body solver (`nbody_solver.py`)** computes exact pairwise accelerations in PyTorch (any device), processing targets in row tiles of `TILE_ELEMENTS` against all sources, so memory is O(N·tile). Integration happens in `engine.step`.
- **Adding or changing a potential touches four places:** `potentials.get_fourier_kernel` (PM), `nbody_solver.solve` (N-body), `engine.precompute_kernel`/`step` param dicts, and the buttons/sliders in `gui.py`.

### Taichi initialisation order

- No module allocates Taichi fields at import, so modules can be imported in any order, even before `ti.init()`. `tests/test_imports.py` enforces this.
- Fields are created by their owners after initialisation: `MeshRenderer.__init__` (`display_pixels`, arrow buffers), `CircleRenderer.__init__`, and the brush-ring/arrow fields inside `main.main()`, which are passed to template kernels.
- `ti.init()` must still run before you construct a `SimulationEngine` (`kernels.make_backend` reads the active arch) or any renderer. `main.py` runs `_init_taichi()` at import, and the tests and tools rely on `import main` doing that.
- Keep new Taichi fields out of module scope.

### Array layout conventions

- PyTorch grids (`density_grid`, `phi`, `fx`, `fy`, `k_sq`, viewport density) are `[row=y, col=x]`; flat index `j * W + i`.
- Taichi fields (`MeshRenderer.display_pixels`) are `[x, y]`; `MeshRenderer._present` transposes the `(VIEW_H, VIEW_W, 3)` image before `from_torch`. Row 0 is the bottom of the screen.
- World coordinates span `[0, PHYSICS_GRID_W * PHYSICS_CELL_SIZE)`. `camera.py` maps screen pixels to world via `cam_x/cam_y` (world-space bottom-left of the view) and `zoom`. Advanced PM views resample grids into the viewport with `grid_sample`, using normalised `[-1, 1]` coordinates from `mesh_renderer.build_view_grid()`.
- The viewport is measured in **render cells** (`RENDER_CELL_SIZE` screen pixels each; `VIEW_W = WIDTH / RENDER_CELL_SIZE`), not pixels.
  - Any code that maps a render-cell index to world coordinates must multiply by `RENDER_CELL_SIZE` before dividing by `zoom`.
  - Anything drawn into the frame is sized in render cells.
  - The shipped config uses `RENDER_CELL_SIZE: 2`, so this bug class shows up immediately.

### Rendering

All image math runs in PyTorch on the simulation device; Taichi only receives finished buffers.
- **PM frame entry point:** `MeshRenderer.render_pm(canvas, engine, camera, color_map, ui_w)` draws the density, potential or bandpass view, the FFT minimap (`fft_thumbnail`, mask from the cached `bandpass_mask`) and the force arrows. `build_view_grid` lives in `mesh_renderer.py`, and `main.py` re-exports it.
- **PM mode:** `MeshRenderer.render` calls `kernels.splat_view` (counts per render cell), then `colorize` (log10 through a cached 257-entry uint8 LUT per theme; empty cells black), then `_present`.
  - `_present` pastes the overlay, then does one upload into `display_pixels`, then `canvas.set_image`.
  - The potential, bandpass and vector-field views are fed by grid-sampled solver arrays instead.
- **Overlays** must be pasted before `canvas.set_image()`, because it snapshots the field immediately. The FFT minimap is passed as `overlay=`.
- **Vector arrows** are built as vertex/colour tensors in `MeshRenderer.render_vectors` and uploaded into `vector_verts`/`vector_colors`.
- **N-body mode:** `CircleRenderer` normalises on-screen particles on the device and uploads them into buffers sized `min(290000, NBODY_MAX_PARTICLES)`.
- Colour themes (`colors.py`) must have exactly **9** RGB entries in 0-255; the gradient interpolates between indices 0-8.

## Known issues

`AUDIT.md` lists verified bugs, performance problems and the XPU plan, with a Status column and a suggested fix order (§6). Check it before changing physics or rendering code.

## Legacy code

`mono_code/` holds earlier single-file versions (`PM_gravity_pytorch*.py`, `PM_LJ.py`, `PPS_pytorch_final.py`, etc.) that predate the modular refactor. They are not imported by the current app. Treat them as reference only and don't edit them unless asked.
