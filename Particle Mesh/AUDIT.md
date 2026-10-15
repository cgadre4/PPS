# Particle Mesh — Code Audit

Audit date: 2026-10-15. Scope: every module (`main.py`, `engine.py`, `pm_solver.py`, `nbody_solver.py`, `potentials.py`, `mesh_renderer.py`, `circle_renderer.py`, `gui.py`, `camera.py`, `image_loader.py`, `colors.py`, `config.json`). I also benchmarked XPU options on the laptop (Intel Core Ultra X7 358H, **Arc B390 iGPU**, 16.7 GB shared, torch 2.13.0+xpu, Taichi 1.7.4).

> **Which version this audits.** The line references in §1–§4 point to the June 2026 copy of the code, which was backed up to Google Drive from the CUDA machine. The version committed in this repo is the August 2026 copy. That copy changes `main.py`, `engine.py`, `pm_solver.py`, `nbody_solver.py`, `mesh_renderer.py` and `config.json` (XPU/Vulkan backend selection, absolute config paths, a smaller config), so line numbers in those files will be off. The findings still apply unless noted. §5.2 reviews the August changes specifically.

Severity: **High** = wrong results, crash, or a feature that silently does nothing. **Medium** = wrong in some settings, or a large waste of performance. **Low** = minor, cosmetic, or code quality.

"Verified" means I reproduced the behaviour or confirmed it in library source. Everything else comes from reading the code.

---

## 1. Top findings

| # | Sev | Finding | Where | Status |
|---|-----|---------|-------|--------|
| 1 | High | The FFT minimap is never displayed, but it is still computed every frame (on by default). **Verified.** | `main.py:365-394`, `mesh_renderer.py:175,185` | Done (PR #1) |
| 2 | High | Coulomb in PM mode ignores the particle's own charge, so + and − particles accelerate identically. | `pm_solver.py:47-51` | Done (PR #1) |
| 3 | High | Friction factor `(1 - friction*dt)` goes negative at high friction × time scale and blows up the velocities. | `pm_solver.py:50-51`, `engine.py:229-230` | Done (PR #1) |
| 4 | High | N-body view ignores zoom: the culling and normalisation use screen pixels as world units. | `main.py:270-278`, `circle_renderer.py:21-22,46-47` | Done (PR #1) |
| 5 | High | N-body mode has no particle cap after startup. One brush stroke spawns ~785k particles, which needs terabytes for the N×N matrices, so it OOMs or hangs. | `main.py:61-64,234-237`, `nbody_solver.py:24-39` | Done (PR #1) |
| 6 | High | Mouse-wheel zoom and brush resize are dead code: Taichi 1.7.4 GGUI emits no wheel events. **Verified in source.** | `main.py:166-172` | Done (PR #6): dead code and legend hints removed |
| 7 | Medium | The k=0 (mean) mode of the potential isn't zeroed. That adds a huge constant to φ, which wrecks float32 precision at large force range and washes out the potential view. | `potentials.py:39-47,79-88` | Done (PR #1) |
| 8 | Medium | Forces are periodic (FFT) but particles are deleted at the edges, not wrapped. Mass near one edge pulls particles at the opposite edge. The GUI calls this mode "UNIVERSE WRAPPED". | `engine.py:198-206`, `gui.py:113` | Done (PR #3) |
| 9 | Medium | Pressure is applied as acceleration `-P∇ρ^γ` rather than `-∇P/ρ`, so it explodes in dense regions. | `potentials.py:106-115` | Done (PR #5) |
| 10 | Medium | Potential, field and bandpass views are wrong whenever `RENDER_CELL_SIZE ≠ 1` (the local `config.json` uses 2). | `main.py:283-296,380-381` | Done (PR #1, #2) |
| 11 | Medium | Particle data is read 3–4 times per frame plus a GPU sync. At 100M particles that is about 6–7 GB of memory traffic per frame. | `engine.py:199-206`, `pm_solver.py`, `mesh_renderer.py:37-46` | Partial (PR #3): sync removed, cell sorting and a shared CIC stencil added; the single fused kernel is option B |
| 12 | Medium | About 1 GB of redundant precomputed grids, and the kernel is stored as full complex when only half is used. | `potentials.py:12-25`, `pm_solver.py:69-74` | Done (PR #3, #6) |
| 13 | High (XPU) | The August Arc port passes XPU tensors into Taichi Vulkan kernels. Taichi copies every argument to the CPU and back on every call. **Verified, measured.** | `pm_solver.py` (August copy) | Done (PR #3) |
| 14 | Medium (XPU) | The port's Vulkan Poisson fallback (12 red-black Gauss-Seidel sweeps) is far from converged, so its gravity is effectively ~12-cell range. | same | Done (PR #3, removed) |
| 15 | Low | Two diverged copies of the project, and neither is under working version control. | — | Done (repo `cgadre4/PPS`) |

---

## 2. Correctness bugs

### 2.1 FFT minimap never appears (High, verified)
`MeshRenderer.render()` / `render_grid()` call `canvas.set_image(display_pixels)` first. `main.py` then calls `overlay_fft()`, which writes into `display_pixels`. In Taichi 1.7.4, `Canvas.set_image` immediately copies the field into an RGBA8 staging buffer (`taichi/ui/canvas.py:48` → `staging_buffer.to_rgba8`). Anything written to the field afterwards is overwritten by the next frame's full repaint.

I reproduced this headlessly (`window.get_image_buffer_as_numpy()`). Written in `main.py`'s order, the overlay quarter stays at background value on every frame; written before `set_image`, it shows.

Since `show_fft` defaults to `True` (it isn't in `config.json`), every frame pays for an extra full-grid `rfft2`, `abs`, sigmoid bandpass, resize and `log1p` for nothing.

**Fix:** run the overlay before `set_image`. That means moving the PIP into `MeshRenderer` and drawing it between `render_color_kernel` and `set_image`. Also reuse the solver's `rho_k` (see 3.3) and refresh the thumbnail only every N frames.

Smaller issues in the same block:
- `interpolate(..., mode='bilinear')` without `antialias=True` aliases badly when shrinking 7680×3841 down to ~576×288. *Fixed in PR #7* (`antialias=True`, which XPU supports).
- The mirrored half is built with a horizontal flip only. For a real input, |F(−k)| = |F(k)| is a point reflection, so the left half should be flipped in both axes (after `fftshift`).

### 2.2 PM Coulomb ignores the sign of the charge (High)
`scatter_mass_kernel` deposits `mass[p]` (±1 in Coulomb mode), so φ is a charge potential. But `gather_and_integrate_kernel` applies `vel += F * dt` with no `* mass[p]`, so a negative charge is pushed the same way as a positive one. The N-body path gets this right: `nbody_solver.py:87-88` multiplies by `masses`.

**Fix:** pass `mass` into the gather kernel and scale `fx, fy` by `q_p` when the potential is Coulomb. A mass/charge-to-mass factor works for all potentials.

Related issues:
- `add_galaxy` (`engine.py:142`) and `load_particles_from_array` (`engine.py:177`) always set `masses = 1`, so particles spawned in Coulomb mode are all positive. Randomise ±1 when `potential_type == 'coulomb'`.

### 2.3 Friction can flip velocities and blow up (High)
`vel = (vel + a*dt) * (1 - friction*dt)`. The friction slider goes up to 5, and the time-scale slider up to 10² gives dt = 0.83, so the factor reaches −3.2. Velocities then flip sign and grow every step. The same formula appears in the N-body path.

**Fix:** `damp = exp(-friction*dt)`, computed once on the host and passed as a float.

### 2.4 N-body circles don't zoom (High)
`main.py:275-276` passes `camera.width, camera.height` (screen pixels). `CircleRenderer` uses them both as the world-space culling window (`circle_renderer.py:46-47`) and as the normalisation divisor (`:21-22`). Zooming therefore only shifts `cam_x/cam_y`. The view never scales, and culling is wrong at any zoom other than 1.

**Fix:** pass `camera.width / camera.zoom`, `camera.height / camera.zoom`.

### 2.5 N-body mode has no runtime particle cap (High)
The 10,000 cap only applies to `initial_particles` (`main.py:61-64`). The brush spawns `π·r²·density` particles per frame, which is 785k at the defaults. The N-body solver builds about ten dense N×N float32 tensors (`diff`, `dx`, `dy`, `r_sq`, `r`, `screening`, `factor`, …). That is already ~4 GB at N = 10k, and it grows with N².

**Fix:** enforce an N-body cap in `add_galaxy` / `load_particles_from_array`. Then tile the computation in blocks of rows (see 3.5) so memory is O(N·block).

### 2.6 Mouse wheel does nothing (High, verified in source)
`taichi/ui` contains no wheel/scroll handling in 1.7.4. GGUI events have no `delta` attribute (that belongs to the legacy `ti.GUI`), and `getattr(ti.ui, 'WHEEL', 'Wheel')` just returns the fallback string. Scroll-to-zoom and scroll-to-resize never fire. Only the Z+drag, arrow-key and slider fallbacks work. The hotkey legend in `gui.py:244-245` advertises scroll.

**Fix:** remove the branch and update the legend, or switch to a windowing layer that reports scroll (see §5, option B).

### 2.7 The potential's mean (k = 0) term isn't removed (Medium)
For gravity and Coulomb, `K_k[0,0] = ∓G/λ²` with λ = 1/force_range. After `irfft2`, φ gets a constant offset of `G·M_total/(λ²·H·W)`.

- With the shipped config (G = 1, 100M particles, 7680² grid, range 100), the offset is ≈1.7×10⁴. At the slider maximum (range 8192) it is ≈1.1×10⁸, where the float32 spacing is 8. `torch.gradient` then differences values that are quantised in steps of 8, which adds noise of up to ~4 units per cell to every force. That is significant for small clumps.
- The potential view plots `log10|φ|`, so the offset dominates the image and the view is close to uniform.
- `denom_safe[0,0] = 1e-8` only triggers when λ² < 1e-8. At range 8192, λ² ≈ 1.49e-8, so the guard doesn't fire.

**Fix:** set `K_k[0,0] = 0` (the standard zero-mean condition for periodic Poisson solves). For harmonic and LJ the k = 0 term is also a pure constant and can be zeroed too.

### 2.8 Periodic forces with non-periodic particles (Medium)
**Fixed** in the XPU port, which takes the truly periodic option. Positions wrap after each drift, CIC wraps its neighbour indices, N-body uses minimum-image displacements, and forces use periodic central differences. Particles are no longer deleted at the edges, which also removes the per-frame `.item()` sync. `reflective_walls` is unchanged.

The FFT treats the universe as a torus, but in `wrap_delete` mode particles that leave are deleted. Near an edge, a particle feels the periodic image of mass on the opposite side. The CIC scatter also clamps rather than wraps (`pm_solver.py:14-15,38-39`), which piles the outermost half-cell of mass onto the edge cells.

Pick one model:
- **Truly periodic:** wrap positions (`pos %= universe`) and wrap the CIC neighbour indices (`(i+1) % W`). There's no need to delete particles, which also removes the per-frame compaction and sync (3.2).
- **Isolated:** zero-pad the grid to 2W×2H for the FFT (Hockney–Eastwood). Each FFT is about 4× the size, but nothing wraps.

The current hybrid is neither. At minimum, rename the "UNIVERSE WRAPPED" button.

### 2.9 Pressure force has the wrong form (Medium)
**Fixed** (PR #5): `a = -∇P/ρ` on a Gaussian-smoothed number density per unit area, with a density floor. In Coulomb mode it uses a separate `|mass|` deposit and is gathered without the charge factor. Measured: a 100× denser clump of the same shape gets a 1.11× push (before, ~100×).

`compute_pressure_forces` returns `-P·∇(ρ^γ)` and adds it straight to the acceleration field. Physically, acceleration is `-∇P/ρ = -K·γ·ρ^(γ-2)·∇ρ`. For isothermal gas (γ = 1) that is `-K·∇ln ρ`. The current form scales with the local density, so dense clumps get enormous kicks while diffuse gas gets almost none.

The raw CIC density is also noisy (shot noise), and its gradient is noisier still.

**Fix:** divide by `max(ρ, ε)` and smooth the density first. A Gaussian filter in k-space is free here, because `rho_k` is already computed.

### 2.10 Advanced views break when `RENDER_CELL_SIZE ≠ 1` (Medium)
**Fixed** on branch `fix/render-cell-size-views`: `main.build_view_grid()` scales by `RENDER_CELL_SIZE` and samples at render-cell centres. The minimap sizing part was fixed in PR #1.

`main.py:291-292` computes `world = screen_x / zoom + cam_x` with `screen_x` in render cells, not pixels. With `RENDER_CELL_SIZE = 2` (the setting in `C:\Python Dev\PPS\Particle Mesh\config.json`), the potential, bandpass and field views show the wrong quarter of the screen at 2× scale.

The minimap size (`main.py:380-381`) is also in pixels but is written into a render-cell-sized buffer.

**Fix:** `world_gx = (screen_x + 0.5) * RENDER_CELL_SIZE / zoom + cam_x` (same for y), and divide the minimap size by `RENDER_CELL_SIZE`.

### 2.11 PM and N-body don't match for the same slider values (Low/Medium)
**Fixed** (PR #5):
- Both solvers use `|a| = G·m/r`. The PM kernel is `-2πG/(k²+λ²)/cell_size²`.
- Harmonic uses 1/N mean-field scaling in both solvers.
- The softening slider is relabelled "Force Softening Length".
- Measured PM vs N-body: gravity and Coulomb within 0.7%, harmonic within 0.2%.
- Default G is retuned from 1.0 to 0.64, keeping the default PM scene's strength.

- **Gravity/Coulomb:** PM solves ∇²φ = Gρ, giving `|g| = G·M/(2π r)`. N-body uses `G·m·dx/r²`, giving `|g| = G·M/r`. The same G is therefore 2π stronger in N-body.
- **Harmonic:** N-body divides by N (`nbody_solver.py:57`); PM does not.
- **Units:** PM deposits mass per cell, not per unit area, so changing `PHYSICS_CELL_SIZE` silently rescales every force by `cell_size²`. Divide `density_grid` by `cell_size**2`.
- **Softening:** the GUI calls it "Plummer" (`gui.py:125`), but PM uses a Gaussian `exp(-k²s²/2)`. Only N-body is Plummer.

### 2.12 Spawned galaxies aren't in equilibrium (Low)
**Fixed** (PR #5):
- Spawn speeds are `sqrt(G·M(<r)·(κr)K₁(κr))`, which includes the force-range screening.
- The spawn-velocity slider is now a multiple of the orbit speed: range −2…2, default 1.
- Measured: a 200k-particle disk keeps its median radius within 7% over 1 s (+20% by 2 s, from the screening approximation). At rest it collapses to 39%.
- *PR #7:* replaced the analytic formula with the measured field, `v = sqrt(r·a_inward)`, from a real field solve (PM) or the N-body solver. Disks now hold their median radius within **0.5% over 2 s**, in both dense-core and uniform modes.

`orbital_speed = sqrt(|G|·100/(r+10))` (`engine.py:131`) is an arbitrary profile. For the 2D PM law, the circular speed is `v² = G·M(<r)/(2π)`. For the "dense core" distribution (r uniform), M(<r) = count·r/R, so `v = sqrt(G·count·r/(2π·R))` (divided by cell_size² once 2.11 is fixed). The current formula makes disks either fly apart or collapse depending on brush size and density. `spawn_vel` then multiplies whatever the result is.

### 2.13 Smaller issues
**Fixed** (PR #6):
- the image list is cached, with a "Rescan Images" button;
- view and bandpass slider ranges come from `config.json`;
- the unused `show_fft_panel` is gone;
- the Coulomb colour tensors are created once;
- `image_loader` reports when an image yields no particles.

Not changed: `time.sleep` throttling (the project now runs on Python 3.13, with high-resolution timers), and `dt` from the target frame time (kept deliberately for deterministic steps; documented in AGENTS.md).

- `gui.py:91-94, 235-237`: the view and bandpass sliders hard-code their ranges (−5…15, 0…5) and ignore the `config.json` ranges, which differ.
- `gui.py:202-205`: five `glob.glob` calls per frame, which is a filesystem query 120 times a second (much worse when run from a Google Drive virtual disk). Cache the list and add a "Rescan" button.
- `gui.py:15`: `show_fft_panel` is never used.
- `main.py:414-417`: `time.sleep` FPS throttling. On Python 3.10 on Windows (the `__pycache__` is cp310), sleep granularity is about 15.6 ms, which caps the frame rate near 64 FPS when the target is 120. Python 3.11+ uses high-resolution timers.
- `main.py:120`: `dt` comes from the *target* frame time, not the measured one, so the simulation runs in slow motion whenever FPS drops. That may be intentional (deterministic steps), but it isn't documented.
- `main.py:258-259`: two small host→device `torch.tensor` uploads per frame in Coulomb N-body rendering. Hoist them out of the loop.
- `image_loader.py`: if the image has very few bright pixels, `total_spawn == 0` returns `None` silently.

---

## 3. Performance and memory

### 3.1 Memory footprint at the shipped config (100M particles, 7680² grid)
**Fixed** (PR #3, #6):
- `kx_grid`/`ky_grid` are gone, and one `k_sq` in rfft2 layout is shared.
- The distance grid is built on demand only.
- Kernels are built directly in rfft2 layout: real float32 for gravity/Coulomb, `rfft2` for harmonic/LJ.
- `K_eff` (kernel × softening) is cached.

At 7680² that removes ~1.2 GB of persistent grids.

| Item | Size |
|------|------|
| `pos`, `vel` (float32, MAX_PARTICLES×2), `masses` (int8) | 1.7 GB |
| `PotentialsManager` `kx_grid`, `ky_grid`, `k_sq`, `r` | 0.94 GB (only `k_sq` and `r` are used after init, and `k_sq` duplicates the solver's) |
| `ParticleMeshSolver` `k_sq`, `density_grid` | 0.47 GB |
| `K_k` stored as full complex64 from `fft2` | 0.47 GB (only half the columns are ever used; gravity/Coulomb kernels are purely real) |
| Per-step temporaries: `rho_k`, `phi_k`, `gaussian`, `phi`, `fx`, `fy` | ~1.5 GB churn |

Savings:
- Drop `kx_grid`/`ky_grid`, share one `k_sq`, and build `r` only when building a harmonic or LJ kernel.
- Store `K_k` in `rfft2` layout, as real float32 for gravity/Coulomb: 0.12 GB instead of 0.47 GB, and real×complex multiplies.
- Precompute `K_eff = K_k · exp(-k²s²/2)` with `K_eff[0,0] = 0`. Rebuild it only when G, range or softening change. That removes one `exp` pass and one complex multiply per step.

### 3.2 Particle memory traffic dominates
Per frame at N = 100M:

| Pass | Traffic |
|------|---------|
| Boundary mask: reads `pos` and builds four 100 MB bool temporaries, then `.all().item()` | ~1.2 GB + a **GPU→CPU sync every frame** |
| Scatter: reads `pos`, `masses` | 0.9 GB |
| Gather + integrate: reads and writes `pos`, `vel` | 3.2 GB |
| Render scatter: reads `pos` | 0.8 GB |

That totals about 6 GB, roughly 6 ms of pure DRAM time even on a ~1 TB/s RTX 4090. The random-access atomics make it worse, so 120 FPS at 100M isn't reachable regardless of the solver.

Fixes:
- **Fuse integrate + boundary + next-step deposit + render splat** into the single gather kernel. After moving a particle, wrap or reflect it, `atomic_add` it into next step's `density_grid`, and if it's on screen, into `visual_density`. Particle traffic per frame drops to one read and one write (~3.3 GB).
- **Remove the per-frame `.item()`.** With periodic wrapping (2.8) nothing needs deleting. Otherwise, compact every N frames or on demand.
- **Sort particles by cell periodically.** The 7680² grid is 236 MB, far larger than any GPU cache, so CIC scatter and gather are cache-miss bound. Measured on the Arc B390 at 16M particles on a 4096² grid: sorting cut scatter from 103 ms to 39 ms and gather from 140 ms to 78 ms. One argsort plus permute costs 81 ms, so doing it every ~50 frames is nearly free. The same applies on CUDA.

### 3.3 Redundant FFTs
**Fixed** in the XPU port: the solver keeps `rho_k`, which the minimap and bandpass view reuse. The softened kernel `K_eff` is also cached instead of recomputing `exp` every step.

`main.py` recomputes `rfft2(density_grid)` for the bandpass render view and again for the FFT minimap, on top of the solver's own forward FFT. Keep `rho_k` on the solver (it's already computed in `solve`) and reuse it. The bandpass mask code is also duplicated verbatim (`main.py:311-320` and `366-375`); extract it into one function and cache the mask until the sliders change.

### 3.4 Force differencing
**Fixed** (PR #3): periodic central differences with `torch.roll` when the boundary is periodic.

`torch.gradient` allocates two new grids and uses one-sided differences at the edges, which doesn't match the periodic solve. Use either a periodic central difference with `torch.roll`, or a single fused kernel that writes `fx`/`fy` in place. Spectral differentiation (`i·k·φ_k`) is exact but needs two inverse FFTs, so only consider it if accuracy matters more than speed.

### 3.5 N-body solver
**Fixed** (PR #6): row-tiled pairwise computation with separate x/y arrays and in-place ops.

| Particles | Before | After |
|---|---|---|
| 10k | 3.35 GiB, 180 ms | 0.13 GiB, 146 ms |
| 20k | 13.4 GiB, 842 ms | 0.13 GiB, 630 ms |

Skipping the screening at maximum range was *not* done: `exp(-r/8192)` is still 0.6 at r = 4096, so skipping it would change the physics. The cap stays at 10k because the solver is speed-bound on the iGPU; it's configurable.

- **Memory:** about ten dense N×N tensors. Process the target particles in row blocks (e.g. 1024–4096 rows) against all sources, so memory is O(N·block) and N = 50k+ becomes feasible.
- **Cost:** `screening = exp(-r/range)` is always computed, even when the range is at its maximum. Skip it when `range ≥ universe size`.

### 3.6 Kernel rebuilds while dragging sliders
**Fixed** (PR #6): sliders call `engine.request_kernel_rebuild()`. The rebuild runs once the value has been still for 0.15 s.

Every frame that a G, range, spring or LJ slider moves rebuilds the kernel. For harmonic and LJ that means a full-grid `fft2` (472 MB of complex output) plus several 236 MB temporaries such as `r**2` and `(σ/r)^12`. Rebuild on slider release, or at most every ~100 ms.

---

## 4. Code quality and maintainability

- **Two diverged copies.** At audit time, the June Google Drive copy and the August laptop copy had drifted apart, and neither was under version control. *Resolved:* the August copy is now this repo. The Google Drive copy (`GDrive\Documents\Scripts\PPS\Particle Mesh`) is stale; don't put a `.git` folder there, because Google Drive syncing can corrupt it.
- **Relative `config.json` paths and import-time side effects.** `main.py`, `engine.py` and `mesh_renderer.py` each load the config with a relative path. `mesh_renderer.py` and `circle_renderer.py` allocate Taichi fields at import. One `config.py` with an absolute path (the August copy already does the path part) plus lazy field allocation would make the modules importable in tests.
  - *Status, PR #6:* `config.py` is now the only reader of `config.json`.
  - *Status, PR #7:* fields are now created by their owners after `ti.init()`, so modules import in any order (`tests/test_imports.py`).
- **`main.py` does too much.** Input handling, view resampling, colouring and overlays all live in one 420-line loop. Move the PM view and overlay logic into `MeshRenderer` (which also fixes 2.1), and keep the key edge-trigger state in a dict.
  - *Status, PR #6:* the PM views, minimap, bandpass and arrows moved into `MeshRenderer.render_pm`, and `main.py` dropped ~110 lines.
  - *Status, PR #7:* the input edge-trigger flags are replaced by a `KeyEdges` helper.
- **No tests.** *Resolved (PR #4 onwards): `tests/`, run with `uv run pytest`, all headless.* Taichi's `ti.cpu` arch plus CPU tensors runs the real kernels with zero-copy, so headless physics checks are cheap:
  - Two-body force against the analytic 2D law, for PM and N-body: this catches the 2π mismatch.
  - Total momentum stays ~0 for a symmetric configuration.
  - Coulomb: an opposite-charge pair attracts and a like-charge pair repels in both solvers: this catches 2.2.
  - Damping never flips velocity sign: this catches 2.3.
- **Dependencies are undeclared here.** The parent `C:\Python Dev\PPS\pyproject.toml` lists `opencv-python`, `pygame` and `torchvision`, which this project doesn't import (they may be for `Ep1/`). `Pillow` and `tkinter` are used lazily.

---

## 5. Running efficiently on Intel XPU (Arc B390)

### 5.1 What's on this machine
- torch **2.13.0+xpu** in `C:\Python Dev\PPS\.venv`. `torch.xpu.is_available()` is True: B390, 96 EUs, 16.7 GB shared memory, 16 MB last-level cache. oneMKL SYCL DFT is installed, so `torch.fft` runs on the GPU.
- Taichi **1.7.4**, which has **no XPU backend**. On Intel it runs on **Vulkan** (verified working on the B390).
- `triton_xpu` 3.7.2 is installed, but **`torch.compile` fails**: "Failed to find C compiler". No MSVC or clang is installed.

### 5.2 Why the August port is slow (verified)
`taichi/lang/kernel_impl.py:777-785`: if a torch tensor's device is anything other than `cpu` (or `cuda` on the CUDA arch), Taichi copies it to a CPU tensor before the launch. The Vulkan runtime then uploads it, runs the kernel, downloads the result, and a callback copies it back to the XPU tensor. That happens **for every ndarray argument on every kernel call**. The August port keeps `pos`, `vel` and the grids on XPU and calls Taichi kernels on them several times per frame.

Measured: scatter on an XPU tensor with 1M particles on a 1440² grid takes **9.45 ms per call**. The same kernel on Taichi-owned Vulkan ndarrays does **4M particles in 2.44 ms**, about 15× faster per particle.

The port also has these problems:
- The Vulkan red-black Gauss-Seidel fallback runs 12 sweeps from φ = 0 each step. Information travels about one cell per sweep, so gravity is truncated to ~12 cells. It also uses Dirichlet (φ = 0) edges and ignores G, force range, softening and potential type.
- Its Taichi N-body kernel uses `dx/r³` (3D gravity), while the torch path uses `dx/r²` (2D).

### 5.3 Benchmarks on the B390 (ms per call)
| Stage | 1M / 2048² | 4M / 2048² | 16M / 4096² | 32M / 4096² |
|---|---|---|---|---|
| torch eager XPU: CIC deposit (`index_add_`) | 2.7 | 11.6 | 105 (39 sorted) | 207 |
| torch eager XPU: FFT solve + gradient | 5.6 | 5.5 | 22.6 | 23.0 |
| torch eager XPU: gather + integrate | 3.4 | 18.5 | 77–140 (78 sorted) | 149 |
| Taichi **Vulkan** scatter (Taichi-owned ndarrays) | — | 2.4 | 27.4 | — |
| Taichi **Vulkan** gather + integrate | — | 3.1 | 15.1 | — |
| Grid hand-off Vulkan ↔ XPU via host (density down, fx/fy up) | — | 18.0 | 94.6 | — |

The 2048² grid is 16 MB and fits the B390's 16 MB last-level cache; 4096² doesn't, which explains the jump in cost. Reading back a finished RGB frame from XPU costs **1.96 ms** at 2880×1800 and **0.55 ms** at 1440×900. `canvas.set_image()` accepts a NumPy `uint8` array, so presenting a torch-rendered frame through GGUI is cheap.

Takeaways:
- Taichi's generated particle kernels are **4–6× faster** than eager PyTorch. Eager PyTorch creates many N-sized temporaries and int64 index tensors.
- The XPU FFT (oneMKL) is fast.
- Moving grids between Taichi Vulkan and torch XPU goes through the CPU and costs more than everything else combined.

### 5.4 Options
**A. Now, no installs: pure torch on XPU, no Taichi compute.** Rewrite the four Taichi compute kernels (scatter, gather + integrate, render splat, colormap) as torch ops (`index_add_`, gather indexing, LUT lookup). Keep Taichi only for the GGUI window on Vulkan, and present with `canvas.set_image(img.cpu().numpy())`. Expected physics cost per frame: ~12 ms at 1M particles / 2048², ~36 ms at 4M. That is playable at 1–2M particles. Add periodic sorting (3.2) for larger grids.

**B. Recommended: torch XPU + Triton kernels.** Install **Visual Studio 2022 Build Tools (C++ workload)**. Triton's Windows XPU backend needs a host C compiler, and PyTorch's Intel GPU guide lists MSVC for `torch.compile` on Windows. Then:
- Write the hot particle passes as `@triton.jit` kernels that operate directly on XPU tensors, with zero copies:
  - one fused "gather → kick → drift → wrap → deposit next density → splat visible" kernel;
  - a colormap kernel.
  Alternatively, `torch.compile` the torch version from option A.
- Keep the FFT in `torch.fft` (oneMKL).
- Keep GGUI only as the window: one ~0.5–2 ms frame readback, and Taichi never sees an XPU tensor.

Native GPU kernels with atomics should be roughly as fast as the Taichi Vulkan numbers above. My estimate is ~12–15 ms per frame at 4M particles / 2048² (60+ FPS), but that is an extrapolation until the compiler is installed and it's measured.

**C. All-Vulkan Taichi.** Particles, rendering and the Poisson solve all live in Taichi Vulkan, so nothing crosses devices. That needs a hand-written Taichi FFT (power-of-two grid; roughly 22 radix-2 passes at 2048²) or a periodic multigrid solver, which can't do the arbitrary LJ/harmonic kernels. You'd own and maintain that FFT, so I'd only take this route if you want to drop PyTorch.

**Not recommended:**
- The current hybrid (XPU tensors into Taichi kernels).
- Intel Extension for PyTorch (IPEX). Upstream torch-xpu already provides everything used here.

### 5.5 Suggested config for the B390
`PHYSICS_GRID_W/H = 2048`, `PHYSICS_CELL_SIZE = 1`, `MAX_PARTICLES ≈ 4–8M`, `initial_particles ≈ 1–2M`, window 2880×1800 with `RENDER_CELL_SIZE = 1` (or 2 for speed once 2.10 is fixed), `show_fft = false` until 2.1 is fixed. Remove Taichi's `device_memory_fraction`; it only applies to CUDA.

---

## 6. Suggested order of work
1. Consolidate into one git repo (§4).
2. ~~Quick correctness fixes: 2.3 damping, 2.7 `K[0,0] = 0`, 2.2 Coulomb charge, 2.4 N-body zoom, 2.5 N-body cap, 2.1 overlay order.~~ **Done** on branch `fix/quick-correctness`. That branch also fixed the minimap's size when `RENDER_CELL_SIZE ≠ 1` and the point-reflection mirroring. The redundant FFTs (3.3) were removed in PR #3; antialiasing is tracked in §7.
3. ~~XPU option A, to get the app running on this laptop.~~ **Done** on branch `feat/xpu-port`. It adds a pure-PyTorch kernel backend with a shared CIC stencil, periodic cell sorting, all rendering in PyTorch, a 2048² grid, reuse of `rho_k`, a cached `K_eff` and a LUT colormap. Measured on the B390: 1M particles ≈ 17 ms/frame (≈59 FPS), 4M ≈ 42 ms. The old path took 115 ms for physics alone at 1M. CUDA machines keep the Taichi kernels via `kernels.TaichiBackend`; that path is parity-tested on Taichi CPU but not run on CUDA here.
4. Install MSVC Build Tools and move to option B (Triton fused particle kernel). The hot paths are already isolated in `kernels_torch.py`, and `TORCH_COMPILE: true` will try `torch.compile` and fall back to eager if it fails. Benchmark against the numbers in step 3.
5. ~~Boundary model (2.8), pressure form (2.9), units and parity (2.11), and a small physics test suite.~~ **Done:** 2.8 in PR #3, the test suite in PR #4, 2.9/2.11/2.12 in PR #5. The cleanup items (6, 2.13, 3.1, 3.5, 3.6, §4) are in PR #6.

The benchmarks were run with throwaway scripts (`bench_xpu.py`, `bench_xpu2.py`, `test_set_image_order.py`) that are not committed. Ask for them to be re-created under a `bench/` folder if needed.

---

## 7. Remaining work (partials and loose ends)

Everything still open after PRs #1–#6, in the order it's being done. Option B is deliberately last.

| # | Item | Source | Status |
|---|---|---|---|
| 1 | Stale docs: config loading, slider rebuilds, N-body tiling, `build_view_grid` location, "redundant FFTs still open" | AGENTS.md, §6 | Done (PR #7) |
| 2 | FFT minimap shrinks without antialiasing | 2.1 | Done (PR #7) |
| 3 | Spawned galaxies drift (+20% radius by 2 s) because the screening formula is an approximation | 2.12 | Done (PR #7) |
| 4 | Six hand-rolled `prev_*` key edge-trigger flags in `main.py` | §4 | Done (PR #7) |
| 5 | Taichi fields allocated at import, forcing `ti.init()` before imports | §4 | Done (PR #7) |
| 6 | **Option B: fused Triton / `torch.compile` particle kernels** (item 11's remaining part) | 5.4, 3.2 | Deferred: needs Visual Studio 2022 Build Tools |

---

## 8. Performance opportunities (2026-10-15 sweep)

A read-only multi-agent sweep produced these findings. Each one was then checked by a correctness reviewer and a performance reviewer, against the measured profile and the installed torch 2.14.1+xpu / Taichi 1.7.4 source. Gains are the reviewers' estimates unless a measurement is given. Triton and `torch.compile` were out of scope (that's §7 item 6).

| ID | Change | Area | Estimated / measured gain | Status |
|---|---|---|---|---|
| F01 | Present frames as packed RGBA8 straight to GGUI (no `from_torch`, no to_rgba8) | render | **Render at 1M measured 6.6–8.2 → 3.1–3.2 ms** (with F09–F11) | Done (PR #8) |
| F09 | Count-indexed colour LUT (one clamp + one gather) | render | ~0.5 ms | Done (PR #8) |
| F10 | Leaner splat: border-ring clamp, x-major output, no bool masks | render | ~0.2–0.3 ms at 1M | Done (PR #8) |
| F11 | Minimap: skip identity band-pass, refresh every `minimap_interval` frames | render | ~0.3–0.8 ms | Done (PR #8) |
| F20 | Force arrows: sample forces only at arrow centres | render | **Arrow view at 1M measured 15.6 → 9.9 ms render** | Done (PR #8) |
| F21 | Potential/bandpass views: abs after sampling, identity band-pass = density grid, cached view grid | render | 0.1–1.8 ms in those views | Done (PR #8) |
| F07 | Cache per-step constant tensors (no H2D copy and sync); in-place wrap | step | ~0.2–0.4 ms | Done (PR #8) |
| F48 | `tools/bench_frame.py` times the real `render_pm` frame (minimap), plus `--zoom` and `--field` | tooling | measurement | Done (PR #8) |
| F03 | Periodic force differences by slicing into persistent buffers (no `torch.roll`) | field | ~1.3–2.2 ms/step | Done (PR #9): measured 2.7 -> 1.2 ms |
| F02 | Fused `grid_sample` gather + `grid_sampler_2d_backward` deposit instead of the hand-built stencil (F05 int32 stencil as fallback) | step | ~2–4.5 ms at 1M, 8–12 ms at 4M | Done (PR #9): measured stencil+deposit+gather 6.5 -> 1.9 ms (1M), 24.6 -> 6.3 ms (4M); step 11.2 -> 6.0 ms (1M), 32.8 -> 16.3 ms (4M) |
| F12 | Skip the mass cast/multiply when every mass is 1 | step | ~0.15–0.3 ms at 1M | Subsumed by F02 (the masses are the sampler gradient; no separate multiply) |
| F04 | Reuse the step's field for spawn orbit speeds (no second solve) | spawn | ~9–20 ms per painting frame | Done (PR #9): painting frame 24.3 -> 7.9 ms |
| F13 | Cheaper periodic sort (smaller keys, permute into swap buffers) | sort | sort hitch ~35 → ~20 ms at 4M | Done (PR #9): int16 tile keys (argsort 9.0 -> 5.3 ms at 4M) + swap buffers |
| F22 | Leaner pressure path (slice gradients, no pow at γ = 1, fused scales) | field | ~2–3.5 ms when pressure is on | Done (PR #9) |
| F24 | Version-counter cache key for `K_eff` (the `id()` key can collide); `irfft2(norm='forward')` A/B | field | correctness; 0–0.3 ms | Done (PR #9): version-counter key; irfft2(norm='forward') 1.36 -> 1.09 ms |
| F27 | Fold uniform gravity (and reflective walls) into the integrate pass | step | 0.5–4 ms in those modes | Done (PR #9): uniform gravity in the kick; vectorised reflective bounce |
| F32 | Opt-in multi-rate field update (`FIELD_INTERVAL`, default 1) | field | ~3 ms average at k = 2 if enabled | Done (PR #9): FIELD_INTERVAL, default 1 |
| F41 | CUDA: smaller Taichi memory pool (`device_memory_GB`) | capacity | gives torch 1.5–3 GB back on CUDA | Done (PR #9): device_memory_GB=0.5 |
| F36 | N-body: one fused Taichi kernel (zero-copy on CUDA, tiny CPU hop on Vulkan) | N-body | 7–30× at 10k; cap could rise to ~25–40k | Planned (perf PR 3) |
| F33/F34 | N-body torch path: fewer passes per tile, symmetric tile pairs | N-body | ~1.5–2.5× | Planned (perf PR 3) |
| F19, R4.1 | Opt-in config tradeoffs: 2048×1280 grid, `RENDER_CELL_SIZE` 3 | config | ~1.5–2.7 ms each | Documented option |
| F05 | int32 stencil | step | ~1–1.6 ms at 1M | Not needed (F02 is faster) |
| F06, F08, F17, F18, F28, F35, F47 | Single force buffer, pipelined present, direct ring VBO, render from density grid, fixed-size splat scatter, far-pair culling, Taichi colour expansion | various | ≤0.5 ms each, or scenario-dependent | Not pursued (uncertain or small) |
| F14–F16, F25, F26, F29–F31, F37–F40, F42–F46, F49, F50, R1–R6 | Time-based or rolling sorts, erase fast path, batched iFFTs, real LJ kernels, `inference_mode`, ImGui trims, XPUGraphs, circle VBO, cap/workspace/chunking changes, CUDA texture present, Taichi fusions and atomics, spectral forces, NGP, a Vulkan particle backend, frame pacing | various | refuted: negligible, slower, or infeasible | Rejected |

The critic's main idea, moving particles into Taichi Vulkan ndarrays on the laptop, was rejected. Each step would hand 2 × 16 MB grids across through the host, costing ~12–18 ms per step (§5.3), which outweighs the 4–6× faster particle kernels at the default 1M particles.
