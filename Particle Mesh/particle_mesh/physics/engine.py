import time
import torch
import math
from particle_mesh.settings import CONFIG, SETTINGS  # noqa: F401  (CONFIG re-exported)
from particle_mesh.physics.potentials import PotentialsManager
from particle_mesh.physics.nbody_solver import DirectNBodySolver
from particle_mesh.physics.pm_solver import ParticleMeshSolver
from particle_mesh.kernels import make_backend
from particle_mesh.runtime.device import const_tensor, get_torch_device
from particle_mesh.view_state import VIEW_FIELDS, ViewState

WIDTH = SETTINGS.width
HEIGHT = SETTINGS.height
FPS_TARGET = SETTINGS.fps_target

PHYSICS_GRID_W = SETTINGS.grid_w
PHYSICS_GRID_H = SETTINGS.grid_h
PHYSICS_CELL_SIZE = SETTINGS.physics_cell_size
MAX_PARTICLES = SETTINGS.max_particles
# Thermal spawn velocity spread for non-gravity potentials (at spawn_velocity = 1)
THERMAL_SPEED = 1250.0
# Direct N-body is O(N^2) work per step. With the fused Taichi kernel ~25k is interactive on the Arc B390
# (10k ~5 ms, 20k ~8-10 ms, 40k ~30 ms per step); the tiled torch fallback is ~30-100x slower.
NBODY_MAX_PARTICLES = SETTINGS.nbody_max_particles
# Slider-driven kernel rebuilds wait until the value has been still this long (seconds)
KERNEL_REBUILD_DELAY = 0.15

POTENTIALS = ('gravity', 'lennard_jones', 'harmonic', 'coulomb')
PRESSURE_MODELS = ('none', 'isothermal', 'polytropic')
# 'open': particles that leave the physics mesh are deleted, and forces come from an isolated (zero-padded)
# field solve with no periodic images. 'periodic': a wrapped torus. 'reflective_walls': bounce at the mesh edge
# (isolated solve). toggle_boundary() cycles in this order.
BOUNDARY_MODES = ('open', 'periodic', 'reflective_walls')
# The force-range slider's maximum means "no screening": the pure 2D law |a| = G * m / r
FORCE_RANGE_OFF = SETTINGS.slider("gravity_range")[1]

UNIVERSE_W = SETTINGS.universe_w
UNIVERSE_H = SETTINGS.universe_h
GRID_W = PHYSICS_GRID_W
GRID_H = PHYSICS_GRID_H

DEVICE = get_torch_device()

def _forward_to_view(name):
    return property(lambda self: getattr(self.view, name), lambda self, value: setattr(self.view, name, value))


class SimulationEngine:
    def __init__(self, initial_particles=None, solver=None):
        """
        initial_particles: startup count (default: config initial_particles). solver: 'pm' or 'nbody' (default:
        config SOLVER). In N-body mode the startup count is capped at NBODY_MAX_PARTICLES.
        """
        self.device = DEVICE
        self.width = WIDTH
        self.height = HEIGHT
        self.universe_w = UNIVERSE_W
        self.universe_h = UNIVERSE_H
        
        # Solver & Mode Settings
        self.solver_type = (solver or SETTINGS.solver).lower()
        self.potential_type = 'gravity' # 'gravity', 'lennard_jones', 'harmonic'
        self.pressure_model = 'none' # 'none', 'isothermal', 'polytropic'
        self.boundary_mode = 'open'  # see BOUNDARY_MODES
        
        # Physics Parameters
        self.G = SETTINGS.slider("gravity")[2]
        self.gravity_range = SETTINGS.slider("gravity_range")[2] # Screen range
        self.friction = SETTINGS.slider("friction")[2] # Friction damping
        
        self.pressure_strength = SETTINGS.slider("pressure")[2]
        self.gamma = SETTINGS.slider("polytropic_gamma")[2]
        
        self.spring_k = SETTINGS.slider("spring_k")[2]
        self.epsilon = SETTINGS.slider("epsilon")[2]
        self.sigma = SETTINGS.slider("sigma")[2]
        self.softening_length = SETTINGS.slider("softening_length")[2]
        
        self.uniform_gravity_mag = SETTINGS.slider("uniform_gravity_mag")[2]
        self.uniform_gravity_angle = SETTINGS.slider("uniform_gravity_angle")[2]
        
        self.uniform_spawn = False
        
        # Rendering & Visualization States (view toggles, colour ranges, band-pass, theme)
        self.view = ViewState.from_settings()
        
        # Particle State Tensors
        if initial_particles is None:
            initial_particles = SETTINGS.initial_particles
        if self.solver_type == 'nbody' and initial_particles > NBODY_MAX_PARTICLES:
            print(f"[Warning] initial_particles ({initial_particles}) is extremely high for O(N^2) Direct N-Body mode.")
            print(f"          Capping to NBODY_MAX_PARTICLES ({NBODY_MAX_PARTICLES}) to prevent a hang or out-of-memory error.")
            initial_particles = NBODY_MAX_PARTICLES
        self.num_active = initial_particles
        self.pos = torch.zeros((MAX_PARTICLES, 2), device=self.device)
        self.vel = torch.zeros((MAX_PARTICLES, 2), device=self.device)
        self.masses = torch.zeros(MAX_PARTICLES, dtype=torch.int8, device=self.device)
        
        # Seed initial particles
        init_cam_x = (self.universe_w - self.width) / 2.0
        init_cam_y = (self.universe_h - self.height) / 2.0
        self.pos[:self.num_active, 0] = (torch.rand(self.num_active, device=self.device) * self.width) + init_cam_x
        self.pos[:self.num_active, 1] = (torch.rand(self.num_active, device=self.device) * self.height) + init_cam_y
        self.vel[:self.num_active] = (torch.rand((self.num_active, 2), device=self.device) - 0.5) * 10.0
        self.masses[:self.num_active] = 1
        
        # Particle-kernel backend: Taichi on shared memory (CUDA/CPU), pure PyTorch otherwise (Intel XPU)
        self.kernels = make_backend(
            self.device,
            SETTINGS.kernel_backend,
            SETTINGS.torch_compile
        )
        # Re-sort particles by grid cell every N steps so CIC scatter/gather hit memory coherently (0 = off)
        self.sort_interval = SETTINGS.sort_interval
        self.step_count = 0
        # Pending slider-driven kernel rebuild (see request_kernel_rebuild)
        self._kernel_dirty_since = None
        # Opt-in multi-rate PM: rebuild the field (deposit + FFT) every FIELD_INTERVAL steps; 1 = every step
        self.field_interval = SETTINGS.field_interval
        self._field_dirty = True       # particles added/removed or kernel changed since the last field solve
        # Freshly spawned gravity disks waiting for their orbit speeds: (start, end, centre_x, centre_y, vel_mult).
        # Applied inside step() from that step's own field, so spawning needs no second solve (AUDIT §8 F04).
        self._pending_orbits = []
        # Sort keys: index of the t x t-cell tile holding each particle. The finest tile whose count fits int16
        # (8x8 cells on 2048^2); int16 argsort is ~1.7x faster than float cell keys (AUDIT §8 F13).
        self._sort_tile = 1
        while (math.ceil(GRID_W / self._sort_tile) * math.ceil(GRID_H / self._sort_tile) > 65536
               and self._sort_tile < 16):
            self._sort_tile *= 2
        self._sort_tiles_x = math.ceil(GRID_W / self._sort_tile)
        self._sort_tiles_y = math.ceil(GRID_H / self._sort_tile)
        self._sort_key16 = self._sort_tiles_x * self._sort_tiles_y <= 65536
        self._sort_alt = None          # swap buffers for the sort permutation (allocated on first sort)

        # Initialize helper managers and solvers
        self.pot_manager = PotentialsManager(GRID_W, GRID_H, PHYSICS_CELL_SIZE, self.device)
        # Direct N-body: fused Taichi kernel by default, tiled torch path if NBODY_KERNEL is "torch"
        self.nbody_solver = DirectNBodySolver(
            self.device, use_taichi=SETTINGS.nbody_kernel != "torch")
        self.pm_solver = ParticleMeshSolver(GRID_W, GRID_H, PHYSICS_CELL_SIZE, self.device, self.pot_manager, self.kernels)

    @property
    def periodic(self):
        return self.boundary_mode == 'periodic'

    def sort_keys(self, n=None):
        """Row-major tile index of each active particle (int16 when the tile count fits, else int32)."""
        n = self.num_active if n is None else n
        t = self._sort_tile
        tiles = (self.pos[:n] * (1.0 / (PHYSICS_CELL_SIZE * t))).to(torch.int32).clamp_(min=0)
        tiles = torch.minimum(tiles, const_tensor((self._sort_tiles_x - 1, self._sort_tiles_y - 1),
                                                  self.device, torch.int32))
        key = tiles[:, 1] * self._sort_tiles_x + tiles[:, 0]
        return (key - 32768).to(torch.int16) if self._sort_key16 else key

    def sort_particles(self):
        """
        Reorders active particles by grid tile. Memory-coherent CIC deposit/gather is ~2-4x faster. The permutation
        is gathered straight into swap buffers (index_select out=), then the buffers are swapped: no copy back.
        """
        n = self.num_active
        if n < 2:
            return
        order = torch.argsort(self.sort_keys(n))
        if self._sort_alt is None:
            self._sort_alt = (torch.empty_like(self.pos), torch.empty_like(self.vel), torch.empty_like(self.masses))
        pos_alt, vel_alt, masses_alt = self._sort_alt
        torch.index_select(self.pos[:n], 0, order, out=pos_alt[:n])
        torch.index_select(self.vel[:n], 0, order, out=vel_alt[:n])
        torch.index_select(self.masses[:n], 0, order, out=masses_alt[:n])
        self._sort_alt = (self.pos, self.vel, self.masses)
        self.pos, self.vel, self.masses = pos_alt, vel_alt, masses_alt

    def request_kernel_rebuild(self):
        """
        Debounced kernel rebuild for slider drags (AUDIT 3.6): the rebuild happens in step() once the parameter
        has been unchanged for KERNEL_REBUILD_DELAY seconds, instead of every frame while dragging (a harmonic/LJ
        rebuild is a full-grid FFT).
        """
        self._kernel_dirty_since = time.perf_counter()

    def precompute_kernel(self):
        """
        Triggers potentials precomputation inside solvers on parameter change.
        """
        self._kernel_dirty_since = None
        self._field_dirty = True
        self.pm_solver.update_kernel(self.potential_type, self._physics_params(), isolated=not self.periodic)

    # ------------------------------------------------------------------ commands (called by the GUI panel)

    def set_potential(self, kind):
        """Switch the interaction potential. Resets every particle's mass/charge (random +/-1 charges for Coulomb,
        1 otherwise) and rebuilds the kernel immediately."""
        if kind not in POTENTIALS:
            raise ValueError(f"Unknown potential: {kind}")
        self.potential_type = kind
        self.masses[:self.num_active] = self._new_masses(self.num_active)
        self.precompute_kernel()

    def set_pressure_model(self, model):
        if model not in PRESSURE_MODELS:
            raise ValueError(f"Unknown pressure model: {model}")
        self.pressure_model = model

    def toggle_boundary(self):
        """Cycle open -> periodic (wrapped) -> reflective walls. The PM kernel follows on the next solve."""
        i = BOUNDARY_MODES.index(self.boundary_mode)
        self.boundary_mode = BOUNDARY_MODES[(i + 1) % len(BOUNDARY_MODES)]
        self._field_dirty = True

    def _inside(self, p):
        """Bool mask of the (M, 2) positions inside the physics mesh [0, universe)."""
        size = const_tensor((float(self.universe_w), float(self.universe_h)), p.device, p.dtype)
        return ((p >= 0) & (p < size)).all(dim=1)

    def _drop_outside(self):
        """Open boundary: delete the particles that left the physics mesh (survivors stay compacted, in order)."""
        n = self.num_active
        inside = self._inside(self.pos[:n])
        kept = int(inside.sum())
        if kept == n:
            return
        self._flush_pending_orbits()   # compaction below would invalidate their index ranges
        self.pos[:kept] = self.pos[:n][inside]
        self.vel[:kept] = self.vel[:n][inside]
        self.masses[:kept] = self.masses[:n][inside]
        self.num_active = kept
        self._field_dirty = True

    def capacity(self):
        """
        Maximum number of active particles for the current solver.
        """
        if self.solver_type == 'nbody':
            return min(MAX_PARTICLES, NBODY_MAX_PARTICLES)
        return MAX_PARTICLES

    def _new_masses(self, count):
        """
        Masses for newly spawned particles: 1, or a random +/-1 charge in Coulomb mode.
        """
        if self.potential_type == 'coulomb':
            return torch.randint(0, 2, (count,), device=self.device, dtype=torch.int8) * 2 - 1
        return torch.ones(count, device=self.device, dtype=torch.int8)

    def _physics_params(self):
        return {
            'G': self.G,
            'force_range': math.inf if self.gravity_range >= FORCE_RANGE_OFF else self.gravity_range,
            'spring_k': self.spring_k,
            'epsilon': self.epsilon,
            'sigma': self.sigma,
            'softening_length': self.softening_length
        }

    def _apply_pending_orbits(self, sample):
        """Give every pending spawned disk its orbit speeds; sample(start, end) returns the field (ax, ay) there."""
        for start, end, world_x, world_y, vel_mult in self._pending_orbits:
            ax, ay = sample(start, end)
            self._set_orbital_velocities(start, end, world_x, world_y, vel_mult, ax, ay)
        self._pending_orbits.clear()

    def _flush_pending_orbits(self):
        """Apply pending orbit speeds right now with a separate field evaluation. Only needed when particles are
        about to be reordered or erased before the next step (the normal path reuses the step's field)."""
        if not self._pending_orbits:
            return
        if self._kernel_dirty_since is not None:
            self.precompute_kernel()   # apply a pending slider change so speeds match the current G
        n = self.num_active
        params = self._physics_params()
        if self.solver_type == 'nbody':
            ax, ay = self.nbody_solver.solve(self.pos[:n], self.masses[:n], 'gravity', params,
                                             self.periodic, self.universe_w, self.universe_h)
            self._apply_pending_orbits(lambda s, e: (ax[s:e], ay[s:e]))
        else:
            s0, e0 = self._pending_orbits[0][:2]
            self.pm_solver.accelerations_at(self.pos, self.masses, n, s0, e0, 'gravity', params,
                                            self.periodic, self.universe_w, self.universe_h)
            self._apply_pending_orbits(
                lambda s, e: self.pm_solver.sample_forces(self.pos[s:e], self.universe_w, self.universe_h))

    def _set_orbital_velocities(self, start, end, world_x, world_y, vel_mult, ax, ay):
        """
        Gives particles start:end (a freshly spawned disk centred on world_x, world_y) the circular speed of the
        field they actually feel (AUDIT 2.12): v = sqrt(r * a_inward), clockwise, times vel_mult. (ax, ay) is the
        field acceleration at those particles from a real field evaluation (the step's PM solve or direct N-body),
        including the new disk itself, existing particles, softening, force-range screening and periodic images.
        """
        d = self.pos[start:end] - torch.tensor([world_x, world_y], device=self.device)
        if self.periodic:
            size = torch.tensor([self.universe_w, self.universe_h], device=self.device, dtype=d.dtype)
            d -= size * torch.round(d / size)        # minimum image: the disk may straddle an edge
        r = d.norm(dim=1).clamp_(min=1e-6)
        a_inward = -(ax * d[:, 0] + ay * d[:, 1]) / r
        speed = torch.sqrt((a_inward * r).clamp_(min=0.0)) * vel_mult
        # Clockwise tangent (sin(theta), -cos(theta)) = (dy, -dx) / r
        self.vel[start:end, 0] = d[:, 1] / r * speed
        self.vel[start:end, 1] = -d[:, 0] / r * speed

    def add_galaxy(self, world_x, world_y, world_radius, count=5000, vel_mult=0.0):
        count = int(count)
        capacity = self.capacity()
        if count <= 0 or self.num_active >= capacity:
            return

        count = min(count, capacity - self.num_active)
        
        if self.uniform_spawn:
            r = torch.sqrt(torch.rand(count, device=self.device)) * world_radius
        else:
            r = torch.rand(count, device=self.device) * world_radius
            
        theta = torch.rand(count, device=self.device) * 2 * math.pi
        
        new_pos = torch.zeros((count, 2), device=self.device)
        new_pos[:, 0] = world_x + r * torch.cos(theta)
        new_pos[:, 1] = world_y + r * torch.sin(theta)
        
        if self.potential_type == 'gravity':
            new_vel = torch.zeros((count, 2), device=self.device)  # set from the measured field below
        else:
            # Thermal velocities; vel_mult = 1 (the slider default) gives the same spread as before
            new_vel = (torch.rand((count, 2), device=self.device) - 0.5) * THERMAL_SPEED * vel_mult
        if self.boundary_mode == 'open':
            # Spawns outside the physics mesh would be deleted on the next step anyway
            inside = self._inside(new_pos)
            new_pos, new_vel = new_pos[inside], new_vel[inside]
            count = len(new_pos)
            if count == 0:
                return

        start, end = self.num_active, self.num_active + count
        self.pos[start:end] = new_pos
        self.vel[start:end] = new_vel
        self.masses[start:end] = self._new_masses(count)
        self.num_active += count
        if self.periodic:
            self.kernels.wrap(self.pos[start:end], count, self.universe_w, self.universe_h)
        self._field_dirty = True
        if self.potential_type == 'gravity' and vel_mult != 0.0:
            self._pending_orbits.append((start, end, world_x, world_y, vel_mult))

    def remove_particles(self, world_x, world_y, world_radius):
        if self.num_active == 0:
            return
        self._flush_pending_orbits()   # compaction below would invalidate their index ranges
        self._field_dirty = True
        uni_pos = torch.tensor([world_x, world_y], dtype=torch.float32, device=self.device)
        d = self.pos[:self.num_active] - uni_pos
        if self.periodic:
            size = const_tensor((float(self.universe_w), float(self.universe_h)), d.device, d.dtype)
            d -= size * torch.round(d / size)        # minimum image: the brush may straddle an edge
        distances = torch.norm(d, dim=1)
        keep_mask = distances > world_radius
        new_count = keep_mask.sum().item()
        
        self.pos[:new_count] = self.pos[:self.num_active][keep_mask]
        self.vel[:new_count] = self.vel[:self.num_active][keep_mask]
        self.masses[:new_count] = self.masses[:self.num_active][keep_mask]
        self.num_active = new_count

    def load_particles_from_array(self, np_coords):
        """
        Seeds particle coordinates from a NumPy array.
        """
        if np_coords is None:
            return
            
        total_spawn = len(np_coords)
        capacity = self.capacity()
        if total_spawn == 0 or self.num_active >= capacity:
            return

        available_slots = capacity - self.num_active
        if total_spawn > available_slots:
            np_coords = np_coords[:available_slots]
            total_spawn = len(np_coords)
            
        coords = torch.from_numpy(np_coords).to(self.device)
        if self.boundary_mode == 'open':
            coords = coords[self._inside(coords)]
            total_spawn = len(coords)
            if total_spawn == 0:
                return
        start, end = self.num_active, self.num_active + total_spawn
        self.pos[start:end] = coords
        self.vel[start:end] = 0.0
        self.masses[start:end] = self._new_masses(total_spawn)
        self.num_active += total_spawn
        self._field_dirty = True

    def step(self, dt):
        if self._kernel_dirty_since is not None and time.perf_counter() - self._kernel_dirty_since >= KERNEL_REBUILD_DELAY:
            self.precompute_kernel()
        if self.num_active == 0:
            return

        n = self.num_active
        # 1. Handle Boundaries. Periodic needs nothing here: positions are wrapped right after each drift.
        # Open deletes escaped particles after the drift (step 3).
        if self.boundary_mode == 'reflective_walls':
            p = self.pos[:n]
            v = self.vel[:n]
            size = const_tensor((float(self.universe_w), float(self.universe_h)), self.device, p.dtype)
            hit = (p < 0) | (p > size)
            v.mul_(torch.where(hit, -0.5, 1.0))     # bounce with half restitution
            p.clamp_(min=0.0)
            torch.minimum(p, size, out=p)

        # Uniform background gravity, folded into the kick of either solver
        uniform = (0.0, 0.0)
        if self.uniform_gravity_mag > 0:
            uniform = self.pot_manager.get_uniform_gravity_forces(self.uniform_gravity_mag, self.uniform_gravity_angle)

        # 2. Run Selected Physics Solver
        params = self._physics_params()
        if self.solver_type == 'nbody':
            ax, ay = self.nbody_solver.solve(
                self.pos[:n],
                self.masses[:n],
                self.potential_type,
                params,
                self.periodic, self.universe_w, self.universe_h
            )
            if self._pending_orbits:
                self._apply_pending_orbits(lambda s, e: (ax[s:e], ay[s:e]))

            # Euler integration for velocity and positions (exponential friction decay never flips sign)
            damping = math.exp(-self.friction * dt)
            self.vel[:n, 0] = (self.vel[:n, 0] + (ax + uniform[0]) * dt) * damping
            self.vel[:n, 1] = (self.vel[:n, 1] + (ay + uniform[1]) * dt) * damping
            self.pos[:n].add_(self.vel[:n], alpha=dt)
            if self.periodic:
                self.kernels.wrap(self.pos, n, self.universe_w, self.universe_h)

        elif self.solver_type == 'pm':
            # Sort on schedule, but never while spawned disks still wait for their orbit speeds (index ranges)
            if self.sort_interval > 0 and self.step_count % self.sort_interval == 0 and not self._pending_orbits:
                self.sort_particles()
            recompute = (self._field_dirty or self.field_interval <= 1
                         or self.step_count % self.field_interval == 0)
            hook = None
            if self._pending_orbits:
                hook = lambda solver: self._apply_pending_orbits(
                    lambda s, e: solver.sample_forces(self.pos[s:e], self.universe_w, self.universe_h))
            self.pm_solver.solve(
                self.pos,
                self.vel,
                self.masses,
                n,
                self.potential_type,
                params,
                dt,
                self.friction,
                self.pressure_model,
                self.pressure_strength,
                self.gamma,
                self.periodic,
                self.universe_w,
                self.universe_h,
                uniform_accel=uniform,
                recompute_field=recompute,
                before_gather=hook
            )
            self._field_dirty = False
        self.step_count += 1

        # 3. Open boundary: particles that drifted out of the physics mesh are gone
        if self.boundary_mode == 'open':
            self._drop_outside()


# View toggles, colour ranges, band-pass and theme live in engine.view (a ViewState); engine.<name> forwards to it
for _name in VIEW_FIELDS:
    setattr(SimulationEngine, _name, _forward_to_view(_name))
