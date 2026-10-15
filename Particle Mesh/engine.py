import os
import torch
import math
import json
from potentials import PotentialsManager
from nbody_solver import DirectNBodySolver
from pm_solver import ParticleMeshSolver

# --- Load Configuration ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

WIDTH = CONFIG["display"]["WIDTH"]
HEIGHT = CONFIG["display"]["HEIGHT"]
FPS_TARGET = CONFIG["display"]["FPS_TARGET"]

PHYSICS_GRID_W = CONFIG["simulation"].get("PHYSICS_GRID_W", 7680)
PHYSICS_GRID_H = CONFIG["simulation"].get("PHYSICS_GRID_H", 7680)
PHYSICS_CELL_SIZE = CONFIG["simulation"]["PHYSICS_CELL_SIZE"]
MAX_PARTICLES = CONFIG["simulation"]["MAX_PARTICLES"]

UNIVERSE_W = int(PHYSICS_GRID_W * PHYSICS_CELL_SIZE)
UNIVERSE_H = int(PHYSICS_GRID_H * PHYSICS_CELL_SIZE)
GRID_W = PHYSICS_GRID_W
GRID_H = PHYSICS_GRID_H

def _get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

DEVICE = _get_torch_device()

class SimulationEngine:
    def __init__(self, initial_particles=None):
        self.device = DEVICE
        self.width = WIDTH
        self.height = HEIGHT
        self.universe_w = UNIVERSE_W
        self.universe_h = UNIVERSE_H
        
        # Solver & Mode Settings
        self.solver_type = CONFIG["simulation"].get("SOLVER", "pm").lower()
        self.potential_type = 'gravity' # 'gravity', 'lennard_jones', 'harmonic'
        self.pressure_model = 'none' # 'none', 'isothermal', 'polytropic'
        self.boundary_mode = 'wrap_delete' # 'wrap_delete', 'reflective_walls'
        
        # Load custom sliders from config
        sliders = CONFIG.get("sliders", {})
        
        # Physics Parameters
        self.G = sliders.get("gravity", [0.01, 100.0, 0.01])[2]
        self.gravity_range = sliders.get("gravity_range", [1.0, 8192.0, 8192.0])[2] # Screen range
        self.friction = sliders.get("friction", [0.0, 5.0, 0.2])[2] # Friction damping
        
        self.pressure_strength = sliders.get("pressure", [0.01, 10000.0, 0.0])[2]
        self.gamma = sliders.get("polytropic_gamma", [0.0, 3.0, 1.0])[2]
        
        self.spring_k = sliders.get("spring_k", [0.001, 10.0, 0.1])[2]
        self.epsilon = sliders.get("epsilon", [1.0, 5000.0, 1000.0])[2]
        self.sigma = sliders.get("sigma", [1.0, 200.0, 40.0])[2]
        self.softening_length = sliders.get("softening_length", [0.1, 10.0, 1.5])[2]
        
        self.uniform_gravity_mag = sliders.get("uniform_gravity_mag", [0.0, 2000.0, 0.0])[2]
        self.uniform_gravity_angle = sliders.get("uniform_gravity_angle", [0.0, 360.0, 270.0])[2]
        
        self.uniform_spawn = False
        
        # Rendering & Visualization States
        self.show_fft = CONFIG["simulation"].get("show_fft", True)
        self.use_bandpass_render = False
        self.show_potential = False
        self.show_field = False
        
        self.pot_min_log = sliders.get("potential_min_log", [-5.0, 10.0, 2.0])[2]
        self.pot_max_log = sliders.get("potential_max_log", [-5.0, 15.0, 6.0])[2]
        self.field_min_log = sliders.get("field_min_log", [-5.0, 10.0, 0.0])[2]
        self.field_max_log = sliders.get("field_max_log", [-5.0, 10.0, 3.0])[2]
        
        self.bp_inner = sliders.get("bp_inner", [-0.5, 5.0, -0.5])[2]
        self.bp_outer = sliders.get("bp_outer", [-0.5, 5.0, 5.0])[2]
        self.bp_transition = sliders.get("bp_transition", [0.001, 1.0, 0.1])[2]
        
        # Particle State Tensors
        if initial_particles is None:
            initial_particles = CONFIG["simulation"].get("initial_particles", 10000 if self.solver_type == 'nbody' else 1000000)
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
        
        # Initialize helper managers and solvers
        self.pot_manager = PotentialsManager(GRID_W, GRID_H, PHYSICS_CELL_SIZE, self.device)
        self.nbody_solver = DirectNBodySolver(self.device)
        self.pm_solver = ParticleMeshSolver(GRID_W, GRID_H, PHYSICS_CELL_SIZE, self.device, self.pot_manager)

    def precompute_kernel(self):
        """
        Triggers potentials precomputation inside solvers on parameter change.
        """
        params = {
            'G': self.G,
            'force_range': self.gravity_range,
            'spring_k': self.spring_k,
            'epsilon': self.epsilon,
            'sigma': self.sigma
        }
        self.pm_solver.update_kernel(self.potential_type, params)

    def add_galaxy(self, world_x, world_y, world_radius, count=5000, vel_mult=0.0):
        count = int(count)
        if count <= 0 or self.num_active >= MAX_PARTICLES:
            return
            
        count = min(count, MAX_PARTICLES - self.num_active)
        
        if self.uniform_spawn:
            r = torch.sqrt(torch.rand(count, device=self.device)) * world_radius
        else:
            r = torch.rand(count, device=self.device) * world_radius
            
        theta = torch.rand(count, device=self.device) * 2 * math.pi
        
        new_pos = torch.zeros((count, 2), device=self.device)
        new_pos[:, 0] = world_x + r * torch.cos(theta)
        new_pos[:, 1] = world_y + r * torch.sin(theta)
        
        if self.potential_type == 'gravity':
            orbital_speed = torch.sqrt(abs(self.G) * 100.0 / (r + 10.0))
            new_vel = torch.zeros((count, 2), device=self.device)
            new_vel[:, 0] = torch.sin(theta) * orbital_speed * vel_mult
            new_vel[:, 1] = -torch.cos(theta) * orbital_speed * vel_mult
        else:
            # Thermal coordinates setup
            new_vel = (torch.rand((count, 2), device=self.device) - 0.5) * 50.0 * vel_mult
            
        start, end = self.num_active, self.num_active + count
        self.pos[start:end] = new_pos
        self.vel[start:end] = new_vel
        self.masses[start:end] = 1
        self.num_active += count

    def remove_particles(self, world_x, world_y, world_radius):
        if self.num_active == 0:
            return
        uni_pos = torch.tensor([world_x, world_y], dtype=torch.float32, device=self.device)
        distances = torch.norm(self.pos[:self.num_active] - uni_pos, dim=1)
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
        if total_spawn == 0 or self.num_active >= MAX_PARTICLES:
            return
            
        available_slots = MAX_PARTICLES - self.num_active
        if total_spawn > available_slots:
            np_coords = np_coords[:available_slots]
            total_spawn = len(np_coords)
            
        start, end = self.num_active, self.num_active + total_spawn
        self.pos[start:end] = torch.from_numpy(np_coords).to(self.device)
        self.vel[start:end] = 0.0
        self.masses[start:end] = 1
        self.num_active += total_spawn

    def step(self, dt):
        if self.num_active == 0:
            return

        # 1. Handle Boundaries
        if self.boundary_mode == 'reflective_walls':
            hit_left = self.pos[:self.num_active, 0] < 0
            hit_right = self.pos[:self.num_active, 0] > self.universe_w
            hit_top = self.pos[:self.num_active, 1] < 0
            hit_bottom = self.pos[:self.num_active, 1] > self.universe_h
            
            self.pos[:self.num_active, 0] = torch.where(hit_left, 0.0, self.pos[:self.num_active, 0])
            self.pos[:self.num_active, 0] = torch.where(hit_right, float(self.universe_w), self.pos[:self.num_active, 0])
            self.vel[:self.num_active, 0] = torch.where(hit_left | hit_right, -self.vel[:self.num_active, 0] * 0.5, self.vel[:self.num_active, 0])

            self.pos[:self.num_active, 1] = torch.where(hit_top, 0.0, self.pos[:self.num_active, 1])
            self.pos[:self.num_active, 1] = torch.where(hit_bottom, float(self.universe_h), self.pos[:self.num_active, 1])
            self.vel[:self.num_active, 1] = torch.where(hit_top | hit_bottom, -self.vel[:self.num_active, 1] * 0.5, self.vel[:self.num_active, 1])
        else:
            in_bounds = (self.pos[:self.num_active, 0] >= 0) & (self.pos[:self.num_active, 0] < self.universe_w) & \
                        (self.pos[:self.num_active, 1] >= 0) & (self.pos[:self.num_active, 1] < self.universe_h)
            if not in_bounds.all().item():
                new_count = in_bounds.sum().item()
                self.pos[:new_count] = self.pos[:self.num_active][in_bounds]
                self.vel[:new_count] = self.vel[:self.num_active][in_bounds]
                self.masses[:new_count] = self.masses[:self.num_active][in_bounds]
                self.num_active = new_count

        if self.num_active == 0:
            return

        # 2. Run Selected Physics Solver
        if self.solver_type == 'nbody':
            params = {
                'G': self.G,
                'force_range': self.gravity_range,
                'spring_k': self.spring_k,
                'epsilon': self.epsilon,
                'sigma': self.sigma,
                'softening_length': self.softening_length
            }
            ax, ay = self.nbody_solver.solve(
                self.pos[:self.num_active], 
                self.masses[:self.num_active], 
                self.potential_type, 
                params
            )
            
            # Euler integration for velocity and positions
            self.vel[:self.num_active, 0] = (self.vel[:self.num_active, 0] + ax * dt) * (1.0 - self.friction * dt)
            self.vel[:self.num_active, 1] = (self.vel[:self.num_active, 1] + ay * dt) * (1.0 - self.friction * dt)
            
            self.pos[:self.num_active, 0] += self.vel[:self.num_active, 0] * dt
            self.pos[:self.num_active, 1] += self.vel[:self.num_active, 1] * dt
            
        elif self.solver_type == 'pm':
            params = {
                'G': self.G,
                'force_range': self.gravity_range,
                'spring_k': self.spring_k,
                'epsilon': self.epsilon,
                'sigma': self.sigma,
                'softening_length': self.softening_length
            }
            self.pm_solver.solve(
                self.pos, 
                self.vel, 
                self.masses, 
                self.num_active, 
                self.potential_type, 
                params, 
                dt, 
                self.friction,
                self.pressure_model,
                self.pressure_strength,
                self.gamma
            )

        # 3. Apply Uniform Gravity Field (CUDA Accelerated)
        if self.uniform_gravity_mag > 0:
            ugx, ugy = self.pot_manager.get_uniform_gravity_forces(self.uniform_gravity_mag, self.uniform_gravity_angle)
            self.vel[:self.num_active, 0] += ugx * dt
            self.vel[:self.num_active, 1] += ugy * dt
            
        # Apply fluid dampening cooling (if Lennard-Jones active in N-body mode)
        if self.potential_type == 'lennard_jones' and self.solver_type == 'nbody':
            self.vel[:self.num_active] *= 0.99
