import taichi as ti
import math
import os
import glob
from image_loader import browse_custom_image

class SimulationGUI:
    def __init__(self, engine, camera, config):
        self.engine = engine
        self.camera = camera
        self.config = config
        
        # UI sizing configurations
        self.ui_w = config["display"].get("ui_width", 0.15)  # Panel width fraction
        self.show_fft_panel = config["simulation"].get("show_fft", True)
        
        # State variables for log sliders
        self.log_G_val = math.log10(self.engine.G)
        self.log_time_val = config["sliders"]["time_scale_log"][2]
        self.friction_val = self.engine.friction
        self.softening_val = self.engine.softening_length
        
        # Image spawner variables
        self.custom_image_path = None
        self.custom_image_name = None

    def render(self, gui, fps_display, interaction_radius, spawn_vel, spawn_density):
        """
        Renders the IMGUI panels on the window canvas.
        Returns:
            updated values for: interaction_radius, spawn_vel, spawn_density
        """
        # Define layout height depending on FFT minimap presence
        # Minimap takes some height at the bottom left
        minimap_h_px = int(self.camera.width * self.ui_w)
        panel_height = 1.0 - (minimap_h_px / float(self.camera.height)) if (self.engine.solver_type == 'pm' and self.engine.show_fft) else 1.0
        
        gui.begin("Simulation Controls", 0.0, 0.0, self.ui_w, panel_height)
        
        # --- System Stats ---
        gui.text("--- SYSTEM STATS ---")
        gui.text(f"Solver: {self.engine.solver_type.upper()}")
        gui.text(f"FPS: {fps_display:.1f} / {self.config['display']['FPS_TARGET']:.0f}")
        gui.text(f"Total Stars: {self.engine.num_active} / {self.engine.capacity()}")
        
        gui.text("")
        gui.text("--- ACTIVE POTENTIAL ---")
        gui.text(f"Selected: {self.engine.potential_type.upper()}")
        if gui.button("Force: Gravity"):
            self.engine.potential_type = 'gravity'
            self.engine.masses[:self.engine.num_active] = 1
            self.engine.precompute_kernel()
        if gui.button("Force: Lennard-Jones"):
            self.engine.potential_type = 'lennard_jones'
            self.engine.masses[:self.engine.num_active] = 1
            self.engine.precompute_kernel()
        if gui.button("Force: Mutual Springs"):
            self.engine.potential_type = 'harmonic'
            self.engine.masses[:self.engine.num_active] = 1
            self.engine.precompute_kernel()
        if gui.button("Force: Coulomb"):
            self.engine.potential_type = 'coulomb'
            # Randomize charges in Coulomb mode
            import torch
            self.engine.masses[:self.engine.num_active] = torch.randint(0, 2, (self.engine.num_active,), device=self.engine.device, dtype=torch.int8) * 2 - 1
            self.engine.precompute_kernel()
            
        # --- Gas Pressure (PM Mode only) ---
        if self.engine.solver_type == 'pm':
            gui.text("")
            gui.text("--- GAS PRESSURE ---")
            gui.text(f"Selected: {self.engine.pressure_model.upper()}")
            if gui.button("Gas: Cold Dust (None)"):
                self.engine.pressure_model = 'none'
            if gui.button("Gas: Isothermal (Linear)"):
                self.engine.pressure_model = 'isothermal'
            if gui.button("Gas: Polytropic (Adiabatic)"):
                self.engine.pressure_model = 'polytropic'
                
        # --- Visual Theme Theme ---
        gui.text("")
        gui.text("--- VISUALS ---")
        gui.text(f"Theme: {self.engine.theme_names[self.engine.color_theme]}")
        if gui.button("Next Color Theme"):
            self.engine.color_theme = (self.engine.color_theme + 1) % len(self.engine.theme_names)
            
        # --- Advanced View Settings (PM Mode only) ---
        if self.engine.solver_type == 'pm':
            gui.text("")
            gui.text("--- VIEW SETTINGS ---")
            self.engine.pot_min_log = gui.slider_float("Potential Min (Log)", self.engine.pot_min_log, -5.0, 15.0)
            self.engine.pot_max_log = gui.slider_float("Potential Max (Log)", self.engine.pot_max_log, -5.0, 15.0)
            self.engine.field_min_log = gui.slider_float("Field Min (Log)", self.engine.field_min_log, -5.0, 15.0)
            self.engine.field_max_log = gui.slider_float("Field Max (Log)", self.engine.field_max_log, -5.0, 15.0)

        # --- Engine Modifiers ---
        gui.text("")
        gui.text("--- ENGINE MODIFIERS ---")
        
        # Universal Force Range (Yukawa screening)
        old_fr = self.engine.gravity_range
        self.engine.gravity_range = gui.slider_float("Force Range", self.engine.gravity_range, self.config["sliders"]["gravity_range"][0], self.config["sliders"]["gravity_range"][1])
        if old_fr != self.engine.gravity_range:
            self.engine.precompute_kernel()
            
        # Gas pressure scaling (PM Mode only)
        if self.engine.solver_type == 'pm':
            self.engine.pressure_strength = gui.slider_float("Gas Pressure P", self.engine.pressure_strength, self.config["sliders"]["pressure"][0], self.config["sliders"]["pressure"][1])
            if self.engine.pressure_model == 'polytropic':
                self.engine.gamma = gui.slider_float("Poly Index (Gamma)", self.engine.gamma, self.config["sliders"]["polytropic_gamma"][0], self.config["sliders"]["polytropic_gamma"][1])
            
        # Boundary selector
        bound_text = "SOLID WALL BOUNCE" if self.engine.boundary_mode == 'reflective_walls' else "UNIVERSE WRAPPED"
        if gui.button(f"Boundary: {bound_text}"):
            if self.engine.boundary_mode == 'reflective_walls':
                self.engine.boundary_mode = 'wrap_delete'
            else:
                self.engine.boundary_mode = 'reflective_walls'
                
        # Friction (damping)
        self.friction_val = gui.slider_float("Friction (Gas Cooling)", self.friction_val, self.config["sliders"]["friction"][0], self.config["sliders"]["friction"][1])
        self.engine.friction = self.friction_val
        
        # Force Softening
        self.softening_val = gui.slider_float("Force Softening (Plummer)", self.softening_val, self.config["sliders"]["softening_length"][0], self.config["sliders"]["softening_length"][1])
        self.engine.softening_length = self.softening_val
        
        # Time Scale Log
        self.log_time_val = gui.slider_float("Time Scale (Log)", self.log_time_val, self.config["sliders"]["time_scale_log"][0], self.config["sliders"]["time_scale_log"][1])
        
        # --- Potential Coefficients ---
        gui.text("")
        gui.text("--- POTENTIAL COEFFICIENTS ---")
        if self.engine.potential_type in ('gravity', 'coulomb'):
            label = "Coulomb Const (Log)" if self.engine.potential_type == 'coulomb' else "Gravity (Log)"
            self.log_G_val = gui.slider_float(label, self.log_G_val, math.log10(self.config["sliders"]["gravity"][0]), math.log10(self.config["sliders"]["gravity"][1]))
            new_G = 10 ** self.log_G_val
            if new_G != self.engine.G:
                self.engine.G = new_G
                self.engine.precompute_kernel()
            val_label = "Active ke" if self.engine.potential_type == 'coulomb' else "Active G"
            gui.text(f"{val_label}: {self.engine.G:.4f}")
            
        elif self.engine.potential_type == 'harmonic':
            old_k = self.engine.spring_k
            self.engine.spring_k = gui.slider_float("Spring K", self.engine.spring_k, self.config["sliders"]["spring_k"][0], self.config["sliders"]["spring_k"][1])
            if old_k != self.engine.spring_k:
                self.engine.precompute_kernel()
                
        elif self.engine.potential_type == 'lennard_jones':
            old_eps = self.engine.epsilon
            old_sig = self.engine.sigma
            self.engine.epsilon = gui.slider_float("LJ Epsilon", self.engine.epsilon, self.config["sliders"]["epsilon"][0], self.config["sliders"]["epsilon"][1])
            self.engine.sigma = gui.slider_float("LJ Sigma", self.engine.sigma, self.config["sliders"]["sigma"][0], self.config["sliders"]["sigma"][1])
            if old_eps != self.engine.epsilon or old_sig != self.engine.sigma:
                self.engine.precompute_kernel()
                
        # --- Uniform Gravity Field ---
        gui.text("")
        gui.text("--- UNIFORM FIELD ---")
        self.engine.uniform_gravity_mag = gui.slider_float("Gravity Mag", self.engine.uniform_gravity_mag, self.config["sliders"]["uniform_gravity_mag"][0], self.config["sliders"]["uniform_gravity_mag"][1])
        self.engine.uniform_gravity_angle = gui.slider_float("Gravity Angle (Deg)", self.engine.uniform_gravity_angle, self.config["sliders"]["uniform_gravity_angle"][0], self.config["sliders"]["uniform_gravity_angle"][1])
        
        # --- Spawner Brush ---
        gui.text("")
        gui.text("--- SPAWNER BRUSH ---")
        interaction_radius = gui.slider_float("Brush Radius", interaction_radius, self.config["sliders"]["brush_radius"][0], self.config["sliders"]["brush_radius"][1])
        spawn_vel = gui.slider_float("Spawn Velocity (CW)", spawn_vel, self.config["sliders"]["spawn_velocity"][0], self.config["sliders"]["spawn_velocity"][1])
        spawn_density = gui.slider_float("Density (stars/px^2)", spawn_density, self.config["sliders"]["spawn_density"][0], self.config["sliders"]["spawn_density"][1])
        
        spawn_type_text = "Uniform Disk" if self.engine.uniform_spawn else "Dense Core"
        gui.text(f"Spawn Mode: {spawn_type_text}")
        if gui.button("Toggle Spawn Mode"):
            self.engine.uniform_spawn = not self.engine.uniform_spawn
            
        # --- Astro-Image Loading ---
        gui.text("")
        gui.text("--- LOAD ASTRO-IMAGE ---")
        if gui.button("Browse Image (Explorer)..."):
            path, name = browse_custom_image()
            if path:
                self.custom_image_path = path
                self.custom_image_name = name
                
        if self.custom_image_path and self.custom_image_name:
            if gui.button(f"Spawn {self.custom_image_name}"):
                # Load image coordinates
                from image_loader import load_image_particles
                grid_w = int(self.camera.width / self.config["simulation"]["RENDER_CELL_SIZE"])
                grid_h = int(self.camera.height / self.config["simulation"]["RENDER_CELL_SIZE"])
                np_coords = load_image_particles(
                    self.custom_image_path,
                    self.camera.cam_x, self.camera.cam_y, self.camera.zoom,
                    spawn_density, self.camera.width, self.camera.height,
                    grid_w, grid_h, self.config["simulation"]["RENDER_CELL_SIZE"],
                    self.engine.universe_w, self.engine.universe_h
                )
                if np_coords is not None:
                    self.engine.load_particles_from_array(np_coords)
                    
        # Scan folder for any local images
        img_list = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'):
            img_list.extend(glob.glob(ext))
        img_list = [name for name in img_list if name != self.custom_image_name]
        
        for img_name in img_list:
            if gui.button(f"Spawn {img_name}"):
                from image_loader import load_image_particles
                grid_w = int(self.camera.width / self.config["simulation"]["RENDER_CELL_SIZE"])
                grid_h = int(self.camera.height / self.config["simulation"]["RENDER_CELL_SIZE"])
                np_coords = load_image_particles(
                    img_name,
                    self.camera.cam_x, self.camera.cam_y, self.camera.zoom,
                    spawn_density, self.camera.width, self.camera.height,
                    grid_w, grid_h, self.config["simulation"]["RENDER_CELL_SIZE"],
                    self.engine.universe_w, self.engine.universe_h
                )
                if np_coords is not None:
                    self.engine.load_particles_from_array(np_coords)

        # --- Bandpass & FFT Minimap (PM Mode only) ---
        if self.engine.solver_type == 'pm':
            gui.text("")
            gui.text("--- BANDPASS & FFT RADAR ---")
            
            show_fft_text = "Compute FFT Minimap: ON" if self.engine.show_fft else "Compute FFT Minimap: OFF"
            if gui.button(show_fft_text):
                self.engine.show_fft = not self.engine.show_fft
                
            bp_render_text = "Apply Filter to Main Render: ON" if self.engine.use_bandpass_render else "Apply Filter to Main Render: OFF"
            if gui.button(bp_render_text):
                self.engine.use_bandpass_render = not self.engine.use_bandpass_render
                
            self.engine.bp_inner = gui.slider_float("High-Pass (Inner)", self.engine.bp_inner, 0.0, 5.0)
            self.engine.bp_outer = gui.slider_float("Low-Pass (Outer)", self.engine.bp_outer, 0.0, 5.0)
            self.engine.bp_transition = gui.slider_float("Filter Softness", self.engine.bp_transition, 0.001, 1.0)
            
        # --- Shortcuts Legend ---
        gui.text("")
        gui.text("--- CONTROL HOTKEYS ---")
        gui.text("LMB: Spawn | RMB: Delete")
        gui.text("Space+LMB Drag: Pan View")
        gui.text("Z+LMB Drag (or Z+Scroll): Zoom")
        gui.text("Z+RMB Drag (or Scroll): Brush Size")
        if self.engine.solver_type == 'pm':
            gui.text("P: Toggle Potential View")
            gui.text("F: Toggle Vector Field")
        gui.text("R: Reset View")
        gui.text("Left/Right Arrow: Color Theme")
        
        gui.text("")
        gui.text("--- DEBUG READOUT ---")
        gui.text(f"Zoom Level: {self.camera.zoom:.2f}x")
        
        if gui.button("Reset View"):
            self.camera.reset()
            
        gui.end()
        
        return interaction_radius, spawn_vel, spawn_density
