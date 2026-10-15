import math
import glob
from particle_mesh.ui.image_loader import browse_custom_image, load_image_particles

IMAGE_PATTERNS = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')


class SimulationGUI:
    def __init__(self, engine, camera, config, palettes):
        self.engine = engine
        self.palettes = palettes  # colour themes (render.colormap.Palettes)
        self.camera = camera
        self.config = config

        # UI sizing configurations
        self.ui_w = config["display"].get("ui_width", 0.15)  # Panel width fraction

        # State variables for log sliders
        self.log_G_val = math.log10(self.engine.G)
        self.log_time_val = config["sliders"]["time_scale_log"][2]
        self.friction_val = self.engine.friction
        self.softening_val = self.engine.softening_length

        # Image spawner variables. The folder is scanned once (and on "Rescan"), not every frame.
        self.custom_image_path = None
        self.custom_image_name = None
        self.image_files = self._scan_images()

    @staticmethod
    def _scan_images():
        files = []
        for pattern in IMAGE_PATTERNS:
            files.extend(glob.glob(pattern))
        return sorted(set(files))

    def _range(self, name, fallback):
        """[min, max] of a slider from config.json (fallback when the entry is missing)."""
        spec = self.config["sliders"].get(name)
        return (spec[0], spec[1]) if spec else fallback

    def _spawn_image(self, path, spawn_density):
        rcs = self.config["simulation"]["RENDER_CELL_SIZE"]
        np_coords = load_image_particles(
            path,
            self.camera.cam_x, self.camera.cam_y, self.camera.zoom,
            spawn_density, self.camera.width, self.camera.height,
            int(self.camera.width / rcs), int(self.camera.height / rcs), rcs,
            self.engine.universe_w, self.engine.universe_h
        )
        if np_coords is not None:
            self.engine.load_particles_from_array(np_coords)

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
            self.engine.set_potential('gravity')
        if gui.button("Force: Lennard-Jones"):
            self.engine.set_potential('lennard_jones')
        if gui.button("Force: Mutual Springs"):
            self.engine.set_potential('harmonic')
        if gui.button("Force: Coulomb"):
            self.engine.set_potential('coulomb')  # randomises +/-1 charges
            
        # --- Gas Pressure (PM Mode only) ---
        if self.engine.solver_type == 'pm':
            gui.text("")
            gui.text("--- GAS PRESSURE ---")
            gui.text(f"Selected: {self.engine.pressure_model.upper()}")
            if gui.button("Gas: Cold Dust (None)"):
                self.engine.set_pressure_model('none')
            if gui.button("Gas: Isothermal (Linear)"):
                self.engine.set_pressure_model('isothermal')
            if gui.button("Gas: Polytropic (Adiabatic)"):
                self.engine.set_pressure_model('polytropic')
                
        # --- Visual Theme Theme ---
        gui.text("")
        gui.text("--- VISUALS ---")
        gui.text(f"Theme: {self.palettes.names[self.engine.color_theme]}")
        if gui.button("Next Color Theme"):
            self.engine.color_theme = self.palettes.cycle(self.engine.color_theme)
            
        # --- Advanced View Settings (PM Mode only) ---
        if self.engine.solver_type == 'pm':
            gui.text("")
            gui.text("--- VIEW SETTINGS ---")
            self.engine.pot_min_log = gui.slider_float("Potential Min (Log)", self.engine.pot_min_log, *self._range("potential_min_log", (-5.0, 15.0)))
            self.engine.pot_max_log = gui.slider_float("Potential Max (Log)", self.engine.pot_max_log, *self._range("potential_max_log", (-5.0, 15.0)))
            self.engine.field_min_log = gui.slider_float("Field Min (Log)", self.engine.field_min_log, *self._range("field_min_log", (-5.0, 15.0)))
            self.engine.field_max_log = gui.slider_float("Field Max (Log)", self.engine.field_max_log, *self._range("field_max_log", (-5.0, 15.0)))

        # --- Engine Modifiers ---
        gui.text("")
        gui.text("--- ENGINE MODIFIERS ---")
        
        # Universal Force Range (Yukawa screening)
        old_fr = self.engine.gravity_range
        self.engine.gravity_range = gui.slider_float("Force Range", self.engine.gravity_range, self.config["sliders"]["gravity_range"][0], self.config["sliders"]["gravity_range"][1])
        if old_fr != self.engine.gravity_range:
            self.engine.request_kernel_rebuild()
            
        # Gas pressure scaling (PM Mode only)
        if self.engine.solver_type == 'pm':
            self.engine.pressure_strength = gui.slider_float("Gas Pressure P", self.engine.pressure_strength, self.config["sliders"]["pressure"][0], self.config["sliders"]["pressure"][1])
            if self.engine.pressure_model == 'polytropic':
                self.engine.gamma = gui.slider_float("Poly Index (Gamma)", self.engine.gamma, self.config["sliders"]["polytropic_gamma"][0], self.config["sliders"]["polytropic_gamma"][1])
            
        # Boundary selector
        bound_text = "SOLID WALL BOUNCE" if self.engine.boundary_mode == 'reflective_walls' else "UNIVERSE WRAPPED"
        if gui.button(f"Boundary: {bound_text}"):
            self.engine.toggle_boundary()
                
        # Friction (damping)
        self.friction_val = gui.slider_float("Friction (Gas Cooling)", self.friction_val, self.config["sliders"]["friction"][0], self.config["sliders"]["friction"][1])
        self.engine.friction = self.friction_val
        
        # Force Softening
        # Softening length in world units: Gaussian filter in PM mode, Plummer epsilon in N-body mode
        self.softening_val = gui.slider_float("Force Softening Length", self.softening_val, self.config["sliders"]["softening_length"][0], self.config["sliders"]["softening_length"][1])
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
                self.engine.request_kernel_rebuild()
            val_label = "Active ke" if self.engine.potential_type == 'coulomb' else "Active G"
            gui.text(f"{val_label}: {self.engine.G:.4f}")
            
        elif self.engine.potential_type == 'harmonic':
            old_k = self.engine.spring_k
            self.engine.spring_k = gui.slider_float("Spring K", self.engine.spring_k, self.config["sliders"]["spring_k"][0], self.config["sliders"]["spring_k"][1])
            if old_k != self.engine.spring_k:
                self.engine.request_kernel_rebuild()
                
        elif self.engine.potential_type == 'lennard_jones':
            old_eps = self.engine.epsilon
            old_sig = self.engine.sigma
            self.engine.epsilon = gui.slider_float("LJ Epsilon", self.engine.epsilon, self.config["sliders"]["epsilon"][0], self.config["sliders"]["epsilon"][1])
            self.engine.sigma = gui.slider_float("LJ Sigma", self.engine.sigma, self.config["sliders"]["sigma"][0], self.config["sliders"]["sigma"][1])
            if old_eps != self.engine.epsilon or old_sig != self.engine.sigma:
                self.engine.request_kernel_rebuild()
                
        # --- Uniform Gravity Field ---
        gui.text("")
        gui.text("--- UNIFORM FIELD ---")
        self.engine.uniform_gravity_mag = gui.slider_float("Gravity Mag", self.engine.uniform_gravity_mag, self.config["sliders"]["uniform_gravity_mag"][0], self.config["sliders"]["uniform_gravity_mag"][1])
        self.engine.uniform_gravity_angle = gui.slider_float("Gravity Angle (Deg)", self.engine.uniform_gravity_angle, self.config["sliders"]["uniform_gravity_angle"][0], self.config["sliders"]["uniform_gravity_angle"][1])
        
        # --- Spawner Brush ---
        gui.text("")
        gui.text("--- SPAWNER BRUSH ---")
        interaction_radius = gui.slider_float("Brush Radius", interaction_radius, self.config["sliders"]["brush_radius"][0], self.config["sliders"]["brush_radius"][1])
        # Gravity: multiple of the circular orbit speed (1 = equilibrium disk, negative = counter-rotating).
        # Other potentials: thermal speed multiplier.
        spawn_vel = gui.slider_float("Spawn Velocity (x orbit)", spawn_vel, self.config["sliders"]["spawn_velocity"][0], self.config["sliders"]["spawn_velocity"][1])
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
                self._spawn_image(self.custom_image_path, spawn_density)

        # Images in the working folder (scanned at startup; rescan on demand)
        if gui.button("Rescan Images"):
            self.image_files = self._scan_images()
        for img_name in self.image_files:
            if img_name != self.custom_image_name and gui.button(f"Spawn {img_name}"):
                self._spawn_image(img_name, spawn_density)

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
                
            self.engine.bp_inner = gui.slider_float("High-Pass (Inner)", self.engine.bp_inner, *self._range("bp_inner", (-0.5, 5.0)))
            self.engine.bp_outer = gui.slider_float("Low-Pass (Outer)", self.engine.bp_outer, *self._range("bp_outer", (-0.5, 5.0)))
            self.engine.bp_transition = gui.slider_float("Filter Softness", self.engine.bp_transition, *self._range("bp_transition", (0.001, 1.0)))
            
        # --- Shortcuts Legend ---
        gui.text("")
        gui.text("--- CONTROL HOTKEYS ---")
        gui.text("LMB: Spawn | RMB: Delete")
        gui.text("Space+LMB / MMB Drag: Pan View")
        gui.text("Z+LMB Drag or Up/Down: Zoom")
        gui.text("Z+RMB Drag: Brush Size")
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
