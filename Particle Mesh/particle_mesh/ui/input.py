"""
Keyboard and mouse handling: edge-detected keys, camera pan/zoom, brush resize, panel toggling, and the brush
action (spawn / erase) for this frame. (Taichi 1.7 GGUI reports no mouse-wheel events, so zoom and brush size use
drags and keys.)
"""
import math
from dataclasses import dataclass

import taichi as ti

# Every key/button the app reads
DEFAULT_KEYS = ('z', 'p', 'f', 'r', 'm', ti.ui.SPACE, ti.ui.RIGHT, ti.ui.LEFT, ti.ui.UP, ti.ui.DOWN,
                ti.ui.LMB, ti.ui.RMB, ti.ui.MMB)
# Clickable "show panel" arrow in the top-left corner (normalised screen coordinates)
ARROW_MAX_X = 0.04
ARROW_MIN_Y = 0.95


class KeyEdges:
    """Per-frame key/button state with rising-edge detection (replaces hand-rolled prev_* flags)."""

    def __init__(self, window, keys):
        self.window = window
        self.keys = tuple(keys)
        self.now = dict.fromkeys(self.keys, False)
        self.before = dict(self.now)

    def update(self):
        self.before = self.now
        self.now = {k: bool(self.window.is_pressed(k)) for k in self.keys}

    def held(self, key):
        return self.now[key]

    def pressed(self, key):
        """True only on the frame the key goes down."""
        return self.now[key] and not self.before[key]


@dataclass
class FrameIntent:
    """What the user asked for this frame, after camera and panel handling."""
    mouse_norm: tuple
    world_x: float
    world_y: float
    world_radius: float
    lmb: bool
    rmb: bool
    hovering_ui: bool
    modifying_view: bool    # Z, Space or a pan drag is active: the mouse steers the view, not the brush
    spawn: bool
    erase: bool
    spawn_count: int


class InputController:
    def __init__(self, window, camera, width, height, brush_limits, keys=None):
        """brush_limits: (min, max) brush radius in screen pixels."""
        self.window = window
        self.keys = keys if keys is not None else KeyEdges(window, DEFAULT_KEYS)
        self.camera = camera
        self.width = width
        self.height = height
        self.brush_limits = brush_limits
        # Drag tracking states
        self.prev_pan_active = False
        self.prev_z_lmb = False
        self.prev_z_rmb = False
        self.prev_mouse_norm = (0.0, 0.0)

    def process(self, engine, palettes, controls, ui_w):
        """Reads this frame's input, applies view/panel changes, and returns the brush action as a FrameIntent."""
        keys, camera, window = self.keys, self.camera, self.window
        width, height = self.width, self.height

        # Capture key states
        keys.update()
        z_pressed = keys.held('z')
        space_pressed = keys.held(ti.ui.SPACE)

        # Keyboard Toggles (Edge-Triggered)
        if keys.pressed(ti.ui.RIGHT):
            engine.color_theme = palettes.cycle(engine.color_theme, 1)
        if keys.pressed(ti.ui.LEFT):
            engine.color_theme = palettes.cycle(engine.color_theme, -1)
        if keys.pressed('p') and engine.solver_type == 'pm':
            engine.show_potential = not engine.show_potential
        if keys.pressed('f') and engine.solver_type == 'pm':
            engine.show_field = not engine.show_field
        if keys.pressed('r'):
            camera.reset()
        if keys.pressed('m'):
            controls.ui_open = not controls.ui_open

        # Key events
        mouse_norm = window.get_cursor_pos()
        for e in window.get_events():
            if e.key == ti.ui.ESCAPE:
                window.running = False

        # Keyboard zoom
        if keys.held(ti.ui.UP):
            camera.zoom_around_mouse(mouse_norm[0] * width, mouse_norm[1] * height, 1.02)
        if keys.held(ti.ui.DOWN):
            camera.zoom_around_mouse(mouse_norm[0] * width, mouse_norm[1] * height, 1.0 / 1.02)

        # Mouse Click/Drag events
        lmb_pressed = keys.held(ti.ui.LMB)
        rmb_pressed = keys.held(ti.ui.RMB)
        mmb_pressed = keys.held(ti.ui.MMB)

        # Panning: Space+LMB or MMB drag
        pan_active = (lmb_pressed and space_pressed) or mmb_pressed
        if pan_active and self.prev_pan_active:
            dx = (mouse_norm[0] - self.prev_mouse_norm[0]) * width
            dy = (mouse_norm[1] - self.prev_mouse_norm[1]) * height
            camera.pan(dx, dy)
        self.prev_pan_active = pan_active

        # Zoom drag (Z+LMB), relative to the viewport centre
        z_lmb_active = lmb_pressed and z_pressed
        if z_lmb_active and self.prev_z_lmb:
            dy = mouse_norm[1] - self.prev_mouse_norm[1]
            camera.zoom_around_mouse(width / 2.0, height / 2.0, math.exp(dy * 5.0))
        self.prev_z_lmb = z_lmb_active

        # Brush resize (Z+RMB drag)
        z_rmb_active = rmb_pressed and z_pressed
        if z_rmb_active and self.prev_z_rmb:
            dy = mouse_norm[1] - self.prev_mouse_norm[1]
            radius = controls.radius * math.exp(dy * 5.0)
            controls.radius = max(self.brush_limits[0], min(self.brush_limits[1], radius))
        self.prev_z_rmb = z_rmb_active

        self.prev_mouse_norm = mouse_norm

        # Protect the panel area (or the "show panel" arrow) from brush clicks
        on_arrow = mouse_norm[0] <= ARROW_MAX_X and mouse_norm[1] >= ARROW_MIN_Y
        hovering_ui = mouse_norm[0] <= ui_w if controls.ui_open else on_arrow
        if keys.pressed(ti.ui.LMB) and not controls.ui_open and on_arrow:
            controls.ui_open = True

        # Translate cursor to simulation coordinates
        world_x, world_y = camera.screen_to_world(mouse_norm[0] * width, mouse_norm[1] * height)
        world_radius = controls.radius / camera.zoom

        modifying_view = z_pressed or space_pressed or pan_active
        brush_free = not modifying_view and not hovering_ui
        return FrameIntent(
            mouse_norm=mouse_norm, world_x=world_x, world_y=world_y, world_radius=world_radius,
            lmb=lmb_pressed, rmb=rmb_pressed, hovering_ui=hovering_ui, modifying_view=modifying_view,
            spawn=lmb_pressed and brush_free, erase=rmb_pressed and brush_free,
            spawn_count=max(1, int(math.pi * (controls.radius ** 2) * controls.spawn_density)),
        )
