"""
Keyboard and mouse handling: edge-detected keys, camera pan/zoom, brush resize, panel toggling, and the brush
action (spawn / erase) for this frame. Zoom is the mouse wheel (towards the cursor; Taichi 1.7 GGUI has no wheel
events, so ui/wheel.py reads them from the Windows message loop), Ctrl+left drag, or the up/down arrows; brush size
is Ctrl+right drag.
"""
import math
import time
from dataclasses import dataclass

import taichi as ti

# Every key/button the app reads
DEFAULT_KEYS = (ti.ui.CTRL, 'p', 'f', 'r', 'm', ti.ui.SPACE, ti.ui.RIGHT, ti.ui.LEFT, ti.ui.UP, ti.ui.DOWN,
                ti.ui.LMB, ti.ui.RMB, ti.ui.MMB)
# Drag sensitivity: zoom / brush radius scale by exp(DRAG_RATE * vertical drag in window heights)
DRAG_RATE = 5.0
# Up/Down arrow zoom speed, ln(zoom factor) per second: 1.02x per frame at 120 FPS, now independent of the frame rate
KEY_ZOOM_RATE = 120.0 * math.log(1.02)
# Longest frame time the key zoom integrates over (a stalled frame must not jump the zoom)
MAX_KEY_DT = 0.1
# Zoom factor per wheel notch, towards the cursor
WHEEL_ZOOM_STEP = 1.2
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
    modifying_view: bool    # Ctrl, Space or a pan drag is active: the mouse steers the view, not the brush
    spawn: bool
    erase: bool
    spawn_count: int


class InputController:
    def __init__(self, window, camera, width, height, brush_limits, keys=None, wheel=None, clock=time.perf_counter):
        """brush_limits: (min, max) brush radius in screen pixels. wheel: callable returning the wheel notches since
        its last call (ui.wheel.WheelHook.take), or None. clock: seconds, for the frame-rate independent key zoom."""
        self.window = window
        self.keys = keys if keys is not None else KeyEdges(window, DEFAULT_KEYS)
        self.camera = camera
        self.width = width
        self.height = height
        self.brush_limits = brush_limits
        # Drag tracking states
        self.prev_pan_active = False
        self.prev_zoom_drag = False
        self.prev_brush_drag = False
        self.zoom_anchor = (0.5, 0.5)     # normalised cursor position where the zoom drag started
        self.prev_mouse_norm = (0.0, 0.0)
        self.wheel = wheel
        self.clock = clock
        self._last_time = None

    def _frame_dt(self):
        """Seconds since the previous frame (1/120 on the first one), at most MAX_KEY_DT."""
        now = self.clock()
        dt = 1.0 / 120.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        return min(max(dt, 0.0), MAX_KEY_DT)

    def process(self, engine, palettes, controls, ui_w):
        """Reads this frame's input, applies view/panel changes, and returns the brush action as a FrameIntent."""
        keys, camera, window = self.keys, self.camera, self.window
        width, height = self.width, self.height

        # Capture key states
        keys.update()
        ctrl_pressed = keys.held(ti.ui.CTRL)
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

        # Keyboard zoom about the cursor, at a fixed rate per second
        dt = self._frame_dt()
        key_zoom = keys.held(ti.ui.UP) - keys.held(ti.ui.DOWN)
        if key_zoom:
            camera.zoom_around_mouse(mouse_norm[0] * width, mouse_norm[1] * height,
                                     math.exp(key_zoom * KEY_ZOOM_RATE * dt))

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

        # Zoom drag (Ctrl+LMB): drag up zooms in, anchored at the point where the drag started
        zoom_drag = lmb_pressed and ctrl_pressed
        if zoom_drag and not self.prev_zoom_drag:
            self.zoom_anchor = mouse_norm
        elif zoom_drag:
            dy = mouse_norm[1] - self.prev_mouse_norm[1]
            camera.zoom_around_mouse(self.zoom_anchor[0] * width, self.zoom_anchor[1] * height,
                                     math.exp(dy * DRAG_RATE))
        self.prev_zoom_drag = zoom_drag

        # Brush resize (Ctrl+RMB drag): drag up grows the brush
        brush_drag = rmb_pressed and ctrl_pressed
        if brush_drag and self.prev_brush_drag:
            dy = mouse_norm[1] - self.prev_mouse_norm[1]
            radius = controls.radius * math.exp(dy * DRAG_RATE)
            controls.radius = max(self.brush_limits[0], min(self.brush_limits[1], radius))
        self.prev_brush_drag = brush_drag

        self.prev_mouse_norm = mouse_norm

        # Protect the panel area (or the "show panel" arrow) from brush clicks
        on_arrow = mouse_norm[0] <= ARROW_MAX_X and mouse_norm[1] >= ARROW_MIN_Y
        hovering_ui = mouse_norm[0] <= ui_w if controls.ui_open else on_arrow
        if keys.pressed(ti.ui.LMB) and not controls.ui_open and on_arrow:
            controls.ui_open = True

        # Wheel zoom towards the cursor; over the panel the wheel belongs to ImGui (scrolling)
        notches = self.wheel() if self.wheel is not None else 0.0
        if notches and not hovering_ui:
            camera.zoom_around_mouse(mouse_norm[0] * width, mouse_norm[1] * height, WHEEL_ZOOM_STEP ** notches)

        # Translate cursor to simulation coordinates
        world_x, world_y = camera.screen_to_world(mouse_norm[0] * width, mouse_norm[1] * height)
        world_radius = controls.radius / camera.zoom

        modifying_view = ctrl_pressed or space_pressed or pan_active
        brush_free = not modifying_view and not hovering_ui
        return FrameIntent(
            mouse_norm=mouse_norm, world_x=world_x, world_y=world_y, world_radius=world_radius,
            lmb=lmb_pressed, rmb=rmb_pressed, hovering_ui=hovering_ui, modifying_view=modifying_view,
            spawn=lmb_pressed and brush_free, erase=rmb_pressed and brush_free,
            spawn_count=max(1, int(math.pi * (controls.radius ** 2) * controls.spawn_density)),
        )
