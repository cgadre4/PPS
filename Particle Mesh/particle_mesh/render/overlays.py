"""
Screen-space overlays drawn on top of the frame: the brush ring and the "show panel" arrow.

Vertices live in persistent host buffers handed straight to GGUI (render/ggui_present), so drawing launches no Taichi
kernel. Canvas.circles/triangles on a field run a copy kernel into a fresh NumPy buffer on every call, and on Vulkan
each one waits for the queue (~1.3 ms of the windowed frame at 1M particles on the B390).
"""
import math

import numpy as np
import taichi as ti

from particle_mesh.render import ggui_present

ARROW_VERTS = [(0.016, 0.985), (0.016, 0.965), (0.006, 0.975),
               (0.028, 0.985), (0.028, 0.965), (0.018, 0.975)]
RING_COLOR = (0.2, 0.8, 0.6)
ERASE_COLOR = (1.0, 0.2, 0.2)
ARROW_COLOR = (0.8, 0.8, 0.8)


class BrushOverlay:
    """Host vertex buffers; Taichi fields only for the public-API fallback, created on first use (after ti.init)."""

    RING_POINTS = 64
    RING_DOT_RADIUS = 0.001

    def __init__(self):
        self.ring_vbo = ggui_present.host_vertex_buffer(self.RING_POINTS)
        self.arrow_vbo = ggui_present.host_vertex_buffer(len(ARROW_VERTS))
        self.arrow_vbo[:, :2] = ARROW_VERTS
        angle = np.arange(self.RING_POINTS, dtype=np.float32) / self.RING_POINTS * np.float32(2.0 * math.pi)
        self._cos, self._sin = np.cos(angle), np.sin(angle)
        self._ring_key = None
        self._fallback_fields = {}

    def draw_ring(self, canvas, mouse_norm, radius_px, width, height, erasing):
        """Brush outline around the cursor: red while erasing, green otherwise. radius_px is in screen pixels."""
        key = (mouse_norm[0], mouse_norm[1], radius_px / float(width), radius_px / float(height))
        if key != self._ring_key:  # vertices change only when the cursor or brush size does
            cx, cy, rx, ry = key
            self.ring_vbo[:, 0] = cx + self._cos * rx
            self.ring_vbo[:, 1] = cy + self._sin * ry
            self._ring_key = key
        color = ERASE_COLOR if erasing else RING_COLOR
        if not ggui_present.draw_host_circles(canvas, self.ring_vbo, self.RING_DOT_RADIUS, color):
            canvas.circles(self._fallback_field("ring", self.ring_vbo), radius=self.RING_DOT_RADIUS, color=color)

    def draw_arrow(self, canvas):
        """Top-left marker shown while the panel is hidden (clicking it reopens the panel)."""
        if not ggui_present.draw_host_triangles(canvas, self.arrow_vbo, ARROW_COLOR):
            canvas.triangles(self._fallback_field("arrow", self.arrow_vbo), color=ARROW_COLOR)

    def _fallback_field(self, name, vbo):
        """A Taichi field holding the buffer's positions, for canvases without the PyCanvas attribute."""
        field = self._fallback_fields.get(name)
        if field is None:
            field = self._fallback_fields[name] = ti.Vector.field(2, dtype=ti.f32, shape=vbo.shape[0])
        field.from_numpy(np.ascontiguousarray(vbo[:, :2]))
        return field
