"""
Screen-space overlays drawn on top of the frame: the brush ring and the "show panel" arrow.
"""
import math

import taichi as ti


@ti.kernel
def init_arrow(arrow_verts: ti.template()):
    arrow_verts[0] = [0.016, 0.985]
    arrow_verts[1] = [0.016, 0.965]
    arrow_verts[2] = [0.006, 0.975]
    arrow_verts[3] = [0.028, 0.985]
    arrow_verts[4] = [0.028, 0.965]
    arrow_verts[5] = [0.018, 0.975]


@ti.kernel
def update_brush_ring(ring_pos_field: ti.template(), center_x: ti.f32, center_y: ti.f32, radius_x: ti.f32, radius_y: ti.f32):
    for i in range(ring_pos_field.shape[0]):
        angle = float(i) / ring_pos_field.shape[0] * 2.0 * math.pi
        ring_pos_field[i] = [center_x + ti.cos(angle) * radius_x, center_y + ti.sin(angle) * radius_y]


class BrushOverlay:
    """The fields are created here (after ti.init), never at import."""

    RING_POINTS = 64

    def __init__(self):
        self.ring_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=self.RING_POINTS)
        self.arrow_verts = ti.Vector.field(2, dtype=ti.f32, shape=6)
        init_arrow(self.arrow_verts)

    def draw_ring(self, canvas, mouse_norm, radius_px, width, height, erasing):
        """Brush outline around the cursor: red while erasing, green otherwise. radius_px is in screen pixels."""
        update_brush_ring(self.ring_pos_field, mouse_norm[0], mouse_norm[1], radius_px / float(width), radius_px / float(height))
        if erasing:
            canvas.circles(self.ring_pos_field, radius=0.001, color=(1.0, 0.2, 0.2))
        else:
            canvas.circles(self.ring_pos_field, radius=0.001, color=(0.2, 0.8, 0.6))

    def draw_arrow(self, canvas):
        """Top-left marker shown while the panel is hidden (clicking it reopens the panel)."""
        canvas.triangles(self.arrow_verts, color=(0.8, 0.8, 0.8))
