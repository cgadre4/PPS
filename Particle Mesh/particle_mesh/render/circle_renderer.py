import taichi as ti
import torch

from particle_mesh.settings import SETTINGS
from particle_mesh.runtime.device import to_taichi

# Only N-body mode draws circles, and it is capped at NBODY_MAX_PARTICLES, so size the buffers to that
MAX_RENDER_PARTICLES = SETTINGS.circle_buffer_size


class CircleRenderer:
    def __init__(self):
        # GGUI vertex buffers, created here (after ti.init) rather than at import
        self.render_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_RENDER_PARTICLES)
        self.render_color_field = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER_PARTICLES)

    def render(self, canvas, pos_tensor, color_tensor, num_active, cam_x, cam_y, screen_w, screen_h, particle_radius):
        """
        Normalises on-screen particles to [0, 1] screen space on the sim device, uploads them to the GGUI
        vertex buffers and draws circle primitives. screen_w / screen_h are the visible world extent.
        """
        if num_active == 0:
            return
        device = pos_tensor.device
        p = pos_tensor[:num_active]
        u = (p[:, 0] - cam_x) / screen_w
        v = (p[:, 1] - cam_y) / screen_h
        on_screen = torch.nonzero((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).squeeze(1)[:MAX_RENDER_PARTICLES]
        count = on_screen.numel()

        # Unused slots are parked off-screen
        pos_buf = torch.full((MAX_RENDER_PARTICLES, 2), -10.0, device=device)
        col_buf = torch.zeros((MAX_RENDER_PARTICLES, 3), device=device)
        if count > 0:
            pos_buf[:count, 0] = u[on_screen]
            pos_buf[:count, 1] = v[on_screen]
            col_buf[:count] = color_tensor[on_screen] / 255.0

        self.render_pos_field.from_torch(to_taichi(pos_buf))
        self.render_color_field.from_torch(to_taichi(col_buf))
        canvas.circles(
            self.render_pos_field,
            radius=particle_radius,
            per_vertex_color=self.render_color_field
        )
