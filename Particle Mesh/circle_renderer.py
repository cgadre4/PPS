import taichi as ti

# --- TAICHI RENDER FIELDS (Zero-Copy Bridge) ---
MAX_RENDER_PARTICLES = 290000  
render_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=MAX_RENDER_PARTICLES)
render_color_field = ti.Vector.field(3, dtype=ti.f32, shape=MAX_RENDER_PARTICLES)

@ti.kernel
def copy_to_taichi_kernel(
    pos_arr: ti.types.ndarray(), 
    color_arr: ti.types.ndarray(), 
    num_particles: ti.i32,
    cam_x: ti.f32,
    cam_y: ti.f32,
    screen_w: ti.f32,
    screen_h: ti.f32
):
    for i in render_pos_field:
        if i < num_particles:
            # Mathematical translation & normalization directly on the GPU shader!
            render_pos_field[i] = [(pos_arr[i, 0] - cam_x) / screen_w, 
                                   (pos_arr[i, 1] - cam_y) / screen_h]
            # Normalize colors instantly without PyTorch memory allocation
            render_color_field[i] = [color_arr[i, 0] / 255.0, 
                                     color_arr[i, 1] / 255.0, 
                                     color_arr[i, 2] / 255.0]
        else:
            # Teleport inactive particles off-screen
            render_pos_field[i] = [-10.0, -10.0]

import torch

class CircleRenderer:
    def __init__(self):
        pass

    def render(self, canvas, pos_tensor, color_tensor, num_active, cam_x, cam_y, screen_w, screen_h, particle_radius):
        """
        Runs the zero-copy copy kernel to GGUI vertex fields, and draws circle primitives.
        Only renders particles that are on the screen.
        """
        if num_active == 0:
            return
            
        # Get mask of particles within camera viewport in world space (fully on GPU)
        pos_on_screen = (pos_tensor[:num_active, 0] >= cam_x) & (pos_tensor[:num_active, 0] <= cam_x + screen_w) & \
                        (pos_tensor[:num_active, 1] >= cam_y) & (pos_tensor[:num_active, 1] <= cam_y + screen_h)
        
        on_screen_indices = torch.nonzero(pos_on_screen).squeeze(1)
        num_on_screen = len(on_screen_indices)
        num_to_draw = min(num_on_screen, MAX_RENDER_PARTICLES)
        
        if num_to_draw > 0:
            on_screen_indices = on_screen_indices[:num_to_draw]
            pos_to_draw = pos_tensor[on_screen_indices]
            color_to_draw = color_tensor[on_screen_indices]
            
            # Pass PyTorch tensors directly in VRAM to Taichi fields
            copy_to_taichi_kernel(
                pos_to_draw, 
                color_to_draw, 
                num_to_draw, 
                float(cam_x), 
                float(cam_y), 
                float(screen_w), 
                float(screen_h)
            )
        else:
            # Teleport all render vertices off-screen if nothing is visible
            copy_to_taichi_kernel(
                pos_tensor, 
                color_tensor, 
                0, 
                float(cam_x), 
                float(cam_y), 
                float(screen_w), 
                float(screen_h)
            )
        
        canvas.circles(
            render_pos_field, 
            radius=particle_radius, 
            per_vertex_color=render_color_field
        )
