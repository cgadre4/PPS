import os
import taichi as ti
import json
import numpy as np

# --- Load Config for Field Sizes ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

WIDTH = CONFIG["display"]["WIDTH"]
HEIGHT = CONFIG["display"]["HEIGHT"]
RENDER_CELL_SIZE = CONFIG["simulation"]["RENDER_CELL_SIZE"]

VIEW_W = int(WIDTH / RENDER_CELL_SIZE)
VIEW_H = int(HEIGHT / RENDER_CELL_SIZE)

# --- TAICHI NATIVE FIELDS ---
display_pixels = ti.Vector.field(3, dtype=ti.u8, shape=(VIEW_W, VIEW_H))
visual_density = ti.field(dtype=ti.f32, shape=(VIEW_W, VIEW_H))

# --- Vector Arrow Field Allocation ---
PIXELS_PER_ARROW = CONFIG["display"].get("pixels_per_arrow", 32)
NUM_ARROWS_X = VIEW_W // PIXELS_PER_ARROW
NUM_ARROWS_Y = VIEW_H // PIXELS_PER_ARROW
TOTAL_ARROWS = NUM_ARROWS_X * NUM_ARROWS_Y

if TOTAL_ARROWS > 0:
    # Each arrow has 6 vertices (3 lines: shaft, left wing, right wing)
    vector_verts = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_ARROWS * 6)
    vector_colors = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_ARROWS * 6)

@ti.kernel
def clear_visual_density():
    for i, j in visual_density:
        visual_density[i, j] = 0.0

@ti.kernel
def render_scatter_kernel(
    pos: ti.types.ndarray(), num_active: ti.i32, cam_x: ti.f32, cam_y: ti.f32, zoom: ti.f32,
    view_w: ti.i32, view_h: ti.i32, render_cell_size: ti.f32
):
    for p in range(num_active):
        px, py = (pos[p, 0] - cam_x) * zoom, (pos[p, 1] - cam_y) * zoom
        if 0 <= px < view_w * render_cell_size and 0 <= py < view_h * render_cell_size:
            gx, gy = ti.cast(px / render_cell_size, ti.i32), ti.cast(py / render_cell_size, ti.i32)
            if 0 <= gx < view_w and 0 <= gy < view_h:
                ti.atomic_add(visual_density[gx, gy], 1.0)

@ti.kernel
def render_color_kernel(color_map: ti.types.ndarray(), min_log: ti.f32, max_log: ti.f32, scale: ti.f32):
    for i, j in visual_density:
        dens = visual_density[i, j]
        if dens == 0.0:
            display_pixels[i, j] = [ti.cast(0, ti.u8), ti.cast(0, ti.u8), ti.cast(0, ti.u8)]
        else:
            log_dens = ti.log(dens * scale + 0.1) / 2.30258509299
            norm_dens = ti.max(0.0, ti.min((log_dens - min_log) / (max_log - min_log), 1.0))
            
            idx_f = norm_dens * 8.0
            idx_lower = ti.cast(ti.floor(idx_f), ti.i32)
            idx_upper = ti.max(0, ti.min(ti.cast(ti.ceil(idx_f), ti.i32), 8))
            
            weight_c = idx_f - ti.cast(idx_lower, ti.f32)
            
            r = color_map[idx_lower, 0] * (1.0 - weight_c) + color_map[idx_upper, 0] * weight_c
            g = color_map[idx_lower, 1] * (1.0 - weight_c) + color_map[idx_upper, 1] * weight_c
            b = color_map[idx_lower, 2] * (1.0 - weight_c) + color_map[idx_upper, 2] * weight_c
            
            display_pixels[i, j] = [ti.cast(r, ti.u8), ti.cast(g, ti.u8), ti.cast(b, ti.u8)]

@ti.kernel
def overlay_fft_kernel(thumb: ti.types.ndarray(), offset_x: ti.i32, offset_y: ti.i32, thumb_w: ti.i32, thumb_h: ti.i32):
    for i, j in ti.ndrange(thumb_w, thumb_h):
        val = ti.cast(thumb[j, i], ti.u8)
        px = offset_x + i
        py = offset_y + j
        if 0 <= px < VIEW_W and 0 <= py < VIEW_H:
            display_pixels[px, py] = [val, val, val]

@ti.kernel
def update_vector_field_kernel(
    fx_view: ti.types.ndarray(), fy_view: ti.types.ndarray(),
    color_map: ti.types.ndarray(),
    min_log: ti.f32, max_log: ti.f32,
    spacing: ti.i32, view_w: ti.i32, view_h: ti.i32,
    length_scale: ti.f32, head_size: ti.f32, head_angle: ti.f32
):
    num_x = view_w // spacing
    num_y = view_h // spacing
    for i, j in ti.ndrange(num_x, num_y):
        idx = i * num_y + j
        px = i * spacing + spacing // 2
        py = j * spacing + spacing // 2
        
        f_x = fx_view[py, px]
        f_y = fy_view[py, px]
        f_mag = ti.sqrt(f_x**2 + f_y**2)
        
        log_dens = ti.log(f_mag + 0.1) / 2.30258509299
        norm_dens = ti.max(0.0, ti.min((log_dens - min_log) / (max_log - min_log), 1.0))
        
        idx_f = norm_dens * 8.0
        idx_lower = ti.cast(ti.floor(idx_f), ti.i32)
        idx_upper = ti.max(0, ti.min(ti.cast(ti.ceil(idx_f), ti.i32), 8))
        weight_c = idx_f - ti.cast(idx_lower, ti.f32)
        
        r = color_map[idx_lower, 0] * (1.0 - weight_c) + color_map[idx_upper, 0] * weight_c
        g = color_map[idx_lower, 1] * (1.0 - weight_c) + color_map[idx_upper, 1] * weight_c
        b = color_map[idx_lower, 2] * (1.0 - weight_c) + color_map[idx_upper, 2] * weight_c
        c = [r / 255.0, g / 255.0, b / 255.0]
        
        dir_x, dir_y = 0.0, 0.0
        if f_mag > 1e-5:
            dir_x = f_x / f_mag
            dir_y = f_y / f_mag
            
        max_val = ti.exp(max_log * 2.30258509299)
        min_val = ti.exp(min_log * 2.30258509299)
        linear_frac = ti.max(0.0, ti.min((f_mag - min_val) / ti.max(1e-5, max_val - min_val), 1.0))
        length = linear_frac * length_scale * spacing
        
        sx = ti.cast(px, ti.f32) / view_w
        sy = ti.cast(py, ti.f32) / view_h
        
        ex = sx + (dir_x * length) / view_w
        ey = sy + (dir_y * length) / view_h
        
        wing_len = length * head_size
        cos_a = ti.cos(head_angle)
        sin_a = ti.sin(head_angle)
        
        wx1 = ex + (-dir_x * cos_a + dir_y * sin_a) * wing_len / view_w
        wy1 = ey + (-dir_x * sin_a - dir_y * cos_a) * wing_len / view_h
        wx2 = ex + (-dir_x * cos_a - dir_y * sin_a) * wing_len / view_w
        wy2 = ey + ( dir_x * sin_a - dir_y * cos_a) * wing_len / view_h
        
        base_idx = idx * 6
        
        vector_verts[base_idx] = [sx, sy]
        vector_verts[base_idx + 1] = [ex, ey]
        
        vector_verts[base_idx + 2] = [ex, ey]
        vector_verts[base_idx + 3] = [wx1, wy1]
        
        vector_verts[base_idx + 4] = [ex, ey]
        vector_verts[base_idx + 5] = [wx2, wy2]
        
        for k in ti.static(range(6)):
            vector_colors[base_idx + k] = c


class MeshRenderer:
    def __init__(self):
        pass

    def render(self, canvas, pos_tensor, num_active, cam_x, cam_y, zoom, color_map_tensor, min_log, max_log, zoom_intensity_scale):
        """
        Runs density scatter onto the mesh grid and maps values to colormaps, drawing an image texture.
        """
        clear_visual_density()
        
        if num_active > 0:
            render_scatter_kernel(
                pos_tensor, 
                num_active, 
                float(cam_x), 
                float(cam_y), 
                float(zoom), 
                VIEW_W, 
                VIEW_H, 
                float(RENDER_CELL_SIZE)
            )
            
        scale = float(max(1.0, zoom) ** zoom_intensity_scale)
        render_color_kernel(color_map_tensor, float(min_log), float(max_log), scale)
        canvas.set_image(display_pixels)

    def render_grid(self, canvas, view_dens_tensor, color_map_tensor, min_log, max_log, scale):
        """
        Renders a grid-sampled potential or bandpass density map directly.
        """
        visual_density.from_torch(view_dens_tensor.transpose(0, 1).contiguous())
        render_color_kernel(color_map_tensor, float(min_log), float(max_log), float(scale))
        canvas.set_image(display_pixels)

    def overlay_fft(self, fft_thumb_tensor, offset_x, offset_y, thumb_w, thumb_h):
        """
        Overlays the grayscale FFT thumbnail in the viewport corner.
        """
        overlay_fft_kernel(fft_thumb_tensor, offset_x, offset_y, thumb_w, thumb_h)

    def render_vectors(self, canvas, view_fx, view_fy, color_map_tensor, min_log, max_log, length_scale, head_size, head_angle, width):
        """
        Calculates arrow grids and draws line primitives over the viewport.
        """
        if TOTAL_ARROWS <= 0:
            return
            
        update_vector_field_kernel(
            view_fx, view_fy,
            color_map_tensor,
            float(min_log), float(max_log),
            PIXELS_PER_ARROW, VIEW_W, VIEW_H,
            float(length_scale), float(head_size), float(head_angle)
        )
        
        canvas.lines(vector_verts, width=width, per_vertex_color=vector_colors)
