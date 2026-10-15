import os
import math
import taichi as ti
import json
import torch

from kernels import to_taichi

# --- Load Config for Field Sizes ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

WIDTH = CONFIG["display"]["WIDTH"]
HEIGHT = CONFIG["display"]["HEIGHT"]
RENDER_CELL_SIZE = CONFIG["simulation"]["RENDER_CELL_SIZE"]

VIEW_W = int(WIDTH / RENDER_CELL_SIZE)
VIEW_H = int(HEIGHT / RENDER_CELL_SIZE)

# --- TAICHI NATIVE FIELDS (GGUI presentation only; all image math happens in PyTorch on the sim device) ---
display_pixels = ti.Vector.field(3, dtype=ti.u8, shape=(VIEW_W, VIEW_H))

# --- Vector Arrow Field Allocation ---
PIXELS_PER_ARROW = CONFIG["display"].get("pixels_per_arrow", 32)  # spacing in render cells
NUM_ARROWS_X = VIEW_W // PIXELS_PER_ARROW
NUM_ARROWS_Y = VIEW_H // PIXELS_PER_ARROW
TOTAL_ARROWS = NUM_ARROWS_X * NUM_ARROWS_Y

if TOTAL_ARROWS > 0:
    # Each arrow has 6 vertices (3 lines: shaft, left wing, right wing)
    vector_verts = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_ARROWS * 6)
    vector_colors = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_ARROWS * 6)


def _gradient(norm, color_map):
    """Interpolate the 9-stop theme gradient. norm in [0, 1], any shape -> (..., 3) float 0-255."""
    idx_f = norm * 8.0
    lower = torch.floor(idx_f).to(torch.int64).clamp_(0, 8)
    upper = (lower + 1).clamp_(max=8)
    weight = (idx_f - lower.to(idx_f.dtype)).unsqueeze(-1)
    return color_map[lower] * (1.0 - weight) + color_map[upper] * weight


def _log_norm(values, min_log, max_log):
    log_v = torch.log10(values + 0.1)
    return ((log_v - min_log) / max(max_log - min_log, 1e-6)).clamp_(0.0, 1.0)


_LUT_CACHE = {}

def _color_lut(color_map):
    """257-entry uint8 lookup table for a theme: 256 gradient steps plus a black entry for empty cells."""
    key = (color_map.data_ptr(), str(color_map.device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        steps = _gradient(torch.linspace(0.0, 1.0, 256, device=color_map.device), color_map)
        lut = torch.cat((steps, torch.zeros((1, 3), device=color_map.device))).to(torch.uint8)
        _LUT_CACHE[key] = lut
    return lut


def colorize(density, color_map, min_log, max_log, scale):
    """(H, W) density -> (H, W, 3) uint8 through the log10 colour map; empty cells stay black."""
    idx = (_log_norm(density * scale, min_log, max_log) * 255.0).round_().to(torch.int64)
    idx = torch.where(density > 0, idx, 256)
    return _color_lut(color_map)[idx]


class MeshRenderer:
    def __init__(self, kernels):
        self.kernels = kernels  # particle backend; provides splat_view on the sim device

    def render(self, canvas, pos_tensor, num_active, cam_x, cam_y, zoom, color_map_tensor, min_log, max_log, zoom_intensity_scale, overlay=None):
        """
        Splats particles onto the viewport grid, maps density through the colour map and presents it.
        overlay: optional (thumb_tensor, thumb_w, thumb_h) grayscale image drawn in the bottom-left corner.
        """
        density = self.kernels.splat_view(pos_tensor, num_active, cam_x, cam_y, zoom, VIEW_W, VIEW_H, RENDER_CELL_SIZE)
        scale = float(max(1.0, zoom) ** zoom_intensity_scale)
        self._present(canvas, colorize(density, color_map_tensor, min_log, max_log, scale), overlay)

    def render_grid(self, canvas, view_dens_tensor, color_map_tensor, min_log, max_log, scale, overlay=None):
        """
        Renders a grid-sampled potential or bandpass density map directly.
        overlay: optional (thumb_tensor, thumb_w, thumb_h) grayscale image drawn in the bottom-left corner.
        """
        self._present(canvas, colorize(view_dens_tensor, color_map_tensor, min_log, max_log, float(scale)), overlay)

    def _present(self, canvas, image, overlay):
        """
        image: (VIEW_H, VIEW_W, 3) uint8 on the sim device, row 0 = bottom of the screen.
        The overlay is pasted before the upload; canvas.set_image() snapshots the field immediately.
        """
        if overlay is not None:
            thumb_tensor, thumb_w, thumb_h = overlay
            h, w = min(thumb_h, VIEW_H), min(thumb_w, VIEW_W)
            image[:h, :w] = thumb_tensor[:h, :w].unsqueeze(-1)
        # display_pixels is indexed [x, y]; one device->host copy of the finished frame when memory isn't shared
        display_pixels.from_torch(to_taichi(image.transpose(0, 1).contiguous()))
        canvas.set_image(display_pixels)

    def render_vectors(self, canvas, view_fx, view_fy, color_map_tensor, min_log, max_log, length_scale, head_size, head_angle, width):
        """
        Builds a grid of force arrows (shaft + two head wings) on the sim device and draws them as lines.
        """
        if TOTAL_ARROWS <= 0:
            return
        device = view_fx.device
        spacing = PIXELS_PER_ARROW
        cx = torch.arange(NUM_ARROWS_X, device=device) * spacing + spacing // 2
        cy = torch.arange(NUM_ARROWS_Y, device=device) * spacing + spacing // 2
        px, py = torch.meshgrid(cx, cy, indexing='ij')           # (nx, ny), arrow index = i * ny + j

        f_x = view_fx[py, px]
        f_y = view_fy[py, px]
        f_mag = torch.sqrt(f_x * f_x + f_y * f_y)
        colors = _gradient(_log_norm(f_mag, min_log, max_log), color_map_tensor) / 255.0

        safe = f_mag > 1e-5
        dir_x = torch.where(safe, f_x / f_mag.clamp(min=1e-5), 0.0)
        dir_y = torch.where(safe, f_y / f_mag.clamp(min=1e-5), 0.0)

        max_val = 10.0 ** max_log
        min_val = 10.0 ** min_log
        linear_frac = ((f_mag - min_val) / max(1e-5, max_val - min_val)).clamp_(0.0, 1.0)
        length = linear_frac * length_scale * spacing

        sx = px.to(torch.float32) / VIEW_W
        sy = py.to(torch.float32) / VIEW_H
        ex = sx + dir_x * length / VIEW_W
        ey = sy + dir_y * length / VIEW_H

        wing = length * head_size
        cos_a, sin_a = math.cos(head_angle), math.sin(head_angle)
        wx1 = ex + (-dir_x * cos_a + dir_y * sin_a) * wing / VIEW_W
        wy1 = ey + (-dir_x * sin_a - dir_y * cos_a) * wing / VIEW_H
        wx2 = ex + (-dir_x * cos_a - dir_y * sin_a) * wing / VIEW_W
        wy2 = ey + (dir_x * sin_a - dir_y * cos_a) * wing / VIEW_H

        verts = torch.stack((sx, sy, ex, ey, ex, ey, wx1, wy1, ex, ey, wx2, wy2), dim=-1).reshape(-1, 2)
        vert_colors = colors.unsqueeze(2).expand(-1, -1, 6, -1).reshape(-1, 3)

        vector_verts.from_torch(to_taichi(verts.contiguous()))
        vector_colors.from_torch(to_taichi(vert_colors.contiguous()))
        canvas.lines(vector_verts, width=width, per_vertex_color=vector_colors)
