import math
import taichi as ti
import torch

from config import CONFIG
from kernels import to_taichi

WIDTH = CONFIG["display"]["WIDTH"]
HEIGHT = CONFIG["display"]["HEIGHT"]
RENDER_CELL_SIZE = CONFIG["simulation"]["RENDER_CELL_SIZE"]

VIEW_W = int(WIDTH / RENDER_CELL_SIZE)
VIEW_H = int(HEIGHT / RENDER_CELL_SIZE)

# --- Vector arrow grid (spacing in render cells) ---
PIXELS_PER_ARROW = CONFIG["display"].get("pixels_per_arrow", 32)
NUM_ARROWS_X = VIEW_W // PIXELS_PER_ARROW
NUM_ARROWS_Y = VIEW_H // PIXELS_PER_ARROW
TOTAL_ARROWS = NUM_ARROWS_X * NUM_ARROWS_Y


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


def build_view_grid(camera, universe_w, universe_h, device):
    """
    grid_sample coordinates that map every viewport render cell onto the physics grid.
    Returns a (1, VIEW_H, VIEW_W, 2) tensor of normalised [-1, 1] coordinates (for align_corners=False).
    """
    grid_w = int(camera.width / RENDER_CELL_SIZE)
    grid_h = int(camera.height / RENDER_CELL_SIZE)

    screen_y, screen_x = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=torch.float32),
        torch.arange(grid_w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    # Render cell (gx, gy) covers screen pixels [gx, gx + 1) * RENDER_CELL_SIZE (same binning as splat_view);
    # sample the physics grid at the world position of that cell's centre.
    world_gx = (screen_x + 0.5) * RENDER_CELL_SIZE / camera.zoom + camera.cam_x
    world_gy = (screen_y + 0.5) * RENDER_CELL_SIZE / camera.zoom + camera.cam_y

    grid_x_norm = (world_gx / universe_w) * 2.0 - 1.0
    grid_y_norm = (world_gy / universe_h) * 2.0 - 1.0
    return torch.stack((grid_x_norm, grid_y_norm), dim=-1).unsqueeze(0)


def _sample(grids, view_grid):
    """grid_sample a (C, H, W) stack of physics grids onto the viewport -> (C, VIEW_H, VIEW_W)."""
    return torch.nn.functional.grid_sample(grids.unsqueeze(0), view_grid, align_corners=False,
                                           padding_mode='zeros').squeeze(0)


class MeshRenderer:
    def __init__(self, kernels):
        self.kernels = kernels  # particle backend; provides splat_view on the sim device
        self._bandpass = None
        self._bandpass_key = None
        # Taichi fields for GGUI presentation only (all image math happens in PyTorch on the sim device).
        # Created here, not at import, so this module can be imported before ti.init().
        self.display_pixels = ti.Vector.field(3, dtype=ti.u8, shape=(VIEW_W, VIEW_H))
        self.vector_verts = self.vector_colors = None
        if TOTAL_ARROWS > 0:
            # Each arrow has 6 vertices (3 lines: shaft, left wing, right wing)
            self.vector_verts = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_ARROWS * 6)
            self.vector_colors = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_ARROWS * 6)

    def bandpass_mask(self, k_sq, inner, outer, transition):
        """Smooth radial band-pass in |k| (rfft2 layout), cached until the sliders or the grid change."""
        key = (id(k_sq), inner, outer, transition)
        if self._bandpass_key != key:
            k_mag = torch.sqrt(k_sq)
            trans = max(transition, 0.001)
            mask = torch.ones_like(k_mag)
            if inner > 0.01:
                mask *= torch.sigmoid((k_mag - inner) / trans * 4.0)
            if outer < 4.99:
                mask *= torch.sigmoid((outer - k_mag) / trans * 4.0)
            self._bandpass, self._bandpass_key = mask, key
        return self._bandpass

    def fft_thumbnail(self, rho_k, mask, size):
        """
        Grayscale log-magnitude spectrum, size x size render cells, DC at the centre.
        Returns an overlay tuple (uint8 tensor, width, height) for render()/render_grid().
        """
        # antialias=True low-pass filters while shrinking (e.g. 2048x1025 -> 216x108); plain bilinear sampling
        # would alias fine spectral detail into moire (AUDIT 2.1)
        thumb_half = torch.nn.functional.interpolate(
            (torch.abs(rho_k) * mask)[None, None], size=(size, size // 2), mode='bilinear', align_corners=False,
            antialias=True
        )[0, 0]
        thumb_half = torch.fft.fftshift(torch.log1p(thumb_half), dim=0)
        # Real input: |F(-kx, -ky)| = |F(kx, ky)|, so the negative-kx half is a point reflection
        thumb = torch.cat([torch.flip(thumb_half, dims=[0, 1]), thumb_half], dim=1)
        thumb = (thumb / (thumb.max() + 1e-5) * 255.0).to(torch.uint8)
        return thumb.contiguous(), thumb.shape[1], size

    def render_pm(self, canvas, engine, camera, color_map, ui_w):
        """
        Draws a full PM-mode frame: the particle density (or the potential / band-passed density view), the
        optional FFT minimap overlay and the optional force-arrow overlay.
        """
        solver = engine.pm_solver
        sim_cfg = CONFIG["simulation"]
        rho_k = solver.rho_k  # density spectrum from this step's solve (no extra FFT)
        mask = None
        if rho_k is not None and (engine.show_fft or engine.use_bandpass_render):
            mask = self.bandpass_mask(solver.k_sq, engine.bp_inner, engine.bp_outer, engine.bp_transition)

        # FFT radar thumbnail, drawn into the frame before canvas.set_image() snapshots it. It spans ui_w of the
        # window width, measured in render cells.
        overlay = None
        if engine.show_fft and rho_k is not None:
            overlay = self.fft_thumbnail(rho_k, mask, int(camera.width * ui_w / RENDER_CELL_SIZE))

        view_grid = None
        if engine.show_potential or engine.use_bandpass_render or engine.show_field:
            view_grid = build_view_grid(camera, engine.universe_w, engine.universe_h, engine.device)

        if engine.show_potential:
            view = _sample(torch.abs(solver.phi)[None], view_grid)[0].clamp_(min=0.0)
            self.render_grid(canvas, view, color_map, engine.pot_min_log, engine.pot_max_log, 1.0, overlay=overlay)
        elif engine.use_bandpass_render and rho_k is not None:
            filtered = torch.fft.irfft2(rho_k * mask, s=solver.density_grid.shape)
            view = _sample(filtered[None], view_grid)[0].clamp_(min=0.0)
            self.render_grid(canvas, view, color_map, sim_cfg["MIN_LOG_DENSITY"], sim_cfg["MAX_LOG_DENSITY"], 1.0,
                             overlay=overlay)
        else:
            self.render(canvas, engine.pos, engine.num_active, camera.cam_x, camera.cam_y, camera.zoom, color_map,
                        sim_cfg["MIN_LOG_DENSITY"], sim_cfg["MAX_LOG_DENSITY"], sim_cfg["ZOOM_INTENSITY_SCALE"],
                        overlay=overlay)

        if engine.show_field:
            forces = _sample(torch.stack((solver.fx, solver.fy)), view_grid)
            disp = CONFIG["display"]
            self.render_vectors(
                canvas, forces[0].contiguous(), forces[1].contiguous(), color_map,
                engine.field_min_log, engine.field_max_log,
                disp.get("arrow_length_scale", 1.5), disp.get("arrow_head_size", 0.25),
                disp.get("arrow_head_angle", 0.5), disp.get("arrow_width", 0.001)
            )

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
        self.display_pixels.from_torch(to_taichi(image.transpose(0, 1).contiguous()))
        canvas.set_image(self.display_pixels)

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

        self.vector_verts.from_torch(to_taichi(verts.contiguous()))
        self.vector_colors.from_torch(to_taichi(vert_colors.contiguous()))
        canvas.lines(self.vector_verts, width=width, per_vertex_color=self.vector_colors)
