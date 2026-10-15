import math
import taichi as ti
import torch

from particle_mesh.settings import SETTINGS
from particle_mesh.runtime.device import to_taichi
from particle_mesh.render.ggui_present import HostFramePresenter

WIDTH = SETTINGS.width
HEIGHT = SETTINGS.height
RENDER_CELL_SIZE = SETTINGS.render_cell_size

VIEW_W = SETTINGS.view_w
VIEW_H = SETTINGS.view_h

# --- Vector arrow grid (spacing in render cells) ---
PIXELS_PER_ARROW = SETTINGS.pixels_per_arrow
NUM_ARROWS_X = VIEW_W // PIXELS_PER_ARROW
NUM_ARROWS_Y = VIEW_H // PIXELS_PER_ARROW
TOTAL_ARROWS = NUM_ARROWS_X * NUM_ARROWS_Y

# FFT minimap: recompute the thumbnail every N frames (the spectrum changes slowly; it is ~0.5-1 ms per refresh)
MINIMAP_INTERVAL = SETTINGS.minimap_interval
# Largest count-indexed colour LUT (entries); beyond this the float colour path is used
MAX_COUNT_LUT = 1 << 20
from particle_mesh.render.colormap import (  # noqa: E402,F401  (re-exported)
    ALPHA, _LUT_CACHE, _color_lut, _gradient, _log_norm, _pack_gray, _pack_rgb, colorize,
)


from particle_mesh.render.viewport import _sample, build_view_grid  # noqa: E402,F401  (re-exported)


class MeshRenderer:
    def __init__(self, kernels, settings=SETTINGS):
        self.kernels = kernels  # particle backend; provides splat_view on the sim device
        # Render settings, captured once (no per-frame config reads)
        self.min_log_density = settings.min_log_density
        self.max_log_density = settings.max_log_density
        self.zoom_intensity_scale = settings.zoom_intensity_scale
        self.arrow_style = (settings.arrow_length_scale, settings.arrow_head_size, settings.arrow_head_angle,
                            settings.arrow_width)
        self._bandpass = None
        self._bandpass_key = None
        self._count_lut = None
        self._count_lut_key = None
        self._view_grid = None
        self._view_grid_key = None
        self._thumb = None
        self._thumb_key = None
        self._frame = 0
        # Presentation: the finished frame goes to GGUI through one persistent (pinned) host buffer
        self._presenter = HostFramePresenter(VIEW_W, VIEW_H)
        self._host = self._presenter.host
        self._host_np = self._presenter.host_np
        # Arrow vertex buffers (Taichi fields; created here, after ti.init, never at import)
        self.vector_verts = self.vector_colors = None
        if TOTAL_ARROWS > 0:
            # Each arrow has 6 vertices (3 lines: shaft, left wing, right wing)
            self.vector_verts = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_ARROWS * 6)
            self.vector_colors = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_ARROWS * 6)

    # ------------------------------------------------------------------ caches

    def bandpass_mask(self, k_sq, inner, outer, transition):
        """
        Smooth radial band-pass in |k| (rfft2 layout), cached until the sliders or the grid change.
        Returns None when the band-pass is the identity (default sliders), so callers can skip the multiply.
        """
        if inner <= 0.01 and outer >= 4.99:
            return None
        key = (k_sq.data_ptr(), inner, outer, transition)
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

    def _counts_lut(self, color_map, min_log, max_log, scale):
        """
        Packed colour for every particle count 0..C: count c maps through the same log10 curve as colorize().
        C is the count at which the curve saturates, so clamping larger counts to C is exact. Returns None if
        that would exceed MAX_COUNT_LUT entries (then the float path is used).
        """
        key = (color_map.data_ptr(), min_log, max_log, scale)
        if self._count_lut_key != key:
            c_max = int(math.ceil(max(10.0 ** max_log - 0.1, 0.0) / max(scale, 1e-12))) + 1
            if c_max > MAX_COUNT_LUT:
                self._count_lut, self._count_lut_key = None, key
                return None
            counts = torch.arange(c_max + 1, device=color_map.device, dtype=torch.float32)
            idx = (_log_norm(counts * scale, min_log, max_log) * 255.0).round_().to(torch.int64)
            idx[0] = 256  # empty cell -> black
            self._count_lut, self._count_lut_key = _color_lut(color_map)[idx], key
        return self._count_lut

    def _view_grid_for(self, camera, engine):
        """build_view_grid, cached while the camera and universe are unchanged."""
        key = (camera.cam_x, camera.cam_y, camera.zoom, engine.universe_w, engine.universe_h)
        if self._view_grid_key != key:
            self._view_grid = build_view_grid(camera, engine.universe_w, engine.universe_h, engine.device)
            self._view_grid_key = key
        return self._view_grid

    # ------------------------------------------------------------------ minimap

    def fft_thumbnail(self, rho_k, mask, size):
        """
        Grayscale log-magnitude spectrum, size x size render cells, DC at the centre.
        Returns an overlay tuple (packed RGBA8 int32 tensor in [x, y] layout, width, height).
        mask None means identity (no multiply).
        """
        mag = torch.abs(rho_k)
        if mask is not None:
            mag.mul_(mask)
        # antialias=True low-pass filters while shrinking (e.g. 2048x1025 -> 216x108); plain bilinear sampling
        # would alias fine spectral detail into moire (AUDIT 2.1)
        thumb_half = torch.nn.functional.interpolate(
            mag[None, None], size=(size, size // 2), mode='bilinear', align_corners=False, antialias=True
        )[0, 0]
        thumb_half = torch.fft.fftshift(torch.log1p(thumb_half), dim=0)
        # Real input: |F(-kx, -ky)| = |F(kx, ky)|, so the negative-kx half is a point reflection
        thumb = torch.cat([torch.flip(thumb_half, dims=[0, 1]), thumb_half], dim=1)
        gray = (thumb / (thumb.max() + 1e-5) * 255.0).to(torch.uint8)       # [y, x]
        return _pack_gray(gray.t()).contiguous(), gray.shape[1], size

    def _minimap(self, rho_k, mask, size):
        """fft_thumbnail, refreshed every MINIMAP_INTERVAL frames (or when its size or band-pass changes)."""
        key = (size, self._bandpass_key if mask is not None else None)
        if self._thumb is None or self._thumb_key != key or self._frame % MINIMAP_INTERVAL == 0:
            self._thumb, self._thumb_key = self.fft_thumbnail(rho_k, mask, size), key
        return self._thumb

    # ------------------------------------------------------------------ frames

    def render_pm(self, canvas, engine, camera, color_map, ui_w, view=None):
        """
        Draws a full PM-mode frame: the particle density (or the potential / band-passed density view), the
        optional FFT minimap overlay and the optional force-arrow overlay. view: a ViewState (default engine.view).
        """
        view = engine.view if view is None else view
        self._frame += 1
        solver = engine.pm_solver
        rho_k = solver.rho_k  # density spectrum from this step's solve (no extra FFT)
        mask = None
        if rho_k is not None and (view.show_fft or view.use_bandpass_render):
            mask = self.bandpass_mask(solver.k_sq, view.bp_inner, view.bp_outer, view.bp_transition)

        # FFT radar thumbnail, pasted into the frame before it is presented. It spans ui_w of the window width,
        # measured in render cells.
        overlay = None
        if view.show_fft and rho_k is not None:
            overlay = self._minimap(rho_k, mask, int(camera.width * ui_w / RENDER_CELL_SIZE))

        if view.show_potential:
            image = _sample(solver.phi, self._view_grid_for(camera, engine)).abs_()
            self.render_grid(canvas, image, color_map, view.pot_min_log, view.pot_max_log, 1.0, overlay=overlay)
        elif view.use_bandpass_render and rho_k is not None:
            # Identity band-pass: the filtered density is just the density grid (no inverse FFT)
            if mask is None:
                filtered = solver.density_grid
            else:
                filtered = torch.fft.irfft2(rho_k * mask, s=solver.density_grid.shape)
            image = _sample(filtered, self._view_grid_for(camera, engine)).clamp_(min=0.0)
            self.render_grid(canvas, image, color_map, self.min_log_density, self.max_log_density, 1.0,
                             overlay=overlay)
        else:
            self.render(canvas, engine.pos, engine.num_active, camera.cam_x, camera.cam_y, camera.zoom, color_map,
                        self.min_log_density, self.max_log_density, self.zoom_intensity_scale,
                        overlay=overlay)

        if view.show_field:
            self.render_vectors(
                canvas, solver.fx, solver.fy, camera, engine.universe_w, engine.universe_h, color_map,
                view.field_min_log, view.field_max_log, *self.arrow_style
            )

    def render(self, canvas, pos_tensor, num_active, cam_x, cam_y, zoom, color_map_tensor, min_log, max_log, zoom_intensity_scale, overlay=None):
        """
        Splats particles onto the viewport grid, maps counts through the colour map and presents it.
        overlay: optional (packed thumb tensor [x, y], thumb_w, thumb_h) drawn in the bottom-left corner.
        """
        counts = self.kernels.splat_view(pos_tensor, num_active, cam_x, cam_y, zoom, VIEW_W, VIEW_H, RENDER_CELL_SIZE)
        scale = float(max(1.0, zoom) ** zoom_intensity_scale)
        lut = self._counts_lut(color_map_tensor, min_log, max_log, scale)
        if lut is not None:
            # Count-indexed LUT: one clamp + one gather per cell (AUDIT §8 F09)
            image = lut[counts.clamp_(max=lut.shape[0] - 1)]
        else:
            image = colorize(counts.to(torch.float32), color_map_tensor, min_log, max_log, scale)
        self._present(canvas, image, overlay)

    def render_grid(self, canvas, view_dens_tensor, color_map_tensor, min_log, max_log, scale, overlay=None):
        """
        Renders a grid-sampled potential or bandpass density map (VIEW_H, VIEW_W) directly.
        overlay: optional (packed thumb tensor [x, y], thumb_w, thumb_h) drawn in the bottom-left corner.
        """
        # Colour through the transposed view so the packed image comes out in GGUI's [x, y] layout
        self._present(canvas, colorize(view_dens_tensor.t(), color_map_tensor, min_log, max_log, float(scale)), overlay)

    def _present(self, canvas, image, overlay):
        """image: (VIEW_W, VIEW_H) packed RGBA8 int32 on the sim device (see HostFramePresenter.present)."""
        self._presenter.present(canvas, image, overlay)

    def render_vectors(self, canvas, fx_grid, fy_grid, camera, universe_w, universe_h, color_map_tensor, min_log, max_log,
                       length_scale, head_size, head_angle, width):
        """
        Grid of force arrows (shaft + two head wings), drawn as lines. The force grids are sampled only at the
        arrow centres (AUDIT §8 F20), not resampled over the whole viewport.
        """
        if TOTAL_ARROWS <= 0:
            return
        device = fx_grid.device
        spacing = PIXELS_PER_ARROW
        cx = torch.arange(NUM_ARROWS_X, device=device) * spacing + spacing // 2
        cy = torch.arange(NUM_ARROWS_Y, device=device) * spacing + spacing // 2
        px, py = torch.meshgrid(cx, cy, indexing='ij')           # (nx, ny), arrow index = i * ny + j

        # World position of each arrow's render-cell centre -> normalised grid_sample coordinates
        world_x = (px.to(torch.float32) + 0.5) * RENDER_CELL_SIZE / camera.zoom + camera.cam_x
        world_y = (py.to(torch.float32) + 0.5) * RENDER_CELL_SIZE / camera.zoom + camera.cam_y
        coords = torch.stack((world_x / universe_w * 2.0 - 1.0, world_y / universe_h * 2.0 - 1.0), dim=-1)[None]
        f_x = torch.nn.functional.grid_sample(fx_grid[None, None], coords, align_corners=False, padding_mode='zeros')[0, 0]
        f_y = torch.nn.functional.grid_sample(fy_grid[None, None], coords, align_corners=False, padding_mode='zeros')[0, 0]

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
