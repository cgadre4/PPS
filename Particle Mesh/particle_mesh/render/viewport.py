"""
Viewport sampling: maps every render cell of the viewport onto the physics grid (for grid_sample).
"""
import torch

from particle_mesh.settings import SETTINGS

RENDER_CELL_SIZE = SETTINGS.render_cell_size


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


def _sample(grid, view_grid):
    """grid_sample one (H, W) physics grid onto the viewport -> (VIEW_H, VIEW_W)."""
    return torch.nn.functional.grid_sample(grid[None, None], view_grid, align_corners=False,
                                           padding_mode='zeros')[0, 0]
