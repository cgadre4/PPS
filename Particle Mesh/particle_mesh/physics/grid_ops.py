"""
Grid-level operations shared by the field solver and the pressure model (pure PyTorch, any device).
"""
import torch


def neg_gradient(src, out_x, out_y, cell_size, periodic, prescaled=False):
    """
    Writes -grad(src) into out_x / out_y (each (H, W); may be non-contiguous views, e.g. the interiors of a padded
    buffer). Central differences in the interior; at the edges they wrap (periodic) or are one-sided (otherwise,
    matching torch.gradient's edge_order=1).

    Uses slices written straight into the outputs instead of torch.roll copies or torch.gradient's temporaries:
    measured 2.7 -> 1.2 ms on a 2048^2 grid on the Arc B390 (AUDIT §8 F03).

    prescaled: src already carries the 1 / (2 * cell_size) factor (the PM solver folds it into its cached kernel),
    so the two full-grid scaling passes are skipped (AUDIT §9 B1: they were 0.94 of 1.70 ms on 8192^2, RTX 5090).
    """
    inv_2dx = 0.5 / cell_size
    # x direction (columns): -d/dx at i = (src[i-1] - src[i+1]) / (2 dx)
    torch.sub(src[:, :-2], src[:, 2:], out=out_x[:, 1:-1])
    # y direction (rows)
    torch.sub(src[:-2, :], src[2:, :], out=out_y[1:-1, :])
    if periodic:
        torch.sub(src[:, -1], src[:, 1], out=out_x[:, 0])
        torch.sub(src[:, -2], src[:, 0], out=out_x[:, -1])
        torch.sub(src[-1, :], src[1, :], out=out_y[0, :])
        torch.sub(src[-2, :], src[0, :], out=out_y[-1, :])
    else:
        # One-sided: -(src[1] - src[0]) / dx = 2 * (src[0] - src[1]) * inv_2dx
        torch.sub(src[:, 0], src[:, 1], out=out_x[:, 0]).mul_(2.0)
        torch.sub(src[:, -2], src[:, -1], out=out_x[:, -1]).mul_(2.0)
        torch.sub(src[0, :], src[1, :], out=out_y[0, :]).mul_(2.0)
        torch.sub(src[-2, :], src[-1, :], out=out_y[-1, :]).mul_(2.0)
    if not prescaled:
        out_x.mul_(inv_2dx)
        out_y.mul_(inv_2dx)
    return out_x, out_y
