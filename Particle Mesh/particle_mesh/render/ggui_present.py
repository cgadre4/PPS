"""
Hands finished frames and overlay vertices to GGUI. The only module that touches Taichi's private canvas API
(PyCanvas.set_image / circles / triangles with host-mapped buffers); callers fall back to the public Canvas API when
that attribute is missing.
"""
import numpy as np
import torch
from taichi.ui.utils import get_field_info

# GGUI vertex layout (taichi.ui.staging_buffer): position 3, normal 3, texcoord 2, colour 4, all float32
VERTEX_STRIDE = 12


def _py_canvas(canvas, method):
    """The PyCanvas behind a ti.ui.Canvas if it has `method`, else None."""
    py_canvas = getattr(canvas, "canvas", None)
    return py_canvas if py_canvas is not None and hasattr(py_canvas, method) else None


def host_vertex_buffer(n):
    """Persistent zeroed host vertex buffer for n vertices in GGUI's layout; 2D positions go in columns 0 and 1."""
    return np.zeros((n, VERTEX_STRIDE), dtype=np.float32)


def draw_host_circles(canvas, vbo, radius, color):
    """
    One-colour circles at the positions in a host_vertex_buffer, handed to PyCanvas.circles as host-mapped memory:
    what Canvas.circles does after its copy_all_to_vbo_particle kernel, minus that kernel and its Vulkan wait.
    Returns False when the PyCanvas attribute is missing (the caller then uses the public API).
    """
    py_canvas = _py_canvas(canvas, "circles")
    if py_canvas is None:
        return False
    py_canvas.circles(get_field_info(vbo), False, False, color, radius)
    return True


def draw_host_triangles(canvas, vbo, color):
    """One-colour triangles (consecutive vertex triples of a host_vertex_buffer); see draw_host_circles."""
    py_canvas = _py_canvas(canvas, "triangles")
    if py_canvas is None:
        return False
    py_canvas.triangles(get_field_info(vbo), get_field_info(None), False, color)
    return True


class HostFramePresenter:
    """Persistent host frame buffer: packed RGBA8 uint32 in GGUI's [x, y] layout (view_w, view_h), pinned when the
    torch accelerator supports it (faster readback)."""

    def __init__(self, view_w, view_h):
        self.view_w = view_w
        self.view_h = view_h
        try:
            self.host = torch.empty((view_w, view_h), dtype=torch.int32, pin_memory=True)
        except RuntimeError:
            self.host = torch.empty((view_w, view_h), dtype=torch.int32)
        self.host_np = self.host.numpy().view(np.uint32)
        self.host_info = get_field_info(self.host_np)

    def present(self, canvas, image, overlay=None):
        """
        image: (view_w, view_h) packed RGBA8 int32 on the sim device, GGUI [x, y] layout, y = 0 at the bottom.
        overlay: optional (packed thumb tensor [x, y], thumb_w, thumb_h), pasted into the bottom-left corner first.

        The frame is copied once into the host buffer and handed to GGUI as a host-mapped uint32 image: exactly
        what Taichi's own Canvas.set_image does after its to_rgba8 conversion, minus that conversion and the
        field.from_torch upload (two Taichi kernel launches and two Vulkan waits per frame; AUDIT §8 F01).
        """
        if overlay is not None:
            thumb, thumb_w, thumb_h = overlay
            w, h = min(thumb_w, self.view_w), min(thumb_h, self.view_h)
            image[:w, :h] = thumb[:w, :h]
        self.host.copy_(image)
        py_canvas = _py_canvas(canvas, "set_image")
        if py_canvas is not None:
            py_canvas.set_image(self.host_info)
        else:  # public-API fallback (e.g. a future Taichi without the PyCanvas attribute)
            rgb = np.ascontiguousarray(self.host_np.view(np.uint8).reshape(self.view_w, self.view_h, 4)[..., :3])
            canvas.set_image(rgb)
