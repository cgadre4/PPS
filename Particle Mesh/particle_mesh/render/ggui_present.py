"""
Hands finished frames to GGUI. The only module that touches Taichi's private canvas API (PyCanvas.set_image with a
host-mapped buffer); it falls back to the public Canvas.set_image when that attribute is missing.
"""
import numpy as np
import torch
from taichi.ui.utils import get_field_info


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
        py_canvas = getattr(canvas, "canvas", None)
        if py_canvas is not None and hasattr(py_canvas, "set_image"):
            py_canvas.set_image(self.host_info)
        else:  # public-API fallback (e.g. a future Taichi without the PyCanvas attribute)
            rgb = np.ascontiguousarray(self.host_np.view(np.uint8).reshape(self.view_w, self.view_h, 4)[..., :3])
            canvas.set_image(rgb)
