"""
Colour mapping: 9-stop theme gradients, log normalisation, packed RGBA8 words and the per-theme 257-entry LUT.
"""
import torch

# Opaque alpha in the packed RGBA8 word (little-endian bytes R, G, B, A), as a signed int32
ALPHA = -(1 << 24)  # 0xFF000000


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


def _pack_rgb(rgb):
    """(..., 3) float/uint8 0-255 -> (...) int32 packed RGBA8 (R in the low byte, alpha 255)."""
    rgb = rgb.to(torch.int32)
    return rgb[..., 0] | (rgb[..., 1] << 8) | (rgb[..., 2] << 16) | ALPHA


def _pack_gray(gray):
    """uint8 gray -> packed RGBA8 int32."""
    return gray.to(torch.int32) * 0x010101 | ALPHA


_LUT_CACHE = {}

def _color_lut(color_map):
    """257-entry packed-RGBA8 table for a theme: 256 gradient steps plus opaque black for empty cells."""
    key = (color_map.data_ptr(), str(color_map.device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        steps = _gradient(torch.linspace(0.0, 1.0, 256, device=color_map.device), color_map)
        rgb = torch.cat((steps, torch.zeros((1, 3), device=color_map.device))).to(torch.uint8)
        lut = _pack_rgb(rgb)
        _LUT_CACHE[key] = lut
    return lut


def colorize(values, color_map, min_log, max_log, scale):
    """Float image (any 2D layout) -> packed RGBA8 int32 of the same shape via log10; zero/empty -> black."""
    idx = (_log_norm(values * scale, min_log, max_log) * 255.0).round_().to(torch.int64)
    idx = torch.where(values > 0, idx, 256)
    return _color_lut(color_map)[idx]


class Palettes:
    """The colour themes as (9, 3) float tensors on the sim device, kept alive for the whole run (the LUT cache
    is keyed by tensor address)."""

    def __init__(self, themes, device):
        self.names = list(themes.keys())
        self.maps = [torch.tensor(themes[name], dtype=torch.float32, device=device) for name in self.names]

    def __len__(self):
        return len(self.names)

    def get(self, index):
        return self.maps[index]

    def cycle(self, index, step=1):
        return (index + step) % len(self.names)
