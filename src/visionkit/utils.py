"""Image conversion helpers and color space utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy array (H, W, C) in uint8."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image, dtype=np.uint8)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a NumPy array (H, W, C) or (H, W) to a PIL Image."""
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def rgb_to_grayscale(array: np.ndarray) -> np.ndarray:
    """Convert an RGB NumPy array to grayscale using luminance weights.

    Uses the ITU-R BT.601 luma formula: Y = 0.299*R + 0.587*G + 0.114*B
    """
    if array.ndim == 2:
        return array
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray = np.dot(array[:, :, :3].astype(np.float64), weights)
    return gray.astype(np.uint8)


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """Convert a single RGB color to HSV.

    Returns (h, s, v) where h is in [0, 360), s and v in [0, 1].
    """
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    c_max = max(r_norm, g_norm, b_norm)
    c_min = min(r_norm, g_norm, b_norm)
    delta = c_max - c_min

    # Hue
    if delta == 0:
        h = 0.0
    elif c_max == r_norm:
        h = 60.0 * (((g_norm - b_norm) / delta) % 6)
    elif c_max == g_norm:
        h = 60.0 * (((b_norm - r_norm) / delta) + 2)
    else:
        h = 60.0 * (((r_norm - g_norm) / delta) + 4)

    # Saturation
    s = 0.0 if c_max == 0 else delta / c_max

    # Value
    v = c_max

    return (h, s, v)


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV to RGB.

    h in [0, 360), s and v in [0, 1]. Returns (r, g, b) in [0, 255].
    """
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if h < 60:
        r1, g1, b1 = c, x, 0.0
    elif h < 120:
        r1, g1, b1 = x, c, 0.0
    elif h < 180:
        r1, g1, b1 = 0.0, c, x
    elif h < 240:
        r1, g1, b1 = 0.0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0.0, c
    else:
        r1, g1, b1 = c, 0.0, x

    return (
        int((r1 + m) * 255),
        int((g1 + m) * 255),
        int((b1 + m) * 255),
    )


def clamp(value: float, low: float = 0.0, high: float = 255.0) -> float:
    """Clamp a numeric value to [low, high]."""
    return max(low, min(high, value))


def validate_image_dimensions(
    width: int, height: int, max_size: int = 4096
) -> None:
    """Raise ValueError if dimensions exceed max_size or are non-positive."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensions must be positive, got {width}x{height}.")
    if width > max_size or height > max_size:
        raise ValueError(
            f"Dimension {max(width, height)} exceeds maximum allowed {max_size}."
        )


def compute_aspect_ratio(
    orig_w: int, orig_h: int, target_w: int | None, target_h: int | None
) -> Tuple[int, int]:
    """Compute new dimensions preserving aspect ratio.

    If only one of target_w/target_h is provided, the other is computed.
    If both are provided, they are returned as-is.
    """
    if target_w is not None and target_h is not None:
        return (target_w, target_h)
    if target_w is not None:
        ratio = target_w / orig_w
        return (target_w, max(1, int(orig_h * ratio)))
    if target_h is not None:
        ratio = target_h / orig_h
        return (max(1, int(orig_w * ratio)), target_h)
    return (orig_w, orig_h)
