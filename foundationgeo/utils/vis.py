from typing import *

import numpy as np
import matplotlib


def colorize_depth(depth: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_depth_affine(depth: np.ndarray, mask: np.ndarray = None, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        depth = np.where(mask, depth, np.nan)

    min_depth, max_depth = np.nanquantile(depth, 0.001), np.nanquantile(depth, 0.999)
    depth = (depth - min_depth) / (max_depth - min_depth)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](depth)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_disparity(disparity: np.ndarray, mask: np.ndarray = None, normalize: bool = True, cmap: str = 'Spectral') -> np.ndarray:
    if mask is not None:
        disparity = np.where(mask, disparity, np.nan)
    
    if normalize:
        min_disp, max_disp = np.nanquantile(disparity, 0.001), np.nanquantile(disparity, 0.999)
        disparity = (disparity - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disparity)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_segmentation(segmentation: np.ndarray, cmap: str = 'Set1') -> np.ndarray:
    colored = matplotlib.colormaps[cmap]((segmentation % 20) / 20)[..., :3]
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_normal(normal: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    normal = normal * [0.5, -0.5, -0.5] + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal


def colorize_error_map(error_map: np.ndarray, mask: np.ndarray = None, cmap: str = 'plasma', value_range: Tuple[float, float] = None):
    vmin, vmax = value_range if value_range is not None else (np.nanmin(error_map), np.nanmax(error_map))
    cmap = matplotlib.colormaps[cmap]
    colorized_error_map = cmap(((error_map - vmin) / (vmax - vmin)).clip(0, 1))[..., :3]
    if mask is not None:
        colorized_error_map = np.where(mask[..., None], colorized_error_map, 0)
    colorized_error_map = np.ascontiguousarray((colorized_error_map.clip(0, 1) * 255).astype(np.uint8))
    return colorized_error_map


def colorize_scalar_field(
    scalar_field: np.ndarray,
    mask: np.ndarray = None,
    cmap: str = 'Spectral',
    clip_quantile: Tuple[float, float] = (0.01, 0.99),
) -> np.ndarray:
    scalar_field = np.asarray(scalar_field, dtype=np.float32)
    if mask is not None:
        scalar_field = np.where(mask, scalar_field, np.nan)

    valid = np.isfinite(scalar_field)
    if np.any(valid):
        qmin, qmax = clip_quantile
        vmin = np.nanquantile(scalar_field, qmin)
        vmax = np.nanquantile(scalar_field, qmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-8
        scalar_field = (scalar_field - vmin) / (vmax - vmin)
    else:
        scalar_field = np.zeros_like(scalar_field, dtype=np.float32)

    colored = np.nan_to_num(matplotlib.colormaps[cmap](scalar_field.clip(0, 1))[..., :3], 0)
    if mask is not None:
        colored = np.where(mask[..., None], colored, 0)
    return np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))


def colorize_vector_field(vector_field: np.ndarray, mask: np.ndarray = None, clip_quantile: float = 0.99) -> np.ndarray:
    """
    Visualize a 2D vector field with HSV colors.

    Hue encodes direction and value encodes magnitude.
    `vector_field` is expected to have shape (H, W, 2).
    """
    if vector_field.ndim != 3 or vector_field.shape[-1] != 2:
        raise ValueError(f"Expected vector_field with shape (H, W, 2), got {vector_field.shape}")

    vector_field = np.asarray(vector_field, dtype=np.float32)
    vx, vy = vector_field[..., 0], vector_field[..., 1]
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    angle = np.arctan2(vy, vx)

    if mask is not None:
        magnitude = np.where(mask, magnitude, np.nan)

    valid = np.isfinite(magnitude)
    if np.any(valid):
        max_magnitude = np.nanquantile(magnitude, clip_quantile)
        max_magnitude = max(max_magnitude, 1e-8)
    else:
        max_magnitude = 1.0

    hue = (angle + np.pi) / (2 * np.pi)
    saturation = np.ones_like(hue, dtype=np.float32)
    value = np.clip(magnitude / max_magnitude, 0, 1)

    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    if mask is not None:
        rgb = np.where(mask[..., None], rgb, 0)
    return np.ascontiguousarray((rgb.clip(0, 1) * 255).astype(np.uint8))
