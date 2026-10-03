"""
Core pixel boost algorithms: interleaved spatial sampling and tile reassembly.
"""

from typing import Callable, List
import numpy as np


def split_interleaved_tiles(crop: np.ndarray, target_size: int = 128) -> np.ndarray:
    """
    Deconstructs a high-res square crop into N interleaved sub-images.
    Instead of blurring down with interpolation, it samples every n-th pixel
    in spatial coordinates (x, y), preserving high-frequency phase information.

    Args:
        crop: H x W x C numpy array (where H == W and H is a multiple of target_size, e.g. 256).
        target_size: Sub-tile dimension (typically 128 for InSwapper).

    Returns:
        Array of shape (n*n, target_size, target_size, C) containing interleaved sub-grids.
    """
    h, w, c = crop.shape
    if h != w:
        raise ValueError(f"Crop must be square, got {h}x{w}")
    n = h // target_size
    if n * target_size != h:
        raise ValueError(f"Crop dimension {h} is not an integer multiple of target_size {target_size}")

    # Reshape: (target_size, n, target_size, n, c) -> transpose -> (n*n, target_size, target_size, c)
    tiles = crop.reshape(target_size, n, target_size, n, c).transpose(1, 3, 0, 2, 4).reshape(n * n, target_size, target_size, c)
    return tiles


def merge_interleaved_tiles(tiles: List[np.ndarray], crop_size: int = 256, target_size: int = 128) -> np.ndarray:
    """
    Reconstructs the full-resolution image from processed sub-grid tiles.

    Args:
        tiles: List or array of n*n processed tiles of shape (target_size, target_size, C).
        crop_size: Output resolution (e.g. 256).
        target_size: Tile resolution (e.g. 128).

    Returns:
        H x W x C numpy array at crop_size resolution.
    """
    n = crop_size // target_size
    c = tiles[0].shape[-1]
    stacked = np.stack(tiles)
    reconstructed = stacked.reshape(n, n, target_size, target_size, c).transpose(2, 0, 3, 1, 4).reshape(crop_size, crop_size, c)
    return reconstructed


def run_pixel_boost_inference(
    crop: np.ndarray,
    infer_fn: Callable[[np.ndarray], np.ndarray],
    crop_size: int = 256,
    sub_size: int = 128
) -> np.ndarray:
    """
    Executes 4-pass interleaved inference using an arbitrary 128px face swap backend.

    Args:
        crop: 256x256 aligned face crop.
        infer_fn: Function that takes a 128x128 image and returns a 128x128 swapped face.
        crop_size: Full crop dimension (default 256).
        sub_size: InSwapper internal dimension (default 128).

    Returns:
        256x256 high-fidelity swapped face image.
    """
    tiles = split_interleaved_tiles(crop, target_size=sub_size)
    processed_tiles = [infer_fn(tile) for tile in tiles]
    return merge_interleaved_tiles(processed_tiles, crop_size=crop_size, target_size=sub_size)
