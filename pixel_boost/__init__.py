"""
inswapper-pixel-boost
~~~~~~~~~~~~~~~~~~~~~

Interleaved grid sub-sampling and 256px upscaling pipeline for 128px InSwapper models.
"""

from .boost import split_interleaved_tiles, merge_interleaved_tiles, run_pixel_boost_inference
from .transform import OneEuroFilter, similarity_transform

__version__ = "0.1.0"
__all__ = [
    "split_interleaved_tiles",
    "merge_interleaved_tiles",
    "run_pixel_boost_inference",
    "OneEuroFilter",
    "similarity_transform",
]
