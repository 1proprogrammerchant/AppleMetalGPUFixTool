"""
triton-graphic – Triton GPU kernel suite for Apple iPhone graphics fixes.

Provides GPU-accelerated detection and repair of visual glitches caused by
Apple Metal TBDR pipeline bugs on 2026-era hardware (M4/M5, A18).
"""
from tritongraphicset import fix_graphics_buffer, fix_half_denorms
from texture_repair import scrub_nan_inf, repair_texture
from depth_fix import fix_depth_buffer, detect_depth_artifacts
from tile_gap_fill import fill_tile_gaps, detect_tile_gaps
from flicker_suppress import suppress_flicker, compute_flicker_mask
from vertex_sanitizer import sanitize_vertices, count_outlier_vertices
from pipeline import TritonGPUFixPipeline

__all__ = [
    # Core fixes
    "fix_graphics_buffer",
    "fix_half_denorms",
    # Texture
    "scrub_nan_inf",
    "repair_texture",
    # Depth
    "fix_depth_buffer",
    "detect_depth_artifacts",
    # Tile gaps
    "fill_tile_gaps",
    "detect_tile_gaps",
    # Flicker
    "suppress_flicker",
    "compute_flicker_mask",
    # Vertices
    "sanitize_vertices",
    "count_outlier_vertices",
    # Pipeline
    "TritonGPUFixPipeline",
]
