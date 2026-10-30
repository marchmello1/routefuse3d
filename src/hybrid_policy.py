"""Inference-time selection between a routed model and a morphology expert."""

from __future__ import annotations

import numpy as np


def hybrid_prediction(
    routed: np.ndarray, specialist: np.ndarray, predicted_contiguous: bool
) -> np.ndarray:
    """Use the routed model for predicted tubes and the expert for blobs."""
    return routed if predicted_contiguous else specialist


def three_way_hybrid_prediction(
    routed: np.ndarray,
    specialist: np.ndarray,
    predicted_contiguous: bool,
    predicted_dense: bool | None,
) -> np.ndarray:
    """Expert for predicted isolated/dense; routed for predicted sparse tubes.

    The choice uses only inference-time heads, never the true validation
    category. Without a density head, retain the original morphology route.
    """
    if not predicted_contiguous:
        return specialist
    return specialist if predicted_dense is True else routed


def three_way_source(
    predicted_category: str,
    voxel_count: int,
    max_specialist_voxels: int,
) -> str:
    """Choose a family from model output and image size, never true category."""
    valid = {
        "isolated_sparse", "isolated_dense", "contiguous_sparse",
        "contiguous_dense", "isolated", "contiguous",
    }
    if predicted_category not in valid:
        raise ValueError(f"Unexpected predicted category: {predicted_category}")
    if voxel_count < 1 or max_specialist_voxels < 1:
        raise ValueError("voxel counts must be positive")
    if voxel_count > max_specialist_voxels:
        return "fallback"
    return "fallback" if predicted_category in ("contiguous_sparse", "contiguous") else "specialist"


def size_only_source(voxel_count: int, max_specialist_voxels: int) -> str:
    """Dispatch without a second inference pass or any semantic metadata."""
    if voxel_count < 1 or max_specialist_voxels < 1:
        raise ValueError("voxel counts must be positive")
    return "specialist" if voxel_count <= max_specialist_voxels else "fallback"


def minimum_size_for_route(predicted_contiguous: bool, isolated_minimum_size: int) -> int:
    """Apply postprocessing using an inference-time route, never the held-out label."""
    if isolated_minimum_size < 1:
        raise ValueError("isolated_minimum_size must be positive")
    return 1 if predicted_contiguous else isolated_minimum_size
