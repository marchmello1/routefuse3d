"""Annotation-free targets for a paper-aligned MSC pretraining experiment.

Input crops must already be source-volume z-scored and clipped to [-5, 20].
The returned channels are intensity, distance-to-signal, tube anisotropy, and
blob anisotropy. This module is separate from the currently running hybrid SSL
pipeline so an ablation cannot silently change its training distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def symmetric_eigenvalues(
    xx: np.ndarray,
    xy: np.ndarray,
    xz: np.ndarray,
    yy: np.ndarray,
    yz: np.ndarray,
    zz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form eigenvalues of symmetric 3x3 tensors, ascending."""
    q = (xx + yy + zz) / 3.0
    p2 = (xx - q) ** 2 + (yy - q) ** 2 + (zz - q) ** 2
    p2 += 2.0 * (xy**2 + xz**2 + yz**2)
    p = np.sqrt(np.maximum(p2 / 6.0, 0.0))
    safe_p = np.where(p > 1e-12, p, 1.0)
    a, d, f = (xx - q) / safe_p, (yy - q) / safe_p, (zz - q) / safe_p
    b, c, e = xy / safe_p, xz / safe_p, yz / safe_p
    determinant = a * d * f + 2.0 * b * c * e - a * e**2 - d * c**2 - f * b**2
    phi = np.arccos(np.clip(determinant / 2.0, -1.0, 1.0)) / 3.0
    largest = q + 2.0 * p * np.cos(phi)
    smallest = q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0)
    middle = 3.0 * q - smallest - largest
    return smallest, middle, largest


def structural_targets(crop: np.ndarray) -> np.ndarray:
    """Return the four MSC targets for one [D,H,W] z-scored crop."""
    image = np.asarray(crop, dtype=np.float32)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("crop must be a finite 3-D array")

    foreground = image > 2.0
    if foreground.any():
        distance = np.minimum(distance_transform_edt(~foreground), 24.0) / 24.0
    else:
        distance = np.ones(image.shape, dtype=np.float32)

    smoothed = gaussian_filter(image, sigma=1.5)
    gradients = np.gradient(smoothed)
    tensor = {}
    for row in range(3):
        for column in range(row, 3):
            tensor[row, column] = gaussian_filter(
                gradients[row] * gradients[column], sigma=2.0
            )

    tube = np.empty(image.shape, dtype=np.float32)
    blob = np.empty(image.shape, dtype=np.float32)
    # Closed-form eigenvalues avoid a costly D*H*W*3*3 decomposition.
    for depth in range(image.shape[0]):
        low, mid, high = symmetric_eigenvalues(
            tensor[0, 0][depth], tensor[0, 1][depth], tensor[0, 2][depth],
            tensor[1, 1][depth], tensor[1, 2][depth], tensor[2, 2][depth],
        )
        denominator = high + 1e-3
        tube[depth] = np.clip(
            (mid - low) / denominator, 0, 1
        )
        blob[depth] = np.clip(low / denominator, 0, 1)

    return np.stack(
        (image, distance.astype(np.float32), tube, blob), axis=0
    )


def normalize_source(volume: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """Apply the source-volume normalization specified in the MSC paper."""
    source = np.asarray(volume, dtype=np.float64)
    mean = float(source.mean())
    scale = max(float(source.std()), 1e-3)
    return np.clip((np.asarray(crop, dtype=np.float32) - mean) / scale, -5, 20)
