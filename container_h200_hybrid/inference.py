"""Predicted-category dispatch between the deployed fallback and ResEnc-L.

The specialist is never run for predicted sparse contiguous structures.
No ground-truth category or held-out metadata is available in this container.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk

import fallback_inference
import specialist_inference
from hybrid_policy import three_way_source


INPUT = Path("/input/images/light-sheet-3d-microscopy")
OUTPUT = Path("/output/images/isolated-biological-structure")


@dataclass
class Runtime:
    fallback: fallback_inference.Runtime
    specialist: object


def choose_source_with_size(predicted_category: str, voxel_count: int) -> str:
    limit = int(os.environ.get("SELMA_MAX_SPECIALIST_VOXELS", "27000000"))
    return three_way_source(predicted_category, voxel_count, limit)


def initialize() -> Runtime:
    return Runtime(
        fallback=fallback_inference.initialize(),
        specialist=specialist_inference.initialize(),
    )


def run(runtime: Runtime) -> None:
    images = sorted(
        path for pattern in ("*.mha", "*.tif", "*.tiff")
        for path in INPUT.glob(pattern)
    )
    if len(images) != 1:
        raise RuntimeError(f"Expected one supported input in {INPUT}, got {len(images)}")
    source_path = images[0]
    source = sitk.ReadImage(str(source_path))
    if source.GetDimension() != 3:
        raise RuntimeError("Expected a 3D input image")
    started = time.perf_counter()
    array = sitk.GetArrayFromImage(source)
    limit = int(os.environ.get("SELMA_MAX_SPECIALIST_VOXELS", "27000000"))
    if array.size > limit:
        # The size guard has already ruled out the specialist, so avoid a
        # duplicate route-only pass on the largest sparse-structure volumes.
        fallback_mask, predicted_category = fallback_inference._predict_standard_with_category(
            runtime.fallback, array
        )
        chosen = "fallback"
    else:
        predicted_category = fallback_inference.predict_category_only(
            runtime.fallback, array
        )
        chosen = choose_source_with_size(predicted_category, int(array.size))
        fallback_mask = None
    if chosen == "specialist":
        # Reuse the already-read image so the specialist path does not pay a
        # second large MHA decode. It still runs its own trained preprocessing.
        specialist_inference.run(
            runtime.specialist, source=source, source_path=source_path,
            output_name="segmentation.mha",
        )
    else:
        if fallback_mask is None:
            fallback_mask, full_category = fallback_inference._predict_standard_with_category(
                runtime.fallback, array
            )
            if full_category != predicted_category:
                raise RuntimeError(
                    f"Route-only/full category mismatch: {predicted_category} vs {full_category}"
                )
        if fallback_mask.shape != array.shape:
            raise RuntimeError("Fallback mask and input grid differ")
        mask = sitk.GetImageFromArray(np.asarray(fallback_mask, dtype=np.uint8))
        mask.CopyInformation(source)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(mask, str(OUTPUT / "segmentation.mha"), useCompression=True)
    print(
        f"predicted_category={predicted_category} chosen={chosen} "
        f"total_seconds={time.perf_counter() - started:.2f}", flush=True
    )
