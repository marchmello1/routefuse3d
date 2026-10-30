from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import SimpleITK as sitk

from src.data import discover_labeled_cases, stratified_fold


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value[:96] or "case"


def convert(payload: tuple[Path, Path, Path, Path]) -> None:
    image_path, target_path, image_output, target_output = payload
    image = sitk.ReadImage(str(image_path))
    target = sitk.ReadImage(str(target_path))
    if image.GetSize() != target.GetSize():
        raise RuntimeError(f"Geometry mismatch: {image_path} and {target_path}")
    sitk.WriteImage(image, str(image_output), useCompression=True)
    sitk.WriteImage(target, str(target_output), useCompression=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cases = discover_labeled_cases(args.data)
    if len(cases) != 82:
        raise RuntimeError(f"Expected 82 labeled volumes, found {len(cases)}")
    images = args.output / "imagesTr"
    labels = args.output / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    jobs = []
    manifest = []
    seen = set()
    for index, case in enumerate(cases):
        identifier = f"selma_{index:03d}_{safe_name(case.image.stem)}"
        if identifier in seen:
            raise RuntimeError(f"Duplicate identifier: {identifier}")
        seen.add(identifier)
        image_output = images / f"{identifier}_0000.nii.gz"
        target_output = labels / f"{identifier}.nii.gz"
        jobs.append((case.image, case.target, image_output, target_output))
        manifest.append(
            {
                "identifier": identifier,
                "image": str(case.image),
                "target": str(case.target),
                "subtype": case.subtype,
                "contiguous": bool(case.contiguous),
            }
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(convert, jobs))

    dataset = {
        "channel_names": {"0": "light-sheet microscopy"},
        "labels": {"background": 0, "structure": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    (args.output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    id_by_image = {str(item["image"]): item["identifier"] for item in manifest}
    splits = []
    for fold in range(5):
        train, validation = stratified_fold(cases, fold)
        splits.append(
            {
                "train": [id_by_image[str(case.image)] for case in train],
                "val": [id_by_image[str(case.image)] for case in validation],
            }
        )
    (args.output / "splits_final.json").write_text(
        json.dumps(splits, indent=2) + "\n"
    )
    print(f"prepared={len(cases)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
