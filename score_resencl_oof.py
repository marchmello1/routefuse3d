"""Score only excluded-fold predictions, pooled and in official structure strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from panoptica import (
    ConnectedComponentsInstanceApproximator,
    InputType,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
)
from panoptica.metrics import Metric
from panoptica.metrics.cldice import _get_skeleton, cl_score
from scipy import ndimage

from src.data import structure_category


def remove_small(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 1:
        return mask
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels]


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(prediction, target).sum() / denominator)


def pq(prediction: np.ndarray, target: np.ndarray) -> float:
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(cca_backend=None),
        instance_matcher=NaiveThresholdMatching(matching_threshold=0.3),
        instance_metrics=[Metric.DSC],
    )
    return float(evaluator.evaluate(prediction, target, verbose=False)["ungrouped"].pq_dsc)


def cldice(prediction: np.ndarray, target: np.ndarray) -> float:
    precision = cl_score(prediction, _get_skeleton(target))
    sensitivity = cl_score(target, _get_skeleton(prediction))
    total = precision + sensitivity
    return 0.0 if total == 0 else float(2 * precision * sensitivity / total)


def geometry_close(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), atol=1e-4)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), atol=1e-3)
        and np.allclose(left.GetDirection(), right.GetDirection(), atol=1e-5)
    )


def score(
    work: Path,
    labels_dir: Path,
    manifest_path: Path,
    minimum: int,
    method: str = "excluded_fold_no_tta",
    partial: bool = False,
) -> dict:
    plan = json.loads((work / "plan.json").read_text())
    manifest = {row["identifier"]: row for row in json.loads(manifest_path.read_text())}
    rows = []
    for fold in plan["folds"]:
        output_dir = work / f"fold_{fold['source_fold']}" / "output"
        expected = set(fold["validation_ids"])
        found = {path.name.removesuffix(".nii.gz") for path in output_dir.glob("*.nii.gz")}
        if (not partial and found != expected) or (partial and not found.issubset(expected)):
            raise ValueError(f"Fold {fold['source_fold']} output mismatch: missing={expected-found}, extra={found-expected}")
        for identifier in fold["validation_ids"]:
            if identifier not in found:
                continue
            if identifier not in manifest:
                raise ValueError(f"Missing source manifest entry: {identifier}")
            target_image = sitk.ReadImage(str(labels_dir / f"{identifier}.nii.gz"))
            prediction_image = sitk.ReadImage(str(output_dir / f"{identifier}.nii.gz"))
            if not geometry_close(target_image, prediction_image):
                raise ValueError(f"Geometry mismatch: {identifier}")
            target = sitk.GetArrayFromImage(target_image) > 0
            prediction = remove_small(sitk.GetArrayFromImage(prediction_image) > 0, minimum)
            source = manifest[identifier]
            category = structure_category(source["subtype"])
            row = {
                "identifier": identifier,
                "source_fold": fold["source_fold"],
                "virtual_fold": fold["virtual_fold"],
                "category": category,
                "dice": dice(prediction, target),
            }
            if source["contiguous"]:
                row["cldice"] = cldice(prediction, target)
            else:
                row["pq"] = pq(prediction, target)
            rows.append(row)
            print(json.dumps(row), flush=True)
    if not partial and len(rows) != 82:
        raise ValueError(f"Expected 82 held-out cases, got {len(rows)}")
    if partial and not rows:
        raise ValueError("No completed held-out cases to score")
    groups = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        names = ("dice", "cldice") if category.startswith("contiguous") else ("dice", "pq")
        groups[category] = {"n": len(subset)}
        groups[category].update({name: float(np.mean([row[name] for row in subset])) for name in names})
    result = {"method": method, "minimum_component_size": minimum, "partial": partial, "n": len(rows), "by_category": groups, "cases": rows}
    (work / ("scores_partial.json" if partial else "scores.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--minimum", type=int, default=9)
    parser.add_argument("--method", default="excluded_fold_no_tta")
    parser.add_argument("--partial", action="store_true", help="Score only completed cases; write scores_partial.json")
    args = parser.parse_args()
    result = score(args.work, args.labels_dir, args.manifest, args.minimum, args.method, args.partial)
    print(json.dumps(result["by_category"], indent=2), flush=True)


if __name__ == "__main__":
    main()
