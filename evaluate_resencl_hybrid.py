"""Out-of-fold gate for a ResEnc-L isolated expert plus routed contiguous model.

This script never evaluates a training case with its own fold checkpoint.  It
compares the existing routed model, the isolated expert, and the hybrid on
the exact same held-out images before any submission image is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from panoptica.metrics.cldice import _get_skeleton

from evaluate_cv import cldice_with_reference_skeleton, dice, panoptic_quality, remove_small
from src.data import read_volume, structure_category
from src.hybrid_policy import (
    hybrid_prediction,
    minimum_size_for_route,
    three_way_hybrid_prediction,
)
from src.specialist_paths import prediction_path


def load_prediction(path: Path, reference: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    target = sitk.ReadImage(str(reference))
    if image.GetSize() != target.GetSize():
        raise ValueError(f"size mismatch: {path}, {reference}")
    if not np.allclose(image.GetSpacing(), target.GetSpacing(), atol=1e-4):
        raise ValueError(f"spacing mismatch: {path}, {reference}")
    if not np.allclose(image.GetOrigin(), target.GetOrigin(), atol=1e-3):
        raise ValueError(f"origin mismatch: {path}, {reference}")
    if not np.allclose(image.GetDirection(), target.GetDirection(), atol=1e-5):
        raise ValueError(f"direction mismatch: {path}, {reference}")
    return sitk.GetArrayFromImage(image) > 0


def routed_prediction_with_density(
    cache_path: Path, threshold: float, shared_weight: float
) -> tuple[np.ndarray, bool, bool | None]:
    with np.load(cache_path) as cached:
        logits = cached["logits"].astype(np.float32)
        route = float(cached["route_score"]) >= 0.45
        density = float(cached["density_score"]) >= 0.5 if "density_score" in cached.files else None
    expert = (
        3 + int(route) * 2 + int(density)
        if logits.shape[0] >= 7 and density is not None
        else (2 if route else 1)
    )
    blended = shared_weight * logits[0] + (1.0 - shared_weight) * logits[expert]
    return blended >= np.log(threshold / (1.0 - threshold)), route, density


def routed_prediction(
    cache_path: Path, threshold: float, shared_weight: float
) -> tuple[np.ndarray, bool]:
    """Backward-compatible morphology-only decoder for existing diagnostics."""
    prediction, route, _ = routed_prediction_with_density(
        cache_path, threshold, shared_weight
    )
    return prediction, route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--routed-cache", type=Path, required=True)
    parser.add_argument("--nnunet-results", type=Path)
    parser.add_argument("--nnunet-template", type=str)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--shared-weight", type=float, default=0.5)
    parser.add_argument("--minimum-size", type=int, default=17)
    parser.add_argument("--all-structures", action="store_true", help="Score contiguous Dice/clDice as well")
    parser.add_argument("--identifiers", nargs="+", help="Only these held-out identifiers (parallel scoring)")
    args = parser.parse_args()
    if (args.nnunet_results is None) == (args.nnunet_template is None):
        parser.error("provide exactly one of --nnunet-results or --nnunet-template")

    manifest = {row["identifier"]: row for row in json.loads(args.manifest.read_text())}
    splits = json.loads(args.splits.read_text())
    requested = set(args.identifiers or [])
    held_out = {identifier for fold in args.folds for identifier in splits[fold]["val"]}
    if requested and not requested <= held_out:
        parser.error(f"not held out by selected folds: {sorted(requested - held_out)}")
    records: list[dict] = []
    for fold in args.folds:
        for identifier in splits[fold]["val"]:
            if requested and identifier not in requested:
                continue
            case = manifest[identifier]
            contiguous = bool(case["contiguous"])
            if contiguous and not args.all_structures:
                continue
            reference = Path(case["target"])
            routed_cache = args.routed_cache / f"fold{fold}_auto_{Path(case['image']).stem}.npz"
            nnunet_path = prediction_path(
                args.nnunet_results, args.nnunet_template, fold, identifier
            )
            if not routed_cache.is_file() or not nnunet_path.is_file():
                raise FileNotFoundError(f"missing OOF prediction: {routed_cache} or {nnunet_path}")
            target = read_volume(reference) > 0
            routed, predicted_contiguous, predicted_dense = routed_prediction_with_density(
                routed_cache, args.threshold, args.shared_weight
            )
            minimum = minimum_size_for_route(predicted_contiguous, args.minimum_size)
            routed = remove_small(routed, minimum)
            specialist = remove_small(load_prediction(nnunet_path, reference), minimum)
            hybrid = hybrid_prediction(routed, specialist, predicted_contiguous)
            hybrid3 = three_way_hybrid_prediction(
                routed, specialist, predicted_contiguous, predicted_dense
            )
            record = {
                "fold": fold,
                "identifier": identifier,
                "case": Path(case["image"]).name,
                "category": structure_category(case["subtype"]),
                "contiguous": contiguous,
                "route_correct": predicted_contiguous == contiguous,
                "predicted_dense": predicted_dense,
            }
            reference_skeleton = _get_skeleton(target) if contiguous else None
            for name, prediction in (
                ("routed", routed), ("specialist", specialist),
                ("hybrid", hybrid), ("hybrid3", hybrid3),
            ):
                record[f"{name}_dice"] = dice(prediction, target)
                if contiguous:
                    record[f"{name}_cldice"] = cldice_with_reference_skeleton(
                        target, prediction, reference_skeleton
                    )
                else:
                    record[f"{name}_pq"] = panoptic_quality(prediction, target)
            records.append(record)
            print(json.dumps(record), flush=True)
    if not records:
        raise RuntimeError("No isolated validation cases were evaluated")
    summary = {}
    for name in ("routed", "specialist", "hybrid", "hybrid3"):
        summary[name] = {}
        for contiguous, label, metrics in (
            (False, "isolated", ("dice", "pq")),
            (True, "contiguous", ("dice", "cldice")),
        ):
            subset = [row for row in records if row["contiguous"] == contiguous]
            if subset:
                summary[name][label] = {
                    metric: float(np.mean([row[f"{name}_{metric}"] for row in subset]))
                    for metric in metrics
                }
                summary[name][label]["n"] = len(subset)
    # The challenge ranks sparse and dense isolated structures separately.
    # A pooled case mean can hide regressions in the smaller stratum.
    by_category = {}
    for category in sorted({row["category"] for row in records}):
        subset = [row for row in records if row["category"] == category]
        by_category[category] = {"n": len(subset)}
        for name in ("routed", "specialist", "hybrid", "hybrid3"):
            metric_names = ("dice", "cldice") if subset[0]["contiguous"] else ("dice", "pq")
            metrics = {
                metric: float(np.mean([row[f"{name}_{metric}"] for row in subset]))
                for metric in metric_names
            }
            metrics["objective"] = float(np.mean(list(metrics.values())))
            by_category[category][name] = metrics
    result = {"summary": summary, "by_category": by_category, "cases": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
