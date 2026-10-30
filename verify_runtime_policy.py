"""Score the packaged routing/post-processing policy on cached OOF logits.

``summarize_cv.py`` can calibrate route and density thresholds after its masks
were generated. Those masks cannot validate a policy that takes a different
expert branch. This audit decodes the cached logits again using ``runtime.json``
and evaluates only that single, explicit policy with the official metric calls.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.special import expit

from src.data import discover_labeled_cases, read_volume, stratified_fold, structure_category


def decode(
    logits: np.ndarray,
    route_score: float,
    density_score: float | None,
    config: dict,
) -> tuple[np.ndarray, str]:
    """Mirror container/inference.py's mask decisions for one OOF model."""
    contiguous = route_score >= float(config.get("route_threshold", 0.5))
    dense = (
        density_score >= float(config.get("density_threshold", 0.5))
        if density_score is not None
        else None
    )
    morphology = "contiguous" if contiguous else "isolated"
    category = f"{morphology}_{'dense' if dense else 'sparse'}" if dense is not None else morphology
    has_strata = logits.shape[0] >= 7
    if has_strata and dense is not None and config.get("use_strata", True):
        expert = logits[3 + int(contiguous) * 2 + int(dense)]
    else:
        expert = logits[2 if contiguous else 1]
    weight = float(config.get(
        f"{category}_shared_weight",
        config.get(f"{morphology}_shared_weight", config.get("shared_weight", 0.65)),
    ))
    threshold = float(config.get(
        f"{category}_threshold",
        config.get(f"{morphology}_threshold", 0.5),
    ))
    probability = expit(weight * logits[0].astype(np.float32)
                        + (1.0 - weight) * expert.astype(np.float32))
    mask = probability >= threshold
    closing = int(config.get(
        f"{category}_closing", config.get("contiguous_closing", 0)
    ))
    if contiguous and closing > 0:
        mask = ndimage.binary_closing(mask, iterations=closing)
    minimum = int(config.get(
        f"{category}_minimum_size",
        config.get(f"{morphology}_minimum_size", 1),
    ))
    if minimum > 1:
        labels, count = ndimage.label(mask)
        if count:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= minimum
            keep[0] = False
            mask = keep[labels]
    return mask.astype(np.uint8), category


def reusable_metric(
    saved: list[dict], case_name: str, contiguous_truth: bool,
    route_score: float, density_score: float | None, config: dict,
) -> dict | None:
    """Reuse a prior official-metric call only for the same decoding decisions.

    ``evaluate_cv.py`` used route=0.45 and density=0.5 by default. Its
    contiguous component sweep is special-cased by *ground truth* morphology,
    so conservatively decline reuse when the original route was wrong.
    """
    original_contiguous = route_score >= 0.45
    selected_contiguous = route_score >= float(config.get("route_threshold", 0.5))
    if original_contiguous != selected_contiguous or original_contiguous != contiguous_truth:
        return None
    original_dense = density_score >= 0.5 if density_score is not None else None
    selected_dense = (
        density_score >= float(config.get("density_threshold", 0.5))
        if density_score is not None else None
    )
    if original_dense != selected_dense:
        return None
    morphology = "contiguous" if selected_contiguous else "isolated"
    category = (
        f"{morphology}_{'dense' if selected_dense else 'sparse'}"
        if selected_dense is not None else morphology
    )
    if selected_contiguous and int(config.get(
        f"{category}_closing", config.get("contiguous_closing", 0)
    )):
        return None
    weight = float(config.get(
        f"{category}_shared_weight",
        config.get(f"{morphology}_shared_weight", config.get("shared_weight", 0.65)),
    ))
    threshold = float(config.get(
        f"{category}_threshold", config.get(f"{morphology}_threshold", 0.5),
    ))
    minimum = int(config.get(
        f"{category}_minimum_size", config.get(f"{morphology}_minimum_size", 1),
    ))
    for record in saved:
        if record.get("case") != case_name or bool(record.get("contiguous")) != contiguous_truth:
            continue
        # The saved evaluator may have been invoked with nondefault routing
        # thresholds. Its correctness flags reveal which branch it actually
        # chose, so decline reuse if that differs from our default assumption.
        if record.get("route_correct") is not True:
            continue
        if density_score is not None:
            true_dense = str(record.get("category", "")).endswith("_dense")
            density_correct = record.get("density_correct")
            if density_correct is None:
                continue
            saved_dense = true_dense if bool(density_correct) else not true_dense
            if saved_dense != original_dense:
                continue
        if bool(record.get("uses_strata")) != bool(
            density_score is not None and config.get("use_strata", True)
        ):
            continue
        if (abs(float(record["shared_weight"]) - weight) < 1e-8
                and abs(float(record["threshold"]) - threshold) < 1e-8
                and int(record["minimum_size"]) == minimum):
            return record
    return None


def pooled_official_metrics(records: list[dict]) -> dict:
    """Report the four challenge metric units, pooling within morphology."""
    isolated = [row for row in records if row["true_category"].startswith("isolated")]
    contiguous = [row for row in records if row["true_category"].startswith("contiguous")]
    if not isolated or not contiguous:
        raise ValueError("Both isolated and contiguous held-out cases are required")
    values = {
        "isolated_dice": float(np.mean([row["dice"] for row in isolated])),
        "isolated_pq": float(np.mean([row["topology"] for row in isolated])),
        "contiguous_dice": float(np.mean([row["dice"] for row in contiguous])),
        "contiguous_cldice": float(np.mean([row["topology"] for row in contiguous])),
    }
    return {
        "n_isolated": len(isolated),
        "n_contiguous": len(contiguous),
        **values,
        "four_metric_mean_proxy": float(np.mean(list(values.values()))),
    }


def audit(
    data: Path, cache: Path, config: dict, folds: list[int], cfos_dense: bool,
    reuse_metrics: bool = False,
) -> dict:
    # Import only in the metric-bearing path: the local unit-test venv does not
    # include Panoptica, while the remote validation env pins official 2.1.7.
    from evaluate_cv import dice, panoptic_quality
    from panoptica.metrics.cldice import _compute_centerline_dice_coefficient

    cases = discover_labeled_cases(data)
    records = []
    reused = 0
    for fold in folds:
        _, validation = stratified_fold(cases, fold)
        saved_by_case: dict[str, list[dict]] = defaultdict(list)
        if reuse_metrics:
            metrics_path = cache / f"fold{fold}_metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(metrics_path)
            for record in json.loads(metrics_path.read_text()):
                saved_by_case[record["case"]].append(record)
        for case_index, case in enumerate(validation, 1):
            path = cache / f"fold{fold}_auto_{case.image.stem}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as cached:
                route_score = float(cached["route_score"])
                density_score = (
                    float(cached["density_score"])
                    if "density_score" in cached.files else None
                )
                saved = reusable_metric(
                    saved_by_case[case.image.name], case.image.name, case.contiguous,
                    route_score, density_score, config,
                ) if reuse_metrics else None
                logits = cached["logits"] if saved is None else None
            if saved is None:
                prediction, predicted_category = decode(logits, route_score, density_score, config)
                target = (read_volume(case.target) > 0).astype(np.uint8)
                topology = (
                    float(_compute_centerline_dice_coefficient(target, prediction))
                    if case.contiguous else panoptic_quality(prediction, target)
                )
                score = dice(prediction, target)
            else:
                reused += 1
                predicted_category = (
                    f"{'contiguous' if case.contiguous else 'isolated'}_"
                    f"{'dense' if density_score >= float(config.get('density_threshold', 0.5)) else 'sparse'}"
                    if density_score is not None
                    else ('contiguous' if case.contiguous else 'isolated')
                )
                topology = float(saved["cldice" if case.contiguous else "pq"])
                score = float(saved["dice"])
            true_category = (
                "isolated_dense" if cfos_dense and case.subtype == "cfos"
                else structure_category(case.subtype)
            )
            records.append({
                "fold": fold, "case": case.image.name, "subtype": case.subtype,
                "true_category": true_category, "predicted_category": predicted_category,
                "dice": score, "topology": topology,
                "objective": (score + topology) / 2,
                "route_correct": predicted_category.startswith("contiguous") == case.contiguous,
                "reused_metric": saved is not None,
            })
            print(
                f"audited fold={fold} case={case_index}/{len(validation)} "
                f"name={case.image.name}",
                file=sys.stderr,
                flush=True,
            )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["true_category"]].append(record)
    categories = {
        key: {
            "n": len(items),
            "dice": float(np.mean([item["dice"] for item in items])),
            "topology": float(np.mean([item["topology"] for item in items])),
            "objective": float(np.mean([item["objective"] for item in items])),
        }
        for key, items in sorted(grouped.items())
    }
    return {
        "method": (
            "packaged_policy_on_cached_oof_logits_with_saved_metric_reuse"
            if reuse_metrics else "packaged_policy_on_cached_oof_logits"
        ),
        "cfos_dense_assumption": cfos_dense,
        "folds": folds,
        "n": len(records),
        "reused_metrics": reused,
        "route_accuracy": float(np.mean([item["route_correct"] for item in records])),
        "macro_category_objective": float(np.mean([item["objective"] for item in categories.values()])),
        "official_style_pooled": pooled_official_metrics(records),
        "categories": categories,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--cfos-dense", action="store_true")
    parser.add_argument("--reuse-metrics", action="store_true")
    args = parser.parse_args()
    if any(fold not in range(5) for fold in args.folds) or len(set(args.folds)) != len(args.folds):
        raise ValueError("folds must be unique integers in [0, 4]")
    config = json.loads(args.runtime.read_text())
    # Panoptica may print a citation banner during its first import. Keep
    # stdout machine-readable so redirected audit reports remain valid JSON.
    with contextlib.redirect_stdout(sys.stderr):
        result = audit(
            args.data, args.cache, config, args.folds, args.cfos_dense,
            args.reuse_metrics,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
