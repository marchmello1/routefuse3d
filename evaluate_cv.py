from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from panoptica import (
    ConnectedComponentsInstanceApproximator,
    InputType,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
)
from panoptica.metrics import Metric
from panoptica.metrics.cldice import _get_skeleton, cl_score
from scipy import ndimage

from src.data import (
    discover_labeled_cases,
    read_volume,
    robust_normalize,
    stratified_fold,
    structure_category,
)
from src.model import RoutedResidualUNet, strip_distributed_prefix


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = prediction.sum() + target.sum()
    return 1.0 if denominator == 0 else float(2 * np.logical_and(prediction, target).sum() / denominator)


def panoptic_quality(prediction: np.ndarray, target: np.ndarray) -> float:
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(cca_backend=None),
        instance_matcher=NaiveThresholdMatching(matching_threshold=0.3),
        instance_metrics=[Metric.DSC],
    )
    return float(evaluator.evaluate(prediction, target, verbose=False)["ungrouped"].pq_dsc)


def cldice_with_reference_skeleton(
    reference: np.ndarray, prediction: np.ndarray, reference_skeleton: np.ndarray
) -> float:
    """Panoptica's clDice formula with the invariant reference skeleton reused."""
    tprec = cl_score(prediction, reference_skeleton)
    tsens = cl_score(reference, _get_skeleton(prediction))
    return float(2 * tprec * tsens / (tprec + tsens))


def remove_small(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 1:
        return mask
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.reshape(-1))
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels]


def minimum_sizes_for_route(use_contiguous: bool, candidates: list[int]) -> list[int]:
    """Sweep the branch the inference container would actually take."""
    return [1] if use_contiguous else candidates


def predict_logits(
    model,
    image: np.ndarray,
    roi: int,
    overlap: float,
    has_density_head: bool,
    has_strata_head: bool,
) -> tuple[np.ndarray, float, float | None]:
    tensor = torch.from_numpy(image[None, None]).cuda()

    def predictor(patch: torch.Tensor) -> torch.Tensor:
        output = model(patch)
        route = output.morphology[:, None, None, None, None].expand_as(output.shared)
        channels = [output.shared, output.isolated, output.contiguous]
        if has_strata_head:
            channels.extend(output.strata[:, index : index + 1] for index in range(4))
        channels.append(route)
        if has_density_head:
            density = output.density[:, None, None, None, None].expand_as(output.shared)
            channels.append(density)
        return torch.cat(channels, dim=1)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = sliding_window_inference(
            tensor,
            roi_size=(roi, roi, roi),
            sw_batch_size=2,
            predictor=predictor,
            overlap=overlap,
            mode="gaussian",
        )
    route_index = 7 if has_strata_head else 3
    density_index = route_index + 1
    route_score = float(torch.sigmoid(logits[0, route_index].mean()).cpu())
    density_score = (
        float(torch.sigmoid(logits[0, density_index].mean()).cpu())
        if has_density_head
        else None
    )
    segmentation_channels = 7 if has_strata_head else 3
    return (
        logits[0, :segmentation_channels].float().cpu().numpy(),
        route_score,
        density_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--roi", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--route-threshold", type=float, default=0.45)
    parser.add_argument("--density-threshold", type=float, default=0.5)
    parser.add_argument("--disable-strata", action="store_true")
    parser.add_argument("--auto-route", action="store_true")
    parser.add_argument("--shared-weights", type=float, nargs="+", default=[0.50, 0.65, 0.80])
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.35, 0.45, 0.55, 0.65],
    )
    parser.add_argument("--minimum-sizes", type=int, nargs="+", default=[1, 5, 17, 33])
    args = parser.parse_args()
    cases = discover_labeled_cases(args.data)
    _, validation = stratified_fold(cases, args.fold)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_base = (
        checkpoint_args.get("base_channels", 32)
        if isinstance(checkpoint_args, dict)
        else 32
    )
    base_channels = int(
        args.base_channels if args.base_channels is not None else checkpoint_base
    )
    state = strip_distributed_prefix(checkpoint["model"])
    has_density_head = any(key.startswith("density_head.") for key in state)
    has_strata_head = any(key.startswith("strata_head.") for key in state)
    model = RoutedResidualUNet(
        base_channels, with_density_head=has_density_head
    ).cuda().eval()
    # Supervised-only diagnostics created before the structural SSL head was
    # introduced have a one-channel reconstruction head. It is unused during
    # segmentation evaluation, so discard only those incompatible tensors.
    for key in ("reconstruction_head.weight", "reconstruction_head.bias"):
        if key in state and state[key].shape != model.state_dict()[key].shape:
            del state[key]
    model.load_state_dict(state, strict=False)
    args.cache.mkdir(parents=True, exist_ok=True)

    records = []
    for case in validation:
        route_tag = "auto" if args.auto_route else "oracle"
        cache_path = args.cache / f"fold{args.fold}_{route_tag}_{case.image.stem}.npz"
        target = (read_volume(case.target) > 0).astype(np.uint8)
        if cache_path.exists():
            cached = np.load(cache_path)
            logits = cached["logits"]
            route_score = float(cached["route_score"])
            density_score = (
                float(cached["density_score"])
                if "density_score" in cached.files
                else None
            )
        else:
            logits, route_score, density_score = predict_logits(
                model,
                robust_normalize(read_volume(case.image)),
                args.roi,
                args.overlap,
                has_density_head,
                has_strata_head,
            )
            payload = {
                "logits": logits.astype(np.float16),
                "route_score": np.asarray(route_score, dtype=np.float32),
            }
            if density_score is not None:
                payload["density_score"] = np.asarray(density_score, dtype=np.float32)
            np.savez_compressed(cache_path, **payload)
        use_contiguous = (
            route_score >= args.route_threshold if args.auto_route else case.contiguous
        )
        reference_skeleton = _get_skeleton(target) if case.contiguous else None
        expert_channel = 2 if use_contiguous else 1
        use_strata = (
            has_strata_head
            and density_score is not None
            and not args.disable_strata
        )
        if use_strata:
            use_dense = density_score >= args.density_threshold
            stratum_index = int(use_contiguous) * 2 + int(use_dense)
            expert_channel = 3 + stratum_index
        for shared_weight in args.shared_weights:
            blended = shared_weight * logits[0] + (1 - shared_weight) * logits[expert_channel]
            probability = 1.0 / (1.0 + np.exp(-blended.astype(np.float32)))
            for threshold in args.thresholds:
                raw = probability >= threshold
                minimum_sizes = minimum_sizes_for_route(use_contiguous, args.minimum_sizes)
                for minimum in minimum_sizes:
                    prediction = remove_small(raw, minimum).astype(np.uint8)
                    record = {
                        "case": case.image.name,
                        "subtype": case.subtype,
                        "category": structure_category(case.subtype),
                        "contiguous": case.contiguous,
                        "shared_weight": shared_weight,
                        "threshold": threshold,
                        "minimum_size": minimum,
                        "route_score": route_score,
                        "density_score": density_score,
                        "density_correct": (
                            (density_score >= args.density_threshold)
                            == structure_category(case.subtype).endswith("_dense")
                            if density_score is not None
                            else None
                        ),
                        "uses_strata": use_strata,
                        "route_correct":
                            (route_score >= args.route_threshold) == case.contiguous,
                        "dice": dice(prediction, target),
                    }
                    if case.contiguous:
                        record["cldice"] = cldice_with_reference_skeleton(
                            target, prediction, reference_skeleton
                        )
                    else:
                        record["pq"] = panoptic_quality(prediction, target)
                    records.append(record)
                    print(json.dumps(record), flush=True)
    output_path = args.cache / f"fold{args.fold}_metrics.json"
    output_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
