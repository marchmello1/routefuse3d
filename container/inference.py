from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage

from model import RoutedResidualUNet, strip_distributed_prefix


INPUT = Path("/input/images/light-sheet-3d-microscopy")
OUTPUT = Path("/output/images/isolated-biological-structure")
RESOURCE_DIR = Path(os.environ.get("SELMA_FALLBACK_RESOURCE_DIR", "/opt/app/resources"))
ATTACHED_MODEL_DIR = Path("/opt/ml/model")


@dataclass
class Runtime:
    models: list[RoutedResidualUNet]
    device: torch.device
    config: dict


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    stride = max(1, int(round((array.size / 1_000_000) ** (1 / 3))))
    sampled = array[::stride, ::stride, ::stride]
    nonzero = sampled[sampled != 0]
    values = nonzero if nonzero.size >= 100 else sampled.reshape(-1)
    low, high = np.percentile(values, (0.5, 99.5))
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0, 1).astype(np.float32)


def initialize() -> Runtime:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "roi": 128,
        "overlap": 0.25,
        "sw_batch_size": 2,
        "channels_last": True,
        "shared_weight": 0.65,
        "route_threshold": 0.5,
        "density_threshold": 0.5,
        "use_strata": True,
        "isolated_threshold": 0.5,
        "contiguous_threshold": 0.5,
        "isolated_minimum_size": 1,
        "contiguous_minimum_size": 1,
        "contiguous_closing": 0,
    }
    # Grand Challenge mounts its optional model volume at /opt/ml/model. That
    # mount hides files baked at the same path, even when no separate model was
    # uploaded. Keep the self-contained submission weights under /opt/app and
    # use an attached model only when it actually contains checkpoints.
    attached = sorted(ATTACHED_MODEL_DIR.glob("*.pt"))
    model_dir = ATTACHED_MODEL_DIR if attached else RESOURCE_DIR
    config_path = model_dir / "runtime.json"
    if config_path.exists():
        config.update(json.loads(config_path.read_text()))
    checkpoints = attached if attached else sorted(RESOURCE_DIR.glob("*.pt"))
    if not checkpoints:
        raise RuntimeError(
            "No .pt checkpoints found in attached or embedded model paths: "
            f"{ATTACHED_MODEL_DIR}, {RESOURCE_DIR}"
        )
    hybrid_count = int(config.get("hybrid_reference_members", 0))
    if hybrid_count:
        reference_prefix = str(config.get("hybrid_reference_prefix", "a_reference_"))
        candidate_prefix = str(config.get("hybrid_candidate_prefix", "b_candidate_"))
        if len(checkpoints) != 2 * hybrid_count or not all(
            path.name.startswith(reference_prefix) for path in checkpoints[:hybrid_count]
        ) or not all(
            path.name.startswith(candidate_prefix) for path in checkpoints[hybrid_count:]
        ):
            raise RuntimeError(
                "Hybrid checkpoint order is invalid: expected reference members "
                "followed by candidate members with configured filename prefixes"
            )
    models = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {})
        base_channels = int(args.get("base_channels", 32)) if isinstance(args, dict) else 32
        source_state = strip_distributed_prefix(checkpoint["model"])
        has_density_head = any(
            key.startswith("density_head.") for key in source_state
        )
        model = RoutedResidualUNet(
            base_channels, with_density_head=has_density_head
        ).to(device)
        # Supervised-only checkpoints were produced before the SSL auxiliary
        # heads gained their current shape.  Those heads are never used by
        # segmentation inference, so load every shape-compatible tensor and
        # reject incompatibilities anywhere in the actual inference path.
        target_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        incompatible = sorted(
            key
            for key, value in source_state.items()
            if key not in target_state or value.shape != target_state[key].shape
        )
        allowed_prefixes = ("reconstruction_head.", "latent_predictor.")
        unexpected = [key for key in incompatible if not key.startswith(allowed_prefixes)]
        if unexpected:
            raise RuntimeError(f"Incompatible inference weights in {path}: {unexpected}")
        missing, unexpected_loaded = model.load_state_dict(compatible, strict=False)
        unsafe_missing = [key for key in missing if not key.startswith(allowed_prefixes)]
        if unsafe_missing or unexpected_loaded:
            raise RuntimeError(
                f"Unsafe partial checkpoint load for {path}: "
                f"missing={unsafe_missing}, unexpected={unexpected_loaded}"
            )
        model.eval()
        if bool(config.get("channels_last", True)) and device.type == "cuda":
            model.to(memory_format=torch.channels_last_3d)
        models.append(model)
    print(
        f"loaded_models={len(models)} model_dir={model_dir} "
        f"device={device} config={config}",
        flush=True,
    )
    return Runtime(models=models, device=device, config=config)


def infer_model(runtime: Runtime, model: RoutedResidualUNet, tensor: torch.Tensor):
    def predictor(patch: torch.Tensor) -> torch.Tensor:
        if bool(runtime.config.get("channels_last", True)) and patch.is_cuda:
            patch = patch.contiguous(memory_format=torch.channels_last_3d)
        output = model(patch)
        route = output.morphology[:, None, None, None, None].expand_as(output.shared)
        channels = [output.shared, output.isolated, output.contiguous]
        if output.strata is not None:
            channels.extend(output.strata[:, index : index + 1] for index in range(4))
        channels.append(route)
        if output.density is not None:
            density = output.density[:, None, None, None, None].expand_as(output.shared)
            channels.append(density)
        return torch.cat(channels, dim=1)

    with torch.inference_mode(), torch.autocast(
        device_type=runtime.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.device.type == "cuda",
    ):
        output = sliding_window_inference(
            tensor,
            roi_size=(int(runtime.config["roi"]),) * 3,
            sw_batch_size=int(runtime.config["sw_batch_size"]),
            predictor=predictor,
            overlap=float(runtime.config["overlap"]),
            mode="gaussian",
        )
    return output[0].float().cpu()


def infer_route_only(runtime: Runtime, model: RoutedResidualUNet, tensor: torch.Tensor):
    """Run the trained morphology/density heads without decoding a mask.

    Keeping MONAI's original tiling and Gaussian overlap makes the spatially
    averaged route logits comparable to those from full segmentation inference.
    """

    def predictor(patch: torch.Tensor) -> torch.Tensor:
        if bool(runtime.config.get("channels_last", True)) and patch.is_cuda:
            patch = patch.contiguous(memory_format=torch.channels_last_3d)
        bottleneck, _ = model.encode(patch)
        pooled = model.pool(bottleneck)
        spatial_shape = patch.shape[2:]
        morphology = model.morphology_head(pooled).reshape(-1, 1, 1, 1, 1)
        channels = [morphology.expand(-1, 1, *spatial_shape)]
        if model.density_head is not None:
            density = model.density_head(pooled).reshape(-1, 1, 1, 1, 1)
            channels.append(density.expand(-1, 1, *spatial_shape))
        return torch.cat(channels, dim=1)

    with torch.inference_mode(), torch.autocast(
        device_type=runtime.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.device.type == "cuda",
    ):
        output = sliding_window_inference(
            tensor,
            roi_size=(int(runtime.config["roi"]),) * 3,
            sw_batch_size=int(runtime.config["sw_batch_size"]),
            predictor=predictor,
            overlap=float(runtime.config["overlap"]),
            mode="gaussian",
        )
    return output[0].float().cpu()


def predict_category_only(runtime: Runtime, array: np.ndarray) -> str:
    """Use trained heads for routing, stopping only at a provably fixed vote."""
    started = time.perf_counter()
    tensor = torch.from_numpy(normalize(array)[None, None]).to(runtime.device)
    route_scores = []
    density_scores = []
    model_count = len(runtime.models)
    density_count = sum(model.density_head is not None for model in runtime.models)
    route_decision = None
    density_decision = None
    for model in runtime.models:
        output = infer_route_only(runtime, model, tensor)
        route_scores.append(float(torch.sigmoid(output[0].mean())))
        if model.density_head is not None:
            density_scores.append(float(torch.sigmoid(output[1].mean())))
        route_decision = forced_threshold_decision(
            route_scores, model_count, float(runtime.config["route_threshold"])
        )
        density_decision = (
            forced_threshold_decision(
                density_scores, density_count,
                float(runtime.config["density_threshold"]),
            ) if density_count else None
        )
        if route_decision is not None and (
            density_count == 0 or density_decision is not None
        ):
            break
    contiguous = (
        route_decision if route_decision is not None else
        float(np.mean(route_scores)) >= float(runtime.config["route_threshold"])
    )
    if density_scores:
        dense = (
            density_decision if density_decision is not None else
            float(np.mean(density_scores)) >= float(runtime.config["density_threshold"])
        )
        category = f"{'contiguous' if contiguous else 'isolated'}_{'dense' if dense else 'sparse'}"
    else:
        category = "contiguous" if contiguous else "isolated"
    print(
        f"route_only_category={category} route_scores={route_scores} "
        f"density_scores={density_scores} members={len(route_scores)}/{model_count} "
        f"seconds={time.perf_counter() - started:.2f}",
        flush=True,
    )
    return category


def forced_threshold_decision(
    observed_scores: list[float], total_members: int, threshold: float
) -> bool | None:
    """Return a threshold decision only when all unseen scores cannot flip it."""
    if total_members <= 0 or len(observed_scores) > total_members:
        raise ValueError("Invalid ensemble member counts")
    if len(observed_scores) == total_members:
        return float(np.mean(observed_scores)) >= threshold
    low = sum(observed_scores) / total_members
    high = (sum(observed_scores) + total_members - len(observed_scores)) / total_members
    epsilon = 1e-5
    if low >= threshold + epsilon:
        return True
    if high < threshold - epsilon:
        return False
    return None


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


def _predict_standard_with_category(runtime: Runtime, array: np.ndarray) -> tuple[np.ndarray, str]:
    started = time.perf_counter()
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    tensor = torch.from_numpy(normalize(array)[None, None]).to(runtime.device)
    accumulated = None
    strata_accumulated = None
    strata_models = 0
    route_scores = []
    density_scores = []
    for model in runtime.models:
        output = infer_model(runtime, model, tensor)
        segmentation = output[:3]
        accumulated = segmentation if accumulated is None else accumulated + segmentation
        has_strata = model.strata_head is not None
        route_index = 7 if has_strata else 3
        if has_strata:
            strata = output[3:7]
            strata_accumulated = (
                strata if strata_accumulated is None else strata_accumulated + strata
            )
            strata_models += 1
        route_scores.append(float(torch.sigmoid(output[route_index].mean())))
        if model.density_head is not None:
            density_scores.append(float(torch.sigmoid(output[route_index + 1].mean())))
    logits = accumulated / len(runtime.models)
    strata_logits = (
        strata_accumulated / strata_models if strata_models > 0 else None
    )
    route_score = float(np.mean(route_scores))
    contiguous = route_score >= float(runtime.config["route_threshold"])
    density_score = float(np.mean(density_scores)) if density_scores else None
    dense = (
        density_score >= float(runtime.config.get("density_threshold", 0.5))
        if density_score is not None
        else None
    )
    category = (
        f"{'contiguous' if contiguous else 'isolated'}_{'dense' if dense else 'sparse'}"
        if dense is not None
        else ("contiguous" if contiguous else "isolated")
    )
    if (
        strata_logits is not None
        and dense is not None
        and bool(runtime.config.get("use_strata", True))
    ):
        stratum_index = int(contiguous) * 2 + int(dense)
        expert = strata_logits[stratum_index]
    else:
        expert = logits[2] if contiguous else logits[1]
    morphology = "contiguous" if contiguous else "isolated"
    shared_key = f"{category}_shared_weight"
    shared_weight = float(
        runtime.config.get(
            shared_key,
            runtime.config.get(
                f"{morphology}_shared_weight", runtime.config["shared_weight"]
            ),
        )
    )
    probability = torch.sigmoid(shared_weight * logits[0] + (1 - shared_weight) * expert)
    threshold = float(
        runtime.config.get(
            f"{category}_threshold", runtime.config[f"{morphology}_threshold"]
        )
    )
    mask = probability.numpy() >= threshold
    closing = int(
        runtime.config.get(
            f"{category}_closing", runtime.config.get("contiguous_closing", 0)
        )
    )
    if contiguous and closing > 0:
        mask = ndimage.binary_closing(
            mask, iterations=closing
        )
    minimum = int(
        runtime.config.get(
            f"{category}_minimum_size",
            runtime.config[f"{morphology}_minimum_size"],
        )
    )
    mask = remove_small(mask, minimum)
    peak_gib = (
        torch.cuda.max_memory_allocated(runtime.device) / (1024**3)
        if runtime.device.type == "cuda"
        else 0.0
    )
    print(
        f"route_score={route_score:.4f} density_score={density_score} "
        f"category={category} "
        f"inference_seconds={time.perf_counter() - started:.2f} peak_memory_gib={peak_gib:.2f}",
        flush=True,
    )
    return mask.astype(np.uint8), category


def _predict_standard(runtime: Runtime, array: np.ndarray) -> np.ndarray:
    return _predict_standard_with_category(runtime, array)[0]


def predict(runtime: Runtime, array: np.ndarray) -> np.ndarray:
    """Optionally choose one fivefold family using a candidate's morphology head.

    The normal single-family path is unchanged. The hybrid path is experimental
    and must pass fivefold paired OOF and A10G runtime gates before submission.
    """
    reference_count = int(runtime.config.get("hybrid_reference_members", 0))
    if not reference_count:
        return _predict_standard(runtime, array)
    if reference_count < 1 or len(runtime.models) != 2 * reference_count:
        raise RuntimeError("Hybrid needs equal nonempty reference/candidate families")

    # Match the held-out audit's candidate-model route decision. The selector
    # forward is counted in runtime, then the chosen fivefold family is run.
    selector = runtime.models[reference_count]
    tensor = torch.from_numpy(normalize(array)[None, None]).to(runtime.device)
    selector_output = infer_model(runtime, selector, tensor)
    route_index = 7 if selector.strata_head is not None else 3
    route_score = float(torch.sigmoid(selector_output[route_index].mean()))
    contiguous = route_score >= float(runtime.config.get("hybrid_route_threshold", 0.45))
    chosen = (
        runtime.models[reference_count:] if contiguous
        else runtime.models[:reference_count]
    )
    # Hold the branch fixed after selecting a family. The full ensemble still
    # estimates sparse/dense density and applies its own frozen mask policy.
    selected_config = dict(runtime.config)
    selected_config.pop("hybrid_reference_members", None)
    selected_config["route_threshold"] = -0.1 if contiguous else 1.1
    print(f"hybrid_route_score={route_score:.4f} family={'candidate' if contiguous else 'reference'}", flush=True)
    return _predict_standard(Runtime(chosen, runtime.device, selected_config), array)


def run(runtime: Runtime) -> None:
    candidates = []
    for pattern in ("*.mha", "*.tif", "*.tiff"):
        candidates.extend(glob.glob(str(INPUT / pattern)))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one input image, found {len(candidates)}")
    source = sitk.ReadImage(candidates[0])
    prediction = predict(runtime, sitk.GetArrayFromImage(source))
    result = sitk.GetImageFromArray(prediction)
    result.CopyInformation(source)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result, str(OUTPUT / "segmentation.mha"), useCompression=True)


if __name__ == "__main__":
    run(initialize())
