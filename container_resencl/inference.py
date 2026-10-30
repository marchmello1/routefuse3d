"""Grand Challenge inference adapter for a validated nnU-Net ResEnc-L export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


INPUT = Path("/input/images/light-sheet-3d-microscopy")
OUTPUT = Path("/output/images/isolated-biological-structure")
EMBEDDED_MODEL = Path(os.environ.get("SELMA_SPECIALIST_RESOURCE_DIR", "/opt/app/resources"))
ATTACHED_MODEL = Path("/opt/ml/model")


def _binary_mask(prediction: np.ndarray) -> np.ndarray:
    mask = np.asarray(prediction > 0, dtype=np.uint8)
    minimum = int(os.environ.get("SELMA_MIN_COMPONENT_SIZE", "1"))
    if minimum < 1:
        raise ValueError("SELMA_MIN_COMPONENT_SIZE must be positive")
    if minimum == 1:
        return mask
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    keep[0] = False
    return keep[labels].astype(np.uint8)


def _model_directory() -> Path:
    # An attached model mount may be present but empty. In that case, use the
    # checkpoints embedded in the image rather than failing on the mount.
    if (ATTACHED_MODEL / "plans.json").is_file():
        return ATTACHED_MODEL
    return EMBEDDED_MODEL


def initialize() -> nnUNetPredictor:
    model = _model_directory()
    manifest_path = model / "export_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing inference export manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    member_mode = os.environ.get("SELMA_ENSEMBLE_MEMBERS", "all")
    if member_mode not in ("all", "final_only", "best_only"):
        raise ValueError(f"Invalid SELMA_ENSEMBLE_MEMBERS: {member_mode}")
    mirroring_setting = os.environ.get("SELMA_MIRRORING", "0")
    array_setting = os.environ.get("SELMA_ARRAY_API", "0")
    if mirroring_setting not in ("0", "1"):
        raise ValueError(f"Invalid SELMA_MIRRORING: {mirroring_setting}")
    if array_setting not in ("0", "1"):
        raise ValueError(f"Invalid SELMA_ARRAY_API: {array_setting}")
    minimum = int(os.environ.get("SELMA_MIN_COMPONENT_SIZE", "1"))
    if minimum < 1:
        raise ValueError("SELMA_MIN_COMPONENT_SIZE must be positive")
    folds = tuple(int(f) for f in manifest["folds"])
    if not folds or len(set(folds)) != len(folds):
        raise RuntimeError("Inference export has no distinct folds")
    if member_mode != "all":
        sources = manifest.get("member_sources")
        if not isinstance(sources, list) or len(sources) != len(folds):
            raise RuntimeError("Ensemble selection requires a complete member_sources manifest")
        checkpoint = (
            "checkpoint_final.pth" if member_mode == "final_only"
            else "checkpoint_best.pth"
        )
        folds = tuple(
            int(source["virtual_fold"])
            for source in sources if source["source_checkpoint"] == checkpoint
        )
        if not folds or len(set(folds)) != len(folds):
            raise RuntimeError(f"No distinct {member_mode} members in export manifest")
    fold_setting = os.environ.get("SELMA_ENSEMBLE_FOLDS", "all")
    if fold_setting != "all":
        try:
            requested = tuple(int(part) for part in fold_setting.split(","))
        except ValueError as exc:
            raise ValueError(f"Invalid SELMA_ENSEMBLE_FOLDS: {fold_setting}") from exc
        if not requested or len(set(requested)) != len(requested):
            raise ValueError(f"Invalid SELMA_ENSEMBLE_FOLDS: {fold_setting}")
        if not set(requested).issubset(folds):
            raise ValueError(f"Requested folds {requested} are not in selected {member_mode} members {folds}")
        folds = requested
    for fold in folds:
        if not (model / f"fold_{fold}" / "checkpoint_final.pth").is_file():
            raise RuntimeError(f"Missing fold {fold} checkpoint in {model}")
    if not (model / "dataset.json").is_file() or not (model / "plans.json").is_file():
        raise RuntimeError(f"Incomplete nnU-Net model metadata in {model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=float(os.environ.get("SELMA_TILE_STEP", "0.5")),
        use_gaussian=True,
        use_mirroring=mirroring_setting == "1",
        perform_everything_on_device=os.environ.get("SELMA_ALL_ON_DEVICE") == "1",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model), use_folds=folds, checkpoint_name="checkpoint_final.pth"
    )
    selected_axes = os.environ.get("SELMA_MIRROR_AXES")
    if selected_axes is not None and selected_axes != "all":
        if not predictor.use_mirroring:
            raise ValueError("SELMA_MIRROR_AXES requires SELMA_MIRRORING=1")
        axes = tuple(int(part) for part in selected_axes.split(","))
        if len(set(axes)) != len(axes) or not axes or any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError(f"Invalid SELMA_MIRROR_AXES: {selected_axes}")
        allowed = predictor.allowed_mirroring_axes
        if allowed is None or not set(axes).issubset(set(allowed)):
            raise ValueError(f"Mirror axes {axes} not allowed by checkpoint {allowed}")
        predictor.allowed_mirroring_axes = axes
    print(
        f"loaded_resencl_folds={folds} ensemble_mode={member_mode} "
        f"mirroring={predictor.use_mirroring} axes={predictor.allowed_mirroring_axes} "
        f"array_api={array_setting} min_component={minimum} "
        f"model={model} device={device}", flush=True,
    )
    return predictor


def run(
    predictor: nnUNetPredictor,
    *,
    source: sitk.Image | None = None,
    source_path: Path | None = None,
    output_name: str | None = None,
) -> None:
    images = sorted(
        path for pattern in ("*.mha", "*.tif", "*.tiff")
        for path in INPUT.glob(pattern)
    )
    if len(images) != 1:
        raise RuntimeError(f"Expected one supported image in {INPUT}, got {len(images)}")
    expected_path = images[0]
    if source_path is not None and source_path != expected_path:
        raise RuntimeError("Provided source path differs from challenge input")
    source_path = expected_path
    destination_name = output_name or (
        source_path.name if source_path.suffix.lower() == ".mha" else "segmentation.mha"
    )
    if Path(destination_name).name != destination_name or not destination_name.endswith(".mha"):
        raise ValueError(f"Output name must be a plain .mha filename: {destination_name}")
    if source is None:
        source = sitk.ReadImage(str(source_path))
    if source.GetDimension() != 3:
        raise RuntimeError(f"Expected 3D image, got {source.GetDimension()}D")
    started = time.perf_counter()
    if os.environ.get("SELMA_ARRAY_API") == "1":
        # Use the same SimpleITK voxel order and spacing convention as
        # nnU-Net's SimpleITKIO, but avoid a second disk read and its
        # multiprocessing file-export path for a single challenge case.
        array = sitk.GetArrayFromImage(source)[None]
        properties = {"spacing": [abs(v) for v in source.GetSpacing()[::-1]]}
        result = predictor.predict_single_npy_array(
            array, properties, output_file_truncated=None,
            save_or_return_probabilities=False,
        )
        if result.shape != array.shape[1:]:
            raise RuntimeError("Array prediction and input voxel grids differ")
        mask = sitk.GetImageFromArray(_binary_mask(result))
        mask.CopyInformation(source)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        destination = OUTPUT / destination_name
        sitk.WriteImage(mask, str(destination), useCompression=True)
        print(
            f"prediction={destination} voxels={np.prod(source.GetSize())} "
            f"seconds={time.perf_counter() - started:.2f} mode=array_api",
            flush=True,
        )
        return
    file_ending = predictor.dataset_json.get("file_ending")
    if file_ending not in (".mha", ".nii.gz"):
        raise RuntimeError(f"Unsupported model file ending: {file_ending}")
    with TemporaryDirectory(prefix="selma_resencl_") as temporary:
        work = Path(temporary)
        predicted_dir = work / "predicted"
        predicted_dir.mkdir()
        name = source_path.stem
        if file_ending == ".mha":
            input_dir = work / "input"
            input_dir.mkdir()
            # nnU-Net's case naming requires _0000 for channel zero. A
            # symlink avoids copying the potentially large original MHA.
            nnunet_input = input_dir / f"{name}_0000.mha"
            if source_path.suffix.lower() == ".mha":
                nnunet_input.symlink_to(source_path)
            else:
                sitk.WriteImage(source, str(nnunet_input), useCompression=True)
        else:
            input_dir = work / "input"
            input_dir.mkdir()
            sitk.WriteImage(source, str(input_dir / f"{name}_0000.nii.gz"), useCompression=True)
        if os.environ.get("SELMA_SEQUENTIAL") == "1":
            predictor.predict_from_files_sequential(
                str(input_dir), str(predicted_dir),
                save_probabilities=False, overwrite=True,
            )
        else:
            predictor.predict_from_files(
                str(input_dir), str(predicted_dir), save_probabilities=False,
                overwrite=True, num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
        predicted_path = predicted_dir / f"{name}{file_ending}"
        if not predicted_path.is_file():
            raise RuntimeError(f"nnU-Net produced no prediction: {predicted_path}")
        predicted = sitk.ReadImage(str(predicted_path))
        if predicted.GetSize() != source.GetSize():
            raise RuntimeError("Prediction and input voxel grids differ")
        for attr in ("GetSpacing", "GetOrigin", "GetDirection"):
            reference = np.asarray(getattr(source, attr)(), dtype=np.float64)
            observed = np.asarray(getattr(predicted, attr)(), dtype=np.float64)
            if not np.allclose(reference, observed, rtol=0, atol=1e-3):
                raise RuntimeError(f"Prediction geometry differs at {attr}")
        mask = sitk.GetImageFromArray(_binary_mask(sitk.GetArrayFromImage(predicted)))
        mask.CopyInformation(source)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        destination = OUTPUT / destination_name
        sitk.WriteImage(mask, str(destination), useCompression=True)
    print(
        f"prediction={destination} voxels={np.prod(source.GetSize())} "
        f"seconds={time.perf_counter() - started:.2f}", flush=True,
    )
