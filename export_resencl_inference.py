"""Export nnU-Net segmentation weights without training-only optimizer state.

The custom SELMA trainers change initialization and schedule, not the inherited
network builder. The export uses the stock trainer name for inference, and must
pass a same-case voxel-equivalence test before it is used in a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch


REQUIRED_KEYS = {
    "network_weights",
    "trainer_name",
    "init_args",
    "inference_allowed_mirroring_axes",
}


def export(
    source: Path, destination: Path, folds: tuple[int, ...],
    file_ending: str = ".nii.gz",
    checkpoint_name: str = "checkpoint_final.pth",
    include_best: bool = False,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if not folds or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
        raise ValueError("Fold list must contain unique fold IDs from 0 through 4")
    plans = source / "plans.json"
    dataset = source / "dataset.json"
    for path in (plans, dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
        json.loads(path.read_text())
    if file_ending not in (".nii.gz", ".mha"):
        raise ValueError(f"Unsupported inference file ending: {file_ending}")
    if checkpoint_name not in ("checkpoint_final.pth", "checkpoint_best.pth"):
        raise ValueError(f"Unsupported source checkpoint: {checkpoint_name}")
    if include_best and checkpoint_name != "checkpoint_final.pth":
        raise ValueError("include_best requires checkpoint_final.pth as the primary member")
    plan_data = json.loads(plans.read_text())
    if file_ending == ".mha" and plan_data.get("image_reader_writer") != "SimpleITKIO":
        raise ValueError("Direct MHA export requires SimpleITKIO training plans")
    members = (
        [(source_fold, source_checkpoint, virtual_fold)
         for virtual_fold, (source_fold, source_checkpoint) in enumerate(
             (fold, variant)
             for fold in folds
             for variant in ("checkpoint_final.pth", "checkpoint_best.pth")
         )]
        if include_best else [(fold, checkpoint_name, fold) for fold in folds]
    )
    for source_fold, source_checkpoint, _ in members:
        path = source / f"fold_{source_fold}" / source_checkpoint
        if not path.is_file():
            raise FileNotFoundError(path)

    destination.mkdir(parents=True)
    shutil.copyfile(plans, destination / plans.name)
    dataset_data = json.loads(dataset.read_text())
    dataset_data["file_ending"] = file_ending
    (destination / dataset.name).write_text(json.dumps(dataset_data, indent=2) + "\n")
    source_trainers: set[str] = set()
    axes_seen: set[tuple[int, ...] | None] = set()
    for source_fold, source_checkpoint, virtual_fold in members:
        path = source / f"fold_{source_fold}" / source_checkpoint
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        missing = REQUIRED_KEYS - checkpoint.keys()
        if missing:
            raise ValueError(f"{path} missing {sorted(missing)}")
        trainer = str(checkpoint["trainer_name"])
        if not trainer.startswith("nnUNetTrainerSelma"):
            raise ValueError(f"Unexpected trainer {trainer} in {path}")
        source_trainers.add(trainer)
        if checkpoint["init_args"]["configuration"] != "3d_fullres":
            raise ValueError(f"Unexpected configuration in {path}")
        weights = checkpoint["network_weights"]
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Empty network weights in {path}")
        axes = checkpoint["inference_allowed_mirroring_axes"]
        axes_seen.add(None if axes is None else tuple(axes))
        if len(axes_seen) > 1:
            raise ValueError("Ensemble checkpoints disagree on mirror axes")
        payload = {
            "network_weights": weights,
            "trainer_name": "nnUNetTrainer",
            "init_args": checkpoint["init_args"],
            "inference_allowed_mirroring_axes": checkpoint["inference_allowed_mirroring_axes"],
        }
        out = destination / f"fold_{virtual_fold}"
        out.mkdir()
        torch.save(payload, out / "checkpoint_final.pth")
    manifest = {
        "source_trainers": sorted(source_trainers),
        "inference_trainer": "nnUNetTrainer",
        "folds": [virtual_fold for _, _, virtual_fold in members],
        "member_sources": [
            {"virtual_fold": virtual_fold, "source_fold": source_fold,
             "source_checkpoint": source_checkpoint}
            for source_fold, source_checkpoint, virtual_fold in members
        ],
        "source_checkpoint": checkpoint_name if not include_best else "final+best",
        "file_ending": file_ending,
        "equivalence_tested": False,
    }
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--file-ending", choices=(".nii.gz", ".mha"), default=".nii.gz")
    parser.add_argument("--checkpoint-name", choices=("checkpoint_final.pth", "checkpoint_best.pth"),
                        default="checkpoint_final.pth")
    parser.add_argument("--include-best", action="store_true",
                        help="Add best alongside final for every source fold (10 members for 5 folds)")
    args = parser.parse_args()
    export(args.source, args.destination, tuple(args.folds), args.file_ending,
           args.checkpoint_name, args.include_best)


if __name__ == "__main__":
    main()
