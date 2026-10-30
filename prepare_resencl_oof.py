"""Prepare immutable per-fold input lists for excluded-fold ResEnc-L inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(images_dir: Path, splits_path: Path, target: Path) -> dict:
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    splits = json.loads(splits_path.read_text())
    if len(splits) != 5:
        raise ValueError("Expected exactly five folds")
    validation_ids = [identifier for fold in splits for identifier in fold["val"]]
    if len(validation_ids) != 82 or len(set(validation_ids)) != 82:
        raise ValueError("Expected 82 distinct held-out cases")
    missing = [
        identifier for identifier in validation_ids
        if not (images_dir / f"{identifier}_0000.nii.gz").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing source images: {missing}")
    target.mkdir(parents=True)
    plan = {"split_sha256": sha256(splits_path), "folds": []}
    for source_fold, fold in enumerate(splits):
        input_dir = target / f"fold_{source_fold}" / "input"
        input_dir.mkdir(parents=True)
        for identifier in fold["val"]:
            source = (images_dir / f"{identifier}_0000.nii.gz").resolve()
            (input_dir / source.name).symlink_to(source)
        plan["folds"].append({
            "source_fold": source_fold,
            "virtual_fold": 2 * source_fold,
            "validation_ids": fold["val"],
        })
    (target / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    plan = prepare(args.images_dir, args.splits, args.target)
    print(f"OOF_PREPARED split_sha256={plan['split_sha256']} cases=82 folds=5")


if __name__ == "__main__":
    main()
