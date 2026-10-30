"""Create an experimental weight average from two same-fold checkpoints.

Only model weights are retained. This is not a training resumption checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import torch


def average_states(left: dict, right: dict, alpha: float) -> dict:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if left.keys() != right.keys():
        raise ValueError("model state keys differ")
    averaged = {}
    for key in left:
        a, b = left[key], right[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError(f"model state {key!r} is not a tensor")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"model state shape/dtype differs for {key!r}")
        if torch.is_floating_point(a):
            averaged[key] = torch.lerp(a, b, alpha)
        elif torch.equal(a, b):
            averaged[key] = a.clone()
        else:
            raise ValueError(f"non-floating model state differs for {key!r}")
    return averaged


def build_average(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    fold: int,
    left_epoch: int,
    right_epoch: int,
    alpha: float,
) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    for path, checkpoint, epoch in (
        (left_path, left, left_epoch),
        (right_path, right, right_epoch),
    ):
        if int(checkpoint["fold"]) != fold:
            raise ValueError(f"fold mismatch in {path}")
        if int(checkpoint["epoch"]) + 1 != epoch:
            raise ValueError(f"epoch mismatch in {path}")
    payload = {
        "fold": fold,
        "model": average_states(left["model"], right["model"], alpha),
        "source_epochs": [left_epoch, right_epoch],
        "alpha": alpha,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=output_path.name + ".", suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if output_path.exists():
            raise FileExistsError(output_path)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--left-epoch", type=int, required=True)
    parser.add_argument("--right-epoch", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    build_average(
        args.left, args.right, args.output, args.fold,
        args.left_epoch, args.right_epoch, args.alpha,
    )
    print(f"AVERAGED_CHECKPOINT {args.output}", flush=True)


if __name__ == "__main__":
    main()
