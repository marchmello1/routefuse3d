"""Interpolate two same-architecture fold checkpoints for a gated experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def blend_states(first: dict, second: dict, alpha: float) -> dict:
    if set(first) != set(second):
        raise ValueError("Checkpoint parameter keys differ")
    blended = {}
    for name in first:
        left, right = first[name], second[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Tensor incompatible: {name}")
        if left.is_floating_point():
            value = (1 - alpha) * left.float() + alpha * right.float()
            blended[name] = value.to(left.dtype)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"Non-floating buffer differs: {name}")
            blended[name] = left.clone()
    return blended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    if args.output.exists():
        raise FileExistsError(args.output)
    a = torch.load(args.first, map_location="cpu", weights_only=False)
    b = torch.load(args.second, map_location="cpu", weights_only=False)
    if a.get("fold") != b.get("fold"):
        raise ValueError("Sources are from different folds")
    state = blend_states(a["model"], b["model"], args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "args": b.get("args", {}),
            "fold": a.get("fold"),
            "epoch": None,
            "weight_soup": {
                "first": str(args.first),
                "second": str(args.second),
                "alpha_second": args.alpha,
            },
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output), "fold": a.get("fold"), "tensors": len(state)}))


if __name__ == "__main__":
    main()
