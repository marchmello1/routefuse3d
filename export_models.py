from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inference-only fold checkpoints")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(args.checkpoint):
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if "model" not in checkpoint:
            raise RuntimeError(f"Missing model state: {source}")
        slim = {
            "model": checkpoint["model"],
            "args": checkpoint.get("args", {}),
            "fold": checkpoint.get("fold", index),
            "epoch": checkpoint.get("epoch"),
        }
        # Keep multiple checkpoints from the same fold (for example the best
        # and final states) as separate ensemble members instead of silently
        # overwriting the first export.
        destination = args.output / (
            f"member{index:02d}_fold{slim['fold']}_{source.stem}.pt"
        )
        torch.save(slim, destination)
        print(f"exported {source} -> {destination}")
    if args.runtime:
        config = json.loads(args.runtime.read_text())
        (args.output / "runtime.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
