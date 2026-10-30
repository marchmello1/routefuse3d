from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members", type=int, default=3)
    args = parser.parse_args()
    candidates = [args.input / f"fold{args.fold}_last.pt"]
    candidates.extend(
        sorted(args.input.glob(f"fold{args.fold}_epoch*.pt"), reverse=True)
    )
    selected = []
    seen = set()
    for path in candidates:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        key = fingerprint(state)
        if key in seen:
            continue
        seen.add(key)
        selected.append((path, checkpoint))
        if len(selected) == args.members:
            break
    if len(selected) < 2:
        raise RuntimeError(f"Need at least two distinct snapshots, found {len(selected)}")
    keys = set(selected[0][1]["model"])
    if any(set(checkpoint["model"]) != keys for _, checkpoint in selected[1:]):
        raise RuntimeError("Snapshot state dictionaries do not match")
    averaged = {}
    for key in sorted(keys):
        tensors = [checkpoint["model"][key].detach().cpu() for _, checkpoint in selected]
        if tensors[0].is_floating_point():
            value = torch.stack([tensor.float() for tensor in tensors]).mean(0)
            averaged[key] = value.to(tensors[0].dtype)
        else:
            averaged[key] = tensors[0]
    latest = selected[0][1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": averaged,
            "args": latest.get("args", {}),
            "fold": args.fold,
            "epoch": latest.get("epoch"),
            "averaged_sources": [str(path) for path, _ in selected],
        },
        args.output,
    )
    print(f"wrote={args.output} sources={[path.name for path, _ in selected]}")


if __name__ == "__main__":
    main()
