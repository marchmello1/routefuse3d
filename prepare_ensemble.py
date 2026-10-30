from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def load_summary(path: Path) -> dict:
    text = path.read_text()
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"No JSON summary found in {path}")
    return json.loads(text[start:])


def model_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_candidates(directory: Path, fold: int, prefer_ema: bool = False) -> list[Path]:
    # Held-out evaluation uses *_best.pt. Export that snapshot first so a
    # one-member-per-fold package contains the same weights as the OOF audit.
    preferred = directory / f"fold{fold}_ema.pt"
    candidates = [directory / f"fold{fold}_best.pt", directory / f"fold{fold}_last.pt"]
    if prefer_ema:
        candidates.insert(0, preferred)
    candidates.extend(
        sorted(directory.glob(f"fold{fold}_epoch*.pt"), reverse=True)
    )
    return [path for path in candidates if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select distinct fold snapshots and export an inference ensemble"
    )
    parser.add_argument(
        "--checkpoint-dir-template",
        required=True,
        help="Directory template containing {fold}, for example /runs/task1_ft_f{fold}",
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--contiguous-postprocess", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members-per-fold", type=int, default=2)
    parser.add_argument(
        "--prefer-ema", action="store_true",
        help="Select supervised EMA first; use only after a paired OOF evaluation",
    )
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(5)),
        help="Held-out folds to export; default is the complete five-fold ensemble",
    )
    args = parser.parse_args()

    if args.members_per_fold < 1:
        raise ValueError("members-per-fold must be positive")
    if not args.folds or len(set(args.folds)) != len(args.folds) or any(
        fold not in range(5) for fold in args.folds
    ):
        raise ValueError("folds must be distinct integers in [0, 4]")
    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("*.pt"):
        stale.unlink()

    manifest: list[dict] = []
    global_fingerprints: dict[str, tuple[int, Path]] = {}
    for fold in args.folds:
        directory = Path(args.checkpoint_dir_template.format(fold=fold))
        selected: list[tuple[Path, dict, str]] = []
        seen: set[str] = set()
        for source in checkpoint_candidates(directory, fold, args.prefer_ema):
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            if "model" not in checkpoint:
                continue
            checkpoint_fold = checkpoint.get("fold")
            if checkpoint_fold is not None and int(checkpoint_fold) != fold:
                raise RuntimeError(
                    f"Checkpoint fold mismatch: expected {fold}, found "
                    f"{checkpoint_fold} in {source}"
                )
            fingerprint = model_fingerprint(checkpoint["model"])
            if fingerprint in seen:
                continue
            if fingerprint in global_fingerprints:
                previous_fold, previous_path = global_fingerprints[fingerprint]
                raise RuntimeError(
                    "Identical model weights found in different folds: "
                    f"fold {previous_fold} ({previous_path}) and fold {fold} "
                    f"({source})"
                )
            seen.add(fingerprint)
            global_fingerprints[fingerprint] = (fold, source)
            selected.append((source, checkpoint, fingerprint))
            if len(selected) == args.members_per_fold:
                break
        if not selected:
            raise RuntimeError(f"No usable checkpoint found for fold {fold} in {directory}")
        for member, (source, checkpoint, fingerprint) in enumerate(selected):
            destination = args.output / f"fold{fold}_member{member}.pt"
            torch.save(
                {
                    "model": checkpoint["model"],
                    "args": checkpoint.get("args", {}),
                    "fold": fold,
                    "epoch": checkpoint.get("epoch"),
                },
                destination,
            )
            manifest.append(
                {
                    "fold": fold,
                    "member": member,
                    "epoch": checkpoint.get("epoch"),
                    "source": str(source),
                    "fingerprint": fingerprint,
                    "file": destination.name,
                }
            )

    summary = load_summary(args.summary)
    isolated = summary["recommended"]["isolated"]
    contiguous = summary["recommended"]["contiguous"]
    if args.contiguous_postprocess:
        sweep = json.loads(args.contiguous_postprocess.read_text())
        if not sweep.get("best"):
            raise RuntimeError("Contiguous post-processing sweep has no candidates")
        contiguous = {**contiguous, **sweep["best"][0]}
    route = summary.get("route_calibration", {})
    density = summary.get("density_calibration") or {}
    runtime = {
        "roi": 128,
        "overlap": 0.25,
        "sw_batch_size": 2,
        "route_threshold": float(route.get("threshold", 0.5)),
        "density_threshold": float(density.get("threshold", 0.5)),
        "isolated_shared_weight": float(isolated["shared_weight"]),
        "contiguous_shared_weight": float(contiguous["shared_weight"]),
        "isolated_threshold": float(isolated["threshold"]),
        "contiguous_threshold": float(contiguous["threshold"]),
        "isolated_minimum_size": int(isolated["minimum_size"]),
        "contiguous_minimum_size": int(contiguous["minimum_size"]),
        "contiguous_closing": int(contiguous.get("closing", 0)),
    }
    if density and summary.get("recommended_by_category"):
        for category, settings in summary["recommended_by_category"].items():
            runtime[f"{category}_shared_weight"] = float(settings["shared_weight"])
            runtime[f"{category}_threshold"] = float(settings["threshold"])
            runtime[f"{category}_minimum_size"] = int(settings["minimum_size"])
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (args.output / "ensemble_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"members": len(manifest), "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
