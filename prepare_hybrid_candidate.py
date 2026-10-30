"""Assemble two independently audited fivefold ensembles for routed inference.

This is a packaging step, not a claim that the hybrid improves held-out scores.
The caller must first inspect the complete paired OOF and runtime audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_runtime(reference: dict, candidate: dict, members: int) -> dict:
    if members < 1:
        raise ValueError("Hybrid requires at least one member per family")
    for key in ("roi", "overlap", "sw_batch_size", "density_threshold"):
        if key in reference and key in candidate and reference[key] != candidate[key]:
            raise ValueError(f"Incompatible family runtime setting: {key}")
    merged = dict(reference)
    for key, value in candidate.items():
        if key.startswith("contiguous_"):
            merged[key] = value
    merged.update({
        "hybrid_reference_members": members,
        "hybrid_reference_prefix": "a_reference_",
        "hybrid_candidate_prefix": "b_candidate_",
        "hybrid_route_threshold": 0.45,
    })
    return merged


def family_members(directory: Path, expected: int) -> list[Path]:
    manifest = json.loads((directory / "ensemble_manifest.json").read_text())
    if len(manifest) != expected or sorted(int(row["fold"]) for row in manifest) != list(range(expected)):
        raise ValueError(f"Expected exactly one checkpoint per fold in {directory}")
    files = []
    for fold in range(expected):
        row = next(row for row in manifest if int(row["fold"]) == fold)
        source = directory / row["file"]
        if not source.is_file() or source.suffix != ".pt":
            raise ValueError(f"Missing checkpoint {source}")
        files.append(source)
    return files


def assemble(reference: Path, candidate: Path, output: Path, members: int = 5) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reference_files = family_members(reference, members)
    candidate_files = family_members(candidate, members)
    ref_config = json.loads((reference / "runtime.json").read_text())
    cand_config = json.loads((candidate / "runtime.json").read_text())
    runtime = merge_runtime(ref_config, cand_config, members)
    output.mkdir(parents=True)
    manifest = []
    for label, files in (("a_reference_", reference_files), ("b_candidate_", candidate_files)):
        for fold, source in enumerate(files):
            destination = output / f"{label}fold{fold}.pt"
            shutil.copy2(source, destination)
            manifest.append({
                "family": label.removesuffix("_"),
                "fold": fold,
                "file": destination.name,
                "sha256": sha256(destination),
                "source": str(source),
            })
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (output / "hybrid_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"members": len(manifest), "runtime": runtime}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(assemble(args.reference, args.candidate, args.output, args.members), indent=2))


if __name__ == "__main__":
    main()
