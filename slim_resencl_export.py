"""Create a self-contained ResEnc-L export with only the selected final folds.

This does not alter the source export or a submitted Docker image. The output
is published only after all copied checkpoint hashes match their originals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def make_slim_export(source: Path, target: Path) -> dict:
    source = source.resolve(strict=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    manifest = json.loads((source / "export_manifest.json").read_text())
    members = manifest["member_sources"]
    if set(manifest["folds"]) != {row["virtual_fold"] for row in members}:
        raise ValueError("Source manifest fold IDs do not match member sources")
    finals = [row for row in members if row["source_checkpoint"] == "checkpoint_final.pth"]
    if len(finals) != 5 or len({row["source_fold"] for row in finals}) != 5:
        raise ValueError("Expected five distinct final source folds")
    for name in ("plans.json", "dataset.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for name in ("plans.json", "dataset.json"):
            shutil.copy2(source / name, temporary / name)
        hashes = {}
        for member in finals:
            fold = int(member["virtual_fold"])
            relative = Path(f"fold_{fold}/checkpoint_final.pth")
            original = source / relative
            if not original.is_file():
                raise FileNotFoundError(original)
            destination = temporary / relative
            destination.parent.mkdir()
            shutil.copy2(original, destination)
            expected = sha256(original)
            actual = sha256(destination)
            if actual != expected:
                raise RuntimeError(f"Checkpoint copy differs: {relative}")
            hashes[str(relative)] = actual
        slim_manifest = dict(manifest)
        slim_manifest["folds"] = [int(row["virtual_fold"]) for row in finals]
        slim_manifest["member_sources"] = finals
        slim_manifest["source_checkpoint"] = "final"
        slim_manifest["slimmed_from_member_count"] = len(members)
        (temporary / "export_manifest.json").write_text(
            json.dumps(slim_manifest, indent=2) + "\n"
        )
        (temporary / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
        os.rename(temporary, target)
        return {"folds": slim_manifest["folds"], "checkpoint_sha256": hashes}
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    result = make_slim_export(args.source, args.target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
