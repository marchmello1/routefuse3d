from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2


def read_tiff(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not decode TIFF: {path}")
    return image


def complete_stacks(root: Path, expected: Path) -> list[list[Path]]:
    grouped: dict[Path, list[tuple[Path, int]]] = defaultdict(list)
    for line in expected.read_text().splitlines():
        relative, size = line.split("\t")
        path = root / relative
        grouped[path.parent].append((path, int(size)))
    stacks = []
    for records in grouped.values():
        records.sort(key=lambda item: item[0].name)
        if all(path.exists() and path.stat().st_size == size for path, size in records):
            stacks.append([path for path, _ in records])
    return sorted(stacks, key=lambda files: str(files[0].parent))


def complete_runs(root: Path, expected: Path, minimum_depth: int) -> list[list[Path]]:
    """Use only size-verified, consecutive TIFF planes from in-flight stacks.

    A partially downloaded full volume is not itself usable. A run of at least
    ``minimum_depth`` verified adjacent planes is a valid 3D crop source and
    keeps SSL pretraining independent of download completion.
    """
    grouped: dict[Path, list[tuple[int, Path, int]]] = defaultdict(list)
    for line in expected.read_text().splitlines():
        relative, size = line.split("\t")
        path = root / relative
        match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
        if match is None:
            continue
        grouped[path.parent].append((int(match.group(1)), path, int(size)))
    runs: list[list[Path]] = []
    for records in grouped.values():
        records.sort(key=lambda item: item[0])
        current: list[Path] = []
        previous: int | None = None
        for index, path, size in records:
            valid = path.is_file() and path.stat().st_size == size
            adjacent = previous is None or index == previous + 1
            if not valid or not adjacent:
                if len(current) >= minimum_depth:
                    runs.append(current)
                current = []
            if valid:
                current.append(path)
                previous = index
            else:
                previous = None
        if len(current) >= minimum_depth:
            runs.append(current)
    return sorted(runs, key=lambda files: str(files[0]))


def crop_stack(
    files: list[Path],
    output: Path,
    patch: int,
    count: int,
    seed: int,
    minimum_depth: int,
) -> int:
    # Axon SCAPE acquisitions are laterally large but contain fewer than 300
    # z-planes. Preserve their native depth rather than dropping this entire
    # 2026 structure class. The downstream loader accepts variable shapes and
    # draws/resamples cubic training crops from the available field of view.
    depth = min(patch, len(files))
    if depth < minimum_depth:
        return 0
    first = read_tiff(files[0])
    height, width = first.shape[-2:]
    if min(height, width) < patch:
        return 0
    rng = np.random.default_rng(seed)
    z_blocks = min(max(1, len(files) // depth), 16)
    z_starts = np.linspace(0, len(files) - depth, z_blocks, dtype=int)
    xy_per_block = math.ceil(count / z_blocks)
    written = 0
    stack_tag = hashlib.sha1(str(files[0]).encode()).hexdigest()[:10]
    for z_start in z_starts:
        coordinates = [
            (
                int(rng.integers(0, height - patch + 1)),
                int(rng.integers(0, width - patch + 1)),
            )
            for _ in range(xy_per_block)
        ]
        crops = np.empty((len(coordinates), depth, patch, patch), dtype=first.dtype)
        for dz, path in enumerate(files[z_start : z_start + depth]):
            plane = read_tiff(path)
            for index, (y, x) in enumerate(coordinates):
                crops[index, dz] = plane[y : y + patch, x : x + patch]
        for index, crop in enumerate(crops):
            if written >= count:
                break
            sample = crop[::4, ::4, ::4].astype(np.float32)
            lo, hi = np.percentile(sample, (1, 99))
            if hi <= lo or np.count_nonzero(sample) < sample.size * 0.002:
                continue
            destination = output / f"{stack_tag}_z{z_start:05d}_{index:03d}.npy"
            np.save(destination, crop, allow_pickle=False)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop complete TIFF stacks into SSL-ready 3D patches")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch", type=int, default=160)
    parser.add_argument("--minimum-depth", type=int, default=64)
    parser.add_argument("--patches-per-stack", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-partial-runs", action="store_true")
    parser.add_argument("--max-stacks", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stacks = (
        complete_runs(args.root, args.expected, args.minimum_depth)
        if args.allow_partial_runs
        else complete_stacks(args.root, args.expected)
    )
    if args.max_stacks > 0:
        stacks = stacks[: args.max_stacks]
    print(f"complete_stacks={len(stacks)}", flush=True)
    total = 0
    for index, files in enumerate(stacks):
        total += crop_stack(
            files,
            args.output,
            args.patch,
            args.patches_per_stack,
            args.seed + index,
            args.minimum_depth,
        )
        print(f"stack={index + 1}/{len(stacks)} patches={total}", flush=True)
    print(f"wrote={total} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
