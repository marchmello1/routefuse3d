"""On-demand SELMA patch loader for structural-completion pretraining.

Unlike the current hybrid SSL loader, this keeps z-scored intensities signed.
It reads the already downloaded NIfTI patches and creates no second copy.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from msc_targets import normalize_source, structural_targets


def read_volume(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        return np.transpose(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32), (2, 1, 0))
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def random_crop(source: np.ndarray, width: int, center: np.ndarray | None = None) -> np.ndarray:
    padding = [(max(0, width - n) // 2, max(0, width - n + 1) // 2) for n in source.shape]
    if any(left or right for left, right in padding):
        source = np.pad(source, padding, mode="constant")
    if center is not None:
        center = center + np.array([left for left, _ in padding])
        starts = [
            max(0, min(n - width, int(c) - width // 2 + random.randint(-width // 4, width // 4)))
            for c, n in zip(center, source.shape)
        ]
    else:
        starts = [random.randrange(n - width + 1) for n in source.shape]
    return source[tuple(slice(start, start + width) for start in starts)]


class MSCPatchDataset(Dataset):
    def __init__(
        self,
        root: Path,
        samples_per_epoch: int,
        crop_size: int = 160,
        manifest: Path | None = None,
        multiscale: bool = False,
        targets_in_workers: bool = False,
    ):
        if manifest is not None:
            self.paths = [Path(x) for x in manifest.read_text().splitlines() if x.strip()]
        else:
            self.paths = (
                sorted(root.rglob("*.nii.gz"))
                + sorted(root.rglob("*.mha"))
                + sorted(root.rglob("*.npy"))
            )
        if not self.paths:
            raise ValueError(f"No source patches under {root}")
        groups: dict[str, list[Path]] = {}
        for path in self.paths:
            parts = path.parts
            structure = path.parent.name
            for marker in ("isolated_structures", "contiguous_structures"):
                if marker in parts and parts.index(marker) + 1 < len(parts):
                    structure = parts[parts.index(marker) + 1]
                    break
            groups.setdefault(structure, []).append(path)
        self.groups = [groups[key] for key in sorted(groups)]
        self.samples_per_epoch = samples_per_epoch
        self.crop_size = crop_size
        self.multiscale = multiscale
        self.targets_in_workers = targets_in_workers

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = random.choice(self.groups[index % len(self.groups)])
        source = read_volume(path)
        width = self.crop_size
        scale = random.choice((1.0, 1.4, 1.875)) if self.multiscale else 1.0
        source_width = int(round(width * scale))
        center = None
        if random.random() < 0.5:
            coarse = source[::4, ::4, ::4]
            mean, std = float(source.mean()), max(float(source.std()), 1e-3)
            signal = np.argwhere((coarse - mean) / std > 2.0)
            if len(signal):
                center = signal[random.randrange(len(signal))] * 4
        crop = random_crop(source, source_width, center=center)
        crop = normalize_source(source, crop)
        if source_width != width:
            factor = (width / source_width,) * 3
            crop = zoom(crop, factor, order=1)[:width, :width, :width]
        for axis in range(3):
            if random.random() < 0.5:
                crop = np.flip(crop, axis)
        if random.random() < 0.5:
            axes = random.choice(((0, 1), (0, 2), (1, 2)))
            crop = np.rot90(crop, random.randrange(4), axes)
        crop = np.ascontiguousarray(crop, dtype=np.float32)
        sample = {"data": torch.from_numpy(crop[None])}
        if self.targets_in_workers:
            sample["targets"] = torch.from_numpy(structural_targets(crop))
        return sample
