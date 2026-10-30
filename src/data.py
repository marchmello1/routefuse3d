from __future__ import annotations

import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
from torch.utils.data import Dataset


def robust_bounds(array: np.ndarray) -> tuple[float, float]:
    # Exact percentiles over 300^3 voxels dominate CPU time while providing no
    # meaningful extra precision. A deterministic spatial sample preserves the
    # intensity distribution and makes both SSL and fine-tuning data loading
    # substantially faster.
    stride = max(1, int(round((array.size / 1_000_000) ** (1 / 3))))
    # Cast only the spatial sample: converting a uint16 raw TIFF crop bank to
    # float32 first needlessly copies the complete source volume every draw.
    sampled = np.asarray(array[::stride, ::stride, ::stride], dtype=np.float32)
    nonzero = sampled[sampled != 0]
    sample = nonzero if nonzero.size >= 100 else sampled.reshape(-1)
    lo, hi = np.percentile(sample, (0.5, 99.5))
    return float(lo), float(hi)


def normalize_with_bounds(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0, 1).astype(np.float32)


def robust_normalize(array: np.ndarray) -> np.ndarray:
    lo, hi = robust_bounds(array)
    return normalize_with_bounds(array, lo, hi)


def read_volume(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        # SSL normalizes a crop after sampling. Keep the source memory-mapped
        # so repeatedly drawing from 300-voxel raw banks does not copy it all.
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if path.name.endswith((".nii", ".nii.gz")):
        array = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
        return np.transpose(array, (2, 1, 0))
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def pad_to_shape(array: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    padding = []
    for current, wanted in zip(array.shape, shape):
        total = max(0, wanted - current)
        padding.append((total // 2, total - total // 2))
    if not any(before or after for before, after in padding):
        return array
    return np.pad(array, padding, mode="constant")


def random_crop(
    image: np.ndarray,
    target: np.ndarray | None,
    crop_size: tuple[int, int, int],
    foreground_probability: float = 0.75,
    foreground_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    image = pad_to_shape(image, crop_size)
    if target is not None:
        target = pad_to_shape(target, crop_size)
    if foreground_points is not None:
        # Cached points are authoritative: a failed foreground draw must be a
        # uniform crop, not a second foreground draw over the full 400³ mask.
        if len(foreground_points) and random.random() < foreground_probability:
            center = foreground_points[random.randrange(len(foreground_points))]
        else:
            center = np.array([random.randrange(n) for n in image.shape])
    elif target is not None and target.any() and random.random() < foreground_probability:
        points = np.argwhere(target > 0)
        center = points[random.randrange(len(points))]
    else:
        center = np.array([random.randrange(n) for n in image.shape])
    starts = []
    for c, n, width in zip(center, image.shape, crop_size):
        jitter = random.randint(-width // 4, width // 4)
        starts.append(max(0, min(n - width, int(c) - width // 2 + jitter)))
    slices = tuple(slice(s, s + w) for s, w in zip(starts, crop_size))
    return image[slices], None if target is None else target[slices]


def augment(image: np.ndarray, target: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    for axis in range(3):
        if random.random() < 0.5:
            image = np.flip(image, axis).copy()
            if target is not None:
                target = np.flip(target, axis).copy()
    if random.random() < 0.5:
        axes = random.choice(((0, 1), (0, 2), (1, 2)))
        k = random.randrange(4)
        image = np.rot90(image, k, axes).copy()
        if target is not None:
            target = np.rot90(target, k, axes).copy()
    if random.random() < 0.35:
        gamma = random.uniform(0.65, 1.55)
        image = np.power(np.clip(image, 0, 1), gamma)
    image = image * random.uniform(0.85, 1.15) + random.uniform(-0.08, 0.08)
    if random.random() < 0.25:
        image = gaussian_filter(image, random.uniform(0.35, 0.9))
    if random.random() < 0.35:
        image = image + np.random.normal(0, random.uniform(0.005, 0.04), image.shape)
    return np.clip(image, 0, 1).astype(np.float32), target


def subtype(path: Path) -> str:
    name = path.name.lower()
    for key in ("plaque", "nuclei", "cfos", "axon", "vessel"):
        if key in name:
            return key
    if any(key in name for key in ("nefh", "nefl", "_pv_")):
        return "axon"
    if "peripheral_nerve" in name:
        return "nerve"
    return "unknown"


def structure_category(name: str) -> str:
    """Map released structures to the four strata used by official ranking.

    The challenge proposal explicitly identifies nuclei/plaque and vessel/axon
    as dense/sparse examples. c-Fos cells are isolated sparse objects, while
    peripheral nerves share the sparse contiguous morphology of axons.
    """
    mapping = {
        "nuclei": "isolated_dense",
        "plaque": "isolated_sparse",
        "cfos": "isolated_sparse",
        "vessel": "contiguous_dense",
        "axon": "contiguous_sparse",
        "nerve": "contiguous_sparse",
    }
    if name not in mapping:
        raise ValueError(f"Unknown SELMA3D structure subtype: {name}")
    return mapping[name]


@dataclass(frozen=True)
class LabeledCase:
    image: Path
    target: Path
    contiguous: bool
    subtype: str


def discover_labeled_cases(root: Path) -> list[LabeledCase]:
    raw_files = sorted(p for p in root.rglob("*.mha") if "/raw/" in str(p))
    gt_files = sorted(p for p in root.rglob("*.mha") if "/gt/" in str(p))
    def canonical(stem: str) -> str:
        return stem.removesuffix("_gt").removesuffix("_raw")

    gt_by_stem = {canonical(p.stem): p for p in gt_files}
    cases = []
    for image in raw_files:
        target = gt_by_stem.get(canonical(image.stem))
        if target is None:
            continue
        cases.append(
            LabeledCase(
                image=image,
                target=target,
                contiguous="contiguous_structures" in str(image),
                subtype=subtype(image),
            )
        )
    return cases


def foreground_coordinates(target: np.ndarray, maximum: int = 100_000) -> np.ndarray:
    """Return the same C-order sampled foreground points without a huge Nx3 array."""
    flat = np.flatnonzero(target)
    if flat.size > maximum:
        flat = flat[:: int(np.ceil(flat.size / maximum))]
    if flat.size == 0:
        return np.empty((0, target.ndim), dtype=np.int32)
    return np.column_stack(np.unravel_index(flat, target.shape)).astype(
        np.int32, copy=False
    )


class LabeledPatchDataset(Dataset):
    def __init__(
        self,
        cases: list[LabeledCase],
        samples_per_epoch: int = 1200,
        crop_size: int = 128,
        cache_volumes: bool = True,
    ):
        self.cases = cases
        grouped: dict[str, list[LabeledCase]] = {}
        for case in cases:
            grouped.setdefault(case.subtype, []).append(case)
        # Uniform subtype sampling intentionally gives the four difficult
        # sparse structures more updates than the already-strong dense vessel
        # and nuclei strata. Official category balancing is applied during
        # validation/model selection, where it belongs.
        self.groups = [grouped[key] for key in sorted(grouped)]
        self.samples_per_epoch = samples_per_epoch
        self.crop_size = (crop_size,) * 3
        # The labeled set is only 82 volumes. Loading and normalizing it once in
        # the parent process lets Linux DataLoader workers share the arrays via
        # copy-on-write instead of repeatedly decoding MHA files every epoch.
        self.cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if cache_volumes:
            def load_case(case: LabeledCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                image = robust_normalize(read_volume(case.image))
                target = (read_volume(case.target) > 0).astype(np.float32)
                return image, target, foreground_coordinates(target)
            with ThreadPoolExecutor(max_workers=min(8, len(cases))) as pool:
                for case, arrays in zip(cases, pool.map(load_case, cases)):
                    self.cache[case.image] = arrays

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        case = random.choice(self.groups[index % len(self.groups)])
        if case.image in self.cache:
            image, target, foreground_points = self.cache[case.image]
        else:
            image = robust_normalize(read_volume(case.image))
            target = (read_volume(case.target) > 0).astype(np.float32)
            foreground_points = foreground_coordinates(target)
        image, target = random_crop(
            image,
            target,
            self.crop_size,
            foreground_points=foreground_points,
        )
        image, target = augment(image, target)
        return {
            "image": torch.from_numpy(image[None]),
            "target": torch.from_numpy(target[None].astype(np.float32)),
            "contiguous": torch.tensor(float(case.contiguous)),
            "dense": torch.tensor(
                float(structure_category(case.subtype).endswith("_dense"))
            ),
        }


class UnlabeledPatchDataset(Dataset):
    def __init__(
        self,
        root: Path,
        samples_per_epoch: int = 16000,
        crop_size: int = 128,
        manifest: Path | None = None,
        multiscale: bool = False,
        focus_prefix: str | None = None,
        focus_probability: float = 0.0,
    ):
        if manifest is None:
            self.paths = (
                sorted(root.rglob("*.nii.gz"))
                + sorted(root.rglob("*.mha"))
                + sorted(root.rglob("*.npy"))
            )
        else:
            self.paths = [Path(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not self.paths:
            raise RuntimeError(f"No unlabeled volumes found under {root}")
        grouped: dict[str, list[Path]] = {}
        for path in self.paths:
            parts = path.parts
            group = path.parent.name
            # Auxiliary raw-data banks may be nested below a biological group.
            # Group by that biological parent, not by an experimental bank name;
            # otherwise a new seed silently changes the sampling distribution.
            if "task2_patches_early300" in parts:
                marker_index = parts.index("task2_patches_early300")
                if marker_index + 1 < len(parts):
                    group = parts[marker_index + 1]
            for marker in ("isolated_structures", "contiguous_structures"):
                if marker in parts and parts.index(marker) + 1 < len(parts):
                    group = parts[parts.index(marker) + 1]
                    break
            grouped.setdefault(group, []).append(path)
        self.groups = [grouped[key] for key in sorted(grouped)]
        self.focus_groups = [
            group for group in self.groups
            if focus_prefix and any(focus_prefix in str(path) for path in group)
        ]
        if not 0.0 <= focus_probability <= 1.0:
            raise ValueError("focus_probability must be between 0 and 1")
        if focus_probability > 0 and not self.focus_groups:
            raise ValueError(f"No unlabeled groups match focus prefix: {focus_prefix}")
        self.focus_probability = focus_probability
        self.samples_per_epoch = samples_per_epoch
        self.crop_size = (crop_size,) * 3
        self.multiscale = multiscale
        # Each DataLoader worker gets its own copy. A corrupt/partially copied
        # compressed volume should not abort a multi-hour distributed run.
        self.unreadable: set[Path] = set()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _read_from_group(self, group: list[Path]) -> np.ndarray:
        last_error: OSError | EOFError | None = None
        for attempt in range(12):
            pool = group if attempt < 8 else random.choice(self.groups)
            choices = [path for path in pool if path not in self.unreadable]
            if not choices:
                continue
            path = random.choice(choices)
            try:
                return read_volume(path)
            except (OSError, EOFError) as error:
                last_error = error
                if path not in self.unreadable:
                    warnings.warn(
                        f"Skipping unreadable unlabeled volume {path}: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self.unreadable.add(path)
        raise RuntimeError("No readable unlabeled volume found after 12 attempts") from last_error

    @staticmethod
    def structural_targets(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate annotation-free geometry targets from an augmented crop.

        The distance channel captures global signal layout.  Tube/blob channels
        are coarse structure-tensor descriptors; calculating them on a 4x grid
        keeps loader cost bounded while retaining the non-local morphology that
        the downstream metrics reward.
        """
        # A global upper-tail threshold keeps the pseudo-signal occupancy near
        # three percent on dense acquisitions while the absolute floor remains
        # meaningful for genuinely sparse stacks.
        threshold = float(np.quantile(image, 0.97)) if image.size else 1.0
        foreground = image >= max(0.05, threshold)
        if foreground.any():
            distance = np.minimum(distance_transform_edt(~foreground), 24.0) / 24.0
        else:
            distance = np.ones_like(image, dtype=np.float32)

        coarse = image[::4, ::4, ::4].astype(np.float32, copy=False)
        gradients = np.gradient(gaussian_filter(coarse, 0.5))
        tensor = np.empty((*coarse.shape, 3, 3), dtype=np.float32)
        for row in range(3):
            for column in range(row, 3):
                value = gaussian_filter(gradients[row] * gradients[column], 0.75)
                tensor[..., row, column] = value
                tensor[..., column, row] = value
        eigenvalues = np.linalg.eigvalsh(tensor)
        denominator = eigenvalues[..., 2] + 1e-5
        tube = np.clip((eigenvalues[..., 1] - eigenvalues[..., 0]) / denominator, 0, 1)
        blob = np.clip(eigenvalues[..., 0] / denominator, 0, 1)
        scale = tuple(wanted / current for wanted, current in zip(image.shape, tube.shape))
        tube = zoom(tube, scale, order=1)[: image.shape[0], : image.shape[1], : image.shape[2]]
        blob = zoom(blob, scale, order=1)[: image.shape[0], : image.shape[1], : image.shape[2]]
        return (
            distance.astype(np.float32),
            tube.astype(np.float32),
            blob.astype(np.float32),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        groups = (
            self.focus_groups
            if self.focus_groups and random.random() < self.focus_probability
            else self.groups
        )
        source = self._read_from_group(groups[index % len(groups)])
        lo, hi = robust_bounds(source)
        # Half of the SSL crops are anchored near bright signal. This prevents
        # the overwhelmingly dark LSM background from dominating pretraining.
        coarse = normalize_with_bounds(source[::4, ::4, ::4], lo, hi)
        nonzero = coarse[coarse > 0]
        points = None
        if nonzero.size:
            cutoff = np.quantile(nonzero, 0.90)
            points = np.argwhere(coarse >= cutoff) * 4
            if len(points) > 25_000:
                points = points[:: int(np.ceil(len(points) / 25_000))]
        target_width = self.crop_size[0]
        scale = random.choice((1.0, 1.4, 1.875)) if self.multiscale else 1.0
        requested_width = int(round(target_width * scale))
        source_shape = tuple(min(size, requested_width) for size in source.shape)
        image, _ = random_crop(
            source,
            None,
            source_shape,
            foreground_probability=0.5,
            foreground_points=points,
        )
        image = normalize_with_bounds(image, lo, hi)
        if image.shape != self.crop_size:
            factor = tuple(target_width / width for width in image.shape)
            image = zoom(image, factor, order=1)
            image = image[:target_width, :target_width, :target_width]
        image, _ = augment(image, None)
        distance, tube, blob = self.structural_targets(image)
        return {
            "image": torch.from_numpy(image[None]),
            "structure": torch.from_numpy(np.stack((distance, tube, blob))),
        }


def stratified_fold(cases: list[LabeledCase], fold: int, folds: int = 5) -> tuple[list[LabeledCase], list[LabeledCase]]:
    grouped: dict[str, list[LabeledCase]] = {}
    for case in cases:
        grouped.setdefault(case.subtype, []).append(case)
    train, validation = [], []
    for group in grouped.values():
        for index, case in enumerate(sorted(group, key=lambda x: x.image.name)):
            (validation if index % folds == fold else train).append(case)
    return train, validation
