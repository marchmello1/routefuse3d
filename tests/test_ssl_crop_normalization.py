import numpy as np
import pytest

import src.data as data

from src.data import UnlabeledPatchDataset, normalize_with_bounds, read_volume, robust_bounds, robust_normalize


def test_source_bounds_then_crop_matches_full_normalization():
    rng = np.random.default_rng(7)
    source = rng.gamma(shape=1.5, scale=2.0, size=(67, 73, 61)).astype(np.float32)
    source[rng.random(source.shape) < 0.18] = 0
    lo, hi = robust_bounds(source)
    full = robust_normalize(source)
    crop = (slice(11, 45), slice(17, 61), slice(9, 49))
    np.testing.assert_array_equal(
        normalize_with_bounds(source[crop], lo, hi), full[crop]
    )
    np.testing.assert_array_equal(
        normalize_with_bounds(source[::4, ::4, ::4], lo, hi),
        full[::4, ::4, ::4],
    )


def test_constant_source_stays_zero_after_crop():
    source = np.zeros((40, 44, 48), dtype=np.float32)
    lo, hi = robust_bounds(source)
    assert not normalize_with_bounds(source[4:20, 8:24, 12:28], lo, hi).any()


def test_uint16_memmap_bounds_and_crop_match_eager_float32(tmp_path):
    rng = np.random.default_rng(17)
    raw = rng.integers(0, 65535, size=(65, 69, 71), dtype=np.uint16)
    raw[::4] = 0
    path = tmp_path / "raw.npy"
    np.save(path, raw)
    mapped = read_volume(path)
    assert isinstance(mapped, np.memmap)
    assert mapped.dtype == np.uint16
    eager = raw.astype(np.float32)
    assert robust_bounds(mapped) == robust_bounds(eager)
    crop = (slice(7, 54), slice(8, 60), slice(5, 65))
    lo, hi = robust_bounds(mapped)
    np.testing.assert_array_equal(
        normalize_with_bounds(mapped[crop], lo, hi),
        normalize_with_bounds(eager[crop], lo, hi),
    )


def test_nested_raw_bank_keeps_biological_group(tmp_path):
    bank = tmp_path / "task2_patches_early300"
    paths = [
        bank / "axon" / "original.npy",
        bank / "axon" / "extra_seed2027" / "additional.npy",
        bank / "vessel" / "original.npy",
    ]
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(map(str, paths)) + "\n")
    dataset = UnlabeledPatchDataset(
        tmp_path, manifest=manifest, focus_prefix="task2_patches_early300",
        focus_probability=0.65,
    )
    assert sorted(len(group) for group in dataset.groups) == [1, 2]
    assert len(dataset.focus_groups) == 2


def test_corrupt_unlabeled_volume_is_skipped_within_group(tmp_path, monkeypatch):
    bad = tmp_path / "bad.nii.gz"
    good = tmp_path / "good.nii.gz"
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{bad}\n{good}\n")
    dataset = UnlabeledPatchDataset(tmp_path, manifest=manifest)
    calls = []

    def fake_read(path):
        calls.append(path)
        if path == bad:
            raise OSError("CRC check failed")
        return np.ones((2, 2, 2), dtype=np.float32)

    monkeypatch.setattr(data, "read_volume", fake_read)
    monkeypatch.setattr(data.random, "choice", lambda values: values[0])
    with pytest.warns(RuntimeWarning, match="Skipping unreadable"):
        np.testing.assert_array_equal(
            dataset._read_from_group(dataset.groups[0]), np.ones((2, 2, 2))
        )
    assert calls == [bad, good]
    dataset._read_from_group(dataset.groups[0])
    assert calls == [bad, good, good]
