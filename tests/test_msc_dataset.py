from pathlib import Path

import numpy as np

from msc_dataset import MSCPatchDataset
from msc_targets import structural_targets


def test_signed_source_normalization_and_shape(tmp_path: Path):
    source = np.zeros((32, 32, 32), dtype=np.float32)
    source[:16] = 8.0
    np.save(tmp_path / "sample.npy", source)
    dataset = MSCPatchDataset(tmp_path, samples_per_epoch=1, crop_size=32)
    data = dataset[0]["data"].numpy()
    assert data.shape == (1, 32, 32, 32)
    assert np.isfinite(data).all()
    assert data.min() < 0 < data.max()
    assert -5 <= data.min() and data.max() <= 20


def test_multiscale_has_fixed_output(tmp_path: Path):
    np.save(tmp_path / "sample.npy", np.ones((20, 20, 20), dtype=np.float32))
    dataset = MSCPatchDataset(tmp_path, samples_per_epoch=1, crop_size=16, multiscale=True)
    assert dataset[0]["data"].shape == (1, 16, 16, 16)


def test_worker_targets_match_the_existing_target_function(tmp_path: Path):
    rng = np.random.default_rng(7)
    np.save(tmp_path / "sample.npy", rng.normal(size=(24, 24, 24)).astype(np.float32))
    dataset = MSCPatchDataset(
        tmp_path, samples_per_epoch=1, crop_size=16, targets_in_workers=True
    )
    sample = dataset[0]
    expected = structural_targets(sample["data"][0].numpy())
    np.testing.assert_array_equal(sample["targets"].numpy(), expected)
