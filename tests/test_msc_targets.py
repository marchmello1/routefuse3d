import numpy as np

from msc_targets import normalize_source, structural_targets, symmetric_eigenvalues


def test_eigenvalues_match_reference():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(1000, 3, 3))
    matrix += matrix.transpose(0, 2, 1)
    actual = np.stack(
        symmetric_eigenvalues(
            matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2],
            matrix[:, 1, 1], matrix[:, 1, 2], matrix[:, 2, 2],
        ), axis=1,
    )
    np.testing.assert_allclose(actual, np.linalg.eigvalsh(matrix), atol=1e-9)


def test_distance_target_and_empty_crop():
    image = np.zeros((16, 16, 16), dtype=np.float32)
    empty = structural_targets(image)
    assert empty.shape == (4, 16, 16, 16)
    assert np.all(empty[1] == 1)
    image[8, 8, 8] = 4
    target = structural_targets(image)
    assert target[1, 8, 8, 8] == 0
    np.testing.assert_allclose(target[1, 8, 8, 9], 1 / 24)
    assert np.isfinite(target).all()


def test_source_normalization():
    source = np.arange(64, dtype=np.float32)
    result = normalize_source(source, np.array([0, 63], dtype=np.float32))
    assert result[0] < 0 < result[1]
