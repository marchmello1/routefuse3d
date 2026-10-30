import numpy as np

from src.data import foreground_coordinates, random_crop


def old_coordinates(target: np.ndarray, maximum: int) -> np.ndarray:
    points = np.argwhere(target > 0)
    if len(points) > maximum:
        points = points[:: int(np.ceil(len(points) / maximum))]
    return points.astype(np.int32, copy=False)


def test_matches_original_for_empty_sparse_and_dense_masks():
    rng = np.random.default_rng(2026)
    for probability in (0.0, 0.001, 0.1, 0.9, 1.0):
        target = (rng.random((29, 31, 37)) < probability).astype(np.float32)
        for maximum in (7, 100_000):
            np.testing.assert_array_equal(
                foreground_coordinates(target, maximum),
                old_coordinates(target, maximum),
            )


def test_cached_points_failed_draw_does_not_rescan_target(monkeypatch):
    image = np.zeros((12, 12, 12), dtype=np.float32)
    target = np.zeros_like(image)
    target[6, 6, 6] = 1
    points = np.array([[6, 6, 6]], dtype=np.int32)
    draws = iter((0.9, 0.0))
    monkeypatch.setattr("src.data.random.random", lambda: next(draws))
    monkeypatch.setattr("src.data.np.argwhere", lambda _: (_ for _ in ()).throw(
        AssertionError("cached points must avoid a full-mask scan")
    ))
    crop_image, crop_target = random_crop(
        image, target, (4, 4, 4), foreground_probability=0.75,
        foreground_points=points,
    )
    assert crop_image.shape == crop_target.shape == (4, 4, 4)
