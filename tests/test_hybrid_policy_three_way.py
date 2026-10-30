import numpy as np

import pytest

from src.hybrid_policy import size_only_source, three_way_hybrid_prediction, three_way_source


def test_three_way_uses_only_predicted_category():
    routed = np.zeros((2, 2, 2), dtype=bool)
    specialist = np.ones((2, 2, 2), dtype=bool)
    assert three_way_hybrid_prediction(routed, specialist, False, False) is specialist
    assert three_way_hybrid_prediction(routed, specialist, False, True) is specialist
    assert three_way_hybrid_prediction(routed, specialist, True, True) is specialist
    assert three_way_hybrid_prediction(routed, specialist, True, False) is routed


def test_missing_density_head_preserves_old_contiguous_route():
    routed = np.zeros((1,), dtype=bool)
    specialist = np.ones((1,), dtype=bool)
    assert three_way_hybrid_prediction(routed, specialist, True, None) is routed


def test_volume_guard_catches_misrouted_large_nerve_without_oracle_label():
    assert three_way_source("isolated_sparse", 64_000_000, 27_000_000) == "fallback"
    assert three_way_source("isolated_sparse", 27_000_000, 27_000_000) == "specialist"
    assert three_way_source("contiguous_dense", 8_000_000, 27_000_000) == "specialist"
    assert three_way_source("contiguous_sparse", 8_000_000, 27_000_000) == "fallback"


def test_size_only_policy_has_exact_boundary():
    assert size_only_source(27_000_000, 27_000_000) == "specialist"
    assert size_only_source(27_000_001, 27_000_000) == "fallback"
    with pytest.raises(ValueError):
        size_only_source(0, 27_000_000)
