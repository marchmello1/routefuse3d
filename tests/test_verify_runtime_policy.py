import numpy as np

from verify_runtime_policy import decode, pooled_official_metrics, reusable_metric


def test_decode_uses_the_runtime_route_and_density_thresholds():
    logits = np.zeros((7, 2, 2, 2), dtype=np.float32)
    logits[3] = 10.0  # isolated sparse
    logits[4] = -10.0  # isolated dense
    logits[5] = -10.0  # contiguous sparse
    logits[6] = 10.0  # contiguous dense
    config = {
        "shared_weight": 0.0,
        "route_threshold": 0.5,
        "density_threshold": 0.5,
    }
    sparse, category = decode(logits, 0.3, 0.3, config)
    assert category == "isolated_sparse"
    assert sparse.all()
    dense, category = decode(logits, 0.3, 0.7, config)
    assert category == "isolated_dense"
    assert not dense.any()
    contiguous, category = decode(logits, 0.7, 0.7, config)
    assert category == "contiguous_dense"
    assert contiguous.all()


def test_decode_applies_category_specific_minimum_size():
    logits = np.full((7, 3, 3, 3), -10.0, dtype=np.float32)
    logits[3, 1, 1, 1] = 10.0
    config = {
        "shared_weight": 0.0,
        "isolated_sparse_minimum_size": 2,
    }
    mask, category = decode(logits, 0.1, 0.1, config)
    assert category == "isolated_sparse"
    assert not mask.any()


def test_reuse_requires_identical_inference_decisions_and_postprocess():
    record = {
        "case": "a.mha", "contiguous": False, "uses_strata": True,
        "category": "isolated_sparse", "route_correct": True,
        "density_correct": True,
        "shared_weight": 0.65, "threshold": 0.45, "minimum_size": 5,
        "dice": 0.75, "pq": 0.5,
    }
    config = {
        "route_threshold": 0.45, "density_threshold": 0.5,
        "isolated_sparse_shared_weight": 0.65,
        "isolated_sparse_threshold": 0.45,
        "isolated_sparse_minimum_size": 5,
    }
    assert reusable_metric([record], "a.mha", False, 0.2, 0.3, config) is record
    assert reusable_metric([record], "a.mha", False, 0.2, 0.3,
                           {**config, "density_threshold": 0.2}) is None
    assert reusable_metric([record], "a.mha", False, 0.2, 0.3,
                           {**config, "isolated_sparse_threshold": 0.55}) is None
    assert reusable_metric([record], "a.mha", False, 0.2, 0.3,
                           {**config, "use_strata": False}) is None


def test_reuse_declines_originally_misrouted_case_and_closing():
    record = {
        "case": "v.mha", "contiguous": True, "uses_strata": False,
        "route_correct": True,
        "shared_weight": 0.5, "threshold": 0.35, "minimum_size": 1,
        "dice": 0.8, "cldice": 0.7,
    }
    config = {
        "route_threshold": 0.45, "contiguous_shared_weight": 0.5,
        "contiguous_threshold": 0.35, "contiguous_minimum_size": 1,
    }
    assert reusable_metric([record], "v.mha", True, 0.8, None, config) is record
    assert reusable_metric([record], "v.mha", True, 0.2, None, config) is None
    assert reusable_metric([record], "v.mha", True, 0.8, None,
                           {**config, "contiguous_closing": 1}) is None


def test_reuse_rejects_saved_metrics_from_different_routing_thresholds():
    record = {
        "case": "a.mha", "contiguous": False, "uses_strata": True,
        "category": "isolated_sparse", "route_correct": True,
        "density_correct": True,
        "shared_weight": 0.65, "threshold": 0.45, "minimum_size": 5,
        "dice": 0.75, "pq": 0.5,
    }
    config = {
        "route_threshold": 0.45, "density_threshold": 0.5,
        "isolated_sparse_shared_weight": 0.65,
        "isolated_sparse_threshold": 0.45,
        "isolated_sparse_minimum_size": 5,
    }
    assert reusable_metric([record], "a.mha", False, 0.2, 0.3, config) is record
    assert reusable_metric([{**record, "route_correct": False}],
                           "a.mha", False, 0.2, 0.3, config) is None
    assert reusable_metric([{**record, "density_correct": False}],
                           "a.mha", False, 0.2, 0.3, config) is None


def test_pooled_metrics_weight_cases_not_subtype_buckets():
    rows = [
        {"true_category": "isolated_sparse", "dice": 0.2, "topology": 0.4},
        {"true_category": "isolated_sparse", "dice": 0.4, "topology": 0.6},
        {"true_category": "isolated_dense", "dice": 0.9, "topology": 0.9},
        {"true_category": "contiguous_sparse", "dice": 0.5, "topology": 0.7},
    ]
    pooled = pooled_official_metrics(rows)
    assert pooled["n_isolated"] == 3
    assert pooled["n_contiguous"] == 1
    assert np.isclose(pooled["isolated_dice"], 0.5)
    assert np.isclose(pooled["isolated_pq"], 1.9 / 3)
    assert np.isclose(pooled["four_metric_mean_proxy"], np.mean([0.5, 1.9 / 3, 0.5, 0.7]))
