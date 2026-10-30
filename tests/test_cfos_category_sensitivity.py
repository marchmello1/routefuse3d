import json
import sys

import summarize_cv


def test_cfos_category_sensitivity(tmp_path, monkeypatch, capsys):
    records = [
        {
            "case": "cfos_0.mha",
            "subtype": "cfos",
            "category": "isolated_sparse",
            "contiguous": False,
            "shared_weight": 0.5,
            "threshold": 0.5,
            "minimum_size": 1,
            "route_score": 0.1,
            "density_score": 0.8,
            "dice": 0.7,
            "pq": 0.6,
        },
        {
            "case": "plaque_0.mha",
            "subtype": "plaque",
            "category": "isolated_sparse",
            "contiguous": False,
            "shared_weight": 0.5,
            "threshold": 0.5,
            "minimum_size": 1,
            "route_score": 0.1,
            "density_score": 0.1,
            "dice": 0.5,
            "pq": 0.4,
        },
        {
            "case": "vessel_0.mha",
            "subtype": "vessel",
            "category": "contiguous_dense",
            "contiguous": True,
            "shared_weight": 0.5,
            "threshold": 0.5,
            "minimum_size": 1,
            "route_score": 0.9,
            "density_score": 0.9,
            "dice": 0.8,
            "cldice": 0.7,
        },
    ]
    (tmp_path / "fold0_metrics.json").write_text(json.dumps(records))

    monkeypatch.setattr(sys, "argv", ["summarize_cv.py", str(tmp_path)])
    summarize_cv.main()
    original_output = capsys.readouterr().out
    original = json.loads(original_output[original_output.index("{"):])
    assert "isolated_sparse" in original["recommended_by_category"]
    assert "isolated_dense" not in original["recommended_by_category"]

    monkeypatch.setattr(sys, "argv", ["summarize_cv.py", str(tmp_path), "--cfos-dense"])
    summarize_cv.main()
    changed_output = capsys.readouterr().out
    changed = json.loads(changed_output[changed_output.index("{"):])
    assert "isolated_dense" in changed["recommended_by_category"]
    assert changed["density_calibration"]["accuracy"] == 1.0
