"""Exact early-stop conditions for route-only fallback inference."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


def load_fallback(monkeypatch):
    monai = types.ModuleType("monai")
    inferers = types.ModuleType("monai.inferers")
    inferers.sliding_window_inference = lambda *_args, **_kwargs: None
    model = types.ModuleType("model")
    model.RoutedResidualUNet = object
    model.strip_distributed_prefix = lambda state: state
    monkeypatch.setitem(sys.modules, "monai", monai)
    monkeypatch.setitem(sys.modules, "monai.inferers", inferers)
    monkeypatch.setitem(sys.modules, "model", model)
    path = Path(__file__).resolve().parents[1] / "container" / "inference.py"
    spec = importlib.util.spec_from_file_location("fallback_bounds_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_bounds_cannot_guess_from_one_vote(monkeypatch):
    fallback = load_fallback(monkeypatch)
    assert fallback.forced_threshold_decision([0.001], 5, 0.5) is None
    assert fallback.forced_threshold_decision([0.001, 0.002], 5, 0.5) is None
    assert fallback.forced_threshold_decision([0.001, 0.002, 0.003], 5, 0.5) is False
    assert fallback.forced_threshold_decision([0.99, 0.98, 0.97], 5, 0.5) is True
    assert fallback.forced_threshold_decision([0.2, 0.8, 0.2, 0.8, 0.2], 5, 0.5) is False


def test_category_early_stops_at_three_strong_members(monkeypatch):
    fallback = load_fallback(monkeypatch)
    calls = []
    scores = [(0.001, 0.002), (0.003, 0.004), (0.002, 0.003)]

    def fake_infer(_runtime, _model, _tensor):
        route, density = scores[len(calls)]
        calls.append(1)
        return torch.logit(torch.tensor([route, density])).reshape(2, 1, 1, 1)

    monkeypatch.setattr(fallback, "infer_route_only", fake_infer)
    models = [types.SimpleNamespace(density_head=object()) for _ in range(5)]
    runtime = fallback.Runtime(
        models=models,
        device=torch.device("cpu"),
        config={"route_threshold": 0.5, "density_threshold": 0.5},
    )
    category = fallback.predict_category_only(runtime, np.ones((4, 4, 4), dtype=np.uint8))
    assert category == "isolated_sparse"
    assert len(calls) == 3
