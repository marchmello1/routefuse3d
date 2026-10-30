"""The draft hybrid must preserve the official image socket's TIFF/MHA scope."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk


def load_adapter(monkeypatch, category: str, called: list[dict]):
    fallback = types.ModuleType("fallback_inference")
    fallback.Runtime = object

    def full_predict(_runtime, array):
        called.append({"fallback_full": True})
        return np.zeros_like(array, dtype=np.uint8), category

    def category_only(_runtime, _array):
        called.append({"route_only": True})
        return category

    fallback._predict_standard_with_category = full_predict
    fallback.predict_category_only = category_only
    specialist = types.ModuleType("specialist_inference")

    def specialist_run(_runtime, *, source, source_path, output_name):
        called.append({"source_path": source_path, "output_name": output_name})
        output = sitk.GetImageFromArray(np.ones_like(sitk.GetArrayFromImage(source), dtype=np.uint8))
        output.CopyInformation(source)
        sitk.WriteImage(output, str(adapter.OUTPUT / output_name))

    specialist.run = specialist_run
    monkeypatch.setitem(sys.modules, "fallback_inference", fallback)
    monkeypatch.setitem(sys.modules, "specialist_inference", specialist)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    path = Path(__file__).resolve().parents[1] / "container_h200_hybrid" / "inference.py"
    spec = importlib.util.spec_from_file_location("hybrid_adapter_test", path)
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, adapter)
    spec.loader.exec_module(adapter)
    return adapter


@pytest.mark.parametrize("suffix", [".mha", ".tif", ".tiff"])
@pytest.mark.parametrize("category,expected_specialist", [
    ("isolated_sparse", True),
    ("contiguous_sparse", False),
])
def test_hybrid_accepts_all_official_image_formats(
    monkeypatch, tmp_path, suffix, category, expected_specialist
):
    called = []
    adapter = load_adapter(monkeypatch, category, called)
    adapter.INPUT = tmp_path / "input"
    adapter.OUTPUT = tmp_path / "output"
    adapter.INPUT.mkdir()
    adapter.OUTPUT.mkdir()
    source = sitk.GetImageFromArray(np.ones((4, 5, 6), dtype=np.uint8))
    sitk.WriteImage(source, str(adapter.INPUT / f"case{suffix}"))
    adapter.run(adapter.Runtime(fallback=object(), specialist=object()))
    result = sitk.ReadImage(str(adapter.OUTPUT / "segmentation.mha"))
    assert result.GetSize() == source.GetSize()
    assert any("source_path" in event for event in called) is expected_specialist
    assert sum("route_only" in event for event in called) == 1
    assert sum("fallback_full" in event for event in called) == (not expected_specialist)
    assert np.unique(sitk.GetArrayFromImage(result)).tolist() == ([1] if expected_specialist else [0])


def test_large_misrouted_case_falls_back(monkeypatch):
    called = []
    adapter = load_adapter(monkeypatch, "isolated_sparse", called)
    assert adapter.choose_source_with_size("isolated_sparse", 64_000_000) == "fallback"
    assert adapter.choose_source_with_size("isolated_sparse", 27_000_000) == "specialist"


@pytest.mark.parametrize("suffix", [".mha", ".tif", ".tiff"])
def test_specialist_array_path_reads_socket_formats(monkeypatch, tmp_path, suffix):
    package = types.ModuleType("nnunetv2")
    inference_pkg = types.ModuleType("nnunetv2.inference")
    predictor_module = types.ModuleType("nnunetv2.inference.predict_from_raw_data")
    predictor_module.nnUNetPredictor = object
    monkeypatch.setitem(sys.modules, "nnunetv2", package)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference", inference_pkg)
    monkeypatch.setitem(
        sys.modules, "nnunetv2.inference.predict_from_raw_data", predictor_module
    )
    path = Path(__file__).resolve().parents[1] / "container_resencl" / "inference.py"
    spec = importlib.util.spec_from_file_location("specialist_adapter_test", path)
    specialist = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, specialist)
    spec.loader.exec_module(specialist)
    specialist.INPUT = tmp_path / "input"
    specialist.OUTPUT = tmp_path / "output"
    specialist.INPUT.mkdir()
    source = sitk.GetImageFromArray(np.ones((4, 5, 6), dtype=np.uint8))
    source.SetSpacing((1.2, 1.3, 2.0))
    source_path = specialist.INPUT / f"case{suffix}"
    sitk.WriteImage(source, str(source_path))
    monkeypatch.setenv("SELMA_ARRAY_API", "1")

    class FakePredictor:
        def predict_single_npy_array(self, array, properties, **_kwargs):
            assert array.shape == (1, 4, 5, 6)
            assert properties["spacing"] == [2.0, 1.3, 1.2]
            return np.ones(array.shape[1:], dtype=np.uint8)

    specialist.run(
        FakePredictor(), source=source, source_path=source_path,
        output_name="segmentation.mha",
    )
    output = sitk.ReadImage(str(specialist.OUTPUT / "segmentation.mha"))
    assert output.GetSize() == source.GetSize()
    assert output.GetSpacing() == source.GetSpacing()
    assert sitk.GetArrayFromImage(output).sum() == 4 * 5 * 6


@pytest.mark.parametrize("mirror_flag,expected", [("0", False), ("1", True)])
def test_specialist_mirroring_is_explicit(monkeypatch, tmp_path, mirror_flag, expected):
    package = types.ModuleType("nnunetv2")
    inference_pkg = types.ModuleType("nnunetv2.inference")
    predictor_module = types.ModuleType("nnunetv2.inference.predict_from_raw_data")
    seen = []

    class FakePredictor:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.use_mirroring = kwargs["use_mirroring"]
            self.allowed_mirroring_axes = (0, 1, 2)

        def initialize_from_trained_model_folder(self, *_args, **_kwargs):
            pass

    predictor_module.nnUNetPredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "nnunetv2", package)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference", inference_pkg)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference.predict_from_raw_data", predictor_module)
    path = Path(__file__).resolve().parents[1] / "container_resencl" / "inference.py"
    spec = importlib.util.spec_from_file_location("specialist_mirroring_test", path)
    specialist = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, specialist)
    spec.loader.exec_module(specialist)
    model = tmp_path / "model"
    (model / "fold_0").mkdir(parents=True)
    (model / "export_manifest.json").write_text('{"folds": [0]}')
    for name in ("plans.json", "dataset.json", "fold_0/checkpoint_final.pth"):
        (model / name).touch()
    monkeypatch.setattr(specialist, "EMBEDDED_MODEL", model)
    monkeypatch.setattr(specialist, "ATTACHED_MODEL", tmp_path / "missing")
    monkeypatch.setenv("SELMA_MIRRORING", mirror_flag)
    monkeypatch.delenv("SELMA_MIRROR_AXES", raising=False)
    specialist.initialize()
    assert seen[0]["use_mirroring"] is expected
    monkeypatch.setenv("SELMA_MIRRORING", "true")
    with pytest.raises(ValueError, match="SELMA_MIRRORING"):
        specialist.initialize()
    monkeypatch.setenv("SELMA_MIRRORING", mirror_flag)
    monkeypatch.setenv("SELMA_ARRAY_API", "yes")
    with pytest.raises(ValueError, match="SELMA_ARRAY_API"):
        specialist.initialize()
    monkeypatch.setenv("SELMA_ARRAY_API", "0")
    monkeypatch.setenv("SELMA_MIN_COMPONENT_SIZE", "0")
    with pytest.raises(ValueError, match="SELMA_MIN_COMPONENT_SIZE"):
        specialist.initialize()


def test_specialist_mirror_axis_subset(monkeypatch, tmp_path):
    package = types.ModuleType("nnunetv2")
    inference_pkg = types.ModuleType("nnunetv2.inference")
    predictor_module = types.ModuleType("nnunetv2.inference.predict_from_raw_data")

    class FakePredictor:
        def __init__(self, **kwargs):
            self.use_mirroring = kwargs["use_mirroring"]
            self.allowed_mirroring_axes = (0, 1, 2)

        def initialize_from_trained_model_folder(self, *_args, **_kwargs):
            pass

    predictor_module.nnUNetPredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "nnunetv2", package)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference", inference_pkg)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference.predict_from_raw_data", predictor_module)
    path = Path(__file__).resolve().parents[1] / "container_resencl" / "inference.py"
    spec = importlib.util.spec_from_file_location("specialist_mirror_axes_test", path)
    specialist = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, specialist)
    spec.loader.exec_module(specialist)
    model = tmp_path / "model"
    (model / "fold_0").mkdir(parents=True)
    (model / "export_manifest.json").write_text('{"folds": [0]}')
    for name in ("plans.json", "dataset.json", "fold_0/checkpoint_final.pth"):
        (model / name).touch()
    monkeypatch.setattr(specialist, "EMBEDDED_MODEL", model)
    monkeypatch.setattr(specialist, "ATTACHED_MODEL", tmp_path / "missing")
    monkeypatch.setenv("SELMA_MIRRORING", "1")
    monkeypatch.setenv("SELMA_MIRROR_AXES", "0,2")
    assert specialist.initialize().allowed_mirroring_axes == (0, 2)
    monkeypatch.setenv("SELMA_MIRROR_AXES", "0,3")
    with pytest.raises(ValueError, match="Invalid"):
        specialist.initialize()


def test_specialist_ensemble_member_selection(monkeypatch, tmp_path):
    package = types.ModuleType("nnunetv2")
    inference_pkg = types.ModuleType("nnunetv2.inference")
    predictor_module = types.ModuleType("nnunetv2.inference.predict_from_raw_data")
    selected = []

    class FakePredictor:
        def __init__(self, **_kwargs):
            self.use_mirroring = False
            self.allowed_mirroring_axes = (0, 1, 2)

        def initialize_from_trained_model_folder(self, _model, *, use_folds, checkpoint_name):
            selected.append((use_folds, checkpoint_name))

    predictor_module.nnUNetPredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "nnunetv2", package)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference", inference_pkg)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference.predict_from_raw_data", predictor_module)
    path = Path(__file__).resolve().parents[1] / "container_resencl" / "inference.py"
    spec = importlib.util.spec_from_file_location("specialist_member_test", path)
    specialist = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, specialist)
    spec.loader.exec_module(specialist)

    model = tmp_path / "model"
    model.mkdir()
    members = []
    for virtual_fold in range(10):
        (model / f"fold_{virtual_fold}").mkdir()
        (model / f"fold_{virtual_fold}" / "checkpoint_final.pth").touch()
        members.append({
            "virtual_fold": virtual_fold,
            "source_fold": virtual_fold // 2,
            "source_checkpoint": (
                "checkpoint_final.pth" if virtual_fold % 2 == 0
                else "checkpoint_best.pth"
            ),
        })
    (model / "export_manifest.json").write_text(
        json.dumps({"folds": list(range(10)), "member_sources": members})
    )
    (model / "plans.json").touch()
    (model / "dataset.json").touch()
    monkeypatch.setattr(specialist, "EMBEDDED_MODEL", model)
    monkeypatch.setattr(specialist, "ATTACHED_MODEL", tmp_path / "missing")
    monkeypatch.delenv("SELMA_MIRROR_AXES", raising=False)

    for mode, expected in (
        ("all", tuple(range(10))),
        ("final_only", (0, 2, 4, 6, 8)),
        ("best_only", (1, 3, 5, 7, 9)),
    ):
        monkeypatch.setenv("SELMA_ENSEMBLE_MEMBERS", mode)
        specialist.initialize()
        assert selected[-1] == (expected, "checkpoint_final.pth")
    monkeypatch.setenv("SELMA_ENSEMBLE_MEMBERS", "invalid")
    with pytest.raises(ValueError, match="SELMA_ENSEMBLE_MEMBERS"):
        specialist.initialize()


def test_specialist_component_filter_is_opt_in(monkeypatch):
    package = types.ModuleType("nnunetv2")
    inference_pkg = types.ModuleType("nnunetv2.inference")
    predictor_module = types.ModuleType("nnunetv2.inference.predict_from_raw_data")
    predictor_module.nnUNetPredictor = object
    monkeypatch.setitem(sys.modules, "nnunetv2", package)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference", inference_pkg)
    monkeypatch.setitem(sys.modules, "nnunetv2.inference.predict_from_raw_data", predictor_module)
    path = Path(__file__).resolve().parents[1] / "container_resencl" / "inference.py"
    spec = importlib.util.spec_from_file_location("specialist_component_filter_test", path)
    specialist = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, specialist)
    spec.loader.exec_module(specialist)
    array = np.zeros((5, 5, 5), dtype=np.uint8)
    array[0, 0, 0] = 1
    array[2:5, 2:5, 2:5] = 1
    monkeypatch.delenv("SELMA_MIN_COMPONENT_SIZE", raising=False)
    assert specialist._binary_mask(array).sum() == 28
    monkeypatch.setenv("SELMA_MIN_COMPONENT_SIZE", "17")
    assert specialist._binary_mask(array).sum() == 27
    monkeypatch.setenv("SELMA_MIN_COMPONENT_SIZE", "0")
    with pytest.raises(ValueError, match="positive"):
        specialist._binary_mask(array)
