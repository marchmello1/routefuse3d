from pathlib import Path

import pytest
import numpy as np

from src.hybrid_policy import hybrid_prediction, minimum_size_for_route
from src.specialist_paths import prediction_path


def test_prediction_path_shared_root():
    assert prediction_path(Path("/runs/shared"), None, 1, "case") == Path(
        "/runs/shared/fold_1/validation/case.nii.gz"
    )


def test_prediction_path_per_fold_template():
    assert prediction_path(None, "/runs/specialist_f{fold}/interim", 2, "case") == Path(
        "/runs/specialist_f2/interim/fold_2/validation/case.nii.gz"
    )


def test_prediction_path_requires_source():
    with pytest.raises(ValueError, match="required"):
        prediction_path(None, None, 0, "case")


@pytest.mark.parametrize("actual_contiguous", [False, True])
@pytest.mark.parametrize("predicted_contiguous", [False, True])
def test_hybrid_uses_inferred_route_not_oracle_label(
    actual_contiguous: bool, predicted_contiguous: bool
):
    routed = np.zeros((2, 2, 2), dtype=bool)
    specialist = np.ones((2, 2, 2), dtype=bool)
    selected = hybrid_prediction(routed, specialist, predicted_contiguous)
    assert np.array_equal(selected, routed if predicted_contiguous else specialist)


@pytest.mark.parametrize("actual_contiguous", [False, True])
@pytest.mark.parametrize("predicted_contiguous", [False, True])
def test_minimum_size_uses_inferred_route_even_when_wrong(
    actual_contiguous: bool, predicted_contiguous: bool
):
    del actual_contiguous
    assert minimum_size_for_route(predicted_contiguous, 17) == (
        1 if predicted_contiguous else 17
    )


def test_minimum_size_rejects_nonpositive_value():
    with pytest.raises(ValueError, match="positive"):
        minimum_size_for_route(False, 0)
