from pathlib import Path

import pytest
import torch

from average_supervised_checkpoints import average_states, build_average


def test_average_states_preserves_float_dtype_and_equal_integer_buffers():
    left = {"weight": torch.tensor([0.0, 2.0]), "count": torch.tensor(3)}
    right = {"weight": torch.tensor([2.0, 4.0]), "count": torch.tensor(3)}
    result = average_states(left, right, 0.5)
    assert torch.equal(result["weight"], torch.tensor([1.0, 3.0]))
    assert result["weight"].dtype == left["weight"].dtype
    assert torch.equal(result["count"], left["count"])


def test_average_states_rejects_mismatched_nonfloat():
    with pytest.raises(ValueError, match="non-floating"):
        average_states({"count": torch.tensor(2)}, {"count": torch.tensor(3)}, 0.5)


def test_build_average_validates_fold_and_epoch(tmp_path: Path):
    left, right, output = (tmp_path / name for name in ("left.pt", "right.pt", "out.pt"))
    torch.save({"fold": 2, "epoch": 19, "model": {"w": torch.tensor([1.0])}}, left)
    torch.save({"fold": 2, "epoch": 29, "model": {"w": torch.tensor([3.0])}}, right)
    build_average(left, right, output, 2, 20, 30, 0.5)
    result = torch.load(output, map_location="cpu", weights_only=False)
    assert result["source_epochs"] == [20, 30]
    assert torch.equal(result["model"]["w"], torch.tensor([2.0]))
    with pytest.raises(FileExistsError):
        build_average(left, right, output, 2, 20, 30, 0.5)
    with pytest.raises(ValueError, match="fold mismatch"):
        build_average(left, right, tmp_path / "bad.pt", 3, 20, 30, 0.5)
