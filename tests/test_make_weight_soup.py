import pytest
import torch

from make_weight_soup import blend_states


def test_blend_states_interpolates_matching_tensors():
    left = {"weight": torch.tensor([0.0, 2.0]), "counter": torch.tensor(3)}
    right = {"weight": torch.tensor([2.0, 4.0]), "counter": torch.tensor(3)}
    result = blend_states(left, right, 0.25)
    assert torch.equal(result["weight"], torch.tensor([0.5, 2.5]))
    assert torch.equal(result["counter"], torch.tensor(3))


def test_blend_states_rejects_architecture_mismatch():
    with pytest.raises(ValueError, match="keys differ"):
        blend_states({"a": torch.ones(1)}, {"b": torch.ones(1)}, 0.5)


def test_blend_states_rejects_nonfloat_mismatch():
    with pytest.raises(ValueError, match="buffer differs"):
        blend_states({"counter": torch.tensor(1)}, {"counter": torch.tensor(2)}, 0.5)
