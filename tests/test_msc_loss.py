import torch

from msc_loss import MaskedStructuralCompletionLoss
from msc_targets import structural_targets


def test_masked_structural_loss_backpropagates():
    image = torch.zeros((2, 1, 16, 16, 16))
    image[:, :, 8, 8, 8] = 4
    prediction = torch.zeros((2, 4, 16, 16, 16), requires_grad=True)
    hidden_everywhere = torch.zeros((2, 1, 1, 1, 1))
    loss = MaskedStructuralCompletionLoss(workers=2)(
        prediction, image, hidden_everywhere
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_visible_voxels_do_not_contribute_to_loss():
    image = torch.zeros((1, 1, 16, 16, 16))
    prediction = torch.zeros((1, 4, 16, 16, 16))
    mask = torch.zeros((1, 1, 2, 2, 2))
    loss_fn = MaskedStructuralCompletionLoss(workers=1)
    baseline = loss_fn(prediction, image, mask)
    prediction[:, :, :8, :8, :8] = 99
    mask[:, :, 0, 0, 0] = 1
    hidden_only = loss_fn(prediction, image, mask)
    # The changed block is visible, so its prediction cannot affect the loss.
    torch.testing.assert_close(hidden_only, baseline)


def test_precomputed_targets_preserve_loss_and_gradient():
    image = torch.zeros((1, 1, 16, 16, 16))
    image[:, :, 3:7, 5:9, 4:11] = 4
    prediction = torch.randn((1, 4, 16, 16, 16), requires_grad=True)
    mask = torch.zeros((1, 1, 2, 2, 2))
    mask[:, :, 0, 1, 1] = 1
    loss_fn = MaskedStructuralCompletionLoss(workers=1)
    old_loss = loss_fn(prediction, image, mask)
    old_gradient = torch.autograd.grad(old_loss, prediction, retain_graph=True)[0]
    targets = torch.from_numpy(structural_targets(image[0, 0].numpy()))[None]
    new_loss = loss_fn(prediction, image, mask, targets)
    new_gradient = torch.autograd.grad(new_loss, prediction)[0]
    torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
    torch.testing.assert_close(new_gradient, old_gradient, rtol=0, atol=0)
