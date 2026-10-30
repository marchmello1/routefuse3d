from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * target).sum(dims)
    denominator = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-x, (3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-x, (1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-x, (1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.minimum(torch.minimum(p1, p2), p3)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(x, 3, stride=1, padding=1)


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    opened = soft_dilate(soft_erode(x))
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_dilate(soft_erode(x))
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    pred_skeleton = soft_skeletonize(prob, iterations)
    target_skeleton = soft_skeletonize(target, iterations)
    dims = tuple(range(2, prob.ndim))
    tprec = (pred_skeleton * target).sum(dims) / (pred_skeleton.sum(dims) + 1e-5)
    tsens = (target_skeleton * prob).sum(dims) / (target_skeleton.sum(dims) + 1e-5)
    return (1.0 - 2.0 * tprec * tsens / (tprec + tsens + 1e-5)).mean()


def boundary_weight(target: torch.Tensor, factor: float = 4.0) -> torch.Tensor:
    dilated = F.max_pool3d(target, 3, stride=1, padding=1)
    eroded = -F.max_pool3d(-target, 3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp_(0, 1)
    return 1.0 + factor * boundary


def isolated_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=boundary_weight(target))
    return 0.55 * soft_dice_loss(logits, target) + 0.45 * bce


def contiguous_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.45 * soft_dice_loss(logits, target) + 0.25 * bce + 0.30 * soft_cldice_loss(logits, target)

