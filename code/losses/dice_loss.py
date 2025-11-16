"""
Dice Loss for Binary Segmentation

This module implements the Dice loss function, which is 1 - Dice coefficient.
The Dice coefficient measures the overlap between prediction and ground truth.

Dice coefficient = 2 * |pred ∩ gt| / (|pred| + |gt|)
Dice loss = 1 - Dice coefficient

This loss function is particularly effective for segmentation tasks with class imbalance.
"""

import torch
import torch.nn as nn


def dice_metric(pred, gt, sigmoid=False, eps=1e-6):
    """
    Compute Dice coefficient (F1 score) for 3D binary segmentation.

    The Dice coefficient measures the overlap between prediction and ground truth:
    Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)

    For empty ground truth (no lesions), we use a special formulation:
    - Returns (1 - mean_prediction) to give high scores when model correctly
      predicts no lesions (i.e., low average prediction values)

    Args:
        pred: Predicted logits or probabilities [B, 1, D, H, W]
        gt: Ground truth binary mask [B, 1, D, H, W] with values in {0, 1}
        sigmoid: Apply sigmoid to pred if True (use when pred is logits)
        eps: Small constant for numerical stability

    Returns:
        Mean Dice coefficient over batch (scalar tensor)
    """
    if sigmoid:
        pred = torch.sigmoid(pred)

    pred = pred.float()
    gt = gt.float()

    # Compute intersection and cardinalities per sample
    intersection = torch.sum(pred * gt, dim=(1, 2, 3, 4))
    pred_sum = torch.sum(pred, dim=(1, 2, 3, 4))
    gt_sum = torch.sum(gt, dim=(1, 2, 3, 4))

    # Standard Dice for samples with lesions
    dice_fg = (2 * intersection + eps) / (pred_sum + gt_sum + eps)

    # Special handling for empty ground truth:
    # Give high score when prediction is also empty (low mean prediction)
    pred_mean = torch.mean(pred, dim=(1, 2, 3, 4))
    dice_bg = 1.0 - pred_mean

    # Choose appropriate metric per sample based on whether GT has lesions
    empty = (gt_sum == 0)
    dice = torch.where(empty, dice_bg, dice_fg)

    return dice.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Loss = 1 - Dice coefficient

    This loss function works well for segmentation tasks, especially when there is
    class imbalance between foreground (lesions) and background.

    Args:
        sigmoid: Apply sigmoid activation to predictions (set True if using logits)

    Example:
        >>> loss_fn = DiceLoss(sigmoid=True)
        >>> pred = torch.randn(2, 1, 96, 96, 96)  # Batch of 2, logits
        >>> gt = torch.randint(0, 2, (2, 1, 96, 96, 96)).float()  # Binary masks
        >>> loss = loss_fn(pred, gt)
    """

    def __init__(self, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, pred, gt):
        """
        Compute Dice loss.

        Args:
            pred: Predicted logits or probabilities [B, 1, D, H, W]
            gt: Ground truth binary mask [B, 1, D, H, W]

        Returns:
            Scalar loss value (1 - Dice coefficient)
        """
        dice = dice_metric(pred, gt, sigmoid=self.sigmoid)
        return 1.0 - dice
