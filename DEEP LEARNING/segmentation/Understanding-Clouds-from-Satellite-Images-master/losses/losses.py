import torch
from typing import List
from pytorch_toolbelt.losses.functional import sigmoid_focal_loss
from torch.nn.modules.loss import _Loss
from pytorch_toolbelt.losses.functional import soft_dice_score
import torch.nn as nn


class FocalLoss(_Loss):
    def __init__(self, alpha=0.5, gamma=2, ignore_index=None):
        """
        Focal loss for multi-class problem.
        https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/focal.py

        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, label_input, label_target):
        num_classes = label_input.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = label_target != self.ignore_index

        for cls in range(num_classes):
            cls_label_target = (label_target == cls).long()
            cls_label_input = label_input[:, cls, ...]

            if self.ignore_index is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += sigmoid_focal_loss(
                cls_label_input, cls_label_target, gamma=self.gamma, alpha=self.alpha
            )
        return loss


class MulticlassDiceLoss(_Loss):
    """Implementation of Dice loss for multiclass (semantic) image segmentation task
    """

    def __init__(
        self,
        classes: List[int] = None,
        from_logits=True,
        weight=None,
        reduction="elementwise_mean",
    ):
        super(MulticlassDiceLoss, self).__init__(reduction=reduction)
        self.classes = classes
        self.from_logits = from_logits
        self.weight = weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if self.from_logits:
            y_pred = y_pred.softmax(dim=1)

        n_classes = y_pred.size(1)
        smooth = 1e-3

        loss = torch.zeros(n_classes, dtype=torch.float, device=y_pred.device)

        if self.classes is None:
            classes = range(n_classes)
        else:
            classes = self.classes

        if self.weight is None:
            weights = [1] * n_classes
        else:
            weights = self.weight

        for class_index, weight in zip(classes, weights):

            dice_target = (y_true == class_index).float()
            dice_output = y_pred[:, class_index, ...]

            num_preds = dice_target.long().sum()

            if num_preds == 0:
                loss[class_index] = 0
            else:
                dice = soft_dice_score(
                    dice_output, dice_target, from_logits=False, smooth=smooth
                )
                loss[class_index] = (1.0 - dice) * weight

        if self.reduction == "elementwise_mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class BCEMulticlassDiceLoss(MulticlassDiceLoss):
    __name__ = "bce_multiclass_dice_loss"

    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce
