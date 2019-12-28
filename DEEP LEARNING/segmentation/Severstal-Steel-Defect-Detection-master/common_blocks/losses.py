import torch.nn as nn


import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class JaccardLoss(nn.Module):
    __name__ = "jaccard_loss"

    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - jaccard(
            y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation
        )


class DiceLoss(nn.Module):
    __name__ = "dice_loss"

    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(
            y_pr,
            y_gt,
            beta=1.0,
            eps=self.eps,
            threshold=None,
            activation=self.activation,
        )


class BCEJaccardLoss(JaccardLoss):
    __name__ = "bce_jaccard_loss"

    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = "bce_dice_loss"

    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce


def iou(pr, gt, eps=1e-7, threshold=None, activation="sigmoid"):
    """
    Source:
        https://github.com/catalyst-team/catalyst/
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation="sigmoid"):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) / (
        (1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps
    )

    return score
