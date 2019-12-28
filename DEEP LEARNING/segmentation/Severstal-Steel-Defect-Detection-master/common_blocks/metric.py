import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        # self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, num_negative, num_positive = dice_channel_torch(
            probs, targets, self.base_threshold
        )
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        # preds = predict(probs, self.base_threshold)
        # iou = compute_iou_batch(preds, targets, classes=[1])
        # self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        # iou = np.nanmean(self.iou_scores)
        return dices


def predict(X, threshold):
    """X is sigmoid output of the model"""
    X_p = np.copy(X)
    preds = (X_p > threshold).astype("uint8")
    return preds


def metric_old(probability, truth, threshold=0.5):
    """Calculates dice of positive and negative images seperately
        probability and truth must be torch tensors

        Seems to be this code averages per image not class
        """
    batch_size = truth.shape[0]  # len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def dice_channel_torch(probability, truth, threshold):
    """
    This competition is evaluated on the mean Dice coefficient.
    The Dice coefficient can be used to compare the pixel-wise agreement between a
    predicted segmentation and its corresponding ground truth. The formula is given by:
    Dice(X,Y)=2∗|X∩Y||X|+|Y|
    where X is the predicted set of pixels and Y is the ground truth.
    The Dice coefficient is defined to be 1 when both X and Y are empty.
    The leaderboard score is the mean of the Dice coefficients for each <ImageId, ClassId> pair in the test set.
    Which means the seperate channel of each mask will be average to Dice score.
    :param probability:
    :param truth:
    :param threshold:
    :return:
    """
    batch_size = truth.shape[0]
    channel_num = truth.shape[1]
    mean_dice_channel = 0.0
    with torch.no_grad():
        for j in range(channel_num):
            channel_dice = dice_single_channel(
                probability[:, j, :, :], truth[:, j, :, :], threshold, batch_size
            )
            mean_dice_channel += channel_dice.sum(0) / (batch_size * channel_num)
    return mean_dice_channel, 1, 1, 1, 1


def dice_single_channel(probability, truth, threshold, batch_size, eps=1e-9):
    p = (probability.view(batch_size, -1) > threshold).float()
    t = (truth.view(batch_size, -1) > 0.5).float()
    dice = (2.0 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return dice


def epoch_log(phase, epoch, epoch_loss, meter, start):
    """logging the metrics at the end of an epoch"""
    dice, dice_neg, dice_pos = meter.get_metrics()
    print("Loss: %0.4f | dice: %0.4f" % (epoch_loss, dice))
    return dice


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    """computes iou for one ground truth mask and predicted mask"""
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    """computes mean iou for a batch of ground truth masks and predicted masks"""
    ious = []
    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels)  # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou
