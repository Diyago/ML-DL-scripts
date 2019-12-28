from typing import Dict
import torch
import numpy as np
from catalyst.dl.core import Callback, RunnerState, CallbackOrder
import cv2
from collections import OrderedDict


def calculate_confusion_matrix_from_arrays(
    prediction: np.array, ground_truth: np.array, num_classes: int
) -> np.array:
    """Calculate confusion matrix for a given set of classes.
    if GT value is outside of the [0, num_classes) it is excluded.
    Args:
        prediction:
        ground_truth:
        num_classes:
    Returns:
    """
    # a long 2xn array with each column being a pixel pair
    replace_indices = np.vstack((ground_truth.flatten(), prediction.flatten()))

    valid_index = replace_indices[0, :] < num_classes
    replace_indices = replace_indices[:, valid_index].T

    # add up confusion matrix
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes), (0, num_classes)],
    )
    return confusion_matrix.astype(np.uint64)


def get_confusion_matrix(y_pred_logits: torch.Tensor, y_true: torch.Tensor):
    num_classes = y_pred_logits.shape[1]
    y_pred = torch.argmax(y_pred_logits, dim=1)
    ground_truth = y_true.cpu().numpy()
    prediction = y_pred.cpu().numpy()

    return calculate_confusion_matrix_from_arrays(ground_truth, prediction, num_classes)


def calculate_tp_fp_fn(confusion_matrix):
    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for index in range(confusion_matrix.shape[0]):
        true_positives[index] = confusion_matrix[index, index]
        false_positives[index] = (
            confusion_matrix[:, index].sum() - true_positives[index]
        )
        false_negatives[index] = (
            confusion_matrix[index, :].sum() - true_positives[index]
        )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def calculate_dice(tp_fp_fn_dict):
    epsilon = 1e-7

    dice = {}

    for i in range(len(tp_fp_fn_dict["true_positives"])):
        tp = tp_fp_fn_dict["true_positives"][i]
        fp = tp_fp_fn_dict["false_positives"][i]
        fn = tp_fp_fn_dict["true_positives"][i]

        dice[i] = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)

        if not 0 <= dice[i] <= 1:
            raise ValueError()

    return dice


class MulticlassDiceMetricCallback(Callback):
    def __init__(
        self,
        prefix: str = "dice",
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params,
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
        self.confusion_matrix = None
        self.class_names = metric_params[
            "class_names"
        ]  # dictionary {class_id: class_name}
        self.class_prefix = metric_params["class_prefix"]

    def _reset_stats(self):
        self.confusion_matrix = None

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        confusion_matrix = get_confusion_matrix(outputs, targets)

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

    def on_loader_end(self, state: RunnerState):

        tp_fp_fn_dict = calculate_tp_fp_fn(self.confusion_matrix)

        batch_metrics: Dict = calculate_dice(tp_fp_fn_dict)

        for metric_id, dice_value in batch_metrics.items():
            if metric_id not in self.class_names:
                continue

            metric_name = self.class_names[metric_id]
            state.metrics.epoch_values[state.loader_name][
                f"{self.class_prefix}_{metric_name}"
            ] = dice_value

        state.metrics.epoch_values[state.loader_name]["mean"] = np.mean(
            [x for x in batch_metrics.values()]
        )

        self._reset_stats()


class CustomSegmentationInferCallback(Callback):
    def __init__(self, return_valid: bool = False):
        super().__init__(CallbackOrder.Internal)
        self.valid_masks = []
        self.probabilities = np.zeros((2220, 350, 525))
        self.return_valid = return_valid

    def on_batch_end(self, state: RunnerState):
        image, mask = state.input
        output = state.output["logits"]
        if self.return_valid:
            for m in mask:
                if m.shape != (350, 525):
                    m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                self.valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(
                    probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                )
            self.probabilities[j, :, :] = probability
