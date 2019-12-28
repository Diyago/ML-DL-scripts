import cv2
import numpy as np
import pandas as pd
from utils import post_process
from dataset import mask2rle
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(
    loaders=None,
    runner=None,
    class_params: dict = None,
    path: str = "",
    sub_name: str = "",
):
    """

    Args:
        loaders:
        runner:
        class_params:
        path:
        sub_name:

    Returns:

    """
    encoded_pixels = []
    image_id = 0
    for _, test_batch in tqdm(enumerate(loaders["test"])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})["logits"]
        for _, batch in enumerate(runner_out):
            for probability in batch:

                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(
                        probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                    )
                    prediction, num_predict = post_process(
                        sigmoid(probability),
                        class_params[str(image_id % 4)][0],
                        class_params[str(image_id % 4)][1],
                    )
                if num_predict == 0:
                    encoded_pixels.append("")
                else:
                    r = mask2rle(prediction)
                    encoded_pixels.append(r)
                image_id += 1

    sub = pd.read_csv(f"{path}/sample_submission.csv")
    sub["EncodedPixels"] = encoded_pixels
    sub.to_csv(
        f"submissions/submission_{sub_name}.csv",
        columns=["Image_Label", "EncodedPixels"],
        index=False,
    )


def predict_blend(
    loaders=None,
    runner=None,
    class_params: dict = None,
    path: str = "",
    sub_name: str = "",
):
    """

    Args:
        loaders:
        runner:
        class_params:
        path:
        sub_name:

    Returns:

    """
    encoded_pixels = []
    image_id = 0
    for _, test_batch in tqdm(enumerate(loaders["test"])):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})["logits"]
        for _, batch in enumerate(runner_out):
            for probability in batch:

                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(
                        probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                    )
                    prediction, num_predict = post_process(
                        (probability),
                        class_params[str(image_id % 4)][0],
                        class_params[str(image_id % 4)][1],
                    )
                if num_predict == 0:
                    encoded_pixels.append("")
                else:
                    r = mask2rle(prediction)
                    encoded_pixels.append(r)
                image_id += 1

    sub = pd.read_csv(f"{path}/sample_submission.csv")
    sub["EncodedPixels"] = encoded_pixels
    sub.to_csv(
        f"submissions/submission_{sub_name}.csv",
        columns=["Image_Label", "EncodedPixels"],
        index=False,
    )
