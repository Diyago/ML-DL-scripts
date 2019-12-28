import matplotlib.pyplot as plt
import cv2
import numpy as np
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def visualize(image, mask, original_image=None, original_mask=None, fontsize: int = 14):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.

    Args:
        image: transformed image
        mask: transformed mask
        original_image:
        original_mask:
        fontsize:

    Returns:

    """
    class_dict = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f"Mask {class_dict[i]}", fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f"Original mask {class_dict[i]}", fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title("Transformed image", fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(
                f"Transformed mask {class_dict[i]}", fontsize=fontsize
            )


def visualize_with_raw(
    image, mask, original_image=None, original_mask=None, raw_image=None, raw_mask=None
):
    """
    Similar to visualize function, but with post-processed image, mask.

    Args:
        image:
        mask:
        original_image:
        original_mask:
        raw_image:
        raw_mask:

    Returns:

    """
    fontsize = 14
    class_dict = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}

    f, ax = plt.subplots(3, 5, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title("Original image", fontsize=fontsize)

    for i in range(4):
        ax[0, i + 1].imshow(original_mask[:, :, i])
        ax[0, i + 1].set_title(f"Original mask {class_dict[i]}", fontsize=fontsize)

    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title("Original image", fontsize=fontsize)

    for i in range(4):
        ax[1, i + 1].imshow(raw_mask[:, :, i])
        ax[1, i + 1].set_title(f"Raw predicted mask {class_dict[i]}", fontsize=fontsize)

    ax[2, 0].imshow(image)
    ax[2, 0].set_title("Transformed image", fontsize=fontsize)

    for i in range(4):
        ax[2, i + 1].imshow(mask[:, :, i])
        ax[2, i + 1].set_title(
            f"Predicted mask with processing {class_dict[i]}", fontsize=fontsize
        )


def plot_with_augmentation(image, mask, augment):
    """
    Wrapper for `visualize` function.

    Args:
        image:
        mask:
        augment:

    Returns:

    """
    augmented = augment(image=image, mask=mask)
    image_flipped = augmented["image"]
    mask_flipped = augmented["mask"]
    visualize(image_flipped, mask_flipped, original_image=image, original_mask=mask)


from scipy.ndimage.morphology import binary_fill_holes


def post_process(
    probability: np.array = None, threshold: float = 0.5, min_size: int = 10
):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored

    Args:
        probability: mask
        threshold: threshold for processing
        min_size: min_size for processing

    Returns:

    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return (predictions), num


def dice(img1: np.array, img2: np.array) -> float:
    """
    Calculate dice of two images
    Args:
        img1:
        img2:

    Returns:

    """
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())


def get_optimal_postprocess(loaders=None, runner=None, logdir: str = ""):
    """
    Calculate optimal thresholds for validation data.

    Args:
        loaders: loaders with necessary datasets
        runner: runner
        logdir: directory with model checkpoints

    Returns:

    """
    loaders["infer"] = loaders["valid"]

    runner.infer(
        model=runner.model,
        loaders=loaders,
        callbacks=[
            CheckpointCallback(resume=f"{logdir}/checkpoints/best.pth"),
            InferCallback(),
        ],
    )
    valid_masks = []
    probabilities = np.zeros((2220, 350, 525))
    for i, (batch, output) in enumerate(
        zip(loaders["infer"].dataset, runner.callbacks[0].predictions["logits"])
    ):
        image, mask = batch
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(
                    probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR
                )
            probabilities[i * 4 + j, :, :] = probability

    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(0, 100, 10):
            t /= 100
            for ms in [
                0,
                100,
                1000,
                5000,
                10000,
                11000,
                14000,
                15000,
                16000,
                18000,
                19000,
                20000,
                21000,
                23000,
                25000,
                27000,
                30000,
                50000,
            ]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=["threshold", "size", "dice"])

        attempts_df = attempts_df.sort_values("dice", ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df["threshold"].values[0]
        best_size = attempts_df["size"].values[0]

        class_params[class_id] = (best_threshold, int(best_size))

    print(class_params)
    return class_params
