import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import os
from segmentation_models_pytorch import Unet, FPN
import sys
from collections import OrderedDict

sys.path.append("..")
from configs.train_params import *


# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask2rle(img):
    """
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def make_mask(row_id, df):
    """Given a row index, return image_id and mask (256, 1600, 4) from the dataframe `df`"""
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos : (pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order="F")
    return fname, masks


def plot(scores, name, fold=0, safe_pic=True):
    if not isDebug:
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])), scores["train"], label=f"train {name}")
        plt.plot(range(len(scores["train"])), scores["val"], label=f"val {name}")
        plt.title(f"{name} plot")
        plt.xlabel("Epoch")
        plt.ylabel(f"{name}")
        plt.legend()
        if safe_pic:
            plt.savefig("./logs/{}_fold_{}.png".format(name, fold))
        else:
            plt.show()


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model_unet(_model_weights, is_inference=False):
    print("Using weights {}".format(_model_weights))
    if _model_weights == "imagenet":
        model = Unet(
            unet_encoder,
            encoder_weights="imagenet",
            classes=4,
            activation=None,
            attention_type=ATTENTION_TYPE,
        )
        if is_inference:
            model.eval()
        return model
    else:
        model = Unet(
            unet_encoder,
            encoder_weights=None,  # "imagenet",
            classes=4,
            activation=None,
            attention_type=ATTENTION_TYPE,
        )
        if is_inference:
            model.eval()
    if _model_weights is not None:
        device = torch.device("cuda")
        model.to(device)
        state = torch.load(
            _model_weights
        )  # , map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        optimizer_state = state["optimizer"]
        return model, optimizer_state
        # new_state_dict = OrderedDict()
        #
        # for k, v in state['state_dict'].items():
        #     if k in model.state_dict():
        #         new_state_dict[k] = v
        # model = model.load_state_dict(new_state_dict)
    return model


def load_model_fpn(_model_weights, is_inference=False):
    print("Using weights {}".format(_model_weights))
    if _model_weights == "imagenet":
        model = FPN(
            unet_encoder, encoder_weights="imagenet", classes=4, activation=None
        )
        if is_inference:
            model.eval()
        return model
    else:
        model = FPN(unet_encoder, encoder_weights=None, classes=4, activation=None)
        if is_inference:
            model.eval()
    if _model_weights is not None:
        device = torch.device("cuda")
        model.to(device)
        state = torch.load(
            _model_weights
        )  # , map_location=lambda storage, loc: storage)
        model.load_state_dict(state["state_dict"])
        # new_state_dict = OrderedDict()
        #
        # for k, v in state['state_dict'].items():
        #     if k in model.state_dict():
        #         new_state_dict[k] = v
        # model = model.load_state_dict(new_state_dict)
    return model
