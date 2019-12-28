#!/usr/bin/env python
# coding: utf-8


get_ipython().system(" python ../input/mlcomp/mlcomp/mlcomp/setup.py")
get_ipython().system(
    "pip install ../input/../input/efficientnet-pytorch/efficientnet-pytorch/EfficientNet-PyTorch-master"
)
get_ipython().system(
    "pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/"
)
get_ipython().system(
    "pip install --no-deps  --no-dependencies ../input/segmentation-models-pytorch/ "
)


import warnings

warnings.filterwarnings("ignore")
import os
import matplotlib.pyplot as plt

import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm_notebook
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.jit import load

from mlcomp.contrib.transform.albumentations import ChannelTranspose
from mlcomp.contrib.dataset.classify import ImageDataset
from mlcomp.contrib.transform.rle import rle2mask, mask2rle
from mlcomp.contrib.transform.tta import TtaWrap


unet_se_resnext50_32x4d = load(
    "/kaggle/input/severstalmodels/unet_se_resnext50_32x4d.pth"
).cuda()
unet_mobilenet2 = load("/kaggle/input/severstalmodels/unet_mobilenet2.pth").cuda()
# unet_resnet34 = load('/kaggle/input/severstalmodels/unet_resnet34.pth').cuda()


import os
from segmentation_models_pytorch import Unet, FPN


ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda"

CLASSES = ["0", "1", "2", "3", "4"]
ACTIVATION = "softmax"


unet_resnet34 = Unet(
    encoder_name=ENCODER, encoder_weights=None, classes=4, activation="sigmoid"
)

state = torch.load("../input/bce-clf/unet_res34_525.pth")
unet_resnet34.load_state_dict(state["model_state_dict"])


unet_resnet34 = unet_resnet34.cuda()
unet_resnet34 = unet_resnet34.eval()


device = torch.device("cuda")
model_senet = Unet(
    "se_resnext50_32x4d", encoder_weights=None, classes=4, activation=None
)
model_senet.to(device)
model_senet.eval()
state = torch.load(
    "../input/senetmodels/senext50_30_epochs_high_threshold.pth",
    map_location=lambda storage, loc: storage,
)
model_senet.load_state_dict(state["state_dict"])


model_fpn91lb = FPN(
    encoder_name="se_resnext50_32x4d", classes=4, activation=None, encoder_weights=None
)
model_fpn91lb.to(device)
model_fpn91lb.eval()
# state = torch.load('../input/fpnseresnext/model_se_resnext50_32x4d_fold_0_epoch_7_dice_0.935771107673645.pth', map_location=lambda storage, loc: storage)
state = torch.load(
    "../input/fpnse50dice944/model_se_resnext50_32x4d_fold_0_epoch_26_dice_0.94392.pth",
    map_location=lambda storage, loc: storage,
)
model_fpn91lb.load_state_dict(state["state_dict"])


model_fpn91lb_pseudo = FPN(
    encoder_name="se_resnext50_32x4d", classes=4, activation=None, encoder_weights=None
)
model_fpn91lb_pseudo.to(device)
model_fpn91lb_pseudo.eval()
# state = torch.load('../input/fpnseresnext/model_se_resnext50_32x4d_fold_0_epoch_7_dice_0.935771107673645.pth', map_location=lambda storage, loc: storage)
state = torch.load(
    "../input/942-finetuned-on-pseudo-to9399/pseudo_fpn_se_resnext50_32x4d_fold_0_epoch_22_dice_0.944/pseudo_fpn_se_resnext50_32x4d_fold_0_epoch_22_dice_0.9446276426315308.pth",
    map_location=lambda storage, loc: storage,
)
model_fpn91lb_pseudo.load_state_dict(state["state_dict"])


ENCODER = "se_resnext50_32x4d"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["0", "1", "2", "3", "4"]
ACTIVATION = "softmax"
fpn_se = FPN(
    encoder_name=ENCODER,
    encoder_weights=None,
    #     encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
state = torch.load("../input/bce-clf/fpn_se13.pth")
fpn_se.to(device)
fpn_se.eval()
fpn_se.load_state_dict(state["model_state_dict"])


ENCODER = "se_resnext50_32x4d"
ENCODER_WEIGHTS = "imagenet"
CLASSES = ["0", "1", "2", "3", "4"]
ACTIVATION = "softmax"
fpn_se2 = FPN(
    encoder_name=ENCODER,
    encoder_weights=None,
    #     encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
state = torch.load("../input/bce-clf/fpn_lovash_9519.pth")
fpn_se2.to(device)
fpn_se2.eval()
fpn_se2.load_state_dict(state["model_state_dict"])


# ### Models' mean aggregator


class Model:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():

            for m in self.models[:-2]:
                res.append(torch.sigmoid(m(x)))
            # last model with 5 classes (+background)
            res.append(torch.sigmoid(self.models[-2](x))[:, 1:, :, :])
            res.append(torch.sigmoid(self.models[-1](x))[:, 1:, :, :])
        res = torch.stack(res)
        res = torch.mean(res, dim=0)
        #         print(res.shape)
        #         print(pred_cls.shape)

        return res


model = Model(
    [
        unet_se_resnext50_32x4d,
        unet_mobilenet2,
        unet_resnet34,
        model_senet,
        model_fpn91lb,
        model_fpn91lb_pseudo,
        fpn_se,
        fpn_se2,
    ]
)


# ### Create TTA transforms, datasets, loaders


def create_transforms(additional):
    res = list(additional)
    # add necessary transformations
    res.extend(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.230, 0.225, 0.223)
                # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            ChannelTranspose(),
        ]
    )
    res = A.Compose(res)
    return res


img_folder = "/kaggle/input/severstal-steel-defect-detection/test_images"
batch_size = 2
num_workers = 0

# Different transforms for TTA wrapper
transforms = [[], [A.HorizontalFlip(p=1)]]

transforms = [create_transforms(t) for t in transforms]
datasets = [
    TtaWrap(ImageDataset(img_folder=img_folder, transforms=t), tfms=t)
    for t in transforms
]
loaders = [
    DataLoader(d, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    for d in datasets
]


# ### Loaders' mean aggregator


thresholds = [0.5, 0.5, 0.5, 0.49]
min_area = [500, 500, 1000, 2000]

res = []
# Iterate over all TTA loaders
total = len(datasets[0]) // batch_size
for loaders_batch in tqdm_notebook(zip(*loaders), total=total):
    preds = []
    image_file = []
    for i, batch in enumerate(loaders_batch):
        features = batch["features"].cuda()
        # p = torch.sigmoid(model(features))
        p = model(features)
        # inverse operations for TTA
        p = datasets[i].inverse(p)
        preds.append(p)
        image_file = batch["image_file"]

    # TTA mean
    preds = torch.stack(preds)
    preds = torch.mean(preds, dim=0)
    preds = preds.detach().cpu().numpy()

    # Batch post processing
    for p, file in zip(preds, image_file):
        file = os.path.basename(file)
        # Image postprocessing
        for i in range(4):
            p_channel = p[i]
            imageid_classid = file + "_" + str(i + 1)
            p_channel = (p_channel > thresholds[i]).astype(np.uint8)
            if p_channel.sum() < min_area[i]:
                p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)

            res.append(
                {
                    "ImageId_ClassId": imageid_classid,
                    "EncodedPixels": mask2rle(p_channel),
                }
            )

df = pd.DataFrame(res)
df.to_csv("submission.csv", index=False)


df = pd.DataFrame(res)
df = df.fillna("")
df.to_csv("submission.csv", index=False)


# In[22]:


import pdb
import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensor
import torch.utils.data as data
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F

BatchNorm2d = nn.BatchNorm2d

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD = [0.229, 0.224, 0.225]

###############################################################################
CONVERSION = [
    "block0.0.weight",
    (64, 3, 7, 7),
    "conv1.weight",
    (64, 3, 7, 7),
    "block0.1.weight",
    (64,),
    "bn1.weight",
    (64,),
    "block0.1.bias",
    (64,),
    "bn1.bias",
    (64,),
    "block0.1.running_mean",
    (64,),
    "bn1.running_mean",
    (64,),
    "block0.1.running_var",
    (64,),
    "bn1.running_var",
    (64,),
    "block1.1.conv_bn1.conv.weight",
    (64, 64, 3, 3),
    "layer1.0.conv1.weight",
    (64, 64, 3, 3),
    "block1.1.conv_bn1.bn.weight",
    (64,),
    "layer1.0.bn1.weight",
    (64,),
    "block1.1.conv_bn1.bn.bias",
    (64,),
    "layer1.0.bn1.bias",
    (64,),
    "block1.1.conv_bn1.bn.running_mean",
    (64,),
    "layer1.0.bn1.running_mean",
    (64,),
    "block1.1.conv_bn1.bn.running_var",
    (64,),
    "layer1.0.bn1.running_var",
    (64,),
    "block1.1.conv_bn2.conv.weight",
    (64, 64, 3, 3),
    "layer1.0.conv2.weight",
    (64, 64, 3, 3),
    "block1.1.conv_bn2.bn.weight",
    (64,),
    "layer1.0.bn2.weight",
    (64,),
    "block1.1.conv_bn2.bn.bias",
    (64,),
    "layer1.0.bn2.bias",
    (64,),
    "block1.1.conv_bn2.bn.running_mean",
    (64,),
    "layer1.0.bn2.running_mean",
    (64,),
    "block1.1.conv_bn2.bn.running_var",
    (64,),
    "layer1.0.bn2.running_var",
    (64,),
    "block1.2.conv_bn1.conv.weight",
    (64, 64, 3, 3),
    "layer1.1.conv1.weight",
    (64, 64, 3, 3),
    "block1.2.conv_bn1.bn.weight",
    (64,),
    "layer1.1.bn1.weight",
    (64,),
    "block1.2.conv_bn1.bn.bias",
    (64,),
    "layer1.1.bn1.bias",
    (64,),
    "block1.2.conv_bn1.bn.running_mean",
    (64,),
    "layer1.1.bn1.running_mean",
    (64,),
    "block1.2.conv_bn1.bn.running_var",
    (64,),
    "layer1.1.bn1.running_var",
    (64,),
    "block1.2.conv_bn2.conv.weight",
    (64, 64, 3, 3),
    "layer1.1.conv2.weight",
    (64, 64, 3, 3),
    "block1.2.conv_bn2.bn.weight",
    (64,),
    "layer1.1.bn2.weight",
    (64,),
    "block1.2.conv_bn2.bn.bias",
    (64,),
    "layer1.1.bn2.bias",
    (64,),
    "block1.2.conv_bn2.bn.running_mean",
    (64,),
    "layer1.1.bn2.running_mean",
    (64,),
    "block1.2.conv_bn2.bn.running_var",
    (64,),
    "layer1.1.bn2.running_var",
    (64,),
    "block1.3.conv_bn1.conv.weight",
    (64, 64, 3, 3),
    "layer1.2.conv1.weight",
    (64, 64, 3, 3),
    "block1.3.conv_bn1.bn.weight",
    (64,),
    "layer1.2.bn1.weight",
    (64,),
    "block1.3.conv_bn1.bn.bias",
    (64,),
    "layer1.2.bn1.bias",
    (64,),
    "block1.3.conv_bn1.bn.running_mean",
    (64,),
    "layer1.2.bn1.running_mean",
    (64,),
    "block1.3.conv_bn1.bn.running_var",
    (64,),
    "layer1.2.bn1.running_var",
    (64,),
    "block1.3.conv_bn2.conv.weight",
    (64, 64, 3, 3),
    "layer1.2.conv2.weight",
    (64, 64, 3, 3),
    "block1.3.conv_bn2.bn.weight",
    (64,),
    "layer1.2.bn2.weight",
    (64,),
    "block1.3.conv_bn2.bn.bias",
    (64,),
    "layer1.2.bn2.bias",
    (64,),
    "block1.3.conv_bn2.bn.running_mean",
    (64,),
    "layer1.2.bn2.running_mean",
    (64,),
    "block1.3.conv_bn2.bn.running_var",
    (64,),
    "layer1.2.bn2.running_var",
    (64,),
    "block2.0.conv_bn1.conv.weight",
    (128, 64, 3, 3),
    "layer2.0.conv1.weight",
    (128, 64, 3, 3),
    "block2.0.conv_bn1.bn.weight",
    (128,),
    "layer2.0.bn1.weight",
    (128,),
    "block2.0.conv_bn1.bn.bias",
    (128,),
    "layer2.0.bn1.bias",
    (128,),
    "block2.0.conv_bn1.bn.running_mean",
    (128,),
    "layer2.0.bn1.running_mean",
    (128,),
    "block2.0.conv_bn1.bn.running_var",
    (128,),
    "layer2.0.bn1.running_var",
    (128,),
    "block2.0.conv_bn2.conv.weight",
    (128, 128, 3, 3),
    "layer2.0.conv2.weight",
    (128, 128, 3, 3),
    "block2.0.conv_bn2.bn.weight",
    (128,),
    "layer2.0.bn2.weight",
    (128,),
    "block2.0.conv_bn2.bn.bias",
    (128,),
    "layer2.0.bn2.bias",
    (128,),
    "block2.0.conv_bn2.bn.running_mean",
    (128,),
    "layer2.0.bn2.running_mean",
    (128,),
    "block2.0.conv_bn2.bn.running_var",
    (128,),
    "layer2.0.bn2.running_var",
    (128,),
    "block2.0.shortcut.conv.weight",
    (128, 64, 1, 1),
    "layer2.0.downsample.0.weight",
    (128, 64, 1, 1),
    "block2.0.shortcut.bn.weight",
    (128,),
    "layer2.0.downsample.1.weight",
    (128,),
    "block2.0.shortcut.bn.bias",
    (128,),
    "layer2.0.downsample.1.bias",
    (128,),
    "block2.0.shortcut.bn.running_mean",
    (128,),
    "layer2.0.downsample.1.running_mean",
    (128,),
    "block2.0.shortcut.bn.running_var",
    (128,),
    "layer2.0.downsample.1.running_var",
    (128,),
    "block2.1.conv_bn1.conv.weight",
    (128, 128, 3, 3),
    "layer2.1.conv1.weight",
    (128, 128, 3, 3),
    "block2.1.conv_bn1.bn.weight",
    (128,),
    "layer2.1.bn1.weight",
    (128,),
    "block2.1.conv_bn1.bn.bias",
    (128,),
    "layer2.1.bn1.bias",
    (128,),
    "block2.1.conv_bn1.bn.running_mean",
    (128,),
    "layer2.1.bn1.running_mean",
    (128,),
    "block2.1.conv_bn1.bn.running_var",
    (128,),
    "layer2.1.bn1.running_var",
    (128,),
    "block2.1.conv_bn2.conv.weight",
    (128, 128, 3, 3),
    "layer2.1.conv2.weight",
    (128, 128, 3, 3),
    "block2.1.conv_bn2.bn.weight",
    (128,),
    "layer2.1.bn2.weight",
    (128,),
    "block2.1.conv_bn2.bn.bias",
    (128,),
    "layer2.1.bn2.bias",
    (128,),
    "block2.1.conv_bn2.bn.running_mean",
    (128,),
    "layer2.1.bn2.running_mean",
    (128,),
    "block2.1.conv_bn2.bn.running_var",
    (128,),
    "layer2.1.bn2.running_var",
    (128,),
    "block2.2.conv_bn1.conv.weight",
    (128, 128, 3, 3),
    "layer2.2.conv1.weight",
    (128, 128, 3, 3),
    "block2.2.conv_bn1.bn.weight",
    (128,),
    "layer2.2.bn1.weight",
    (128,),
    "block2.2.conv_bn1.bn.bias",
    (128,),
    "layer2.2.bn1.bias",
    (128,),
    "block2.2.conv_bn1.bn.running_mean",
    (128,),
    "layer2.2.bn1.running_mean",
    (128,),
    "block2.2.conv_bn1.bn.running_var",
    (128,),
    "layer2.2.bn1.running_var",
    (128,),
    "block2.2.conv_bn2.conv.weight",
    (128, 128, 3, 3),
    "layer2.2.conv2.weight",
    (128, 128, 3, 3),
    "block2.2.conv_bn2.bn.weight",
    (128,),
    "layer2.2.bn2.weight",
    (128,),
    "block2.2.conv_bn2.bn.bias",
    (128,),
    "layer2.2.bn2.bias",
    (128,),
    "block2.2.conv_bn2.bn.running_mean",
    (128,),
    "layer2.2.bn2.running_mean",
    (128,),
    "block2.2.conv_bn2.bn.running_var",
    (128,),
    "layer2.2.bn2.running_var",
    (128,),
    "block2.3.conv_bn1.conv.weight",
    (128, 128, 3, 3),
    "layer2.3.conv1.weight",
    (128, 128, 3, 3),
    "block2.3.conv_bn1.bn.weight",
    (128,),
    "layer2.3.bn1.weight",
    (128,),
    "block2.3.conv_bn1.bn.bias",
    (128,),
    "layer2.3.bn1.bias",
    (128,),
    "block2.3.conv_bn1.bn.running_mean",
    (128,),
    "layer2.3.bn1.running_mean",
    (128,),
    "block2.3.conv_bn1.bn.running_var",
    (128,),
    "layer2.3.bn1.running_var",
    (128,),
    "block2.3.conv_bn2.conv.weight",
    (128, 128, 3, 3),
    "layer2.3.conv2.weight",
    (128, 128, 3, 3),
    "block2.3.conv_bn2.bn.weight",
    (128,),
    "layer2.3.bn2.weight",
    (128,),
    "block2.3.conv_bn2.bn.bias",
    (128,),
    "layer2.3.bn2.bias",
    (128,),
    "block2.3.conv_bn2.bn.running_mean",
    (128,),
    "layer2.3.bn2.running_mean",
    (128,),
    "block2.3.conv_bn2.bn.running_var",
    (128,),
    "layer2.3.bn2.running_var",
    (128,),
    "block3.0.conv_bn1.conv.weight",
    (256, 128, 3, 3),
    "layer3.0.conv1.weight",
    (256, 128, 3, 3),
    "block3.0.conv_bn1.bn.weight",
    (256,),
    "layer3.0.bn1.weight",
    (256,),
    "block3.0.conv_bn1.bn.bias",
    (256,),
    "layer3.0.bn1.bias",
    (256,),
    "block3.0.conv_bn1.bn.running_mean",
    (256,),
    "layer3.0.bn1.running_mean",
    (256,),
    "block3.0.conv_bn1.bn.running_var",
    (256,),
    "layer3.0.bn1.running_var",
    (256,),
    "block3.0.conv_bn2.conv.weight",
    (256, 256, 3, 3),
    "layer3.0.conv2.weight",
    (256, 256, 3, 3),
    "block3.0.conv_bn2.bn.weight",
    (256,),
    "layer3.0.bn2.weight",
    (256,),
    "block3.0.conv_bn2.bn.bias",
    (256,),
    "layer3.0.bn2.bias",
    (256,),
    "block3.0.conv_bn2.bn.running_mean",
    (256,),
    "layer3.0.bn2.running_mean",
    (256,),
    "block3.0.conv_bn2.bn.running_var",
    (256,),
    "layer3.0.bn2.running_var",
    (256,),
    "block3.0.shortcut.conv.weight",
    (256, 128, 1, 1),
    "layer3.0.downsample.0.weight",
    (256, 128, 1, 1),
    "block3.0.shortcut.bn.weight",
    (256,),
    "layer3.0.downsample.1.weight",
    (256,),
    "block3.0.shortcut.bn.bias",
    (256,),
    "layer3.0.downsample.1.bias",
    (256,),
    "block3.0.shortcut.bn.running_mean",
    (256,),
    "layer3.0.downsample.1.running_mean",
    (256,),
    "block3.0.shortcut.bn.running_var",
    (256,),
    "layer3.0.downsample.1.running_var",
    (256,),
    "block3.1.conv_bn1.conv.weight",
    (256, 256, 3, 3),
    "layer3.1.conv1.weight",
    (256, 256, 3, 3),
    "block3.1.conv_bn1.bn.weight",
    (256,),
    "layer3.1.bn1.weight",
    (256,),
    "block3.1.conv_bn1.bn.bias",
    (256,),
    "layer3.1.bn1.bias",
    (256,),
    "block3.1.conv_bn1.bn.running_mean",
    (256,),
    "layer3.1.bn1.running_mean",
    (256,),
    "block3.1.conv_bn1.bn.running_var",
    (256,),
    "layer3.1.bn1.running_var",
    (256,),
    "block3.1.conv_bn2.conv.weight",
    (256, 256, 3, 3),
    "layer3.1.conv2.weight",
    (256, 256, 3, 3),
    "block3.1.conv_bn2.bn.weight",
    (256,),
    "layer3.1.bn2.weight",
    (256,),
    "block3.1.conv_bn2.bn.bias",
    (256,),
    "layer3.1.bn2.bias",
    (256,),
    "block3.1.conv_bn2.bn.running_mean",
    (256,),
    "layer3.1.bn2.running_mean",
    (256,),
    "block3.1.conv_bn2.bn.running_var",
    (256,),
    "layer3.1.bn2.running_var",
    (256,),
    "block3.2.conv_bn1.conv.weight",
    (256, 256, 3, 3),
    "layer3.2.conv1.weight",
    (256, 256, 3, 3),
    "block3.2.conv_bn1.bn.weight",
    (256,),
    "layer3.2.bn1.weight",
    (256,),
    "block3.2.conv_bn1.bn.bias",
    (256,),
    "layer3.2.bn1.bias",
    (256,),
    "block3.2.conv_bn1.bn.running_mean",
    (256,),
    "layer3.2.bn1.running_mean",
    (256,),
    "block3.2.conv_bn1.bn.running_var",
    (256,),
    "layer3.2.bn1.running_var",
    (256,),
    "block3.2.conv_bn2.conv.weight",
    (256, 256, 3, 3),
    "layer3.2.conv2.weight",
    (256, 256, 3, 3),
    "block3.2.conv_bn2.bn.weight",
    (256,),
    "layer3.2.bn2.weight",
    (256,),
    "block3.2.conv_bn2.bn.bias",
    (256,),
    "layer3.2.bn2.bias",
    (256,),
    "block3.2.conv_bn2.bn.running_mean",
    (256,),
    "layer3.2.bn2.running_mean",
    (256,),
    "block3.2.conv_bn2.bn.running_var",
    (256,),
    "layer3.2.bn2.running_var",
    (256,),
    "block3.3.conv_bn1.conv.weight",
    (256, 256, 3, 3),
    "layer3.3.conv1.weight",
    (256, 256, 3, 3),
    "block3.3.conv_bn1.bn.weight",
    (256,),
    "layer3.3.bn1.weight",
    (256,),
    "block3.3.conv_bn1.bn.bias",
    (256,),
    "layer3.3.bn1.bias",
    (256,),
    "block3.3.conv_bn1.bn.running_mean",
    (256,),
    "layer3.3.bn1.running_mean",
    (256,),
    "block3.3.conv_bn1.bn.running_var",
    (256,),
    "layer3.3.bn1.running_var",
    (256,),
    "block3.3.conv_bn2.conv.weight",
    (256, 256, 3, 3),
    "layer3.3.conv2.weight",
    (256, 256, 3, 3),
    "block3.3.conv_bn2.bn.weight",
    (256,),
    "layer3.3.bn2.weight",
    (256,),
    "block3.3.conv_bn2.bn.bias",
    (256,),
    "layer3.3.bn2.bias",
    (256,),
    "block3.3.conv_bn2.bn.running_mean",
    (256,),
    "layer3.3.bn2.running_mean",
    (256,),
    "block3.3.conv_bn2.bn.running_var",
    (256,),
    "layer3.3.bn2.running_var",
    (256,),
    "block3.4.conv_bn1.conv.weight",
    (256, 256, 3, 3),
    "layer3.4.conv1.weight",
    (256, 256, 3, 3),
    "block3.4.conv_bn1.bn.weight",
    (256,),
    "layer3.4.bn1.weight",
    (256,),
    "block3.4.conv_bn1.bn.bias",
    (256,),
    "layer3.4.bn1.bias",
    (256,),
    "block3.4.conv_bn1.bn.running_mean",
    (256,),
    "layer3.4.bn1.running_mean",
    (256,),
    "block3.4.conv_bn1.bn.running_var",
    (256,),
    "layer3.4.bn1.running_var",
    (256,),
    "block3.4.conv_bn2.conv.weight",
    (256, 256, 3, 3),
    "layer3.4.conv2.weight",
    (256, 256, 3, 3),
    "block3.4.conv_bn2.bn.weight",
    (256,),
    "layer3.4.bn2.weight",
    (256,),
    "block3.4.conv_bn2.bn.bias",
    (256,),
    "layer3.4.bn2.bias",
    (256,),
    "block3.4.conv_bn2.bn.running_mean",
    (256,),
    "layer3.4.bn2.running_mean",
    (256,),
    "block3.4.conv_bn2.bn.running_var",
    (256,),
    "layer3.4.bn2.running_var",
    (256,),
    "block3.5.conv_bn1.conv.weight",
    (256, 256, 3, 3),
    "layer3.5.conv1.weight",
    (256, 256, 3, 3),
    "block3.5.conv_bn1.bn.weight",
    (256,),
    "layer3.5.bn1.weight",
    (256,),
    "block3.5.conv_bn1.bn.bias",
    (256,),
    "layer3.5.bn1.bias",
    (256,),
    "block3.5.conv_bn1.bn.running_mean",
    (256,),
    "layer3.5.bn1.running_mean",
    (256,),
    "block3.5.conv_bn1.bn.running_var",
    (256,),
    "layer3.5.bn1.running_var",
    (256,),
    "block3.5.conv_bn2.conv.weight",
    (256, 256, 3, 3),
    "layer3.5.conv2.weight",
    (256, 256, 3, 3),
    "block3.5.conv_bn2.bn.weight",
    (256,),
    "layer3.5.bn2.weight",
    (256,),
    "block3.5.conv_bn2.bn.bias",
    (256,),
    "layer3.5.bn2.bias",
    (256,),
    "block3.5.conv_bn2.bn.running_mean",
    (256,),
    "layer3.5.bn2.running_mean",
    (256,),
    "block3.5.conv_bn2.bn.running_var",
    (256,),
    "layer3.5.bn2.running_var",
    (256,),
    "block4.0.conv_bn1.conv.weight",
    (512, 256, 3, 3),
    "layer4.0.conv1.weight",
    (512, 256, 3, 3),
    "block4.0.conv_bn1.bn.weight",
    (512,),
    "layer4.0.bn1.weight",
    (512,),
    "block4.0.conv_bn1.bn.bias",
    (512,),
    "layer4.0.bn1.bias",
    (512,),
    "block4.0.conv_bn1.bn.running_mean",
    (512,),
    "layer4.0.bn1.running_mean",
    (512,),
    "block4.0.conv_bn1.bn.running_var",
    (512,),
    "layer4.0.bn1.running_var",
    (512,),
    "block4.0.conv_bn2.conv.weight",
    (512, 512, 3, 3),
    "layer4.0.conv2.weight",
    (512, 512, 3, 3),
    "block4.0.conv_bn2.bn.weight",
    (512,),
    "layer4.0.bn2.weight",
    (512,),
    "block4.0.conv_bn2.bn.bias",
    (512,),
    "layer4.0.bn2.bias",
    (512,),
    "block4.0.conv_bn2.bn.running_mean",
    (512,),
    "layer4.0.bn2.running_mean",
    (512,),
    "block4.0.conv_bn2.bn.running_var",
    (512,),
    "layer4.0.bn2.running_var",
    (512,),
    "block4.0.shortcut.conv.weight",
    (512, 256, 1, 1),
    "layer4.0.downsample.0.weight",
    (512, 256, 1, 1),
    "block4.0.shortcut.bn.weight",
    (512,),
    "layer4.0.downsample.1.weight",
    (512,),
    "block4.0.shortcut.bn.bias",
    (512,),
    "layer4.0.downsample.1.bias",
    (512,),
    "block4.0.shortcut.bn.running_mean",
    (512,),
    "layer4.0.downsample.1.running_mean",
    (512,),
    "block4.0.shortcut.bn.running_var",
    (512,),
    "layer4.0.downsample.1.running_var",
    (512,),
    "block4.1.conv_bn1.conv.weight",
    (512, 512, 3, 3),
    "layer4.1.conv1.weight",
    (512, 512, 3, 3),
    "block4.1.conv_bn1.bn.weight",
    (512,),
    "layer4.1.bn1.weight",
    (512,),
    "block4.1.conv_bn1.bn.bias",
    (512,),
    "layer4.1.bn1.bias",
    (512,),
    "block4.1.conv_bn1.bn.running_mean",
    (512,),
    "layer4.1.bn1.running_mean",
    (512,),
    "block4.1.conv_bn1.bn.running_var",
    (512,),
    "layer4.1.bn1.running_var",
    (512,),
    "block4.1.conv_bn2.conv.weight",
    (512, 512, 3, 3),
    "layer4.1.conv2.weight",
    (512, 512, 3, 3),
    "block4.1.conv_bn2.bn.weight",
    (512,),
    "layer4.1.bn2.weight",
    (512,),
    "block4.1.conv_bn2.bn.bias",
    (512,),
    "layer4.1.bn2.bias",
    (512,),
    "block4.1.conv_bn2.bn.running_mean",
    (512,),
    "layer4.1.bn2.running_mean",
    (512,),
    "block4.1.conv_bn2.bn.running_var",
    (512,),
    "layer4.1.bn2.running_var",
    (512,),
    "block4.2.conv_bn1.conv.weight",
    (512, 512, 3, 3),
    "layer4.2.conv1.weight",
    (512, 512, 3, 3),
    "block4.2.conv_bn1.bn.weight",
    (512,),
    "layer4.2.bn1.weight",
    (512,),
    "block4.2.conv_bn1.bn.bias",
    (512,),
    "layer4.2.bn1.bias",
    (512,),
    "block4.2.conv_bn1.bn.running_mean",
    (512,),
    "layer4.2.bn1.running_mean",
    (512,),
    "block4.2.conv_bn1.bn.running_var",
    (512,),
    "layer4.2.bn1.running_var",
    (512,),
    "block4.2.conv_bn2.conv.weight",
    (512, 512, 3, 3),
    "layer4.2.conv2.weight",
    (512, 512, 3, 3),
    "block4.2.conv_bn2.bn.weight",
    (512,),
    "layer4.2.bn2.weight",
    (512,),
    "block4.2.conv_bn2.bn.bias",
    (512,),
    "layer4.2.bn2.bias",
    (512,),
    "block4.2.conv_bn2.bn.running_mean",
    (512,),
    "layer4.2.bn2.running_mean",
    (512,),
    "block4.2.conv_bn2.bn.running_var",
    (512,),
    "layer4.2.bn2.running_var",
    (512,),
    "logit.weight",
    (1000, 512),
    "fc.weight",
    (1000, 512),
    "logit.bias",
    (1000,),
    "fc.bias",
    (1000,),
]


###############################################################################
class ConvBn2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


#############  resnext50 pyramid feature net #######################################
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py


# bottleneck type C
class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):
        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(
            in_channel, channel, kernel_size=3, padding=1, stride=stride
        )
        self.conv_bn2 = ConvBn2d(
            channel, out_channel, kernel_size=3, padding=1, stride=1
        )

        if is_shortcut:
            self.shortcut = ConvBn2d(
                in_channel, out_channel, kernel_size=1, padding=0, stride=stride
            )

    def forward(self, x):
        z = F.relu(self.conv_bn1(x), inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z, inplace=True)
        return z


class ResNet34(nn.Module):
    def __init__(self, num_class=1000):
        super(ResNet34, self).__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            BasicBlock(64, 64, 64, stride=1, is_shortcut=False),
            *[BasicBlock(64, 64, 64, stride=1, is_shortcut=False) for i in range(1, 3)],
        )
        self.block2 = nn.Sequential(
            BasicBlock(64, 128, 128, stride=2, is_shortcut=True),
            *[
                BasicBlock(128, 128, 128, stride=1, is_shortcut=False)
                for i in range(1, 4)
            ],
        )
        self.block3 = nn.Sequential(
            BasicBlock(128, 256, 256, stride=2, is_shortcut=True),
            *[
                BasicBlock(256, 256, 256, stride=1, is_shortcut=False)
                for i in range(1, 6)
            ],
        )
        self.block4 = nn.Sequential(
            BasicBlock(256, 512, 512, stride=2, is_shortcut=True),
            *[
                BasicBlock(512, 512, 512, stride=1, is_shortcut=False)
                for i in range(1, 3)
            ],
        )
        self.logit = nn.Linear(512, num_class)

    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        logit = self.logit(x)
        return logit


class Resnet34_classification(nn.Module):
    def __init__(self, num_class=4):
        super(Resnet34_classification, self).__init__()
        e = ResNet34()
        self.block = nn.ModuleList([e.block0, e.block1, e.block2, e.block3, e.block4])
        e = None  # dropped
        self.feature = nn.Conv2d(512, 32, kernel_size=1)  # dummy conv for dim reduction
        self.logit = nn.Conv2d(32, num_class, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        for i in range(len(self.block)):
            x = self.block[i](x)
            # print(i, x.shape)

        x = F.dropout(x, 0.5, training=self.training)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)
        logit = self.logit(x)
        return logit


model_classification = Resnet34_classification()
model_classification.load_state_dict(
    torch.load(
        "../input/clsification/00007500_model.pth",
        map_location=lambda storage, loc: storage,
    ),
    strict=True,
)


class TestDataset(Dataset):
    """Dataset for test prediction"""

    def __init__(self, root, df, mean, std):
        self.root = root
        df["ImageId"] = df["ImageId_ClassId"].apply(lambda x: x.split("_")[0])
        self.fnames = df["ImageId"].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = Compose([Normalize(mean=mean, std=std, p=1), ToTensor()])

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname)
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return fname, images

    def __len__(self):
        return self.num_samples


def sharpen(p, t=0.5):
    if t != 0:
        return p ** t
    else:
        return p


augment = ["null"]


def get_classification_preds(net, test_loader):
    test_probability_label = []
    test_id = []

    net = net.cuda()
    for t, (fnames, images) in enumerate(tqdm(test_loader)):
        batch_size, C, H, W = images.shape
        images = images.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1:  # null
                logit = net(images)
                probability = torch.sigmoid(logit)

                probability_label = sharpen(probability, 0)
                num_augment += 1

            if "flip_lr" in augment:
                logit = net(torch.flip(images, dims=[3]))
                probability = torch.sigmoid(logit)

                probability_label += sharpen(probability)
                num_augment += 1

            if "flip_ud" in augment:
                logit = net(torch.flip(images, dims=[2]))
                probability = torch.sigmoid(logit)

                probability_label += sharpen(probability)
                num_augment += 1

            probability_label = probability_label / num_augment

        probability_label = probability_label.data.cpu().numpy()

        test_probability_label.append(probability_label)
        test_id.extend([i for i in fnames])

    test_probability_label = np.concatenate(test_probability_label)
    return test_probability_label, test_id


sample_submission_path = (
    "../input/severstal-steel-defect-detection/sample_submission.csv"
)
test_data_folder = "../input/severstal-steel-defect-detection/test_images"
batch_size = 1

# mean and std
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
threshold_label = [0.50, 0.50, 0.50, 0.50]
probability_label, image_id = get_classification_preds(model_classification, testset)
predict_label = probability_label > np.array(threshold_label).reshape(1, 4, 1, 1)

image_id_class_id = []
encoded_pixel = []
for b in range(len(image_id)):
    for c in range(4):
        image_id_class_id.append(image_id[b] + "_%d" % (c + 1))
        if predict_label[b, c] == 0:
            rle = ""
        else:
            rle = "1 1"
        encoded_pixel.append(rle)

df_classification = pd.DataFrame(
    zip(image_id_class_id, encoded_pixel), columns=["ImageId_ClassId", "EncodedPixels"]
)


df = pd.read_csv("submission.csv")
df = df.fillna("")

if 1:
    df["Class"] = df["ImageId_ClassId"].str[-1].astype(np.int32)
    df["Label"] = (df["EncodedPixels"] != "").astype(np.int32)
    pos1 = ((df["Class"] == 1) & (df["Label"] == 1)).sum()
    pos2 = ((df["Class"] == 2) & (df["Label"] == 1)).sum()
    pos3 = ((df["Class"] == 3) & (df["Label"] == 1)).sum()
    pos4 = ((df["Class"] == 4) & (df["Label"] == 1)).sum()

    num_image = len(df) // 4
    num = len(df)
    pos = (df["Label"] == 1).sum()
    neg = num - pos

    print("")
    print("\t\tnum_image = %5d(1801)" % num_image)
    print("\t\tnum  = %5d(7204)" % num)
    print("\t\tneg  = %5d(6172)  %0.3f" % (neg, neg / num))
    print("\t\tpos  = %5d(1032)  %0.3f" % (pos, pos / num))
    print("\t\tpos1 = %5d( 128)  %0.3f  %0.3f" % (pos1, pos1 / num_image, pos1 / pos))
    print("\t\tpos2 = %5d(  43)  %0.3f  %0.3f" % (pos2, pos2 / num_image, pos2 / pos))
    print("\t\tpos3 = %5d( 741)  %0.3f  %0.3f" % (pos3, pos3 / num_image, pos3 / pos))
    print("\t\tpos4 = %5d( 120)  %0.3f  %0.3f" % (pos4, pos4 / num_image, pos4 / pos))

df_mask = df.copy()
df_label = df_classification.copy()


assert np.all(df_mask["ImageId_ClassId"].values == df_label["ImageId_ClassId"].values)
print(
    (df_mask.loc[df_label["EncodedPixels"] == "", "EncodedPixels"] != "").sum()
)  # 202
df_mask.loc[df_label["EncodedPixels"] == "", "EncodedPixels"] = ""


# df_mask.to_csv("submission.csv", index=False)
df_mask.to_csv(
    "submission.csv", columns=["ImageId_ClassId", "EncodedPixels"], index=False
)
if 1:
    df_mask["Class"] = df_mask["ImageId_ClassId"].str[-1].astype(np.int32)
    df_mask["Label"] = (df_mask["EncodedPixels"] != "").astype(np.int32)
    pos1 = ((df_mask["Class"] == 1) & (df_mask["Label"] == 1)).sum()
    pos2 = ((df_mask["Class"] == 2) & (df_mask["Label"] == 1)).sum()
    pos3 = ((df_mask["Class"] == 3) & (df_mask["Label"] == 1)).sum()
    pos4 = ((df_mask["Class"] == 4) & (df_mask["Label"] == 1)).sum()

    num_image = len(df_mask) // 4
    num = len(df_mask)
    pos = (df_mask["Label"] == 1).sum()
    neg = num - pos

    print("")
    print("\t\tnum_image = %5d(1801)" % num_image)
    print("\t\tnum  = %5d(7204)" % num)
    print("\t\tneg  = %5d(6172)  %0.3f" % (neg, neg / num))
    print("\t\tpos  = %5d(1032)  %0.3f" % (pos, pos / num))
    print("\t\tpos1 = %5d( 128)  %0.3f  %0.3f" % (pos1, pos1 / num_image, pos1 / pos))
    print("\t\tpos2 = %5d(  43)  %0.3f  %0.3f" % (pos2, pos2 / num_image, pos2 / pos))
    print("\t\tpos3 = %5d( 741)  %0.3f  %0.3f" % (pos3, pos3 / num_image, pos3 / pos))
    print("\t\tpos4 = %5d( 120)  %0.3f  %0.3f" % (pos4, pos4 / num_image, pos4 / pos))


# ### Visualization


get_ipython().run_line_magic("matplotlib", "inline")

df = pd.read_csv("submission.csv")[:40]
df["Image"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[0])
df["Class"] = df["ImageId_ClassId"].map(lambda x: x.split("_")[1])

for row in df.itertuples():
    img_path = os.path.join(img_folder, row.Image)
    img = cv2.imread(img_path)
    mask = (
        rle2mask(row.EncodedPixels, (1600, 256))
        if isinstance(row.EncodedPixels, str)
        else np.zeros((256, 1600))
    )
    if mask.sum() == 0:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(20, 60))
    axes[0].imshow(img / 255)
    axes[1].imshow(mask * 60)
    axes[0].set_title(row.Image)
    axes[1].set_title(row.Class)
    plt.show()
