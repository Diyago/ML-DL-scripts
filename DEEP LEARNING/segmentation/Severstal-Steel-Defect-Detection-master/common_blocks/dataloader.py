import os
from sklearn.model_selection import StratifiedKFold, KFold
import cv2
import joblib
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from .utils import make_mask
from .metric import Meter, epoch_log
from albumentations import (
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,
    CropNonEmptyMaskIfExists,
    Resize,
    Compose,
    RandomBrightnessContrast,
    VerticalFlip,
    RandomBrightness,
    RandomContrast,
)
from albumentations.pytorch import ToTensor
import sys

sys.path.append("..")
from configs.train_params import *

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class SteelDataset(Dataset):
    def __init__(self, df, data_folder, mean, std, phase, df_full, data_folder_full):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()
        self.df_full = df_full
        self.data_folder_full = data_folder_full
        self.df_full_index = self.df_full.reset_index(drop=True).index.tolist()

    def __getitem__(self, idx):
        cur_item_type = np.random.choice(["pseudo", "full"], p=[0.3, 0.7])
        if self.phase == "val" or cur_item_type == "pseudo":
            image_id, mask = make_mask(idx, self.df)
            image_path = os.path.join(self.root, image_id)
        else:
            idx = np.random.choice(self.df_full_index)
            image_id, mask = make_mask(idx, self.df_full)
            image_path = os.path.join(self.data_folder_full, image_id)

        img = cv2.imread(image_path)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]  # 1x256x1600x4
        mask = mask[0].permute(2, 0, 1)  # 1x4x256x1600
        return img, mask
        # {'features': img, 'masks': mask, 'mask': mask.argmax(axis=1)}

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        if crop_image_size is not None:
            list_transforms.extend(
                [
                    CropNonEmptyMaskIfExists(
                        crop_image_size[0], crop_image_size[1], p=0.85
                    ),
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    RandomBrightnessContrast(
                        p=0.1, brightness_limit=0.1, contrast_limit=0.1
                    ),
                ]
            )
        else:
            list_transforms.extend(
                [
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    RandomBrightnessContrast(
                        p=0.1, brightness_limit=0.1, contrast_limit=0.1
                    ),
                ]
            )
    list_transforms.extend([Normalize(mean=mean, std=std, p=1), ToTensor()])
    list_trfms = Compose(list_transforms)
    return list_trfms


def provider_trai_test_split(
    data_folder, df_path, phase, mean=None, std=None, batch_size=8, num_workers=4
):
    """
    Returns dataloader for the model training
    """

    df = pd.read_csv(df_path)
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df["ImageId"], df["ClassId"] = zip(*df["ImageId_ClassId"].str.split("_"))
    df["ClassId"] = df["ClassId"].astype(int)
    df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
    df["defects"] = df.count(axis=1)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["defects"], random_state=69
    )
    df = train_df if phase == "train" else val_df
    data_folder_cur = lb_test if phase == "train" else data_folder
    image_dataset = SteelDataset(df, data_folder_cur, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


def provider_cv(
    fold,
    total_folds,
    data_folder,
    df_path,
    phase,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
):
    if isDebug:
        df = pd.read_csv(df_path).head(200)
        df["ImageId"], df["ClassId"] = zip(*df["ImageId_ClassId"].str.split("_"))
        df["ClassId"] = df["ClassId"].astype(int)
        df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
        df["defects"] = df.count(axis=1)

        kfold = KFold(
            total_folds, shuffle=True, random_state=69
        )  # StratifiedKFold(total_folds, shuffle=True, random_state=69)
        train_idx, val_idx = list(kfold.split(df))[fold]  # , df["defects"]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    else:
        df = pd.read_csv(df_path)
        df["ImageId"], df["ClassId"] = zip(*df["ImageId_ClassId"].str.split("_"))
        df["ClassId"] = df["ClassId"].astype(int)
        df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
        df["defects"] = df.count(axis=1)
        folds_idx = joblib.load(FOLDS_ids)
        train_idx, val_idx = list(folds_idx)[fold]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        df = pd.read_csv(lb_test)
        df["ImageId"], df["ClassId"] = zip(*df["ImageId_ClassId"].str.split("_"))
        df["ClassId"] = df["ClassId"].astype(int)
        df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
        df["defects"] = df.count(axis=1)

    df_cur = df if phase == "train" else val_df
    folder_cur = test_data_folder if phase == "train" else data_folder
    print(df.shape)

    image_dataset = SteelDataset(
        df_cur, folder_cur, mean, std, phase, train_df, data_folder
    )
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


def provider_cv___(
    fold,
    total_folds,
    data_folder,
    df_path,
    phase,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
):
    """

    :param fold:
    :param total_folds:
    :param data_folder:
    :param df_path:
    :param phase:
    :param mean:
    :param std:
    :param batch_size:
    :param num_workers:
    :return:

    # example of usage
    dataloader = provider_cv(
    fold=0,
    total_folds=5,
    data_folder=data_folder,
    df_path=train_df_path,
    phase="train",
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    batch_size=16,
    num_workers=4,
)
    """
    if isDebug:
        df = pd.read_csv(df_path).head(200)
        df["ImageId"], df["ClassId"] = zip(*df["ImageId_ClassId"].str.split("_"))
        df["ClassId"] = df["ClassId"].astype(int)
        df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
        df["defects"] = df.count(axis=1)

        kfold = KFold(
            total_folds, shuffle=True, random_state=69
        )  # StratifiedKFold(total_folds, shuffle=True, random_state=69)
        train_idx, val_idx = list(kfold.split(df))[fold]  # , df["defects"]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    else:
        df = pd.read_csv(df_path)
        df["ImageId"], df["ClassId"] = zip(*df["ImageId_ClassId"].str.split("_"))
        df["ClassId"] = df["ClassId"].astype(int)
        df = df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")
        df["defects"] = df.count(axis=1)
        folds_idx = joblib.load(FOLDS_ids)
        train_idx, val_idx = list(folds_idx)[fold]
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    df = train_df if phase == "train" else val_df
    print(df.shape)

    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader
