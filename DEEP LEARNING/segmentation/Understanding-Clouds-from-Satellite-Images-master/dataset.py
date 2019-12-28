import os
import cv2


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
import warnings
from augs import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)

warnings.filterwarnings("once")


def get_img(x: str = "img_name", folder: str = "train_images"):
    """
    Return image based on image name and folder.

    Args:
        x: image name
        folder: folder with images

    Returns:

    """
    image_path = os.path.join(folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = "", shape: tuple = (1400, 2100)):
    """
    Decode rle encoded mask.

    Args:
        mask_rle: encoded mask
        shape: final shape

    Returns:

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def make_mask(
    df: pd.DataFrame, image_name: str = "img.jpg", shape: tuple = (1400, 2100)
):
    """
    Create mask based on df, image name and shape.

    Args:
        df: dataframe with cloud dataset
        image_name: image name
        shape: final shape

    Returns:

    """

    encoded_masks = df.loc[df["im_id"] == image_name, "EncodedPixels"]
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def mask2rle(img):
    """
    Convert mask to rle.

    Args:
        img:

    Returns:

    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class CloudDataset(Dataset):
    def __init__(
        self,
        path: str = "",
        df: pd.DataFrame = None,
        datatype: str = "train",
        img_ids: np.array = None,
        transforms=albu.Compose([albu.HorizontalFlip()]),
        preprocessing=None,
        preload: bool = False,
        image_size: tuple = (320, 640),
        augmentation: str = "default",
        filter_bad_images: bool = False,
    ):
        """

        Args:
            path: path to data
            df: dataframe with data
            datatype: train|valid|test
            img_ids: list of imagee ids
            transforms: albumentation transforms
            preprocessing: preprocessing if necessary
            preload: whether to preload data
            image_size: image size for resizing
            augmentation: name of augmentation settings
            filter_bad_images: to filter out bad images
        """

        self.df = df
        self.path = path
        self.datatype = datatype if datatype == "test" else "train"
        if self.datatype != "test":
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        # list of bad images from discussions
        self.bad_imgs = [
            "046586a.jpg",
            "1588d4c.jpg",
            "1e40a05.jpg",
            "41f92e5.jpg",
            "449b792.jpg",
            "563fc48.jpg",
            "8bd81ce.jpg",
            "c0306e5.jpg",
            "c26c635.jpg",
            "e04fea3.jpg",
            "e5f2f24.jpg",
            "eda52f2.jpg",
            "fa645da.jpg",
        ]
        if filter_bad_images:
            self.img_ids = [i for i in self.img_ids if i not in self.bad_imgs]
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.dir_name = (
            f"{self.path}/preload_{augmentation}_{image_size[0]}_{image_size[1]}"
        )

        self.preload = preload
        self.preloaded = False
        if self.preload:
            self.save_processed_()
            self.preloaded = True

    def save_processed_(self):
        """
        Saves train images with augmentations, to speed up training.

        Returns:

        """
        os.makedirs(self.dir_name, exist_ok=True)
        self.dir_name += f"/{self.datatype}"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            for i, e in enumerate(self.img_ids):
                img, mask = self.__getitem__(i)
                np.save(f"{self.dir_name}/{e}_mask.npy", mask)
                np.save(f"{self.dir_name}/{e}_img.npy", img)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        if self.preloaded and self.datatype != "valid":
            img = np.load(f"{self.dir_name}/{image_name}_img.npy")
            mask = np.load(f"{self.dir_name}/{image_name}_mask.npy")

        else:
            mask = make_mask(self.df, image_name)
            image_path = os.path.join(self.data_folder, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed["image"]
                mask = preprocessed["mask"]

        return img, mask

    def __len__(self):
        return len(self.img_ids)


class CloudDatasetClassification(Dataset):
    def __init__(
        self,
        path: str = "",
        df: pd.DataFrame = None,
        datatype: str = "train",
        img_ids: np.array = None,
        transforms=albu.Compose([albu.HorizontalFlip()]),
        preprocessing=None,
        preload: bool = False,
        image_size: tuple = (320, 640),
        augmentation: str = "default",
        one_hot_labels: dict = None,
        filter_bad_images: bool = False,
    ):
        """

        Args:
            path: path to data
            df: dataframe with data
            datatype: train|valid|test
            img_ids: list of imagee ids
            transforms: albumentation transforms
            preprocessing: preprocessing if necessary
            preload: whether to preload data
            image_size: image size for resizing
            augmentation: name of augmentation settings
            one_hot_labels: dictionary with labels for images
            filter_bad_images: to filter out bad images
        """
        self.df = df
        self.path = path
        self.datatype = datatype if datatype == "test" else "train"
        if self.datatype != "test":
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.bad_imgs = [
            "046586a.jpg",
            "1588d4c.jpg",
            "1e40a05.jpg",
            "41f92e5.jpg",
            "449b792.jpg",
            "563fc48.jpg",
            "8bd81ce.jpg",
            "c0306e5.jpg",
            "c26c635.jpg",
            "e04fea3.jpg",
            "e5f2f24.jpg",
            "eda52f2.jpg",
            "fa645da.jpg",
        ]
        if filter_bad_images:
            self.img_ids = [i for i in self.img_ids if i not in self.bad_imgs]
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.dir_name = (
            f"{self.path}/preload_{augmentation}_{image_size[0]}_{image_size[1]}"
        )
        self.one_hot_labels = one_hot_labels

        self.preload = preload
        self.preloaded = False
        if self.preload:
            self.save_processed_()
            self.preloaded = True

    def save_processed_(self):

        os.makedirs(self.dir_name, exist_ok=True)
        self.dir_name += f"/{self.datatype}"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            for i, e in enumerate(self.img_ids):
                img, mask = self.__getitem__(i)
                np.save(f"{self.dir_name}/{e}_img.npy", img)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        if self.preloaded and self.datatype != "valid":
            img = np.load(f"{self.dir_name}/{image_name}_img.npy")

        else:
            image_path = os.path.join(self.data_folder, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transforms(image=img)
            img = augmented["image"]
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img)
                img = preprocessed["image"]

            label = self.one_hot_labels[image_name]

        return img, label

    def __len__(self):
        return len(self.img_ids)


def prepare_loaders(
    path: str = "",
    bs: int = 4,
    num_workers: int = 0,
    preprocessing_fn=None,
    preload: bool = False,
    image_size: tuple = (320, 640),
    augmentation: str = "default",
    task: str = "segmentation",
):
    """
    Prepare dataloaders for catalyst.

    At first reads dataframe with the data and prepares it to be used in dataloaders.
    Creates dataloaders and returns them.

    Args:
        path: path to data
        bs: batch size
        num_workers: number of workers
        preprocessing_fn: preprocessing
        preload: whether to save augmented data on disk
        image_size: image size to resize
        augmentation: augmentation name
        task: segmentation or classification

    Returns:

    """

    train = pd.read_csv(f"{path}/train.csv")
    train["label"] = train["Image_Label"].apply(lambda x: x.split("_")[1])
    train["im_id"] = train["Image_Label"].apply(lambda x: x.split("_")[0])

    id_mask_count = (
        train.loc[~train["EncodedPixels"].isnull(), "Image_Label"]
        .apply(lambda x: x.split("_")[0])
        .value_counts()
        .reset_index()
        .rename(columns={"index": "img_id", "Image_Label": "count"})
    )
    train_ids, valid_ids = train_test_split(
        id_mask_count["img_id"].values,
        random_state=42,
        shuffle=True,
        # stratify=id_mask_count['count'],
        test_size=0.1,
    )

    if task == "classification":
        train_df = train[~train["EncodedPixels"].isnull()]
        classes = train_df["label"].unique()
        train_df = train_df.groupby("im_id")["label"].agg(set).reset_index()
        for class_name in classes:
            train_df[class_name] = train_df["label"].map(
                lambda x: 1 if class_name in x else 0
            )

        img_2_ohe_vector = {
            img: np.float32(vec)
            for img, vec in zip(train_df["im_id"], train_df.iloc[:, 2:].values)
        }

    sub = pd.read_csv(f"{path}/sample_submission.csv")
    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])
    test_ids = (
        sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values
    )

    if task == "segmentation":
        if preload:
            _ = CloudDataset(
                path=path,
                df=train,
                datatype="train",
                img_ids=id_mask_count["img_id"].values,
                transforms=get_training_augmentation(
                    augmentation=augmentation, image_size=image_size
                ),
                preprocessing=get_preprocessing(preprocessing_fn),
                preload=preload,
                image_size=(320, 640),
            )

        train_dataset = CloudDataset(
            path=path,
            df=train,
            datatype="train",
            img_ids=train_ids,
            transforms=get_training_augmentation(
                augmentation=augmentation, image_size=image_size
            ),
            preprocessing=get_preprocessing(preprocessing_fn),
            preload=preload,
            image_size=(320, 640),
        )
        valid_dataset = CloudDataset(
            path=path,
            df=train,
            datatype="valid",
            img_ids=valid_ids,
            transforms=get_validation_augmentation(image_size=image_size),
            preprocessing=get_preprocessing(preprocessing_fn),
            preload=preload,
            image_size=(320, 640),
        )

    elif task == "classification":
        if preload:
            _ = CloudDatasetClassification(
                path=path,
                df=train,
                datatype="train",
                img_ids=id_mask_count["img_id"].values,
                transforms=get_training_augmentation(
                    augmentation=augmentation, image_size=image_size
                ),
                preprocessing=get_preprocessing(preprocessing_fn),
                preload=preload,
                image_size=(320, 640),
                one_hot_labels=img_2_ohe_vector,
            )

        train_dataset = CloudDatasetClassification(
            path=path,
            df=train,
            datatype="train",
            img_ids=train_ids,
            transforms=get_training_augmentation(
                augmentation=augmentation, image_size=image_size
            ),
            preprocessing=get_preprocessing(preprocessing_fn),
            preload=preload,
            image_size=(320, 640),
            one_hot_labels=img_2_ohe_vector,
        )
        valid_dataset = CloudDatasetClassification(
            path=path,
            df=train,
            datatype="valid",
            img_ids=valid_ids,
            transforms=get_validation_augmentation(image_size=image_size),
            preprocessing=get_preprocessing(preprocessing_fn),
            preload=preload,
            image_size=(320, 640),
            one_hot_labels=img_2_ohe_vector,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataset = CloudDataset(
        path=path,
        df=sub,
        datatype="test",
        img_ids=test_ids,
        transforms=get_validation_augmentation(image_size=image_size),
        preprocessing=get_preprocessing(preprocessing_fn),
        preload=preload,
        image_size=(320, 640),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs // 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

    return loaders
