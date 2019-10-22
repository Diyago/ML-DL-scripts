import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from attrdict import AttrDict
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader
from imgaug import augmenters as iaa
from functools import partial
from itertools import product
import multiprocessing as mp
from scipy.stats import gmean
from tqdm import tqdm
import json
from steppy.base import BaseTransformer

from .utils import from_pil, to_pil, binary_from_rle, ImgAug


class ImageReader(BaseTransformer):
    def __init__(self, train_mode, x_columns, y_columns, target_format="png"):
        self.train_mode = train_mode
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.target_format = target_format

    def transform(self, meta):
        X_ = meta[self.x_columns].values

        X = self.load_images(X_, filetype="png", grayscale=False)
        if self.train_mode:
            y_ = meta[self.y_columns].values
            y = self.load_images(y_, filetype=self.target_format, grayscale=True)
        else:
            y = None

        return {"X": X, "y": y}

    def load_images(self, filepaths, filetype, grayscale=False):
        X = []
        for i in range(filepaths.shape[1]):
            column = filepaths[:, i]
            X.append([])
            for filepath in tqdm(column):
                if filetype == "png":
                    data = self.load_image(filepath, grayscale=grayscale)
                elif filetype == "json":
                    data = self.read_json(filepath)
                else:
                    raise Exception("files must be png or json")
                X[i].append(data)
        return X

    def load_image(self, img_filepath, grayscale):
        image = Image.open(img_filepath, "r")
        if not grayscale:
            image = image.convert("RGB")
        else:
            image = image.convert("L").point(lambda x: 0 if x < 128 else 255, "1")
        return image

    def read_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        masks = [to_pil(binary_from_rle(rle)) for rle in data]
        return masks


class XYSplit(BaseTransformer):
    def __init__(self, train_mode, x_columns, y_columns):
        self.train_mode = train_mode
        super().__init__()
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.columns_to_get = None
        self.target_columns = None

    def transform(self, meta):
        X = meta[self.x_columns[0]].values
        if self.train_mode:
            y = meta[self.y_columns[0]].values
        else:
            y = None

        return {"X": X, "y": y}


class ImageSegmentationBaseDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        train_mode,
        image_transform,
        image_augment_with_target,
        mask_transform,
        image_augment,
        image_source="memory",
    ):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
        else:
            self.y = None

        self.train_mode = train_mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.image_augment = (
            image_augment if image_augment is not None else ImgAug(iaa.Noop())
        )
        self.image_augment_with_target = (
            image_augment_with_target
            if image_augment_with_target is not None
            else ImgAug(iaa.Noop())
        )

        self.image_source = image_source

    def __len__(self):
        if self.image_source == "memory":
            return len(self.X[0])
        elif self.image_source == "disk":
            return self.X.shape[0]

    def __getitem__(self, index):
        if self.image_source == "memory":
            load_func = self.load_from_memory
        elif self.image_source == "disk":
            load_func = self.load_from_disk
        else:
            raise NotImplementedError("Possible loading options: 'memory' and 'disk'!")

        Xi = load_func(self.X, index, filetype="png", grayscale=False)

        if self.y is not None:
            Mi = self.load_target(self.y, index, load_func)
            Xi, *Mi = from_pil(Xi, *Mi)
            Xi, *Mi = self.image_augment_with_target(Xi, *Mi)
            Xi = self.image_augment(Xi)
            Xi, *Mi = to_pil(Xi, *Mi)

            if self.mask_transform is not None:
                Mi = [self.mask_transform(m) for m in Mi]

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)

            Mi = torch.cat(Mi, dim=0)
            return Xi, Mi
        else:
            Xi = from_pil(Xi)
            Xi = self.image_augment(Xi)
            Xi = to_pil(Xi)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi

    def load_from_memory(self, data_source, index, **kwargs):
        return data_source[0][index]

    def load_from_disk(self, data_source, index, *, filetype, grayscale=False):
        if filetype == "png":
            img_filepath = data_source[index]
            return self.load_image(img_filepath, grayscale=grayscale)
        elif filetype == "json":
            json_filepath = data_source[index]
            return self.read_json(json_filepath)
        else:
            raise Exception("files must be png or json")

    def load_image(self, img_filepath, grayscale):
        image = Image.open(img_filepath, "r")
        if not grayscale:
            image = image.convert("RGB")
        else:
            image = image.convert("L").point(lambda x: 0 if x < 128 else 1)
        return image

    def read_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        masks = [to_pil(binary_from_rle(rle)) for rle in data]
        return masks

    def load_target(self, data_source, index, load_func):
        raise NotImplementedError


class ImageSegmentationJsonDataset(ImageSegmentationBaseDataset):
    def load_target(self, data_source, index, load_func):
        Mi = load_func(data_source, index, filetype="json")
        return Mi


class ImageSegmentationPngDataset(ImageSegmentationBaseDataset):
    def load_target(self, data_source, index, load_func):
        Mi = load_func(data_source, index, filetype="png", grayscale=True)
        Mi = from_pil(Mi)
        target = [to_pil(Mi == class_nr) for class_nr in [0, 1]]
        return target


class ImageSegmentationTTADataset(ImageSegmentationBaseDataset):
    def __init__(self, tta_params, tta_transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tta_params = tta_params
        self.tta_transform = tta_transform

    def __getitem__(self, index):
        if self.image_source == "memory":
            load_func = self.load_from_memory
        elif self.image_source == "disk":
            load_func = self.load_from_disk
        else:
            raise NotImplementedError("Possible loading options: 'memory' and 'disk'!")

        Xi = load_func(self.X, index, filetype="png", grayscale=False)
        Xi = from_pil(Xi)

        if self.image_augment is not None:
            Xi = self.image_augment(Xi)

        if self.tta_params is not None:
            tta_transform_specs = self.tta_params[index]
            Xi = self.tta_transform(Xi, tta_transform_specs)
        Xi = to_pil(Xi)

        if self.image_transform is not None:
            Xi = self.image_transform(Xi)

        return Xi


class ImageSegmentationLoaderBasic(BaseTransformer):
    def __init__(self, train_mode, loader_params, dataset_params, augmentation_params):
        super().__init__()
        self.train_mode = train_mode
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)
        self.augmentation_params = AttrDict(augmentation_params)

        self.mask_transform = None
        self.image_transform = None

        self.image_augment_train = None
        self.image_augment_inference = None
        self.image_augment_with_target_train = None
        self.image_augment_with_target_inference = None

        self.dataset = None

    def transform(self, X, y, X_valid=None, y_valid=None, **kwargs):
        if self.train_mode and y is not None:
            flow, steps = self.get_datagen(X, y, True, self.loader_params.training)
        else:
            flow, steps = self.get_datagen(X, None, False, self.loader_params.inference)

        if X_valid is not None and y_valid is not None:
            valid_flow, valid_steps = self.get_datagen(
                X_valid, y_valid, False, self.loader_params.inference
            )
        else:
            valid_flow = None
            valid_steps = None

        return {
            "datagen": (flow, steps),
            "validation_datagen": (valid_flow, valid_steps),
        }

    def get_datagen(self, X, y, train_mode, loader_params):
        if train_mode:
            dataset = self.dataset(
                X,
                y,
                train_mode=True,
                image_augment=self.image_augment_train,
                image_augment_with_target=self.image_augment_with_target_train,
                mask_transform=self.mask_transform,
                image_transform=self.image_transform,
                image_source=self.dataset_params.image_source,
            )
        else:
            dataset = self.dataset(
                X,
                y,
                train_mode=False,
                image_augment=self.image_augment_inference,
                image_augment_with_target=self.image_augment_with_target_inference,
                mask_transform=self.mask_transform,
                image_transform=self.image_transform,
                image_source=self.dataset_params.image_source,
            )

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps

    def load(self, filepath):
        params = joblib.load(filepath)
        self.loader_params = params["loader_params"]
        return self

    def save(self, filepath):
        params = {"loader_params": self.loader_params}
        joblib.dump(params, filepath)


class ImageSegmentationLoaderBasicTTA(ImageSegmentationLoaderBasic):
    def __init__(self, loader_params, dataset_params, augmentation_params):
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)
        self.augmentation_params = AttrDict(augmentation_params)

        self.mask_transform = None
        self.image_transform = None

        self.image_augment_train = None
        self.image_augment_inference = None
        self.image_augment_with_target_train = None
        self.image_augment_with_target_inference = None

        self.dataset = None

    def transform(self, X, tta_params, **kwargs):
        flow, steps = self.get_datagen(X, tta_params, self.loader_params.inference)
        valid_flow = None
        valid_steps = None
        return {
            "datagen": (flow, steps),
            "validation_datagen": (valid_flow, valid_steps),
        }

    def get_datagen(self, X, tta_params, loader_params):
        dataset = self.dataset(
            tta_params=tta_params,
            tta_transform=self.augmentation_params.tta_transform,
            X=X,
            y=None,
            train_mode=False,
            image_augment=self.image_augment_inference,
            image_augment_with_target=self.image_augment_with_target_inference,
            mask_transform=self.mask_transform,
            image_transform=self.image_transform,
            image_source=self.dataset_params.image_source,
        )

        datagen = DataLoader(dataset, **loader_params)
        steps = len(datagen)
        return datagen, steps


class ImageSegmentationLoaderResizePad(ImageSegmentationLoaderBasic):
    def __init__(self, train_mode, loader_params, dataset_params, augmentation_params):
        super().__init__(train_mode, loader_params, dataset_params, augmentation_params)

        self.image_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.dataset_params.MEAN, std=self.dataset_params.STD
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [transforms.Lambda(to_array), transforms.Lambda(to_tensor)]
        )

        self.image_augment_train = ImgAug(
            self.augmentation_params["image_augment_train"]
        )
        self.image_augment_with_target_train = ImgAug(
            self.augmentation_params["image_augment_with_target_train"]
        )
        self.image_augment_inference = ImgAug(
            self.augmentation_params["image_augment_inference"]
        )
        self.image_augment_with_target_inference = ImgAug(
            self.augmentation_params["image_augment_with_target_inference"]
        )

        if self.dataset_params.target_format == "png":
            self.dataset = ImageSegmentationPngDataset
        elif self.dataset_params.target_format == "json":
            self.dataset = ImageSegmentationJsonDataset
        else:
            raise Exception("files must be png or json")


class ImageSegmentationLoaderPadTTA(ImageSegmentationLoaderBasicTTA):
    def __init__(self, loader_params, dataset_params, augmentation_params):
        super().__init__(loader_params, dataset_params, augmentation_params)

        self.image_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.dataset_params.MEAN, std=self.dataset_params.STD
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [transforms.Lambda(to_array), transforms.Lambda(to_tensor)]
        )

        self.image_augment_inference = ImgAug(
            self.augmentation_params["image_augment_inference"]
        )
        self.image_augment_with_target_inference = ImgAug(
            self.augmentation_params["image_augment_with_target_inference"]
        )
        self.dataset = ImageSegmentationTTADataset


class ImageSegmentationLoaderResize(ImageSegmentationLoaderBasic):
    def __init__(self, train_mode, loader_params, dataset_params, augmentation_params):
        super().__init__(train_mode, loader_params, dataset_params, augmentation_params)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.dataset_params.h, self.dataset_params.w)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.dataset_params.MEAN, std=self.dataset_params.STD
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.dataset_params.h, self.dataset_params.w), interpolation=0
                ),
                transforms.Lambda(to_array),
                transforms.Lambda(to_tensor),
            ]
        )

        self.image_augment_train = ImgAug(
            self.augmentation_params["image_augment_train"]
        )
        self.image_augment_with_target_train = ImgAug(
            self.augmentation_params["image_augment_with_target_train"]
        )

        if self.dataset_params.target_format == "png":
            self.dataset = ImageSegmentationPngDataset
        elif self.dataset_params.target_format == "json":
            self.dataset = ImageSegmentationJsonDataset
        else:
            raise Exception("files must be png or json")


class ImageSegmentationLoaderResizeTTA(ImageSegmentationLoaderBasicTTA):
    def __init__(self, loader_params, dataset_params, augmentation_params):
        super().__init__(loader_params, dataset_params, augmentation_params)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.dataset_params.h, self.dataset_params.w)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.dataset_params.MEAN, std=self.dataset_params.STD
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.dataset_params.h, self.dataset_params.w), interpolation=0
                ),
                transforms.Lambda(to_array),
                transforms.Lambda(to_tensor),
            ]
        )

        self.dataset = ImageSegmentationTTADataset


class MetaTestTimeAugmentationGenerator(BaseTransformer):
    def __init__(self, **kwargs):
        self.tta_transformations = AttrDict(kwargs)

    def transform(self, X, **kwargs):
        X_tta_rows, tta_params, img_ids = [], [], []
        for i in range(len(X)):
            rows, params, ids = self._get_tta_data(i, X[i])
            tta_params.extend(params)
            img_ids.extend(ids)
            X_tta_rows.extend(rows)
        X_tta = np.array(X_tta_rows)
        return {"X_tta": X_tta, "tta_params": tta_params, "img_ids": img_ids}

    def _get_tta_data(self, i, row):
        original_specs = {
            "ud_flip": False,
            "lr_flip": False,
            "rotation": 0,
            "color_shift": False,
        }
        tta_specs = [original_specs]

        ud_options = [True, False] if self.tta_transformations.flip_ud else [False]
        lr_options = [True, False] if self.tta_transformations.flip_lr else [False]
        rot_options = [0, 90, 180, 270] if self.tta_transformations.rotation else [0]
        if self.tta_transformations.color_shift_runs:
            color_shift_options = list(
                range(1, self.tta_transformations.color_shift_runs + 1, 1)
            )
        else:
            color_shift_options = [False]

        for ud, lr, rot, color in product(
            ud_options, lr_options, rot_options, color_shift_options
        ):
            if ud is False and lr is False and rot == 0 and color is False:
                continue
            else:
                tta_specs.append(
                    {
                        "ud_flip": ud,
                        "lr_flip": lr,
                        "rotation": rot,
                        "color_shift": color,
                    }
                )

        img_ids = [i] * len(tta_specs)
        X_rows = [row] * len(tta_specs)
        return X_rows, tta_specs, img_ids


class TestTimeAugmentationGenerator(BaseTransformer):
    def __init__(self, **kwargs):
        self.tta_transformations = AttrDict(kwargs)

    def transform(self, X, **kwargs):
        X_tta, tta_params, img_ids = [], [], []
        X = X[0]
        for i in range(len(X)):
            images, params, ids = self._get_tta_data(i, X[i])
            tta_params.extend(params)
            img_ids.extend(ids)
            X_tta.extend(images)
        return {"X_tta": [X_tta], "tta_params": tta_params, "img_ids": img_ids}

    def _get_tta_data(self, i, row):
        original_specs = {
            "ud_flip": False,
            "lr_flip": False,
            "rotation": 0,
            "color_shift": False,
        }
        tta_specs = [original_specs]

        ud_options = [True, False] if self.tta_transformations.flip_ud else [False]
        lr_options = [True, False] if self.tta_transformations.flip_lr else [False]
        rot_options = [0, 90, 180, 270] if self.tta_transformations.rotation else [0]
        if self.tta_transformations.color_shift_runs:
            color_shift_options = list(
                range(1, self.tta_transformations.color_shift_runs + 1, 1)
            )
        else:
            color_shift_options = [False]

        for ud, lr, rot, color in product(
            ud_options, lr_options, rot_options, color_shift_options
        ):
            if ud is False and lr is False and rot == 0 and color is False:
                continue
            else:
                tta_specs.append(
                    {
                        "ud_flip": ud,
                        "lr_flip": lr,
                        "rotation": rot,
                        "color_shift": color,
                    }
                )

        img_ids = [i] * len(tta_specs)
        X_rows = [row] * len(tta_specs)
        return X_rows, tta_specs, img_ids


class TestTimeAugmentationAggregator(BaseTransformer):
    def __init__(self, tta_inverse_transform, method, nthreads):
        self.tta_inverse_transform = tta_inverse_transform
        self.method = method
        self.nthreads = nthreads

    @property
    def agg_method(self):
        methods = {"mean": np.mean, "max": np.max, "min": np.min, "gmean": gmean}
        return partial(methods[self.method], axis=-1)

    def transform(self, images, tta_params, img_ids, **kwargs):
        _aggregate_augmentations = partial(
            aggregate_augmentations,
            images=images,
            tta_params=tta_params,
            tta_inverse_transform=self.tta_inverse_transform,
            img_ids=img_ids,
            agg_method=self.agg_method,
        )
        unique_img_ids = set(img_ids)
        threads = min(self.nthreads, len(unique_img_ids))
        with mp.pool.ThreadPool(threads) as executor:
            averages_images = executor.map(_aggregate_augmentations, unique_img_ids)
        return {"aggregated_prediction": averages_images}


def aggregate_augmentations(
    img_id, images, tta_params, tta_inverse_transform, img_ids, agg_method
):
    tta_predictions_for_id = []
    for image, tta_param, ids in zip(images, tta_params, img_ids):
        if ids == img_id:
            tta_prediction = tta_inverse_transform(image, tta_param)
            tta_predictions_for_id.append(tta_prediction)
        else:
            continue
    tta_averaged = agg_method(np.stack(tta_predictions_for_id, axis=-1))
    return tta_averaged


def to_array(x):
    x_ = x.convert("L")  # convert image to monochrome
    x_ = np.array(x_)
    x_ = x_.astype(np.float32)
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_
