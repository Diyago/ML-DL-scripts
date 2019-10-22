import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

from .utils import get_crop_pad_sequence, reseed


def _perspective_transform_augment_images(self, images, random_state, parents, hooks):
    result = images
    if not self.keep_size:
        result = list(result)

    matrices, max_heights, max_widths = self._create_matrices(
        [image.shape for image in images], random_state
    )

    for i, (M, max_height, max_width) in enumerate(
        zip(matrices, max_heights, max_widths)
    ):
        warped = cv2.warpPerspective(images[i], M, (max_width, max_height))
        if warped.ndim == 2 and images[i].ndim == 3:
            warped = np.expand_dims(warped, 2)
        if self.keep_size:
            h, w = images[i].shape[0:2]
            warped = ia.imresize_single_image(warped, (h, w))

        result[i] = warped

    return result


iaa.PerspectiveTransform._augment_images = _perspective_transform_augment_images

affine_seq = iaa.Sequential(
    [
        # General
        iaa.SomeOf(
            (1, 2),
            [
                iaa.Fliplr(0.5),
                iaa.Affine(
                    rotate=(-2, 2), translate_percent={"x": (-0.04, 0.04)}, mode="edge"
                )  # symmetric  (-0.25, 0.25)
                # iaa.CropAndPad(percent=((0.0, 0.10), (0.0, 0.05), (0.0, 0.10), (0.0, 0.05)), pad_mode='symmetric')
            ],
        ),
        # >>> aug = iaa.CropAndPad(px=((0, 10), (0, 5), (0, 10), (0, 5)))
        #   pads the top and bottom by a random value from the range 0px to 10px
        #    and the left and right by a random value in the range 0px to 5px.
        # Deformations
        iaa.Sometimes(0.05, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
        iaa.Sometimes(0.05, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
    ],
    random_order=True,
)
intensity_seq = iaa.Sequential([iaa.Noop()], random_order=False)
"""
intensity_seq = iaa.Sequential([
iaa.Invert(0.3),
iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
iaa.OneOf([
    iaa.Noop(),
    iaa.Sequential([
        iaa.OneOf([
            iaa.Add((-10, 10)),
            iaa.AddElementwise((-10, 10)),
            iaa.Multiply((0.95, 1.05)),
            iaa.MultiplyElementwise((0.95, 1.05)),
        ]),
    ]),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.AverageBlur(k=(2, 5)),
        iaa.MedianBlur(k=(3, 5))
    ])
])
], random_order=False)
"""

tta_intensity_seq = iaa.Sequential([iaa.Noop()], random_order=False)


def resize_pad_seq(resize_target_size, pad_method, pad_size):
    seq = iaa.Sequential(
        [
            iaa.Scale({"height": resize_target_size, "width": resize_target_size}),
            PadFixed(pad=(pad_size, pad_size), pad_method=pad_method),
            affine_seq,
        ],
        random_order=False,
    )
    return seq


def pad_to_fit_net(divisor, pad_mode, rest_of_augs=iaa.Noop()):
    return iaa.Sequential(InferencePad(divisor, pad_mode), rest_of_augs)


class PadFixed(iaa.Augmenter):
    PAD_FUNCTION = {"reflect": cv2.BORDER_REFLECT_101, "edge": cv2.BORDER_REPLICATE}

    def __init__(
        self,
        pad=None,
        pad_method=None,
        name=None,
        deterministic=False,
        random_state=None,
    ):
        super().__init__(name, deterministic, random_state)
        self.pad = pad
        self.pad_method = pad_method

    def _augment_images(self, images, random_state, parents, hooks):
        result = []
        for i, image in enumerate(images):
            image_pad = self._pad(image)
            result.append(image_pad)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _pad(self, img):
        img_ = img.copy()

        if self._is_expanded_grey_format(img):
            img_ = np.squeeze(img_, axis=-1)

        h_pad, w_pad = self.pad
        img_ = cv2.copyMakeBorder(
            img_.copy(),
            h_pad,
            h_pad,
            w_pad,
            w_pad,
            PadFixed.PAD_FUNCTION[self.pad_method],
        )

        if self._is_expanded_grey_format(img):
            img_ = np.expand_dims(img_, axis=-1)

        return img_

    def get_parameters(self):
        return []

    def _is_expanded_grey_format(self, img):
        if len(img.shape) == 3 and img.shape[2] == 1:
            return True
        else:
            return False


def test_time_augmentation_transform(image, tta_parameters):
    if tta_parameters["ud_flip"]:
        image = np.flipud(image)
    if tta_parameters["lr_flip"]:
        image = np.fliplr(image)
    if tta_parameters["color_shift"]:
        random_color_shift = reseed(intensity_seq, deterministic=False)
        image = random_color_shift.augment_image(image)
    image = rotate(image, tta_parameters["rotation"])
    return image


def test_time_augmentation_inverse_transform(image, tta_parameters):
    image = per_channel_rotation(image.copy(), -1 * tta_parameters["rotation"])

    if tta_parameters["lr_flip"]:
        image = per_channel_fliplr(image.copy())
    if tta_parameters["ud_flip"]:
        image = per_channel_flipud(image.copy())
    return image


def per_channel_flipud(x):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = np.flipud(channel)
    return x_


def per_channel_fliplr(x):
    x_ = x.copy()
    for i, channel in enumerate(x):
        x_[i, :, :] = np.fliplr(channel)
    return x_


def per_channel_rotation(x, angle):
    return rotate(x, angle, axes=(1, 2))


def rotate(image, angle, axes=(0, 1)):
    if angle % 90 != 0:
        raise Exception("Angle must be a multiple of 90.")
    k = angle // 90
    return np.rot90(image, k, axes=axes)


class RandomCropFixedSize(iaa.Augmenter):
    def __init__(self, px=None, name=None, deterministic=False, random_state=None):
        super(RandomCropFixedSize, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state
        )
        self.px = px
        if isinstance(self.px, tuple):
            self.px_h, self.px_w = self.px
        elif isinstance(self.px, int):
            self.px_h = self.px
            self.px_w = self.px
        else:
            raise NotImplementedError

    def _augment_images(self, images, random_state, parents, hooks):

        result = []
        seeds = random_state.randint(0, 10 ** 6, (len(images),))
        for i, image in enumerate(images):
            seed = seeds[i]
            image_cr = self._random_crop(seed, image)
            result.append(image_cr)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = []
        return result

    def _random_crop(self, seed, image):
        height, width = image.shape[:2]

        np.random.seed(seed)
        if height > self.px_h:
            crop_top = np.random.randint(height - self.px_h)
        elif height == self.px_h:
            crop_top = 0
        else:
            raise ValueError("To big crop height")
        crop_bottom = crop_top + self.px_h

        np.random.seed(seed + 1)
        if width > self.px_w:
            crop_left = np.random.randint(width - self.px_w)
        elif width == self.px_w:
            crop_left = 0
        else:
            raise ValueError("To big crop width")
        crop_right = crop_left + self.px_w

        if len(image.shape) == 2:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right]
        else:
            image_cropped = image[crop_top:crop_bottom, crop_left:crop_right, :]
        return image_cropped

    def get_parameters(self):
        return []


class InferencePad(iaa.Augmenter):
    def __init__(
        self,
        divisor=2,
        pad_mode="symmetric",
        name=None,
        deterministic=False,
        random_state=None,
    ):
        super(InferencePad, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state
        )
        self.divisor = divisor
        self.pad_mode = pad_mode

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def _augment_images(self, images, random_state, parents, hooks):

        result = []
        for i, image in enumerate(images):
            image_padded = self._pad_image(image)
            result.append(image_padded)
        return result

    def _pad_image(self, image):
        height = image.shape[0]
        width = image.shape[1]

        pad_sequence = self._get_pad_sequence(height, width)
        augmenter = iaa.Pad(px=pad_sequence, keep_size=False, pad_mode=self.pad_mode)
        return augmenter.augment_image(image)

    def _get_pad_sequence(self, height, width):
        pad_vertical = self._get_pad(height)
        pad_horizontal = self._get_pad(width)
        return get_crop_pad_sequence(pad_vertical, pad_horizontal)

    def _get_pad(self, dim):
        if dim % self.divisor == 0:
            return 0
        else:
            return self.divisor - dim % self.divisor

    def get_parameters(self):
        return [self.divisor, self.pad_mode]
