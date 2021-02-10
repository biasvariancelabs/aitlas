import random

import cv2
import numpy as np
from imgaug import augmenters as iaa

from aitlas.base import BaseTransforms


def _blend(img1, img2, alpha):
    return img1 * alpha + (1 - alpha) * img2


_alpha = np.asarray([0.25, 0.25, 0.25, 0.25]).reshape((1, 1, 4))


def _grayscale(img):
    return np.sum(_alpha * img, axis=2, keepdims=True)


def saturation(img, alpha):
    gs = _grayscale(img)
    return _blend(img, gs, alpha)


def brightness(img, alpha):
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)


def contrast(img, alpha):
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 4)
    return _blend(img, gs, alpha)


class SpaceNet6Transforms(BaseTransforms):
    """
    SpaceNet6 transforms.
    """

    def __call__(self, sample):
        """
        Parameters
        ----------
            sample : dict
                containing the data to be transformed, and a few other variables
        """
        # Unpack sample
        ###################################
        # Data
        image = sample.get("image", None)
        mask = sample.get("mask", None)
        # Crop size
        crop_size = sample.get("crop_size", 0)
        # Transform probabilities
        rot_prob = sample.get("rot_prob", 0)
        scale_prob = sample.get("scale_prob", 0)
        color_aug_prob = sample.get("color_aug_prob", 0)
        gamma_aug_prob = sample.get("gamma_aug_prob", 0)
        gauss_aug_prob = sample.get("gauss_aug_prob", 0)
        elastic_aug_prob = sample.get("elastic_aug_prob", 0)
        flipud_prob = sample.get("flipud_prob", 0)
        fliplr_prob = sample.get("fliplr_prob", 0)
        rot90_prob = sample.get("rot90_prob", 0)
        channel_swap_prob = sample.get("channel_swap_prob", 0)
        ###################################
        # Start transforms
        pad = max(0, crop_size - image.shape[0])
        image = cv2.copyMakeBorder(src=image, top=0, bottom=pad, left=0, right=0, borderType=cv2.BORDER_CONSTANT,
                                   value=0.0)
        mask = cv2.copyMakeBorder(src=mask, top=0, bottom=pad, left=0, right=0, borderType=cv2.BORDER_CONSTANT,
                                  value=0.0)
        if random.random() < rot_prob:
            rotation_matrix = cv2.getRotationMatrix2D(center=(image.shape[0] // 2, image.shape[1] // 2),
                                                      angle=random.randint(0, 10) - 5, scale=1.0)
            image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=image.shape[:2], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(src=mask, M=rotation_matrix, dsize=mask.shape[:2], flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)
        if random.random() < scale_prob:
            rotation_matrix = cv2.getRotationMatrix2D(center=(image.shape[0] // 2, image.shape[1] // 2), angle=0,
                                                      scale=random.uniform(0.5, 2.0))
            image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=image.shape[:2], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(src=mask, M=rotation_matrix, dsize=mask.shape[:2], flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        x0 = random.randint(0, image.shape[1] - crop_size)
        y0 = random.randint(0, image.shape[0] - crop_size)
        image = image[y0: y0 + crop_size, x0: x0 + crop_size]
        mask = mask[y0: y0 + crop_size, x0: x0 + crop_size]
        if random.random() < color_aug_prob:
            image = saturation(image, 0.8 + random.random() * 0.4)
        if random.random() < color_aug_prob:
            image = brightness(image, 0.8 + random.random() * 0.4)
        if random.random() < color_aug_prob:
            image = contrast(image, 0.8 + random.random() * 0.4)
        if random.random() < gamma_aug_prob:
            gamma = 0.8 + 0.4 * random.random()
            image = np.clip(image, a_min=0.0, a_max=None)
            image = np.power(image, gamma)
        if random.random() < gauss_aug_prob:
            gauss = np.random.normal(10.0, 10.0 ** 0.5, image.shape)
            image += gauss - np.min(gauss)
        if random.random() < elastic_aug_prob:
            el_det = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2).to_deterministic()
            image = el_det.augment_image(image)
        if random.random() < flipud_prob:
            image = np.flipud(image)
            mask = np.flipud(mask)
        if random.random() < fliplr_prob:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if random.random() < rot90_prob:
            k = random.randint(0, 3)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        if random.random() < channel_swap_prob:
            c1 = random.randint(0, 3)
            c2 = random.randint(0, 3)
            image[:, :, [c1, c2]] = image[:, :, [c2, c1]]
        # Return results
        return image, mask
