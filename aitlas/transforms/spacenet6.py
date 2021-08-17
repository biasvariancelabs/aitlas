import random

import cv2
import numpy as np

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
        # Data
        image = sample.get("image", None)
        mask = sample.get("mask", None)
        # Crop size
        crop_size = 512
        # Transform probabilities
        rot_prob = 0.7
        flip_lr_prob = 0.5
        ###################################
        # Start transforms
        pad = max(0, crop_size - image.shape[0])
        image = cv2.copyMakeBorder(src=image, top=0, bottom=pad, left=0, right=0, borderType=cv2.BORDER_CONSTANT,
                                   value=0.0)
        mask = cv2.copyMakeBorder(src=mask, top=0, bottom=pad, left=0, right=0, borderType=cv2.BORDER_CONSTANT,
                                  value=0.0)
        # Rotate image
        if random.random() < rot_prob:
            rotation_matrix = cv2.getRotationMatrix2D(center=(image.shape[0] // 2, image.shape[1] // 2),
                                                      angle=random.randint(0, 10) - 5, scale=1.0)
            image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=image.shape[:2], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(src=mask, M=rotation_matrix, dsize=mask.shape[:2], flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)
        # Scale image (because scale_prob = 1)
        rotation_matrix = cv2.getRotationMatrix2D(center=(image.shape[0] // 2, image.shape[1] // 2), angle=0,
                                                  scale=random.uniform(0.5, 2.0))
        image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=image.shape[:2], flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(src=mask, M=rotation_matrix, dsize=mask.shape[:2], flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
        # Random crop
        x0 = random.randint(0, image.shape[1] - crop_size)
        y0 = random.randint(0, image.shape[0] - crop_size)
        image = image[y0: y0 + crop_size, x0: x0 + crop_size]
        mask = mask[y0: y0 + crop_size, x0: x0 + crop_size]
        # Apply these functions (because color_aug_prob = 1)
        image = saturation(image, 0.8 + random.random() * 0.4)
        image = brightness(image, 0.8 + random.random() * 0.4)
        image = contrast(image, 0.8 + random.random() * 0.4)
        # Left-right flip
        if random.random() < flip_lr_prob:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        # Return results
        return image, mask
