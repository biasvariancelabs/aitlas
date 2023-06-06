"""Classes and methods for image transformations specific for the Spacenet6 dataset."""

import random

import cv2
import numpy as np

from aitlas.base import BaseTransforms


def _blend(img1, img2, alpha):
    """
    Blend two images together using the specified alpha.

    :param img1: First input image
    :type img1: numpy.ndarray
    :param img2: Second input image
    :type img2: numpy.ndarray
    :param alpha: blending factor
    :type alpha: float
    :return: Blended image
    :rtype: numpy.ndarray
    """
    return img1 * alpha + (1 - alpha) * img2


_alpha = np.asarray([0.25, 0.25, 0.25, 0.25]).reshape((1, 1, 4))


def _grayscale(img):
    """
    Transform an image to grayscale.

    :param img: Input image
    :type img: numpy.ndarray
    :return: grayscale image
    :rtype: numpy.ndarray
    """
    return np.sum(_alpha * img, axis=2, keepdims=True)


def saturation(img, alpha):
    """
    Adjust the saturation of an image.

    :param img: input image
    :type img: numpy.ndarray
    :param alpha: saturation factor
    :type alpha: float
    :return: image with adjusted saturation
    :rtype: numpy.ndarray
    """
    gs = _grayscale(img)
    return _blend(img, gs, alpha)


def brightness(img, alpha):
    """
    Adjust the brightness of an image.

    :param img: input image
    :type img: numpy.ndarray
    :param alpha: brightness factor
    :type alpha: float
    :return: image with adjusted brightness
    :rtype: numpy.ndarray
    """
    gs = np.zeros_like(img)
    return _blend(img, gs, alpha)


def contrast(img, alpha):
    """
    Adjust the contrast of an image.

    :param img: input image
    :type img: numpy.ndarray
    :param alpha: contrast factor
    :type alpha: float
    :return: image with adjusted contrast
    :rtype: numpy.ndarray
    """
    gs = _grayscale(img)
    gs = np.repeat(gs.mean(), 4)
    return _blend(img, gs, alpha)


class SpaceNet6Transforms(BaseTransforms):
    """
    SpaceNet6 specific image transformations.
    """

    def __call__(self, sample):
        """
        Apply transformations to the sample.
        The transformations include:
        - random crop to 512x512
        - random rotation with probability 0.7
        - random scale between 0.5 and 2.0
        - random left-right flip with probability 0.5
        - random color augmentation (saturation, brightness, contrast)


        :param sample: a dictionary containing image and mask data
        :type sample: dict
        :return: image and mask after applying transformations
        :rtype: tuple
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
        image = cv2.copyMakeBorder(
            src=image,
            top=0,
            bottom=pad,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=0.0,
        )
        mask = cv2.copyMakeBorder(
            src=mask,
            top=0,
            bottom=pad,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=0.0,
        )
        # Rotate image
        if random.random() < rot_prob:
            rotation_matrix = cv2.getRotationMatrix2D(
                center=(image.shape[0] // 2, image.shape[1] // 2),
                angle=random.randint(0, 10) - 5,
                scale=1.0,
            )
            image = cv2.warpAffine(
                src=image,
                M=rotation_matrix,
                dsize=image.shape[:2],
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                src=mask,
                M=rotation_matrix,
                dsize=mask.shape[:2],
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        # Scale image (because scale_prob = 1)
        rotation_matrix = cv2.getRotationMatrix2D(
            center=(image.shape[0] // 2, image.shape[1] // 2),
            angle=0,
            scale=random.uniform(0.5, 2.0),
        )
        image = cv2.warpAffine(
            src=image,
            M=rotation_matrix,
            dsize=image.shape[:2],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            src=mask,
            M=rotation_matrix,
            dsize=mask.shape[:2],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Random crop
        x0 = random.randint(0, image.shape[1] - crop_size)
        y0 = random.randint(0, image.shape[0] - crop_size)
        image = image[y0 : y0 + crop_size, x0 : x0 + crop_size]
        mask = mask[y0 : y0 + crop_size, x0 : x0 + crop_size]
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
