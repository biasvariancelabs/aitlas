import cv2
import math
import numpy as np

from aitlas.base import BaseTransforms


class CompositeTransforms(BaseTransforms):
    """
    Compose transforms from list to apply them sequentially.
    """

    def __init__(self, transforms):
        """
        Parameters
        ----------
            transforms : list
                The list of transformations to be applied.

        Raises
        ------
            ValueError
                If the transforms list is None or empty
        """
        super().__init__()
        if transforms is None or len(transforms) < 1:
            raise ValueError("The list of transformations must not be empty")
        self.transforms = transforms

    def __call__(self, sample):
        """
        Sequentially applies each transformation to the passed parameter.

        Parameters
        ----------
            sample : dict, with keys: "image" and "mask".
                The data to be transformed.

        Returns
        -------
            tuple
                The transformed image and mask.
        """
        for transform in self.transforms:
            image, mask = transform(sample)
            sample = {"image": image, "mask": mask}
        return sample["image"], sample["mask"]


class RandomFlip(BaseTransforms):
    """
    Random flip transformation.
    """

    def __init__(self, flip_code=None):
        """
        Parameters
        ----------
            flip_code: int, optional, default is random.
                Control parameter for the flip.
                If it is 0, then it flips around x-axis,
                else if it is a positive value it flips around y-axis,
                and negative value flips around both axes.
        """
        super().__init__()
        if flip_code is None:
            self.flip_code = np.random.randint(low=-1, high=1)
        else:
            self.flip_code = flip_code

    def __call__(self, sample):
        """
        Applies the random flip transformation.

        Parameters
        ----------
            sample : dict, with keys "image" and "mask".
                The data to be transformed.

        Returns
        -------
            tuple
                The transformed image and mask.
        """
        # Unpack data
        image = sample.get("image", None)
        mask = sample.get("mask", None)
        # Apply transformation
        if image is not None:
            image = cv2.flip(src=image, flipCode=self.flip_code)
        if mask is not None:
            mask = cv2.flip(src=mask, flipCode=self.flip_code)
        return image, mask


class RandomShiftScaleRotate(BaseTransforms):
    """
    Random shift, scale and rotate transformation.
    """

    def __init__(self, shift_limit=0., scale_limit=0.1, rotate_limit=30):
        """
        Parameters
        ----------
            shift_limit : float, optional, default 0
                The shifting limit.
            scale_limit : float, optional, default 0.1
                The scaling limit.
            rotate_limit : int, optional, default 30
                The rotation limit.
        """
        super().__init__()
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

    def __call__(self, sample):
        """
        Applies a random rotate, shift and scale transformation.

        Parameters
        ----------
            sample : dict, with keys "image" and "mask".
                The data to be transformed.

        Returns
        -------
            tuple
                The transformed image and mask.
        """

        def apply(image_):
            """
            Actually applies the transformation.

            Parameters
            ----------
                image_ : an image
                    To which the transformation is applied.

            Returns
            -------
                an image
                    The transformed image.

            Raises
            ------
                cv2.error: (-215:Assertion failed) !ssize.empty() in function 'remapBilinear'
                    If the data passed is empty
            """
            height, width = image_.shape[:2]
            cc = math.cos(angle / 180 * math.pi) * scale
            ss = math.sin(angle / 180 * math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])
            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx * width, height / 2 + dy * height])
            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            return cv2.warpPerspective(src=image_, M=mat, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)

        # Randomly initialize parameters
        angle = np.random.uniform(low=-self.rotate_limit, high=self.rotate_limit)
        scale = np.random.uniform(low=1 - self.scale_limit, high=1 + self.scale_limit)
        dx = np.random.uniform(low=-self.shift_limit, high=self.shift_limit)
        dy = np.random.uniform(low=-self.shift_limit, high=self.shift_limit)
        # Unpack data
        image = sample.get("image", None)
        mask = sample.get("mask", None)
        # Apply transformations
        try:
            image = apply(image)
            mask = apply(mask)
        except:
            pass
        return image, mask


class CenterCrop(BaseTransforms):

    def __init__(self, size=1024):
        super().__init__()
        self.size = size

    def __call__(self, sample):
        image = sample.get("image", None)
        mask = sample.get("mask", None)
        image = image[138:1162, 138:1162, :]
        mask = mask[138:1162, 138:1162, :]
        return image, mask


class Transpose(BaseTransforms):
    def __call__(self, sample):
        image = sample.get("image", None)
        mask = sample.get("mask", None)
        if mask is None:
            return image.transpose(2, 0, 1).astype("float32") / 255
        else:
            return image.transpose(2, 0, 1).astype("float32") / 255, mask.transpose(2, 0, 1).astype("float32") / 255


class SpaceNet5Transforms(BaseTransforms):
    """
    Applies random transformations during training.

    Notes
    -----
        Inspired by https://github.com/CosmiQ/cresi/blob/master/cresi/net/augmentations/transforms.py
    """

    def __call__(self, sample):
        """
        Applies the specified transformations.

        Parameters
        ----------
            sample : dict, with keys "image" and "mask".
                The data to be transformed.

        Returns
        -------
            tuple
                The transformed image and mask.
        """
        return CompositeTransforms([
            RandomFlip(),
            RandomShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30),
            CenterCrop(1024),
            Transpose()  # because of the transpose operation
        ])(sample)
