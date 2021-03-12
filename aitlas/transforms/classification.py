from torchvision import transforms

from ..base import BaseTransforms


class ResizeCenterCropFlipHVToTensor(BaseTransforms):
    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

        return data_transforms(sample)


class ResizeCenterCropToTensor(BaseTransforms):
    def __call__(self, sample):
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        return data_transforms(sample)
