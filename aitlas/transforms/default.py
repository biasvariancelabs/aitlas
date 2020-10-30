import torchvision.transforms as transforms

from ..base import BaseTransforms


class DefaultTransforms(BaseTransforms):
    def __call__(self, input, target=None):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        return transform(input)
