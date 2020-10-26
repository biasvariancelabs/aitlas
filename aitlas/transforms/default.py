import torchvision.transforms as transforms

from ..base import BaseTransforms


class DefaultTransforms(BaseTransforms):
    def __call__(self, input, target=None):
        return self.transform(input)

    def load_transforms(self):
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )