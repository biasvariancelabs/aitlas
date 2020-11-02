import matplotlib.pyplot as plt

from ..base import BaseVisualization
from ..utils import pil_loader


class ImageMaskPredictionVisualization(BaseVisualization):
    def __init__(self, y_true, y_pred, y_prob, labels, file, **kwargs):
        super().__init__(y_true, y_pred, y_prob, labels, file, **kwargs)
        self.image = kwargs.get("image")

    def plot(self):
        image = pil_loader(self.image)
        fig = self.plot_segmenation(image, self.y_prob, self.labels)
        fig.savefig(self.output_file, format="png")

    def plot_segmenation(self, img, probs, labels):
        """Display image and preditions from model"""

        # Show the image
        fig = plt.figure(figsize=(10, 10))

        # plot image
        plt.subplot(1, len(labels) + 1, 1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        # plot masks
        for i in range(len(probs)):
            plt.subplot(1, len(labels) + 1, i + 2)
            plt.imshow(probs[i])
            plt.title(labels[i])
            plt.axis("off")

        plt.tight_layout()

        return fig


def display_image_segmentation(image, y_true, y_pred, y_prob, labels, file):
    viz = ImageMaskPredictionVisualization(
        y_true, y_pred, y_prob, labels, file, image=image
    )
    viz.plot()
