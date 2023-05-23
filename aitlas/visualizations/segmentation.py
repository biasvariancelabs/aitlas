"""Classes and methods for visualizations for segmentation tasks."""
import matplotlib.pyplot as plt
import numpy as np
from ..base import BaseDetailedVisualization
from ..utils import pil_loader
from PIL import Image, ImageOps


class ImageMaskPredictionVisualization(BaseDetailedVisualization):
    """
    Class for visualizing the image mask predictions.

    """

    def __init__(self, y_true, y_pred, y_prob, labels, file, **kwargs):
        """
        Initialisation

        :param y_true: The ground truth labels
        :type y_true: array-like of shape (n_samples,)
        :param y_pred: The predicted labels
        :type y_pred: array-like of shape (n_samples,)
        :param y_prob: The predicted probabilities
        :type y_prob: list of float
        :param labels: The class labels
        :type labels: list of str
        :param file: The output file path
        :type file: str

        """
        super().__init__(y_true, y_pred, y_prob, labels, file, **kwargs)
        self.image = kwargs.get("image")

    def plot(self):
        """
        Plots the image mask predictions and saves the plot to the output file.
        """
        image = pil_loader(self.image)
        fig = self.plot_segmenation(image, self.y_prob, self.labels)
        fig.savefig(self.output_file, format="png")

    def plot_segmenation(self, img, probs, labels):
        """
        Displays the image and the predicted segmentation masks for each label.

        :param img: The input image
        :type img: array-like or PIL image
        :param probs: The predicted probabilities
        :type probs: list of float
        :param labels: The class labels
        :type labels: list of str
        :return: The figure containing the plots
        :rtype: matplotlib.figure.Figure
        """

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
    """
    Displays the predicted segmentation masks for each label.

    :param image: The input image
    :type image: array-like or PIL image
    :param y_true: The ground truth labels
    :type y_true: array-like of shape (n_samples,)
    :param y_pred: The predicted labels
    :type y_pred: array-like of shape (n_samples,)
    :param y_prob: The predicted probabilities
    :type y_prob: list of float
    :param labels: The class labels
    :type labels: list of str
    :param file: The output file path
    :type file: str
    """
    viz = ImageMaskPredictionVisualization(
        y_true, y_pred, y_prob, labels, file, image=image
    )
    viz.plot()


def save_predicted_masks(y_pred, labels, base_filepath_name):
    """
    Saves the predicted masks to the specified file path.

    :param y_pred: The predicted labels
    :type y_pred: array-like of shape (n_samples,)
    :param labels: The class labels
    :type labels: list of str
    :param base_filepath_name: The base file path name
    :type base_filepath_name: str
    """
    for i in range(len(labels)):
        mask = Image.fromarray(y_pred[i].astype(np.uint8) * 255)
        # mask = ImageOps.invert(mask)
        mask.save("{}_{}.png".format(base_filepath_name, labels[i]))
