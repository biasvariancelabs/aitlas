"""Classes and methods for visualizations for classification tasks."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as skmetrics
import math

from ..base import BaseDetailedVisualization, BaseVisualization
from ..utils import pil_loader


def plot_confusion_matrix(
    confusion_matrix, axes, class_label, class_names, fontsize=14
):
    """
    Plots a confusion matrix.

    :param confusion_matrix: The confusion matrix to plot.
    :type confusion_matrix: array-like of shape (n_classes, n_classes)
    :param axes: The matplotlib axes object to plot on.
    :type axes: matplotlib.axes.Axes
    :param class_label: The label of the class for the confusion matrix.
    :type class_label: str
    :param class_names: The names of the classes.
    :type class_names: list of str
    :param fontsize: The fontsize for the plot, defaults to 14.
    :type fontsize: int, optional
    """
    df_cm = pd.DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    )
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize
    )
    axes.set_ylabel("True label")
    axes.set_xlabel("Predicted label")
    if len(class_label) >= 20:
        class_label = class_label[0:20] + "..."
    axes.set_title(class_label)


def plot_multilabel_confusion_matrix(cm_array, labels, dataset_name, output_file):
    """
    Plots multiple confusion matrices in a .pdf format for multilabel tasks.

    :param cm_array: The array of confusion matrices.
    :type cm_array: list of array-like of shape (n_classes, n_classes)
    :param labels: The labels for the classes.
    :type labels: list of str
    :param dataset_name: The name of the dataset.
    :type dataset_name: str
    :param output_file: The file path to save the plot.
    :type output_file: str
    """

    rows = math.ceil(math.sqrt(len(labels)))
    columns = rows
    if (rows * columns - len(labels)) >= rows:
        columns -= 1

    figure_height = rows * 1.75
    figure_width = columns * 2.0
    figure, ax = plt.subplots(rows, columns, figsize=(figure_width, figure_height))

    # figure.suptitle("Confusion matrix of predictions for {}".format(dataset_name), fontsize=20)
    for axes, cfs_matrix, label in zip(ax.flatten(), cm_array, labels):
        plot_confusion_matrix(cfs_matrix, axes, label, ["N", "P"])
    num_ax_remove = rows * columns - len(labels)
    for i in range(num_ax_remove):
        ax[-1, columns - 1 - i].axis("off")
    figure.tight_layout()
    # figure.savefig(output_file, format="png")
    figure.savefig(output_file, format="pdf", bbox_inches="tight")

    return figure


def plot_multiclass_confusion_matrix(cm_array, labels, dataset_name, output_file):
    # get the confusion matrix
    """
    Plots multiple confusion matrices .pdf format useful for multiclass tasks.

    :param cm_array: The array of confusion matrices.
    :type cm_array: list of array-like of shape (n_classes, n_classes)
    :param labels: The labels for the classes.
    :type labels: list of str
    :param dataset_name: The name of the dataset.
    :type dataset_name: str
    :param output_file: The file path to save the plot.
    :type output_file: str
    """
    df_cm = pd.DataFrame(cm_array, index=labels, columns=labels)

    # plot confusion matrix, different dimensions for different number of labels
    if len(labels) < 10:
        figure = plt.figure(figsize=(8, 5))
    elif len(labels) < 20:
        figure = plt.figure(figsize=(10, 7))
    elif len(labels) < 32:
        figure = plt.figure(figsize=(12, 9))
    else:
        figure = plt.figure(figsize=(20, 15))
    ax = plt.axes()
    # ax.set_title("Confusion matrix of predictions for {}".format(dataset_name))
    sns.set(font_scale=1)
    sns.heatmap(df_cm, cmap="YlGnBu", ax=ax, annot=True, fmt="g")
    plt.yticks(rotation=0)
    figure.tight_layout()

    # figure.savefig(output_file, format="png")
    figure.savefig(output_file, format="pdf", bbox_inches="tight")

    return figure


class PrecisionRecallCurve(BaseDetailedVisualization):
    def plot(self):
        """
        Generates and plots the precision recall curve.

        :return: matplotlib.figure.Figure object with the plot
        :rtype: matplotlib.figure.Figure
        """
        figure = plt.figure()

        # plot pr curve for each class
        for i in range(len(self.labels)):
            filtered_true = [1 if x == i else 0 for x in self.y_true]
            filtered_pred = [1 if x == i else 0 for x in self.y_pred]
            precision, recall, _ = skmetrics.precision_recall_curve(
                filtered_true, filtered_pred
            )
            plt.plot(recall, precision, lw=2, label=self.labels[i])

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall curve")

        figure.savefig(self.output_file, format="png")

        return figure


class ImageLabelsVisualization(BaseDetailedVisualization):
    """
    Class for visualising predictions for an image.
    """

    def __init__(self, y_true, y_pred, y_prob, labels, file, **kwargs):
        """
        Initialize the ImageLabelsVisualization class.

        :param y_true: Ground truth (correct) labels.
        :type y_true: array-like of shape (n_samples,)
        :param y_pred: Predicted labels, as returned by a classifier.
        :type y_pred: array-like of shape (n_samples,)
        :param y_prob: The predicted probabilities.
        :type y_prob: list of float
        :param labels: The labels for the classes.
        :type labels: list of str
        :param file: The file path to save the plot.
        :type file: str
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(y_true, y_pred, y_prob, labels, file, **kwargs)
        self.image = kwargs.get("image")

    def plot(self):
        """
        Plots the image with the predictions.

        :return: matplotlib.figure.Figure object with the plot
        :rtype: matplotlib.figure.Figure
        """
        image = pil_loader(self.image)
        fig = self.plot_prediction(image, self.y_prob, self.labels)
        fig.savefig(self.output_file, format="png")

    def plot_prediction(self, img, probs, classes):
        """Display image and predictions from model

        :param img: Image to plot.
        :type img: array-like or PIL image
        :param prob: The predicted probabilities.
        :type prob: list of float
        :param classes: The labels for the classes.
        :type classes: list of str
        :return: matplotlib.figure.Figure object with the plot
        :rtype: matplotlib.figure.Figure
        """

        # Convert results to dataframe for plotting
        result = pd.DataFrame({"p": probs}, index=classes)

        # Show the image
        fig = plt.figure(figsize=(16, 5))
        ax = plt.subplot(1, 2, 1)
        ax.imshow(img)

        # Set title to be the actual class
        ax.set_title("", size=20)

        ax = plt.subplot(1, 2, 2)
        # Plot a bar plot of predictions
        result.sort_values("p")["p"].plot.barh(color="blue", edgecolor="k", ax=ax)
        plt.xlabel("Predicted Probability")
        plt.tight_layout()

        return fig


def display_image_labels(image, y_true, y_pred, y_prob, labels, output_file):
    viz = ImageLabelsVisualization(
        y_true, y_pred, y_prob, labels, output_file, image=image
    )
    viz.plot()


def precision_recall_curve(y_true, y_pred, y_prob, labels, output_file):
    viz = PrecisionRecallCurve(y_true, y_pred, y_prob, labels, output_file)
    viz.plot()
