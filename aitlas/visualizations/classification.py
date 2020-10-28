import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from ..base import BaseVisualization
from ..utils import pil_loader


class ConfusionMatrix(BaseVisualization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels = kwargs.get("labels")
        self.y_true = kwargs.get("y_true")
        self.y_pred = kwargs.get("y_pred")
        self.output_file = kwargs.get("file", "cm.png")

    def plot(self):
        # get the confusion matrix
        cm = sk_confusion_matrix(self.y_true, self.y_pred)
        df_cm = pd.DataFrame(cm, index=self.labels, columns=self.labels)

        # plot confusion matrix
        figure = plt.figure()
        ax = plt.axes()
        ax.set_title("Confusion matrix of predictions")
        sns.set(font_scale=0.9)
        sns.heatmap(df_cm, cmap="PiYG", ax=ax)
        plt.yticks(rotation=0)

        confusion_matrix_plot_name = self.output_file
        figure.savefig(confusion_matrix_plot_name, format="png")
        # plt.clf()

        return figure


class ImageLabelsVisualization(BaseVisualization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image = kwargs.get("image")
        self.probs = kwargs.get("probs")
        self.labels = kwargs.get("labels")
        self.output_file = kwargs.get("file", "classification.png")

    def plot(self):
        image = pil_loader(self.image)
        fig = self.plot_prediction(image, self.probs, self.labels)
        fig.savefig(self.output_file, format="png")

    def plot_prediction(self, img, probs, classes):
        """Display image and preditions from model"""

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


def confusion_matrix(labels, y_true, y_pred, output_file):
    """Wrapper for the call for easier usage"""
    viz = ConfusionMatrix(
        labels=labels, y_true=y_true, y_pred=y_pred, file=output_file,
    )
    return viz.plot()


def display_image_labels(image, probs, labels, output_file):
    viz = ImageLabelsVisualization(
        image=image, probs=probs, labels=labels, file=output_file
    )
    viz.plot()
