import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as skmetrics

from ..base import BaseVisualization
from ..utils import pil_loader


class ConfusionMatrix(BaseVisualization):
    def __init__(self, y_true, y_pred, y_prob, labels, file, **kwargs):
        super().__init__(y_true, y_pred, y_prob, labels, file, **kwargs)

    def plot(self):
        # get the confusion matrix
        cm = skmetrics.confusion_matrix(self.y_true, self.y_pred, normalize="true")
        df_cm = pd.DataFrame(cm, index=self.labels, columns=self.labels)

        # plot confusion matrix
        figure = plt.figure()
        ax = plt.axes()
        ax.set_title("Confusion matrix of predictions")
        sns.set(font_scale=1)
        sns.heatmap(df_cm, cmap="YlGnBu", ax=ax, annot=True)
        plt.yticks(rotation=0)

        figure.savefig(self.output_file, format="png")

        return figure


class PrecisionRecallCurve(BaseVisualization):
    def plot(self):
        """Generate plot"""
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


class ImageLabelsVisualization(BaseVisualization):
    def __init__(self, y_true, y_pred, y_prob, labels, file, **kwargs):
        super().__init__(y_true, y_pred, y_prob, labels, file, **kwargs)
        self.image = kwargs.get("image")

    def plot(self):
        image = pil_loader(self.image)
        fig = self.plot_prediction(image, self.y_prob, self.labels)
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


def confusion_matrix(y_true, y_pred, y_prob, labels, output_file):
    """Wrapper for the call for easier usage"""
    viz = ConfusionMatrix(y_true, y_pred, y_prob, labels, output_file)
    return viz.plot()


def display_image_labels(image, y_true, y_pred, y_prob, labels, output_file):
    viz = ImageLabelsVisualization(
        y_true, y_pred, y_prob, labels, output_file, image=image
    )
    viz.plot()


def precision_recall_curve(y_true, y_pred, y_prob, labels, output_file):
    viz = PrecisionRecallCurve(y_true, y_pred, y_prob, labels, output_file)
    viz.plot()
