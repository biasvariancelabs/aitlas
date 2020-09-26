import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..base import BaseVisualization


class ConfusionMatrix(BaseVisualization):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = kwargs.get("dataset")
        self.y_true = kwargs.get("y_true")
        self.y_pred = kwargs.get("y_pred")
        self.output_file = kwargs.get("file")

    def plot(self):
        # get the confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        df_cm = pd.DataFrame(
            cm, index=self.dataset.labels(), columns=self.dataset.labels()
        )

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
