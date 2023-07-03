"""Base class for implementing visualizations."""


class BaseVisualization:
    """Base class for visualizations"""

    def __init__(self, cm, labels, file, **kwargs):
        self.cm = cm
        self.labels = labels
        self.output_file = file

    def plot(self):
        raise NotImplementedError("Please implement the `plot` method.")


class BaseDetailedVisualization(BaseVisualization):
    """Base class for visualizations"""

    def __init__(self, y_true, y_pred, y_prob, labels, file, **kwargs):
        super().__init__(None, labels, file, **kwargs)

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def plot(self):
        raise NotImplementedError("Please implement the `plot` method.")
