class BaseVisualization:
    """Base class for visualizations"""

    def __init__(self, y_true, y_pred, y_prob, labels, file, **kwargs):
        """Get default params"""
        self.labels = labels
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.output_file = file

    def plot(self):
        raise NotImplementedError("Please implement the `plot` method.")
