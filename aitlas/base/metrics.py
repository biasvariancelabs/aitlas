
class BaseMetric:
    """Base class for implementing metrics """

    def __init__(self, **kwargs):
        pass

    def calculate(self, y_true, y_pred):
        raise NotImplementedError("Please implement you metric calculation logic here.")
