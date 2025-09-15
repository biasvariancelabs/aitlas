import torch.nn as nn
from .models import BaseModel
from .schemas import BaseFoundationModelSchema

class FoundationModel(BaseModel):
    """Base class for foundation models.
    """

    schema = BaseFoundationModelSchema

    def __init__(self, config):
        super().__init__(config)
        self.backbone = self.load_backbone()

    def load_backbone(self) -> nn.Module:
        """Load the backbone model

        Each foundation model should implement this method to load its backbone.

        :return: The backbone model
        :rtype: torch.nn.Module
        """
        raise NotImplementedError("Each foundation model should implement 'load_backbone' to load its backbone.")

    def forward_features(self, x, **kwargs):
        """Forward pass through the model to get the feature embeddings.

        Each foundation model should implement this method to define its forward_features pass.

        :param x: Input image tensor
        :type x: torch.Tensor
        **kwargs: Additional arguments that might be required for the forward_features pass, such a as wave_list for certain models
        :return: Output feature embeddings or predictions
        :rtype: torch.Tensor
        """
        raise NotImplementedError("Each foundation model should implement 'forward_features' to define its forward pass to get the feature embaddings.")