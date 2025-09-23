import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .ScaleMAE.scale_mae import MaskedAutoencoderViT, vit_base_patch16, vit_large_patch16, vit_huge_patch14

class ScaleMAE(FoundationModel):
    """AiTLAS wrapper class for Scale-MAE model
    
    .. note:: Based on https://github.com/bair-climate-initiative/scale-mae
    """

    name = "Scale-MAE"

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the Scale-MAE backbone model from Github or from a local path (if available).
        """

        # Define checkpoints for all backbones available on Huggingface
        backbone_checkpoints = {
            'vit_base_patch16': None,
            'vit_large_patch16': [
                'scalemae-vitlarge-800.pth',
                'vit_large_patch16_224_fmow_rgb_scalemae-98ed9821.pth'
            ],
            'vit_huge_patch14': None
        }

        # Get the fixed output size from the config (default is 224)
        fixed_output_size = getattr(self.config, 'fixed_output_size', 224)

        if self.config.pretrained: # Load pretrained weights
            if self.config.local_model_path:
                # Check if the provided local path exists
                if not os.path.exists(self.config.local_model_path):
                    print(f"Provided local model path does not exist: {self.config.local_model_path}")
                    print("Weights will be downloaded from Huggingface instead.")
                    # Check if backbone is supported
                    if self.config.backbone_name not in backbone_checkpoints:
                        raise ValueError(f"Unsupported or missing backbone: '{self.config.backbone_name}'. Supported names are: {list(backbone_checkpoints.keys())}")
                    else:
                        # Check if backbone has weights available
                        if backbone_checkpoints[self.config.backbone_name] is None:
                            raise ValueError(f"No pretrained weights are available for backbone '{self.config.backbone_name}'.")
                        else: # Download the weights and load the model
                            # For now, just load the first checkpoint available for the backbone
                            checkpoint_name = backbone_checkpoints[self.config.backbone_name][0]
                            self.config.local_model_path = hf_hub_download(repo_id="isaaccorley/vit_large_patch16_224_fmow_rgb_scalemae", filename=checkpoint_name, local_dir=os.path.dirname(self.config.local_model_path))
                            checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                            backbone = globals()[self.config.backbone_name](fixed_output_size=fixed_output_size)
                            msg = backbone.load_state_dict(checkpoint, strict=False)
                            print("Successfully loaded checkpoint:", checkpoint_name)
                else:
                    print(f"Loading weights from the provided local path: {self.config.local_model_path}")
                    checkpoint = torch.load(self.config.local_model_path, weights_only=False)
                    checkpoint_name = os.path.basename(self.config.local_model_path)
                    # Find the backbone name corresponding to the checkpoint
                    self.backbone_name = None
                    for name, checkpoints in backbone_checkpoints.items():
                        if checkpoints and checkpoint_name in checkpoints:
                            self.backbone_name = name
                            break
                    backbone = globals()[self.backbone_name](fixed_output_size=fixed_output_size)
                    msg = backbone.load_state_dict(checkpoint, strict=False)
                    print("Successfully loaded checkpoint:", checkpoint_name)
        else: # Load model without pretrained weights
            raise NotImplementedError("Loading model without pretrained weights is not supported.")

        # Replace the head with identity if it exists
        if hasattr(backbone, 'head'):
            backbone.head = nn.Identity()

        # This method MUST return the loaded backbone object for the parent class.
        return backbone
    
    def forward_features(self, x, input_res):
        """ Forward pass through the Scale-MAE model to get the feature embeddings.
        
        Args:
            x (torch.Tensor): A batch of RGB image tensors, e.g., shape [N, 3, 224, 224].
            input_res (torch.Tensor): A 1D tensor of the Ground Sampling Distance for
                                    each image in the batch, e.g., shape [N].
        """
        
        # Check if backbone is loaded
        if self.backbone is None:
            raise RuntimeError(
                "The backbone model has not been loaded. "
                "Please call the .load_backbone() method before the forward pass."
            )

        # Pass both the image tensor and its resolution to the model
        embedding = self.backbone.forward(x, knn_feats=True, input_res=input_res)

        return embedding


