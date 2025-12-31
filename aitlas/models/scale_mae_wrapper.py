import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from ..base.foundation import FoundationModel
from .ScaleMAE.scale_mae import MaskedAutoencoderViT, scalemae_vit_large_patch16
from aitlas.models.registries import BACKBONE_REGISTRY


class ScaleMAE(FoundationModel):
    """AiTLAS wrapper class for Scale-MAE model
    
    .. note:: Based on https://github.com/bair-climate-initiative/scale-mae
    """

    name = "Scale-MAE"

    # Define checkpoints for all backbones available on Huggingface
    BACKBONE_CHECKPOINTS = {
        'scalemae_vit_large_patch16': [
            {
                'filename': 'scalemae-vitlarge-800.pth', 
                'repo_id': 'isaaccorley/vit_large_patch16_224_fmow_rgb_scalemae',
                'description': 'Scale-MAE ViT large model trained on fMoW RGB'
            },
            {
                'filename': 'vit_large_patch16_224_fmow_rgb_scalemae-98ed9821.pth', 
                'repo_id': 'isaaccorley/vit_large_patch16_224_fmow_rgb_scalemae',
                'description': 'Scale-MAE ViT large model trained on fMoW RGB (alternate checkpoint)'
            }
        ]
    }

    def __init__(self, config):    
        super().__init__(config)

    def load_backbone(self):
        """ Loads the Scale-MAE backbone model from Huggingface or from a local path (if available).
        """

        # Get the fixed output size from the config (default is 224)
        fixed_output_size = getattr(self.config, 'fixed_output_size', 224)

        if self.config.pretrained: # Load pretrained weights
            if self.config.local_model_path:
                # Check if the provided local path exists
                if not os.path.exists(self.config.local_model_path):
                    print(f"Provided local model path does not exist: {self.config.local_model_path}")
                    print("Weights will be downloaded from Huggingface instead.")
                    # Check if backbone is supported
                    if self.config.backbone_name not in self.BACKBONE_CHECKPOINTS:
                        raise ValueError(f"Unsupported or missing backbone: '{self.config.backbone_name}'. Supported names are: {list(self.BACKBONE_CHECKPOINTS.keys())}")
                    else:
                        # Check if backbone has weights available
                        if self.BACKBONE_CHECKPOINTS[self.config.backbone_name] is None:
                            raise ValueError(f"No pretrained weights are available for backbone '{self.config.backbone_name}'.")
                        else: # Download the weights and load the model
                            # For now, just load the first checkpoint available for the backbone
                            temp_checkpoint_name = self.BACKBONE_CHECKPOINTS[self.config.backbone_name][0]
                            checkpoint_name = temp_checkpoint_name['filename']
                            repo_id = temp_checkpoint_name['repo_id']
                            self.config.local_model_path = hf_hub_download(repo_id=repo_id, filename=checkpoint_name, local_dir=os.path.dirname(self.config.local_model_path))
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
                    for name, checkpoint_list in self.BACKBONE_CHECKPOINTS.items():
                        # Ensure the list of checkpoints is not None before iterating
                        if checkpoint_list:
                            # Create a list of all filenames for the current backbone
                            filenames = [ckpt['filename'] for ckpt in checkpoint_list]
                            if checkpoint_name in filenames:
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

for variant in ScaleMAE.BACKBONE_CHECKPOINTS.keys():
    BACKBONE_REGISTRY.register(variant)(ScaleMAE)