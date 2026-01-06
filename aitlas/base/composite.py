import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import BaseModel
from ..models.registries import BACKBONE_REGISTRY, NECK_REGISTRY, DECODER_REGISTRY, HEAD_REGISTRY
from .schemas import CompositeModelSchema

class CompositeModel(BaseModel):
    """Composite model consisting of backbone, neck, decoder, and head.
    """

    schema = CompositeModelSchema

    def __init__(self, config):
        super().__init__(config)

        # BACKBONE
        # Prepare backbone config
        backbone_config = dict(self.config)
        
        # Keys reserved for the orchestrator
        orchestrator_reserved_keys = [
            "task_type", 
            "necks",
            "decoder_name", "decoder_params", 
            "head_name", "head_params"
        ]
        
        for key in orchestrator_reserved_keys:
            backbone_config.pop(key, None)

        # Enforce defaults
        if "out_indices" not in backbone_config or backbone_config["out_indices"] is None:
             backbone_config["out_indices"] = [1, 2, 3, 4]

        # Instantiate backbone
        backbone_cls = BACKBONE_REGISTRY.get(self.config.backbone_name)
        self.backbone = backbone_cls(backbone_config)
        
        # Ensure backbone reports its channels   
        current_channels = self._get_feature_info(self.backbone)

        # Instatiate components
        # NECK
        self.necks = nn.Sequential()
        
        if self.config.necks:
            layers = []
            for i, neck_conf in enumerate(self.config.necks):
                # Copy to avoid modifying the original config
                params = neck_conf.copy()
                neck_name = params.pop("name", None)
                # Check if 'name' is missing
                if not neck_name:
                    raise ValueError(f"Neck config at index {i} is missing 'name'.")
                
                # Get the neck class from the registry
                neck_cls = NECK_REGISTRY.get(neck_name)
                
                # Instantiate the neck
                neck_instance = self._instantiate_component(
                    neck_cls, 
                    current_channels, 
                    **params
                )
                layers.append(neck_instance)

                # Update channels for the next component in the chain
                if hasattr(neck_instance, "process_channel_list"):
                    current_channels = neck_instance.process_channel_list(current_channels)
                elif hasattr(neck_instance, "out_channels"):
                    current_channels = neck_instance.out_channels
                
            self.necks = nn.Sequential(*layers)

        # DECODER
        self.decoder = None
        if self.config.decoder_name:
            decoder_cls = DECODER_REGISTRY.get(self.config.decoder_name)
            
            self.decoder = self._instantiate_component(
                decoder_cls, 
                current_channels,
                **self.config.decoder_params
            )
            
            # Update channels
            if hasattr(self.decoder, "out_channels"):
                current_channels = self.decoder.out_channels

        # HEAD
        head_cls = HEAD_REGISTRY.get(self.config.head_name)
        self.head = self._instantiate_component(
            head_cls, 
            current_channels, 
            num_classes=self.config.num_classes, 
            **self.config.head_params
        )

        # Guardrails for different tasks
        task = self.config.task_type
        
        # Feature extraction (backbone only is allowed)
        if task == "feature extraction":
            pass 

        # Prediction Tasks (head is mandatory)
        elif task in ["classification", "segmentation", "object detection", "change detection"]:
            if self.head is None:
                raise ValueError(
                    f"Task type is '{task}', but no 'head_name' was provided. "
                    f"For {task}, a head is required to produce predictions. "
                    f"Please specify a head in the config."
                )

    def _get_feature_info(self, backbone):
        """
        Function to find output channels for any backbone.
        """
        # Option 1: Check if backbone wrapper contains the attribute 
        # (not implemented at the moment, might be in the future)
        if hasattr(backbone, "feature_info") and backbone.feature_info:
            return backbone.feature_info

        # Access the raw underlying backbone
        raw_backbone = getattr(backbone, "backbone", backbone)
        out_indices = self.config.get("out_indices", [1, 2, 3, 4])
        
        found_channels = None

        # Option 2: Check for standard attributes (timm, Swin, etc.)
        if hasattr(raw_backbone, "feature_info"): 
            # timm style: [{'num_chs': 64, ...}, ...]
            found_channels = [x['num_chs'] for x in raw_backbone.feature_info]
        elif hasattr(raw_backbone, "embed_dims"):
            # Swin, hierarchical transformers
            found_channels = raw_backbone.embed_dims

        # Option 3: Inspect structure (isotropic ViTs)
        if found_channels is None:
            # Check for 'encoder' block
            if hasattr(raw_backbone, "encoder") and len(raw_backbone.encoder) > 0:
                first_block = raw_backbone.encoder[0]
                # Look for LayerNorm (norm1) to find embedding dimension
                if hasattr(first_block, "norm1") and hasattr(first_block.norm1, "normalized_shape"):
                    dim = first_block.norm1.normalized_shape[0]
                    # Isotropic: same dim for all layers
                    num_blocks = len(raw_backbone.encoder)
                    found_channels = [dim] * num_blocks
                    
                    # Create a list long enough to cover the requested indices
                    max_idx = max(out_indices) if out_indices else 4
                    found_channels = [dim] * (max_idx + 1)

        # Option 4: Forward pass on a dummy input
        # Not implemented yet. TODO: Implement if needed

        # Filter channels based on out_indices
        if found_channels:
            # Map 1-based indices (1,2,3,4) to 0-based list access
            final_list = []
            for idx in out_indices:
                list_idx = idx - 1 if idx > 0 else idx
                if 0 <= list_idx < len(found_channels):
                    final_list.append(found_channels[list_idx])
                else:
                    # If index is out of bounds (e.g. isotropic ViT returning 1 tensor), reuse the last channel
                    final_list.append(found_channels[-1])
            return final_list

        raise ValueError(
            f"Could not detect channels for {self.config.backbone_name}. "
        )
    
    def _instantiate_component(self, cls, current_channels, **kwargs):
        """
        Smart helper to instantiate a component.
        """
        try:
            sig = inspect.signature(cls.__init__)
        except ValueError:
             # Fallback for complex decorators
             return cls(current_channels, **kwargs)
             
        params = sig.parameters

        # Prepare data forms
        # If it is a list
        if isinstance(current_channels, list):
            channels_list = current_channels
            # Handle case where list is empty
            single_channel = channels_list[-1] if channels_list else 0
        else:
            # If it is an int
            channels_list = [current_channels]
            single_channel = current_channels

        # Determine target index if specific layer is requested
        in_index = kwargs.get('in_index', -1)
        try:
            target_single_channel = channels_list[in_index]
        except IndexError:
            target_single_channel = single_channel

        # Matching logic
        # ScalarHead (explicit 'in_dim')
        if "in_dim" in params:
             kwargs["in_dim"] = target_single_channel

        # SegmentationHead, RegressionHead, ASPP decoder (explicit 'in_channels')
        elif "in_channels" in params:
            # 'in_channels' implies a single integer input
            kwargs["in_channels"] = target_single_channel

        # Necks, transformer decoders (explicit 'channel_list')
        elif "channel_list" in params:
            kwargs["channel_list"] = channels_list
            
        # UPerNet (explicit 'embed_dim')
        elif "embed_dim" in params:
            kwargs["embed_dim"] = channels_list

        # Simple heads (requires 'dim' as input)
        elif "dim" in params:
            kwargs["dim"] = target_single_channel

        return cls(**kwargs)

    def forward(self, x):
        # Load backbone and get feature embeddings
        features = self.backbone(x)
        
        # Pass through the neck(s)
        features = self.necks(features)
        
        # Pass through the decoder, if it exists
        if self.decoder: 
            features = self.decoder(features)

        # If no head, return the features directly
        if self.head is None:
            return features    

        # Pass through head to get final predictions
        logits = self.head(features)
        
        # Standard segmentation upsampling
        if self.config.task_type == "segmentation":
            # Upsample logits to match input image resolution (H, W)
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(
                    logits, 
                    size=x.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
             
        return logits