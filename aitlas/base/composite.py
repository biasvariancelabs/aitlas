import inspect
import torch
import torch.nn.functional as F
from .models import BaseModel
from aitlas.models.registries import BACKBONE_REGISTRY, NECK_REGISTRY, DECODER_REGISTRY, HEAD_REGISTRY
from .schemas import CompositeModelSchema

class CompositeModel(BaseModel):
    """Composite model consisting of backbone, neck, decoder, and head.
    """

    schema = CompositeModelSchema

    def __init__(self, config):
        super().__init__(config)

        # Prepare backbone config
        backbone_config = dict(self.config)
        
        # Keys reserved for the orchestrator
        orchestrator_reserved_keys = [
            "task_type", 
            "neck_name", "neck_params",
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
        if getattr(self.backbone, "feature_info", None) is None:
             raise ValueError(
                 f"Backbone '{self.config.backbone_name}' has no channel information. "
                 f"Please implement `get_feature_info()` in the {backbone_cls.__name__} wrapper "
                 f"or set `self.feature_info` in its __init__."
             )
             
        current_channels = self.backbone.feature_info

        # Instatiate components
        # NECK
        self.neck = None
        if self.config.neck_name:
            neck_cls = NECK_REGISTRY.get(self.config.neck_name)
            
            # Smartly instantiate neck (handles 'channel_list' or 'in_channels')
            self.neck = self._instantiate_component(
                neck_cls, 
                current_channels, 
                **self.config.get("neck_params", {})
            )
            
            # Update current_channels for the next stage
            # For TerraTorch necks that use 'process_channel_list'
            if hasattr(self.neck, "process_channel_list"):
                current_channels = self.neck.process_channel_list(current_channels)
            # For other necks that could use 'out_channels'
            elif hasattr(self.neck, "out_channels"):
                current_channels = self.neck.out_channels

        # DECODER
        self.decoder = None
        if self.config.decoder_name:
            decoder_cls = DECODER_REGISTRY.get(self.config.decoder_name)
            
            self.decoder = self._instantiate_component(
                decoder_cls, 
                current_channels, 
                num_classes=self.config.num_classes, 
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
        
        # Pass through neck and decoder if they exist
        if self.neck: 
            features = self.neck(features)
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