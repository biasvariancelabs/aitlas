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

        # BACKBONE
        # Prepare backbone config by removing keys meant for orchestrator/sub-components
        backbone_config = dict(self.config)
        
        # These keys that are strictly for the orchestrator or sub-components
        orchestrator_reserved_keys = [
            "task_type", 
            "neck_name", "neck_params",
            "decoder_name", "decoder_params", 
            "head_name", "head_params"
        ]
        
        for key in orchestrator_reserved_keys:
            backbone_config.pop(key, None)

        # Enforce defaults for backbone if missing
        if "out_indices" not in backbone_config or backbone_config["out_indices"] is None:
             backbone_config["out_indices"] = [1, 2, 3, 4]

        # Instantiate backbone
        backbone_cls = BACKBONE_REGISTRY.get(self.config.backbone_name)
        self.backbone = backbone_cls(backbone_config) # Clean config passed here
        current_channels = self.backbone.feature_info

        # Instantiate sub-components 
        # NECK
        self.neck = None
        if self.config.neck_name:
            neck_cls = NECK_REGISTRY.get(self.config.neck_name)
            self.neck = neck_cls(
                in_channels=current_channels, 
                **self.config.get("neck_params", {})
            )
            # Update channels if the neck changes them
            if hasattr(self.neck, "out_channels"):
                current_channels = self.neck.out_channels

        # DECODER
        self.decoder = None
        if self.config.decoder_name:
            decoder_cls = DECODER_REGISTRY.get(self.config.decoder_name)
            self.decoder = decoder_cls(
                in_channels=current_channels,
                num_classes=self.config.num_classes,
                **self.config.decoder_params
            )
            # Update channels (decoders usually output a single rich feature map)
            if hasattr(self.decoder, "out_channels"):
                current_channels = self.decoder.out_channels

        # HEAD
        # Defaults to "Default" via schema if missing
        head_cls = HEAD_REGISTRY.get(self.config.head_name)
        self.head = head_cls(
            in_channels=current_channels,
            num_classes=self.config.num_classes,
            **self.config.head_params
        )

    def forward(self, x):
        # Load backbone and get feature embeddings
        features = self.backbone(x)
        
        # Pass through neck and decoder if they exist
        if self.neck: 
            features = self.neck(features)
        if self.decoder: 
            features = self.decoder(features)
            
        # Pass through head to get final predictions
        logits = self.head(features)
        
        # Standard segmentation upsampling
        if self.config.task_type == "segmentation":
            # Upsample logits to match input image resolution (H, W)
            # x is the original input tensor
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(
                    logits, 
                    size=x.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
             
        return logits