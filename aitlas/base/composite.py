import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import BaseModel
from ..models.registries import BACKBONE_REGISTRY, NECK_REGISTRY, DECODER_REGISTRY, HEAD_REGISTRY
from .schemas import CompositeModelSchema
from ..models.necks import NeckSequential

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
             backbone_config["out_indices"] = [0, 1, 2, 3]

        # Instantiate backbone
        backbone_cls = BACKBONE_REGISTRY.get(self.config.backbone_name)
        self.backbone = backbone_cls(backbone_config)
        
        # Ensure backbone reports its channels   
        self.current_channels = self._get_feature_info(self.backbone)

        # Instatiate components
        # NECK(S)
        layers = []
        if self.config.necks:
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
                    self.current_channels, 
                    **params
                )
                layers.append(neck_instance)

                # Update channels for the next component in the chain
                if hasattr(neck_instance, "process_channel_list"):
                    self.current_channels = neck_instance.process_channel_list(self.current_channels)
                elif hasattr(neck_instance, "out_channels"):
                    self.current_channels = neck_instance.out_channels

        # Create Sequential container for necks        
        self.necks = NeckSequential(*layers)

        # DECODER
        self.decoder = None
        if self.config.decoder_name:
            decoder_cls = DECODER_REGISTRY.get(self.config.decoder_name)
            
            self.decoder = self._instantiate_component(
                decoder_cls, 
                self.current_channels,
                **self.config.decoder_params
            )
            
            # Update channels
            if hasattr(self.decoder, "out_channels"):
                self.current_channels = self.decoder.out_channels

        # HEAD
        self.head = None
        if self.config.head_name:
            head_cls = HEAD_REGISTRY.get(self.config.head_name)
            self.head = self._instantiate_component(
                head_cls, 
                self.current_channels, 
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

    def forward(self, x=None, **kwargs):
        # Check for 'tim' in backbone self.config.backbone_name for TerraMind's Thinking in Modalities
        if "tim" in self.config.backbone_name:
            backbone_fn = self.backbone.thinking_in_modalities
        else:
            # Standard forward method
            backbone_fn = self.backbone
        
        # Load backbone and get feature embeddings
        if x is not None:
            features = backbone_fn(x, **kwargs)
        else:
            features = backbone_fn(**kwargs)

        cur_shapes, cur_channels = self._get_feature_shape(features)
        print(f"Feature shapes (backbone): {cur_shapes}")
        print(f"Feature channels (backbone): {cur_channels}")
        
        # Pass through the neck(s)
        # Use len(self.necks) because it is an nn.Sequential object
        if len(self.necks) > 0:
            features = self.necks(features, **kwargs)

        cur_shapes, cur_channels = self._get_feature_shape(features)
        print(f"Feature shapes (necks): {cur_shapes}")
        print(f"Feature channels (necks): {cur_channels}")
        
        # Pass through the decoder, if it exists
        if self.decoder is not None: 
            features = self.decoder(features)

        cur_shapes, cur_channels = self._get_feature_shape(features)
        print(f"Feature shapes (decoder): {cur_shapes}")
        print(f"Feature channels (decoder): {cur_channels}")

        # If no head, return the features directly
        if self.head is None:
            return features    

        # Pass through head to get final predictions
        logits = self.head(features)

        cur_shapes, cur_channels = self._get_feature_shape(logits)
        print(f"Feature shapes (head): {cur_shapes}")
        print(f"Feature channels (head): {cur_channels}")
        
        # Standard segmentation upsampling
        if self.config.task_type == "segmentation":
            # Upsample logits to match input image resolution (H, W)
            logits = self._upsample_logits(logits, x, kwargs)
             
        cur_shapes, cur_channels = self._get_feature_shape(logits)
        print(f"Final output shapes: {cur_shapes}")
        print(f"Final output channels: {cur_channels}")

        return logits
    
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
        # Handle the case where out_indices in config is None
        out_indices = self.config.get("out_indices")
        if out_indices is None:
            out_indices = [1, 2]
        
        found_channels = None

        # Option 1: Check for standard attributes (timm, Swin, etc.)
        if hasattr(raw_backbone, "feature_info"): 
            # timm style: [{'num_chs': 64, ...}, ...]
            found_channels = [x['num_chs'] for x in raw_backbone.feature_info]
        elif hasattr(raw_backbone, "embed_dims"):
            # Swin, hierarchical transformers
            found_channels = raw_backbone.embed_dims

        # Option 2: Inspect structure (isotropic ViTs, Presto)
        if found_channels is None:
            # Case A: Check for 'encoder'
            if hasattr(raw_backbone, "encoder"):
                encoder_obj = raw_backbone.encoder
                # Check if it is a list/container (e.g., AnySat, Panopticon, standard ViT)
                if hasattr(encoder_obj, "__len__") and len(encoder_obj) > 0:
                    first_block = encoder_obj[0]
                    if hasattr(first_block, "norm1") and hasattr(first_block.norm1, "normalized_shape"):
                        dim = first_block.norm1.normalized_shape[0]
                        num_blocks = len(encoder_obj)
                        found_channels = [dim] * num_blocks        
                # Check if it is a single module (e.g., Presto)
                else:
                    if hasattr(encoder_obj, "norm"):
                        norm_layer = encoder_obj.norm
                        if hasattr(norm_layer, "normalized_shape"):
                            # LayerNorm((128,), eps=1e-05, ...)
                            dim = norm_layer.normalized_shape[0]
                            found_channels = [dim]
            # Case B: Check for 'blocks'
            elif hasattr(raw_backbone, "blocks") and len(raw_backbone.blocks) > 0:
                first_block = raw_backbone.blocks[0]
                if hasattr(first_block, "norm1") and hasattr(first_block.norm1, "normalized_shape"):
                    dim = first_block.norm1.normalized_shape[0]
                    num_blocks = len(raw_backbone.blocks) 
                    found_channels = [dim] * num_blocks

        # Option 3: Inspect the structure (ResNet backbone)
        if found_channels is None:
            if hasattr(raw_backbone, "encoder_q"):
                enc = raw_backbone.encoder_q
                last_known_dim = None
                # Iterate through all layers in the Sequential blocks
                for child in enc.children():
                    # We are looking for the ResNet stages
                    if isinstance(child, torch.nn.Sequential) and len(child) > 0:
                        last_block = child[-1]                        
                        # Logic for ResNet18/34 (BasicBlock uses bn2)
                        if hasattr(last_block, "bn2") and hasattr(last_block.bn2, "num_features"):
                            last_known_dim = last_block.bn2.num_features                       
                        # Logic for ResNet50/101 (Bottleneck uses bn3)
                        elif hasattr(last_block, "bn3") and hasattr(last_block.bn3, "num_features"):
                            last_known_dim = last_block.bn3.num_features
                # If we found dimensions, return ONLY the final one as a single-item list.
                if last_known_dim:
                    found_channels = [last_known_dim] # [512] for ResNet18, [2048] for ResNet50

        # Option 4: AnySat detection
        if found_channels is None:
            # AnySat wrapper nests the model AnySatModule -> model (AnySatEncoder)     
            if hasattr(raw_backbone, "model"):
                possible_inner = raw_backbone.model
                if hasattr(possible_inner, "spatial_encoder") or hasattr(possible_inner, "projector_s2"):
                    raw_backbone = possible_inner
            # Inspect the structure
            if hasattr(raw_backbone, "blocks") and hasattr(raw_backbone, "spatial_encoder"):
                # It is AnySat
                if len(raw_backbone.blocks) > 0:
                    first_block = raw_backbone.blocks[0]
                    if hasattr(first_block, "norm1") and hasattr(first_block.norm1, "normalized_shape"):
                        base_dim = first_block.norm1.normalized_shape[0] # Usually 768  
                        # Check for 'dense' output mode in the config
                        # AnySat 'dense' mode typically concatenates features or upscales, doubling channels (1536)
                        backbone_params = self.config.get("backbone_params", {})
                        # Also check root config in case params are merged
                        output_mode = backbone_params.get("output", self.config.get("output"))
                        if output_mode == "dense":
                             # AnySat 'dense' output is usually 2x the base dimension
                            final_dim = base_dim * 2 
                            # Dense usually returns a single tensor, not a list of blocks
                            found_channels = [final_dim] 
                        else:
                            # 'tile', 'patch', 'all' modes return the base dimension
                            final_dim = base_dim
                            found_channels = [final_dim] * len(raw_backbone.blocks)
        
        # Option 5: CROMA detection
        if found_channels is None:
            # Check for CROMA-specific attributes (Optical Projection Head)
            if hasattr(raw_backbone, "s2_GAP_FFN") and isinstance(raw_backbone.s2_GAP_FFN, torch.nn.Sequential):
                # CROMA uses a specific Sequential head for GAP: LayerNorm -> Linear -> GELU -> Linear
                # We access the last Linear layer to get the output dimension (usually 768).
                if len(raw_backbone.s2_GAP_FFN) > 0:
                    last_layer = raw_backbone.s2_GAP_FFN[-1]
                    if hasattr(last_layer, "out_features"):
                        found_channels = [last_layer.out_features]
        
        # Option 6: Panopticon detection
        if found_channels is None:
            # Check for Panopticon wrapper structure
            target_model = raw_backbone
            if hasattr(target_model, "model") and hasattr(target_model.model, "blocks"):
                # Drill down to the VisionTransformer inside PanopticonModule
                target_model = target_model.model       
            # Now check for standard ViT blocks
            if hasattr(target_model, "blocks"):
                # Get the last block to determine final output dimension
                blocks = target_model.blocks
                if len(blocks) > 0:
                    last_block = blocks[-1]            
                    # Detect dimension from the first LayerNorm or Linear layer in the block
                    if hasattr(last_block, "norm1") and hasattr(last_block.norm1, "normalized_shape"):
                        # norm1: LayerNorm((768,), ...)
                        dim = last_block.norm1.normalized_shape[0]
                        # Panopticon returns all tokens [B, N, 768] (dense=True) or CLS [B, 768] (dense=False)
                        # In both cases, the feature dimension 'C' seen by heads/necks is 768.
                        found_channels = [dim]
        
        # Option 7: Forward pass on a dummy input
        # Not implemented yet. TODO: Implement if needed

        # Filter channels based on out_indices
        if found_channels:
            final_list = []
            for idx in out_indices:
                # If index is valid, append the corresponding channel
                if 0 <= idx < len(found_channels):
                    final_list.append(found_channels[idx])
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
        if isinstance(self.current_channels, list):
            channels_list = self.current_channels
            # Handle case where list is empty
            single_channel = channels_list[-1] if channels_list else 0
        else:
            # If it is an int
            channels_list = [self.current_channels]
            single_channel = self.current_channels

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

    def _get_feature_shape(self, features):
        """
        Helper to find feature shapes and channels from backbone output.
        """

        # Case 1: Single dictionary (standard for some models)
        if isinstance(features, dict):
            feature_list = [v for v in features.values() if isinstance(v, torch.Tensor)]
        
        # Case 2: List or tuple
        elif isinstance(features, (list, tuple)):
            if len(features) > 0:
                first_elem = features[0]
                
                # Case 2a: List of dictionaries (TerraMind style)
                if isinstance(first_elem, dict):
                    feature_list = []
                    for item in features:
                        if isinstance(item, dict):
                            feature_list.extend([v for v in item.values() if isinstance(v, torch.Tensor)])
                
                # Case 2b: List of tensors (standard)
                else:
                    # Special handling for Galileo to limit to first 4 features
                    if "galileo" in self.config.backbone_name and len(features) > 4:
                        feature_list = list(features[:4])
                    else:
                        feature_list = features
            else:
                feature_list = [] # Empty list handling
        
        # Case 3: Single tensor
        else:
            feature_list = [features]

        current_shapes = []
        current_channels = []

        for f in feature_list:
            shape = list(f.shape)
            current_shapes.append(shape)
            
            # Logic to find "channel" dim based on tensor rank
            if len(shape) == 6:
                # Galileo 6D: [B, H, W, T, G, D] -> Last dim is D
                current_channels.append(shape[-1])

            elif len(shape) == 5:
                # Galileo 5D (space-only): [B, H, W, G, D] -> last dim is D
                current_channels.append(shape[-1])

            elif len(shape) == 4: 
                # Standard PyTorch [B, C, H, W]
                if "galileo" in self.config.backbone_name:
                    current_channels.append(shape[-1]) # Galileo sometimes permutes to [B, T, H, W] or similar where D is last
                else:
                    current_channels.append(shape[1]) # Standard [B, C, H, W]

            elif len(shape) == 3: 
                # [B, N, C] -> Token-like (patch) -> Last dim is C
                current_channels.append(shape[2])

            elif len(shape) == 2: 
                # [B, C] -> Vector (tile)
                current_channels.append(shape[1])
                
            else:
                current_channels.append("?")

        return current_shapes, current_channels

    
    def _upsample_logits(self, logits, x, kwargs):
        """
        Helper to safely determine target size and upsample.
        """
        target_size = None

        # Helper to find spatial dims in a potential input object
        def get_shape(obj):
            # Case A: It's a Tensor [B, C, H, W]
            if isinstance(obj, torch.Tensor) and obj.ndim >= 3:
                return obj.shape[-2:]
            # Case B: It's a dictionary containing Tensors
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, torch.Tensor) and v.ndim >= 3:
                        return v.shape[-2:]
            return None

        # Option 1: Check explicit 'x'
        if x is not None:
            target_size = get_shape(x)

        # Option 2: Check kwargs (for 'inputs', 's2', 'x_dict' etc.)
        if target_size is None:
            for v in kwargs.values():
                target_size = get_shape(v)
                if target_size: break

        # Option 3: Fallback (If we can't find input size, don't resize)
        if target_size is None:
            target_size = logits.shape[-2:]

        # Perform Interpolation
        if logits.shape[-2:] != target_size:
            return F.interpolate(
                logits, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        return logits