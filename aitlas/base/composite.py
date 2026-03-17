import inspect
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from aitlas.base import BaseMulticlassClassifier, BaseMultilabelClassifier, BaseObjectDetection, BaseSegmentationClassifier, BaseChangeDetection, FoundationModel, BaseInputAdapter
from .models import BaseModel
from ..models.registries import BACKBONE_REGISTRY, NECK_REGISTRY, DECODER_REGISTRY, HEAD_REGISTRY, ADAPTER_REGISTRY
from .schemas import CompositeModelSchema, CompositeClassificationSchema, CompositeSegmentationSchema, CompositeObjectDetectionSchema
from ..models.necks import NeckSequential

class CompositeModelArchitectureMixin:
    """Composite model consisting of backbone, neck, decoder, and head.
    """

    def setup_composite(self):

        # DATA-MODEL ADAPTER
        adapter_name = getattr(self.config, "adapter_name", None)
        if adapter_name:
            adapter_cls = ADAPTER_REGISTRY.get(adapter_name)
            self.input_adapter = adapter_cls(self.config)
        else:
            self.input_adapter = BaseInputAdapter(self.config)

        # BACKBONE
        # Prepare backbone config
        backbone_config = dict(self.config)
        
        # Keys reserved for the orchestrator
        orchestrator_reserved_keys = [
            "task_type", 
            "necks",
            "decoder_name", "decoder_params", 
            "head_name", "head_params",
            "freeze_modules",
            "forward_params",
            "backbone_setup_calls",
            "learning_rate", "weight_decay", 
            "threshold", "freeze",
            "metrics", "mode",
            "step_size", "gamma",
            "adapter_name", "selection",
            "bands", "bands_s1", 
            "bands_s2", "bands_l8"
        ]
        
        for key in orchestrator_reserved_keys:
            backbone_config.pop(key, None)

        # Instantiate backbone
        backbone_cls = BACKBONE_REGISTRY.get(self.config.backbone_name)
        self.model.backbone = backbone_cls(backbone_config)

        # Get the list of possible methods to call, defaulting to an empty list if None
        setup_calls = getattr(self.config, "backbone_setup_calls", []) or []
        
        for call_info in setup_calls:
            method_name = call_info.get("method")
            params = call_info.get("params", {})
            
            if not method_name:
                continue # Skip if the user forgot to specify a method name
                
            # Verify the backbone actually has this method
            if hasattr(self.model.backbone, method_name):
                # Get the method from the backbone and execute it with the provided parameters
                method = getattr(self.model.backbone, method_name)
                method(**params) 
            else:
                # Raise a warning if the method is not found in the backbone
                warnings.warn(f"Setup method '{method_name}' not found in backbone '{self.config.backbone_name}'. Skipping.") 
        
        # Ensure backbone reports its output indices and channels
        self.out_indices = self.model.backbone.out_indices
        self.current_channels = self._get_feature_info(self.model.backbone)

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

                # If the neck asks for specific 'indices' (e.g., SelectIndices), validate the config input
                self._validate_indices(neck_name, params)
                
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
        self.model.necks = NeckSequential(*layers)

        # DECODER
        self.model.decoder = None
        if self.config.decoder_name:
            decoder_cls = DECODER_REGISTRY.get(self.config.decoder_name)
            
            self.model.decoder = self._instantiate_component(
                decoder_cls, 
                self.current_channels,
                **self.config.decoder_params
            )
            
            # Update channels
            if hasattr(self.model.decoder, "out_channels"):
                self.current_channels = self.model.decoder.out_channels

        # HEAD
        self.model.head = None
        if self.config.head_name:
            head_cls = HEAD_REGISTRY.get(self.config.head_name)
            self.model.head = self._instantiate_component(
                head_cls, 
                self.current_channels, 
                num_classes=self.config.num_classes, 
                **self.config.head_params
            )

        # Guardrails for different tasks
        self.task = self.config.task_type
        
        # Feature extraction (backbone only is allowed)
        if self.task == "feature extraction":
            if len(self.model.necks) > 0 or self.model.decoder is not None or self.model.head is not None:
                warnings.warn(
                    f"Task is '{self.task}', but necks, decoder, or head were found in the config. "
                    f"They are not used for pure feature extraction and will be ignored/removed."
                )
                # Force them to be empty/None
                self.model.necks = NeckSequential() # Empty sequential, so that len() == 0 is safe
                self.model.decoder = None
                self.model.head = None
            pass 

        # Prediction tasks (head is mandatory)
        elif self.task in ["multiclass classification", "multilabel classification", "segmentation", "object detection", "change detection"]:
            if self.model.head is None:
                raise ValueError(
                    f"Task type is '{self.task}', but no 'head_name' was provided. "
                    f"For {self.task}, a head is required to produce predictions. "
                    f"Please specify a head in the config."
                )
            
        # Freeze componenets listed in the config
        self._apply_freezing()

    def forward(self, x=None, **kwargs):
        """Standard forward pass through the composite model.
        """

        # Pass the input through the adapter if it exists, and get any dynamic kwargs it produces
        dynamic_adapter_kwargs = {}
        if x is not None and hasattr(self, "input_adapter"):
            x, dynamic_adapter_kwargs = self.input_adapter(x)

        # Get the forward() params. If missing or None, default to an empty dict {}
        raw_forward_params = getattr(self.config, "forward_params", {}) or {}
        
        # Copy to a new dictionary
        static_forward_kwargs = dict(raw_forward_params)
        
        # Merge with any dynamic kwargs passed into the adapter and forward function
        static_forward_kwargs.update(dynamic_adapter_kwargs)
        static_forward_kwargs.update(kwargs)

        # Check for 'tim' in backbone self.config.backbone_name for TerraMind's Thinking in Modalities
        if "tim" in self.config.backbone_name:
            backbone_fn = self.model.backbone.thinking_in_modalities
        else:
            # Standard forward method
            backbone_fn = self.model.backbone
        
        # Load backbone and get feature embeddings
        if x is not None:
            features = backbone_fn(x, **static_forward_kwargs)
        else:
            features = backbone_fn(**static_forward_kwargs)

        cur_shapes, cur_channels = self._get_feature_shape(features)
        #print(f"Feature shapes (backbone): {cur_shapes}")
        #print(f"Feature channels (backbone): {cur_channels}")

        # Standardize features -> List[Tensor]
        features = self._standardize_features(features)

        # Check and dynamically rebuild necks, decoders and heads if needed
        self._check_and_rebuild_components(features)
        
        # Infer image size for necks if not provided in kwargs
        kwargs = self._infer_image_size(x, kwargs)
        
        # Pass through the neck(s)
        # Use len(self.model.necks) because it is an nn.Sequential object
        if len(self.model.necks) > 0:
            features = self.model.necks(features, **kwargs)

        cur_shapes, cur_channels = self._get_feature_shape(features)
        #print(f"Feature shapes (necks): {cur_shapes}")
        #print(f"Feature channels (necks): {cur_channels}")
        
        # Pass through the decoder, if it exists
        if self.model.decoder is not None: 
            features = self.model.decoder(features)

        cur_shapes, cur_channels = self._get_feature_shape(features)
        #print(f"Feature shapes (decoder): {cur_shapes}")
        #print(f"Feature channels (decoder): {cur_channels}")

        # If no head, return the features directly
        if self.model.head is None:
            return features    

        # Pass through head to get final predictions
        logits = self.model.head(features)

        cur_shapes, cur_channels = self._get_feature_shape(logits)
        #print(f"Feature shapes (head): {cur_shapes}")
        #print(f"Feature channels (head): {cur_channels}")
        
        # Standard segmentation upsampling
        if self.task == "segmentation":
            # Upsample logits to match input image resolution (H, W)
            logits = self._upsample_logits(logits, x, kwargs)
             
        cur_shapes, cur_channels = self._get_feature_shape(logits)
        #print(f"Final output shapes: {cur_shapes}")
        #print(f"Final output channels: {cur_channels}")

        return logits

    def predict(self, x=None, **kwargs):
        """Inference pass that returns probabilities/values instead of logits.
        """

        # Run standard forward pass to get logits
        logits = self.forward(x, **kwargs)

        # Apply task-specific activation
        if self.task == "multiclass classification":
            return torch.softmax(logits, dim=1)
        
        elif self.task == "multilabel classification":
            return torch.sigmoid(logits)

        elif self.task == "segmentation":
            #return torch.argmax(logits, dim=1) # TODO: change back to argmax after testing
            return torch.softmax(logits, dim=1)
        
        return logits

    def _get_feature_info(self, backbone):
        """Function to find output channels for any backbone.
        """
        # Option 1: Check if backbone wrapper contains the attribute 
        # (not implemented at the moment, might be in the future)
        if hasattr(backbone, "feature_info") and backbone.feature_info:
            return backbone.feature_info

        # Access the raw underlying backbone
        raw_backbone = getattr(backbone, "backbone", backbone)
        
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

        # Filter channels based on output indices
        if found_channels:
            final_list = []
            for idx in self.out_indices:
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
        """Smart helper to instantiate a component.
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
            """Helper to find feature shapes and channels from backbone output.
            """

            # Option 1: Single dictionary (standard for some models)
            if isinstance(features, dict):
                feature_list = [v for v in features.values() if isinstance(v, torch.Tensor)]
            
            # Option 2: List or tuple
            elif isinstance(features, (list, tuple)):
                if len(features) > 0:
                    first_elem = features[0]
                    
                    # Option 2a: List of dictionaries (TerraMind style)
                    if isinstance(first_elem, dict):
                        feature_list = []
                        for item in features:
                            if isinstance(item, dict):
                                feature_list.extend([v for v in item.values() if isinstance(v, torch.Tensor)])
                    
                    # Option 2b: List of tensors (standard)
                    else:
                        # Special handling for Galileo to limit to first 4 features (Raw Output only)
                        if "galileo" in self.config.backbone_name and len(features) > 4:
                            feature_list = list(features[:4])
                        else:
                            feature_list = features
                else:
                    feature_list = [] # Empty list handling
            
            # Option 3: Single tensor
            else:
                feature_list = [features]

            current_shapes = []
            current_channels = []

            for f in feature_list:
                if not isinstance(f, torch.Tensor):
                    # Guard against non-tensor elements slipping through
                    current_channels.append("?")
                    continue
                    
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
                    # Once standardized to 4D, it is always Channel-First (dim 1).
                    current_channels.append(shape[1]) 

                elif len(shape) == 3: 
                    # [B, N, C] -> Token-like (patch) -> Last dim is C
                    current_channels.append(shape[2])

                elif len(shape) == 2: 
                    # [B, C] -> Vector (tile)
                    current_channels.append(shape[1])
                    
                else:
                    current_channels.append("?")

            return current_shapes, current_channels

    def _standardize_features(self, features) -> list[torch.Tensor]:
        """Standardizes the output of various backbones into a uniform List[Tensor] format
        expected by necks and heads.
        """
        
        # Option 1: Single Tensor (e.g., standard ViT with one output)
        if isinstance(features, torch.Tensor):
            return [features]

        # Option 2: Tuple (specific to Galileo with average_features=False)
        elif isinstance(features, tuple):
            # features[0] is the spatial embedding: (B, H', W', T, 7, D)
            raw_tensor = features[0]
            # Average pool over T (dim 3) and 7 (dim 4) -> (B, H', W', D)
            pooled = raw_tensor.mean(dim=(3, 4))
            # Permute to standard image format (B, D, H', W')
            permuted = pooled.permute(0, 3, 1, 2)
            return [permuted]

        # Option 3: Dictionary (specific to CROMA)
        elif isinstance(features, dict):
            # Access the backbone modalities safely
            if hasattr(self.model.backbone, 'backbone') and hasattr(self.model.backbone.backbone, 'modalities'):
                modalities = self.model.backbone.backbone.modalities
            else:
                # Fallback: if we can't find the attribute, assume it's the full joint model
                modalities = ['optical', 'sar']

            # Robust checks
            if len(modalities) == 1 and modalities[0] == 'optical':
                return [features['optical_encodings']]
            elif len(modalities) == 1 and modalities[0] == 'sar':
                return [features['sar_encodings']]
            else:
                # Default to joint for ['optical', 'sar'], ['sar', 'optical'], or defaults
                return [features['joint_encodings']]

        # Option 4: Already a list
        elif isinstance(features, list):
            # Optional: Validate contents are tensors
            for i, feat in enumerate(features):
                if not isinstance(feat, torch.Tensor):
                    raise TypeError(f"Expected features[{i}] to be a Tensor, but got {type(feat)}.")
            return features

        # Option 5: Unknown type
        else:
            raise TypeError(f"Expected features to be a Tensor, tuple, dict, or list of Tensors, but got {type(features)}.")

    def _infer_image_size(self, x, kwargs: dict) -> dict:
        """Attempts to infer the spatial image size (H, W) from inputs 'x' or 'kwargs'
        and injects 'image_size' into kwargs if found.
        """
        # If 'image_size' is already provided explicitly, trust it and return early
        if "image_size" in kwargs:
            return kwargs

        ref_tensor = None

        # Try input 'x' (standard backbones like ViT)
        if x is not None:
            if isinstance(x, torch.Tensor):
                ref_tensor = x
            elif isinstance(x, dict):
                # Grab the first tensor found in the dictionary values
                for v in x.values():
                    if isinstance(v, torch.Tensor):
                        ref_tensor = v
                        break

        # If 'x' failed, scan kwargs (multimodal backbones like CROMA, Presto)
        if ref_tensor is None:
            for k, v in kwargs.items():
                # Look for tensors with spatial dimensions (ndim >= 4)
                # Common keys: x_optical, x_sar, s2, images, etc.
                if isinstance(v, torch.Tensor) and v.ndim >= 4:
                    ref_tensor = v
                    break
                elif isinstance(v, dict):
                     # Handle cases like Presto where input might be nested in a dict
                     for sub_v in v.values():
                         if isinstance(sub_v, torch.Tensor) and sub_v.ndim >= 4:
                             ref_tensor = sub_v
                             break
                
                if ref_tensor is not None:
                    break

        # Apply the size if a reference tensor was found
        if ref_tensor is not None:
            # We assume standard format (B, C, H, W) or (B, T, C, H, W)
            kwargs["image_size"] = ref_tensor.shape[-2:]

        return kwargs
    
    def _check_and_rebuild_components(self, features: list[torch.Tensor]):
        """Checks feature shapes and dynamically rebuilds necks, decoders, and heads if needed.
        """
        
        # Get current (temporary) feature shapes
        _, temp_channels = self._get_feature_shape(features)

        # Compare with self.current_channels to see if rebuilding is needed
        if temp_channels != self.current_channels:

            # Print a warning about dynamic rebuilding
            warnings.warn(
                "Detected change in feature channels from backbone output. "
                "Dynamically rebuilding necks, decoder, and head to match new channels from {self.current_channels} to {temp_channels}. "
            )
            
            # Update current channels
            self.current_channels = temp_channels

            # Get the device and dtype from the features
            # We assume features is a list of tensors at this point
            temp_tensor = features[0] if isinstance(features, list) else features
            device = temp_tensor.device
            dtype = temp_tensor.dtype

            # Get the current mode (train/eval)
            is_training_mode = self.model.training

            # Rebuild NECK(S)
            layers = []
            if self.config.necks:
                for i, neck_conf in enumerate(self.config.necks):
                    params = neck_conf.copy()
                    neck_name = params.pop("name", None)
                    neck_cls = NECK_REGISTRY.get(neck_name)
                    
                    # Instantiate using existing helper
                    neck_instance = self._instantiate_component(
                        neck_cls, 
                        self.current_channels, 
                        **params
                    )
                    layers.append(neck_instance)

                    # Update channels for next component
                    if hasattr(neck_instance, "process_channel_list"):
                        self.current_channels = neck_instance.process_channel_list(self.current_channels)
                    elif hasattr(neck_instance, "out_channels"):
                        self.current_channels = neck_instance.out_channels
            
            # Replace and move to device
            self.model.necks = NeckSequential(*layers).to(device=device, dtype=dtype)
            # Put back to original mode (train/eval)
            self.model.necks.train(is_training_mode)

            # Rebuild DECODER
            if self.config.decoder_name:
                decoder_cls = DECODER_REGISTRY.get(self.config.decoder_name)
                self.model.decoder = self._instantiate_component(
                    decoder_cls, 
                    self.current_channels,
                    **self.config.decoder_params
                ).to(device=device, dtype=dtype)
                
                # Put back to original mode (train/eval)
                self.model.decoder.train(is_training_mode)

                if hasattr(self.model.decoder, "out_channels"):
                    self.current_channels = self.model.decoder.out_channels
            
            # Rebuild HEAD
            if self.config.head_name:
                head_cls = HEAD_REGISTRY.get(self.config.head_name)
                self.model.head = self._instantiate_component(
                    head_cls, 
                    self.current_channels, 
                    num_classes=self.config.num_classes, 
                    **self.config.head_params
                ).to(device=device, dtype=dtype)

                # Put back to original mode (train/eval)      
                self.model.head.train(is_training_mode)     
    
    def _validate_indices(self, neck_name: str, params: dict):
        """Internal helper to validate requested indices against available backbone features.
        """
        num_indices = len(self.out_indices)

        if "indices" not in params or num_indices is None:
            return

        requested_indices = params["indices"]
        
        # Handle case where config might be a single int
        if isinstance(requested_indices, int):
            requested_indices = [requested_indices]

        # Get max and min indices for validation
        max_idx = max(requested_indices) if requested_indices else -1
        min_idx = min(requested_indices) if requested_indices else 0

        # Check upper bound
        if max_idx >= num_indices:
            raise ValueError(
                f"Configuration error in neck '{neck_name}': "
                f"Requested index {max_idx} is out of bounds. "
                f"The backbone only outputs {len(self.out_indices)} "
                f"feature maps (indices 0 to {len(self.out_indices)-1})."
            )

        # Check lower bound
        if min_idx < 0:
            raise ValueError(f"Configuration error in neck '{neck_name}': Indices cannot be negative.")
    
    def _upsample_logits(self, logits, x, kwargs):
        """Helper to safely determine target size and upsample.
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
    
    def _apply_freezing(self):
        """Helper to freeze components (backbone, necks, decoder, head) based on the config.
        """
        
        # Helper function to freeze a specific PyTorch module
        def freeze_module(module):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False

        # Handle legacy AiTLAS `freeze` boolean (if the user sets 'freeze: True', we assume they mean 'freeze backbone')
        if getattr(self.config, "freeze", False):
            freeze_module(self.model.backbone)
            
        # Handle `freeze_modules`
        freeze_list = getattr(self.config, "freeze_modules", [])
        
        if "backbone" in freeze_list:
            freeze_module(self.model.backbone)
            
        if "necks" in freeze_list:
            freeze_module(self.model.necks)
            
        if "decoder" in freeze_list:
            freeze_module(self.model.decoder)
            
        if "head" in freeze_list:
            warnings.warn(
                "You are freezing the prediction head! This is usually not recommended unless doing zero-shot evaluation.",
                UserWarning
            )
            freeze_module(self.model.head)
    

# Task-specific composite models that inherit from the mixin and the appropriate base class for their task type
class CompositeMultiClassificationModel(CompositeModelArchitectureMixin, BaseMulticlassClassifier):
    """Composite model for multiclass classification tasks.
    """

    schema = CompositeClassificationSchema

    def __init__(self, config):

        # Initialize the mixin to set up the composite architecture
        super().__init__(config)

        # Build the specific backbone/neck/decoder/head pipeline based on the config
        self.setup_composite()

    pass

class CompositeMultilabelClassificationModel(CompositeModelArchitectureMixin, BaseMultilabelClassifier):
    """Composite model for multilabel classification tasks.
    """

    schema = CompositeClassificationSchema

    def __init__(self, config):
        
        # Initialize the mixin to set up the composite architecture
        super().__init__(config)

        # Build the specific backbone/neck/decoder/head pipeline based on the config
        self.setup_composite()

    pass

class CompositeSegmentationModel(CompositeModelArchitectureMixin, BaseSegmentationClassifier):
    """Composite model for segmentation and change detection tasks.
    """

    schema = CompositeSegmentationSchema

    def __init__(self, config):
        
        # Initialize the mixin to set up the composite architecture
        super().__init__(config)

        # Build the specific backbone/neck/decoder/head pipeline based on the config
        self.setup_composite()

    pass

class CompositeObjectDetectionModel(CompositeModelArchitectureMixin, BaseObjectDetection):
    """Composite model for object detection tasks.
    """

    schema = CompositeObjectDetectionSchema

    def __init__(self, config):
        
        # Initialize the mixin to set up the composite architecture
        super().__init__(config)

        # Build the specific backbone/neck/decoder/head pipeline based on the config
        self.setup_composite()

    pass

class CompositeChangeDetectionModel(CompositeModelArchitectureMixin, BaseChangeDetection):
    """Composite model for change detection tasks.
    """

    schema = CompositeSegmentationSchema # Reuses the same schema as segmentation since change detection is a special case of segmentation

    def __init__(self, config):
        
        # Initialize the mixin to set up the composite architecture
        super().__init__(config)

        # Build the specific backbone/neck/decoder/head pipeline based on the config
        self.setup_composite()

    pass

class CompositeFeatureExtractionModel(CompositeModelArchitectureMixin, FoundationModel):
    """
    Lightweight model for raw feature extraction. 
    Only requires a backbone. No head, no training loop.
    """
    
    # Use the base schema without any task-specific classifier fields
    schema = CompositeModelSchema 
    
    def __init__(self, config):
        # Initialize the standard BaseModel
        super().__init__(config)
        
        # Build the architecture (Backbone, Neck)
        self.setup_composite()

    def load_backbone(self):
        """
        Overrides the FoundationModel requirement.
        We return None here because our Mixin handles instantiation 
        inside setup_composite().
        """
        return None
    
    pass

# Factory composite model that can be used for any task type based on the config
class CompositeModel:
    """Factory that returns the correct task-specific composite model.
    """
    
    def __new__(cls, config):
        
        # Extract task_type from the config object/dict
        task = getattr(config, "task_type", config.get("task_type", "")).lower()

        if task == "feature extraction":
            return CompositeFeatureExtractionModel(config)
        
        elif task == "multiclass classification":
            return CompositeMultiClassificationModel(config)
            
        elif task == "multilabel classification":
            return CompositeMultilabelClassificationModel(config)
            
        elif task == "segmentation":
            return CompositeSegmentationModel(config)
        
        elif task == "object detection":
            return CompositeObjectDetectionModel(config)
        
        elif task == "change detection":
            return CompositeChangeDetectionModel(config)
        
        else:
            raise ValueError(f"Task type '{task}' is not currently supported.")