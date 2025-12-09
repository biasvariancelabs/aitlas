import pytest
import torch
import torch.nn as nn
import os
from unittest.mock import patch, MagicMock
from aitlas.models import TerraMind

# Constants for testing
EMBED_DIM = 768
BATCH_SIZE = 2
CHANNELS = 12  # Example S2L2A
HEIGHT = 224
WIDTH = 224

# Mock Backbone of the TerraMind model
class MockTerraMindBackbone(nn.Module):
    """
    A mock class to simulate the actual TerraMind ViT backbone.
    """
    def __init__(self, modalities=None):
        super().__init__()
        # Simulate attributes required by the wrapper
        self.head = nn.Linear(EMBED_DIM, 10)
        self.modalities = modalities
        
    def forward(self, d=None, **kwargs):
        """
        Simulates the forward logic of TerraMind backbone.
        TerraMind usually returns a list of embeddings or tokens.
        """
        # Return a dummy embedding list simulating transformer layers
        # Shape: List of (B, Seq_Len, Embed_Dim)
        batch_size = BATCH_SIZE
        if isinstance(d, torch.Tensor):
            batch_size = d.shape[0]
        elif isinstance(d, dict):
            # Get batch size from first value
            batch_size = next(iter(d.values())).shape[0]
            
        return [torch.randn(batch_size, 196, EMBED_DIM)]

    def load_state_dict(self, state_dict, strict=True):
        """ Mock load_state_dict to avoid actual weight loading errors. """
        pass

# Fixtures
@pytest.fixture
def dummy_terramind_input():
    """ Creates a dummy input tensor dictionary. """
    return {"S2L2A": torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)}

@pytest.fixture
def config_terramind_tiny_local():
    """ 
    Valid config for TerraMind Tiny loading locally. 
    """
    return {
        "local_model_path": "/tmp/mock_path/TerraMind_v1_tiny.pt",
        "backbone_name": "terramind_v1_tiny",
        "pretrained": True,
        "modalities": ["S2L2A"]
    }

@pytest.fixture
def config_terramind_inferred_local():
    """ 
    Config where backbone_name is missing but filename allows inference.
    """
    return {
        "local_model_path": "/tmp/mock_path/TerraMind_v1_base.pt",
        "backbone_name": "terramind_v1_base",
        "pretrained": True,
        "modalities": ["S2L2A"]
    }

@pytest.fixture
def config_terramind_tiny_download():
    """ 
    Valid config for TerraMind Tiny triggering download.
    """
    return {
        "local_model_path": "/tmp/cache/TerraMind_v1_tiny.pt",
        "backbone_name": "terramind_v1_tiny",
        "pretrained": True,
        "modalities": ["S2L2A"]
    }

# Tests
# Model instantiation tests
@patch('aitlas.models.terramind_wrapper.torch.load')
@patch('aitlas.models.terramind_wrapper.os.path.exists')
@patch('aitlas.models.terramind_wrapper.checkpoint_filter_fn')
# Patch the specific backbone class in the wrapper namespace so globals() finds it
@patch('aitlas.models.terramind_wrapper.terramind_v1_tiny', side_effect=MockTerraMindBackbone)
def test_instantiation_terramind_local(
    mock_backbone_cls, mock_filter_fn, mock_os_exists, mock_torch_load, config_terramind_tiny_local
):
    """ 
    Tests loading from a local file with explicit backbone name.
    """
    # Setup
    mock_os_exists.return_value = True # File exists
    mock_torch_load.return_value = {}  # Return empty state dict
    mock_filter_fn.side_effect = lambda ckpt, model: ckpt # Pass through
    
    # Initialize
    model = TerraMind(config_terramind_tiny_local)
    
    # Assertions
    assert isinstance(model, TerraMind)
    assert isinstance(model.backbone, MockTerraMindBackbone)
    
    # Check that modalities were passed to the backbone constructor
    assert model.backbone.modalities == ["S2L2A"]
    
    # Check that the head was replaced with Identity
    assert isinstance(model.backbone.head, nn.Identity)
    
    # Ensure we loaded the state dict from local path
    mock_torch_load.assert_called_with(config_terramind_tiny_local["local_model_path"], weights_only=False)


@patch('aitlas.models.terramind_wrapper.torch.load')
@patch('aitlas.models.terramind_wrapper.os.path.exists')
@patch('aitlas.models.terramind_wrapper.checkpoint_filter_fn')
@patch('aitlas.models.terramind_wrapper.terramind_v1_base', side_effect=MockTerraMindBackbone)
def test_instantiation_terramind_inferred_backbone(
    mock_backbone_cls, mock_filter_fn, mock_os_exists, mock_torch_load, config_terramind_inferred_local
):
    """ 
    Tests that the wrapper correctly infers 'terramind_v1_base' from filename 'TerraMind_v1_base.pt'.
    """
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}
    mock_filter_fn.side_effect = lambda ckpt, model: ckpt

    # Initialize
    model = TerraMind(config_terramind_inferred_local)

    # Assertions
    # It should have deduced the name and instantiated the mock for v1_base
    assert model.backbone_name == "terramind_v1_base"
    assert isinstance(model.backbone, MockTerraMindBackbone)
    mock_backbone_cls.assert_called_once()


@patch('aitlas.models.terramind_wrapper.hf_hub_download')
@patch('aitlas.models.terramind_wrapper.torch.load')
@patch('aitlas.models.terramind_wrapper.os.path.exists')
@patch('aitlas.models.terramind_wrapper.checkpoint_filter_fn')
@patch('aitlas.models.terramind_wrapper.terramind_v1_tiny', side_effect=MockTerraMindBackbone)
def test_instantiation_terramind_download(
    mock_backbone_cls, mock_filter_fn, mock_os_exists, mock_torch_load, mock_hf_download, config_terramind_tiny_download
):
    """ Tests fallback to Hugging Face download when local file is missing. """
    
    # Setup
    mock_os_exists.return_value = False # Trigger download
    expected_download_path = "/tmp/cache/TerraMind_v1_tiny.pt"
    mock_hf_download.return_value = expected_download_path
    mock_torch_load.return_value = {}
    mock_filter_fn.side_effect = lambda ckpt, model: ckpt

    # Initialize
    model = TerraMind(config_terramind_tiny_download)
    
    # Assertions
    expected_repo = 'ibm-esa-geospatial/TerraMind-1.0-tiny'
    expected_filename = 'TerraMind_v1_tiny.pt'
    
    # Check download was called
    mock_hf_download.assert_called_once_with(
        repo_id=expected_repo, 
        filename=expected_filename, 
        local_dir=os.path.dirname(config_terramind_tiny_download["local_model_path"])
    )
    
    # Check we loaded the downloaded file
    mock_torch_load.assert_called_with(expected_download_path, weights_only=False)


# Band selection tests
@patch('aitlas.models.terramind_wrapper.torch.load')
@patch('aitlas.models.terramind_wrapper.os.path.exists')
@patch('aitlas.models.terramind_wrapper.checkpoint_filter_fn')
@patch('aitlas.models.terramind_wrapper.terramind_v1_tiny', side_effect=MockTerraMindBackbone)
@patch('aitlas.models.terramind_wrapper.select_modality_patch_embed_weights')
def test_select_input_bands(
    mock_select_weights, mock_backbone_cls, mock_filter_fn, mock_os_exists, mock_torch_load, config_terramind_tiny_local
):
    """ Tests that select_input_bands correctly calls the utility function. """
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}
    mock_filter_fn.side_effect = lambda ckpt, model: ckpt
    
    model = TerraMind(config_terramind_tiny_local)
    mock_select_weights.return_value = model.backbone
    
    # Define bands to select
    bands_to_select = {"S2L2A": ["B01", "B02", "B03", "B04", "B05"]}
    
    # Call the method
    model.select_input_bands(bands_to_select)
    
    # Assert
    # Verify the external utility function was called with the correct arguments
    mock_select_weights.assert_called_once()
    call_args = mock_select_weights.call_args
    assert call_args.kwargs['model'] == model.backbone
    assert call_args.kwargs['bands'] == bands_to_select
    # Check that PRETRAINED_BANDS was passed (it's imported as default arg in wrapper)
    assert 'pretrained_bands' in call_args.kwargs 


# Error handling tests
def test_raises_error_if_pretrained_is_false_terramind():
    """ Tests that NotImplementedError is raised if pretrained=False. """
    config = {
        "local_model_path": "any",
        "backbone_name": "terramind_v1_tiny",
        "pretrained": False,
        "modalities": ["S2L2A"]
    }
    with pytest.raises(NotImplementedError, match="Loading model without pretrained weights is not supported"):
        TerraMind(config)

@patch('aitlas.models.terramind_wrapper.os.path.exists')
def test_raises_error_invalid_backbone_terramind(mock_os_exists):
    """ Tests ValueError for unknown backbone names during download attempt. """
    mock_os_exists.return_value = False # Trigger download logic
    config = {
        "local_model_path": "any/path/model.pt",
        "backbone_name": "invalid_terramind_model",
        "pretrained": True,
        "modalities": ["S2L2A"]
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        TerraMind(config)


# Forward pass tests
@patch('aitlas.models.terramind_wrapper.torch.load')
@patch('aitlas.models.terramind_wrapper.os.path.exists')
@patch('aitlas.models.terramind_wrapper.checkpoint_filter_fn')
@patch('aitlas.models.terramind_wrapper.terramind_v1_tiny', side_effect=MockTerraMindBackbone)
def test_forward_features_terramind(
    mock_backbone_cls, mock_filter_fn, mock_os_exists, mock_torch_load, 
    config_terramind_tiny_local, dummy_terramind_input
):
    """ Tests forward pass execution. """
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}
    mock_filter_fn.side_effect = lambda ckpt, model: ckpt
    
    model = TerraMind(config_terramind_tiny_local)

    # Pass input
    output = model.forward_features(dummy_terramind_input)
    
    # Assert
    assert isinstance(output, list)
    # Check shape: Mock returns list containing one tensor of shape (B, 196, Embed_Dim)
    assert output[-1].shape == (BATCH_SIZE, 196, EMBED_DIM)