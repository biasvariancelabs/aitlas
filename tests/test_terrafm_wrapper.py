import pytest
import torch
import torch.nn as nn
import os
from unittest.mock import patch, MagicMock
from aitlas.models import TerraFM

# Constants for testing
EMBED_DIM = 768  # TerraFM Base dimension
BATCH_SIZE = 2
CHANNELS = 6  # Example multispectral input
HEIGHT = 224
WIDTH = 224

# Mock Backbone of the TerraFM model
class MockTerraFMBackbone(nn.Module):
    """
    A mock class to simulate the actual TerraFM ViT backbone.
    """
    def __init__(self):
        super().__init__()
        # Simulate attributes required by the wrapper
        self.head = nn.Linear(EMBED_DIM, 10)
        self.patch_embed = nn.Identity() # Just to have some attribute

    def forward_features(self, x):
        """
        Simulates the forward_features logic of TerraFM backbone.
        """
        # Return a dummy embedding of shape (B, Embed_Dim)
        batch_size = x.shape[0]
        return torch.randn(batch_size, EMBED_DIM)

    def load_state_dict(self, state_dict, strict=True):
        """ Mock load_state_dict to avoid actual weight loading errors. """
        pass

# Fixtures
@pytest.fixture
def dummy_terrafm_input():
    """ Creates a dummy input tensor. """
    return torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)

@pytest.fixture
def config_terrafm_base_local():
    """ 
    Valid config for TerraFM Base loading locally. 
    Note: filename must match the one in wrapper's dictionary for inference logic.
    """
    return {
        "local_model_path": "/tmp/mock_path/TerraFM-B.pth",
        "backbone_name": "terrafm_base",
        "pretrained": True
    }

@pytest.fixture
def config_terrafm_base_download():
    """ 
    Valid config for TerraFM Base triggering download.
    """
    return {
        "local_model_path": "/tmp/cache/TerraFM-B.pth",
        "backbone_name": "terrafm_base",
        "pretrained": True
    }

# Tests
# Model instantiation tests
@patch('aitlas.models.terrafm_wrapper.torch.load')
@patch('aitlas.models.terrafm_wrapper.os.path.exists')
# We must patch 'terrafm_base' inside the wrapper module so globals()[name] finds the mock
@patch('aitlas.models.terrafm_wrapper.terrafm_base', side_effect=MockTerraFMBackbone)
def test_instantiation_terrafm_local(
    mock_backbone_cls, mock_os_exists, mock_torch_load, config_terrafm_base_local
):
    """ 
    Tests loading from a local file. 
    Tests the logic where the wrapper infers 'terrafm_base' from the filename 'TerraFM-B.pth'.
    """
    
    # Setup
    mock_os_exists.return_value = True # File exists
    mock_torch_load.return_value = {}  # Return empty state dict
    
    # Initialize
    model = TerraFM(config_terrafm_base_local)
    
    # Assertions
    assert isinstance(model, TerraFM)
    assert isinstance(model.backbone, MockTerraFMBackbone)
    
    # Check that the head was replaced with Identity (wrapper logic)
    assert isinstance(model.backbone.head, nn.Identity)
    
    # Ensure we loaded the state dict from local path
    mock_torch_load.assert_called_with(config_terrafm_base_local["local_model_path"], weights_only=False)

@patch('aitlas.models.terrafm_wrapper.hf_hub_download')
@patch('aitlas.models.terrafm_wrapper.torch.load')
@patch('aitlas.models.terrafm_wrapper.os.path.exists')
@patch('aitlas.models.terrafm_wrapper.terrafm_base', side_effect=MockTerraFMBackbone)
def test_instantiation_terrafm_download(
    mock_backbone_cls, mock_os_exists, mock_torch_load, mock_hf_download, config_terrafm_base_download
):
    """ Tests fallback to Hugging Face download when local file is missing. """
    
    # Setup
    mock_os_exists.return_value = False # Trigger download
    expected_download_path = "/tmp/cache/TerraFM-B.pth"
    mock_hf_download.return_value = expected_download_path
    mock_torch_load.return_value = {}

    # Initialize
    model = TerraFM(config_terrafm_base_download)
    
    # Assertions
    expected_repo = 'MBZUAI/TerraFM'
    expected_filename = 'TerraFM-B.pth'
    
    # Check download was called
    mock_hf_download.assert_called_once_with(
        repo_id=expected_repo, 
        filename=expected_filename, 
        local_dir=os.path.dirname(config_terrafm_base_download["local_model_path"])
    )
    
    # Check we loaded the downloaded file
    mock_torch_load.assert_called_with(expected_download_path, weights_only=False)

# Error handling tests
def test_raises_error_if_pretrained_is_false_terrafm():
    """ Tests that NotImplementedError is raised if pretrained=False. """
    config = {
        "local_model_path": "any",
        "backbone_name": "terrafm_base",
        "pretrained": False
    }
    with pytest.raises(NotImplementedError, match="Loading model without pretrained weights is not supported"):
        TerraFM(config)

@patch('aitlas.models.terrafm_wrapper.os.path.exists')
def test_raises_error_invalid_backbone_terrafm(mock_os_exists):
    """ Tests ValueError for unknown backbone names during download attempt. """
    mock_os_exists.return_value = False # Trigger download logic
    config = {
        "local_model_path": "any/path/model.pth",
        "backbone_name": "invalid_terrafm_model",
        "pretrained": True
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        TerraFM(config)

@patch('aitlas.models.terrafm_wrapper.os.path.exists')
def test_raises_error_no_weights_for_backbone_terrafm(mock_os_exists):
    """ 
    Tests ValueError if backbone exists in keys but is set to None (e.g. terrafm_large). 
    """
    mock_os_exists.return_value = False
    config = {
        "local_model_path": "any/path/model.pth",
        "backbone_name": "terrafm_large", # This is defined as None in wrapper code
        "pretrained": True
    }
    with pytest.raises(ValueError, match="No pretrained weights are available for backbone"):
        TerraFM(config)

# Forward pass tests
@patch('aitlas.models.terrafm_wrapper.torch.load')
@patch('aitlas.models.terrafm_wrapper.os.path.exists')
@patch('aitlas.models.terrafm_wrapper.terrafm_base', side_effect=MockTerraFMBackbone)
def test_forward_pass_terrafm(
    mock_backbone_cls, mock_os_exists, mock_torch_load, 
    config_terrafm_base_local, dummy_terrafm_input
):
    """ Tests forward pass execution. """
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}
    
    model = TerraFM(config_terrafm_base_local)

    # Pass input
    output = model.forward_features(dummy_terrafm_input)
    
    # Assert
    assert isinstance(output, torch.Tensor)
    # Check shape: (Batch_Size, Embed_Dim)
    assert output.shape == (BATCH_SIZE, EMBED_DIM)