import pytest
import torch
import torch.nn as nn
import os
from unittest.mock import patch, MagicMock

from aitlas.models import SatMAE_plusplus
from aitlas.models.SatMAE_plusplus.models_mae import MaskedAutoencoderViT, mae_vit_large
from aitlas.models.SatMAE_plusplus.models_mae_group_channels import MaskedAutoencoderGroupChannelViT, mae_vit_large_multispectral

# Create state dictionaries
def create_dummy_standard_state_dict():
    """Creates a state dict for the standard SatMAE++ model."""
    model = mae_vit_large()
    model.head = nn.Identity()
    return model.state_dict()

def create_dummy_multispectral_state_dict():
    """Creates a state dict for the multispectral SatMAE++ model."""
    model = mae_vit_large_multispectral()
    model.head = nn.Identity()
    return model.state_dict()

# Pytest fixtures for creating temporary checkpoint files
@pytest.fixture(scope="session")
def dummy_standard_checkpoint(tmp_path_factory):
    """Creates a temporary checkpoint file for the standard model."""
    # Filename must match an entry in the wrapper's backbone_checkpoints dict
    path = tmp_path_factory.mktemp("checkpoints") / "checkpoint_ViT-L_pretrain_fmow_rgb.pth"
    torch.save(create_dummy_standard_state_dict(), path)
    return str(path)

@pytest.fixture(scope="session")
def dummy_multispectral_checkpoint(tmp_path_factory):
    """Creates a temporary checkpoint file for the multispectral model."""
    # Filename must match an entry in the wrapper's backbone_checkpoints dict
    path = tmp_path_factory.mktemp("checkpoints") / "checkpoint_ViT-L_pretrain_fmow_sentinel.pth"
    torch.save(create_dummy_multispectral_state_dict(), path)
    return str(path)

# Pytest fixtures for model configurations
@pytest.fixture
def config_standard(dummy_standard_checkpoint):
    """Provides a valid config for the standard model loading from a local file."""
    return {"local_model_path": dummy_standard_checkpoint, "backbone_name": "mae_vit_large", "pretrained": True}

@pytest.fixture
def config_multispectral(dummy_multispectral_checkpoint):
    """Provides a valid config for the multispectral model loading from a local file."""
    return {"local_model_path": dummy_multispectral_checkpoint, "backbone_name": "mae_vit_large_multispectral", "pretrained": True}

# Test cases
# Model instantiation tests
def test_instantiation_standard(config_standard):
    """Tests instantiation for the standard (RGB) backbone."""
    model = SatMAE_plusplus(config_standard)
    assert isinstance(model, SatMAE_plusplus)
    assert isinstance(model.backbone, MaskedAutoencoderViT)

def test_instantiation_multispectral(config_multispectral):
    """Tests instantiation for the multispectral backbone."""
    model = SatMAE_plusplus(config_multispectral)
    assert isinstance(model, SatMAE_plusplus)
    assert isinstance(model.backbone, MaskedAutoencoderGroupChannelViT)

# Forward pass tests
def test_forward_pass_standard(config_standard):
    """Tests the forward_features pass for the standard model."""
    model = SatMAE_plusplus(config_standard)
    dummy_input = torch.randn(2, 3, 224, 224)  # (N, C, H, W)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 1024)  # ViT-Large embedding dimension is 1024

def test_forward_pass_multispectral(config_multispectral):
    """Tests the forward_features pass for the multispectral model."""
    model = SatMAE_plusplus(config_multispectral)
    dummy_input = torch.randn(2, 10, 96, 96)  # Expects a smaller image as input (H, W) = (96, 96)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 1024)

# Download and error handling tests
@patch('aitlas.models.satmae_plusplus_wrapper.mae_vit_large_multispectral')
@patch('aitlas.models.satmae_plusplus_wrapper.hf_hub_download')
@patch('torch.load')
def test_fallback_to_hf_hub_download(mock_torch_load, mock_hf_download, mock_model_factory):
    """Tests the fallback to Huggingface hub download if local_model_path is invalid."""
    mock_model_instance = MagicMock()
    mock_model_factory.return_value = mock_model_instance
    mock_torch_load.return_value = create_dummy_multispectral_state_dict()
    
    non_existent_path = "/path/to/non_existent/model.pth"
    mock_hf_download.return_value = non_existent_path # Simulate download returning the path

    config = {
        "local_model_path": non_existent_path,
        "backbone_name": "mae_vit_large_multispectral",
        "pretrained": True,
    }
    
    model = SatMAE_plusplus(config)

    mock_hf_download.assert_called_once_with(
        repo_id='mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel', 
        filename='checkpoint_ViT-L_pretrain_fmow_sentinel.pth',
        local_dir=os.path.dirname(non_existent_path)
    )
    
    mock_torch_load.assert_called_once_with(non_existent_path, weights_only=False)
    mock_model_factory.assert_called_once()
    mock_model_instance.load_state_dict.assert_called_once()


def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if config has pretrained=False."""
    config = {
        "pretrained": False,
        "backbone_name": "any_name",
        "local_model_path": "any_path"
    }
    with pytest.raises(NotImplementedError, match="Loading model without pretrained weights is not supported."):
        SatMAE_plusplus(config)


@pytest.mark.parametrize(
    "backbone_name",
    ["unsupported_vit_model", "mae_vit_base"] # mae_vit_base is a valid model but not listed in the checkpoints dict
)
def test_raises_error_for_invalid_backbones_on_download(backbone_name):
    """Tests that a ValueError is raised for invalid backbones when attempting to download."""
    config = {
        "local_model_path": "/path/to/non_existent/model.pth", # Force download path
        "backbone_name": backbone_name,
        "pretrained": True
    }
    # The error message can be one of two, so we use a regex to catch either
    with pytest.raises(ValueError, match="Unsupported or missing backbone|No pretrained weights are available for backbone"):
        SatMAE_plusplus(config)