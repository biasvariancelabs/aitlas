import pytest
import torch
import torch.nn as nn
import os
from unittest.mock import patch, MagicMock

from aitlas.models import GASSL
from aitlas.models.GASSL import MoCo, gassl_moco_resnet50, MoCo_geo, gassl_moco_geo_resnet50

# Create dummy raw checkpoint dictionaries
def create_dummy_raw_moco_dict():
    """Creates a raw state dict for the standard MoCo model, mimicking the original file structure."""
    model = gassl_moco_resnet50()
    # Add the 'module.' prefix to simulate DataParallel saving
    state_dict = {"module." + k: v for k, v in model.state_dict().items()}
    # Wrap it in the outer dictionary with metadata
    return {"state_dict": state_dict, "epoch": 199}

def create_dummy_raw_moco_geo_dict():
    """Creates a raw state dict for the MoCo_geo model."""
    model = gassl_moco_geo_resnet50()
    state_dict = {"module." + k: v for k, v in model.state_dict().items()}
    return {"state_dict": state_dict, "epoch": 199}

# Pytest fixtures for temporary checkpoint files
@pytest.fixture(scope="session")
def dummy_moco_checkpoint(tmp_path_factory):
    """Creates a temporary .pth.tar file for the standard MoCo model."""
    path = tmp_path_factory.mktemp("checkpoints") / "moco.pth.tar"
    torch.save(create_dummy_raw_moco_dict(), path)
    return str(path)

@pytest.fixture(scope="session")
def dummy_moco_geo_checkpoint(tmp_path_factory):
    """Creates a temporary .pth.tar file for the MoCo_geo model."""
    path = tmp_path_factory.mktemp("checkpoints") / "moco_geo+tp.pth.tar"
    torch.save(create_dummy_raw_moco_geo_dict(), path)
    return str(path)

# Configuration fixtures
@pytest.fixture
def config_moco(dummy_moco_checkpoint):
    """Provides a valid config for the standard MoCo model."""
    return {"local_model_path": dummy_moco_checkpoint, "backbone_name": "gassl_moco_resnet50", "pretrained": True}

@pytest.fixture
def config_gassl_full(dummy_moco_geo_checkpoint):
    """Provides a valid config for the full GASSL (MoCo_geo) model."""
    return {"local_model_path": dummy_moco_geo_checkpoint, "backbone_name": "gassl_moco_geo_resnet50", "pretrained": True}

# Tests
# Model instantiation tests
def test_instantiation_moco(config_moco):
    """Tests correct instantiation and backbone type for the standard MoCo model."""
    model = GASSL(config_moco)
    assert isinstance(model, GASSL) and isinstance(model.backbone, MoCo)

def test_instantiation_moco_geo(config_gassl_full):
    """Tests correct instantiation and backbone type for the MoCo_geo model."""
    model = GASSL(config_gassl_full)
    assert isinstance(model, GASSL) and isinstance(model.backbone, MoCo_geo)

# Forward pass tests
def test_forward_pass_moco(config_moco):
    """Tests the forward_features pass for the standard MoCo model."""
    model = GASSL(config_moco)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward_features(dummy_input)
    # ResNet-50 feature dimension before the classifier has a size of 128
    assert output.shape == (2, 128)

def test_forward_pass_moco_geo(config_gassl_full):
    """Tests the forward_features pass for the MoCo_geo model."""
    model = GASSL(config_gassl_full)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 128)

# Downlaod and error handling tests
@patch('aitlas.models.gassl_wrapper.gassl_moco_resnet50')
@patch('aitlas.models.gassl_wrapper.GASSL._download_from_zenodo')
@patch('torch.load')
def test_fallback_to_zenodo_download(mock_torch_load, mock_download, mock_model_factory):
    """Tests the fallback to Zenodo download if local_model_path is invalid."""
    mock_model_instance = MagicMock()
    mock_model_factory.return_value = mock_model_instance
    mock_download.return_value = None 
    # torch.load should return a raw checkpoint dictionary
    mock_torch_load.return_value = create_dummy_raw_moco_dict()

    non_existent_path = "/path/to/non_existent/model.pth"
    config = {
        "local_model_path": non_existent_path,
        "backbone_name": "gassl_moco_resnet50",
        "pretrained": True,
    }
    model = GASSL(config)

    # Verify that download was called with the correct parameters
    mock_download.assert_called_once_with(
        record_id='7379715', 
        checkpoint_name='moco.pth.tar', # Defaults to the first in the list
        local_model_path=non_existent_path
    )
    
    # Verify torch.load and model factory were called
    mock_torch_load.assert_called_once_with(non_existent_path, map_location='cpu')
    mock_model_factory.assert_called_once()

    # Verify that load_state_dict was called on the model instance
    mock_model_instance.load_state_dict.assert_called_once()

def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if config has pretrained=False."""
    config = {
        "pretrained": False,
        "backbone_name": "any_name",
        "local_model_path": "any_path"
    }
    with pytest.raises(NotImplementedError, match="Loading model without pretrained weights is not supported."):
        GASSL(config)

@pytest.mark.parametrize(
    "backbone_name",
    ["unsupported_gassl_model", "gassl_moco_resnet18"]
)
def test_raises_error_for_invalid_backbones_on_download(backbone_name):
    """Tests that a ValueError is raised for invalid backbones when downloading."""
    config = {
        "local_model_path": "/path/to/non_existent/model.pth",
        "backbone_name": backbone_name,
        "pretrained": True
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        GASSL(config)