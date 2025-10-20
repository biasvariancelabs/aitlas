import pytest
import torch
import torch.nn as nn
import os
from unittest.mock import patch, MagicMock

from aitlas.models import GASSL
from aitlas.models.GASSL import MoCo, gassl_moco_resnet50, MoCo_geo, gassl_moco_geo_resnet50

# Helper functions to create realistic dummy raw checkpoints
def create_dummy_raw_moco_dict():
    """Creates a raw state dict for the MoCo model with an MLP head."""
    model = gassl_moco_resnet50()
    state_dict = {"module." + k: v for k, v in model.state_dict().items()}
    return {"state_dict": state_dict, "epoch": 199}

def create_dummy_raw_moco_geo_dict():
    """Creates a raw state dict for the MoCo_geo model with an MLP head."""
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

def test_forward_pass_backbone_default_moco(config_moco):
    """Tests the default forward pass for the 2048-dim backbone embedding."""
    model = GASSL(config_moco)
    dummy_input = torch.randn(2, 3, 224, 224)
    # The default call should now return the 2048-dim backbone embedding
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 2048)

def test_forward_pass_mlp_embedding_moco(config_moco):
    """Tests the forward pass for the 128-dim MLP embedding."""
    model = GASSL(config_moco)
    dummy_input = torch.randn(2, 3, 224, 224)
    # To get the 128-dim embedding, we must now explicitly ask for it
    output = model.forward_features(dummy_input, return_all_embeddings=False)
    assert output.shape == (2, 128)

def test_forward_pass_restores_fc_layer(config_moco):
    """Tests that the fc layer is restored after extracting backbone embeddings."""
    model = GASSL(config_moco)
    dummy_input = torch.randn(2, 3, 224, 224)
    original_fc = model.backbone.encoder_q.fc
    
    # Call the method that modifies the fc layer (now the default behavior)
    model.forward_features(dummy_input)
    
    # Check that the fc layer is the same object as before
    assert model.backbone.encoder_q.fc is original_fc

# Download and error handling tests
@patch('aitlas.models.gassl_wrapper.gassl_moco_resnet50')
@patch('aitlas.models.gassl_wrapper.GASSL._download_from_zenodo')
@patch('torch.load')
def test_fallback_to_zenodo_download(mock_torch_load, mock_download, mock_model_factory):
    """Tests the fallback to Zenodo download if local_model_path is invalid."""
    mock_model_instance = MagicMock()
    mock_model_factory.return_value = mock_model_instance
    mock_download.return_value = None 
    mock_torch_load.return_value = create_dummy_raw_moco_dict()

    non_existent_path = "/path/to/non_existent/model.pth"
    config = {
        "local_model_path": non_existent_path,
        "backbone_name": "gassl_moco_resnet50",
        "pretrained": True
    }
    GASSL(config)

    mock_download.assert_called_once()
    mock_torch_load.assert_called_once_with(non_existent_path, map_location='cpu')
    mock_model_instance.load_state_dict.assert_called_once()

def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if config has pretrained=False."""
    config = {
        "local_model_path": "any_path",
        'backbone_name': "any_model",
        "pretrained": False
    } 
    with pytest.raises(NotImplementedError):
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