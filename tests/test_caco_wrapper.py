import pytest
import torch
import torch.nn as nn
import os
from unittest.mock import patch, MagicMock

from aitlas.models import CACo 
from aitlas.models.CACo import MoCoV2CACoModule, caco_resnet18, caco_resnet50

# Helper functions and fixtures
def create_dummy_raw_caco_dict(model_name='caco_resnet18'):
    """
    Creates a realistic dummy raw state_dict for a CACo encoder.
    The CACo checkpoint is just the state_dict of the ResNet encoder,
    not a wrapper dict like SeCo.
    """
    if model_name == 'caco_resnet18':
        model = caco_resnet18()
    elif model_name == 'caco_resnet50':
        model = caco_resnet50()
    else:
        raise ValueError("Unsupported model name for dummy dict creation.")
        
    # The checkpoint is the state dict of the encoder_q
    state_dict = model.encoder_q.state_dict()
    return state_dict

@pytest.fixture(scope="session")
def dummy_caco_resnet18_checkpoint(tmp_path_factory):
    """
    Creates a temporary .pth file for the CACo ResNet-18 model.
    The filename matches the one in the wrapper's config.
    """
    path = tmp_path_factory.mktemp("checkpoints") / "resnet18_caco_geo_100k_1000.pth"
    torch.save(create_dummy_raw_caco_dict('caco_resnet18'), path)
    return str(path)

@pytest.fixture(scope="session")
def dummy_caco_resnet50_checkpoint(tmp_path_factory):
    """
    Creates a temporary .pth file for the CACo ResNet-50 model.
    The filename matches the one in the wrapper's config.
    """
    path = tmp_path_factory.mktemp("checkpoints") / "resnet50_caco_geo_100k_1000.pth"
    torch.save(create_dummy_raw_caco_dict('caco_resnet50'), path)
    return str(path)

@pytest.fixture
def config_caco_resnet18(dummy_caco_resnet18_checkpoint):
    """Provides a valid config for the CACo ResNet-18 model with a local file."""
    return {
        "local_model_path": dummy_caco_resnet18_checkpoint,
        "backbone_name": "caco_resnet18",  # Included for completeness
        "pretrained": True
    }

@pytest.fixture
def config_caco_resnet50(dummy_caco_resnet50_checkpoint):
    """Provides a valid config for the CACo ResNet-50 model with a local file."""
    return {
        "local_model_path": dummy_caco_resnet50_checkpoint,
        "backbone_name": "caco_resnet50", # Included for completeness
        "pretrained": True
    }

# Tests
# Model instantiation tests
def test_instantiation_caco_resnet18(config_caco_resnet18):
    """Tests correct instantiation and backbone type for CACo ResNet-18."""
    model = CACo(config_caco_resnet18)
    assert isinstance(model, CACo)
    assert isinstance(model.backbone, MoCoV2CACoModule)

def test_instantiation_caco_resnet50(config_caco_resnet50):
    """Tests correct instantiation and backbone type for CACo ResNet-50."""
    model = CACo(config_caco_resnet50)
    assert isinstance(model, CACo)
    assert isinstance(model.backbone, MoCoV2CACoModule)


# Forward pass tests
def test_forward_pass_resnet18(config_caco_resnet18):
    """Tests the forward pass for the 512-dim embedding of ResNet-18."""
    model = CACo(config_caco_resnet18)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 512)

def test_forward_pass_resnet50(config_caco_resnet50):
    """Tests the forward pass for the 2048-dim embedding of ResNet-50."""
    model = CACo(config_caco_resnet50)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 2048)
    
def test_forward_pass_raises_error_if_backbone_not_loaded(config_caco_resnet18):
    """Tests that a RuntimeError is raised if forward is called before load."""
    model = CACo(config_caco_resnet18)
    
    # Manually remove the backbone after initialization
    model.backbone = None 
    
    dummy_input = torch.randn(2, 3, 224, 224)
    with pytest.raises(RuntimeError, match="The backbone model has not been loaded"):
        model.forward_features(dummy_input)

# Download and error handling tests
@patch('aitlas.models.caco_wrapper.caco_resnet50')
@patch('aitlas.models.caco_wrapper.CACo._download_from_cornell')
@patch('torch.load')
def test_fallback_to_cornell_download(mock_torch_load, mock_download, mock_model_factory):
    """Tests the fallback to Cornell download if local_model_path does not exist."""
    
    # Setup mocks
    mock_model_instance = MagicMock(spec=MoCoV2CACoModule)
    mock_encoder_q = MagicMock()
    mock_encoder_k = MagicMock()
    mock_model_instance.encoder_q = mock_encoder_q
    mock_model_instance.encoder_k = mock_encoder_k
    mock_model_factory.return_value = mock_model_instance
    
    mock_download.return_value = None 
    
    dummy_dict = create_dummy_raw_caco_dict('caco_resnet50')
    mock_torch_load.return_value = dummy_dict

    # Config
    non_existent_path = "/path/to/non_existent/model.pth"
    config = {
        "local_model_path": non_existent_path,
        "backbone_name": "caco_resnet50", # This is crucial for the download path
        "pretrained": True
    }

    CACo(config)

    # Assertions
    expected_filename = 'resnet50_caco_geo_100k_1000.pth' 
    mock_download.assert_called_once_with(
        checkpoint_name=expected_filename, 
        local_model_path=non_existent_path
    )
    
    mock_torch_load.assert_called_once_with(non_existent_path, map_location='cpu', weights_only=False)

    mock_encoder_q.load_state_dict.assert_called_once_with(dummy_dict, strict=True)
    mock_encoder_k.load_state_dict.assert_called_once_with(dummy_dict, strict=True)

def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if config has pretrained=False."""
    config = {
        "local_model_path": "any_path",
        'backbone_name': "any_model",
        "pretrained": False
    } 
    with pytest.raises(NotImplementedError, match="Loading model without pretrained weights is not supported."):
        CACo(config)

@pytest.mark.parametrize(
    "backbone_name",
    ["unsupported_caco_model", "caco_resnet34", "resnet18"] # Test invalid names
)
def test_raises_error_for_invalid_backbones_on_download(backbone_name):
    """Tests that a ValueError is raised for invalid backbones when downloading."""
    config = {
        "local_model_path": "/path/to/non_existent/model.pth", # Force download path
        "backbone_name": backbone_name,
        "pretrained": True
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        CACo(config)