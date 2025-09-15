import pytest
import torch
import torch.nn as nn
import os
from unittest.mock import patch, MagicMock

from aitlas.models import DOFA_v2
from aitlas.models.DOFA.dofa_v2 import OFAViT, vit_base_patch16


def create_dummy_vit_base_state_dict():
    """Creates a state dict from a fresh model instance.
    """

    model = vit_base_patch16()
    return model.state_dict()

@pytest.fixture(scope="session")
def dummy_checkpoint_path(tmp_path_factory):
    """Creates a temporary dummy checkpoint file that perfectly matches the model.
    """
    chkpt_dir = tmp_path_factory.mktemp("checkpoints")
    file_path = chkpt_dir / "DOFA_ViT_base_e100.pth"
    perfect_state_dict = create_dummy_vit_base_state_dict()
    torch.save(perfect_state_dict, file_path)
    return str(file_path)

@pytest.fixture
def base_config(dummy_checkpoint_path):
    """Provides a basic, valid configuration as a dictionary.
    """

    basic_config = {
        "local_model_path": dummy_checkpoint_path,
        "backbone_name": "vit_base_patch16",
        "pretrained": True
    }

    return basic_config

def test_model_instantiation_from_local_path(base_config):
    """Tests the model instantiation for a vali local checkpoint.
    """

    try:
        model = DOFA_v2(base_config)
        
        assert isinstance(model, DOFA_v2)
        assert model.backbone is not None
        assert isinstance(model.backbone, OFAViT)
        assert isinstance(model.backbone.head, nn.Identity)
        
    except Exception as e:
        pytest.fail(f"Model instantiation from local path failed: {e}")

def test_forward_features_pass(base_config):
    """Tests if the loaded model can perform a forward_features pass without errors.
    """

    model = DOFA_v2(base_config)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    wave_list = [0.665, 0.56, 0.49]  # Example wavelengths for RGB channels
    batch_size = dummy_input.shape[0]
    expected_shape = (batch_size, 768)
    
    try:
        output = model.forward_features(dummy_input, wave_list)
        assert isinstance(output, torch.Tensor)
        assert output.shape == expected_shape
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")

# The patch path should point to the file where DOFA_v2 is defined, e.g., 'dofa_v2_wrapper.py'
@patch('aitlas.models.dofa_v2_wrapper.vit_base_patch16')
@patch('aitlas.models.dofa_v2_wrapper.hf_hub_download')
@patch('torch.load')
def test_fallback_to_huggingface_download(mock_torch_load, mock_hf_download, mock_vit_factory, dummy_checkpoint_path):
    """Tests the fallback mechanism, now also mocking the model factory function.
    """

    mock_model_instance = MagicMock()
    mock_vit_factory.return_value = mock_model_instance
    
    mock_hf_download.return_value = dummy_checkpoint_path
    mock_torch_load.return_value = create_dummy_vit_base_state_dict()

    config = {
        "local_model_path": "/path/to/non_existent/model.pth",
        "backbone_name": "vit_base_patch16",
        "pretrained": True
    }

    model = DOFA_v2(config)

    mock_torch_load.assert_called_once()
    mock_hf_download.assert_called_once()
    mock_vit_factory.assert_called_once()
    mock_model_instance.load_state_dict.assert_called_once()

def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if pretrained is set to False.
    """

    config = {
        "pretrained": False,
        "backbone_name": "any_vit_model",
        "local_model_path": "/any/path"
    }

    with pytest.raises(NotImplementedError, match="Loading model without pretrained weights is not supported."):
        model = DOFA_v2(config)

@pytest.mark.parametrize(
    "backbone_name",
    ["unsupported_vit_model", "vit_small_patch16", "vit_huge_patch14"]
)
def test_raises_error_for_invalid_backbones(backbone_name):
    """Tests that ValueErrors are raised for invalid backbones when downloading.
    """
    
    config = {
        "local_model_path": "/path/to/non_existent/model.pth",
        "backbone_name": backbone_name,
        "pretrained": True
    }
    
    with pytest.raises(ValueError, match="Unsupported or missing backbone|No pretrained weights are available for backbone"):
        model = DOFA_v2(config)
        model.load_backbone()