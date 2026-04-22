import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitlas.models import ScaleMAE
from aitlas.models.ScaleMAE.scale_mae import (
    MaskedAutoencoderViT,
    scalemae_vit_large_patch16,
)


def create_dummy_vit_large_state_dict():
    """Creates a state dict from a fresh model instance."""

    model = scalemae_vit_large_patch16(fixed_output_size=224)
    return model.state_dict()


@pytest.fixture(scope="session")
def dummy_checkpoint_path(tmp_path_factory):
    """Creates a temporary dummy checkpoint file that perfectly matches the model."""

    chkpt_dir = tmp_path_factory.mktemp("checkpoints")
    file_path = chkpt_dir / "scalemae-vitlarge-800.pth"
    perfect_state_dict = create_dummy_vit_large_state_dict()
    torch.save(perfect_state_dict, file_path)
    return str(file_path)


@pytest.fixture
def base_config(dummy_checkpoint_path):
    """Provides a basic, valid configuration as a dictionary."""

    basic_config = {
        "local_model_path": dummy_checkpoint_path,
        "backbone_name": "scalemae_vit_large_patch16",
        "pretrained": True,
    }

    return basic_config


def test_model_instantiation_from_local_path(base_config):
    """Tests the model instantiation for a valid local checkpoint."""

    try:
        model = ScaleMAE(base_config)

        assert isinstance(model, ScaleMAE)
        assert model.backbone is not None
        assert isinstance(model.backbone, MaskedAutoencoderViT)

    except Exception as e:
        pytest.fail(f"Model instantiation from local path failed: {e}")


# In test_scale_mae_wrapper.py


def test_forward_features_pass(base_config):
    """Tests if the loaded model can perform a forward_features pass without errors."""

    model = ScaleMAE(base_config)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Create a dummy ground sample distance (GSD) tensor (e.g., representing 1.0 m/pixel)
    dummy_res = torch.tensor([1.0])
    expected_shape = (1, 1024)  # For a large model

    try:
        # Call the method with both arguments
        output = model.forward_features(dummy_input, input_res=dummy_res)
        print(output.shape)
        print(output)
        assert output.shape == expected_shape

    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")


# The patch path should point to the file where Scale-MAE is defined, e.g., 'scale_mae_wrapper.py'
@patch("aitlas.models.scale_mae_wrapper.scalemae_vit_large_patch16")
@patch("aitlas.models.scale_mae_wrapper.hf_hub_download")
@patch("torch.load")
def test_fallback_to_huggingface_download(
    mock_torch_load, mock_hf_download, mock_vit_factory, dummy_checkpoint_path
):
    """Tests the fallback mechanism, now also mocking the model factory function."""

    mock_model_instance = MagicMock()
    mock_vit_factory.return_value = mock_model_instance

    mock_hf_download.return_value = dummy_checkpoint_path
    mock_torch_load.return_value = create_dummy_vit_large_state_dict()

    config = {
        "local_model_path": "/path/to/non_existent/model.pth",
        "backbone_name": "scalemae_vit_large_patch16",
        "pretrained": True,
    }

    model = ScaleMAE(config)

    mock_torch_load.assert_called_once()
    mock_hf_download.assert_called_once()
    mock_vit_factory.assert_called_once()
    mock_model_instance.load_state_dict.assert_called_once()


def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if pretrained is set to False."""

    config = {
        "pretrained": False,
        "backbone_name": "any_vit_model",
        "local_model_path": "/any/path",
    }

    with pytest.raises(
        NotImplementedError,
        match="Loading model without pretrained weights is not supported.",
    ):
        model = ScaleMAE(config)


@pytest.mark.parametrize(
    "backbone_name", ["unsupported_vit_model", "vit_small_patch16", "vit_huge_patch14"]
)
def test_raises_error_for_invalid_backbones(backbone_name):
    """Tests that ValueErrors are raised for invalid backbones when downloading."""

    config = {
        "local_model_path": "/path/to/non_existent/model.pth",
        "backbone_name": backbone_name,
        "pretrained": True,
    }

    with pytest.raises(
        ValueError,
        match="Unsupported or missing backbone|No pretrained weights are available for backbone",
    ):
        model = ScaleMAE(config)
        model.load_backbone()
