import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitlas.models import SatMAE
from aitlas.models.SatMAE import (
    MaskedAutoencoderGroupChannelViT,
    MaskedAutoencoderTemporalViT,
    MaskedAutoencoderViT,
    satmae_vit_large,
    satmae_vit_large_multispectral,
    satmae_vit_large_temporal,
)


# Create state dictionaries
def create_dummy_standard_state_dict():
    """Creates a state dict for the standard SatMAE model."""
    model = satmae_vit_large()
    model.head = nn.Identity()
    return model.state_dict()


def create_dummy_multispectral_state_dict():
    """Creates a state dict for the multispectral SatMAE model."""
    model = satmae_vit_large_multispectral()
    model.head = nn.Identity()
    return model.state_dict()


def create_dummy_temporal_state_dict():
    """Creates a state dict for the temporal SatMAE model."""
    model = satmae_vit_large_temporal()
    model.head = nn.Identity()
    return model.state_dict()


# Pytest fixtures
@pytest.fixture(scope="session")
def dummy_standard_checkpoint(tmp_path_factory):
    """Creates a temporary checkpoint file for the standard model."""
    path = tmp_path_factory.mktemp("checkpoints") / "fmow_pretrain.pth"
    torch.save(create_dummy_standard_state_dict(), path)
    return str(path)


@pytest.fixture(scope="session")
def dummy_multispectral_checkpoint(tmp_path_factory):
    """Creates a temporary checkpoint file for the multispectral model."""
    path = tmp_path_factory.mktemp("checkpoints") / "pretrain-vit-large-e199.pth"
    torch.save(create_dummy_multispectral_state_dict(), path)
    return str(path)


@pytest.fixture(scope="session")
def dummy_temporal_checkpoint(tmp_path_factory):
    """Creates a temporary checkpoint file for the temporal model."""
    path = tmp_path_factory.mktemp("checkpoints") / "pretrain_fmow_temporal.pth"
    torch.save(create_dummy_temporal_state_dict(), path)
    return str(path)


# Configuration fixtures
@pytest.fixture
def config_standard(dummy_standard_checkpoint):
    """Provides a valid config for the standard model."""
    return {
        "local_model_path": dummy_standard_checkpoint,
        "backbone_name": "satmae_vit_large",
        "pretrained": True,
    }


@pytest.fixture
def config_multispectral(dummy_multispectral_checkpoint):
    """Provides a valid config for the multispectral model."""
    return {
        "local_model_path": dummy_multispectral_checkpoint,
        "backbone_name": "satmae_vit_large_multispectral",
        "pretrained": True,
    }


@pytest.fixture
def config_temporal(dummy_temporal_checkpoint):
    """Provides a valid config for the temporal model."""
    return {
        "local_model_path": dummy_temporal_checkpoint,
        "backbone_name": "satmae_vit_large_temporal",
        "pretrained": True,
    }


# Test cases
# Model instantiation tests
def test_instantiation_standard(config_standard):
    """Tests instantiation for the standard (RGB) backbone."""
    model = SatMAE(config_standard)
    assert isinstance(model, SatMAE) and isinstance(
        model.backbone, MaskedAutoencoderViT
    )


def test_instantiation_multispectral(config_multispectral):
    """Tests instantiation for the multispectral backbone."""
    model = SatMAE(config_multispectral)
    assert isinstance(model, SatMAE) and isinstance(
        model.backbone, MaskedAutoencoderGroupChannelViT
    )


def test_instantiation_temporal(config_temporal):
    """Tests that the temporal backbone can be instantiated (error is raised on forward pass)."""
    model = SatMAE(config_temporal)
    assert isinstance(model, SatMAE) and isinstance(
        model.backbone, MaskedAutoencoderTemporalViT
    )


# Forward pass tests
def test_forward_pass_standard(config_standard):
    """Tests the forward_features pass for the standard model."""
    model = SatMAE(config_standard)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 1024)


def test_forward_pass_multispectral(config_multispectral):
    """Tests the forward_features pass for the multispectral model."""
    model = SatMAE(config_multispectral)
    dummy_input = torch.randn(2, 10, 224, 224)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 1024)


def test_forward_features_raises_error_for_temporal_model(config_temporal):
    """Tests that a NotImplementedError is raised when calling forward_features on a temporal model."""
    model = SatMAE(config_temporal)
    dummy_input = torch.randn(1, 3, 3, 224, 224)

    expected_error_msg = "Loading a temporal SatMAE model is not supported due to a bug in the source code for encoding timestamps."

    with pytest.raises(NotImplementedError, match=expected_error_msg):
        model.forward_features(dummy_input)


# Download and error handling tests
@patch("aitlas.models.satmae_wrapper.satmae_vit_large_multispectral")
@patch("aitlas.models.satmae_wrapper.SatMAE._download_from_zenodo")
@patch("torch.load")
def test_fallback_to_zenodo_download(
    mock_torch_load, mock_download, mock_model_factory, dummy_multispectral_checkpoint
):
    """Tests the fallback to Zenodo download if local_model_path is invalid."""
    mock_model_instance = MagicMock()
    mock_model_factory.return_value = mock_model_instance
    mock_download.return_value = None
    mock_torch_load.return_value = create_dummy_multispectral_state_dict()

    non_existent_path = "/path/to/non_existent/model.pth"

    config = {
        "local_model_path": non_existent_path,
        "backbone_name": "satmae_vit_large_multispectral",
        "pretrained": True,
    }
    model = SatMAE(config)

    mock_download.assert_called_once_with(
        record_id="7338613",
        checkpoint_name="pretrain-vit-large-e199.pth",
        local_model_path=non_existent_path,
    )

    mock_torch_load.assert_called_once_with(non_existent_path, weights_only=False)

    mock_model_factory.assert_called_once()
    mock_model_instance.load_state_dict.assert_called_once()


def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if config has pretrained=False."""
    config = {
        "pretrained": False,
        "backbone_name": "any_name",
        "local_model_path": "any_path",
    }
    with pytest.raises(
        NotImplementedError,
        match="Loading model without pretrained weights is not supported.",
    ):
        SatMAE(config)


@pytest.mark.parametrize("backbone_name", ["unsupported_vit_model", "mae_vit_base"])
def test_raises_error_for_invalid_backbones_on_download(backbone_name):
    """Tests that a ValueError is raised for invalid backbones when downloading."""
    config = {
        "local_model_path": "/path/to/non_existent/model.pth",
        "backbone_name": backbone_name,
        "pretrained": True,
    }
    with pytest.raises(
        ValueError,
        match="Unsupported or missing backbone|No pretrained weights are available for backbone",
    ):
        SatMAE(config)
