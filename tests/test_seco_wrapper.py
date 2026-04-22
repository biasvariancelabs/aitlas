import os
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitlas.models import SeCo
from aitlas.models.SeCo import MoCoV2Module, seco_resnet18, seco_resnet50


# Helper functions and fixtures
def create_dummy_raw_seco_dict(model_name="seco_resnet18"):
    """Creates a realistic dummy raw checkpoint for a SeCo model."""
    if model_name == "seco_resnet18":
        model = seco_resnet18()
    elif model_name == "seco_resnet50":
        model = seco_resnet50()
    else:
        raise ValueError("Unsupported model name for dummy dict creation.")

    state_dict = model.state_dict()
    return {"state_dict": state_dict, "epoch": 99}


@pytest.fixture(scope="session")
def dummy_seco_resnet18_checkpoint(tmp_path_factory):
    """Creates a temporary .ckpt file for the SeCo ResNet-18 model."""
    path = tmp_path_factory.mktemp("checkpoints") / "seco_resnet18_100k.ckpt"
    torch.save(create_dummy_raw_seco_dict("seco_resnet18"), path)
    return str(path)


@pytest.fixture(scope="session")
def dummy_seco_resnet50_checkpoint(tmp_path_factory):
    """Creates a temporary .ckpt file for the SeCo ResNet-50 model."""
    path = tmp_path_factory.mktemp("checkpoints") / "seco_resnet50_100k.ckpt"
    torch.save(create_dummy_raw_seco_dict("seco_resnet50"), path)
    return str(path)


@pytest.fixture
def config_seco_resnet18(dummy_seco_resnet18_checkpoint):
    """Provides a valid config for the SeCo ResNet-18 model."""
    return {
        "local_model_path": dummy_seco_resnet18_checkpoint,
        "backbone_name": "seco_resnet18",
        "pretrained": True,
    }


@pytest.fixture
def config_seco_resnet50(dummy_seco_resnet50_checkpoint):
    """Provides a valid config for the SeCo ResNet-50 model."""
    return {
        "local_model_path": dummy_seco_resnet50_checkpoint,
        "backbone_name": "seco_resnet50",
        "pretrained": True,
    }


# Tests
# Model instantiation tests
def test_instantiation_seco_resnet18(config_seco_resnet18):
    """Tests correct instantiation and backbone type for SeCo ResNet-18."""
    model = SeCo(config_seco_resnet18)
    assert isinstance(model, SeCo)
    assert isinstance(model.backbone, MoCoV2Module)


def test_instantiation_seco_resnet50(config_seco_resnet50):
    """Tests correct instantiation and backbone type for SeCo ResNet-50."""
    model = SeCo(config_seco_resnet50)
    assert isinstance(model, SeCo)
    assert isinstance(model.backbone, MoCoV2Module)


# Forward pass tests
def test_forward_pass_resnet18(config_seco_resnet18):
    """Tests the forward pass for the 512-dim embedding of ResNet-18."""
    model = SeCo(config_seco_resnet18)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 512)


def test_forward_pass_resnet50(config_seco_resnet50):
    """Tests the forward pass for the 2048-dim embedding of ResNet-50."""
    model = SeCo(config_seco_resnet50)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model.forward_features(dummy_input)
    assert output.shape == (2, 2048)


# Download and error handling tests
@patch("aitlas.models.seco_wrapper.seco_resnet50")
@patch("aitlas.models.seco_wrapper.SeCo._download_from_zenodo")
@patch("torch.load")
def test_fallback_to_zenodo_download(
    mock_torch_load, mock_download, mock_model_factory
):
    """Tests the fallback to Zenodo download if local_model_path does not exist."""
    # Setup mocks
    mock_model_instance = MagicMock()
    mock_model_instance.load_state_dict.return_value = (
        MagicMock()
    )  # Mock the load_state_dict return
    mock_model_factory.return_value = mock_model_instance
    mock_download.return_value = None
    mock_torch_load.return_value = create_dummy_raw_seco_dict("seco_resnet50")

    non_existent_path = "/path/to/non_existent/model.ckpt"
    config = {
        "local_model_path": non_existent_path,
        "backbone_name": "seco_resnet50",
        "pretrained": True,
    }

    SeCo(config)

    mock_download.assert_called_once()
    mock_torch_load.assert_called_once_with(
        non_existent_path, map_location="cpu", weights_only=False
    )
    mock_model_instance.load_state_dict.assert_called_once()


def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if config has pretrained=False."""
    config = {
        "local_model_path": "any_path",
        "backbone_name": "any_model",
        "pretrained": False,
    }
    with pytest.raises(NotImplementedError):
        SeCo(config)


@pytest.mark.parametrize("backbone_name", ["unsupported_seco_model", "seco_resnet34"])
def test_raises_error_for_invalid_backbones_on_download(backbone_name):
    """Tests that a ValueError is raised for invalid backbones when downloading."""
    config = {
        "local_model_path": "/path/to/non_existent/model.ckpt",
        "backbone_name": backbone_name,
        "pretrained": True,
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        SeCo(config)
