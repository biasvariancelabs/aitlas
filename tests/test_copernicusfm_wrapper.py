import os
from unittest.mock import patch

import pytest
import torch
from torch import nn

from aitlas.models import CopernicusFM


# Constants for testing
EMBED_DIM = 768  # Base dimension
BATCH_SIZE = 2
CHANNELS = 4
HEIGHT = 224
WIDTH = 224
METADATA_DIM = 4  # [lon, lat, time, area]


# Mock Backbone of the CopernicusFM model
class MockCopernicusBackbone(nn.Module):
    """
    A mock class to simulate the actual CopernicusFM ViT backbone.
    """

    def __init__(self):
        super().__init__()
        # Simulate the 'head' attribute which the wrapper attempts to replace
        self.head = nn.Linear(EMBED_DIM, 10)

    def forward_features(
        self,
        x,
        metadata,
        wavelengths=None,
        bandwidths=None,
        language_embed=None,
        input_mode="spectral",
        kernel_size=None,
    ):
        """
        Simulates the forward pass logic of the backbone.
        """
        B = x.shape[0]
        # Return (B, D)
        return torch.randn(B, EMBED_DIM)


# Fixtures
@pytest.fixture
def dummy_copernicus_input_spectral():
    """Creates a dummy input dictionary for CopernicusFM (Spectral Mode)."""
    return {
        "x": torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
        "metadata": torch.randn(BATCH_SIZE, METADATA_DIM),  # [lon, lat, time, area]
        "wavelengths": [490.0, 560.0, 665.0, 842.0],
        "bandwidths": [65.0, 35.0, 30.0, 115.0],
        "input_mode": "spectral",
    }


@pytest.fixture
def dummy_copernicus_input_variable():
    """Creates a dummy input dictionary for CopernicusFM (Variable Mode)."""
    return {
        "x": torch.randn(BATCH_SIZE, 1, HEIGHT, WIDTH),  # Usually 1 channel for variable mode
        "metadata": torch.randn(BATCH_SIZE, METADATA_DIM),
        "language_embed": torch.randn(2048),  # Llama embedding size
        "input_mode": "variable",
    }


@pytest.fixture
def config_copernicus_base():
    """
    Valid config for CopernicusFM Base.
    Note: We must use the specific filename expected by the wrapper's lookup logic.
    """
    return {
        "local_model_path": "/tmp/mock_path/CopernicusFM_ViT_base_varlang_e100.pth",
        "backbone_name": "copernicusfm_base",
        "pretrained": True,
    }


# Tests
# Model instantiation tests
@patch("aitlas.models.copernicusfm_wrapper.torch.load")
@patch("aitlas.models.copernicusfm_wrapper.os.path.exists")
# We patch the backbone class in the wrapper to return our MockBackbone
@patch(
    "aitlas.models.copernicusfm_wrapper.copernicusfm_base",
    side_effect=MockCopernicusBackbone,
)
def test_instantiation_copernicus_local(
    mock_backbone_cls, mock_os_exists, mock_torch_load, config_copernicus_base
):
    """Tests loading from a local file."""

    # Setup
    mock_os_exists.return_value = True  # File exists
    mock_torch_load.return_value = {}  # Return empty state dict

    # Initialize
    model = CopernicusFM(config_copernicus_base)

    # Assertions
    assert isinstance(model, CopernicusFM)
    # Ensure the backbone is our mock
    assert isinstance(model.backbone, MockCopernicusBackbone)
    # Check that the head was replaced with Identity (wrapper logic)
    assert isinstance(model.backbone.head, nn.Identity)

    # Ensure we tried to load the state dict from local path
    mock_torch_load.assert_called_with(
        config_copernicus_base["local_model_path"], weights_only=False
    )


@patch("aitlas.models.copernicusfm_wrapper.hf_hub_download")
@patch("aitlas.models.copernicusfm_wrapper.torch.load")
@patch("aitlas.models.copernicusfm_wrapper.os.path.exists")
@patch(
    "aitlas.models.copernicusfm_wrapper.copernicusfm_base",
    side_effect=MockCopernicusBackbone,
)
def test_instantiation_copernicus_download(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    mock_hf_download,
    config_copernicus_base,
):
    """Tests fallback to Hugging Face download when local file is missing."""

    # Setup
    mock_os_exists.return_value = False  # Trigger download
    expected_download_path = "/tmp/cache/CopernicusFM_ViT_base_varlang_e100.pth"
    mock_hf_download.return_value = expected_download_path
    mock_torch_load.return_value = {}

    # Initialize
    model = CopernicusFM(config_copernicus_base)

    # Assertions
    expected_repo = "wangyi111/Copernicus-FM"
    expected_filename = "CopernicusFM_ViT_base_varlang_e100.pth"

    # Check download was called
    mock_hf_download.assert_called_once_with(
        repo_id=expected_repo,
        filename=expected_filename,
        local_dir=os.path.dirname(config_copernicus_base["local_model_path"]),
    )

    # Check we loaded the downloaded file
    mock_torch_load.assert_called_with(expected_download_path, weights_only=False)


def test_raises_error_if_pretrained_is_false_copernicus():
    """Tests that NotImplementedError is raised if pretrained=False."""
    config = {
        "local_model_path": "any",
        "backbone_name": "copernicusfm_base",
        "pretrained": False,
    }
    with pytest.raises(
        NotImplementedError,
        match="Loading model without pretrained weights is not supported",
    ):
        CopernicusFM(config)


@patch("aitlas.models.copernicusfm_wrapper.os.path.exists")
def test_raises_error_invalid_backbone_copernicus(mock_os_exists):
    """Tests ValueError for unknown backbone names during download attempt."""
    mock_os_exists.return_value = False  # Trigger download logic where check happens
    config = {
        "local_model_path": "any/path/model.pth",
        "backbone_name": "invalid_backbone_name",
        "pretrained": True,
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        CopernicusFM(config)


# Forward pass tests
@patch("aitlas.models.copernicusfm_wrapper.torch.load")
@patch("aitlas.models.copernicusfm_wrapper.os.path.exists")
@patch(
    "aitlas.models.copernicusfm_wrapper.copernicusfm_base",
    side_effect=MockCopernicusBackbone,
)
def test_forward_pass_copernicus_spectral(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    config_copernicus_base,
    dummy_copernicus_input_spectral,
):
    """Tests forward pass in 'spectral' mode."""
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}

    model = CopernicusFM(config_copernicus_base)

    # Unpack dictionary to pass args
    output = model.forward_features(**dummy_copernicus_input_spectral)

    # Assert
    assert isinstance(output, torch.Tensor)
    # Expected shape (B, EmbedDim)
    assert output.shape == (BATCH_SIZE, EMBED_DIM)


@patch("aitlas.models.copernicusfm_wrapper.torch.load")
@patch("aitlas.models.copernicusfm_wrapper.os.path.exists")
@patch(
    "aitlas.models.copernicusfm_wrapper.copernicusfm_base",
    side_effect=MockCopernicusBackbone,
)
def test_forward_pass_copernicus_variable(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    config_copernicus_base,
    dummy_copernicus_input_variable,
):
    """Tests forward pass in 'variable' mode (language embeddings)."""
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}

    model = CopernicusFM(config_copernicus_base)

    # Unpack dictionary to pass args
    output = model.forward_features(**dummy_copernicus_input_variable)

    # Assert
    assert isinstance(output, torch.Tensor)
    # Expected shape (B, EmbedDim)
    assert output.shape == (BATCH_SIZE, EMBED_DIM)
