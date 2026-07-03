import os
from unittest.mock import patch

import pytest
import torch
from torch import nn

from aitlas.models import CROMA


# Constants for testing
EMBED_DIM = 768  # CROMA Base dimension
BATCH_SIZE = 2
HEIGHT = 120  # CROMA uses 120x120 images
WIDTH = 120
SAR_CHANNELS = 2
OPTICAL_CHANNELS = 12  # B10 omitted


# Mock Backbone of the CROMA model
class MockCROMABackbone(nn.Module):
    """
    A mock class to simulate the actual CROMA ViT backbone.
    """

    def __init__(self):
        super().__init__()
        # Simulate attributes required by the wrapper
        self.head = nn.Linear(EMBED_DIM, 10)
        self.modalities = ["sar", "optical"]

    def forward(self, x_sar=None, x_optical=None):
        """
        Simulates the forward pass logic of CROMA backbone.
        Returns a dictionary of embeddings.
        """
        # Determine batch size from whichever input is present
        if x_sar is not None:
            B = x_sar.shape[0]
        elif x_optical is not None:
            B = x_optical.shape[0]
        else:
            B = 1

        # Simulate patches (e.g., 100 patches)
        num_patches = 100

        return {
            "SAR_encodings": torch.randn(B, num_patches, EMBED_DIM),
            "SAR_GAP": torch.randn(B, EMBED_DIM),
            "optical_encodings": torch.randn(B, num_patches, EMBED_DIM),
            "optical_GAP": torch.randn(B, EMBED_DIM),
            "joint_encodings": torch.randn(B, num_patches, EMBED_DIM),
            "joint_GAP": torch.randn(B, EMBED_DIM),
        }


# Fixtures
@pytest.fixture
def dummy_croma_input():
    """Creates a dummy input dictionary for CROMA."""
    return {
        "x_sar": torch.randn(BATCH_SIZE, SAR_CHANNELS, HEIGHT, WIDTH),
        "x_optical": torch.randn(BATCH_SIZE, OPTICAL_CHANNELS, HEIGHT, WIDTH),
        "modalities": ["sar", "optical"],
    }


@pytest.fixture
def dummy_croma_input_sar_only():
    """Creates a dummy input dictionary for CROMA (SAR only)."""
    return {
        "x_sar": torch.randn(BATCH_SIZE, SAR_CHANNELS, HEIGHT, WIDTH),
        "x_optical": None,
        "modalities": ["sar"],
    }


@pytest.fixture
def config_croma_base():
    """
    Valid config for CROMA Base.
    """
    return {
        "local_model_path": "/tmp/mock_path/CROMA_base.pt",
        "backbone_name": "croma_base",
        "pretrained": True,
    }


# Tests
# Model instantiation tests
@patch("aitlas.models.croma_wrapper.torch.load")
@patch("aitlas.models.croma_wrapper.os.path.exists")
# We must patch 'croma_base' inside the wrapper module because the wrapper uses globals()['croma_base']
@patch("aitlas.models.croma_wrapper.croma_base", side_effect=MockCROMABackbone)
def test_instantiation_croma_local(
    mock_backbone_cls, mock_os_exists, mock_torch_load, config_croma_base
):
    """Tests loading from a local file."""

    # Setup
    mock_os_exists.return_value = True  # File exists
    mock_torch_load.return_value = {}  # Return empty state dict

    # Initialize
    model = CROMA(config_croma_base)

    # Assertions
    assert isinstance(model, CROMA)
    assert isinstance(model.backbone, MockCROMABackbone)
    # Check that the head was replaced with Identity (wrapper logic)
    assert isinstance(model.backbone.head, nn.Identity)

    # Ensure we loaded the state dict from local path
    mock_torch_load.assert_called_with(config_croma_base["local_model_path"], weights_only=False)


@patch("aitlas.models.croma_wrapper.hf_hub_download")
@patch("aitlas.models.croma_wrapper.torch.load")
@patch("aitlas.models.croma_wrapper.os.path.exists")
@patch("aitlas.models.croma_wrapper.croma_base", side_effect=MockCROMABackbone)
def test_instantiation_croma_download(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    mock_hf_download,
    config_croma_base,
):
    """Tests fallback to Hugging Face download when local file is missing."""

    # Setup
    mock_os_exists.return_value = False  # Trigger download
    expected_download_path = "/tmp/cache/CROMA_base.pt"
    mock_hf_download.return_value = expected_download_path
    mock_torch_load.return_value = {}

    # Initialize
    model = CROMA(config_croma_base)

    # Assertions
    expected_repo = "antofuller/CROMA"
    expected_filename = "CROMA_base.pt"

    # Check download was called
    mock_hf_download.assert_called_once_with(
        repo_id=expected_repo,
        filename=expected_filename,
        local_dir=os.path.dirname(config_croma_base["local_model_path"]),
    )

    # Check we loaded the downloaded file
    mock_torch_load.assert_called_with(expected_download_path, weights_only=False)


def test_raises_error_if_pretrained_is_false_croma():
    """Tests that NotImplementedError is raised if pretrained=False."""
    config = {
        "local_model_path": "any",
        "backbone_name": "croma_base",
        "pretrained": False,
    }
    with pytest.raises(
        NotImplementedError,
        match="Loading model without pretrained weights is not supported",
    ):
        CROMA(config)


@patch("aitlas.models.croma_wrapper.os.path.exists")
def test_raises_error_invalid_backbone_croma(mock_os_exists):
    """Tests ValueError for unknown backbone names during download attempt."""
    mock_os_exists.return_value = False  # Trigger download logic
    config = {
        "local_model_path": "any/path/model.pt",
        "backbone_name": "invalid_croma_model",
        "pretrained": True,
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        CROMA(config)


@patch("aitlas.models.croma_wrapper.os.path.exists")
def test_raises_error_no_weights_for_backbone_croma(mock_os_exists):
    """Tests ValueError if backbone exists in keys but has no checkpoint defined."""
    # This assumes there might be a key in the dictionary defined as None,
    # effectively simulating the 'if backbone_checkpoints[...] is None' check.
    pass


# Forward pass tests
@patch("aitlas.models.croma_wrapper.torch.load")
@patch("aitlas.models.croma_wrapper.os.path.exists")
@patch("aitlas.models.croma_wrapper.croma_base", side_effect=MockCROMABackbone)
def test_forward_pass_croma_full(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    config_croma_base,
    dummy_croma_input,
):
    """Tests forward pass with both SAR and Optical."""
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}

    model = CROMA(config_croma_base)

    # Unpack dictionary to pass args
    output = model.forward_features(**dummy_croma_input)

    # Assert
    assert isinstance(output, dict)
    # Check for specific keys expected from CROMA
    expected_keys = [
        "SAR_encodings",
        "SAR_GAP",
        "optical_encodings",
        "optical_GAP",
        "joint_GAP",
    ]
    for key in expected_keys:
        assert key in output
        assert isinstance(output[key], torch.Tensor)

    # Check shape of GAP (Global Average Pooling)
    assert output["joint_GAP"].shape == (BATCH_SIZE, EMBED_DIM)


@patch("aitlas.models.croma_wrapper.torch.load")
@patch("aitlas.models.croma_wrapper.os.path.exists")
@patch("aitlas.models.croma_wrapper.croma_base", side_effect=MockCROMABackbone)
def test_forward_pass_croma_modality_update(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    config_croma_base,
    dummy_croma_input_sar_only,
):
    """Tests that modalities are updated in the backbone correctly."""
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}

    model = CROMA(config_croma_base)

    # Initially mock modalities might be both
    assert model.backbone.modalities == ["sar", "optical"]

    # Pass SAR only input
    _ = model.forward_features(**dummy_croma_input_sar_only)

    # Check if backbone modalities were updated
    assert model.backbone.modalities == ["sar"]
