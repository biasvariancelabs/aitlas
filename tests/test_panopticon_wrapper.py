import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitlas.models import Panopticon


# Constants for testing
EMBED_DIM = 768
BATCH_SIZE = 2
CHANNELS = 4
HEIGHT = 224
WIDTH = 224
PATCH_SIZE = 14
NUM_PATCHES = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)  # 256


# Mock Backbone of the Panopticon model (DinoV2 ViT)
class MockPanopticonBackbone(nn.Module):
    """
    A mock class to simulate the actual Panopticon ViT backbone.
    We need this to test the wrapper without loading the heavy weights.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_dict, dense_features=False):
        """
        Simulates the forward pass logic of the backbone.
        """
        imgs = x_dict.get("imgs")
        B = imgs.shape[0]

        if dense_features:
            # Return (B, N + 1, D)
            return torch.randn(B, NUM_PATCHES + 1, EMBED_DIM)
        else:
            # Return (B, D)
            return torch.randn(B, EMBED_DIM)


# Fixtures
@pytest.fixture
def dummy_panopticon_input():
    """Creates a dummy input dictionary for Panopticon."""
    return {
        "imgs": torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
        # Random wavelengths/IDs
        "chn_ids": torch.randint(0, 1000, (BATCH_SIZE, CHANNELS)),
    }


@pytest.fixture
def config_panopticon_base():
    """Valid config for Panopticon Base."""
    return {
        "local_model_path": "/tmp/mock_path/panopticon_vitb14.pth",
        "backbone_name": "panopticon_base",
        "pretrained": True,
    }


# Tests
# Model instantiation tests
@patch("aitlas.models.panopticon_wrapper.torch.load")
@patch("aitlas.models.panopticon_wrapper.os.path.exists")
# WE PATCH THE CLASS IN THE WRAPPER'S NAMESPACE TO RETURN OUR MOCK
@patch(
    "aitlas.models.panopticon_wrapper.panopticon_base",
    side_effect=MockPanopticonBackbone,
)
def test_instantiation_panopticon_local(
    mock_backbone_cls, mock_os_exists, mock_torch_load, config_panopticon_base
):
    """Tests loading from a local file."""

    # Setup
    mock_os_exists.return_value = True  # File exists
    mock_torch_load.return_value = {}  # Return empty state dict

    # Initialize
    model = Panopticon(config_panopticon_base)

    # Assertions
    assert isinstance(model, Panopticon)
    # Ensure the backbone is our mock
    assert isinstance(model.backbone, MockPanopticonBackbone)
    # Ensure we tried to load the state dict
    mock_torch_load.assert_called_with(
        config_panopticon_base["local_model_path"], weights_only=False
    )


@patch("aitlas.models.panopticon_wrapper.hf_hub_download")
@patch("aitlas.models.panopticon_wrapper.torch.load")
@patch("aitlas.models.panopticon_wrapper.os.path.exists")
@patch(
    "aitlas.models.panopticon_wrapper.panopticon_base",
    side_effect=MockPanopticonBackbone,
)
def test_instantiation_panopticon_download(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    mock_hf_download,
    config_panopticon_base,
):
    """Tests fallback to Hugging Face download when local file is missing."""

    # Setup
    mock_os_exists.return_value = False  # Trigger download
    expected_download_path = "/tmp/cache/panopticon_vitb14.pth"
    mock_hf_download.return_value = expected_download_path
    mock_torch_load.return_value = {}

    # Initialize
    model = Panopticon(config_panopticon_base)

    # Assertions
    expected_repo = "lewaldm/panopticon"
    expected_filename = "panopticon_vitb14.pth"

    # Check download was called
    mock_hf_download.assert_called_once_with(
        repo_id=expected_repo,
        filename=expected_filename,
        local_dir=os.path.dirname(config_panopticon_base["local_model_path"]),
    )

    # Check we loaded the downloaded file
    mock_torch_load.assert_called_with(expected_download_path, weights_only=False)


def test_raises_error_if_pretrained_is_false_panopticon():
    """Tests that NotImplementedError is raised if pretrained=False."""
    config = {
        "local_model_path": "any",
        "backbone_name": "panopticon_base",
        "pretrained": False,
    }
    with pytest.raises(
        NotImplementedError,
        match="Loading model without pretrained weights is not supported",
    ):
        Panopticon(config)


@patch("aitlas.models.panopticon_wrapper.os.path.exists")
def test_raises_error_invalid_backbone_panopticon(mock_os_exists):
    """Tests ValueError for unknown backbone names."""
    mock_os_exists.return_value = False  # Trigger checks
    config = {
        "local_model_path": "any",
        "backbone_name": "invalid_backbone_name",
        "pretrained": True,
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        Panopticon(config)


# Forward pass tests
@patch("aitlas.models.panopticon_wrapper.torch.load")
@patch("aitlas.models.panopticon_wrapper.os.path.exists")
@patch(
    "aitlas.models.panopticon_wrapper.panopticon_base",
    side_effect=MockPanopticonBackbone,
)
def test_forward_pass_panopticon_pooled(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    config_panopticon_base,
    dummy_panopticon_input,
):
    """Tests forward pass returning the CLS token (pooled)."""
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}

    model = Panopticon(config_panopticon_base)

    output = model.forward_features(dummy_panopticon_input, dense_features=False)

    # Assert
    assert isinstance(output, torch.Tensor)
    # Expected shape (B, EmbedDim)
    assert output.shape == (BATCH_SIZE, EMBED_DIM)


@patch("aitlas.models.panopticon_wrapper.torch.load")
@patch("aitlas.models.panopticon_wrapper.os.path.exists")
@patch(
    "aitlas.models.panopticon_wrapper.panopticon_base",
    side_effect=MockPanopticonBackbone,
)
def test_forward_pass_panopticon_dense(
    mock_backbone_cls,
    mock_os_exists,
    mock_torch_load,
    config_panopticon_base,
    dummy_panopticon_input,
):
    """Tests forward pass returning patch tokens (dense)."""
    # Setup
    mock_os_exists.return_value = True
    mock_torch_load.return_value = {}

    model = Panopticon(config_panopticon_base)

    output = model.forward_features(dummy_panopticon_input, dense_features=True)

    # Assert
    assert isinstance(output, torch.Tensor)
    # Expected shape (B, NumPatches + 1, EmbedDim)
    assert output.shape == (BATCH_SIZE, NUM_PATCHES + 1, EMBED_DIM)
