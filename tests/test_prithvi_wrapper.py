import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitlas.models import Prithvi
from aitlas.models.prithvi_wrapper import (
    PrithviMAE,
    PrithviViT,
    prithvi_eo_v1_base,
    prithvi_eo_v2_base_tl,
    prithvi_eo_v2_huge,
    prithvi_eo_v2_huge_tl,
    prithvi_eo_v2_large,
    prithvi_eo_v2_large_tl,
    prithvi_eo_v2_tiny_tl,
)


# Helper functions and fixtures
def create_dummy_prithvi_dict(model_factory_func):
    """
    Creates a realistic dummy state_dict for a Prithvi model
    by instantiating it via its factory function.
    """
    # Instantiate the model to get its structure
    model = model_factory_func()
    # The checkpoint contains the state_dict of the PrithviMAE model
    state_dict = model.state_dict()
    return state_dict


@pytest.fixture(scope="session")
def dummy_prithvi_v1_base_checkpoint(tmp_path_factory):
    """Creates a temporary .pt file for the Prithvi V1 Base model."""
    path = tmp_path_factory.mktemp("checkpoints") / "Prithvi_EO_V1_100M.pt"
    torch.save(create_dummy_prithvi_dict(prithvi_eo_v1_base), path)
    return str(path)


@pytest.fixture(scope="session")
def dummy_prithvi_v2_tiny_checkpoint(tmp_path_factory):
    """Creates a temporary .pt file for the Prithvi V2 Tiny TL model."""
    path = tmp_path_factory.mktemp("checkpoints") / "Prithvi_EO_V2_tiny_TL.pt"
    torch.save(create_dummy_prithvi_dict(prithvi_eo_v2_tiny_tl), path)
    return str(path)


@pytest.fixture(scope="session")
def dummy_prithvi_v2_base_checkpoint(tmp_path_factory):
    """Creates a temporary .pt file for the Prithvi V2 Base TL model."""
    path = tmp_path_factory.mktemp("checkpoints") / "Prithvi_EO_V2_100M_TL.pt"
    torch.save(create_dummy_prithvi_dict(prithvi_eo_v2_base_tl), path)
    return str(path)


@pytest.fixture
def config_prithvi_v1_base(dummy_prithvi_v1_base_checkpoint):
    """Provides a valid config for Prithvi V1 Base with a local file."""
    return {
        "local_model_path": dummy_prithvi_v1_base_checkpoint,
        "backbone_name": "prithvi_eo_v1_base",  # Optional, but good for clarity
        "pretrained": True,
    }


@pytest.fixture
def config_prithvi_v2_tiny(dummy_prithvi_v2_tiny_checkpoint):
    """Provides a valid config for Prithvi V2 Tiny TL with a local file."""
    return {
        "local_model_path": dummy_prithvi_v2_tiny_checkpoint,
        "backbone_name": "prithvi_eo_v2_tiny_tl",  # Optional, but good for clarity
        "pretrained": True,
    }


# Tests
# Model instantiation tests
def test_instantiation_prithvi_v1_base(config_prithvi_v1_base):
    """Tests correct instantiation and backbone type for Prithvi V1 Base."""
    model = Prithvi(config_prithvi_v1_base)
    assert isinstance(model, Prithvi)
    # The factory functions return PrithviMAE instances
    assert isinstance(model.backbone, PrithviMAE)
    # Confirm it's a ViT-base model
    assert model.backbone.encoder.embed_dim == 768


def test_instantiation_prithvi_v2_tiny(config_prithvi_v2_tiny):
    """Tests correct instantiation and backbone type for Prithvi V2 Tiny TL."""
    model = Prithvi(config_prithvi_v2_tiny)
    assert isinstance(model, Prithvi)
    assert isinstance(model.backbone, PrithviMAE)
    # Confirm it's a ViT-tiny
    assert model.backbone.encoder.embed_dim == 192


# Forward pass tests
B, T_IN, C, H_IMG, W_IMG = 2, 3, 6, 224, 224  # Input dimensions for the test
PATCH_T, PATCH_H, PATCH_W = 1, 16, 16  # Patch dimensions
N_H, N_W = H_IMG // PATCH_H, W_IMG // PATCH_W  # Spatial grid size (14, 14)
NUM_PATCHES = T_IN * N_H * N_W  # Total patches (3 * 14 * 14 = 588)
EMBED_DIM_BASE = 768  # Embedding dim for ViT-base
EFFECTIVE_T = T_IN // PATCH_T  # Effective time dim (3 // 1 = 3)


def test_forward_pass_v1_base_raw_features(config_prithvi_v1_base):
    """Tests the forward pass for raw ViT features (V1 Base)."""
    model = Prithvi(config_prithvi_v1_base)
    dummy_input = torch.randn(B, T_IN, C, H_IMG, W_IMG)
    temp_coords = torch.randint(1, 365, (B, T_IN, 2)).float()
    loc_coords = torch.rand(B, 2).float() * 90

    assert model.backbone.encoder.num_frames == T_IN
    assert model.backbone.encoder.patch_embed.patch_size == [PATCH_T, PATCH_H, PATCH_W]
    assert model.backbone.encoder.embed_dim == EMBED_DIM_BASE

    output_tensor = model.forward_features(
        dummy_input,
        temporal_coords=temp_coords,
        location_coords=loc_coords,
        segmentation_features=False,
    )
    # Check that it's a tensor, not a list
    assert isinstance(output_tensor, torch.Tensor)

    # Expected shape: [Batch, 1 + NumPatches, EmbedDim]
    assert output_tensor.shape == (B, 1 + NUM_PATCHES, EMBED_DIM_BASE)


def test_forward_pass_v1_base_segmentation_features(config_prithvi_v1_base):
    """Tests the forward pass for segmentation-ready features (V1 Base)."""
    model = Prithvi(config_prithvi_v1_base)
    dummy_input = torch.randn(B, T_IN, C, H_IMG, W_IMG)
    temp_coords = torch.randint(1, 365, (B, T_IN, 2)).float()
    loc_coords = torch.rand(B, 2).float() * 90

    # Ensure the loaded backbone has the expected config
    assert model.backbone.encoder.num_frames == T_IN
    assert model.backbone.encoder.patch_embed.patch_size == [PATCH_T, PATCH_H, PATCH_W]
    assert model.backbone.encoder.embed_dim == EMBED_DIM_BASE
    # Check properties used in prepare_features
    assert model.backbone.encoder.patch_embed.input_size[0] == T_IN

    output_tensor = model.forward_features(
        dummy_input,
        temporal_coords=temp_coords,
        location_coords=loc_coords,
        segmentation_features=True,  # Trigger the segmentation path
    )
    # Check that it's a tensor, not a list
    assert isinstance(output_tensor, torch.Tensor)

    # Expected shape: [Batch, EffectiveT * EmbedDim, GridH, GridW]
    expected_channels = EFFECTIVE_T * EMBED_DIM_BASE  # 3 * 768 = 2304
    assert output_tensor.shape == (B, expected_channels, N_H, N_W)


# Download and error handling tests (prithvi_eo_v2_base_tl as an example)
@patch("aitlas.models.prithvi_wrapper.prithvi_eo_v2_base_tl")
@patch("aitlas.models.prithvi_wrapper.hf_hub_download")
@patch("torch.load")
def test_fallback_to_hf_download(mock_torch_load, mock_hf_download, mock_model_factory):
    """Tests the fallback to Hugging Face download if local_model_path does not exist."""

    # Setup mocks
    mock_model_instance = MagicMock(spec=PrithviMAE)
    # Mock the state_dict loading capability
    mock_model_instance.load_state_dict.return_value = MagicMock()
    mock_model_factory.return_value = mock_model_instance

    downloaded_path = "/fake/download/path/Prithvi_EO_V2_100M_TL.pt"
    mock_hf_download.return_value = downloaded_path

    dummy_dict = create_dummy_prithvi_dict(prithvi_eo_v2_base_tl)
    mock_torch_load.return_value = dummy_dict

    # Config
    non_existent_path = "/path/to/non_existent/model.pt"
    config = {
        "local_model_path": non_existent_path,
        "backbone_name": "prithvi_eo_v2_base_tl",  # Crucial for selecting repo/file
        "pretrained": True,
    }

    # Instantiate the wrapper, triggering the load_backbone logic
    model_wrapper = Prithvi(config)

    # Assertions
    expected_repo_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL"
    expected_filename = "Prithvi_EO_V2_100M_TL.pt"
    mock_hf_download.assert_called_once_with(
        repo_id=expected_repo_id,
        filename=expected_filename,
        local_dir=os.path.dirname(
            non_existent_path
        ),  # It uses the dir of the non-existent path
    )

    mock_torch_load.assert_called_once_with(downloaded_path, weights_only=False)
    mock_model_factory.assert_called_once()
    mock_model_instance.load_state_dict.assert_called_once_with(
        dummy_dict, strict=False
    )
    # Check that the backbone was actually set
    assert model_wrapper.backbone is mock_model_instance


def test_raises_error_if_pretrained_is_false():
    """Tests that a NotImplementedError is raised if config has pretrained=False."""
    config = {
        "local_model_path": "any_path",
        "backbone_name": "any_model",
        "pretrained": False,
    }
    with pytest.raises(
        NotImplementedError,
        match="Loading model without pretrained weights is not supported.",
    ):
        Prithvi(config)


@pytest.mark.parametrize(
    "backbone_name",
    ["unsupported_prithvi_model", "vit_base", "prithvi"],  # Test invalid names
)
@patch("aitlas.models.prithvi_wrapper.hf_hub_download")
def test_raises_error_for_invalid_backbones_on_download(
    mock_hf_download, backbone_name
):
    """Tests that a ValueError is raised for invalid backbones when download is triggered."""
    config = {
        "local_model_path": "/path/to/non_existent/model.pth",  # Force download path
        "backbone_name": backbone_name,
        "pretrained": True,
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        Prithvi(config)
