import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from aitlas.models import Galileo
from aitlas.models.Galileo import Encoder
from aitlas.models.Galileo.utils import (
    CONFIG_FILENAME,
    ENCODER_FILENAME,
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)


# Helper functions and fixtures
def create_dummy_galileo_folder(tmp_path_factory, model_name: str, embed_dim: int):
    """
    Creates a temporary folder with a dummy encoder.pt and config.json
    for the Galileo model.
    """
    folder = tmp_path_factory.mktemp(model_name)

    # Create dummy config.json
    # The Encoder.load_from_folder expects this structure
    encoder_config = {"embedding_size": embed_dim, "depth": 2, "max_patch_size": 4}
    config_data = {"model": {"encoder": encoder_config}}

    with open(folder / CONFIG_FILENAME, "w") as f:
        json.dump(config_data, f)

    # Create dummy encoder.pt
    # We instantiate a real Encoder (from the wrapper's code) to get a valid state_dict
    model = Encoder(**encoder_config)
    torch.save(model.state_dict(), folder / ENCODER_FILENAME)

    return folder


@pytest.fixture(scope="session")
def dummy_galileo_nano_folder(tmp_path_factory):
    """Creates a temporary folder for the Galileo Nano model."""
    return create_dummy_galileo_folder(tmp_path_factory, "galileo_nano", embed_dim=128)


@pytest.fixture(scope="session")
def dummy_galileo_base_folder(tmp_path_factory):
    """Creates a temporary folder for the Galileo Base model."""
    return create_dummy_galileo_folder(tmp_path_factory, "galileo_base", embed_dim=768)


@pytest.fixture
def config_galileo_nano(dummy_galileo_nano_folder):
    """Provides a valid config for Galileo Nano with a local file."""
    return {
        # The wrapper logic derives the folder from the file path
        "local_model_path": str(dummy_galileo_nano_folder / ENCODER_FILENAME),
        "backbone_name": "galileo_nano",
        "pretrained": True,
    }


@pytest.fixture
def config_galileo_base(dummy_galileo_base_folder):
    """Provides a valid config for Galileo Base with a local file."""
    return {
        "local_model_path": str(dummy_galileo_base_folder / ENCODER_FILENAME),
        "backbone_name": "galileo_base",
        "pretrained": True,
    }


# Tests
# Model instantiation tests
def test_instantiation_galileo_nano(config_galileo_nano):
    """Tests correct instantiation and backbone type for Galileo Nano."""
    model = Galileo(config_galileo_nano)
    assert isinstance(model, Galileo)
    assert isinstance(model.backbone, Encoder)
    # Check embedding dim from our dummy config
    assert model.backbone.embedding_size == 128


def test_instantiation_galileo_base(config_galileo_base):
    """Tests correct instantiation and backbone type for Galileo Base."""
    model = Galileo(config_galileo_base)
    assert isinstance(model, Galileo)
    assert isinstance(model.backbone, Encoder)
    # Check embedding dim from our dummy config
    assert model.backbone.embedding_size == 768


# Forward pass tests
B, T = 2, 3
H, W = 32, 32
PATCH_SIZE = 4
H_p, W_p = H // PATCH_SIZE, W // PATCH_SIZE  # Patched grid size (8, 8)
D_NANO = 128  # Embedding dim for nano

# Get channel group counts
C_st_g = len(SPACE_TIME_BANDS_GROUPS_IDX)
C_sp_g = len(SPACE_BAND_GROUPS_IDX)
C_t_g = len(TIME_BAND_GROUPS_IDX)
C_s_g = len(STATIC_BAND_GROUPS_IDX)


@pytest.fixture
def dummy_galileo_input():
    """Creates a dummy input dictionary in the AiTLAS-standard format."""
    return {
        "s1": torch.randn(B, T, 2, H, W),  # B, T, C, H, W
        "s2": torch.randn(B, T, 10, H, W),  # B, T, C, H, W
        "srtm": torch.randn(B, 2, H, W),  # B, C, H, W
        "era5": torch.randn(B, T, 2),  # B, T, C
        "landscan": torch.randn(B, 1),  # B, C
        "latlon": torch.randn(B, 2),  # B, 2
        "months": torch.randint(0, 12, (B, T)),  # B, T
    }


def test_forward_pass_galileo_nano_average_features(
    config_galileo_nano, dummy_galileo_input
):
    """Tests the forward pass for averaged features (Nano)."""
    model = Galileo(config_galileo_nano)

    output_tensor = model.forward_features(
        dummy_galileo_input,
        patch_size=PATCH_SIZE,
        average_features=True,
        normalize=False,  # Skip normalization for dummy data
    )

    # Check that it's a single tensor
    assert isinstance(output_tensor, torch.Tensor)

    # Expected shape: [Batch, EmbedDim]
    assert output_tensor.shape == (B, D_NANO)


def test_forward_pass_galileo_nano_raw_features(
    config_galileo_nano, dummy_galileo_input
):
    """Tests the forward pass for raw, non-averaged features (Nano)."""
    model = Galileo(config_galileo_nano)

    output_tuple = model.forward_features(
        dummy_galileo_input,
        patch_size=PATCH_SIZE,
        average_features=False,  # Trigger the raw feature path
        normalize=False,
    )

    # Check that it's a tuple
    assert isinstance(output_tuple, tuple)

    # As per the code, it returns all 9 tensors
    assert len(output_tuple) == 9

    # Unpack the 4 embedding tensors
    s_t_x, sp_x, t_x, st_x = output_tuple[:4]

    # Check shapes
    # [B, H_p, W_p, T, C_st_g, D]
    assert s_t_x.shape == (B, H_p, W_p, T, C_st_g, D_NANO)
    # [B, H_p, W_p, C_sp_g, D]
    assert sp_x.shape == (B, H_p, W_p, C_sp_g, D_NANO)
    # [B, T, C_t_g, D]
    assert t_x.shape == (B, T, C_t_g, D_NANO)
    # [B, C_s_g, D]
    assert st_x.shape == (B, C_s_g, D_NANO)


# Download and error handling tests
@patch("aitlas.models.galileo_wrapper.shutil.rmtree")
@patch("aitlas.models.galileo_wrapper.shutil.move")
@patch("aitlas.models.galileo_wrapper.Encoder.load_from_folder")
@patch("aitlas.models.galileo_wrapper.hf_hub_download")
@patch("aitlas.models.galileo_wrapper.os.path.exists")
def test_fallback_to_hf_download_galileo(
    mock_os_exists,
    mock_hf_download,
    mock_load_from_folder,
    mock_shutil_move,
    mock_shutil_rmtree,
):
    """Tests the fallback to Hugging Face download if local_model_path does not exist."""

    # Setup mocks
    mock_os_exists.return_value = False  # Trigger the download logic

    # Mock the return paths from hf_hub_download
    # It downloads encoder.pt and config.json
    downloaded_pt_path = "/fake/download/models/nano/encoder.pt"
    downloaded_cfg_path = "/fake/download/models/nano/config.json"
    mock_hf_download.side_effect = [downloaded_pt_path, downloaded_cfg_path]

    mock_backbone = MagicMock(spec=Encoder)
    mock_load_from_folder.return_value = mock_backbone

    # Config
    non_existent_path = "/path/to/non_existent/encoder.pt"
    config = {
        "local_model_path": non_existent_path,
        "backbone_name": "galileo_nano",  # Crucial for selecting repo/file
        "pretrained": True,
    }

    model_wrapper = Galileo(config)

    expected_repo_id = "nasaharvest/galileo"
    expected_subfolder = "models/nano"
    local_dir = os.path.dirname(non_existent_path)

    # Check that hf_hub_download was called correctly for both files
    assert mock_hf_download.call_count == 2
    mock_hf_download.assert_any_call(
        repo_id=expected_repo_id,
        filename=ENCODER_FILENAME,
        subfolder=expected_subfolder,
        local_dir=local_dir,
    )
    mock_hf_download.assert_any_call(
        repo_id=expected_repo_id,
        filename=CONFIG_FILENAME,
        subfolder=expected_subfolder,
        local_dir=local_dir,
    )

    # Check that the model was loaded from the correct folder
    mock_load_from_folder.assert_called_once_with(
        folder=Path(local_dir), device=torch.device("cpu")
    )

    # Check that the backbone was actually set
    assert model_wrapper.backbone is mock_backbone


def test_raises_error_if_pretrained_is_false_galileo():
    """Tests that a NotImplementedError is raised if config has pretrained=False."""
    config = {
        "local_model_path": "any_path",
        "backbone_name": "galileo_nano",
        "pretrained": False,
    }
    with pytest.raises(
        NotImplementedError,
        match="Loading model without pretrained weights is not supported.",
    ):
        Galileo(config)


@pytest.mark.parametrize(
    "backbone_name",
    ["unsupported_galileo_model", "vit_base", "prithvi"],  # Test invalid names
)
@patch("aitlas.models.galileo_wrapper.os.path.exists")
def test_raises_error_for_invalid_backbones_on_download_galileo(
    mock_os_exists, backbone_name
):
    """Tests that a ValueError is raised for invalid backbones when download is triggered."""
    mock_os_exists.return_value = False  # Trigger download

    config = {
        "local_model_path": "/path/to/non_existent/model.pth",  # Force download path
        "backbone_name": backbone_name,
        "pretrained": True,
    }
    with pytest.raises(ValueError, match="Unsupported or missing backbone"):
        Galileo(config)
