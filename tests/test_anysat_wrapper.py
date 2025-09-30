import pytest
import torch
import os
from unittest.mock import patch, MagicMock

# Import the wrapper and the components it uses
from aitlas.models import AnySat
from aitlas.models.AnySat.anysat import AnySatModule, AnySatEncoder


def create_dummy_anysat_state_dict():
    """Creates a state dict from a fresh `anysat_base` instance.
    The checkpoint is wrapped in a dictionary with a 'state_dict' key,
    mimicking the actual pretrained file structure.
    """
    model = AnySatModule(model_size='base')
    # The weights belong to the inner AnySatEncoder module
    return {'state_dict': model.model.state_dict()}

@pytest.fixture(scope="session")
def dummy_checkpoint_path(tmp_path_factory):
    """Creates a temporary dummy checkpoint file that perfectly matches the model.
    """
    chkpt_dir = tmp_path_factory.mktemp("checkpoints")
    file_path = chkpt_dir / "AnySat.pth"  # Match the name in the config
    perfect_state_dict = create_dummy_anysat_state_dict()
    torch.save(perfect_state_dict, file_path)
    return str(file_path)

@pytest.fixture
def base_config(dummy_checkpoint_path):
    """Provides a basic, valid configuration pointing to the dummy local checkpoint.
    """
    return {
        "local_model_path": dummy_checkpoint_path,
        "backbone_name": "anysat_base",
        "pretrained": True,
    }


# Test Cases
def test_model_instantiation_from_local_path(base_config):
    """Tests the model instantiation for a valid local checkpoint.
    """
    try:
        model = AnySat(base_config)
        
        assert isinstance(model, AnySat)
        assert model.backbone is not None
        assert isinstance(model.backbone, AnySatModule)
        assert isinstance(model.backbone.model, AnySatEncoder)
        
    except Exception as e:
        pytest.fail(f"Model instantiation from local path failed: {e}")

def test_forward_features_pass_no_flash_attn_output_type_tile(base_config):
    """Tests if the loaded model can perform a forward_features pass without flash attention without errors. Output type: tile
    """
    base_config['flash_attn'] = False

    model = AnySat(base_config)

    # Create a dummy input dictionary for a single modality (e.g., Sentinel-2)
    dummy_input = {
        's2': torch.randn(1, 5, 10, 64, 64), # B, T, C, H, W
        's2_dates': torch.randint(0, 365, (1, 5)) # B, T
    }
    
    # patch_size is the ground size in meters (e.g., 640 m for a 64 x 64 S2 tile)
    patch_size = 640
    
    # Expected output shape for 'tile'
    expected_shape = (1, 768)

    try:
        # Call the method with the dictionary and patch size
        output = model.forward_features(dummy_input, patch_size=patch_size, output_type='tile')
        assert output.shape == expected_shape
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")

def test_forward_features_pass_with_flash_attn_output_type_tile(base_config):
    """Tests if the loaded model can perform a forward_features pass with flash attention without errors. Output type: tile
    Flash attention requires flash-attn package and a cuda device. 
    """
    base_config['flash_attn'] = True

    model = AnySat(base_config)
    model.half().to('cuda')

    # Create a dummy input dictionary for a single modality (e.g., Sentinel-2)
    dummy_input = {
        's2': torch.randn(1, 5, 10, 64, 64, device='cuda', dtype=torch.float16), # B, T, C, H, W
        's2_dates': torch.randint(0, 365, (1, 5), device='cuda', dtype=torch.int16) # B, T
    }
    
    # patch_size is the ground size in meters (e.g., 640 m for a 64 x 64 S2 tile)
    patch_size = 640
    
    # Expected output shape for 'tile'
    expected_shape = (1, 768)

    try:
        # Call the method with the dictionary and patch size
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.forward_features(dummy_input, patch_size=patch_size, output_type='tile')
            assert output.shape == expected_shape
            assert output.dtype == torch.float16
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")

def test_forward_features_pass_no_flash_attn_output_type_patch(base_config):
    """Tests if the loaded model can perform a forward_features pass without flash attention without errors. Output type: patch
    """
    base_config['flash_attn'] = False

    model = AnySat(base_config)

    # Create a dummy input dictionary for a single modality (e.g., Sentinel-2)
    dummy_input = {
        's2': torch.randn(1, 5, 10, 64, 64), # B, T, C, H, W
        's2_dates': torch.randint(0, 365, (1, 5)) # B, T
    }
    
    # patch_size is the ground size in meters (e.g., 640 m for a 64x64 S2 tile)
    patch_size = 640
    
    # Expected output shape for 'patch'
    expected_shape = (1, 1, 1, 768)

    try:
        # Call the method with the dictionary and patch size
        output = model.forward_features(dummy_input, patch_size=patch_size, output_type='patch')
        assert output.shape == expected_shape
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")

def test_forward_features_pass_with_flash_attn_output_type_patch(base_config):
    """Tests if the loaded model can perform a forward_features pass with flash attention without errors. Output type: patch
    Flash attention requires flash-attn package and a cuda device. 
    """
    base_config['flash_attn'] = True

    model = AnySat(base_config)
    model.half().to('cuda')

    # Create a dummy input dictionary for a single modality (e.g., Sentinel-2)
    dummy_input = {
        's2': torch.randn(1, 5, 10, 64, 64, device='cuda', dtype=torch.float16), # B, T, C, H, W
        's2_dates': torch.randint(0, 365, (1, 5), device='cuda', dtype=torch.int16) # B, T
    }
    
    # patch_size is the ground size in meters (e.g., 640 m for a 64 x 64 S2 tile)
    patch_size = 640
    
    # Expected output shape for 'patch'
    expected_shape = (1, 1, 1, 768)

    try:
        # Call the method with the dictionary and patch size
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.forward_features(dummy_input, patch_size=patch_size, output_type='patch')
            assert output.shape == expected_shape
            assert output.dtype == torch.float16
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")

def test_forward_features_pass_no_flash_attn_output_type_dense(base_config):
    """Tests if the loaded model can perform a forward_features pass without flash attention without errors. Output type: dense
    """
    base_config['flash_attn'] = False

    model = AnySat(base_config)

    # Create a dummy input dictionary for a single modality (e.g., Sentinel-2)
    dummy_input = {
        's2': torch.randn(1, 5, 10, 64, 64), # B, T, C, H, W
        's2_dates': torch.randint(0, 365, (1, 5)) # B, T
    }
    
    # patch_size is the ground size in meters (e.g., 640 m for a 64x64 S2 tile)
    patch_size = 640
    
    # Expected output shape for 'dense'
    expected_shape = (1, 64, 64, 2*768)

    try:
        # Call the method with the dictionary and patch size
        output = model.forward_features(dummy_input, patch_size=patch_size, output_type='dense', output_modality='s2') # 'dense' also requires 'output_modality'
        assert output.shape == expected_shape
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")

def test_forward_features_pass_with_flash_attn_output_type_dense(base_config):
    """Tests if the loaded model can perform a forward_features pass with flash attention without errors. Output type: dense
    Flash attention requires flash-attn package and a cuda device. 
    """
    base_config['flash_attn'] = True

    model = AnySat(base_config)
    model.half().to('cuda')

    # Create a dummy input dictionary for a single modality (e.g., Sentinel-2)
    dummy_input = {
        's2': torch.randn(1, 5, 10, 64, 64, device='cuda', dtype=torch.float16), # B, T, C, H, W
        's2_dates': torch.randint(0, 365, (1, 5), device='cuda', dtype=torch.int16) # B, T
    }
    
    # patch_size is the ground size in meters (e.g., 640 m for a 64 x 64 S2 tile)
    patch_size = 640
    
    # Expected output shape for 'dense'
    expected_shape = (1, 64, 64, 2*768)

    try:
        # Call the method with the dictionary and patch size
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.forward_features(dummy_input, patch_size=patch_size, output_type='dense', output_modality='s2') # 'dense' also requires 'output_modality'
            assert output.shape == expected_shape
            assert output.dtype == torch.float32 # A different output dtype than previously
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")

def test_forward_features_pass_no_flash_attn_output_type_all(base_config):
    """Tests if the loaded model can perform a forward_features pass without flash attention without errors. Output type: all
    """
    base_config['flash_attn'] = False

    model = AnySat(base_config)

    # Create a dummy input dictionary for a single modality (e.g., Sentinel-2)
    dummy_input = {
        's2': torch.randn(1, 5, 10, 64, 64), # B, T, C, H, W
        's2_dates': torch.randint(0, 365, (1, 5)) # B, T
    }
    
    # patch_size is the ground size in meters (e.g., 640 m for a 64x64 S2 tile)
    patch_size = 640
    
    # Expected output shape for 'all'
    expected_shape = (1, 2, 768)

    try:
        # Call the method with the dictionary and patch size
        output = model.forward_features(dummy_input, patch_size=patch_size, output_type='all')
        assert output.shape == expected_shape
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")

def test_forward_features_pass_with_flash_attn_output_type_all(base_config):
    """Tests if the loaded model can perform a forward_features pass with flash attention without errors. Output type: all
    Flash attention requires flash-attn package and a cuda device. 
    """
    base_config['flash_attn'] = True

    model = AnySat(base_config)
    model.half().to('cuda')

    # Create a dummy input dictionary for a single modality (e.g., Sentinel-2)
    dummy_input = {
        's2': torch.randn(1, 5, 10, 64, 64, device='cuda', dtype=torch.float16), # B, T, C, H, W
        's2_dates': torch.randint(0, 365, (1, 5), device='cuda', dtype=torch.int16) # B, T
    }
    
    # patch_size is the ground size in meters (e.g., 640 m for a 64 x 64 S2 tile)
    patch_size = 640
    
    # Expected output shape for 'dense'
    expected_shape = (1, 2, 768)

    try:
        # Call the method with the dictionary and patch size
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model.forward_features(dummy_input, patch_size=patch_size, output_type='all')
            assert output.shape == expected_shape
            assert output.dtype == torch.float16 # A different output dtype than previously
        
    except Exception as e:
        pytest.fail(f"Model forward_features pass failed: {e}")


# The patch path must point to where the factory function is *imported* by the wrapper
@patch('aitlas.models.anysat_wrapper.anysat_base')
@patch('aitlas.models.anysat_wrapper.hf_hub_download')
@patch('torch.load')
def test_fallback_to_huggingface_download(mock_torch_load, mock_hf_download, mock_anysat_factory, dummy_checkpoint_path, tmp_path):
    """Tests the fallback to HuggingFace download if the local path is invalid.
    """

    mock_model_instance = MagicMock()
    mock_model_instance.model = MagicMock()
    mock_anysat_factory.return_value = mock_model_instance
    
    mock_hf_download.return_value = dummy_checkpoint_path
    mock_torch_load.return_value = create_dummy_anysat_state_dict()

    destination_file = tmp_path / "model.pth"
    config = {
        "local_model_path": str(destination_file),
        "backbone_name": "anysat_base",
        "pretrained": True,
    }

    model = AnySat(config)

    mock_hf_download.assert_called_once()
    mock_torch_load.assert_called_once_with(str(destination_file), weights_only=False)
    mock_anysat_factory.assert_called_once()
    
    mock_model_instance.model.load_state_dict.assert_called_once()
    
def test_raises_error_if_pretrained_is_false(base_config):
    """Tests that a NotImplementedError is raised if pretrained is set to False.
    """
    base_config["pretrained"] = False
    
    with pytest.raises(NotImplementedError, match="Loading model without pretrained weights is not supported."):
        AnySat(base_config).load_backbone()

@pytest.mark.parametrize(
    "backbone_name",
    ["unsupported_model", "anysat_small"] # Test a completely wrong name and one without weights
)
def test_raises_error_for_invalid_backbones(backbone_name, base_config):
    """Tests that ValueErrors are raised for invalid backbones when downloading.
    """
    config = {
        "local_model_path": "/path/to/non_existent/model.pth",
        "backbone_name": backbone_name,
        "pretrained": True,
    }
    
    # Match either of the expected error messages
    with pytest.raises(ValueError, match="Unsupported or missing backbone|No pretrained weights are available for backbone"):
        AnySat(config).load_backbone()