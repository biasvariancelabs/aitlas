import pytest
import torch
import os
from unittest.mock import patch, MagicMock

from aitlas.models.presto_wrapper import Presto, PrestoSchema
from aitlas.models.Presto.utils import prepare_presto_input, INPUT_PRESTO_S2_BANDS
from aitlas.models.Presto.presto import PrestoModel, presto_default

# Fixtures
def create_dummy_presto_state_dict():
    """Creates a state dict for a dummy Presto model."""
    return PrestoModel.construct().state_dict()

@pytest.fixture(scope="session")
def dummy_presto_checkpoint(tmp_path_factory):
    """Creates a temporary checkpoint file for the Presto model."""
    path = tmp_path_factory.mktemp("checkpoints") / "model-bfa691d3.pth"
    torch.save(create_dummy_presto_state_dict(), path)
    return str(path)

@pytest.fixture
def config_presto(dummy_presto_checkpoint):
    """Provides a valid DICTIONARY config for the Presto model."""
    return {
        "local_model_path": dummy_presto_checkpoint,
        "backbone_name": "presto_default",
        "pretrained": True,
        "month": 6,
    }

@pytest.fixture
def dummy_input_list():
    """Creates a valid input list of dicts with both S1 and S2 data."""
    batch_size, T, H, W = 2, 12, 8, 8
    C_s1 = 2 # Sentinel-1
    C_s2 = 10 # Sentinel-2
    
    return [
        {
            "s1": torch.randn(T, C_s1, H, W),
            "s2": torch.randn(T, C_s2, H, W),
            "latlons": torch.randn(2, H, W),
            "dynamic_world": torch.randint(0, 9, (T, H, W))
        }
        for _ in range(batch_size)
    ]

# Tests
# Test the utility function
class TestPreparePrestoInputs:
    def test_happy_path_with_s1_and_s2(self):
        """Tests the utility with both S1 and S2 data to check merging."""
        b, t, h, w = 2, 12, 8, 8
        s1 = torch.randn(b, t, 2, h, w)
        s2 = torch.randn(b, t, 10, h, w)
        latlons = torch.randn(b, 2, h, w)

        x, dw, lats, mnths = prepare_presto_input(s1=s1, s2=s2, latlons=latlons)

        assert x.shape == (b, t, 17, h, w)
        assert not torch.all(x[:, :, 0:2, :, :] == 0)
        assert not torch.all(x[:, :, 2:12, :, :] == 0)

    def test_defaults(self):
        """Tests if defaults for dynamic_world and months are applied correctly."""
        b, t, h, w = 2, 12, 8, 8
        s1 = torch.randn(b, t, 2, h, w)
        latlons = torch.randn(b, 2, h, w)
        default_month = 5

        _, dw, _, mnths = prepare_presto_input(s1=s1, latlons=latlons, default_month=default_month)

        assert torch.all(dw == 9)
        assert torch.all(mnths == torch.fmod(torch.arange(5, 5 + t, dtype=torch.long), 12))

# Test the Presto wrapper class
class TestPrestoWrapper:
    
    # Integration tests
    def test_instantiation(self, config_presto):
        """Tests real instantiation and model loading from a local file."""
        model = Presto(config_presto)
        assert isinstance(model, Presto)
        assert isinstance(model.backbone, PrestoModel)

    def test_forward_pass_happy_path(self, config_presto, dummy_input_list):
        """Tests a real forward pass with both S1 and S2 data."""
        B, D, H, W = 2, 128, 8, 8
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = Presto(config_presto).to(device)

        for sample in dummy_input_list:
            for key, tensor in sample.items():
                if isinstance(tensor, torch.Tensor):
                    sample[key] = tensor.to(device)
        
        output = model.forward_features(dummy_input_list)
        
        assert output.shape == (B, D, H, W)
        assert output.device.type == device

    # Unit tests
    @patch('aitlas.models.presto_wrapper.presto_default')
    @patch('aitlas.models.presto_wrapper.hf_hub_download')
    @patch('aitlas.models.presto_wrapper.torch.load')
    def test_hf_download_fallback(self, mock_torch_load, mock_hf_download, mock_presto_default, config_presto):
        """Tests the download fallback logic in isolation."""
        mock_model_instance = MagicMock()
        mock_presto_default.return_value = mock_model_instance
        mock_torch_load.return_value = create_dummy_presto_state_dict()
        
        config_presto['local_model_path'] = "/non/existent/path.pth"
        mock_hf_download.return_value = "/non/existent/path.pth"
        
        Presto(config_presto)
        
        mock_hf_download.assert_called_once()
        mock_torch_load.assert_called_with(mock_hf_download.return_value, weights_only=False)

    def test_forward_pass_empty_input(self, config_presto):
        """Tests the error case for an empty input list."""
        model = Presto(config_presto)
        with pytest.raises(ValueError, match="Input list cannot be empty"):
            model.forward_features([])

    def test_forward_pass_raises_error_if_latlons_missing(self, config_presto, dummy_input_list):
        """Tests that a ValueError is raised if the latlons key is missing."""
        for sample in dummy_input_list:
            del sample['latlons']
            
        model = Presto(config_presto)
        
        with pytest.raises(ValueError, match="`latlons` tensor is required"):
            model.forward_features(dummy_input_list)