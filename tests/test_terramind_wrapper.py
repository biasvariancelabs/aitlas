from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from aitlas.models import TerraMind


EMBED_DIM = 768
BATCH_SIZE = 2
CHANNELS = 12
HEIGHT = 224
WIDTH = 224


# Mocks
class MockTerraMindBackbone(nn.Module):
    """
    Mock for the standard TerraMind backbone (feature extraction).
    """

    def __init__(self, modalities=None):
        super().__init__()
        self.head = nn.Linear(EMBED_DIM, 10)
        self.modalities = modalities

    def forward(self, d=None, **kwargs):
        """Standard forward pass returning embeddings."""
        batch_size = BATCH_SIZE
        if isinstance(d, torch.Tensor):
            batch_size = d.shape[0]
        elif isinstance(d, dict) and d:
            batch_size = next(iter(d.values())).shape[0]

        # Returns list of (B, Seq_Len, Embed_Dim)
        return [torch.randn(batch_size, 196, EMBED_DIM)]

    def load_state_dict(self, state_dict, strict=True):
        pass


class MockTerraMindGenerateBackbone(nn.Module):
    """
    Mock for the TerraMind any-to-any image generation backbone.
    """

    def __init__(self, modalities=None, output_modalities=None, pretrained=False):
        super().__init__()
        self.head = nn.Linear(EMBED_DIM, 10)
        self.modalities = modalities
        self.output_modalities = output_modalities
        self.pretrained = pretrained

    def forward(self, d=None, standardize=True, verbose=True, timesteps=None, **kwargs):
        """Generation forward pass returning image dictionary."""
        batch_size = BATCH_SIZE
        if isinstance(d, torch.Tensor):
            batch_size = d.shape[0]
        elif isinstance(d, dict) and d:
            batch_size = next(iter(d.values())).shape[0]

        # Return dict simulating generated images
        return {"S2L2A_gen": torch.randn(batch_size, CHANNELS, HEIGHT, WIDTH)}

    def load_state_dict(self, state_dict, strict=True):
        pass


class MockTerraMindTiMBackbone(nn.Module):
    """
    Mock for the TerraMind Thinking in Modalities (TiM) backbone.
    """

    def __init__(self, modalities=None, tim_modalities=None, pretrained=False):
        super().__init__()
        self.head = nn.Linear(EMBED_DIM, 10)
        self.modalities = modalities
        self.tim_modalities = tim_modalities
        self.pretrained = pretrained

    def forward(self, d=None, **kwargs):
        """TiM forward pass."""
        batch_size = BATCH_SIZE
        if isinstance(d, torch.Tensor):
            batch_size = d.shape[0]
        elif isinstance(d, dict) and d:
            batch_size = next(iter(d.values())).shape[0]

        # Simulating TiM output (wrapper expects list/tensor return)
        return [torch.randn(batch_size, 196, EMBED_DIM)]

    def load_state_dict(self, state_dict, strict=True):
        pass


# Fixtures
@pytest.fixture
def dummy_input():
    return {"S2L2A": torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)}


@pytest.fixture
def config_standard():
    """Configuration for a standard encoder."""
    return {
        "local_model_path": "/tmp/mock/TerraMind_v1_tiny.pt",
        "backbone_name": "terramind_v1_tiny",
        "pretrained": True,
        "modalities": ["S2L2A"],
    }


@pytest.fixture
def config_generate():
    """Configuration for any-to-any image generation model."""
    return {
        "local_model_path": "/tmp/mock/TerraMind_v1_tiny.pt",
        "backbone_name": "terramind_v1_tiny_generate",
        "pretrained": True,
        "modalities": ["S2L2A"],
        "output_modalities": ["S1RTC"],
    }


@pytest.fixture
def config_tim():
    """Configuration for Thinking in Modalities (TiM) model."""
    return {
        "local_model_path": "/tmp/mock/TerraMind_v1_tiny.pt",
        "backbone_name": "terramind_v1_tiny_tim",
        "pretrained": True,
        "modalities": ["S2L2A"],
        "tim_modalities": ["S1RTC"],
    }


# Tests
# Standard encoder tests
@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny",
    side_effect=MockTerraMindBackbone,
)
def test_standard_instantiation(mock_cls, mock_filter, mock_exists, mock_load, config_standard):
    """Test initializing the standard feature extraction model."""
    mock_exists.return_value = True
    mock_load.return_value = {}
    mock_filter.side_effect = lambda c, m: c

    model = TerraMind(config_standard)

    assert isinstance(model.backbone, MockTerraMindBackbone)
    assert model.backbone.modalities == ["S2L2A"]
    # Verify the head was replaced
    assert isinstance(model.backbone.head, nn.Identity)
    mock_cls.assert_called_once()
    mock_filter.assert_called_once()


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny",
    side_effect=MockTerraMindBackbone,
)
def test_forward_features_success(
    mock_cls, mock_filter, mock_exists, mock_load, config_standard, dummy_input
):
    """Test forward_features returns embeddings for standard backbone."""
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_standard)
    output = model.forward_features(dummy_input)

    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0].shape == (BATCH_SIZE, 196, EMBED_DIM)


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny",
    side_effect=MockTerraMindBackbone,
)
def test_generate_images_failure_on_standard_model(
    mock_cls, mock_filter, mock_exists, mock_load, config_standard, dummy_input
):
    """
    Test that calling generate_images on a standard backbone raises TypeError.
    """
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_standard)

    # Expect TypeError because backbone_name ("terramind_v1_tiny") does not contain "generate"
    with pytest.raises(TypeError, match="does not support image generation"):
        model.generate_images(dummy_input, timesteps=50)


# Any-to-any image generation model tests
@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn_generate")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny_generate",
    side_effect=MockTerraMindGenerateBackbone,
)
def test_generate_instantiation(mock_cls, mock_filter_gen, mock_exists, mock_load, config_generate):
    """Test initializing the any-to-any image generation model."""
    mock_exists.return_value = True
    mock_load.return_value = {}
    mock_filter_gen.side_effect = lambda c, m: c

    model = TerraMind(config_generate)

    # Verify we got the image generation backbone
    assert isinstance(model.backbone, MockTerraMindGenerateBackbone)

    # Verify constructor args specific to generation
    _, kwargs = mock_cls.call_args
    assert kwargs["output_modalities"] == ["S1RTC"]

    # Verify the specific filter function was called
    mock_filter_gen.assert_called_once()


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn_generate")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny_generate",
    side_effect=MockTerraMindGenerateBackbone,
)
def test_generate_images_success(
    mock_cls, mock_filter_gen, mock_exists, mock_load, config_generate, dummy_input
):
    """Test the generate_images method executes correctly on generate backbone."""
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_generate)

    # Look at the backbone forward method
    model.backbone.forward = MagicMock(side_effect=model.backbone.forward)

    # Perform any-to-any image generation
    result = model.generate_images(dummy_input, timesteps=50, standardize=False)

    # Assertions
    assert isinstance(result, dict)
    assert "S2L2A_gen" in result

    # Check arguments passed to backbone.forward
    model.backbone.forward.assert_called_once()
    kwargs = model.backbone.forward.call_args.kwargs
    assert kwargs["timesteps"] == 50
    assert kwargs["standardize"] is False
    assert kwargs["verbose"] is True
    # Ensure input copy was passed, not exact reference if implemented, but mock checks value
    assert isinstance(kwargs["d"], dict)


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn_generate")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny_generate",
    side_effect=MockTerraMindGenerateBackbone,
)
def test_forward_features_failure_on_generate_model(
    mock_cls, mock_filter_gen, mock_exists, mock_load, config_generate, dummy_input
):
    """
    Test that calling forward_features on a any-to-any image generation backbone raises TypeError.
    """
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_generate)

    with pytest.raises(TypeError, match="does not support feature embeddings"):
        model.forward_features(dummy_input)


# Thinking in Modalities (TiM) tests
@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn_tim")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny_tim",
    side_effect=MockTerraMindTiMBackbone,
)
def test_tim_instantiation(mock_cls, mock_filter_tim, mock_exists, mock_load, config_tim):
    """Test initializing the Thinking in Modalities (TiM) model."""
    mock_exists.return_value = True
    mock_load.return_value = {}
    mock_filter_tim.side_effect = lambda c, m: c

    model = TerraMind(config_tim)

    # Verify we got the TiM backbone
    assert isinstance(model.backbone, MockTerraMindTiMBackbone)

    # Verify constructor args specific to TiM
    _, kwargs = mock_cls.call_args
    assert kwargs["tim_modalities"] == ["S1RTC"]

    # Verify the specific filter function was called
    mock_filter_tim.assert_called_once()


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn_tim")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny_tim",
    side_effect=MockTerraMindTiMBackbone,
)
def test_thinking_in_modalities_success(
    mock_cls, mock_filter_tim, mock_exists, mock_load, config_tim, dummy_input
):
    """Test the thinking_in_modalities method executes correctly."""
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_tim)

    # Perform Thinking in Modalities
    output = model.thinking_in_modalities(dummy_input)

    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0].shape == (BATCH_SIZE, 196, EMBED_DIM)


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn_tim")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny_tim",
    side_effect=MockTerraMindTiMBackbone,
)
def test_forward_features_failure_on_tim_model(
    mock_cls, mock_filter_tim, mock_exists, mock_load, config_tim, dummy_input
):
    """
    Test that calling forward_features on a TiM backbone raises TypeError.
    """
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_tim)

    with pytest.raises(TypeError, match="does not support feature embeddings"):
        model.forward_features(dummy_input)


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny",
    side_effect=MockTerraMindBackbone,
)
def test_thinking_in_modalities_failure_on_standard_model(
    mock_cls, mock_filter, mock_exists, mock_load, config_standard, dummy_input
):
    """
    Test that calling thinking_in_modalities on a standard backbone raises TypeError.
    """
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_standard)

    with pytest.raises(TypeError, match="does not support Thinking in Modalities"):
        model.thinking_in_modalities(dummy_input)


# Shared utilities & band selection guardrails
@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny",
    side_effect=MockTerraMindBackbone,
)
@patch("aitlas.models.terramind_wrapper.select_modality_patch_embed_weights")
def test_select_input_bands_utility(
    mock_select_util, mock_cls, mock_filter, mock_exists, mock_load, config_standard
):
    """Test that band selection utility is called correctly on standard model."""
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_standard)
    mock_select_util.return_value = model.backbone  # Utility returns modified model

    bands = {"S2L2A": ["B04", "B03", "B02"]}
    model.select_input_bands(bands)

    mock_select_util.assert_called_once()
    assert mock_select_util.call_args.kwargs["bands"] == bands
    assert mock_select_util.call_args.kwargs["model"] == model.backbone


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn_generate")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny_generate",
    side_effect=MockTerraMindGenerateBackbone,
)
def test_select_input_bands_failure_on_generate(
    mock_cls, mock_filter_gen, mock_exists, mock_load, config_generate
):
    """Test that selecting bands on a generate model raises TypeError."""
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_generate)
    bands = {"S2L2A": ["B04"]}

    with pytest.raises(TypeError, match="does not support selecting input bands"):
        model.select_input_bands(bands)


@patch("aitlas.models.terramind_wrapper.torch.load")
@patch("aitlas.models.terramind_wrapper.os.path.exists")
@patch("aitlas.models.terramind_wrapper.checkpoint_filter_fn_tim")
@patch(
    "aitlas.models.terramind_wrapper.terramind_v1_tiny_tim",
    side_effect=MockTerraMindTiMBackbone,
)
def test_select_input_bands_failure_on_tim(
    mock_cls, mock_filter_tim, mock_exists, mock_load, config_tim
):
    """Test that selecting bands on a TiM model raises TypeError."""
    mock_exists.return_value = True
    mock_load.return_value = {}

    model = TerraMind(config_tim)
    bands = {"S2L2A": ["B04"]}

    with pytest.raises(TypeError, match="does not support selecting input bands"):
        model.select_input_bands(bands)
