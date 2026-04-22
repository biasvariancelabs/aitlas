import pytest
import torch

from aitlas.models import BIT


@pytest.fixture(scope="module")
def bit_model():
    """
    A pytest fixture that instantiates the BIT once per test module.

    Using `scope="module"` is efficient as it prevents reloading the model
    for every single test function.
    """
    mock_config = {
        "pretrained": True,
        "num_classes": 2,
        # "in_channels": 3 not implemented yet (would have to add the field to the schema)
    }
    try:
        model = BIT(mock_config)
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        pytest.fail(f"Failed to instantiate BIT: {e}")


def test_bit_forward_pass(bit_model):
    """
    Test the forward pass of the BIT model.
    """
    batch_size = 2
    channels = 3
    height = 256
    width = 256

    # Create dummy input tensors representing two images
    img1 = torch.randn(batch_size, channels, height, width)
    img2 = torch.randn(batch_size, channels, height, width)

    try:
        outputs = bit_model(img1, img2)
        assert outputs.shape == (
            batch_size,
            bit_model.num_classes,
            height,
            width,
        ), f"Expected output shape {(batch_size, bit_model.num_classes, height, width)}, but got {outputs.shape}"
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")
