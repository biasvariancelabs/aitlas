import pytest
import torch

from aitlas.models import SiamCRNN


@pytest.fixture(scope="module")
def siamcrnn_model():
    """
    A pytest fixture that instantiates the SiamCRNN once per test module.

    Using `scope="module"` is efficient as it prevents reloading the model
    for every single test function.
    """
    mock_config = {
        "pretrained": True,
        "num_classes": 2,
        # "in_channels": 3 not implemented yet (would have to add the field to the schema)
    }
    try:
        model = SiamCRNN(mock_config)
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        pytest.fail(f"Failed to instantiate SiamCRNN: {e}")


def test_siamcrnn_forward_pass(siamcrnn_model):
    """
    Test the forward pass of the SiamCRNN model.
    """
    batch_size = 2
    channels = 3
    height = 256
    width = 256

    # Create dummy input tensors representing two images
    img1 = torch.randn(batch_size, channels, height, width)
    img2 = torch.randn(batch_size, channels, height, width)

    try:
        outputs = siamcrnn_model(img1, img2)
        assert outputs.shape == (
            batch_size,
            siamcrnn_model.num_classes,
            height,
            width,
        ), f"Expected output shape {(batch_size, siamcrnn_model.num_classes, height, width)}, but got {outputs.shape}"
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")
