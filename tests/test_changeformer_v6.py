import pytest
import torch

from aitlas.models import ChangeFormerV6


@pytest.fixture(scope="module")
def changeformer_v6_model():
    """
    A pytest fixture that instantiates the ChangeFormerV6 once per test module.

    Using `scope="module"` is efficient as it prevents reloading the model
    for every single test function.
    """
    mock_config = {
        "pretrained": True,
        "num_classes": 2,
        # "in_channels": 3 not implemented yet (would have to add the field to the schema)
    }
    try:
        model = ChangeFormerV6(mock_config)
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        pytest.fail(f"Failed to instantiate ChangeFormerV6: {e}")


def test_changeformer_v6_forward_pass(changeformer_v6_model):
    """
    Test the forward pass of the ChangeFormerV6 model.
    """
    batch_size = 2
    channels = 3
    height = 256
    width = 256

    # Create dummy input tensors representing two images
    img1 = torch.randn(batch_size, channels, height, width)
    img2 = torch.randn(batch_size, channels, height, width)

    try:
        outputs = changeformer_v6_model(img1, img2)
        assert (
            outputs.shape
            == (
                batch_size,
                changeformer_v6_model.num_classes,
                height,
                width,
            )
        ), f"Expected output shape {(batch_size, changeformer_v6_model.num_classes, height, width)}, but got {outputs.shape}"
    except Exception as e:
        pytest.fail(f"Forward pass failed: {e}")
