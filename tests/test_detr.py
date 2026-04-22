import pytest
import torch

from aitlas.models import DETR


@pytest.fixture(scope="module")
def detr_model():
    """
    A pytest fixture that instantiates the DETR once per test module.
    """
    mock_config = {
        "pretrained": True,
        "num_classes": 20,
    }
    try:
        # We need to make sure the model is on CPU for testing purposes if no GPU
        model = DETR(mock_config)
        return model
    except Exception as e:
        pytest.fail(f"Failed to instantiate DETR: {e}")


def test_detr_eval_forward_pass(detr_model):
    """
    Test the forward pass of the DETR model in evaluation mode.
    """
    detr_model.model.eval()
    batch_size = 2
    channels = 3
    height = 512
    width = 512

    # Create dummy input images as a list (Aitlas format)
    inputs = [torch.randn(channels, height, width) for _ in range(batch_size)]

    try:
        # In eval mode, it should return a list of dictionaries
        outputs = detr_model(inputs)

        assert isinstance(outputs, list), "Output should be a list"
        assert (
            len(outputs) == batch_size
        ), f"Expected {batch_size} outputs, got {len(outputs)}"

        for output in outputs:
            assert isinstance(output, dict), "Each output should be a dictionary"
            assert "boxes" in output
            assert "scores" in output
            assert "labels" in output

    except Exception as e:
        pytest.fail(f"Eval forward pass failed: {e}")


def test_detr_train_forward_pass(detr_model):
    """
    Test the forward pass of the DETR model in training mode.
    """
    detr_model.model.train()
    batch_size = 2
    channels = 3
    height = 512
    width = 512

    # Create dummy input images as a list (Aitlas format)
    inputs = [torch.randn(channels, height, width) for _ in range(batch_size)]

    # Create dummy targets (Aitlas format: list of dicts with 'boxes' and 'labels')
    targets = []
    for _ in range(batch_size):
        num_objs = 3
        boxes = torch.tensor(
            [[10, 10, 50, 50], [100, 100, 150, 150], [200, 200, 300, 300]],
            dtype=torch.float32,
        )
        labels = torch.randint(0, 20, (num_objs,))
        targets.append({"boxes": boxes, "labels": labels})

    try:
        # In training mode, it should return a dictionary of losses
        outputs = detr_model(inputs, targets)

        assert isinstance(outputs, dict), "Output should be a dictionary of losses"
        assert "classification_loss" in outputs
        assert "regression_loss" in outputs

        # Check if losses are scalars
        assert outputs["classification_loss"].dim() == 0 or (
            outputs["classification_loss"].dim() == 1
            and outputs["classification_loss"].shape[0] == 1
        )
        assert outputs["regression_loss"].dim() == 0 or (
            outputs["regression_loss"].dim() == 1
            and outputs["regression_loss"].shape[0] == 1
        )

    except Exception as e:
        pytest.fail(f"Training forward pass failed: {e}")
