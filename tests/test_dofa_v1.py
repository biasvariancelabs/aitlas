import pytest
import torch

from aitlas.models import DOFA_v1

@pytest.fixture(scope="module")
def dofa_model():
    """
    A pytest fixture that instantiates the DOFA_v1 once per test module.
    
    Using `scope="module"` is efficient as it prevents reloading the model
    for every single test function.
    """
    mock_config = {
        "backbone_name": "vit_base_dofa",
        "pretrained": True
    }
    try:
        model = DOFA_v1(mock_config)
        model.eval()  # Set the model to evaluation mode
        print(model)
        return model
    except Exception as e:
        pytest.fail(f"Failed to instantiate DOFA_v1: {e}")

@pytest.fixture
def dummy_data():
    """
    A pytest fixture that provides a dummy image tensor and corresponding
    wavelengths for a single 3-channel RGB image.
    """
    batch_size = 1
    num_channels = 3
    image_size = 224
    image_tensor = torch.rand(batch_size, num_channels, image_size, image_size)
    
    # Wavelengths for a standard 3-channel RGB image.
    wavelengths = [0.665, 0.56, 0.49]
    
    return image_tensor, wavelengths

def test_dofa_forward_pass(dofa_model, dummy_data):
    """
    Tests the forward pass of the DOFA_v1 to ensure it produces an
    embedding of the correct shape.
    """
    # Arrange: Get the model and data from our fixtures
    image_tensor, wavelengths = dummy_data
    
    # Act: Perform the forward pass
    with torch.no_grad():
        embedding = dofa_model(image_tensor, wave_list=wavelengths)
        
    # Assert: Check if the output has the expected shape
    # For ViT-Base, the embedding dimension is 768.
    batch_size = image_tensor.shape[0]
    expected_shape = (batch_size, 768)
    
    assert embedding.shape == expected_shape, \
        f"Output shape is incorrect. Expected {expected_shape}, but got {embedding.shape}"

def test_dofa_model_has_backbone(dofa_model):
    """
    Tests that the DOFA_v1 instance has a 'backbone' attribute
    which is a torch.nn.Module.
    """
    assert hasattr(dofa_model, 'backbone'), "Model does not have a 'backbone' attribute."
    assert isinstance(dofa_model.backbone, torch.nn.Module), "'backbone' is not a torch.nn.Module."