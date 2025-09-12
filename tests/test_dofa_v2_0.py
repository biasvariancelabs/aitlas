# This is a simple test file for the DOFA_v2 model
import pytest
import torch
from aitlas.models import DOFA_v2


@pytest.fixture(scope="module")
def dofa_model_instance():

    # Create the mock config
    mock_config = {
        "local_model_path": "/home/dragik/tadej/aitlas/tests/checkpoints/DOFA_ViT_base_e100.pth",
        "backbone_name": "vit_base_patch16",
        "pretrained": True
    }

    try:
        model = DOFA_v2(mock_config)
        #print(model)
        return model
    except Exception as e:
        pytest.fail(f"Failed to instantiate DOFA_v2: {e}")


def test_dofa_model_is_correct_instance(dofa_model_instance):
    """
    Test to ensure the fixture successfully provides a DOFA_v2 instance.
    """
    print(f"Model instance successfully created!")
    # The fixture already handles instantiation, so this test just verifies the type.
    assert isinstance(dofa_model_instance, DOFA_v2), "The model instance is not of type DOFA_v2"