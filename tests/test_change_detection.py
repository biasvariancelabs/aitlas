import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from aitlas.base import BaseChangeDetection


# ---- Dummy subclass for testing ----
class DummyChangeDetection(BaseChangeDetection):
    schema = None  # not used

    def __init__(self):
        # Initialize nn.Module
        nn.Module.__init__(self)

        # Minimal config object with threshold
        class C:
            pass

        self.config = C()
        self.config.threshold = 0.5
        self.learning_rate = 1e-3

        # Attributes used in tests
        self.num_classes = 2
        self.in_channels = 3
        self.device = torch.device("cpu")

        # Define a trivial conv model and assign to self.model
        self.model = nn.Conv2d(self.in_channels * 2, self.num_classes, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)

    def load_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


@pytest.fixture(scope="module")
def dummy_model():
    # Instantiate without config (your DummyChangeDetection sets defaults)
    return DummyChangeDetection()


def make_dataloader(batch_size=2, num_samples=4, channels=3, height=32, width=32, num_classes=2):
    # Create dummy dataset: pairs of images and one-hot masks
    img1 = torch.randn(num_samples, channels, height, width)
    img2 = torch.randn(num_samples, channels, height, width)
    masks = torch.randint(0, num_classes, (num_samples, height, width))
    masks_onehot = nn.functional.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dataset = [((img1[i], img2[i]), masks_onehot[i]) for i in range(num_samples)]
    return DataLoader(dataset, batch_size=batch_size)


# ---- Tests ----


def test_forward(dummy_model):
    x1 = torch.randn(2, 3, 32, 32)
    x2 = torch.randn(2, 3, 32, 32)
    outputs = dummy_model.forward(x1, x2)
    assert outputs.shape == (2, dummy_model.num_classes, 32, 32)


def test_get_predicted(dummy_model):
    outputs = torch.randn(2, 2, 32, 32)  # logits
    probs, preds = dummy_model.get_predicted(outputs)
    assert probs.shape == preds.shape
    assert preds.dtype == torch.long
    # Check thresholding works
    assert ((probs >= 0.5).long() == preds).all()


def test_predict_output_per_batch(dummy_model):
    dataloader = make_dataloader()
    batches = list(dummy_model.predict_output_per_batch(dataloader, description="testing"))
    assert len(batches) > 0
    inputs, outputs, labels = batches[0]
    img1, img2 = inputs
    assert outputs.shape[0] == labels.shape[0]


def test_train_epoch(dummy_model):
    dataloader = make_dataloader()
    optimizer = dummy_model.load_optimizer()
    criterion = nn.CrossEntropyLoss()
    loss = dummy_model.train_epoch(0, dataloader, optimizer, criterion, iterations_log=1)
    assert isinstance(loss, float)
    assert loss >= 0.0
