import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from aitlas.base import BaseChangeDetection, BaseModel
from aitlas.base.metrics import SegmentationRunningScore


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

        # Attributes normally set by BaseChangeDetection.__init__
        self.use_amp = False
        self.amp_dtype = None
        self.scaler = None
        self.running_metrics = SegmentationRunningScore(self.num_classes, self.device)

        # Define a trivial conv model and assign to self.model
        self.model = nn.Conv2d(self.in_channels * 2, self.num_classes, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)

    def load_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


class EchoChangeDetection(DummyChangeDetection):
    """Returns the first input image as logits, so a test controls the prediction exactly."""

    def forward(self, x1, x2):
        return x1


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


def test_evaluate_model_binarises_soft_labels():
    # One-hot labels as produced by resizing a mask: border pixels have values between 0 and 1
    change = torch.tensor([[1.0, 0.6], [0.4, 0.0]])
    labels = torch.stack([1 - change, change])  # channels: no change, change
    # Logits whose prediction (sigmoid >= 0.5) follows the 0.5 rule exactly
    logits = torch.where(labels >= 0.5, 10.0, -10.0)
    dataloader = DataLoader([((logits, torch.zeros_like(logits)), labels)], batch_size=1)

    model = EchoChangeDetection()
    model.running_metrics.reset()
    model.evaluate_model(dataloader)

    # With labels binarised at 0.5, the prediction matches the ground truth for both classes.
    # Casting the soft labels to integers would remove the 0.6 and 0.4 pixels from both
    # classes and give an IoU below 1.
    iou_per_class = model.running_metrics.iou()["IOU per Class"]
    assert list(iou_per_class) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    "cuda_available, bf16_supported, expected",
    [(True, True, torch.bfloat16), (True, False, None), (False, False, None)],
)
def test_select_amp_dtype(monkeypatch, cuda_available, bf16_supported, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        torch.cuda, "is_bf16_supported", lambda including_emulation=True: bf16_supported
    )
    # bfloat16 when supported, otherwise None (float32). Never float16.
    assert BaseChangeDetection._select_amp_dtype() == expected


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="autocast only changes precision on a GPU"
)
def test_evaluation_runs_in_float32_with_mixed_precision():
    model = DummyChangeDetection()
    model.device = torch.device("cuda")
    model.model.to(model.device)
    model.use_amp = True
    model.amp_dtype = torch.bfloat16

    # Even inside an active float16 autocast context, evaluation must produce float32 outputs
    with torch.amp.autocast("cuda", dtype=torch.float16):
        batches = list(model.predict_output_per_batch(make_dataloader(), description="testing"))
    _, outputs, _ = batches[0]
    assert outputs.dtype == torch.float32


@pytest.mark.parametrize(
    "val_loss, best_loss, expected",
    [
        (float("nan"), None, False),
        (float("nan"), 0.5, False),
        (0.5, None, True),
        (0.4, 0.5, True),
        (0.6, 0.5, False),
        (0.0, None, True),
        (0.1, 0.0, False),
    ],
)
def test_is_new_best(val_loss, best_loss, expected):
    assert BaseModel._is_new_best(val_loss, best_loss) is expected
