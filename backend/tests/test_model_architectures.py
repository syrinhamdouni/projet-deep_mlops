from pathlib import Path

import pytest
import torch

from src.models.model import checkpoint_architecture, create_model_for_architecture


@pytest.mark.parametrize(
    "architecture",
    ["baseline_cnn", "efficientnet_b0", "densenet121"],
)
def test_classifier_architectures_return_requested_number_of_classes(architecture):
    model = create_model_for_architecture(
        architecture, num_classes=4, pretrained=False
    ).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 4)


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        create_model_for_architecture("unknown", num_classes=4, pretrained=False)


def test_legacy_baseline_checkpoint_is_detected():
    assert checkpoint_architecture({}, Path("models/baseline.pt")) == "baseline_cnn"
    assert checkpoint_architecture({}, Path("models/best.pt")) == "efficientnet_b0"
