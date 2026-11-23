from __future__ import annotations

import torch.nn as nn
import torch
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    densenet121,
    efficientnet_b0,
)


def checkpoint_architecture(payload: dict, checkpoint_path) -> str:
    architecture = payload.get("architecture")
    if architecture is not None:
        return architecture
    if checkpoint_path.name.lower() == "baseline.pt":
        return "baseline_cnn"
    return "efficientnet_b0"


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def create_model(num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def create_densenet121(
    num_classes: int, pretrained: bool = True, dropout: float = 0.3
) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


def create_model_for_architecture(
    architecture: str,
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    if architecture == "baseline_cnn":
        return BaselineCNN(num_classes)
    if architecture == "efficientnet_b0":
        return create_model(num_classes, pretrained, dropout)
    if architecture == "densenet121":
        return create_densenet121(num_classes, pretrained, dropout)
    raise ValueError(f"Unsupported architecture: {architecture}")
