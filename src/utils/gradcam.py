"""Grad-CAM for EfficientNet-B0 explanations."""

from __future__ import annotations

import torch
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _inputs, output) -> None:
        self.gradients = output[0].detach()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[:, class_index].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1).relu()
        heatmap = heatmap / (heatmap.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return heatmap
