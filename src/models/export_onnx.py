from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.model import checkpoint_architecture, create_model_for_architecture


def export(checkpoint: Path, output: Path, opset: int = 17) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    classes = payload["classes"]
    architecture = checkpoint_architecture(payload, checkpoint)
    model = create_model_for_architecture(
        architecture,
        len(classes),
        pretrained=False,
        dropout=payload.get("dropout", 0.3),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, payload.get("image_size", 224), payload.get("image_size", 224))
    torch.onnx.export(
        model, dummy, output, input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "architecture": architecture,
                "classes": classes,
                "image_size": payload.get("image_size", 224),
                "training_class_distribution": payload.get(
                    "training_class_distribution"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/model.onnx"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
