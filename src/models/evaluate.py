"""Evaluate a trained checkpoint on the held-out test manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.models.model import checkpoint_architecture, create_model_for_architecture
from src.models.train import ManifestDataset
from src.utils.tracking import dataset_version, mlflow_tracking_uri


def evaluate(
    checkpoint: Path,
    processed_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    image_size: int = 224,
    mlflow_uri: str | None = None,
    split: str = "test",
) -> dict:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
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
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    loader = DataLoader(
        ManifestDataset(processed_dir / f"{split}.csv", image_size, False),
        batch_size=batch_size,
    )
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions.extend(model(images).argmax(dim=1).tolist())
            targets.extend(labels.tolist())
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "f1_weighted": f1_score(targets, predictions, average="weighted", zero_division=0),
        "classification_report": classification_report(
            targets, predictions, target_names=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{architecture}_{split}_metrics.json"
    if architecture == "efficientnet_b0":
        output_name = f"{split}_metrics.json"
    output = output_dir / output_name
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    mlflow.set_tracking_uri(
        mlflow_uri or mlflow_tracking_uri(checkpoint.resolve().parent.parent)
    )
    with mlflow.start_run(run_name=f"{architecture}-{split}-evaluation"):
        mlflow.log_params(
            {
                "architecture": architecture,
                "dataset_version": dataset_version(processed_dir),
                "split": split,
            }
        )
        mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, float)})
        mlflow.log_artifact(str(output))
        mlflow.log_artifact(str(checkpoint))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mlflow-uri")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    args = parser.parse_args()
    print(evaluate(**vars(args)))


if __name__ == "__main__":
    main()
