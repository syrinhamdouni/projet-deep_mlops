"""Small CNN baseline used for a reproducible architecture comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.train import ManifestDataset, class_distribution, evaluate
from src.models.model import BaselineCNN
from src.utils.tracking import (
    dataset_version,
    mlflow_tracking_uri,
    register_pytorch_model,
)


def train_baseline(
    processed_dir: Path = Path("data/processed"),
    model_dir: Path = Path("models"),
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 224,
) -> dict[str, float]:
    classes = json.loads((processed_dir / "classes.json").read_text(encoding="utf-8"))
    train_dataset = ManifestDataset(processed_dir / "train.csv", image_size, True)
    train_distribution = class_distribution(train_dataset.rows, classes)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ManifestDataset(processed_dir / "val.csv", image_size, False),
        batch_size=batch_size,
    )
    model = BaselineCNN(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "baseline.pt"
    metrics_path = model_dir / "baseline_metrics.json"
    mlflow.set_tracking_uri(mlflow_tracking_uri(processed_dir.resolve().parent.parent))
    with mlflow.start_run(run_name="baseline_cnn"):
        data_version = dataset_version(processed_dir)
        mlflow.log_params(
            {
                "architecture": "baseline_cnn",
                "dataset_version": data_version,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "num_classes": len(classes),
                "image_size": image_size,
                "device": "cpu",
            }
        )
        training_started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            sample_count = 0
            for images, labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                sample_count += labels.size(0)
            mlflow.log_metric("train_loss", total_loss / sample_count, step=epoch)
        metrics = evaluate(model, val_loader, torch.device("cpu"))
        mlflow.log_metrics(metrics)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "classes": classes,
                "architecture": "baseline_cnn",
                "image_size": image_size,
                "training_class_distribution": train_distribution,
                "validation_metrics": metrics,
            },
            checkpoint_path,
        )
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_metric("training_time_seconds", time.perf_counter() - training_started)
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(metrics_path))
        register_pytorch_model(model.eval(), "baseline_cnn", data_version, image_size)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    print(train_baseline(**vars(args)))


if __name__ == "__main__":
    main()
