"""Train EfficientNet-B0 and track the run in MLflow."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import mlflow
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.models.model import create_model_for_architecture
from src.utils.tracking import (
    dataset_version,
    mlflow_tracking_uri,
    register_pytorch_model,
)


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, image_size: int, train: bool) -> None:
        import csv

        with manifest.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["path"]).convert("RGB") as image:
            return self.transform(image), int(row["label"])


def class_distribution(rows: list[dict[str, str]], classes: list[str]) -> dict[str, float]:
    counts = Counter(int(row["label"]) for row in rows)
    total = sum(counts.values())
    return {
        class_name: counts.get(label, 0) / total
        for label, class_name in enumerate(classes)
    }


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    return {
        "accuracy": accuracy_score(targets, predictions),
        "f1_weighted": f1_score(targets, predictions, average="weighted", zero_division=0),
    }


def train(args: argparse.Namespace) -> dict[str, float]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processed = Path(args.processed_dir)
    classes = json.loads((processed / "classes.json").read_text(encoding="utf-8"))
    architecture = getattr(args, "architecture", "efficientnet_b0")
    if architecture not in {"efficientnet_b0", "densenet121"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ManifestDataset(processed / "train.csv", args.image_size, True)
    train_distribution = class_distribution(train_dataset.rows, classes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ManifestDataset(processed / "val.csv", args.image_size, False),
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    model = create_model_for_architecture(
        architecture, len(classes), args.pretrained, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    metrics: dict[str, float] = {}
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / (
        "best.pt" if architecture == "efficientnet_b0" else f"{architecture}_best.pt"
    )
    metrics_path = model_dir / (
        "metrics.json" if architecture == "efficientnet_b0" else f"{architecture}_metrics.json"
    )
    mlflow.set_tracking_uri(
        args.mlflow_uri or mlflow_tracking_uri(processed.resolve().parent.parent)
    )
    with mlflow.start_run(run_name=architecture):
        mlflow.log_params(
            {key: value for key, value in vars(args).items() if value is not None}
        )
        mlflow.log_param("architecture", architecture)
        data_version = dataset_version(processed)
        mlflow.log_param("dataset_version", data_version)
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("loss_function", "CrossEntropyLoss")
        mlflow.log_param("num_classes", len(classes))
        mlflow.log_param("device", str(device))
        training_started = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            sample_count = 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(images.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                sample_count += labels.size(0)
            mlflow.log_metric("train_loss", total_loss / sample_count, step=epoch)
            metrics = evaluate(model, val_loader, device)
            mlflow.log_metrics(metrics, step=epoch)
            if metrics["f1_weighted"] > best_f1:
                best_f1 = metrics["f1_weighted"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "image_size": args.image_size,
                        "architecture": architecture,
                        "dropout": args.dropout,
                        "training_class_distribution": train_distribution,
                        "validation_metrics": metrics,
                    },
                    checkpoint_path,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        mlflow.log_metric("training_time_seconds", time.perf_counter() - training_started)
        metrics_path.write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(checkpoint_path))
        mlflow.log_artifact(str(metrics_path))
        best_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(best_checkpoint["state_dict"])
        register_pytorch_model(
            model.eval().cpu(), architecture, data_version, args.image_size
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        nargs="?",
        const=True,
        type=parse_bool,
    )
    pretrained_group.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false"
    )
    parser.set_defaults(pretrained=True)
    parser.add_argument(
        "--architecture",
        choices=["efficientnet_b0", "densenet121"],
        default="efficientnet_b0",
    )
    parser.add_argument("--mlflow-uri")
    print(train(parser.parse_args()))


if __name__ == "__main__":
    main()
