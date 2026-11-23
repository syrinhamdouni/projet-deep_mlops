"""Assign an MLflow registry alias to a selected FarmPulse model version."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from src.utils.tracking import mlflow_tracking_uri


def promote(architecture: str, version: str, stage: str, project_root: Path) -> None:
    mlflow.set_tracking_uri(mlflow_tracking_uri(project_root))
    client = mlflow.MlflowClient()
    registered_name = f"FarmPulse-{architecture}"
    client.set_model_version_tag(registered_name, version, "stage", stage)
    client.set_registered_model_alias(registered_name, stage.lower(), version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=["baseline_cnn", "efficientnet_b0", "densenet121"],
        required=True,
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--stage", choices=["Staging", "Production"], required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    promote(**vars(args))


if __name__ == "__main__":
    main()
