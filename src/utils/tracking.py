from pathlib import Path
import hashlib

import mlflow
import torch


def mlflow_tracking_uri(project_root: Path) -> str:
    database_path = (project_root / "mlflow.db").resolve().as_posix()
    return f"sqlite:///{database_path}"


def dataset_version(processed_dir: Path) -> str:
    project_root = processed_dir.resolve().parent.parent
    digest = hashlib.sha256()
    dvc_metadata = project_root / "data" / "raw.dvc"
    if dvc_metadata.exists():
        digest.update(dvc_metadata.read_bytes())

    for name in ("classes.json", "train.csv", "val.csv", "test.csv"):
        manifest = processed_dir / name
        if not manifest.exists():
            raise FileNotFoundError(f"Required dataset manifest is missing: {manifest}")
        digest.update(name.encode("utf-8"))
        digest.update(manifest.read_bytes())

    return f"sha256:{digest.hexdigest()}"


def register_pytorch_model(
    model, architecture: str, data_version: str, image_size: int = 224
) -> str:
    input_example = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        output_example = model(input_example)
    signature = mlflow.models.infer_signature(
        input_example.numpy(), output_example.numpy()
    )
    model_info = mlflow.pytorch.log_model(
        model,
        name="model",
        tags={"architecture": architecture, "dataset_version": data_version},
        input_example=input_example,
        signature=signature,
        serialization_format="pt2",
    )
    registered_name = f"FarmPulse-{architecture}"
    version = mlflow.register_model(model_info.model_uri, registered_name)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(registered_name, version.version, "stage", "Staging")
    client.set_model_version_tag(
        registered_name, version.version, "dataset_version", data_version
    )
    client.set_registered_model_alias(
        registered_name, "staging", str(version.version)
    )
    return f"{registered_name}/versions/{version.version}"
