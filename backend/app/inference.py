from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class InferenceEngine:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.session = None
        self.classes: list[str] = []
        self.image_size = 224
        self.architecture = "unavailable"
        self.training_class_distribution = None
        if model_path.exists() and metadata_path.exists():
            import onnxruntime as ort

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.classes = metadata["classes"]
            self.image_size = int(metadata.get("image_size", 224))
            self.architecture = metadata.get("architecture", "efficientnet_b0")
            self.training_class_distribution = metadata.get(
                "training_class_distribution"
            )
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def predict(self, image_bytes: bytes, top_k: int = 3) -> tuple[list[dict[str, float | str]], float]:
        if not self.loaded:
            raise RuntimeError("Model artifact is not loaded")
        try:
            image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a valid image") from exc
        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        logits = self.session.run(None, {"image": np.transpose(array, (2, 0, 1))[None, ...]})[0][0]
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        indexes = np.argsort(probabilities)[::-1][:top_k]
        elapsed = time.perf_counter()
        result = [
            {"label": self.classes[index], "confidence": float(probabilities[index])}
            for index in indexes
        ]
        return result, (time.perf_counter() - elapsed) * 1000
