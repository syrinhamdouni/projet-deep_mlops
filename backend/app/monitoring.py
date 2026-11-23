from __future__ import annotations

import math
import threading
from collections import Counter


class PredictionMonitor:
    def __init__(self, training_distribution: dict[str, float] | None) -> None:
        self.training_distribution = training_distribution
        self._lock = threading.Lock()
        self._count = 0
        self._confidence_sum = 0.0
        self._latency_sum = 0.0
        self._class_counts: Counter[str] = Counter()

    def record(self, label: str, confidence: float, latency_ms: float) -> None:
        with self._lock:
            self._count += 1
            self._confidence_sum += confidence
            self._latency_sum += latency_ms
            self._class_counts[label] += 1

    def snapshot(self) -> dict:
        with self._lock:
            counts = dict(self._class_counts)
            count = self._count
            confidence_sum = self._confidence_sum
            latency_sum = self._latency_sum

        result = {
            "prediction_count": count,
            "mean_confidence": confidence_sum / count if count else None,
            "mean_latency_ms": latency_sum / count if count else None,
            "predicted_class_counts": counts,
            "drift_status": "unavailable",
            "prediction_distribution_psi": None,
        }
        if not self.training_distribution:
            return result
        if count < 30:
            result["drift_status"] = "collecting"
            return result

        classes = set(self.training_distribution) | set(counts)
        epsilon = 1e-6
        smoothing = epsilon * len(classes)
        psi = 0.0
        for class_name in classes:
            expected = (self.training_distribution.get(class_name, 0.0) + epsilon) / (
                1.0 + smoothing
            )
            observed = (counts.get(class_name, 0) + epsilon) / (
                count + smoothing
            )
            psi += (observed - expected) * math.log(observed / expected)
        result["prediction_distribution_psi"] = psi
        result["drift_status"] = "drift_suspected" if psi >= 0.2 else "stable"
        return result
