
from __future__ import annotations

import math
import sqlite3
import threading
from collections import Counter
from pathlib import Path


class PredictionMonitor:
    def __init__(
        self,
        training_distribution: dict[str, float] | None,
        db_path: str | Path | None = None,
    ) -> None:
        self.training_distribution = training_distribution
        self._lock = threading.Lock()

        # None = temporary in-memory database (useful for tests).
        self.db_path = str(db_path) if db_path is not None else ":memory:"

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30,
        )

        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    latency_ms REAL NOT NULL
                )
                """
            )
            self._connection.commit()

    def record(self, label: str, confidence: float, latency_ms: float) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO predictions (label, confidence, latency_ms)
                VALUES (?, ?, ?)
                """,
                (label, confidence, latency_ms),
            )
            self._connection.commit()

    def snapshot(self) -> dict:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT label, confidence, latency_ms
                FROM predictions
                """
            ).fetchall()

        count = len(rows)
        counts = Counter(row[0] for row in rows)
        confidence_sum = sum(row[1] for row in rows)
        latency_sum = sum(row[2] for row in rows)

        result = {
            "prediction_count": count,
            "mean_confidence": confidence_sum / count if count else None,
            "mean_latency_ms": latency_sum / count if count else None,
            "predicted_class_counts": dict(counts),
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
            expected = (
                self.training_distribution.get(class_name, 0.0) + epsilon
            ) / (1.0 + smoothing)

            observed = (
                counts.get(class_name, 0) + epsilon
            ) / (count + smoothing)

            psi += (observed - expected) * math.log(observed / expected)

        result["prediction_distribution_psi"] = psi
        result["drift_status"] = (
            "drift_suspected" if psi >= 0.2 else "stable"
        )

        return result

    def close(self) -> None:
        with self._lock:
            self._connection.close()
