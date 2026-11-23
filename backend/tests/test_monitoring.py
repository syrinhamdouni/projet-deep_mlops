
from backend.app.monitoring import PredictionMonitor


def test_prediction_distribution_drift_is_flagged_after_minimum_samples():
    monitor = PredictionMonitor({"healthy": 0.5, "diseased": 0.5})

    for _ in range(30):
        monitor.record("healthy", 0.95, 10.0)

    metrics = monitor.snapshot()

    assert metrics["prediction_count"] == 30
    assert metrics["drift_status"] == "drift_suspected"
    assert metrics["prediction_distribution_psi"] >= 0.2

    monitor.close()


def test_predictions_are_persisted_in_sqlite(tmp_path):
    db_path = tmp_path / "monitoring.db"

    monitor = PredictionMonitor(
        {"healthy": 0.5, "diseased": 0.5},
        db_path,
    )

    monitor.record("healthy", 0.95, 12.0)
    monitor.record("diseased", 0.85, 18.0)

    monitor.close()

    # Reopen the same database to verify persistence.
    reopened_monitor = PredictionMonitor(
        {"healthy": 0.5, "diseased": 0.5},
        db_path,
    )

    metrics = reopened_monitor.snapshot()

    assert metrics["prediction_count"] == 2
    assert metrics["predicted_class_counts"] == {
        "healthy": 1,
        "diseased": 1,
    }
    assert abs(metrics["mean_confidence"] - 0.90) < 1e-9
    assert abs(metrics["mean_latency_ms"] - 15.0) < 1e-9

    reopened_monitor.close()


def test_empty_monitoring_database_returns_empty_metrics(tmp_path):
    monitor = PredictionMonitor(
        {"healthy": 0.5, "diseased": 0.5},
        tmp_path / "empty.db",
    )

    metrics = monitor.snapshot()

    assert metrics["prediction_count"] == 0
    assert metrics["mean_confidence"] is None
    assert metrics["mean_latency_ms"] is None
    assert metrics["drift_status"] == "collecting"

    monitor.close()
