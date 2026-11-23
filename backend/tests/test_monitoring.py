from backend.app.monitoring import PredictionMonitor


def test_prediction_distribution_drift_is_flagged_after_minimum_samples():
    monitor = PredictionMonitor({"healthy": 0.5, "diseased": 0.5})
    for _ in range(30):
        monitor.record("healthy", 0.95, 10.0)

    metrics = monitor.snapshot()
    assert metrics["prediction_count"] == 30
    assert metrics["drift_status"] == "drift_suspected"
    assert metrics["prediction_distribution_psi"] >= 0.2
