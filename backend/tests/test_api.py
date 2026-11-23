import asyncio

from backend.app.main import health, monitoring_metrics, predict
from fastapi import UploadFile
from io import BytesIO


def test_health_contract():
    response = asyncio.run(health())
    assert response.status in {"ok", "degraded"}
    assert response.model_version


def test_rejects_non_image_upload():
    upload = UploadFile(BytesIO(b"not an image"), filename="sample.txt")
    try:
        asyncio.run(predict(upload))
    except Exception as error:
        assert getattr(error, "status_code", None) == 400
    else:
        raise AssertionError("Invalid content type should be rejected")


def test_monitoring_endpoint_reports_empty_state():
    response = asyncio.run(monitoring_metrics())
    assert response["prediction_count"] >= 0
