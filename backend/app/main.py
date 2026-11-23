from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.inference import InferenceEngine
from backend.app.monitoring import PredictionMonitor
from backend.app.schemas import HealthResponse, PredictionResponse

ROOT = Path(__file__).resolve().parents[2]
engine = InferenceEngine(ROOT / "models/model.onnx", ROOT / "models/model.json")
prediction_monitor = PredictionMonitor(
    engine.training_class_distribution,
    ROOT / "backend" / "monitoring.db",
)
app = FastAPI(title="FarmPulse Plant Disease API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if engine.loaded else "degraded",
        model_loaded=engine.loaded,
        device="cpu",
        model_version="onnx-1",
    )


@app.get("/model/metadata")
async def metadata() -> dict:
    return {
        "architecture": engine.architecture,
        "classes": engine.classes,
        "model_version": "onnx-1",
    }


@app.get("/monitoring/metrics")
async def monitoring_metrics() -> dict:
    return prediction_monitor.snapshot()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are accepted")
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded; export models/model.onnx first")
    started = time.perf_counter()
    try:
        predictions, _ = engine.predict(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    latency = (time.perf_counter() - started) * 1000
    prediction_monitor.record(
        str(predictions[0]["label"]),
        float(predictions[0]["confidence"]),
        latency,
    )
    return PredictionResponse(
        predicted_class=str(predictions[0]["label"]),
        confidence=float(predictions[0]["confidence"]),
        top_predictions=predictions,
        latency_ms=latency,
        model_version="onnx-1",
    )
