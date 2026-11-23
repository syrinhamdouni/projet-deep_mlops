from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[Prediction]
    latency_ms: float = Field(ge=0)
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_version: str
