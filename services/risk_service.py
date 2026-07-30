"""
ProjectIQ Risk Service
-----------------------
Serves the trained TensorFlow risk model behind a FastAPI app. Predicts
expected cost overrun % and schedule delay (days) for a construction
project given its planning-stage features.

Run standalone:
    uvicorn services.risk_service:app --reload --port 8002
"""
import os

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "ml", "model")

app = FastAPI(title="ProjectIQ Risk Service", version="1.0.0")

# Allows the standalone frontend (opened as a local file or hosted separately,
# e.g. GitHub Pages) to call this API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_preprocessor = None
_feature_spec = None


def _load_artifacts():
    global _model, _preprocessor, _feature_spec
    if _model is None:
        _model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "risk_model.keras"))
        _preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
        _feature_spec = joblib.load(os.path.join(MODEL_DIR, "feature_spec.joblib"))
    return _model, _preprocessor, _feature_spec


VALID_PROJECT_TYPES = ["Residential", "Commercial", "Industrial", "Infrastructure"]


class ProjectFeatures(BaseModel):
    project_type: str = Field(..., description=f"One of: {VALID_PROJECT_TYPES}")
    size_sqm: float = Field(..., gt=0)
    planned_budget_aed: float = Field(..., gt=0)
    planned_duration_days: float = Field(..., gt=0)
    num_suppliers: int = Field(..., ge=0)
    supplier_reliability_score: float = Field(..., ge=0, le=1, description="0-1 historical on-time delivery rate")
    num_subcontractors: int = Field(0, ge=0)
    design_change_orders: int = Field(0, ge=0)
    weather_risk_days: int = Field(0, ge=0)
    site_congestion_score: float = Field(0.5, ge=0, le=1)
    equipment_utilization_pct: float = Field(70.0, ge=0, le=100)
    permit_delay_days: int = Field(0, ge=0)
    labor_turnover_pct: float = Field(15.0, ge=0, le=100)


class RiskResponse(BaseModel):
    predicted_cost_overrun_pct: float
    predicted_delay_days: float
    predicted_final_cost_aed: float
    predicted_final_duration_days: float
    risk_level: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=RiskResponse)
def predict(features: ProjectFeatures):
    if features.project_type not in VALID_PROJECT_TYPES:
        raise HTTPException(400, f"Unknown project_type. Valid: {VALID_PROJECT_TYPES}")

    model, preprocessor, spec = _load_artifacts()

    import pandas as pd

    row = pd.DataFrame(
        [
            {
                "size_sqm": features.size_sqm,
                "planned_budget_aed": features.planned_budget_aed,
                "planned_duration_days": features.planned_duration_days,
                "num_suppliers": features.num_suppliers,
                "supplier_reliability_score": features.supplier_reliability_score,
                "num_subcontractors": features.num_subcontractors,
                "design_change_orders": features.design_change_orders,
                "weather_risk_days": features.weather_risk_days,
                "site_congestion_score": features.site_congestion_score,
                "equipment_utilization_pct": features.equipment_utilization_pct,
                "permit_delay_days": features.permit_delay_days,
                "labor_turnover_pct": features.labor_turnover_pct,
                "project_type": features.project_type,
            }
        ]
    )
    X = preprocessor.transform(row)
    pred = model.predict(X, verbose=0)[0]
    cost_overrun_pct = float(pred[0])
    delay_days = max(0.0, float(pred[1]))

    final_cost = features.planned_budget_aed * (1 + cost_overrun_pct / 100)
    final_duration = features.planned_duration_days + delay_days

    if cost_overrun_pct < 5 and delay_days < 10:
        risk_level = "Low"
    elif cost_overrun_pct < 15 and delay_days < 30:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return RiskResponse(
        predicted_cost_overrun_pct=round(cost_overrun_pct, 2),
        predicted_delay_days=round(delay_days, 1),
        predicted_final_cost_aed=round(final_cost, -2),
        predicted_final_duration_days=round(final_duration, 1),
        risk_level=risk_level,
    )
