from __future__ import annotations

import logging
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.model_loader import get_model, get_model_version
from api.schemas import PredictionRequest, PredictionResponse
from training.config import RAW_DATA_PATH

app = FastAPI(title="Employee Attrition Prediction API")
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "prediction_api_requests_total",
    "Total prediction API requests",
)
PREDICTION_COUNT = Counter(
    "prediction_results_total",
    "Total predictions by class",
    ["prediction"],
)
REQUEST_LATENCY = Histogram(
    "prediction_api_request_latency_seconds",
    "Prediction API request latency",
)

default_values = None
all_features = None


@app.on_event("startup")
async def startup_event():
    global default_values, all_features

    get_model()

    df = pd.read_csv(RAW_DATA_PATH)
    drop_columns = [
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "EmployeeNumber",
        "Attrition",
    ]
    df = df.drop(columns=drop_columns, errors="ignore")
    all_features = df.columns.tolist()

    default_values = {}
    for col in all_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            default_values[col] = df[col].median()
        else:
            mode = df[col].mode()
            default_values[col] = mode.iloc[0] if not mode.empty else "Unknown"


@app.get("/")
async def root():
    return {
        "service": "employee-attrition-api",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": get_model() is not None,
        "model_version": get_model_version(),
    }


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    REQUEST_COUNT.inc()
    started = time.perf_counter()

    try:
        model = get_model()
        input_dict = request.model_dump()
        full_input = {
            col: input_dict.get(col, default_values[col])
            for col in all_features
        }

        df_input = pd.DataFrame([full_input])
        probability = float(model.predict_proba(df_input)[0][1])
        prediction = int(probability >= 0.5)

        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        return PredictionResponse(
            attrition_probability=round(probability, 4),
            prediction=prediction,
        )
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - started)
