"""
API de predicción de churn (FastAPI).

Endpoints:
    GET  /         mensaje de bienvenida
    GET  /health   estado de la API y datos del modelo cargado
    POST /predict  probabilidad de fuga para uno o varios clientes

Ejecutar en local:
    uvicorn api.main:app --host 0.0.0.0 --port 8001
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/churn_model.joblib"))

bundle: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}")
    bundle.update(joblib.load(MODEL_PATH))
    yield


app = FastAPI(
    title="ChurnTel API",
    description="Predicción de abandono (churn) de clientes de telecomunicaciones.",
    version="1.0.0",
    lifespan=lifespan,
)


EXAMPLE = {
    "tenure": 4, "monthly_charges": 85.0, "total_charges": 340.0,
    "contract": "Month-to-month", "internet_service": "Fiber optic",
    "payment_method": "Electronic check", "tech_support": "No",
    "online_security": "No", "paperless_billing": "Yes", "senior_citizen": 0,
}


class PredictRequest(BaseModel):
    customers: List[Dict[str, Any]] = Field(
        ..., min_length=1, examples=[[EXAMPLE]],
        description="Lista de clientes. Las variables que falten se completan "
                    "con el valor típico de los datos de entrenamiento.",
    )


class Prediction(BaseModel):
    customer_id: Any = None
    probabilidad_fuga: float
    prediccion: int
    riesgo: str


class PredictResponse(BaseModel):
    modelo: str
    version: str
    predicciones: List[Prediction]


def risk_level(p: float) -> str:
    if p < 0.30:
        return "bajo"
    if p < 0.60:
        return "medio"
    return "alto"


def to_frame(customers: List[Dict[str, Any]]) -> pd.DataFrame:
    features, defaults = bundle["features"], bundle["defaults"]
    rows = []
    for c in customers:
        rows.append({f: defaults[f] if c.get(f) is None else c[f] for f in features})
    df = pd.DataFrame(rows, columns=features)
    for f in features:
        if isinstance(defaults[f], float):
            df[f] = pd.to_numeric(df[f], errors="coerce").fillna(defaults[f])
        else:
            df[f] = df[f].astype(str)
    return df


@app.get("/")
def root():
    return {"mensaje": "Bienvenido a la API ChurnTel. Documentación en /docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": bundle.get("model_name"),
        "version": bundle.get("version"),
        "corrida_mlflow": bundle.get("source_run"),
        "entrenado": bundle.get("trained_at"),
        "metricas_test": bundle.get("metrics"),
        "variables": bundle.get("features"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = to_frame(req.customers)
        proba = bundle["pipeline"].predict_proba(X)[:, 1]
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Entrada inválida: {error}")

    preds = [
        Prediction(
            customer_id=c.get("customer_id"),
            probabilidad_fuga=round(float(p), 4),
            prediccion=int(p >= 0.5),
            riesgo=risk_level(float(p)),
        )
        for c, p in zip(req.customers, proba)
    ]
    return PredictResponse(modelo=bundle["model_name"], version=bundle["version"],
                           predicciones=preds)
