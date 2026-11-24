"""
Entrena y empaqueta el modelo final de churn para servirlo en la API.

Modelo elegido: regresión logística balanceada (C=1.0), corrida
`logreg_C1.0_balanced` de MLflow. Tiene un ROC-AUC casi igual al mejor
(0.842 vs 0.844) pero un recall mucho mayor (0.78 vs 0.50): detecta a 8 de
cada 10 clientes que se van, que es lo que importa para retenerlos. Además es
interpretable (coeficientes).

Usa el mismo preprocesamiento y la misma partición que models/train_mlflow.py.

Uso:
    python models/export_model.py --data base_de_datos_proyecto.csv
    # opcional: registrar el modelo en MLflow como "churn_model"
    python models/export_model.py --data base_de_datos_proyecto.csv --tracking http://localhost:5000
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
MODEL_NAME = "churn_model"
MODEL_VERSION = "1.0.0"
RUN_NAME = "logreg_C1.0_balanced"
PARAMS = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}


def to_snake(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = [to_snake(c) for c in df.columns]
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce").fillna(0)
    y = df["churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)
    X = df.drop(columns=["churn", "customer_id"], errors="ignore")
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(**PARAMS))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="base_de_datos_proyecto.csv")
    parser.add_argument("--out", default="models/churn_model.joblib")
    parser.add_argument("--tracking", default=None,
                        help="URI de MLflow para registrar el modelo (opcional)")
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

    # Valores por defecto para variables que no lleguen en una petición:
    # mediana en numéricas y valor más frecuente en categóricas.
    defaults = {}
    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            defaults[col] = float(X_train[col].median())
        else:
            defaults[col] = str(X_train[col].mode().iloc[0])

    bundle = {
        "pipeline": pipe,
        "features": X_train.columns.tolist(),
        "defaults": defaults,
        "metrics": metrics,
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "source_run": RUN_NAME,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"Modelo guardado en {out}")
    print(json.dumps({"version": MODEL_VERSION, "metrics": metrics}, indent=2))

    if args.tracking:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(args.tracking)
        mlflow.set_experiment("churn_telecomunicaciones")
        with mlflow.start_run(run_name=f"modelo_final_{RUN_NAME}"):
            mlflow.log_params({**PARAMS, "modelo": "LogisticRegression",
                               "version": MODEL_VERSION})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipe, name="model", serialization_format="cloudpickle",
                registered_model_name=MODEL_NAME)
        print(f"Modelo registrado en MLflow como '{MODEL_NAME}'")


if __name__ == "__main__":
    main()
