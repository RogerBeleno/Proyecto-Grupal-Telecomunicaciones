"""
Entrenamiento de modelos de churn con registro de experimentos en MLflow.
Proyecto MIAD - Despliegue de soluciones analíticas (Telecomunicaciones).

Uso (en la EC2, con el servidor MLflow corriendo en el puerto 5000):
    python train_mlflow.py --data base_de_datos_proyecto.csv --tracking http://localhost:5000
"""
import argparse
import re

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
EXPERIMENT = "churn_telecomunicaciones"


def to_snake(name: str) -> str:
    """CustomerID -> customer_id, MonthlyCharges -> monthly_charges."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name.strip())
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = [to_snake(c) for c in df.columns]

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(0)  # clientes con tenure 0

    y = df["churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)
    X = df.drop(columns=["churn", "customer_id"], errors="ignore")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


# Cada entrada = una corrida (run) en MLflow
EXPERIMENTS = [
    ("logistic_regression", LogisticRegression, {"C": 1.0, "max_iter": 1000}),
    ("logistic_regression_C0.1", LogisticRegression, {"C": 0.1, "max_iter": 1000}),
    ("logistic_regression_balanced", LogisticRegression,
     {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
    ("random_forest", RandomForestClassifier,
     {"n_estimators": 200, "max_depth": 8, "random_state": SEED}),
    ("random_forest_balanced", RandomForestClassifier,
     {"n_estimators": 300, "max_depth": 10, "class_weight": "balanced",
      "random_state": SEED}),
    ("gradient_boosting", GradientBoostingClassifier,
     {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
      "random_state": SEED}),
]


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="base_de_datos_proyecto.csv")
    parser.add_argument("--tracking", default="http://localhost:5000")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking)
    mlflow.set_experiment(EXPERIMENT)

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    results = []
    for run_name, estimator_cls, params in EXPERIMENTS:
        with mlflow.start_run(run_name=run_name):
            pipe = Pipeline([
                ("prep", build_preprocessor(X_train)),
                ("clf", estimator_cls(**params)),
            ])
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)

            mlflow.log_param("modelo", estimator_cls.__name__)
            mlflow.log_params(params)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipe, "model", serialization_format="cloudpickle")

            results.append({"run": run_name, **metrics})
            print(f"{run_name:32s} AUC={metrics['roc_auc']:.3f} "
                  f"F1={metrics['f1']:.3f} Recall={metrics['recall']:.3f}")

    print("\nResumen (ordenado por ROC-AUC):")
    print(pd.DataFrame(results).sort_values("roc_auc", ascending=False)
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
