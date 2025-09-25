"""FastAPI service that serves the latest MLflow-trained Iris classifier."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
import pandas as pd

app = FastAPI(title="Iris Classifier Service", version="0.1.0")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm")
    sepal_width: float = Field(..., description="Sepal width in cm")
    petal_length: float = Field(..., description="Petal length in cm")
    petal_width: float = Field(..., description="Petal width in cm")


@lru_cache(maxsize=1)
def _load_model_bundle() -> Dict[str, Any]:
    """Load latest MLflow model and associated metadata."""
    explicit_model_uri = os.getenv("MODEL_URI")
    explicit_labels_uri = os.getenv("LABELS_URI")

    if explicit_model_uri:
        model = mlflow.sklearn.load_model(explicit_model_uri)
        label_mapping = mlflow.artifacts.load_dict(
            explicit_labels_uri
            or f"{explicit_model_uri}/label_encoder_classes.json"
        )
        classes = label_mapping["classes"]
        run_id = os.getenv("RUN_ID", "explicit-model")
        return {"model": model, "classes": classes, "run_id": run_id}

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "iris_classification",
    )
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_name}' not found. "
            "Train the model before starting the API."
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No runs found for experiment '{experiment_name}'. "
            "Train the model before starting the API."
        )

    run = runs[0]
    model_uri = f"runs:/{run.info.run_id}/model"

    model = mlflow.sklearn.load_model(model_uri)
    label_mapping = mlflow.artifacts.load_dict(
        f"runs:/{run.info.run_id}/model/label_encoder_classes.json"
    )
    classes = label_mapping["classes"]

    return {"model": model, "classes": classes, "run_id": run.info.run_id}


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health check endpoint."""
    bundle = _load_model_bundle()
    return {"status": "ok", "run_id": bundle["run_id"]}


@app.post("/predict")
def predict(features: IrisFeatures) -> Dict[str, Any]:
    """Predict Iris species from the provided measurements."""
    bundle = _load_model_bundle()
    model = bundle["model"]
    classes = bundle["classes"]

    dataframe = pd.DataFrame([features.model_dump()])
    try:
        preds = model.predict(dataframe)
        probs = model.predict_proba(dataframe)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predicted_index = int(preds[0])
    if not 0 <= predicted_index < len(classes):
        raise HTTPException(
            status_code=500,
            detail="Prediction index out of range",
        )

    predicted_label = classes[predicted_index]
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(max(probs[0]))

    response: Dict[str, Any] = {
        "prediction": predicted_label,
        "prediction_index": predicted_index,
        "probability": probability,
        "run_id": bundle["run_id"],
    }
    response.update(
        {
            f"proba_{classes[i]}": float(prob)
            for i, prob in enumerate(probs[0])
        }
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
