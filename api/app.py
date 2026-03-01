import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Prévision Pathologies API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

linear_model = joblib.load(os.path.join(BASE_DIR, "model", "linear.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "model", "random_forest.pkl"))
gb_model = joblib.load(os.path.join(BASE_DIR, "model", "gradient_boosting.pkl"))
kmeans_model = joblib.load(os.path.join(BASE_DIR, "model", "kmeans.pkl"))


class PathologyFeatures(BaseModel):
    annee: int
    region: int
    patho_niv1: str

class ClusterFeatures(BaseModel):
    annee: int
    region: int
    Ntop: float
    prev: float



@app.post("/predict/linear")
def predict_linear(payload: PathologyFeatures):
    X = pd.DataFrame([payload.model_dump()])
    pred_log = linear_model.predict(X)[0]
    pred_ntop = np.expm1(pred_log)

    return {
        "model": "Linear Regression",
        "log_Ntop": float(pred_log),
        "predicted_Ntop": float(pred_ntop)
    }


@app.post("/predict/random_forest")
def predict_rf(payload: PathologyFeatures):
    X = pd.DataFrame([payload.model_dump()])
    pred_log = rf_model.predict(X)[0]
    pred_ntop = np.expm1(pred_log)

    return {
        "model": "Random Forest",
        "log_Ntop": float(pred_log),
        "predicted_Ntop": float(pred_ntop)
    }


@app.post("/predict/gradient_boosting")
def predict_gb(payload: PathologyFeatures):
    X = pd.DataFrame([payload.model_dump()])
    pred_log = gb_model.predict(X)[0]
    pred_ntop = np.expm1(pred_log)

    return {
        "model": "Gradient Boosting",
        "log_Ntop": float(pred_log),
        "predicted_Ntop": float(pred_ntop)
    }



@app.post("/predict/cluster")
def predict_cluster(payload: ClusterFeatures):
    X = pd.DataFrame([payload.model_dump()])
    cluster = kmeans_model.predict(X)[0]

    return {
        "model": "KMeans Clustering",
        "assigned_cluster": int(cluster)
    }


@app.get("/health")
def health():
    return {"status": "ok"}