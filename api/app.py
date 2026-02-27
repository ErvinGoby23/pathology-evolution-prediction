import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Prévision Pathologies API")

# =========================
# Chargement des modèles
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

linear_model = joblib.load(os.path.join(BASE_DIR, "model", "linear.pkl"))
rf_model = joblib.load(os.path.join(BASE_DIR, "model", "random_forest.pkl"))
gb_model = joblib.load(os.path.join(BASE_DIR, "model", "gradient_boosting.pkl"))

# =========================
# Schéma des données entrée
# =========================

class PathologyFeatures(BaseModel):
    annee: int
    region: int
    patho_niv1: str

# =========================
# Linear Regression
# =========================

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

# =========================
# Random Forest
# =========================

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

# =========================
# Gradient Boosting
# =========================

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

# =========================
# Health check
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}