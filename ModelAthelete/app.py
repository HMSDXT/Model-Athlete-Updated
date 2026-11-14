import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# Load model
# ============================================================

MODEL_PATH = "models/sports_performance_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Sports Psychology AI Model",
    description=(
        "AI-based model for assessing "
        "psychological factors associated "
        "with sports performance."
    ),
    version="1.0"
)


# ============================================================
# Input schema
# ============================================================

class AthleteData(BaseModel):

    Motivation: float
    Anxiety: float
    Confidence: float
    Stress: float
    Resilience: float
    Mental_Toughness: float


# ============================================================
# Home
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Sports Psychology AI Model API",
        "status": "running"
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(data: AthleteData):

    athlete = pd.DataFrame([{
        "Motivation": data.Motivation,
        "Anxiety": data.Anxiety,
        "Confidence": data.Confidence,
        "Stress": data.Stress,
        "Resilience": data.Resilience,
        "Mental_Toughness": data.Mental_Toughness
    }])


    prediction = model.predict(athlete)[0]

    prediction = max(
        0,
        min(100, prediction)
    )


    if prediction < 50:

        category = "Low"

    elif prediction < 75:

        category = "Moderate"

    else:

        category = "High"


    return {
        "predicted_performance": round(
            float(prediction),
            2
        ),

        "performance_category": category
    }