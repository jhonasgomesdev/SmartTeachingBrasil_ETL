from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="SmartTeaching Predict API")

class PredictionInput(BaseModel):
    media_nota: float
    qtd_disciplinas: int

# Carrega o modelo ao iniciar
model = joblib.load("models/course_model.joblib")

@app.post("/predict")
def predict(input_data: PredictionInput):
    df = pd.DataFrame([input_data.dict()])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].max()
    return {"predicted_label": int(pred), "probability": round(float(proba), 2)}