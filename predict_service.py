from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import traceback
import json
from pathlib import Path
from models.train import train_model  # ✅ Função de treino

# ============================================================
# 🚀 Inicialização da API
# ============================================================
app = FastAPI(title="SmartTeaching Prediction Service")

# ============================================================
# 1️⃣ Carregamento do modelo
# ============================================================
MODEL_PATH = "models/course_model.joblib"

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    print(f"✅ Modelo carregado com features: {feature_names}")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    model, feature_names = None, []

# ============================================================
# 2️⃣ Carregamento dinâmico dos cursos
# ============================================================
COURSES_PATH = Path("data/courses.json")
try:
    with open(COURSES_PATH, "r", encoding="utf-8") as f:
        COURSES = json.load(f)
    print(f"✅ {len(COURSES)} cursos carregados de {COURSES_PATH}")
except Exception as e:
    print(f"❌ Erro ao carregar cursos: {e}")
    COURSES = []

# ============================================================
# 3️⃣ Schema de entrada (payload esperado)
# ============================================================
class PredictionRequest(BaseModel):
    media_exatas: float
    media_humanas: float
    media_biologicas: float
    E_I: float
    S_N: float
    T_F: float
    J_P: float
    perfil_mbti: float
    perfil_vocacional: float


# ============================================================
# 4️⃣ Endpoint principal de predição
# ============================================================
@app.post("/predict")
def predict(data: PredictionRequest):
    print("📩 Payload recebido:", data.dict())

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    try:
        # 🔹 Monta o dicionário completo com todas as features esperadas
        input_data = {
            "media_exatas": data.media_exatas,
            "media_humanas": data.media_humanas,
            "media_biologicas": data.media_biologicas,
            "E/I": getattr(data, "E_I", 3.0),
            "S/N": getattr(data, "S_N", 3.0),
            "T/F": getattr(data, "T_F", 3.0),
            "J/P": getattr(data, "J_P", 3.0),
            "perfil_mbti": data.perfil_mbti,
            "perfil_vocacional": data.perfil_vocacional
        }

        # 🔹 Garante a ordem das colunas igual ao treino do modelo
        X = pd.DataFrame([[input_data[col] for col in feature_names]], columns=feature_names)

        # 🔹 Predição principal
        pred_label = int(model.predict(X)[0])

        # 🔹 Obtém probabilidades de cada classe
        if hasattr(model.named_steps["clf"], "predict_proba"):
            probs = model.named_steps["clf"].predict_proba(X)[0]
        else:
            probs = np.zeros(len(set(model.named_steps["clf"].classes_)))
            probs[pred_label] = 1.0

        # ============================================================
        # 🎓 Mapeamento de afinidade MBTI / Vocacional por área
        # ============================================================
        afinidade_mbti = {
            "Exatas": data.perfil_mbti / 5,
            "Humanas": 1 - abs(data.perfil_mbti - 2.5) / 5,
            "Biológicas": abs(data.perfil_mbti - 3.5) / 5,
            "Negócios": data.perfil_mbti / 4.5
        }

        afinidade_vocacional = {
            "Exatas": (data.perfil_vocacional * 0.8) / 5,
            "Humanas": (data.perfil_vocacional * 1.0) / 5,
            "Biológicas": (data.perfil_vocacional * 0.9) / 5,
            "Negócios": (data.perfil_vocacional * 0.95) / 5
        }

        area_index_map = {"Humanas": 0, "Exatas": 1, "Biológicas": 2, "Negócios": 0}

        resultados = []
        for curso in COURSES:
            area = curso["area"]
            nome = curso["nome"]

            idx = area_index_map.get(area.split("/")[0], 0)
            base_score = probs[min(idx, len(probs) - 1)]

            # 🔹 Ajuste com base na média por área
            if area == "Exatas":
                media_area = data.media_exatas
            elif area == "Humanas":
                media_area = data.media_humanas
            elif area == "Biológicas":
                media_area = data.media_biologicas
            else:
                media_area = (data.media_exatas + data.media_humanas + data.media_biologicas) / 3

            media_norm = media_area / 10

            # 🔹 Cálculo ponderado final
            score_final = (
                (base_score * 0.5)
                + (afinidade_mbti.get(area, 0.0) * 0.2)
                + (afinidade_vocacional.get(area, 0.0) * 0.1)
                + (media_norm * 0.2)
            )

            score_final *= np.random.uniform(0.97, 1.03)

            resultados.append({
                "nome": nome,
                "area": area,
                "score": round(float(score_final), 3)
            })

        resultados = sorted(resultados, key=lambda x: x["score"], reverse=True)[:10]

        return {
            "PredictedLabel": pred_label,
            "Probability": float(max(probs)),
            "CursosRecomendados": resultados
        }

    except Exception as e:
        print("❌ Erro interno no modelo:", traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(e))


# ============================================================
# 5️⃣ Endpoint de re-treinamento
# ============================================================
@app.post("/train")
def retrain_model():
    try:
        print("🔁 Iniciando re-treinamento do modelo...")
        result = train_model(MODEL_PATH)

        global model, feature_names
        model_bundle = joblib.load(MODEL_PATH)
        model = model_bundle["model"]
        feature_names = model_bundle["features"]

        print("✅ Novo modelo carregado com sucesso!")
        return {
            "message": "Modelo reentreinado com sucesso.",
            "score": result["score"],
            "importance": result["importance"]
        }

    except Exception as e:
        print("❌ Erro no re-treinamento:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))