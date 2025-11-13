from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import traceback
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ============================================================
# 🚀 Inicialização da API
# ============================================================
app = FastAPI(title="SmartTeaching Prediction Service v2")

# ============================================================
# 1️⃣ Carregamento do modelo otimizado
# ============================================================
MODEL_PATH = "models/course_model_v2.joblib"

try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    feature_names = model_bundle["features"]
    print(f"✅ Modelo v2 carregado com features: {feature_names}")
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
# 4️⃣ Função auxiliar para gerar features adicionais
# ============================================================
def compute_extra_features(data: PredictionRequest):
    media_global = np.mean([data.media_exatas, data.media_humanas, data.media_biologicas])
    dif_exatas_humanas = round(data.media_exatas - data.media_humanas, 3)
    dif_exatas_bio = round(data.media_exatas - data.media_biologicas, 3)
    dif_humanas_bio = round(data.media_humanas - data.media_biologicas, 3)
    return media_global, dif_exatas_humanas, dif_exatas_bio, dif_humanas_bio


# ============================================================
# 5️⃣ Endpoint principal de predição
# ============================================================
@app.post("/predict")
def predict(data: PredictionRequest):
    print("📩 Payload recebido:", data.dict())

    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    try:
        media_global, dif_exatas_humanas, dif_exatas_bio, dif_humanas_bio = compute_extra_features(data)

        # 🔹 Monta o dataframe de entrada
        X = pd.DataFrame([[
            data.media_exatas,
            data.media_humanas,
            data.media_biologicas,
            media_global,
            dif_exatas_humanas,
            dif_exatas_bio,
            dif_humanas_bio,
            data.E_I,
            data.S_N,
            data.T_F,
            data.J_P,
            data.perfil_mbti,
            data.perfil_vocacional
        ]], columns=feature_names)

        # 🔹 Predição principal
        pred_label = int(model.predict(X)[0])
        label_map = {0: "Biológicas", 1: "Exatas", 2: "Humanas", 3: "Negócios"}
        label_text = label_map.get(pred_label, "Desconhecido")

        # 🔹 Probabilidade por classe
        probs = model.predict_proba(X)[0]
        prob_max = float(np.max(probs))

        # 🔹 Log da predição
        print(f"🎯 Predição: {label_text} (confiança: {prob_max:.3f})")

        # ============================================================
        # 🎓 Mapeamento de recomendação de cursos
        # ============================================================
        area_index_map = {"Biológicas": 0, "Exatas": 1, "Humanas": 2, "Negócios": 3}
        resultados = []

        for curso in COURSES:
            area = curso["area"]
            nome = curso["nome"]

            idx = area_index_map.get(area.split("/")[0], 0)
            base_score = probs[min(idx, len(probs) - 1)]

            media_area = (
                data.media_exatas if area == "Exatas" else
                data.media_humanas if area == "Humanas" else
                data.media_biologicas if area == "Biológicas" else
                media_global
            )

            score_final = (
                base_score * 0.6
                + (media_area / 10) * 0.3
                + (prob_max) * 0.1
            ) * np.random.uniform(0.98, 1.02)

            resultados.append({
                "nome": nome,
                "area": area,
                "score": round(float(score_final), 3)
            })

        resultados = sorted(resultados, key=lambda x: x["score"], reverse=True)[:10]

        return {
            "PredictedLabel": label_text,
            "Confidence": round(prob_max, 3),
            "CursosRecomendados": resultados
        }

    except Exception as e:
        print("❌ Erro interno no modelo:", traceback.format_exc())
        raise HTTPException(status_code=422, detail=str(e))


# ============================================================
# 6️⃣ Health-check simples
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "modelo": "v2", "features": feature_names}