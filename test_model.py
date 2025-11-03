import joblib
import pandas as pd

# Carrega o modelo
model = joblib.load("models/course_model.joblib")

# Cria um exemplo de entrada
data = pd.DataFrame([{"media_nota": 8.5, "qtd_disciplinas": 6}])

# Faz a predição
pred = model.predict(data)
proba = model.predict_proba(data)

print(f"Predição: {pred[0]}")
print(f"Probabilidades: {proba[0]}")