import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sqlalchemy import create_engine
import joblib
import numpy as np
import os

# ============================================================
# 🔹 Conexão com o banco OLAP
# ============================================================
def get_engine():
    db_url = os.getenv(
        "OLAP_DATABASE_URL",
        "postgresql+psycopg2://postgres:360elite@localhost:5432/SmartTeachingOLAP"
    )
    return create_engine(db_url, connect_args={"options": "-c client_encoding=utf8"})


# ============================================================
# 🔹 Carrega dados
# ============================================================
def load_data_from_olap():
    engine = get_engine()
    df_hist = pd.read_sql("SELECT * FROM fato_historico", engine)
    df_perf = pd.read_sql("SELECT * FROM fato_perfil", engine)

    if df_hist.empty or df_perf.empty:
        raise RuntimeError("❌ Fato histórico ou perfil estão vazios. Rode o ETL primeiro.")

    print("✅ Dados carregados do OLAP:")
    print(f"   - fato_historico: {len(df_hist)} registros")
    print(f"   - fato_perfil: {len(df_perf)} registros")

    # --------------------------------------------------------
    # 🔸 Médias de notas por área
    # --------------------------------------------------------
    medias = (
        df_hist.groupby(["aluno_id", "area_conhecimento"])["nota"]
        .mean()
        .unstack(fill_value=0)
        .reset_index()
    )

    medias.columns = ["aluno_id", "media_biologicas", "media_exatas", "media_humanas"][:len(medias.columns)]
    for col in medias.columns[1:]:
        medias[col] = medias[col] / 10

    df = pd.merge(medias, df_perf, on="aluno_id", how="left")

    # --------------------------------------------------------
    # 🔸 Feature Engineering
    # --------------------------------------------------------
    df["media_global"] = df[["media_exatas", "media_humanas", "media_biologicas"]].mean(axis=1)
    df["dif_exatas_humanas"] = (df["media_exatas"] - df["media_humanas"]).round(3)
    df["dif_exatas_bio"] = (df["media_exatas"] - df["media_biologicas"]).round(3)
    df["dif_humanas_bio"] = (df["media_humanas"] - df["media_biologicas"]).round(3)

    # --------------------------------------------------------
    # 🔸 Cria rótulo (label) de área predominante
    # --------------------------------------------------------
    area_map = {
        "Exatas": 1,
        "Humanas": 2,
        "Biológicas": 0,
        "Negócios": 3
    }

    df["label"] = df["area_vocacional_predominante"].map(area_map)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    print("\n📊 Distribuição de classes (labels):")
    print(df["label"].value_counts())

    return df


# ============================================================
# 🔹 Função principal de treino (com GridSearchCV)
# ============================================================
def train_model(model_path="models/course_model_v2.joblib"):
    df = load_data_from_olap()

    feature_cols = [
        "media_exatas", "media_humanas", "media_biologicas",
        "media_global", "dif_exatas_humanas", "dif_exatas_bio", "dif_humanas_bio",
        "E/I", "S/N", "T/F", "J/P", "perfil_mbti", "perfil_vocacional"
    ]
    X = df[feature_cols]
    y = df["label"]

    # --------------------------------------------------------
    # 🔍 Ajuste de hiperparâmetros com GridSearchCV
    # --------------------------------------------------------
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [4, 6, 8, None],
        "clf__min_samples_split": [2, 4, 6],
        "clf__min_samples_leaf": [1, 2, 3]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=1)
    grid.fit(X, y)

    print("\n🔎 Melhor combinação de parâmetros encontrada:")
    print(grid.best_params_)

    # --------------------------------------------------------
    # 🧠 Re-treina modelo final com LOOCV (avaliação realista)
    # --------------------------------------------------------
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    best_model = grid.best_estimator_
    for train_idx, test_idx in loo.split(X):
        best_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = best_model.predict(X.iloc[test_idx])
        y_true.append(y.iloc[test_idx].values[0])
        y_pred.append(pred[0])

    score = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Acurácia geral com LOOCV: {score:.3f}")

    print("\n📋 Relatório detalhado:")
    print(classification_report(y_true, y_pred, digits=3))

    print("\n📈 Matriz de confusão:")
    print(confusion_matrix(y_true, y_pred))

    # --------------------------------------------------------
    # 💾 Salva modelo final
    # --------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": best_model,
        "features": feature_cols
    }, model_path)

    importances = best_model.named_steps["clf"].feature_importances_
    feature_importance = {col: float(imp) for col, imp in zip(feature_cols, importances)}

    print("\n📊 Importância das features:")
    for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   {k:<25} → {v:.3f}")

    print("\n🏁 Treinamento concluído com sucesso!")
    return score


# ============================================================
# 🔹 Execução direta
# ============================================================
if __name__ == "__main__":
    acc = train_model()
    print(f"\n✅ Acurácia final do modelo: {acc:.3f}")