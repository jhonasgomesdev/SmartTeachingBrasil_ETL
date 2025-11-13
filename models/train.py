import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sqlalchemy import create_engine
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
# 🔹 Função para carregar e preparar dados
# ============================================================
def load_data_from_olap():
    engine = get_engine()
    df_hist = pd.read_sql("SELECT * FROM fato_historico", engine)
    df_perf = pd.read_sql("SELECT * FROM fato_perfil", engine)

    if df_hist.empty or df_perf.empty:
        raise RuntimeError("Fato histórico ou perfil estão vazios. Rode o ETL primeiro.")

    print("✅ Dados carregados do OLAP:")
    print(f"   - fato_historico: {len(df_hist)} registros")
    print(f"   - fato_perfil: {len(df_perf)} registros")

    # ============================================================
    # 🔸 1️⃣ Calcula médias por área
    # ============================================================
    area_map = {
        "Exatas": "media_exatas",
        "Humanas": "media_humanas",
        "Biológicas": "media_biologicas"
    }

    df_hist["coluna_media"] = df_hist["area_conhecimento"].map(area_map)
    medias = (
        df_hist.groupby(["aluno_id", "coluna_media"])["nota"]
        .mean()
        .unstack()
        .reset_index()
        .fillna(0)
    )

    # Normaliza de 0–1
    for col in ["media_exatas", "media_humanas", "media_biologicas"]:
        if col in medias.columns:
            medias[col] = medias[col] / 10

    # ============================================================
    # 🔸 2️⃣ Merge com o fato_perfil (MBTI + Vocacional)
    # ============================================================
    df_temp = pd.merge(medias, df_perf, on="aluno_id", how="left")

    # 🔹 Seleciona apenas as colunas relevantes
    df = df_temp[[
        "aluno_id", "media_exatas", "media_humanas", "media_biologicas",
        "E/I", "S/N", "T/F", "J/P", "perfil_mbti", "perfil_vocacional",
        "area_vocacional_predominante"
    ]].fillna(0)

    # ============================================================
    # 🔸 3️⃣ Encoding da variável alvo (área vocacional)
    # ============================================================
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["area_vocacional_predominante"])
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))

    print("\n📊 Mapeamento das classes (área_vocacional_predominante):")
    print(label_map)

    print("\n📊 Distribuição de classes:")
    print(df["label"].value_counts())

    return df, le


# ============================================================
# 🔹 Função principal de treino com LOOCV
# ============================================================
def train_model(model_path="models/course_model.joblib"):
    df, label_encoder = load_data_from_olap()

    feature_cols = [
        "media_exatas", "media_humanas", "media_biologicas",
        "E/I", "S/N", "T/F", "J/P", "perfil_mbti", "perfil_vocacional"
    ]
    X = df[feature_cols]
    y = df["label"]

    # ============================================================
    # 🧠 Validação Leave-One-Out (ideal para bases pequenas)
    # ============================================================
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42))
        ])

        pipe.fit(X_train, y_train)
        y_pred.append(pipe.predict(X_test)[0])
        y_true.append(y_test.values[0])

    score = accuracy_score(y_true, y_pred)

    # ============================================================
    # 🔹 Re-treina o modelo final completo
    # ============================================================
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42))
    ])
    final_pipe.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": final_pipe,
        "features": feature_cols,
        "label_encoder": label_encoder
    }, model_path)

    importances = final_pipe.named_steps["clf"].feature_importances_
    feature_importance = {col: float(imp) for col, imp in zip(feature_cols, importances)}

    # ============================================================
    # 📊 Relatórios e resultados
    # ============================================================
    print("\n✅ Modelo treinado com LOOCV!")
    print(f"🎯 Acurácia média: {score:.3f}")
    print("\n📈 Matriz de confusão:")
    print(confusion_matrix(y_true, y_pred))
    print("\n📋 Relatório de classificação:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\n📊 Importância das features:")
    for k, v in feature_importance.items():
        print(f"   {k:<20} → {v:.3f}")

    return {
        "score": float(score),
        "features": feature_cols,
        "importance": feature_importance
    }


# ============================================================
# 🔹 Execução direta
# ============================================================
if __name__ == "__main__":
    result = train_model()
    print("\n🏁 Treinamento concluído com sucesso!")
    print(result)