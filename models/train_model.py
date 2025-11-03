import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sqlalchemy import create_engine
import os

def get_engine():
    # String de conexão sem caracteres especiais
    db_url = os.getenv("OLAP_DATABASE_URL", "postgresql+psycopg2://postgres:360elite@localhost:5432/SmartTeachingOLAP")
    
    # Se tiver caracteres especiais na senha, codifique corretamente
    if any(char in db_url for char in ['ç', 'ã', 'õ', 'á', 'é', 'í', 'ó', 'ú']):
        # Para senhas com caracteres especiais, use quote_plus
        parsed = urllib.parse.urlparse(db_url)
        safe_password = urllib.parse.quote_plus(parsed.password)
        safe_db_url = f"{parsed.scheme}://{parsed.username}:{safe_password}@{parsed.hostname}:{parsed.port}{parsed.path}"
        engine = create_engine(safe_db_url, connect_args={"options": "-c client_encoding=utf8"})
    else:
        engine = create_engine(db_url, connect_args={"options": "-c client_encoding=utf8"})
    
    return engine

def load_data_from_olap():
    engine = get_engine()
    df_hist = pd.read_sql("select * from fato_historico", engine)
    if df_hist.empty:
        raise RuntimeError("fato_historico vazio — rode o ETL antes de treinar.")
    agg = df_hist.groupby("aluno_id").agg(media_nota=("nota", "mean"), qtd_disciplinas=("disciplina", "count")).reset_index()
    return agg

def train():
    df = load_data_from_olap()
    # Label sintético: media >= 8 => 0 (Exatas), else 1 (Humanas) — apenas baseline
    df["label"] = (df["media_nota"] < 8).astype(int)
    X = df[["media_nota", "qtd_disciplinas"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=50, random_state=42))])
    pipe.fit(X_train, y_train)
    print("score:", pipe.score(X_test, y_test))
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/course_model.joblib")
    print("Modelo salvo: models/course_model.joblib")

if __name__ == "__main__":
    train()