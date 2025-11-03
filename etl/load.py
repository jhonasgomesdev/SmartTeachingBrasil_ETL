from sqlalchemy import create_engine
import os
import urllib.parse

def get_engine():
    # Lê a URL de conexão (ajuste user/senha/banco conforme seu OLAP)
    db_url = os.getenv("OLAP_DATABASE_URL", "postgresql+psycopg2://postgres:360elite@localhost:5432/SmartTeachingOLAP")

    # Codifica a URL para evitar problemas de acentuação
    safe_url = urllib.parse.quote_plus(db_url)
    
    # Garante que o encoding usado na conexão seja UTF-8
    engine = create_engine(db_url, connect_args={"options": "-c client_encoding=utf8"})
    return engine

def load_dfs(dfs):
    engine = get_engine()
    if not dfs["alunos"].empty:
        dfs["alunos"].to_sql("dim_aluno", engine, if_exists="replace", index=False)
    if not dfs["itens_historico"].empty:
        dfs["itens_historico"].to_sql("fato_historico", engine, if_exists="replace", index=False)
    if not dfs["itens_questionario"].empty:
        dfs["itens_questionario"].to_sql("fato_questionario", engine, if_exists="replace", index=False)
    print("✅ Dados carregados no OLAP com sucesso!")