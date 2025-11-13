from sqlalchemy import create_engine
import os
import urllib.parse

def get_engine():
    db_url = os.getenv(
        "OLAP_DATABASE_URL",
        "postgresql+psycopg2://postgres:360elite@localhost:5432/SmartTeachingOLAP"
    )
    return create_engine(db_url, connect_args={"options": "-c client_encoding=utf8"})


def load_dfs(dfs):
    """
    Carrega todos os DataFrames transformados no banco OLAP.
    Espera receber o dicionário retornado por transformar_dados().
    """
    engine = get_engine()

    # ✅ Carrega o fato_perfil
    if "fato_perfil" in dfs and not dfs["fato_perfil"].empty:
        dfs["fato_perfil"].to_sql("fato_perfil", engine, if_exists="replace", index=False)
        print("✅ fato_perfil carregado no OLAP com sucesso!")

    # ✅ Carrega o fato_historico
    if "fato_historico" in dfs and not dfs["fato_historico"].empty:
        dfs["fato_historico"].to_sql("fato_historico", engine, if_exists="replace", index=False)
        print("✅ fato_historico carregado no OLAP com sucesso!")

    # (Opcional) se quiser manter dimensões auxiliares:
    if "alunos" in dfs and not dfs["alunos"].empty:
        dfs["alunos"].to_sql("dim_aluno", engine, if_exists="replace", index=False)
        print("✅ dim_aluno carregado no OLAP com sucesso!")

    print("🏁 Todos os dados foram carregados no OLAP com êxito!")