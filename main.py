from etl.extract import extract_data, flatten_data
from etl.load import load_dfs
from etl.transform import transformar_dados


def main():
    print("🚀 Iniciando pipeline ETL...")

    # ============================================================
    # 1️⃣ EXTRAÇÃO — consome a API OLTP e gera os DataFrames brutos
    # ============================================================
    raw = extract_data("https://localhost:7033/api/export/alunos-detalhados")
    dfs = flatten_data(raw)

    print(f"🔗 Extraídos {len(dfs['alunos'])} alunos")
    for k, v in dfs.items():
        print(f"  - {k}: {len(v)}")

    # ============================================================
    # 2️⃣ TRANSFORMAÇÃO — gera fatos analíticos (perfil + histórico)
    # ============================================================
    resultados = transformar_dados(
        dfs["alunos"],
        dfs["historicos"],
        dfs["itens_historico"],
        dfs["questionarios"],
        dfs["itens_questionario"],
        dfs["perguntas"],
        dfs["opcoes"]
    )

    df_fato_perfil = resultados["fato_perfil"]
    df_fato_historico = resultados["fato_historico"]

    print("✅ Transformações concluídas com sucesso!")
    print(f"   - fato_perfil: {len(df_fato_perfil)} registros")
    print(f"   - fato_historico: {len(df_fato_historico)} registros")

    # ============================================================
    # 3️⃣ CARGA — envia os dados processados ao banco OLAP
    # ============================================================
    dfs["fato_perfil"] = df_fato_perfil
    dfs["fato_historico"] = df_fato_historico
    load_dfs(dfs)


    print("✅ Dados carregados no OLAP com sucesso!")
    print("🏁 ETL concluído com êxito!")


if __name__ == "__main__":
    main()