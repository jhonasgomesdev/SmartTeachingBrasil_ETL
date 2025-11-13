import pandas as pd

# ============================================================
# 🔹 Função principal de transformação
# ============================================================
def transformar_dados(df_alunos, df_historicos, df_itens_historico,
                      df_questionarios, df_itens_questionario, df_perguntas, df_opcoes):
    """
    Responsável por transformar os dados brutos extraídos do OLTP
    em estruturas analíticas prontas para o OLAP.
    """

    print("🔄 Iniciando transformações...")

    # ============================================================
    # 1️⃣ PERFIL MBTI — cálculo das médias das quatro dimensões
    # ============================================================
    df_mbti = df_itens_questionario.merge(
        df_perguntas, left_on="pergunta_id", right_on="id", suffixes=("_item", "_pergunta")
    )

    # 🔹 Normaliza o nome da coluna para evitar diferenças de maiúsculas/minúsculas
    df_mbti["tipo"] = df_mbti["tipo"].astype(str).str.strip().str.upper()

    # 🔹 Filtra apenas as dimensões MBTI
    df_mbti = df_mbti[df_mbti["tipo"].isin(["E/I", "S/N", "T/F", "J/P"])]

    if df_mbti.empty:
        raise RuntimeError("Nenhuma pergunta MBTI encontrada — verifique se as dimensões E/I, S/N, T/F, J/P existem no banco.")

    # 🔹 Calcula as médias das respostas por dimensão MBTI
    df_mbti_agrupado = (
        df_mbti.groupby(["aluno_id", "tipo"])["resposta_valor"]
        .mean()
        .unstack(fill_value=0)
        .reset_index()
    )

    print("✅ Perfis MBTI transformados com sucesso!")
    print(df_mbti_agrupado.head())


    # ============================================================
    # 2️⃣ PERFIL VOCACIONAL — cálculo baseado em perguntas RIASEC
    # ============================================================
    df_voc = (
        df_itens_questionario
        .merge(df_questionarios, left_on="questionario_id", right_on="id", suffixes=("_item", "_questionario"))
        .merge(df_perguntas, left_on="pergunta_id", right_on="id", suffixes=("_item", "_pergunta"))
    )

    # 🔹 Detecta automaticamente a coluna que representa o tipo do questionário
    col_tipo_q = None
    possiveis_colunas = ["tipo_questionario", "tipo_item", "tipo", "tipo_questionario_item"]

    for c in df_voc.columns:
        if c.lower() in possiveis_colunas:
            col_tipo_q = c
            break

    if not col_tipo_q:
        raise KeyError(f"Não foi encontrada a coluna de tipo do questionário nas colunas: {df_voc.columns.tolist()}")
    else:
        print(f"📘 Coluna de tipo do questionário detectada automaticamente: {col_tipo_q}")

    # 🔹 Normaliza e filtra apenas os questionários vocacionais
    df_voc[col_tipo_q] = df_voc[col_tipo_q].astype(str).str.upper()
    df_voc = df_voc[df_voc[col_tipo_q].str.contains("VOCACIONAL", na=False)]

    if not df_voc.empty:
        print(f"✅ Perguntas vocacionais encontradas: {len(df_voc)}")

        # 🔹 Garante que a coluna aluno_id exista corretamente
        if "aluno_id_item" in df_voc.columns:
            df_voc["aluno_id"] = df_voc["aluno_id_item"]
        elif "aluno_id_questionario" in df_voc.columns:
            df_voc["aluno_id"] = df_voc["aluno_id_questionario"]
        else:
            raise KeyError(f"Coluna de aluno não encontrada em df_voc. Colunas disponíveis: {df_voc.columns.tolist()}")

        # 🔹 Mapeia automaticamente a área RIASEC com base no tipo da pergunta
        df_voc["area_riasec"] = df_voc["tipo_pergunta"].str.strip().str.capitalize()

        # 🔹 Calcula médias por área RIASEC para cada aluno
        medias_areas = (
            df_voc.groupby(["aluno_id", "area_riasec"])["resposta_valor"]
            .mean()
            .unstack(fill_value=0)
            .reset_index()
        )

        # 🔹 Normaliza as colunas esperadas (RIASEC simplificado)
        riasec_cols = ["Exatas", "Humanas", "Biológicas", "Negócios"]
        for col in riasec_cols:
            if col not in medias_areas.columns:
                medias_areas[col] = 0.0

        # 🔹 Determina área predominante e perfil de dispersão
        medias_areas["area_vocacional_predominante"] = medias_areas[riasec_cols].idxmax(axis=1)
        medias_areas["perfil_vocacional"] = medias_areas[riasec_cols].std(axis=1)

        print("✅ Perfil vocacional (RIASEC) gerado com sucesso!")
        print(medias_areas.head())

        df_vocacional = medias_areas[["aluno_id", "perfil_vocacional", "area_vocacional_predominante"]]
    else:
        print("⚠️ Nenhuma pergunta vocacional encontrada.")
        df_vocacional = pd.DataFrame(columns=["aluno_id", "perfil_vocacional", "area_vocacional_predominante"])


    # ============================================================
    # 3️⃣ FATO PERFIL — junção do MBTI + Vocacional
    # ============================================================
    df_fato_perfil = df_mbti_agrupado.merge(df_vocacional, on="aluno_id", how="left")

    # Substituir nulos
    df_fato_perfil["perfil_vocacional"].fillna(0, inplace=True)
    df_fato_perfil["area_vocacional_predominante"].fillna("N/A", inplace=True)

    # Criar o índice médio MBTI (média das quatro dimensões)
    df_fato_perfil["perfil_mbti"] = df_fato_perfil[["E/I", "S/N", "T/F", "J/P"]].mean(axis=1)

    print("✅ Fato de perfil consolidado com sucesso!")
    print(df_fato_perfil.head())


    # ============================================================
    # 4️⃣ FATO HISTÓRICO — médias de notas por área
    # ============================================================
    df_hist = df_itens_historico.merge(
        df_historicos, left_on="historico_id", right_on="id", suffixes=("_item", "_hist")
    )

    # 🔹 Garante que a coluna aluno_id esteja presente corretamente
    if "aluno_id_item" in df_hist.columns:
        df_hist["aluno_id"] = df_hist["aluno_id_item"]
    elif "aluno_id_hist" in df_hist.columns:
        df_hist["aluno_id"] = df_hist["aluno_id_hist"]
    else:
        raise KeyError("Nenhuma coluna aluno_id encontrada no histórico.")

    # 🔹 Classifica as disciplinas em áreas (Exatas, Humanas, Biológicas)
    df_hist["area_conhecimento"] = df_hist["disciplina"].apply(classificar_area_disciplina)

    # 🔹 Calcula a média das notas por aluno e área
    df_fato_historico = (
        df_hist.groupby(["aluno_id", "area_conhecimento"])["nota"]
        .mean()
        .reset_index()
    )

    print("✅ Fato histórico consolidado com sucesso!")
    print(df_fato_historico.head())


    # ============================================================
    # 5️⃣ Retorno final (dicionário para carga)
    # ============================================================
    return {
        "fato_perfil": df_fato_perfil,
        "fato_historico": df_fato_historico
    }


# ============================================================
# 🔹 Função auxiliar — classificação automática das disciplinas
# ============================================================
def classificar_area_disciplina(nome_disciplina):
    """
    Classifica a disciplina automaticamente em uma das 3 grandes áreas.
    """
    nome = nome_disciplina.lower()

    if any(x in nome for x in ["mat", "fis", "quim", "algoritmo", "calc"]):
        return "Exatas"
    elif any(x in nome for x in ["bio", "saúde", "anat", "fisio", "med"]):
        return "Biológicas"
    else:
        return "Humanas"