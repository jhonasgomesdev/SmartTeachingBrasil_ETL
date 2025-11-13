import requests
import pandas as pd

def extract_data(url: str):
    """
    Faz a extração de dados da API e retorna o JSON bruto.
    """
    print(f"🔗 Extraindo dados de {url}...")
    response = requests.get(url, verify=False)  # verify=False evita erro de certificado local
    response.raise_for_status()
    data = response.json()
    print(f"✅ {len(data)} registros extraídos.")
    return data


def flatten_data(raw):
    alunos_data, historicos_data, itens_hist_data = [], [], []
    questionarios_data, itens_quest_data, perguntas_data, opcoes_data = [], [], [], []

    for aluno in raw:
        alunos_data.append({
            "id": aluno["id"],
            "nome": aluno["nome"],
            "email": aluno["email"],
            "cidade": aluno["cidade"],
            "data_nascimento": aluno["dataNascimento"]
        })

        # HISTÓRICOS
        for h in aluno.get("historicosEscolares", []):
            historicos_data.append({
                "id": h["id"],
                "aluno_id": aluno["id"],
                "serie": h["serie"],
                "ano": h["ano"]
            })

            for item in h.get("itens", []):
                itens_hist_data.append({
                    "historico_id": h["id"],
                    "aluno_id": aluno["id"],
                    "disciplina": item["disciplina"],
                    "area_conhecimento": item["areaConhecimento"],
                    "nota": item["nota"],
                    "frequencia": item["frequencia"]
                })

        # QUESTIONÁRIOS
        for q in aluno.get("questionarios", []):
            questionarios_data.append({
                "id": q["id"],
                "aluno_id": aluno["id"],
                "tipo": q["tipo"],
                "nome": q["nome"],
                "descricao": q.get("descricao")
            })

            for iq in q.get("itens", []):
                itens_quest_data.append({
                    "id": iq["id"],
                    "questionario_id": q["id"],
                    "aluno_id": aluno["id"],
                    "sequencial": iq["sequencial"],
                    "data_resposta": iq["dataResposta"],
                    "resposta_texto": iq["respostaTexto"],
                    "resposta_valor": iq.get("respostaValor"),
                    "pergunta_id": iq["pergunta"]["id"],
                    "opcao_id": iq["opcao"]["id"] if iq.get("opcao") else None
                })

                # PERGUNTAS
                p = iq["pergunta"]
                perguntas_data.append({
                    "id": p["id"],
                    "tipo": p["tipo"],
                    "titulo": p["titulo"],
                    "descricao": p["descricao"]
                })

                # OPÇÕES
                if iq.get("opcao"):
                    o = iq["opcao"]
                    opcoes_data.append({
                        "id": o["id"],
                        "descricao": o["descricao"],
                        "valor": o["valor"]
                    })

    dfs = {
        "alunos": pd.DataFrame(alunos_data),
        "historicos": pd.DataFrame(historicos_data),
        "itens_historico": pd.DataFrame(itens_hist_data),
        "questionarios": pd.DataFrame(questionarios_data),
        "itens_questionario": pd.DataFrame(itens_quest_data),
        "perguntas": pd.DataFrame(perguntas_data),
        "opcoes": pd.DataFrame(opcoes_data),
    }

    print("📊 Dados tabulares criados:")
    for k, df in dfs.items():
        print(f"   - {k}: {len(df)}")

    return dfs