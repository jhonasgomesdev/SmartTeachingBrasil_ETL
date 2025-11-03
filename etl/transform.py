import pandas as pd

def flatten(alunos_json):
    alunos_rows = []
    historico_rows = []
    itens_hist_rows = []
    questionario_rows = []
    itens_q_rows = []

    for a in alunos_json:
        alunos_rows.append({
            "aluno_id": a["id"],
            "nome": a["nome"],
            "email": a.get("email"),
            "cidade": a.get("cidade"),
            "data_nascimento": a.get("dataNascimento")
        })
        for h in a.get("historicosEscolares", []):
            historico_rows.append({
                "historico_id": h["id"],
                "aluno_id": a["id"],
                "serie": h.get("serie"),
                "ano": h.get("ano")
            })
            for it in h.get("itens", []):
                itens_hist_rows.append({
                    "historico_id": h["id"],
                    "aluno_id": a["id"],
                    "disciplina": it.get("disciplina"),
                    "area": it.get("areaConhecimento"),
                    "nota": it.get("nota"),
                    "frequencia": it.get("frequencia")
                })
        for q in a.get("questionarios", []):
            questionario_rows.append({
                "questionario_id": q.get("id", None),
                "aluno_id": a["id"],
                "tipo": q.get("tipo"),
                "nome": q.get("nome")
            })
            for iq in q.get("itens", []):
                itens_q_rows.append({
                    "questionario_id": q.get("id", None),
                    "aluno_id": a["id"],
                    "pergunta_id": iq.get("perguntaId"),
                    "pergunta": iq.get("pergunta"),
                    "opcao_id": iq.get("opcaoId"),
                    "resposta_texto": iq.get("respostaTexto"),
                    "resposta_valor": iq.get("respostaValor"),
                    "data_resposta": iq.get("dataResposta")
                })

    return {
        "alunos": pd.DataFrame(alunos_rows),
        "historicos": pd.DataFrame(historico_rows),
        "itens_historico": pd.DataFrame(itens_hist_rows),
        "questionarios": pd.DataFrame(questionario_rows),
        "itens_questionario": pd.DataFrame(itens_q_rows)
    }