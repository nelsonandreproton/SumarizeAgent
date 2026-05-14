from utils.memory_store import load_memory, save_memory, add_semantic
from datetime import datetime


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
def compress_memory(user_id: str):

    data = load_memory(user_id)

    episodic = data.get("episodic", [])

    if len(episodic) < 10:
        return "Nada para comprimir ainda."

    # pega últimos eventos
    recent = episodic[-20:]

    # cria resumo simples
    summary = build_summary(recent)

    # guarda como memória semântica consolidada
    add_semantic(
        user_id,
        fact=summary,
        confidence=0.9
    )

    # limpa memória episódica antiga (mantém só recentes)
    data["episodic"] = episodic[-10:]

    save_memory(user_id, data)

    return summary


# -----------------------------
# SIMPLE SUMMARIZER (SEM LLM AINDA)
# -----------------------------
def build_summary(events):

    questions = [e["question"] for e in events]
    answers = [e["answer"] for e in events]

    summary = f"""
Resumo automático ({datetime.utcnow().isoformat()}):

O utilizador fez {len(events)} interações recentes.
Principais temas incluem:

- perguntas sobre documentos e análise de conteúdo
- exploração de informação contextual
- evolução progressiva de entendimento

Últimas perguntas:
- {questions[-3:] if len(questions) >= 3 else questions}

Insight geral:
O sistema está a construir conhecimento incremental sobre o utilizador e os seus documentos.
""".strip()

    return summary