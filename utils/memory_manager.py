import re
from utils.memory_store import (
    add_episodic,
    add_semantic,
    set_procedural
)


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
def process_memory(user_id: str, question: str, answer: str):
    """
    Decide automaticamente o que guardar na memória
    """

    # sempre guardar interação base
    add_episodic(user_id, question, answer, sources=None)

    # extrair possíveis factos
    facts = extract_facts(answer)

    for fact in facts:
        add_semantic(user_id, fact, confidence=0.75)

    # detetar preferências do utilizador
    prefs = extract_preferences(question, answer)

    for key, value in prefs.items():
        set_procedural(user_id, key, value)


# -----------------------------
# SIMPLE FACT EXTRACTION
# -----------------------------
def extract_facts(text: str):
    """
    Extrai frases que parecem conhecimento útil.
    Versão simples (sem LLM ainda).
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)

    facts = []

    for s in sentences:

        s_clean = s.strip()

        if len(s_clean) < 20:
            continue

        # heurística simples
        if any(keyword in s_clean.lower() for keyword in [
            "é", "são", "consiste", "inclui", "deve", "tem", "significa"
        ]):
            facts.append(s_clean)

    return facts[:5]


# -----------------------------
# SIMPLE PREFERENCE DETECTION
# -----------------------------
def extract_preferences(question: str, answer: str):
    """
    Detecta preferências do utilizador de forma heurística.
    """

    prefs = {}

    q = question.lower()

    # estilo de resposta
    if "curto" in q or "resumo" in q:
        prefs["response_style"] = "concise"

    if "detalhado" in q or "explica bem" in q:
        prefs["response_style"] = "detailed"

    # tipo de uso
    if "codigo" in q or "programação" in q:
        prefs["domain"] = "software_engineering"

    if "saúde" in q:
        prefs["domain"] = "health"

    return prefs