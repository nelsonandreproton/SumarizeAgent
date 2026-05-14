from openai import OpenAI

from utils.memory_store import get_memory_context
from utils.memory_rag import search_memory


# -----------------------------
# LM STUDIO CLIENT
# -----------------------------
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # obrigatório mas ignorado
)

MODEL = "qwen2.5-3b-instruct"


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def ask_llm(user_id, question, sources, history):

    # 1. memória estruturada
    memory_context = get_memory_context(user_id)

    # 2. memória relevante (RAG)
    memory_hits = search_memory(user_id, question)

    memory_rag_context = "\n".join([
        f"[MEM-{m['type']}] {m['text']} (score: {m['score']:.2f})"
        for m in memory_hits
    ])

    # 3. documentos com referências numeradas
    docs_context = "\n\n".join([
        f"[DOC {i+1} | {s['source']} | chunk {s['chunk_id']}]\n{s['text']}"
        for i, s in enumerate(sources)
    ])

    messages = [
        {
            "role": "system",
            "content": f"""
És um assistente inteligente.

Usa SEMPRE:
- documentos
- memória do utilizador
- memória relevante (RAG)

REGRA DE CITAÇÃO OBRIGATÓRIA:
Sempre que usares informação de um documento, cita inline no formato:
(segundo [nome do ficheiro], chunk [N])
Exemplo: "O contrato foi assinado em 2023 (segundo relatorio.pdf, chunk 4)."
Nunca respondas sem citar as fontes usadas.

MEMÓRIA ESTRUTURADA:
{memory_context}

MEMÓRIA RELEVANTE:
{memory_rag_context}
"""
        }
    ]

    # 4. histórico Gradio
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    # 5. pergunta atual
    messages.append({
        "role": "user",
        "content": f"""
DOCUMENTOS:
{docs_context}

PERGUNTA:
{question}
"""
    })

    # 6. chamada LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=800
    )

    return response.choices[0].message.content