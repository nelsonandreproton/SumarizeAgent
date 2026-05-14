from openai import OpenAI

from utils.vector_store import get_all_chunks


client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

MODEL = "qwen2.5-3b-instruct"

MAX_CHARS_PER_DOC = 12000


def compare_documents(doc_id_a: str, filename_a: str, doc_id_b: str, filename_b: str) -> str:

    chunks_a = get_all_chunks(doc_id_a)
    chunks_b = get_all_chunks(doc_id_b)

    text_a = _flatten(chunks_a, MAX_CHARS_PER_DOC)
    text_b = _flatten(chunks_b, MAX_CHARS_PER_DOC)

    prompt = f"""Compara os dois documentos abaixo e produz um relatório de diferenças estruturado.

Para cada diferença encontrada, indica:
- O que existe em "{filename_a}" mas não em "{filename_b}"
- O que existe em "{filename_b}" mas não em "{filename_a}"
- O que existe em ambos mas com conteúdo diferente (ex: datas, valores, cláusulas alteradas)

Formato da resposta:
## Apenas em {filename_a}
- ...

## Apenas em {filename_b}
- ...

## Diferente entre os dois
- [tema]: em {filename_a} diz "...", em {filename_b} diz "..."

## Resumo
...

---

DOCUMENTO A ({filename_a}):
{text_a}

---

DOCUMENTO B ({filename_b}):
{text_b}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "És um assistente especializado em análise e comparação de documentos legais e contratuais."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=1500
    )

    return response.choices[0].message.content


def _flatten(chunks: list, max_chars: int) -> str:
    parts = []
    total = 0

    for c in chunks:
        text = c["text"]

        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(text[:remaining])
            break

        parts.append(text)
        total += len(text)

    return "\n\n".join(parts)
