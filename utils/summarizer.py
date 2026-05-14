from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

MODEL = "qwen2.5-3b-instruct"


def summarize(text):

    prompt = f"""
Faz um resumo claro e estruturado do seguinte documento.

Inclui:
- principais tópicos
- decisões importantes
- ações importantes
- conclusão final

Documento:
{text[:15000]}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "És um assistente especializado em sumarização de documentos."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1200
    )

    return response.choices[0].message.content