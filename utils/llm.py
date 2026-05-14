from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

MODEL = "qwen2.5-3b-instruct"


def ask_llm(question, sources, history):

    context = "\n\n".join([s["text"] for s in sources])

    messages = [
        {
            "role": "system",
            "content": "És um assistente de documentos. Usa apenas o contexto."
        }
    ]

    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({
        "role": "user",
        "content": f"CONTEXTO:\n{context}\n\nPERGUNTA:\n{question}"
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=800
    )

    return response.choices[0].message.content