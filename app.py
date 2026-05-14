import gradio as gr

from rag import ingest_document
from utils.rag_search import search_document
from utils.llm import ask_llm
from utils.vector_store import reset_collection


document_loaded = False


def upload_document(file):

    global document_loaded

    reset_collection()

    chunks = ingest_document(file.name)

    document_loaded = True

    return f"Documento indexado com {chunks} chunks"


def chat(message, history):

    if not document_loaded:
        return "Primeiro faz upload de um documento."

    sources = search_document(message)

    answer = ask_llm(message, sources, history)

    # formatar resposta com fontes
    formatted_sources = "\n".join([
        f"- {s['source']} (chunk {s['chunk_id']}) | score: {s['score']}"
        for s in sources
    ])

    return f"""{answer}

---

📚 Fontes:
{formatted_sources}
"""

with gr.Blocks() as app:

    gr.Markdown("# Local RAG Agent")

    upload = gr.File()

    upload_button = gr.Button("Indexar Documento")

    upload_status = gr.Textbox()

    upload_button.click(
        upload_document,
        inputs=upload,
        outputs=upload_status
    )

    chatbot = gr.ChatInterface(
    fn=chat
)

app.launch()