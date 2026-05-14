import gradio as gr

from rag import ingest_document
from utils.rag_search import search_document
from utils.llm import ask_llm
from utils.vector_store import reset_collection

from utils.memory_manager import process_memory
from utils.memory_compressor import compress_memory


document_loaded = False
USER_ID = "default_user"


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

    answer = ask_llm(
        USER_ID,
        message,
        sources,
        history
    )

    process_memory(
        USER_ID,
        message,
        answer
    )

    compress_memory(USER_ID)

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

    gr.Markdown("# 📚 Local RAG Agent + Memory")

    upload = gr.File()

    upload_button = gr.Button("Indexar Documento")

    upload_status = gr.Textbox()

    upload_button.click(
        upload_document,
        inputs=upload,
        outputs=upload_status
    )

    chatbot = gr.ChatInterface(fn=chat)


app.launch()