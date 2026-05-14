import os
import gradio as gr

from rag import ingest_document
from utils.rag_search import search_document
from utils.llm import ask_llm
from utils.vector_store import reset_collection
from utils.comparator import compare_documents

from utils.memory_manager import process_memory
from utils.memory_compressor import compress_memory


USER_ID = "default_user"


# -------------------------------------------------------
# CHAT TAB — multi-doc
# -------------------------------------------------------

def upload_documents(files, doc_count):
    if not files:
        return "Nenhum ficheiro selecionado.", doc_count

    added = 0
    names = []

    for file in files:
        n_chunks, _ = ingest_document(file.name)
        added += n_chunks
        names.append(os.path.basename(file.name))

    new_count = doc_count + len(files)
    label = ", ".join(names)
    return f"✅ {len(files)} documento(s) indexado(s): {label} | Total: {new_count} doc(s)", new_count


def clear_documents():
    reset_collection()
    return "🗑️ Todos os documentos removidos.", 0


def chat(message, history, doc_count):
    if doc_count == 0:
        return "Primeiro faz upload de pelo menos um documento."

    sources = search_document(message)

    answer = ask_llm(
        USER_ID,
        message,
        sources,
        history
    )

    process_memory(USER_ID, message, answer)
    compress_memory(USER_ID)

    unique_sources = {(s['source'], s['chunk_id']) for s in sources}
    formatted_sources = "\n".join(
        sorted(f"- {src} (chunk {cid})" for src, cid in unique_sources)
    )

    return f"""{answer}

---

📚 Fontes consultadas:
{formatted_sources}
"""


# -------------------------------------------------------
# COMPARE TAB
# -------------------------------------------------------

def load_doc_a(file, state):
    if file is None:
        return "⚠️ Seleciona um ficheiro primeiro.", state

    n_chunks, doc_id = ingest_document(file.name)
    filename = os.path.basename(file.name)

    state = dict(state)
    state["doc_id_a"] = doc_id
    state["filename_a"] = filename

    return f"✅ {filename} — {n_chunks} chunks carregados", state


def load_doc_b(file, state):
    if file is None:
        return "⚠️ Seleciona um ficheiro primeiro.", state

    n_chunks, doc_id = ingest_document(file.name)
    filename = os.path.basename(file.name)

    state = dict(state)
    state["doc_id_b"] = doc_id
    state["filename_b"] = filename

    return f"✅ {filename} — {n_chunks} chunks carregados", state


def run_comparison(state):
    if not state.get("doc_id_a") or not state.get("doc_id_b"):
        return "⚠️ Carrega os dois documentos antes de comparar."

    return compare_documents(
        state["doc_id_a"],
        state["filename_a"],
        state["doc_id_b"],
        state["filename_b"],
    )


# -------------------------------------------------------
# UI
# -------------------------------------------------------

with gr.Blocks(title="RAG Agent") as app:

    gr.Markdown("# 📚 Local RAG Agent + Memory")

    with gr.Tab("Chat com Documentos"):

        doc_count = gr.State(0)

        with gr.Row():
            upload = gr.File(
                label="Adicionar documentos (PDF ou DOCX)",
                file_count="multiple",
                file_types=[".pdf", ".docx"]
            )
            with gr.Column():
                upload_button = gr.Button("Indexar Documentos", variant="primary")
                clear_button = gr.Button("Limpar Todos", variant="stop")
                upload_status = gr.Textbox(label="Estado", interactive=False, lines=2)

        upload_button.click(
            upload_documents,
            inputs=[upload, doc_count],
            outputs=[upload_status, doc_count],
            show_progress="full"
        )

        clear_button.click(
            clear_documents,
            inputs=[],
            outputs=[upload_status, doc_count]
        )

        chatbot = gr.ChatInterface(
            fn=chat,
            additional_inputs=[doc_count]
        )

    with gr.Tab("Comparar Documentos"):

        compare_state = gr.State({
            "doc_id_a": None,
            "filename_a": None,
            "doc_id_b": None,
            "filename_b": None,
        })

        gr.Markdown("### Carrega dois documentos para comparar as diferenças")

        with gr.Row():
            with gr.Column():
                file_a = gr.File(label="Documento A", file_types=[".pdf", ".docx"])
                load_a_btn = gr.Button("Carregar A", variant="secondary")
                status_a = gr.Textbox(label="Estado A", interactive=False, value="Aguarda documento...")

            with gr.Column():
                file_b = gr.File(label="Documento B", file_types=[".pdf", ".docx"])
                load_b_btn = gr.Button("Carregar B", variant="secondary")
                status_b = gr.Textbox(label="Estado B", interactive=False, value="Aguarda documento...")

        compare_button = gr.Button("🔍 Comparar Documentos", variant="primary")
        comparison_output = gr.Markdown(value="")

        load_a_btn.click(
            load_doc_a,
            inputs=[file_a, compare_state],
            outputs=[status_a, compare_state],
            show_progress="full"
        )

        load_b_btn.click(
            load_doc_b,
            inputs=[file_b, compare_state],
            outputs=[status_b, compare_state],
            show_progress="full"
        )

        compare_button.click(
            run_comparison,
            inputs=[compare_state],
            outputs=[comparison_output],
            show_progress="full"
        )


app.launch()
