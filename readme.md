# 📚 Local RAG Agent (Offline AI Document Assistant)

Um sistema local de Retrieval-Augmented Generation (RAG) que permite fazer perguntas sobre documentos PDF e DOCX usando um modelo LLM local (via LM Studio) e embeddings locais.

---

## 🚀 Features

- 📄 Upload de documentos (PDF / DOCX)
- ✂️ Chunking inteligente de texto
- 🧠 Embeddings locais (SentenceTransformers)
- 🗃️ Vector database (ChromaDB)
- 🔎 Semantic search + reranking
- 💬 Chat interface (Gradio)
- 📚 Respostas baseadas em contexto real do documento
- 📌 Fontes com ficheiro + chunk + score

---

## 🏗️ Arquitetura

Document → Chunking → Embeddings → ChromaDB
↓
User Query → Embedding → Search → Rerank → LLM (LM Studio) → Answer

---

## ⚙️ Tecnologias

- Python
- Gradio 6
- ChromaDB
- SentenceTransformers
- LM Studio (local LLM)
- Hugging Face embeddings (opcional)

---

## 🧠 Como funciona

1. Faz upload de um PDF ou DOCX
2. O sistema divide o documento em chunks
3. Gera embeddings e guarda no ChromaDB
4. Quando perguntas algo:
   - Faz semantic search
   - Reordena resultados (reranker)
   - Envia contexto ao LLM local
   - Devolve resposta com fontes

---

## ▶️ Como correr

```bash
pip install -r requirements.txt
python app.py

📦 Estrutura do projeto

app.py
rag.py
utils/
    llm.py
    embeddings.py
    chunker.py
    vector_store.py
    rag_search.py
    reranker.py

📌 Próximas melhorias
6) 🧠 Memory Layer (histórico inteligente)

Em vez de apenas chat history:
Guardar perguntas frequentes
Guardar insights extraídos automaticamente
Criar resumos persistentes por documento
Evolução de memória por utilizador

7) 🤖 Agent Mode (nível avançado)

Integração com agentes usando smolagents:

“Vai buscar informação ao documento”
“Cria lista de tarefas baseada no conteúdo”
“Extrai entidades importantes (nomes, datas, eventos)”
“Gera emails automaticamente com base no documento”

📦 Framework: https://huggingface.co/docs/smolagents/index

🔮 Roadmap futuro
Streaming de tokens
Multi-document workspace
UI estilo ChatGPT/NotebookLM
OCR para PDFs escaneados
Reranker avançado
Memory persistente cross-session

🧑‍💻 Autor
Nelson André