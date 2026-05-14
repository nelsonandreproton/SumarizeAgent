# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
python app.py

# Install dependencies
pip install -r requirements.txt

# Run tests (pytest)
pytest

# Run a single test
pytest tests/test_rag.py::test_ingest_document
```

LM Studio must be running locally on `http://localhost:1234` with the `qwen2.5-3b-instruct` model loaded before starting the app.

## Architecture

**Request flow:**

1. User uploads PDF/DOCX via Gradio → `app.py:upload_document`
2. `rag.py:ingest_document` extracts text, splits into 800-word overlapping chunks (150-word overlap)
3. Chunks embedded via `utils/embeddings.py` (SentenceTransformer `intfloat/multilingual-e5-small`, E5 requires `"passage: "` prefix on all texts to be indexed)
4. Embeddings stored in ChromaDB (`chroma_db/` dir, collection `"documents"`) via `utils/vector_store.py`
5. On chat: query embedded with `"query: "` prefix → cosine similarity search → reranked by `utils/reranker.py` (CrossEncoder `BAAI/bge-reranker-base`)
6. LLM called via `utils/llm.py` using OpenAI-compatible API pointed at LM Studio

**Memory layer** (`utils/memory_*.py`):

- `memory_store.py` — per-user JSON files in `memory/`, three types: `episodic` (Q&A pairs), `semantic` (extracted facts), `procedural` (preferences)
- `memory_manager.py` — auto-classifies each interaction; extracts facts via Portuguese keyword heuristics; detects user preferences from question phrasing
- `memory_rag.py` — bag-of-words overlap scoring to surface relevant past interactions (no embeddings — simple but intentional MVP)
- `memory_compressor.py` — when episodic count ≥ 10, condenses oldest 20 entries into a semantic summary and trims episodic to last 10

**Agent mode** (`utils/agent_tools.py`, `utils/agent_runner.py`):

- `smolagents.ToolCallingAgent` with `OpenAIServerModel` pointing at LM Studio — same model as chat tab
- 5 tools: `SearchDocumentTool`, `ExtractEntitiesTool`, `TaskListTool`, `GenerateEmailTool`, `SummarizeTool`
- All tools use `search_document()` internally — they operate on whatever docs are in ChromaDB at query time
- `run_agent(query)` returns `(final_answer, steps)` — steps are `ActionStep` logs (tool calls + observations)
- Agent tab shares `doc_count` state with chat tab via `additional_inputs` — must index docs in chat tab first
- `ExtractEntitiesTool` and `TaskListTool` use heuristic regex (no extra LLM call) for speed; `GenerateEmailTool` retrieves context and lets the agent LLM compose the final email
- `max_steps=6` — prevents runaway loops on small local models

**Document comparison** (`utils/comparator.py`):

- Upload two docs via "Comparar Documentos" tab — each gets its own `document_id`, both coexist in ChromaDB
- `compare_documents()` fetches all chunks for each doc via `vector_store.get_all_chunks()`, truncates to 12K chars each, sends structured diff prompt to LLM
- Output format: three sections — only in A / only in B / different between both + summary
- The chat tab's "Indexar Documento" calls `reset_collection()` which wipes comparison docs — tabs are independent workflows

**Key constraints:**
- `app.py` uses `gr.State` for `doc_count` (chat tab) and `compare_state` (compare tab) — never global mutable dicts
- Chat tab: multi-doc mode — accumulates docs without reset; "Limpar Todos" calls `reset_collection()`; `doc_count` state passed via `additional_inputs` to `ChatInterface`
- Compare tab: two-doc mode — docs also accumulate; switching to chat tab and clearing wipes compare docs too (shared ChromaDB collection)
- `utils/summarizer.py` is a standalone summarizer (not wired into the Gradio app) — can be called independently
- `memory_store.py` uses `datetime.utcnow()` — should be migrated to `datetime.now(UTC)` (Python 3.12+ deprecation)

## LLM / Embedding Models

| Component | Model |
|-----------|-------|
| LLM | `qwen2.5-3b-instruct` via LM Studio (`http://localhost:1234/v1`) |
| Embeddings | `intfloat/multilingual-e5-small` (SentenceTransformers) |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder) |

E5 embedding prefix rule: `get_embeddings(texts, is_query=False)` — pass `is_query=True` for search queries (uses `"query: "` prefix), default uses `"passage: "` for indexing.
