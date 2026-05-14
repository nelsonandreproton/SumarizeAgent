from utils.embeddings import model
from utils.vector_store import search
from utils.reranker import rerank


def search_document(question):

    embedding = model.encode(question)

    results = search(embedding, top_k=10)

    docs = results["documents"]
    metas = results["metadatas"]
    scores = results["distances"]

    # rerank só pelos textos
    ranked_docs = rerank(question, docs, top_k=3)

    # reconstruir fontes associadas aos docs selecionados
    final_sources = []

    for doc in ranked_docs:

        idx = docs.index(doc)

        final_sources.append({
            "text": doc,
            "source": metas[idx]["source"],
            "chunk_id": metas[idx]["chunk_id"],
            "score": round(1 - scores[idx], 3)  # converte distância → relevância
        })

    return final_sources