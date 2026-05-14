from utils.vector_store import query_chunks
from utils.embeddings import get_embeddings


# -----------------------------
# MAIN SEARCH FUNCTION
# -----------------------------
def search_document(query, document_id=None, n_results=8):

    query_embedding = get_embeddings(query)[0]

    results = query_chunks(
        query_embedding=query_embedding,
        n_results=n_results,
        document_id=document_id
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted = []

    for doc, meta, dist in zip(documents, metadatas, distances):

        formatted.append({
            "text": doc,
            "source": meta.get("filename", "unknown"),
            "document_id": meta.get("document_id"),
            "chunk_id": meta.get("chunk_id"),
            "score": float(1 - dist)  # cosine similarity approx
        })

    # ordenar por relevância
    formatted.sort(key=lambda x: x["score"], reverse=True)

    return formatted


# -----------------------------
# DEBUG VIEW (OPTIONAL)
# -----------------------------
def search_debug(query):

    results = search_document(query)

    print("\n🔎 SEARCH RESULTS\n")

    for r in results:
        print(f"[{r['score']:.3f}] {r['source']} - chunk {r['chunk_id']}")
        print(r["text"][:200])
        print("-" * 50)

    return results