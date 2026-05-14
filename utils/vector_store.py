import chromadb
import uuid

from chromadb.config import Settings


# -----------------------------
# CLIENT + COLLECTION
# -----------------------------
client = chromadb.PersistentClient(
    path="chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name="documents"
)


# -----------------------------
# RESET (opcional)
# -----------------------------
def reset_collection():
    global collection

    client.delete_collection("documents")

    collection = client.get_or_create_collection(
        name="documents"
    )


# -----------------------------
# ADD CHUNKS (MULTI-DOC SUPPORTED)
# -----------------------------
def add_chunks(chunks, embeddings, filename, document_id):

    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):

        chunk_id = f"{document_id}_{i}"

        ids.append(chunk_id)

        metadatas.append({
            "document_id": document_id,
            "filename": filename,
            "chunk_id": i
        })

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )


# -----------------------------
# GET ALL CHUNKS FOR A DOCUMENT
# -----------------------------
def get_all_chunks(document_id: str):

    results = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"]
    )

    chunks = []

    for doc, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({
            "text": doc,
            "chunk_id": meta.get("chunk_id"),
            "filename": meta.get("filename")
        })

    chunks.sort(key=lambda x: x["chunk_id"])

    return chunks


# -----------------------------
# QUERY
# -----------------------------
def query_chunks(query_embedding, n_results=8, document_id=None):

    where_filter = None

    if document_id:
        where_filter = {"document_id": document_id}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    return results