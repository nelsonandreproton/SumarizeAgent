import chromadb
import uuid

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)


def add_chunks(chunks, embeddings, filename):

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatas = [
        {
            "source": filename,
            "chunk_id": i
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )


def search(query_embedding, top_k=5):

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0]
    }


def reset_collection():

    global collection

    try:
        client.delete_collection("documents")
    except:
        pass

    collection = client.get_or_create_collection(
        name="documents"
    )