from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)


# -----------------------------
# MAIN FUNCTION (ALINHADA COM RAG)
# -----------------------------
def get_embeddings(texts, is_query: bool = False):

    if isinstance(texts, str):
        texts = [texts]

    # E5 requires different prefixes: "query: " for search, "passage: " for indexing
    prefix = "query: " if is_query else "passage: "
    texts = [f"{prefix}{t}" for t in texts]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()