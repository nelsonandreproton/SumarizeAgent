from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)


# -----------------------------
# MAIN FUNCTION (ALINHADA COM RAG)
# -----------------------------
def get_embeddings(texts):

    if isinstance(texts, str):
        texts = [texts]

    # E5 funciona melhor com prefixos (importante!)
    texts = [f"passage: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()