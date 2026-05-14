from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)


def create_embeddings(chunks):

    embeddings = model.encode(chunks)

    return embeddings