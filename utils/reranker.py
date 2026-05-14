from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")


def rerank(query, chunks, top_k=3):

    pairs = [(query, chunk) for chunk in chunks]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [chunk for chunk, score in ranked[:top_k]]