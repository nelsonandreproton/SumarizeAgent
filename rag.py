from utils.pdf_reader import read_pdf
from utils.chunker import chunk_text
from utils.embeddings import create_embeddings
from utils.vector_store import add_chunks

from pathlib import Path


def ingest_document(path):

    text = read_pdf(path)

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    filename = Path(path).name

    add_chunks(
        chunks,
        embeddings,
        filename
    )

    return len(chunks)