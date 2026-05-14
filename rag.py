import uuid
from pathlib import Path

from pypdf import PdfReader
import docx

from utils.vector_store import add_chunks
from utils.embeddings import get_embeddings


# -----------------------------
# ENTRY POINT
# -----------------------------
def ingest_document(file_path: str):

    file_path = Path(file_path)

    document_id = str(uuid.uuid4())

    if file_path.suffix.lower() == ".pdf":
        text = extract_pdf(file_path)

    elif file_path.suffix.lower() in [".docx"]:
        text = extract_docx(file_path)

    else:
        raise ValueError("Formato não suportado (apenas PDF e DOCX)")

    chunks = chunk_text(text)

    embeddings = get_embeddings(chunks)

    add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        filename=file_path.name,
        document_id=document_id
    )

    return len(chunks)


# -----------------------------
# PDF LOADER
# -----------------------------
def extract_pdf(file_path: Path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# -----------------------------
# DOCX LOADER
# -----------------------------
def extract_docx(file_path: Path):

    doc = docx.Document(file_path)

    return "\n".join([p.text for p in doc.paragraphs])


# -----------------------------
# CHUNKING (SIMPLE BUT EFFECTIVE)
# -----------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks