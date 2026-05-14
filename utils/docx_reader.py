from docx import Document


def read_docx(path):
    doc = Document(path)

    return "\n".join(
        paragraph.text for paragraph in doc.paragraphs
    )