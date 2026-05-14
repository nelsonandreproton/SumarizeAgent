import re
from smolagents import Tool

from utils.rag_search import search_document


class SearchDocumentTool(Tool):
    name = "search_document"
    description = (
        "Searches the loaded document(s) for information relevant to a query. "
        "Returns the most relevant text passages with their source and chunk number. "
        "Use this before any other tool to retrieve context from the document."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The question or topic to search for in the document",
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        results = search_document(query, n_results=5)

        if not results:
            return "No relevant content found in the document."

        parts = []
        for r in results:
            parts.append(
                f"[{r['source']} | chunk {r['chunk_id']} | score {r['score']:.2f}]\n{r['text']}"
            )

        return "\n\n---\n\n".join(parts)


class ExtractEntitiesTool(Tool):
    name = "extract_entities"
    description = (
        "Extracts named entities from document content: people, organizations, "
        "dates, locations, monetary values, and other key terms. "
        "First searches the document for the given topic, then extracts entities from the results."
    )
    inputs = {
        "topic": {
            "type": "string",
            "description": "Topic or section to focus entity extraction on (e.g. 'parties involved', 'payment terms')",
        }
    }
    output_type = "string"

    def forward(self, topic: str) -> str:
        results = search_document(topic, n_results=6)

        if not results:
            return "No content found to extract entities from."

        combined = "\n\n".join(r["text"] for r in results)

        entities = _extract_entities_heuristic(combined)

        if not entities:
            return f"No clear entities found for topic: {topic}"

        lines = [f"Entities extracted from document (topic: {topic}):"]
        for category, items in entities.items():
            if items:
                lines.append(f"\n**{category}:**")
                for item in items:
                    lines.append(f"  - {item}")

        return "\n".join(lines)


class TaskListTool(Tool):
    name = "extract_task_list"
    description = (
        "Reads the document and generates a structured list of action items, "
        "tasks, obligations, or next steps mentioned in the content. "
        "Useful for contracts, meeting notes, project documents."
    )
    inputs = {
        "focus": {
            "type": "string",
            "description": "Optional focus area (e.g. 'deliverables', 'deadlines', 'responsibilities'). Use 'all' for general task extraction.",
        }
    }
    output_type = "string"

    def forward(self, focus: str) -> str:
        query = focus if focus.lower() != "all" else "tasks obligations deliverables deadlines responsibilities"
        results = search_document(query, n_results=8)

        if not results:
            return "No actionable content found in the document."

        combined = "\n\n".join(r["text"] for r in results)

        tasks = _extract_tasks_heuristic(combined)

        if not tasks:
            return "No clear action items detected. Consider using search_document for more specific queries."

        lines = ["**Action items / Tasks extracted from document:**\n"]
        for i, task in enumerate(tasks, 1):
            lines.append(f"{i}. {task}")

        return "\n".join(lines)


class GenerateEmailTool(Tool):
    name = "generate_email"
    description = (
        "Generates a professional email based on content from the document. "
        "Searches the document for relevant context, then drafts an email. "
        "Returns the full email text (subject + body) ready to send."
    )
    inputs = {
        "instructions": {
            "type": "string",
            "description": "What the email should be about and who it is for (e.g. 'email to client summarizing contract payment terms')",
        }
    }
    output_type = "string"

    def forward(self, instructions: str) -> str:
        results = search_document(instructions, n_results=5)

        if not results:
            context = "(no document content found — writing from instructions only)"
        else:
            context = "\n\n".join(
                f"[{r['source']} chunk {r['chunk_id']}]: {r['text']}"
                for r in results
            )

        return f"[EMAIL DRAFT — instructions: {instructions}]\n\nDOCUMENT CONTEXT RETRIEVED:\n{context}\n\n[The agent LLM will compose the final email using this context.]"


class SummarizeTool(Tool):
    name = "summarize_section"
    description = (
        "Retrieves and summarizes a specific section or topic from the document. "
        "More focused than search_document — returns a concise summary of the relevant content."
    )
    inputs = {
        "topic": {
            "type": "string",
            "description": "The section or topic to summarize (e.g. 'payment terms', 'termination clauses', 'project scope')",
        }
    }
    output_type = "string"

    def forward(self, topic: str) -> str:
        results = search_document(topic, n_results=5)

        if not results:
            return f"No content found for topic: {topic}"

        parts = []
        for r in results:
            parts.append(f"[{r['source']} | chunk {r['chunk_id']}]\n{r['text']}")

        return f"Content retrieved for '{topic}':\n\n" + "\n\n---\n\n".join(parts)


# -------------------------------------------------------
# Heuristic helpers (no LLM — fast, local)
# -------------------------------------------------------

def _extract_entities_heuristic(text: str) -> dict:
    entities = {
        "Dates": [],
        "Monetary values": [],
        "Organizations / People": [],
        "Locations": [],
    }

    # dates: dd/mm/yyyy, dd-mm-yyyy, month year, etc.
    date_patterns = [
        r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
        r"\b(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+(?:de\s+)?\d{4}\b",
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",
    ]
    for pat in date_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group().strip()
            if val not in entities["Dates"]:
                entities["Dates"].append(val)

    # monetary
    money_patterns = [
        r"\b(?:EUR|USD|GBP|€|\$|£)\s*[\d\.,]+\b",
        r"\b[\d\.,]+\s*(?:euros?|dollars?|reais?)\b",
    ]
    for pat in money_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group().strip()
            if val not in entities["Monetary values"]:
                entities["Monetary values"].append(val)

    # capitalized sequences (names, orgs) — min 2 consecutive capitalized words
    for m in re.finditer(r"\b([A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+)+)\b", text):
        val = m.group().strip()
        if len(val) > 4 and val not in entities["Organizations / People"]:
            entities["Organizations / People"].append(val)

    return {k: v[:10] for k, v in entities.items()}  # cap at 10 per category


def _extract_tasks_heuristic(text: str) -> list:
    task_keywords = [
        r"(?:deve|deverá|deverão|shall|must|should|has to|have to|is required to|são obrigados?)",
        r"(?:entregar|fornecer|submeter|apresentar|completar|realizar|executar|enviar)",
        r"(?:prazo|deadline|até|by|before|no máximo)",
        r"(?:responsável|responsible|assigned to|a cargo de)",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    tasks = []

    for s in sentences:
        s = s.strip()
        if len(s) < 20 or len(s) > 300:
            continue
        for pat in task_keywords:
            if re.search(pat, s, re.IGNORECASE):
                if s not in tasks:
                    tasks.append(s)
                break

    return tasks[:15]
