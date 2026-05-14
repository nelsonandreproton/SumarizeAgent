import os
import json
import uuid
from datetime import datetime
from pathlib import Path


MEMORY_DIR = Path("memory")
MEMORY_DIR.mkdir(exist_ok=True)


# -----------------------------
# UTIL: file per user
# -----------------------------
def _get_user_file(user_id: str):
    return MEMORY_DIR / f"{user_id}.json"


# -----------------------------
# INIT MEMORY FILE
# -----------------------------
def init_user_memory(user_id: str):

    file = _get_user_file(user_id)

    if not file.exists():
        data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "episodic": [],
            "semantic": [],
            "procedural": []
        }

        with open(file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# -----------------------------
# LOAD MEMORY
# -----------------------------
def load_memory(user_id: str):

    file = _get_user_file(user_id)

    if not file.exists():
        init_user_memory(user_id)

    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# SAVE MEMORY
# -----------------------------
def save_memory(user_id: str, data: dict):

    file = _get_user_file(user_id)

    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# ADD EPISODIC MEMORY
# -----------------------------
def add_episodic(user_id: str, question: str, answer: str, sources=None):

    data = load_memory(user_id)

    data["episodic"].append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources or []
    })

    save_memory(user_id, data)


# -----------------------------
# ADD SEMANTIC MEMORY
# -----------------------------
def add_semantic(user_id: str, fact: str, confidence: float = 0.8):

    data = load_memory(user_id)

    data["semantic"].append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "fact": fact,
        "confidence": confidence
    })

    save_memory(user_id, data)


# -----------------------------
# ADD PROCEDURAL MEMORY
# -----------------------------
def set_procedural(user_id: str, key: str, value: str):

    data = load_memory(user_id)

    # overwrite or update preference
    existing = {p["key"]: p for p in data["procedural"]}

    existing[key] = {
        "key": key,
        "value": value,
        "timestamp": datetime.utcnow().isoformat()
    }

    data["procedural"] = list(existing.values())

    save_memory(user_id, data)


# -----------------------------
# GET FULL MEMORY CONTEXT
# -----------------------------
def get_memory_context(user_id: str):

    data = load_memory(user_id)

    context = []

    # procedural (system preferences)
    for p in data["procedural"]:
        context.append(f"[PREF] {p['key']} = {p['value']}")

    # semantic (facts)
    for s in data["semantic"][-20:]:
        context.append(f"[FACT] {s['fact']}")

    # episodic (recent interactions)
    for e in data["episodic"][-10:]:
        context.append(f"[Q] {e['question']}")
        context.append(f"[A] {e['answer']}")

    return "\n".join(context)