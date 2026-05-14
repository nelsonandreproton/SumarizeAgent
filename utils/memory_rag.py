from utils.memory_store import load_memory
import math


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def search_memory(user_id: str, query: str, top_k: int = 5):

    data = load_memory(user_id)

    results = []

    # search semantic memory
    for item in data.get("semantic", []):
        score = simple_score(query, item["fact"])

        results.append({
            "type": "semantic",
            "text": item["fact"],
            "score": score
        })

    # search episodic memory
    for item in data.get("episodic", []):
        combined = item["question"] + " " + item["answer"]

        score = simple_score(query, combined)

        results.append({
            "type": "episodic",
            "text": combined,
            "score": score
        })

    # sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


# -----------------------------
# VERY SIMPLE SCORING (MVP)
# -----------------------------
def simple_score(query: str, text: str):

    q_words = set(query.lower().split())
    t_words = set(text.lower().split())

    overlap = len(q_words.intersection(t_words))

    if len(t_words) == 0:
        return 0.0

    return overlap / math.sqrt(len(t_words))