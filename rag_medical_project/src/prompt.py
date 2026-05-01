from __future__ import annotations

from typing import Any, Dict, List


SYSTEM_PROMPT = (
    "You are a medical question-answering assistant. "
    "Answer only from the retrieved context. "
    "If the context is insufficient, explicitly say the provided evidence is insufficient. "
    "Do not invent facts, do not cite knowledge outside the provided context, and be medically cautious."
)


def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})
        header = (
            f"[Chunk {i}] PMID={meta.get('pmid', '')}; "
            f"Topic={meta.get('topic', '')}; "
            f"Year={meta.get('year', '')}; "
            f"Title={meta.get('title', '')}"
        )
        lines.append(header)
        lines.append(chunk.get("document", ""))
    return "\n\n".join(lines)


def build_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context = format_context(retrieved_chunks)
    return (
        f"System instruction:\n{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Instructions:\n"
        "- Use only the retrieved context.\n"
        "- Be concise, factual, and cautious.\n"
        "- If the evidence is insufficient, say so clearly.\n"
        "- Do not fabricate statistics, diagnoses, or treatment recommendations.\n"
        "- Where possible, mention which study topic the evidence refers to.\n\n"
        "Answer:"
    )
