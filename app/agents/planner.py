from __future__ import annotations

import re

from app.state import AgentState


_FOLLOW_UP_MARKERS = {
    "it",
    "that",
    "this",
    "they",
    "them",
    "those",
    "more",
    "also",
    "why",
    "how about",
}
_MULTI_HOP_MARKERS = {"and", "compare", "difference", "versus", " vs ", "both"}


def _history_text(history: list[dict[str, str]]) -> str:
    return " ".join(
        message.get("content", "")
        for message in history[-4:]
        if message.get("content")
    )


def plan(state: AgentState) -> AgentState:
    question = state["question"].strip()
    history = state.get("conversation_history", [])
    lowered = question.lower()
    history_text = _history_text(history)
    is_follow_up = bool(history) and (
        len(lowered.split()) < 8
        or any(marker in lowered for marker in _FOLLOW_UP_MARKERS)
    )
    query = f"{history_text} {question}".strip() if is_follow_up else question
    is_multi_hop = any(marker in lowered for marker in _MULTI_HOP_MARKERS)

    if is_multi_hop:
        question_type = "multi_hop"
    elif lowered.startswith(("what", "who", "when", "where")):
        question_type = "fact_lookup"
    elif lowered.startswith(("how", "why")):
        question_type = "explanation"
    else:
        question_type = "general"

    return {
        "query": query,
        "question_type": question_type,
        "is_follow_up": is_follow_up,
        "is_multi_hop": is_multi_hop,
    }
