from __future__ import annotations

import re

from app.state import AgentState, ClaimAssessment, RetrievedChunk


def _recency_key(chunk: RetrievedChunk) -> tuple[str, str]:
    return (str(chunk["published"]), str(chunk["version"]))


def _sentences(chunk: RetrievedChunk) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]) if sentence.strip()]


def _fact_tokens(sentence: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_/-]{2,}", sentence.lower()))


def _numbers(sentence: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", sentence))


def _document_key(chunk: RetrievedChunk) -> str:
    if chunk.get("doc_id"):
        return str(chunk["doc_id"])
    return str(chunk["chunk_id"]).rsplit(":", 1)[0]


def _scope_terms(sentence: str) -> set[str]:
    terms = _fact_tokens(sentence)
    return terms & {"starter", "growth", "scale", "eu", "us", "android", "ios", "python", "go"}


def _proposition_tokens(sentence: str) -> set[str]:
    tokens = _fact_tokens(sentence)
    tokens -= _numbers(sentence)
    tokens -= {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "to", "for", "and", "or", "with", "from", "on", "in", "by", "can", "may", "will", "should"}
    return tokens


def _is_contradictory(first: str, second: str) -> bool:
    first_tokens = _proposition_tokens(first)
    second_tokens = _proposition_tokens(second)
    if not first_tokens or not second_tokens:
        return False

    shared_tokens = first_tokens & second_tokens
    similarity = len(shared_tokens) / len(first_tokens | second_tokens)
    if similarity < 0.7 or _scope_terms(first) != _scope_terms(second):
        return False

    first_numbers = _numbers(first)
    second_numbers = _numbers(second)
    if first_numbers and second_numbers and first_numbers != second_numbers:
        return True

    first_negative = bool(re.search(r"\b(no|not|never|cannot|can't|doesn't|isn't)\b", first.lower()))
    second_negative = bool(re.search(r"\b(no|not|never|cannot|can't|doesn't|isn't)\b", second.lower()))
    return first_negative != second_negative and similarity >= 0.8


def _find_conflicts(evidence: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], list[dict[str, str]]]:
    conflicts: list[RetrievedChunk] = []
    details: list[dict[str, str]] = []
    for index, first_chunk in enumerate(evidence):
        for second_chunk in evidence[index + 1 :]:
            if _document_key(first_chunk) == _document_key(second_chunk):
                continue
            for first_sentence in _sentences(first_chunk):
                for second_sentence in _sentences(second_chunk):
                    if not _is_contradictory(first_sentence, second_sentence):
                        continue
                    for chunk in (first_chunk, second_chunk):
                        if chunk not in conflicts:
                            conflicts.append(chunk)
                    newer = max(first_chunk, second_chunk, key=_recency_key)
                    details.extend(
                        {
                            "chunk_id": str(chunk["chunk_id"]),
                            "claim": sentence,
                            "published": str(chunk["published"]),
                            "version": str(chunk["version"]),
                            "reliability_note": (
                                "Newer published/versioned evidence is preferred."
                                if chunk is newer
                                else "Older published/versioned evidence is retained for comparison."
                            ),
                        }
                        for chunk, sentence in (
                            (first_chunk, first_sentence),
                            (second_chunk, second_sentence),
                        )
                    )
                    return conflicts, details
    return conflicts, details


def verify(state: AgentState) -> AgentState:
    evidence = state.get("retrieved_chunks", [])
    assessments: list[ClaimAssessment] = []

    if not evidence:
        assessments.append(
            {
                "claim": state["question"],
                "status": "insufficient_evidence",
                "verdict": "insufficient_evidence",
                "evidence": [],
                "evidence_chunk_ids": [],
                "reasoning": "No retrieved evidence establishes the claim.",
                "conflicting_sources": [],
                "conflict_details": [],
                "recency_note": "No chunks were retrieved.",
            }
        )
    else:
        conflicting_sources, conflict_details = _find_conflicts(evidence)
        best_score = max(chunk["score"] for chunk in evidence)
        evidence_chunk_ids = [chunk["chunk_id"] for chunk in evidence]
        if best_score < 0.25:
            status = "insufficient_evidence"
            reasoning = "Retrieved evidence scores are too low to establish the claim."
        elif conflicting_sources:
            status = "conflicting_evidence"
            reasoning = "At least two sources make materially incompatible claims about the same proposition."
        elif best_score < 0.45:
            status = "partially_supported"
            reasoning = "Retrieved evidence provides only partial support for the claim."
        else:
            status = "supported"
            reasoning = "Retrieved evidence supports the claim without an explicit contradiction."

        newest = max(evidence, key=_recency_key)
        recency_note = (
            f"Most recent retrieved evidence is published {newest['published']} "
            f"at version {newest['version']}."
        )
        assessments.append(
            {
                "claim": state["question"],
                "status": status,
                "verdict": status,
                "evidence": evidence,
                "evidence_chunk_ids": evidence_chunk_ids,
                "reasoning": reasoning,
                "conflicting_sources": conflicting_sources,
                "conflict_details": conflict_details,
                "recency_note": recency_note,
            }
        )

    return {"claim_assessments": assessments}
