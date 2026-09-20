from __future__ import annotations

from typing import Any, TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str | int
    title: str
    text: str
    published: str
    version: str
    score: float


class ClaimAssessment(TypedDict):
    claim: str
    status: str
    verdict: str
    evidence: list[RetrievedChunk]
    evidence_chunk_ids: list[str | int]
    reasoning: str
    conflicting_sources: list[RetrievedChunk]
    conflict_details: list[dict[str, str]]
    recency_note: str


class AgentState(TypedDict, total=False):
    question: str
    conversation_history: list[dict[str, str]]
    query: str
    question_type: str
    is_follow_up: bool
    is_multi_hop: bool
    retrieved_chunks: list[RetrievedChunk]
    claim_assessments: list[ClaimAssessment]
    answer: str
