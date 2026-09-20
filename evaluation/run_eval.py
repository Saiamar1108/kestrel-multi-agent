from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from langsmith import trace

from app.graph.workflow import workflow


INPUT_PATH = Path("results/eval_questions.jsonl")
OUTPUT_PATH = Path("results/eval_results.jsonl")
CITATION_PATTERN = re.compile(r"\[([^\]]+)\s+\u2014\s+[^\]]+\]")


def load_questions(path: Path = INPUT_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ordered_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    standalone: list[dict[str, Any]] = []
    conversation_order: list[str] = []

    for question in questions:
        conversation_id = question.get("conversation_id")
        if conversation_id is None:
            standalone.append(question)
        else:
            conversation_key = str(conversation_id)
            if conversation_key not in grouped:
                conversation_order.append(conversation_key)
            grouped[conversation_key].append(question)

    ordered: list[dict[str, Any]] = []
    ordered.extend(standalone)
    for conversation_id in conversation_order:
        ordered.extend(sorted(grouped[conversation_id], key=lambda item: item["turn"]))
    return ordered


def cited_chunk_ids(answer: str) -> list[str | int]:
    citations: list[str | int] = []
    for raw_chunk_id in CITATION_PATTERN.findall(answer):
        chunk_id: str | int = raw_chunk_id
        if raw_chunk_id.isdigit():
            chunk_id = int(raw_chunk_id)
        if chunk_id not in citations:
            citations.append(chunk_id)
    return citations


def score_result(
    result: dict[str, Any], question: dict[str, Any], retrieved_ids: list[str | int], citations: list[str | int]
) -> dict[str, float | bool | None]:
    expected_ids = question["expected_chunk_ids"]
    expected_id_set = set(expected_ids)
    retrieved_id_set = set(retrieved_ids)
    cited_id_set = set(citations)

    return {
        "expected_chunk_recall": (
            len(expected_id_set & retrieved_id_set) / len(expected_id_set)
            if expected_id_set
            else None
        ),
        "citation_chunk_recall": (
            len(expected_id_set & cited_id_set) / len(expected_id_set)
            if expected_id_set
            else None
        ),
        "answer_exact_match": (
            result.get("answer", "").strip() == question["expected_answer"].strip()
            if isinstance(question.get("expected_answer"), str)
            else None
        ),
    }


def evaluate(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    histories: dict[str, list[dict[str, str]]] = defaultdict(list)
    results: list[dict[str, Any]] = []

    for question in ordered_questions(questions):
        conversation_id = question.get("conversation_id")
        history = histories[str(conversation_id)] if conversation_id is not None else []
        started = time.perf_counter()
        trace_url = None
        with trace(
            "Kestrel workflow",
            inputs={"question": question["question"], "conversation_history": list(history)},
            project_name=None,
        ) as run:
            state = workflow.invoke(
                {
                    "question": question["question"],
                    "conversation_history": list(history),
                }
            )
            if hasattr(run, "get_url"):
                trace_url = run.get_url()
        latency_seconds = time.perf_counter() - started

        answer = state.get("answer", "")
        retrieved_chunks = state.get("retrieved_chunks", [])
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        citations = cited_chunk_ids(answer)
        assessments = state.get("claim_assessments", [])
        verifier_verdict = assessments[0]["status"] if assessments else None
        langsmith_run_url = trace_url

        results.append(
            {
                "question_id": question["question_id"],
                "answer": answer,
                "citations": citations,
                "retrieved_chunk_ids": retrieved_ids,
                "verifier_verdict": verifier_verdict,
                "scores": score_result(state, question, retrieved_ids, citations),
                "latency_seconds": latency_seconds,
                "langsmith_run_url": langsmith_run_url,
            }
        )

        if conversation_id is not None:
            history.extend(
                [
                    {"role": "user", "content": question["question"]},
                    {"role": "assistant", "content": answer},
                ]
            )

    return results


def write_results(results: list[dict[str, Any]], path: Path = OUTPUT_PATH) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")


def main() -> None:
    write_results(evaluate(load_questions()))


if __name__ == "__main__":
    main()
