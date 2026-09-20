from __future__ import annotations

import os
import re
import time
from pathlib import Path

from app.state import AgentState, RetrievedChunk


def _load_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GROQ_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if "YOUR_KEY=" in key:
        key = key.replace("YOUR_KEY=", "")
    if key:
        os.environ["GROQ_API_KEY"] = key
    return key


def normalize_citations(text: str, evidence: list[RetrievedChunk]) -> str:
    text = re.sub(r"[【\(\[]\s*([a-zA-Z0-9_\-]+:\d+)[^\]】\)]*[\]】\)]", r"[\1]", text)
    chunk_map = {str(c["chunk_id"]): c["title"] for c in evidence}
    for chunk_id, title in chunk_map.items():
        canonical = f"[{chunk_id} — {title}]"
        pattern = re.compile(r"\[\s*" + re.escape(chunk_id) + r"\b[^\]]*\]")
        text = pattern.sub(canonical, text)
    return text


def _clean_claim(text: str) -> str:
    lines = [re.sub(r"^#+\s*", "", line).strip() for line in text.splitlines()]
    text = " ".join(line for line in lines if line)
    text = re.split(r"\s+[—-]\s+", text, maxsplit=1)[0]
    clauses = re.split(r";\s+|\s+because\s+|\s+so that\s+", text, maxsplit=1)
    return clauses[0].strip().rstrip(".")


def fallback_synthesize(
    state: AgentState,
    status: str,
    evidence: list[RetrievedChunk],
    conflict_details: list[dict[str, str]],
) -> str:
    if status == "insufficient_evidence" or not evidence:
        return "The corpus does not contain sufficient evidence to answer this question."

    if status == "conflicting_evidence" and conflict_details:
        lines = []
        assessment = state.get("claim_assessments", [])[0] if state.get("claim_assessments") else {}
        sources = {str(c["chunk_id"]): c for c in assessment.get("conflicting_sources", [])}
        for detail in conflict_details:
            cid = detail["chunk_id"]
            source = sources.get(cid)
            title = source["title"] if source else "Source"
            claim = _clean_claim(detail["claim"])
            lines.append(f"{claim} [{cid} — {title}]")
        if sources:
            newest = max(
                sources.values(),
                key=lambda c: (str(c.get("published", "")), str(c.get("version", ""))),
            )
            lines.append(
                f"The newer evidence is version {newest.get('version', '')} published {newest.get('published', '')} [{newest['chunk_id']} — {newest['title']}]."
            )
        return " ".join(lines[:5])

    claims = []
    for chunk in evidence[:3]:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            cleaned = _clean_claim(sentence)
            if cleaned and len(cleaned.split()) > 4:
                claims.append(f"{cleaned} [{chunk['chunk_id']} — {chunk['title']}]")
                break
        if len(claims) >= 3:
            break
    if not claims:
        return "The corpus does not contain sufficient evidence to answer this question."
    return " ".join(claims)


def synthesize(state: AgentState) -> AgentState:
    assessment = state.get("claim_assessments", [])[0] if state.get("claim_assessments") else {}
    status = assessment.get("verdict", assessment.get("status", "insufficient_evidence"))
    evidence = assessment.get("evidence", [])
    conflict_details = assessment.get("conflict_details", [])

    if status == "insufficient_evidence" or not evidence:
        return {"answer": "The corpus does not contain sufficient evidence to answer this question."}

    api_key = _load_groq_key()
    if api_key:
        try:
            from groq import Groq

            client = Groq(api_key=api_key)

            evidence_snippets = []
            for c in evidence[:4]:
                clean_text = re.sub(r"#+\s*", "", c["text"])
                snippet = (
                    f"[Chunk ID: {c['chunk_id']} | Title: {c['title']} | "
                    f"Published: {c.get('published', '')} | Version: {c.get('version', '')}]\n"
                    f"{clean_text[:600]}"
                )
                evidence_snippets.append(snippet)

            evidence_str = "\n---\n".join(evidence_snippets)
            is_multi_hop = (
                state.get("is_multi_hop", False)
                or status == "conflicting_evidence"
                or bool(conflict_details)
            )
            max_sentences = 5 if is_multi_hop else 3

            system_prompt = f"""Synthesize a direct, concise answer using ONLY the provided evidence.

RULES:
1. Output ONLY the direct answer. Never copy source paragraphs, section titles, or headings verbatim. Rephrase into your own words.
2. Do NOT output headings or verbatim source titles such as 'Queryable retention by plan' or 'Under version 2...'.
3. Sentence count: Maximum {max_sentences} concise sentences.
4. Add inline citations immediately after each supported claim using EXACTLY this format:
   [chunk_id — title]
   (e.g., [spec-beacons:3 — Beacons: Alerting Specification])
5. If the evidence does not contain sufficient information to answer the question, output ONLY:
   'The corpus does not contain sufficient evidence to answer this question.'
6. For conflicts, summarize the competing facts and identify the newer/more authoritative source based on publication date/version metadata.
7. No intro, outro, preamble, or extra sections."""

            user_prompt = f"Question: {state['question']}\nVerdict: {status}\nEvidence:\n{evidence_str}"

            for attempt in range(3):
                try:
                    res = client.chat.completions.create(
                        model="groq/compound-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                    )
                    raw_answer = res.choices[0].message.content.strip()
                    raw_answer = re.sub(r"^[#\s]+", "", raw_answer)
                    answer = normalize_citations(raw_answer, evidence)
                    return {"answer": answer}
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        break
        except Exception:
            pass

    answer = fallback_synthesize(state, status, evidence, conflict_details)
    answer = normalize_citations(answer, evidence)
    return {"answer": answer}

