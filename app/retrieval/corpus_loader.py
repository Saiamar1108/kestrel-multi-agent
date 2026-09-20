from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel


EXPECTED_CORPUS_SHA256 = "b401a4f906446f4e93f1d58c714918acf40d94d81a80c6358e5b7edd46cbbb41"
REQUIRED_FIELDS = [
    "chunk_id",
    "doc_id",
    "title",
    "category",
    "owner",
    "source_url",
    "published",
    "version",
    "text",
]


class CorpusChunk(BaseModel):
    chunk_id: str | int
    doc_id: str
    title: str
    category: str
    owner: str
    source_url: str
    published: str
    version: str
    text: str


def _verify_corpus_hash(path: str | Path) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)

    actual_hash = digest.hexdigest()
    if actual_hash != EXPECTED_CORPUS_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {file_path}. Expected {EXPECTED_CORPUS_SHA256}, "
            f"got {actual_hash}."
        )


def load_corpus(path: str = "corpus.jsonl") -> list[CorpusChunk]:
    file_path = Path(path)
    _verify_corpus_hash(file_path)

    chunks: list[CorpusChunk] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(f"Malformed JSONL in {file_path} on line {line_number}: empty line.")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {file_path} on line {line_number}: {exc.msg}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Malformed JSON in {file_path} on line {line_number}: each record must be an object."
                )

            missing_fields = [field for field in REQUIRED_FIELDS if field not in record]
            if missing_fields:
                raise ValueError(
                    f"Missing required fields in {file_path} on line {line_number}: "
                    f"{', '.join(missing_fields)}"
                )

            try:
                chunks.append(CorpusChunk(**record))
            except Exception as exc:  # pragma: no cover - surfaced as clear validation error
                raise ValueError(
                    f"Invalid corpus record in {file_path} on line {line_number}: {exc}"
                ) from exc

    return chunks


def corpus_stats(chunks: list[CorpusChunk]) -> dict:
    categories = sorted({chunk.category for chunk in chunks})
    return {
        "total_chunks": len(chunks),
        "unique_documents": len({chunk.doc_id for chunk in chunks}),
        "categories": categories,
    }


__all__ = ["CorpusChunk", "load_corpus", "corpus_stats"]
