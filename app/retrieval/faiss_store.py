from __future__ import annotations

import re
from collections import Counter

import faiss
import numpy as np

from app.retrieval.corpus_loader import CorpusChunk
from app.retrieval.embeddings import LocalEmbedder

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "in", "is", "it", "many", "of", "on", "or",
    "that", "the", "this", "to", "was", "what", "when", "where", "which", "who", "with"
}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_/-]+", text.lower()) if len(t) > 1]


def _rerank_score(query: str, chunk: CorpusChunk, faiss_score: float) -> float:
    query_tokens = [w for w in _tokenize(query) if w not in STOPWORDS]
    if not query_tokens:
        return faiss_score

    full_text = f"{chunk.title} {chunk.text}".lower()
    text_tokens = _tokenize(full_text)
    tf = Counter(text_tokens)

    unique_q = set(query_tokens)
    overlap_ratio = len(unique_q & set(tf.keys())) / len(unique_q) if unique_q else 0.0

    title_tokens = set(_tokenize(chunk.title))
    title_match = len(unique_q & title_tokens) / len(unique_q) if unique_q else 0.0

    proper_nouns = [
        w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z0-9_/-]+\b", query)
        if w.lower() not in STOPWORDS
    ]
    pn_match = (
        sum(1 for pn in set(proper_nouns) if pn in full_text) / len(set(proper_nouns))
        if proper_nouns
        else 0.0
    )

    boost = 0.35 * overlap_ratio + 0.25 * pn_match + 0.15 * title_match
    return faiss_score + boost


class FAISSStore:
    def __init__(self) -> None:
        self.embedder = LocalEmbedder()
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[CorpusChunk] = []

    def build(self, chunks: list[CorpusChunk]) -> None:
        if not chunks:
            self.index = faiss.IndexFlatIP(0)
            self.chunks = []
            return

        texts = [f"{chunk.title}\n{chunk.text}" for chunk in chunks]
        vectors = self.embedder.embed_texts(texts)
        matrix = np.asarray(vectors, dtype="float32")

        dimension = matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(matrix)
        self.chunks = list(chunks)

    def search(self, query: str, k: int = 5) -> list[tuple[CorpusChunk, float]]:
        if self.index is None:
            raise ValueError("FAISS index has not been built yet.")

        if not self.chunks:
            return []

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        candidate_k = max(k * 5, 25)
        if candidate_k > len(self.chunks):
            candidate_k = len(self.chunks)

        query_vector = np.asarray(self.embedder.embed_text(query), dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, candidate_k)

        candidates: list[tuple[CorpusChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            reranked_score = _rerank_score(query, chunk, float(score))
            candidates.append((chunk, reranked_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

