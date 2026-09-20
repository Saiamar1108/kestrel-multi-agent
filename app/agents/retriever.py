from __future__ import annotations

from functools import lru_cache

from app.retrieval.corpus_loader import load_corpus
from app.retrieval.faiss_store import FAISSStore
from app.state import AgentState, RetrievedChunk


@lru_cache(maxsize=1)
def _store() -> FAISSStore:
    store = FAISSStore()
    store.build(load_corpus())
    return store


def retrieve(state: AgentState) -> AgentState:
    results = _store().search(state["query"], k=5)
    retrieved_chunks: list[RetrievedChunk] = [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "text": chunk.text,
            "published": chunk.published,
            "version": chunk.version,
            "score": score,
        }
        for chunk, score in results
    ]
    return {"retrieved_chunks": retrieved_chunks}
