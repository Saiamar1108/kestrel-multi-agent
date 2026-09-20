from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Simple local embedding wrapper using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in embedding]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in row] for row in embeddings]
