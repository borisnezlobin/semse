"""Local sentence-transformers embedding wrapper."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_one(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0]


if __name__ == "__main__":
    e = Embedder()
    out = e.embed_batch(["hello world", "the quick brown fox", "dinner plans tomorrow", "rent payment", "Sarah's birthday"])
    print("shape:", out.shape)
    print("dtype:", out.dtype)
    print("first 5 dims of vec[0]:", out[0][:5])
