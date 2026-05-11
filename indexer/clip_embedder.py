"""CLIP image+text embedder. Both modalities live in the same 512-dim space.

We use `sentence-transformers/clip-ViT-B-32`, which exposes the same image and
text encoders OpenAI's CLIP shipped, with a sentence-transformers wrapper.
HEIC support requires `pillow-heif` (registered globally on import).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageFile
from sentence_transformers import SentenceTransformer

# Register HEIC/HEIF loaders with PIL so .heic photos open transparently.
pillow_heif.register_heif_opener()
# Apple sometimes truncates .heic on disk; tell PIL to load best-effort.
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLIP_MODEL = "clip-ViT-B-32"
CLIP_DIM = 512


class ClipEmbedder:
    """Singleton wrapper. The model is heavy to load; share one across the run."""

    _instance: "ClipEmbedder | None" = None

    @classmethod
    def shared(cls) -> "ClipEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.model = SentenceTransformer(CLIP_MODEL)

    def embed_images(self, paths: list[Path], batch_size: int = 32) -> tuple[np.ndarray, list[int]]:
        """Returns (embeddings, kept_indices) — kept_indices points back into `paths`
        for entries that loaded successfully. Failed loads are silently dropped.
        """
        kept: list[int] = []
        images: list[Image.Image] = []
        for i, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            images.append(img)
            kept.append(i)
        if not images:
            return np.empty((0, CLIP_DIM), dtype="float32"), []
        vecs = self.model.encode(
            images,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        return vecs.astype("float32"), kept

    def embed_text(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def embed_text_one(self, text: str) -> np.ndarray:
        return self.embed_text([text])[0]


if __name__ == "__main__":
    import sys
    e = ClipEmbedder()
    # Quick sanity: text-text similarity should be high for related phrases.
    t = e.embed_text(["a photo of a sunset", "evening sky", "spreadsheet of finances"])
    sim_close = float(np.dot(t[0], t[1]))
    sim_far = float(np.dot(t[0], t[2]))
    print(f"sunset ↔ evening sky:    {sim_close:.3f}")
    print(f"sunset ↔ spreadsheet:    {sim_far:.3f}")
    if len(sys.argv) > 1:
        from pathlib import Path
        img_path = Path(sys.argv[1])
        v, _ = e.embed_images([img_path])
        for label in ["a sunset", "a meme", "a screenshot of code", "a person", "an animal"]:
            tv = e.embed_text_one(label)
            sim = float(np.dot(tv, v[0]))
            print(f"  '{label:30s}'  sim={sim:.3f}")
