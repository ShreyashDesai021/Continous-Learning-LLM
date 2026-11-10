"""BGE-M3 embedding client with D: drive cache management and local embedding cache."""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Union
import numpy as np

# Point HuggingFace home explicitly to D: drive
HF_STORAGE_DIR = "runtime/research/models"
os.environ.setdefault("HF_HOME", HF_STORAGE_DIR)
os.environ["TRANSFORMERS_CACHE"] = HF_STORAGE_DIR

from sentence_transformers import SentenceTransformer


class BGEM3EmbeddingClient:
    """Wrapper for BAAI/bge-m3 dense embeddings running on CPU with disk caching."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str = HF_STORAGE_DIR,
        device: str = "cpu",
        normalize: bool = True
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.normalize = normalize
        
        # Local vector cache directory to prevent re-computing on identical chunks
        self.vector_cache_dir = self.cache_dir.parent / "vector_cache"
        self.vector_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load model to avoid memory overhead when not actively embedding."""
        if self._model is None:
            print(f"[Embedding] Loading {self.model_name} onto {self.device} (cache: {self.cache_dir})...")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_dir)
            )
        return self._model

    def _get_cache_key(self, text: str) -> str:
        """Generates MD5 hash for cached vector lookup."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """Embeds a list of texts using BGE-M3 dense representation with caching."""
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []

        for idx, text in enumerate(texts):
            key = self._get_cache_key(text)
            cache_file = self.vector_cache_dir / f"{key}.npy"
            if cache_file.exists():
                vec = np.load(cache_file)
                embeddings.append((idx, vec))
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(idx)

        if texts_to_compute:
            computed_vecs = self.model.encode(
                texts_to_compute,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )
            for idx, vec in zip(indices_to_compute, computed_vecs):
                key = self._get_cache_key(texts[idx])
                cache_file = self.vector_cache_dir / f"{key}.npy"
                np.save(cache_file, vec)
                embeddings.append((idx, vec))

        # Sort back to original order
        embeddings.sort(key=lambda x: x[0])
        return np.array([x[1] for x in embeddings], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string."""
        vecs = self.embed_texts([query], batch_size=1)
        return vecs[0]
