"""
embeddings.py
=============
Sentence-transformer embeddings for menu items, add-ons, and live carts.

Architecture
------------
* Model  : ``all-MiniLM-L6-v2``  (~22 MB, 384-d, <10 ms/encode on CPU)
* Storage: FAISS IndexFlatIP for cosine-similarity retrieval
* Cart embedding = mean-pool of item embeddings ⊕ lightweight TF-IDF
  weighting so higher-priced / rarer items contribute more.

Cold-Start Handling
-------------------
New / unseen add-ons get an embedding from their *name + category
description* — no interaction history needed.  This is a key advantage
over collaborative-filtering baselines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

# Lazy-loaded model handle (avoids import-time download)
_model = None

def _get_model():
    """Lazy-load sentence-transformer model (CPU)."""
    global _model
    if _model is None:
        import torch  # Fix for transformers NameError bug (nn is not defined)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device="cpu",
        )
        logger.info("Loaded sentence-transformer model on CPU")
    return _model


def encode_items(
    texts: Sequence[str],
    batch_size: int = 256,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of item descriptions into 384-d embeddings.

    Parameters
    ----------
    texts : list[str]
        e.g. ["Paneer Tikka Pizza", "Garlic Bread", "Coke 300ml"]
    batch_size : int
        Encoding batch size.

    Returns
    -------
    np.ndarray of shape (len(texts), 384)
    """
    model = _get_model()
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # unit vectors for cosine sim
    )


def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """Build a FAISS Inner-Product index (cosine sim on unit vecs)."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    logger.info("Built FAISS index: %d vectors, dim=%d", index.ntotal, dim)
    return index


def save_index(index: "faiss.IndexFlatIP", name: str = "addon_index") -> Path:
    """Persist FAISS index to disk."""
    import faiss

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = EMBEDDINGS_DIR / f"{name}.faiss"
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index → %s", path)
    return path


def load_index(name: str = "addon_index") -> "faiss.IndexFlatIP":
    """Load a persisted FAISS index."""
    import faiss

    path = EMBEDDINGS_DIR / f"{name}.faiss"
    return faiss.read_index(str(path))


def cart_embedding(item_embeddings: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Compute a single 384-d cart embedding from item embeddings.

    Uses weighted mean-pooling (default: uniform weights).
    """
    if weights is None:
        weights = np.ones(len(item_embeddings), dtype=np.float32)
    weights = weights / weights.sum()
    emb = (item_embeddings * weights[:, None]).sum(axis=0)
    # re-normalise to unit vector
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb
