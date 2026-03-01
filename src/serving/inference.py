"""
inference.py
============
End-to-end inference pipeline for Cart Super Add-On recommendations.

Flow (per request, target < 200 ms total)
------------------------------------------
1. Encode current cart items → cart_embedding           (~10 ms)
2. FAISS top-K candidate retrieval                       (~2 ms)
3. Feature vector assembly for each candidate            (~5 ms)
4. LightGBM ranking score                                (~1 ms)
5. Return top-N ranked add-ons                           (~0 ms)
                                               TOTAL ≈ ~18 ms ✅
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AddonRecommendation:
    """A single add-on recommendation returned by the pipeline."""
    addon_id: str
    addon_name: str
    category: str
    price: float
    score: float
    rank: int


@dataclass
class InferencePipeline:
    """
    Holds pre-loaded artefacts (model, index, metadata) and
    exposes a ``recommend()`` method that runs the full pipeline.

    Designed to be instantiated **once** at server startup.
    """

    # Will be populated in Step 3
    booster: Any = None
    faiss_index: Any = None
    addon_metadata: Any = None
    embedding_model: Any = None
    feature_columns: list[str] = field(default_factory=list)

    def load(self) -> "InferencePipeline":
        """Load all artefacts from disk into memory."""
        raise NotImplementedError("Step 3: implement artefact loading")

    def recommend(
        self,
        cart_items: list[dict[str, Any]],
        top_n: int = 5,
        excluded_ids: set[str] | None = None,
    ) -> list[AddonRecommendation]:
        """
        Given current cart items, return top-N add-on recommendations.

        Parameters
        ----------
        cart_items : list[dict]
            Each dict has keys: item_id, name, category, price, qty
        top_n : int
            Number of recommendations to return.
        excluded_ids : set[str]
            Add-on IDs already in the cart (to avoid duplicates).

        Returns
        -------
        list[AddonRecommendation]
        """
        raise NotImplementedError("Step 3: implement full inference pipeline")

    def recommend_sequential(
        self,
        cart_history: list[list[dict[str, Any]]],
        top_n: int = 5,
    ) -> list[AddonRecommendation]:
        """
        Handle sequential cart updates.

        ``cart_history`` is an ordered list of cart snapshots.
        The latest snapshot is used for ranking; earlier snapshots
        contribute a momentum / recency signal.
        """
        raise NotImplementedError("Step 3: implement sequential updates")
