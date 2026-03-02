"""
session_graph.py
================
Tier 1: Temporal Session Graph — the core innovation.

Builds an in-memory graph of cart items using NetworkX, where:
  * Nodes = current cart items (with attributes: price, category, cuisine, spice)
  * Edges = temporal co-occurrence with PMI-weighted edge strengths
  * Edge weight = PMI(item_i, item_j, meal_context) × recency_decay

Provides graph-level metrics (centrality, clustering) and candidate scoring
based on edge weights to potential add-on items.

Performance: <10ms for typical carts (1–10 items).
"""

from __future__ import annotations

import json
import math
import time
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PMI_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "pmi_matrix.json"

# ── Default co-occurrence data (fallback when no PMI matrix exists) ───────
# These represent common Indian food pairings with approximate PMI scores.
DEFAULT_CO_OCCURRENCES: dict[str, dict[str, float]] = {
    "biryani":      {"raita": 0.78, "salan": 0.72, "gulab_jamun": 0.45, "lassi": 0.41,
                     "kebab": 0.35, "papad": 0.30, "onion_rings": 0.25},
    "butter_chicken": {"naan": 0.82, "jeera_rice": 0.65, "raita": 0.40, "dal_makhani": 0.55,
                       "tandoori_roti": 0.60, "lassi": 0.35},
    "dal_makhani":  {"naan": 0.75, "jeera_rice": 0.60, "raita": 0.35, "papad": 0.30},
    "paneer_tikka": {"naan": 0.70, "mint_chutney": 0.55, "onion_rings": 0.30, "lassi": 0.35},
    "dosa":         {"sambhar": 0.80, "coconut_chutney": 0.75, "vada": 0.55,
                     "filter_coffee": 0.50, "idli": 0.30},
    "idli":         {"sambhar": 0.78, "coconut_chutney": 0.72, "vada": 0.50,
                     "filter_coffee": 0.48},
    "pizza":        {"garlic_bread": 0.72, "coke": 0.55, "fries": 0.50,
                     "pasta": 0.35, "brownie": 0.30},
    "burger":       {"fries": 0.80, "coke": 0.60, "shake": 0.45, "nuggets": 0.40},
    "noodles":      {"spring_roll": 0.65, "manchurian": 0.55, "fried_rice": 0.50,
                     "sweet_corn_soup": 0.45, "momos": 0.40},
    "fried_rice":   {"manchurian": 0.60, "spring_roll": 0.50, "sweet_corn_soup": 0.45,
                     "chilli_chicken": 0.55},
    "thali":        {"papad": 0.50, "lassi": 0.35, "gulab_jamun": 0.40, "raita": 0.30},
    "chole":        {"bhature": 0.85, "lassi": 0.45, "onion_rings": 0.25},
    "pav_bhaji":    {"lassi": 0.50, "masala_papad": 0.35, "gulab_jamun": 0.30},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PMI MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

class PMIMatrix:
    """
    Pointwise Mutual Information matrix for item co-occurrence.

    Loaded from disk (produced by batch pipeline) or initialised with
    sensible defaults for common Indian food pairings.
    """

    def __init__(self, data: dict[str, dict[str, float]] | None = None) -> None:
        self._matrix = data or {}

    @classmethod
    def load(cls, path: Path | None = None) -> "PMIMatrix":
        """Load PMI matrix from JSON file, falling back to defaults."""
        fpath = path or PMI_MATRIX_PATH
        if fpath.exists():
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Loaded PMI matrix from %s (%d items)", fpath, len(data))
                return cls(data)
            except Exception as e:
                logger.warning("Failed to load PMI matrix: %s — using defaults", e)
        logger.info("Using default co-occurrence PMI matrix (%d items)",
                     len(DEFAULT_CO_OCCURRENCES))
        return cls(dict(DEFAULT_CO_OCCURRENCES))

    def save(self, path: Path | None = None) -> None:
        """Persist PMI matrix to JSON."""
        fpath = path or PMI_MATRIX_PATH
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(self._matrix, f, indent=2)
        logger.info("Saved PMI matrix → %s", fpath)

    def get_pmi(self, item_a: str, item_b: str) -> float:
        """
        Return PMI score between two items (symmetric lookup).
        Returns 0.0 if the pair is not found.
        """
        a, b = self._normalise(item_a), self._normalise(item_b)
        # Try both directions
        score = self._matrix.get(a, {}).get(b, 0.0)
        if score == 0.0:
            score = self._matrix.get(b, {}).get(a, 0.0)
        return score

    def get_candidates(self, item: str) -> dict[str, float]:
        """Return all known co-occurring items and their PMI scores."""
        key = self._normalise(item)
        direct = dict(self._matrix.get(key, {}))

        # Also find reverse lookups
        for other_item, pairs in self._matrix.items():
            if key in pairs and key != other_item:
                if other_item not in direct:
                    direct[other_item] = pairs[key]

        return direct

    def update(self, item_a: str, item_b: str, pmi_score: float) -> None:
        """Update a single PMI entry."""
        a, b = self._normalise(item_a), self._normalise(item_b)
        if a not in self._matrix:
            self._matrix[a] = {}
        self._matrix[a][b] = pmi_score

    @property
    def items(self) -> list[str]:
        """All items in the matrix."""
        all_items = set(self._matrix.keys())
        for pairs in self._matrix.values():
            all_items.update(pairs.keys())
        return sorted(all_items)

    @staticmethod
    def _normalise(name: str) -> str:
        """Normalise item name for lookup."""
        return name.lower().strip().replace(" ", "_").replace("-", "_")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL SESSION GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class SessionGraph:
    """
    In-memory temporal session graph built on every cart modification.

    Nodes represent cart items with rich attributes.
    Edges represent co-occurrence relationships weighted by PMI × recency_decay.

    Usage::

        pmi = PMIMatrix.load()
        graph = SessionGraph(pmi_matrix=pmi, decay_constant=3600)

        # User adds Chicken Biryani
        graph.add_item({
            "name": "Chicken Biryani",
            "price": 250, "category": "main",
            "cuisine": "indian", "spice_level": "high",
        })

        # Get candidate scores for recommendations
        candidates = graph.get_candidate_scores()
        # → {"raita": 0.75, "salan": 0.69, "gulab_jamun": 0.43, ...}
    """

    def __init__(
        self,
        pmi_matrix: PMIMatrix | None = None,
        decay_constant: float = 3600.0,
    ) -> None:
        self.graph = nx.Graph()
        self.pmi = pmi_matrix or PMIMatrix.load()
        self.decay_constant = decay_constant
        self._item_add_times: dict[str, float] = {}

    def add_item(self, item: dict[str, Any]) -> None:
        """
        Add an item to the session graph.

        Creates a node with attributes and computes edges to all existing
        nodes based on PMI × recency decay.

        Parameters
        ----------
        item : dict
            Must have 'name'. Optional: price, category, cuisine, spice_level.
        """
        name = self._node_id(item)
        now = time.monotonic()

        # Add node with attributes
        self.graph.add_node(name, **{
            "price": item.get("price", 0),
            "category": item.get("category", "other"),
            "cuisine": item.get("cuisine_type", item.get("cuisine", "unknown")),
            "spice_level": item.get("spice_level", "medium"),
            "added_at": now,
        })
        self._item_add_times[name] = now

        # Compute edges to all existing nodes
        for existing_node in list(self.graph.nodes):
            if existing_node == name:
                continue

            pmi_score = self.pmi.get_pmi(name, existing_node)
            if pmi_score <= 0:
                continue

            # Recency decay: exp(-time_diff / decay_constant)
            time_diff = abs(now - self._item_add_times.get(existing_node, now))
            recency = math.exp(-time_diff / self.decay_constant)

            edge_weight = pmi_score * recency
            self.graph.add_edge(name, existing_node, weight=edge_weight,
                                pmi=pmi_score, recency=recency)

        logger.debug("Added node '%s' — graph: %d nodes, %d edges",
                      name, self.graph.number_of_nodes(), self.graph.number_of_edges())

    def remove_item(self, item: dict[str, Any] | str) -> None:
        """Remove an item node and all its edges."""
        name = self._node_id(item) if isinstance(item, dict) else self._normalise(item)
        if name in self.graph:
            self.graph.remove_node(name)
            self._item_add_times.pop(name, None)

    def get_candidate_scores(
        self,
        top_k: int = 20,
        min_score: float = 0.05,
    ) -> list[dict[str, Any]]:
        """
        Score potential add-on candidates based on graph edge weights.

        For each cart item, look up its PMI neighbours that are NOT in the
        current cart, aggregate scores, and return the top-K candidates.

        Returns
        -------
        list[dict] with keys: name, score, contributing_items
        """
        cart_items = set(self.graph.nodes)
        candidate_scores: dict[str, float] = {}
        candidate_sources: dict[str, list[str]] = {}

        for node in cart_items:
            neighbours = self.pmi.get_candidates(node)
            for candidate, pmi_score in neighbours.items():
                cand_norm = self._normalise(candidate)
                if cand_norm in cart_items:
                    continue  # Already in cart

                # Apply recency decay
                now = time.monotonic()
                node_time = self._item_add_times.get(node, now)
                recency = math.exp(-abs(now - node_time) / self.decay_constant)
                score = pmi_score * recency

                if score < min_score:
                    continue

                # Aggregate across multiple cart items
                candidate_scores[cand_norm] = candidate_scores.get(cand_norm, 0) + score
                if cand_norm not in candidate_sources:
                    candidate_sources[cand_norm] = []
                candidate_sources[cand_norm].append(node)

        # Sort and return top-K
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
        results = []
        for name, score in sorted_candidates[:top_k]:
            results.append({
                "name": name,
                "graph_score": round(score, 4),
                "contributing_items": candidate_sources.get(name, []),
            })

        return results

    # ── Graph Metrics ─────────────────────────────────────────────────────────

    def compute_graph_features(self) -> dict[str, Any]:
        """
        Compute graph-level features for the ML pipeline.

        Features:
        - node_count, edge_count
        - avg_degree, max_degree
        - avg_clustering_coefficient
        - graph_density
        - avg_edge_weight, max_edge_weight
        - avg_centrality
        """
        g = self.graph
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()

        if n_nodes == 0:
            return {
                "graph_node_count": 0, "graph_edge_count": 0,
                "graph_avg_degree": 0, "graph_max_degree": 0,
                "graph_avg_clustering": 0, "graph_density": 0,
                "graph_avg_edge_weight": 0, "graph_max_edge_weight": 0,
                "graph_avg_centrality": 0,
            }

        degrees = [d for _, d in g.degree()]
        edge_weights = [d.get("weight", 0) for _, _, d in g.edges(data=True)]

        # Clustering coefficient
        clustering = nx.average_clustering(g, weight="weight") if n_nodes > 1 else 0.0

        # Degree centrality
        centrality = nx.degree_centrality(g) if n_nodes > 1 else {n: 0 for n in g.nodes}

        return {
            "graph_node_count": n_nodes,
            "graph_edge_count": n_edges,
            "graph_avg_degree": round(sum(degrees) / max(n_nodes, 1), 2),
            "graph_max_degree": max(degrees) if degrees else 0,
            "graph_avg_clustering": round(clustering, 4),
            "graph_density": round(nx.density(g), 4),
            "graph_avg_edge_weight": round(
                sum(edge_weights) / max(len(edge_weights), 1), 4
            ),
            "graph_max_edge_weight": round(max(edge_weights), 4) if edge_weights else 0,
            "graph_avg_centrality": round(
                sum(centrality.values()) / max(len(centrality), 1), 4
            ),
        }

    def path_length_to_candidate(self, candidate_name: str) -> float:
        """
        Compute shortest weighted path length from any cart node to candidate.
        Returns infinity if unreachable.
        """
        cand = self._normalise(candidate_name)
        if cand not in self.graph:
            return float("inf")

        min_path = float("inf")
        for node in self.graph.nodes:
            if node == cand:
                continue
            try:
                length = nx.shortest_path_length(
                    self.graph, node, cand, weight="weight"
                )
                min_path = min(min_path, length)
            except nx.NetworkXNoPath:
                continue
        return min_path

    def reset(self) -> None:
        """Clear the entire graph (new session)."""
        self.graph.clear()
        self._item_add_times.clear()

    @property
    def size(self) -> int:
        return self.graph.number_of_nodes()

    @staticmethod
    def _node_id(item: dict | str) -> str:
        if isinstance(item, str):
            return SessionGraph._normalise(item)
        return SessionGraph._normalise(item.get("name", "unknown"))

    @staticmethod
    def _normalise(name: str) -> str:
        return name.lower().strip().replace(" ", "_").replace("-", "_")
