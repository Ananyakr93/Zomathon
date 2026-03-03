"""
metrics.py
==========
Evaluation metrics for the CartComplete ranking system.

Primary Metrics
---------------
* NDCG@K  — the competition's core metric
* MRR     — mean reciprocal rank
* Hit@K   — fraction of sessions where a relevant add-on is in top-K
* P95 Latency — must be < 200 ms end-to-end
"""

from __future__ import annotations

import time
import logging
from typing import Callable

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score

logger = logging.getLogger(__name__)


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute NDCG@K averaged over query groups (cart sessions).
    """
    scores: list[float] = []
    idx = 0
    for g in groups:
        yt = y_true[idx : idx + g]
        ys = y_score[idx : idx + g]
        if yt.sum() > 0:  # skip sessions with no positive
            scores.append(
                ndcg_score(yt.reshape(1, -1), ys.reshape(1, -1), k=k)
            )
        idx += g
    return float(np.mean(scores)) if scores else 0.0


def mrr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Mean Reciprocal Rank across query groups."""
    rrs: list[float] = []
    idx = 0
    for g in groups:
        yt = y_true[idx : idx + g]
        ys = y_score[idx : idx + g]
        order = np.argsort(-ys)
        ranked_labels = yt[order]
        positives = np.where(ranked_labels > 0)[0]
        if len(positives) > 0:
            rrs.append(1.0 / (positives[0] + 1))
        else:
            rrs.append(0.0)
        idx += g
    return float(np.mean(rrs))


def hit_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
) -> float:
    """Fraction of sessions with at least one relevant add-on in top-K."""
    hits = 0
    total = 0
    idx = 0
    for g in groups:
        yt = y_true[idx : idx + g]
        ys = y_score[idx : idx + g]
        top_k_idx = np.argsort(-ys)[:k]
        if yt[top_k_idx].sum() > 0:
            hits += 1
        total += 1
        idx += g
    return hits / total if total > 0 else 0.0


def auc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Compute Area Under the ROC Curve.
    Probability that a randomly chosen positive item ranks higher than a random negative item.
    """
    # AUC requires both positive and negative classes
    if len(np.unique(y_true)) > 1:
        return float(roc_auc_score(y_true > 0, y_score))
    return 0.0


def measure_latency(
    fn: Callable,
    *args,
    n_runs: int = 100,
    **kwargs,
) -> dict[str, float]:
    """
    Benchmark a function's latency.

    Returns
    -------
    dict with keys: mean_ms, median_ms, p95_ms, p99_ms
    """
    timings: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

    arr = np.array(timings)
    result = {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }
    logger.info("Latency: %s", result)
    return result
