"""
test_metrics.py
===============
Unit tests for src.evaluation.metrics
"""

import numpy as np
import pytest

from src.evaluation.metrics import ndcg_at_k, mrr, hit_at_k, measure_latency


class TestNDCGAtK:
    def test_perfect_ranking(self):
        y_true = np.array([1, 0, 0, 0, 0])
        y_score = np.array([5, 4, 3, 2, 1])
        groups = np.array([5])
        assert ndcg_at_k(y_true, y_score, groups, k=5) == pytest.approx(1.0)

    def test_empty_groups(self):
        y_true = np.array([])
        y_score = np.array([])
        groups = np.array([])
        assert ndcg_at_k(y_true, y_score, groups, k=5) == 0.0


class TestMRR:
    def test_first_position(self):
        y_true = np.array([1, 0, 0])
        y_score = np.array([3, 2, 1])
        groups = np.array([3])
        assert mrr(y_true, y_score, groups) == pytest.approx(1.0)

    def test_second_position(self):
        y_true = np.array([0, 1, 0])
        y_score = np.array([1, 3, 2])
        groups = np.array([3])
        # After sort by score desc: [1,0,0] → first relevant at pos 0 → 1/1
        # Actually score order: idx1(3) > idx2(2) > idx0(1), labels: [1,0,0] → pos 0 → 1.0
        assert mrr(y_true, y_score, groups) == pytest.approx(1.0)


class TestHitAtK:
    def test_hit(self):
        y_true = np.array([0, 0, 1, 0, 0])
        y_score = np.array([1, 2, 5, 3, 0])
        groups = np.array([5])
        assert hit_at_k(y_true, y_score, groups, k=3) == 1.0

    def test_miss(self):
        y_true = np.array([0, 0, 0, 0, 1])
        y_score = np.array([5, 4, 3, 2, 1])
        groups = np.array([5])
        assert hit_at_k(y_true, y_score, groups, k=3) == 0.0


class TestLatency:
    def test_returns_stats(self):
        stats = measure_latency(lambda: sum(range(100)), n_runs=10)
        assert "p95_ms" in stats
        assert stats["p95_ms"] >= 0
