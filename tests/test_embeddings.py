"""
test_embeddings.py
==================
Unit tests for src.features.embeddings
"""

import numpy as np
import pytest


class TestCartEmbedding:
    def test_uniform_weights(self):
        from src.features.embeddings import cart_embedding

        vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        result = cart_embedding(vecs)
        assert result.shape == (3,)
        # mean of [1,0,0] and [0,1,0] ∝ [0.5, 0.5, 0]
        assert np.dot(result, result) == pytest.approx(1.0, abs=1e-5)

    def test_weighted(self):
        from src.features.embeddings import cart_embedding

        vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        weights = np.array([3.0, 1.0])
        result = cart_embedding(vecs, weights)
        # should be closer to [1,0,0] direction
        assert result[0] > result[1]
