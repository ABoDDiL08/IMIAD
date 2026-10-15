"""Tests for bounded-memory pixel AUROC."""

import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from imiad.metrics import pixel_auroc_histogram


class HistogramPixelAurocTest(unittest.TestCase):
    def test_perfect_separation_and_ties(self) -> None:
        mask = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        separated = np.array([[0.0, 0.1], [0.9, 1.0]], dtype=np.float32)
        tied = np.full((2, 2), 0.5, dtype=np.float32)
        self.assertEqual(pixel_auroc_histogram([separated], [mask], 64), 1.0)
        self.assertEqual(pixel_auroc_histogram([tied], [mask], 64), 0.5)

    def test_approximation_close_to_exact(self) -> None:
        rng = np.random.default_rng(7)
        mask = rng.integers(0, 2, (200, 200), dtype=np.uint8)
        scores = (rng.random((200, 200)) + mask * 0.1).astype(np.float32)
        exact = roc_auc_score(mask.ravel(), scores.ravel())
        approximate = pixel_auroc_histogram([scores], [mask], 4096)
        self.assertLess(abs(approximate - exact), 0.001)

    def test_single_class_is_undefined(self) -> None:
        scores = np.arange(16, dtype=np.float32).reshape(4, 4)
        self.assertIsNone(pixel_auroc_histogram([scores],
                                                [np.zeros((4, 4), dtype=np.uint8)], 16))


if __name__ == "__main__":
    unittest.main()
