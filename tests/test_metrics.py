"""Tests for bounded-memory pixel AUROC."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from imiad.metrics import evaluate, image_classification_metrics, pixel_auroc_histogram


class ImageClassificationMetricsTest(unittest.TestCase):
    def test_confusion_matrix_and_rates(self) -> None:
        metrics, counts = image_classification_metrics(
            [1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
        self.assertEqual(counts, {"TP": 1, "FP": 1, "TN": 1, "FN": 1})
        self.assertEqual(metrics, {"Precision": 0.5, "Recall": 0.5,
                                   "F1": 0.5, "FPR": 0.5})

    def test_all_predicted_positive(self) -> None:
        metrics, counts = image_classification_metrics(
            [1] * 93 + [0] * 107, [1.0] * 200, 0.5)
        self.assertEqual(counts, {"TP": 93, "FP": 107, "TN": 0, "FN": 0})
        self.assertEqual(metrics["Precision"], 93 / 200)
        self.assertEqual(metrics["Recall"], 1.0)
        self.assertEqual(metrics["FPR"], 1.0)

    def test_undefined_denominators(self) -> None:
        metrics, counts = image_classification_metrics([1, 1], [0.1, 0.2], 0.5)
        self.assertEqual(counts, {"TP": 0, "FP": 0, "TN": 0, "FN": 2})
        self.assertEqual(metrics, {"Precision": None, "Recall": 0.0,
                                   "F1": 0.0, "FPR": None})

    def test_score_equal_to_threshold_is_negative(self) -> None:
        metrics, counts = image_classification_metrics([0, 1], [0.5, 0.5], 0.5)
        self.assertEqual(counts, {"TP": 0, "FP": 0, "TN": 1, "FN": 1})


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

    def test_disk_backed_inputs_match_arrays(self) -> None:
        scores = np.arange(16, dtype=np.float32).reshape(4, 4)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[2:, :] = 1
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            map_path, mask_path = Path(temp) / "map.npy", Path(temp) / "mask.npy"
            np.save(map_path, scores)
            np.save(mask_path, mask)
            self.assertEqual(pixel_auroc_histogram([map_path], [mask_path], 16),
                             pixel_auroc_histogram([scores], [mask], 16))

    def test_all_metrics_match_disk_backed_inputs(self) -> None:
        normal = np.zeros((8, 8), dtype=np.float32)
        anomaly = normal.copy()
        anomaly[2:5, 2:5] = 1.0
        normal_mask = np.zeros((8, 8), dtype=np.uint8)
        anomaly_mask = (anomaly > 0).astype(np.uint8)
        arrays = evaluate([0, 1], [0.0, 1.0],
                          [normal, anomaly], [normal_mask, anomaly_mask],
                          0.3, 25, pixel_auroc_bins=32)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            paths = [root / name for name in ("normal.npy", "anomaly.npy",
                                               "normal_mask.npy", "anomaly_mask.npy")]
            for path, array in zip(paths, (normal, anomaly, normal_mask, anomaly_mask)):
                np.save(path, array)
            disk = evaluate([0, 1], [0.0, 1.0], paths[:2], paths[2:],
                            0.3, 25, pixel_auroc_bins=32)
            self.assertEqual(arrays, disk)


if __name__ == "__main__":
    unittest.main()
