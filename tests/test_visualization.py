"""Review image selection and PNG output without a model dependency."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from imiad.visualization import CaseVisualizer


class CaseVisualizerTest(unittest.TestCase):
    def test_wrong_predictions_and_top_true_anomaly_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            image = root / "example.png"
            Image.fromarray(np.full((32, 40, 3), 120, dtype=np.uint8)).save(image)
            anomaly_map = np.linspace(0, 1, 32 * 40, dtype=np.float32).reshape(32, 40)
            mask = np.zeros((32, 40), dtype=np.uint8)
            mask[10:20, 10:20] = 1
            visualizer = CaseVisualizer(root, threshold=0.5, max_correct=1, max_side=24)
            for index, (label, score) in enumerate(((0, 0.7), (1, 0.3),
                                                     (1, 0.8), (1, 0.9), (0, 0.1))):
                visualizer.consider(index, image, label, score, anomaly_map, mask)
            result = visualizer.finish()
            self.assertEqual(result["counts"], {"TP": 2, "FP": 1, "TN": 1, "FN": 1})
            self.assertEqual(result["saved"], 3)
            with (visualizer.directory / "manifest.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["case"] for row in rows}, {"TP", "FP", "FN"})
            self.assertEqual(len(rows), 3)
            for row in rows:
                for key in ("case_image", "heatmap", "overlay"):
                    self.assertTrue((visualizer.directory / row[key]).is_file())
            self.assertFalse(list((visualizer.directory / "tp").glob("00002_*")))


if __name__ == "__main__":
    unittest.main()
