"""Small end-to-end checks using generated folder data and random backbones."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from imiad.run import run
from imiad.tiling import GridTiler


class PipelineTest(unittest.TestCase):
    def _make_dataset(self, root: Path) -> dict:
        folders = {name: root / name for name in ("train", "normal", "anomaly", "masks")}
        for folder in folders.values():
            folder.mkdir()
        rng = np.random.default_rng(12)
        for i in range(2):
            Image.fromarray(rng.integers(90, 150, (64, 64, 3), dtype=np.uint8)).save(
                folders["train"] / f"train{i}.png")
        Image.fromarray(rng.integers(90, 150, (64, 64, 3), dtype=np.uint8)).save(
            folders["normal"] / "normal.png")
        anomaly = rng.integers(90, 150, (64, 64, 3), dtype=np.uint8)
        anomaly[20:35, 20:35] = [255, 0, 0]
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:35, 20:35] = 255
        Image.fromarray(anomaly).save(folders["anomaly"] / "anomaly.png")
        Image.fromarray(mask).save(folders["masks"] / "anomaly_mask.png")
        return {
            "format": "folders", "train_normal_dir": str(folders["train"]),
            "test_normal_dir": str(folders["normal"]),
            "test_anomaly_dir": str(folders["anomaly"]),
            "mask_dir": str(folders["masks"]), "mask_suffix": "_mask",
            "image_size": [64, 64], "batch_size": 1, "num_workers": 0,
        }

    def _check_run(self, model: str) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            cfg = {
                "experiment": {"name": "smoke", "device": "cpu", "output_dir": str(root / "runs")},
                "data": self._make_dataset(root),
                "model": {"name": model, "backbone": "resnet18",
                          "layers": ["layer2"], "pretrained": False},
                "input_tiling": {"enabled": model == "patchcore", "tile_size": [32, 32],
                                 "stride": [32, 32]},
                "tiled_ensemble": {"enabled": model == "cflow", "tile_size": [32, 32],
                                   "stride": [32, 32]},
                "patchcore": {"coreset_ratio": 0.1, "num_neighbors": 1},
                "cflow": {"condition_vector": 16, "coupling_blocks": 1,
                          "fiber_batch_size": 8, "epochs": 1},
                "evaluation": {"fps_warmup": 0, "aupro_thresholds": 25},
            }
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            result = Path(run(config_file))
            report = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["dataset"]["test_images"], 2)
            for name in ("I-AUROC", "P-AUROC", "AUPRO", "train_time_seconds", "inference_fps"):
                self.assertIsNotNone(report["metrics"][name], name)
            self.assertGreater(report["metrics"]["inference_fps"], 0)
            self.assertTrue((result.parent / "predictions.csv").exists())
            self.assertIsInstance(yaml.safe_load((result.parent / "config_resolved.yaml").read_text()), dict)

    def test_patchcore_input_tiling(self) -> None:
        self._check_run("patchcore")

    def test_cflow_tiled_ensemble(self) -> None:
        self._check_run("cflow")

    def test_tiler_coverage(self) -> None:
        import torch

        image = torch.rand(2, 3, 70, 71)
        tiler = GridTiler((32, 32), (16, 16))
        self.assertTrue(torch.allclose(tiler.untile(tiler.tile(image)), image))
        tiles = tiler.tile(image)
        pooled = torch.nn.functional.avg_pool2d(tiles, 8)
        self.assertEqual(tiler.untile(pooled).shape, (2, 3, 9, 9))
        border_tiler = GridTiler((32, 32), (16, 16), remove_border_count=2)
        self.assertTrue(torch.allclose(border_tiler.untile(border_tiler.tile(image)), image))


if __name__ == "__main__":
    unittest.main()
