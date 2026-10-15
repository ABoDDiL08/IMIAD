"""Small end-to-end checks using generated folder data and random backbones."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from imiad.config import load_config
from imiad.coreset_disk import select_coreset
from imiad.data import ImageDataset, Record, load_records, proportional_test_sample
from imiad.patchcore import kcenter_coreset
from imiad.run import run
from imiad.tiling import GridTiler


class PipelineTest(unittest.TestCase):
    def test_proportional_test_sampling_by_defect_type(self) -> None:
        records = (
            [Record(Path(f"ok_{i}"), None, 0, "OK") for i in range(300)]
            + [Record(Path(f"defect1_{i}"), None, 1, "defect1") for i in range(250)]
            + [Record(Path(f"defect2_{i}"), None, 1, "defect2") for i in range(100)]
        )
        selected = proportional_test_sample(records, 100, seed=42)
        self.assertEqual(len(selected), 100)
        self.assertEqual({name: sum(r.defect_type == name for r in selected)
                          for name in ("OK", "defect1", "defect2")},
                         {"OK": 46, "defect1": 39, "defect2": 15})
        self.assertEqual(selected, proportional_test_sample(records, 100, seed=42))
        self.assertNotEqual(selected, proportional_test_sample(records, 100, seed=43))
        self.assertIs(proportional_test_sample(records, 650, seed=42), records)

    def test_real_iad_preserves_defect_types_for_sampling(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            image_dir = root / "realiad_raw" / "item"
            image_dir.mkdir(parents=True)
            train = [{"image_path": "train_C1_0.png", "anomaly_class": "OK"}]
            test = []
            for name, count in (("OK", 3), ("scratch", 2), ("dent", 1)):
                for index in range(count):
                    image_name = f"{name}_C1_{index}.png"
                    (image_dir / image_name).touch()
                    item = {"image_path": image_name, "anomaly_class": name}
                    if name != "OK":
                        mask_name = f"{name}_C1_{index}_mask.png"
                        (image_dir / mask_name).touch()
                        item["mask_path"] = mask_name
                    test.append(item)
            (image_dir / "train_C1_0.png").touch()
            metadata = root / "item.json"
            metadata.write_text(json.dumps({"meta": {"normal_class": "OK"},
                                            "train": train, "test": test}), encoding="utf-8")
            _, selected = load_records({"format": "real_iad", "root": str(root),
                                        "metadata": str(metadata), "category": "item",
                                        "camera": "C1", "resolution": "raw",
                                        "strict_files": True, "max_test": 4})
            self.assertEqual(len(selected), 4)
            self.assertEqual({name: sum(r.defect_type == name for r in selected)
                              for name in ("OK", "scratch", "dent")},
                             {"OK": 2, "scratch": 1, "dent": 1})

    def _make_dataset(self, root: Path, variable: bool = False) -> dict:
        folders = {name: root / name for name in ("train", "normal", "anomaly", "masks")}
        for folder in folders.values():
            folder.mkdir()
        rng = np.random.default_rng(12)
        for i in range(2):
            height, width = (64 + 8 * i, 64 + 16 * i) if variable else (64, 64)
            Image.fromarray(rng.integers(90, 150, (height, width, 3), dtype=np.uint8)).save(
                folders["train"] / f"train{i}.png")
        normal_size = (64, 68) if variable else (64, 64)
        Image.fromarray(rng.integers(90, 150, (*normal_size, 3), dtype=np.uint8)).save(
            folders["normal"] / "normal.png")
        anomaly_size = (70, 74) if variable else (64, 64)
        anomaly = rng.integers(90, 150, (*anomaly_size, 3), dtype=np.uint8)
        anomaly[20:35, 20:35] = [255, 0, 0]
        mask = np.zeros(anomaly_size, dtype=np.uint8)
        mask[20:35, 20:35] = 255
        Image.fromarray(anomaly).save(folders["anomaly"] / "anomaly.png")
        Image.fromarray(mask).save(folders["masks"] / "anomaly_mask.png")
        return {
            "format": "folders", "train_normal_dir": str(folders["train"]),
            "test_normal_dir": str(folders["normal"]),
            "test_anomaly_dir": str(folders["anomaly"]),
            "mask_dir": str(folders["masks"]), "mask_suffix": "_mask",
            "image_size": None if variable else [64, 64], "batch_size": 1, "num_workers": 0,
        }

    def _check_run(self, model: str, tile_resize: tuple[int, int] = (32, 32),
                   storage: str = "disk", neighbors: int = 1,
                   shared_cflow: bool = False) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            cfg = {
                "experiment": {"name": "smoke", "device": "cpu", "output_dir": str(root / "runs")},
                "data": self._make_dataset(root, variable=model == "patchcore" or shared_cflow),
                "model": {"name": model, "backbone": "resnet18",
                          "layers": ["layer2"], "pretrained": False},
                "input_tiling": {"enabled": model == "patchcore" or shared_cflow, "tile_size": [32, 32],
                                 "stride": [12, 12], "tile_resize": list(tile_resize)},
                "tiled_ensemble": {"enabled": model == "cflow" and not shared_cflow, "tile_size": [32, 32],
                                   "stride": [32, 32]},
                "patchcore": {"coreset_ratio": 0.1, "num_neighbors": neighbors,
                              "embedding_storage": storage, "tile_batch_size": 3,
                              "distance_bank_chunk_size": 7},
                "cflow": {"condition_vector": 16, "coupling_blocks": 1,
                          "fiber_batch_size": 8, "epochs": 1, "tile_batch_size": 3},
                "evaluation": {"fps_warmup": 0, "aupro_thresholds": 25,
                               "pixel_auroc_bins": 128},
            }
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            if model == "patchcore":
                records, _ = load_records(load_config(config_file)["data"])
                self.assertEqual(ImageDataset(records, None)[1]["image"].shape[-2:], (72, 80))
            result = Path(run(config_file))
            report = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["phase"], "completed")
            progress_log = (result.parent / "progress.log").read_text(encoding="utf-8")
            if model == "patchcore":
                self.assertIn("train_coreset_selection", progress_log)
            self.assertIn("test_inference", progress_log)
            self.assertIn("test_metrics_pixel_histogram", progress_log)
            self.assertEqual(report["metric_details"]["P-AUROC"],
                             {"method": "histogram", "bins": 128})
            self.assertEqual(report["dataset"]["test_images"], 2)
            for name in ("I-AUROC", "P-AUROC", "AUPRO", "train_time_seconds", "inference_fps"):
                self.assertIsNotNone(report["metrics"][name], name)
            self.assertGreater(report["metrics"]["inference_fps"], 0)
            self.assertTrue((result.parent / "predictions.csv").exists())
            self.assertFalse(list(result.parent.glob("imiad_patchcore_*")))
            self.assertIsInstance(yaml.safe_load((result.parent / "config_resolved.yaml").read_text()), dict)

    def test_patchcore_input_tiling(self) -> None:
        self._check_run("patchcore", tile_resize=(48, 48))

    def test_patchcore_input_tiling_without_resize(self) -> None:
        self._check_run("patchcore", tile_resize=(32, 32))

    def test_patchcore_memory_storage_and_neighbor_weighting(self) -> None:
        self._check_run("patchcore", storage="memory", neighbors=3)

    def test_cflow_tiled_ensemble(self) -> None:
        self._check_run("cflow")

    def test_cflow_shared_input_tiling(self) -> None:
        self._check_run("cflow", tile_resize=(48, 40), shared_cflow=True)

    def test_tiler_coverage(self) -> None:
        image = torch.rand(2, 3, 70, 71)
        tiler = GridTiler((32, 32), (16, 16))
        self.assertTrue(torch.allclose(tiler.untile(tiler.tile(image)), image))
        tiles = tiler.tile(image)
        pooled = torch.nn.functional.avg_pool2d(tiles, 8)
        self.assertEqual(tiler.untile(pooled).shape, (2, 3, 9, 9))
        border_tiler = GridTiler((32, 32), (16, 16), remove_border_count=2)
        self.assertTrue(torch.allclose(border_tiler.untile(border_tiler.tile(image)), image))

    def test_tiler_fractional_feature_stride(self) -> None:
        image = torch.ones(2, 3, 70, 71)
        for border in (0, 1):
            tiler = GridTiler((32, 32), (12, 12), remove_border_count=border)
            tiles = tiler.tile(image)
            features = torch.ones(len(tiles), 3, 5, 7)
            merged = tiler.untile(features)
            self.assertEqual(merged.shape, (2, 3, 11, 16))
            self.assertTrue(torch.allclose(merged, torch.ones_like(merged)))

    def test_streamed_tiles_and_map_join_match_full_grid(self) -> None:
        image = torch.rand(1, 3, 70, 71)
        tiler = GridTiler((32, 32), (12, 12))
        full_tiles = tiler.tile(image)
        full_maps = full_tiles.mean(dim=1, keepdim=True)
        expected = tiler.untile(full_maps)
        streamed = []
        accumulator = None
        for indices, tiles in tiler.iter_tiles(image, batch_size=3):
            self.assertLessEqual(len(tiles), 3)
            self.assertTrue(torch.allclose(tiles, full_tiles[indices]))
            if accumulator is None:
                accumulator = tiler.map_accumulator()
            accumulator.add(indices, tiles.mean(dim=1, keepdim=True))
            streamed.extend(indices)
        self.assertEqual(streamed, list(range(len(full_tiles))))
        self.assertTrue(torch.allclose(accumulator.result(), expected))

    def test_disk_coreset_matches_memory_on_small_data(self) -> None:
        embeddings = torch.rand(41, 12)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "embeddings.bin"
            embeddings.numpy().tofile(path)
            torch.manual_seed(19)
            expected = kcenter_coreset(embeddings, ratio=0.2)
            torch.manual_seed(19)
            actual = select_coreset(path, 41, 12, ratio=0.2, eps=0.9, chunk_size=7)
            self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
