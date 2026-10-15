"""Global coordinates, shared training, and tile-batch-independent scoring."""

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from imiad.cflow import TiledCFlowRunner, global_positional_encoding_2d
from imiad.config import DEFAULTS, load_config
from imiad.models import build_runner
from imiad.tiling import GridTiler


class TinyEncoder(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.channels = [3, 3]

    def forward(self, images):
        return {"layer2": F.avg_pool2d(images, 4), "layer3": F.avg_pool2d(images, 8)}


class SharedCFlowTest(unittest.TestCase):
    def context(self, image, mode="padding"):
        tiler = GridTiler((32, 24), (16, 12), mode)
        tiler.tile(image)
        return {"grid": tiler.grid, "tile_size": tiler.tile_size,
                "stride": tiler.stride, "mode": mode}

    def encoding(self, context, indices, height=8, width=6):
        return global_positional_encoding_2d(16, height, width, torch.device("cpu"),
                                             indices, **context)

    def test_overlap_and_image_batch_offsets(self):
        context = self.context(torch.zeros(2, 3, 48, 36))
        encoding = self.encoding(context, [0, 1, 2, 4])
        # Horizontal and vertical overlaps represent identical global centers.
        torch.testing.assert_close(encoding[0, :, :, 3:], encoding[1, :, :, :3])
        torch.testing.assert_close(encoding[0, :, 4:, :], encoding[2, :, :4, :])
        torch.testing.assert_close(encoding[0], encoding[3])
        self.assertFalse(torch.allclose(encoding[0], encoding[1]))

    def test_resize_uses_raw_coordinates(self):
        context = self.context(torch.zeros(1, 3, 48, 36))
        encoding = self.encoding(context, [1])
        # First cell center is x=12+24/6/2-0.5, y=32/8/2-0.5.
        torch.testing.assert_close(encoding[0, 0, 0, 0], torch.tensor(13.5).sin())
        torch.testing.assert_close(encoding[0, 8, 0, 0], torch.tensor(1.5).sin())

    def test_padding_and_interpolation_coordinates(self):
        image = torch.zeros(1, 3, 39, 29)
        padded = self.encoding(self.context(image), [3])
        interpolated = self.encoding(self.context(image, "interpolation"), [3])
        torch.testing.assert_close(padded[0, 0, 0, 0], torch.tensor(13.5).sin())
        expected_x = (12 + 2) * 29 / 36 - 0.5
        expected_y = (16 + 2) * 39 / 48 - 0.5
        torch.testing.assert_close(interpolated[0, 0, 0, 0], torch.tensor(expected_x).sin())
        torch.testing.assert_close(interpolated[0, 8, 0, 0], torch.tensor(expected_y).sin())

    def cfg(self):
        cfg = deepcopy(DEFAULTS)
        cfg["model"].update(name="cflow", backbone="resnet18", layers=["layer2", "layer3"], pretrained=False)
        cfg["input_tiling"].update(tile_size=(32, 24), stride=(16, 12),
                                    tile_resize=(24, 32), remove_border_count=2)
        cfg["cflow"].update(condition_vector=16, coupling_blocks=1,
                             fiber_batch_size=13, epochs=1, tile_batch_size=3)
        return cfg

    @patch("imiad.cflow.FeatureEncoder", TinyEncoder)
    def test_shared_fit_and_prediction_batch_invariance(self):
        torch.manual_seed(7)
        cfg = self.cfg()
        runner = build_runner(cfg, torch.device("cpu"))
        self.assertIsInstance(runner, TiledCFlowRunner)
        self.assertFalse(hasattr(runner, "models"))
        before = [p.detach().clone() for p in runner.model.decoders.parameters()]
        history = runner.fit([{"image": torch.randn(2, 3, 39, 29)}])
        self.assertTrue(torch.isfinite(torch.tensor(history)).all())
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before, runner.model.decoders.parameters())))
        image = torch.randn(1, 3, 39, 29)
        reference = runner.predict(image)
        for size in (1, 2, 7):
            runner.tile_batch_size = size
            score, anomaly_map = runner.predict(image)
            self.assertEqual(anomaly_map.shape, (1, 39, 29))
            torch.testing.assert_close(score, reference[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(anomaly_map, reference[1], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(score, anomaly_map.flatten(1).amax(1))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            checkpoint = Path(temp) / "model.pt"
            runner.save(checkpoint)
            restored = build_runner(cfg, torch.device("cpu"))
            restored.model.load_state_dict(torch.load(checkpoint, weights_only=True))
            torch.testing.assert_close(restored.predict(image)[1], reference[1])

    def test_config_modes(self):
        cfg = self.cfg()
        cfg["data"].update(format="folders", train_normal_dir="train",
                            test_normal_dir="normal", test_anomaly_dir="anomaly")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            path = Path(temp) / "config.yaml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            self.assertIsNone(load_config(path)["data"]["image_size"])
            cfg["tiled_ensemble"]["enabled"] = True
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
                load_config(path)
            cfg["tiled_ensemble"]["enabled"] = False
            cfg["data"]["image_size"] = [64, 64]
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native-size"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
