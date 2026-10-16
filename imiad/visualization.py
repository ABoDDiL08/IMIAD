"""Save a small, reviewable set of image-level PatchCore predictions."""

from __future__ import annotations

import csv
import heapq
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


_STOPS = np.array([[20, 35, 110], [0, 175, 220], [255, 220, 40],
                   [220, 35, 35]], dtype=np.float32)
_PALETTE = np.stack([
    np.interp(np.linspace(0, 3, 256), np.arange(4), _STOPS[:, channel])
    for channel in range(3)
], axis=1).astype(np.uint8)


class CaseVisualizer:
    """Save every FP/FN and at most N highest-scoring true anomalies."""

    def __init__(self, run_dir: Path, threshold: float, max_correct: int,
                 max_side: int) -> None:
        self.directory = run_dir / "visualizations"
        self.directory.mkdir()
        self.threshold = threshold
        self.max_correct = max_correct
        self.max_side = max_side
        self.counts: Counter[str] = Counter()
        self.saved: dict[int, dict] = {}
        self.top_correct: list[tuple[float, int]] = []

    def _render(self, index: int, case: str, image_path: Path,
                score: float, anomaly_map: np.ndarray,
                mask: np.ndarray) -> dict:
        height, width = anomaly_map.shape
        scale = min(1.0, self.max_side / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        with Image.open(image_path) as source:
            original = source.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        scores = np.asarray(Image.fromarray(
            np.asarray(anomaly_map, dtype=np.float32)).resize(size, Image.Resampling.BILINEAR))
        low, high = np.percentile(scores, [1, 99])
        if high <= low:
            indices = np.zeros(scores.shape, dtype=np.uint8)
        else:
            indices = (np.clip((scores - low) * (255.0 / (high - low)), 0, 255)
                       .astype(np.uint8))
        heatmap = Image.fromarray(_PALETTE[indices])
        overlay = Image.blend(original, heatmap, 0.45)
        gt = Image.fromarray((np.asarray(mask) != 0).astype(np.uint8) * 255)
        gt = gt.resize(size, Image.Resampling.NEAREST).convert("RGB")
        panels = (original, heatmap, overlay, gt)
        sheet = Image.new("RGB", (size[0] * 4, size[1] + 48), "white")
        draw = ImageDraw.Draw(sheet)
        name = image_path.name.encode("ascii", "replace").decode("ascii")
        draw.text((8, 4), f"{case} | score={score:.5f} | threshold={self.threshold:.5f} | {name}",
                  fill="black")
        for column, (panel, title) in enumerate(zip(
                panels, ("Original", "Anomaly heatmap", "Overlay", "GT mask"))):
            left = column * size[0]
            draw.text((left + 8, 27), title, fill="black")
            sheet.paste(panel, (left, 48))
        stem = f"{index:05d}_{case.lower()}"
        case_dir = self.directory / case.lower()
        case_dir.mkdir(exist_ok=True)
        paths = {"case_image": case_dir / f"{stem}_case.png",
                 "heatmap": case_dir / f"{stem}_heatmap.png",
                 "overlay": case_dir / f"{stem}_overlay.png"}
        sheet.save(paths["case_image"])
        heatmap.save(paths["heatmap"])
        overlay.save(paths["overlay"])
        return {"source": str(image_path), "case": case, "score": score,
                "threshold": self.threshold,
                **{key: str(path.relative_to(self.directory)) for key, path in paths.items()}}

    def consider(self, index: int, image_path: Path, label: int, score: float,
                 anomaly_map: np.ndarray, mask: np.ndarray) -> None:
        predicted = int(score > self.threshold)
        case = ("TP" if label else "FP") if predicted else ("FN" if label else "TN")
        self.counts[case] += 1
        if case == "TN":
            return
        if case == "TP" and (self.max_correct == 0 or (
                len(self.top_correct) == self.max_correct
                and score <= self.top_correct[0][0])):
            return
        self.saved[index] = self._render(index, case, image_path, score, anomaly_map, mask)
        if case == "TP":
            heapq.heappush(self.top_correct, (score, index))
            if len(self.top_correct) > self.max_correct:
                _, evicted = heapq.heappop(self.top_correct)
                old = self.saved.pop(evicted)
                for key in ("case_image", "heatmap", "overlay"):
                    (self.directory / old[key]).unlink()

    def finish(self) -> dict:
        with (self.directory / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "source", "case", "score", "threshold", "case_image", "heatmap", "overlay"])
            writer.writeheader()
            writer.writerows(self.saved[index] for index in sorted(self.saved))
        return {"directory": str(self.directory), "image_threshold": self.threshold,
                "counts": {case: self.counts[case] for case in ("TP", "FP", "TN", "FN")},
                "saved": len(self.saved)}
