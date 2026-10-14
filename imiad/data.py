"""Real-IAD single-camera and metadata-free folder datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class Record:
    image: Path
    mask: Path | None
    label: int


def _existing_image(root: Path, category: str, rel: Path, resolution: str) -> Path:
    candidates = [
        root / f"realiad_{resolution}" / category / rel,
        root / category / rel,
        root / category / category / rel,
        root / rel,
    ]
    return next((p for p in candidates if p.is_file()), candidates[0])


def _metadata_path(root: Path, category: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    candidates = [
        root / "realiad_jsons" / "realiad_jsons_sv" / f"{category}.json",
        root / "realiad_jsons_sv" / f"{category}.json",
        root / "realiad_jsons" / f"{category}.json",
    ]
    return next((p for p in candidates if p.is_file()), candidates[0])


def real_iad_records(data: dict) -> tuple[list[Record], list[Record]]:
    root = Path(data["root"])
    category = data["category"]
    metadata_path = _metadata_path(root, category, data.get("metadata"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    normal_name = metadata.get("meta", {}).get("normal_class", "OK")
    camera = str(data["camera"]).upper().replace("CAMERA", "C")
    if not re.fullmatch(r"C[1-9]", camera):
        raise ValueError("data.camera must be C1, C2, ... or Camera1, Camera2, ...")
    pattern = re.compile(rf"(?:^|_){re.escape(camera)}(?:_|\.)", re.IGNORECASE)
    resolution = str(data.get("resolution", "raw"))
    sets: list[list[Record]] = [[], []]
    missing: list[str] = []
    for split_idx, split in enumerate(("train", "test")):
        for item in metadata[split]:
            rel = Path(item["image_path"])
            if not pattern.search(rel.name):
                continue
            image = _existing_image(root, category, rel, resolution)
            label = int(item["anomaly_class"] != normal_name)
            mask_rel = item.get("mask_path")
            mask = _existing_image(root, category, Path(mask_rel), resolution) if mask_rel else None
            if not image.is_file() or (mask is not None and not mask.is_file()):
                missing.append(str(image if not image.is_file() else mask))
                continue
            if split_idx == 0 and label:
                raise ValueError("Real-IAD train split contains an anomalous label")
            if split_idx == 1 and label and mask is None:
                raise ValueError(f"Anomalous test image has no mask: {image}")
            sets[split_idx].append(Record(image, mask, label))
    if missing and data["strict_files"]:
        raise FileNotFoundError(f"{len(missing)} Real-IAD files missing; first: {missing[0]}")
    if not sets[0] or not sets[1]:
        raise ValueError("No train/test images matched the selected Real-IAD camera and category")
    return sets[0], sets[1]


def _images(directory: Path, recursive: bool) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in EXTENSIONS)


def folder_records(data: dict) -> tuple[list[Record], list[Record]]:
    recursive = bool(data.get("recursive", True))
    train_dir, normal_dir, anomaly_dir = (Path(data[key]) for key in (
        "train_normal_dir", "test_normal_dir", "test_anomaly_dir"))
    train = [Record(p, None, 0) for p in _images(train_dir, recursive)]
    test = [Record(p, None, 0) for p in _images(normal_dir, recursive)]
    mask_root = Path(data["mask_dir"]) if data.get("mask_dir") else None
    suffix = data.get("mask_suffix", "")
    extension = data.get("mask_extension", ".png")
    if not str(extension).startswith("."):
        extension = "." + str(extension)
    missing = []
    for p in _images(anomaly_dir, recursive):
        rel = p.relative_to(anomaly_dir).with_name(p.stem + suffix + extension)
        candidates = [mask_root / rel, mask_root / rel.name] if mask_root else []
        mask = next((m for m in candidates if m.is_file()), None)
        if mask is None:
            missing.append(str(p))
        test.append(Record(p, mask, 1))
    if not train or not test:
        raise ValueError("Folder dataset has an empty train or test split")
    if missing:
        raise FileNotFoundError(
            f"Missing masks for {len(missing)} anomalous test images; first: {missing[0]}. "
            "Set data.mask_dir/mask_suffix/mask_extension in YAML."
        )
    return train, sorted(test, key=lambda x: str(x.image))


def load_records(data: dict) -> tuple[list[Record], list[Record]]:
    if data["format"] == "real_iad":
        train, test = real_iad_records(data)
    else:
        train, test = folder_records(data)
    if data.get("max_train") is not None:
        train = train[:int(data["max_train"])]
    if data.get("max_test") is not None:
        # Stratified truncation keeps both classes in small smoke runs.
        cap = int(data["max_test"])
        normal = [r for r in test if r.label == 0][: max(1, cap // 2)]
        anomaly = [r for r in test if r.label == 1][: max(1, cap - len(normal))]
        test = normal + anomaly
    if not train or not test:
        raise ValueError("Dataset cap left an empty train or test split")
    return train, test


class ImageDataset(Dataset):
    def __init__(self, records: list[Record], size: tuple[int, int] | None) -> None:
        self.records = records
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record.image) as raw:
            image = raw.convert("RGB")
            if self.size is not None:
                image = image.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
            image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        if record.mask is None:
            mask_tensor = torch.zeros(image_tensor.shape[-2:], dtype=torch.uint8)
        else:
            with Image.open(record.mask) as raw:
                mask = raw.convert("L")
                if self.size is not None:
                    mask = mask.resize((self.size[1], self.size[0]), Image.Resampling.NEAREST)
                mask_tensor = torch.from_numpy((np.asarray(mask) > 0).astype(np.uint8))
            if mask_tensor.shape != image_tensor.shape[-2:]:
                raise ValueError(f"Image and mask dimensions differ: {record.image} / {record.mask}")
        return {"image": image_tensor, "mask": mask_tensor,
                "label": torch.tensor(record.label, dtype=torch.uint8),
                "path": str(record.image)}
