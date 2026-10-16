"""Load and validate one experiment configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULTS = {
    "experiment": {"name": "imiad", "seed": 42, "device": "auto", "output_dir": "runs"},
    "data": {"format": "real_iad", "image_size": None, "batch_size": 1,
             "num_workers": 0, "max_train": None, "max_test": None,
             "strict_files": True},
    "model": {"name": "patchcore", "backbone": "wide_resnet50_2",
              "layers": ["layer2", "layer3"], "pretrained": True},
    "input_tiling": {"enabled": True, "tile_size": [128, 128],
                     "stride": [128, 128], "tile_resize": None, "mode": "padding",
                     "remove_border_count": 0},
    "tiled_ensemble": {"enabled": False, "tile_size": [128, 128],
                       "stride": [128, 128], "aggregation": "mean"},
    "patchcore": {"coreset_ratio": 0.1, "num_neighbors": 9,
                  "projection_eps": 0.9, "tile_batch_size": 8,
                  "embedding_storage": "disk", "distance_bank_chunk_size": 4096},
    "cflow": {"condition_vector": 128, "coupling_blocks": 8,
              "clamp_alpha": 1.9, "fiber_batch_size": 64,
              "learning_rate": 0.0001, "epochs": 10, "tile_batch_size": 8},
    "evaluation": {"aupro_fpr_limit": 0.3, "aupro_thresholds": 200,
                   "pixel_auroc_bins": None,
                   "visualization": {"enabled": False, "image_threshold": None,
                                     "train_quantile": 0.99, "correct_anomalies": 5,
                                     "max_side": 1600},
                   "fps_warmup": 2, "save_maps": False, "save_checkpoint": False},
}


def _merge(dst: dict, src: dict, context: str = "") -> None:
    for key, value in src.items():
        name = f"{context}.{key}" if context else key
        if key not in dst:
            # Data adapters have distinct, documented keys.
            if context != "data" or key not in {
                "root", "metadata", "category", "camera", "resolution",
                "train_normal_dir", "test_normal_dir", "test_anomaly_dir",
                "mask_dir", "mask_suffix", "mask_extension", "recursive",
            }:
                raise ValueError(f"Unknown configuration key: {name}")
            dst[key] = value
        elif isinstance(dst[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected mapping at {name}")
            _merge(dst[key], value, name)
        else:
            dst[key] = value


def _pair(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        value = [value, value]
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        not isinstance(n, int) or n < 1 for n in value
    ):
        raise ValueError(f"{name} must be two positive integers")
    return tuple(value)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("YAML root must be a mapping")
    cfg = deepcopy(DEFAULTS)
    _merge(cfg, source)
    data, model = cfg["data"], cfg["model"]
    if data["format"] not in {"real_iad", "folders"}:
        raise ValueError("data.format must be real_iad or folders")
    if model["name"] not in {"patchcore", "cflow"}:
        raise ValueError("model.name must be patchcore or cflow")
    if model["name"] == "patchcore" and cfg["tiled_ensemble"]["enabled"]:
        raise ValueError("TiledEnsemble is supported for CFlow in this project")
    if cfg["input_tiling"]["enabled"] and cfg["tiled_ensemble"]["enabled"]:
        raise ValueError("input_tiling and tiled_ensemble cannot both be enabled")
    if data["image_size"] is not None:
        data["image_size"] = _pair(data["image_size"], "data.image_size")
    if cfg["input_tiling"]["enabled"] and data["image_size"] is not None:
        raise ValueError("Input Tiling reads native-size images; set data.image_size: null")
    if model["name"] == "cflow" and not cfg["input_tiling"]["enabled"] and data["image_size"] is None:
        raise ValueError("CFlow without Input Tiling requires a fixed data.image_size")
    for group in ("input_tiling", "tiled_ensemble"):
        tile = _pair(cfg[group]["tile_size"], f"{group}.tile_size")
        stride = _pair(cfg[group]["stride"], f"{group}.stride")
        if any(s > t for s, t in zip(stride, tile)):
            raise ValueError(f"{group}.stride cannot exceed tile_size")
        cfg[group]["tile_size"], cfg[group]["stride"] = tile, stride
    resize = cfg["input_tiling"]["tile_resize"]
    cfg["input_tiling"]["tile_resize"] = (
        cfg["input_tiling"]["tile_size"] if resize is None else _pair(resize, "input_tiling.tile_resize")
    )
    if not 0 < cfg["patchcore"]["coreset_ratio"] <= 1:
        raise ValueError("patchcore.coreset_ratio must be in (0, 1]")
    if not 0 < cfg["patchcore"]["projection_eps"] < 1:
        raise ValueError("patchcore.projection_eps must be in (0, 1)")
    if cfg["patchcore"]["num_neighbors"] < 1:
        raise ValueError("patchcore.num_neighbors must be positive")
    if cfg["patchcore"]["tile_batch_size"] < 1:
        raise ValueError("patchcore.tile_batch_size must be positive")
    if cfg["patchcore"]["distance_bank_chunk_size"] < 1:
        raise ValueError("patchcore.distance_bank_chunk_size must be positive")
    if cfg["patchcore"]["embedding_storage"] not in {"memory", "disk"}:
        raise ValueError("patchcore.embedding_storage must be memory or disk")
    if cfg["cflow"]["condition_vector"] % 4:
        raise ValueError("cflow.condition_vector must be divisible by 4")
    if cfg["cflow"]["fiber_batch_size"] < 1 or cfg["cflow"]["epochs"] < 1:
        raise ValueError("cflow.fiber_batch_size and cflow.epochs must be positive")
    if type(cfg["cflow"]["tile_batch_size"]) is not int or cfg["cflow"]["tile_batch_size"] < 1:
        raise ValueError("cflow.tile_batch_size must be a positive integer")
    if data["batch_size"] < 1 or data["num_workers"] < 0:
        raise ValueError("data.batch_size must be positive and num_workers nonnegative")
    for field in ("max_train", "max_test"):
        value = data[field]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"data.{field} must be a positive integer or null")
    if cfg["evaluation"]["aupro_thresholds"] < 2:
        raise ValueError("evaluation.aupro_thresholds must be at least two")
    bins = cfg["evaluation"]["pixel_auroc_bins"]
    if bins is not None and (type(bins) is not int or bins < 2):
        raise ValueError("evaluation.pixel_auroc_bins must be null or an integer >= 2")
    visual = cfg["evaluation"]["visualization"]
    threshold = visual["image_threshold"]
    if threshold is not None and (not isinstance(threshold, (int, float))
                                  or not float("-inf") < threshold < float("inf")):
        raise ValueError("evaluation.visualization.image_threshold must be a finite number or null")
    if not 0 < visual["train_quantile"] <= 1:
        raise ValueError("evaluation.visualization.train_quantile must be in (0, 1]")
    if type(visual["correct_anomalies"]) is not int or visual["correct_anomalies"] < 0:
        raise ValueError("evaluation.visualization.correct_anomalies must be nonnegative")
    if type(visual["max_side"]) is not int or visual["max_side"] < 64:
        raise ValueError("evaluation.visualization.max_side must be at least 64")
    if cfg["evaluation"]["fps_warmup"] < 0:
        raise ValueError("evaluation.fps_warmup must be nonnegative")
    if cfg["input_tiling"]["remove_border_count"] < 0:
        raise ValueError("input_tiling.remove_border_count must be nonnegative")
    if cfg["evaluation"]["aupro_fpr_limit"] <= 0 or cfg["evaluation"]["aupro_fpr_limit"] > 1:
        raise ValueError("evaluation.aupro_fpr_limit must be in (0, 1]")
    if data["format"] == "real_iad":
        for field in ("root", "category", "camera"):
            if not data.get(field):
                raise ValueError(f"data.{field} is required for Real-IAD")
    else:
        for field in ("train_normal_dir", "test_normal_dir", "test_anomaly_dir"):
            if not data.get(field):
                raise ValueError(f"data.{field} is required for folders")
    # Relative paths are relative to the YAML file, making runs cwd-independent.
    for field in ("root", "metadata", "train_normal_dir", "test_normal_dir",
                  "test_anomaly_dir", "mask_dir"):
        if data.get(field):
            p = Path(data[field])
            data[field] = str((path.parent / p).resolve() if not p.is_absolute() else p)
    out = Path(cfg["experiment"]["output_dir"])
    cfg["experiment"]["output_dir"] = str((path.parent / out).resolve() if not out.is_absolute() else out)
    cfg["_config_path"] = str(path)
    return cfg
