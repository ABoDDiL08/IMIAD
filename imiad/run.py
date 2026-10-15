"""Run a complete experiment from a YAML file."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from importlib.metadata import version
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .config import load_config
from .data import ImageDataset, load_records
from .metrics import evaluate
from .models import build_runner


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch cannot access a GPU")
    return torch.device(name)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _yaml_safe(value):
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    return value


def run(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg["experiment"]["output_dir"]) / (
        f"{cfg['experiment']['name']}_{timestamp}_{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg["_run_dir"] = str(run_dir)
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(_yaml_safe({k: v for k, v in cfg.items() if not k.startswith("_")}),
                       sort_keys=False, allow_unicode=True), encoding="utf-8")
    report = {
        "status": "running", "phase": "initializing", "run_id": run_dir.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg["_config_path"],
        "model": cfg["model"]["name"],
        "versions": {"torch": torch.__version__, "timm": version("timm"),
                     "FrEIA": version("FrEIA")},
        "metric_details": {"P-AUROC": {
            "method": "exact" if cfg["evaluation"]["pixel_auroc_bins"] is None else "histogram",
            "bins": cfg["evaluation"]["pixel_auroc_bins"],
        }},
        "metrics": {"I-AUROC": None, "P-AUROC": None, "AUPRO": None,
                    "train_time_seconds": None, "inference_fps": None},
    }
    report_path = run_dir / "metrics.json"

    def save_report() -> None:
        temporary_path = report_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                             allow_nan=False), encoding="utf-8")
        temporary_path.replace(report_path)

    def progress(phase: str, **details: object) -> None:
        report["phase"] = phase
        report["progress"] = details
        message = f"[{datetime.now(timezone.utc).isoformat()}] {phase} {details}"
        print(message, flush=True)
        with (run_dir / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        save_report()

    save_report()
    try:
        _seed(int(cfg["experiment"]["seed"]))
        device = _device(cfg["experiment"]["device"])
        report["device"] = str(device)
        progress("loading_data")
        train_records, test_records = load_records(cfg["data"], seed=int(cfg["experiment"]["seed"]))
        report["dataset"] = {"train_images": len(train_records),
                             "test_images": len(test_records),
                             "test_anomalies": sum(x.label for x in test_records),
                             "test_class_counts": dict(Counter(x.defect_type for x in test_records))}
        train_set = ImageDataset(train_records, cfg["data"]["image_size"])
        test_set = ImageDataset(test_records, cfg["data"]["image_size"])
        train_loader = DataLoader(train_set, batch_size=cfg["data"]["batch_size"],
                                  shuffle=True, num_workers=cfg["data"]["num_workers"])
        # One test image per forward makes CFLOW likelihood normalization stable.
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                                 num_workers=cfg["data"]["num_workers"])
        progress("building_model", train_images=len(train_records), test_images=len(test_records))
        model = build_runner(cfg, device)
        cpu_tiled_patchcore = cfg["model"]["name"] == "patchcore" and cfg["input_tiling"]["enabled"]
        cpu_input_tiling = cfg["input_tiling"]["enabled"]
        _sync(device)
        start = time.perf_counter()
        progress("train_start", train_images=len(train_records))
        history = (model.fit(train_loader, progress=progress)
                   if cfg["model"]["name"] == "patchcore" else model.fit(train_loader))
        _sync(device)
        report["metrics"]["train_time_seconds"] = time.perf_counter() - start
        progress("train_complete", train_time_seconds=report["metrics"]["train_time_seconds"])
        if history is not None:
            report["train_loss"] = history
        if cfg["evaluation"]["save_checkpoint"]:
            model.save(run_dir / "model.pt")

        warmup = int(cfg["evaluation"]["fps_warmup"])
        if warmup:
            progress("test_warmup", warmup_runs=warmup)
            first = next(iter(test_loader))["image"]
            if not cpu_input_tiling:
                first = first.to(device)
            for _ in range(warmup):
                model.predict(first)
            _sync(device)

        image_labels: list[int] = []
        image_scores: list[float] = []
        maps: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        inference_seconds = 0.0
        rows: list[dict] = []
        map_dir = run_dir / "maps"
        if cfg["evaluation"]["save_maps"]:
            map_dir.mkdir()
        progress("test_inference", images_done=0, images_total=len(test_records))
        for index, batch in enumerate(test_loader):
            image = batch["image"] if cpu_input_tiling else batch["image"].to(device)
            _sync(device)
            tick = time.perf_counter()
            scores, anomaly_maps = model.predict(image)
            _sync(device)
            inference_seconds += time.perf_counter() - tick
            score = float(scores[0].detach().cpu())
            anomaly_map = anomaly_maps[0].detach().float().cpu().numpy()
            mask = batch["mask"][0].numpy()
            label = int(batch["label"][0])
            if not np.isfinite(anomaly_map).all() or not np.isfinite(score):
                raise ValueError(f"Non-finite prediction for {batch['path'][0]}")
            image_scores.append(score)
            image_labels.append(label)
            maps.append(anomaly_map)
            masks.append(mask)
            rows.append({"path": batch["path"][0], "label": label, "score": score})
            if cfg["evaluation"]["save_maps"]:
                np.save(map_dir / f"{index:06d}.npy", anomaly_map)
            progress("test_inference", images_done=index + 1, images_total=len(test_records))
        report["metrics"].update(evaluate(
            image_labels, image_scores, maps, masks,
            cfg["evaluation"]["aupro_fpr_limit"],
            cfg["evaluation"]["aupro_thresholds"],
            pixel_auroc_bins=cfg["evaluation"]["pixel_auroc_bins"],
            progress=progress,
        ))
        report["metrics"]["inference_fps"] = len(test_records) / inference_seconds
        report["fps_scope"] = (
            "one CPU image at a time; tile transfers, model forward, CPU nearest-neighbor search "
            "and map joining included; data loading and metric calculation excluded"
            if cpu_tiled_patchcore else
            "one CPU image at a time; tile transfers, shared CFLOW forward and CPU map joining "
            "included; data loading and metric calculation excluded"
            if cpu_input_tiling else
            "model forward only; one preloaded image at a time; device synchronized"
        )
        progress("writing_outputs")
        with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label", "score"])
            writer.writeheader()
            writer.writerows(rows)
        report["status"] = "completed"
        progress("completed")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        message = f"Run failed during {report['phase']}: {report['error']}"
        print(message, flush=True)
        with (run_dir / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_report()
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IMIAD experiment from YAML")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
