# IMIAD

One YAML-driven project for **PatchCore + Input Tiling** and **CFlow + TiledEnsemble**. The models, coreset search, tile merging, dataset loading, evaluation, and per-run JSON reporting are implemented here. Timm provides pretrained backbones and FrEIA provides invertible flow blocks; Anomalib is a reference, not a runtime dependency.

## Setup

Use a Python environment with PyTorch installed for your CUDA driver, then install:

```powershell
pip install -r requirements.txt
```

Run from the project root:

```powershell
python -m imiad.run --config configs/real_iad_patchcore.yaml
python -m imiad.run --config configs/real_iad_cflow_ensemble.yaml
```

Each YAML is independent. `experiment.output_dir` and relative dataset paths are resolved relative to the YAML file. Change `data.root`, `data.metadata`, category, and camera for your Real-IAD installation. The supplied examples point to the existing `D:/VinRobotics/audiojack/audiojack` data and single-view JSON. `camera: C1` filters filenames with `_C1_` and uses only the official `train` and `test` split in that JSON. Missing referenced files fail by default. Normal training images are taken only from the train split, even if their source path contains `NG`.

For a dataset without metadata, copy `configs/folders_patchcore.yaml` or `configs/folders_cflow_ensemble.yaml` and set four paths: `train_normal_dir`, `test_normal_dir`, `test_anomaly_dir`, and `mask_dir`. An anomaly image `test/defect/sub/a.jpg` maps to `ground_truth/defect/sub/a_mask.png` with the shown suffix. A flat mask directory is also accepted. All anomalous test images need masks to compute pixel metrics. To include multiple defect classes, put them under `test_anomaly_dir` and mirror the relative class folders under `mask_dir`.

## Algorithms

PatchCore follows Anomalib's frozen Timm feature extractor, pooled layer embeddings, k-center coreset, nearest-neighbor search, and anomaly-map generation. Coreset selection uses a sparse random projection controlled by `patchcore.projection_eps`, then farthest-first search. Input Tiling splits the preprocessed image before feature extraction, then merges its feature embeddings with overlap averaging before memory-bank fitting or querying. Tile stride should align with backbone feature resolution (for `layer2`, a multiple of 8). `remove_border_count` is measured in **feature cells** and may only be used when overlap keeps every output cell covered.

CFlow follows Anomalib's frozen feature extractor and FrEIA conditional flow decoders, with the same sinusoidal position code and log-probability loss. TiledEnsemble trains one CFlow per tile position, predicts each tile, averages overlap maps, and uses the maximum of the merged map as its image score. This is a standalone adaptation of the Anomalib pipeline; its validation thresholding, normalization, seam smoothing, and visualization jobs are not part of this project's scoring path. The scores and metrics therefore need not match an Anomalib run numerically.

`data.image_size` resizes image and mask to a fixed `(height, width)` before tiling. Images use bilinear resize and ImageNet normalization; masks use nearest-neighbor resize. The same preprocessing is applied to training and test images.

## Output and metric definitions

Every invocation creates a unique `runs/<experiment>_<UTC time>_<id>/` containing `config_resolved.yaml`, `metrics.json`, and on success `predictions.csv`. Optional `model.pt` and `maps/*.npy` are controlled by the YAML. Failed runs leave `metrics.json` with `status: failed` and an `error.txt` traceback.

`metrics.json` stores:

- `I-AUROC`: AUROC of image labels and model image scores.
- `P-AUROC`: AUROC of every pixel in test masks and anomaly maps.
- `AUPRO`: 8-connected per-region overlap integrated over background FPR from 0 to `evaluation.aupro_fpr_limit` (default 0.3), divided by that FPR limit. Thresholds are a linspace over observed scores (default 200). This is an approximation whose granularity is configurable.
- `train_time_seconds`: full fitting time including batches and coreset selection or all tile-flow optimizations, excluding model construction and metric evaluation.
- `inference_fps`: number of test images divided by synchronized forward time. Images are preloaded on the compute device; data loading, mask loading, file writes, and metric calculation are excluded. CFlow evaluates one image per forward to avoid batch-dependent likelihood scaling.

AUROC/AUPRO is JSON `null` if the test split lacks the corresponding positive/negative classes or connected anomaly regions. Use an uncapped test split for final reporting. `data.max_train` and `data.max_test` can limit the dataset for smoke tests; `max_test` keeps some normal and anomalous examples when possible. The implementation keeps anomaly maps in RAM for pixel metrics, so choose an appropriate `image_size` for large evaluations.

The installed reference environment on this machine had CPU-only PyTorch. Full 512-pixel WideResNet50 and four CFlow tile models will be slow there; select a CUDA-enabled PyTorch environment for full experiments. Each tile CFlow is resident on the selected device, so GPU memory grows with the tile count.

Reference code: [Anomalib PatchCore](https://github.com/open-edge-platform/anomalib/tree/dc087d5c998d960ed739ae9276a5c051b8be4d9e/src/anomalib/models/image/patchcore), [CFlow](https://github.com/open-edge-platform/anomalib/tree/dc087d5c998d960ed739ae9276a5c051b8be4d9e/src/anomalib/models/image/cflow), and [TiledEnsemble](https://github.com/open-edge-platform/anomalib/tree/dc087d5c998d960ed739ae9276a5c051b8be4d9e/src/anomalib/pipelines/tiled_ensemble).
