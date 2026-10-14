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

PatchCore follows Anomalib's frozen Timm feature extractor, pooled layer embeddings, k-center coreset, nearest-neighbor search, and anomaly-map generation. Coreset selection uses a sparse random projection controlled by `patchcore.projection_eps`, then farthest-first search. With Input Tiling enabled, the loader preserves the raw image and mask (`data.image_size: null`). `input_tiling.tile_size` and `stride` are measured on the raw image. The grid is padded if needed, then **each tile is resized to `input_tiling.tile_resize` before the backbone**. When `tile_resize` equals `tile_size`, the resize call is skipped.

Low-memory PatchCore keeps **one shared backbone and one global memory bank**. `patchcore.tile_batch_size` bounds how many tiles enter the backbone in one forward pass; the full image and unused tiles remain on CPU. Training pools features from every tile into the same coreset candidate set. With `patchcore.embedding_storage: disk`, candidate embeddings and their random projections are temporarily stored on disk during farthest-first selection and deleted after fitting. `memory` keeps candidates in RAM instead. The selected memory bank and nearest-neighbor search stay on CPU; `patchcore.distance_bank_chunk_size` bounds the distance matrix. During inference each tile is scored independently against this one bank, its anomaly map is resized to the **raw tile size**, and overlapping maps are averaged on CPU. The image score is the maximum patch score across tiles, with the configured PatchCore neighbor weighting. Blur is applied after the image map has been joined. `remove_border_count` is now measured in pixels of the native-size score map; it requires enough overlap to cover every output pixel. This tile-score merging differs numerically from the earlier feature-level untiling implementation and from Anomalib's one-model-per-position TiledEnsemble.

CFlow follows Anomalib's frozen feature extractor and FrEIA conditional flow decoders, with the same sinusoidal position code and log-probability loss. TiledEnsemble trains one CFlow per tile position, predicts each tile, averages overlap maps, and uses the maximum of the merged map as its image score. This is a standalone adaptation of the Anomalib pipeline; its validation thresholding, normalization, seam smoothing, and visualization jobs are not part of this project's scoring path. The scores and metrics therefore need not match an Anomalib run numerically.

For CFlow + TiledEnsemble, `data.image_size` remains a required fixed `(height, width)` because the number of position-specific models is fixed when the ensemble is created. Its images use bilinear resize and ImageNet normalization; masks use nearest-neighbor resize. PatchCore + Input Tiling instead requires `data.image_size: null`, keeps masks at native resolution, and applies bilinear resize only to image tiles when `tile_resize` differs from `tile_size`. The same preprocessing is applied to training and test images.

Set `data.batch_size: 1` for PatchCore when raw training images have different dimensions; PyTorch's default batch collation cannot stack differently sized images.

## Output and metric definitions

Every invocation creates a unique `runs/<experiment>_<UTC time>_<id>/` containing `config_resolved.yaml`, `metrics.json`, and on success `predictions.csv`. Optional `model.pt` and `maps/*.npy` are controlled by the YAML. Failed runs leave `metrics.json` with `status: failed` and an `error.txt` traceback.

`metrics.json` stores:

- `I-AUROC`: AUROC of image labels and model image scores.
- `P-AUROC`: AUROC of every pixel in test masks and anomaly maps.
- `AUPRO`: 8-connected per-region overlap integrated over background FPR from 0 to `evaluation.aupro_fpr_limit` (default 0.3), divided by that FPR limit. Thresholds are a linspace over observed scores (default 200). This is an approximation whose granularity is configurable.
- `train_time_seconds`: full fitting time including batches and coreset selection or all tile-flow optimizations, excluding model construction and metric evaluation.
- `inference_fps`: number of test images divided by synchronized inference time. For tiled PatchCore, this includes tile CPU-to-GPU transfers, backbone forwards, CPU nearest-neighbor search, map joining and blur. For CFlow, the image is preloaded on the compute device. Data loading, mask loading, file writes, and metric calculation are excluded. CFlow evaluates one image per forward to avoid batch-dependent likelihood scaling.

AUROC/AUPRO is JSON `null` if the test split lacks the corresponding positive/negative classes or connected anomaly regions. Use an uncapped test split for final reporting. `data.max_train` and `data.max_test` can limit the dataset for smoke tests; `max_test` keeps some normal and anomalous examples when possible. PatchCore's temporary training files are created inside the run directory and automatically removed after fitting. The final memory bank and all test anomaly maps/masks for pixel metrics still occupy CPU RAM; native-resolution evaluation can therefore need considerable RAM.

The installed reference environment on this machine had CPU-only PyTorch. Full 512-pixel WideResNet50 and four CFlow tile models will be slow there; select a CUDA-enabled PyTorch environment for full experiments. Each tile CFlow is resident on the selected device, so GPU memory grows with the tile count.

Reference code: [Anomalib PatchCore](https://github.com/open-edge-platform/anomalib/tree/dc087d5c998d960ed739ae9276a5c051b8be4d9e/src/anomalib/models/image/patchcore), [CFlow](https://github.com/open-edge-platform/anomalib/tree/dc087d5c998d960ed739ae9276a5c051b8be4d9e/src/anomalib/models/image/cflow), and [TiledEnsemble](https://github.com/open-edge-platform/anomalib/tree/dc087d5c998d960ed739ae9276a5c051b8be4d9e/src/anomalib/pipelines/tiled_ensemble).
