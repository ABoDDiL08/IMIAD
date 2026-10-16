# Setup


```powershell
pip install -r requirements.txt
```

Chỉnh sửa nội dung đường dẫn đến data phù hợp trong configs/folders_patchcore.yaml và configs/folders_cflow_ensemble.yaml

```powershell
python -m imiad.run --config configs/real_iad_patchcore.yaml
python -m imiad.run --config configs/real_iad_cflow_ensemble.yaml
python -m imiad.run --config configs/real_iad_cflow_input_tiling.yaml
```

When `evaluation.visualization.enabled: true`, the runner calibrates an image-score
threshold from the configured quantile of normal training images, unless
`evaluation.visualization.image_threshold` is set explicitly. It saves every
false positive and false negative plus the five highest-scoring true anomalies
under `runs/<run_id>/visualizations/`. Each case has a review sheet (original,
anomaly heatmap, overlay, and ground-truth mask), a separate heatmap and overlay,
and a row in `manifest.csv`. The threshold and TP/FP/TN/FN counts are recorded
in `metrics.json`. This is a diagnostic threshold based on training images, not
a validated operating threshold; use a threshold from separate normal validation
data for deployment decisions. The visual output is resized to `max_side` for
bounded RAM and uses per-image 1st/99th-percentile color scaling.

Test anomaly maps and masks are staged as temporary `.npy` files and read back
one image at a time for pixel metrics. The temporary files are deleted after a
normal run. `evaluation.save_maps: true` keeps the raw anomaly maps separately.
For native-size test images, use `evaluation.pixel_auroc_bins` to avoid the exact
P-AUROC implementation concatenating all pixels in RAM.

