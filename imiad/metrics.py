"""Image AUROC, pixel AUROC and normalized area under PRO/FPR curve."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if np.unique(labels).size < 2:
        return None
    return float(roc_auc_score(labels, scores))


def aupro(maps: list[np.ndarray], masks: list[np.ndarray],
          fpr_limit: float = 0.3, num_thresholds: int = 200) -> float | None:
    """Integrate mean connected-region overlap against background FPR.

    Each 8-connected anomaly component contributes equally. The integral is
    divided by `fpr_limit` and uses a linear interpolation at that limit.
    """
    if not maps or len(maps) != len(masks):
        return None
    bounds = [(float(np.min(m)), float(np.max(m))) for m in maps]
    low, high = min(b[0] for b in bounds), max(b[1] for b in bounds)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Anomaly maps must contain finite values")
    if low == high:
        return 0.0
    thresholds = np.linspace(high + np.finfo(np.float32).eps, low, num_thresholds)
    background_total = 0
    false_positives = np.zeros(len(thresholds), dtype=np.float64)
    region_sum = np.zeros(len(thresholds), dtype=np.float64)
    region_count = 0
    for scores, mask in zip(maps, masks):
        scores = np.asarray(scores, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if scores.shape != mask.shape:
            raise ValueError("Anomaly map and mask shapes differ")
        background = np.sort(scores[~mask])
        background_total += background.size
        false_positives += background.size - np.searchsorted(background, thresholds, side="left")
        components, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        for idx in range(1, count + 1):
            region = np.sort(scores[components == idx])
            region_sum += (region.size - np.searchsorted(region, thresholds, side="left")) / region.size
            region_count += 1
    if not region_count or not background_total:
        return None
    fpr = false_positives / background_total
    pro = region_sum / region_count
    order = np.argsort(fpr)
    fpr, pro = fpr[order], pro[order]
    unique_fpr, first = np.unique(fpr, return_index=True)
    # Duplicate FPRs may occur; keep their maximum overlap.
    max_pro = np.maximum.reduceat(pro, first)
    if unique_fpr[0] > 0:
        unique_fpr = np.r_[0.0, unique_fpr]
        max_pro = np.r_[0.0, max_pro]
    left = unique_fpr < fpr_limit
    x = np.r_[unique_fpr[left], fpr_limit]
    y = np.r_[max_pro[left], np.interp(fpr_limit, unique_fpr, max_pro)]
    return float(trapezoid(y, x) / fpr_limit)


def evaluate(image_labels: list[int], image_scores: list[float],
             maps: list[np.ndarray], masks: list[np.ndarray],
             fpr_limit: float, thresholds: int) -> dict:
    image_auroc = _auroc(np.asarray(image_labels), np.asarray(image_scores))
    pixel_labels = np.concatenate([np.asarray(x, dtype=np.uint8).reshape(-1) for x in masks])
    pixel_scores = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in maps])
    pixel_auroc = _auroc(pixel_labels, pixel_scores)
    return {
        "I-AUROC": image_auroc,
        "P-AUROC": pixel_auroc,
        "AUPRO": aupro(maps, masks, fpr_limit, thresholds),
    }
