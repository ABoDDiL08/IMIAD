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


def pixel_auroc_histogram(maps: list[np.ndarray], masks: list[np.ndarray],
                          bins: int, progress=None) -> float | None:
    """Bound RAM by counting positive/negative pixel scores in fixed bins."""
    if bins < 2:
        raise ValueError("Histogram P-AUROC requires at least two bins")
    if len(maps) != len(masks):
        raise ValueError("Anomaly maps and masks must have the same length")
    if not maps:
        return None
    low = min(float(np.min(scores)) for scores in maps)
    high = max(float(np.max(scores)) for scores in maps)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Anomaly maps must contain finite values")
    positives = np.zeros(bins, dtype=np.int64)
    negatives = np.zeros(bins, dtype=np.int64)
    for image_index, (scores, mask) in enumerate(zip(maps, masks), start=1):
        flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        flat_mask = np.asarray(mask, dtype=np.uint8).reshape(-1)
        if flat_scores.size != flat_mask.size or np.shape(scores) != np.shape(mask):
            raise ValueError("Anomaly map and mask shapes differ")
        for start in range(0, flat_scores.size, 1_000_000):
            end = min(start + 1_000_000, flat_scores.size)
            values = flat_scores[start:end]
            positive = flat_mask[start:end] != 0
            positives += np.histogram(values[positive], bins=bins, range=(low, high))[0]
            negatives += np.histogram(values[~positive], bins=bins, range=(low, high))[0]
        if progress is not None:
            progress("test_metrics_pixel_histogram", images_done=image_index,
                     images_total=len(maps), bins=bins)
    positive_count, negative_count = int(positives.sum()), int(negatives.sum())
    if not positive_count or not negative_count:
        return None
    negative_below = np.cumsum(negatives, dtype=np.float64) - negatives
    return float(np.dot(positives, negative_below + 0.5 * negatives)
                 / (positive_count * negative_count))


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
             fpr_limit: float, thresholds: int, pixel_auroc_bins: int | None = None,
             progress=None) -> dict:
    if progress is not None:
        progress("test_metrics_image_auroc")
    image_auroc = _auroc(np.asarray(image_labels), np.asarray(image_scores))
    if pixel_auroc_bins is None:
        if progress is not None:
            progress("test_metrics_pixel_arrays", images_total=len(maps))
        pixel_labels = np.concatenate([np.asarray(x, dtype=np.uint8).reshape(-1) for x in masks])
        pixel_scores = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in maps])
        if progress is not None:
            progress("test_metrics_pixel_auroc", pixel_count=len(pixel_scores))
        pixel_auroc = _auroc(pixel_labels, pixel_scores)
        del pixel_labels, pixel_scores
    else:
        if progress is not None:
            progress("test_metrics_pixel_histogram", images_done=0,
                     images_total=len(maps), bins=pixel_auroc_bins)
        pixel_auroc = pixel_auroc_histogram(maps, masks, pixel_auroc_bins, progress)
    if progress is not None:
        progress("test_metrics_aupro", images_total=len(maps))
    aupro_score = aupro(maps, masks, fpr_limit, thresholds)
    return {
        "I-AUROC": image_auroc,
        "P-AUROC": pixel_auroc,
        "AUPRO": aupro_score,
    }
