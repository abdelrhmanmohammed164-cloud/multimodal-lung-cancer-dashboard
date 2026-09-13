#!/usr/bin/env python3
"""
ct_segmentation_utils.py
================================================================
Shared lung + nodule-candidate segmentation logic, used by BOTH:
  - extract_ct_slices.py (coarse scan, to pick WHICH slices to keep)
  - segment_lung_and_nodules.py (full segmentation on the kept slices,
    for radiomics feature extraction)

Pulling this out into one module means both scripts always agree on
what counts as "lung tissue" and "a nodule candidate" -- no drift
between the slice-selection stage and the feature-extraction stage.

Two-stage approximate segmentation (no ground-truth masks in NLST):
  STAGE 1 -- Lung segmentation: intensity thresholding + morphological
             cleanup + connected-component selection.
  STAGE 2 -- Nodule-candidate detection: within the lung mask only,
             finds denser (soft-tissue-like) blobs as approximate
             nodule regions. Coarse proxy, not verified segmentation.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import opening, closing, disk, remove_small_objects

MIN_LUNG_AREA_FRAC = 0.01     # a lung region must be >=1% of the image to count
MAX_LUNG_REGIONS = 2          # keep at most the 2 largest (left + right lung)
MIN_NODULE_AREA_PX = 4        # ignore candidate blobs smaller than this (noise)
MAX_NODULE_AREA_FRAC = 0.15   # ignore blobs implausibly large (likely mis-segmentation)


def segment_lung(slice_img):
    """STAGE 1. slice_img: uint8 (H,W), lung-windowed. Returns a boolean
    lung mask."""
    try:
        thresh = threshold_otsu(slice_img)
    except ValueError:
        return np.zeros_like(slice_img, dtype=bool)  # flat image, nothing to segment

    dark_mask = slice_img <= thresh
    cleared = clear_border(dark_mask)
    filled = ndi.binary_fill_holes(cleared)
    filled = opening(filled, disk(2))
    filled = closing(filled, disk(3))

    labeled = label(filled)
    if labeled.max() == 0:
        return np.zeros_like(slice_img, dtype=bool)

    props = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
    min_area = MIN_LUNG_AREA_FRAC * slice_img.size
    keep_labels = [r.label for r in props[:MAX_LUNG_REGIONS] if r.area >= min_area]

    return np.isin(labeled, keep_labels)


def find_nodule_candidates(slice_img, lung_mask):
    """STAGE 2. Within lung_mask only, finds brighter (soft-tissue-like)
    blobs as approximate nodule candidates. Returns (candidate_mask,
    list_of_region_properties)."""
    if not lung_mask.any():
        return np.zeros_like(lung_mask), []

    lung_pixels = slice_img[lung_mask]
    if lung_pixels.max() == lung_pixels.min():
        return np.zeros_like(lung_mask), []

    try:
        local_thresh = threshold_otsu(lung_pixels)
    except ValueError:
        return np.zeros_like(lung_mask), []

    candidate_mask = (slice_img > local_thresh) & lung_mask
    candidate_mask = remove_small_objects(candidate_mask, min_size=MIN_NODULE_AREA_PX + 1)

    labeled = label(candidate_mask)
    max_area = MAX_NODULE_AREA_FRAC * slice_img.size
    regions = [r for r in regionprops(labeled) if MIN_NODULE_AREA_PX <= r.area <= max_area]

    keep_labels = [r.label for r in regions]
    final_mask = np.isin(labeled, keep_labels)
    return final_mask, regions


def score_slice(slice_img):
    """Convenience wrapper for the COARSE SCAN in extract_ct_slices.py:
    runs both stages on one already lung-windowed uint8 slice and
    returns a single scalar 'how nodule-like is this slice' score, plus
    the region list (for debugging/manifest logging).

    Score = area of the single largest candidate region on this slice.
    Using the largest region (not total count or summed area) keeps the
    score robust to many tiny noise blobs scattered across a slice --
    one real dense structure should outscore a handful of small ones.
    """
    lung_mask = segment_lung(slice_img)
    if not lung_mask.any():
        return 0.0, []
    _, regions = find_nodule_candidates(slice_img, lung_mask)
    if not regions:
        return 0.0, []
    largest_area = max(r.area for r in regions)
    return float(largest_area), regions
