#!/usr/bin/env python3
"""
ct_dicom_io_utils.py
================================================================
Shared low-level DICOM reading helpers for the 3D patch extraction
pipeline (extract_ct_patches_3d.py). Deliberately duplicated (not
imported) from extract_ct_slices.py -- that script is already
validated on real data, and this keeps the 3D pipeline decoupled so
changes here can't accidentally destabilize the working 2D pipeline.
"""

import os
import numpy as np
import pydicom


def is_real_dcm(filename):
    """True .dcm files only -- excludes macOS AppleDouble resource-fork
    files (e.g. '._image1.dcm')."""
    return filename.lower().endswith(".dcm") and not filename.startswith("._")


def find_series_dirs(ct_dir, pid):
    """Every SeriesInstanceUID folder for a patient."""
    pid_dir = os.path.join(ct_dir, str(pid))
    if not os.path.isdir(pid_dir):
        return []
    series_dirs = []
    for root, _dirs, files in os.walk(pid_dir):
        if any(is_real_dcm(f) for f in files):
            series_dirs.append(root)
    return series_dirs


def pick_best_series_cheaply(series_dirs):
    """Cheap: counts real .dcm files per candidate series folder (no
    pixel reads), picks the one with the most files."""
    best_dir, best_count = None, 0
    for sd in series_dirs:
        try:
            n = sum(1 for f in os.listdir(sd) if is_real_dcm(f))
        except OSError:
            continue
        if n > best_count:
            best_dir, best_count = sd, n
    return best_dir, best_count


def load_headers_only(series_dir):
    """DICOM headers only (stop_before_pixels=True) -- sort order without
    decoding pixel data."""
    dcm_files = [f for f in os.listdir(series_dir) if is_real_dcm(f)]
    headers = []
    for f in dcm_files:
        try:
            path = os.path.join(series_dir, f)
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            headers.append((path, ds))
        except Exception:
            continue
    return headers


def sort_key(ds):
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    return float(getattr(ds, "InstanceNumber", 0))


def get_z_position(ds):
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    return None


def get_pixel_spacing(ds):
    """Returns (row_spacing_mm, col_spacing_mm), falling back to 1.0 if
    unavailable (only used for documentation/manifest logging here, not
    for physical resampling)."""
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None:
        return (1.0, 1.0)
    return (float(ps[0]), float(ps[1]))


def decode_one_slice(path):
    """Full pixel data for one file, converted to HU. Returns None
    (instead of raising) if this file fails to decode."""
    try:
        ds_full = pydicom.dcmread(path)
        arr = ds_full.pixel_array.astype(np.float32)
        slope = float(getattr(ds_full, "RescaleSlope", 1))
        intercept = float(getattr(ds_full, "RescaleIntercept", 0))
        return arr * slope + intercept
    except Exception:
        return None


def apply_lung_window(slice_hu, center=-600, width=1500):
    """Standard CT windowing: clips to the lung HU range, rescales to
    0-255."""
    lo, hi = center - width / 2, center + width / 2
    clipped = np.clip(slice_hu, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return (normalized * 255).astype(np.uint8)
