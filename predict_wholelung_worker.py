#!/usr/bin/env python3
"""
predict_wholelung_worker.py
================================================================
Runs ONLY the whole-lung 3D branch (MedicalNet-pretrained ResNet-10,
via MONAI) and prints its result as JSON to stdout. Isolated in its own
process for the same reason predict_ct_worker.py is: PyTorch's OpenMP
runtime must never share a process with CatBoost/XGBoost's.

IMPORTANT -- read before using this branch's output for anything:
This branch's probability is INFORMATIONAL / DOCUMENTATION ONLY. A
rigorous, cross-validated test (see project notes) showed that adding
or substituting this branch into the official 3-branch fusion does NOT
improve validated performance (CV AUC 0.7962 vs 0.7963/0.7964 for the
official fusion -- statistically indistinguishable). The official,
deployed risk score therefore continues to use ONLY the original 3
branches (clinical + 2D CT + 3D-MIL). This worker's output is surfaced
in the dashboard/performance views purely for transparency and future reference,
clearly labeled as such, and is never weighted into final_prob.

All DICOM I/O and lung-segmentation helper functions below are copied
verbatim from predict_ct_worker.py (already tested, single-patient
signature) rather than from extract_whole_lung_volume.py's batch/
manifest-oriented versions, so this worker is self-contained and uses
the exact same proven logic as the rest of the deployed pipeline.

RUN:
    python3 predict_wholelung_worker.py \
        --dicom-dir /path/to/new_patient/DICOM_FOLDER \
        --wholelung-models-dir ct_wholelung_pretrained_outputs \
        --device auto
"""

import argparse
import json
import os
import sys

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import pydicom
    from skimage.transform import resize
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.measure import label, regionprops
    from skimage.morphology import opening, closing, disk
except ImportError as e:
    print(json.dumps({"error": f"Missing package: {e}"}))
    sys.exit(1)

try:
    from monai.networks.nets import resnet10, resnet18, resnet34
except ImportError as e:
    print(json.dumps({"error": f"Missing MONAI: {e}. Install with: pip install monai huggingface_hub"}))
    sys.exit(1)

RESNET_BUILDERS = {10: resnet10, 18: resnet18, 34: resnet34}

# same constants as extract_whole_lung_volume.py / train_wholelung_pretrained.py
# -- must match exactly so a new patient's volume is built identically to
# the volumes the models were trained on
WORK_XY = 128
TARGET_Z = 64
TARGET_XY = 128
BBOX_MARGIN_FRAC = 0.06


# =============================================================================
# DICOM I/O + segmentation (copied verbatim from predict_ct_worker.py)
# =============================================================================
def is_real_dcm(filename):
    return filename.lower().endswith(".dcm") and not filename.startswith("._")


def find_series_dirs(dicom_dir):
    series_dirs = []
    for root, _dirs, files in os.walk(dicom_dir):
        if any(is_real_dcm(f) for f in files):
            series_dirs.append(root)
    return series_dirs


def pick_best_series_cheaply(series_dirs):
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


def decode_one_slice(path):
    try:
        ds_full = pydicom.dcmread(path)
        arr = ds_full.pixel_array.astype(np.float32)
        slope = float(getattr(ds_full, "RescaleSlope", 1))
        intercept = float(getattr(ds_full, "RescaleIntercept", 0))
        return arr * slope + intercept
    except Exception:
        return None


def apply_lung_window(slice_hu, center=-600, width=1500):
    lo, hi = center - width / 2, center + width / 2
    clipped = np.clip(slice_hu, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return (normalized * 255).astype(np.uint8)


MIN_LUNG_AREA_FRAC = 0.01
MAX_LUNG_REGIONS = 2


def segment_lung(slice_img):
    try:
        thresh = threshold_otsu(slice_img)
    except ValueError:
        return np.zeros_like(slice_img, dtype=bool)
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


# =============================================================================
# Whole-lung volume extraction (same algorithm as extract_whole_lung_volume.py)
# =============================================================================
def extract_whole_lung_volume(dicom_dir):
    series_dirs = find_series_dirs(dicom_dir)
    if not series_dirs:
        raise RuntimeError("No DICOM series found in dicom-dir")
    best_dir, n_files = pick_best_series_cheaply(series_dirs)
    if best_dir is None or n_files < 10:
        raise RuntimeError("No valid CT volume found (need >= 10 slices)")

    headers = load_headers_only(best_dir)
    if len(headers) < 10:
        raise RuntimeError("No valid CT volume found (need >= 10 slices)")
    headers.sort(key=lambda ph: sort_key(ph[1]))
    n_total = len(headers)

    processed_slices = []
    lung_present = np.zeros(n_total, dtype=bool)
    y_min, y_max, x_min, x_max = WORK_XY, 0, WORK_XY, 0

    for i, (path, _hdr) in enumerate(headers):
        arr = decode_one_slice(path)
        if arr is None:
            processed_slices.append(np.zeros((WORK_XY, WORK_XY), dtype=np.uint8))
            continue
        windowed = apply_lung_window(arr)
        small = resize(windowed, (WORK_XY, WORK_XY), preserve_range=True,
                        anti_aliasing=True).astype(np.uint8)
        processed_slices.append(small)

        lung_mask = segment_lung(small)
        if lung_mask.any():
            lung_present[i] = True
            ys, xs = np.where(lung_mask)
            y_min, y_max = min(y_min, ys.min()), max(y_max, ys.max())
            x_min, x_max = min(x_min, xs.min()), max(x_max, xs.max())

    lung_z_indices = np.where(lung_present)[0]
    if len(lung_z_indices) == 0:
        z_lo, z_hi = 0, n_total
        y0, y1, x0, x1 = 0, WORK_XY, 0, WORK_XY
    else:
        z_lo, z_hi = int(lung_z_indices.min()), int(lung_z_indices.max()) + 1
        margin_y = int(round((y_max - y_min) * BBOX_MARGIN_FRAC))
        margin_x = int(round((x_max - x_min) * BBOX_MARGIN_FRAC))
        y0 = max(0, y_min - margin_y)
        y1 = min(WORK_XY, y_max + margin_y + 1)
        x0 = max(0, x_min - margin_x)
        x1 = min(WORK_XY, x_max + margin_x + 1)

    raw_volume = np.stack([processed_slices[i][y0:y1, x0:x1] for i in range(z_lo, z_hi)], axis=0)
    final_volume = resize(raw_volume, (TARGET_Z, TARGET_XY, TARGET_XY),
                           preserve_range=True, anti_aliasing=True).astype(np.uint8)
    return final_volume


class WholeLungPretrainedClassifier(nn.Module):
    def __init__(self, resnet_depth=10):
        super().__init__()
        builder = RESNET_BUILDERS[resnet_depth]
        self.backbone = builder(pretrained=False, spatial_dims=3, n_input_channels=1,
                                 feed_forward=False, shortcut_type="B", bias_downsample=False)
        in_features = 512 if resnet_depth in (10, 18, 34) else 2048
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def get_device(override="auto"):
    if override == "cpu":
        return torch.device("cpu")
    if override == "mps":
        return torch.device("mps")
    if override == "cuda":
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dicom-dir", required=True)
    p.add_argument("--wholelung-models-dir", required=True)
    p.add_argument("--resnet-depth", type=int, default=10)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        device = get_device(args.device)
        volume = extract_whole_lung_volume(args.dicom_dir)
        vol_t = torch.from_numpy(volume.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

        fold_probs = []
        for fold_i in range(1, 6):
            path = os.path.join(args.wholelung_models_dir, f"model_fold{fold_i}.pt")
            if not os.path.exists(path):
                continue
            model = WholeLungPretrainedClassifier(args.resnet_depth).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(vol_t)).cpu().numpy()[0]
            fold_probs.append(float(prob))

        if not fold_probs:
            raise RuntimeError(f"No whole-lung fold models found in {args.wholelung_models_dir}")

        print(json.dumps({
            "wholelung_pretrained_prob": float(np.mean(fold_probs)),
            "wholelung_pretrained_folds": fold_probs,
            "device": str(device),
            "note": "Reference/documentation branch only -- not included in the official fused risk score.",
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
