#!/usr/bin/env python3
"""
predict_ct_worker.py
================================================================
Runs ONLY the two CT branches (old 2D CNN + new 3D-MIL) and prints the
result as JSON to stdout. Deliberately never imports catboost/xgboost
-- called as a SEPARATE PROCESS by predict_new_patient.py so PyTorch's
OpenMP runtime never has to coexist with CatBoost/XGBoost's in the same
process (that conflict was causing a hard crash on Apple Silicon).

RUN (called automatically by predict_new_patient.py, or standalone):
    python3 predict_ct_worker.py \
        --dicom-dir /path/to/new_patient/DICOM_FOLDER \
        --cnn2d-models-dir ct_model_outputs \
        --mil3d-models-dir ct_cnn_3d_mil_outputs \
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
    from skimage.morphology import opening, closing, disk, remove_small_objects
except ImportError as e:
    print(json.dumps({"error": f"Missing package: {e}"}))
    sys.exit(1)

try:
    from torchvision import models as tv_models
except ImportError as e:
    print(json.dumps({"error": f"Missing torchvision: {e}"}))
    sys.exit(1)

# =============================================================================
# DICOM I/O
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


# =============================================================================
# Segmentation
# =============================================================================
MIN_LUNG_AREA_FRAC = 0.01
MAX_LUNG_REGIONS = 2
MIN_NODULE_AREA_PX = 4
MAX_NODULE_AREA_FRAC = 0.15


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


def find_nodule_candidates(slice_img, lung_mask):
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
    candidate_mask = remove_small_objects(candidate_mask, min_size=MIN_NODULE_AREA_PX)
    labeled = label(candidate_mask)
    max_area = MAX_NODULE_AREA_FRAC * slice_img.size
    regions = [r for r in regionprops(labeled) if MIN_NODULE_AREA_PX <= r.area <= max_area]
    keep_labels = [r.label for r in regions]
    return np.isin(labeled, keep_labels), regions


# =============================================================================
# BRANCH: OLD 2D CNN
# =============================================================================
N_SLICES_2D = 5
CROP_SPACING_2D = 8
IMG_SIZE_2D = 224


def extract_2d_slices_for_patient(dicom_dir):
    series_dirs = find_series_dirs(dicom_dir)
    if not series_dirs:
        raise RuntimeError("No DICOM series found in dicom-dir")
    best_dir, n_files = pick_best_series_cheaply(series_dirs)
    if best_dir is None or n_files < 3:
        raise RuntimeError("No valid CT volume found")
    headers = load_headers_only(best_dir)
    headers.sort(key=lambda ph: sort_key(ph[1]))
    n_total = len(headers)

    scores = [0.0] * n_total
    processed_cache = {}
    for i, (path, _hdr) in enumerate(headers):
        arr = decode_one_slice(path)
        if arr is None:
            continue
        windowed = apply_lung_window(arr)
        small = resize(windowed, (IMG_SIZE_2D, IMG_SIZE_2D), preserve_range=True,
                        anti_aliasing=True).astype(np.uint8)
        processed_cache[i] = small
        lung_mask = segment_lung(small)
        if lung_mask.any():
            _, regions = find_nodule_candidates(small, lung_mask)
            scores[i] = max((r.area for r in regions), default=0.0)

    ranked = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
    chosen = []
    for idx in ranked:
        if scores[idx] <= 0:
            break
        if all(abs(idx - c) >= 4 for c in chosen):
            chosen.append(idx)
        if len(chosen) == N_SLICES_2D:
            break
    if len(chosen) < N_SLICES_2D:
        mid = n_total // 2
        offsets = [(i - N_SLICES_2D // 2) * CROP_SPACING_2D for i in range(N_SLICES_2D)]
        for off in offsets:
            idx = max(0, min(n_total - 1, mid + off))
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == N_SLICES_2D:
                break
    chosen = sorted(set(chosen))[:N_SLICES_2D]
    while len(chosen) < N_SLICES_2D:
        chosen.append(chosen[-1])

    slices = []
    for idx in chosen:
        if idx not in processed_cache:
            arr = decode_one_slice(headers[idx][0])
            processed_cache[idx] = resize(apply_lung_window(arr), (IMG_SIZE_2D, IMG_SIZE_2D),
                                           preserve_range=True, anti_aliasing=True).astype(np.uint8)
        slices.append(processed_cache[idx])
    return np.stack(slices)


def build_2d_model():
    model = tv_models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def slices_to_base64_pngs(slices):
    """Encode the raw uint8 slice arrays as base64 PNGs for UI display."""
    try:
        import base64
        import io as _io
        from PIL import Image
        out = []
        for i in range(slices.shape[0]):
            img = Image.fromarray(slices[i], mode="L")
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode())
        return out
    except Exception:
        return []


def predict_old_2d_cnn(dicom_dir, models_dir, device):
    slices = extract_2d_slices_for_patient(dicom_dir)
    slice_thumbnails_b64 = slices_to_base64_pngs(slices)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    imgs = []
    for i in range(slices.shape[0]):
        arr_3ch = np.stack([slices[i]] * 3, axis=-1).astype(np.float32) / 255.0
        arr_3ch = (arr_3ch - mean) / std
        imgs.append(torch.from_numpy(arr_3ch).permute(2, 0, 1).float())
    batch = torch.stack(imgs).to(device)

    fold_probs = []
    for fold_i in range(1, 6):
        path = os.path.join(models_dir, f"model_fold{fold_i}.pt")
        if not os.path.exists(path):
            continue
        model = build_2d_model().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(batch).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
        fold_probs.append(float(probs.max()))

    if not fold_probs:
        raise RuntimeError(f"No 2D CNN fold models found in {models_dir}")
    return float(np.mean(fold_probs)), fold_probs, slice_thumbnails_b64


# =============================================================================
# BRANCH: NEW 3D-MIL
# =============================================================================
SCAN_XY_3D = 160
PATCH_XY_3D = 48
PATCH_Z_3D = 32
CROP_SCAN_PX_3D = 40
MAX_Z_EXTENT_FRAC = 0.12
MAX_Z_EXTENT_SLICES = 15
MIN_FILL_RATIO = 0.25
MIN_VOXEL_VOLUME = 20
MAX_INSTANCES_3D = 3
EMBED_DIM_3D = 128


def coarse_volumetric_scan(headers):
    n_total = len(headers)
    scan_volume = np.zeros((n_total, SCAN_XY_3D, SCAN_XY_3D), dtype=np.uint8)
    candidate_volume = np.zeros((n_total, SCAN_XY_3D, SCAN_XY_3D), dtype=bool)
    for i, (path, _hdr) in enumerate(headers):
        arr = decode_one_slice(path)
        if arr is None:
            continue
        windowed = apply_lung_window(arr)
        small = resize(windowed, (SCAN_XY_3D, SCAN_XY_3D), preserve_range=True,
                        anti_aliasing=True).astype(np.uint8)
        scan_volume[i] = small
        lung_mask = segment_lung(small)
        nodule_mask, _regions = find_nodule_candidates(small, lung_mask)
        candidate_volume[i] = nodule_mask
    return scan_volume, candidate_volume


def get_3d_components(candidate_volume, n_total):
    labeled, n_labels = ndi.label(candidate_volume, structure=np.ones((3, 3, 3)))
    if n_labels == 0:
        return []
    components = []
    for lbl in range(1, n_labels + 1):
        coords = np.argwhere(labeled == lbl)
        volume = len(coords)
        if volume < MIN_VOXEL_VOLUME:
            continue
        z0, y0, x0 = coords.min(axis=0)
        z1, y1, x1 = coords.max(axis=0) + 1
        z_extent_slices = int(z1 - z0)
        y_extent, x_extent = int(y1 - y0), int(x1 - x0)
        z_extent_frac = z_extent_slices / max(n_total, 1)
        bbox_volume = max(z_extent_slices * y_extent * x_extent, 1)
        fill_ratio = volume / bbox_volume
        if z_extent_frac > MAX_Z_EXTENT_FRAC or z_extent_slices > MAX_Z_EXTENT_SLICES:
            continue
        if fill_ratio < MIN_FILL_RATIO:
            continue
        centroid = coords.mean(axis=0)
        components.append({
            "centroid_z": float(centroid[0]), "centroid_y": float(centroid[1]),
            "centroid_x": float(centroid[2]), "voxel_volume": int(volume),
        })
    components.sort(key=lambda c: c["voxel_volume"], reverse=True)
    return components


def crop_full_res_patch(headers, native_rows, native_cols, z_center, y_scan, x_scan):
    n_total = len(headers)
    z_lo = int(round(z_center - PATCH_Z_3D / 2))
    z_lo = max(0, min(n_total - PATCH_Z_3D, z_lo))
    z_hi = z_lo + PATCH_Z_3D
    y_native = y_scan * (native_rows / SCAN_XY_3D)
    x_native = x_scan * (native_cols / SCAN_XY_3D)
    crop_h = int(round(CROP_SCAN_PX_3D * (native_rows / SCAN_XY_3D)))
    crop_w = int(round(CROP_SCAN_PX_3D * (native_cols / SCAN_XY_3D)))
    y0 = int(max(0, min(native_rows - crop_h, y_native - crop_h / 2)))
    x0 = int(max(0, min(native_cols - crop_w, x_native - crop_w / 2)))

    patch_slices = []
    for i in range(z_lo, z_hi):
        path, _hdr = headers[i]
        arr = decode_one_slice(path)
        if arr is None:
            patch_slices.append(np.zeros((PATCH_XY_3D, PATCH_XY_3D), dtype=np.uint8))
            continue
        windowed = apply_lung_window(arr)
        crop = windowed[y0:y0 + crop_h, x0:x0 + crop_w]
        resized = resize(crop, (PATCH_XY_3D, PATCH_XY_3D), preserve_range=True,
                          anti_aliasing=True).astype(np.uint8)
        patch_slices.append(resized)
    return np.stack(patch_slices, axis=0)


def extract_3d_patches_for_patient(dicom_dir):
    series_dirs = find_series_dirs(dicom_dir)
    if not series_dirs:
        raise RuntimeError("No DICOM series found in dicom-dir")
    best_dir, n_files = pick_best_series_cheaply(series_dirs)
    if best_dir is None or n_files < PATCH_Z_3D:
        raise RuntimeError("No valid CT volume found")
    headers = load_headers_only(best_dir)
    headers.sort(key=lambda ph: sort_key(ph[1]))
    n_total = len(headers)
    native_rows = int(getattr(headers[0][1], "Rows", 512))
    native_cols = int(getattr(headers[0][1], "Columns", 512))

    _scan_volume, candidate_volume = coarse_volumetric_scan(headers)
    components = get_3d_components(candidate_volume, n_total)[:MAX_INSTANCES_3D]

    patches = []
    if components:
        for cand in components:
            patch = crop_full_res_patch(headers, native_rows, native_cols,
                                         cand["centroid_z"], cand["centroid_y"], cand["centroid_x"])
            patches.append(patch)
    else:
        z_c, y_c, x_c = n_total / 2, SCAN_XY_3D / 2, SCAN_XY_3D / 2
        patches.append(crop_full_res_patch(headers, native_rows, native_cols, z_c, y_c, x_c))
    return patches


def conv_block_3d(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_ch),
        nn.ReLU(inplace=True),
    )


class InstanceEncoder3D(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM_3D):
        super().__init__()
        self.features = nn.Sequential(
            conv_block_3d(1, 16), nn.MaxPool3d(2),
            conv_block_3d(16, 32), nn.MaxPool3d(2),
            conv_block_3d(32, 64), nn.MaxPool3d(2),
            conv_block_3d(64, embed_dim),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)


class AttentionMIL3D(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM_3D):
        super().__init__()
        self.encoder = InstanceEncoder3D(embed_dim)
        self.attention = nn.Sequential(nn.Linear(embed_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, 1))

    def forward(self, x, mask):
        B, K = x.shape[0], x.shape[1]
        x_flat = x.view(B * K, *x.shape[2:])
        emb = self.encoder(x_flat).view(B, K, -1)
        attn_logits = self.attention(emb).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=1)
        bag_emb = (attn_weights.unsqueeze(-1) * emb).sum(dim=1)
        logits = self.classifier(bag_emb).squeeze(-1)
        return logits, attn_weights


def predict_3d_mil(dicom_dir, models_dir, device):
    patches = extract_3d_patches_for_patient(dicom_dir)
    k = MAX_INSTANCES_3D
    vols = np.zeros((k, PATCH_Z_3D, PATCH_XY_3D, PATCH_XY_3D), dtype=np.float32)
    mask = np.zeros((k,), dtype=np.float32)
    for i, p in enumerate(patches[:k]):
        vols[i] = p.astype(np.float32) / 255.0
        mask[i] = 1.0
    vols_t = torch.from_numpy(vols).unsqueeze(1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    fold_probs = []
    for fold_i in range(1, 6):
        path = os.path.join(models_dir, f"model_fold{fold_i}.pt")
        if not os.path.exists(path):
            continue
        model = AttentionMIL3D().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits, _attn = model(vols_t, mask_t)
            prob = torch.sigmoid(logits).cpu().numpy()[0]
        fold_probs.append(float(prob))

    if not fold_probs:
        raise RuntimeError(f"No 3D-MIL fold models found in {models_dir}")
    return float(np.mean(fold_probs)), fold_probs, len(patches)


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
    p.add_argument("--cnn2d-models-dir", required=True)
    p.add_argument("--mil3d-models-dir", required=True)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        device = get_device(args.device)
        old_cnn_prob, old_cnn_folds, slice_thumbnails_b64 = predict_old_2d_cnn(
            args.dicom_dir, args.cnn2d_models_dir, device)
        mil_prob, mil_folds, n_patches = predict_3d_mil(args.dicom_dir, args.mil3d_models_dir, device)
        print(json.dumps({
            "old_cnn_prob": old_cnn_prob, "old_cnn_folds": old_cnn_folds,
            "mil_prob": mil_prob, "mil_folds": mil_folds, "n_3d_patches": n_patches,
            "slice_thumbnails_b64": slice_thumbnails_b64,
            "device": str(device),
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
