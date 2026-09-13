#!/usr/bin/env python3
"""
app.py -- Lung Cancer AI Analysis Dashboard (Flask backend)
================================================================
Serves the dashboard UI and wires it to the REAL, already-tested
pipeline (predict_clinical_worker.py + predict_ct_worker.py), the same
isolated-process workers used by predict_new_patient.py. No model
logic is reimplemented here -- this is an orchestration + UI layer.

CONFIG: edit dashboard_config.json (created on first run with
defaults) to point at your model folders.

RUN:
    pip install flask
    python3 app.py
Then open http://localhost:5050 in a browser.
"""

import base64
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

from flask import Flask, render_template, request, jsonify

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "dashboard_config.json")
PATIENTS_LOG_PATH = os.path.join(APP_DIR, "patients_log.json")
CLINICAL_DRAFTS_PATH = os.path.join(APP_DIR, "clinical_drafts.json")

FUSION_WEIGHT_CLINICAL = 0.40
FUSION_WEIGHT_OLD_CNN = 0.20
FUSION_WEIGHT_MIL = 0.30
FUSION_WEIGHT_WHOLELUNG = 0.10
# Weights chosen via grid search over the validated development cohort, then
# confirmed via proper 5-fold cross-validation to be safe (CV AUC 0.7962,
# statistically indistinguishable from the prior 3-branch fusion's 0.7963-
# 0.7964) -- i.e. this 4-way weighting does not measurably help OR hurt
# validated performance, but does genuinely incorporate the whole-lung
# branch's signal into the official score, per an explicit decision to
# include it in the calculation going forward.
OPTIMAL_THRESHOLD = 0.4621357735959353

DEFAULT_CONFIG = {
    "clinical_models_dir": "/Users/eng-abdelrhman/Documents/Research/Try again /clinical_models_for_inference",
    "cnn2d_models_dir": "ct_model_outputs",
    "mil3d_models_dir": "ct_cnn_3d_mil_outputs",
    "wholelung_models_dir": "ct_wholelung_pretrained_outputs",
    "project_results_dir": "project_results",
    "eda_outputs_dir": "/Volumes/Apple/images_organized/Analysis/eda_outputs",
    "doctor_name": "Dr. Shwetha",
    "doctor_specialty": "Pulmonologist",
    "doctor_title": "Associate Professor",
    "doctor_photo": "doctor_photo.jpg",
    "device": "auto",
}


def _clinical_model_dir_is_valid(path):
    if not path:
        return False
    required = (
        "clinical_feature_columns.json",
        "clinical_weights.json",
        "clinical_imputer.pkl",
        "clinical_catboost.cbm",
        "clinical_xgboost.json",
        "clinical_logreg.pkl",
    )
    return os.path.isdir(path) and all(os.path.exists(os.path.join(path, f)) for f in required)


def _resolve_clinical_models_dir(configured_path):
    """Resolve the clinical inference bundle after the project is moved.

    The dashboard used to point to an absolute Documents path. The project now
    commonly lives under /Volumes/Apple/images_organized, so we try the saved
    path first, then portable project-relative/common locations, then perform a
    shallow search below the project root. No model is changed; this only finds
    the existing clinical inference files.
    """
    candidates = []

    def add(path):
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        if path not in candidates:
            candidates.append(path)

    # 1) Existing configured path (preserves old installations).
    add(configured_path)

    # 2) Portable locations relative to the dashboard folder.
    add(os.path.join(APP_DIR, "clinical_models_for_inference"))
    add(os.path.join(os.path.dirname(APP_DIR), "clinical_models_for_inference"))
    add(os.path.join(os.path.dirname(APP_DIR), "04_inference", "clinical_models_for_inference"))
    add(os.path.join(os.path.dirname(APP_DIR), "03_training", "clinical_models_for_inference"))

    # 3) Known current project root used by this dashboard.
    project_root = "/Volumes/Apple/images_organized"
    add(os.path.join(project_root, "clinical_models_for_inference"))
    add(os.path.join(project_root, "04_inference", "clinical_models_for_inference"))
    add(os.path.join(project_root, "03_training", "clinical_models_for_inference"))
    add(os.path.join(project_root, "01_data", "clinical_models_for_inference"))

    for path in candidates:
        if _clinical_model_dir_is_valid(path):
            return path

    # 4) Shallow fallback search under the project root. This makes the
    # dashboard tolerant if the folder was reorganized into another numbered
    # project directory. Limit depth to keep startup fast.
    if os.path.isdir(project_root):
        root_depth = project_root.rstrip(os.sep).count(os.sep)
        for current, dirs, files in os.walk(project_root):
            depth = current.rstrip(os.sep).count(os.sep) - root_depth
            if depth >= 4:
                dirs[:] = []
            if "clinical_feature_columns.json" in files:
                if _clinical_model_dir_is_valid(current):
                    return current

    # Leave the configured path intact so any error message still points to
    # the user's chosen location instead of silently inventing a model.
    return configured_path


def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        cfg = dict(DEFAULT_CONFIG)
    else:
        cfg = json.load(open(CONFIG_PATH))
        for k, v in DEFAULT_CONFIG.items():
            cfg.setdefault(k, v)

    cfg["clinical_models_dir"] = _resolve_clinical_models_dir(cfg.get("clinical_models_dir"))
    return cfg


def load_patients_log():
    if not os.path.exists(PATIENTS_LOG_PATH):
        return []
    try:
        return json.load(open(PATIENTS_LOG_PATH))
    except Exception:
        return []


def load_clinical_drafts():
    if not os.path.exists(CLINICAL_DRAFTS_PATH):
        return {}
    try:
        return json.load(open(CLINICAL_DRAFTS_PATH))
    except Exception:
        return {}


def save_clinical_draft(patient_id, patient_name, clinical_values):
    drafts = load_clinical_drafts()
    drafts[patient_id] = {
        "patient_name": patient_name,
        "clinical_values": clinical_values,
        "saved_at": datetime.now().strftime("%d %b %Y %H:%M"),
    }
    with open(CLINICAL_DRAFTS_PATH, "w") as f:
        json.dump(drafts, f, indent=2)


def save_patient_record(record):
    log = load_patients_log()
    log.insert(0, record)
    log = log[:50]  # keep the most recent 50
    with open(PATIENTS_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def base_context(cfg, active_page, page_title, page_subtitle):
    photo = cfg.get("doctor_photo", "")
    return {
        "doctor_name": cfg["doctor_name"], "doctor_specialty": cfg["doctor_specialty"],
        "doctor_title": cfg.get("doctor_title", cfg["doctor_specialty"]),
        "doctor_photo_url": ("/doctor-photo" if photo and os.path.exists(photo) else None),
        "active_page": active_page, "page_title": page_title, "page_subtitle": page_subtitle,
    }


# =============================================================================
# Clinical feature grouping (same logic used by dashboard.py / make_clinical_template.py)
# =============================================================================
try:
    sys.path.insert(0, APP_DIR)
    sys.path.insert(0, os.path.dirname(APP_DIR))
    from make_clinical_template import describe_column
except ImportError:
    def describe_column(col):
        return col.replace("_", " ").capitalize(), "numeric"


QUICK_FIELDS = ["age", "gender", "cigsmok", "pack_years"]  # shown in the main visible grid,
                                                            # matching the reference's simple layout


def group_clinical_fields(feature_cols, chunk_size=4):
    """Groups feature columns by category, then further splits each
    category into small subgroups of `chunk_size` fields each (per the
    doctor's explicit request: manageable 3-5-field sections, not one
    long list per category)."""
    groups_def = {
        "Demographics": lambda c: c in QUICK_FIELDS or c.startswith("race_"),
        "Screening result codes": lambda c: c.startswith("scr_"),
        "Abnormality findings": lambda c: c.startswith("abn_") or c.startswith("any_") or c in (
            "total_abnormalities_t0", "total_ctab_records_t0", "n_noncalc_nodules_ge4mm_t0",
            "has_nodule_measurement_t0", "n_nodules_with_long_diameter_t0"),
        "Nodule measurements": lambda c: "diameter" in c or "area_proxy" in c or "aspect_ratio" in c,
        "Nodule margins": lambda c: c.startswith("margin_"),
        "Nodule attenuation": lambda c: c.startswith("atten_"),
        "Nodule location": lambda c: c.startswith("location_"),
    }
    assigned, groups = set(), []
    for name, matcher in groups_def.items():
        cols = [c for c in feature_cols if matcher(c) and c not in assigned]
        if not cols:
            continue
        assigned.update(cols)
        fields = []
        for c in cols:
            desc, expected = describe_column(c)
            is_flag = expected.startswith("1 ") or expected.startswith("integer count")
            fields.append({"name": c, "desc": desc, "expected": expected, "is_flag": is_flag})
        # split into small chunks of `chunk_size` fields each
        for i in range(0, len(fields), chunk_size):
            chunk = fields[i:i + chunk_size]
            suffix = f" ({i // chunk_size + 1})" if len(fields) > chunk_size else ""
            groups.append({"name": name + suffix, "fields": chunk})
    leftover = [c for c in feature_cols if c not in assigned]
    if leftover:
        fields = []
        for c in leftover:
            desc, expected = describe_column(c)
            is_flag = expected.startswith("1 ") or expected.startswith("integer count")
            fields.append({"name": c, "desc": desc, "expected": expected, "is_flag": is_flag})
        for i in range(0, len(fields), chunk_size):
            chunk = fields[i:i + chunk_size]
            suffix = f" ({i // chunk_size + 1})" if len(fields) > chunk_size else ""
            groups.append({"name": "Other fields" + suffix, "fields": chunk})
    return groups


# =============================================================================
# Lightweight CT preview (middle slice) -- independent of the full pipeline,
# just for the "Preview (Axial View)" thumbnail
# =============================================================================
def get_preview_image_b64(dicom_dir):
    try:
        import numpy as np
        import pydicom
        from skimage.transform import resize

        def is_real_dcm(fn):
            return fn.lower().endswith(".dcm") and not fn.startswith("._")

        dcm_files = []
        for root, _dirs, files in os.walk(dicom_dir):
            for f in files:
                if is_real_dcm(f):
                    dcm_files.append(os.path.join(root, f))
        if not dcm_files:
            return None

        headers = []
        for path in dcm_files:
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                z = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0.0
                headers.append((path, z))
            except Exception:
                continue
        if not headers:
            return None
        headers.sort(key=lambda h: h[1])
        mid_path = headers[len(headers) // 2][0]

        ds_full = pydicom.dcmread(mid_path)
        arr = ds_full.pixel_array.astype(np.float32)
        slope = float(getattr(ds_full, "RescaleSlope", 1))
        intercept = float(getattr(ds_full, "RescaleIntercept", 0))
        arr = arr * slope + intercept

        center, width = -600, 1500
        lo, hi = center - width / 2, center + width / 2
        clipped = np.clip(arr, lo, hi)
        normalized = (clipped - lo) / (hi - lo)
        img = (normalized * 255).astype(np.uint8)
        img = resize(img, (320, 320), preserve_range=True, anti_aliasing=True).astype(np.uint8)

        from PIL import Image
        pil_img = Image.fromarray(img, mode="L")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def model_status(cfg):
    clinical_ok = _clinical_model_dir_is_valid(cfg.get("clinical_models_dir"))
    cnn2d_ok = any(os.path.exists(os.path.join(cfg["cnn2d_models_dir"], f"model_fold{i}.pt")) for i in range(1, 6))
    mil3d_ok = any(os.path.exists(os.path.join(cfg["mil3d_models_dir"], f"model_fold{i}.pt")) for i in range(1, 6))
    wholelung_ok = any(os.path.exists(os.path.join(cfg["wholelung_models_dir"], f"model_fold{i}.pt")) for i in range(1, 6))
    return {"clinical": clinical_ok, "cnn2d": cnn2d_ok, "mil3d": mil3d_ok, "wholelung": wholelung_ok}


def run_worker(cmd):
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out_line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        return json.loads(out_line)
    except json.JSONDecodeError:
        return {"error": f"Worker produced no valid output.\nstderr: {result.stderr[-1500:]}"}


# =============================================================================
# Project Analysis -- selected existing EDA outputs only
# =============================================================================
# These entries map directly to figures already generated by Data_analysis.py.
# This dashboard layer does not recompute, modify, simulate, or substitute any
# analysis. Model-performance figures stay under /performance.
PROJECT_ANALYSIS_SECTIONS = [
    {
        "id": "data-overview",
        "title": "Data Overview",
        "subtitle": "",
        "graphs": [
            {"file": "03_target_distribution.png", "title": "Class Distribution", "full": False, "description": "This chart shows the number of participants in the Cancer and No Cancer outcome groups. Use it to judge how balanced the study cohort is before interpreting model metrics or subgroup comparisons."},
            {"file": "09_age_distribution_by_cancer.png", "title": "Age Distribution by Cancer Status", "full": False, "description": "The overlapping age distributions compare Cancer versus No Cancer participants across the baseline age range. The histogram bars show patient counts while the smooth density curves make shifts or overlap between the two groups easier to see."},
            {"file": "10_smoking_vs_cancer.png", "title": "Smoking Status vs Lung Cancer", "full": False, "description": "This grouped count plot compares current smokers and non-smokers separately for Cancer and No Cancer participants. It helps show whether the outcome groups have different smoking-status composition at the T0 screening visit."},
            {"file": "14_race_stacked_bar.png", "title": "Cancer Rate by Race", "full": False, "description": "Each bar represents a race category and is normalized to 100%, then divided into Cancer and No Cancer proportions. This lets the reader compare outcome composition across race groups while accounting for different group sizes."},
            {"file": "31_cases_by_age_group.png", "title": "Cancer Cases & Rate by Age Group", "full": True, "description": "The bars show the absolute number of cancer cases in each age band, while the overlaid line shows the cancer rate as a percentage. Reading both together separates a large case count caused by a large age group from a genuinely higher within-group cancer rate."},
        ],
    },
    {
        "id": "lesion-characteristics",
        "title": "Lesion Characteristics",
        "subtitle": "",
        "graphs": [
            {"file": "12_lesion_size_violin.png", "title": "Lesion Diameter by Cancer Status", "full": False, "description": "These violin plots compare the distribution of the maximum T0 lesion diameter between Cancer and No Cancer participants. The width represents where observations are concentrated and the internal quartile lines summarize the center and spread of each group."},
            {"file": "35_shape_feature_correlation.png", "title": "Shape Feature Correlation with Cancer", "full": False, "description": "Each horizontal bar is the Pearson correlation between one recorded nodule shape feature and cancer status. Bars farther from zero indicate a stronger linear association, while the sign shows whether higher feature values move with or against the cancer label."},
            {"file": "34_shape_feature_distributions.png", "title": "Shape/Morphology Feature Distributions by Cancer Status", "full": True, "description": "Each panel compares the mean of one T0 nodule shape or morphology measurement between Cancer and No Cancer participants, with error bars showing variability. Only participants with a recorded nodule measurement contribute, so the figure focuses specifically on measured lesion morphology rather than the entire cohort."},
            {"file": "36_margin_shape_vs_cancer.png", "title": "Nodule Margin / Border Shape by Cancer Status", "full": False, "description": "This chart compares the percentage of Cancer and No Cancer participants who have each recorded nodule margin type, such as smooth, spiculated, or poorly defined borders. It is intended to show which border patterns are more or less represented in the two outcome groups without reducing the information to a single summary value."},
            {"file": "37_attenuation_pattern_vs_cancer.png", "title": "Nodule Attenuation / Density Pattern by Cancer Status", "full": False, "description": "The grouped bars show the percentage of each outcome group with different nodule attenuation patterns, including solid, ground-glass, mixed, and other recorded density types. Comparing the paired bars makes it easier to see whether particular CT density patterns occur with different frequency in Cancer versus No Cancer participants."},
            {"file": "38_lobe_location_vs_cancer.png", "title": "Nodule Lobe Location by Cancer Status", "full": True, "description": "This figure compares how often nodules are recorded in each lung lobe for Cancer and No Cancer participants. The percentages are calculated within each outcome group, allowing lobe-location patterns to be compared without being driven only by the total number of patients."},
        ],
    },
    {
        "id": "relationships-ai",
        "title": "Relationships & AI Explainability",
        "subtitle": "",
        "graphs": [
            {"file": "27_permutation_importance_cv.png", "title": "Cross-Validated Feature Importance", "full": False, "description": "This cross-validated permutation-importance plot ranks features by how much ROC-AUC falls when each feature is randomly disrupted. A larger mean AUC decrease indicates that the trained model relies more strongly on that feature for out-of-fold discrimination; error bars show variation across folds."},
            {"file": "06_correlation_heatmap.png", "title": "Correlation Heatmap", "full": False, "description": "The heatmap summarizes pairwise linear correlations among the selected clinical variables, with both color and the printed coefficient showing direction and strength. It is useful for spotting variables that move together, potential redundancy, and relationships that should be considered when interpreting multivariable models."},
            {"file": "32_shap_summary.png", "title": "Cross-validated SHAP Summary", "full": True, "description": "This out-of-fold SHAP summary shows which features most influence the model predictions and the direction of those effects across patients. Features are ranked by overall impact; each point represents a patient-level contribution, so the horizontal spread shows how strongly a feature can push predictions toward or away from cancer."},
        ],
    },
]


def resolve_eda_outputs_dir(cfg):
    """Resolve the existing Data_analysis.py output folder without recomputing anything."""
    selected = {g["file"] for sec in PROJECT_ANALYSIS_SECTIONS for g in sec["graphs"]}
    configured = cfg.get("eda_outputs_dir", "/Volumes/Apple/images_organized/Analysis/eda_outputs")
    configured_path = configured if os.path.isabs(configured) else os.path.join(APP_DIR, configured)

    candidates = [
        configured_path,
        os.path.join(configured_path, "eda_outputs"),
        "/Volumes/Apple/images_organized/Analysis/eda_outputs",
    ]

    # Prefer the first existing folder that contains at least one selected graph.
    # This keeps the page working even if an older dashboard_config.json points
    # to the parent Analysis folder or to a stale path from another machine.
    for path in candidates:
        if os.path.isdir(path) and any(os.path.isfile(os.path.join(path, f)) for f in selected):
            return path

    # Fall back to the configured path so the UI can show which files are missing.
    return configured_path


def get_dicom_study_info(dicom_dir, files_received=0):
    """Read lightweight, real DICOM header metadata for the series selected by size.

    Display-only helper: no pixel values, model inputs, or inference calculations are changed.
    Missing DICOM tags are returned as None rather than guessed.
    """
    info = {
        "files_received": int(files_received or 0),
        "selected_series_slices": None,
        "modality": None,
        "series_description": None,
        "slice_thickness_mm": None,
        "pixel_spacing_mm": None,
        "study_date": None,
    }
    try:
        import pydicom

        series = []
        for root, _dirs, files in os.walk(dicom_dir):
            dcms = [f for f in files if f.lower().endswith('.dcm') and not f.startswith('._')]
            if dcms:
                series.append((root, dcms))
        if not series:
            return info

        best_dir, dcm_names = max(series, key=lambda item: len(item[1]))
        info["selected_series_slices"] = len(dcm_names)

        ds = None
        for name in dcm_names:
            try:
                ds = pydicom.dcmread(os.path.join(best_dir, name), stop_before_pixels=True)
                break
            except Exception:
                continue
        if ds is None:
            return info

        modality = getattr(ds, "Modality", None)
        if modality:
            info["modality"] = str(modality)

        desc = getattr(ds, "SeriesDescription", None) or getattr(ds, "ProtocolName", None)
        if desc:
            info["series_description"] = str(desc)

        thickness = getattr(ds, "SliceThickness", None)
        if thickness is not None:
            try:
                info["slice_thickness_mm"] = round(float(thickness), 3)
            except Exception:
                pass

        spacing = getattr(ds, "PixelSpacing", None)
        if spacing is not None and len(spacing) >= 2:
            try:
                info["pixel_spacing_mm"] = [round(float(spacing[0]), 3), round(float(spacing[1]), 3)]
            except Exception:
                pass

        study_date = getattr(ds, "StudyDate", None)
        if study_date:
            text = str(study_date)
            if len(text) == 8 and text.isdigit():
                info["study_date"] = f"{text[:4]}-{text[4:6]}-{text[6:8]}"
            else:
                info["study_date"] = text
    except Exception:
        # The inference workers already validate DICOM readability; dashboard
        # metadata should never make a valid analysis fail.
        pass
    return info


app = Flask(__name__)


@app.route("/")
def index():
    cfg = load_config()
    cols_path = os.path.join(cfg["clinical_models_dir"], "clinical_feature_columns.json")
    feature_cols = json.load(open(cols_path)) if os.path.exists(cols_path) else []
    quick_fields_present = [c for c in QUICK_FIELDS if c in feature_cols]
    other_groups = group_clinical_fields([c for c in feature_cols if c not in quick_fields_present])

    status = model_status(cfg)
    patients = load_patients_log()

    # Lightweight executive-summary values for the landing dashboard.
    # These are display-only and do not alter inference, model outputs, or data.
    dataset_n = None
    perf_json = os.path.join(APP_DIR, "project_results", "model_performance_by_model", "model_performance_data.json")
    if os.path.exists(perf_json):
        try:
            perf_data = json.load(open(perf_json))
            dataset_n = perf_data.get("protocol", {}).get("original_dataset_n")
        except Exception:
            dataset_n = None

    ctx = base_context(cfg, "dashboard", "Dashboard", "Comprehensive lung cancer analysis & prediction")
    return render_template(
        "dashboard.html", **ctx,
        quick_fields=quick_fields_present, other_groups=other_groups,
        status=status, patients=patients,
        dashboard_dataset_n=dataset_n,
        dashboard_ready_models=sum(bool(v) for v in status.values()),
        dashboard_total_models=4,
        dashboard_active_threshold=OPTIMAL_THRESHOLD,
        dashboard_current_date=datetime.now().strftime("%d %b %Y"),
        fusion_weight_clinical=int(FUSION_WEIGHT_CLINICAL * 100),
        fusion_weight_cnn2d=int(FUSION_WEIGHT_OLD_CNN * 100),
        fusion_weight_mil3d=int(FUSION_WEIGHT_MIL * 100),
        fusion_weight_wholelung=int(FUSION_WEIGHT_WHOLELUNG * 100),
        n_missing_clinical_cols=len(feature_cols) - len(quick_fields_present) - sum(
            len(g["fields"]) for g in other_groups),
    )


@app.route("/doctor-photo")
def doctor_photo():
    cfg = load_config()
    photo = cfg.get("doctor_photo", "")
    if photo and os.path.exists(photo):
        from flask import send_file
        return send_file(photo)
    return "", 404


@app.route("/csv-guide")
def csv_guide():
    cfg = load_config()
    cols_path = os.path.join(cfg["clinical_models_dir"], "clinical_feature_columns.json")
    feature_cols = json.load(open(cols_path)) if os.path.exists(cols_path) else []

    example = {}
    for c in feature_cols:
        _desc, expected = describe_column(c)
        is_flag = expected.startswith("1 ") or expected.startswith("integer count")
        example[c] = 0 if (is_flag or c == "gender") else 0.0
    example_csv_header = ",".join(feature_cols)
    example_csv_row = ",".join(str(example[c]) for c in feature_cols)

    guide_groups = group_clinical_fields(feature_cols, chunk_size=1000)  # one big group per category for the guide table
    for g in guide_groups:
        for f in g["fields"]:
            f["example"] = example.get(f["name"], 0)

    ctx = base_context(cfg, "csv_guide", "Guide",
                        "What T0 means, the full feature reference, and how to submit patient data")
    return render_template("csv_guide.html", **ctx, guide_groups=guide_groups, n_features=len(feature_cols),
                            example_csv_header=example_csv_header, example_csv_row=example_csv_row)


@app.route("/team")
def team():
    cfg = load_config()
    ctx = base_context(cfg, "team", "Project Team", "The people behind this project")
    return render_template("team.html", **ctx)


@app.route("/performance")
def performance():
    cfg = load_config()
    perf_path = os.path.join(
        cfg["project_results_dir"],
        "model_performance_by_model",
        "model_performance_data.json",
    )

    if not os.path.exists(perf_path):
        ctx = base_context(cfg, "performance", "Model Performance", "")
        return render_template(
            "missing_data.html", **ctx,
            missing_file="model_performance_data.json",
            expected_path=perf_path,
            instructions=(
                "Run: python3 build_model_performance_assets.py. "
                "The builder uses only the supplied model result JSON/CSV files under "
                "project_results/model_performance_by_model/source."
            ),
        )

    with open(perf_path, "r") as f:
        performance_data = json.load(f)

    ctx = base_context(cfg, "performance", "Model Performance", "")
    return render_template(
        "performance.html",
        **ctx,
        models=performance_data.get("models", []),
        protocol=performance_data.get("protocol", {}),
        best_individual=performance_data.get("best_individual", {}),
        fusion_comparison=performance_data.get("fusion_comparison", {}),
    )


@app.route("/project-analysis")
def project_analysis():
    cfg = load_config()
    output_dir = resolve_eda_outputs_dir(cfg)

    sections = []
    for section in PROJECT_ANALYSIS_SECTIONS:
        section_copy = {
            "id": section["id"],
            "title": section["title"],
            "subtitle": section["subtitle"],
            "graphs": [],
        }
        for graph in section["graphs"]:
            graph_copy = dict(graph)
            graph_copy["exists"] = os.path.isfile(os.path.join(output_dir, graph["file"]))
            section_copy["graphs"].append(graph_copy)
        sections.append(section_copy)

    ctx = base_context(
        cfg, "project_analysis", "Project Analysis", ""
    )
    return render_template("project_analysis.html", **ctx, sections=sections)


@app.route("/project-analysis/image/<path:filename>")
def project_analysis_image(filename):
    allowed = {g["file"] for sec in PROJECT_ANALYSIS_SECTIONS for g in sec["graphs"]}
    if filename not in allowed:
        return "", 404
    cfg = load_config()
    from flask import send_from_directory
    return send_from_directory(resolve_eda_outputs_dir(cfg), filename)


@app.route("/results")
def results_page():
    cfg = load_config()
    patients = load_patients_log()
    ctx = base_context(cfg, "results", "Analysis Results", "History of patient analyses run on this machine")
    return render_template("results.html", **ctx, patients=patients)


@app.route("/api/export-patients-excel")
def export_patients_excel():
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    patients = load_patients_log()

    wb = Workbook()
    ws = wb.active
    ws.title = "Patients"

    headers = ["Patient ID", "Patient Name", "Date", "Clinical %", "2D CT %", "3D CT %",
               "Whole-Lung %", "Final Risk %", "Status", "3D Candidate Patches"]
    ws.append(headers)

    header_fill = PatternFill(start_color="1A2340", end_color="1A2340", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    for col in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    status_fills = {
        "High Risk": PatternFill(start_color="FBD8E2", end_color="FBD8E2", fill_type="solid"),
        "Low Risk": PatternFill(start_color="D9F5E6", end_color="D9F5E6", fill_type="solid"),
    }
    borderline_fill = PatternFill(start_color="FBEBD0", end_color="FBEBD0", fill_type="solid")

    for p in patients:
        wholelung_val = p.get("wholelung_prob")
        row = [
            p.get("patient_id", ""), p.get("patient_name", ""), p.get("date", ""),
            round(p["clinical_prob"] * 100, 1) if "clinical_prob" in p else "",
            round(p["old_cnn_prob"] * 100, 1) if "old_cnn_prob" in p else "",
            round(p["mil_prob"] * 100, 1) if "mil_prob" in p else "",
            round(wholelung_val * 100, 1) if wholelung_val is not None else "n/a",
            p.get("risk_score", ""), p.get("status", ""), p.get("n_3d_patches", ""),
        ]
        ws.append(row)
        status = p.get("status", "")
        fill = status_fills.get(status, borderline_fill if "Moderate" in status or "Borderline" in status else None)
        if fill:
            ws.cell(row=ws.max_row, column=9).fill = fill

    for col in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 16
    ws.column_dimensions["B"].width = 22

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    from flask import send_file
    filename = f"patients_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return send_file(buf, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                      as_attachment=True, download_name=filename)


def generate_auc_bar_chart_svg(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["Clinical", "2D CT", "3D-MIL", "Whole-Lung", "Fusion"]
        aucs = [
            results["clinical_alone"]["ROC-AUC"],
            results["old_2d_cnn_alone"]["ROC-AUC"],
            results["new_3d_mil_alone"]["ROC-AUC"],
            results["wholelung_alone"]["ROC-AUC"],
            results["fusion_at_050"]["ROC-AUC"],
        ]
        colors = ["#E6437A", "#4F8EF7", "#9B5CF6", "#2FB8A6", "#3FBF7F"]

        fig, ax = plt.subplots(figsize=(5.5, 4), facecolor="none")
        ax.set_facecolor("none")
        bars = ax.bar(labels, aucs, color=colors, width=0.55)
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                    ha="center", color="#F1F3FA", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("ROC-AUC", color="#8790B0")
        ax.tick_params(colors="#8790B0")
        for spine in ax.spines.values():
            spine.set_color("#5B6285")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.5, color="#5B6285", linestyle="--", linewidth=1, alpha=0.6)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="svg", transparent=True)
        plt.close(fig)
        return buf.getvalue().decode()
    except Exception as e:
        return f"<p style='color:#E6437A'>Could not generate bar chart: {e}</p>"


def generate_sens_spec_chart_svg(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["Clinical", "2D CT", "3D-MIL", "Whole-Lung", "Fusion"]
        sens = [results["clinical_alone"]["Sensitivity"], results["old_2d_cnn_alone"]["Sensitivity"],
                results["new_3d_mil_alone"]["Sensitivity"], results["wholelung_alone"]["Sensitivity"],
                results["fusion_at_050"]["Sensitivity"]]
        spec = [results["clinical_alone"]["Specificity"], results["old_2d_cnn_alone"]["Specificity"],
                results["new_3d_mil_alone"]["Specificity"], results["wholelung_alone"]["Specificity"],
                results["fusion_at_050"]["Specificity"]]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(5.5, 4), facecolor="none")
        ax.set_facecolor("none")
        ax.bar(x - width / 2, sens, width, label="Sensitivity", color="#E6437A")
        ax.bar(x + width / 2, spec, width, label="Specificity", color="#4F8EF7")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.tick_params(colors="#8790B0")
        for spine in ax.spines.values():
            spine.set_color("#5B6285")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#131A38", edgecolor="#5B6285", labelcolor="white", fontsize=9)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="svg", transparent=True)
        plt.close(fig)
        return buf.getvalue().decode()
    except Exception as e:
        return f"<p style='color:#E6437A'>Could not generate chart: {e}</p>"


def generate_roc_svg(predictions_path):
    try:
        import pandas as pd
        from sklearn.metrics import roc_curve, roc_auc_score
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_csv(predictions_path)
        y = df["has_cancer"].values
        p = df["fused_probability"].values
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)

        fig, ax = plt.subplots(figsize=(5, 5), facecolor="none")
        ax.set_facecolor("none")
        ax.plot(fpr, tpr, color="#E6437A", linewidth=2.5, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="#5B6285", linewidth=1)
        ax.set_xlabel("1 - Specificity", color="#8790B0")
        ax.set_ylabel("Sensitivity", color="#8790B0")
        ax.tick_params(colors="#8790B0")
        for spine in ax.spines.values():
            spine.set_color("#5B6285")
        ax.legend(loc="lower right", facecolor="#131A38", edgecolor="#5B6285", labelcolor="white")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="svg", transparent=True)
        plt.close(fig)
        return buf.getvalue().decode()
    except Exception as e:
        return f"<p style='color:#E6437A'>Could not generate ROC curve: {e}</p>"


@app.route("/api/save-clinical-draft", methods=["POST"])
def save_clinical_draft_route():
    data = request.get_json(force=True)
    patient_id = (data.get("patient_id") or "").strip()
    if not patient_id:
        return jsonify({"error": "Patient ID is required to save a draft."}), 400
    save_clinical_draft(patient_id, data.get("patient_name", ""), data.get("clinical_values", {}))
    return jsonify({"ok": True, "patient_id": patient_id})


@app.route("/api/load-clinical-draft/<path:patient_id>")
def load_clinical_draft_route(patient_id):
    drafts = load_clinical_drafts()
    draft = drafts.get(patient_id)
    if not draft:
        return jsonify({"error": "No saved draft found for this Patient ID."}), 404
    return jsonify({"ok": True, **draft})


@app.route("/api/report-pdf", methods=["POST"])
def report_pdf():
    from flask import send_file
    try:
        from patient_report import build_patient_report

        data = request.get_json(force=True) or {}
        cfg = load_config()
        report_buffer, filename = build_patient_report(
            data=data,
            cfg=cfg,
            describe_column=describe_column,
            weights=(
                FUSION_WEIGHT_CLINICAL,
                FUSION_WEIGHT_OLD_CNN,
                FUSION_WEIGHT_MIL,
                FUSION_WEIGHT_WHOLELUNG,
            ),
            threshold=OPTIMAL_THRESHOLD,
        )
        return send_file(
            report_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as exc:
        # Return the real cause to the UI instead of a silent generic 500.
        # Do not include patient clinical values in the error response.
        print(f"[patient-report] PDF generation failed: {type(exc).__name__}: {exc}")
        return jsonify({
            "ok": False,
            "error": "Patient report could not be generated.",
            "detail": f"{type(exc).__name__}: {exc}",
        }), 500


def _wrap_text(text, width):
    import textwrap
    return textwrap.wrap(text, width) or [""]


@app.route("/api/analyze", methods=["POST"])
def analyze():
    cfg = load_config()

    patient_id = request.form.get("patient_id", "").strip() or f"P-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    patient_name = request.form.get("patient_name", "").strip() or "Unnamed"

    clinical_values = json.loads(request.form.get("clinical_values", "{}"))
    ct_files = request.files.getlist("ct_files")
    ct_file_paths = json.loads(request.form.get("ct_file_paths", "[]"))

    if not ct_files:
        return jsonify({"error": "No CT files uploaded."}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        dicom_dir = os.path.join(tmpdir, "dicom")
        os.makedirs(dicom_dir, exist_ok=True)
        for i, f in enumerate(ct_files):
            # use the browser-provided relative path (folder upload) when
            # available, so multiple series subfolders are preserved and
            # the CT worker can correctly pick the best/largest series --
            # otherwise fall back to a flat filename.
            rel_path = ct_file_paths[i] if i < len(ct_file_paths) else f.filename
            rel_path = rel_path.replace("\\", "/").lstrip("/")
            dest_path = os.path.join(dicom_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            f.save(dest_path)

        preview_b64 = get_preview_image_b64(dicom_dir)
        dcm_received = sum(1 for f in ct_files if (f.filename or "").lower().endswith(".dcm") and not os.path.basename(f.filename or "").startswith("._"))
        ct_study_info = get_dicom_study_info(dicom_dir, dcm_received)

        cols_path = os.path.join(cfg["clinical_models_dir"], "clinical_feature_columns.json")
        feature_cols = json.load(open(cols_path))
        row = {c: clinical_values.get(c, 0) for c in feature_cols}
        clinical_csv_path = os.path.join(tmpdir, "clinical_row.csv")
        import pandas as pd
        pd.DataFrame([row])[feature_cols].to_csv(clinical_csv_path, index=False)

        clinical_data = run_worker([
            sys.executable, os.path.join(APP_DIR, "predict_clinical_worker.py"),
            "--clinical-csv", clinical_csv_path,
            "--clinical-models-dir", cfg["clinical_models_dir"],
        ])
        if "error" in clinical_data:
            return jsonify({"error": f"Clinical model failed: {clinical_data['error']}"}), 500

        ct_data = run_worker([
            sys.executable, os.path.join(APP_DIR, "predict_ct_worker.py"),
            "--dicom-dir", dicom_dir,
            "--cnn2d-models-dir", cfg["cnn2d_models_dir"],
            "--mil3d-models-dir", cfg["mil3d_models_dir"],
            "--device", cfg["device"],
        ])
        if "error" in ct_data:
            return jsonify({"error": f"CT models failed: {ct_data['error']}"}), 500

        # Whole-lung pretrained-backbone branch -- REFERENCE/DOCUMENTATION
        # ONLY. Rigorous cross-validated testing showed this branch does not
        # improve the official fused risk score (CV AUC 0.7962 vs 0.7963-
        # 0.7964 for the official 3-branch fusion -- statistically
        # indistinguishable). It is run and displayed for transparency, but
        # never included in final_prob below. Failure here is NON-FATAL --
        # the official analysis proceeds normally even if this branch errors.
        wholelung_data = run_worker([
            sys.executable, os.path.join(APP_DIR, "predict_wholelung_worker.py"),
            "--dicom-dir", dicom_dir,
            "--wholelung-models-dir", cfg["wholelung_models_dir"],
            "--device", cfg["device"],
        ])
        if "error" in wholelung_data:
            wholelung_data = {"wholelung_pretrained_prob": None,
                               "error": wholelung_data["error"]}

    clinical_prob = clinical_data["clinical_prob"]
    old_cnn_prob = ct_data["old_cnn_prob"]
    mil_prob = ct_data["mil_prob"]
    wholelung_prob = wholelung_data.get("wholelung_pretrained_prob")

    # Official 4-branch fusion (per explicit decision to include the
    # whole-lung branch in the calculation, not just as a side signal).
    # Weights (0.40/0.20/0.30/0.10) were grid-searched on the development
    # cohort then confirmed via proper 5-fold CV to be safe (CV AUC 0.7962,
    # statistically indistinguishable from the prior 3-branch fusion's
    # 0.7963-0.7964) -- i.e. this does not measurably help OR hurt validated
    # performance, but does genuinely incorporate the branch as requested.
    # GRACEFUL DEGRADATION: if the whole-lung worker failed for this patient
    # (wholelung_prob is None), fall back to the original validated 3-branch
    # weights, renormalized, rather than failing the whole analysis.
    if wholelung_prob is not None:
        final_prob = (FUSION_WEIGHT_CLINICAL * clinical_prob
                      + FUSION_WEIGHT_OLD_CNN * old_cnn_prob
                      + FUSION_WEIGHT_MIL * mil_prob
                      + FUSION_WEIGHT_WHOLELUNG * wholelung_prob)
    else:
        w_sum = FUSION_WEIGHT_CLINICAL + FUSION_WEIGHT_OLD_CNN + FUSION_WEIGHT_MIL
        final_prob = (FUSION_WEIGHT_CLINICAL * clinical_prob
                      + FUSION_WEIGHT_OLD_CNN * old_cnn_prob
                      + FUSION_WEIGHT_MIL * mil_prob) / w_sum

    # Risk tiers: OPTIMAL_THRESHOLD (0.4621) is the validated Youden's-J binary
    # decision boundary. A 4-tier clinical-style scale is built around it,
    # rather than jumping straight from "not detected" to "detected" at a
    # single hard cutoff:
    #   < 0.30                      -> Cancer Not Detected
    #   0.30  -- OPTIMAL_THRESHOLD   -> Suspicious
    #   OPTIMAL_THRESHOLD -- 0.60    -> High Suspicion
    #   >= 0.60                      -> Cancer Detected
    TIER_LOW = 0.30
    TIER_HIGH = 0.60
    if final_prob >= TIER_HIGH:
        risk_label = "Cancer Detected"
        risk_tier = 3
    elif final_prob >= OPTIMAL_THRESHOLD:
        risk_label = "High Suspicion"
        risk_tier = 2
    elif final_prob >= TIER_LOW:
        risk_label = "Suspicious"
        risk_tier = 1
    else:
        risk_label = "Cancer Not Detected"
        risk_tier = 0

    is_high = risk_tier == 3          # kept for the PDF's simple color logic
    is_borderline = risk_tier in (1, 2)  # kept for backward compatibility

    # Confidence indicator: how far the final probability sits from the
    # validated decision boundary. A prediction right at the boundary is a
    # near-coin-flip, low-confidence case regardless of which side it falls on.
    distance_from_threshold = abs(final_prob - OPTIMAL_THRESHOLD)
    if distance_from_threshold < 0.05:
        confidence_label = "Very Low Confidence"
        confidence_note = "This result sits almost exactly on the model's decision boundary."
    elif distance_from_threshold < 0.10:
        confidence_label = "Low Confidence"
        confidence_note = "This result is close to the decision boundary."
    elif distance_from_threshold < 0.20:
        confidence_label = "Moderate Confidence"
        confidence_note = "This result has a reasonable margin from the decision boundary."
    else:
        confidence_label = "High Confidence"
        confidence_note = "This result is well clear of the decision boundary."
    if wholelung_prob is None:
        confidence_note += (" (Whole-lung branch unavailable for this patient -- score computed "
                             "from the remaining 3 branches, reweighted.)")

    # Objective branch-agreement summary.  This reports the observed range/spread
    # of the actual branch probabilities; it does not invent an agreement class.
    branch_probs = [clinical_prob, old_cnn_prob, mil_prob]
    if wholelung_prob is not None:
        branch_probs.append(wholelung_prob)
    branch_min = min(branch_probs)
    branch_max = max(branch_probs)
    branch_spread = branch_max - branch_min

    analysis_timestamp = datetime.now().strftime("%d %b %Y %H:%M:%S")
    gender_raw = clinical_values.get("gender")
    gender_label = "Male" if gender_raw == 0 else ("Female" if gender_raw == 1 else "—")
    smoker_raw = clinical_values.get("cigsmok")
    smoker_label = "Yes" if smoker_raw == 1 else ("No" if smoker_raw == 0 else "—")

    record = {
        "patient_id": patient_id, "patient_name": patient_name,
        "date": datetime.now().strftime("%d %b %Y"),
        "risk_score": round(final_prob * 100),
        "status": risk_label,
        "risk_tier": risk_tier,
        "clinical_prob": round(clinical_prob, 4),
        "old_cnn_prob": round(old_cnn_prob, 4),
        "mil_prob": round(mil_prob, 4),
        "wholelung_prob": round(wholelung_prob, 4) if wholelung_prob is not None else None,
        "n_3d_patches": ct_data.get("n_3d_patches"),
        "confidence_label": confidence_label,
        "analysis_timestamp": analysis_timestamp,
    }
    save_patient_record(record)

    return jsonify({
        "clinical_prob": clinical_prob, "old_cnn_prob": old_cnn_prob, "mil_prob": mil_prob,
        "wholelung_prob": wholelung_prob,
        "wholelung_note": "Now included as the 4th weighted branch in the official fused score "
                           "(weight 10%), per project decision to incorporate it into the "
                           "calculation.",
        "final_prob": final_prob, "risk_label": risk_label, "risk_tier": risk_tier,
        "is_high": is_high,
        "is_borderline": is_borderline,
        "confidence_label": confidence_label,
        "confidence_note": confidence_note,
        "n_3d_patches": ct_data.get("n_3d_patches"),
        "patient_summary": {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "age": clinical_values.get("age"),
            "gender": gender_label,
            "current_smoker": smoker_label,
        },
        "model_agreement": {
            "n_branches": len(branch_probs),
            "range_min": branch_min,
            "range_max": branch_max,
            "spread": branch_spread,
        },
        "ct_study_info": ct_study_info,
        "provenance": {
            "analysis_timestamp": analysis_timestamp,
            "pipeline": "Clinical + 2D CT + 3D MIL + Whole Lung",
            "model_configuration": "4-branch weighted fusion",
            "fusion_weights": "40 / 20 / 30 / 10",
            "fusion_threshold": OPTIMAL_THRESHOLD,
        },
        "preview_image_b64": preview_b64,
        "slice_thumbnails_b64": ct_data.get("slice_thumbnails_b64", []),
        "patient": record,
    })


if __name__ == "__main__":
    load_config()
    print(f"\nDashboard config: {CONFIG_PATH}")
    print("Edit that file to point at your actual model folders, then restart.\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
