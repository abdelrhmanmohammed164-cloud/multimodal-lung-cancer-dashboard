#!/usr/bin/env python3
"""
app.py -- Live clinical-branch demo of the lung cancer risk dashboard
======================================================================
This is a stripped-down, publicly-deployable companion to the full
dashboard in the parent repository. It runs ONLY the clinical branch
(CatBoost + XGBoost + Logistic Regression ensemble) of the four-branch
multimodal pipeline described in the manuscript. The 2D CT, 3D-MIL, and
whole-lung branches all need imaging models that are far too large to
host on a free tier, so they are not part of this demo.

The score this page returns is the clinical-branch probability alone,
NOT the validated 4-branch fused score reported in the paper. It exists
to let a reviewer try the clinical model interactively without cloning
the repository or installing anything.
"""

import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from flask import Flask, render_template, request, jsonify
from xgboost import XGBClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "clinical_models_for_inference")

QUICK_FIELDS = ["age", "gender", "cigsmok", "pack_years"]


# -----------------------------------------------------------------------
# Field descriptions (condensed from the full project's
# make_clinical_template.py, kept local so this demo has no dependency
# on the rest of the dashboard codebase)
# -----------------------------------------------------------------------
LOCATION_NAMES = {
    "right_upper_lobe": "right upper lobe", "right_middle_lobe": "right middle lobe",
    "right_lower_lobe": "right lower lobe", "left_upper_lobe": "left upper lobe",
    "lingula": "lingula", "left_lower_lobe": "left lower lobe",
    "other_location": "another/unspecified location",
}
MARGIN_NAMES = {
    "spiculated": "spiculated (spiky) margins", "smooth": "smooth margins",
    "poorly_defined": "poorly-defined margins", "margin_undetermined": "margin could not be determined",
}
ATTEN_NAMES = {
    "soft_tissue": "soft-tissue attenuation", "ground_glass": "ground-glass attenuation",
    "mixed_attenuation": "mixed (part-solid) attenuation", "fluid_water": "fluid/water attenuation",
    "fat_attenuation": "fat attenuation", "other_attenuation": "another attenuation type",
    "attenuation_undetermined": "attenuation could not be determined",
}
ABN_NAMES = {
    "noncalc_nodule_ge4mm": "non-calcified nodule(s) ≥4mm",
    "noncalc_micronodule_lt4mm": "non-calcified micronodule(s) <4mm",
    "benign_calcified_nodule": "benign-appearing calcified nodule(s)",
    "atelectasis": "atelectasis (partial lung collapse)",
    "pleural_thickening_effusion": "pleural thickening or effusion",
    "hilar_mediastinal_adenopathy_mass": "hilar/mediastinal adenopathy or mass",
    "chest_wall_abnormality": "chest wall abnormality",
    "consolidation": "consolidation",
    "emphysema": "emphysema",
    "cardiovascular_abnormality": "cardiovascular abnormality",
    "fibrosis_scar_reticular_opacity": "fibrosis, scarring, or reticular opacity",
    "sixplus_nodules_not_suspicious": "6+ small non-suspicious nodules",
    "other_significant_above_diaphragm": "other significant finding above the diaphragm",
    "other_significant_below_diaphragm": "other significant finding below the diaphragm",
    "other_minor_abnormality": "other minor abnormality",
}


def describe_column(col):
    if col == "age":
        return "Patient's age at T0 screening", "integer (years)"
    if col == "gender":
        return "Patient's sex", "0 = Male, 1 = Female"
    if col == "cigsmok":
        return "Current smoker flag at T0", "1 = currently smoking, 0 = not currently smoking"
    if col == "pack_years":
        return "Smoking history", "numeric (pack-years)"
    if col.startswith("race_"):
        suffix = col.replace("race_", "")
        return f"Race category flag ('{suffix}')", "1 if this category applies, else 0"
    if col == "scr_days0":
        return "Days from enrollment to the T0 screening visit", "integer (days)"
    if re.match(r"scr_res0_\d+$", col):
        code = col.split("_")[-1]
        return f"T0 screening result code = {code}", "1 if this code applies, else 0"
    if re.match(r"scr_iso0_\d+$", col):
        code = col.split("_")[-1]
        return f"T0 screening isolated-finding code = {code}", "1 if this code applies, else 0"
    if col.startswith("abn_") and (col.endswith("_count_t0") or col.endswith("_present_t0")):
        core = col[len("abn_"):-len("_count_t0")] if col.endswith("_count_t0") else col[len("abn_"):-len("_present_t0")]
        label = ABN_NAMES.get(core, core.replace("_", " "))
        if col.endswith("_count_t0"):
            return f"Number of findings of: {label}", "integer count (0 if none)"
        return f"Whether the patient has any: {label}", "1 = present, 0 = absent"
    if col.startswith("margin_") and (col.endswith("_count_t0") or col.endswith("_present_t0")):
        core = col[len("margin_"):-len("_count_t0")] if col.endswith("_count_t0") else col[len("margin_"):-len("_present_t0")]
        label = MARGIN_NAMES.get(core, core.replace("_", " "))
        if col.endswith("_count_t0"):
            return f"Number of nodules with {label}", "integer count (0 if none)"
        return f"Whether any nodule has {label}", "1 = yes, 0 = no"
    if col.startswith("atten_") and (col.endswith("_count_t0") or col.endswith("_present_t0")):
        core = col[len("atten_"):-len("_count_t0")] if col.endswith("_count_t0") else col[len("atten_"):-len("_present_t0")]
        label = ATTEN_NAMES.get(core, core.replace("_", " "))
        if col.endswith("_count_t0"):
            return f"Number of nodules with {label}", "integer count (0 if none)"
        return f"Whether any nodule has {label}", "1 = yes, 0 = no"
    if col.startswith("location_") and (col.endswith("_count_t0") or col.endswith("_present_t0")):
        core = col[len("location_"):-len("_count_t0")] if col.endswith("_count_t0") else col[len("location_"):-len("_present_t0")]
        label = LOCATION_NAMES.get(core, core.replace("_", " "))
        if col.endswith("_count_t0"):
            return f"Number of nodules located in the {label}", "integer count (0 if none)"
        return f"Whether any nodule is located in the {label}", "1 = yes, 0 = no"
    if col.startswith("any_"):
        rest = col[len("any_"):].replace("_t0", "")
        rest = re.sub(r"ge(\d+)mm", r">=\1mm", rest)
        rest = re.sub(r"le(\d+)mm", r"<=\1mm", rest)
        rest = rest.replace("_", " ")
        return f"Whether the patient has any: {rest}", "1 = yes, 0 = no"
    if col == "n_noncalc_nodules_ge4mm_t0":
        return "Total number of non-calcified nodules ≥4mm", "integer count"
    if col == "n_nodules_with_long_diameter_t0":
        return "Number of nodules with a recorded long-axis diameter", "integer count"
    if col == "has_nodule_measurement_t0":
        return "Whether at least one nodule has a recorded measurement", "1 = yes, 0 = no"
    if col == "total_abnormalities_t0":
        return "Total count of all recorded abnormalities at T0", "integer count"
    if col == "total_ctab_records_t0":
        return "Total number of CT abnormality-table records at T0", "integer count"
    m = re.match(r"(max|mean|median|min)_long_diameter_mm_t0", col)
    if m:
        return f"{m.group(1).capitalize()} long-axis nodule diameter", "numeric (millimeters)"
    m = re.match(r"(max|mean|median|min)_perp_diameter_mm_t0", col)
    if m:
        return f"{m.group(1).capitalize()} perpendicular nodule diameter", "numeric (millimeters)"
    if col in ("max_nodule_area_proxy_t0", "mean_nodule_area_proxy_t0"):
        return f"{col.split('_')[0].capitalize()} nodule area (proxy measure)", "numeric"
    if col in ("max_nodule_aspect_ratio_t0", "mean_nodule_aspect_ratio_t0"):
        return f"{col.split('_')[0].capitalize()} nodule aspect ratio (long/perp diameter)", "numeric ratio"
    return "Field from the NLST screening record", "numeric"


def group_clinical_fields(feature_cols, chunk_size=20):
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


FEATURE_COLS = json.load(open(os.path.join(MODELS_DIR, "clinical_feature_columns.json")))
WEIGHTS = json.load(open(os.path.join(MODELS_DIR, "clinical_weights.json")))
IMPUTER = joblib.load(os.path.join(MODELS_DIR, "clinical_imputer.pkl"))
LOGREG = joblib.load(os.path.join(MODELS_DIR, "clinical_logreg.pkl"))
CATBOOST = CatBoostClassifier()
CATBOOST.load_model(os.path.join(MODELS_DIR, "clinical_catboost.cbm"))
XGB = XGBClassifier()
XGB.load_model(os.path.join(MODELS_DIR, "clinical_xgboost.json"))

app = Flask(__name__)


@app.route("/")
def index():
    quick_fields_present = [c for c in QUICK_FIELDS if c in FEATURE_COLS]
    other_groups = group_clinical_fields([c for c in FEATURE_COLS if c not in quick_fields_present])
    quick_field_defs = []
    for c in quick_fields_present:
        desc, expected = describe_column(c)
        quick_field_defs.append({"name": c, "desc": desc, "expected": expected})
    return render_template("index.html", quick_fields=quick_field_defs, other_groups=other_groups)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}
        row = {c: data.get(c, 0) for c in FEATURE_COLS}
        X = pd.DataFrame([row])[FEATURE_COLS].values
        X_imputed = IMPUTER.transform(X)

        p_cat = float(CATBOOST.predict_proba(X_imputed)[:, 1][0])
        p_xgb = float(XGB.predict_proba(X_imputed)[:, 1][0])
        p_lr = float(LOGREG.predict_proba(X_imputed)[:, 1][0])

        total_w = WEIGHTS["cat"] + WEIGHTS["xgb"] + WEIGHTS["lr"]
        clinical_prob = (WEIGHTS["cat"] * p_cat + WEIGHTS["xgb"] * p_xgb + WEIGHTS["lr"] * p_lr) / total_w

        if clinical_prob >= 0.60:
            label = "Higher risk range"
        elif clinical_prob >= 0.30:
            label = "Intermediate risk range"
        else:
            label = "Lower risk range"

        return jsonify({
            "clinical_prob": clinical_prob,
            "p_cat": p_cat, "p_xgb": p_xgb, "p_lr": p_lr,
            "label": label,
        })
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
