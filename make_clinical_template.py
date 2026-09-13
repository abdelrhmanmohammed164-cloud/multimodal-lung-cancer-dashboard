#!/usr/bin/env python3
"""
make_clinical_template.py  (v2 -- with field guide)
================================================================
Generates TWO files for a new patient:

  1. patient_clinical_row.csv       -- the fillable template (same as
     before: exact required column names/order, count/present columns
     pre-filled with 0)
  2. patient_clinical_field_guide.csv -- a plain-English guide, ONE ROW
     PER COLUMN, explaining what each column means and what kind of
     value belongs there (flag, count, measurement, or a coded
     category), inferred from the column's naming pattern.

IMPORTANT: descriptions for "coded category" columns (scr_res0_*,
scr_iso0_*) are inferred from naming convention only -- their exact
numeric code meanings come from NLST's own data dictionary, which
should be checked against the source codebook, not guessed here.

RUN:
    python3 make_clinical_template.py \
        --clinical-models-dir clinical_models_for_inference \
        --output patient_clinical_row.csv \
        --guide-output patient_clinical_field_guide.csv
"""

import argparse
import json
import os
import re
import sys

try:
    import pandas as pd
except ImportError:
    print("Missing pandas. Install with: pip install pandas")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clinical-models-dir", default="clinical_models_for_inference")
    p.add_argument("--output", default="patient_clinical_row.csv")
    p.add_argument("--guide-output", default="patient_clinical_field_guide.csv")
    return p.parse_args()


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
    "noncalc_nodule_ge4mm": "non-calcified nodule(s) >=4mm",
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
    """Returns (description, expected_value) inferred from the column
    name's pattern. Falls back to a generic note if no pattern matches."""
    if col == "age":
        return "Patient's age at T0 screening", "integer (years)"
    if col == "gender":
        return "Patient's sex", "0 = Male, 1 = Female (current model-ready dataset coding)"
    if col == "cigsmok":
        return "Current smoker flag at T0", "1 = currently smoking, 0 = not currently smoking"
    if col == "pack_years":
        return "Smoking history", "numeric (pack-years)"
    if col.startswith("race_"):
        suffix = col.replace("race_", "")
        return f"One-hot race category flag ('{suffix}', per your NLST coding)", "1 if this category applies, else 0"
    if col == "scr_days0":
        return "Days from enrollment to the T0 screening visit", "integer (days)"
    if re.match(r"scr_res0_\d+$", col):
        code = col.split("_")[-1]
        return (f"T0 screening RESULT code = {code} (categorical -- verify exact meaning "
                 f"of code {code} against your NLST data dictionary)"), "1 if this code applies, else 0"
    if re.match(r"scr_iso0_\d+$", col):
        code = col.split("_")[-1]
        return (f"T0 screening ISOLATED-FINDING code = {code} (categorical -- verify exact "
                 f"meaning of code {code} against your NLST data dictionary)"), "1 if this code applies, else 0"

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
        return "Total number of non-calcified nodules >=4mm", "integer count"
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

    return "Column meaning not auto-recognized -- check your NLST data dictionary directly", "check codebook"


def main():
    args = parse_args()
    cols_path = os.path.join(args.clinical_models_dir, "clinical_feature_columns.json")
    if not os.path.exists(cols_path):
        print(f"ERROR: {cols_path} not found. Run save_clinical_models_for_inference.py first.")
        sys.exit(1)

    cols = json.load(open(cols_path))
    print(f"Loaded {len(cols)} required columns from {cols_path}")

    # --- template CSV (same as before) ---
    row = {}
    n_defaulted, n_blank = 0, 0
    for c in cols:
        if c.endswith("_count_t0") or c.endswith("_present_t0") or c.startswith("any_"):
            row[c] = 0
            n_defaulted += 1
        else:
            row[c] = ""
            n_blank += 1
    df = pd.DataFrame([row])[cols]
    df.to_csv(args.output, index=False)

    # --- field guide CSV (new) ---
    guide_rows = []
    for c in cols:
        desc, expected = describe_column(c)
        default = row[c] if row[c] != "" else "(fill in manually)"
        guide_rows.append({"column_name": c, "description": desc, "expected_value": expected,
                            "template_default": default})
    guide_df = pd.DataFrame(guide_rows)
    guide_df.to_csv(args.guide_output, index=False)

    print(f"\nSaved template     -> {args.output}")
    print(f"Saved field guide  -> {args.guide_output}")
    print(f"  {n_defaulted} count/present columns pre-filled with 0")
    print(f"  {n_blank} columns left BLANK for manual entry")
    print("\nOpen the field guide alongside the template -- each row explains what to put "
          "in the matching column. Codes flagged 'verify against your NLST data dictionary' "
          "need checking against your own source codebook, not guessed.")


if __name__ == "__main__":
    main()
