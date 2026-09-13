#!/usr/bin/env python3
"""
show_patient_values.py
================================================================
Reads a real patient's clinical row (exported from your NLST CSV) and
prints every value next to its column name, grouped exactly like the
dashboard's form -- so you can copy each value into the matching field
by hand without hunting through 106+ columns.

RUN:
    python3 show_patient_values.py \
        --patient-csv patient_125951_clinical.csv \
        --clinical-models-dir clinical_models_for_inference
"""

import argparse
import json
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("Missing pandas. Install with: pip install pandas")
    sys.exit(1)

try:
    from make_clinical_template import describe_column
except ImportError:
    def describe_column(col):
        return col.replace("_", " ").capitalize(), "numeric"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patient-csv", required=True, help="1-row CSV with the real patient's clinical values")
    p.add_argument("--clinical-models-dir", default="clinical_models_for_inference")
    return p.parse_args()


GROUPS = {
    "Demographics": lambda c: c in ("age", "gender", "cigsmok", "pack_years") or c.startswith("race_"),
    "Screening result codes": lambda c: c.startswith("scr_"),
    "Abnormality findings": lambda c: c.startswith("abn_") or c.startswith("any_") or c in (
        "total_abnormalities_t0", "total_ctab_records_t0", "n_noncalc_nodules_ge4mm_t0",
        "has_nodule_measurement_t0", "n_nodules_with_long_diameter_t0"),
    "Nodule measurements": lambda c: "diameter" in c or "area_proxy" in c or "aspect_ratio" in c,
    "Nodule margins": lambda c: c.startswith("margin_"),
    "Nodule attenuation": lambda c: c.startswith("atten_"),
    "Nodule location": lambda c: c.startswith("location_"),
}


def main():
    args = parse_args()
    row = pd.read_csv(args.patient_csv).iloc[0]

    cols_path = os.path.join(args.clinical_models_dir, "clinical_feature_columns.json")
    if not os.path.exists(cols_path):
        print(f"ERROR: {cols_path} not found.")
        sys.exit(1)
    feature_cols = json.load(open(cols_path))

    missing = [c for c in feature_cols if c not in row.index]
    if missing:
        print(f"WARNING: patient CSV is missing {len(missing)} expected columns: {missing[:5]}...")

    assigned = set()
    print("=" * 70)
    print(f"VALUES TO ENTER IN THE DASHBOARD -- {args.patient_csv}")
    print("=" * 70)

    only_nonzero_groups = ("Screening result codes", "Abnormality findings",
                            "Nodule margins", "Nodule attenuation", "Nodule location")

    for group_name, matcher in GROUPS.items():
        cols = [c for c in feature_cols if matcher(c) and c not in assigned]
        if not cols:
            continue
        assigned.update(cols)

        if group_name in only_nonzero_groups:
            nonzero = [(c, row[c]) for c in cols if c in row.index and row[c] not in (0, "0", 0.0)]
            print(f"\n--- {group_name} ({len(cols)} fields total, {len(nonzero)} non-zero -- "
                  f"the rest can stay at 0) ---")
            for c, v in nonzero:
                desc, _ = describe_column(c)
                print(f"  {c} = {v}   ({desc})")
        else:
            print(f"\n--- {group_name} ---")
            for c in cols:
                v = row[c] if c in row.index else "MISSING"
                desc, _ = describe_column(c)
                print(f"  {c} = {v}   ({desc})")

    leftover = [c for c in feature_cols if c not in assigned]
    if leftover:
        print(f"\n--- Other fields ---")
        for c in leftover:
            v = row[c] if c in row.index else "MISSING"
            print(f"  {c} = {v}")


if __name__ == "__main__":
    main()
