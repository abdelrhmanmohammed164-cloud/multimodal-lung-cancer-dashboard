#!/usr/bin/env python3
"""
predict_clinical_worker.py
================================================================
Runs ONLY the clinical branch (CatBoost + XGBoost + LogisticRegression)
and prints its result as JSON to stdout. Deliberately never imports
torch -- this is called as a SEPARATE PROCESS by predict_new_patient.py
so CatBoost/XGBoost's OpenMP runtime never has to coexist with
PyTorch's in the same process, which was causing a hard crash
(EXC_BAD_ACCESS / SIGSEGV in libomp) on Apple Silicon.

RUN (called automatically by predict_new_patient.py, or standalone):
    python3 predict_clinical_worker.py \
        --clinical-csv patient_clinical_row.csv \
        --clinical-models-dir clinical_models_for_inference
"""

import argparse
import json
import os
import sys

try:
    import numpy as np
    import pandas as pd
    import joblib
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
except ImportError as e:
    print(json.dumps({"error": f"Missing package: {e}"}))
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clinical-csv", required=True)
    p.add_argument("--clinical-models-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        row = pd.read_csv(args.clinical_csv).iloc[[0]]
        cols = json.load(open(os.path.join(args.clinical_models_dir, "clinical_feature_columns.json")))
        weights = json.load(open(os.path.join(args.clinical_models_dir, "clinical_weights.json")))
        imputer = joblib.load(os.path.join(args.clinical_models_dir, "clinical_imputer.pkl"))

        missing = [c for c in cols if c not in row.columns]
        if missing:
            print(json.dumps({"error": f"Clinical CSV missing columns: {missing}"}))
            sys.exit(1)

        X_new = row[cols].values
        X_new_imputed = imputer.transform(X_new)

        cat = CatBoostClassifier()
        cat.load_model(os.path.join(args.clinical_models_dir, "clinical_catboost.cbm"))
        xgb = XGBClassifier()
        xgb.load_model(os.path.join(args.clinical_models_dir, "clinical_xgboost.json"))
        lr = joblib.load(os.path.join(args.clinical_models_dir, "clinical_logreg.pkl"))

        p_cat = float(cat.predict_proba(X_new_imputed)[:, 1][0])
        p_xgb = float(xgb.predict_proba(X_new_imputed)[:, 1][0])
        p_lr = float(lr.predict_proba(X_new_imputed)[:, 1][0])

        total_w = weights["cat"] + weights["xgb"] + weights["lr"]
        prob = (weights["cat"] * p_cat + weights["xgb"] * p_xgb + weights["lr"] * p_lr) / total_w

        print(json.dumps({"clinical_prob": prob, "p_cat": p_cat, "p_xgb": p_xgb, "p_lr": p_lr}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
