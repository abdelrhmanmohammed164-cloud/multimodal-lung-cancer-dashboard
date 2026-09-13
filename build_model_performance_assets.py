#!/usr/bin/env python3
"""Recalculate Model Performance from supplied OOF probabilities.

Validation rule
---------------
The project supplies out-of-fold (OOF) probabilities, not a separately trained
final hold-out prediction file for each individual branch. To avoid optimizing
classification thresholds on the same labels later used for reporting, this
script uses the common OOF cohort and creates one fixed stratified 50/50
secondary split:

* threshold-validation half: used ONLY to select each model's threshold;
* held-out evaluation half: used ONLY to report final threshold-dependent
  metrics and threshold-independent discrimination/calibration metrics.

The threshold criterion is Youden's J (Sensitivity + Specificity - 1), with
Balanced Accuracy, MCC and F1 used only as deterministic tie-breakers.

The four-way Fusion probability is reconstructed from the current dashboard's
existing fixed weights: Clinical 0.40, 2D 0.20, 3D-MIL 0.30, Whole Lung 0.10.
The weights are NOT re-optimized here.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

APP_DIR = Path(__file__).resolve().parent
SOURCE_DIR = APP_DIR / "project_results" / "model_performance_by_model" / "source"
OUT_DIR = APP_DIR / "project_results" / "model_performance_by_model"
IMG_DIR = APP_DIR / "static" / "images" / "performance_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

COMMON_FILE = SOURCE_DIR / "all_branches_merged.csv"
RANDOM_STATE = 42
BOOTSTRAP_N = 1000

# Current dashboard fusion weights. Do not modify here unless the deployed
# dashboard itself is intentionally changed in a separate task.
FUSION_WEIGHTS = {
    "clinical_prob": 0.40,
    "old_2d_cnn_prob": 0.20,
    "cnn3d_mil_probability": 0.30,
    "wholelung_pretrained_probability": 0.10,
}

MODELS = [
    {
        "slug": "clinical",
        "name": "Clinical Model",
        "column": "clinical_prob",
        "technical_name": "CatBoost + XGBoost + Logistic Regression (weighted soft voting)",
        "input": "T0 clinical and radiology-derived tabular features",
    },
    {
        "slug": "2d",
        "name": "2D Model",
        "column": "old_2d_cnn_prob",
        "technical_name": "2D CT CNN",
        "input": "Selected axial CT slices",
    },
    {
        "slug": "3d-mil",
        "name": "3D Model — MIL",
        "column": "cnn3d_mil_probability",
        "technical_name": "3D multiple-instance learning CT model",
        "input": "3D nodule-candidate patches",
    },
    {
        "slug": "whole-lung",
        "name": "Whole Lung Model",
        "column": "wholelung_pretrained_probability",
        "technical_name": "MedicalNet-pretrained ResNet-10 (3D, MONAI)",
        "input": "Whole-lung 3D CT volume",
    },
    {
        "slug": "fusion",
        "name": "Final Model — Fusion",
        "column": "fusion_probability",
        "technical_name": "Four-way weighted probability fusion",
        "input": "Clinical 40% + 2D 20% + 3D-MIL 30% + Whole Lung 10%",
    },
]

COLORS = {
    "clinical": "#E36C9A",
    "2d": "#5B8DEF",
    "3d-mil": "#9B6DE3",
    "whole-lung": "#45B9A8",
    "fusion": "#F2A65A",
}


def load_common_data():
    if not COMMON_FILE.exists():
        raise FileNotFoundError(f"Missing source file: {COMMON_FILE}")
    df = pd.read_csv(COMMON_FILE)
    required = ["pid", "has_cancer"] + list(FUSION_WEIGHTS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Source data missing required columns: {missing}")

    # Strictly use patients with complete probability outputs for all four
    # current branches, so every individual model and Fusion are evaluated on
    # the exact same cases.
    before = len(df)
    df = df.dropna(subset=required).copy()
    if len(df) != before:
        print(f"Dropped {before-len(df)} row(s) with missing model probabilities.")

    # Reconstruct Fusion from current fixed deployed weights.
    df["fusion_probability"] = sum(w * df[c].astype(float) for c, w in FUSION_WEIGHTS.items())

    # Sanity-check against the provided four-way reference column if present.
    if "fused_probability_4WAY_reference_only" in df.columns:
        max_diff = float(np.max(np.abs(df["fusion_probability"] - df["fused_probability_4WAY_reference_only"])))
        if max_diff > 1e-10:
            raise RuntimeError(f"Four-way fusion reconstruction mismatch (max diff {max_diff}).")

    df["has_cancer"] = df["has_cancer"].astype(int)
    # Preserve the supplied row order because the project's existing 50/50
    # development/test split was created from this ordering with random_state=42.
    # Keeping it reproduces the established split rather than silently creating
    # a different one.
    return df.reset_index(drop=True)


def classification_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision / PPV": float(precision),
        "NPV": float(npv),
        "F1-score": float(f1_score(y, pred, zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "Cohen's Kappa": float(cohen_kappa_score(y, pred)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def select_validation_threshold(y, p):
    fpr, tpr, thresholds = roc_curve(y, p)
    candidates = []
    for fpr_i, tpr_i, threshold in zip(fpr, tpr, thresholds):
        if not np.isfinite(threshold):
            continue
        metrics = classification_metrics(y, p, float(threshold))
        youden = float(tpr_i + (1.0 - fpr_i) - 1.0)
        key = (
            youden,
            metrics["Balanced Accuracy"],
            metrics["MCC"],
            metrics["F1-score"],
            -abs(float(threshold) - 0.5),
        )
        candidates.append((key, float(threshold), metrics, youden))
    if not candidates:
        threshold = 0.5
        metrics = classification_metrics(y, p, threshold)
        return threshold, metrics, metrics["Sensitivity"] + metrics["Specificity"] - 1.0
    _, threshold, metrics, youden = max(candidates, key=lambda x: x[0])
    return threshold, metrics, youden


def bootstrap_auc_ci(y, p, n_boot=BOOTSTRAP_N, seed=RANDOM_STATE):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bootstrap_auc_difference_ci(y, p_fusion, p_other, n_boot=BOOTSTRAP_N, seed=RANDOM_STATE):
    y = np.asarray(y, dtype=int)
    p_fusion = np.asarray(p_fusion, dtype=float)
    p_other = np.asarray(p_other, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(roc_auc_score(y[idx], p_fusion[idx]) - roc_auc_score(y[idx], p_other[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _midrank(x):
    order = np.argsort(x)
    sx = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x), dtype=float)
    out[order] = ranks
    return out


def _fast_delong(pred_sorted, n_pos):
    m = n_pos
    n = pred_sorted.shape[1] - m
    k = pred_sorted.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pred_sorted[r, :m])
        ty[r] = _midrank(pred_sorted[r, m:])
        tz[r] = _midrank(pred_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]]); sy = np.array([[float(sy)]])
    return aucs, sx / m + sy / n


def delong_pvalue(y, p1, p2):
    y = np.asarray(y, dtype=int)
    order = np.argsort(-y)
    n_pos = int(y[order].sum())
    pred = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(pred, n_pos)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ cov @ contrast.T).item())
    if variance <= 0:
        return 1.0
    z = abs(float(aucs[0] - aucs[1])) / math.sqrt(variance)
    return float(2.0 * norm.sf(z))


def descriptive_oof_stability(df, model_col):
    """Five stratified evaluation slices of the already-OOF probabilities.

    This is deliberately labelled as OOF evaluation stability, not model
    retraining CV, because original patient-level training-fold IDs are not
    supplied consistently for all five displayed models (including Fusion).
    """
    y = df["has_cancer"].to_numpy(dtype=int)
    p = df[model_col].to_numpy(dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    vals = []
    for _, idx in skf.split(np.zeros(len(y)), y):
        vals.append(float(roc_auc_score(y[idx], p[idx])))
    return vals


def _valid_image(path):
    try:
        return Path(path).is_file() and Path(path).stat().st_size > 1024
    except OSError:
        return False


def save_model_plots(model, validation, test, result, full_df):
    slug = model["slug"]
    col = model["column"]
    color = COLORS[slug]
    y_test = test["has_cancer"].to_numpy(dtype=int)
    p_test = test[col].to_numpy(dtype=float)
    y_val = validation["has_cancer"].to_numpy(dtype=int)
    p_val = validation[col].to_numpy(dtype=float)
    threshold = result["optimal_threshold"]

    expected = {
        "roc_pr": f"{slug}_roc_pr.png",
        "threshold": f"{slug}_threshold_validation.png",
        "calibration": f"{slug}_calibration.png",
        "confusion": f"{slug}_confusion_matrix.png",
        "stability": f"{slug}_oof_stability.png",
    }
    if all(_valid_image(IMG_DIR / fn) for fn in expected.values()):
        return expected

    # ROC + Precision-Recall
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    fpr, tpr, _ = roc_curve(y_test, p_test)
    axes[0].plot(fpr, tpr, color=color, lw=2.2, label=f"AUC = {result['metrics']['ROC-AUC']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[0].set_title("ROC Curve — held-out evaluation")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=.22)

    precision, recall, _ = precision_recall_curve(y_test, p_test)
    axes[1].plot(recall, precision, color=color, lw=2.2, label=f"PR-AUC = {result['metrics']['PR-AUC']:.3f}")
    axes[1].set_title("Precision–Recall Curve — held-out evaluation")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=.22)
    fig.suptitle(model["name"], fontsize=15)
    fig.tight_layout()
    fname = f"{slug}_roc_pr.png"
    fig.savefig(IMG_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Threshold validation plot
    thresholds = np.linspace(0.05, 0.95, 181)
    sens, spec, bal, f1s, mccs = [], [], [], [], []
    for t in thresholds:
        mm = classification_metrics(y_val, p_val, float(t))
        sens.append(mm["Sensitivity"]); spec.append(mm["Specificity"])
        bal.append(mm["Balanced Accuracy"]); f1s.append(mm["F1-score"]); mccs.append(mm["MCC"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(thresholds, sens, label="Sensitivity", lw=1.9)
    ax.plot(thresholds, spec, label="Specificity", lw=1.9)
    ax.plot(thresholds, bal, label="Balanced Accuracy", lw=2.2, color=color)
    ax.plot(thresholds, f1s, label="F1", lw=1.5, linestyle=":")
    ax.axvline(threshold, color="#222222", linestyle="--", lw=1.5,
               label=f"Selected t = {threshold:.3f}")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Validation — validation half only")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=.22)
    fig.tight_layout()
    fname_threshold = f"{slug}_threshold_validation.png"
    fig.savefig(IMG_DIR / fname_threshold, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", color=color, lw=2,
            label=f"Brier = {result['metrics']['Brier Score']:.3f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed cancer frequency")
    ax.set_title("Calibration — held-out evaluation")
    ax.legend(fontsize=9)
    ax.grid(alpha=.22)
    fig.tight_layout()
    fname_cal = f"{slug}_calibration.png"
    fig.savefig(IMG_DIR / fname_cal, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Confusion matrix
    pred = (p_test >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.7, 5.1))
    im = ax.imshow(cm, cmap="Blues")
    vmax = cm.max()
    for (r, c), value in np.ndenumerate(cm):
        ax.text(c, r, str(value), ha="center", va="center", fontsize=16,
                color="white" if value > vmax * .55 else "#222222")
    ax.set_xticks([0, 1], ["No Cancer", "Cancer"])
    ax.set_yticks([0, 1], ["No Cancer", "Cancer"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — t={threshold:.3f}")
    fig.tight_layout()
    fname_cm = f"{slug}_confusion_matrix.png"
    fig.savefig(IMG_DIR / fname_cm, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # OOF stability plot (same method for all displayed models, including Fusion)
    vals = result["oof_stability_auc"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(1, 6)
    ax.bar(x, vals, color=color, edgecolor="#555555", alpha=.85)
    mean_v = float(np.mean(vals))
    ax.axhline(mean_v, color="#333333", linestyle="--", lw=1.4, label=f"Mean = {mean_v:.3f}")
    ax.set_xticks(x, [f"Fold {i}" for i in x])
    ax.set_ylim(0.45, min(1.0, max(vals) + 0.12))
    ax.set_ylabel("ROC-AUC")
    ax.set_title("OOF Evaluation Stability — 5 stratified slices")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.22)
    fig.tight_layout()
    fname_stab = f"{slug}_oof_stability.png"
    fig.savefig(IMG_DIR / fname_stab, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "roc_pr": fname,
        "threshold": fname_threshold,
        "calibration": fname_cal,
        "confusion": fname_cm,
        "stability": fname_stab,
    }


def build_analysis_text(model_name, metrics, threshold, validation_youden, best_individual_name=None, best_individual_metrics=None, fusion_compare=None):
    sens = metrics["Sensitivity"]
    spec = metrics["Specificity"]
    auc = metrics["ROC-AUC"]
    bal = metrics["Balanced Accuracy"]
    f1 = metrics["F1-score"]
    mcc = metrics["MCC"]
    balance_gap = abs(sens - spec)

    if model_name == "Final Model — Fusion":
        base = (
            f"The four-way Fusion achieved ROC-AUC {auc:.3f}, Balanced Accuracy {bal:.3f}, "
            f"F1 {f1:.3f}, and MCC {mcc:.3f} on the held-out evaluation half. "
            f"Its validation-selected threshold was {threshold:.3f} (validation Youden J {validation_youden:.3f})."
        )
        if best_individual_metrics:
            d_auc = auc - best_individual_metrics["ROC-AUC"]
            d_bal = bal - best_individual_metrics["Balanced Accuracy"]
            d_sens = sens - best_individual_metrics["Sensitivity"]
            d_spec = spec - best_individual_metrics["Specificity"]
            base += (
                f" Compared with the strongest individual model ({best_individual_name}), Fusion changed ROC-AUC by {d_auc:+.3f}, "
                f"Balanced Accuracy by {d_bal:+.3f}, Sensitivity by {d_sens:+.3f}, and Specificity by {d_spec:+.3f}."
            )
        if fusion_compare:
            p = fusion_compare.get("delong_p_vs_best_individual")
            ci = fusion_compare.get("auc_difference_95_ci")
            if p is not None and ci:
                base += (
                    f" The paired AUC comparison versus {best_individual_name} gave DeLong p={p:.3g}; "
                    f"the bootstrap 95% CI for the AUC difference was [{ci[0]:+.3f}, {ci[1]:+.3f}]."
                )
        return base

    balance_phrase = "well balanced" if balance_gap < 0.08 else ("sensitivity-favoring" if sens > spec else "specificity-favoring")
    return (
        f"At the validation-selected threshold of {threshold:.3f}, the held-out evaluation performance was "
        f"ROC-AUC {auc:.3f}, Sensitivity {sens:.3f}, Specificity {spec:.3f}, Balanced Accuracy {bal:.3f}, "
        f"F1 {f1:.3f}, and MCC {mcc:.3f}. The operating point is {balance_phrase} "
        f"(Sensitivity–Specificity gap {balance_gap:.3f}). The threshold was selected only from the validation half by "
        f"maximizing Youden J ({validation_youden:.3f}); the evaluation half was not used to tune it."
    )


def main():
    df = load_common_data()

    # The supplied 3D-MIL OOF file contains 1637 cases, while the original
    # project cohort is 1638. To make all model comparisons mathematically fair,
    # the analysis uses the common 1637-patient cohort present across all four
    # model probability outputs. No data values are modified.
    n_original = 1638
    n_common = len(df)

    indices = np.arange(n_common)
    val_idx, test_idx = train_test_split(
        indices,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=df["has_cancer"].to_numpy(dtype=int),
    )
    validation = df.iloc[val_idx].sort_values("pid").reset_index(drop=True)
    test = df.iloc[test_idx].sort_values("pid").reset_index(drop=True)

    model_results = []
    for model in MODELS:
        col = model["column"]
        y_val = validation["has_cancer"].to_numpy(dtype=int)
        p_val = validation[col].to_numpy(dtype=float)
        y_test = test["has_cancer"].to_numpy(dtype=int)
        p_test = test[col].to_numpy(dtype=float)

        threshold, val_threshold_metrics, val_youden = select_validation_threshold(y_val, p_val)
        metrics = classification_metrics(y_test, p_test, threshold)
        metrics.update({
            "ROC-AUC": float(roc_auc_score(y_test, p_test)),
            "PR-AUC": float(average_precision_score(y_test, p_test)),
            "Brier Score": float(brier_score_loss(y_test, p_test)),
            "ROC-AUC 95% CI": bootstrap_auc_ci(y_test, p_test),
        })
        stability = descriptive_oof_stability(df, col)

        item = {
            **model,
            "n_patients": n_common,
            "n_cancer": int(df["has_cancer"].sum()),
            "n_no_cancer": int((1 - df["has_cancer"]).sum()),
            "validation_n": int(len(validation)),
            "evaluation_n": int(len(test)),
            "optimal_threshold": float(threshold),
            "threshold_selection": {
                "criterion": "Maximum Youden J on threshold-validation half; deterministic tie-breakers: Balanced Accuracy, MCC, F1",
                "validation_youden_j": float(val_youden),
                "validation_balanced_accuracy": float(val_threshold_metrics["Balanced Accuracy"]),
                "validation_sensitivity": float(val_threshold_metrics["Sensitivity"]),
                "validation_specificity": float(val_threshold_metrics["Specificity"]),
            },
            "metrics": metrics,
            "oof_stability_auc": stability,
            "oof_stability_mean": float(np.mean(stability)),
            "oof_stability_std": float(np.std(stability, ddof=1)),
        }
        model_results.append(item)

    # Determine strongest individual model on held-out balanced classification.
    individuals = model_results[:4]
    best_individual = max(individuals, key=lambda x: (x["metrics"]["Balanced Accuracy"], x["metrics"]["MCC"], x["metrics"]["F1-score"]))
    fusion = model_results[-1]

    y_test = test["has_cancer"].to_numpy(dtype=int)
    p_fusion = test["fusion_probability"].to_numpy(dtype=float)
    p_best = test[best_individual["column"]].to_numpy(dtype=float)
    fusion_compare = {
        "best_individual": best_individual["name"],
        "delong_p_vs_best_individual": delong_pvalue(y_test, p_fusion, p_best),
        "auc_difference": float(fusion["metrics"]["ROC-AUC"] - best_individual["metrics"]["ROC-AUC"]),
        "auc_difference_95_ci": bootstrap_auc_difference_ci(y_test, p_fusion, p_best),
        "metric_deltas": {
            metric: float(fusion["metrics"][metric] - best_individual["metrics"][metric])
            for metric in ["ROC-AUC", "PR-AUC", "Accuracy", "Sensitivity", "Specificity", "Precision / PPV", "NPV", "F1-score", "Balanced Accuracy", "MCC"]
        },
    }

    # Add analysis text and create plots after all metrics are known.
    for item in model_results:
        compare = fusion_compare if item["slug"] == "fusion" else None
        item["analysis"] = build_analysis_text(
            item["name"], item["metrics"], item["optimal_threshold"],
            item["threshold_selection"]["validation_youden_j"],
            best_individual_name=best_individual["name"],
            best_individual_metrics=best_individual["metrics"],
            fusion_compare=compare,
        )
        item["images"] = save_model_plots(item, validation, test, item, df)

    # Fusion comparison plot.
    metric_names = ["ROC-AUC", "Sensitivity", "Specificity", "Balanced Accuracy", "F1-score", "MCC"]
    labels = [m["name"].replace(" Model", "") for m in model_results]
    x = np.arange(len(model_results))
    width = 0.13
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    for i, metric in enumerate(metric_names):
        vals = [m["metrics"][metric] for m in model_results]
        ax.bar(x + (i - (len(metric_names)-1)/2)*width, vals, width, label=metric)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
    ax.set_title("Held-out Performance — Individuals vs Final Fusion")
    ax.legend(ncol=3, fontsize=9); ax.grid(axis="y", alpha=.22)
    fig.tight_layout()
    fusion_comparison_img = "fusion_vs_individuals.png"
    fig.savefig(IMG_DIR / fusion_comparison_img, dpi=180, bbox_inches="tight")
    plt.close(fig)
    fusion["images"]["comparison"] = fusion_comparison_img

    # Persist exact split predictions so every dashboard number can be audited.
    validation_out = validation[["pid", "has_cancer"] + [m["column"] for m in MODELS]].copy()
    test_out = test[["pid", "has_cancer"] + [m["column"] for m in MODELS]].copy()
    validation_out.to_csv(OUT_DIR / "threshold_validation_predictions.csv", index=False)
    test_out.to_csv(OUT_DIR / "heldout_evaluation_predictions.csv", index=False)

    payload = {
        "protocol": {
            "original_dataset_n": n_original,
            "common_eligible_n": n_common,
            "common_eligible_reason": "One original patient has no 3D-MIL OOF probability; all five displayed models use the same 1,637 complete cases for fair comparison.",
            "validation_n": int(len(validation)),
            "validation_cancer": int(validation["has_cancer"].sum()),
            "evaluation_n": int(len(test)),
            "evaluation_cancer": int(test["has_cancer"].sum()),
            "random_state": RANDOM_STATE,
            "probability_source": "Supplied out-of-fold probabilities. Each patient's branch probability was generated without fitting that branch on that patient's own outcome.",
            "threshold_rule": "Each model threshold is selected independently on the threshold-validation half using maximum Youden J; Balanced Accuracy, MCC, and F1 are tie-breakers. Threshold is then frozen before evaluation.",
            "important_limitation": "The project does not supply a separately trained, untouched final-test prediction file for each individual branch. Therefore the leakage-safe separation is performed at the OOF-prediction/threshold-selection layer: threshold labels from the held-out evaluation half are never used to choose thresholds.",
            "fusion_weights": FUSION_WEIGHTS,
            "fusion_weights_changed": False,
        },
        "models": model_results,
        "best_individual": {
            "name": best_individual["name"],
            "balanced_accuracy": best_individual["metrics"]["Balanced Accuracy"],
        },
        "fusion_comparison": fusion_compare,
    }

    with open(OUT_DIR / "model_performance_data.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(OUT_DIR / "validation_protocol.json", "w") as f:
        json.dump(payload["protocol"], f, indent=2)

    print(f"Built Model Performance for {n_common} common eligible patients.")
    print(f"Validation: {len(validation)} | Held-out evaluation: {len(test)}")
    for item in model_results:
        m = item["metrics"]
        print(
            f"{item['name']}: threshold={item['optimal_threshold']:.4f}, "
            f"AUC={m['ROC-AUC']:.4f}, Sens={m['Sensitivity']:.4f}, "
            f"Spec={m['Specificity']:.4f}, BalAcc={m['Balanced Accuracy']:.4f}, MCC={m['MCC']:.4f}"
        )


if __name__ == "__main__":
    main()
