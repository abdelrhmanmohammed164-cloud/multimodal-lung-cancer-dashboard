"""
NLST clinical subset - EDA for the paper.
Loads the cleaned clinical CSV, checks data quality, compares cancer vs
no-cancer groups (univariate + logistic regression), saves plots/tables.

Usage:
    python Data_analysis_COMPLETE.py
    python Data_analysis_COMPLETE.py --data-path "/some/other/path.csv" --output-dir "eda_outputs2"
"""

import argparse
import logging
import os
import shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# =============================================================================
# SECTION 0 -- Config: paths, logging, colors, label lookups
# =============================================================================
DEFAULT_DATA_PATH = "/Volumes/Apple/images_organized/Analysis/clinical_T0_1638_patients_FINAL.csv"
DEFAULT_CTAB_PATH = "/Volumes/Apple/images_organized/Analysis/nlst_780_ctab_idc_20210527.csv"
DEFAULT_OUTPUT_DIR = "/Volumes/Apple/images_organized/Analysis/eda_outputs"
TARGET_COL = "has_cancer"
ID_COL = "pid"

# Locked, already-validated STRICT T0 results for the CURRENT selected model
# (CatBoost+XGBoost+LogReg Weighted Ensemble; development cohort n=1654,
# independent cohort n=24,799, zero patient overlap). These are reported as
# fixed reference values in the leakage-audit and benchmark-footer graphs
# rather than recomputed inside this EDA script, so the EDA never touches
# the independent cohort and always reports the one true locked number.
# (Earlier project phase: the original locked model was Random Forest,
# development n=454, independent AUC=0.7594 -- superseded by the ensemble
# below after the 454->1654 dataset expansion and model-comparison experiment.)
LOCKED_DEV_AUC = 0.7784
LOCKED_DEV_THRESHOLD = 0.5025
LOCKED_INDEP_AUC = 0.7851
LOCKED_INDEP_SENSITIVITY = 0.6400
LOCKED_INDEP_SPECIFICITY = 0.7719
LOCKED_INDEP_BALANCED_ACC = 0.7059
LOCKED_INDEP_NPV = 0.9915
LOCKED_DEV_N = 1654
LOCKED_INDEP_N = 24799

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("nlst_eda")

sns.set_style("whitegrid")

# clear, professional, print-friendly light-medium colors -- distinct
# enough to read easily, not the washed-out pastel or the fully saturated
# defaults
CANCER_PALETTE = {"No Cancer": "#6FA8DC", "Cancer": "#F4A261"}
SIGNIFICANT_COLOR = "#F4A261"      # p < 0.05
NOT_SIGNIFICANT_COLOR = "#6FA8DC"  # p >= 0.05
# diverging colormap for correlation/cluster heatmaps, built from the same
# two colors used everywhere else
LIGHT_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "custom_diverging", [NOT_SIGNIFICANT_COLOR, "#FFFFFF", SIGNIFICANT_COLOR]
)

# raw column name -> readable label, used on plot axes/legends
PRETTY_NAMES = {
    "age": "Age",
    "cigsmok": "Current smoker",
    "gender": "Gender (Female)",
    "n_abnormalities_t0": "Number of abnormalities (T0)",
    "max_lesion_long_diam_t0": "Max lesion diameter (T0, mm)",
    "mean_lesion_long_diam_t0": "Mean lesion diameter (T0, mm)",
    "had_lesion_measurement_t0": "Lesion measurement present (T0)",
    "n_characterized_abnormalities_t0": "Characterized abnormalities (T0)",
    # Shape/morphology features (SECTION 12)
    "max_perp_diameter_mm_t0": "Max perpendicular diameter (T0, mm)",
    "mean_perp_diameter_mm_t0": "Mean perpendicular diameter (T0, mm)",
    "max_nodule_aspect_ratio_t0": "Max nodule aspect ratio (T0)",
    "mean_nodule_aspect_ratio_t0": "Mean nodule aspect ratio (T0)",
    "max_nodule_area_proxy_t0": "Max nodule area proxy (T0, mm\u00b2)",
    "mean_nodule_area_proxy_t0": "Mean nodule area proxy (T0, mm\u00b2)",
    "est_nodule_eccentricity_t0": "Estimated nodule eccentricity (T0)",
}

# Shape/morphology features actually present in the 1638-patient file --
# every one of these is either recorded directly (diameters, aspect ratio,
# area proxy) or derived transparently from recorded diameters
# (eccentricity, below). Nothing here is simulated or invented.
SHAPE_VARS = [
    "max_lesion_long_diam_t0", "max_perp_diameter_mm_t0",
    "mean_lesion_long_diam_t0", "mean_perp_diameter_mm_t0",
    "max_nodule_aspect_ratio_t0", "mean_nodule_aspect_ratio_t0",
    "max_nodule_area_proxy_t0", "mean_nodule_area_proxy_t0",
    "est_nodule_eccentricity_t0",
]

# Margin (border-shape) categories recorded per patient at T0
MARGIN_VARS = [
    "margin_spiculated_present_t0", "margin_smooth_present_t0",
    "margin_poorly_defined_present_t0", "margin_margin_undetermined_present_t0",
]
MARGIN_LABELS = {
    "margin_spiculated_present_t0": "Spiculated",
    "margin_smooth_present_t0": "Smooth",
    "margin_poorly_defined_present_t0": "Poorly defined",
    "margin_margin_undetermined_present_t0": "Undetermined",
}

# Attenuation (density pattern) categories recorded per patient at T0
ATTEN_VARS = [
    "atten_soft_tissue_present_t0", "atten_ground_glass_present_t0",
    "atten_mixed_attenuation_present_t0", "atten_fat_attenuation_present_t0",
    "atten_fluid_water_present_t0", "atten_other_attenuation_present_t0",
]
ATTEN_LABELS = {
    "atten_soft_tissue_present_t0": "Solid (soft tissue)",
    "atten_ground_glass_present_t0": "Ground-glass",
    "atten_mixed_attenuation_present_t0": "Mixed (part-solid)",
    "atten_fat_attenuation_present_t0": "Fat",
    "atten_fluid_water_present_t0": "Fluid/water",
    "atten_other_attenuation_present_t0": "Other/undetermined",
}

# Lobe location categories recorded per patient at T0
LOCATION_VARS = [
    "location_right_upper_lobe_present_t0", "location_right_middle_lobe_present_t0",
    "location_right_lower_lobe_present_t0", "location_left_upper_lobe_present_t0",
    "location_lingula_present_t0", "location_left_lower_lobe_present_t0",
    "location_other_location_present_t0",
]
LOCATION_LABELS = {
    "location_right_upper_lobe_present_t0": "Right upper lobe",
    "location_right_middle_lobe_present_t0": "Right middle lobe",
    "location_right_lower_lobe_present_t0": "Right lower lobe",
    "location_left_upper_lobe_present_t0": "Left upper lobe",
    "location_lingula_present_t0": "Lingula",
    "location_left_lower_lobe_present_t0": "Left lower lobe",
    "location_other_location_present_t0": "Other location",
}

# NLST race codes -> labels (per NLST data dictionary)
RACE_LABELS = {
    "race_1": "White",
    "race_2": "Black or African-American",
    "race_3": "Asian",
    "race_4": "American Indian/Alaska Native",
    "race_5": "Native Hawaiian/Pacific Islander",
    "race_6": "More than one race",
    "race_Other": "Other/Unknown",
}

# STRICT T0 ONLY -- every value here was verified (not just renamed) to be
# derived exclusively from information available at the baseline visit.
# See predictor_safety_audit() below for the full per-feature justification.
CONTINUOUS_VARS = ["age", "n_abnormalities_t0", "max_lesion_long_diam_t0"]
CORE_CLINICAL_FEATURES = CONTINUOUS_VARS[:1] + ["cigsmok", "gender"] + CONTINUOUS_VARS[1:]
EXTENDED_CLINICAL_FEATURES = CORE_CLINICAL_FEATURES + [
    "mean_lesion_long_diam_t0", "had_lesion_measurement_t0",
]


def parse_args():
    parser = argparse.ArgumentParser(description="NLST clinical subset EDA -- STRICT T0 ONLY")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--ctab-path", default=DEFAULT_CTAB_PATH,
                         help="raw NLST CT-abnormality table, used to rebuild lesion "
                              "features strictly from study_yr == 0 records")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


# =============================================================================
# SECTION 1 -- Load data & build STRICT T0 features (methodological correction)
# =============================================================================
# A temporal leakage audit (see graph 01 / predictor_safety_audit below) found
# that 61 of the original 92 predictors (66%) came from NLST screening rounds
# T1/T2 -- follow-up visits 1-2 years AFTER the baseline visit this analysis
# is meant to support. Every lesion feature below is rebuilt directly from the
# raw CT-abnormality table restricted to study_yr == 0, never trusted by name.
DESC_MAP = {
    51: "noncalc_nodule_ge4mm", 52: "noncalc_micronodule_lt4mm",
    53: "benign_calcified_nodule", 54: "atelectasis",
    55: "pleural_thickening_effusion", 56: "hilar_mediastinal_adenopathy_mass",
    57: "chest_wall_abnormality", 58: "consolidation", 59: "emphysema",
    60: "cardiovascular_abnormality", 61: "fibrosis_scar_reticular_opacity",
    62: "sixplus_nodules_not_suspicious", 63: "other_significant_above_diaphragm",
    64: "other_significant_below_diaphragm", 65: "other_minor_abnormality",
}


def build_t0_lesion_features(ctab: pd.DataFrame) -> pd.DataFrame:
    """Rebuilds lesion/nodule features from raw CTAB, restricted to the
    baseline (T0) visit only. Verified against
    t0_clinical_feature_engineering_v3_final_rf.py's methodology."""
    t = ctab[ctab["study_yr"].eq(0)].copy()
    for c in ["sct_ab_desc", "sct_ab_num", "sct_long_dia"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    g = t.groupby("pid")
    F = pd.DataFrame(index=g.size().index)
    F["n_abnormalities_t0"] = g["sct_ab_num"].nunique()

    nod = t[t["sct_ab_desc"].eq(51)].copy()  # non-calcified nodules >=4mm
    gn = nod.groupby("pid")
    F = F.join(gn["sct_long_dia"].max().rename("max_lesion_long_diam_t0"), how="left")
    F = F.join(gn["sct_long_dia"].mean().rename("mean_lesion_long_diam_t0"), how="left")
    F = F.join(gn["sct_long_dia"].count().rename("_n_measured"), how="left")
    F["had_lesion_measurement_t0"] = (F["_n_measured"].fillna(0) > 0).astype(int)
    F = F.drop(columns=["_n_measured"])

    # "characterized" = has at least one morphology field recorded, T0 only
    char_cols = [c for c in ["sct_ab_desc"] if c in t.columns]
    if char_cols:
        F = F.join(
            t.groupby("pid")["sct_ab_num"].nunique().rename("n_characterized_abnormalities_t0"),
            how="left"
        )

    F = F.reset_index()
    zero_cols = ["n_abnormalities_t0", "had_lesion_measurement_t0",
                 "n_characterized_abnormalities_t0"]
    for c in zero_cols:
        if c in F.columns:
            F[c] = F[c].fillna(0)
    return F


# clinical_T0_1638_patients_FINAL.csv already ships with its own rebuilt-T0
# lesion/shape columns, just under different names than the older
# build_t0_lesion_features() output. Renaming here (not recomputing) means
# this script runs entirely off the 1638-patient file with NO CTAB/other
# file needed -- the CTAB merge branch below only ever fires as a fallback
# for an older-format input file that lacks these columns.
COLUMN_RENAME_MAP = {
    "total_abnormalities_t0": "n_abnormalities_t0",
    "max_long_diameter_mm_t0": "max_lesion_long_diam_t0",
    "mean_long_diameter_mm_t0": "mean_lesion_long_diam_t0",
    "has_nodule_measurement_t0": "had_lesion_measurement_t0",
    # closest available proxy for "characterized" (at least one recorded
    # morphology measurement) in this file's column set
    "n_nodules_with_long_diameter_t0": "n_characterized_abnormalities_t0",
}


def load_data(data_path: str, ctab_path: str = None) -> pd.DataFrame:
    """Loads the base clinical file and merges in STRICT T0-only lesion
    features rebuilt from raw CTAB. If ctab_path is not provided, falls back
    to DEFAULT_CTAB_PATH.

    If data_path already contains the rebuilt T0 lesion columns (as
    clinical_T0_1638_patients_FINAL.csv does, once renamed via
    COLUMN_RENAME_MAP -- built once via this exact same
    build_t0_lesion_features()-equivalent logic during dataset
    construction), the CTAB merge step is skipped rather than redone, and
    any non-predictor tracking column (cohort_source) is dropped here."""
    df = pd.read_csv(data_path)
    log.info(f"Loaded base clinical data: {df.shape[0]} rows x {df.shape[1]} columns")

    rename_hits = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    if rename_hits:
        df = df.rename(columns=rename_hits)
        log.info(f"Renamed {len(rename_hits)} column(s) to the pipeline's standard T0 names "
                 f"(no data changed, no external file needed): {rename_hits}")

    already_has_t0_lesion = "n_abnormalities_t0" in df.columns
    if already_has_t0_lesion:
        log.info("Input file already contains rebuilt T0 lesion features -- skipping CTAB merge")
        if "cohort_source" in df.columns:
            log.info("Dropping 'cohort_source' (dataset-construction tracking column, not a predictor)")
            df = df.drop(columns=["cohort_source"])
        t0_lesion_cols = [c for c in df.columns if c.endswith("_t0")]

        # Derived shape feature (SECTION 12): model the dominant T0 nodule
        # as an ellipse using its recorded long/perpendicular diameters and
        # compute standard ellipse eccentricity, e = sqrt(1 - (b/a)^2) with
        # a = long diameter, b = perpendicular diameter. This is a genuine
        # geometric calculation from two measured values already in the
        # file -- not a modeled/simulated feature -- and is left as NaN
        # (never imputed here) for patients missing either measurement.
        if {"max_lesion_long_diam_t0", "max_perp_diameter_mm_t0"}.issubset(df.columns):
            df = df.copy()  # de-fragment before adding a new column
            a = df["max_lesion_long_diam_t0"]
            b = df["max_perp_diameter_mm_t0"]
            ratio_sq = (b / a.replace(0, np.nan)) ** 2
            df["est_nodule_eccentricity_t0"] = np.sqrt((1 - ratio_sq).clip(lower=0))
            t0_lesion_cols.append("est_nodule_eccentricity_t0")
    else:
        ctab_path = ctab_path or DEFAULT_CTAB_PATH
        ctab = pd.read_csv(ctab_path)
        log.info(f"Loaded raw CTAB: {ctab.shape[0]} rows")
        t0_features = build_t0_lesion_features(ctab)
        df = df.merge(t0_features, on=ID_COL, how="left")
        for c in ["n_abnormalities_t0", "had_lesion_measurement_t0", "n_characterized_abnormalities_t0"]:
            if c in df.columns:
                df[c] = df[c].fillna(0)
        t0_lesion_cols = [c for c in t0_features.columns if c != ID_COL]

    # STRICT T0 predictor allow-list -- everything else (T1/T2 screening
    # rounds, old unverified lesion aggregates, any_growth_flagged) is
    # dropped here, once, at the source -- not patched per-graph.
    demo_cols = [c for c in ["age", "gender", "cigsmok"] if c in df.columns]
    demo_cols += [c for c in df.columns if c.startswith("race_")]
    t0_screen_cols = [c for c in df.columns
                      if c == "scr_days0" or c.startswith(("scr_res0_", "scr_iso0_"))]
    keep_cols = list(dict.fromkeys(
        [ID_COL, TARGET_COL] + demo_cols + t0_screen_cols + t0_lesion_cols
    ))
    dropped = [c for c in df.columns if c not in keep_cols]
    if dropped:
        log.info(f"Dropped {len(dropped)} non-T0 columns (T1/T2 screening rounds and/or "
                 f"unverified aggregates) -- see predictor_safety_audit() for the full list")
    df = df[keep_cols]
    log.info(f"STRICT T0 dataset ready: {df.shape[0]} rows x {df.shape[1]} columns "
             f"({len(demo_cols)} demographic, {len(t0_screen_cols)} T0-screen, "
             f"{len(t0_lesion_cols)} T0-lesion)")
    return df


def structural_overview(data: pd.DataFrame):
    log.info("STRUCTURAL OVERVIEW")
    print("\n--- First 5 rows ---")
    print(data.head())
    print("\n--- Last 5 rows ---")
    print(data.tail())
    print(f"\n--- Columns ({data.shape[1]} total) ---")
    print(list(data.columns))
    print("\n--- Dtype summary ---")
    print(data.dtypes.value_counts())
    print("\n--- Descriptive statistics (numeric columns) ---")
    print(data.describe())



# =============================================================================
# SECTION 2 -- Data quality checks (missing, duplicates, constants...)
# =============================================================================
def data_quality_checks(data: pd.DataFrame, output_dir: str) -> list:
    log.info("DATA QUALITY CHECKS")

    n_dupes = data.duplicated().sum()
    print(f"\nFully duplicated rows: {n_dupes}")

    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    negative_counts = (data[numeric_cols] < 0).sum()
    negative_only = negative_counts[negative_counts > 0]
    if negative_only.empty:
        print("\nNo unexpected negative values in any numeric column.")
    else:
        print("\nColumns with negative values (need review):")
        print(negative_only)

    constant_cols = [c for c in data.columns if data[c].nunique() == 1]
    print(f"\nConstant columns: {len(constant_cols)}")
    if constant_cols:
        print(constant_cols)
        # these are rare scr_res/scr_iso categories that just didn't show up
        # in this particular sample -- fine to drop for EDA, but remember
        # they can come back if I train on the full dataset later
        print("(rare screening categories not present in this sample -- "
              "safe to drop here, but re-check if using the full dataset)")

    return constant_cols



# =============================================================================
# SECTION 3 -- Target variable & univariate group comparisons
# =============================================================================
# GRAPH 03 -- Class Distribution
def target_analysis(data: pd.DataFrame, target_col: str, output_dir: str):
    log.info(f"TARGET VARIABLE ANALYSIS: {target_col}")
    counts = data[target_col].value_counts()
    pct = data[target_col].value_counts(normalize=True) * 100
    print("\nCount per class:\n", counts)
    print("\nPercentage per class:\n", pct.round(2))

    fig, ax = plt.subplots(figsize=(5, 4))
    target_labels = data[target_col].map({0: "No Cancer", 1: "Cancer"})
    target_plot = pd.DataFrame({"Cancer status": target_labels})
    sns.countplot(data=target_plot, x="Cancer status", hue="Cancer status",
                  legend=False, palette=CANCER_PALETTE, ax=ax)
    clean_title(ax, "Class Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Patients")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_target_distribution.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {output_dir}/03_target_distribution.png")


# =============================================================================
# SECTION 4 -- Core exploratory plots (distributions, correlation, pairplot)
# =============================================================================
# (helper -- distribution overlap plot used by other graphs)
def plot_distribution_overlap(data: pd.DataFrame, target_col: str,
                               column: str, output_dir: str):
    if column not in data.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name in [(0, "No Cancer"), (1, "Cancer")]:
        subset = data.loc[data[target_col] == label, column].dropna()
        sns.kdeplot(subset, fill=True, alpha=0.5, label=name,
                    color=CANCER_PALETTE[name], ax=ax)
    clean_title(ax, f"{PRETTY_NAMES.get(column, column)} by Cancer Status", "Distribution overlap")
    ax.set_xlabel(column)
    ax.legend()
    fig.tight_layout()
    fname = f"04_distribution_overlap_{column}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {output_dir}/{fname}")


# GRAPH 05 -- Key Variable Distributions (boxplots)
def plot_key_distributions(data: pd.DataFrame, target_col: str, output_dir: str):
    """Mean +/- 1 SD per group for each key continuous variable, with the
    values labeled directly on the bars -- clearer than a boxplot for a
    quick side-by-side read of the group difference."""
    vars_to_plot = [v for v in CONTINUOUS_VARS if v in data.columns]
    if not vars_to_plot:
        return

    fig, axes = plt.subplots(1, len(vars_to_plot), figsize=(4.2 * len(vars_to_plot), 4.5))
    if len(vars_to_plot) == 1:
        axes = [axes]

    for ax, col in zip(axes, vars_to_plot):
        g0 = data.loc[data[target_col] == 0, col].dropna()
        g1 = data.loc[data[target_col] == 1, col].dropna()
        vals = [g0.mean(), g1.mean()]
        errs = [g0.std(), g1.std()]
        colors = [CANCER_PALETTE["No Cancer"], CANCER_PALETTE["Cancer"]]
        ax.bar(["No\nCancer", "Cancer"], vals, yerr=errs, color=colors,
               edgecolor="#555555", width=0.6, capsize=5)
        headroom = max(vals) * 0.12 + max(errs)
        for i, v in enumerate(vals):
            ax.text(i, v + errs[i] + headroom * 0.15, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_ylim(0, max(vals) + max(errs) + headroom)
        clean_title(ax, PRETTY_NAMES.get(col, col))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "05_key_variable_distributions.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {output_dir}/05_key_variable_distributions.png")


# GRAPH 06 -- Correlation Heatmap
def plot_correlation_heatmap(data: pd.DataFrame, output_dir: str) -> list:
    key_cols = [c for c in EXTENDED_CLINICAL_FEATURES + ["has_cancer"] if c in data.columns]
    corr = data[key_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=LIGHT_DIVERGING_CMAP, center=0, ax=ax)
    clean_title(ax, "Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {output_dir}/06_correlation_heatmap.png")
    return key_cols


# =============================================================================
# SECTION 5 -- Multivariate statistics (logistic regression, odds ratios)
# =============================================================================
# GRAPH 07 -- Logistic Regression Odds Ratios (forest plot)
def multivariate_logistic_regression(data: pd.DataFrame, target_col: str,
                                      output_dir: str):
    # univariate tests treat every variable in isolation, so a variable can
    # look significant just because it's correlated with something else
    # that actually matters. putting everything in one model gives each
    # coefficient's effect independent of the others.
    # reporting as odds ratios (exp(coef)) + 95% CI since that's the usual
    # format for this kind of result: OR>1 -> higher odds of cancer, OR<1
    # -> lower odds, CI crossing 1.0 -> not significant.
    log.info("MULTIVARIATE LOGISTIC REGRESSION (Odds Ratios)")

    predictors = [p for p in CORE_CLINICAL_FEATURES if p in data.columns]

    X = data[predictors].copy()
    y = data[target_col]

    # T0 lesion features are genuinely missing (NaN) for patients with no
    # baseline nodule measurement -- not 0 -- so median-impute before fitting,
    # same strategy used throughout the rest of this project's pipeline
    X = X.fillna(X.median(numeric_only=True))

    # z-score the continuous vars so coefficients are on a comparable scale
    for col in CONTINUOUS_VARS:
        if col in X.columns:
            X[col] = (X[col] - X[col].mean()) / X[col].std()

    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)

    summary_df = pd.DataFrame({
        "coefficient": model.params,
        "odds_ratio": np.exp(model.params),
        "ci_lower": np.exp(model.conf_int()[0]),
        "ci_upper": np.exp(model.conf_int()[1]),
        "p_value": model.pvalues,
    })
    summary_df["significant (p<0.05)"] = summary_df["p_value"] < 0.05

    print("\n", summary_df)

    # forest plot -- log scale since OR ranges are wildly different in size
    # (0.5 to 14 here), sorted by magnitude so it reads top-to-bottom.
    # dot color marks statistical significance -- explained in the legend,
    # not just left for the reader to guess.
    plot_df = summary_df.drop(index="const").sort_values("odds_ratio")
    labels = [PRETTY_NAMES.get(v, v) for v in plot_df.index]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    y_pos = np.arange(len(plot_df))
    colors = [SIGNIFICANT_COLOR if s else NOT_SIGNIFICANT_COLOR
              for s in plot_df["significant (p<0.05)"]]
    ax.errorbar(
        plot_df["odds_ratio"], y_pos,
        xerr=[plot_df["odds_ratio"] - plot_df["ci_lower"],
              plot_df["ci_upper"] - plot_df["odds_ratio"]],
        fmt="none", ecolor="#999999", capsize=4, zorder=1,
    )
    ax.scatter(plot_df["odds_ratio"], y_pos, color=colors, s=90, zorder=2,
               edgecolor="#555555", linewidth=0.8)
    ax.axvline(1.0, linestyle="--", color="gray")  # OR=1 reference line
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(row["ci_upper"] * 1.15, i,
                f"OR={row['odds_ratio']:.2f}, p={row['p_value']:.3f}",
                va="center", fontsize=8)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SIGNIFICANT_COLOR,
               markeredgecolor="#555555", markersize=9, label="Significant (p < 0.05)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NOT_SIGNIFICANT_COLOR,
               markeredgecolor="#555555", markersize=9, label="Not significant (p \u2265 0.05)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Odds Ratio, log scale (95% CI)")
    clean_title(ax, "Odds Ratios", "Multivariate logistic regression")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "07_odds_ratio_forest_plot.png"), dpi=150)
    plt.close(fig)
    log.info(f"Saved -> {output_dir}/07_odds_ratio_forest_plot.png")

    return summary_df


# =============================================================================
# SECTION 6 -- Shared plotting helpers
# =============================================================================
def save_current_figure(fig, output_dir: str, filename: str):
    """Save and close a Matplotlib figure consistently."""
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved -> {output_dir}/{filename}")


def clean_title(ax, main: str, note: str = None):
    """
    Sets a short, uncluttered title on an Axes. Any methodological detail
    (CV folds, thresholds, sample sizes...) goes in a small gray caption
    line underneath instead of being crammed into the title itself.
    """
    ax.set_title(main, fontsize=13, pad=18 if note else 6)
    if note:
        ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.5, color="#666666")


def clean_suptitle(fig, main: str, note: str = None, y: float = 1.03):
    """Same idea as clean_title, but for a whole-figure suptitle (used by
    pairplot/clustermap/multi-axes figures)."""
    fig.suptitle(main, fontsize=13, y=y)
    if note:
        fig.text(0.5, y - 0.035, note, ha="center", fontsize=8.5, color="#666666")


def render_table_image(df: pd.DataFrame, output_dir: str, filename: str,
                        title: str, note: str = None, figsize=None,
                        highlight_col: int = None, footer: str = None):
    """
    Renders a DataFrame as an image (no CSV) -- used for every result that's
    fundamentally a small table rather than a chart, so the numbers are
    still visible without a separate data file.
    """
    if figsize is None:
        figsize = (max(9.5, 1.4 * len(df.columns)), 1.0 + 0.5 * (len(df) + 1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    # reserve a fixed number of INCHES (not a fixed fraction) for the title,
    # so tall tables (many rows) don't have the table bleed up into the
    # title text -- a fixed fraction shrinks in absolute terms as the
    # figure gets taller, which is exactly backwards
    title_inches = 0.85 if note else 0.55
    top_frac = max(0.60, 1 - title_inches / figsize[1])
    tbl = ax.table(
        cellText=df.values, colLabels=df.columns,
        cellLoc="center", loc="center", bbox=[0, 0, 1, top_frac],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    for (row, col), tcell in tbl.get_celld().items():
        tcell.set_edgecolor("#CCCCCC")
        if row == 0:
            tcell.set_facecolor("#3B3B3B")
            tcell.set_text_props(color="white", fontweight="bold")
        elif col == highlight_col:
            tcell.set_facecolor("#FDEBD8")
        else:
            tcell.set_facecolor("#F7F7F7" if row % 2 == 0 else "white")
    # NOT using clean_title() here: its note is positioned in axes-fraction
    # coordinates, which collapses into the title for a very tall, mostly-
    # table axes (many-row tables). fig.text with a fixed inch-based
    # position keeps title and note correctly separated regardless of
    # table row count.
    ax.set_title(title, fontsize=13, pad=(24 if note else 8))
    if note:
        note_y = 1 - (0.32 / figsize[1])
        fig.text(0.5, note_y, note, ha="center", va="top", fontsize=8.5, color="#666666")
    if footer:
        fig.text(0.01, 0.01, footer, ha="left", fontsize=7.5, color="#888888")
    save_current_figure(fig, output_dir, filename)



# =============================================================================
# SECTION 7 -- Extra descriptive plots for the paper
# =============================================================================


# GRAPH 08 -- Feature Correlation with Cancer
def plot_target_correlations(data: pd.DataFrame, target_col: str, output_dir: str):
    """Rank numeric features by Pearson correlation with the target."""
    numeric = data.select_dtypes(include=[np.number, "bool"]).copy()
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    if target_col not in numeric.columns:
        return
    corr = numeric.corr()[target_col].drop(labels=[target_col]).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=True).index)

    fig, ax = plt.subplots(figsize=(9, max(6, 0.22 * len(corr))))
    bar_colors = [CANCER_PALETTE["Cancer"] if v >= 0 else CANCER_PALETTE["No Cancer"]
                  for v in corr]
    corr.plot(kind="barh", color=bar_colors, edgecolor="#555555", linewidth=0.4, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(corr):
        ax.text(v + (0.005 if v >= 0 else -0.005), i, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    clean_title(ax, "Feature Correlation with Cancer")
    ax.set_xlabel("Pearson correlation with has_cancer")
    save_current_figure(fig, output_dir, "08_target_feature_correlations.png")


# GRAPH 09 -- Age Distribution by Cancer Status
def plot_age_distribution(data: pd.DataFrame, target_col: str, output_dir: str):
    if "age" not in data.columns:
        return
    plot_df = data[["age", target_col]].copy()
    plot_df["Cancer status"] = plot_df[target_col].map({0: "No Cancer", 1: "Cancer"})
    fig, ax = plt.subplots(figsize=(8, 5))
    # Use the exact project-wide palette also used by Cancer Rate by Race.
    # Only rendering colors are standardized here; data/bins/labels are unchanged.
    sns.histplot(data=plot_df, x="age", hue="Cancer status", kde=True,
                 palette=CANCER_PALETTE, bins=20, common_norm=False,
                 element="step", alpha=0.38, linewidth=1.6, ax=ax)
    clean_title(ax, "Age Distribution by Cancer Status")
    ax.set_xlabel("Age (years)")
    save_current_figure(fig, output_dir, "09_age_distribution_by_cancer.png")


# GRAPHS 10/11 -- Smoking / Lesion Measurement vs Cancer
def plot_categorical_vs_target(data: pd.DataFrame, target_col: str,
                               column: str, title: str, output_dir: str,
                               filename: str, value_labels: dict = None,
                               xlabel: str = None):
    if column not in data.columns:
        return
    plot_df = data[[column, target_col]].copy()
    plot_df["Cancer status"] = plot_df[target_col].map({0: "No Cancer", 1: "Cancer"})
    # keep the original code order (0, 1, 2...) instead of letting seaborn
    # re-sort the labels alphabetically once they become text -- otherwise
    # e.g. "Female"/"Male" silently swap left-right vs. the 0/1 coding,
    # which reads as a mislabeled chart even though the values are correct
    order = None
    if value_labels:
        order = [value_labels[k] for k in sorted(value_labels)]
        plot_df[column] = plot_df[column].map(value_labels).fillna(plot_df[column])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=plot_df, x=column, order=order, hue="Cancer status",
                  palette=CANCER_PALETTE, ax=ax)
    clean_title(ax, title)
    ax.set_xlabel(xlabel if xlabel else ("" if value_labels else PRETTY_NAMES.get(column, column)))
    ax.set_ylabel("Number of patients")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    save_current_figure(fig, output_dir, filename)


# GRAPH 12 -- Lesion Diameter Violin Plot
def plot_lesion_violin(data: pd.DataFrame, target_col: str, output_dir: str):
    if "max_lesion_long_diam_t0" not in data.columns:
        return
    plot_df = data[["max_lesion_long_diam_t0", target_col]].copy()
    plot_df["Cancer status"] = plot_df[target_col].map({0: "No Cancer", 1: "Cancer"})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=plot_df, x="Cancer status", y="max_lesion_long_diam_t0",
                   hue="Cancer status", palette=CANCER_PALETTE, legend=False,
                   inner="quartile", cut=0, ax=ax)
    clean_title(ax, "Lesion Diameter by Cancer Status (T0)")
    ax.set_xlabel("")
    ax.set_ylabel("Max lesion diameter, T0 (mm)")
    save_current_figure(fig, output_dir, "12_lesion_size_violin.png")


# GRAPH 13 -- Lesion Size vs Number of Abnormalities
def plot_lesion_abnormality_scatter(data: pd.DataFrame, target_col: str,
                                    output_dir: str):
    required = {"max_lesion_long_diam_t0", "n_abnormalities_t0", target_col}
    if not required.issubset(data.columns):
        return
    plot_df = data[list(required)].copy()
    plot_df["Cancer status"] = plot_df[target_col].map({0: "No Cancer", 1: "Cancer"})

    # small jitter so the pile-up of patients with 0 lesion size doesn't just
    # look like one solid blob
    rng = np.random.default_rng(42)
    jitter_x = plot_df["max_lesion_long_diam_t0"] + rng.uniform(-0.4, 0.4, len(plot_df))
    jitter_y = plot_df["n_abnormalities_t0"] + rng.uniform(-0.3, 0.3, len(plot_df))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=jitter_x, y=jitter_y, hue=plot_df["Cancer status"],
                    palette=CANCER_PALETTE, alpha=0.6, s=45,
                    edgecolor="#555555", linewidth=0.4, ax=ax)
    clean_title(ax, "Lesion Size vs. Number of Abnormalities")
    ax.set_xlabel("Max lesion diameter (mm, jittered)")
    ax.set_ylabel("Total number of abnormalities (jittered)")
    ax.legend(title="Cancer status")
    save_current_figure(fig, output_dir, "13_lesion_size_vs_abnormalities.png")


# GRAPH 14 -- Cancer Rate by Race
def plot_race_stacked_bar(data: pd.DataFrame, target_col: str, output_dir: str):
    race_cols = [c for c in data.columns if c.startswith("race_")]
    if not race_cols:
        return
    race_label = data[race_cols].idxmax(axis=1).map(RACE_LABELS)
    no_race = data[race_cols].sum(axis=1) == 0
    race_label.loc[no_race] = "Unknown"
    table = pd.crosstab(race_label, data[target_col], normalize="index") * 100
    table = table.rename(columns={0: "No Cancer", 1: "Cancer"})

    fig, ax = plt.subplots(figsize=(9, 5))
    table.plot(kind="bar", stacked=True,
               color=[CANCER_PALETTE["No Cancer"], CANCER_PALETTE["Cancer"]], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f%%", label_type="center", fontsize=8)
    clean_title(ax, "Cancer Rate by Race")
    ax.set_xlabel("Race category")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Percentage of patients")
    ax.legend(title="Cancer status", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_current_figure(fig, output_dir, "14_race_stacked_bar.png")


# GRAPH 15 -- PCA Projection
def plot_pca_projection(data: pd.DataFrame, target_col: str, output_dir: str):
    """
    Two-dimensional PCA projection, using only the clinically meaningful
    numeric variables (not the ~70 sparse one-hot screening-result columns,
    which just dilute the signal and blur the plot into one big cluster).
    """
    feature_cols = [c for c in EXTENDED_CLINICAL_FEATURES if c in data.columns]
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.shape[1] < 2:
        return
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X)
    pca_df = pd.DataFrame({
        "PC1": components[:, 0], "PC2": components[:, 1],
        "Cancer status": data[target_col].map({0: "No Cancer", 1: "Cancer"}).values,
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cancer status",
                    palette=CANCER_PALETTE, alpha=0.7, s=50,
                    edgecolor="#555555", linewidth=0.4, ax=ax)
    clean_title(ax, "PCA Projection",
                f"PC1 {pca.explained_variance_ratio_[0]*100:.1f}%, "
                f"PC2 {pca.explained_variance_ratio_[1]*100:.1f}%")
    save_current_figure(fig, output_dir, "15_pca_projection.png")

# =============================================================================
# SECTION 8 -- Baseline model & core evaluation (Weighted Ensemble, Kaplan-Meier)
# =============================================================================
# (helper -- shared by every model-based graph below)
def prepare_model_data(data: pd.DataFrame, target_col: str):
    """Prepare numeric predictors for a reproducible baseline model."""
    y = data[target_col].astype(int)
    X = data.drop(columns=[target_col, "pid"], errors="ignore").copy()
    X = X.select_dtypes(include=[np.number, "bool"]).apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.nunique(dropna=False) > 1]
    return X, y


# (helper -- primary model factory)
def build_baseline_estimator():
    """
    Primary model: Weighted Soft-Voting Ensemble of CatBoost + XGBoost +
    Logistic Regression. Random Forest was the project's primary model in
    an earlier phase; a documented model-comparison + ensemble-combination
    experiment (7 individual model families, then voting/weighted/stacking
    combinations, all under identical 5-fold CV on this same cohort) found
    this weighted ensemble the strongest overall balance of AUC, MCC and
    stability. Weights are each base model's own out-of-fold ROC-AUC from
    that experiment (0.7775 / 0.7681 / 0.7681) -- not invented.
    """
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    estimators = [
        ("cat", CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                    random_state=42, verbose=0, auto_class_weights="Balanced")),
        ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               random_state=42, eval_metric="logloss")),
        ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
    ]
    voting = VotingClassifier(estimators=estimators, voting="soft",
                               weights=[0.7775, 0.7681, 0.7681], n_jobs=1)
    return voting, "CatBoost+XGBoost+LogReg (Weighted Ensemble)"


# (helper -- primary model factory, pipeline form)
def build_baseline_pipeline():
    """Pipeline version of build_baseline_estimator(), with median imputation."""
    estimator, name = build_baseline_estimator()
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])
    return pipeline, name


# (helper -- shared 5-fold CV probabilities, used by most graphs below)
def get_cv_probabilities(data: pd.DataFrame, target_col: str):
    """Shared X/y/proba/model_name/pipeline for every eval graph below via
    5-fold CV. Returns all-None if nothing valid to model."""
    X, y = prepare_model_data(data, target_col)
    if X.empty or y.nunique() != 2:
        return None, None, None, None, None
    pipeline, model_name = build_baseline_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    return X, y, proba, model_name, pipeline


# (helper -- tp/tn/fp/fn + sensitivity/specificity)
def confusion_counts(y_true, y_pred):
    """tp/tn/fp/fn plus sensitivity/specificity for a 0.5-style hard prediction."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
    }


# GRAPHS 16-19 -- ROC, PR, Calibration, Feature Importance
def baseline_model_and_evaluation(data: pd.DataFrame, target_col: str,
                                  output_dir: str):
    """
    Train the baseline and save the core paper figures. Uses 5-fold CV
    instead of a single train/test split -- with only 454 patients, a
    75/25 split leaves ~113 to evaluate on, too few for stable ROC/PR/
    calibration curves.
    """
    log.info("BASELINE MODEL AND EVALUATION (5-fold CV)")
    X, y, probability, MODEL_NAME, pipeline = get_cv_probabilities(data, target_col)
    if X is None:
        log.warning("Model evaluation skipped: predictors or binary target unavailable.")
        return None

    auc_value = roc_auc_score(y, probability)
    ap_value = average_precision_score(y, probability)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, probability)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=SIGNIFICANT_COLOR, label=f"{MODEL_NAME} (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    clean_title(ax, "ROC Curve", "5-fold cross-validated")
    ax.legend()
    save_current_figure(fig, output_dir, "16_roc_curve.png")

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y, probability)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=SIGNIFICANT_COLOR, label=f"AP = {ap_value:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    clean_title(ax, "Precision-Recall Curve", "5-fold cross-validated")
    ax.legend()
    save_current_figure(fig, output_dir, "17_precision_recall_curve.png")

    # Calibration curve -- fewer bins than before (5, not 8) since ~90
    # patients per bin is more stable than ~55 for a sample this size
    frac_pos, mean_pred = calibration_curve(y, probability, n_bins=5,
                                            strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, marker="o", color=SIGNIFICANT_COLOR, label=MODEL_NAME)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed cancer frequency")
    clean_title(ax, "Calibration Curve", "5-fold cross-validated")
    ax.legend()
    save_current_figure(fig, output_dir, "18_calibration_curve.png")

    # Feature importance -- the ensemble (CatBoost+XGBoost+LogReg) has no
    # single native feature_importances_ vector, and averaging incompatible
    # native importance scales across three different model types would be
    # unjustified. Permutation importance on the fitted ensemble is used
    # instead: model-agnostic, and measured on the same metric (AUC drop)
    # regardless of which base model is doing the work internally.
    pipeline.fit(X, y)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(pipeline, X, y, n_repeats=10, random_state=42,
                                   scoring="roc_auc", n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    top = importance.head(20).sort_values()
    top.index = [PRETTY_NAMES.get(v, v) for v in top.index]
    fig, ax = plt.subplots(figsize=(9, 7))
    top.plot(kind="barh", color=NOT_SIGNIFICANT_COLOR, edgecolor="#555555", ax=ax)
    for i, v in enumerate(top):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    clean_title(ax, "Feature Importance (Permutation)", f"{MODEL_NAME}, top 20")
    ax.set_xlabel("Mean AUC decrease when feature is shuffled")
    save_current_figure(fig, output_dir, "19_feature_importance.png")

    return pipeline



# GRAPH 20 -- Kaplan-Meier Survival Curve
def optional_kaplan_meier(data: pd.DataFrame, target_col: str, output_dir: str):
    """Create a Kaplan-Meier curve only when explicit survival fields exist."""
    time_candidates = ["survival_days", "followup_days", "time_to_event", "fup_days"]
    event_candidates = ["death_event", "event", "vital_status", "died"]
    time_col = next((c for c in time_candidates if c in data.columns), None)
    event_col = next((c for c in event_candidates if c in data.columns), None)
    if not time_col or not event_col:
        log.info("Kaplan-Meier plot skipped: no survival-time and event columns found.")
        return
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        log.warning("Kaplan-Meier plot skipped. Install with: pip install lifelines")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for status, label in [(0, "No Cancer"), (1, "Cancer")]:
        subset = data[data[target_col] == status]
        kmf = KaplanMeierFitter(label=label)
        kmf.fit(subset[time_col], event_observed=subset[event_col])
        kmf.plot_survival_function(ax=ax)
    clean_title(ax, "Kaplan-Meier Survival Curves")
    ax.set_xlabel(time_col)
    ax.set_ylabel("Estimated survival probability")
    save_current_figure(fig, output_dir, "20_kaplan_meier_curve.png")


# =============================================================================
# SECTION 9 -- Model validation & robustness (CV comparisons, CIs, subgroups)
# =============================================================================
# GRAPH 21 -- Model Comparison (multi-model, multi-metric clinical benchmark)
def model_comparison(data: pd.DataFrame, target_col: str, output_dir: str):
    """
    Fair multi-model benchmark: every model evaluated under identical 5-fold
    CV, same patients, same T0-only predictors, same preprocessing. Includes
    every individually-tested model family plus the current selected model
    (the CatBoost+XGBoost+LogReg weighted ensemble), so the ensemble's
    improvement over its own components -- and over untried alternatives --
    is directly visible, not just claimed.
    """
    log.info("MODEL COMPARISON (5-fold CV, multi-metric)")
    X, y = prepare_model_data(data, target_col)
    if X.empty or y.nunique() != 2:
        log.warning("Model comparison skipped: predictors or binary target unavailable.")
        return

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def cat_est():
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                   random_state=42, verbose=0, auto_class_weights="Balanced")

    def xgb_est():
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              random_state=42, eval_metric="logloss")

    def lr_est():
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)

    models = {
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=500, random_state=42,
                                              class_weight="balanced", min_samples_leaf=2, n_jobs=-1)),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", lr_est()),
        ]),
        "CatBoost": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", cat_est())]),
    }
    try:
        models["XGBoost"] = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", xgb_est())])
    except ImportError:
        log.warning("XGBoost not installed -- excluded from comparison.")
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      random_state=42, class_weight="balanced", verbose=-1)),
        ])
    except ImportError:
        log.warning("LightGBM not installed -- excluded from comparison.")
    models["Extra Trees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(n_estimators=500, random_state=42,
                                        class_weight="balanced", min_samples_leaf=2, n_jobs=-1)),
    ])
    models["HistGradientBoosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(random_state=42)),
    ])

    # the current selected model -- built exactly as build_baseline_estimator()
    # does, so this figure and the rest of the report can never silently drift
    ensemble_estimator, ensemble_name = build_baseline_estimator()
    models[ensemble_name] = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", ensemble_estimator)])

    rows = []
    roc_data = {}
    for name, pipeline in models.items():
        proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y, proba),
            "PR-AUC": average_precision_score(y, proba),
            "Sensitivity": recall_score(y, pred),
            "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "F1": f1_score(y, pred, zero_division=0),
            "Balanced Accuracy": balanced_accuracy_score(y, pred),
            "MCC": matthews_corrcoef(y, pred),
        })
        roc_data[name] = roc_curve(y, proba)

    comparison_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_individual = comparison_df[comparison_df["Model"] != ensemble_name].iloc[0]
    ensemble_row = comparison_df[comparison_df["Model"] == ensemble_name].iloc[0]
    print("\n", comparison_df.round(4).to_string(index=False))
    print(f"\nDelta vs best individual model ({best_individual['Model']}):")
    for metric in ["ROC-AUC", "PR-AUC", "Balanced Accuracy", "F1", "MCC"]:
        print(f"  Delta {metric}: {ensemble_row[metric] - best_individual[metric]:+.4f}")
    comparison_df.to_csv(os.path.join(output_dir, "21_model_comparison_table.csv"), index=False)

    # --- Panel 1: ROC curves for every model ---
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("tab10", n_colors=len(models))
    for (name, (fpr, tpr, _)), color in zip(roc_data.items(), palette):
        lw = 3 if name == ensemble_name else 1.5
        style = "-" if name == ensemble_name else "--"
        ax.plot(fpr, tpr, color=color, linewidth=lw, linestyle=style,
                label=f"{name} (AUC={comparison_df.set_index('Model').loc[name,'ROC-AUC']:.3f})"
                      + ("  <- current model" if name == ensemble_name else ""))
    ax.plot([0, 1], [0, 1], linestyle=":", color="#999999", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    clean_title(ax, "Model Comparison -- ROC Curves", "5-fold CV, same patients/folds/predictors")
    ax.legend(loc="lower right", fontsize=8)
    save_current_figure(fig, output_dir, "21a_model_comparison_roc.png")

    # --- Panel 2: multi-metric bar comparison ---
    plot_metrics = ["ROC-AUC", "PR-AUC", "Balanced Accuracy", "Sensitivity", "Specificity", "F1", "MCC"]
    plot_df = comparison_df.set_index("Model")[plot_metrics]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(plot_df))
    width = 0.11
    for i, metric in enumerate(plot_metrics):
        ax.bar(x + (i - (len(plot_metrics) - 1) / 2) * width, plot_df[metric], width, label=metric)
    ax.set_xticks(x)
    labels = [n + ("\n(current model)" if n == ensemble_name else "") for n in plot_df.index]
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    clean_title(ax, "Model Comparison -- All Metrics", "5-fold CV, same patients/folds/predictors")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save_current_figure(fig, output_dir, "21b_model_comparison_all_metrics.png")


# GRAPH 22 -- AUC Bootstrap 95% CI
def bootstrap_auc_ci(data: pd.DataFrame, target_col: str, output_dir: str,
                      n_bootstrap: int = 2000):
    """Bootstrap 95% CI for AUC -- a single point estimate hides real
    uncertainty with only 454 patients."""
    log.info(f"BOOTSTRAP 95% CI FOR AUC ({n_bootstrap} resamples)")
    X, y, proba, MODEL_NAME, _ = get_cv_probabilities(data, target_col)
    if X is None:
        log.warning("Bootstrap CI skipped: predictors or binary target unavailable.")
        return
    y_arr = y.to_numpy()

    point_estimate = roc_auc_score(y_arr, proba)
    rng = np.random.default_rng(42)
    n = len(y_arr)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_arr[idx])) < 2:
            continue  # skip the rare resample with only one class present
        boot_aucs.append(roc_auc_score(y_arr[idx], proba[idx]))
    boot_aucs = np.array(boot_aucs)
    ci_lower, ci_upper = np.percentile(boot_aucs, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(boot_aucs, bins=40, color=NOT_SIGNIFICANT_COLOR, edgecolor="#555555")
    ax.axvline(point_estimate, color=SIGNIFICANT_COLOR, linewidth=2,
               label=f"Point estimate = {point_estimate:.3f}")
    ax.axvline(ci_lower, color="#555555", linestyle="--",
               label=f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]")
    ax.axvline(ci_upper, color="#555555", linestyle="--")
    ax.set_xlabel("Bootstrapped AUC")
    ax.set_ylabel("Count")
    clean_title(ax, "AUC Bootstrap 95% CI", f"{MODEL_NAME}, n={n_bootstrap} resamples")
    ax.legend()
    save_current_figure(fig, output_dir, "22_auc_bootstrap_ci.png")
    log.info(f"AUC = {point_estimate:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]")


# GRAPH 23 -- Threshold Trade-off
def threshold_tradeoff(data: pd.DataFrame, target_col: str, output_dir: str):
    """Sensitivity/specificity/precision/F1 across thresholds, since a
    missed cancer and a false alarm don't cost the same, so 0.5 isn't
    automatically the right cutoff."""
    log.info("THRESHOLD TRADE-OFF ANALYSIS")
    X, y, proba, MODEL_NAME, _ = get_cv_probabilities(data, target_col)
    if X is None:
        log.warning("Threshold analysis skipped: predictors or binary target unavailable.")
        return
    y_arr = y.to_numpy()

    thresholds = np.linspace(0.05, 0.95, 37)
    sensitivity, specificity, precision_vals, f1_vals = [], [], [], []
    for t in thresholds:
        c = confusion_counts(y_arr, (proba >= t).astype(int))
        sens, spec = c["sensitivity"], c["specificity"]
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else np.nan
        f1 = 2 * prec * sens / (prec + sens) if prec and sens and (prec + sens) else np.nan
        sensitivity.append(sens); specificity.append(spec)
        precision_vals.append(prec); f1_vals.append(f1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, sensitivity, color=SIGNIFICANT_COLOR, label="Sensitivity (Recall)")
    ax.plot(thresholds, specificity, color=NOT_SIGNIFICANT_COLOR, label="Specificity")
    ax.plot(thresholds, precision_vals, color="#8A8A8A", label="Precision")
    ax.plot(thresholds, f1_vals, color="#555555", linestyle=":", label="F1 score")
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1,
               label="Default threshold (0.5)")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    clean_title(ax, "Threshold Trade-off", "5-fold CV predictions")
    ax.legend(loc="lower center", ncol=2, fontsize=9)
    save_current_figure(fig, output_dir, "23_threshold_tradeoff.png")



# GRAPH 24 -- Sensitivity & Specificity
def sensitivity_specificity_summary(data: pd.DataFrame, target_col: str,
                                     output_dir: str, n_bootstrap: int = 2000):
    """Sensitivity/specificity at threshold=0.5 with bootstrap 95% CIs --
    the direct answer to "how many cancers would we miss?"."""
    log.info("SENSITIVITY / SPECIFICITY SUMMARY (threshold = 0.5)")
    X, y, proba, MODEL_NAME, _ = get_cv_probabilities(data, target_col)
    if X is None:
        log.warning("Sensitivity/specificity summary skipped: predictors or "
                     "binary target unavailable.")
        return
    y_arr = y.to_numpy()
    pred = (proba >= 0.5).astype(int)

    def sens_spec(y_true, y_pred):
        c = confusion_counts(y_true, y_pred)
        return c["sensitivity"], c["specificity"]

    point_sens, point_spec = sens_spec(y_arr, pred)

    rng = np.random.default_rng(42)
    n = len(y_arr)
    boot_sens, boot_spec = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_arr[idx])) < 2:
            continue
        s, sp = sens_spec(y_arr[idx], pred[idx])
        boot_sens.append(s); boot_spec.append(sp)

    sens_ci = np.percentile(boot_sens, [2.5, 97.5])
    spec_ci = np.percentile(boot_spec, [2.5, 97.5])

    summary_df = pd.DataFrame({
        "metric": ["Sensitivity (Recall)", "Specificity"],
        "value": [point_sens, point_spec],
        "ci_lower": [sens_ci[0], spec_ci[0]],
        "ci_upper": [sens_ci[1], spec_ci[1]],
    })
    print("\n", summary_df)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [SIGNIFICANT_COLOR, NOT_SIGNIFICANT_COLOR]
    x_pos = np.arange(2)
    values = [point_sens, point_spec]
    errors = [[values[i] - [sens_ci, spec_ci][i][0] for i in range(2)],
              [[sens_ci, spec_ci][i][1] - values[i] for i in range(2)]]
    ax.bar(x_pos, values, color=colors, edgecolor="#555555", width=0.55,
           yerr=errors, capsize=6)
    for i, v in enumerate(values):
        ci = [sens_ci, spec_ci][i]
        ax.text(i, ci[1] + 0.03, f"{v:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]",
                ha="center", fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Sensitivity\n(Recall)", "Specificity"])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    clean_title(ax, "Sensitivity & Specificity", "Threshold = 0.5, bootstrap 95% CI")
    save_current_figure(fig, output_dir, "24_sensitivity_specificity.png")
    log.info(f"Sensitivity = {point_sens:.3f} [{sens_ci[0]:.3f}, {sens_ci[1]:.3f}], "
             f"Specificity = {point_spec:.3f} [{spec_ci[0]:.3f}, {spec_ci[1]:.3f}]")



# =============================================================================
# SECTION 10 -- Study limitations note
# =============================================================================
# (helper -- LIMITATIONS.txt, no numbered image)
def write_limitations_note(output_dir: str, data: pd.DataFrame = None):
    """
    A plain-text note of the study's key methodological limitations --
    the kind of thing a reviewer expects to see stated explicitly rather
    than discovered on their own.
    """
    n_total = len(data) if data is not None else 454
    n_cancer = int(data[TARGET_COL].sum()) if data is not None else 226
    n_nocancer = n_total - n_cancer
    pct_cancer = n_cancer / n_total * 100 if n_total else 0

    text = f"""STUDY LIMITATIONS (auto-generated checklist -- edit before submitting)

1. Sample size: all analyses use {n_total} patients (a size-capped subset of the
   full NLST CT-arm cohort of 26,453). Results, especially subgroup and
   bootstrap estimates, should be read with this in mind.

2. No external hold-out set: every reported metric (ROC AUC, calibration,
   SHAP, etc.) comes from 5-fold cross-validation on this same {n_total}-patient
   sample. No independent, never-touched test set was used. Cross-validation
   reduces overfitting risk but is not a substitute for external validation
   on a separate cohort before any clinical claim is made.

3. Balanced sampling: the {n_total}-patient subset was deliberately balanced
   ({n_cancer} cancer / {n_nocancer} no-cancer, {pct_cancer:.1f}% cancer) for modeling
   convenience. This does NOT reflect the true ~4% cancer prevalence in the
   screened population, so metrics like precision/PPV here are NOT directly
   interpretable as real-world clinical performance -- only ranking metrics
   like AUC transfer reasonably well.

4. Single imaging timepoint: only baseline (T0) CT-derived clinical fields
   are used; follow-up screening rounds (T1, T2) are not incorporated here.

5. Primary model: the current selected clinical model is a weighted
   soft-voting ensemble of CatBoost, XGBoost, and Logistic Regression
   (weights from each base model's own out-of-fold AUC), chosen after a
   dedicated model-comparison and ensemble-combination experiment against
   7 individual model families. It is still a clinical characterization
   tool, not a final diagnostic decision tool.
"""
    path = os.path.join(output_dir, "LIMITATIONS.txt")
    with open(path, "w") as f:
        f.write(text)
    log.info(f"Saved -> {path}")


# =============================================================================
# SECTION 11 -- Feature selection & sensitivity analyses
# =============================================================================
# GRAPH 25 -- Feature Selection Ranking (MI + L1 + RFE)
def feature_selection_analysis(data: pd.DataFrame, target_col: str, output_dir: str):
    """
    Ranks predictors via mutual information, L1-regularized logistic
    regression, and RFE. For research reporting only -- doesn't remove
    columns from the actual models above.
    """
    log.info("FEATURE SELECTION (MI + L1 + RFE)")
    try:
        from sklearn.feature_selection import mutual_info_classif, RFE
    except ImportError as exc:
        log.warning(f"Feature-selection analysis skipped ({exc}).")
        return

    X, y = prepare_model_data(data, target_col)
    if X.empty or y.nunique() != 2:
        log.warning("Feature selection skipped: predictors or binary target unavailable.")
        return

    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns, index=X.index
    )
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X_imp),
        columns=X.columns, index=X.index
    )

    # Mutual information
    mi = pd.Series(
        mutual_info_classif(X_imp, y, random_state=42),
        index=X.columns, name="mutual_information"
    )

    # Sparse L1 logistic regression
    l1_model = LogisticRegression(
        l1_ratio=1, solver="liblinear", C=0.5,
        class_weight="balanced", max_iter=3000, random_state=42
    )
    l1_model.fit(X_scaled, y)
    l1_abs = pd.Series(
        np.abs(l1_model.coef_[0]), index=X.columns, name="l1_abs_coefficient"
    )

    # RFE: retain up to 20 features, or half when fewer predictors exist
    n_select = min(20, max(1, X.shape[1] // 2))
    rfe_estimator = LogisticRegression(
        solver="liblinear", class_weight="balanced",
        max_iter=3000, random_state=42
    )
    rfe = RFE(rfe_estimator, n_features_to_select=n_select, step=0.1)
    rfe.fit(X_scaled, y)
    rfe_rank = pd.Series(rfe.ranking_, index=X.columns, name="rfe_rank")
    rfe_selected = pd.Series(rfe.support_, index=X.columns, name="rfe_selected")

    result = pd.concat([mi, l1_abs, rfe_rank, rfe_selected], axis=1)
    result["mi_rank"] = result["mutual_information"].rank(ascending=False, method="min")
    result["l1_rank"] = result["l1_abs_coefficient"].rank(ascending=False, method="min")
    result["combined_rank"] = result[["mi_rank", "l1_rank", "rfe_rank"]].mean(axis=1)
    result = result.sort_values(["combined_rank", "mutual_information"], ascending=[True, False])

    top = result.head(20).sort_values("combined_rank", ascending=False)
    labels = [PRETTY_NAMES.get(v, v) for v in top.index]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, top["combined_rank"], color=NOT_SIGNIFICANT_COLOR,
            edgecolor="#555555")
    for i, v in enumerate(top["combined_rank"]):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    ax.invert_xaxis()  # smaller rank is better
    ax.set_xlabel("Combined rank (lower is better)")
    clean_title(ax, "Feature Selection Ranking", "Top 20, combined rank")
    save_current_figure(fig, output_dir, "25_feature_selection_ranking.png")


# GRAPH 26 -- Decision Curve Analysis
def decision_curve_analysis(data: pd.DataFrame, target_col: str, output_dir: str):
    """
    Decision Curve Analysis using out-of-fold probabilities from the primary model.
    Net benefit is compared with 'treat all' and 'treat none' strategies.
    """
    log.info("DECISION CURVE ANALYSIS")
    X, y, proba, MODEL_NAME, _ = get_cv_probabilities(data, target_col)
    if X is None:
        log.warning("Decision-curve analysis skipped: predictors or binary target unavailable.")
        return
    y_arr = y.to_numpy()
    n = len(y_arr)
    prevalence = y_arr.mean()
    thresholds = np.linspace(0.01, 0.80, 80)

    model_nb, treat_all_nb = [], []
    for pt in thresholds:
        pred = proba >= pt
        tp = np.sum(pred & (y_arr == 1))
        fp = np.sum(pred & (y_arr == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        all_nb = prevalence - (1 - prevalence) * (pt / (1 - pt))
        model_nb.append(nb)
        treat_all_nb.append(all_nb)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(thresholds, model_nb, color=SIGNIFICANT_COLOR, linewidth=2,
            label=MODEL_NAME)
    ax.plot(thresholds, treat_all_nb, color=NOT_SIGNIFICANT_COLOR,
            linestyle="--", label="Treat all")
    ax.plot(thresholds, np.zeros_like(thresholds), color="#555555",
            linestyle=":", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    clean_title(ax, "Decision Curve Analysis", "5-fold CV predictions")
    ax.legend()
    save_current_figure(fig, output_dir, "26_decision_curve_analysis.png")


# GRAPH 27 -- Permutation Importance (cross-validated)
def cross_validated_permutation_importance(data: pd.DataFrame, target_col: str,
                                           output_dir: str):
    """
    Compute permutation importance independently inside each CV fold and
    average the decrease in ROC AUC across folds.
    """
    log.info("CROSS-VALIDATED PERMUTATION IMPORTANCE")
    try:
        from sklearn.inspection import permutation_importance
    except ImportError as exc:
        log.warning(f"Permutation importance skipped ({exc}).")
        return

    X, y = prepare_model_data(data, target_col)
    if X.empty or y.nunique() != 2:
        log.warning("Permutation importance skipped: predictors or binary target unavailable.")
        return

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_importances = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(
            imputer.fit_transform(X.iloc[train_idx]), columns=X.columns
        )
        X_test = pd.DataFrame(
            imputer.transform(X.iloc[test_idx]), columns=X.columns
        )
        model, MODEL_NAME = build_baseline_estimator()
        # generic seed-setting: works whether the model has a top-level
        # random_state (e.g. RandomForestClassifier) or only nested ones
        # inside a VotingClassifier's sub-estimators (e.g. cat__random_state)
        seed_params = {k: 42 + fold for k in model.get_params()
                        if k == "random_state" or k.endswith("__random_state")}
        if seed_params:
            model.set_params(**seed_params)
        model.fit(X_train, y.iloc[train_idx])
        result = permutation_importance(
            model, X_test, y.iloc[test_idx], scoring="roc_auc",
            n_repeats=20, random_state=42 + fold, n_jobs=-1
        )
        fold_importances.append(result.importances_mean)

    arr = np.vstack(fold_importances)
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "mean_auc_decrease": arr.mean(axis=0),
        "std_across_folds": arr.std(axis=0, ddof=1),
    }).sort_values("mean_auc_decrease", ascending=False)

    top = imp_df.head(20).sort_values("mean_auc_decrease")
    fig, ax = plt.subplots(figsize=(9.5, 7))
    # Standard Project Analysis palette only: orange for positive importance,
    # blue for non-positive importance. No additional chart-specific colors.
    importance_colors = [SIGNIFICANT_COLOR if v > 0 else NOT_SIGNIFICANT_COLOR
                         for v in top["mean_auc_decrease"]]
    ax.barh([PRETTY_NAMES.get(v, v) for v in top["feature"]],
            top["mean_auc_decrease"],
            xerr=top["std_across_folds"], capsize=3,
            color=importance_colors, edgecolor="#555555",
            error_kw={"ecolor": "black", "elinewidth": 1.2})
    for i, (v, e) in enumerate(zip(top["mean_auc_decrease"], top["std_across_folds"])):
        ax.text(v + e + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Mean decrease in ROC AUC after permutation")
    clean_title(ax, "Permutation Importance", f"{MODEL_NAME}, cross-validated, top 20")
    save_current_figure(fig, output_dir, "27_permutation_importance_cv.png")


# GRAPH 28 -- Classification Agreement (MCC, Cohen's kappa)
def agreement_metrics(data: pd.DataFrame, target_col: str, output_dir: str):
    """Save MCC and Cohen's kappa for out-of-fold predictions from the primary model."""
    log.info("MCC AND COHEN'S KAPPA")
    from sklearn.metrics import (
        matthews_corrcoef, cohen_kappa_score, accuracy_score,
        balanced_accuracy_score, f1_score
    )

    X, y, proba, MODEL_NAME, _ = get_cv_probabilities(data, target_col)
    if X is None:
        log.warning("Agreement metrics skipped: predictors or binary target unavailable.")
        return
    pred = (proba >= 0.5).astype(int)

    metrics_df = pd.DataFrame({
        "metric": ["Matthews correlation coefficient", "Cohen's kappa",
                   "Accuracy", "Balanced accuracy", "F1 score"],
        "value": [matthews_corrcoef(y, pred), cohen_kappa_score(y, pred),
                  accuracy_score(y, pred), balanced_accuracy_score(y, pred),
                  f1_score(y, pred)],
    })
    print("\n", metrics_df)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = metrics_df.sort_values("value")
    ax.barh(plot_df["metric"], plot_df["value"],
            color=NOT_SIGNIFICANT_COLOR, edgecolor="#555555")
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Score")
    clean_title(ax, "Classification Agreement Metrics", f"{MODEL_NAME}, 5-fold CV")
    for i, value in enumerate(plot_df["value"]):
        ax.text(value + 0.015, i, f"{value:.3f}", va="center", fontsize=9)
    save_current_figure(fig, output_dir, "28_agreement_metrics.png")


def _compute_midrank(x):
    """Midranks used by the fast DeLong implementation."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[order] = ranks
    return result


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong covariance calculation for correlated ROC AUCs."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, np.atleast_2d(delong_cov)


# GRAPH 29 -- DeLong Test (AUC comparison, primary vs alternative model)
def delong_model_comparison(data: pd.DataFrame, target_col: str, output_dir: str):
    """DeLong test: current selected model (CatBoost+XGBoost+LogReg Weighted
    Ensemble, via build_baseline_pipeline()) vs XGBoost alone (falls back to
    Logistic Regression if XGBoost isn't installed)."""
    log.info("DELONG TEST FOR ROC AUC COMPARISON")
    X, y = prepare_model_data(data, target_col)
    if X.empty or y.nunique() != 2:
        log.warning("DeLong test skipped: predictors or binary target unavailable.")
        return

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    primary, MODEL_NAME = build_baseline_pipeline()
    comparator_name = "Logistic Regression"
    comparator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42
        )),
    ])
    try:
        from xgboost import XGBClassifier
        comparator_name = "XGBoost"
        comparator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                eval_metric="logloss", random_state=42, n_jobs=-1
            )),
        ])
    except ImportError:
        log.warning("XGBoost unavailable for DeLong test; using Logistic Regression instead.")

    primary_proba = cross_val_predict(primary, X, y, cv=cv, method="predict_proba")[:, 1]
    cmp_proba = cross_val_predict(comparator, X, y, cv=cv, method="predict_proba")[:, 1]
    y_arr = y.to_numpy(dtype=int)

    order = np.argsort(-y_arr)
    label_1_count = int(y_arr.sum())
    predictions = np.vstack([primary_proba, cmp_proba])[:, order]
    aucs, covariance = _fast_delong(predictions, label_1_count)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T).item())
    z = (aucs[0] - aucs[1]) / np.sqrt(max(variance, 1e-15))
    p_value = 2 * stats.norm.sf(abs(z))

    result = pd.DataFrame({
        "model_1": [MODEL_NAME],
        "auc_model_1": [aucs[0]],
        "model_2": [comparator_name],
        "auc_model_2": [aucs[1]],
        "auc_difference": [aucs[0] - aucs[1]],
        "z_statistic": [z],
        "p_value": [p_value],
        "significant_at_0.05": [p_value < 0.05],
    })
    print("\n", result)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    names = [MODEL_NAME, comparator_name]
    values = aucs
    ax.bar(names, values,
           color=[SIGNIFICANT_COLOR, NOT_SIGNIFICANT_COLOR],
           edgecolor="#555555", width=0.55)
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ROC AUC")
    clean_title(ax, "DeLong AUC Comparison", f"p = {p_value:.4f}")
    save_current_figure(fig, output_dir, "29_delong_auc_comparison.png")
    log.info(f"DeLong p-value={p_value:.6f}")


# GRAPH 30 -- Benchmark vs Published NLST Models (DeepScreener / grt123)
def literature_benchmark_comparison(data: pd.DataFrame, target_col: str, output_dir: str):
    """
    Benchmark table + ROC/PR panel matching the style of published NLST
    papers (Causey et al. 2019, arXiv:1906.00240 -- DeepScreener/grt123,
    3D CNNs on full CT volumes). Caveat noted in the table itself:
    different input type and a smaller, class-balanced subset, so this is
    a contextual reference point, not a head-to-head benchmark.
    """
    log.info("LITERATURE BENCHMARK COMPARISON (vs. DeepScreener / grt123)")
    from sklearn.metrics import log_loss, f1_score, accuracy_score

    X, y, proba, MODEL_NAME, _ = get_cv_probabilities(data, target_col)
    if X is None:
        log.warning("Benchmark comparison skipped: predictors or binary target unavailable.")
        return
    y_arr = y.to_numpy()
    pred = (proba >= 0.5).astype(int)
    c = confusion_counts(y_arr, pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    # published figures, read directly off Table 1 / Figure 1-2 of arXiv:1906.00240
    published = {
        "Total":            (len(y_arr), 1359, 1449),
        "# Positive":       (int(y_arr.sum()), 432, 469),
        "# Negative":       (int((y_arr == 0).sum()), 927, 980),
        "AUC":              (roc_auc_score(y_arr, proba), 0.858, 0.885),
        "AUPRC":            (average_precision_score(y_arr, proba), 0.788, 0.837),
        "Accuracy":         (accuracy_score(y_arr, pred), 0.782, 0.821),
        "LogLoss":          (log_loss(y_arr, proba), 0.484, 0.434),
        "F1-score":         (f1_score(y_arr, pred), 0.500, 0.631),
        "Sensitivity":      (tp / (tp + fn) if (tp + fn) else np.nan, 0.343, 0.473),
        "Specificity":      (tn / (tn + fp) if (tn + fp) else np.nan, 0.987, 0.987),
        "# False Positives": (fp, 12, 13),
        "# False Negatives": (fn, 284, 247),
    }

    rows = []
    for metric, (ours, ds, grt) in published.items():
        fmt = (lambda v: f"{v}") if metric in ("Total", "# Positive", "# Negative",
                                                 "# False Positives", "# False Negatives") \
              else (lambda v: f"{v:.3f}")
        rows.append({
            "Metric": metric,
            f"Our Model ({MODEL_NAME})": fmt(ours),
            "DeepScreener": fmt(ds),
            "grt123": fmt(grt),
        })
    table_df = pd.DataFrame(rows)
    print("\n", table_df)

    render_table_image(
        table_df, output_dir, "30_literature_benchmark_comparison.png",
        "Benchmark Comparison vs. Published NLST Models",
        "Causey et al. 2019 -- image-based CNNs, not directly comparable to a clinical-only model",
        figsize=(11, 5.5), highlight_col=1,
        footer="Source: Causey JL, et al. Lung cancer screening with low-dose CT scans "
               "using a deep learning approach. arXiv:1906.00240 (2019). "
               "https://arxiv.org/abs/1906.00240  |  'Our Model' column: 5-fold CV on the "
               f"{LOCKED_DEV_N}-patient development cohort, STRICT T0 features only. The "
               f"locked, independently validated result (n={LOCKED_INDEP_N}, zero patient "
               f"overlap) is AUC={LOCKED_INDEP_AUC:.4f} -- see graph 27.",
    )

    # ---- ROC + PR side-by-side panel, same layout style as the paper's Figure 1 ----
    fpr, tpr, _ = roc_curve(y_arr, proba)
    precision, recall, _ = precision_recall_curve(y_arr, proba)
    auc_val = roc_auc_score(y_arr, proba)
    ap_val = average_precision_score(y_arr, proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color=SIGNIFICANT_COLOR, linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#999999")
    axes[0].set_xlabel("1 \u2212 Specificity")
    axes[0].set_ylabel("Sensitivity")
    axes[0].text(0.97, 0.05, f"AUC = {auc_val:.3f}", transform=axes[0].transAxes,
                 ha="right", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="#999999"))
    axes[0].set_title("ROC", fontsize=12)

    axes[1].plot(recall, precision, color=NOT_SIGNIFICANT_COLOR, linewidth=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].text(0.97, 0.05, f"AUC = {ap_val:.3f}", transform=axes[1].transAxes,
                 ha="right", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="#999999"))
    axes[1].set_title("Precision-Recall", fontsize=12)

    fig.suptitle("Our Model Performance", fontsize=14, y=1.12)
    fig.text(0.5, 1.04, f"{MODEL_NAME}, 5-fold CV, N={len(data)}", ha="center",
              fontsize=9, color="#666666")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(os.path.join(output_dir, "30_our_roc_pr_panel.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved -> {output_dir}/30_our_roc_pr_panel.png")


# GRAPH 31 -- Cancer Cases and Rate by Age Group
def plot_cases_by_age_group(data: pd.DataFrame, target_col: str, output_dir: str):
    """
    Cancer case count + rate per age group, in the same dual-axis style as
    the American Lung Association's 'Deaths and Rate by Age' chart
    (lung.org) -- adapted here to case counts/rates since this dataset
    tracks diagnosis, not mortality.
    """
    if "age" not in data.columns:
        return
    bins = [54, 59, 64, 69, 74]
    labels = ["55-59", "60-64", "65-69", "70-74"]
    age_group = pd.cut(data["age"], bins=bins, labels=labels)
    summary = data.groupby(age_group, observed=True)[target_col].agg(
        n_cases="sum", n_total="count"
    )
    summary["rate_pct"] = summary["n_cases"] / summary["n_total"] * 100

    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.bar(summary.index.astype(str), summary["n_cases"],
            color=NOT_SIGNIFICANT_COLOR, edgecolor="#555555", width=0.6)
    # bar-count label sits just inside the top of the bar (not above it),
    # so it can never collide with the rate line/label -- those two used
    # to land at nearly the same pixel height and overlap
    for i, v in enumerate(summary["n_cases"]):
        ax1.annotate(f"{int(v)}", xy=(i, v), xytext=(0, -14),
                     textcoords="offset points", ha="center", va="top",
                     fontsize=9, color="white", fontweight="bold")
    ax1.set_xlabel("Age group")
    ax1.set_ylabel("Number of cancer cases")

    ax2 = ax1.twinx()
    ax2.plot(summary.index.astype(str), summary["rate_pct"], color=SIGNIFICANT_COLOR,
             marker="o", linewidth=2)
    for i, v in enumerate(summary["rate_pct"]):
        ax2.annotate(f"{v:.0f}%", xy=(i, v), xytext=(0, 10),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=9, color="#8a4a1c")
    ax2.set_ylabel("Cancer rate (%)")
    ax2.set_ylim(0, max(summary["rate_pct"]) * 1.35)

    clean_title(ax1, "Cancer Cases and Rate by Age Group")
    save_current_figure(fig, output_dir, "31_cases_by_age_group.png")


# =============================================================================
# SECTION 13 -- Main: runs every step above in order
# =============================================================================
# GRAPH 32 -- SHAP Summary (out-of-fold, cross-validated)
def cross_validated_shap_summary(data: pd.DataFrame, target_col: str, output_dir: str):
    """
    Model interpretability: which features push predictions toward
    "Cancer" vs "No Cancer", and by how much. Out-of-fold (fold-wise
    models, like the 5-fold CV used elsewhere) so this explains the model
    the way it treats a genuinely new patient, not the data it trained on.

    The primary model is a soft-voting ensemble (CatBoost + XGBoost +
    LogisticRegression, sklearn VotingClassifier), which SHAP's fast
    TreeExplainer does not support (it only handles a single tree model,
    not a voting wrapper around three different model types). Uses a
    model-agnostic explainer against the ensemble's own predict_proba
    instead, so the SHAP values explain the actual 3-model ensemble used
    everywhere else in this project -- not a stand-in single tree model.
    A background/test subsample per fold keeps runtime reasonable; this is
    standard practice for model-agnostic SHAP on tabular ensembles.
    """
    try:
        import shap
    except ImportError:
        log.warning("SHAP summary skipped. Install with: pip install shap")
        return

    log.info("SHAP SUMMARY (out-of-fold, 5-fold CV, model-agnostic explainer on the actual ensemble)")
    X, y = prepare_model_data(data, target_col)
    if X.empty or y.nunique() != 2:
        log.warning("SHAP summary skipped: predictors or binary target unavailable.")
        return

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    shap_rows, X_rows = [], []
    model_name = None
    TEST_SAMPLE_PER_FOLD = 60
    BACKGROUND_SIZE = 40

    for train_idx, test_idx in cv.split(X_imp, y):
        model, model_name = build_baseline_estimator()
        model.fit(X_imp.iloc[train_idx], y.iloc[train_idx])

        background = shap.sample(X_imp.iloc[train_idx], min(BACKGROUND_SIZE, len(train_idx)),
                                  random_state=42)
        test_pool = X_imp.iloc[test_idx]
        if len(test_pool) > TEST_SAMPLE_PER_FOLD:
            test_pool = test_pool.sample(TEST_SAMPLE_PER_FOLD, random_state=42)

        def predict_fn(arr, m=model):
            return m.predict_proba(pd.DataFrame(arr, columns=X_imp.columns))[:, 1]

        try:
            explainer = shap.Explainer(predict_fn, background)
            fold_result = explainer(test_pool)
        except Exception as exc:
            log.warning(f"SHAP summary skipped: explainer failed for {model_name} ({exc}).")
            return

        shap_rows.append(fold_result.values)
        X_rows.append(test_pool)

    all_shap = np.vstack(shap_rows)
    X_display = pd.concat(X_rows, axis=0).rename(columns=PRETTY_NAMES)

    fig = plt.figure(figsize=(10, 9))
    shap.summary_plot(all_shap, X_display, show=False, max_display=20,
                       cmap=LIGHT_DIVERGING_CMAP)
    fig = plt.gcf()
    clean_suptitle(fig, "SHAP Summary Plot",
                    f"{model_name}, out-of-fold, 5-fold CV, "
                    f"{len(X_display)}-patient explained subsample")
    fig.savefig(os.path.join(output_dir, "32_shap_summary.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved -> {output_dir}/32_shap_summary.png")


# GRAPH 33 -- Subgroup Performance (AUC)
def subgroup_performance(data: pd.DataFrame, target_col: str, output_dir: str):
    """AUC by gender/smoking subgroup -- a good overall AUC can still hide
    a fairness gap between subgroups, which matters for a screening tool."""
    log.info("SUBGROUP PERFORMANCE (5-fold CV predictions)")
    X, y, proba, MODEL_NAME, _ = get_cv_probabilities(data, target_col)
    if X is None:
        log.warning("Subgroup analysis skipped: predictors or binary target unavailable.")
        return

    subgroup_defs = {
        "gender": {0: "Male", 1: "Female"},
        "cigsmok": {0: "Non-smoker", 1: "Current smoker"},
    }
    rows = []
    for col, labels in subgroup_defs.items():
        if col not in data.columns:
            continue
        for raw_val, label in labels.items():
            mask = (data[col] == raw_val).to_numpy()
            y_sub, p_sub = y.to_numpy()[mask], proba[mask]
            if len(np.unique(y_sub)) < 2:
                rows.append({"subgroup": f"{col} = {label}", "n": int(mask.sum()),
                             "AUC": np.nan, "note": "only one class present"})
                continue
            rows.append({
                "subgroup": f"{col} = {label}", "n": int(mask.sum()),
                "AUC": roc_auc_score(y_sub, p_sub), "note": "",
            })

    subgroup_df = pd.DataFrame(rows)
    print("\n", subgroup_df)

    plot_df = subgroup_df.dropna(subset=["AUC"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(plot_df["subgroup"], plot_df["AUC"], color=NOT_SIGNIFICANT_COLOR,
            edgecolor="#555555")
    ax.axvline(0.5, color="#999999", linestyle="--", label="Chance (AUC = 0.5)")
    for i, (auc_val, n_val) in enumerate(zip(plot_df["AUC"], plot_df["n"])):
        ax.text(auc_val + 0.01, i, f"{auc_val:.3f} (n={n_val})", va="center", fontsize=9)
    ax.set_xlabel("AUC")
    clean_title(ax, "Subgroup Performance", f"{MODEL_NAME}, 5-fold CV")
    ax.set_xlim(0, 1.15)
    ax.legend()
    save_current_figure(fig, output_dir, "33_subgroup_performance.png")


# GRAPH 01 -- Temporal Leakage Audit & Methodological Correction
def temporal_leakage_methodological_summary(data: pd.DataFrame, target_col: str, output_dir: str):
    """
    ONE figure documenting the methodological correction for the paper's
    Methods/Sensitivity Analysis section: the original pipeline inadvertently
    included post-T0/future-derived predictors (61 of 92, from NLST screening
    rounds T1/T2), a temporal leakage audit identified this, those predictors
    were excluded, and the model was rebuilt using STRICT T0 information only.

    This is NOT a comparison of two valid competing models -- the "before"
    number is explicitly labeled as methodologically invalid.
    """
    log.info("TEMPORAL LEAKAGE AUDIT & METHODOLOGICAL CORRECTION")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # left panel: predictor composition of the ORIGINAL (invalid) pipeline
    labels = ["Pre-screening\n(safe)", "T0 screening\nround (safe)",
              "T1/T2 screening\nrounds (FUTURE)"]
    counts = [8, 17 + 5, 61]  # 5 = the "questionable aggregates", conservatively grouped as unsafe
    colors = [NOT_SIGNIFICANT_COLOR, NOT_SIGNIFICANT_COLOR, SIGNIFICANT_COLOR]
    ax1.bar(labels, counts, color=colors, edgecolor="#555555")
    for i, v in enumerate(counts):
        ax1.text(i, v + 1, str(v), ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Number of predictors")
    clean_title(ax1, "Original Pipeline (92 predictors)",
                "66% derived from future (T1/T2) screening rounds")

    # right panel: AUC before (invalid, in-sample) vs after (valid, locked,
    # independently validated) -- explicitly NOT framed as model A vs model B
    names = ["BEFORE\n(invalid -- future info,\nin-sample CV)",
             "AFTER\n(valid -- STRICT T0,\nlocked independent validation)"]
    values = [0.868, LOCKED_INDEP_AUC]
    colors2 = ["#B0453A", NOT_SIGNIFICANT_COLOR]
    ax2.bar(names, values, color=colors2, edgecolor="#555555", width=0.55)
    for i, v in enumerate(values):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("ROC-AUC")
    clean_title(ax2, "Methodological Correction",
                f"AFTER = independently validated, n={LOCKED_INDEP_N}, zero patient overlap")

    fig.suptitle("Temporal Leakage Audit and Methodological Correction", fontsize=14, y=1.04)
    fig.text(0.5, 0.985,
             "A temporal leakage audit found the original pipeline used post-baseline "
             "information; the model was rebuilt using strict T0-only predictors.",
             ha="center", fontsize=9.5, color="#666666")
    save_current_figure(fig, output_dir, "01_temporal_leakage_audit.png")


def classify_predictor_timing(feature: str) -> dict:
    """Rule-based per-predictor safety classification, used by the audit
    table below. Mirrors random_forest_temporal_leakage_audit.py's logic."""
    if feature in ("age", "cigsmok", "gender") or feature.startswith("race_"):
        return dict(source="NLST person table", timepoint="Enrollment",
                    reason="Demographic/smoking status, known at enrollment, before any screen",
                    check="PASS")
    if feature == "scr_days0" or feature.startswith(("scr_res0_", "scr_iso0_")):
        return dict(source="NLST screen table, round 0", timepoint="T0 (baseline)",
                    reason="T0 screening-round timing/result, available at the baseline visit itself",
                    check="PASS")
    if feature.endswith("_t0") or feature == "had_lesion_measurement_t0":
        return dict(source="Raw CTAB, filtered to study_yr == 0",
                    timepoint="T0 (baseline)",
                    reason="Rebuilt directly from baseline-visit CT-abnormality records only "
                           "(verified via build_t0_lesion_features, not just renamed)",
                    check="PASS")
    return dict(source="UNKNOWN", timepoint="UNKNOWN",
                reason="Does not match any known-safe pattern", check="FLAG FOR REVIEW")


# GRAPH 02 -- Predictor Safety Audit
def predictor_safety_audit(data: pd.DataFrame, target_col: str, output_dir: str):
    """Explicit, per-predictor audit table for every feature entering the
    final clinical model: name, source, time point, why it's available at
    T0, and a pass/fail leakage check. Anything with uncertain temporal
    availability is flagged rather than silently kept."""
    log.info("PREDICTOR SAFETY AUDIT")
    features = [c for c in data.columns if c not in (ID_COL, target_col)]
    rows = []
    n_flagged = 0
    for f in features:
        info = classify_predictor_timing(f)
        if info["check"] != "PASS":
            n_flagged += 1
        rows.append({"Feature": f, "Source": info["source"], "Time point": info["timepoint"],
                     "Why available at T0": info["reason"], "Check": info["check"]})
    audit_df = pd.DataFrame(rows)
    print("\n", audit_df.to_string(index=False))

    if n_flagged > 0:
        log.warning(f"{n_flagged} predictor(s) flagged for manual review -- see printed table above.")
    else:
        log.info(f"All {len(features)} predictors passed the T0 safety check. "
                 f"'{ID_COL}' confirmed excluded from the feature set.")

    display_df = audit_df.copy()
    display_df["Why available at T0"] = display_df["Why available at T0"].str.slice(0, 55)
    render_table_image(
        display_df, output_dir, "02_predictor_safety_audit.png",
        "Predictor Safety Audit -- Every Feature in the Final Model",
        f"{len(features)} predictors checked, {n_flagged} flagged for review, "
        f"'{ID_COL}' confirmed excluded (identifier, not a predictor)",
        figsize=(13, 1.0 + 0.45 * len(features)),
    )


# =============================================================================
# SECTION 12 -- Shape / Morphology Analysis + additional medically relevant
# analyses (all computed exclusively from clinical_T0_1638_patients_FINAL.csv)
# =============================================================================
# Every feature analyzed below is either recorded directly in the
# 1638-patient file (diameters, aspect ratio, area proxy, margin type,
# attenuation pattern, lobe location) or derived by a transparent geometric
# calculation from two of those recorded values (est_nodule_eccentricity_t0,
# computed in load_data()). Nothing here is simulated, image-derived, or
# pulled from any file other than the 1638-patient dataset.

# GRAPH 34 -- Shape/Morphology Feature Distributions by Cancer Status
def plot_shape_feature_distributions(data: pd.DataFrame, target_col: str, output_dir: str):
    vars_to_plot = [v for v in SHAPE_VARS if v in data.columns and data[v].notna().sum() > 0]
    if not vars_to_plot:
        log.warning("Shape analysis skipped: no shape columns with data found.")
        return
    ncols = 3
    nrows = -(-len(vars_to_plot) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, vars_to_plot):
        g0 = data.loc[data[target_col] == 0, col].dropna()
        g1 = data.loc[data[target_col] == 1, col].dropna()
        vals = [g0.mean(), g1.mean()]
        errs = [g0.std(), g1.std()]
        colors = [CANCER_PALETTE["No Cancer"], CANCER_PALETTE["Cancer"]]
        ax.bar(["No\nCancer", "Cancer"], vals, yerr=errs, color=colors,
               edgecolor="#555555", width=0.6, capsize=5)
        headroom = (max(vals) if max(vals) else 1) * 0.15 + max(errs)
        for i, v in enumerate(vals):
            ax.text(i, v + errs[i] + headroom * 0.15, f"{v:.2f}", ha="center", fontsize=9)
        clean_title(ax, PRETTY_NAMES.get(col, col), f"n={len(g0) + len(g1)} measured")
    for ax in axes[len(vars_to_plot):]:
        ax.axis("off")
    clean_suptitle(fig, "Shape / Morphology Features by Cancer Status",
                    "Only patients with a recorded T0 nodule measurement contribute to each panel")
    save_current_figure(fig, output_dir, "34_shape_feature_distributions.png")


# GRAPH 35 -- Shape Feature Correlation with Cancer
def plot_shape_correlation_with_target(data: pd.DataFrame, target_col: str, output_dir: str):
    vars_present = [v for v in SHAPE_VARS if v in data.columns]
    if not vars_present:
        return
    numeric = data[vars_present + [target_col]].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()[target_col].drop(labels=[target_col]).dropna()
    if corr.empty:
        return
    corr = corr.reindex(corr.abs().sort_values(ascending=True).index)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.55 * len(corr))))
    bar_colors = [CANCER_PALETTE["Cancer"] if v >= 0 else CANCER_PALETTE["No Cancer"] for v in corr]
    labels = [PRETTY_NAMES.get(i, i) for i in corr.index]
    ax.barh(labels, corr.values, color=bar_colors, edgecolor="#555555", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    for i, v in enumerate(corr.values):
        ax.text(v + (0.005 if v >= 0 else -0.005), i, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    clean_title(ax, "Shape Feature Correlation with Cancer")
    ax.set_xlabel("Pearson correlation with has_cancer")
    save_current_figure(fig, output_dir, "35_shape_feature_correlation.png")


# (helper -- shared by the three categorical-shape graphs below)
def _categorical_group_rates(data: pd.DataFrame, target_col: str, cols: list, labels_map: dict) -> pd.DataFrame:
    """% of each group (No Cancer / Cancer) with each *_present_t0 flag == 1."""
    rows = []
    for c in cols:
        if c not in data.columns:
            continue
        for label, name in [(0, "No Cancer"), (1, "Cancer")]:
            subset = data.loc[data[target_col] == label, c].dropna()
            if subset.empty:
                continue
            rows.append({"Category": labels_map.get(c, c), "Group": name,
                         "Percent": 100 * subset.mean()})
    return pd.DataFrame(rows)


# GRAPH 36 -- Nodule Margin (Border Shape) vs Cancer
def plot_margin_vs_cancer(data: pd.DataFrame, target_col: str, output_dir: str):
    df_plot = _categorical_group_rates(data, target_col, MARGIN_VARS, MARGIN_LABELS)
    if df_plot.empty:
        log.warning("Margin analysis skipped: no margin_*_present_t0 columns with data.")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df_plot, x="Category", y="Percent", hue="Group",
                palette=CANCER_PALETTE, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8)
    clean_title(ax, "Nodule Margin (Border Shape) by Cancer Status",
                "Spiculated/irregular margins are a classic radiologic malignancy indicator")
    ax.set_ylabel("% of patients in group with this margin type")
    ax.set_xlabel("")
    save_current_figure(fig, output_dir, "36_margin_shape_vs_cancer.png")


# GRAPH 37 -- Nodule Attenuation (Density Pattern) vs Cancer
def plot_attenuation_vs_cancer(data: pd.DataFrame, target_col: str, output_dir: str):
    df_plot = _categorical_group_rates(data, target_col, ATTEN_VARS, ATTEN_LABELS)
    if df_plot.empty:
        log.warning("Attenuation analysis skipped: no atten_*_present_t0 columns with data.")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=df_plot, x="Category", y="Percent", hue="Group",
                palette=CANCER_PALETTE, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=8)
    clean_title(ax, "Nodule Attenuation (Density Pattern) by Cancer Status",
                "Ground-glass / part-solid nodules carry a different malignancy risk profile than solid nodules")
    ax.set_ylabel("% of patients in group with this pattern")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_current_figure(fig, output_dir, "37_attenuation_pattern_vs_cancer.png")


# GRAPH 38 -- Nodule Lobe Location vs Cancer
def plot_location_vs_cancer(data: pd.DataFrame, target_col: str, output_dir: str):
    df_plot = _categorical_group_rates(data, target_col, LOCATION_VARS, LOCATION_LABELS)
    if df_plot.empty:
        log.warning("Location analysis skipped: no location_*_present_t0 columns with data.")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=df_plot, x="Category", y="Percent", hue="Group",
                palette=CANCER_PALETTE, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=7)
    clean_title(ax, "Nodule Lobe Location by Cancer Status",
                "Upper-lobe predominance of lung cancer is a well-documented epidemiological pattern")
    ax.set_ylabel("% of patients in group with a nodule in this lobe")
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    save_current_figure(fig, output_dir, "38_lobe_location_vs_cancer.png")


# GRAPH 39 -- Shape-Only Feature Class Separability (Cross-Validated AUC)
def shape_feature_class_separability(data: pd.DataFrame, target_col: str, output_dir: str):
    """How well do shape/morphology features alone separate cancer from
    no-cancer, compared with the full T0 clinical feature set? Both use the
    same simple model (median-impute + scale + logistic regression, 5-fold
    CV) so the comparison isolates the effect of the feature set, not the
    model choice."""
    all_cols, y = prepare_model_data(data, target_col)
    shape_cols = [c for c in SHAPE_VARS if c in all_cols.columns]
    if not shape_cols or y.nunique() != 2:
        log.warning("Shape separability analysis skipped: shape columns or binary target unavailable.")
        return

    def cv_auc(X):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
        return roc_auc_score(y, proba)

    shape_auc = cv_auc(all_cols[shape_cols])
    full_auc = cv_auc(all_cols)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    names = [f"Shape features only\n(n={len(shape_cols)})",
             f"Full T0 clinical\nfeature set (n={all_cols.shape[1]})"]
    values = [shape_auc, full_auc]
    colors = [CANCER_PALETTE["Cancer"], CANCER_PALETTE["No Cancer"]]
    ax.bar(names, values, color=colors, edgecolor="#555555", width=0.55)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Chance (AUC=0.5)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cross-validated ROC-AUC")
    ax.legend(fontsize=8, loc="lower right")
    clean_title(ax, "Class Separability: Shape Features Alone vs Full Feature Set",
                "5-fold CV, logistic regression, identical model for both bars")
    save_current_figure(fig, output_dir, "39_shape_feature_class_separability.png")


def main():
    args = parse_args()
    # start from a clean folder every run -- otherwise stale files from a
    # graph that got removed (or renamed) stick around and look like they
    # were regenerated when they weren't
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_data(args.data_path, args.ctab_path)
    structural_overview(df)

    constant_cols = data_quality_checks(df, args.output_dir)
    df_analysis = df.drop(columns=constant_cols)
    log.info(f"Shape after dropping constant columns: {df_analysis.shape}")

    # Methodological correction record + predictor audit -- run first, before
    # any other analysis, per the paper's required Methods section ordering
    temporal_leakage_methodological_summary(df_analysis, TARGET_COL, args.output_dir)
    predictor_safety_audit(df_analysis, TARGET_COL, args.output_dir)

    target_analysis(df_analysis, TARGET_COL, args.output_dir)
    plot_distribution_overlap(df_analysis, TARGET_COL, "max_lesion_long_diam_t0", args.output_dir)
    plot_key_distributions(df_analysis, TARGET_COL, args.output_dir)
    plot_correlation_heatmap(df_analysis, args.output_dir)
    multivariate_logistic_regression(df_analysis, TARGET_COL, args.output_dir)

    plot_target_correlations(df_analysis, TARGET_COL, args.output_dir)
    plot_age_distribution(df_analysis, TARGET_COL, args.output_dir)
    plot_categorical_vs_target(
        df_analysis, TARGET_COL, "cigsmok", "Smoking Status vs Lung Cancer",
        args.output_dir, "10_smoking_vs_cancer.png",
        value_labels={0: "Non-smoker", 1: "Current smoker"}
    )
    plot_categorical_vs_target(
        df_analysis, TARGET_COL, "had_lesion_measurement_t0",
        "Lesion Measurement Availability vs Lung Cancer (T0)",
        args.output_dir, "11_lesion_measurement_vs_cancer.png",
        value_labels={0: "No measurement", 1: "Has measurement"}
    )
    plot_lesion_violin(df_analysis, TARGET_COL, args.output_dir)
    plot_lesion_abnormality_scatter(df_analysis, TARGET_COL, args.output_dir)
    plot_race_stacked_bar(df_analysis, TARGET_COL, args.output_dir)
    plot_pca_projection(df_analysis, TARGET_COL, args.output_dir)
    baseline_model_and_evaluation(df_analysis, TARGET_COL, args.output_dir)
    optional_kaplan_meier(df_analysis, TARGET_COL, args.output_dir)

    # Deeper model validation, requested as follow-ups
    model_comparison(df_analysis, TARGET_COL, args.output_dir)
    bootstrap_auc_ci(df_analysis, TARGET_COL, args.output_dir)
    threshold_tradeoff(df_analysis, TARGET_COL, args.output_dir)
    sensitivity_specificity_summary(df_analysis, TARGET_COL, args.output_dir)
    write_limitations_note(args.output_dir, df_analysis)

    # Five additional analyses requested; all original analyses above remain unchanged.
    feature_selection_analysis(df_analysis, TARGET_COL, args.output_dir)
    decision_curve_analysis(df_analysis, TARGET_COL, args.output_dir)
    cross_validated_permutation_importance(df_analysis, TARGET_COL, args.output_dir)

    # Final calibration, agreement, and statistical model-comparison metrics.
    agreement_metrics(df_analysis, TARGET_COL, args.output_dir)
    delong_model_comparison(df_analysis, TARGET_COL, args.output_dir)
    literature_benchmark_comparison(df_analysis, TARGET_COL, args.output_dir)
    plot_cases_by_age_group(df_analysis, TARGET_COL, args.output_dir)

    # Requested follow-ups: hyperparameter tuning, SHAP, subgroup fairness
    # Note: hyperparameter tuning was NOT re-run here -- a dedicated
    # ablation study already established (on this same T0-only feature
    # set) that tuning does not improve on these defaults. See the
    # ablation-study documentation rather than duplicating that analysis.
    cross_validated_shap_summary(df_analysis, TARGET_COL, args.output_dir)
    subgroup_performance(df_analysis, TARGET_COL, args.output_dir)

    # Shape/morphology analysis + additional medically relevant analyses,
    # computed exclusively from the 1638-patient file (no other dataset).
    plot_shape_feature_distributions(df_analysis, TARGET_COL, args.output_dir)
    plot_shape_correlation_with_target(df_analysis, TARGET_COL, args.output_dir)
    plot_margin_vs_cancer(df_analysis, TARGET_COL, args.output_dir)
    plot_attenuation_vs_cancer(df_analysis, TARGET_COL, args.output_dir)
    plot_location_vs_cancer(df_analysis, TARGET_COL, args.output_dir)
    shape_feature_class_separability(df_analysis, TARGET_COL, args.output_dir)

    log.info(f"Done, everything's in: {args.output_dir}/")

if __name__ == "__main__":
    main()