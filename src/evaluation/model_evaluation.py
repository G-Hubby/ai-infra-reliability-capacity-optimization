"""
model_evaluation.py
====================
AI Infrastructure Reliability & Capacity Optimization
AI Infrastructure Reliability & Capacity Optimization
California Science and Technology University (CSTU), August 2026
Author: Subhashish Mitra

Performs post-training evaluation across all four ML models:
  1. SHAP Feature Attribution  — top-4 drivers per model
  2. Subgroup Bias Audit       — regional RMSE/R² gaps (CHF, SDP); suite-status F1 gap (CLS);
                                 bucket RMSE gap (TEE)
  3. Operationalization Report — PSI drift monitor + refresh cadence schedule

Loads saved model artifacts from models/ and corpora from data/.
Outputs:
  - Console report (full bias audit + SHAP rankings)
  - matplotlib plots saved to outputs/plots/
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import r2_score, f1_score, mean_squared_error, classification_report
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs" / "plots"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_artifact(name: str) -> dict:
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Run src/training/model_training.py first."
        )
    return joblib.load(path)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def banner(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def section(title: str):
    print(f"\n  --- {title} ---")


# ---------------------------------------------------------------------------
# 1.  SHAP-equivalent Feature Attribution (Permutation Importance)
# ---------------------------------------------------------------------------
# Note: permutation importance is used here as a model-agnostic SHAP proxy.
# For production SHAP TreeExplainer, install the `shap` package and swap in
# shap.TreeExplainer(model).shap_values(X).
# ---------------------------------------------------------------------------

def compute_feature_importance(model, X, y, feature_names: list, n_repeats: int = 20) -> pd.DataFrame:
    """Return sorted feature importances via permutation."""
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=SEED)
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": result.importances_mean,
        "std":        result.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def plot_feature_importance(imp_df: pd.DataFrame, model_name: str, metric_label: str):
    """Bar chart of top-8 feature importances."""
    top = imp_df.head(8)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#7c3aed" if i == 0 else "#a78bfa" for i in range(len(top))]
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1],
                   xerr=top["std"][::-1], capsize=3, alpha=0.9)
    ax.set_xlabel(f"Permutation Importance ({metric_label})")
    ax.set_title(f"{model_name} — Feature Attribution (SHAP-equivalent)", fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    out = OUTPUTS_DIR / f"{model_name.lower()}_feature_importance.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"      → Plot saved: {out}")


# ---------------------------------------------------------------------------
# 2.  Bias Audit
# ---------------------------------------------------------------------------

BIAS_REGIONS = ["NA", "EU", "APAC", "LATAM"]
REGION_ENC   = {"NA": 0, "EU": 1, "APAC": 2, "LATAM": 3}
SUITE_STATUSES = ["Operational", "PlannedUpgrade", "RetrofitScheduled", "AtRisk", "DecommissionPending"]


def bias_audit_sdp(model, df: pd.DataFrame, features: list):
    """SDP regional RMSE gap audit. Threshold: < 1.0 day."""
    section("SDP Bias Audit — Regional RMSE Gap")
    X   = df[features].values
    y   = df["supply_delay_days"].values
    pred = model.predict(X)
    df   = df.copy()
    df["pred"] = pred
    df["residual_sq"] = (df["supply_delay_days"] - df["pred"]) ** 2

    results = {}
    for region in BIAS_REGIONS:
        mask  = df["region"] == region
        rmse_ = float(np.sqrt(df.loc[mask, "residual_sq"].mean()))
        results[region] = rmse_
        print(f"    {region:6s}: RMSE = {rmse_:.3f} days")

    gap   = max(results.values()) - min(results.values())
    status = "✅ PASS" if gap < 1.0 else "🚩 FLAG"
    print(f"    RMSE gap (max-min): {gap:.3f} days  |  Threshold: 1.0 d  |  {status}")
    return results, gap, status


def bias_audit_chf(model, df: pd.DataFrame, features: list):
    """CHF regional R² gap audit."""
    section("CHF Bias Audit — Regional R² Gap")
    X    = df[features].values
    y    = df["headroom_mw"].values
    pred = model.predict(X)
    df   = df.copy()
    df["pred"] = pred

    results = {}
    for region in BIAS_REGIONS:
        mask = df["region"] == region
        r2   = r2_score(df.loc[mask, "headroom_mw"], df.loc[mask, "pred"])
        results[region] = r2
        print(f"    {region:6s}: R² = {r2:.3f}")

    gap   = max(results.values()) - min(results.values())
    worst = min(results, key=results.get)
    best  = max(results, key=results.get)
    status = "🚩 FLAG" if gap > 0.20 else "✅ PASS"
    print(f"    R² gap ({best}={results[best]:.3f} vs {worst}={results[worst]:.3f}): {gap:.3f}  |  {status}")
    return results, gap, status


def bias_audit_cls(model, df: pd.DataFrame, features: list):
    """CLS suite-status F1 gap audit."""
    section("CLS Bias Audit — Suite-Status F1 Gap")
    X    = df[features].values
    y    = df["upgrade_label"].values
    pred = model.predict(X)
    df   = df.copy()
    df["pred"] = pred

    # Per-status F1 (upgrade_label == 1 → Upgrade class)
    results = {}
    for status_val in SUITE_STATUSES:
        mask = df["suite_status"] == status_val
        if mask.sum() < 5:
            continue
        sub_y    = df.loc[mask, "upgrade_label"].values
        sub_pred = df.loc[mask, "pred"].values
        if len(np.unique(sub_y)) < 2:
            continue
        f1 = f1_score(sub_y, sub_pred, average="macro", zero_division=0)
        results[status_val] = f1
        print(f"    {status_val:25s}: F1 = {f1:.3f}")

    if results:
        gap    = max(results.values()) - min(results.values())
        worst  = min(results, key=results.get)
        best   = max(results, key=results.get)
        status = "🚩 FLAG" if gap > 0.15 else "✅ PASS"
        print(f"    F1 gap ({best}={results[best]:.3f} vs {worst}={results[worst]:.3f}): {gap:.3f}  |  {status}")
    else:
        gap, status = 0.0, "N/A"
    return results, gap, status


def bias_audit_tee(model, df: pd.DataFrame, features: list):
    """TEE bucket RMSE gap audit (signal quality flag)."""
    section("TEE Bias Audit — Bucket RMSE Gap")
    BUCKETS = ["GPU-Training", "GPU-Inference", "Storage", "Networking", "Compute"]
    X    = df[features].values
    y    = df["effort_weeks"].values
    pred = model.predict(X)
    df   = df.copy()
    df["pred"] = pred
    df["residual_sq"] = (df["effort_weeks"] - df["pred"]) ** 2

    results = {}
    for bucket in BUCKETS:
        mask  = df["bucket"] == bucket
        if mask.sum() < 5:
            continue
        rmse_ = float(np.sqrt(df.loc[mask, "residual_sq"].mean()))
        results[bucket] = rmse_
        print(f"    {bucket:18s}: RMSE = {rmse_:.3f} wk")

    if results:
        gap    = max(results.values()) - min(results.values())
        status = "🚩 FLAG (signal quality — not demographic)" if gap > 1.0 else "✅ PASS"
        print(f"    Bucket RMSE gap: {gap:.3f} wk  |  {status}")
    else:
        gap, status = 0.0, "N/A"
    return results, gap, status


# ---------------------------------------------------------------------------
# 3.  Operationalization Report
# ---------------------------------------------------------------------------

OPERATIONALIZATION = [
    {"Model": "CHF", "Refresh Cadence": "Quarterly",    "PSI Trigger": "> 0.20",
     "Inference Latency": "< 10 ms (CPU)",  "SHAP Latency": "50–100 ms", "Notes": "EU R² gap: Bayesian recalibration planned"},
    {"Model": "SDP", "Refresh Cadence": "Monthly",      "PSI Trigger": "> 0.20",
     "Inference Latency": "< 10 ms (CPU)",  "SHAP Latency": "50–100 ms", "Notes": "Vendor lead-time signal expansion planned"},
    {"Model": "CLS", "Refresh Cadence": "Semi-Annual",  "PSI Trigger": "> 0.20",
     "Inference Latency": "< 10 ms (CPU)",  "SHAP Latency": "50–100 ms", "Notes": "Decom recall class-weight tuning planned"},
    {"Model": "TEE", "Refresh Cadence": "Quarterly",    "PSI Trigger": "> 0.20",
     "Inference Latency": "< 10 ms (CPU)",  "SHAP Latency": "50–100 ms", "Notes": "TF-IDF enrichment planned"},
]


def print_operationalization():
    section("Operationalization & Refresh Schedule")
    header = f"  {'Model':<6} {'Cadence':<14} {'PSI Trigger':<13} {'Inference':<16} {'Notes'}"
    print(header)
    print("  " + "-" * 80)
    for row in OPERATIONALIZATION:
        print(
            f"  {row['Model']:<6} {row['Refresh Cadence']:<14} {row['PSI Trigger']:<13} "
            f"{row['Inference Latency']:<16} {row['Notes']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("CSE599 Capstone — Model Evaluation, SHAP & Bias Audit")
    print("CSTU · Subhashish Mitra · August 2026")
    print("=" * 62)

    # ── Load corpora ─────────────────────────────────────────────────────────
    mdr_df    = pd.read_csv(DATA_DIR / "mdr_corpus.csv")
    supply_df = pd.read_csv(DATA_DIR / "supply_corpus.csv")
    task_df   = pd.read_csv(DATA_DIR / "task_corpus.csv")

    # ── CHF ──────────────────────────────────────────────────────────────────
    banner("CHF — Capacity Headroom Forecaster")
    chf_art  = load_artifact("chf_hgb.pkl")
    chf_mod  = chf_art["model"]
    chf_feat = chf_art["features"]
    X_chf = supply_df[chf_feat].values
    y_chf = supply_df["headroom_mw"].values

    section("SHAP Feature Attribution (Permutation Importance)")
    chf_imp = compute_feature_importance(chf_mod, X_chf, y_chf, chf_feat)
    print(chf_imp.head(4).to_string(index=False))
    plot_feature_importance(chf_imp, "CHF", "R² drop")

    chf_bias, chf_gap, chf_status = bias_audit_chf(chf_mod, supply_df, chf_feat)

    # ── SDP ──────────────────────────────────────────────────────────────────
    banner("SDP — Supply Delay Predictor")
    sdp_art  = load_artifact("sdp_gbr.pkl")
    sdp_mod  = sdp_art["model"]
    sdp_feat = sdp_art["features"]
    X_sdp = mdr_df[sdp_feat].values
    y_sdp = mdr_df["supply_delay_days"].values

    section("SHAP Feature Attribution (Permutation Importance)")
    sdp_imp = compute_feature_importance(sdp_mod, X_sdp, y_sdp, sdp_feat)
    print(sdp_imp.head(4).to_string(index=False))
    plot_feature_importance(sdp_imp, "SDP", "R² drop")

    sdp_bias, sdp_gap, sdp_status = bias_audit_sdp(sdp_mod, mdr_df, sdp_feat)

    # ── CLS ──────────────────────────────────────────────────────────────────
    banner("CLS — Topology Upgrade Classifier")
    cls_art  = load_artifact("cls_hgb.pkl")
    cls_mod  = cls_art["model"]
    cls_feat = cls_art["features"]
    X_cls = mdr_df[cls_feat].values
    y_cls = mdr_df["upgrade_label"].values

    section("SHAP Feature Attribution (Permutation Importance)")
    cls_imp = compute_feature_importance(cls_mod, X_cls, y_cls, cls_feat)
    print(cls_imp.head(4).to_string(index=False))
    plot_feature_importance(cls_imp, "CLS", "F1 drop")

    section("Classification Report (Full Corpus)")
    pred_cls = cls_mod.predict(X_cls)
    print(classification_report(y_cls, pred_cls, target_names=["NoUpgrade", "Upgrade"]))

    cls_bias, cls_gap, cls_status = bias_audit_cls(cls_mod, mdr_df, cls_feat)

    # ── TEE ──────────────────────────────────────────────────────────────────
    banner("TEE — Task Effort Estimator")
    tee_art  = load_artifact("tee_hgb.pkl")
    tee_mod  = tee_art["model"]
    tee_feat = tee_art["features"]
    X_tee = task_df[tee_feat].values
    y_tee = task_df["effort_weeks"].values

    section("SHAP Feature Attribution (Permutation Importance)")
    tee_imp = compute_feature_importance(tee_mod, X_tee, y_tee, tee_feat)
    print(tee_imp.head(4).to_string(index=False))
    plot_feature_importance(tee_imp, "TEE", "R² drop")

    tee_bias, tee_gap, tee_status = bias_audit_tee(tee_mod, task_df, tee_feat)

    # ── Bias Audit Summary ───────────────────────────────────────────────────
    banner("Bias Audit Summary")
    rows = [
        ("SDP", "Regional RMSE gap",      f"{sdp_gap:.3f} days", "< 1.0 d",  sdp_status),
        ("CHF", "Regional R² gap",         f"{chf_gap:.3f} R²",  "—",         chf_status),
        ("TEE", "Bucket RMSE gap",         f"{tee_gap:.3f} wk",  "—",         tee_status),
        ("CLS", "Suite-status F1 gap",     f"{cls_gap:.3f} F1",  "—",         cls_status),
    ]
    print(f"\n  {'Model':<6} {'Dimension':<26} {'Gap':<16} {'Threshold':<12} {'Status'}")
    print("  " + "-" * 75)
    for row in rows:
        print(f"  {row[0]:<6} {row[1]:<26} {row[2]:<16} {row[3]:<12} {row[4]}")

    # ── Operationalization ───────────────────────────────────────────────────
    banner("Operationalization & Refresh Schedule")
    print_operationalization()

    print(f"\n\n✅  Evaluation complete.  Plots: {OUTPUTS_DIR}/")
    print("    Review bias flags and apply remediations documented in Future Work.")


if __name__ == "__main__":
    main()
