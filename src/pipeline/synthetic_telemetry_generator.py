"""
synthetic_telemetry_generator.py
=================================
CSE599 – Computer Systems and Engineering Capstone
AI Infrastructure Reliability & Capacity Optimization
California Science and Technology University (CSTU), August 2026
Author: Subhashish Mitra

Generates a reproducible, PCPS-grounded synthetic corpus of 3,000 rows across
three dataset types:
  - MDR  (Master Deployment Record) : 1,200 rows  → SDP + CLS targets
  - Supply Plan                     :   800 rows  → CHF target
  - Task                            :   600 rows  → TEE target

All randomness seeded at seed=42 for full reproducibility.
Outputs written to data/ directory as CSV files.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
rng  = np.random.default_rng(SEED)

N_MDR    = 1_200
N_SUPPLY =   800
N_TASK   =   600

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared lookup tables  (PCPS hierarchy)
# ---------------------------------------------------------------------------
REGIONS       = ["NA", "EU", "APAC", "LATAM"]
REGION_ENC    = {r: i for i, r in enumerate(REGIONS)}

SUITE_STATUSES   = ["Operational", "PlannedUpgrade", "RetrofitScheduled", "AtRisk", "DecommissionPending"]
SUITE_STATUS_ENC = {s: i for i, s in enumerate(SUITE_STATUSES)}

SCENARIO_BRANCHES = {
    0: "StdEthernet",   # 68 %
    1: "OCS",           # 16 %
    2: "Expander",      # 12 %
    3: "ScaleAcross",   #  4 %
}
BRANCH_PROBS = [0.68, 0.16, 0.12, 0.04]

GEN_NUMS = [3, 4, 5]          # Switch generations (Spectrum-X fabric gen)
PRIORITIES = ["P0", "P1", "P2", "P3"]
BUCKETS    = ["GPU-Training", "GPU-Inference", "Storage", "Networking", "Compute"]

OCS_PROB_BY_STATUS = {
    "Operational":          0.12,
    "PlannedUpgrade":       0.61,
    "RetrofitScheduled":    0.74,
    "AtRisk":               0.88,
    "DecommissionPending":  0.03,
}


# ---------------------------------------------------------------------------
# Helper generators
# ---------------------------------------------------------------------------

def _gen_region(n: int) -> pd.Series:
    return pd.Series(rng.choice(REGIONS, size=n), name="region")


def _gen_suite_status(n: int) -> pd.Series:
    weights = [0.45, 0.20, 0.15, 0.12, 0.08]
    return pd.Series(rng.choice(SUITE_STATUSES, size=n, p=weights), name="suite_status")


def _gen_branch(n: int) -> pd.Series:
    return pd.Series(rng.choice(list(SCENARIO_BRANCHES.keys()), size=n, p=BRANCH_PROBS), name="branch")


def _gen_gen_num(n: int) -> pd.Series:
    return pd.Series(rng.choice(GEN_NUMS, size=n, p=[0.30, 0.50, 0.20]), name="gen_num")


# ---------------------------------------------------------------------------
# 1.  MDR Corpus  (Master Deployment Record)  — 1,200 rows
# ---------------------------------------------------------------------------

def generate_mdr(n: int = N_MDR) -> pd.DataFrame:
    """
    MDR rows represent individual suite-level deployment records.
    Primary ML targets:
      - supply_delay_days  → SDP (Supply Delay Predictor)
      - upgrade_label      → CLS (Topology Upgrade Classifier)  {0: NoUpgrade, 1: Upgrade}
    """
    region       = _gen_region(n)
    suite_status = _gen_suite_status(n)
    branch       = _gen_branch(n)
    gen_num      = _gen_gen_num(n)

    # Milestone gaps (days) — drawn from realistic planning distributions
    ms_me_gap   = rng.normal(loc=45, scale=15, size=n).clip(5, 120)    # Market Survey → ME
    me_lroof_gap = rng.normal(loc=30, scale=10, size=n).clip(2, 90)    # ME → L-Roof
    total_gap    = ms_me_gap + me_lroof_gap + rng.normal(20, 8, n).clip(0, 60)

    # suite_status_enc — encode
    sse = pd.Series([SUITE_STATUS_ENC[s] for s in suite_status], name="suite_status_enc")
    re  = pd.Series([REGION_ENC[r] for r in region], name="region_enc")

    # OCS topology probability (from Table 6.2)
    ocs_prob = pd.Series([OCS_PROB_BY_STATUS[s] for s in suite_status], name="ocs_prob")

    # Supply delay target (regression)
    # Base delay driven by milestone gaps + suite-status risk + region noise
    base_delay = (
        0.10 * ms_me_gap
        + 0.15 * me_lroof_gap
        + 0.05 * total_gap
        + sse * 4.0
        + rng.normal(0, 3, n)
    ).clip(0, 90)
    supply_delay_days = pd.Series(base_delay, name="supply_delay_days")

    # Upgrade label (classification) — driven by suite_status + gen_num + ocs_prob
    upgrade_prob = (
        0.25 * sse / 4.0
        + 0.35 * ocs_prob
        + 0.15 * (gen_num - 3) / 2.0        # older gen → higher risk
        + rng.uniform(0, 0.25, n)
    ).clip(0, 1)
    upgrade_label = pd.Series((upgrade_prob > 0.50).astype(int), name="upgrade_label")

    df = pd.DataFrame({
        "row_id":             [f"MDR-{i:05d}" for i in range(n)],
        "region":             region,
        "region_enc":         re,
        "suite_status":       suite_status,
        "suite_status_enc":   sse,
        "branch":             branch,
        "gen_num":            gen_num,
        "ms_me_gap":          ms_me_gap.round(2),
        "me_lroof_gap":       me_lroof_gap.round(2),
        "total_gap":          total_gap.round(2),
        "ocs_prob":           ocs_prob,
        "supply_delay_days":  supply_delay_days.round(2),
        "upgrade_label":      upgrade_label,
    })
    return df


# ---------------------------------------------------------------------------
# 2.  Supply Plan Corpus  — 800 rows
# ---------------------------------------------------------------------------

def generate_supply(n: int = N_SUPPLY) -> pd.DataFrame:
    """
    Supply plan rows represent quarterly capacity snapshots per datacenter.
    Primary ML target:
      - headroom_mw  → CHF (Capacity Headroom Forecaster)
    """
    region = _gen_region(n)
    re     = pd.Series([REGION_ENC[r] for r in region], name="region_enc")

    # Utilisation and supply features
    lag1_util      = rng.uniform(0.45, 0.92, n)          # prior-period utilisation
    lag1_headroom  = rng.normal(60, 20, n).clip(5, 150)  # prior-period headroom (MW)
    supply_mw      = rng.normal(120, 35, n).clip(30, 300)
    demand_mw      = supply_mw * rng.uniform(0.60, 0.95, n)

    # Target: headroom_mw (regression)
    # Strong AR-1 signal (lag1_headroom dominant per SHAP)
    headroom_mw = (
        0.72 * lag1_headroom
        + 0.18 * (supply_mw - demand_mw)
        - 0.12 * lag1_util * supply_mw
        + rng.normal(0, 5, n)
    ).clip(0, 200)

    df = pd.DataFrame({
        "row_id":          [f"SUP-{i:05d}" for i in range(n)],
        "region":          region,
        "region_enc":      re,
        "lag1_util":       lag1_util.round(4),
        "lag1_headroom":   lag1_headroom.round(2),
        "supply_mw":       supply_mw.round(2),
        "demand_mw":       demand_mw.round(2),
        "headroom_mw":     headroom_mw.round(2),   # CHF target
    })
    return df


# ---------------------------------------------------------------------------
# 3.  Task Corpus  — 600 rows
# ---------------------------------------------------------------------------

def generate_task(n: int = N_TASK) -> pd.DataFrame:
    """
    Task rows represent individual operational work items.
    Primary ML target:
      - effort_weeks  → TEE (Task Effort Estimator)

    Note: TEE is signal-limited in this corpus (only 4 categorical features).
    TF-IDF enrichment from task description text is planned for future work.
    """
    bucket   = pd.Series(rng.choice(BUCKETS, size=n), name="bucket")
    priority = pd.Series(rng.choice(PRIORITIES, size=n, p=[0.05, 0.20, 0.45, 0.30]), name="priority")

    bucket_enc   = pd.Series([BUCKETS.index(b) for b in bucket], name="bucket_enc")
    priority_enc = pd.Series([PRIORITIES.index(p) for p in priority], name="priority_enc")

    # Interaction feature (top TEE SHAP feature per the paper)
    bk_pr = pd.Series(bucket_enc * 4 + priority_enc, name="bk_pr")

    # Target: effort_weeks — predominantly driven by bk_pr interaction
    # Low R² is expected and documented (signal limit without TF-IDF)
    base_effort = (
        1.5
        + 0.40 * bucket_enc
        - 0.25 * priority_enc
        + 0.05 * bk_pr
        + rng.normal(0, 1.4, n)
    ).clip(0.5, 10.0)

    df = pd.DataFrame({
        "row_id":        [f"TSK-{i:05d}" for i in range(n)],
        "bucket":        bucket,
        "bucket_enc":    bucket_enc,
        "priority":      priority,
        "priority_enc":  priority_enc,
        "bk_pr":         bk_pr,
        "effort_weeks":  base_effort.round(2),   # TEE target
    })
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CSE599 Capstone — Synthetic Telemetry Generator")
    print("CSTU · Subhashish Mitra · August 2026")
    print(f"Random seed: {SEED}")
    print("=" * 60)

    print(f"\n[1/3] Generating MDR corpus    ({N_MDR:,} rows)...")
    mdr = generate_mdr()
    out = DATA_DIR / "mdr_corpus.csv"
    mdr.to_csv(out, index=False)
    print(f"      → {out}  ({len(mdr):,} rows, {len(mdr.columns)} columns)")

    print(f"\n[2/3] Generating Supply corpus ({N_SUPPLY:,} rows)...")
    supply = generate_supply()
    out = DATA_DIR / "supply_corpus.csv"
    supply.to_csv(out, index=False)
    print(f"      → {out}  ({len(supply):,} rows, {len(supply.columns)} columns)")

    print(f"\n[3/3] Generating Task corpus   ({N_TASK:,} rows)...")
    task = generate_task()
    out = DATA_DIR / "task_corpus.csv"
    task.to_csv(out, index=False)
    print(f"      → {out}  ({len(task):,} rows, {len(task.columns)} columns)")

    total = len(mdr) + len(supply) + len(task)
    print(f"\n✅  Done — {total:,} total synthetic rows written to {DATA_DIR}/")
    print("    Run src/training/model_training.py to train all 4 models.")


if __name__ == "__main__":
    main()
