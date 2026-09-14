# AI Infrastructure Reliability & Capacity Optimization

**Emerging Technologies Program**
California Science and Technology University (CSTU) · Milpitas, CA
Subhashish Mitra · August 2026

---

## Overview

This project delivers an end-to-end machine learning platform for **AI data-center infrastructure reliability and capacity planning**. It addresses four tightly coupled operational problems — supply delay prediction, capacity headroom forecasting, task effort estimation, and topology upgrade risk classification — within a unified Spectrum-X fabric environment instrumented by the Fabric360 and FlowQ telemetry systems.

The platform spans a four-tier architecture: telemetry ingestion, ML pipeline, agentic mitigation workflow, and closed-loop feedback. A natural-language AI Assistant layer routes operator queries through a policy-aware inference engine to automated action executors, completing the reliability loop without manual intervention.

---

## System Architecture

| Layer | Components |
|---|---|
| **Telemetry & Observability** | Fabric360 (error counters, congestion metrics, buffer pressure, link health) · FlowQ (topology graph, event logs, state changes) |
| **Data Processing & ML** | Feature engineering (reliability, capacity, temporal, TF-IDF, composite) · Model training · Evaluation (metrics, SHAP, Bias Audit) |
| **Agentic Workflow** | NL Query Interface → Model Router → Inference Engine → Policy Engine → Action Executor |
| **Infrastructure** | Spectrum-X Fabric · 100G / 200G / 400G switches · Training Pods A–D |

![System Architecture](figures/system-architecture.png)

---

## Models

| Model | Abbrev. | Task | Algorithm | CV Metric | Result | Target |
|---|---|---|---|---|---|---|
| Capacity Headroom Forecaster | **CHF** | Regression | LightGBM | CV R² | **0.815 ±0.122** | ≥ 0.75 ✅ |
| Supply Delay Predictor | **SDP** | Regression | XGBoost | CV R² | **0.736 ±0.049** | ≥ 0.75 ⚠️ |
| Topology Upgrade Classifier | **CLS** | Classification | LightGBM | CV F1 | **0.427 ±0.024** | Decomm. Recall ≥ 0.70 ⚠️ |
| Task Effort Estimator | **TEE** | Regression | LightGBM | CV R² | **0.003 ±0.055** | ≥ 0.25 ⚠️ |

**CHF** meets its target comfortably. **SDP** falls 1.9pp short — addressable with additional supply-chain signal. **CLS** achieves the baseline F1 target but falls short on decommission recall (0.550 vs. 0.70), a known class-imbalance issue. **TEE** is feature-limited; TF-IDF enrichment of task description text is the primary recommended intervention.

---

## Synthetic Corpus

All models are trained on a reproducible synthetic dataset:

| Schema | Records | Seed |
|---|---|---|
| MDR / SDP / CLS | 1,200 | 42 |
| Supply Plan / CHF | 800 | 42 |
| Task / TEE | 600 | 42 |

No proprietary or production data is used. Full schemas are documented in Appendix A of the project report.

---

## SHAP Explainability

SHAP analysis was applied to all four models:

- **CHF**: Rack headroom trajectory and historical buffer pressure are top drivers
- **SDP**: Vendor lead-time variance and order backlog depth dominate slippage prediction
- **CLS**: Link utilization rate and error counter trends drive upgrade classification
- **TEE**: Task complexity composite and team queue depth are the strongest available signals

---

## Bias Audit

A structured bias audit was conducted across regional and subgroup dimensions:

- No systematic regional parity violations detected for CHF and SDP
- CLS exhibits elevated false-negative rate on decommission-class samples — flagged for threshold calibration
- TEE parity analysis is inconclusive given near-zero R²; revisit after feature enrichment

---

## Agentic AI Assistant

The AI Assistant provides a closed-loop, natural-language interface for infrastructure operators:

1. **NL Query Interface** — parses free-text operator queries
2. **Model Router** — selects the appropriate ML model(s) for the query context
3. **Inference Engine** — runs prediction and returns ranked recommendations
4. **Policy Engine** — validates recommended actions against operational constraints
5. **Action Executor** — dispatches approved mitigations to the fabric control plane

![AI Assistant Architecture](figures/ai-assistant-architecture.png)

---

## Repository Structure

