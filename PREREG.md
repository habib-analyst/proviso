# PREREG.md — proviso (Project #61)

Locked 2026-10-22 ~04:20 PKT, before measurement.

## Paper being reproduced (arXiv 2610.27115, Yan et al.)
PROVISO: a controller that decides which action to commit AFTER the agent
proposes an action and BEFORE it executes. It encodes trusted observations,
external observations, the proposed action, and an alternative into
provenance-typed memory, relates them through role-specific queries, and
commits the action with the highest calibrated value learned from outcomes.
Reported: +13.84 F1 on SkillsBench, +7.07 on ToolSafe vs strongest baseline,
with no baseline beating it in BOTH security and utility.

## What this build does (mechanism demonstration, synthetic scope)
A deterministic, zero-LLM, CPU-only harness demonstrating the paper's core
mechanism: provenance-awareness beats isolation-judging WITHOUT an
always-deny utility collapse. It does NOT reproduce the paper's benchmarks
(SkillsBench/ToolSafe are not used); it tests the same mechanism on synthetic
scripted traces with planted provenance attacks.

## Experimental design (fixed before measurement)
- **Trace generator** (`src/proviso/traces.py`): seeded scripted episodes.
  Each episode = user request (TRUSTED provenance) + tool outputs (EXTERNAL
  provenance, some carrying planted prompt-injection instructions) +
  proposed action + safe alternative + ground-truth label
  (unsafe = proposed action executes injected external content or is
  intrinsically harmful; safe otherwise).
- **Invisible attacks**: on attacked episodes the proposed action's text is
  token-identical to a benign twin's action, so any judge that sees only the
  action text cannot discriminate by text alone. The ONLY signal is
  provenance: payload content traces to EXTERNAL observations and was never
  authorized by TRUSTED observations.
- **Isolation judge** (`isolation.py`): TF-IDF + logistic regression on the
  proposed action text ONLY. Judges in isolation, provenance-blind.
- **PROVISO controller** (`controller.py`): provenance-typed memory —
  observations tagged [TRUSTED]/[EXTERNAL] plus role-specific queries
  (provenance-alignment features: does action content trace to trusted
  authorization? does it contain tokens that appear only in external
  observations? injection-marker overlap; recipient/target authorization
  mismatch). TF-IDF over role-tagged text + engineered provenance features,
  logistic regression, isotonic calibration (calibrated value), commits
  argmax(V(proposed), V(alternative)).
- **Decision rule shared by both**: allow the proposed action iff its
  predicted value >= value of the safe alternative; else block
  (commit alternative). Primary metric: F1 with positive class =
  "unsafe proposed action blocked".
- **Stats** (`stats.py`): paired bootstrap over episodes (B=2000, fixed seed)
  on F1 differences; 95% percentile CIs.
- **Splits**: train split (mixed) fits both models; held-out eval split is
  partitioned into attacked slice and benign slice. All gates evaluated on
  eval only.
- **Calibration check**: report Brier score / reliability gap of the
  controller's value head (the "calibrated value" claim).

## Locked bars (from KILLGATE.md)
- Gate 2: F1_iso(benign) - F1_iso(attacked) >= 0.15, 95% CI excl. 0.
- Gate 3: F1_prov(attacked) - F1_iso(attacked) >= 0.10, 95% CI excl. 0;
  utility loss on benign <= 0.02 both directions.
