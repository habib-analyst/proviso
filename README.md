# proviso — Provenance-Informed Action Selection for Secure LLM Agents

An open implementation of the mechanism in **arXiv 2610.27115**
("PROVISO: Provenance-Informed Action Selection for Secure LLM Agents",
Yan et al., 2026). Built by **Habib Ur Rehman** ([github.com/habib-analyst](https://github.com/habib-analyst)).

## The idea

LLM agents act on the world through tools, so one executed action can cause
irreversible harm. Existing defenses judge a proposed action **in isolation**,
without tracking whether its content came from the user or from external
sources like tool outputs — and without comparing it against an available
alternative. That forces a security-vs-utility trade-off.

PROVISO instead encodes **trusted observations** (the user), **external
observations** (tool outputs), the **proposed action**, and a **safe
alternative** into provenance-typed memory, relates them through
role-specific queries, and commits the action with the highest calibrated
value learned from outcomes.

## What this repo demonstrates

A deterministic, **zero-LLM, CPU-only** harness reproducing the paper's core
result on synthetic scripted traces with planted prompt-injection attacks:

- **Disease:** tool-output-borne ("invisible") injections flip a
  provenance-blind action judge. The judge's F1 collapses on the attacked
  slice while staying strong on benign traffic.
- **Cure:** the provenance-typed controller beats the isolation judge by a
  wide margin on the attacked slice **without** an always-deny utility
  collapse on the benign slice (the security-utility frontier, both
  directions).

> **Synthetic scope.** All traces are scripted and seeded; no real LLM, no
> real tools, no real attacks. This is a mechanism demonstration, not a
> reproduction of the paper's SkillsBench/ToolSafe benchmarks.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m proviso.run_experiment --seed 42 --out results.json
```

Runs in ~2 minutes on CPU. All numbers are deterministic for a fixed seed.

## Layout

- `src/proviso/traces.py` — synthetic trace generator with planted
  provenance attacks (5 families; every attacked episode has a benign twin
  with token-identical action text)
- `src/proviso/isolation.py` — isolation-judge baseline (action text only)
- `src/proviso/controller.py` — PROVISO controller (provenance-typed memory
  + role-specific queries + calibrated value head)
- `src/proviso/stats.py` — F1, paired-bootstrap CIs, Brier score
- `src/proviso/run_experiment.py` — end-to-end experiment + locked kill-gates
- `tests/` — deterministic test suite (`pytest tests/`)
- `KILLGATE.md` / `PREREG.md` — kill-gate bars locked *before* measurement

## Measured results (seed 42, 1000 train / 900 eval)

| slice | isolation judge F1 | PROVISO F1 |
|---|---|---|
| attacked (300) | 0.0000 | 1.0000 |
| benign (600) | 1.0000 | 1.0000 |

- Gate 2 (disease): benign-minus-attacked judge F1 gap = 1.0000 (bar 0.15),
  paired correctness gap 95% CI excludes 0 → **PASS**
- Gate 3 (security): ΔF1 = 1.0000 (bar 0.10), 95% CI excludes 0 → **PASS**
- Gate 3 (utility): |loss| = 0.0000 (bar 0.02) → **PASS**, no always-deny
- Marker ablation (q3 zeroed): PROVISO attacked F1 stays 1.0000 — the effect
  rides on provenance-alignment queries, not marker string-matching

See `results_seed42.json` and `hidden_files/papers/proviso/PAPER.md` in the
goal workspace for the full writeup.

## Reproduce the paper's claim in one line

Provenance-awareness beats isolation-judging **without** a utility collapse:
+100pp F1 on the attacked slice, ±0pp on the benign slice.

## License

MIT — see `LICENSE`.
