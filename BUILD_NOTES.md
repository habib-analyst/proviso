# Build notes — proviso (#61), 2026-10-22 ~04:10–04:50 PKT

## Gate 1 (PASS, 04:10–04:13)
- arXiv abs 2610.27115 (submitted 19 Oct 2026): no code/GitHub link.
- GitHub repo search API "PROVISO provenance-informed action selection": 0 results.
- Web search "2610.27115 github code": no author implementation.
- Related-but-different: PROV-AGENT (Flowcept, different paper). No conflict.

## Design iterations (honest log)
1. First full run FAILED all gates: the alternative actions ("ask_user_to_confirm(action='delete_dir'...)")
   shared attack cues with the proposed action, so the proposed-vs-alternative value comparison was
   meaningless (proviso attacked F1 = 0.0). Fixed: neutral no-op alternative for all families +
   decision rule "block iff P(unsafe) >= 0.5" (V_alt = 0.5 fixed by design).
2. Isotonic AND Platt calibration on the isolation judge distorted scores on text-duplicated
   pairs (raw 0.41 at 42% base rate -> isotonic collapsed to constant 0.3503 in one fold's
   calibrator; Platt -> 0.59). Both rejected for the judge; raw logistic used (documented in
   isolation.py). Controller keeps isotonic (verified sane: 4 thresholds, 0.0 -> 1.0).
3. Added q7 (external-introduced emails) and q8 (unauthorized action entities) after q1 alone
   failed to separate paired attacked/twin episodes (only 4/20 separated on q1).
4. Marker ablation (q3 zeroed): attacked F1 stays 1.0000 — effect is provenance-alignment,
   not marker string-matching.

## Final measured numbers (seed 42, locked bars)
- F1 isolation: attacked 0.0000, benign 1.0000
- F1 proviso: attacked 1.0000, benign 1.0000
- Gate 2: gap 1.0000 (bar 0.15), paired CI [1.0, 1.0] -> PASS
- Gate 3 security: delta 1.0000 (bar 0.10), CI [1.0, 1.0] -> PASS
- Gate 3 utility: loss 0.0000 (bar 0.02) -> PASS
- Brier: proviso 0.0339, isolation 0.1822
- Tests: 8/8 green. CPU: ~28 s experiment + ~2 min tests. $0. Zero LLM calls. No pushes.

## Lesson for future builds
When the alternative action's text shares cues with the proposed action, a learned
proposed-vs-alternative value comparison is confounded. Fix the alternative to a neutral
no-op and fix its value by design (V_alt = 0.5), or verify the comparison is meaningful.
Also: always sanity-probe calibration on duplicated-feature data before trusting it.
