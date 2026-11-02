# REVIEW_BU: red team of BU_draft.md (Opus subagent, 2026-11-01; no model.pt loaded). Verdict: GO-with-fixes, launch only if L2 = DRIFT. All adopted in BU_v1.md

- **B1:** wait for L1 / L2. If dis already holds, the bundle is eval-only at $0. Register as conditional on L2 DRIFT; run the penalty only on drifting seeds.
- **B2:** the norm guard is missing, and score_lh's guard needs a paired CTRL (dis had no hold_log bins). Add BUN-CTRL (the same 8k FT with --hold_log, no penalty), or guard against BUN's own step 0 at 1.25×.
- **B3 (mechanism risk):**
  - Hold-pass inputs equal pass d's (cue S_d), so "pass ≥ 5 → write ~0" fits every penalised pass and could suppress dis's untrained-pass computation.
  - §240's d 5 evidence is mixed (FH − CTRL +0.46 / −0.12), from inits with pass-5 compute ~0.55.
  - Fix: a C2 retention gate vs dis (d 6 / 7 ≥ dis − 0.05; WS d 6 passes 5-6 ≥ 0.90); report d 8-10 at K' 8 / 12 / 16; P(C2 kept) ~0.6.
- **B4:** keep hold_start 500 (the FT re-ramps the fade over 500 steps). Ship args_diff.json.
- **B5:** part_halthead tests d 1..6 only, and score_hx hardcodes dis runs. Reuse the L1 build (d 1..7); parametrise the scorer with fixtures.
- **m6:** gate mapping:
  - C1 = score_ds RETAIN;
  - C2 thresholds are new (retention form);
  - C6 = score_hx.hold plus SEED-VAR;
  - LP0-W on CTRL windows with step > 1000 (dis_s2 itself hit 15.9);
  - C5 = score_hx C5-EXTRAP.
- **m7:** delete "joint training would be equivalent". Scope: post-hoc head on a frozen loop.
- **m8:** P(C6) ~0.6 (d 5-6 hold untrained); ~$1.0 per arm; a mem_check with the penalty + distract on.
- **m9:** label "BUNDLE-PASS (scoped)". Scope: plan given, chain order, REG frames 0-5, one lineage, per-pass aux (no C4), and C2 within K = 8 (K' > K untested; the penalty fixes K' > 8 by design).
