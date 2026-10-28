# REVIEW_DC: red team of DC_draft.md (cff1b394), 2026-10-28

Opus subagent, read-only (scratchpad dcrt/backref.py).
Verdict: **NO-GO as written; GO-with-fixes once 1–3 are adopted, with 4–7 in the registration.** All adopted in DC_v1.md.

1. **BLOCKER:** the EXTRAP bars are met by a plain shift of the +1 / +2 offset, since the inits' d 5 / 6 are 0.863 / 0.415 (s1).
   - The "+0.25" clauses are slack (step 0 at d 7 / 8 ≈ 0.12 / 0.09; CTRL 0.08 / 0.09).
   - Fix: score depth past the training range, DC(6 + k) − CTRL(4 + k) ≥ 0.20 at k 2, d 9–10 > chance + 0.15, name H-ARCH, and redo the OC.
2. **BLOCKER:** the norm guard reads GAMED by construction. Pooled bins over d U{1..6} vs U{1..4} give DC / CTRL ≈ 1.7, since ‖S_8‖ grows ~550 per step.
   - Fix: log norm_guard per d and compare on matched d ≤ 4.
3. **FIX:** keep hold_start 500. The fade restarts (inject_aw.py:386) and gates the read, so a penalty at step 0 trains the last-cue write toward zero.
4. **FIX:** C2-RISE binds on d 12 only (d 11 needs 10 passes = think_k). K′ 16 isn't in eval_acc (KMAX 14), so register run_long with K′ 8, 10, 16, 64, 128 and d 1..12. Cues 11–12 are unseen at K′ 10 (scaffolded). Chance 0.091.
5. **FIX:** ADDR needs a build (ws covers d ≤ d_max only) and should stay descriptive.
   - Back-references are 18–23% of steps at t ≥ 5; unseen lags 7–8 are too rare to bind.
   - Stratify by lag and j = 0. Add a binding regular-op WS at passes 7–8 (d 8) vs CTRL 5–6 (d 6).
6. **FIX:** add CTRL s2. A CTRL-calibrated LP0 cap may false-fail DC, so run a $0 1k-step pilot first and fix the cap.
7. **FIX:** add an UNINFORMATIVE label (d 5–6 < 0.9); IN-RANGE at K′ 10; HOLD-LONG3 extended to d 5–6; the OC block, censoring and math backing; disclose the shared trunk.
8. **NOTE:** cost is ≈ 1.45× §240 FH (≈ $1.0 per arm, $3–3.5 total). Set admit caps from host G-TIME. Peak memory is est. 19 GiB at d 6, K 10 (micro 4 possible).
