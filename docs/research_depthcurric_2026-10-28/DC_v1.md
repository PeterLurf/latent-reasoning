# DEPTH-CURRIC v1 (2026-10-28)

**Status:** v1 = DC_draft.md (cff1b394) with all of the red team's findings (REVIEW_DC.md: NO-GO as written; 1–3 blockers adopted, 4–8 in the registration) and Audit's two points. Where they differ, this file wins.
- Theory's choice under the user's standing rules (the C2 line).
- Evidence: §241 STEPS; §242 NOT-NORM; Audit's locus probes 4e6b4a94 (a COMPUTE failure past the trained depth; old-state back-references at chance).

## 1. Hypotheses, and the offset model (math backing)
- Let f_M(k) = accuracy at k steps past model M's trained depth, at K′ 8 (so d = d_max + k).
- Trained on d ≤ 4, the inits give f(1) = 0.86 / 0.69, f(2) = 0.42 / 0.21, f(≥ 3) ≈ chance (0.09–0.12).
- **H-ARCH:** the limit is architectural and moves with training. f_DC(k) ≈ f_CTRL(k), i.e. acc_DC(6 + k) ≈ acc_CTRL(4 + k).
- **H-NARROW:** narrow training caused the limit, and a wider depth range gives a step function that extrapolates further. f_DC(k) > f_CTRL(k).
- **Prediction table:**
  - H-ARCH: acc_DC(8) ≈ acc_CTRL(6) (0.42 / 0.21), and acc_DC(9) ≈ chance.
  - H-NARROW: acc_DC(8) ≥ acc_CTRL(6) + 0.20, and acc_DC(9) ≥ 0.24.
- The draft's absolute bars (d 7 ≥ 0.60, d 8 ≥ 0.40) would be met by H-ARCH in s1 (fix 1), so they are dropped.

## 2. Arms (fine-tunes from §240's models; one RTX 3090 each)
- **DC:** dc_s1 / dc_s2 from lh_fh_s1 / lh_fh_s2: DF law, d ~ U{1..6}, think_k 10.
- **CTRL:** dc_ctrl_s1 / dc_ctrl_s2 from the same inits, identical except d ~ U{1..4} (Audit (1); fix 6 adds CTRL s2).
- **Both arms:** think_k 10; §240's hold-increment penalty λ 1.6 with **hold_start 500** (fix 3: the fade restarts, and penalising during the ramp would push the last-cue write toward zero); far pass T ~ U{16..40}; 8k steps; §240's lr.
- **Args diff DC vs CTRL = {d_max 6}** (single change). Both arms vs §240 FH = {think_k 10} + init + name.
- **Before launch:** a $0 local 1k-step pilot of dc_s1 and dc_ctrl_s1 (fix 6), reading only gn_plain windows and per-d norm_guard. The LP0 cap is then fixed in a Note.
- One trunk lineage (s233_askname_s1 → … → lh_fh_*), disclosed: 2 data orders, not independent seeds.

## 3. Evals
- run_long `--kprime_list 8,10,16,64,128 --long_ds 1..12` on TEST2-DF, one card / eval_bs per model, for DC, CTRL and both inits. It writes eval_long_dc.json {by_d {acc, s_norm}} (fix 4).
- §238's DF parts on DC / CTRL.
- A new ws_long part: WS per pass at d 1..10 with per-item op / src_j (from regen, joined by sig) (fix 5).
- The per-d norm_guard is logged per bin (fix 2).

## 4. Readings (theory/score_dc.py, fixtures before output). Per seed s, DC_s vs CTRL_s.
- **UNINFORMATIVE gate:** DC d 5–6 at K′ 8 ≥ 0.90, else UNINFORMATIVE (the wider training didn't take).
- **EXTRAP (binding):** Δ2 = acc_DC(8) − acc_CTRL(6) ≥ 0.20, and acc_DC(9) ≥ 0.24 (chance 0.091 + 0.15), at K′ 16.
- **STEP-READ (binding):** mean WS on regular ops (non-back-reference) at passes 7–8 of d 8 (DC) minus the same at passes 5–6 of d 6 (CTRL) ≥ 0.15.
- **C2-RISE (binding, past the budget, d 12 only; fix 4):** acc_DC(12, K′ 16) − acc_DC(12, K′ 10) ≥ 0.10 and acc(12, K′ 16) ≥ 0.25. Labelled scaffolded: the cue at passes 11–12 is shown only when K′ > 10.
- **NOLOSS:** HOLD-LONG3 at d 1–6 (K′ 64 / 128: acc ≥ acc(8) − 0.02), §238's gates at d 3–4, and retention vs lh_fh_* (drop > 0.05 named).
- **LP0:** per the warm-start checklist and the pilot Note. **Norm guard:** per d, matched d ≤ 4, DC / CTRL > 1.25 means GAMED.
- **Descriptive:**
  - ADDR: WS on back-reference ops, stratified by lag (seen ≤ 6 / unseen) and by j = 0 vs j ≥ 1;
  - f_M(k) curves; the ‖S‖ ratio.
- **Labels per seed, first match:**
  1. GAMED / LP0-FAIL / NOLOSS-FAIL (named);
  2. UNINFORMATIVE;
  3. H-ARCH (|acc_DC(6 + k) − acc_CTRL(4 + k)| ≤ 0.10 for k 1, 2, 3);
  4. PASS (EXTRAP, STEP-READ and C2-RISE);
  5. EXTRAP (EXTRAP and STEP-READ, C2-RISE fails);
  6. IN-RANGE-ONLY (otherwise).
- **Overall:** PASS / EXTRAP if both seeds agree; SPLIT / FAIL otherwise.

## 5. OC (2 seeds)
- P(UNINFORMATIVE) ≈ 0.1 per seed.
- Given informative:
  - P(EXTRAP + STEP-READ | H-NARROW) ≈ 0.6, | H-ARCH ≈ 0.03;
  - P(C2-RISE | EXTRAP) ≈ 0.3.
- Prior H-NARROW 0.4.
- Per seed: P(EXTRAP or better) ≈ 0.9 × (0.4 × 0.6 + 0.6 × 0.03) ≈ 0.23; P(PASS) ≈ 0.07.
- Both seeds (correlated lineage, ρ ≈ 0.5): P(overall EXTRAP or better) ≈ 0.14; P(PASS) ≈ 0.03.
- Under no effect (H-ARCH): P(EXTRAP, both) < 0.01.

## 6. Censoring
As §240: a crash means resume (dedupe); missing parts mean pending, FAIL at --final; a missing step-0 ref means pending.

## 7. Cost (fix 8; Audit (2))
- ≈ 1.45× §240 FH step time ≈ $1.0 per arm, four arms ≈ $4. Admit caps set from host G-TIME; micro 4 if memory allows (est. 19 GiB vs 21). Guard $8.
- Running: realised ~$13.25 + $8 = $21.25.
- **Pre-registered fallback (not run): ADDR-LEVER.** If DC is H-ARCH or IN-RANGE-ONLY in both seeds, the next lever targets old-state addressability (candidates: step-keyed writes, or a larger delta-head share). It gets its own derivation and red team then. Estimated guard $5, so the running total would be $26.25 ≤ $30.
