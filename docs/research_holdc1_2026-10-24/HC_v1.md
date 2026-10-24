# HOLD-C1 v1: do C1 (serial causal use) and the §230 long-horizon hold coexist? (2026-10-24)

**Status:** v1 = HC_draft.md (43cce7d8), changed as listed here; where they differ, this file wins. Red team: REVIEW_HC.md (GO with fixes 1–7; 8–10 also adopted). Literature (addendum in LIT_FIXEDHOLD_NOVELTY_2026-10-21.md):
- S1 OK, narrow wording; motivated by Viakhirev 2608.18222 ("settle but EM = 0": settling and computing are separable).
- S2 narrow: cite 2609.01924's swap-persistence test; new only as a state splice with a 128-pass hold.
- S3 descriptive, reworded: HC is history-using in 1 of 2 lineages by the registered rule.

## 1. Cells and models (fixes 4, 5)
- **Family:** VERSION-BANK D3-T. Battery VB-C1 = DEEP, PROBE-GAIN, SERIAL-ONSET, VALUE-SWAP, STALL.
- **Roles per seed:** BASE / FT-CTRL / HC (the §230-style post-formation anchored far fine-tune).

| seed | BASE | FT-CTRL | HC |
|---|---|---|---|
| s1 | s213_d3t_s1 | s215_ftctrl_s1 | s230_fhftfix_s1 |
| s2 | s216_d3t_s2 | s224_ftctrl_s2 | **s224_fhft_s2** (no new s232 cell: within a seed the batch streams are identical, so a re-run would not be a replicate) |

- **Blind s3, pre-registered now:** D3-T s3 base (48k), FT-CTRL s3, HC s3 (λ fixed 1.6); ≈ $3.3, cap $4.
  - **Trigger:** s1 and s2 both read COEXIST / COEXIST-SHIFTED, or they SPLIT.
  - An s3 base that fails VB-C1 is censored (UNREAD), not a fail.
- RD-FT is deferred: it runs only if FAR-CHANGED fires (descriptive).
- s215_ftctrl_s1's training card is confirmed from the queue logs as a build check.

## 2. Stage A on one cloud RTX 3090 (fix 1)
- The C1 battery (vbank_eval acc, probe, vswap, stall) + part_d3hist OLD-SWAP + the new c1hold part, on the 6 existing models. All on one 3090, fp32, bs 32, in staged dirs (committed eval files never overwritten).
- The existing base re-reads on the 3090 cover G-BASE.
- ≈ 4.1 h ≈ $0.90; cap $2.5.
- (K′ 128 at bs 32 needs ≈ 13 GB, beyond the 8 GB cards.)

## 3. Readings (fixes 2, 3, 7, 8)
- **HOLD:** H ≥ 0.95 at every K′ 24–128. **DEPTH:** ≥ FT-CTRL − 0.03. **NOLOSS:** as §230. **VB-C1:** as registered for the family.
- **C1-KEPT:** each decisive C1 statistic of HC is within a pre-registered margin of BASE and of FT-CTRL. Margins are pre-registered (14–100× the measured null noise; worded "within pre-registered margins").

| statistic | margin |
|---|---|
| PROBE-GAIN, PROBE(2) | 0.03 |
| follow_ii, own_ii, STALL, DEEP | 0.02 |
| **PROBE(3–6)** (now decisive) | 0.05 |
| **onset slope** = slope of per-height MEAN onset (replaces the quantised slope_h; paired SD 0.031) | 0.10 |
| mean onset | 0.5 passes |

- **VALUE-SWAP interpretation (fix 3):**
  - own_iv HIGH = the answer is restored from history. Recompute-from-V_0 shows as own_ii up with own_iv down (the draft had this backwards).
  - The G-D'd part_d3hist **OLD-SWAP** arm measures history content reads directly.
  - Arm (v) (a consistent-donor-history splice) **plus a V_0-masked variant** separates "rejects an inconsistent splice" from "lost serial use". It shows chain-level use; step-level use comes from the V_0-masked variant.
  - The "harm only through shared weights" sentence is withdrawn as non-constraining. The anchor trains the shared read policy, so a read shift (carry_likeness 0.83 → 0.60 / 0.69) is the expected first-order effect.
- **Labels, per seed:**
  - **COEXIST** = HOLD ∧ VB-C1 ∧ C1-KEPT ∧ DEPTH ∧ NOLOSS.
  - **COEXIST-SHIFTED** (pre-registered, fix 7) = COEXIST, except that the history-use shift fires (carry_likeness drop ≥ 0.10 vs FT-CTRL, or OLD-SWAP follow ≥ 0.10), while VALUE-SWAP follow_ii and arm (v)'s V_0-masked follow stay within margin. It counts as COEXIST for the claim and for triggering s3, worded "C1 kept, with a shifted read policy".
  - **C1-HARMED:** a decisive C1 statistic outside its margin vs FT-CTRL.
  - **HOLD-FAIL**, **FAR-CHANGED** (descriptive: a read shift with C1 kept).
  - If one reference is unreadable, C1-KEPT reads on the other, with a flag. If both are unreadable → UNREAD.
- **Claim:** COEXIST (or COEXIST-SHIFTED) in s1 and s2; with s3 run, in ≥ 2 of 3.

## 4. c1hold part (fix 6)
- SWAP-HOLD (does the spliced, latent-computed answer persist to K′ 128) and arm (v), built from the existing edit / ver_sub / v0_off_after hooks.
- **Identity gate inside the part:** a per-pair check, plus exact pooled-count equality with eval_vb's vswap.
- Memory check at the actual c1hold shape on the 3090. G-D required.

## 5. OC, cost
- **OC:**
  - C1-KEPT passes under no harm, all decisive statistics, 2 seeds: 0.954 (reproduced; the independence product is conservative).
  - It catches harms of 2–4 points.
  - COEXIST under no C1 harm ≈ 0.66. Theory prior ≈ 0.25–0.30.
- **Cost:** Stage A ≈ $0.90 (cap $2.5) + contingent s3 ≈ $3.3 (cap $4) → **family cap $7**. Realised at registration $0.
