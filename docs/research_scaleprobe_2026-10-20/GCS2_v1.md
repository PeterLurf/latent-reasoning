# GSM-CACHE-SWAP-2 v1 (§221 on registration): §220's battery behind a fidelity gate (2026-10-21)

**Status:** v1 for Audit. Cap $0. Successor to §220, which closed UNREAD at stage 0 (c9ddcd0a). Draft: GCS2_draft.md (865bf267). Audit's rulings on the draft are folded in: data reuse with provenance plus a bitwise subset; CODI20 determinism; reporting.

## 0. Honesty lines
- **This design is made after seeing §220's GATE outcomes and nothing else.** Seen: agree, acc_bs1, acc_official, official_self, the asserts.
- **No swap-arm, descriptive or NECESSARY value of any §220 model has been read by anyone.** They are sealed: scratch/s220/SEALED_score_s220_arms.json, sha 7c3822cf…, uncommitted.
- **G-FID(b) is already known to pass.** Both bs-1 accuracies are in §220's gate blocks: CODI20 0.3791 against 0.379; zen 0.4245640636846095, bitwise equal to §209 G1′. So (b) is a recorded fact, not new evidence.
- **The new evidence in G-FID is (a)**: port fidelity at bs 1 and the bitwise re-run subset.
- **§220 stays UNREAD forever.** Its sealed appendix wording ("pad-free bs-1 harness, gate-failed, not a reading") applies to §220 only.

## 1. Inherited unchanged
GCS_v1.md at 3a88762e plus Addendum 1, with Notes 220-1 and 220-2:
- models, stages, pins, items (HF gsm8k test, sha 3d497ddb…), bs 1, parser;
- exact-length pairs (rng 22020, ≤ 3 uses per donor) and the graph-matched rule;
- every arm and every assert;
- permutation floors, readings (CACHE-CARRIER-GSM), label precedence, the outcome → action table (v1 §6), OC, censoring (rule b), wording;
- the stage-1 trigger: CARRIER-PORTS in either GPT-2 model.

## 2. The change: G-R → G-FID (G-R is removed)
- **G-FID(a), port fidelity:**
  - **zen-gpt2:** the official zhenyi4/codi test.py at **batch size 1**, where no pads exist and the semantics are well defined, on all 1,319 items. Our port's bs-1 INTACT must agree per item at ≥ 0.97 (parsed predictions, U10) with |Δacc| ≤ 0.01. The official repo is pinned by commit sha and run from its own directory (untrusted-code rule).
  - **CODI20:** no official path exists, because the port is its training harness (codi_port). (a) is a **bitwise determinism** check instead (Audit (2)).
  - **Stage-1 CODI-1B** (only if triggered): as zen, with the official test.py at bs 1.
- **G-FID(a′), bitwise re-run subset (Audit (1)):** for each stage-0 model, the **first 50 primary pairs by list index (pairs[0:50])** of its pinned eval_gcs.json are re-run with s220_gcs.py at the pinned provenance (same card, dtype, bs 1). Three things must reproduce **bitwise**:
  - INTACT generated-token sha for each of those 50 recipients (intact_sha);
  - CACHE-SWAP generated-token sha (pairs[i].sha.cache);
  - LEAK-SEG generated-token sha (pairs[i].sha.leak).
  - **Any mismatch voids the reuse, and the full battery re-runs** under this design before any reading.
- **G-FID(b), established accuracy (recorded fact):** CODI20 |acc − 0.379| ≤ 0.025; zen |acc − 0.4246| ≤ 0.025; CODI-1B 0.556 ± 0.03 as a flag.
- **Padded official-semantics runs** (virtual bs 128 and the 220-3 shuffles) are reported descriptively. They are never a gate.

## 3. Data (reuse with provenance)
| model | file | sha256 | provenance |
|---|---|---|---|
| CODI20 | boxes/runs_fsq/gsm_codi_20ep_s1/eval_gcs.json | 82aa84138eea9a390d713cdb0f09c4f7e78deac295a09519ce2f8d716b6bef36 | committed 7cf3913c (01:55 10-21); s220_gcs.py from a clean tree (script_git_status empty), run commit 59a90e07; RTX 2070, fp32 (LoRA merged), bs 1; transformers 5.18.0 |
| zen-gpt2 | scratch/s220/zen_codi_gpt2/eval_gcs.json | fec1f9ec1bda121d33485b8a325fb92e963756b3b0ef1f1fd195ba6b512de769 | committed 064fa9f7 (03:14 10-21); clean tree, run commit b1a6ae32; RTX 3070, fp32 (LoRA merged), bs 1; transformers 5.18.0 |

- Both files were produced by s220_gcs.py at the G-D-reviewed code (e5ea9325). The later f2a3850b change touches only the --official_shuffle branch.
- The G-FID(a′) re-runs use the same script commit as each file and record their own sha.
- **Order:**
  1. §221 is registered and v1 frozen.
  2. score_s221 is written and fixture-checked.
  3. G-FID runs.
  4. Only then is the seal lifted, and score_s221 reads the pinned files.

## 4. Reporting
- §221's readings are computed from the pinned bytes, each line carrying a provenance tag: *"generated under §220's run, sealed, unread until §221's freeze (sha …)"*.
- The §220 appendix wording is not used for §221 readings.

## 5. OC, labels, controls
- Inherited unchanged from GCS_v1 §5–§8 and Addendum 1. P(pass | no effect) ≈ 0; P(pass | floor + 0.35) 0.98–0.99 at about 480 / 545 pairs.
- G-FID is a known-answer gate: bs 1 is deterministic with no pads, so P(fail | faithful port) ≈ 0. A failure means a port fault, and the design reads UNREAD.
- **Realised $: $0.**

## 6. Scorer
- theory/score_s221.py = score_s220 with G-R replaced by G-FID ((a) agreement or determinism, (a′) bitwise subset, (b) accuracy) and the provenance sha pins asserted.
- Fixture-checked before the seal lifts. The sealed sidecar mechanism stays for any model that reads UNREAD.
