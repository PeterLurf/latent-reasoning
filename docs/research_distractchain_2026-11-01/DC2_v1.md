# DISTRACT-CHAIN v1 (§257): do chain-reading distractors force the loop to select steps by content, while keeping C2? (C1 scope, C2) (2026-11-01)

**Status:** v1. Theory's design at Audit's request (stop line lowered to $12 for this design). Scorer theory/score_dch.py, committed before any output.

## 1. Why (and the math)
**§256 result (Note 256-1):** distractor training unlocked depth extrapolation (d 6 0.98 / 0.96). But it taught a surface skip rule: pass over ops that introduce a fresh name from a fresh given.
- Chain-reading distractors break that rule: H-CHAINSRC 0.69 / 0.81 (j = 2 0.32 / 0.47).
- So does a given placed just before its op: H-GIVPOS before_op 0.34 / 0.65.

**Why chain-reading inserts force content selection.** Let I_j be the insert "X has c more than S_j", placed right after S_j's sentence (j < d). At pass j + 1 the op block contains two sentences whose source is S_j: S_{j+1}'s and I_j's. Three candidate rules:
- **Positional** ("the next op sentence after the last used"): picks I_j whenever it precedes S_{j+1}'s sentence, so it is wrong on those items.
- **The §256 surface rule** ("skip ops whose source is a fresh given"): does not skip I_j, whose source is a chain name, so it is wrong the same way.
- **Content** ("the op sentence whose subject is the cued S_{j+1}"): always right. The cue at pass j + 1 names S_{j+1}, so the key is available.

With I_j present on a fraction q of training items, any non-content rule pays answer CE on those items, while the content rule pays none. So the gradient favours content selection, provided it is learnable from the warm init.
- The given-before-op variant (a fresh given placed directly before its insert op) removes the given-block-position cue in the same way.

## 2. Arms (continue from the §256 models; RTX 3090; micro 2 pinned; 8k final)
| Arm | Runs | Init | Training inserts |
|---|---|---|---|
| DCH | dch_s1 / dch_s2 | dis_s1 / dis_s2 (model.pt, sha asserted on the host) | 0-2 inserts per item, each drawn uniformly from: §256 type (add_c, fresh given); chain-reading add_c I_j (j uniform over 1..d−1, placed after S_j); given-before-op add_c |

**Why continue rather than retrain:**
- dis_s* already extrapolate. 8k of fine-tuning (~$0.6 per cell) keeps the family under budget; a 16k retrain from rcb costs ~$1.2 per cell.
- C2 retention is measured (d 6 ≥ 0.90).
- Disclosed: the §256 inserts stay in the mix, so the new types are added, not substituted.

**Chain-reading inserts in training use add_c only, frames 0-5.** Held-out chain-reading families use other ops.

## 3. Readings (per seed; binding; host-built sets with committed shas; meta carries c, v, j, op and position per item)
**HELD-OUT (never trained):**
- **H-CHAINSRC-OP:** chain-reading inserts with sub_c / mul_c ("X has c fewer than S_j", "X has c times as many as S_j"), j = 1..d−1.
- **H-LIVE:** a two-step insert chain (§256: 0.63 / 0.66; inside 0.13 / 0.27).
- Each needs acc K' 8 ≥ 0.90 and min over positions ≥ 0.80.

**TRAINED-TYPE (in-distribution sanity):** H-CHAINSRC (add_c) ≥ 0.90 and H-GIVPOS ≥ 0.90.

**RETAIN (TEST2):** d 1-4 ≥ 0.98; NOT-DECODER d 4 at K' 2 ≤ 0.20; SWAP4; LP5-D4.

**C2 retention:** TEST2 d 6 ≥ 0.90 at K' 8.

**Guards:** CUE-ONLY; STABLE.

**Labels (first match):**
- VOID / UNSTABLE;
- NOT-CAUSAL;
- C2-LOST;
- CONTENT-SELECTOR (HELD-OUT + TRAINED-TYPE);
- TRAINED-TYPE-ONLY;
- FAIL.
- Per arm: CONTENT-SELECTOR k/2.

**Descriptive:**
- H-OP, H-EXTEND (stopping), K3;
- TEST-PERM: a rise means free lookup;
- d 7-9;
- per-item failure reading on chain-reading items: does the answer equal the chain with I_j applied as step j + 1? (needs c / v in meta).

## 4. OC (per seed)
- P(CONTENT-SELECTOR | content selection is learnable from the dis init in 8k) ≈ 0.6; P(… | it is not, i.e. a new surface cue is found) ≈ 0.1.
- P(C2-LOST) ≈ 0.15.
- **Censoring:** VOID-RERUN once; 8k final; no extension.

## 5. Cost and RAM
- 2 × 8k (~1.1 s/step ≈ 2.4 h) + full eval ≈ $0.6-0.8 per cell. Cell cap $1.2, family guard **$2.6** (credit $15.27 → ≥ $12.6).
- **RAM:** every set is built on the host with committed shas; inits come from an unbuffered predeploy with host sha asserts; module / eval / render shas are asserted against the git blobs (Note 255-6).
