# RED TEAM: SELF-DEPTH-CHAIN round 2 draft (research5/SD2_draft.md, proposed §216)

Clock 2026-10-18 00:18–00:40 EDT (`date`). Opus red-team subagent for Theory. **$0: no GPU, no repo edits** (CPU arithmetic only, 1 thread, nice 19, `python -B`; `find -newer` shows no file written under D:/Latent).
Read: SD2_draft.md (all 600 lines); SELFDEPTH_v1.md (frozen); REDTEAM_SELFDEPTH.md; THEORY §215 + notes 215-1…215-5 (there is no 215-6) + the FT-SELF result + 214-4 / 214-5; LIT_ITERATED_SELFTRAIN_2026-10-17.md; theory/score_s215.py, score_s213 (hist24, acc_gain_after10, META); boxes/vbank.py (main, vb_loss, rec_targets, CurGate), vbank_eval.py, vbank_diag.py (part_hist24, part_chainlabel_fresh, part_selfdepth, _acc_n2_items), selfdepth.py, nested170.test_set; scratch/s215_build_pool.py; the draft's scripts.
Raw files: eval_vb / log / args of s213_d3t_s1, s215_ftself_s1, **s215_ftctrl_s1 (eval_vb only)**; FT-SELF eval_hist24 / eval_selfdepth / eval_chainlabel_fresh; D3-T eval_hist24; the other B-8 VB cells' eval_vb. **Not read:** s215_ftctrl_s1/eval_hist24.json, eval_d3hist.json.
Scripts (research5/rt_sd2/): `rt_curves.py` → rt_curves.json, `rt_literal.py` → rt_literal.json, `rt_oc.py` → rt_oc.json. Tags as the draft.

---

## 0. Verdict

| # | finding | class |
|---|---|---|
| B1 | The s2 lineage cannot be scored as written: R1-s2 "by score_s215's cell() unchanged" is **censored by ARGS-DIFF** (BASE hard-coded `s213_d3t_s1`; `seed` is in neither META nor ALLOWED), and gate1() / part_selfdepth read s215 paths; every s2 mechanism bar is anchored on s1's curves | **BLOCKER** |
| B2 | The ×2 row lets a dead / guard-stopped s2 base, a failed GATE-1-s2 or a failed R1-s2 drop out of the denominator ("1 of 1 competent lineage"): violates dead = fail and C2's "not a lucky subset" | **BLOCKER** |
| M1 | K′-BEYOND-2 is labelled "C2-literal", but ≥ 99 % of it is gain at passes ≤ 20 = K under every panel (round 1's measured KB was 104 % within K) | MAJOR |
| M2 | H_prop is a one-level coincidence; DBS-2 is not "a direct test of H_prop" (H_add passes it too, M2 0.896); the "H_prop = constant +1.15 for this step" claim is false for M2 | MAJOR |
| M3 | One lineage: sampling SE 0.006 is the wrong error bar; at plausible lineage σ 0.06–0.12, P(DBS-2 \| H_shift) 0.08–0.25, mechanism mislabel 13–29 % | MAJOR |
| M4 | Dose: "48k per n = 2× round 1's absorption point" reads one 100-step log value; by 1k blocks round 1 absorbed at ≈ 56k per n, so round 2's new window gets ≈ 0.85× | MAJOR |
| M5 | Reused score_s215 censors pass by default when eval_selfdepth is missing (LABEL-AUDIT, CONTAMINATED); "pending" never becomes failure | MAJOR |
| M6 | MEM "robust / lost" sub-label is defined on round 1's numbers so that round 1 reads "robust"; it renames the registered trade and is proposed to frame the user's OK | MAJOR |
| M7 | KEEPS-SELF-STAB-2 (decided by 0.011 in round 1) has no per-item store (part_hist24 keeps per-n means only) | MAJOR |
| m1–m13 | P1 already decidable; N5 scope; one file per K′-grid reading; per-item BS-CHECK; H_T power; K′24 granularity; s2 gating; D3-T s2 resume; retention of pool_r1; eval_deep contract details; admission headroom; ¬ABSORB-2 follow-up; disclosure | MINOR |

Shaded headline (rt_oc.json): **P(SD2-C2) ≈ 0.06** (0.045–0.10; draft 0.11), P(SD2-SELF) ≈ 0.25 (draft 0.40), P(SD2-C2 \| H_shift) ≈ 0.03, \| H_grow 0.14, \| H_full 0.22; **×2 ≈ 0.005** (0.002–0.014; draft 0.02–0.04).

---

## 1. Blockers

### B1. The s2 lineage's scoring path voids or mislabels it

- **Failure scenario (exact).** Draft §2.4: "R1-s2 scoring: score_s215's registered cell() unchanged". s216_r1_s2 runs with seed 2 (G-S5 "plus seed"). score_s215.args_diff compares with `BASE = "s213_d3t_s1"` and keeps every shared key not in S213.META ∪ ALLOWED; `seed` is in neither (score_s213.py l.34–35, score_s215.py l.36–54). So `shared_diffs = {"seed": (1, 2)}` → ARGS_BAD → "FAILURE (ARGS-DIFF; rule b)" → v1 §9's round-2 condition fails → "lineage stopped at round 1" after ≈ $1.6–2.4 of spend. Also: gate1() reads `s215_gate1` (s1's GATE-1, not GATE-1-s2); part_selfdepth hard-codes `s215_pool`, `s215_gate1`, the cursor fallback `s215_ftself_s1` and contamination vs test n_max 20 (vbank_diag l.1377–1410).
- **Mislabel scenario.** Every mechanism bar is an s1 number: R1-s2 MARGIN-GROWS 0.63 = D3-T **s1**'s exact-H_shift 0.480 + 0.15; R2-s2 MARGIN-GROWS-2 0.785 and SHRINK 0.60 are midpoints of **FT-SELF s1**'s panels. If R1-s2's own curve reads, e.g., M 0.80 (more growth than s1), R2-s2's exact H_shift has M2 ≈ 0.80 and reads GROW / DBS-2 with zero round-2 growth. The B-8 family spans M(n 9–12) 0.31–0.49 and n50 9.51–10.31 (rt_curves B8_cells_n50), so s2's base can sit anywhere in that band.
- **Fix.** Register `score_s216.r1_cell(lineage)`: ARGS-DIFF vs s215_ftself_s1/args.json with {seed, init_from, self_pool, out, self_micro} allowed; gate / pool / cursor / contamination paths from the lineage (test n_max 24). **Mechanism bars by formula, per lineage:** R1-s2's H_shift panel = its own base's a(9..12) translated (+0.15 for MG, as v1); R2-s2's H_shift / H_prop panels and midpoints = sd2_tables' formula applied to R1-s2's own eval_vb, frozen as a formula now. DBS-2 (absolute 0.80) stays absolute. Add an R1-s2 fixture (seed 2, own gate) to G-S12.

### B2. The ×2 row counts only the lineages that survive

- **Failure scenario.** §2.4 COMPETENT-BASE includes "not dead (48k)" and "GATE-1-s2 passes"; §4.5 says a non-competent base reads "SD2-C2 instance in 1 of 1 competent lineage". A guard stop, host loss or NaN in s216_d3t_s2 is therefore an *exclusion*, not a failure; so is a labeller that fails GATE-1-s2. "Lineage stopped at round 1" (R1-s2 ¬ABSORB / FT-DIVERGED / LABEL-NOISY / CONTAMINATED) has no defined ×2 value. That is the censoring rule broken (dead / diverged / contaminated cells fail every reading; never-reached events count as failures) and exactly C2's "lucky subset" problem.
- **Fix.** COMPETENT-BASE := the 48k run **completed** ∧ acc_n2 ≥ 0.99 ∧ a_24 ≥ 0.97 at every n 1–8 (entry step reported). That is the only exclusion. A dead base, a failed GATE-1-s2 / GATE-2-s2 (both labellers), any R1-s2 or R2-s2 censor, and R1-s2 failing v1 §9's condition each make **SD2-C2 ×2 FAIL**, worded "SD2-C2 in 1 of 2 competent lineages". A rerun of a dead base needs the user's OK and is disclosed.

---

## 2. Majors

### M1. K′-BEYOND-2 is not a K′ > K reading

- On FT-SELF, accuracy saturates at K′ = n, and the coda finishes one reduction: a_n(n) = 0.999 / 0.906 / 0.642 / 0.355 vs a(n) 0.999 / 0.893 / 0.633 / 0.344 (n 13–16); a_{n−1}(16) = 0.336 [MEASURED eval_vb; rt_curves].
- Round 1's KB 0.155 = **0.1615 from passes 15–16 (≤ K 16) + (−0.0065) from passes > 16**.
- Round 2 on the draft's own panels (rt_literal): KB2 = 0.159 / 0.292 / 0.513 (H_shift / H_prop / H_add), of which passes > 20 give −0.003 / −0.006 / +0.002. RISE-PAST-K-2 (n 19, 20 vs K′ 20) is ≈ 0 by construction.
- At n ≤ K + 1 the C2 quantity "accuracy rises with latent steps past the trained budget" cannot be non-zero. **Literal C2 needs n ≥ K + 2 = 22.**
- **Fix.** Rename KB2 "pass gain at out-of-support depth (within K)": descriptive, mechanism only, never "C2-literal". Add **LIT-C2-2 = mean_{n 22..24}[a(n) − a_20(n)] ≥ 0.10** (eval_deep) as the C2-literal row on SD2-C2's line. Panels: H_shift 0.005, H_prop 0.038, H_add 0.030, H_full floor ≈ 0.71 [DERIVED rt_literal]. It passes only if the boundary stops binding, which is the honest C2 statement.

### M2. H_prop is a coincidence at one level; DBS-2 does not test it

- **margin/B ratio, round 1 / base** [DERIVED rt_curves]:

  | accuracy level | 95 % | 90 % | 80 % | 70 % | 60 % | 50 % | 40 % |
  |---|---|---|---|---|---|---|---|
  | ratio | 1.50 | 1.20 | 1.14 | 1.08 | 1.03 | **1.00** | 0.975 |

  The 0.288 = 0.288 match is where a monotone trend crosses 1.
- **Back-cast.** H_prop applied base → round 1 (a1(12 + d) = a0(8 + 2d/3)) predicts 0.949 / 0.816 / 0.603 / 0.378, M 0.687, against the measured 0.999 / 0.893 / 0.633 / 0.344, M 0.717. The 50 %-slope is predicted 0.225 /n and measured 0.289 (base 0.337). Round 1 looks like a **translation by ≈ +5 with slight sharpening**, not a stretch.
- **The fit uses 1 degree of freedom on 2 points of different kind.** Base = 48k from scratch on gold, lr 1e-4, curriculum. Round 1 = 12k FT on self labels at 3e-5. At fixed B 8 the recipe spread of margin/B is 0.19–0.29, and D3-T is the top of the family. Yang Thm 2's proportionality (constant K − H headroom) is assumption-distant (graph connectivity, gradient flow).
- **Consequence.** H_add (round 1 translated +5.15: the same n50 20.61 as H_prop) gives **M2 0.896** vs H_prop 0.852. Both pass DBS-2 and MARGIN-GROWS-2. So the draft's §3.4 line "H_prop coincides with a constant +1.15 … differ only under a different window" is false for M2, and "DBS-2 is a direct test of H_prop" (§0, §5.1) is false: DBS-2 tests *renewed growth vs H_shift*.
- **Fix.**
  - Reword §0 / §3.4 / §5.1 accordingly.
  - Add descriptive **WIDTH-2 = n50 − n90** (eval_deep): H_shift 1.54, H_add 1.58, H_prop 2.23; round 1 went 1.23 → 1.53, i.e. ×1.24 where H_prop predicts ×1.5. WIDTH is the H_prop-specific split; n50 is not.
  - Drop the "0.50 face-value" sensitivity from any headline.

### M3. One lineage: the right error bar is lineage noise, not 0.006

- No FT-seed replicate exists. FT-CTRL vs D3-T (M_9–12 0.484 vs 0.480) is an FT that barely moved the model, so it bounds nothing for a self-item FT. Kuo's 0.09–0.12 is from scratch.
- At Gaussian σ on M2 [DERIVED rt_curves lineage_noise_OC]:

  | σ | P(DBS-2 \| H_shift) | P(DBS-2 \| H_prop) | LR | P(MG-2 mislabel \| H_shift / H_prop) |
  |---|---|---|---|---|
  | 0.06 | 0.08 | 0.80 | 9.6 | 0.13 / 0.13 |
  | 0.09 | 0.18 | 0.72 | 4.0 | 0.23 / 0.23 |
  | 0.12 | 0.25 | 0.67 | 2.7 | 0.29 / 0.29 |

- **What one lineage can claim:** "a DBS-2 instance" (existence), with the mechanism label "consistent with", never "H_prop confirmed / refuted".
- **Fix.** Replace "14 SE / 10 SE" discrimination language with the table above. State P(pass \| no effect) for DBS-2 as 0.05–0.25 (σ-dependent). The headline OC uses P(DBS \| H_shift) ≈ 0.12.

### M4. The new-window dose is ≈ 0.85× round 1's absorption point, not 2×

- **Round 1's self_answer_ce, 1k-block means** [MEASURED s215_ftself_s1/log.jsonl]: 0.274, 0.049, **0.021 (3k)**, 0.030, 0.016, 0.018, **0.0044 (7k)**, 0.0033 …
  - The draft's 0.0095 at 3k is a single 100-step window.
  - It stays < 0.02 only after step 5.7k (182k items ≈ 46k per n); the first block ≤ 0.005 is at 7k (≈ 224k items ≈ 56k per n).
- **Round 2** gives the new window 48k per n over all 12k steps (≈ 28k by step 7k, with lr already below 2.5e-5). The other half of the self items is n 9–12 that FT-SELF solves at 1.000 (near-zero-gradient rehearsal [ASSUMED]).
- The new window is also harder: 16 serial passes, T to 61.
- **Fix.**
  - P(ABSORB-2) ≈ 0.6.
  - Pre-state the ¬ABSORB-2 follow-up (v1 §9's "one disclosed re-run with the user's OK; else stop" is not carried over).
  - Keep the 4k / 8k checkpoints until read, for a descriptive acc_n2 n 13–16 learning curve (≈ 10 min each on the 3070; RT round 1 §7.5). This replaces T2.

### M5. Censors that pass when their input is missing

- score_s215.cell():
  - `la = sd.get("label_audit") or {}`; `noisy = … and bool(la) and …`, so a missing eval_selfdepth gives **no LABEL-NOISY**.
  - `contam` reads `pool_test_hits` with default 0, so it gives **not CONTAMINATED**.
  - SD-SELF does not need eval_selfdepth, so a cell whose selfdepth part died (the 3070 was lost mid-chain at 22:56 on 10-17) would read SD2-SELF / SD2-C2 **uncensored**.
  - A missing eval_vb returns "pending" forever.
- **Fix.** score_s216 gets a `final` mode. A missing label_audit, contamination block, eval_vb, eval_deep (already in the draft), eval_steppos / lfk or hist24 then reads as a censor (audit / contamination) or a failure (readings). Add a fixture for each.

### M6. The MEM "robust / lost" sub-label is a post-hoc rescue

- **Where its numbers come from.** "Robust" = STAB_self < 0.5 ∧ STAB_dup_newest ≥ 0.10 ∧ intact a_24 = 1.0. Its conditions are round 1's numbers (dup_newest 0.877; self-ablated a_24 0.43–0.65 vs D3-T 0.21–0.28). STAB_dup_newest ≥ 0.10 is already part of CONTENT-STABILISES.
- **It is the registered MEM-TRADED renamed.** "Robust" ≡ "KEEPS-SELF-STAB failed on the self-share term only".
- **Its interpretation is the trade itself.** "Stable without own-slot history" is what MEM-TRADED means ("memory channel lost"). FT-SELF's bank-ablated a_24 also rose (0.18–0.27 → 0.28–0.53): less use of history in general.
- **It is not exhaustive.** STAB_self < 0.5 ∧ CONTENT-STABILISES ∧ intact a_24 < 1 at some n 3–6 gets no label.
- **Fix.**
  - Neutral descriptive names ("MEM-TRADED: own-slot share" / "MEM-TRADED: content"), marked "defined after round 1's readout".
  - Keep it out of the user-OK framing.
  - Score the §215 pair's MEM attribution before asking the user: FT-CTRL's hist24 has been on disk since 22:51, and the recipe does not change across branches.
  - Show the user the raw arms.

### M7. Per-item store behind KEEPS-SELF-STAB-2

part_hist24 writes per-n means only (vbank_diag l.948–949). Round 1's MEM-TRADED was decided by 0.011, and at n 3–6 every 0.001 is one item. Under the store-items rule, add per-arm per-item answers at n 3–8 (both K′ n + 2 and 24) to E2's edit list.

---

## 3. Minors

- **m1 P1 is already read.** FT-CTRL's STOP-MOVED (acc_gain_after10, from eval_vb only, which is unblinded) has mean gain **−0.021** (n 11–14: −0.019 / +0.004 / −0.036 / −0.034) [DERIVED rt_curves]. So P2 cannot occur unless CTRL is censored. Only the MEM branches (M-S / M-F / C-X) are pending. No branch runs SD2 against evidence that the self items didn't cause the gain.
- **m2 N5.**
  - No leak into labeller choice (fixed by rule + GATE on fresh items), thresholds (frozen) or fallbacks (GATE-based).
  - The risk is informal go/no-go. Pre-state that no launch / stop of any SD2 cell depends on N5, and run N5 after s216_r2_s1 has launched.
  - Keep FT-SELF's eval_deep: it is the true zero-round-2 level.
  - D3-T / FT-CTRL baselines are ≈ chance at n ≥ 13 (0.08–0.14 measured at n 13–16) and cost 3070 time: optional.
- **m3 One file per K′-grid reading.**
  - STOP-LOC-2's G(13..16) straddle n 16 / 17; the "n ≤ 16 from eval_vb" notation would mix files. Compute STOP-LOC-2, n50, WIDTH-2 and LIT-C2-2 entirely from eval_deep.
  - BS-CHECK per item: eval_deep n 13–16 vs E2's items_n2, at most 3 mismatches per n. Round-1 precedent: items_n2 (length-sorted chunks, rollout to n + 2) equals eval_vb acc_n2 (test-order chunks, rollout to 24) exactly at every n 9–16 [MEASURED], so fp32 chunking flips are ≈ 0.
- **m4 H_T.**
  - DBS-2's window has the same untrained-row exposure as round 1's DBS window (+12 rows: T ≤ 73 vs 61, then T ≤ 59 vs 47), so round 1's no-T-effect precedent covers it. n 20 has ≈ 45 % of items at T ≥ 63.
  - The ≥ 100-items-per-band rule qualifies at about n 20–21 only (n 19 ≈ 10 % ≥ 63; n 22 ≈ 3 % ≤ 61).
  - Within-n T tracks bracket structure, so a flag is confounded with structure.
  - Add M2 on T ≤ 61 as descriptive.
- **m5 K′24-STABLE granularity.** n 1 has 415 distinct test items (multiplicity ≤ 7). At the 0.990 edge one distinct item moves a_24(1) by up to 0.007, so P(K′24-STABLE) ≈ 0.5–0.6. FT-DIVERGED (a_24 < 0.97) is a whole-cell censor with no term in the draft's OC: add ≈ 0.9.
- **m6 s2 gating.** Pre-state that R2-s2 runs whatever R1-s2's mechanism label. This overrides v1 §9's "SHIFT → optional", to avoid outcome selection, and must be disclosed. State whether the user's SD2 OK covers an R1-s2 MEM-TRADED.
- **m7** D3-T s2 (no init_from, calibrate() at step 0) needs its own resume smoke; G-S4 is init_from-specific.
- **m8 Retention.** v1 §8 deletes pool_r1 after audit, but SD2 excludes its syms. Keep it (or its syms hash) until the s216 pools are built.
- **m9 eval_deep contract.** Fix the items_sha256 serialisation (json separators, order). Make the scorer assert card class 3070 / fp32 / tf32 off. Keep the test-order position per item.
- **m10 Admission headroom.**
  - The upper projection $1.27 vs admission $1.35 is 6 %, and the 3090 launch-bound speed varies up to ≈ 2× across hosts.
  - Expect refusals: pre-state the "not run, user told" path for R2 cells.
  - The 3070 queue is ≈ 20–25 h for the family.
- **m11 Fallback wording.** Under the BASE-LABELLED fallback, SD2-SELF's "the labeller was itself self-trained" is false: pre-state the alternative wording.
- **m12 Disclosure.** testset_ext.py read T and y-class shares of the n 17–24 test items before registration. These are generator statistics, not model outputs, so harmless, but the "sealed / exclusion-only" text should say so.
- **m13 Seed 1.**
  - Round 2's plain batches, coins, self step indices and self U values are identical to round 1's [VERIFIED vbank.py draw_step]. The 384k plain items (n ≤ 8, all at 1.000) are seen a second time.
  - This is no confound for comparator-free readings. It is a plus: round 1 → round 2 differences are init + self items only. Disclose it.

---

## 4. OC (rt_oc.json; every prior [ASSUMED])

| | draft | **RT shaded** |
|---|---|---|
| P(ABSORB-2) | 0.75 | **0.60** (M4) |
| kept | LF 0.9 · VB 0.9 · K′24 0.70 · RETAIN 0.95 = 0.539 | LF 0.9 · VB 0.9 · K′24 **0.60** · RETAIN 0.95 · **¬DIVERGED 0.90** = 0.416 |
| priors given ABSORB-2 (shift / grow / full / shrink) | 0.542 / 0.375 / 0.025 / 0.058 | 0.60 / 0.30 / 0.02 / 0.08 |
| P(DBS-2 \| H) | 0.05 / 0.60 / 0.95 / 0 | **0.12** / 0.55 / 0.90 / 0 (M3) |
| P(SD2-SELF) | 0.404 | **0.249** |
| **P(SD2-C2)** | 0.111 | **0.064** (0.045–0.099; 0.073 keeping the ×1.5 prior) |
| P(SD2-C2 \| H_shift / H_grow / H_full) | 0.020 / 0.242 / 0.384 | 0.030 / 0.137 / 0.224 |
| SD2-C2-MEM | 0.028 | 0.013 |
| **SD2-C2 ×2** | 0.02–0.04 | **≈ 0.005** (0.002–0.014; P(s2 competent) 0.75, R1-s2 OK 0.70) |
| LIT-C2-2 (new) | — | ≈ P(H_full) × 0.9 × SD2-SELF ≈ 0.005 |

How to read it: a DBS-2 pass is evidence for renewed growth (likelihood ratio ≈ 4–10), not for H_prop specifically, and not literal C2. SD2's main expected product is SD2-SELF (depth 16 in support) plus a mechanism instance.

---

## 5. The draft's 12 open questions

1. **(b)**, with B1 and B2 fixed (lineage-relative bars; dead = fail).
2. **Confirm N5, restricted.** After registration and after s216_r2_s1 launches; FT-SELF's eval_deep is required; D3-T's and CTRL's are optional; no decision depends on it.
3. **eval_bs 16 + per-item BS-CHECK.** Every K′-grid reading at n ≥ 13 comes from eval_deep.
4. **T1 (micro 8): yes**, cheap insurance against a refused host (needs G-S7 + diff review). **T2: no**; use the 4k / 8k checkpoint evals.
5. **MEM sub-label:** neutral names, marked post hoc, kept out of the user-OK framing. Score the pair's MEM first.
6. **R2-CTRL: never by default.** Offer it only if the user asks the MEM-dose question.
7. **Shade** as in §4; keep the pre-stated ×1.5 as a sensitivity row. Headline ≈ 0.06.
8. **Confirm K′24-STABLE unchanged.** Add FT-DIVERGED to the OC.
9. **For s1, confirm** 0.80 / 0.785 / 0.60 / M_far 0.60. s2's bars come by formula from its own R1. Add LIT-C2-2 (≥ 0.10) and WIDTH-2. The H_T flag is descriptive at n 20–21.
10. **Mixed windows** (round-1 semantics; $0.15 is not worth a composition change).
11. **Seed 1** (keep, and disclose the identical plain stream).
12. **Halting on P2 is right.** P2 is already excluded by CTRL's eval_vb (−0.021). Ask the user after the pair's MEM is scored.

*Red team completed 2026-10-18 00:40 EDT. CPU arithmetic only; no repo file written (checked with find -newer).*
