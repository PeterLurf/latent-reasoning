# PROSQA-OOD port: registration-ready design (wave-4 lead 05, 2026-11-03)

Covers backlog **D1 PROSQA-OOD** (Lane D) and **BM1 PRONTOQA-DEPTH** (§13.2, sister split). Sources read: Audit's `PROSQA_OOD_PORT.md` and `EXTERNAL_SOTA.md` §4 (`docs/research_audit_2026-11-03/`); `docs/LIT_SUBLAT_PROSQA_OOD_2026-11-03.md`; `boxes/prosqa168s2.py` (all 660 l), `boxes/prosqa168.py`, `boxes/prosqa_sym.py` (bfs / sig_of / reach_targets), `boxes/inject_aw.py` hold_step (l. 159–200), `boxes/deltamem.py` (zero_state l. 86, write_seq l. 179), `prosqa/` top level (make_long.py, make_deep.py, eval_p8.py, gpt2c.py rollout l. 80–112, analyze.parse l. 45–70, pod_p8c.sh, pod_prosqa_g.sh, yaml configs, ckpt/ and data/ listings), THEORY §191 / §201 T2 / §261 / Note 168-6 / l. 13588. Read-only; nothing is registered; no stored number was recomputed.

## 1. Goal

Headline H2 (backlog §12.8): the §191 latent loop on a **public** depth / structure split with same-split own-run baselines, under C1 removal gates, 3 seeds, Wilson CIs. The claim sought is "beats own-run CoT-SFT and Coconut on 7-hop / 8-hop, does not trail on Dense; DAG is the stretch", scoped as *text-step supervision, target given (the question), plan-free, GPT-2 small, one lineage*.

Two corrections to the Audit memo, found in the repo:
1. The "three existing Coconut seeds" `prosqa/ckpt/g_s0–2_checkpoint_50` are PROTOCOL6 **addendum G** models (`prosqa/patch_mask.py`: the answer cannot attend to the question; `pod_prosqa_g.sh`). The only plain official-Coconut checkpoint on the standard split is `prosqa/ckpt/checkpoint_50` (seed 0, `prosqa_1gpu.yaml`, PROTOCOL6 C/D; `scratch/pods.txt` l. 5). The `p8c_coco_s1–4` seeds are trained on deep p8 data, not the standard split. So Stage 2 needs **two new official-Coconut seeds** (~$3 each), and the g_* models are a descriptive "Coconut, answer-masked" row only.
2. "Chen 2603.21676's prediction (168-13)" is **withdrawn as mis-cited** (THEORY l. 13588: Chen's loss demands the final answer at every recurrence; it says nothing about path supervision on DAGs). The fetched title is "Thinking Deeper, Not Longer" (Hung-Hsuan Chen, depth-recurrent transformers). It is not cited here for the DAG risk.

A third finding changes the port's shape: in `prosqa168s2.py` the think input is already `[EOT] + question` (l. 140–141), and the question ends "Is R a A or B?". **The §261 target cue is inherent**: every pass sees the question and nothing else, byte-identical across passes (so `--cache_lower` is exact, l. 148–150). No cue code is needed. The write is from **every token** of the pass (`DM.gpt_forward_mem(...)[1]`, l. 146; write_seq is sequential over T with padding masked), not bun's one ':' token. The frontier question (§5) is therefore "can an all-token DeltaMem write carry a set?", which is testable by probe rather than assumed impossible.

## 2. Data: regenerated splits (`prosqa/gen_prosqa.py`, new, ~250 l)

SubLAT (2610.40283) releases no generator, data, sizes or licence (Appendix A.1 only names the splits; re-fetched 2026-11-03). Coconut (2412.06769, App. A.2, Alg. 1; HTML fetched) gives the base law, which we re-implement; percentages below are derived from the algorithm's thresholds, as the text does not state them.

**Base law G(N, mu, w):** nodes added one at a time; node 0 = Entity (root); a new node draws `poisson(mu)` in-edges (capped by the candidate count) from a candidate set chosen at random: ~35 % descendants of node 0 plus node 1 (excluding node 1's descendants), ~35 % descendants of node 0 plus node 2 (excluding node 0's descendants), ~30 % all nodes; candidates weighted `1.5 * depth_to_root + 1` (we expose the 1.5 as `w`). Parentless nodes get entity names, the rest concept names. Target = a leaf descended from the root; foil = a leaf not reachable from the root; options permuted. Coconut's train statistics: 23.0 nodes, 36.0 edges, shortest path 3.8, 1.6 shortest paths. Item schema = Coconut's (`question / answer / steps / idx_to_symbol / edges / root / target / neg_target`), plus `split`, `hops` (shortest root → target distance), `n_paths`, `n_disjoint`, `ext` (an integer split code, so `prosqa/eval_p8.py` l. 52–57 groups by it unchanged), `gen_seed`.

| split | knobs (fixed here; the registration copies them) | n | tokens (est.) |
|---|---|---|---|
| ID test | the official `prosqa/data/prosqa_test.json` (500), untouched | 500 | ~300 |
| 7-hop | base law, `w` 3.0, N ~ U{26..34}; **rejection: hops = 7**; foil's subtree depth ≥ 3 (so the foil side is not shallow); other knobs as train | 500 | ~400 |
| 8-hop | as 7-hop with hops = 8 | 500 | ~430 |
| DAG | base law, N ~ U{22..26}, hops 3–6 drawn from the train hop histogram; **rejection: ≥ 2 internally vertex-disjoint root → target paths** (unit-capacity max-flow) **and** ≥ 1 merge node (two paths rejoin before the target); foil attached to an off-path branch that is reachable from a non-root entity, as ProsQA-Ext's two-path form (2609.35643) | 500 | ~320 |
| Dense | base law, `mu` 1.5 → **3.0**, N as train, hops drawn from the train histogram | 500 | ~560 |
| len-ctl | base law at train knobs, plus k decoy edges among nodes unreachable from the root (root reachability, hops, target and foil unchanged), k chosen per item so the statement count matches Dense's mean (~70) | 500 | ~560 |

Every split: 500 items, generator seed `20261103 + split_index`, vocabulary `ENT17` / `CON38` (`prosqa168s2.py` l. 48–51; 34 concept nodes fit 38), names drawn per item, premise order shuffled, options swapped with p 0.5. The option-level §57 proximity cue is neutralised by `theory/prosqa_na.py neutralise()` (as p8_na; §59 read 0.50 at every depth) and measured. **Dedupe:** no OOD graph equals a train graph by (sorted edges, root, target, neg_target) after canonical BFS relabelling from the root (§191's criterion), asserted at generation; `make_long.extend`'s hung chains are not used (Note 168-6: unbranched chains are not SubLAT's hops).

Train: the official 17,886-item `prosqa_train.json` with §191's per-draw RELABEL (`relabel()` l. 77–99), declared as augmentation under "same data" (no new graphs); the vocabulary assert at l. 294 stays. Validation: `prosqa_valid.json` (300) for the test500-style trajectory; the final checkpoint is primary and SubLAT's best-val choice is reported descriptively (val selection is a second look).

**ProntoQA sister split (BM1, same harness).** ProntoQA (2210.01240; OOD version 2305.15269) is a linear is-a chain over a fictional ontology with the same "Every X is a Y." surface form. We render it into the ProsQA schema: a root entity, one gold chain of h hops, and ProntoQA-OOD-style distractor chains hanging from non-root entities; question "Is R a A or B?" with B the end of a distractor chain (so `candidate_margins` applies unchanged). Train hops 1–5 (ProntoQA's standard range), test hops {2, 4, 6, 8, 10, 12}, 500 per hop, 17.9k train items to match ProsQA's size, same relabel. The backlog scout reports its generator as Apache-2.0; the repo was unreachable today (404), so the build verifies the licence before any item is written. It registers separately (own OC block) after ProsQA Stage 1 and shares `gen_prosqa.py --law chain`, the trainer and the scorer.

## 3. Model and training (`boxes/prosqa_aw.py`, a copy of `prosqa168s2.py` with asserted edits)

Recipe = §191 A exactly (`stepaux`, K 10, aux_step 1, aux_fade 12000,12000, `--aux_reweight`, `--relabel`, `--cache_lower`, `--ckpt_passes`, `--fast_add`, `--chunk_delta 64`, bs 64, lr 1e-4, 12k steps, `--test500_every 1000`), plus:

- `--data std` (l. 257 choices and the hard-coded loader at l. 284 → a FILES map as `prosqa168.py` l. 25–27): train / valid / test from `prosqa/data/`, OOD from `prosqa/ood/*.json`, md5 of every file written to args.json and asserted on the host.
- `HIST_STD` / `AUX_W_STD`: the 168-12 reweight formula `w_t = E_bar / E_t, E_t = sum_{d>=t} c_d / d` recomputed on the standard histogram {3: 7140, 4: 7645, 5: 2541, 6: 560} (l. 42–45, and the assert at l. 379 switched by `--data`); K 10 keeps passes 7–10 as hold passes.
- `--hold_inc lam` (0 = off, bitwise): §240's L_inc ported to a per-item d. After `dec_loss` (l. 334) the detached states S_0..S_K are kept; for each item and each t in H_i = {d_i + 3 .. K}, S_pen = write(sg S_{t-1}, x) with its own lower blocks, r_t = ||S_pen − sg S_{t−1}||² / ||sg S_{d_i+2}||², L_inc = lam · mean_i mean_{t in H_i} r_t, backward-ed chunk by chunk as `hold_step` does (inject_aw.py l. 171–185). The cue is the same x at every pass, so no cue swap exists. To bound cost, two hold passes per item are drawn per step (`--hold_sub 2`); lam 0.1, start 500 (§240 defaults). Hold is **not** a zero-init pathway; its usage monitor is the logged comp / hold ratio.
- Eval block `ood` (after `acc_by_depth_kprime`, l. 556): per split, `acc_by_kprime` for K' in 0..24 (every integer, so the stability halt is computed from the same sweep), `n`, `wilson95`; `removal` = {k0, donor (the `path_indep` derangement l. 588–591, K' 10 from a donor's S_10), noise (Gaussian S with the item's per-head norm)}; `halt_stab` = first t with sign(M_t) unchanged for 3 passes (Note 259-11's rule), its accuracy and mean t by hops; `aux_path` (descriptive): at each pass the greedy aux step's object (l. 599) is classed gold-path / other-valid-path / unreachable / option, using `prosqa_sym.bfs` on the item's edges; `frontier_probe` (§5).
- `--time_steps 50` on the host before launch (l. 453–458 exists) to fix the s/step; the memo's 7.0 s/step was p8_na at ~450 tokens, standard items are shorter.

Baselines, same data, same relabel, 3 seeds each: **CoT-SFT** = `--arm textcot --data std` (Coconut's CoT format, l. 208–210; greedy CoT then candidate scoring); **No-CoT** = `boxes/prosqa168.py --data prosqa_std` (exists, l. 25) plus an `--ood_dir` eval; **Coconut** = official code via a `pod_p8c.sh`-style script on `prosqa_1gpu.yaml` (seeds 1, 2 new; seed 0 = `ckpt/checkpoint_50`), evaluated on the OOD JSONs with `eval_p8.py`-style scoring through `gpt2c.rollout(K=6)` with `mask_answer=False` (l. 71, 100); **CODI** = the public checkpoint cited (2502.21074). The §208 AB / QO options (l. 118–138) stay off.

## 4. Gates (before any spend; Experiments runs them after registration)

1. **GEN-STATS** (CPU, $0): 5,000 base-law draws reproduce Coconut's train statistics within nodes 23.0 ± 1.0, edges 36.0 ± 2.0, shortest path 3.8 ± 0.2, n shortest paths 1.6 ± 0.2, and the train hop histogram within ± 3 points per hop. Fails → the law is wrong; no split is written.
2. **SPLIT-INVARIANTS** per item (asserted in the generator, reported in `ood_stats.json`): target reachable and foil unreachable from the root (BFS); `hops` equals the split's bar; DAG `n_disjoint ≥ 2` with a merge; Dense mean in-degree ≥ 2.8; `steps` is an edge path root → target; names within ENT17 / CON38; dedupe vs train; proximity cue 0.50 ± 0.03 per split; len-ctl token count within ± 5 % of Dense's.
3. **FLAGS-OFF BITWISE:** `prosqa_aw --data na` with every new flag at its default equals `prosqa168s2` on 8 items, CPU fp32: `torch.equal` on the loss, the margins and the aux logits, and an identical 20-step loss trace. `--hold_inc 0` with the flag present is included in the same gate (p = 0 equivalence).
4. **HOLD gradflow:** `gradflow_check` on L_inc alone: non-zero gradient in the memory's qkv / gate / W_o and blocks 6–11, zero into the detached S_{t−1}, zero into the aux decoder.
5. **mem_check** (l. 347–365, worst relabel) at the chosen micro; eval smoke on Dense at `eval_bs 16` (the longest items, ~560 tokens × K' 24).
6. **SMOKE at full data scale:** `--train_n 64 --test_n 16 --steps 20 --sched_scale` on the 4060 through every eval block, including the OOD block on all five splits and the scorer on its output.
7. **Coconut path equivalence:** `eval_p8.py --mode nocot` and `rollout(K=6)` on `prosqa_test.json` with `checkpoint_50` reproduce the stored PROTOCOL6 numbers (`prosqa/res/protocol6_prosqa.json`, `kprime_test_e50.json`) to 1e-4 before any OOD JSON is scored.
8. **Host:** data md5 asserted; `det` wrap not needed (no bitwise claim on GPU); eval.json partial-write (l. 528–531) kept so a late crash loses no block.

## 5. The frontier-state question on DAG / Dense

The memo's risk is that a one-node state cannot hold a search frontier. Two things test it instead of assuming it:

- **FRONTIER probe** (descriptive, probe labels from the simulator, which §4 of EXTERNAL_SOTA allows for reads, not targets): from the flattened delta-head state after pass t (as `step_recover`, l. 620–630, `nested170.logreg_acc`), fit (a) the gold node at hop t (single label) and (b) the BFS reachable set at distance ≤ t (one logistic head per concept slot, `reach_targets`-style, `prosqa_sym.py` l. 356). Fit on 4,000 train items, test on 500 ID and 500 DAG items. Readings: SET-STATE if (b) macro-F1 ≥ 0.80 on DAG; NODE-STATE if (a) ≥ 0.80 and (b) < 0.60; NEITHER otherwise.
- **AUX-PATH on DAG:** the share of passes whose greedy aux object lies on *any* root → target path vs the gold path. If accuracy on DAG is high while gold-path share is low, the loop found another path (set-like behaviour); if both are low with ID high, the aux's single-path supervision is the limit.

Both are read beside the DAG binding cell; neither changes a label.

## 6. Staged plan, cost (RTX 3090, $0.21/h)

**Stage 0 ($0.3, eval-only, pre-stated → Audit ack → compute; cloud card, sha-verified model.pt, per the local-RAM rule).** Generator + five OOD splits + stats. Evaluate `s191_stepaux_relabel_s1` (A) at K' 0 / 6 / 8 / 9 / 10 / 12 / 24 and `s191_textcot_relabel_s1` (B), `checkpoint_50` (official Coconut, K 6) and g_s0–2 (answer-masked, descriptive) on ID + 5 splits. A / B were trained on p8_na hung chains, so this is a mechanism read, not a protocol number. Pre-stated gate: **GO** if A ≥ 0.60 on both 7-hop and 8-hop at K' = hops + 2 (not the best K': that would be post hoc) and A ≥ checkpoint_50 − 0.10 on DAG or Dense; **REDESIGN** (set-valued write, MC2 / MC3 first) if A trails checkpoint_50 by > 0.10 on both DAG and Dense; else **NOTE** and proceed to Stage 1.

**Stage 1 pilot (~$1.5).** `pq_pilot_s1`: the §3 recipe, `--stop_at 3000` of the 12k schedule (a new flag ported from `inject_aw.py` l. 225 / 543 / 655: ckpt at the stop step, then the eval runs from it; absent = bitwise), one 3090 (~5 s/step est. → ~4.2 h ≈ $0.9) + evals. GO if test500 ≥ 0.70, K' 0 on ID ≤ 0.60, 7-hop at K' 9 ≥ 0.50, hold comp / hold ratio ≥ 3 (the penalty bites); PARK if test500 < 0.55 or K' 0 ≥ test500 − 0.05; else NOTE (budget / LR re-warm before redesign, process rule 3).

**Stage 2 (~$24; guard $30).** Ours 3 × 12k (~17 h each ≈ $3.6 → $10.7); CoT-SFT 3 × ~4.5 h ($2.8); No-CoT 3 × ~2 h ($1.3); Coconut 2 new seeds (~16 h each, $6.7); evals on all arms, FLOPs Meter + bs-1 s/item on one card ($1.5). ProntoQA sister (BM1, separate registration): 3 + 3 seeds at ~1.5 s/step ≈ $4.

## 7. Binding and descriptive criteria (per seed; Wilson 95 % CI; NEAR = the point estimate within 0.03 of a bar; k / 3 seeds reported, recipes compared on per-seed margins, Note 269-1)

| id | cell | bar | reads |
|---|---|---|---|
| ID-FID | ID test, K' 10 | ≥ 0.85 (fidelity only, never evidence: §57 shortcut) | binding |
| REMOVE | ID and each OOD split with acc ≥ 0.70 | K' 0 ≤ 0.60 AND donor ≤ 0.60 AND noise ≤ 0.60 | binding (C1, shortcut-exclusion arm) |
| DEPTH-7/8 | 7-hop, 8-hop at K' = hops + 2 | ≥ own CoT-SFT (same seed) + 0.10 AND ≥ own Coconut − 0.05 | binding |
| C2-CURVE | 8-hop | acc(K' 10) ≥ acc(K' 6) + 0.05 (rising past Coconut's T 6 and within our trained K) AND acc(K' 10) ≥ acc(K' 8) | binding (C2, needs 3 / 3) |
| DENSE | Dense at K' 10 | ≥ own Coconut − 0.05 | binding |
| LEN-CTL | len-ctl at K' 10 | ≥ ID − 0.05 (length is not a proxy; the §191 p8_ctl gap was 0.18) | binding |
| DAG | DAG at K' 10 | ≥ own Coconut − 0.10 (stretch: ≥ Coconut) | binding, stretch |
| C6-HOLD | 8-hop | acc(K' 24) ≥ acc(K' 10) − 0.03 | binding |
| C5-STAB | all splits | halt_stab acc ≥ fixed-K'(hops + 2) acc − 0.03; mean halt within hops + 1 ± 1.5 | descriptive (§201 T2 pattern) |
| BASELINE-FID | own CoT-SFT ID ≥ 0.70 (§191 B read 0.938 relabelled); Coconut ID ≥ 0.90 | binding on the baselines; a failing baseline flags the comparison |
| FRONTIER / AUX-PATH | §5 | — | descriptive |

**Labels** (theory/score_pq.py): per seed **SOTA-OOD** = ID-FID ∧ REMOVE ∧ DEPTH-7/8 ∧ DENSE ∧ LEN-CTL; **DEPTH-ONLY** = DEPTH-7/8 ∧ REMOVE ∧ ¬DENSE; **STRUCT-FAIL** = REMOVE ∧ ¬DEPTH-7/8 ∧ ¬DENSE; **SHORTCUT** = ¬REMOVE on any split with acc ≥ 0.70 (overrides everything); **UNDERFIT** = ¬ID-FID. Arm label = k / 3; the H2 claim needs SOTA-OOD 3 / 3 with the 7-hop and 8-hop per-seed margins over both baselines > 2 seed-sd (SubLAT's own convention) and C2-CURVE 3 / 3; C6 adds C6-HOLD 3 / 3. **OC block:** P(SOTA-OOD 3/3) ~0.20; P(DEPTH-ONLY) ~0.35 (the all-token write carries chains, Dense branching breaks selection, as Note 260-5's reachability-following predicts); P(STRUCT-FAIL) ~0.15; P(SHORTCUT) ~0.10 (binary task, 2602.22441); P(UNDERFIT) ~0.10; mixed ~0.10. If DEPTH-ONLY: Dense / DAG go to the memory lane (MC2 KEEP-k, MC3 entity-keyed write), not to a seed rescue. **Censoring:** VOID-RERUN once per cell on host failure or n_nonfinite > 0 (same seed, same args, logs kept under `.void`); a late eval crash → `--eval_only` from the kept model.pt, never a retrain; splits are frozen by md5 at registration and never re-drawn; OOD splits scored once, after all arms land, by the committed scorer; no checkpoint selection (final step; best-val descriptive); no K' chosen after looking (hops + 2 and 10 are fixed here).

## 8. Scorer sketch (`theory/score_pq.py`, ~150 l, fixtures before any output)

Reads per run: `eval.json` → `acc`, `train_fit_relabel.acc_by_kprime`, `s168s2.acc_by_depth_kprime`, `ood[split].{acc_by_kprime, n, removal.{k0, donor, noise}, halt_stab, frontier_probe, aux_path}`, `n_nonfinite`; `args.json` → md5s, seed, hold_inc, data; `log.jsonl` → test500 trajectory, hold ratios. Baselines: textcot / direct `eval.json` (`acc`, `ood`), Coconut `prosqa/res_ood/<ckpt>_<split>.json` (`acc`, `by_ext`). Labels as §7 with `wilson(k, n)` and NEAR per cell; output `theory/results/score_pq.json` + one reading line. Fixtures: 8 synthetic run sets (one per label, plus NEAR on DEPTH-7, a missing OOD block → "incomplete", a baseline fidelity fail → "flagged"); the field-path check (grep each field in `prosqa_aw.py`) is on the G-D checklist.

## 9. Verified arXiv ids (abs pages fetched 2026-11-03, titles matched)

2610.40283 SubLAT (Zeng & Zhang); 2412.06769 Coconut (Hao et al.); 2609.35643 ProsQA-Ext ("Not All Thinking is Created Equal", Cheng & Zhang); 2210.01240 ProntoQA (Saparov & He); 2305.15269 ProntoQA-OOD (Saparov et al.); 2602.22441 Cui et al. (weak vs strong supervision, shortcut caveat); 2502.21074 CODI (Shen et al.); 2603.21676 Chen (depth-recurrent; **not** a path-supervision claim, see §1). Cited via existing notes only: 2604.04902, 2611.02333.

## 10. RED-TEAM (separate sub-agent; see the appendix added after its return)

<!-- red-team section is appended below once the sub-agent returns -->
