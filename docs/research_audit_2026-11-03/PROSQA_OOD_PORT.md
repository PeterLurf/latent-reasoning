# ProsQA-OOD port: feasibility and design memo

Audit research subagent, 2026-11-03. Follows `EXTERNAL_SOTA.md` §3 (target: the ProsQA structural-OOD splits of SubLAT, GPT-2 124M). Read-only; nothing here is registered.

## 1. What SubLAT (2610.40283) actually defines, and what is released

Fetched abs + HTML on 2026-11-03: "Latent Path Exploration via Potential-Guided Subtrajectory Balance", Zeng & Zhang, v1 29 Oct 2026.

- **Splits.** Appendix A.1 names them only: "7-hop, 8-hop, DAG, and Dense ... The 7-hop and 8-hop splits extend reasoning depth, while DAG and Dense introduce changes in graph structure." No item counts, graph sizes, edge densities, generator, or seeds. Table 4 gives in-distribution sizes only (17,886 / 300 / 500 = Coconut's split).
- **Release.** No GitHub / HF / project URL on the abs or HTML page. GPT-2 numbers (Table 2, mean ± sd over seeds 11/12/13, 5 inference runs each) are in `EXTERNAL_SOTA.md` §1; CODI's row is a public checkpoint; No-CoT is absent. Setup: T = 6 latents, 50 epochs, bs 128, LoRA lr 1e-4, bf16, init from a CoT-SFT checkpoint.
- **Upstream generators.** Coconut (2412.06769, Appendix A.2, Algorithm 1) released the data (`github.com/facebookresearch/coconut`, `data/prosqa_*.json`; our copy `D:\Latent\prosqa\data\`, keys `question / answer / steps / idx_to_symbol / edges / root / target / neg_target`; train depth histogram 3: 7,140, 4: 7,645, 5: 2,541, 6: 560) but **no generator script**. ProsQA-Ext (2609.35643, Cheng & Zhang, 28 Sep 2026) lists no code release either.

**Consequence.** The OOD splits must be regenerated and the definitions are ours, so SubLAT's Table 2 is not directly comparable; only same-split own-run baselines are (the `EXTERNAL_SOTA.md` §4 protocol already requires them). Regeneration spec, from Coconut Alg. 1 (nodes 0, 1 first; each new node gets Poisson(1.5) in-edges from existing nodes; labels track descent from 0 / 1 / both / neither; weights favour deeper nodes; entity = node 0; a leaf descended from 0 only = correct concept, from 1 only = foil; options swapped; ~23 nodes / 36 edges, mean shortest path 3.8, ~1.6 shortest paths, max 6):

| split | our definition (to be fixed in the registration) |
|---|---|
| 7-hop / 8-hop | Alg. 1 with the depth weighting raised and rejection on shortest-path length = 7 / 8; node count free (~25–35) |
| DAG | Alg. 1 with the correct concept forced to have ≥ 2 vertex-disjoint root paths and the foil reachable from an off-path branch (ProsQA-Ext's two-path form) |
| Dense | Alg. 1 with Poisson mean 1.5 → 3.0 (≈ 70 edges), depth distribution as train |

Every split: 500 items, same 17-entity / 38-concept vocabulary (`boxes/prosqa168s2.py` ENT17 / CON38), structural dedupe against train by (edges, root, target, neg_target) as in §191's gate, and the §59 proximity-cue check (the ID set has a 0.91 adjacency heuristic, §57). `prosqa/make_long.py extend()` (hung chains, the p8 recipe) is **not** SubLAT's 7-hop and is kept only as a descriptive bridge to §191.

## 2. Mapping ProsQA onto the current loop

Current best (`docs/sota/sota_arch.json`, bun §259): K = 8 passes over text + a cue naming S_t + ' The answer is:'; the ':' state writes DeltaMem (`boxes/deltamem.py`); a final direct pass reads it; aux = per-pass value (mod 11).

| element | arithmetic loop | ProsQA port |
|---|---|---|
| per-pass state | one running value in Z_11 | **one node on the proof path** (steps[t-1] of the gold CoT: "Every X is a Y") — the form §168 s2 / §191 already used and that reached 0.950 with the think passes carrying the computation (K′ 0 / 1 at chance; THEORY.md:14565–14583). A set-valued BFS frontier (`boxes/prosqa_sym.py reach_targets`) is the "right" state but is simulator-built and disallowed by `EXTERNAL_SOTA.md` §4. |
| cue | names S_t (plan given) | **the question itself** ("Is R a A or B?") repeated each pass = §261's TARGET-CUE form, plan-free and legitimate (Coconut sees the same). A per-step name cue would hand over the path and is excluded. Byte-identical pass inputs also make `--cache_lower` fully effective. |
| aux target | 11-way CE at ':' | teacher-forced text step via the read-only aux decoder of `prosqa168s2.py` (concept names are multi-BPE), §184 reweighting, allowed under revised C4 (text-CoT supervision). |
| answer | number token | candidate-margin scoring over the two "### <concept>." strings (`boxes/prosqa168.py candidate_margins`), chance 0.5. |
| halting | answer-stability rule | the same rule; §201 T2 already showed the learned halt on §191 A tracks depth + 1 up to 14 (THEORY.md:15267). |

**What breaks.** (a) *Branching*: a single-node state cannot backtrack; on DAG / Dense the loop must pick the right child greedily or hold a superposition in the additive heads. §260 found the plan-free loop "follows any read-chain" (Note 260-5), the same failure mode. (b) *Binary answer*: the ID shortcut (§57) makes ID accuracy uninformative; only OOD and the K′ 0 removal read count. (c) *No mod 11*: the number-token parts of `boxes/inject_aw_eval.py` (`ws`, `probe`, `lp5`, `halthead`) need a concept-token readout; `swap_k`, NOT-DECODER, `halt`, `stab` port as they are. (d) *Memorisation*: the fixed 17.9k split memorised without per-draw relabelling (168-16 / 168-17); SubLAT and Coconut train 50 epochs on the same fixed set, so relabelling (names, premise order, option order; no new graphs) must be declared as augmentation under "same data". (e) *Hold penalty / C6*: the hold-inc term reads ':' logits over number tokens; untested on text-valued states.

## 3. Reuse inventory

| reuse as is | path | role |
|---|---|---|
| STEP-AUX trainer + eval, relabel, K′ sweep, candidate scoring, DeltaMem-R | `boxes/prosqa168s2.py` (660 l), `boxes/prosqa168.py` (186 l) | base harness (0.950 on p8_na) |
| memory | `boxes/deltamem.py` (311 l) | unchanged |
| BFS / signature helpers | `boxes/prosqa_sym.py` (558 l) | descriptive frontier probes, `sig` dedupe |
| Coconut / CoT / No-CoT own-run harness, 3 trained Coconut seeds | `prosqa/pod_prosqa*.sh`, `prosqa_1gpu.yaml`, `p8_coconut.yaml`, `p8_nocot.yaml`, `p8c_cot_s*.yaml`, `prosqa/ckpt/g_s0–2_checkpoint_50`, `prosqa/eval_p8.py --mode nocot/cot` | baselines; Coconut OOD is eval-only |
| kept models | `s191_stepaux_relabel_s1` (A), `s191_textcot_relabel_s1` (B) (191-4) | Stage 0 reads |
| scorer pattern | `theory/score_s191.py` (91 l) | template |

**New code (estimates).** `prosqa/gen_prosqa.py`, Alg. 1 + three knobs + stats + dedupe: ~200 lines. `boxes/prosqa_aw.py`, a §234-style copy of `prosqa168s2.py` with asserted edits (answer-write ':' on the question + ' The answer is:', `--cue target`, hold-inc on concept logits, `--cache_lower`, answer-stability halting, K′-by-hop eval; flags-off bitwise `prosqa168s2`): ~250 lines. OOD eval parts + FLOPs Meter + s/item: ~150. Scorer: ~120. Coconut-harness eval over the new JSONs: ~40. Total ≈ 750 lines; `inject_aw.py` is untouched (its item path is OW-specific: `tok_ow`, `NUM_IDS`).

## 4. Staged plan

Cost basis: §191 A measured **7.0 s/step** (83,823 s / 12k steps, bs 64, K 10, ~450-token inputs, RTX 3090; THEORY.md:14575). The 1.3–1.7 s/step figure is for ~100-token arithmetic items and does not apply; `--cache_lower` with a constant cue may cut ~30 %. $0.21/h.

**Stage 0 ($0, local 4060 / 2070 evals; Audit ack before compute, per the pre-state rule).**
1. Generator + five splits (ID test + 4 OOD), stats table (nodes, edges, shortest path, n paths, proximity-cue rate), dedupe.
2. Eval s191 A at K′ 6 / 8 / 10 / 12 / hops + 2, s191 B (CoT) and the three existing Coconut seeds on all splits. Caveat up front: A / B were trained on p8_na (hung chains to depth 10), not the standard split, so this is a mechanism read, not a protocol number.
- **GO** if A ≥ 0.60 on 7-hop / 8-hop at its best K′ and A ≥ own-run Coconut − 0.05 on DAG or Dense. **REDESIGN (set-valued state)** if A trails Coconut by > 0.10 on both. Else NOTE.

**Stage 1 (pilot, ~$2).** `prosqa_aw` on the standard train split + relabel, target cue, K 8, hold penalty, 3k steps of a 12k schedule, 1 seed, 3090: ~6 h ≈ $1.3 + evals. GO: test500 ≥ 0.70, K′ 0 ≤ 0.60 (removal), 7-hop ≥ 0.50 at K′ 8; PARK: test500 < 0.55 or K′ 0 ≈ K′ 8.

**Stage 2 (full, ~$25, under the $30 auto-approve line).** Pre-registered with OC block and censoring rule; OOD splits scored once (blinded).
- Ours: 3 seeds × 12k steps ≈ 3 × 23 h ≈ $15.
- Own-run baselines, same split: CoT-SFT 3 × 4.5 h ≈ $3; No-CoT 3 × ~2 h ≈ $1.5; Coconut from the three existing seeds (eval only; a same-code retrain ≈ $4 if wanted); CODI public checkpoint cited.
- Readings: ID + 4 OOD at K′ = 6 (every published arm) and K′ = hops + 2 (the C2 curve: rising past trained K in every seed, or no C2 claim); NOT-DECODER / K′ 0 and step-swap on the OOD splits; FLOPs per item (Meter) and bs-1 s/item on one card. §191 A read 9.84e11 FLOPs, 0.069 s/item vs CoT 1.10e11, 0.70 s/item (THEORY.md:15202), so C3 stays split.
- Target: beat own-run CoT-SFT and Coconut on 7-hop / 8-hop by > 2 sd and not trail on Dense; DAG is the stretch.

## 5. Risks and the biggest unknown

- The splits are ours; comparison is within our table only. Mitigation: publish generator and seeds with the result.
- ID shortcut (§57) and memorisation (168-17): relabel is mandatory and declared; the ID number is reported, never read as evidence.
- The proof path is one of ~1.6 shortest paths; the aux may supervise a path the loop cannot find greedily on DAG items (Chen 2603.21676's prediction, 168-13).
- Length proxy: §191 A read 0.640 vs B 0.816 on the input-length control (191-2); Dense raises length without depth, so a `p8_ctl`-style control is needed.
- Seed basin effects (SEED-1 / basin-entry notes): 3 seeds minimum, entry steps reported.

**Single biggest unknown:** whether a one-token-per-pass DeltaMem write can carry a *frontier* rather than a path node. Every ProsQA success of ours (§191: 0.950, clean removal, depth 14 at K′ 12) was on unbranched chain lookups (168-6's scope note); DAG / Dense are the first branching test, and §260's plan-free loop already showed chain-following without selection. Stage 0's $0 read of s191 A on the regenerated DAG / Dense splits answers this before any spend.

Verified ids (abs pages fetched 2026-11-03): 2610.40283, 2412.06769, 2609.35643. Cited via existing notes only: 2604.04902, 2603.21676.
