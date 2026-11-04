**Note 263-4 (2026-11-04 02:23 EST; QB0 PRE-STATEMENT: question-blind sufficiency of the training law. 07_REDTEAM HIGH-1. Written before any full output.)**
- **Run status.** Only <= 20-item smokes have run. They printed only 'smoke ok' and parse agreement 20/20.
- **Label.** Reads generator output only: $0 CPU, one process, <= 3 cores (OMP / MKL / OPENBLAS / torch threads 3). No model is loaded; torch is imported only because inject_ow / inject_aw / of_parts import it.
- **Script.** theory/read_qb0.py -> theory/results/read_qb0.json.
  - Binding keys: 'binding' (labels, t_bun_d2_4_sufficient, QB0_headline).
  - Descriptive keys: 'descriptive_gb_agreement', 'E', 'T-of-ask'.
- **Scope.** QB0 measures a property of the DATA GENERATOR, not of any model. Nothing here is evidence about what gb, bun or of compute.

**0. The binding question**
- **(A) BINDING.** Is any of the three question-blind policies (LP, RT, RT-S; §2) QB-SUFFICIENT at each of d 2, 3, 4 on bun's training law (T-bun chain items)?
  - Labels per policy × population × d cell: QB-SUFFICIENT >= 0.99, QB-PARTIAL 0.90-0.99, QB-INSUFFICIENT < 0.90.
  - Value level: P(the policy's stop value == the answer). RT / RT-S use the analytic expectation over their tie draws; the realised seeded value is reported beside.
  - Cells: d 1, 2, 3, 4, the d 2-4 pool, and the pool over all d. Populations: T-bun chain items (headline) and T-of chain items.
  - QB0_headline = YES iff some policy is QB-SUFFICIENT at d 2, d 3 and d 4 separately on T-bun.
- **Prediction: YES, through LP, at exactly 1.000 at d 2, 3 and 4** (§4).
- **Meaning if YES (a design guard, not a claim about any model).**
  - On bun's law a policy that never reads the question is right on every chain item at d >= 2.
  - So plan-free arms trained on this law receive no gradient pressure toward question use on chain items at d >= 2.
  - A retargeting lever (--retarget_p: chain items whose question asks a non-S_d entity) is therefore a required design guard for any plan-free arm trained on bun's law.
  - This holds independently of the §260 mechanism.
- **Meaning if NO.** The training law already penalises question-blind selection at d >= 2. --retarget_p is then optional for bun-based arms.
- **Ask items (of's law) are reported separately.** They quantify the one-hop question pressure that of's law adds (§5).
- **(B) DESCRIPTIVE ONLY: agreement with gb** (§6). gb's numbers are already published (Note 260-5), so this is not a blind test.

**1. Populations: exact generator calls, seeds, d range and counts**
- **No new seeds for the populations.** Every training draw replays the trainer's own seed families from step 1, as in boxes/inject_aw.py main() draw_items / ask_items with the run's args.json.
- **Exactness gate (binding before any number is read).**
  - EXCL = build_sets(mul_s_p 0)['excl'] | inject_aw.reg_excl(args); len asserted == args excl_n (206599).
  - REG TEST2 sha asserted == args test2_sha.
  - Steps 1..1000 are replayed per run. Their distract_n and distract_types counts (+ ask_n for of) must equal the run's first log.jsonl bin (step 1000):
    - bun_s1: n {0: 21383, 1: 21423, 2: 21194}; types {chain 16229, givop 23813, s256 23769}.
    - of_s1: n {0: 16008, 1: 16132, 2: 15860}; types {chain 12214, givop 17804, s256 17834}; ask_n 16000.
  - If the gate fails, the script aborts and nothing is scored.
- **Sample sizes.**
  - Each population is the shortest (step, k)-ordered prefix of those replayed steps with >= its count AND >= 1,000 items at each d 1-4. This is asserted reachable within steps 1..1000; otherwise the run aborts.
  - The d range is 1-4 (d_max 4, d = default_rng(seed + 22331).integers(1, 5) per step).
  - Expected counts: chain items ~5,000 per d; ask items ~2,000 per d.
- **T-bun chain items** (boxes/runs_fsq/bun_s1/args.json: seed 1, bs 64, mul_s_p 0, mul_s_p3 1.0, distract_mix s256,chain,givop, distract_p 1.0, nmax 2, perm_p 0).
  - Calls:
    - stream = OW.Stream(22930, EXCL, FRAME_TRAIN, order_mode topo, dag, mul_s_p3 1.0, mul_s_p 0), 64 draws per step;
    - per item k of step s: coin default_rng((1, 25196, s, k)) < 1.0, then IR.distract_mix(r, 1, s, k, (s256, chain, givop), nmax 2), which draws from default_rng((1, 25205, s, k)).
  - Items: >= 20,000 (~steps 1..313).
  - gb_s1 (seed 1, --cue generic; the cue changes no draw) trained on exactly these items.
- **T-of chain items** (of_s1/args.json: + perm_p 0.5, ask_p 0.25, so 48 chain + 16 ask items per step).
  - Perm-then-insert: if d >= 2 and default_rng((1, 25161, s, k)) < 0.5, apply PR.train_perm(r, 1, s, k) (permutation default_rng((1, 25160, s, k))); then the distract_mix as above.
  - Items: >= 20,000 (~steps 1..417).
- **T-of ask items.**
  - Calls: OW.Stream(26210, EXCL, ...), 16 per step at the step's d; perm coin default_rng((1, 26211, s, k)) < 0.5 (d >= 2), then PR.permuted with default_rng((1, 26212, s, k)); class, X, v_X and text from ask_render.ask_of.
  - Items: >= 8,000 (~steps 1..500).
- **E: descriptive eval families (REG TEST2 = build_sets(0)['test2'], 1,000 per cell, the eval parts' own builders and seeds).**
  - TEST2 d 1-8.
  - TEST-PERM d 2-4: default_rng(25150 + d).
  - LATE-PREMISE d 2-4: of_parts.late_order.
  - INSERT-INDIST: insert_eval_sets on d 4, 7 conditions.
  - IR.held_out_sets on d 4, shas asserted == scratch/s256_ho_shas.json:
    - H-OP sub_c / mul_c / add_v;
    - H-CHAINSRC;
    - H-GIVPOS;
    - H-EXTEND;
    - H-LIVE;
    - K3.
  - IR.held_out2_sets on d 4, shas asserted == scratch/s257_ho_shas.json:
    - H-CHAINSRC-OP sub_c / mul_c;
    - H-CHAINSRC-LIVE.
  - Every named family was found.
- **Where policies or constructions are undefined.** Every policy is defined on every item, since each graph has >= 1 given. The constructions are absent at some d:
  - chain inserts and perm do not occur at d 1;
  - ask class (iv) exists only at d 2;
  - ask classes are 3 at d 1, 3 and 4, and 4 at d 2.

**2. Policies (none reads the question)**
- **The graph (shared by all three policies).**
  - Every sentence except the final question is parsed.
  - A given "Y has v u." is a root.
  - An op sentence defines B from A (+ C), with edges A -> B and C -> B.
  - Mirrored frames are resolved as OW.solve resolves them.
  - Parse check: the parser's value of the asked entity == it['answer'] == OW.solve(text), and every spine value == it['values']. Ask items: val(X) == v_X and val(S_d) == answer. A mismatch is excluded and counted (expected 0).
- **LP.**
  - L(root) = 0; L(n) = 1 + max L(inputs); L* = max L.
  - End = the node with L = L* whose defining sentence is latest in the text (tie a).
  - Start = the earliest root on a longest path to End (tie b).
  - Reported: LP, LP-ENT, LP-STOP = L* - d, LP-START, tie share.
- **RT (07_REDTEAM's walk).**
  - Start = the first given sentence in text order.
  - At a node, candidates = the entities whose op sentence reads it.
  - No candidates: stop and predict the node.
  - One candidate: move to it.
  - Several candidates:
    - exactly one non-sink (a candidate with a reader): move to it;
    - several non-sinks: uniform among them;
    - all sinks: uniform among all candidates.
  - The draw is default_rng((27150, crc32(cell), item, 0)). Seed 27150 is unused in boxes/, theory/ and scratch/*.py (grep 2026-11-04).
  - Reported: realised RT / RT-ENT with Wilson CIs, the analytic expectation (exact 1/k branch probabilities), RT-START (start == S_0) and RT-STOP (stop == S_d).
- **RT-S.** RT with start = the first given that has a reader. Draws (..., 1).
- **Q1 (ask items only).** X's own defining sentence applied to its inputs' values. Q1-ONEHOP = every input is a root.

**3. Derivation facts from the builders**
- **(F1) The first sentence is uniform over the roots.**
  - topo order (NW.topo_shuffle) picks the first sentence uniformly from {S_0, G_t, D_k}.
  - P(it is a distractor) = E[n_dis / (1 + n_v + n_dis)] ~ 0.278, 0.296, 0.266 and 0.241 at d = 1, 2, 3, 4.
  - So c(d) = P(RT does not start on a distractor) ~ 0.72, 0.70, 0.73, 0.76.
- **(F2) Insert givens are never first, outside one cell.** Every insert given goes immediately before S1's sentence or later. S_0 precedes S1, and perm keeps the given slots. The only exception is H-GIVPOS 'start'.
- **(F3) From a spine node, RT and RT-S reach S_{d-1}.** There they pick uniformly among S_d and the chain inserts at j = d-1 (all sinks), so they are right with probability 1/(1 + m).
  - Per insert, P(chain at j = d-1) = r = 1/(3(d-1)).
  - E[1/(1+m)] = 1 - r/2 + r²/9 = 0.846, 0.920 and 0.946 at d = 2, 3, 4.
- **(F4) LP ties.** A chain insert at j = d-1 ties S_d at L = d. In unpermuted text S_d's sentence is later, so tie (a) picks S_d.
  - At d 1, a length-1 insert ties S_1, and LP fails when an insert op lies after S1: P = 5/12.
  - In T-of, a permuted item with S_d's sentence before S_{d-1}'s loses the tie: P = 1, 3/5 and 12/23 at d = 2, 3, 4.
- **(F5) Text order matters to RT / RT-S only through the start.** So perm does not change them, and T-of equals T-bun for RT and RT-S.
- Value coincidences add ~0.09 × (1 - entity-level P).

**4. BINDING predictions (value level; entity level in brackets where it differs)**

  **T-bun chain items**

  | Policy | d 1 | d 2 | d 3 | d 4 | d 2-4 | all d |
  |---|---|---|---|---|---|---|
  | LP | 0.625 (0.583) INSUFF | **1.000 SUFF** | **1.000 SUFF** | **1.000 SUFF** | 1.000 SUFF | ~0.906, PARTIAL / INSUFF boundary |
  | RT | ~0.75 INSUFF | ~0.63 INSUFF | ~0.70 INSUFF | ~0.74 INSUFF | ~0.69 INSUFF | ~0.71 INSUFF |
  | RT-S | 1.000 SUFF | ~0.86 INSUFF | ~0.93 PARTIAL | ~0.95 PARTIAL | ~0.91 PARTIAL | ~0.935 PARTIAL |

  **T-of chain items**

  | Policy | d 1 | d 2 | d 3 | d 4 | d 2-4 | all d |
  |---|---|---|---|---|---|---|
  | LP | 0.625 INSUFF | ~0.867 (0.852) INSUFF | ~0.958 PARTIAL | ~0.975 PARTIAL | ~0.93 PARTIAL | ~0.856 INSUFF |
  | RT | as T-bun | as T-bun | as T-bun | as T-bun | as T-bun | as T-bun |
  | RT-S | as T-bun | as T-bun | as T-bun | as T-bun | as T-bun | as T-bun |

- **QB0_headline predicted YES**, through LP on T-bun at d 2, 3 and 4.
- **On of's law no policy is predicted QB-SUFFICIENT at d 2-4.** Perm-then-insert already removes question-blind sufficiency on some chain items, about 15% at d 2 by LP-ENT.

**5. Ask items (of's law; reported separately, no QB label)**
- **Use, stated in advance: how much one-hop question pressure of's law adds.**
  - Pressure(d) = 1 - max over {LP, RT, RT-S} of P(policy == v_X), against Q1.
  - If Pressure >= 0.5 at each of d 2-4, of's law supplies one-hop question pressure on its ask items. --retarget_p is then required for bun-based plan-free arms and recommended, not required, for of-based ones.
  - Otherwise --retarget_p is required for both.
  - Ask items are one-hop and target S_0 / S_1 / a distractor / S_2 (d 2). They give no pressure on multi-hop chain items, which §4 covers.
- **Predictions (entity level; value adds ~0.09 × the rest).**
  - LP: 0.333 / 0.25 / 0 / 0 at d = 1, 2, 3, 4. LP is right on class (i) at d 1 (S_1 = S_d) and class (iv) at d 2.
  - RT: ~0.31 / 0.10 / 0.06 / 0.06. RT is right on class (iii) when the first distractor opens the text, and on (i) at d 1.
  - RT-S: 0.333 / 0.25 / 0 / 0.
  - Q1 = 1.000; Q1-ONEHOP = 1.000.
  - **Predicted Pressure ~0.68 / 0.85 / 0.85 at d = 2, 3, 4 (value level),** so >= 0.5 at each of d 2-4.

**6. DESCRIPTIVE ONLY: agreement with gb (Note 260-5)**
- **Not a blind test.** gb's numbers were published before this note. By construction, every policy is already predicted to disagree with gb:
  - on H-LIVE: LP 1.000, RT ~0.76 and RT-S 1.000 vs gb 0.396 / 0.408;
  - on TEST2 d 1-4: RT ~0.73 vs gb 1.000.
  - So QB0 as THE mechanism of §260's plan-free collapse is predicted refuted for all three policies.
- **Families: 10 (gb_s1 / gb_s2), from boxes/runs_fsq/gb_s{1,2}/eval_acc.json and eval_insert_ho.json.**

  | Family | gb_s1 / gb_s2 |
  |---|---|
  | TEST2 d 1-4 | 1.000 / 1.000 |
  | INSERT-INDIST | 0.999 / 0.999 |
  | H-OP | 0.952 / 0.955 |
  | H-CHAINSRC | 1.000 / 1.000 |
  | H-GIVPOS | 0.993 / 0.990 |
  | H-EXTEND | 0.081 / 0.019 |
  | H-LIVE | 0.396 / 0.408 |
  | K3 | 0.998 / 0.994 |
  | H-CHAINSRC-OP | 0.882 / 0.858 |
  | H-CHAINSRC-LIVE | 0.058 / 0.414 |

- **Excluded.**
  - TEST2 d 5-8 (gb 0.994 / 0.915, 0.851 / 0.441, 0.246 / 0.066, 0.054 / 0.081):
    - every policy is depth-unbounded;
    - gb's d >= 5 accuracy measures depth extrapolation (capacity), on which a selection policy makes no prediction;
    - including them would fail clause (ii) in both seeds whatever the mechanism.
  - TEST-PERM and LATE-PREMISE: gb has no numbers for these.
- **Policy values per family (entity level; c4 = RT's per-item no-distractor-start share on TEST2 d 4, ~0.76).**

  | Family | LP | RT | RT-S |
  |---|---|---|---|
  | TEST2 d 1-4 | 1 | ~0.73 | 1 |
  | INSERT-INDIST | 1 | c4 | 1 |
  | H-OP | 1 | c4 | 1 |
  | K3 | 1 | c4 | 1 |
  | H-LIVE | 1 | c4 | 1 |
  | H-CHAINSRC | 1 | 0.833 c4 | 0.833 |
  | H-CHAINSRC-OP | 1 | 0.833 c4 | 0.833 |
  | H-GIVPOS | 1 | 2/3 c4 | 2/3 |
  | H-EXTEND | 0 | 0 | 0 |
  | H-CHAINSRC-LIVE | 2/3 | c4/3 | 1/3 |

  - TEST-PERM and LATE-PREMISE equal TEST2 per d for all three policies.
- **Readout (descriptive), per seed.**
  - (i) Families with policy value <= 0.10 have gb <= 0.45.
  - (ii) Families with policy value >= 0.95 have gb >= 0.85.
  - Readout labels:
    - QB0-SUPPORTED: no clause-seed failure.
    - QB0-PARTIAL: exactly one clause fails in exactly one seed.
    - QB0-REFUTED: 2-4 failures.
    - QB0-UNINFORMATIVE: a clause has no families.
  - Also reported: Spearman, and the mid-band families where |gb - policy| <= 0.10.
- **Predicted readout.**
  - LP: REFUTED (H-LIVE fails clause (ii) in both seeds).
  - RT-S: REFUTED (H-LIVE).
  - RT: UNINFORMATIVE. Clause (ii) has no families, and the within-0.10 count is 0 of 9 in both seeds.
  - LP's mid-band reading for H-CHAINSRC-LIVE: consistent iff gb by_pos j = 3 <= 0.45 and j = 1, 2 >= 0.85. gb's by_pos have not been read.
- **CONJECTURE (post-hoc, not tested here; no binding use until it is pre-stated on a fresh test).**
  - A recency selection, 'follow the newest live read-chain', may fit gb's profile.
  - H-LIVE collapses although every spine-following question-blind policy is right there.

**7. Summary of uses**
- **(A)** decides whether --retarget_p is a required design guard for plan-free arms on bun's law (predicted: required).
- **§5** decides whether it is also required for of-based arms (predicted: recommended, not required).
- **(B)** is description only. It bears on the §260 mechanism, which stays open, with the post-hoc conjecture above.
- **Nothing here is a claim about any model's internal use of the question.**
