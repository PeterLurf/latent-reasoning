# SELF-DEPTH-CHAIN ($0 DRAFT): self-training VERSION-BANK / SLVA to greater latent depth with its own re-rooted single-step chain as the labeller

Started 2026-10-17 01:37 EDT (clock). Opus inventor + derivation subagent for Theory. **Not registered; no training, no GPU, no repo edits.**
Scripts and outputs: `scratchpad/research4/scripts/` (keep_rates.py, item_space.py, support_oc.py, gate_oc2.py, compose.py) and `research4/*.json`.
Tags: [MEASURED] read from a raw file (path given), [PROVED], [PROVED UNDER X], [DERIVED], [COMPUTED] (exact numerics here), [ASSUMED], [CONJECTURE].

---

## 0. One-screen summary

| item | design |
|---|---|
| Question | Does training D3-T's latent loop on **self-labelled deeper items** (n 9–12, labelled by its own re-rooted fresh single step, no gold) give latent depth **inside** that range, and does it **extrapolate past it** (n 13–16, untouched §170 test items)? |
| Labeller | RE-ROOT CHAIN (213-33): 1 pass on x(cur) → argmax R(V_1) → re-render → parse → repeat. **KEEP = parses at every step ∧ exactly n steps** (Theory / Audit; n = the item's operator count, read from the input). Plus label-free **COMPOSE-VALID** (frame, one contiguous span + value, leftmost-ready order, canonical re-render), needed to build slot-aligned R2 targets |
| Targets | R1: the chain's final value. R2: E_self,t = the chain's own step decodes composed onto the original slot frame (HOLD past n). No gold field exists in the training pool |
| Round 1 (Stage 1) | FT from D3-T model.pt, 12k steps, lr 3e-5 cosine, per-step coin p_self 0.5 (self batch from a 480k fresh, test-excluded, base-labelled pool at n 9–12, else D3-T's plain n ≤ 8 batch), K = min(n + U{0..4}, 16), c = 0; + FT-CTRL (p_self 0) |
| Gate per round | GATE-q on 2,000 held-out fresh items per new n vs gold (evaluation only): post-filter value precision **pooled over the window ≥ 0.995 and ≥ 0.99 at every n**, post-filter state precision (E_self = gold E_t) pooled ≥ 0.99, keep ≥ 0.70 at every n. P(pass \| π 0.997) ≈ 0.998, P(pass \| π 0.993) ≈ 0.01 [COMPUTED] |
| How far rounds can go | Labels are not binding to n 20 on D3-T (213-38: horizon 20, post-filter ≥ 0.997, keep 0.78 at n 20). The binding limits are the untouched test region and memory. **Practical R ≤ 2 (B 16, test on the extended §170 set n 17–20).** A third round (B 20) needs test n 21–24, a labeller row past 20 and micro ≥ 4 |
| Primary C2 reading | **DEPTH-BEYOND-SELF**: mean acc_n2 over n B+1..B+4 (round 1: n 13–16) ≥ 0.80. Mechanism: **MARGIN-GROWS** (≥ 0.63) vs **SHIFT** (≈ 0.48 predicted if the stop just moves) |
| Derivation | Prop S0: the support boundary ρ + r ≤ 8 moves to ≤ B; "stop at B" is zero-loss again. Prop S2: post-filter label noise ≤ φ(1 − P)/P with measured φ ≈ 0.005 (upper 0.010); no avalanche unless κφn ≥ 1 (κ ≥ 8 at n 12). Prop S4: SHIFT is the default prediction |
| OC (one seed) | P(SD-SELF) ≈ 0.45; P(MARGIN-GROWS) ≈ 0.25; **P(SD-C2) ≈ 0.075**; P(SD-C2 \| no effect) ≤ 0.01 |
| Cost | FT-SELF $0.52–0.96 (micro 2 projected) or $0.34–0.58 (self K extra 2); FT-CTRL $0.30–0.43; **cap $1.5 for the pair**; labelling and evals local, $0 |
| Relation to D4 | Complementary: D4 decouples pass index / age; SELF-DEPTH moves the progress support, the branch 213-32 leans toward and the one pauses provably cannot reach (D0a). They combine sequentially |

---

## 1. Facts this rests on

| fact | value | tag / source |
|---|---|---|
| D3-T latent depth | acc_n2 n 9–16: 0.923 / 0.603 / 0.266 / 0.127 / 0.141 / 0.127 / 0.135 / 0.115; K′ 8 at n 9–12: 0.329 / 0.133 / 0.127 / 0.118 | MEASURED s213_d3t_s1/eval_vb.json |
| STOPS-AT-10 | ACC-GAIN-AFTER-10 −0.003 (D3-T), −0.001 (D3-F) | MEASURED 213-26 |
| Fresh single step | D3-T m 12–16: 0.987 / 0.991 / 0.980 / 0.973 / 0.970; D3-F 0.991 / 0.984 / 0.965 / 0.923 / 0.845 | MEASURED 213-19 / 213-25 |
| CHAIN-LABEL strict precision n 9–16 | D3-T 1.000 / 0.995 / 0.986 / 0.977 / 0.970 / 0.958 / 0.932 / 0.897; D3-F 1.000 / 0.996 / 0.991 / 0.982 / 0.953 / 0.912 / 0.861 / 0.764 | MEASURED 213-35 / 213-36 |
| **Keep rate under KEEP** (= steps_minus_n "0" share) | D3-T 1.000 / 0.996 / 0.986 / 0.977 / 0.970 / 0.958 / 0.932 / 0.898; D3-F 1.000 / 0.996 / 0.991 / 0.982 / 0.953 / 0.912 / 0.861 / 0.764 | MEASURED, keep_rates.py on both eval_chainlabel.json |
| **Kept-but-wrong** (keep − strict) | D3-T 1 item at n 10, 1 at n 16 (of 1,000 each), else 0; D3-F 0 at every n. **Post-filter precision ≥ 0.999 at every n 9–16 in both cells** | MEASURED (same) |
| **CHAIN-LABEL-FRESH** (213-38; fresh items, seed 21337, §170 + SERIAL exclusion, collide 0.000 at every n) | D3-T keep n 9–20: 1.000 / 0.996 / 0.988 / 0.976 / 0.972 / 0.952 / 0.943 / 0.917 / 0.892 / 0.839 / 0.809 / 0.781; **post-filter precision ≥ 0.997 at every n** (3 kept-wrong at n 15, 1 at n 19); **horizon 20**. D3-F keep 0.999 / 0.994 / 0.993 / 0.968 / 0.957 / 0.926 / 0.854 / 0.753 / 0.627 / 0.444 / 0.383 / 0.282, **horizon 17**. Fresh vs test strict within 0.02 at n 9–16 in both cells | MEASURED s213_d3t_s1 / s213_d3f_s1 eval_chainlabel_fresh.json (3070 / 4060) |
| Labelling speed (fresh, D3-T, 3070, fp32, bs 32) | 0.0059 / 0.0081 / 0.0073 / 0.0085 s per item at n 9–12; 0.011–0.015 at n 13–16; 0.015–0.018 at n 17–20 | MEASURED (same file, sec_per_item) |
| Filter's error capture φ = kept-wrong / all non-strict | D3-T: test n 9–16 2 / 285, fresh n 9–20 4 / 939; **pooled 6 / 1,224 = 0.005 (95 % upper ≈ 0.010)**. D3-F 0 / 528 (test) | DERIVED from the above |
| Chain ≈ independence product | measured − predicted within ±0.005 to n 15 (D3-T) | MEASURED 213-35 |
| Every value-correct chain is chain-exact | chain_exact == precision_strict at every n, both cells | MEASURED keep_rates.json |
| Memory channel to keep | D3-T K′24 stability carried by SLVA own-slot history: STAB_self 0.782 (D3-F 0.556); acc_k24 1.000 at n 1–8 (D3-F 0.835 at n 1) | MEASURED 213-30, eval_vb |
| Not size-keyed | PROBE(12) extra-size slope −0.035 (SE 0.017) | MEASURED RT F7 (D4 v1 §1.1) |
| Training slot count | n ≤ 8 draws: T max 33, p99 29; n 12: T mean 38, p99 43, max 47; n 13–16: T mean 41 / 44 / 47 / 50 | COMPUTED item_space.json (200k / 50k draws) |
| Item space vs exclusion | 0 exclusion hits and 0 within-pool duplicates in 50k fresh draws at every n 4–20 (n 1–3 are tiny strata: 89 % / 2.8 % / 0.02 % hits) | COMPUTED item_space.json |
| §170 test set extendable | test_set(1000, n_max 20) is identical to test_set(1000) on n ≤ 16 (same rng order) | COMPUTED item_space.json |
| D3-T run | 48k on a 3090, 0.407 s/step at n ≤ 8 (log 16k–48k), ≈ $1.12; mem_check 11.28 GiB (bs 64, K 12, bf16); G-TIME 0.462 s | MEASURED log.jsonl, s213_d3t_s1_mem, _gtime |

---

## 2. Mechanism

### 2.1 The labeller and the keep filter

**Chain** (213-33, unchanged): cur ← the item's expression; repeat: x = [EOT] + render(cur) + [" ="]; one pass (rollout K = 1); dec = argmax R(V_1); s = rerender(dec); cur ← parse(s); stop when cur has no operator (label = its value) or at a step cap (n + 4, as 213-37). **The chain always uses the model's own decode; it never calls the simulator.**

**Per-step label-free checks** (compose.py, the reference implementation; none uses a gold value or any arithmetic):

| check | definition | why |
|---|---|---|
| PARSE | s is one valid expression (vbank_diag._parse_tree) | 213-33's chain condition |
| STEPS | the chain ends after **exactly n** steps; n = #operator tokens of the input | Theory / Audit |
| FRAME | dec[0] = EOT, dec[−1] = " =" | slot composition |
| SPAN | dec's VOID slots form one contiguous run (≥ 2 slots), the next slot holds a value class, no other slot changed | the R2 format (value on the span's last slot, as vbank.recode_targets) |
| ORDER | that run + value slot is exactly the leftmost-ready handle of the **parsed input** (its operands, operator, own brackets) | the base's reduction order; computed from the input alone, no value |
| CANON | rerender(dec) == render(parse(rerender(dec))) | the next input is the decode itself |

- **KEEP (primary, Theory / Audit) = PARSE at every step ∧ STEPS.** It needs no answer and no gold. It suffices for the R1 answer label.
- **COMPOSE-VALID = KEEP ∧ FRAME ∧ SPAN ∧ ORDER ∧ CANON at every step.** It is needed to build R2 slot targets. Predicted extra drop ≈ 0, because every value-correct chain is chain-exact in both cells [MEASURED]; S0-1 measures it.
- **Fixtures** (compose.py, CPU, 3,000 items each, half the steps corrupted) [COMPUTED]:
  - value on the span's first slot: KEEP passes all 3,000, COMPOSE-VALID catches all corrupted chains (0 wrong E_self valid). **KEEP alone would admit non-canonical R2 targets, so COMPOSE-VALID is required for R2;**
  - broken frame: same pattern (KEEP blind, COMPOSE-VALID catches);
  - a 0-reduction (copy) step: STEPS catches it;
  - a wrong value: both pass (2,601 / 3,000 kept-wrong). **No label-free check sees value errors**; their measured rate is φ (§3, Prop S2).

**R2 target composition** [PROVED by construction; gate in §7]: E_0 = the classes of the original x. At step k the live slots of E_{k−1} (non-VOID, frame excluded) are, in order, the tokens of render(cur_{k−1}) (by CANON at step k − 1). E_k = E_{k−1} with those slots overwritten by dec_k's interior slots. HOLD after n. On chain-exact chains E_self = vbank.recode_targets(root) at every t: **gate 0 diffs on 2,000 items** (the 100k-item run is in §10).

**Pool record** (no gold field): input class ids + per step (j_lo, j_hi, j_val, value) in the original frame, about 85 B per item. Generator outputs t and root are dropped at draw time; the builder re-parses the input syms.

### 2.2 Rounds (general spec; round 1 is Stage 1)

| element | spec |
|---|---|
| Round q | adds window W_q of new depths; support boundary **B_q = 8 + Σ_{q′ ≤ q} |W_{q′}|**. Round 1: W_1 = {9, 10, 11, 12}, B_1 = 12 |
| Labeller | the round-start model. q = 1: D3-T model.pt, local 3070, fp32, bs 32, $0. q ≥ 2: the FT'd model (fp32, local after the drain), the self-improvement path. **Pre-stated fallback:** if GATE-q fails on the FT'd model but passes on the base D3-T labeller (whose fresh labels reach n 20, 213-38), round q is labelled by the base and reads **BASE-LABELLED** (disclosed). The labeller is never chosen on pool or test items |
| Fresh items | nested170.generate(rng_pool, n), rng_pool seed 21800 + 100q (21900 for q = 1), drawn until **120k kept per n** (480k kept for round 1; draws ≈ 120k / keep, so the discard cost is ≈ 1 − keep: about 1 % in round 1, 5 % in round 2, 17 % at n 17–20). Each kept item is drawn at most once (fresh-once; the trainer logs any reuse). Local labelling time [MEASURED rate, 213-38]: ≈ 1.0 h for round 1, ≈ 1.8 h for round 2 on the 3070 |
| Reach | Not label-limited to n 20 on D3-T. Round q needs **≥ 4 untouched test n past B_q**: the §170 set extends to n 20 with identical n ≤ 16 items (S0-2), so B ≤ 16 (R ≤ 2 rounds of W 4). B 20 needs test_set(n_max 24), a labeller row at n 21–24 (unmeasured; keep extrapolates to ≈ 0.65 at n 24 [ASSUMED]) and memory at K 24, T ≈ 75 (micro ≥ 4 or activation checkpointing) |
| Exclusion (Audit) | exact rendered-syms match against test_set(1000, **n_max 20**) ∪ the 2,200-item SERIAL pool ∪ the gate items; excluded draws are redrawn; counts reported. Expected 0 hits at n ≥ 4 [COMPUTED] |
| Gate items | 2,000 per new n, rng seed 21860 + 100q (21960 for q = 1), same exclusion, **never trained**. Gold (generator t, recode_targets(root)) is read here **only** |
| **GATE-q (the per-round precision gate)** | On the gate items, after KEEP ∧ COMPOSE-VALID, against gold (evaluation only): **π_pool ≥ 0.995** (post-filter value precision pooled over the window) ∧ **π_n ≥ 0.99 at every n** ∧ **π^E_pool ≥ 0.99** (E_self equals gold E_t at every t) ∧ **keep_n ≥ 0.70**. Keep is a cost, not a quality, term. OC [COMPUTED, gate_oc2.json, kept ≈ 1,560–2,000 per n]: P(pass \| π 0.999) = 1.000; **P(pass \| π 0.997) ≈ 0.998**; P(pass \| π 0.995) ≈ 0.5; **P(pass \| π 0.993) ≈ 0.01**; P(pass \| π 0.99) ≈ 0. Expected on D3-T's labels (π ≥ 0.997 measured): pass |
| Mixture | per step, coin p_self 0.5 from rng_s = seed + 21800. The plain batch is drawn every step from D3-T's streams (rng, rng_k), so CTRL sees identical plain batches; on a self step it is discarded and a self batch is drawn (Lee 2502.01612 style within the self half: 50 % the newest window, 50 % uniform over earlier windows; round 1 has one window) |
| Self curriculum (round 1) | CurGate on the self-batch answer_ce: stage A n ∈ {9, 10} → stage B n ∈ {9..12} after 2 bins < 0.3 or a 4k-step cap (the existing class) |
| Budgets | plain: D3-T's K = min(n + U{0..4}, kmax) (unchanged, n + U ≤ 12). Self: the same rule with **kmax = B_q + 4** (16 in round 1). R1 window [n, K], R2 at t = 1..K (vb_loss as is) |
| Optimiser | FT from model.pt (strict load, **no calibrate()**), new AdamW, lr 3e-5 peak, lr_at's 100-step warm-up then cosine to 0 over the round's steps, wd 0.01, bs 64, seed 1, c = 0 (--c_sched 0,1), --cur 8 |
| Steps | **12k per round** (6k self steps ≈ 384k self items). Lee et al. use 1.5k–5k fine-tuning steps per round [checked, 2502.01612 full text]; 12k matches D4's FT and leaves room for a 4-depth window |
| Micro | per batch type: plain micro 1 (D3-T bitwise); self micro from the forced self-shape mem_check (projected 2 at K 16, T 47) |
| Stop rule | Before round q + 1, stop if any of these holds:<br>• GATE-(q+1) fails on both labellers;<br>• round q reads FT-DIVERGED, ¬ABSORB, LABEL-AUDIT < 0.99, or MEM-TRADED;<br>• fewer than 4 untouched test n would remain past B_{q+1};<br>• the cap is reached;<br>• there is no user OK (each round past 1 needs its own).<br>A round that reads SHIFT does not stop the ratchet by itself; it only removes the reason to expect a growing margin (§6) |

**Lee et al. checked** (2502.01612, full text via arXiv HTML): +1 length (or +1 digit) per round; ~50,000 self-labelled examples per round; models **fine-tuned** from the previous round on D_0 ∪ … ∪ D_r with the newest round up-sampled to 50 %; filters: length filtering plus majority vote over 3–5 independently seeded models; 1.5k–5k steps per round; accelerated variants sample a wider range of difficulties per round. They report error avalanche without filtering, and that **model-generated (biased) errors hurt more than random label noise**. 2511.07378 (Huang, Wen, Singh, Chi, Chen) proves for constant-depth transformers on state tracking with **text CoT** that recursive self-training on own traces at length 2^k length-generalises to 2^{k+1}, via attention concentration. Neither has a latent loop. Neither uses a labeller other than the model's own output at the new length. **Our labeller is a different procedure (the re-rooted fresh step) whose precision at the new depth is already ≈ 1.000 post-filter**, so round 1 needs no bootstrapping. That is why a 4-depth window is safe here, where Lee steps +1.

### 2.3 Base choice (the user's)

| | D3-T (recommended) | D3-F |
|---|---|---|
| Keep / post-filter precision n 9–12 | 1.000 / 0.996 / 0.986 / 0.977; ≥ 0.999 | 1.000 / 0.996 / 0.991 / 0.982; 1.000 (marginally better) |
| Keep n 13–16 (round-2 labels), test / fresh | 0.970 / 0.958 / 0.932 / 0.898; fresh 0.972 / 0.952 / 0.943 / 0.917 | 0.953 / 0.912 / 0.861 / 0.764; fresh 0.957 / 0.926 / 0.854 / 0.753 (fresh step 0.923 / 0.845 at m 15–16) |
| Fresh keep n 17–20; labelling horizon (213-38) | 0.892 / 0.839 / 0.809 / 0.781; **horizon 20** | 0.627 / 0.444 / 0.383 / 0.282; horizon 17 |
| K′24-STABLE at baseline | met (1.000 at n 1–8) | **not met** (0.835 at n 1), so a kept reading starts failed |
| KEEPS-SELF-STAB baseline | STAB_self 0.782 | 0.556 (just above 0.5) |
| DEPTH baseline | 0.480 | 0.494 |

- **D3-T is recommended whenever rounds may go past n 12, and on the kept readings.**
- D3-F is defensible only for round 1 alone (n ≤ 12), with K′24-STABLE reported as baseline-failed.

### 2.4 Flags and code (none exists today) [flagged §10]

- **New flags:** `--init_from`, `--self_pool`, `--p_self 0.5`, `--self_cur 10,12 --self_gate 0.3:2:4000`, `--self_kmax 16`, `--self_kextra 4`, `--mem_check_self`.
- **Changed vs D3-T args.json:** `--lr 3e-5 --steps 12000 --cur 8 --c_sched 0,1`. ARGS-DIFF must show only these.
- **Trainer edits:**
  1. init_from: strict load, skip calibrate() / ratio_test (vbank.py 602–615 always calibrates on the non-resume branch [VERIFIED]), copy calibration.json, write args.json.
  2. A self-pool loader that rejects any record carrying root / t / gold keys.
  3. rec_targets uses the record's E_self (HOLD past n) when present.
  4. Per-batch-type micro.
  5. The pool cursor and rng_s go into ckpt_last.pt, so resume is state-complete.
  6. mem_check_self: the longest round items at K = B + 4.
- **No model change:** SLVA's parameters are unchanged, so VE.load's strict load works with the existing factory [VERIFIED vbank_eval.py 38–45]. This is simpler than D4, which needed pause_vec.

---

## 3. Mathematics

Notation (D4's): ρ_{t−1} = reductions before pass t; r_t = n − ρ_{t−1} = operators alive at pass t; B_q = the round-q support boundary; s_q(m) = the round-q model's fresh single-step exactness at m operators; P_q(n) = chain strict precision; φ = the share of non-strict chains that pass KEEP.

**Prop S0 (support identity under self-labelling) [PROVED, extends D0].**
- **Identity.** On every training pass of every item in the round-q mixture (gold n ≤ 8, or self n ∈ 9..B_q with a COMPOSE-VALID chain), the **target** schedule has exactly one reduction per pass t ≤ n and HOLD after, so ρ_{t−1} + r_t = n ≤ B_q.
  - Proof: COMPOSE-VALID ∧ STEPS gives one contiguous handle VOIDed per step for n steps. Wrong values change no count.
  - Like D0, this is about targets. The network's state may leave it.
- **Strong form.** Every state with ρ + r > B_q is outside the round-q support, from pass 1 of every n > B_q item.
- **(a) The old stop now pays loss.** "ρ ≥ 8 ⇒ finished" (or any rule that answers before pass n) mislabels the R2 target at passes 9..n of every self item, and the R1 target if it fires before n. Its loss is positive [PROVED]. That it is unlearned at the optimum is [ASSUMED] (optimisation).
- **(b) The new stop is free.** "ρ ≥ B_q ⇒ finished" agrees with every R1 / R2 target of the round-q mixture [PROVED]. So the D3-T identifiability problem reappears one boundary out: **no loss term separates "stop at ~B_q" from "no stop"** on the support.
- **(c) What C2 means here.** In the self range 9..B_q, depth is **in support**: "beyond gold-supervised depth", not beyond training. The C2 test is n > B_q, where only inductive bias decides.
  - Round 1 at n 15–16 also needs K′ 17–18, past the trained maximum K = 16. That is the North-star C2 form (K′ > K).

**Prop S1 (pass-level support of the round-1 mixture) [COMPUTED exact; support_oc.py].** P(an R pass | t ≥ t*), pass-weighted, and E[# unfinished passes at t ≥ t_d per item] (the cost of an input-keyed deadline at t_d):

| t* / t_d | 9 | 10 | 11 | 12 | 13 | 14–16 |
|---|---|---|---|---|---|---|
| D3-T (plain only) | 0 | 0 | 0 | 0 | — | — |
| **Round 1** (p 0.5, self n 9–12) | 0.500 / 1.25 | 0.400 / 0.75 | 0.283 / 0.375 | 0.145 / 0.125 | **0 / 0** | 0 / 0 |
| Round 2 (self 9–16, kmax 20) | 0.643 / 2.25 | 0.609 / 1.75 | 0.568 / 1.31 | 0.517 / 0.94 | 0.455 / 0.63 | 0.375 / 0.19 / 0.06 at 14 / 15 / 16; **0 at ≥ 17** |

- Round 1 makes any pass- or age-keyed deadline at t_d ≤ 12 cost loss: 1.25 unfinished passes per item at t_d 9, 0.125 at 12. **t_d ≥ 13 stays free.**
- Unlike D4, these are unfinished passes **with ρ + r > 8** (real new work), not paused histories.
- So SELF-DEPTH moves both the progress boundary and any pass / age deadline to ≥ B + 1. **It cannot tell the clocks apart** (§8).

**Prop S2 (label precision and noise propagation).**
- **(a) Post-filter precision [PROVED, definitions].**
  - kept = P + φ(1 − P), so π = P / (P + φ(1 − P)) ≥ 1 − φ(1 − P)/P.
  - D3-T n 12: P 0.976, φ ≤ 0.010 (95 % upper of the pooled 6 / 1,224), so π ≥ 0.9997. At n 16: π ≥ 0.999. At n 20 (P 0.781): π ≥ 0.997, which matches the measured ≥ 0.997 (213-38).
- **(b) Recursion [DERIVED UNDER A1–A3].**
  - A1 (noise sensitivity): fine-tuning on a pool whose kept-wrong share is w raises the per-step error at the trained lengths by at most κw, κ ≥ 0.
  - A2 (clean labels don't hurt): COMPOSE-VALID, value-correct targets do not raise s's error at their own lengths.
  - A3 (independence): P_q(n) = Π_{m ≤ n} s_q(m), measured within ±0.005.
  - Then w_q ≤ φ(1 − P_q(n)) ≤ φ n ē_q (union bound), so ē_{q+1} ≤ ē_q (1 + κφn_q), and after Q rounds ē_Q ≤ ē_0 · exp(κφ Σ_q n_q).
  - **Avalanche (geometric growth per round) needs κφn ≥ 1.** At φ ≤ 0.010 and n 12 that means κ ≥ 8: a wrong label would have to corrupt more than 8× its own mass of other inputs. At n 20, κ ≥ 5.
  - Worst case at κ = 1 over two rounds of W 4 (n 9..16, Σn = 100): ē grows ≤ e^{1.0} ≈ 2.7× at φ's upper bound, and **≤ 1.6× at the measured φ 0.005**. For round 1 alone (base labels), w ≤ φ(1 − P) ≤ 0.0003 at n 12.
  - **Label noise is not the binding constraint for rounds to n 20 on D3-T.** Discards cost ≈ 1 − keep in labelling only (local, $0).
  - If rounds are labelled by the base (the fallback), A1's recursion does not apply: noise enters each round once, at the base's φ.
- **(c) Bias, not rate, is the risk [CONJECTURE, Lee B.4].** A systematic error needs a pattern class c with P(wrong | kept, c) ≥ ½. With global kept-wrong ≤ 1e-3, such a c has mass ≤ 2e-3.
  - D3-T's in-range transitions are exact (1.000 at n ≤ 8, the full 4 × 11 × 11 table seen in gold training), and its long-step errors self-announce (unparseable). So we conjecture no such class.
  - GATE-q and LABEL-AUDIT measure it rather than assume it.

**Prop S3 (does FT on deeper items change the single step?).**
- **(a) [PROVED, first order in logit space].** Pass 1 of a self item at n is exactly a fresh single step at m = n. Its R2 target is the model's own current argmax decode (the labeller was the same weights).
  - The CE gradient with target = argmax raises that logit against all others, so **the pass-1 term cannot flip the argmax at its own input** to first order.
  - Parameter sharing spreads the update to other inputs [ASSUMED small, A2].
- **(b) [DERIVED].** Passes t ≥ 2 pull the in-trajectory decode toward the fresh-step decode of the re-rendered state. This is the intended distillation; the targets equal canonical E_t when COMPOSE-VALID.
- **(c) [ASSUMED].** The plain stream (50 % gold n ≤ 8) protects m ≤ 8 and the in-range trajectory.
- **(d) [CONJECTURE].** s(m) at m > B moves up, because the core sees more long, aged states. Round 2's GATE and LABELLER-KEPT read it.

**Prop S4 (where the stop goes; predictions) [CONJECTURE; the three hypotheses].**

| hypothesis | statement | n 13 / 14 / 15 / 16 (round 1) | mean |
|---|---|---|---|
| **H_shift** | the stop moves with the support (S0b): acc(B + k) ≈ D3-T's acc(8 + k) | 0.92 / 0.60 / 0.27 / 0.13 | **0.48** |
| H_grow | the margin scales with B (e.g. a count code whose precision falls with magnitude, Weber-like): δ(12) ≈ 1.5 δ(8) | 0.95 / 0.82 / 0.60 / 0.38 | 0.69 |
| H_full | no stop ≤ 16 (a progress-free loop) | ≥ 0.9 / ≥ 0.9 / ≥ 0.8 / ≥ 0.8 | ≥ 0.85 |

- Under H_shift, STOP-LOC is about 13–14 and THINK-GAIN-BEYOND is about 0.30, the base's G(8) = 0.303 [COMPUTED]. So THINK-GAIN-BEYOND does not discriminate.
- Why H_shift is the default: D0 / S0 say the loss is silent past B, and D3-T's own margin past its support was about +1 (0.923 at n 9).
- The case for H_grow: varied depths discourage counting (Bansal 2202.05826; Kuo 2606.29983), and the trained slot count grows to T 47, covering most n 13 items (T p99 47) [COMPUTED].

**(v) Cost and memory** [D3-T MEASURED; projections ASSUMED: D4 v1's split, GiB = 2 + 0.58 Kb + 0.0258 Σ(t + 1) at T 33, scaled ×T/33 (linear) and ×(T/33)² (bank); s/step ∝ Kb (launch-bound) up to ×T/33 (compute-bound)].

| shape | peak GiB (micro 1) | micro | s/step |
|---|---|---|---|
| D3-T plain (K 12, T 33): model check | 11.3 (measured 11.28) | 1 | 0.41 |
| Round-1 self batch (K 16, T 47) | 23.2 | 2 | 1.10–1.57 |
| Round-1 self, self_kextra 2 (K 14) | 19.8 | 1 | 0.48–0.69 |
| Round-2 self (K 20, T 61) | 43.7 | 4 | 2.75–5.1 |
| Round-2 self, self_kextra 2 (K 18) | 38.0 | 2 | 1.24–2.29 |

| cell | hours | $ at $0.18–0.26 / h |
|---|---|---|
| FT-SELF round 1, 12k (kextra 4) | 2.9–3.7 | **0.52–0.96** |
| FT-SELF round 1 (self_kextra 2, fallback) | 1.9–2.2 | 0.34–0.58 |
| FT-CTRL 12k | 1.66 | **0.30–0.43** |
| Round 2, 12k (self_kextra 2 / 4) | 3.7–5.5 / 6.3–10.2 | 0.67–1.43 / 1.13–2.64 |

- **Local ($0):**
  - the round-1 pool: 480k kept items at the measured 0.0059–0.0085 s per item, ≈ 1.0 h on the 3070. Round 2 (n 13–16, 0.011–0.015 s per item) ≈ 1.8 h (213-38);
  - GATE-1, about 5 min;
  - evals, about 3.5 h per cell (eval_vb alone took 3,884 s).
- A 2× host spread is possible (memory "host speed variance"), so G-TIME decides go / no-go.

---

## 4. Goodhart routes

| route | how | status | read by |
|---|---|---|---|
| **Moved progress stop** "ρ ≥ B ⇒ finished" | zero-loss on the new support (S0b) | **irreducible by loss**; it *is* H_shift | DEPTH-BEYOND-SELF, MARGIN-GROWS, STOP-LOC |
| Pass / age deadline at t_d ≥ B + 1 | zero-loss (S1: P(R pass \| t ≥ 13) = 0) | open | STOP-LOC; D4 on top (§8) |
| Size / T-keyed rule | T > 47 out of support (n 16 T mean 50) | open | NOT-SIZE-KEYED slope; T-split at n 13–16 (descriptive) |
| Kept-but-wrong values | label-free checks are blind to values (fixture) | bounded: φ ≈ 0.005 (upper 0.010), post-filter π ≥ 0.997 to n 20 | GATE π; LABEL-AUDIT |
| Non-canonical R2 targets (value slot, frame, order) | KEEP admits them (fixture) | **closed by COMPOSE-VALID** (0 wrong E_self valid) | GATE π^E |
| Labeller drift across rounds | FT changes s(m) | open (round 2 only) | GATE-2, LABELLER-KEPT |
| Fresh-vs-trajectory input mismatch | targets are fresh-step decodes, applied to aged in-trajectory states | intended (it is the distillation); targets equal canonical E_t | decode-level readings stay descriptive (RT F1) |
| Pool memorisation | items repeat | closed: fresh-once pools | MEMO-GAP |
| Gold leak | root / t reach the targets | closed: no gold field; builder runs with the simulator patched to raise | G-S0 |
| Seed reuse reproduces the test set | a pool seed = 17100 | closed by exclusion against test_set(n_max 20) | G-S1 counts |
| In-range erosion | the plain share halves | open | FT-DIVERGED censor; K′24-STABLE |
| MEM-TRADED | new long trajectories retrain SLVA's history reads (Lemma D3: gradient on aged unfinished states, sign unknown) | open | KEEPS-SELF-STAB |
| One-pass precompute / coda shortcut | — | as §211 / §213 (coda V_0-free, pin 4) | PROBE, VALUE-SWAP, THINK-NECESSARY at n 9..B |
| Gate selection | — | the gate uses held-out items; the pool is audited after the fact | LABEL-AUDIT vs GATE π (CI overlap) |

---

## 5. Readings (comparator-free per cell)

Conditions: local fp32, eval_bs 32, one local card class for every SD cell, §170 test set (1,000 per n; round 2 extends to n 20 with test_set(n_max 20)). Files: the §213 contract (eval_vb, steppos, lfk, rs, d3hist), eval_vbhalt (acc at every K′ 0..24), eval_hist24, eval_chainlabel (+ _fresh), and a new eval_selfdepth.json (MEMO-GAP, LABEL-AUDIT, size slope). Chance c = 0.10.

| reading | definition | role | base D3-T |
|---|---|---|---|
| **ABSORB** (DEPTH-SELF-RANGE) | min over n 9..B of acc_n2 ≥ 0.95 | in-support: "beyond gold-supervised depth" | 0.127 (n 12) |
| DEPTH-GENERAL | §213: mean acc_n2 n 9–12 ≥ 0.80; **in support once B ≥ 12** | reported, labelled IN-SUPPORT | 0.480 |
| **DEPTH-BEYOND-SELF** | mean acc_n2 over n B+1..B+4 ≥ **0.80** | **primary C2** | (analogue at B 8: 0.480) |
| **MARGIN-GROWS** | mean acc_n2 over n B+1..B+4 ≥ **0.63** (base 0.480 + 0.15) | mechanism: GROW vs SHIFT | — |
| STOP-MOVED | ACC-GAIN-AFTER-10 (213-26) ≥ 0.10 | replaces STOPS-AT-10 (mostly in support) | −0.003 |
| STOP-LOC | G(k) = mean over readable n ∈ [k+1, min(k+4, 16)] of acc(K′ n+2) − acc(K′ k); STOP = the smallest k ∈ 10..14 with G(k) ≤ 0.02 (readable: acc(K′ n+2) ≥ 0.18 at ≥ 2 n, else never finishing = failure); **STOP-TO-SUPPORT iff STOP ≥ B + 1**; FIN-K per n beside it (D4 v1 def) | descriptive location of the moved stop (H_shift: 13–14) | 10 |
| THINK-GAIN-BEYOND | mean over n B+1..B+4 of acc(K′ n+2) − acc(K′ B) ≥ 0.10 | C2-literal, **non-discriminating** (H_shift ≈ 0.30) | G(8) 0.303 |
| K′-BEYOND | n 15–16: acc(K′ n+2) − acc(K′ 16) (K′ past trained max K 16) | C2-literal, reported | — |
| C2-TAIL | acc(K′ 24) ≥ acc(K′ n+2) − 0.05 at every readable n B+1..B+4 | no overthinking past training | trivially met at chance |
| LENGTH-FREE kept | §213 NO-KNEE m 9 / m 10 ∧ LFK (NO-KNEE is now in the trained range: a kept check, not a test) | kept | ✓ |
| STEP-GENERAL / LABELLER-KEPT | fresh m 12–16 ≥ 0.95; chain strict n 13–16 ≥ base − 0.02 | kept; gates round 2 | ✓ |
| VB-C1 kept | §211 definitions at n 3–8 (STALL stays untrained); VALUE-SWAP / STALL on n 9..B descriptive | kept | ✓ |
| THINK-NECESSARY-DEEP | acc at K′ 0 and K′ 1 ≤ c + 0.05 at n 9..B | C1 at the new depth (descriptive) | — |
| K′24-STABLE kept | acc_k24 ≥ 0.99 at every n 1–8; at n 9..B descriptive | kept | ✓ |
| KEEPS-SELF-STAB | D4 v1 def: CONTENT-STABILISES ∧ STAB_self ≥ 0.5 (hist24, n 3..8) | must-preserve, beside the claim; else **MEM-TRADED** | 0.782 |
| NOT-SIZE-KEYED | PROBE(12) extra-size slope (RT F7) > −0.10 at n 9–16 | Goodhart read | −0.035 |
| MEMO-GAP | \|acc(K′ n+2) on 2,000 consumed pool items − on 2,000 fresh items\| ≤ 0.02, n 9..B | Goodhart read | — |
| LABEL-AUDIT | gold precision of 5,000 random consumed pool items (eval-only, after training) ≥ 0.99 and inside GATE's CI | contamination check | — |

**Claims.**
- **SD-SELF** = ABSORB ∧ LENGTH-FREE ∧ VB-C1 ∧ K′24-STABLE: "latent depth extended from 8 to B by the model's own one-step labels; no gold beyond n 8; in support".
- **SD-C2** = SD-SELF ∧ DEPTH-BEYOND-SELF.
- **SD-C2-MEM** = SD-C2 ∧ KEEPS-SELF-STAB, reported on the same line. MEM-TRADED means "depth gained, memory channel lost".
- **Mechanism label:**
  - FULL (DEPTH-BEYOND-SELF);
  - GROW (MARGIN-GROWS ∧ ¬FULL);
  - SHIFT (¬MARGIN-GROWS ∧ STOP-TO-SUPPORT ∧ mean ≥ 0.40);
  - SHRINK (mean n B+1..B+4 < 0.40).
- **Attribution (with CTRL):** SELF-ATTRIBUTED iff FT-SELF reads STOP-MOVED ∧ CTRL reads ACC-GAIN-AFTER-10 < 0.10. **FT-EFFECT** iff CTRL ≥ 0.10 (re-heat alone moved it; attribution void).

**C4 / C1 notes (pre-stated wording).**
- **C4:** no gold label beyond n 8 enters training; deep labels come from the model's own single step. That step was trained with simulator-built per-step states at n ≤ 8 (R2), so **strict and revised C4 stay unmet**. The honest phrase is "depth beyond the gold-supervised range from self-generated labels".
- **C1:** the decode / re-render is a **training-time labeller only**. Test-time computation is fully latent: no decode, no re-render, no snaps, as in §213. VB-C1 is kept; THINK-NECESSARY-DEEP and VALUE-SWAP on n 9..B show serial latent use at the new depth.
- **C3 (descriptive):** if ABSORB holds, the latent loop matches its own explicit chain (decode → text → re-encode, n prelude re-encodes) at n 9..B with one prelude pass per item. FLOPs latent vs chain are reported (not computed here).

**Censoring (rule b).**
- These fail every reading of the cell:
  - a dead run (log < steps);
  - **FT-DIVERGED** (acc_n2 < 0.99 at any n 1–8, or acc_k24 < 0.97 at any n 1–8);
  - LABEL-AUDIT < 0.99 (CONTAMINATED);
  - an ARGS-DIFF mismatch (void).
- A failed GATE-q means round q is not run (no cell). Earlier rounds stand.
- Never-finishing is a failure: an unreadable STOP-LOC cell counts as not finished. Chance-level items count as wrong in every DEPTH reading.
- Entry is not an issue (FT from an entered model). The first 1k-bin self answer_ce is reported.

---

## 6. OC (one seed per cell; realised spend $0)

Priors for FT-SELF round 1 on D3-T [ASSUMED]:
- P(ABSORB) 0.65. Dense R2 at every pass; the base learned the 5 → 8 stage in ≤ 8k steps at 1e-4. Risk: lr 3e-5 with 6k self steps.
- Given ABSORB: H_shift 0.55 / H_grow 0.30 / H_full 0.10 / SHRINK 0.05.

| reading | \| H_shift (no extrapolation effect) | \| H_grow | \| H_full (effect) | unconditional | basis |
|---|---|---|---|---|---|
| ABSORB | — | — | — | 0.65 | ASSUMED |
| STOP-MOVED | 0.95 | 0.95 | 1.0 | ≈ 0.72 (0.3 without ABSORB) | n 11–12 are in support; sampling SE ≈ 0.01 |
| MARGIN-GROWS (≥ 0.63) | **0.07** | 0.85 | 1.0 | **≈ 0.25** | H_shift predicts 0.48; sampling SE 0.008; the margin-transfer uncertainty dominates [ASSUMED] |
| DEPTH-BEYOND-SELF (≥ 0.80) | **≤ 0.01** | 0.25 | 0.9 | ≈ 0.11 | as above |
| kept (LF 0.9 × VB-C1 0.9 × K′24 0.85) | 0.69 | 0.69 | 0.69 | — | ASSUMED, independent of the extrapolation |
| **SD-SELF** | — | — | — | **≈ 0.45** | ABSORB × kept |
| **SD-C2** | **≤ 0.01** | 0.17 | **0.62** | **≈ 0.075** | DBS × kept |
| SD-C2-MEM (× KEEPS-SELF-STAB 0.7) | ≤ 0.01 | 0.12 | 0.43 | ≈ 0.05 | KEEPS-SELF-STAB 0.7 [ASSUMED; D4 used 0.7 for an FT] |
| MEM-TRADED | — | — | — | ≈ 0.3 | ASSUMED |
| CTRL: STOP-MOVED | — | — | — | ≤ 0.02 | D0: CTRL adds no support [ASSUMED] |
| GATE-q (pooled ≥ 0.995 ∧ per-n ≥ 0.99) | P(pass \| π 0.997) **0.998**; P(pass \| π 0.995) ≈ 0.5; P(pass \| π 0.993) ≈ 0.01 | — | — | ≈ 1 for GATE-1 and for a base-labelled GATE-2 (measured fresh π ≥ 0.997, 213-38) | COMPUTED exact binomial (gate_oc2.json) |

- **How to read it:**
  - SD-C2 positive is strong: P ≤ 0.01 under no effect, 0.62 under H_full.
  - A DEPTH-BEYOND-SELF null is weak against H_grow (0.25).
  - MARGIN-GROWS is the informative mechanism split: 0.07 vs 0.85.
  - SD-SELF is the most likely positive, and it is in-support by construction.
- **Rate claims** need ≥ 3 seeds and the user's OK.
- **Round 2 (if run):** H_shift predicts n 17–20 ≈ 0.92 / 0.60 / 0.27 / 0.13 again. The OC is restated after round 1, by the pre-stated shift: SHIFT in round 1 → P(H_grow | round 2) ×0.5; GROW → ×1.5 (cap 0.8).

---

## 7. Experiments

### Stage 0 ($0, local, eval-only or CPU; Audit's OK for the GPU rows)

| step | what | status / cost | rule |
|---|---|---|---|
| S0-1 | **213-37 / 213-38 CHAIN-LABEL-FRESH: done** (D3-T horizon 20, post-filter ≥ 0.997, FRESH-MATCHES-TEST in both cells, collide 0.000). Remaining add-on: (a) COMPOSE-VALID per n, (b) π^E (E_self == recode_targets(gold) at every t, eval-only), (c) value-slot agreement, on the same fresh items | (a)–(c) need compose.py's checks inside the part (~5 min, 3070) | (a)–(c) feed GATE-1's state-precision term |
| S0-2 | item space / exclusion / T support | **done** here: 0 hits, 0 duplicates in 50k at n 4–20; test_set(n_max 20) identical on n ≤ 16; training T max 33 → 47 after round 1 | the small-strata trap does not apply at n ≥ 9 |
| S0-3 | composition gate | **done: 100,000 items, 850,429 steps, 0 failures**; fixtures in §2.1 | 0 diffs (as gate_rerender) |
| S0-4 | GATE-1 on D3-T: 2,000 fresh gate items per n 9–12 (rng 21960, §2.2), KEEP + COMPOSE-VALID vs gold | ~5 min, 3070 | §2.2 GATE |
| S0-5 | pool build, round 1: 480k items, base labeller, simulator patched to raise | ≤ 1.1 h, 3070 | G-S0 / G-S1 |
| S0-6 | support numerics (S1 table), gate OC, cost model | **done** (support_oc.json) | G-S3 must reproduce the S1 table from 20k logged draws |

### Gates before any spend (Audit's diff review)

| gate | check |
|---|---|
| G-S0 NO-GOLD | Pool records have no root / t / gold key (loader assert). The builder runs with N.apply, N.value, N.eval_tokens, N.cot_steps, VB.recode_targets and vbank_diag.reduce_k monkeypatched to raise, after generation. The gate items are scored in a separate process. grep: the trainer never reads the gate file |
| G-S1 EXCLUSION | pool ∩ (test_set(1000, n_max 20) ∪ SERIAL ∪ gate) = ∅ by exact syms; counts logged; the n computed from syms equals the generator n (asserted, then the generator n is dropped) |
| G-S2 COMPOSE | compose.py gate 0 diffs on 100k; trainer E_self == compose.py on 1,000 pool records bitwise |
| G-S3 MIXTURE | 20k logged draws reproduce S1's table (P(R pass \| t ≥ 9) 0.500 ± 0.01; 0 at t ≥ 13); self-n law per stage; plain batches hash-identical to CTRL on plain steps (first 200) |
| G-S4 INIT | `--init_from --keep_ckpt_steps 0 --stop_at 1`: ckpt_model_0.pt bitwise base model.pt on every key; **calibrate() not called (asserted log line; red-team trap: calibrate overwrites β / β⁰)**; calibration.json equals the base's; D3-F base: wpe sha unchanged |
| G-S5 ARGS-DIFF | only the §2.4 keys differ from D3-T's args.json |
| G-S6 LOAD | VE.load strict on the FT model.pt with the existing factory; type == args arch |
| G-S7 PLAIN-PATH | p_self 0 and no pool: 1 step bitwise vs pre-edit code (slva / pa / carry construct + 1 step) |
| G-S8 GRADFLOW | scratch/gradflow_check.py on a self batch: R1 and R2 grads nonzero on SLVA / core / R / coda; decorators by ast (no no_grad on the new paths) |
| G-S9 MEM | `--mem_check_self` at the forced shape (the longest n = B items, K = B + 4, bs 64), on the FT host; micro chosen ≤ 21.5 GiB; else self_kextra 2 (disclosed ARGS-DIFF); eval n 16 K′ 24 as before |
| G-S10 TIME | G-TIME on forced self + plain batches on the FT host; projected pair ≤ $1.5 at that host's price, else no launch |
| G-S11 OFF-HOST | full-state ckpt mirror every 2k (incl. pool cursor, rng_s, CurGate state) + a tested resume (user rule 10-15) |
| G-S12 FIXTURES | score_sd fixtures: SHIFT / GROW / FULL / SHRINK panels, CTRL FT-EFFECT, dead, FT-DIVERGED, CONTAMINATED, unreadable STOP-LOC |

### Stage 1 (needs the user's OK)

| option | what | cap | answers |
|---|---|---|---|
| **S1-A (recommended)** | FT-SELF round 1 (D3-T, n 9–12, 12k) + FT-CTRL (p_self 0), one 3090 host, sequential, matched plain streams | **pair $1.5** (projected $0.82–1.39) | ABSORB, the SHIFT / GROW split and SD-C2 with attribution. CTRL equals D4's FT-CTRL if both pin micro 1 and differ only in kmax (plain K never exceeds 12), so it can be **shared** |
| S1-B | FT-SELF round 1 only | $1.0 | the same, attribution descriptive (P(CTRL moves the stop) ≤ 0.02 [ASSUMED]) |
| S1-C (Lee-faithful) | 4 rounds × 3k steps (n 9; 10; 11; 12), each labelled by the previous round's model, GATE per round | $1.5 | bootstrapping and labeller drift at +1 per round; same final support |
| Round 2 (any option; separate OK) | n 13–16, labelled by the round-1 model's chain (GATE-2), test on the extended §170 set n 17–20 (new eval contract, memory at n 20 K′ 24) | $2.0 (projected $0.67–1.43 with self_kextra 2) | does the ratchet continue; does the margin grow with B |

- **The trade-off.** S1-A is cheap, controlled, and its CTRL doubles as D4's. S1-C costs about the same and tests the iterative property, but its later rounds are labelled by fine-tuned weights, for little gain: the base labels n ≤ 12 at post-filter ≥ 0.999 already.
- **Pre-stated follow-ups:**
  - SHIFT → the ratchet is real but linear in $; round 2 is optional. Combine with D4 pauses (§8) to ask whether the residual stop at ~B + 2 is age-keyed.
  - GROW / FULL → round 2, and a second seed for a rate claim.
  - ¬ABSORB → a longer FT (24k) or lr 1e-4 as a disclosed one-change re-run, needing the user's OK; else stop.
  - MEM-TRADED → report "depth gained, memory channel lost". No C6 improvement may be claimed.

---

## 8. Relation to D4

- **What each tests.**
  - D4 (computed PAUSE passes) decouples pass index and lineage age from progress; it cannot touch a progress-keyed stop (D0a).
  - SELF-DEPTH adds unfinished training passes with **ρ + r > 8**: the support / progress branch.
  - 213-32's synthesis leans toward a content-visible progress or support stop (age inert, bank index inert, S0-3 unreadable). Under that branch only SELF-DEPTH can act.
- **Discrimination is one-sided.** SELF-DEPTH moves both a progress boundary and any pass / age deadline ≤ B (S1 table). A SELF-DEPTH success therefore **does not identify the clock**. D4 does (L1–L3 ladder, START-D).
- **Combining.**
  - (a) **Sequential (recommended):** run SELF-DEPTH round 1, then D4-style pauses as an FT on the SD model. If SHIFT holds there, pauses ask whether the stop at ~B + 2 is pass / age-keyed (D4 moves it) or progress-keyed (only another round moves it).
  - (b) Simultaneous pauses on self batches: Kb up to B + 4 + 8 = 24 at T 47. Projected ≫ 24 GiB (micro 4), with 2–3× the cost. Not now.
  - (c) The shared FT-CTRL (S1-A).
  - (d) D4's decode-free FIN-K and ladder can be read on SD cells descriptively; L1 (> 12) is in support there.
- **If both are funded**, the order is SD round 1 first (cheaper, the branch Stage 0 favours), then D4's 48k or FT.

---

## 9. Top risks and falsifiers

1. **SHIFT** (the stop moves with the support; P ≈ 0.55 given ABSORB). *Falsifier of "self-depth extrapolates":* ¬MARGIN-GROWS ∧ STOP-TO-SUPPORT.
2. **¬ABSORB** (the FT budget or lr is too small to learn n 11–12; P ≈ 0.35). *Read:* ABSORB per n plus the self answer_ce curve.
3. **Label bias** (systematic kept-wrong). *Read:* GATE π, LABEL-AUDIT (both ≈ 1 expected).
4. **MEM-TRADED** (P ≈ 0.3). *Read:* KEEPS-SELF-STAB.
5. **In-range erosion / FT damage.** *Read:* the FT-DIVERGED censor.
6. **Memory or time overrun** (micro 2 doubles self-step time). *Rules:* G-S9 / G-S10; the fallback is self_kextra 2.
7. **Size-keyed residual** (T > 47 at n ≥ 15). *Read:* NOT-SIZE-KEYED, the T-split.
8. **Bootstrapping failure in later rounds** (labeller drift). *Read:* GATE-2, LABELLER-KEPT; stop rule.

---

## 10. Unverified in code or data (flagged) and verification log

- **No trainer code exists:**
  - init_from (with the calibrate bypass), the self-pool loader, E_self in rec_targets, per-type micro, the pool cursor in checkpoints, mem_check_self;
  - the pool builder, the GATE scorer, eval_selfdepth.json, the score_sd scorer;
  - the n 17–20 eval extension.
  - compose.py is the reference for E_self only.
- **Memory and time projections are extrapolations** of D4 v1's split (ASSUMED). G-S9 / G-S10 decide.
- The labelling throughput is 213-38's measured sec_per_item on the 3070 (fp32, bs 32). A host labeller (round 2 on the FT model) is not timed.
- φ rests on 6 kept-wrong items (D3-T, test + fresh) and 0 (D3-F). Its 95 % upper bound (0.010) is what the derivation uses.
- The labeller row past n 20 is unmeasured, so a third round (B 20) is out of scope here.
- Value-slot placement of fresh decodes (SPAN) is not measured on the model yet; S0-1(c) measures it. KEEP alone is blind to it (fixture).
- All OC priors are [ASSUMED]; per-n H_grow / H_full curves are [CONJECTURE].
- **Lee et al. 2502.01612** and **2511.07378**: read from the arXiv HTML (WebFetch, 2026-10-17); details as quoted in §2.2. Literature should verify before any citation in THEORY.

**Verification log (CPU, ≤ 1 thread each).**
- keep_rates.json: keep / strict / kept-wrong / post-filter precision per n, both cells (§1).
- item_space.json: exclusion hits, duplicates, T per n 1–20; the test_set(n_max 20) identity.
- support_oc.json: the S1 table, gate OC, H_shift numbers, cost / memory model.
- compose_smoke.json: the composition gate (2,000 items, 16,949 steps, 0 failures) and four corruption fixtures.
- compose_gate.json: the 100k-item composition gate, **100,000 items, 850,429 steps, 0 failures** (E_self == recode_targets at every t; KEEP, COMPOSE-VALID and the label hold on every item), with the same four fixtures.
- gate_oc2.json: exact binomial OC of GATE-q (pooled ≥ 0.995 ∧ per-n ≥ 0.99) at 213-38's keep rates.
- Read, not re-run: s213_d3t_s1 / s213_d3f_s1 eval_chainlabel_fresh.json (213-38: keep, prec_kept, sec_per_item).

*Draft completed 2026-10-17 (Opus subagent for Theory; $0; nothing committed or registered).*
