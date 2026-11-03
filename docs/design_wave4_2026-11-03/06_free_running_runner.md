# 06 FREE-RUNNING PASS RUNNER (shared infrastructure), design 2026-11-03

Lead design, wave 4. Read-only: no code, no runs, no numbers from stored outputs.

## 1. Goal and backlog IDs

**Goal.** One pass runner in which pass t's cue is chosen at run time, by the model, a search, or a schedule. Today the cue is precomputed: `passes()` builds x_1..x_Td and caches one lower per distinct input. The runner must:
- reproduce `passes()` bitwise on the gold cues;
- support teacher forcing (TF) and scheduled sampling (SS) in training;
- carry the eval parts of every consumer below.

| ID | What it needs from the runner |
|---|---|
| P1 SELF-CUE | name readout -> next cue; stop on the question's entity; TF + SS |
| P2 BACKWARD-PLAN | read-only (no-write) "who does X depend on?" hops, then a forward phase with the popped names |
| P8 CONFIDENCE-GAIN | per-pass candidate branching with sentence-span masks; decoder max-prob gain; STOP if none |
| HY1 TPLV | P1 + the {clean, permuted} x {d <= 4, d 6-8} crossover |
| TH2 MIRROR-PAIR | free runs on question-swapped twins; a question-mask control |
| AP5 latent search | beam over cue names (expand / prune), pluggable verifier |

**Precondition.** Lane C1 PLAN-PROBE ($0.15) should read first. It is the cheaper shared falsifier: if no one-pass head reaches 0.9 on next entity or parent, P1 / P2 park and only Build-A's GOLD / SEQ core is worth writing (§2.9).

## 2. Design

### 2.1 The step

Notation for item i:
- prefix P_i = [EOT] + enc(" " + text_i), of length L_i. The text includes the question.
- suffix at pass t: Q_{i,t} = enc(" " + QUESTION(u_i, c_{i,t})) + SFX.
- optional name tail N.
- pass input: x_{i,t} = P_i ⊕ Q_{i,t} ⊕ N.

Write row: w_i = L_i + |Q_{i,t}| - 1, the ':' row. The pass is InjectAW's:

H6 = lower(x) (blocks 0-5, no memory);  h = blocks 6-11(read_fixed(H6, S_{t-1}));  S_t = write_seq(H6[w], h[w], S_{t-1}).

**Prefix identity.** Blocks 0-5 are causal and memory-free, and positions are absolute (`wpe_slice`, POS_OFFSET 0 for bun). Hence H6(x_{i,t})[j] = H6(P_i)[j] for j < L_i, exactly in exact arithmetic.

**Blocks 6-11 cannot be cached.** `read_fixed` adds the read at every token (`deltamem.py` 158-176). Caching them would need E1's q-only read, which is a model change.

### 2.2 API: new module `boxes/fr_runner.py`

```python
class Prompt(m, prefixes, dev, cache="per_input" | "prefix_kv")        # shared per batch
class Runner(m, prompt, S0=None):
    step(cues, span_mask=None, write=True, name_q=None) -> StepOut(S, h, wpos, name_logits)
    expand(index)        # index_select of S, prefix rows and trajectories (branching / beam)
    decode(items)        # m.dec_logits(S, items, dev), unchanged
class Policy: first(items) -> cues;  next(items, t, out, hist) -> (cues, done)
run(m, items, dev, bs, policy, kset, aux_ts=(), on_state=None, edit=None) -> (answers by K', aux, traj)
```

`run()` keeps `passes()`' signature and outputs, and adds a per-item trajectory: names, stop pass, r_t, first deviation, name log-probs.

Policies:
- **GOLD:** spine[min(t, d)].
- **SEQ:** wraps any existing `seqs_of_t`, so `cue_only`, `cue_wrong`, `cue_clock`, the §261 target cue and the §252 generic cue run unchanged.
- **SELF** (P1), **BACKWARD** (P2), **CGAIN** (P8), **BEAM** (AP5).

### 2.3 Prompt caching

**`per_input` (the bitwise mode)**
- Builds x_t exactly as `cue_inputs` / `passes()` do (`pad`, then `m.lower(x_t)`), memoised on the input bytes.
- Then calls `m.pass_aw(S, x, mask, h_low)` itself.
- GOLD therefore cannot drift from the audited path.

**`prefix_kv` (fast)**
- `lower(P)` runs once per batch on the padded prefixes, keeping blocks 0-5's K / V (about 15 MB at B 100, L 120).
- Suffix rows run blocks 0-5 alone:
  - positions L_i + k;
  - SDPA with a boolean mask: key j is allowed iff j < L_i, or j is a suffix key at or before the query.
- h_low = [H6(P_i)[:L_i] ; H6(Q_{i,t})], right-padded, then `m.pass_aw` unchanged. The only new kernel is the suffix lower.
- Lower cost per batch, in token-blocks (L 100, suffix 11):

| d 4, K 8 | Cost |
|---|---|
| bun's per-distinct cache | 444 |
| per-pass recompute (forced by run-time cues) | 888 |
| prefix_kv | 188 |

**Padding**
- `pad` right-pads with EOT under causal-only attention, so pad rows are never attended.
- Per-pass, per-item lengths may differ (P2's phrasing). `passes()`' one-padding assert becomes per-pass masks plus the assert x[w] == ':'.

### 2.4 Span masks (P8; the TH2 / QMASK controls)

- `span_mask` holds per-item token spans [a, b) of the prefix whose keys are hidden from later queries in all 12 blocks. A row in a span still sees itself, so there are no NaN rows.
- The lower cache is keyed by (prefix, mask).
- Spans come from `insert_render.statements` (18-28), mapped to token offsets. G-SPAN asserts that the re-encoded spans concatenate to enc(" " + text).
- `span_mask=None` keeps `_block` (`is_causal=True`). A non-empty mask uses a copy, `_block_m`, which differs only in the attention call.

### 2.5 Branching

- `expand(index)` turns B rows into B' rows.
- **CGAIN** flattens the C_i candidates per item with a segment id and takes the argmax per segment.
- **BEAM** keeps 3 rows per item.
- Prefix caches are shared by index, so a branch costs one S copy plus a suffix lower.

### 2.6 Name readout

**No new parameters**
- The readout is the LM-head logits restricted to V_N:
  - the 40 NAMES tokens, each one token (asserted, `of_parts.py` 61);
  - plus one STOP token for P2 (G-VOCAB).
- Nothing is zero-initialised, so the usage-monitor rule does not bind. The name-CE gradient norm is still logged per 100 steps.

**`same_pass` (default)**
- N = enc(" Next:") is appended after ':'. The logits are read at the last row; the write row stays ':'.
- `pass_aw` takes the write row from mask.sum - 1, so this mode needs `step_core(S, x, mask, h_low, w)`: `pass_aw`'s body with an explicit row. With N empty, it is gated bitwise against `pass_aw`.
- The ':' row is upstream of N under causal attention, so S_t is unchanged in exact arithmetic. It is not bitwise, because T changes.
- The name sees S_{t-1}, the text and the current cue c_t. That is a text-pointer hop, like `name_swap`'s (`of_parts.py` 56-82).

**`post_write`**
- A read-only forward on S_t that asks " Who is next?". It doubles the pass cost.
- It is only for consumers that need S_t (AP5).

**P1 stop**
- The run stops when the emitted name equals A_q = spine[d]. Later passes repeat A_q, which is bun's own min(t, d) hold schedule.

### 2.7 Training: TF, SS, DAgger labels

New `inject_aw` flags. All are default-off, so the defaults are today's path:
- `--cue_src {gold, self}`;
- `--name_ce λ_n`;
- `--name_q {none, same_pass}`;
- `--ss p0,p1,s0,s1` (linear);
- `--lower_mode {per_input, prefix_kv}`.

Rollout, per item and pass:
- **Pass 1:** c_1 = spine[1].
- **Correct prefix:** r_t = max{r : c_{1..r} = spine[1..r]}.
- **Expert label (DAgger relabel, 1011.0686):** n*_t = spine[min(r_t + 1, d)].
- **Next cue:** c_{t+1} = argmax_{V_N} z_t with probability p_ss, else n*_t.
  - The draw uses a dedicated `default_rng((seed, SEED_SS, step, global_item_index, t))`. The torch RNG is untouched, so p = 0 is bitwise.
  - Under TF, r_t = min(t, d) and n*_t = the gold c_{t+1}. SS at p = 0 is TF (1506.03099).

Loss: L = CE_ans + λ_aux·aux + λ_n·L_name, where L_name = mean over t <= d of CE(z_t[V_N], n*_t).
- The repeated A_q labels after pass d would dominate, so they are excluded beyond one stop pass.

Censoring under SS:
- aux at pass t only while r_t = t;
- answer CE only on recovered items (r_K = d). Otherwise derailed runs train the decoder's text shortcut (Note 264-1).
- `ss_cens` (the masked share) is logged per bin.

`hold_step`:
- It still receives the gold S_d input `xs[-1]`.
- Under SS, the penalty is restricted to items with r_d = d. Otherwise it would freeze unrecovered states and block recovery writes.

Lower cache:
- TF uses today's per-distinct cache.
- SS uses `prefix_kv`. This is a recipe flag, gated in fp64 (G-FR1) plus an fp32 loss-trajectory smoke.

### 2.8 Eval parts: `boxes/fr_parts.py`, registered in `run()` like `of_parts` / `sc_parts`

**`fr_gold`.** The G-FR0 replica on the host: bitwise flag, max |ΔS|.

**`fr_self` (P1 / HY1).**
- Sets: TEST2 d 1-8 at K' 0..16; TEST-PERM d 2-4; insert_ho H-EXTEND / H-LIVE / H-CHAINSRC; HY1 cells.
- Per item: names, stop, r_K, first deviation, name log-probs, preds, WS readout per pass.
- Controls:
  - **ORACLE:** GOLD cues.
  - **NAME-SHUFFLE:** the item's own free-run names permuted over passes 2..K (seeded; stop name last).
  - **QMASK:** the question sentence span-masked. Read on **H-EXTEND items**: on clean items the chain simply ends at S_d, so "stop at the end of reachability" passes without the question.
  - **FR-SWAP-K:** `swap_pairs_d` pairs run free, then the recipient's S_K is replaced by the donor's.

**`fr_backward` (P2).**
- Phase 1: write=False hops with " Who does X depend on? The answer is:", starting from A_q, until STOP or 8 hops.
- Phase 2: forward with the popped names.
- Records the parent accuracy per hop and the answers.

**`fr_cgain` (P8 stage 1; gb_s1, generic cue, no training).**
- Candidates: the unchosen op-like sentences j.
- For each candidate: the input with every other op-like sentence span-masked.
- Score: Δ_j = maxprob(dec(S'_j)) - maxprob(dec(S_{t-1})).
- Choose argmax_j; STOP if max_j Δ_j <= 0.
- Records the chosen order vs gold.

**`fr_beam` (AP5).**
- Beam 3 over in-text names; post_write verifier.
- Score: Σ_t log max_v p_dec(v | S_t). The final answer is the best score at K.

**`fr_mirror` (TH2).**
- Pairs from H-CHAINSRC items (X = S_j op c inserted after S_j, j < d). Member A asks S_d; member B asks X; the text is otherwise the same. `OW.solve` is asserted.
- **PAIR-BOTH** = both members follow their own branch at the fork. A question-blind policy chooses identically for both members, so PAIR-BOTH = 0 for it by construction (TH2's prod 1/b bound). QMASK must give about 0.
- On bun, which has no name head, the policy is CGAIN or BEAM.

### 2.9 Staged build

| Build | Contents | Size |
|---|---|---|
| Build-A | GOLD / SEQ / SELF, per_input + prefix_kv, TF / SS trainer branch | about 350 lines |
| Build-B | masks, expand, CGAIN / BEAM / BACKWARD / mirror | about 350 lines |

Build-B is built only when P8, AP5, P2 or TH2 is registered.

## 3. Code touch points (read for this memo)

| File | Lines | Use |
|---|---|---|
| boxes/inject_aw_eval.py | 32-33 PARTS / AW_ONLY; 44-47 `seq_std`; 50-52 `n_distinct`; 55-82 `passes`; 85-92 `answers_by_kprime`; 95-96 `tokd`; 1297-1349 `run()` dispatch | GOLD / SEQ reference; `fr_*` registration (the s270 / s271 edit pattern) |
| boxes/inject_aw.py | 47-56 `pass_aw` (row = mask.sum - 1, ':' assert); 58-59 `write`; 61-63 `colon_logits`; 76-96 `tok_aw`; 159-199 `hold_step`; 361-419 `loss_reg` (cue choice at 373); 421-431 `loss_ask` | per_input calls `pass_aw`; `step_core` mirrors it; the `--cue_src self` branch at line 373 |
| boxes/inject_st.py | 44-46 `cue_str`; 49-63 `tok_st` (CUE_LEN 7); 70-80 `cue_inputs` (one-padding assert) | suffix builder; TF path unchanged |
| boxes/nested170.py | 166-170 `pad`; 228-233 `aux_prompt` (pass token t); 241-243 `lower`; 245-252 `think` | padding law; aux indexing by pass |
| boxes/deltamem.py | 43-48 `wpe_slice`; 158-176 `read_fixed`; 179-223 `write_seq`; 251-260 `_block`; 263-269 `gpt_lower` | prefix_kv = `_block` with past K / V and a mask |
| boxes/of_parts.py | 56-82 `part_name_swap` | in-pass cue-token substitution precedent |
| boxes/sc_parts.py | 36-97 `part_cue_clock` | SEQ equivalence target |
| boxes/insert_render.py | 18-28 `statements`; 107-150 `held_out` | spans; mirror pairs |

The bodies of `inject_aw.py`, `inject_aw_eval.py` and `deltamem.py` are untouched, apart from registration lines and the `loss_reg` branch.

## 4. Gates

Every gate runs on a cloud host. The GPU bitwise gates run inside `scratch/s222_det_wrap.py`.

| Gate | Test | Pass |
|---|---|---|
| G-FR0 | GOLD / per_input vs `passes()`, bun_s1, TEST2 d 1-8 (200 / d), K' 0..14, aux_ts (1, 2) | every S_t `torch.equal`; answers / aux identical |
| G-FR0b | SEQ on `cue_only`, `cue_wrong` (all, at_2), `cue_clock` STD / ER / LF / CY, target, generic | same |
| G-FR0c | `step_core` (N empty) vs `pass_aw` | bitwise |
| G-FR1 | prefix_kv vs per_input in `.double()`, d 1-8 x 50, K' 14, incl. variable-length cues | max_t ‖ΔS_t‖/‖S_t‖ and max |Δ dec logit| <= 1e-9; fp32 flips descriptive |
| G-FR2 OFF-BIT | new trainer, defaults, vs a reference tar, bun recipe, 60 steps (gate_s271 pattern) | ckpt (model / opt / rngs / stream) + logs equal |
| G-FR3 p=0 | (a) self + ss 0 + name_ce 0 vs gold: bitwise. (b) name_q same_pass + name_ce 0: fp64 S_t and loss <= 1e-9 | as stated |
| G-FR4 | empty mask == None (bitwise); `_block_m` all-causal vs `_block` (fp64 <= 1e-12); `expand(arange)` identity; 3 identical branches == 1 | as stated |
| G-SPAN / G-VOCAB | span re-encode; STOP / " Next:" token counts | asserted |
| SMOKE | self, ss 0->0.5 over 20 steps, name_ce 1, prefix_kv | rc 0, finite losses, `ss_cens` > 0 at some step |
| GRADFLOW | gate_s251 PROBE copy, d >= 2 | answer-CE grads nonzero on mem qkv / gate / ln_w / wo; name-CE grads nonzero on blocks 0-11, finite |
| MEM_CHECK | peak with SS + name_q at d_max, 1 GiB headroom; eval peak with CGAIN C 12 and beam 3 | micro / eval_bs chosen |

## 5. Criteria (pre-registration-ready)

**Infrastructure (binding).** G-FR0 / 0b / 0c, G-FR1 and G-FR2 must all pass before any consumer output is read. Otherwise the run is VOID.

**First consumer: the P1 / HY1 pilot.** From bun_s1, 2k steps, 1 seed, 2 arms: TF (λ_n 1) and SS (p 0 -> 0.5 over 0.5k-1.5k). Every rate carries a Wilson 95% CI. NEAR = the CI contains the threshold.

| Row | Reading (fr_self, K' 16) | Bar | Status |
|---|---|---|---|
| NAME-2 | P(c_2 = spine[2]), d 3-4 | >= 0.6, else PARK | binding |
| FREE-D3 | free-run acc, d 3 | >= 0.80 | binding |
| STOP@Sd | P(the first A_q emission is at pass d), d 2-4 clean AND H-EXTEND; never-stop = fail | >= 0.80 clean, >= 0.60 H-EXTEND | binding |
| MECH-D | free-run WS readout, passes 2..d, d 3-4 | >= 0.50 | binding (mechanism at the claimed depth) |
| NOT-DECODER | FR-SWAP-K follow_U, d 3 | >= 0.80 | binding (shortcut exclusion) |
| CUES-MATTER | FREE-D3 - NAME-SHUFFLE | >= 0.30 | binding |
| Q-USE | STOP@Sd(H-EXTEND) - STOP@Sd(H-EXTEND + QMASK) | >= 0.40 | binding |
| H-LIVE, H-CHAINSRC, TEST-PERM, d 6-8, HY1 crossover | acc + trajectory census | — | descriptive (C2 needs >= 3 seeds in the full stage) |
| SS - TF | paired McNemar on FREE-D3 | — | descriptive; picks the full arm |

**Verdict.** GO = every binding row passes in one arm. NEAR-GO = a binding row is NEAR. PARK = NAME-2 < 0.6 in both arms. Otherwise MIXED.

**OC block (priors, committed before output).**

| Outcome | Prior |
|---|---|
| G-FR0 passes first time | 0.85 |
| G-FR1 passes | 0.8 |
| TF NAME-2 >= 0.9 | 0.6 |
| TF GO | 0.35 (STOP on H-EXTEND is the weak row; the §260 generic cue gave H-EXTEND 0.08) |
| SS GO | 0.4 |
| PARK | 0.2 |
| NOT-DECODER fails | 0.15 |

**Censoring.**
- Never-stop by K' 16: censored from the stop-pass distribution, but counted as a failure in STOP@Sd and FREE-D3.
- Out-of-item names are deviations. They are never filtered.
- G-SPAN build failures are dropped and reported. Above 1%, the run is VOID.

## 6. Scorer sketch: `theory/score_fr.py`, committed before output

**Fields read**
- `eval_fr_gold.json`: `bitwise`, `max_rel_dS`, `n`.
- `eval_fr_self.json`: `items[set][d]` = {`ans`, `pred`[N][17], `names`[N][16], `stop`[N] (-1 = never), `r`[N], `first_dev`[N], `ws_ok`[N][16]}.
  - set ∈ {clean, shuffle, oracle, test_perm, ho:H-EXTEND, ho:H-EXTEND:qmask, ho:*, hy1:*}.
- `eval_fr_swap.json`: `follow_U`, `floor`.

**Labels:** VOID / PARK / MIXED / NEAR-GO / GO per arm.

**Fixtures:** GO; PARK; a NEAR edge (0.80 at n 500); all never-stop; `bitwise` false -> VOID; a missing field -> raise. Every field path is grepped in `fr_parts.py` (scorer field-path rule).

## 7. Cost (RTX 3090 at about $0.2/h)

| Item | $ |
|---|---|
| Gates + smoke + gradflow + mem_check (about 0.4 h) | 0.08 |
| P1 / HY1 pilot, 2 x 2k (0.62 steps/s, so about 0.9 h each, +25% for name_q / SS) | 0.45 |
| fr_self + controls on 2 ckpts | 0.10 |
| fr_cgain on gb_s1 (C <= 12, K 8) | 0.10 |
| fr_mirror + fr_beam | 0.10 |
| **Total** | **about 0.85** |

## 8. Red team

**Provenance.** The required separate sub-agent could not be spawned: "Concurrent subagent limit reached". This section is the lead's own adversarial pass, and it is **not** independent. Theory should run a separate red-team agent on this file before registration.

1. **STOP and Q-USE are passable without the question.** On clean items, "stop when no op reads the current entity" lands on S_d. **Fixed:** STOP@Sd is binding on H-EXTEND, and Q-USE is read on H-EXTEND.
2. **`pass_aw` takes the write row from the mask.** The " Next:" tail would move the write to the tail token. **Fixed:** `step_core` with an explicit w, plus G-FR0c.
3. **The hold penalty under SS** froze unrecovered states. **Fixed:** restricted to r_d = d.
4. **Name-CE dilution** by the repeated A_q labels. **Fixed:** mean over t <= d plus one stop pass.
5. **Answer-CE censoring on recovered items** biases SS toward easy items and weakens the recovery signal. The `ss_cens` share is logged. If it exceeds 0.3 at the end, the SS arm is labelled CENSOR-HEAVY (descriptive).
6. **The name readout may be pure text pointer-chasing** (S unused). That is acceptable for plan-free C1, but it is not latent planning, and the claim must say "text-space plan, latent value" (HY1 framing). Plan names are training labels, so this is **not C4**.
7. **NAME-SHUFFLE keeps the stop name last.** Otherwise the shuffled run would stop early and the drop would be inflated.
8. **prefix_kv in bf16 training** sits further from per_input than fp32 does. G-FR1 checks only the math; the fp32 / bf16 loss-trajectory smoke is the guard. Any doubt: train SS with per_input (+~40% lower FLOPs, about +$0.1).
9. **The cheaper falsifier is PLAN-PROBE** (§1). Build-A without prefix_kv is enough for P1. The masks / branching (Build-B) wait for a consumer.
10. **TH2 on bun has no name head.** fr_mirror's result there measures CGAIN / BEAM, not P1. Rerun it on the P1 checkpoint before using it to decide P1 vs P2.
11. **The DAgger relabel after an off-chain write** asks for recovery on a contaminated S. That is intended, but STOP@Sd should be reported split by "ever deviated" (descriptive).
12. **SS Bernoulli seeding** must use the global batch index across micro chunks, or chunks would replay the same coins. Assert uniqueness in the smoke.

## 9. Verified arXiv ids (abs page fetched; title matched)

| id | Title |
|---|---|
| 1506.03099 | Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks |
| 1011.0686 | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger) |
| 1610.09038 | Professor Forcing: A New Algorithm for Training Recurrent Networks |
| 2412.06769 | Training Large Language Models to Reason in a Continuous Latent Space |
| 2309.06180 | Efficient Memory Management for Large Language Model Serving with PagedAttention |
| 2312.07104 | SGLang: Efficient Execution of Structured Language Model Programs |
| 2305.10601 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models |
| 2601.21358 | Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization |
| 2505.12514 | Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought |
| 2407.20311 | Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process |
| 2212.13894 | LAMBADA: Backward Chaining for Automated Reasoning in Natural Language |
