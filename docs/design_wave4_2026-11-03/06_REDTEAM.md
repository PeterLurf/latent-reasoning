# 06 FREE-RUNNING PASS RUNNER: independent red team, 2026-11-04

Target: `docs/design_wave4_2026-11-03/06_free_running_runner.md` (the "doc"). Read-only review: no runs, no reads of stored run outputs. Code claims were checked against the cited lines.

## Verdict: GO-WITH-FIXES for the GOLD / SEQ runner core. NO-GO on the P1 pilot as written.

- The runner core (GOLD / SEQ, `per_input`, G-FR0 / 0b / 0c) is sound and cheap, and it is needed anyway (QUERY-HANDOFF, fr_self evals).
- The P1 / HY1 pilot's criteria can't test what they claim:
  - the stop is decided by the harness;
  - the SS progress index freezes at the first deviation;
  - NAME-SHUFFLE can't fail at d 3;
  - the binding name rows sit on chain-ordered text, where "next sentence's subject" solves the plan;
  - the base is bun, not the order-free of_s1.
- Fix findings 1-5 before registration.

## Findings

### 1. HIGH: STOP is the harness's job, so STOP@Sd and Q-USE are passable without the question.

**Evidence**
- Doc §2.6 "P1 stop": "The run stops when the emitted name equals A_q = spine[d]". The runner knows A_q from the item.
- H-EXTEND (`boxes/insert_render.py:135-138`) inserts `X1 = S_4 + c1` and `X2 = X1 + c2` after S_4.
- A question-blind "follow the reader of c_t" policy emits spine[2], spine[3], spine[4] = A_q on that item. The only reader of S_3 is S_4. So it hits A_q at the right pass, and the harness then stops it before it could ever continue to X1.
- STOP@Sd(H-EXTEND) is therefore about as high for the blind policy as for a question-using one.
- QMASK can't change a stop the harness makes, so Q-USE is about 0 whatever the model does. The binding row fails without carrying information.
- Red-team item #1 in the doc claims this is fixed. It is not: the fix moved the reading to H-EXTEND but kept the stop in the harness.

**Fix**
- The model must decide the stop. The runner must not see A_q. Two options:
  - (a) an explicit STOP token in V_N (it already exists for P2, G-VOCAB);
  - (b) stop = the model re-emits its current cue (n_t = c_t), which is the hold.
- STOP@Sd = P(the first model-side stop happens at cue pass d, with no off-chain cue before it).
- On H-EXTEND, emitting X1 after A_q is the failure the row exists to catch.
- Read Q-USE where the question is needed to choose:
  - at the **H-CHAINSRC fork**: at the pass after cue S_j (`insert_render.py:121-123`; X reads S_j), measure P(next = spine[j+1]) vs P(next = X), with and without QMASK;
  - and on H-EXTEND, as P(stop rather than X1), with and without QMASK.

### 2. HIGH: the SS progress index r_t is a strict prefix, so after one deviation, labels, censoring and the hold all freeze.

**Evidence**
- Doc §2.7 defines r_t = max{r : c_{1..r} = spine[1..r]} and n*_t = spine[min(r_t+1, d)].
- After the first off-chain cue at pass r+1, r_t stays r for every later t, even after the model emits spine[r+1]. Consequences:
  - the DAgger label then says "spine[r+1]" forever, which teaches repetition after recovery;
  - "answer CE only on recovered items (r_K = d)" holds only for items that never deviated, so recovery is never rewarded;
  - aux "while r_t = t" and `hold_step` "r_d = d" censor the same set.
- §263 NAME-SWAP (THEORY Note 263-3, l. 24479 ff.) shows two things:
  - re-cueing an already-computed entity returns its value (follow 1.000);
  - a jump-ahead cue doesn't compute (follow about the floor).
- So the dependency-correct progress measure is a subsequence, not a prefix.

**Fix**
- Define p_t = the largest r such that spine[1..r] occurs **as a subsequence** of c_{1..t}.
- Set n*_t = spine[min(p_t+1, d)]. "Recovered" means p_K = d.
- Gate it with a CPU fixture before any run:
  - deviate, then recover → the label advances;
  - jump ahead, then fill in → p counts the late fill-in, but only once spine[r] follows spine[r-1].
- State as DERIVED-UNDER-ASSUMPTIONS that a stale value written by a jump-ahead cue is overwritten by the later correct write. The delta rule has erasure only in the delta heads, so this is an assumption, not a fact. Report a "jumped-ahead-then-recovered" stratum.

### 3. HIGH: NAME-SHUFFLE keeps the subsequence order at d 3, so CUES-MATTER can't fail informatively.

**Evidence**
- Doc §2.8: the names are permuted over passes 2..K (K' 16) with the stop name kept last.
- At d 3 the free names over passes 2..16 are {c_2 once, A_q 14 times}. With A_q pinned at pass 16, c_2 always precedes the final A_q pass. The run c_1 … c_2 … A_q(final) therefore always contains the chain subsequence spine[1], spine[2], spine[3].
- By Finding 2's NAME-SWAP mechanism, the final pass recomputes A_q correctly. The shuffle answer at K' 16 is about the free-run answer.
- At d 4, the order {c_2, c_3} is kept with probability 1/2.
- The control is not exposed to the intervention (memory: control-must-be-exposed).

**Fix**
- Shuffle only the distinct names of passes 2..stop, as a derangement that breaks the chain subsequence.
- Read the answer at K' = the stop pass (not 16), and also at K' = d.
- Pre-state the fraction of shuffles that keep the chain subsequence (it must be 0 by construction).

### 4. HIGH: the binding name rows are on chain-ordered TEST2, and the base is bun_s1 rather than of_s1.

**Evidence**
- Backlog §12.5 (EXPERIMENT_BACKLOG.md:443-444): "Plan-free gates need branch families + TEST-PERM. On clean chain-order text, 'next sentence's subject' solves the plan."
- The doc's binding NAME-2, FREE-D3 and MECH-D rows are on TEST2 d 3-4 (clean). TEST-PERM is descriptive.
- bun's lineage is order-bound: Note 251-16 measured TEST-PERM about 0.15-0.20, and 257-3 found dch order-bound. On bun, TEST-PERM fails even with gold cues, so it can't be binding there.
- of_s1 / of_s2 (Note 263-3) are order-free over op sentences, with TEST-PERM d 4 = 0.982 / 0.961 under gold cues. They are the SOTA head, and their model.pt is kept.
- bun's model.pt is kept only "until §264-§269 eval-only parts finish" (Note 263-3, Next). It may not exist at pilot time.

**Fix**
- Base: **of_s1**.
- Make NAME-2 and FREE-D3 binding on **TEST-PERM d 3-4**. Keep clean TEST2 as descriptive, and report the gap between the two.
- The G-FR0 / G-FR2 reference recipe becomes of's recipe. It has the perm and ask-item flags, which `loss_items` routes through `loss_ask` (inject_aw.py:421-431). That makes OFF-BIT exercise the code paths that will actually be used.

### 5. MED-HIGH: c_1 = spine[1] is given, which leaks the plan's first step.

**Evidence**
- Doc §2.7: "Pass 1: c_1 = spine[1]". On H-LIVE, the first computed entity on the path is not the first in reachability order, and at d 1 the whole plan is given.

**Fix**
- Choose c_1 at run time: from a name readout on a pass-0 read-only forward with the generic cue (`GEN_CUE`, inject_aw.py:73), on S_0.
- Otherwise, scope every claim "plan given at pass 1" and add NAME-1 as a binding row on H-LIVE.

### 6. MED: G-FR3(a) is bitwise only if the trainer memoises lowers on input bytes. The doc routes SS to prefix_kv.

**Evidence**
- Today's training builds one lower per distinct input with grad and reuses it across passes (inject_aw.py:368-370, `hls`).
- If SS recomputes per pass, the forward values are equal, but the lower's gradients become a sum of K graph contributions instead of one accumulated tensor. The summation order changes, so (a) "self + ss 0 vs gold: bitwise" fails. With prefix_kv, it is not bitwise by construction.

**Fix**
- In training as in eval, memoise `m.lower` on the batch tensor bytes (`per_input`).
- At p = 0 every pass tensor equals gold's `xs[j]`. The memo hits, the graph is identical, and (a) can be bitwise.

### 7. MED: drop `prefix_kv` from Build-A. It is a new kernel for a saving of about $0.1.

**Evidence**
- The doc's own table, in token-blocks, with blocks 6-11 counted (8 passes x 111 = 888):

| Mode | Total |
|---|---|
| today | 444 + 888 = 1332 |
| per-pass `per_input` | 888 + 888 = 1776 (+33%) |
| prefix_kv | 188 + 888 = 1076 |

- The extra cost for the 2 x 0.9 h pilot is about $0.06-0.1 (doc §8.8 says about +$0.1).
- prefix_kv brings in:
  - a boolean-mask SDPA path, which takes a different kernel from `is_causal=True` (deltamem.py:256);
  - G-FR1;
  - bf16 drift;
  - a mem_check of its own.
- If it is ever needed, `core.py:141-147` already has a past-K/V plus boolean-mask forward (`pos0`, `kv_mask`) to copy, so no new kernel would be needed.

**Fix:** per_input only. prefix_kv becomes a Build-B item, gated by G-FR1 when AP5 or CGAIN cost demands it.

### 8. MED: G-FR4 "3 identical branches == 1" bitwise is not guaranteed on GPU.

**Evidence**
- `expand` changes B. cuBLAS can pick a different GEMM algorithm by M, even under `s222_det_wrap.py` (math SDP plus deterministic algorithms, l. 13-14). That makes results deterministic for a given shape, not invariant across shapes.
- The project already hit this (memory: "Repro: match batch size").

**Fix:** compare the branches against each other in one B' batch (bitwise), and against B = 1x in fp64 (<= 1e-12).

### 9. MED: the tied LM head means name-CE moves the input embeddings.

**Evidence**
- `core.py:159-160`: `logits = h @ wte.T`. Name-CE on V_N updates the 40 name rows of `wte`. Those rows are read in the text, in every cue (QUESTION's A token) and in the decoder input.
- So "No new parameters" understates the change: the gold-cue pathway itself is perturbed.

**Fix**
- Log a pre/post GOLD-cue fr_gold accuracy (RETAIN, binding: TEST-PERM d 4 >= of - 0.02).
- Optionally add a frozen-wte arm, with grads to blocks only.

### 10. MED: the infrastructure doc does not state the decision each consumer makes, and the consumer order is pre-of.

**Evidence**
- The §1 table lists what each consumer needs, not what it decides.
- The doc's own §8.10 says TH2 decides P1 vs P2 and needs Build-B on bun, then a rerun on P1.
- After fixes 1 and 4, P1's Q-USE fork row measures the same question-blind bound (prod 1/b; 2505.12514) on P1's own policy. TH2 is then redundant for that decision.
- of makes **P2 BACKWARD** newly attractive, for three reasons:
  - phase 1 is a write=False one-hop text QA ("who does X depend on?") with no compounding through S and no SS;
  - phase 2 runs the popped names through of, which follows gold names order-free (TEST-PERM 0.98);
  - the stop is natural, at a GIVEN.
- On bun, phase 2 fails on TEST-PERM.

**Fix**
- Add a "Decision" column:

| Consumer | Decision |
|---|---|
| P1 | GO → full 3-seed P1; PARK → P2 |
| P2 | GO → the plan-free C1 claim is "question-conditioned text plan, latent value" |
| P8 | stage-1 GO → Q-SELECT critic |
| AP5 | descriptive search bound |
| TH2 | dropped (subsumed) |
| HY1 | rides on P1 |
| QUERY-HANDOFF | missing from the consumer table; backlog l. 774 says it runs before P1 on this runner. It needs a query-patch hook inside `read_fixed`, which `edit(t, S)` can't express |

- Under north-star priority C1 first, the order is:
  - QUERY-HANDOFF ($0, diagnostic);
  - then P1-TF and P2 as two arms of one pilot on of_s1;
  - SS only if TF shows exposure bias (NAME-2 high but FREE low);
  - Build-B consumers only on registration.

### 11. MED: PLAN-PROBE, as specified, is cheap but not decisive.

**Evidence**
- Backlog l. 445: a plain GPT-2 fine-tune with heads for next / parent / is-ancestor.
- The parent head is a one-hop local text lookup and will almost surely pass 0.9. It can't park P2.
- The next-entity head is ambiguous only at forks (H-CHAINSRC) and at H-EXTEND's stop. Pooled over all positions it is dominated by unambiguous steps and will pass.
- "If one pass cannot, the in-loop heads won't either" does not hold for P2: its multi-hop plan is built over several write=False passes.
- No stored output answers it:
  - §263 NAME-SWAP measures whether the loop *follows* a name, not whether it *predicts* one;
  - §260 (Note 260-5: the loop follows reachability and does not stop, H-EXTEND 0.08 / 0.02) is a prior that the implicit policy is question-blind, not a probe of a trained name head.

**Fix**
- Re-specify PLAN-PROBE as fork-stratified:
  - binding next-entity accuracy at the H-CHAINSRC fork pass, plus stop-vs-X1 on H-EXTEND;
  - each with a QMASK control;
  - TEST-PERM ordering.
- Use the of_s1 trunk rather than plain GPT-2, so the reading transfers.
- Park only P1 on failure; P2 parks only if the parent head is below 0.9.
- Cost unchanged ($0.15).

### 12. MED: the math backing and OC block are incomplete under the standing rules.

**Evidence**
- The doc has no claim tags. These need tags:
  - prefix identity: PROVED, given causal masking and absolute wpe (deltamem.py:43-48, 263-269);
  - S_t unchanged by the " Next:" tail: PROVED in exact arithmetic;
  - DAgger relabel optimality: DERIVED-UNDER-ASSUMPTIONS (the stale-write overwrite, Finding 2);
  - "SS at p 0 is TF": PROVED;
  - name readout feasible in one pass: CONJECTURE, tied to the PLAN-PROBE reading.
- The OC block gives priors on events, but not the required P(pass rule | no effect) and P(pass | effect) at the panel's seed count (1 seed). For example: P(GO | question-blind policy) must be about 0 after fixes 1 and 3. Before them it is not, which is how findings 1 and 3 show up.
- The 1-seed pilot must be scoped "one lineage, pilot".

**Fix:** add a tagged derivation section and a null-policy OC row computed by a CPU simulation of the blind reachability policy on the eval items.

### 13. LOW: STOP@Sd has an off-by-one ambiguity.

The readout at pass t emits c_{t+1}. Under gold, A_q is emitted at pass d-1 and cued at pass d. "First A_q emission at pass d" mixes the emission index with the cue index. Define every trajectory field by cue pass and say so in the scorer fixtures.

### 14. LOW: SS anneals only to p = 0.5, but the eval is fully free.

The share of fully self-generated d 3 trajectories in training is about 0.5^2 = 0.25. Anneal to p1 = 1.0 by 1.5k, or report SS as "partial".

### 15. LOW: `hold_step` and the name tail.

`hold_step` writes through `m.write` → `pass_aw`, whose write row comes from the mask (inject_aw.py:54, 179-182). The `xs[-1]` it receives must be the tail-free gold S_d input, or the ':' assert trips. Add this to the SMOKE.

### 16. LOW: code citations hold.

- `pass_aw` row = mask.sum-1, with the ':' assert (inject_aw.py:54-55).
- The `passes()` one-padding assert (inject_aw_eval.py:63).
- The lower-cache-per-distinct-input semantics (inject_aw_eval.py:66).
- `read_fixed` at every token (deltamem.py:158-176).
- `statements()` (insert_render.py:18-28).

G-FR0 is achievable. GOLD reuses identical batch tensors, keeps the batch composition and shape (no dropping of stopped rows; SELF holds A_q, so B is constant), and runs inside the det wrap. Requirements:
- the same `bs` and item order as `passes()`;
- no batch shrinking under any policy (assert it);
- the ambient autocast context copied from `run()`.

## Minimal staged plan I would approve

| Stage | Contents | $ | Gate / exit |
|---|---|---|---|
| S0 (doc, $0) | Apply fixes 1-5 and 10-12: a model-side stop; subsequence progress; a derangement shuffle read at the stop pass; base of_s1 with binding rows on TEST-PERM plus the H-CHAINSRC fork plus H-EXTEND stop-vs-X1; c_1 chosen at run time or scoped; a Decision column; claim tags; a null-policy OC row from a CPU sim | 0 | Audit G-D |
| S1 | PLAN-PROBE, re-specified fork-stratified (Finding 11) | 0.15 | Parks P1 only if the fork head is < 0.9 |
| S2 Build-A0 (about 150 lines) | `fr_runner.py` GOLD / SEQ / SELF with `per_input` memo only; `step_core`; `fr_gold` | ~0.05 | G-FR0 / 0b / 0c in det wrap, plus the no-shrink assert |
| S3 | QUERY-HANDOFF on of_s1 (diagnostic; needs a `read_fixed` query hook) | ~0.05 | descriptive |
| S4 trainer branch | `--cue_src self --name_q same_pass --name_ce`, TF only (no SS, no prefix_kv) | ~0.05 gates | G-FR2 OFF-BIT on of's recipe; G-FR3 with the lower memo; GRADFLOW; MEM_CHECK |
| S5 pilot | of_s1, 2k steps, two arms: P1-TF and P2 (phase 1 as name-ask items, phase 2 on of with popped names) | ~0.5 | the fixed binding rows; Wilson CIs; one lineage |
| S6 (conditional) | SS arm (p 0→1), only if P1-TF NAME-2 >= 0.6 but FREE-D3 < 0.8 | ~0.25 | as S5 |
| Build-B | masks, expand, CGAIN / BEAM, prefix_kv with G-FR1 | — | only when P8 or AP5 registers; TH2 dropped as subsumed |

Total through S5 is about $0.8, well inside the $30 line.
