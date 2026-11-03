# MC3 ENTITY-KEYED MULTI-WRITE: design (wave 4, 2026-11-03 17:20 EST)

Lead design. Read-only. No numbers were computed from stored outputs. Every claim about code cites lines that were read for this design.

## 1. Goal and backlog IDs

**Covers:**
- backlog §13.3 MC3 ENTITY-KEYED MULTI-WRITE;
- the MC2 KEEP-k RECALL eval (as a text-free variant);
- MC1 KEY-GEOMETRY (consumed from §269 part 2, "keys");
- HX4 GATE-USAGE (head dropout and usage monitors);
- §13.0 "memory capacity": addressability, not raw size, is the limit.

**Question.** Can bun's DeltaMem become a **name-addressable table**? The intended table:
- pass 1 writes (entity -> value) at the end of every statement;
- each later ':' write is a by-name delta update of one entry;
- the question entity reads the answer back.

Is that table **used** when the text no longer carries a needed value?

**Why it matters:**
- §13.0 predicts that bun's ':' keys are value- or constant-keyed. §269 keys predicts P(delta VALUE or CONSTANT) ~0.6.
- If so, nothing in bun can retrieve an older quantity by name.
- Retrieval by name is the precondition of D4 BRANCH-DAG (merges), of TH5 DATAFLOW-ALL-WRITE, and of GSM8K-style reuse of earlier quantities.
- It is also the "MEMORY" route of §269 triage.

**NK lesson (Note 262-3).** A name-keyed pathway added beside a working path stayed unused (ablation drop 0.004), and the CTRL arm with the same training did the work. So MC3:
1. makes the table **necessary** on a binding eval (MASK-GIVEN);
2. carries a **same-training, no-table** arm (CTRL-COLON);
3. carries an **entity-shuffled** arm (CTRL-SHUF);
4. gates on an **ablation drop** and a **row-level causal swap**, not on accuracy.

Targets: C1 scope (memory addressing / merges), C6 (table retention), and the precondition for D4 / T-lane. No C2 claim at pilot.

## 2. Design

### 2.1 Today's write and read (code as read)

**Pass t input.** `[EOT] + enc(" " + text + " " + QUESTION(u, S_min(t,d))) + SFX` (inject_aw.py `tok_aw` 76-93; inject_st.py `tok_st` 49-63).

**Write.** `InjectAW.pass_aw` (inject_aw.py 47-56):
- runs blocks 0-11 with `read_fixed` at block 6;
- calls `DM.write_seq` on the single gathered ':' row: `h6[ar,pos][:,None]`, `h[ar,pos][:,None]`, `mask=None`.

**Read.** `read_fixed` (deltamem.py 158-176) reads the pass-start S at every token. The decoder and aux read through `deltas5.fwd_read` (deltas5.py:91, used at inject_ow.py 86-102).

**Head updates in `write_seq` (deltamem.py 179-223), with u = LN_w(h11) because γ = 1:**
- additive: `Sa += φ(k) vᵀ`, φ = elu + 1 > 0;
- delta: `Sd ← α Sd (I − β k kᵀ) + β v kᵀ`, with β ∈ (0, 2) and k, q L2-normalised.

**The T > 1 path already exists:**
- `mask` (B, T): padded rows leave the state unchanged (211-218);
- the FAST_ADD masked einsum (194-200);
- CHUNK_DELTA masked (233-234).

bun runs with `fast_add` (bun_s1 args.json:20).

### 2.2 MC3 mechanism

**Statement-end positions.**
- E_b = the positions of the token "." in item b's pass input before the cue.
- The text is `" ".join(rendered statements) + question` (nested_words.py `_finish` 146-160). Frames, GIVEN ("{A} has {v} {u}.", line 82) and inserts contain exactly one sentence-final ".", and the cue and SFX contain none.
- GPT-2's pre-tokeniser always splits "." from letters and digits, so E_b has exactly one position per statement (gate G-SEG).
- Because attention is causal and pass 1 starts from S = 0, the pass-1 hidden states at E_b do not depend on the cue. They equal a cue-free "scan" of the text, so **no extra forward pass is needed**.

**Pass-1 multi-write.** The new module calls `write_rows(mem, h6[R], h11[R], S0, mask_add, mask_del, s_se)`, a copy of `write_seq` with per-family masks. Its inputs:
- **R** = [E_b in text order] + [':'], padded to the batch max;
- **mask_del** = valid rows (statement ends and ':');
- **mask_add** = the ':' row only (role split, §2.4).

At statement-end rows, the delta heads use a per-head gate s ∈ [0, 1] (`s = clamp(w, 0, 1)`, `w` an nn.Parameter of size 6):

  α_se = 1 − s (1 − α),  β_se = s β

- s = 0 is an exact identity: α_se = 1.0 and β_se = 0.0, so Sd is unchanged bit for bit. This is the equivalence gate and the CTRL-COLON arm.
- s = 1 is the plain delta write. It is the training init of MC3 and SHUF (a non-zero-init pathway, rule 8).
- `w` is a **projected** parameter: after each optimiser step it is clamped into [0, 1]. A plain `clamp` in the forward would leave `w` dead once an update pushed it past a bound.

**Pass detection.**
- A pass is "first" for item b iff its S_b is exactly zero: `S.flatten(1).abs().amax(1) == 0`.
- So every caller that starts from `zero_state` gets the multi-write automatically: training `loss_reg`, the hold far pass (inject_aw.py 186-196), `ws_readout` (134-151), and eval `passes()` (inject_aw_eval.py 55-82).
- The alternative is threading a pass index through about 20 eval parts, each a possible silent omission.
- `m.se_off = True` forces the ':' only write (SE-ABLATE).
- A counter `m.se_first_n` is logged per part (gate G-PATHS).

**Later passes (t ≥ 2) and hold passes.**
- Unchanged: the ':' only write (the `write` path; inject_aw.py 58-59).
- "By-name update" is not hard-wired. The recall loss (§2.3) rewards k_colon(t) aligning with the key of S_t's statement-end entry, so that the delta erase `β (Sd k) kᵀ` replaces that entry.
- This is measured as KEY-ALIGN (§4), not assumed.

**Read by question entity.**
- No new read: `read_fixed` already reads at every position, including the question's name token in the decoder input and the recall query's name token.
- Unlike NK, there is no new read projection (the read was the unused part there).

### 2.3 Training signals (identical in all three arms)

**(a) bun's recipe.** `--init_model` bun_s1 with bun_s1's args: hold penalty, §257 insert mix, aux, decoder CE. Each arm's args differ from bun_s1 only by the flags below, as shown by an args diff.

**(b) Text-free KEEP-k recall (`--rec_w 0.5`).**
- **When:** after pass t ∈ {1..d} ∪ {K}.
- **Queries:** two per item.
  - X_old ~ U(known(t) \ {S_t}), where known(t) = {S_0, G_*, D_*, insert givens} ∪ {S_1..S_t}. Inserted op-statement names are excluded.
  - X_new = S_t (for t ≤ d).
- **Input:** `q = [EOT] + enc(" " + QUESTION(u, X)) + SFX`, 12 tokens, read from S_t through `fwd_read` (no text).
- **Loss:** full-vocabulary CE at ':' against v_X, as in `loss_ask` (inject_aw.py 421-431).
- **Why text-free:** the only source of v_X is S. X_old is the name-addressing test; X_new alone can be solved by "say the latest value".

**(c) MASK-GIVEN (`--mask_given_p 0.5`, per item).**
- In passes t ≥ 2 and in the decoder input, the number token of every given / distractor statement is replaced by " some". It is one token, so lengths, padding and E_b are unchanged.
- Pass 1 sees the full text.
- A pass t with a `*_v` op then gets G_t's value only from the table.
- This is what makes the table **necessary**, which NK lacked.

**(d) Additive-family read dropout (`--hd_drop 0.1`).**
- Per item and per pass (t ≥ 2, plus the decoder), zero the additive half of W_o's input.
- Implemented as a `register_forward_pre_hook` on `mem.wo`. It is the same hook site as `read_mask` (inject_aw_eval.py 1015-1041), and deltamem.py is untouched.
- Purpose: stop bun's additive register from carrying everything, which pushes transport into the delta table. It also gives graded ablation signals.

### 2.4 Additive vs delta roles

- **Statement-end rows write the delta heads only** (`--se_heads delta`).
  - Additive heads are φ > 0 accumulators: a table of n entries reads back as `Σ_i (φ(q)·φ(k_i)) v_i` with all weights positive, so there is no exact retrieval and no by-name overwrite (§12.3).
  - Their norm also grows with every write, which loads the hold penalty's normaliser (§2.5).
- **The ':' row writes both families, as in bun.** The additive heads stay bun's working register; this protects retention.
- `--se_heads both` is a descriptive follow-up, not part of the pilot.

### 2.5 Interaction with the hold penalty (inject_aw.py `hold_step` 159-199)

- **Which passes it touches:** the penalty re-runs `m.write(S_{t-1}, x_{S_d cue})` at t ≥ d+3. S ≠ 0 there, so the ':' only write applies and no statement rows are re-written.
- **Normaliser:** the penalty is `||S_pen − S||² / ||S_{d+2}||²`. The table raises `||S_{d+2}||`, so the effective penalty on ':' updates weakens by about `||S_bun|| / ||S_mc3||` squared.
  - Logged: `norm_guard_by_d` s8 / sd2 (already emitted, 641) plus a new `se_share` = `||Sd_table||` / `||Sd||` after pass 1.
  - If `se_share` > 0.7 and `hold_ratio` rises above bun's by more than 2x, the full stage adds `--hold_inc` x (`se_share` correction). Pre-stated; not in the pilot.
- **Decay of the table:** every ':' write decays the whole delta state by α and erases along k_colon. Over K' 128 the table survives only if hold-pass α → 1, which the penalty rewards because α < 1 makes `S_pen − S ∝ (1 − α) S`.
  - This is a prediction, read by TABLE-RETENTION (§4).

### 2.6 Arms (pilot, seed 1, from bun_s1, `--steps 8000 --stop_at 4000`)

The 4k read follows the FD1 rule (formation markers need ≥ 4k).

| Arm | Statement-end writes | s init / trainable | Everything else |
|---|---|---|---|
| **MC3** | the true pass-1 rows | 1 / yes | (a)-(d) |
| **CTRL-SHUF** | rows from an extra pass-1 forward on the **entity-shuffled** text: one derangement π over the item's names applied to every statement; same token lengths and "." positions; values unchanged. The ':' row comes from the true pass 1 | 1 / yes | (a)-(d) |
| **CTRL-COLON** (shortcut-exclusion arm) | none (s frozen 0: the same code path, bitwise bun at step 0) | 0 / no | (a)-(d) |

How to read the arms:
- **MC3 vs CTRL-COLON:** does the architecture add anything over the same training? This is the §262 CTRL lesson.
- **MC3 vs CTRL-SHUF:** does the *binding* matter, beyond extra writes and compute? The writes happen (the control is exposed), but name ↔ value is wrong.

## 3. Code touch points (all new code in a new module; frozen modules stay byte-identical)

The new module is `boxes/inject_mc3.py`, built by `scratch/mc3_build/mk_inject_mc3.py` following the §234 pattern: `inject_aw.main()` copied with listed edits, each asserted to apply once.

| Where (read) | Change |
|---|---|
| deltamem.py 179-223 `write_seq`; 226-248 `delta_chunk` | `write_rows()` in the new module: a copy with `mask_add` / `mask_del` and the s-gate on statement rows. FAST_ADD einsum masked by `mask_add`; delta loop or `delta_chunk` with (α_se, β_se) substituted per row |
| inject_aw.py 45-63 `InjectAW` | `InjectMC3(InjectAW)` overrides `pass_aw`: zero-state detect; gather R = E_b + ':' rows; SHUF branch (second forward on `x_shuf`, statement rows from it). Keeps the `pos` / ':' assert (55) |
| nk.py 64-91 | precedent for subclassing `InjectAW` and asserting token positions |
| inject_aw.py 76-93 `tok_aw` | `tok_mc3`: adds `se_pos` (asserted), `think_cues_masked` (" some" substitution at passes ≥ 2), `dec_masked`, `x_shuf` rows and `known` (name -> value) |
| inject_aw.py 361-419 `loss_reg` | pass loop 372-393: pass input j ≥ 2 picks the masked row for masked items; recall CE after each pass write; `dec_logits` on `dec_masked` for masked items (404) |
| inject_aw.py 421-431 `loss_ask` | template for `loss_recall` (':' CE, full vocabulary) |
| inject_aw.py 453-465 `mem_train` / `mem_eval` | unchanged call sites; the longest MC_TR items now carry the most statement rows plus the SHUF forward |
| inject_aw.py 529-541 | g_add calibration is skipped under `--init_model` (unchanged) |
| inject_aw_eval.py 55-82 `passes()`, 85-95 `answers_by_kprime` | reused unchanged (zero-detect carries the multi-write); new parts in `boxes/inject_mc3_eval.py` |
| inject_aw_eval.py 1015-1041 | the wo-hook pattern for `--hd_drop` and for the ADD / DEL ablations |
| nested_words.py 128-160 `_statements` / `_finish` | `order`, names and values used to build `known`, the shuffle and the mask (no edit) |

New flags: `--se_write {0, 1}`, `--se_init`, `--se_train`, `--se_heads`, `--se_shuf`, `--rec_w`, `--mask_given_p`, `--hd_drop`.

## 4. Gates (before any spend)

- **G-EQ0 (p = 0, every flag):**
  - Each new flag at 0 / off, with `inject_mc3` vs `inject_aw`, gives **bitwise** identical `log.jsonl` losses over 20 steps on bun_s1 (fp32 and bf16, det wrap).
  - `--args_only` args diff = exactly the module shas.
  - Also smoke the plain path after the edit (memory rule).
- **G-EQ1 (s = 0):**
  - `--se_write 1 --se_init 0` vs bun_s1: S_1..S_8 and decoder logits bitwise on 64 TEST2 items (fp32), including hold-step norms.
  - `write_rows(mask_add = mask_del = mask, s = 1)` equals `DM.write_seq(mask)` bitwise, FAST_ADD on and off, CHUNK_DELTA 0 and 16.
  - fp64: `write_rows` equals an explicit Python loop over rows.
- **G-SEG:** on 20k training draws plus every eval set (including §257 inserts and TEST-PERM):
  - `len(E_b) == len(order) + n_inserts`;
  - `x[E_b] == "."`;
  - decoding the prefix up to E_b[i] gives the first i statements.
- **G-MASK / G-SHUF:**
  - token length is unchanged;
  - only the given-value tokens change (mask) / only name tokens change (shuf);
  - π has no fixed point;
  - `ow_words.solve` on the shuffled text returns the permuted bindings;
  - masked texts are never pass-1 inputs.
- **G-PATHS:** for every eval part, `se_first_n` equals the number of zero-state starts. `part_lp5`'s mean-state edit (inject_aw_eval.py 118-135) is non-zero, which is asserted.
- **Gradflow (`gradflow_check`):** on one batch, nonzero grad for `w` (s), `qkv` k / v rows and `gate` from the recall loss **through the statement rows only**. Detach the ':' row and check the grad is still nonzero. `hd_drop` hooks are inactive in `no_grad` eval.
- **mem_check** at `--d_max 4` with 2 inserts plus the SHUF forward (≤ 21 GiB; `auto_micro`).
- **Usage monitor (rule 8):** per 1k bin, log:
  - s per head;
  - `se_share`;
  - recall acc for X_old / X_new;
  - LATEST-FOLLOW (pred = v_{S_t} on X_old queries with a different value);
  - masked vs clean train acc.

## 5. Pre-registration table

**Eval items:**
- **MG-NEED** = TEST2 d 3-4 items with at least one `*_v` op at t ≥ 2. The masked given is needed by a later pass (stratum by construction; about 0.6 of items, n ≥ 500 per d).
- **MG-NEED-6** = the same at d 6 (descriptive).

All rates carry Wilson 95% CIs. **NEAR** = a point within one CI half-width of its bar; it is claimed neither way.

| ID | Field (K' 8 unless stated) | Type | Bar |
|---|---|---|---|
| B1 USE-RG | MG-NEED d 3-4 acc (masked passes ≥ 2 + decoder) | binding | MC3 ≥ 0.80 |
| B2 OVER-COLON | B1(MC3) − B1(CTRL-COLON), paired items | binding | ≥ 0.20 |
| B3 OVER-SHUF | B1(MC3) − B1(CTRL-SHUF) | binding | ≥ 0.20 |
| B4 SE-ABLATE | B1(MC3) − B1(MC3, `se_off`) | binding | ≥ 0.20 |
| B5 ROW-SWAP (mechanism) | on MG-NEED d 4: replace only the statement-end row of G_t with the row from a donor forward where G_t's value a' ≠ a (same tokens otherwise). Follow = answer equals the donor-recomputed answer | binding | follow ≥ 0.50; the non-needed-row swap (a distractor's) keeps the answer ≥ 0.95 |
| B6 RETAIN | clean TEST2 d 1-4 acc (min over d) | binding | ≥ 0.97 |
| B7 NAME-KEYED | text-free recall at K' = d on TEST2 d 4: chain_old (S_j, j <= d-2) and givens; LATEST-FOLLOW on chain_old queries whose value differs from v_{S_d} | binding | MC3 chain_old >= 0.60 and givens >= 0.60 and LATEST-FOLLOW <= 0.30 (chance 1/11). A position-keyed table can pass B1 via the masked statement's slot in the text; a text-free query cannot |
| D1 | text-free recall of givens / S_j (j ≤ d−2) / S_d at K' = d, 32, 128 (TABLE-RETENTION) | descriptive | — |
| D2 | KEY-NAME score (the §269 keys statistic) on statement-end and ':' keys; KEY-ALIGN = cos(k_colon(t), k_se(S_t's statement)) per delta head | descriptive | prediction: ALIGN > 0.5 in at least 2 heads if by-name updates form |
| D3 | ADD-ABLATE / DEL-ABLATE on B1 and on clean d 1-4 | descriptive | — |
| D4 | clean d 5-8, TEST-PERM d 2-4, add_s / mul_s step WS, MG-NEED-6, LATEST-FOLLOW | descriptive (one seed: no C2 claim) | — |

**Labels:**
- **GO** = B1-B7 all pass.
- **PARK-UNUSED** = B4 < 0.05 or B2 < 0.05 (the NK outcome).
- **KILL-RETAIN** = B6 < 0.95. Redesign with `hd_drop` 0.05 before any other change (rule 3).
- **NOTE** otherwise.

**Full stage (on GO; pre-stated now, registered separately):**
- MC3 and CTRL-COLON, 3 seeds each, 8k. Seed 1 resumes the pilot ckpt; seeds 2 and 3 are data-order seeds from bun_s1 / bun_s2, disclosed as "one lineage".
- Per seed: B1 ≥ 0.85, B2 / B4 / B5 as above, B6, and C6 (K' 128 hold at d 1-6 as bun).
- Any d 6-7 row is reported as a per-seed margin vs bun − 0.05 with NEAR-SEED (σ_seed 0.090, Note 269-2).
- A C2 claim would additionally need B5 at d 6-7 (rule 2).

**OC (outcome priors):**

| Outcome | Prior | Reason |
|---|---|---|
| GO | ~0.30 | — |
| PARK-UNUSED | ~0.30 | CTRL-COLON packs 1-3 givens into the pass-1 ':' write: 6 delta heads x rank 1, so this is possible |
| KILL-RETAIN | ~0.15 | — |
| NOTE | ~0.25 | — |

- P(B5 follow ≥ 0.5 \| GO) ~0.7.
- P(CTRL-SHUF B1 < CTRL-COLON B1) ~0.5 (misleading table). That is why B2 and B3 are both binding.

**Censoring:**
- A part that crashes gets VOID-RERUN once, as its own line (each part is its own output line).
- MG-NEED excludes items whose masked given is unused (stratum definition, stated before data).
- B5 items need pass ≤ t−1 WS correct in the un-swapped run. Excluded items are counted and reported.
- If the count is < 200, B5 is VOID-LOWN, not failed.
- Value coincidences (a' = recomputed a) are excluded at donor construction.

## 6. Scorer sketch (`theory/score_mc3.py`)

**Fields read, by name:**
- `eval_mc3.json` parts:
  - `mg.by_d[d].acc / n / items.pred`;
  - `mg_se_off.by_d[d].acc`;
  - `rowswap.d4.{follow, keep_nonneeded, n_cond, n_excl}`;
  - `tfrec.by_k[K'].{given, chain_old, chain_new, latest_follow}`;
  - `keys.{name_score, align}`;
  - `abl.{add, del}`.
- `eval.json` `acc_by_d` for RETAIN.
- `train_summary.json`.
- The usage log fields in `log.jsonl` bins: `se_s`, `se_share`, `rec_old`, `rec_new`.

Pairing for B2 / B3 is by `sig`, asserted equal across arms.

**Labels:** GO / PARK-UNUSED / KILL-RETAIN / NOTE / VOID-LOWN / VOID-RERUN, with NEAR flags. Wilson and paired-difference statistics come from `theory/statlib.py` if ST1 has landed, else a local stdlib copy.

**Fixtures (10):**
- one per label;
- NEAR on B1;
- NEAR on B2;
- a sig mismatch (must raise);
- a missing `rowswap` part (VOID-RERUN);
- `n_cond` 150 (VOID-LOWN);
- `latest_follow` present but `chain_old` absent (must raise).

Run twice; byte-identical output.

## 7. Cost (RTX 3090, ~$0.2/h)

**Basis:** bun runs at ~1.7 s/step (Note 268-1 rates). MC3 extras:
- statement rows ~+1%;
- recall 2 x 12-token forwards x ≤ 5 passes ~+5%;
- SHUF forward +8% (SHUF arm only).

That gives ~1.8-1.9 s/step.

| Stage | Hours | $ |
|---|---|---|
| Gates (local CPU / 4060 smoke, bitwise; Vast smoke for mem_check) | — | ~0.1 |
| Pilot: 3 arms x 4k steps | 3 x ~2.1 | ~1.3 |
| Pilot evals (MG, se_off, rowswap, tfrec, keys, abl, acc d 1-8; ~0.1 h each) | — | ~0.1 |
| **Pilot total** | — | **~1.5 (guard $2.5)** |
| Full (on GO): 5 more cells x 8k (seed 1 resumes 4k) | — | ~4.6 |
| Full evals | — | ~0.5 |
| **Full total** | — | **~5 (guard $8)** |

## 8. Dependencies and order

- **§269 keys (MC1).**
  - If delta keys read **NAME**, bun's ':' path is already addressable. MC3 then reduces to the statement-end writes plus MASK-GIVEN, and CTRL-COLON is expected to be strong; the prior on B2 drops to ~0.15. Re-price before registering.
  - If **VALUE / CONSTANT**, as above.
- **MC2 stage 0** (text-free KEEP-k on bun, $0.1, pre-stated): gives bun's LATEST-FOLLOW and chain_old baseline. It shares the `tfrec` part code; build it first.
- **§263:** if ORDER-FREE passes, the full stage inits from of_s1 instead (name-based selection already trained). The pilot stays on bun_s1, so it is not blocked.
- **I5 triage** = MEMORY raises MC3's priority; SELECTION lowers it.

## 9. Literature (each id fetched from arxiv.org/abs on 2026-11-03; titles matched)

| Id | Paper | Use here |
|---|---|---|
| 2011.07831 | Schlag et al., *Learning Associative Inference Using Fast Weight Memory* | per-input writes |
| 2102.11174 | Schlag et al., *Linear Transformers Are Secretly Fast Weight Programmers* | the delta rule; capacity |
| 2406.06484 | Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* | `delta_chunk` |
| 2412.06464 | Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule* | α-gated erase |
| 2312.04927 | Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models* | MQAR / KEEP-k |
| 2402.18668 | Arora et al., *Simple linear attention language models balance the recall-throughput tradeoff* | linear-state recall limits |
| 2305.02363 | Kim & Schuster, *Entity Tracking in Language Models* | entity-state tracking |
| 2310.17191 | Feng & Steinhardt, *How do Language Models Bind Entities in Context?* | binding IDs |
| 1612.03969 | Henaff et al., *Tracking the World State with Recurrent Entity Networks* | per-sentence entity slots (closest ancestor) |
| 2004.13342 | Zhou et al., *Scheduled DropHead: A Regularization Method for Transformer Models* | `--hd_drop` |

## 10. RED-TEAM

**Status: PENDING the required separate red-team sub-agent.** Two spawn attempts were refused (the session's 20-subagent concurrency limit). The items below are the lead's own pre-red-team pass. They do **not** satisfy rule 8 of the lead rules: the caller must run the separate red team on this file before registration.

1. **MAJOR: a position-keyed table passes B1.**
   - The masked statement keeps its slot in the text, so pass t can find "Kim has some eggs." and query the table by position.
   - Fixed by B7 (text-free recall plus LATEST-FOLLOW, binding).
   - Descriptive extra: MG-NEED under the §263 independent-only given/distractor shuffle.
2. **MAJOR: CTRL-SHUF can be worse than "no table"** (a misleading memory). So B2 (vs CTRL-COLON) is binding alongside B3.
   - SHUF's extra forward also gives it ~8% more compute. That is disclosed, and it favours SHUF, so the comparison is conservative.
3. **MAJOR: CTRL-COLON may pack the 1-3 needed givens into the pass-1 ':' write** (one rank-1 update per delta head, 6 heads).
   - This is the NK outcome, and it is the PARK branch, not a bug.
   - Cheap pre-check: MC2 stage 0 on bun (below).
4. **MINOR: s at a projection bound.** Fixed by projecting `w` after each step (§2.2).
5. **MINOR: zero-state detection.** An eval edit that zeroes S after pass t ≥ 1 would re-trigger the multi-write.
   - G-PATHS asserts the detection count per part.
   - No current part zeroes S: inject_aw_eval.py has no `zeros_like(S)`, and lp5 substitutes a mean, which is non-zero.
6. **MINOR: the decoder last hop (Note 264-1).** Under MASK-GIVEN the decoder must read G_d from the table when op d is `*_v`.
   - Report B1 split by "needed given at t ≤ d−1" vs "at t = d", so that table use by the passes and by the decoder are separable.
7. **MINOR: value coincidence (1/11).**
   - ROW-SWAP excludes donors whose recomputed answer equals the original.
   - B1 chance is about 0.09, so the bars are far above it.
8. **MINOR: tokenisation of ".".** Checked: frames ending in "{A}'s." and insert texts (insert_render.py 26-28, 37, 115-248 use the same GIVEN / FRAMES) split "." off under GPT-2's pre-tokeniser. G-SEG asserts it on every set.

**Cheapest falsifiers (run before the pilot):**
- **MC2 stage 0 text-free KEEP-k on bun_s1** (~$0.05, eval-only, pre-stated). If chain_old is already ≥ 0.6 with LATEST-FOLLOW ≤ 0.3, the ':' path is already addressable: MC3 is re-priced toward the statement-end rows only, and B2's prior drops.
- **§269 keys** (already registered): NAME on the delta heads gives the same re-pricing.
- **1k-step MASK-GIVEN-only probe on CTRL-COLON** (~$0.1). If masked accuracy on MG-NEED reaches ≥ 0.8 without any table, MC3's necessity argument fails and the pilot is not run.
