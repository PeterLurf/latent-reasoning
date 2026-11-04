Status: completed 2026-11-04 by Theory's agent; not red-teamed yet

# MT JOINT: one looped model, two task families, no task flag (wave-4 design 10, v1; supersedes the truncated v0 "MT1 JOINT-4 / MT4 / MT2" draft at bdb925d7)

Theory design agent, 2026-11-04 02:19 EST. Read-only: no training, no computation on stored run outputs. The only files read under `boxes/runs_fsq/` were `of_s1/args.json` (for the args diff) and the directory listing. One CPU script ran on its own simulated draws: `mt_oc.py` (numpy only, `python -I`, seed 27460) in the session scratchpad, with output `mt_oc_out.txt`. Its numbers are tagged DATA-SIM. Before registration they move to `docs/research_mt_2026-11-04/` (§13).

## 0. Decisions in one paragraph

- **Families: WP + SO.**
  - WP = word programs, the REG stream that of_s1 was trained on (§263).
  - SO = shuffled objects, exactly as §271 N1 builds it.
  - KIN (kinship-lite) and PQ (ProsQA) from v0 are deferred (§2.3). Rule: a family enters a joint model only after its own single-task registration has passed its mechanism gate. This is the §174 precedent (§174 was "read only if the single-task references exist"). Neither KIN nor PQ has such a registration, and neither has a generator.
- **One design.** Both families use the same trunk, DeltaMem, K 8, aux / hold / readout code and the same hyper-parameters. Every pass input is `[EOT] + text + cue_t + " The answer is:"`, where cue_t = "the item's own question, asked of step min(t, d)". The step handle is whatever the item's text exposes: an entity name in WP, a swap label in SO. These are the frozen cues of the two single-task references, so the per-family comparisons stay clean. A cue shared across families is not used: §260 shows the plan-free cue loses WP depth (§2.2).
- **Init: of_s1** (the same file as §271's LOOP-SI).
- **Exposure is matched by construction.**
  - The run is 16k steps, block-balanced, with exactly 8k steps per family.
  - Every per-step RNG stream is family-local, so the SO half's batches are bitwise LOOP-SI s_k's batches.
  - The cosine-LR exposure of the SO steps matches LOOP-SI's within 0.8% (J-3, PROVED and computed).
- **Staging: no reading before §271.**
  - Stage 1 launches only after §271 S1 GO. It is a single 8k-step pilot (4k SO steps), gated on §271's own four S1 GO rows plus WP retention rows.
  - Stage 2 (3 seeds × 16k) launches only after §271 S2 has landed and been scored.
- **Cost:** expected ≈ $7.9 if every stage runs. Guards: $2.0 / $7.5 / $1.5.

## 1. Question, claim, scope

**Question.** Can the looped GPT-2 + DeltaMem model, with one set of weights and no task token, serve two structurally different serial families? WP is a plan-given dataflow lookup over numbers. SO is pointer tracking over people under transpositions. For each family, we test whether the model keeps:
- its in-range accuracy;
- its per-pass C1 causal gates (NOT-DECODER, SWAP, LP5 / K'-curve);
- its depth behaviour,

each relative to the matched single-task model. And does either family help or hurt the other?

**Why now (north star).**
- The user's standing direction (10-05) is "work towards a task-agnostic method". Every C1 result so far is one model per task.
- §174 (two-task INJECT, 10-09) was registered but never launched.
- The priority order is C1 > C6 > C4 = C2 > C3 > C5 > C7. This design binds C1 per family, carries C6 (hold) per family, and makes C2 claims only at ≥ 3 seeds.

**Claim if MT-PASS (§6.3).** "One looped pretrained LM, trained jointly on two serial task families with no task flag, keeps each family's in-range accuracy and per-pass causal gates, and its depth extrapolation, within the stated margins of the matched single-task models."

**Scope, written into every label:**
- one lineage (of_s1 ← bun_s1 ← … ← s233_askname_s1; 3 data-order seeds; Audit rule 2, until X7 / §268);
- WP plan-given (the cue names S_t), REG frames 0-5, op order free (§263), givens / S_0 in trained placement (Note 263-3);
- SO chain-ordered swap sentences with a step-index cue (§271's scope);
- hand-built per-pass aux (C4 not claimed);
- d ≤ 4 / m ≤ 4 trained.

**Not claimed.** A task-agnostic *cue*: the cue is family-rendered by one rule (§2.2). Transfer beyond the two families. Shared versus partitioned memory content: this is descriptive only (§6.4 D4).

## 2. Families, cue, readout (one design, same code and settings)

### 2.1 What is shared and what is family data

| component | WP | SO | shared? |
|---|---|---|---|
| trunk / DeltaMem / W_o / K 8 / write source ':' | of_s1 | of_s1 | yes, one parameter set |
| pass input | `[EOT] + text + " " + cue_t (+ " The answer is:")` | `[EOT] + text + cue_t + " The answer is:"` | one rule (write_src answer) |
| cue_t | `QUESTION(u, S_min(t,d))` = "How many {u} does {A} have?" (`nested_words.py:83`, `inject_st.py:44-46`) | `cue_si` = " After swap {L(min(t,m))}, who {verb} {obj}?" (`so_words.py:91`, worktree) | rule shared; the text fills the handle |
| aux target, pass t | R_step[t] "<S_t> has <v_t> <u>." | R_step[t] "After swap L(t), <S_t> {verb} {obj}." (`so_words.py:104`) | yes (full-vocab CE at vidx, `AUX_W[t-1]`) |
| answer CE | full vocabulary on VOC_f[ans] (`inject_aw.py:363`) | full vocabulary on VOC_f[ans] (`inject_so.py:294`, worktree) | yes (VOC_f per step, §3) |
| eval readout | argmax over NUM_IDS (11 numbers) | argmax over the item's people_ids (`so_tok.pick`) | one rule: argmax over the item's candidate set |
| hold penalty, hold_inc 1.6 | passes d+3..K | passes m+3..K | yes (`hold_step`, `inject_aw.py:159`, "one d per step") |
| training law extras | of's: perm_p 0.5 perm-then-insert, ask_p 0.25, distract_p 1.0 (s256, chain, givop), mul_s_p 0 | none (§271: untouched swaps are the holds) | data of each family, unchanged from its single-task reference |

Code citations marked "worktree" refer to `scratch/_tmp/wt_s272/boxes/`. That is where Experiments' uncommitted §271 build sits (`so_words.py`, `so_tok.py`, `inject_so.py`, `inject_so_eval.py`; checked 02:1x EST). This design depends on the **committed, Audit-G-D'd** form of those files. It is not built until they land (§8 G0).

### 2.2 Why a family-rendered cue and not one shared string (decision)

Three options were weighed:
- **(a) PT (adopted, binding):** each family keeps its frozen single-task cue under the shared rule "re-ask the item's question about step t".
  - The joint model is then compared against single-task models that saw *exactly* the same pass inputs, so retention and interference are interpretable.
  - The rule is task-agnostic in form. A new family supplies text and a step handle, not code.
- **(b) GENERIC (shared constant cue " <question> What is the next value?") was rejected as binding.**
  - DATA: §260 (Note 260-5) shows plan-free WP stops tracking at depth (d 7 0.25 / 0.07; H-EXTEND 0.08 / 0.02).
  - A shared generic cue would therefore fail WP retention for a reason already known, not because of the joint training.
  - §271 S3 GENERIC tests the plan-free cue on SO single-task. Only if both single-task plan-free lines pass does a JOINT-GENERIC arm become informative. Deferred (§5, stage 3b).
- **(c) Index cues for both families** (label WP's op sentences "Step 3: …") was rejected. It changes WP's rendering against every WP reference (TEST2, TEST-PERM, LATE-PREMISE). Under §263's op-order freedom, a label would also be a text position, not a dataflow step.

**Family identifiability (PROVED by construction).** The first cue token is " How" in WP and " After" in SO (index cue). So a deterministic one-token rule identifies the family on every training item, and "no task flag" means the text identifies the task, as in §174. G2 asserts this on 20k items per family.

### 2.3 Families deferred from v0, with reasons

- **KIN (kinship-lite).**
  - It tests value-keyed retrieval: the fact read at step t is keyed by v_{t-1}. No trained model in the project does this.
  - Adding it to a joint run before a single-task KIN result exists would read an unformed mechanism as "interference".
  - Route: its own single-task registration first, using v0's generator spec (kept in Appendix A). Then a JOINT-3 extension under this document's rules.
- **PQ (ProsQA).**
  - (i) Cost: ~500-token items, v0 estimated 4-7 s/step, 3-4× a WP / SO step.
  - (ii) Validity: NORTH_STAR records a proximity heuristic scoring 0.91 on standard ProsQA (§57). No-CoT GPT-2 already reads 0.63 at depth 3 on p8 data (THEORY §168, line 10441). So the C1 gates would need the shortcut-free set and a think-gain bar.
  - (iii) No single-task loop result on PQ exists in this lineage.
  - Route: as KIN. Wave-4 doc 05 (ProsQA OOD port) is the natural single-task home.
- **MT2 (leave-one-family-out)** needs ≥ 3 families. It is dropped from this registration.

## 3. Training recipe and exact args diff

### 3.1 Joint draw (block-balanced; family-local RNG)

- Steps come in blocks of two, (2j−1, 2j). Each block holds one WP step and one SO step, in an order drawn by `rng_f = default_rng((seed, 27461, j))`.
- Batches are single-family. This keeps the one-d-per-step law (`inject_aw.py:552`; worktree `inject_so.py:503`) and `hold_step`'s per-chunk assumption (`inject_aw.py:159-161`).
- Each family f keeps its own step counter s_f. All of its per-step randomness is drawn from family-local generators:
  - `rng_d_f`: d / m per step;
  - `rng_far_f`: hold T_far per step;
  - the per-item seeded coins (seed, CONST, **s_f**, k), as in `draw_items` (`inject_aw.py:316-346`).
- **SO half.** rng_d_so is seeded as LOOP-SI's rng_d (`seed + SEED_D`, worktree `inject_so.py:149`), and the SO stream is `SO.Stream(SO.SEED_TRAIN, …)` (worktree `inject_so.py:159-164`). So the j-th SO batch of jt_sk is **bitwise** the j-th training batch of §271's loop_si_sk (G1b).
- **WP half.** This is of's law, but with data-order seed `seed + 100` for rng_d_wp, rng_far_wp and the coins. The reason: `OW.Stream` is seeded by the constant `OW.SEED_TRAIN` (`ow_words.py:147-159`), so seed 1 would replay of_s1's own training items (a second epoch). G1c reports the overlap of item signatures with a CPU replay of of_s1's law, first 1k WP batches.
- **Exact exposure (PROVED).** At step 2N, each family has had exactly N steps × 64 items. NASK = 16 ask items apply on WP steps only (`inject_aw.py:266-268`), so WP steps carry 48 chain + 16 ask items, as of_s1 did. SO steps carry 64 items, as LOOP-SI.
- **VOC_f.** The value vocabulary is bound per step: NUM_IDS on WP steps, VAL_IDS on SO steps.
  - Today it is one run-level variable (worktree `inject_so.py:148`), used at its loss / acc sites (worktree 281, 294, 311, 322, 350, 357).
  - Under `--joint` it becomes `VOCS[f]`. With `--joint` off, it is the same object as now (G1a bitwise).

### 3.2 Args diff vs `of_s1/args.json` (WP reference) and vs LOOP-SI (SO reference)

| arg | of_s1 (read today) | LOOP-SI (§271 / N1_v1 §2) | jt_sk (this design) | why |
|---|---|---|---|---|
| init_model / sha | bun_s1_init/model.pt / 1ad3517b… | of_s1_init/model.pt / sha at copy | **= LOOP-SI's file and sha** | lineage head; same start as the SO reference (N1_v1 §2) |
| src | ow | so | ow (WP) + so (SO) via `--joint wp,so` | the two families |
| cue | step | index | step on WP steps; `--so_cue index` on SO steps | each family's frozen cue (§2.2) |
| steps | 8000 | 8000 | **16000** | exact per-family exposure 8k (J-3); disclosed second change |
| stop_at | 0 | 4000 (pilot s1) | 8000 (stage-1 pilot, s1) | the pilot has 4k SO steps = §271 S1 pilot exposure |
| perm_p / ask_p / ask_perm_p / distract_p / distract_mix / mul_s_p | 0.5 / 0.25 / 0.5 / 1.0 / s256,chain,givop / 0 | 0 / 0 / – / 0 / "" / (inert) | of's values, **WP steps only** (gated; worktree `inject_so.py:138-139`'s "OW-only off under src so" assert becomes "off on SO steps") | each family's own law |
| so_offset / so_n / so_modes / so_names_excl | – | 1 / 3,4,5 / num,ord / Eve,Bob | same as LOOP-SI, **SO steps only** | §271 F3 / F12a |
| perm_scope | op | op | op | §270 pending; one change at a time |
| seed | 1 | 1 / 2 / 3 | 1 / 2 / 3 (WP half: data seed +100) | 3 data-order seeds; §3.1 |
| grad_cos_every (new, default 0) | – | – | 500 | D1 diagnostic; no RNG / grad side effect (G4) |
| eval_parts | acc,test_perm,insert_ho,swap_kd,lp5h,stab | the §271 list (N1_v1 §3.6) | the union, run through per-family **eval views** (§3.3) | each family scored by its own scorer |
| **unchanged (all three)** | think_k 8, d_max 4, bs 64, lr 1e-4, micro 2 (pinned; G5 headroom), aux_step 1.0, aux_fade 48000,48000, aux_reweight, aux_target value, fast_add, cache_lower, write_src answer, hold_inc 1.6, hold_start 500, hold_far 16 / 16-40, hold_log, init_trunk s233 (sha c00f7ec0), mem_budget 21, eval_bs 250, ckpt_every 1000, keep_final_ckpt false (s1 pilot: true, for the resume) | | | |

The single-change rule is satisfied *per family* by construction:
- Relative to LOOP-SI s_k, the SO half differs only in being interleaved with WP steps on the same weights, and in the cosine horizon. The LR exposure is matched (J-3).
- Relative to of_s1, the WP half is an of-law continuation from of_s1 itself, interleaved with SO.
- `args_diff` (Audit) lists {joint, so_cue, steps, stop_at, grad_cos_every, eval_parts, out, the WP data-seed offset} against both references.

### 3.3 Per-family eval views (no eval-module edits)

- Both eval modules read the cue and set shas from `<out>/args.json`: `inject_aw_eval.run` at line 1297-1298; worktree `inject_so_eval.run` at 633-634 and `cue_of` at 100-104.
- After training, the joint trainer writes `<out>/fam_wp/args.json` (src ow, cue step, the WP set shas) and `<out>/fam_so/args.json` (src so, cue index, so_set_shas).
- It then calls `inject_aw_eval.run(<out>/fam_wp, …, model=(m, g))` and `inject_so_eval.run(<out>/fam_so, …, model=(m, g))`. The `model=` argument already exists in both.
- `score_fp`-style WP rows and `score_n1.seed_loop` (`theory/score_n1.py:298`) then read the views unchanged.
- Both eval modules stay byte-identical and sha-asserted (G1a).

### 3.4 New logging (descriptive; changes nothing that is trained)

- `family_n`: steps per family in each 1k bin. Asserted equal ±1.
- `loss_f` / `aux_f` / `acc_f`: per-family losses and accuracies in each 1k bin.
- `leak_f`: in each 1k bin, the share of answer positions whose full-vocabulary argmax lies in the *other* family's VOC and outside the item's own (J-6).
- `grad_cos`, every 500 steps:
  - Probe batches: two fixed batches of 32, one per family, seeded (27462, f) and excluded from training; plain loss_items; hold penalty off; no optimiser step.
  - Logged: ‖g_f‖; cos(g_WP, g_SO) over all trainable parameters and per module group {blocks 0-5, blocks 6-11, DeltaMem write (qkv, gate), W_o / read}; and the Adam-preconditioned cosine cos(g_WP ⊘ (√v̂ + ε), g_SO), with v̂ from the optimiser's `exp_avg_sq`.
  - The params' `.grad` is restored to None, and the torch / numpy RNG states are saved and restored (G4).

## 4. Mechanism and mathematical backing

Notation: θ = shared parameters, L_f(θ) = family-f loss, g_f = ∇L_f, η_s = LR at step s, P = Adam's diagonal preconditioner.

- **J-1 (PROVED, by construction) Families interact only through weights and optimiser state.**
  - DeltaMem's state S starts at 0 for every item, and every batch holds one family.
  - So no WP state is ever read by an SO pass, or the reverse.
  - Every cross-family effect is a parameter effect: shared θ, plus Adam's shared moment estimates.
  - Consequences: "shared vs partitioned memory" cannot be read from training behaviour. It can only be probed by out-of-distribution transplants (D4). And interference must show up as a change in the per-family functions that each single-task reference also computes.
- **J-2 (DERIVED-UNDER-ASSUMPTIONS) First-order interference.**
  - Assumptions: small steps (first-order Taylor), and P approximately constant over a window.
  - Then one step on family B changes L_A by ΔL_A = −η gᴬᵀ P gᴮ + O(η²).
  - Over a block (one WP and one SO step), the expected change of L_A, relative to a single-task run that takes its A-step only, is −η gᴬᵀ P gᴮ.
  - Prediction: the sign of the window-median *preconditioned* cosine c̃ = cos(P^{1/2} gᴬ, P^{1/2} gᴮ) predicts the sign of the per-family transfer Δ_f.
    - c̃ < 0 (CONFLICT): joint is behind single-task at matched exposure.
    - c̃ > 0 (ALIGNED): joint is ahead.
    - |c̃| small: no first-order effect, so retention is expected.
  - Limits stated:
    - Adam's v̂ mixes both families' gradients, so P is not the single-task P (second-order mismatch, not bounded here).
    - The hold penalty's gradients are excluded from the probe.
    - First-order effects are not long-horizon effects.
  - This is the gradient-affinity reading of Fifty et al. (2109.04617). PCGrad (2001.06782) is the remedy if CONFLICT holds (routing only, not in this registration).
- **J-3 (PROVED; computed) Exposure and LR matching.**
  - lr_at (`boxes/fsq120.py:740-743`) = lr·min(1, s/100)·½(1 + cos(πs/T)).
  - Block balance gives exactly N SO steps by step 2N.
  - Summed over the SO steps, Σ lr is: joint T 16000 to step 8000 = 3248.4; LOOP-SI T 8000 to step 4000 = 3223.5 (ratio 1.0077). Full runs: 3975.0 vs 3950.0 (ratio 1.0063). Units are lr/1e-4. The difference is entirely the 100-step warm-up counted once per run.
  - The mean LR per SO step is 0.812 vs 0.806 (pilot) and 0.497 vs 0.494 (full).
  - So "same examples at the same LR exposure" holds to < 1%. What differs is that the SO steps of the joint run are interleaved with WP steps, which is the treatment.
- **J-4 (PROVED) The question-blind floor differs between families.**
  - In SO the asked object o* is uniform over the item's n objects, independent of the swaps (§271 item law, N1_v1 §3.1). The final map h_m is a bijection.
  - So any policy that does not read o* names h_m(o*) on exactly one of the n possible o* values. P(correct) = 1/n exactly. Expected over n ~ U{3,4,5}, this is (1/3 + 1/4 + 1/5)/3 = 0.261.
  - In WP's chain items, the question-blind "follow the readers, stop at no reader" rule is correct on 100% of training items (backlog §15 QB0; a code-reading finding). The measurement script `theory/read_qb0.py` exists but is uncommitted; its result is not read here.
  - So the joint law is the first law in this lineage where half the steps penalise question-blindness.
  - CONJECTURE (descriptive, D6): this pressure transfers to WP's question use. It is testable only if lane-07's retarget / twin-pair part is built; not binding.
- **J-5 (PROVED) Answer types are separable at the readout.**
  - NUM_IDS (" 0" … " 10") and VAL_IDS (38 name tokens, worktree `so_tok.py:19-21`) are disjoint token sets.
  - Each eval readout is an argmax over the item's own candidate set, so a cross-family type error is invisible to accuracy by construction.
  - Hence the separate `leak_f` log (§3.4). It measures what the restricted readout hides.
- **J-6 (CONJECTURE) Name-token double role, the likeliest interference channel.**
  - WP entity names come from `NAMES` (`nested_words.py:33-34`), and SO's people are `NAMES_SO` ⊂ `NAMES` (worktree `so_words.py:25`).
  - In WP a name in the cue is the *selection key* and the value is a number. of_s1's NAME-SWAP follow is 1.000 (Note 263-3): it selects computed entities by name.
  - In SO a name is the *value*, and the cue's key is a swap label.
  - The shared ':' write must therefore learn "name in cue → read that entity's state" (WP) beside "write the current holder's name" (SO).
  - Predictions:
    - (i) `leak_SO` (numbers predicted on SO items) is > 0.05 in the first 1k bin and < 0.01 by the end. `leak_WP` stays < 0.01 throughout.
    - (ii) If interference appears, it is concentrated in WP's NAME-SWAP / SWAP4 and SO's SWAP-MID rows, not in in-range accuracy.
- **J-7 (DATA + DERIVED) Stage-1 gate power.**
  - The SO GO rows are §271's. Note 271-2 gives P(GO | mechanism) 0.991 at the in-range seed spread σ 0.011, and 0.534 at σ 0.090 (conservative).
  - P(GO | SO memory unused) = 0 by the Note 271-2 argument: the decoder input carries no pass count, so K'2 predictions equal K'8 predictions item by item. That argument is per item and holds unchanged in the joint model.
  - The WP GO rows: DATA-SIM, `mt_oc.py`, normal approximation. Assumptions: TEST2 d 4 true 0.993, bar 0.97, σ 0.011, n 1,000. Pass probability ≈ 0.95. With the 0.98 final bar, the in-range block passes 0.83.

## 5. Stages (cheapest decisive first)

| stage | trigger | cells | decides | expected $ | guard |
|---|---|---|---|---|---|
| 0 | §271 G0 committed | build + gates G0-G6 (§8), CPU + one host slice billed to stage 1 | buildable, bitwise-safe | 0 | – |
| **1** | **§271 S1 scored GO** (or GO after its EXTEND) | jt_s1 `--steps 16000 --stop_at 8000` (4k SO + 4k WP steps), gate-eval subset on ckpt_8000 | does joint training keep both mechanisms at matched SO exposure? | 1.0 | **2.0** |
| 2 | stage-1 GO **and** §271 S2 landed + scored | jt_s1 resume 8k→16k (new dir); jt_s2, jt_s3 (16k); full per-family batteries; MT4-lite eval parts (D4 / D5) | binding labels (§6) | 5.4 (+2.0 rerun reserve) | **7.5** |
| 3a | R-WP-DEPTH FAIL or NEAR-SEED | ctl_wp_s1: of_s1 + 8k of-law WP-only steps, WP data seed 101 (bitwise the jt_s1 WP half's batches) | attributes a WP depth loss to interference vs continued-training drift | 1.0 | **1.5** |
| 3b | not registered here | JOINT-GENERIC (both families, plan-free cue) only if §271 S3 GENERIC passes C2-1 and a plan-free WP line passes C2; JOINT-3 / 4 after KIN / PQ single-task passes | task-agnostic cue / more families | – | own registration |

**Stage-1 decision rule (pilot, jt_s1 at step 8000; every row on 500-1,000 fixed TEST items; Wilson CI + NEAR shown; letters on raw values).**
- SO rows (§271's S1 GO rows verbatim):
  - L1p: NUM n 5 m 1-4 ≥ 0.90 each;
  - CWS_t ≥ 0.90 at passes 2-4, n 5 m 4;
  - M3p: SWAP-MID (5, 4, t 2) follow_cf ≥ 0.50 and ≥ MEAN + 0.30 (1,000 pairs);
  - M4p: K'2 on the S_2 ≠ S_4 stratum, acc vs S_2 ≥ 0.80 and vs S_4 ≤ 0.05.
- WP rows:
  - TEST2 d 1-4 ≥ 0.97 each;
  - TEST-PERM d 2-4 ≥ 0.90;
  - NOT-DECODER: d 4 at K'2 ≤ 0.20;
  - SWAP4: follow_U ≥ 0.50 and ≥ floor + 0.30.
- **GO** = all rows pass. Seeds 2-3 launch once §271 S2 is scored.
- **PARK (stop the line)** = any SO L1p < 0.50, or WP TEST2 d 4 < 0.90 (gross interference). The note reads JOINT-INTERFERES-EARLY with the failing rows. Stage 2 is voided, and the next registration is MT3 (gradient surgery, J-2), only if D1 reads CONFLICT.
- **EXTEND** = otherwise, on a miss. jt_s1 continues to 16k (this is stage 2's s1 resume, billed to stage 2) and is re-gated on the same rows at 8k SO exposure. Seeds 2-3 launch only on GO at 16k.
- **Descriptive at stage 1 (pre-stated):** T-SO-early = jt_s1 pilot minus §271's loop_si_s1 pilot on the four SO rows. Same TEST items, so paired: McNemar p beside each CI. Plus `leak_f`, `grad_cos` and `family_n`.
- **Report text (Note 271-3 / Audit):** in-range SO accuracy at m ≤ 3 is not tracking evidence (shortcut floors f 0.96-1.0 on MOVED items). Tracking evidence comes only from CWS, SWAP-MID and K'2.

**Dependency on §271, stated.**
- Stage 1 needs §271 S1 GO, read from score_n1's s1 output. If §271 PARKs, this design is void: no single-task SO mechanism exists to retain.
- If §271's DIRECT gate fires (DEPTH-SPLIT), stage 1 still runs. That gate concerns the one-pass control, not the loop.
- Stage 2 needs §271 S2's three LOOP-SI seeds landed and scored, because R-SO-DEPTH is a contrast against them.
- No joint output is read, and no joint cell launches, before the corresponding §271 reading exists. So this design cannot pre-empt §271's single-task result.
- It also uses none of §271's control arms (DIRECT-SI / DIRECT-OF), only its loop references.

## 6. Criteria

All criteria follow Audit standing rules 1-5:
- Wilson 95% CI on every proportion-bar cell;
- NEAR when the bar lies within 2 SE of the reading;
- letters on raw values;
- per-seed margins, the seed mean ± SD and NEAR-SEED (σ_seed 0.090, Note 269-2) on every depth row;
- "one lineage" in every label.

### 6.1 Per-seed binding: JOINT-C1(seed), read at 16k

| # | row | cell | bar |
|---|---|---|---|
| W1 | WP in-range | TEST2 d 1-4, K' 8 | ≥ 0.98 each |
| W2 | WP order | TEST-PERM d 2-4 ≥ 0.90; LATE-PREMISE d 2-4 ≥ 0.85 | as §263 / §270 |
| W3 | WP NOT-DECODER (C1) | d 4 at K' 2 | ≤ 0.20 |
| W4 | WP SWAP4 (C1) | swap_kd d 4, SWAP-MID t 4 | follow_U ≥ 0.50 and ≥ floor + 0.30 |
| W5 | WP LP5-D4 (C1) | lp5h d 4 | intact − mean_at_t ≥ 0.30 (raw difference, as score_of) |
| W6 | WP C6 | long_k128 d 1-6 | K' 64 / 128 ≥ K' 8 − 0.02 |
| S1 | SO in-range | NUM n 5 m 1-4 (L1) | ≥ 0.95 each |
| S2 | SO mechanism in range (C1) | CWS_t t 2-4 at m 4 ≥ 0.90; M3 SWAP-MID (5, 4, t 2); M4 NOT-DECODER (K'2 → S_2 ≥ 0.80, → S_4 ≤ 0.05) | §271 rows |
| S3 | SO K'-curve (C1) | KC at n 5 m 8 MOVED: K' 0 / 1 ≤ f(5,8) + 0.05 (f = 0.316, frozen, Note 271-3); K' 4 ≤ 0.302; K'8 − K'4 ≥ 0.40 | §271 KC |
| S4 | SO C6 | H1: n 5 m 1-4, K' 63 and 64 ≥ K' 8 − 0.02 | §271 H1 |
| ST | STABLE | `score_ao.stable` on the mixed-loss log (`theory/score_ao.py:28`) | as every lineage scorer |

- **JOINT-C1(seed)** = all rows pass. **Arm JOINT-C1** = ≥ 2/3 seeds (3/3 reported).
- W1-W6 are `score_fp`'s RETAIN set without its C2 rows. S1-S4 are `score_n1`'s loop rows, called on `<out>/fam_so`.

### 6.2 Arm-level binding rows (3 seeds; depth and transfer)

- **R-WP-DEPTH.** The seed mean of jt acc at d 6 ≥ of_s1 d 6 − 0.05 **and** the seed mean at d 7 ≥ of_s1 d 7 − 0.05.
  - of_s1's values are 0.971 / 0.909 (Note 263-3), so the bars are 0.921 / 0.859.
  - NEAR-SEED if a seed mean is within σ_seed/√3 = 0.052 of its bar.
  - Per-seed margins are listed.
  - This is depth retention moved from per-seed to the arm. Reason (DATA-SIM, §7): a truly retained, of-like seed passes the per-seed d 6 / d 7 rows only 0.50 of the time at σ 0.09.
- **R-SO-DEPTH.** Δ_SO = mean over jt_s1..s3 minus mean over loop_si_s1..s3 of acc (NUM n 5 MOVED, m 8, K' 8). Bar: Δ_SO ≥ −0.10.
  - NEAR-SEED if |Δ_SO + 0.10| < σ_seed·√(2/3) = 0.073.
  - The paired-by-seed differences are listed (same SO batches, J-3).
  - The same contrast at m 6 is descriptive.
- **JOINT-C2 (C2 claim: ≥ 3 seeds, one lineage, retention-type).** Two conditions:
  - §271's depth rows on fam_so in ≥ 2/3 joint seeds: C2-1 (m 6 ≥ 0.85, m 8 ≥ 0.80), M1 (CWS t 5-8 ≥ 0.90), M2 (SWAP-MID at (5, 8, t 4)), M5 strata;
  - and R-WP-DEPTH passes.
  - Audit rule 3 is met for SO by M2 / M1 at depth. For WP the C2 part is retention-only (as §270), so no d 6-7 WP causal gates are claimed.
  - C2-KPRIME on SO (m 10, K' 12 vs K' 8) is reported as in §271, with no separate label here.

### 6.3 Labels

- **MT-PASS** = arm JOINT-C1 ∧ R-WP-DEPTH ∧ R-SO-DEPTH ∧ ST in every seed.
- **NO-INTERFERENCE** = R-WP-DEPTH ∧ R-SO-DEPTH.
- **INTERFERENCE-f** = the family's depth row fails outside its NEAR-SEED band.
  - For WP, the stage-3a control then labels the cause: INTERFERENCE-WP if ctl_wp_s1 passes W1-W6 and keeps its seed d 6 / d 7 above the bars; else DRIFT-WP, meaning continued of-law training alone loses depth.
- **TRANSFER+-f** (descriptive, claim-limited):
  - SO: Δ_SO ≥ +0.10 with all three paired differences > 0.
  - WP: seed-mean d 7 ≥ of_s1 + 0.10.
  - Never claimed as an improvement under σ_seed (Note 263-3's rule).
- **JOINT-C2** as in §6.2.
- **FAIL** with the failing rows. A reference that never lands counts as below (§9).
- Every label carries the scope sentence of §1.

### 6.4 Descriptive (pre-listed; no label depends on them)

- **D1 GRAD-COS.** Window medians over steps 4k-8k and 12k-16k, per module group, raw and preconditioned:
  - CONFLICT if c̃ < −0.10;
  - ORTHOGONAL if |c̃| ≤ 0.10;
  - ALIGNED if c̃ > 0.10.
  - Test of J-2: does sign(c̃) agree with sign(Δ_f)?
- **D2 LEAK.** `leak_f` per 1k bin (J-5, J-6).
- **D3 K'-curves.** WP acc_kprime by d (rule 4); SO kcurve. Joint vs references.
- **D4 MT4-lite cross-family transplant** (eval-only, stage 2, built only if stage-1 GO):
  - S after pass 2 of a WP d 4 donor, written into an SO n 5 m 4 recipient at pass 2, and the reverse.
  - Recipient acc is compared with (i) an in-family random-donor swap and (ii) §174's length-matched in-family null (donor token length ±2).
  - Expected: both families drop to near the in-family swap floor. The value spaces are disjoint (J-5), so a "shared" reading is unlikely and uninformative.
  - The informative contrast is (ii): PARTITIONED if acc is below the length-matched null by ≥ 0.10.
- **D5 TASK-ID.** A logistic probe of family from the flattened S after pass 1 (2,000 / 500 balanced). It has a token-count baseline and the same probe on of_s1's un-adapted state. P ≥ 0.95 is expected and proves nothing alone (J-1).
- **D6 QB.** The joint law's blind-sufficiency: SO 1/n (J-4; G0 checks by simulation); WP from `read_qb0` once committed. A WP twin-pair question-use read only if lane-07's part exists.
- **D7.** §271's descriptive parts on fam_so (BBH three / five / seven, PERM-LABEL, OFFSET, TRANSFER, LOOKUP, beta_so), side by side with loop_si.
- **D8.** WP insert_ho CONTENT and §263 NAME-SWAP on fam_wp (J-6 (ii)).

## 7. Predictions and OC block

**Predictions (Theory, pre-output; conditional on §271 S1 GO unless stated).**

| # | prediction | P | basis |
|---|---|---|---|
| P1 | stage-1 GO | 0.60 | J-7 power ≈ 0.94 if both mechanisms survive; prior 0.65 that SO forms at 4k inside a mix |
| P2 | W1-W6 per seed | 0.80 | the WP half is of's own law; of retained bun's rows through 8k of a changed law |
| P3 | arm JOINT-C1 | 0.50 | |
| P4 | R-WP-DEPTH | 0.60 | DATA-SIM 0.72 at no shift (σ 0.09); some drift expected |
| P5 | R-SO-DEPTH | 0.65 | J-1 / J-6: weight-only coupling, disjoint value types |
| P6 | INTERFERENCE on ≥ 1 family | 0.30 | J-6 channel |
| P7 | TRANSFER+-SO / TRANSFER+-WP | 0.08 / 0.05 | different cue type and value space |
| P8 | MT-PASS | 0.30 | |
| P9 | D1 full-parameter c̃ ORTHOGONAL / CONFLICT | 0.75 / 0.10 | high-dimensional gradients of unrelated losses |
| P10 | sign(c̃) agrees with sign(Δ) in both families | 0.55 | J-2 is first-order; Adam mixing |
| P11 | leak_SO > 0.05 in bin 1 and < 0.01 at 16k; leak_WP < 0.01 throughout | 0.50 / 0.85 | J-6 (i) |
| P12 | JOINT-C2, given §271 N1-PASS | 0.45 | |

Falsifiers:
- MT-PASS with D1 CONFLICT falsifies J-2's link at this scale.
- An INTERFERENCE row with leak < 0.01 and intact W3 / S2 falsifies J-6's channel claim (the loss is elsewhere).
- PARK at stage 1 while §271 passed GO falsifies "weight-only coupling is benign" (J-1's consequence).

**OC block (DATA-SIM, `mt_oc.py`, 20k sims per cell).** Seed effects are clipped normals shared within a family (ρ 0.5); items are binomial; bars as registered.

| quantity | truth | σ_seed 0.09 | σ 0.05 | σ 0.03 |
|---|---|---|---|---|
| R-WP-DEPTH P(pass) | no shift | 0.724 | 0.896 | 0.979 |
| | depth −0.05 | 0.368 | 0.336 | 0.259 |
| | depth −0.10 | 0.100 | 0.018 | 0.000 |
| | depth −0.15 | 0.014 | 0.000 | 0.000 |
| R-SO-DEPTH P(pass) | Δ 0 | 0.914 | 0.992 | 1.000 |
| | Δ −0.05 | 0.755 | 0.884 | 0.971 |
| | Δ −0.10 | 0.504 | 0.505 | 0.500 |
| | Δ −0.20 | 0.090 | 0.008 | 0.000 |
| | Δ −0.30 | 0.004 | 0.000 | 0.000 |

| JOINT-C1 (seed / arm ≥ 2/3) | seed | arm |
|---|---|---|
| retained, SO mechanism rows pass w.p. 0.9 | 0.736 | 0.828 |
| retained, SO mechanism 0.7 | 0.570 | 0.605 |
| WP in-range true 0.975 (slight loss) | 0.308 | 0.227 |
| SO mechanism absent (memory unused) | 0.000 | 0.000 |
| WP mechanism rows broken (0.2) | 0.152 | 0.062 |

- **P(MT-PASS | no effect of interest).** If the joint model is a WP model that ignores SO's memory, SO S2 / S3 fail → 0.000. If SO depth is lost by ≥ 0.20, R-SO-DEPTH gives ≤ 0.09.
- **P(MT-PASS | true no-interference)** ≈ JOINT-C1 0.6-0.83 × R-WP 0.72-0.98 × R-SO 0.91-1.0 ≈ 0.40-0.80, depending on σ_seed. NEAR-SEED marks the ambiguous band; registered bars stay as they are.
- **Why the per-seed C2 rows were moved to the arm (DATA-SIM).** With depth rows bound per seed at σ 0.09, an of-like truly retained model passes only 0.499 per seed (0.499 arm), and a combined JOINT-2 label only 0.25. That is a detector of seed luck, not of interference.
- **Stage 1.** P(GO | both mechanisms) ≈ 0.991 × 0.95 ≈ 0.94. P(GO | SO memory unused) = 0 (J-7).
- **Realised $** is reported per stage beside its guard.

## 8. Build and gates (Experiments; flags default-preserving)

**Code (built only after §271's files are committed and Audit-G-D'd):**
- `boxes/inject_so.py` gains:
  - `--joint` (default ""), `--so_cue`, `--grad_cos_every` (default 0);
  - the block-balanced draw with family-local RNG and step counters (§3.1);
  - `VOCS[f]` at the VOC sites;
  - WP-only gating of perm / ask / distract;
  - per-family `WS_ITEMS` (today OW-only, worktree `inject_so.py:386`);
  - per-family mem_check items (worktree 388-391 for SO, the OW branch for WP);
  - the §3.4 logs;
  - the eval views (§3.3).
- No other module changes. `inject_aw.py`, `inject_aw_eval.py`, `inject_so_eval.py`, `so_words.py`, `so_tok.py`, `inject_st.py`, `inject_ow.py`, `nested170.py`, `deltamem.py` (343d4e5d…) and `ow_words.py` stay byte-identical and sha-asserted in args.json.
- New: `theory/score_mt.py`. It imports `score_n1.seed_loop` and the RETAIN rows (score_fp form), adds §6.2-§6.3, and uses the `wilson` / `near` pair. Fixtures are committed before any stage-1 output.

**Gates (G0-G3 on CPU, Python-3.11-safe; G4-G6 on one host, billed to stage 1):**
- **G0** §271's generator / sets / floors committed and sha-matched (floors_so.json 1db54c39…). J-4 check: 20k simulated SO items, blind-policy accuracy within 0.01 of the analytic 1/n per n. Family identifiability: the first cue token is " How" vs " After" on 20k items per family.
- **G1a p = 0 equivalence.** With `--joint ""`, two runs must reproduce the frozen paths: of_s1's frozen args (src ow) and loop_si_s1's args (src so). For each, the first 50 CPU fp32 steps must be bitwise equal (loss, gn_preclip, ihash, det wrap), compared against the same commit with the new code absent.
- **G1b SO-half identity.** With `--joint wp,so`, the item-signature sequence of the first 200 SO steps equals a CPU replay of loop_si_s1's training draw. This is a replay of the law, not a read of stored outputs.
- **G1c WP-half freshness.** The overlap of item signatures between the first 1k WP batches and a CPU replay of of_s1's law is reported, expected ≈ 0.
- **G2 render.** Every pass input ends in ':'. Within an item every cue row shares one padding (`cue_inputs`, `inject_st.py:70-79`). WP steps carry 48 chain + 16 ask items and SO steps 64. `family_n` equals 50/50 ± 1 per 1k bin.
- **G3 smokes at full data scale.** 20 steps each of `--joint wp,so`, the plain `--src ow` path and the plain `--src so` path (plain-path rule). Eval views on 8 items per part against fixtures. `score_mt` fixtures pass. At least 14 fixtures: all-pass; each binding row failing alone (table-driven); NEAR; NEAR-SEED; missing §271 reference → VOID-REFERENCE; a missing joint seed → FAIL in the count; PARK; EXTEND; INTERFERENCE-WP vs DRIFT-WP via the ctl; TRANSFER+ never labelled "improved".
- **G4 gradflow and side-effect check (host, bf16).**
  - Per family, the answer CE, aux CE and hold penalty each give non-zero grads at DeltaMem qkv / gate / W_o and the trunk (`_inject_aw_gradprobe_s268.py` pattern).
  - The grad_cos hook leaves `.grad` None and the RNG states unchanged: 100 steps with `--grad_cos_every 50` vs 0 must give bitwise-equal ihash and loss under the det wrap.
- **G5 mem_check per family** at its longest training item (the WP nmax-insert shape; SO n 5 m 4) and at eval (WP d 12 K' 14; SO n 5 m 12 K' 64). Micro 2 pinned, budget ≤ 20 GiB (1 GiB headroom).
- **G6 time probe.** `--time_steps 300` with the joint draw. If s/step > 2.3, stage 2 is re-costed before launch (§10).

## 9. Censoring (conservative; standing rules 10-14)

- **VOID-RERUN once per cell** for non-finite loss, OOM, host loss or drain fault. Resume uses sha-verified ckpts in a new dir. Off-host ckpt mirror every 1k steps; drain waiters armed at launch.
- **A joint seed that never reaches a row** (second failure, or a missing or unparseable eval file) counts as **FAIL** in the k/3 count. The event-only reading (seeds that reached it) is reported beside it.
- **A §271 reference seed that never lands.** R-SO-DEPTH stays pending. If §271 S2 closes with fewer than 3 LOOP-SI seeds, the row is computed against the landed seeds with the bar unchanged and labelled REFERENCE-SHORT. MT-PASS then cannot be awarded (it counts as not passed). This is conservative: no claim rests on a partial reference.
- Swap cells with n_pairs < 1,000 are VOID cells and count as FAIL for that seed (§271 rule).
- Stage-1 PARK voids stage 2. EXTEND is the pre-registered continuation, not a rerun.
- **Blinding:** logs are grep-only until `score_mt.py` and its fixtures are committed. No joint cell is read before the matching §271 reading exists (§5).

## 10. Cost (Vast RTX 3090, ~$0.21/h)

**Basis:**
- WP step 1.65 s (§263 measured, of recipe).
- SO step 1.9 s (§271's token-count extrapolation; §271 S1's G6 probe replaces it before stage 1 launches).
- Joint mean ≈ 1.78 s/step.
- grad_cos adds 2 forward/backward passes of 32 items every 500 steps (< 0.1%).

| item | computation | $ |
|---|---|---|
| stage 1 jt_s1 0→8k | 8,000 × 1.78 s = 3.9 h | 0.83 |
| gate eval (both gate subsets) + boot / host gates / drain | ~0.8 h | 0.17 |
| **stage 1** | | **1.0 (guard 2.0; wall cap 6 h)** |
| jt_s1 resume 8k→16k + full batteries (of ~0.7 h + SO ~0.8 h) | 3.9 h + 1.5 h | 1.20 |
| jt_s2, jt_s3 16k each + batteries | 2 × (7.9 h + 1.5 h + 0.25 h) | 4.06 |
| MT4-lite D4 / D5 eval-only | ~0.7 h | 0.15 |
| **stage 2** | | **5.4; + one VOID-RERUN reserve 2.0 → guard 7.5; wall cap 11 h per 16k cell** |
| stage 3a ctl_wp_s1 (conditional) | 8,000 × 1.65 s = 3.7 h + of battery 0.7 h | **1.0 (guard 1.5)** |

- **Total:** expected ≈ $7.4 if stages 1-2 run (≈ $8.4 with stage 3a). Guards $2.0 + $7.5 + $1.5 = **$11.0**, under the $30 auto-approve rule.
- If the projection after the stage-1 probe exceeds a guard, MT4-lite is dropped first, then the rerun reserve is re-requested.
- GPU class: every cell on RTX 3090, the class of of_s1 and loop_si (basin-entry rule). eval_gpu is recorded.

## 11. Process

- Registration text (THEORY §) carries §3.2's args diff, §4's tags, §6's rows, §7 and §9.
- Literature gets the draft first (theory-keep-literature-in-loop). Lead questions:
  - (i) joint training of looped / recurrent-depth models on algorithmic families after 2209.11142 and 2502.08482: is retention vs single-task reported anywhere at matched exposure?
  - (ii) gradient-affinity (2109.04617) as a predictor of transfer sign in recurrent models.
- An independent red team is still owed (v0's attempt was blocked by agent-pool saturation; §12 lists the known weak points to start from).
- Audit G-D on the build diff: full functions, the family-local RNG plumbing, the VOC sites, WP-only gating, grad_cos side effects.
- Pre-statement pushed and acked before any $0 read. Output files are staged only when finished.

## 12. Known weak points (for the red team) and changes from v0

**Open, disclosed:**
1. **Warm-start asymmetry.** WP has the full bun lineage plus of's 8k steps; SO has 8k steps from of_s1. This is the same asymmetry LOOP-SI has, so the joint-vs-single contrast is fair. But "task-agnostic from scratch" is not tested. A from-scratch joint lineage is a separate, much larger design.
2. **Adam-state sharing.** Second moments mix both families (J-2's limit). An optimiser-state-split arm would separate the two couplings; not proposed.
3. **R-SO-DEPTH power.** The design detects Δ ≤ −0.20 reliably. At σ 0.09 it cannot tell Δ −0.05 from 0 (OC); NEAR-SEED marks this. More seeds would help, but X7 (an independent lineage) matters more.
4. **Cue heterogeneity.** WP's cue names an entity and SO's names a position. The design shows one rule with family-rendered handles, not one cue string (§2.2).
5. **WP-half data seed +100.** The joint WP items are fresh, unlike a hypothetical of_s1 replay. ctl_wp uses the identical WP batches.
6. **§270 FULL-PERM.** If it lands FULL-ORDER-C1 2/2 before stage 2, perm_scope stays op here (one change at a time). A joint-on-fp follow-up would be its own registration.

**Changes from v0 (finding → resolution):**

| v0 | v1 |
|---|---|
| init bun_s1, cost basis 0.62 s/step | of_s1 (lineage head, = §271 init); §263-measured 1.65 s/step + §271's 1.9 |
| four families, v0's own SO generator (persons hold colours, entity-named swap cue, single colour value) | WP + SO; SO is §271's built generator / cue / readout verbatim, so the single-task reference is §271 itself; KIN / PQ deferred (§2.3) with v0's specs in Appendix A |
| stage 0 learnability cells sg_f (2k) as MATCH references | the matched references are of_s1 (WP) and §271 LOOP-SI s1-s3 (SO); exposure matched exactly (J-3) instead of 2k vs a spread 2k |
| per-step uniform family draw | block-balanced, family-local RNG (exact exposure; the SO half is bitwise LOOP-SI's batches) |
| 2 seeds, MATCH bar −0.02 per seed | 3 seeds; per-seed binding on high-power in-range / mechanism rows; depth on arm means with NEAR-SEED (OC-justified) |
| new eval code per family | per-family eval views over the two existing (§271-committed) eval modules |
| THEORY line refs 10439 / 11142 | stale; now §168 line 10441 and §174 line 11144 (both read today) |
| no math tags | J-1-J-7 tagged |

## 13. Files

- This document.
- The scratchpad scripts `mt_lr.py` and `mt_oc.py` and the output `mt_oc_out.txt` move to `docs/research_mt_2026-11-04/` at registration. Their seeds are 27460-27462, unused (grep of docs/THEORY.md, boxes/*.py and theory/*.py at 02:2x EST: 0 hits), to be re-checked at registration.
- No code, run directory or THEORY entry was created by this draft.

## 14. Citations

arXiv IDs as verified in v0 (abs pages fetched 2026-11-03, titles matched; not re-fetched today):
- 1807.03819 Universal Transformers
- 2209.11142 A Generalist Neural Algorithmic Learner
- 2502.08482 Loop-Aligned Reasoning (RELAY)
- 2110.14056 transfer of algorithmic reasoning
- 2001.06782 PCGrad
- 2109.04617 task groupings / gradient affinity
- 2210.09261 BBH
- 2210.10749 Shortcuts to Automata
- 2602.08332 Thinking States

For SO, see N1_v1 §14.

## Appendix A. Deferred family specs (v0, condensed; for their own single-task registrations)

- **KIN (kinship-lite).**
  - Item: persons from NAMES (6-10); relations {mother, father, sister, brother, aunt, uncle, cousin, friend} as an arbitrary partial function (Y, r) → X, unique per key.
  - Text: facts "<X> is the <r> of <Y>."; a chain P_0 and then "Let <slot_t> be the <r_t> of <slot_{t-1}>." (slots A-H); question "Who is <slot_d>?".
  - Extra facts: 2-6 about the same persons, plus 0-2 about unused names, in random order.
  - Cue: "Who is <slot_min(t,d)>?". Aux: "<slot_t> is <v_t>.". Candidates: the item's persons.
  - redraw: the same skeleton with the table re-drawn.
  - Risks: a textual two-lookup at d ≤ 2 inside one pass (NOT-DECODER at K'2 on d 4); a last-mentioned-subject heuristic (WS split by recency).
- **PQ (ProsQA p8_na).**
  - Relabel via §191 `relabel()` (`prosqa168s2.py:77-99`) onto single-token vocabularies. The test file is relabelled once with a fixed rng, because `relabel` asserts on train only (line 82).
  - Values: the gold path nodes. Readout: {opt_true, opt_false} (chance 0.5).
  - Bar: THINK-GAIN (acc K' 8 − K' 0 ≥ 0.20 at depth 4) rather than a chance-based decoder bar, because one-pass GPT-2 reads 0.63 at depth 3 (§168, line 10441).
  - Must use the shortcut-free set (NORTH_STAR, §57).
  - These line citations were not re-read for this v1 (PQ is deferred). They are verified when PQ's registration is drafted.
