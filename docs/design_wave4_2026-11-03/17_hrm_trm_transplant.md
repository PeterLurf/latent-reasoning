Status: completed 2026-11-04 by Theory's agent; not red-teamed yet

# 17. DSX: DETACHED DEEP SUPERVISION for answer-only depth extension of the of loop (HRM / TRM transplant) — design (2026-11-03 draft, completed 2026-11-04)

**Provenance.** The 11-03 draft (DSR: deep-supervised recursion from the askname init, a C4 formation test) was cut off before its red team. This file rewrites it in place. The rewrite does four things:
- it checks each HRM / TRM mechanism against its precondition (§2);
- it re-targets the design at the current best model, of_s1 / of_s2 (Note 263-3), and at C2 / C1 rather than C4 formation (§3);
- it keeps one transplant, detached deep supervision;
- it adds the math, gradient-flow plan, predictions, OC block, censoring, staged plan and cost.

The draft's askname-init C4 arm is kept in Appendix A as deferred, with the reason.

Work was read-only. There was no training and no computation on stored run outputs. The only stored files opened were `boxes/runs_fsq/of_s1/args.json`, `args_diff.json` and `host_shas.json`, plus `bun_s1/args.json`, all for recipe flags. Every code citation below was re-read on 2026-11-04 against the current tree. The line numbers in the draft were stale: inject_aw.py is now 739 lines, not 695.

**Backlog IDs.** AP1 (A8) DEEP-SUPERVISED RECURSION (main). It touches CA1 SG-WRITE (detach), D1 / §265 (depth horizon, same failure), and H6 / RL2 (halting; not adopted).

**Priority fit (NORTH_STAR: C1 > C6 > C4 = C2 > C3 > C5 > C7).**
- Primary: **C2**, depth past the trained depth with K' > K, which is the registered construct and is untested (standing rule 4).
- Guarded and gated: **C1** (RETAIN in range, plus the causal gates at d 6-7 that standing rule 3 requires) and **C6** (K' 64 / 128 hold).
- Descriptive: C4-partial. Answer-only extension, but with the inherited step-value anchor on d <= 4, so it is not a C4 claim under Note 266-1.
- C5 is not designed for.

**Scope line (pre-stated):** one lineage (askname -> P-2 -> ... -> bun -> of); plan given (the step cue names S_t); op sentences in any order (§263), givens / S_0 in trained placement; REG frames 0-5; step-value aux on d <= 4 only; answers only on d 5-8; RTX 3090.

## 0. Summary

### 0.1 One-paragraph plan

of_s1 fails past depth 8 for a known reason. Its first errors fall at passes it never trained to compute (passes >= 5; X4a, Note 264-1). The same pass index also degrades as op sentences are added (pass 7: 0.91 at d 7, 0.59 / 0.80 at d 8). Past d 8 the lineage sits at chance: bun's d 9 is 0.16-0.25 and d 10 is 0.08-0.12 (Note 269-4c).

HRM / TRM train a recurrent step from final answers only. They put the answer loss after every supervision segment (deep supervision) and **detach** the carried state between segments. Mapped onto our loop (segment = pass), that becomes **DSX**:
- of_s1's recipe continues unchanged on d <= 4;
- 25% of steps draw d 5-8 items trained by **answers only**;
- each of these items gets CE(decoder(S_t), a) at every pass t in T_d = {d..K};
- S_{t-1} is detached before each supervised pass, and passes 1..d-1 run without a graph.

**Stage 1** (~$2.4) runs DSX s1 against two controls from the same init. CONT continues of_s1's recipe with no deep items (the extra-training confound). DSF is the same deep supervision with no detach (the single-change control for the HRM / TRM detach). A **2k mechanism gate** reads the in-run pass-8 ':' readout at d 8 (WS). **Stage 2** (~$3.0) adds DSX seeds 2-3, because a C2 claim needs >= 3 seeds. The primary read is d 9-10 at K' = d + 2 > K = 8, which is beyond the trained depth and beyond the trained pass budget. Family guard **$7.5**.

### 0.2 Decision rule (pre-stated)

- **Stage 1a (2k):** GO if DSX s1's ws_readout['8']['8'] >= CONT s1's + 0.10 at the 2k bin, and DSX's WS d 1-4 at passes t <= d stays >= 0.95. PARK if DSX ws['8']['8'] <= CONT + 0.02 at both the 1k and 2k bins, or if the d 1-4 WS clause fails. Otherwise NOTE: continue.
- **Stage 1b (4k, s1):** per-seed label (§9.3). Stage 2 runs only if DSX s1 is not VOID / UNSTABLE / DRIFT and its pooled d 9-10 accuracy at K' = d + 2 is >= init + 0.10.
  - If the pre-stated attribution reads DETACH-COSTS (§9.4), Stage 2 runs DSF instead of DSX. The report then says deep supervision without the HRM / TRM detach.
- **Claim:** a C2 claim needs C2-EXT plus RETAIN in **3/3** seeds of the run arm, and CONT not passing C2-EXT. If CONT passes, the label is CONFOUNDED. A C1 + C2 bundle claim also needs C1-DEEP and C6 in 3/3. Two of three seeds is reported as "2/3, no C2 claim".

## 1. What HRM and TRM do (the borrowed mechanisms)

Sources: HRM 2506.21734 (Wang et al.) and TRM 2510.04871 (Jolicoeur-Martineau).
- Repo bib: in docs/LIT_PASSUSE_PRIORART_2026-10-15.md rows 77-78 (abs fetched), docs/LIT_GENLAT_VERIFY_2026-10-13.md rows 48-49 (claims confirmed against the full text), and docs/LIT_WAVE4_N1_MC3_2026-11-04.md lines 61 / 256 (this draft's TRM numbers verified).
- Not yet in docs/LITERATURE.md (Unresolved #7).

| # | Mechanism (one line) | Paper detail (as verified in the repo's notes) |
|---|---|---|
| M1 | **Deep supervision:** the same final-answer loss after every supervision segment, carrying the state forward | HRM: up to M_max segments. TRM: N_sup = 16; "deep supervision 19 -> 39% on ARC" (GENLAT_VERIFY 48) |
| M2 | **Detach between segments:** z carried forward with stop-gradient, so memory is constant in the segment count | HRM Fig. 4 pseudocode (GENLAT_VERIFY 49) |
| M3 | **1-step gradient inside a segment** (HRM), justified by the DEQ implicit-function theorem at a fixed point | HRM; "assumes a fixed point, which TRM shows is not reached" (GENLAT_VERIFY 49) |
| M4 | **Inner latent recursion:** n updates of z per output update, T − 1 recursions without gradient, then 1 with full backprop | TRM n = 6, T = 3. The 1-step gradient gives 56.5 vs 87.4 with full backprop through the last recursion (Sudoku-Extreme; WAVE4 256) |
| M5 | **Two timescales:** a slow H module and a fast L module | HRM; TRM drops the hierarchy for one network |
| M6 | **Learned halting:** a Q-head (HRM, Q-learning) or BCE on own-answer correctness (TRM, 86.1 -> 87.4) | TRM §4.6 (GENLAT_VERIFY 48) |
| M7 | **Tiny network trained from scratch**, heavy augmentation, EMA 0.999 | TRM: 5M parameters, no seeds or error bars (GENLAT_VERIFY 48) |
| M8 | **Output head reads only the latent** (no input path at the answer) | both |

## 2. Preconditions: does each mechanism's requirement hold here?

Rule (transplant-preconditions lesson): name the mechanism, name what it needs, and check what each pass of *our* loop can actually see.

Our loop, from code read on 11-04:
- **Pass t:** h = Trunk(x_t; read_6(S_{t-1})), then S_t = write_seq(h6[':'], h11[':'], S_{t-1}).
  - pass_aw, boxes/inject_aw.py:47-56;
  - the read at block mem.layer via DM.read_fixed, inject_aw.py:52;
  - the write at deltamem.py:179-223.
- **State:** 6 additive heads, S_a <- S_a + Σ φ(k) vᵀ with φ = elu + 1 and no decay (deltamem.py:195-198 FAST_ADD, 206). 6 gated-delta heads, S_d <- α(S_d − β (S_d k) kᵀ) + β v kᵀ (deltamem.py:208-210).
- **Decoder:** reads S_K plus the full text and the question. dec_logits, inject_ow.py:86-90, through deltas5.fwd_read, which uses the **same** block-6 read_fixed (deltas5.py:91-98).
- **Aux:** reads S_t through a text-free prompt, for t <= d (inject_ow.py:92-103; inject_aw.py:427-432).
- **Pass input:** x_t = [EOT] + text + QUESTION(u, S_min(t, d)) + " The answer is:" (tok_aw, inject_aw.py:76-93). So the plan is given, and passes t > d see the S_d cue again.

| # | Mechanism needs | Here | Tag | Verdict |
|---|---|---|---|---|
| M1 | (i) The final answer is *reachable* from the state after each supervised segment, so every segment's loss has an informative direction. (ii) The same target at every segment. | (ii) holds. (i) holds only for t >= d. Every REG op is a bijection on Z_11 (add / sub / mul by constants and givens; §267 Why), so from S_t with t < d − 1 the answer needs >= 2 more hops. The text decoder adds at most one: DECODER-LAST-HOP, d 4 at K' 3 = 1.000 / 1.000 (Note 264-4), and answer at K' = k equals v_{k+1} (Note 255-2). At t = d − 1 the target is reachable only *through the decoder shortcut*, which is the wrong gradient for C1. | DERIVED (bijection) + DATA (264-4, 255-2) | **Adopt, restricted to T_d = {d..K}.** At t > d it is TRM's "stay put once correct" (THEORY line 11924), i.e. an answer-hold target. Using d to pick T_d is a plan-given scope disclosure (Unresolved #3). |
| M2 | The next segment can use the carried state without the state gradient co-adapting writer and reader. That needs the state to be read by a map that the per-segment loss already trains. | Holds. The decoder that ℓ_t trains reads S_t through the same block-6 read_fixed that pass t + 1 uses (inject_aw.py:52; deltas5.py:96), so decoder-readability of S_t trains the read that the next pass uses. The writer-side co-adaptation (later losses shaping pass t's write) is what detach removes, by design (§5). | DERIVED (code) | **Adopt** as the transplant's core, as plain stop-gradient between supervised passes. |
| M3 | A fixed point S* = F(S*) with small ‖J‖, so that (I − J)^{-1} ≈ I. | Fails. (a) On computing passes t <= d the state is not at a fixed point: each pass writes a new value by construction. (b) On the additive block the direct Jacobian ∂S_a,t/∂S_a,t−1 = I exactly (deltamem.py:195-198, 206), so I − J is singular on that block and the IFT approximation has no justification. Delta block: direct J = α(I − β k kᵀ), a contraction. The total J through the read path is unmeasured. | (a) PROVED by construction; (b) DERIVED for the direct path, CONJECTURE for the total | **Not transplanted as an implicit gradient.** Detach is plain truncation (TBPTT-1). It is justified by protection and memory, not by IFT. Note: TRM's own ablation (56.5 vs 87.4) is about within-segment gradient. Our within-segment gradient (the trunk's 12 blocks of one pass) is full backprop, the better TRM arm. |
| M4 | Re-applying the step under a fixed input refines z, i.e. the step is a contractive refinement operator. | Fails. Under the step cue, a repeated pass re-applies the same hop. Additive heads add φ(k)vᵀ again, so ‖S_a(K')‖ ~ (K'/d) ‖S_a(d)‖ under aligned re-writes (Note 264-2 mechanics). Overthinking from additive growth is measured (§175). Gated-delta heads are idempotent up to (1 − α). | DERIVED (264-2) + DATA (§175) | **Not adopted** (the draft's n = 2 inner micro-passes are dropped). |
| M5 | Two state modules with different update rates; the slow one sets context so the fast one avoids premature convergence. | The role of the slow module (choosing the sub-goal per step) is filled by the **given plan cue**: one trunk, one memory. A learned slow state could only matter in the plan-free regime (§260 GENERIC, §261 TARGET-CUE). | DATA (cue design) + CONJECTURE (plan-free value) | **Not adopted here;** deferred to the plan-free lane. |
| M6 | The halting signal (own-answer correctness) is predictable from the hidden state, and the steps needed vary by item. | C5 r = 3 is already EXTRAP (Note 263-3), but it is plan-clocked: khalt = d − 1 at >= 0.976 for d 3-4 (Note 264-4). A BCE head would learn the same clock. ACT / PonderNet collapse to one loop on a looped LM (RecurTrace 2609.03379, verified in WAVE4 line 246). C5 is low priority. | DATA | **Not adopted.** The existing post-hoc halthead part already reads this at no training cost. |
| M7 | Training from scratch on small data, where EMA and augmentation fight overfitting. | Fails: pretrained GPT-2, an unbounded generated stream. | DATA | **Not adopted** (EMA would also be a second change). |
| M8 | The latent holds the whole solution state, so an output head can read the latent alone. | Not established. S accumulates and the decoder reads text plus S; the decoder shortcut is DATA (264-1 / 264-4). An S-only decoder is the M3 / S11 lane of design 19 (red-teamed GO-WITH-FIXES, Stages 0-1). | DATA | **Not adopted here,** to avoid a second simultaneous change and duplicated work. |

**Net:** one transplant survives, **M1 + M2 = detached deep supervision on the passes where the answer is reachable (t >= d)**. Its C2 lever is not M1 / M2 by themselves. It is what they make possible: answer-only training of the passes that X4a shows are untrained (5-8), with the earlier passes' writes shielded from those deep-item losses (§5 (a)).

## 3. Why this transplant, on this init

- **Which payoff it targets.** X4a (Note 264-1) locates of / bun's depth failure in untrained late passes, which degrade with the op-sentence count. ERROR-PROPAGATION is SERIAL / MIXED (Note 269-4a): wrong values carry forward, so the memory carries the chain. A fix to the late passes is therefore a C2 fix.
  - DSX trains passes 5-8 directly, from answers, on real d 5-8 chains. Its sibling §265 DENSE-DISTRACT trains the op-sentence count at d <= 4 (inserts).
- **Why of_s1 removes the old answer-only failure.** Answer-only extension from the P-2 seed was "rare" (§255: EXTEND 1/4). The failure was **basin entry**: d 3-4 never entered in s3 / s4, while the anchor held in 4/4 (Note 255-1). of_s1 is already inside the basin at the target depths (TEST2 d 6 / 7 = 0.971 / 0.909 at K' 8, Note 263-3). So the precondition that killed AEX, an unformed pass, does not hold here (DATA).
- **Why detach.** HRM and TRM always detach between supervision segments. On our loop, detach is what keeps a deep item's answer CE away from the pass-s writes (s < d) that the d <= 4 anchor relies on (§5 (a), PROVED). It also lets passes 1..d − 1 run without a graph, which cuts the train peak at the d 8 shape (§5 (d)). The cost is credit assignment to earlier passes (§5 (b)). DSF measures that cost.
- **Why C2 is read at d 9-10 and K' > 8.** Training at d <= 8 moves the trained depth to 8. So the claim must be read beyond it.
  - d 10 needs passes 9-10. Their cues (S_9, S_10) appear only at K' >= 9 / 10, and K = 8 passes never trained there.
  - "acc rises from K' 8 to K' 12 at d 10" is therefore exactly the registered C2 construct "K' > K helps" (standing rule 4), not a fixed-K depth row.
- **What it does not claim.** C4: the d <= 4 step-value aux continues (Note 266-1 ruling; this is the §251-type C4-partial). C3, C5, C7: none.

## 4. Design

### 4.1 Notation and loss

Notation as §259 / §263:
- items with depth d and answer a ∈ Z_11;
- K = 8;
- x_t is the pass-t input (cue S_min(t, d));
- Dec(S) = dec_logits(S, items) restricted to the 11 NUM_IDS;
- F_θ(S, x) = InjectAW.write.

**Step law (new; one d per step, as now).** With `--ext_p p --ext_lo 5`:
- a coin from a new generator, default_rng(seed + 27200), fires with probability p;
- if it fires, d = rng_d.integers(ext_lo, d_max + 1); otherwise d = rng_d.integers(1, ext_lo).

With ext_p 0 and ext_lo 5, rng_d's draws are integers(1, 5), the same calls and values as of_s1's integers(1, d_max + 1) at d_max 4 (gate G1b). ext_p = 0.25, so d ≤ 4 keeps 75% of steps.

**d <= 4 steps:** of_s1's loss_reg exactly (inject_aw.py:394-452):
- the decoder CE at S_K through the full graph;
- the step-value aux at t <= d, weight AUX_W;
- the ask items (loss_ask, 454-466);
- the hold penalty (hold_step, 159-199).

**Extension steps (d >= ext_lo), under `--ds_ext 1`:**
- aux off, because `--aux_d_max 4` already removes it (inject_aw.py:427);
- the decoder CE at K is replaced by
  - L_ext = (1 / |T_d|) Σ_{t ∈ T_d} CE(Dec(S_t), a), with T_d = {d, …, K} and uniform weights (HRM);
- **DSX** (`--ds_detach 1`):
  - S_0..S_{d−1} are computed under no_grad;
  - for t ∈ T_d: S_t = F_θ(sg S_{t−1}, x_t), then ℓ_t = CE(Dec(S_t), a);
- **DSF** (`--ds_detach 0`): the same ℓ_t on the full graph from S_0.

Ask items and the hold penalty run on extension steps as on any step.
- hold_step's in-batch H = {d+3..K} is empty for d >= 6.
- The far pass (T_far ~ U{16..40}) still trains the hold for the first hold_far_n items.
- A fix is needed for d >= 7 (§6, G2).

### 4.2 Arms and init

Init for all arms: **of_s1 final weights**.
- `--init_model boxes/runs_fsq/of_s1_init/model.pt`, a byte copy of of_s1/model.pt, following §263's bun_s1_init convention.
- `--init_model_sha` is the prefix computed at build (not in of_s1/host_shas.json).
- `--args_ref` is a frozen copy of boxes/runs_fsq/of_s1/args.json (sha256 8fb1fa5a…, host_shas.json).

| Arm | Run names | Change vs of_s1 recipe | Role |
|---|---|---|---|
| **DSX** | dsx_s1, dsx_s2, dsx_s3 (training seeds 1-3) | `--ext_p 0.25 --ds_ext 1 --ds_detach 1` + common | treatment (the transplant) |
| **DSF** | dsf_s1 | `--ext_p 0.25 --ds_ext 1 --ds_detach 0` + common | single-change control for detach |
| **CONT** | cont_s1, cont_s2 | `--ext_p 0` + common | extra-training / warm-restart confound |
| DSX-o2 (optional, Stage 2) | dsx_o2 | DSX, init of_s2 final weights, seed 2 | second init (still one lineage) |

### 4.3 Exact args diff vs of_s1/args.json (keys that differ; everything else identical)

| key | of_s1 | DSX | DSF | CONT | note |
|---|---|---|---|---|---|
| init_model | boxes/runs_fsq/bun_s1_init/model.pt | boxes/runs_fsq/of_s1_init/model.pt | same | same | warm from of_s1 |
| init_model_sha / _sha256 | 1ad3517b… | of_s1 model.pt sha | same | same | computed at build |
| steps | 8000 | 4000 | 4000 | 4000 | new cosine; 100-step warm-up (fsq120.lr_at:740-743) |
| d_max | 4 | 8 | 8 | 8 | CONT at 8 for the d 5-8 WS / swap_kd / lp5h ranges only; its d law is d <= 4 |
| aux_d_max | 0 | 4 | 4 | 4 | inert for CONT (d <= 4 only) |
| ext_p / ext_lo (new) | — | 0.25 / 5 | 0.25 / 5 | 0 / 5 | |
| ds_ext / ds_detach (new) | — | 1 / 1 | 1 / 0 | 0 / — | |
| micro | 2 | pinned from mem_check (expect 4) | pinned (expect 4-8) | 2 | non-semantic; in the diff (auto_micro-headroom lesson) |
| eval_parts | acc,test_perm,insert_ho,swap_kd,lp5h,stab | acc,test_perm,swap_kd,lp5h,ws_deep,stab | same | same | + the long_k128 line (§9) |
| seed | 1 | 1 / 2 / 3 | 1 | 1 / 2 | |
| out, args_ref, keep_final_ckpt | — | per run; keep_final_ckpt only for the Stage-1a stop / resume | | | bookkeeping |

**Unchanged (listed for the reviewer):**
- think_k 8, bs 64, lr 1e-4;
- aux_step 1.0, aux_reweight, aux_fade 48000,48000;
- cue step, write_src answer, cache_lower, fast_add;
- distract_mix s256,chain,givop with distract_p 1.0;
- perm_p 0.5, ask_p 0.25, ask_perm_p 0.5;
- hold_inc 1.6, hold_start 500, hold_far 16 / 16-40, hold_log;
- mul_s_p 0, mul_s_p3 1.0.

hold_start 500 is kept, so the warm run trains its first 500 steps without the penalty. This follows the §265 warm-from-bun convention, applies to all arms, and is disclosed.

## 5. Math backing and gradient flow

Let θ be the shared parameters. S_t = F_θ(S_{t−1}, x_t). ℓ_t = CE(Dec_θ(S_t), a) for t ∈ T_d. Write g_t = ∂ℓ_t/∂S_t and J_r = ∂S_r/∂S_{r−1}.

**DSF (full graph):**
∇_θ L = (1/|T_d|) Σ_{t ∈ T_d} [ ∂ℓ_t/∂θ|_Dec + Σ_{s ≤ t} g_tᵀ (Π_{r=s+1}^{t} J_r) ∂F/∂θ|_{pass s} ].

**DSX:** sg zeroes every J_r on the supervised path and every s < d term:
∇_θ L = (1/|T_d|) Σ_{t ∈ T_d} [ ∂ℓ_t/∂θ|_Dec + g_tᵀ ∂F/∂θ|_{pass t, S_{t−1} const} ].

Consequences:

**(a) PROVED (stop-gradient semantics).** Under DSX, an extension item's loss puts exactly zero gradient on the activations of its passes s < d. That includes their ':' write rows h6 / h11[':'] and their k, v, α, β.
- Mechanism: the AW-SOURCE-DRIFT path is the s < t terms that pull the pass-s code off v_s (Notes 251-8..11). That path is cut for every deep item.
- What remains is the **shared-weight path**. ∂F/∂θ|_{pass t} moves θ, and θ also computes pass s.
- So "protection" means protection of the pass-s codes from directional pressure, not from interference.
- G3 checks the zero (hooks on S_s, s < d).

**(b) DERIVED: the credit-assignment cost.** Suppose item i has an earlier error, v̂_{j} ≠ v_j for some j < d. Every op is a bijection, so a is not a function of S_{d−1}'s carried value. g_d then pushes pass d and the decoder to produce a from a wrong state, i.e. to compute from text (the decoder shortcut) or to memorise.
- The share of such items is 1 − P(passes 1..d−1 right). In bun this is ~0.3-0.5 at d 8 (X4a conditional reliabilities, Note 264-1).
- Under DSF, the s < d terms can also *fix* the earlier pass (useful credit), and they can also distort it.
- Read: DSF − DSX on C2-EXT measures the net (§9.4). NOT-DECODER at d 8 (K' 6) is read descriptively for the shortcut.

**(c) DERIVED (M3's failure, in symbols).** HRM's 1-step gradient replaces ∂S*/∂θ = (I − J)^{-1} ∂F/∂θ by ∂F/∂θ.
- On the additive block the direct J is the identity (S_a,t = S_a,t−1 + Σ_τ φ(k_τ) v_τᵀ; deltamem.py:195-198), so (I − J_direct)^{-1} does not exist.
- On passes t <= d, S_t ≠ S_{t−1} by construction.
- DSX's gradient is therefore not an approximation to any implicit gradient here. It is TBPTT with window 1, and the design claims nothing beyond §5 (a) and (d).

**(d) DERIVED: memory.** DSX keeps graphs for |T_d| = K − d + 1 <= 4 passes plus |T_d| decoder calls. DSF keeps all K = 8 passes plus |T_d| decoders. At the d 8 + 2-insert shape, the train peak ratio is ~ (1 + 1) / (8 + 1) of the pass activations plus fixed terms.
- Prediction P5: DSX peak <= 0.75 × DSF peak, both measured by mem_check at d_max 8 (G5).

**(e) DERIVED: answer-hold.** For t > d, ℓ_t is an answer-hold target. With the cue fixed at S_d, it rewards S_t staying decoder-equivalent to S_d. That is TRM's "stay put after correct", and it complements the hold penalty, which bounds ‖ΔS‖ on t >= d + 3. It acts only on d 5-7 items, at passes d + 1..8. It does not touch d <= 4 items, so C6 rests on the existing hold penalty.

**(f) CONJECTURE: why this should lift d 9-10.** The step map is index-free: no pass embedding, and the cue carries the sub-goal (tok_aw, inject_aw.py:76-93). Training passes 5-8 to compute on 5-8-op chains therefore trains the same map that passes 9-10 apply. The open risks are two:
- the state distribution after 8+ writes (additive norm grows with the write count, Note 264-2);
- op counts of 9-10 at d 9-10, beyond the trained 8 + 2 inserts = 10 op sentences, which is the §265 / §241 axis.

**Gradient-flow check plan (scratch/gradflow_check.py pattern; read from the decorators down):**
- G3a: per term, alone. Each ℓ_t (t ∈ T_d) gives nonzero grad norms on mem.qkv, mem.gate, mem.ln_w, mem.wo and the trunk. ℓ_aux on an extension step gives exactly 0 (aux_d_max). ℓ_ask and the hold term are unchanged vs the flags-off tree.
- G3b: under DSX, S entering pass d has requires_grad False. Exactly |T_d| grad-carrying write_seq calls per extension batch (forward-hook count). Passes 1..d − 1 run inside torch.no_grad.
- G3c: retain_grad on S_s for s < d in a debug path. Under DSX ‖∂L_ext/∂S_s‖ = 0; under DSF it is > 0 (so the control's extra path exists).
- G3d: no new @torch.no_grad on any part of the DS path (copied-module no_grad lesson; ast check of the decorators on loss_ds and on the d <= 4 path).
- G3e: d <= 4 steps, flags on vs flags off: identical loss and grads, bitwise, under the deterministic wrapper.

## 6. Code touch points (all re-read 2026-11-04)

**boxes/inject_aw.py (739 lines):**
- Flags, added after `--perm_scope` (250-251): `--ext_p` (float, default 0), `--ext_lo` (int, default 0 = off), `--ds_ext {0,1}` (default 0), `--ds_detach {0,1}` (default 1).
- Asserts near 261-272:
  - ds_ext needs ext_p > 0, write_src answer, cue step, not ckpt_passes, and not cue_ans;
  - ext_lo needs aux_d_max == ext_lo − 1;
  - ext_lo <= d_max.
- d draw (588): currently `d = int(rng_d.integers(1, a.d_max + 1))`. It becomes the §4.1 law under ext_lo > 0, with the new coin generator stored in the resume checkpoint beside rng_far (restore 562-563, save 705).
- **loss_reg (394-452):** on items with d >= ext_lo under ds_ext, a new branch replaces the pass loop and `la`:
  - passes 1..d − 1 run under no_grad when detaching;
  - passes t ∈ T_d then run with `m.write(S.detach(), …)`;
  - ℓ_t = CE(m.dec_logits(S_t, items, dev)[:, NUM_IDS-index], a);
  - `la` = the mean ℓ_t.
  - Ssv for the hold stats is still filled with detached states.
  - The d <= 4 path is untouched.
  - stat logs `ds_ce_by_pass {t: ce}` and `ds_acc_by_d_pass {d: {t: acc}}`.
- **hold_step (159-199), line 170:** `den = state_n(Ssv[d + 2])` raises IndexError for d >= 7, since Ssv holds S_0..S_K. It becomes `Ssv[min(d + 2, K)]`, which is bitwise for d <= 6. H (168) is already empty for d >= 6. The far pass (185-198) is unaffected (T_far − 1 >= 15 >= d + 2).
- **Logging (≈640-690):** the per-1k-bin record gains `ds_ce_by_pass`, `ds_acc_by_d_pass` and `ext_share`. `ws_readout` (675) already covers d 1..d_max through WS_ITEMS (486), so CONT and DSX both log d 5-8 per-pass WS. This is the Stage-1a gate field.
- mem_train (488-494) runs loss_items on MC_TR, which is at d_max with 2 inserts (478-484). At d_max 8 it therefore measures the deep shape. Under ext_p > 0, MC_TR must exercise the DS branch (a flag so mem_train treats MC_TR as extension items).
- Disjointness: EXCL (281) plus the TP_TEXTS assert, which already spans d 1..d_max (317-318). Add an assert that no extension-step sig is in TEST2 d 5-12, the ws_deep sets, or the swap_kd / lp5h pair sets of d 5-8 (G7).

**boxes/inject_aw_eval.py:** no change.
- part_acc (104-114) covers TEST_DS = 1..12 (inject_ow.py:52) at K' 0..KMAX 14 (35).
- part_swap_kd (391-434) and part_lp5h (510-533) loop d 3..d_max, so d_max 8 in args.json yields the d 6-7 C1 gates of standing rule 3. Fields: by_d[d].swap_mid[k].follow_U / floor, n_pairs, t_mid = ceil(d/2), and by_d[d].intact[k] / mean_at_t[k].
- part_ws_deep (1111-1120): TEST2 d 5-8, set shas asserted.
- part_stab (1282-…): d 1-7 at K' 0..14.
- run_long (558-575) with `--kprime_list 8,16,32,64,128 --long_ds 1,2,3,4,5,6 --long_name eval_long_k128.json`.

**theory/score_dsx.py (new):** wilson / near copied from theory/score_tx.py:40-50; stable from theory/score_ao.py:28.

## 7. Gates (build; Audit G-D)

- **G1 flags-off bitwise:** the new flags at their defaults replay of_s1's recipe from its own init (bun_s1_init) for 100 steps. The step-100 log loss must match the pre-change tree bitwise.
- **G1b CONT law identity:** from the of_s1 init, `--d_max 8 --ext_p 0 --ext_lo 5 --aux_d_max 4` vs `--d_max 4`. Identical d sequence (dseq) and bitwise step-100 loss. Checks that the d_max change only widens logging and eval ranges and the EXCL / TP sets: excl_n is logged and a difference is disclosed.
- **G2 hold_step at d 7-8:** no exception. Stats are finite. d <= 6 output is bitwise unchanged vs the old line 170.
- **G3a-e gradflow** (§5).
- **G4 forward identity:** DSX and DSF give the same ℓ_t values on one batch at step 1 under the deterministic wrapper (GPU bitwise gates need it). Detach changes gradients only.
- **G5 mem_check** at d_max 8 + 2 inserts, micro pinned, 1 GiB headroom under the 21 GiB budget, for each of DSX / DSF / CONT. Eval peak unchanged (d 12 × K' 14). Both peaks are recorded (P5).
- **G6 smoke:** 200 steps at full data scale on the cloud host (Python 3.11, py311 API guard), including >= 10 extension steps at each d 5-8. Local GPU work is on hold (RAM bit flips).
- **G7 disjointness** asserts (§6), with counts logged (0 expected).
- **G8 scorer fixtures** (§10) pass before launch. Per-host gate script on the §265 pattern (scratch/gate_s271p.py): TRAIN requires gates_pass.json.

## 8. Predictions (falsifiable; probabilities stated before any output)

| ID | Prediction | P | Basis |
|---|---|---|---|
| P1 | DSX s1 at 4k: TEST2 d 8 at K' 8 >= 0.80 (IN-DEPTH) | 0.70 | in-basin init (Note 263-3 d 7 0.909); 1k answer-only d 8 steps |
| P2 | Per DSX seed: C2-EXT (pooled d 9-10 at K' = d + 2 >= 0.40 and >= init + 0.20) | 0.45 | §5 (f); op count 9-10 just past the trained 10 |
| P2b | DSX d 10: acc(K' 12) − acc(K' 8) >= 0.25 (K'-GAIN) | 0.50 | at K' 8, d 10 gets at most v_9 via the decoder hop, so ≈ floor (P(v_9 = a) ≈ 1/11) |
| P3 | CONT: C2-EXT fails (pooled d 9-10 within ±0.05 of init) | 0.90 | CONT trains d <= 4 only |
| P4 | Detach attribution (§9.4): DETACH-NEUTRAL 0.50 / DETACH-COSTS 0.30 / DETACH-PROTECTS 0.10 / other 0.10 | — | the anchor held 4/4 under full-graph answer CE (Note 255-1), so drift on a formed loop is unlikely; §5 (b) credit loss favours DSF slightly |
| P5 | DSX train peak <= 0.75 × DSF's (G5) | 0.80 | §5 (d) |
| P6 | DSX RETAIN (d 1-4 >= 0.98, TEST-PERM d 2-4 >= 0.90, SWAP4, LP5-D4, NOT-DECODER) | 0.85 | anchor continues on 75% of steps |
| P7 | C1-DEEP for DSX given IN-DEPTH (SWAP-MID d 6 and d 7 at K' = d; LP5 d 6-7 drop >= 0.30; NOT-ONE-PASS d 7) | 0.60 | ERROR-PROPAGATION SERIAL / MIXED (269-4a) |
| P8 | DSX C6 retained (K' 64 / 128 >= K' 8 − 0.02 at d 1-6) | 0.75 | hold penalty on; T_d answer-hold (§5 (e)) |
| P9 | K'-curve at d 7-8 flat past K' = d (|acc(14) − acc(d)| <= 0.03) | 0.85 | the plan-clocked hold (264-2 / 264-4) |
| P10 | Under DSX, the gradient to pass s < d states from L_ext is exactly 0 (G3c) | 1.0 | PROVED (§5 (a)) |
| P11 | DSX d 11-12 at K' = d + 2: below 0.30 | 0.70 | op count 11-12 beyond trained (§265 axis) |

## 9. Stages, criteria, labels

All reads are on raw values with no smoothing. Wilson 95% CIs go on every proportion cell, and NEAR is flagged when a bar lies within 2 SE (standing rule 1). For C2 rows a NEAR-SEED flag is added when within σ_seed 0.09 of the bar (rule 5), with each seed's margin plus the seed mean ± SD. Scope wording is "one lineage" throughout.

### 9.0 Stage 0 ($0; pre-stated reads, run only after Audit's ack, two runs byte-identical)

theory/read_dsx_pre.py (to be committed before running) reads:
- (a) of_s1 / of_s2 eval_acc.json by_d['5'..'12'].acc_kprime['8'..'14']. This defines **INIT** = of_s1's pooled d 9-10 at K' = d + 2, with n = 2000. It is also the reference for P1 / P2. of_s2's value is used for dsx_o2.
- (b) of_s1 log.jsonl final-bin ws_readout d 1-4 (the RETAIN WS reference).
- (c) The speed basis: of_s1 args.json hostprobe_sps 0.5668, i.e. 1.76 s/step (already read: DATA).

If INIT >= 0.40, the absolute part of C2-EXT is void, and only "init + 0.20" binds (pre-stated).

### 9.1 Stage 1a — mechanism gate (DSX s1 and CONT s1, `--stop_at 2000`, keep the checkpoint)

Fields: log.jsonl 1k / 2k bins, ws_readout[d][t] (200 TEST2 items per d, no_grad, never fed back).
- **GO:** DSX ws['8']['8'] >= CONT ws['8']['8'] + 0.10 at the 2k bin, AND DSX ws[d][t] >= 0.95 for every d 1-4 and t <= d.
- **PARK:** DSX ws['8']['8'] <= CONT + 0.02 at both bins, OR the d 1-4 clause fails at 2k (DRIFT). PARK stops DSX and CONT. dsf_s1 is not launched. The family closes at Stage 1a, with the result descriptive.
- **NOTE:** otherwise, continue to 4k.

The gate tests the mechanism the next stage needs, the pass-8 compute at d 8 (gate-on-mechanism lesson), not the answer accuracy, which the decoder's last hop can supply.

### 9.2 Stage 1b — DSX s1, CONT s1 resumed to 4k; DSF s1 4k (launched at Stage-1a GO / NOTE)

Evals: the eval_parts line plus long_k128. Binding is per DSX seed, through score_dsx.py:

| Criterion | Field(s) | Bar |
|---|---|---|
| **IN-DEPTH** (mechanism) | eval_acc by_d['8'].acc_kprime['8']; eval_ws_deep by_d['8'] passes 7, 8 | >= 0.80; WS >= 0.85 each |
| **C2-EXT** | pooled eval_acc by_d['9'].acc_kprime['11'] and by_d['10'].acc_kprime['12'] (n 2000) | >= 0.40 AND >= INIT + 0.20 |
| **K'-GAIN** | by_d['10'].acc_kprime['12'] − acc_kprime['8'] (n 1000) | >= 0.25 |
| **RETAIN** | eval_acc by_d['1'..'4'].acc (K' 8); eval_test_perm d 2-4; NOT-DECODER by_d['4'].acc_kprime['2']; SWAP4 eval_swap_kd by_d['4'].swap_mid['4'].follow_U; LP5-D4 eval_lp5h pooled d 3-4 intact − mean_at_t at K' = d | >= 0.98; >= 0.90; <= 0.20; >= max(0.50, floor + 0.30); >= 0.30 |
| **C1-DEEP** (rule 3) | eval_swap_kd by_d['6'], ['7'].swap_mid[str(d)].follow_U (n_pairs >= 500); eval_lp5h by_d['6'], ['7'] intact[d] − mean_at_t[d]; NOT-ONE-PASS by_d['7'].acc_kprime['3'] | each >= max(0.50, floor + 0.30); each >= 0.30; <= 0.20 |
| **C6** | eval_long_k128 by_d['1'..'6'].acc['64'], ['128'] | >= acc['8'] − 0.02 at every d |
| **STABLE** | score_ao.stable on log.jsonl / train_summary.json | guard |
| Descriptive | the K'-curve d 7-12 (K' 8..14); NOT-DECODER d 8 at K' 6; d 11-12; ds_acc_by_d_pass; norm guard s8 / sd2 by d; DSF − DSX deltas; peak GiB; s / step | — |

**Labels per seed (first match):**
- VOID (rerun failed / parts missing → pending until the rerun, then FAIL per §11);
- UNSTABLE;
- DRIFT (RETAIN fails);
- NOT-EXTENDED (IN-DEPTH fails);
- NO-C2 (C2-EXT or K'-GAIN fails);
- C2-ONLY (C2 + RETAIN pass; C1-DEEP or C6 fails);
- **DSX-PASS** (all pass).

CONT and DSF are scored on the same table, descriptively.

### 9.3 Stage 2 (conditional, §0.2): dsx_s2, dsx_s3, cont_s2 at 4k (+ optional dsx_o2)

Arm verdicts:
- **C2 claim** iff C2-EXT, K'-GAIN and RETAIN hold in 3/3 DSX seeds, and neither CONT seed passes C2-EXT. If a CONT seed passes, the label is CONFOUNDED.
- **C1 + C2 bundle claim** iff DSX-PASS in 3/3.
- 2/3 → "2/3, no C2 claim (rule 5 needs >= 3 seeds)".
- <= 1/3 → FAIL; the line is parked.

Claim wording: "answer-only deep supervision on d 5-8 moves the depth horizon of the of loop: K' > K helps at d 9-10, k/3 training seeds, one init, one lineage, plan given".

dsx_o2 is reported beside the result. It does not count toward the 3 seeds, because it is a second init, not a second lineage.

### 9.4 Detach attribution (DSX s1 vs DSF s1; descriptive, pre-stated)

Δ = DSF − DSX, with pooled d 9-10 C2-EXT values.
- **DETACH-COSTS** if Δ >= 0.10. Stage 2 then switches to DSF.
- **DETACH-PROTECTS** if DSF is DRIFT and DSX is not.
- **DETACH-NEUTRAL** if |Δ| <= 0.05 and neither is DRIFT. Stage 2 stays DSX, which is the cheaper and lower-peak arm.
- **MIXED** otherwise.

Single seed each: this reads the mechanism, and is not a claim.

## 10. Scorer (theory/score_dsx.py; fixtures before launch)

Reads:
- `log.jsonl` (1k bins: ws_readout, by_d, ds_acc_by_d_pass);
- `eval_acc.json` (by_d[d].acc, acc_kprime);
- `eval_test_perm.json`;
- `eval_swap_kd.json` (by_d[d].swap_mid[k].follow_U / floor, n_pairs, t_mid);
- `eval_lp5h.json` (by_d[d].intact / mean_at_t);
- `eval_ws_deep.json` (by_d[d][t]);
- `eval_long_k128.json` (by_d[d].acc[k]);
- `train_summary.json` (stable);
- of_s1 / of_s2 `eval_acc.json` (INIT).

Field paths are grepped against the module before commit (scorer field-path lesson).

Modes:
- `--gate2k` emits GO / PARK / NOTE;
- `--final` emits per-seed labels, the arm verdict, the attribution, and `desc_missing` when a descriptive file is absent.

Fixtures (>= 12): PASS; C2-ONLY; DRIFT; NOT-EXTENDED; NO-C2 (bar met but K'-GAIN fails); CONFOUNDED (CONT passes); NEAR (C2-EXT CI covers the bar); NEAR-SEED; INIT >= 0.40 (absolute bar void); pending (missing part); VOID → FAIL after rerun; gate2k GO / PARK / NOTE; DETACH-COSTS switch.

## 11. OC block and censoring

**OC (per DSX seed; "effect" = deep answer-only training lifts d 9-10; "null" = it does not, beyond the trained depth):**
- P(DSX-PASS | effect) ≈ 0.85 (IN-DEPTH) × 0.70 (C2 | IN) × 0.90 (RETAIN) × 0.70 (C1-DEEP) × 0.85 (C6) ≈ **0.32**. P(C2 + RETAIN | effect) ≈ 0.85 × 0.70 × 0.90 ≈ **0.54**.
- Under the null, P(C2-EXT) <= 0.03: +0.20 over INIT at n 2000 is > 9 SE, and the K'-GAIN also has to hold.
- With seed correlation ρ ≈ 0.5, P(3/3 C2 | effect) ≈ 0.54 × (0.54 + 0.5 × 0.46)² ≈ **0.32**; P(3/3 | null) < 0.001. P(3/3 DSX-PASS | effect) ≈ 0.32 × (0.32 + 0.34)² ≈ 0.14.
- With prior P(effect) = 0.5: P(C2 claim) ≈ 0.16; P(bundle claim) ≈ 0.07.
- Stage 1a: P(GO | effect) ≈ 0.75; P(GO | null) ≈ 0.10; P(PARK | effect) ≈ 0.10.
- **Realised $:** reported per cell and per stage against §12's guards.

**Conservative censoring:**
- A run that does not reach 4k, or whose eval crashes, is VOID-RERUN once (fresh host, off-host checkpoint every 1k, keep_final_ckpt for the rerun).
- After one rerun, a criterion whose event was never observed (missing part, unfinished run) counts as **FAIL**, never as missing. The event-only reading is reported beside it.
- There is no extension past 4k: a "still rising" seed is scored at 4k as is.
- No scoring on partial logs. The Stage-1a gate reads complete 1k bins only.
- Binding evals run on cloud hosts only.

## 12. Cost (Vast RTX 3090 ~$0.21/h; base ~1.65 s/step)

Step-time model:
- d <= 4 steps ≈ 1.65 s (of_s1's host probe was 1.76 s);
- extension steps ≈ 1.4× (5-8 op sentences + up to 2 inserts; |T_d| <= 4 decoder calls in place of aux), giving DSX ≈ 1.65 × (0.75 + 0.25 × 1.4) ≈ **1.85 s/step**;
- DSF ≈ 1.9 s/step;
- CONT ≈ 1.7 s/step (the per-bin WS now covers d 1-8).

Per 4k run: ~2.1 h train plus ~1.5 h setup / eval (acc d 1-12 × K' 0..14, swap_kd / lp5h d 3-8, long_k128, ws_deep, stab) ≈ 3.6 h ≈ **$0.75**.

| Stage | Cells | Expected | Guard |
|---|---|---|---|
| 0 | stored reads | $0 | $0 |
| 1a | dsx_s1, cont_s1 to 2k | 2 × ~1.3 h ≈ $0.55 | |
| 1b | dsx_s1, cont_s1 resume 2k → 4k + evals; dsf_s1 4k + evals | ≈ $1.85 | **$3.5 (Stage 1 total)** |
| 2 | dsx_s2, dsx_s3, cont_s2 (+ optional dsx_o2) | ≈ $2.25 (+$0.75) | **$4.0** |
| Family | | ≈ $4.7-5.4 | **$7.5** (<= $30; under the $30 auto-approve rule) |

One VOID-RERUN fits inside each stage guard. If a projection exceeds a guard, dsx_o2 is dropped first, then cont_s2.

## 13. Red team

Pending: an independent sub-agent is required before registration (wave-4 README caveat). Questions for it:
1. Does T_d = {d..K} (which uses d) count as plan-given scope, or is it a step label?
2. Is CONT the right confound control, or should it also see d 5-8 *text* with no loss? (CONT's d law excludes them.)
3. Is the 2k gate's +0.10 on 200 items (diff SE ≈ 0.05) too loose?
4. Is the §6 hold_step fix a semantic change for d 7-8 (den := ‖S_K‖)?
5. Does the ask-item stream at d 5-8 (one-hop questions on S_0 / S_1 / S_2 / distractors, ask_render.py:21-39) leak deep step values? Read: no. Classes i-iv ask v_0, v_1, v_2 (with S_1 given) or a distractor, independent of d.

## 14. Literature (ids verified in repo notes; no new fetches in this rewrite)

- 2506.21734 HRM and 2510.04871 TRM: §1 sources. TRM's 1-step-gradient ablation is 56.5 vs 87.4 with full backprop through the last recursion. "k = 4 of 7 truncation did not help" (GENLAT_VERIFY 48).
- 2603.21676 (Chen, *Thinking Deeper, Not Longer*): per-step intermediate supervision harms extrapolation; supervises only the final output (WAVE4 242). DSX's extension items follow that: answers only. The inherited d <= 4 step aux is the part Chen would predict hurts. CONT vs DSX does not test that. Not claimed.
- 2202.05826 (Bansal, progressive loss; LITERATURE 82, 189): recurrence extrapolation from training with truncated, no-grad prefixes. DSX's no-grad passes 1..d − 1 plus supervised suffix are its closest analogue.
- 2211.09961 (path independence; LITERATURE 148); 2502.05171 (Huginn, truncated backprop; LITERATURE 54); 2409.15647 (Fan, looped length generalisation; LITERATURE 84).
- 2609.03379 (RecurTrace: ACT / PonderNet collapse to one loop; WAVE4 246): why M6 is not adopted.
- 2604.01577 (fast-slow latent recurrence, train-short test-long; LITERATURE 155): the M5 relative, deferred.
- Novelty: the repo notes contain no claim that deep supervision with detach has been applied to a pretrained looped LM with an external DeltaMem state. Literature should check this before any wording beyond "transplant".

## 15. Unresolved

1. **hold_step d >= 7** (inject_aw.py:170) needs `min(d + 2, K)`. This is a code change beyond the flags, bitwise for d <= 6. Audit should rule.
2. **Second lineage** (rule 5: >= 2 lineages once §268 lands). DSX has one init lineage. dsx_o2 is a second init, not a lineage.
3. **d in the loss index set T_d:** plan-given scope vs label. The plan-free alternative (DS at every t through an S-only decoder) depends on design 19's S-only result.
4. **Ordering with §265 DENSE-DISTRACT:** same horizon, different lever. If §265 lands first and passes, its ddp model is the stronger init / comparator. The Stage 0 bars would be re-stated before any DSX output.
5. **Wrong-state credit (§5 (b)):** DSX cannot repair an earlier-pass error and may push the decoder shortcut at pass d. NOT-DECODER d 8 is descriptive only.
6. **The warm-restart cosine** (lr 1e-4 on a converged model) can by itself move d 5-8. CONT controls it at d <= 4 data only.
7. **HRM / TRM ids are not in docs/LITERATURE.md** (only in the LIT_* notes). Literature to add.
8. **of_s1 model.pt sha prefix** is not in host_shas.json; it must be computed and asserted at build.
9. **Independent red team** pending (§13).

## Appendix A. Deferred: the 11-03 draft's DSR formation arms (askname init, C4)

The draft proposed per-pass answer CE with detach and an S-only decoder from s233_askname_s1 (AO recipe, d <= 2, 12k), with SAO / DST controls, a TRM BCE halt head, and n = 2 inner micro-passes on a target-cue arm. Deferred, for these reasons:
- **(i)** Under an S-only decoder, the deep-supervision target at t < d is near-uniform (bijections; §2 M1). So for formation, DS reduces to CE at t >= d, which is the AO / AEX loss plus the detach. The AO line's measured failure on new seeds is basin entry (Note 255-1), and detach does not address entry.
- **(ii)** The S-only decoder is design 19's lane.
- **(iii)** §267 EXACT-EM-11 is the registered C4 formation route (no step value in the loss).
- **(iv)** Inner micro-passes fail M4's precondition, and the halt head adds nothing over the plan clock (M6).

If design 19's S-only result and §267 both land, the draft's Stage A (DSR vs SAO at 4k, NO-DRIFT gate on probe u1|v1) is the natural follow-up. Its formation criteria are kept in git history (the 11-03 draft text).
