# Teaching latent reasoning from discrete data: what our proposals showed, what the field does, and eight new methods (2026-10-02)

Literature session, at the user's direct request ("read through the results of all past proposals… do deep research on
new ways we can teach latent reasoning from a discrete dataset").
- Sources for our results: `docs/THEORY.md` (§§6–85, executive summary), `docs/PROPOSALS.md`, `docs/NORTH_STAR.md`,
  `report/findings.md`, `docs/GSM8K_REPRO.md`, `docs/CURRENT_TASKS.md`.
- All arXiv IDs verified on 2026-10-02 (Semantic Scholar batch or arXiv API).

"Discrete dataset" here means data whose supervision is discrete: final answers, plus (optionally) text reasoning steps
or written states, as in GSM8K-Aug, permutation tracking and ProsQA.

## 1. What our proposals showed (≈ 85 sections, condensed to seven rules)

**Successes:**

| Result | Where |
|---|---|
| Anytime per-step answer supervision: 8/10 extrapolate vs 3/9 | Pilot D, §17 |
| F1 on GPT-2 with per-pass targets: exact to d14 (from final answers only: 0/3) | §54b |
| GT-distilled loops reach 2× training depth (17/20) | §37 |
| Restart (re-anchoring) teacher distilled label-free: 19/20 failures fixed | Pilot L, §32 |
| Second label-free rung: 12/20 | M-2, §38 |
| Written-state restarts: S3 0.995 | §64 |
| GPT-2 chunked written state: 1.000 at 17–32 ops (text CoT 0.14) | §66 |
| S5 certified exact to 512 and 2,048 | §71 |
| Edited written checkpoints change answers as predicted (IIA 0.84 set / 1.00 single) | CLR-1, §45 |
| Discrete latent + exact marginalisation: GPT-2 S3 10/10 once features are standardised | §73 |
| Free table: S4/A5/S5 10/10 when length-1/2 problems are present | §76(A) |
| Published step-supervised GPT-2 latents carry work: SIM-CoT+Coconut 0.448 vs 0.051 with no thoughts; CODI 0.425 vs 0.221 | GSM8K_REPRO |

**Failures:**

| Result | Where |
|---|---|
| Final-answer-only acquisition: F1-S 1/3, then 0/10 fresh seeds | §61, §61b |
| §67: 1/6 at 6 codes | §67 |
| RL-style 0–1/10 | §69 |
| Lengths ≥ 3 only: 0/10 on every order-60 group | §76, §82 |
| The gradient norm at init is the same for short and long batches (≈ 0.0055 at lengths 1–8), so the failure is direction or symmetry, not vanishing gradient | §85 (iii) |
| Continuous loops past their range: S3 one loop 0.03 at 13–16; even true labels give only 0.66/0.06 | §64, §65 |
| Coconut GPT-2 on ProsQA is a one-pass conduit | G, J, Q |
| CODI and SIM-CoT peak at their trained 6 latents | §13 |
| Jitter, clocks, clamps, step-adversarial training and age transplants don't fix it | §§23, 28, 36, 39, 58 |
| No in-pass composition: held-out table rows at chance; unseen k-op words 0.03–0.12 | §76(B), §80 |
| Paraphrase transfer partial or null | §74, §79 |
| Every shared-parameter cell fails where the table succeeds | §75, §77 premise, note 81-0 |
| Halting keys on counts: WS-pad stops after 2 values on 164/225 long items | §78/§84 |
| A reward-learned halt stops at a constant 2 segments | §50 |
| Benchmark shortcut: ProsQA proximity cue 0.911; latent arms at chance without it | §57, PROTOCOL8 am. 3 |

**Seven rules these imply:**
1. **Per-step targets are load-bearing.** Every success had a per-step target that was either given (labels, text,
   short problems) or produced by an in-range teacher (restarts). Every final-answer-only attempt failed. This is
   Wies et al. 2204.02892's theorem in practice. Kim & Suzuki 2410.08633 show the same for parity.
2. **Range.** Any quantity that grows with the problem (latent step count, history length, value count, context)
   must stay inside its trained range. Written-state restarts do this; one continuous loop can't, even with true labels
   (§65).
3. **Discrete or snapped states don't drift** (§67 hard codes, §73). Continuous loop states lose validity (§64).
4. **No in-pass composition at our data scale** (§76(B), §80). Externalising each step, in text or as one latent per
   step, works.
5. **Stop decisions must not see a count** (Prop. 10, §78/§84).
6. **Condition pretrained features** (anisotropy; §72b/§73).
7. **Audit benchmarks for shortcuts before claiming latent work** (ProsQA).

## 2. How the field teaches latents from discrete data (2024–2026)

| Family | Representative (verified) | Supervision from the discrete data | Fits our rules? |
|---|---|---|---|
| Curriculum replacement of text steps | Coconut 2412.06769; ICoT-SI 2405.14838; Distill-S2→S1 2407.06023 | text steps, removed over stages | Rule 1 weakly (steps vanish); our GPT-2 runs are one-pass conduits |
| Self-distillation of hidden states | CODI 2502.21074 (answer-token alignment); KaVa 2510.02312 (compressed KV-cache trajectory alignment); System-1.5 2505.18962; SemCoT 2510.24940 | teacher states, per step for KaVa | Rule 1 (dense) ✓; no range control |
| Per-latent step decoding | SIM-CoT 2509.20317 (auxiliary decoder per latent); PMPS 2609.09928 (prototype-mediated soft alignment of latents to unequal-length steps); 2606.20075 (generative reconstruction > geometric imitation; latent–step mutual information predicts accuracy) | each latent must decode its step | Rule 1 ✓, plus C7-style readout |
| Discrete codebooks for steps | Planning tokens 2310.05707; Token Assorted 2502.03275 (VQ-VAE codes for early steps); CIRF 2605.28292 (one functional token per reasoning unit); Abstract-CoT 2604.22709 (reserved vocab, bottleneck-by-masking ↔ self-distillation loop, then RL); DLR 2606.29712 (rendered-CoT clustering); Latent-SFT 2510.15522 (latent vocab within the pretrained vocabulary, Gumbel) | step segmentation into codes | Rules 1 + 3 ✓; none certifies or restarts |
| Latent-variable / EM over thoughts | TRICE 2312.02179; BoLT 2503.18866 (EM bootstrapping of latent thoughts); Latent Thought Models 2502.01567 (variational posterior over thought vectors); LaDiR 2510.04573 (VAE step blocks + latent diffusion) | final answers plus a learned posterior | Rule 1 only via the posterior; our §67/§76 show EM needs short footholds |
| Loop / iteration alignment | RELAY 2502.08482 (loop iteration ↔ CoT step, then distil into AR); TRM 2510.04871; Saunshi 2502.17416; Ouro 2510.25741 | per-iteration step labels | Rules 1 + 2 (looped length generalisation) ✓ |
| Frozen LM + recurrent latent proposer | LRT 2609.01117 (small recurrent reasoner refines latents fed to a frozen LLM; answer-only supervision); PLaT 2601.21358 (planner + separate decoder, answer-aware textual stopping) | answers; decoder readout | Rule 6 (frozen LM) ✓; halting via text |
| Fidelity loops | ReLAT 2606.06252 (question → latent → question reconstruction at test time) | the input itself | the C7 round trip as an objective |
| RL after warm start | HRPO 2505.18454; Abstract-CoT's RL stage; LTO 2509.26314 (latent reward model) | rewards | Works only after supervised warm-up, consistent with our §69 |

**What nobody does** (gaps that match our rules):
- (a) restarts or history cuts *in latent space* with a step-level target;
- (b) certificates on learned codes;
- (c) a measured threshold for how much step supervision is needed;
- (d) count-free halting;
- (e) shortcut-audited, depth-beyond-one-pass evaluation.

## 3. Eight new methods (ranked by expected information per GPU-hour)

Each entry gives what to do, why (our rule and evidence), the closest prior, a first test, and a kill criterion.

### M1. Latent restarts distilled from the certified written-state solver ★ (rules 1–4, 7)
- **What.** The teacher is §71's WC solver (S5, exact to 2,048). The student is GPT-2 that, inside each chunk, carries
  the state as **one latent per op**, not text.
  - Each latent gets a SIM-CoT-style auxiliary decoder loss onto the teacher's written state at that op.
  - Every k ops, the latent is **snapped** to the nearest codebook entry (the 120 S5 states) and the history is cut:
    a latent restart.
  - Text appears only at the end.
- **Why.** It keeps everything that worked (per-step targets, range, snapping) and asks the one open question: can the
  carried state be latent, with C1 measurable as removal of the latents → chance?
- **Prior.** SIM-CoT, KaVa and RELAY supervise steps but never restart. CLR-1 restarts in text. **New as a
  combination.**
- **First test.** S5 under DeltaProduct's protocol (train 128 / test 512), 3 seeds, local GPUs.
- **Kill.** Accuracy at 512 < 0.9, or removing the latents doesn't hurt (bypass).

### M2. Step codes from the discrete trace, certified (rules 1, 3, 6)
- **What.** Assign every text step a discrete code. Use the exact symbolic state where it exists (S5 element, ProsQA
  frontier), else VQ or clustering of teacher hidden states at step boundaries (Token Assorted, CIRF, Abstract-CoT).
  - Train GPT-2 (standardised readout, §73) to emit codes instead of steps.
  - Certify the code→state map exhaustively where states are finite, and run the C7 edit test on codes.
- **Why.** Codes give §76-style short labels from real data (every step labels a transition) while staying latent.
- **Prior.** Planning tokens, Token Assorted, CIRF, Abstract-CoT. None certifies or edit-tests codes. **New:** the
  certificate and edit test.
- **First test.** S5 and S3 boxes (finite states), then GSM8K with codes for the *operation* and value tokens kept
  as text.
- **Kill.** Code certificate < 1.0 on finite tasks, or the edit IIA doesn't beat §45's text checkpoints.

### M3. The step-supervision threshold on real data (rule 1; extends §85)
- **What.** Keep text or code targets on a random fraction p of steps per instance, and latent-only on the rest. Use
  exact marginalisation over codes where they are discrete. Sweep p ∈ {1, 0.3, 0.1, 0.03, 0.01}.
- **Why.** §85(i) measures the threshold on the free table. M3 asks the same of an LM on S5 and GSM8K. The answer
  says how far "from a discrete dataset" can go towards "from final answers" (C4).
- **Prior.** Wies 2204.02892 (theory), ICoT-SI (removal schedule, but all-or-nothing per step). **New:** the
  threshold curve.
- **Kill.** None needed. It is a measurement; report p* with CIs.

### M4. Random-start, tight-window latent training (rule 2)
- **What.** Cut long discrete traces into windows: start at a random step t, give the teacher's state at t (encoded or
  snapped), run exactly k latent steps (tight, no slack), and supervise the state at t + k.
  - The latent loop never sees an absolute step count beyond k.
  - At inference, chain windows with snapped restarts.
- **Why.** §65: one loop with true labels still fails out of range. Ruiz & Gu 2507.02782 show that state coverage
  (starting from other sequences' states) fixes recurrent length generalisation. Prop. 4: tight budgets identify unit
  speed.
- **Prior.** Ruiz & Gu (continuous LRNNs, no discrete restart); TBPTT. **New:** windowed latent training with snapped
  restarts on a pretrained LM.
- **First test.** S3/S5 single loop vs §65's 0.66 at 9–16.
- **Kill.** No gain over §65 at 9–16 and 17–24.

### M5. Count-free halting from discrete traces (rule 5; generalises §84)
- **What.** Train the stop decision on windows of long traces (M4's sampler), where the target is "work remaining =
  0", never on whole short problems. Every training window has a small absolute value count but a varied position in
  the problem.
  - If only short traces exist (the §78 split), use §84's decoys, padded to the test range.
  - Add PLaT-style answer-aware stopping as a baseline.
- **Why.** The §78 failure is Newman 2010.07174's EOS-length attractor. Windows decorrelate count from stop, as
  randomised PE does for position (Ruoss 2305.16843).
- **Kill.** Oracle-stop (§84 Step 1) recovers < 0.10. Then halting isn't the bottleneck.

### M6. Round-trip latents: decode → re-encode consistency as a training loss (C7 by construction)
- **What.** For each latent, decode it to step text (or a code) with the auxiliary decoder, re-encode that text with
  the model, and add ‖h − h′‖ (or a KL on the answer). Optionally add ReLAT's question reconstruction as an
  input-anchored term.
- **Why.** 2606.20075 finds generative reconstruction preserves latent–step mutual information better than geometric
  imitation, and that this information predicts accuracy. Our C7 test (§34: round trip 0.947; F4 failed to train)
  becomes the objective.
- **Prior.** SIM-CoT (decode only), ReLAT (test-time, question only). **New:** a training-time round-trip loss on
  steps.
- **Kill.** Round-trip agreement doesn't beat the input-only control (2509.13316's warning).

### M7. Semi-supervised EM over discrete step codes on GSM8K (rules 1, 3; real data)
- **What.** Treat each GSM8K step as a discrete latent: operation × operand slots, with values computed by a
  calculator or written. Run exact marginalisation (or beam EM) when the step text is hidden, and supervise when shown
  (M3's p). Initialise from the D&W CoT checkpoint.
- **Why.** Our exact-marginal machinery works when footholds exist (§76 std). Text-labelled steps supply them on real
  data. BoLT and TRICE show EM over thoughts helps, without certificates.
- **First test.** The §78 split, measured by step count, with ICoT-SI and text-CoT arms.
- **Kill.** Below text CoT at matched data on in-range steps.

### M8. Loop-aligned composition: per-iteration step targets for in-pass composition (rule 4)
- **What.** Loop GPT-2's middle block k times, aligning iteration i with step i of the trace (RELAY). Then test
  unseen k-op words (§80(b)) and held-out (state, op) rows (§76(B)).
- **Why.** §80 shows composition doesn't emerge from end-state targets. Saunshi 2502.17416 and RELAY suggest
  iteration-aligned depth can supply it.
- **Prior.** RELAY (does alignment; doesn't test held-out composition). **New:** the held-out composition test.
- **Kill.** Unseen-word accuracy stays below 0.5.

## 4. Suggested order
1. **M1, then M2** on S5/S3. They reuse the certified §71 solver and the §73 cell, take one week of local GPUs, and test
   C1 + C2 + C7 in the one setting where everything else already works.
2. **M3 and M4** (cheap ablations that set the supervision and range budgets).
3. **M5** once §84 Step 1 reports.
4. **M7 and M8** on GSM8K and composition, once M1–M4 fix the recipe.
5. RL only as a final stage after a supervised warm start (HRPO, Abstract-CoT), never from scratch (§69).

**Evaluation rules for all eight:**
- removal and bypass tests (latents → 0);
- shortcut-audited data (ProsQA-NA, generator checks);
- per-seed reporting (≥ 5 seeds);
- the C7 edit test on whatever is carried.

## Addendum (2026-10-02, after Audit's review of 685bef0): scope and NORTH_STAR tags

Audit's point is right. In M1 the state space (120 true S5 states) and the per-op labels are **given**, so the latents
are a supervised carrier. A removal test can show that the carried latents are *used* (C1 as NORTH_STAR defines it:
removal → chance, step-specific blocking shows serial use). It cannot show they were *discovered* from final answers
(C4).

| Method | Conditions it can bear on | Explicitly not |
|---|---|---|
| M1 latent restarts from the WC teacher | C1 (use: removal and step blocking), C2 (512/2,048), C6 (snapping), C7 (decoder plus edit) | **C4**: states and per-op labels are given |
| **M1-U (new, the C4 variant)** | C1 + C4 | — |
| M2 certified step codes | C6, C7 (certificate, edit) | C4 (codes come from labelled steps); C1 only if a removal test is added |
| M3 supervision threshold | the C4 boundary (how far below per-step labels one can go) | — |
| M4 random-start windows | C2, C6 | C4 (window targets are teacher states) |
| M5 count-free halting | C5 | — |
| M6 round-trip loss | C7 | — |
| M7 semi-supervised EM on GSM8K | C3; partial C4 (labelled fraction p < 1) | — |
| M8 loop-aligned composition | C1 (in-pass work) | C4 (iteration-aligned step labels) |

**M1-U, the C4 ablation.**
- **Setup.** The same architecture as M1, but with a **learned** codebook of size M ≥ 120 (e.g. M = 240, cf. §76's
  M = 2|G|). There is **no per-op decoder loss**; supervision is the final state only, with exact marginalisation over
  codes at snap points.
- **Kill line.** Certified in < 4/5 seeds at 512 → no C4 claim for the M1 family.
- **Prior expectation from rule 1** (§61b, §76/§82 ge3, §85): it fails unless short problems supply footholds. So run
  two curricula:
  - with length-1/2 problems (std), which bears on C4 only in §76's qualified sense;
  - lengths ≥ 3 only, the strict C4 test.
- Report both, with M3's threshold p* as the bridge.
