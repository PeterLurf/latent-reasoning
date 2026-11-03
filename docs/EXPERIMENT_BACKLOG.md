# EXPERIMENT BACKLOG (living, ranked; owner Theory)

Started 2026-11-03 from 11 parallel research passes:
- 5 Opus agents: C1/C4 training; C2/C5/C6 depth; C3/C7; architecture; scale and task generality.
- 6 Fable agents: C1 order-free and plan-free selection; C7 editable latents; the GPT-2 medium WS-vs-aux divergence; C4 by RL and latent EM; a red team of the scorecard; natural-language task bridges.

**arXiv ids:** each research agent fetched arxiv.org/abs/<id> and checked that the title matches. Literature spot-checks them.

**Rules:**
- Registrations go to Audit one at a time, from the top of §1.
- $0 reads follow the pre-statement → push → Audit ack → compute rule. Binding evals run on cloud hosts only (local RAM bit-flips).
- After every scored result, update the ranks and the `Status` column and add a changelog line.

**Score** = prior × value / max(cost of the first falsifying stage, $0.25).
- Value by north-star priority: C1 5, C6 4.5, C2 = C4 4, C3 3, C5 2, C7 1, and 5 for a "small-benchmark SOTA" (user: beat SOTA on smaller tasks before GSM8K).
- For a diagnostic, the prior is P(the result changes a decision), not P(success).

**Baseline:** bun_s1/s2 (§259), plus §263 ORDER-FREE (of_s1/s2), which is running.
- **C4 caveat (C1/C4 agent):** the §262/§263 one-hop co-training items for S_2 | S_1 are rewritten with simulator values. ORDER-FREE is therefore not a C4 route, and any C4 version needs prefix problems (K3).

## 1. Ranked queue (top of backlog)

| # | ID | Experiment | Targets | First stage, $ | Score | Depends on | Status |
|---|---|---|---|---|---|---|---|
| 1 | X1+X2 | CUE-CLOCK + FREEZE-vs-FIXED-POINT: are C5 / C6 plan-clocked? | C5, C6 scope | eval $0.2 | 12.6 | – | queued |
| 2 | X5 | ONE-HOP-PIPELINE baseline + swap-at-t curve | C1 framing | eval $0.2 | 10.0 | askname ckpt | queued |
| 3 | X4+D3+D5+E1+H5 | C2 diagnostic bundle: first-error pass, length-matched d4, K′12/16, centroid snap, re-sweep, contrast decode, latent beam | C2 | eval $0.3 | 8.0 | – | queued |
| 4 | M1 | ECHO-GATE: medium's pass-2 write vs echo (re-gate §258) | scale, C1 | eval $0.3 | 6.7 | p2m_s1b model.pt | queued |
| 5 | D1-S1 | DENSE-DISTRACT: 0–8 inserts decouple op-sentence count from depth | C2, C1 | 2k $0.2 | 6.4 | – | queued |
| 6 | P1 | SELF-CUE: the loop names its next entity (plan-free at test) | C1, C5, C7 | 2k $0.35 | 5.0 | §263 read | after §263 |
| 7 | K3-S0 | PREFIX-KL: the model's own d = t prefix answer anchors the onset code | C4 | 2k $0.3 | 4.7 | prefix builder | queued |
| 8 | P2 | BACKWARD-PLAN: parent lookup from S_d to a given, then forward | C1, C5, C7 | 2k $0.4 | 4.4 | §263 read | after §263 |
| 9 | P3 | TARGET-CONTRAST: §261 target cue + hold target on off-target reachable ops | C1, C5 | 2k $0.35 | 4.0 | §261 unpark | parked dep |
| 10 | A1 | SENTENCE-LOCAL POSITION IDS (order-blind encoding) | C1 order | S0 $0, S1 $0.5 | 3.5 | value falls if §263 passes | after §263 |
| 11 | R-bundle | C7 $0 reads: TEXT-FREE READOUT, DAS-EDIT on bun, CODEBOOK / STATE-ENCODER edit, TEXT-CHECKPOINT RESTART | C7, C1 | eval $0.3 | 2.5 | Audit ruling: does a codebook count as re-encode? | queued |
| 12 | X3 | TEXT-CoT-D4 same-data baseline (+ Coconut-KV arm) | C2, C3 | $1 | 2.4 | §227 trainer on REG | queued |
| 13 | S1 | BASE READINESS LADDER (probe 8 bases; valid only if medium < small) | scale | $1 | 2.1 | – | queued |
| 14 | M2 | COLON-CE on medium | scale, C1, C2 | $1.3 | 1.5 | M1 | after M1 |
| 15 | K1 | EXACT-EM-11 (discrete value write + forward-DP marginal likelihood) | C4, C7 | stage A $2 | 0.9 | categorical head | queued |
| 16 | N1 | SHUFFLED OBJECTS (BBH tracking) chain | small SOTA, C2 | $4 | 0.69 | step-index cue | queued |

Sequencing:
- **X1 decides the C5 line.** If C5 / C6 are plan-clocked, C5 work moves to plan-free settings (P1 / P2 + H2 source-reached halting) and the in-pass halts (H1) drop.
- **§263 decides A1:** if ORDER-FREE passes, A1 becomes a robustness arm; if it fails, A1 is next for C1.
- **The four eval cells (#1–#4: X1+X2, X5, the C2 bundle, M1) can share one cloud host.**

## 2. Diagnostics and scorecard red team (X)
Key fact: the per-pass cue is spine[min(t, d)], so after pass d the cue repeats S_d. The hold was trained on exactly those passes, and C5 halts when the answer is stable. The scorecard does not separate "the model stopped computing" from "the plan stopped changing".
- **X1 CUE-CLOCK** — C5, C6, C1.
  - Feed EARLY-REPEAT cues (S_1, S_2, S_2, …) and LATE-FRESH cues (a new distractor name each pass after S_d). Read rel_inc, ‖S‖ and pred per pass.
  - If the freeze follows the cue, C5 / C6 are re-scoped to "under a repeated cue".
  - Prior: 2607.20519 (halting readouts vs trajectory), 2606.29983 (spurious loop-length correlation), 2310.02226 (pause tokens). Cost $0.2.
- **X2 FREEZE-vs-FIXED-POINT** — C6.
  - Cycle the cues S_1..S_d repeatedly to K′64. Idempotent writes mean C6 is a property of the loop; collapse or ‖S‖ growth means it is a freeze. Bar: ≥ K′8 − 0.02.
  - 2509.23314, 2409.15647. Same cell as X1.
- **X3 TEXT-CoT-D4** — C2, C3.
  - A GPT-2 text-CoT trained on REG d ≤ 4 with the same distractor mix.
  - Latent-specific C2 survives only if text d 7 / 8 < bun − 0.10. A venue will also demand a Coconut-KV arm.
  - Merges T6 (the matched-FLOP extrapolation framing, plus majority vote) and S2 (the compute-matched protocol: FLOPs, serial layers, wall-clock at bs 1 / 64, extra positions; a pause-token arm at matched positions).
  - 2412.06769, 2502.21074, 2404.15758, 2412.13171, 2408.03314, 2512.21711. Cost $1 (+$2 KV arm, +$1.5 filler arm).
- **X4 LENGTH-NOT-DEPTH** — C2.
  - First-error pass on d 8 (concentrated at 5–7 = untrained-pass compute; uniform = accumulation; at 1–2 = length).
  - d 4 padded to d 8 length.
  - d 8 at K′12 / 16.
  - 2409.15647, 2507.02199. Cost $0.2.
- **X5 ONE-HOP-PIPELINE** — C1, C3.
  - Run the askname model as a text-substitution pipeline over d 1–9, plus a full swap-after-every-pass follow curve at d 4–7.
  - If the pipeline gets d 7 ≈ 1.0, C1 is worded "latent value transport under an external program".
  - 2602.08783, 2607.06648, 2512.21711. Cost $0.2.
- **X6 PROBE TRANSFER to passes 5–7** — C1, C2, C4 wording.
  - Apply the pass-1..4 linear readout, without a refit, at passes 5–7 of d 6–7: one mechanism or a different one?
  - Also re-score H-CHAINSRC-OP / H-LIVE on bun; the SOTA row inherits them from dch.
  - 2602.22441, 2606.01243. Cost $0–0.2.
- **X7 STATS-AND-SEEDS** — all; needed for publication.
  - The two seeds share the rcb→dis→dch lineage.
  - Report bootstrap CIs, frames 6–7, held-out names, and a train/test overlap check.
  - Plus 3 final-stage seeds ($3) or 1 full-lineage seed ($5), after §263. 2505.20896.

## 3. C1 scope: plan-free and order-free selection (P, A1)
Base recipe: the §262/§263 build.
- **P1 SELF-CUE**
  - Mechanism: a second ':' question, "Which entity is next?", produces a name that fills the next pass's cue. It is teacher-forced in training and argmax at test. The loop stops when it emits the question's entity.
  - Scope: plan-free at test only; the plan names are training labels, a C4 cost.
  - 2601.21358 (latent planner), 2407.20311 (iGSM pre-planning), 2606.03741 (subgoal persistence).
  - Pilot: 2k from bun_s1.
    - GO if free-running d 3 ≥ 0.8 and stop-at-S_d ≥ 0.8.
    - PARK if name accuracy at pass 2 < 0.6.
    - Fallback: scheduled sampling. Cost $0.35.
- **P2 BACKWARD-PLAN**
  - Mechanism: phase 1 asks "Who does X depend on?" from S_d until it reaches a GIVEN; parent names are derivable from the text. Phase 2 computes forward with a generic cue. This gives goal anchoring and a natural stop.
  - 2402.11917, 2508.08222, 2212.13894 (LAMBADA), 2411.19865.
  - Pilot: 2k, K = 2 × 4. GO if phase-1 accuracy ≥ 0.9 and H-EXTEND ≥ 0.6. Cost $0.4.
  - Answer-reward variant (**K-RL P2r**, REINFORCE pointer + STOP, group 8; 2205.09712, 2305.04091; $2.5) is the C4 version.
- **P3 TARGET-CONTRAST**: the §261 target cue, plus live-chain inserts that get the hold target.
  - 2302.00093, 2502.13913, 2311.09277, 1906.07132.
  - Pilot: 2k; GO if H-EXTEND and H-LIVE ≥ 0.6 and d 3 ≥ 0.9. Cost $0.35.
- **P4 ANCESTOR-SET PRE-PASS**: a multi-hot ancestor set (BFS superposition) as a goal filter.
  - 2505.12514, 2412.04703.
  - Pilot: a $0.05 probe on gb_s1, then 2k. GO if F1 ≥ 0.9 at d 4 and H-LIVE ≥ 0.6. Cost $0.4.
- **P5 POINTER HEAD**: query = the ':' state, keys = the sentence-final '.' positions; the pointer gates the write source, so the head is forced into use (NK lesson).
  - 2505.20896, 2402.11917.
  - Pilot: GO if pointer accuracy ≥ 0.9 on permuted items and the ablation drop ≥ 0.2. Cost $0.4.
- **P6 LATENT LOOKAHEAD**: predict the next ':' state; a consistency score rejects off-chain steps.
  - 2603.20219, 2502.04404.
  - Pilot: a $0.05 AUC read on the gb dumps first. Cost $0.3.
- **A1 SENTENCE-LOCAL POSITION IDS**: position ids restart per statement, so selection must go through content. This attacks pass-1 positional selection, which NK did not.
  - 2305.19466 (NoPE), 2405.20671, 2405.18719 (CoPE).
  - Pilot:
    - S0: zero-shot on bun, $0.
    - S1: 2k perm-then-insert; GO if conditional pass-t WS on permuted items ≥ 0.6 (bun 0.20–0.37) and TEST2 ≥ 0.95.
    - Arm b: per-sentence position-block shuffle. Cost $0.5.

## 4. C4: answer-only formation (K)
Mechanism to beat (Notes 251-8..11): on d > t items the pass-t ':' code drifts at onset; on d = t items it holds at 1.0. ACUE and GATE-CURRIC did not stop the drift.

Structural gift: 11 is prime, so every op (+c, ×c with c ≠ 0) is a bijection on Z_11.
- **K1 EXACT-EM-11**
  - Mechanism:
    - An 11-way categorical write head, initialised from the tied LM rows for ' 0'..' 10'. The memory is reset to emb(v_{t−1}), so each pass is Markov.
    - The loss is the marginal likelihood by forward DP: 1 + 11(d − 1) conditional passes.
    - The gradient is posterior − prior, and at init the posterior equals the prior (v_1), so there is no onset drift.
    - Because the ops are bijections, the posterior collapses inductively.
  - Qualifier: the state space is designed (no per-item labels, but not a task-agnostic state).
  - 2312.02179 (TRICE), 2512.20169, 1805.10872, 2212.12393, 2503.18866.
  - Pilot:
    - Stage A, the AO-REP recipe at d_max 2 with 2 seeds: WS d 2 pass 1 ≥ 0.8 at 1k / 2k (AO 0.09–0.27), with the K′1-vs-K′2 shortcut exclusion. Cost ~$2.
    - Stage B: d ≤ 4 extrapolation. Cost ~$10.
  - Control: **K4 CONTINUOUS-WRITE GRPO** (noise on the write; $0–1).
- **K2 SAMPLED-VALUE RLOO / Gumbel-ST** on K1's head: the version that scales to large value spaces. The gradient cosine against the exact EM gradient serves as an oracle.
  - 2511.06411, 2509.19170, 2604.27998, 2403.09629. Cost $4, after K1.
- **K3 PREFIX-KL**: KL(readout_t(full) ‖ sg readout_t(depth-t prefix problem)), plus answer CE on both. The model's own d = t computation anchors the onset code.
  - 2502.21074, 1703.01780, 2502.01612.
  - Pilot:
    - S0: the AO recipe + prefix_kl, 2k. GO if WS d 2 pass 1 ≥ 0.8 and ≥ CTRL + 0.3. Cost $0.3 on cloud.
    - S1: from aw_plaw2 with aux off, WS d 3 pass 2 (AEX0 0.22). Cost $1.
- **K5 DISCRETE-GRPO**: sample v_t, write W·e(v_t). 2604.27998, 2512.11816, 2608.01593, 2402.03300. Cost $1.5; needs the discrete tied write (ARCH_UPGRADES #5).
- **K6 HARD-EM, BIJECTION BACK-SUBSTITUTION**: keep a pseudo-label chain only if the model's own next-pass conditionals reproduce the answer. Unchanged architecture.
  - 2411.04282 (LaTRO), 2512.20169.
  - Pilot: precision ≥ 0.995 at d 3; §103's cap was 0.98. Cost $3.
- **K7 LATENT-STaR / SELF-LABEL DEPTH LADDER**: label d 5–6 items with bun's own answers (stable at r = 3 and consistent across distractor re-draws), plus its own per-pass readouts.
  - 2203.14465, 2312.06585, 2502.01612, 2503.18866, 2608.11233.
  - Pilot: 8k; d 9 ≥ 0.6 (bun 0.19). Cost $1.2–2. Re-read SELFDEPTH / ITERATED_SELFTRAIN first. Also counts as C2 (D7).
- **K8 ANSWER-ONLY INIT LADDER**: the ':' readout is trained only on d = 1 final answers, then K3. 2609.19717, 2602.01148, 2405.14838. Cost $2.
- **K9 ANSWER-ONLY DEPTH CURRICULUM from bun** (aux off, d 1→4).
  - PARK if WS < 0.2 by 2k. Cost $0.35. Overlaps the K3 control.
- **K10 DISCRETE→CONTINUOUS ANNEAL**: gated write emb(v) → h, after K1. 2505.18454 (HRPO), 2505.16552. Cost $2.
- **K11 PARAPHRASE CONSISTENCY (UDA)**: unlabelled frames, KL to the template readout. 1904.12848, 2001.07685, 2502.19907. Cost $3.
- **K12 RECON ANCHOR**: low prior (it pins the sentence, not the value). 2606.20075, 2509.20317. Cost $0.5.

## 5. C2 / C6: depth horizon and stability (D)
Facts:
- d 6 0.97, d 7 0.86, d 8 ~0.5.
- §241: the op-sentence count hurts, not length.
- §243 DEPTH-CURRIC failed.
- OFFSET was withdrawn (241-0).
- ARCH_UPGRADES #2, the decay-gated state, is still a candidate.

Entries:
- **D1 DENSE-DISTRACT**: d ≤ 4 with 0–8 inserts.
  - 2606.29983, 2409.15647, 2604.07822.
  - Pilot:
    - S1, 2k: GO if d 8 passes 1–4 WS ≥ 0.97 and accuracy ≥ bun + 0.10.
    - S2, 2 × 8k: d 8 ≥ 0.75. Cost $1.4.
- **D2 HOLD-INTERLEAVE**: re-ask the previous cue mid-chain, which decorrelates pass index from step.
  - 2202.05826, 2502.05171.
  - Needs an aux / hold re-index build plus a p = 0 equivalence gate. Cost $1.4.
- **D3 CENTROID-SNAP** ($0): replace writes with per-value, per-family centroids.
  - Tells drift from selection. 2510.14095, 2610.11570.
- **D4 CORRUPT-AND-RECOVER**: with p 0.1, pass t−1 gets a wrong-value centroid write.
  - 2510.14095, 2605.21488; 2502.05171 is counter-evidence. Cost $1.4.
- **D5 RE-SWEEP** ($0): run the cue sequence a second time; report the flip matrix. 2602.08100, 2211.09961.
- **E1 CONTRAST-DEPTH** ($0.3): logits(S_K′) − α·logits(S_K′−r), or state extrapolation; then self-distil.
  - 2610.02185 (LoopCD), 2602.11451 (LoopFormer).
- **H5 LATENT BEAM + ANSWER-LL VERIFIER** ($0): 16 perturbed writes, beam of 3. 2505.13308, 2510.07745.
- **D6 RANDOM-START MEMORY**: S_0 = another item's S_K. 2211.09961, 2605.21488, 2410.07145. Cost $1.4.
- **D7 = K7.**
- **D8 NORMED RECALL READ**: a second read at block 3, with RMSNorm on the read outputs. 2604.15259, 2604.11791, 2609.36636. Cost $1.4; only if D3 points to scale.
- **D9 TRAJECTORY REWARD**: Δlog p(a) per pass. 2602.10520 (RLTT), 2603.19714, 2509.25020. Cost $1.

## 6. C5 halting (H)
Gated by X1.
- **H1 IN-PASS HALT / SCALE-FREE CONVERGENCE EXIT** ($0)
  - Halt on rel_inc_t / median(earlier passes) < ρ, the pre-write ':' state, or printed-value stability, with no decoder probes.
  - 2603.15051, 2505.18962, 2509.23314, 2107.05407.
  - Pass: within 0.01 of r = 3 on a fresh draw, with ≥ 1.5 fewer passes.
- **H2 SOURCE-REACHED HALT**: an "Is X given?" readout as the stop signal in P2. 2607.20519, 2602.08220. Cost $0.1, on top of P2.

## 7. C3 efficiency (E)
- **E2 EARLY-EXIT WRITE**: the write comes from block 8–9 through a tuned-lens affine map.
  - 2303.08112, 2404.16710, 2207.07061.
  - Pilot: S0 $0 with ℓ = 9, TEST2 ≥ 0.95. FT $0.6.
- **E3 TOP-K ROW REFRESH**: recompute the cue rows plus the top-k attended prompt rows.
  - 2510.02312 (KaVa), 2506.18582 (PCCoT).
  - Pilot: S0 $0 with k ∈ {4, 8, 16}.
- **E4 MoD / MoR ROUTED RECOMPUTE**: capacity 1 → 0.25, with a copy gate.
  - 2404.02258, 2507.10524, 2110.07732. Cost $1.5.
- **E5 HUGINN RETROFIT**: prelude 0–3 / core 4–8 / coda 9–11, trained with K ~ U(2, 8). Needs a large refactor and an equivalence gate.
  - 2502.05171, 2511.07384, 2410.20672, 2510.25741, 2604.11791, 2502.17416, 2409.15647, 2602.11451. Cost $3.
- **Compute protocol**: see X3.

## 8. C7 readable and editable (R)
- §247: the write recomputes v_t from the text and ignores appended text.
- §167: S5 state is editable in a ≤ 64-d subspace.
- §248 TEXT-WRITE was declined as circular.

Entries:
- **R1 TEXT-FREE READOUT (Patchscope)** ($0): inject S_t into a cue-only prompt or another item's prompt.
  - 2401.06102, 2403.10949, 2412.08686, 2604.04902. It also decides C3's cue-row reads.
- **R2 DAS-EDIT on bun** ($0–0.5): k ∈ {16, 64, 256}; IIA ≥ 0.8 = DECOMPOSABLE.
  - 2303.02536, 2112.00826, 2311.17030 (illusion nulls).
- **R3 CODEBOOK / STATE-ENCODER edit** ($0–1): text → write increment through a per-value mean or a fitted encoder, from clean items only. Uses the §247 bars and a shuffled-pair control.
  - 2512.15674, 2412.08686. Needs an Audit ruling on "re-encode".
- **R4 CYCLE-CONSISTENCY** ($0): translator plus encoder. 2405.17653, 2406.04093.
- **R5 TEXT-CHECKPOINT RESTART** ($0): replace S_t's op sentence with "S_t has v_t u.", reset S, run the remaining cues.
  - 2404.18988, 2307.13702. Remaining-depth confound.
- **R6 TOKEN-WRITE**: the write is taken after the ':' value token (teacher-forced gold in training).
  - The exposure control must drop accuracy.
  - 2502.03275, 2609.21662, 2606.29712. Cost $1.5.
- **R7 VALUE-DROPOUT**: R6, plus the step-t operand text masked with p 0.3. 2509.01544, 2607.06648, 2512.21711. Cost $2–2.5.
- **R8 FSQ WRITE BOTTLENECK**: 2606.29712. Cost $2. Overlaps K1 / K5.
- **R9 PER-CHECKPOINT FAITHFULNESS TRAJECTORY** ($0 with checkpoint retention): 2607.06648, 2307.08678.
- **R10 PREMISE-CORRECTION WRITE**: low prior. Cost $0.8.
- Framing: cite both 2512.21711 (latents as shortcuts / placeholders) and 2606.01243 (latents as faithful causal hubs).

## 9. Architecture (A)
Lesson from NK: any pathway added beside DeltaMem needs DeltaMem-head dropout (p 0.3–0.5) during training and an ablation gate; otherwise a negative result can't be read.
- **A2 MULTI-SITE READ** at blocks 2 / 6 / 10, each with zero-init W_o. 2002.09402, 2401.02412, 2207.06881. Cost $1.
- **A3 GATED CROSS-ATTENTION EPISODIC SLOTS**: one slot per pass, softmax read, tanh gate at 0.
  - 2204.14198, 2008.02217, 2203.08913, 2407.04841.
  - Pilot: d 8 ≥ 0.75 and slot ablation ≥ 0.3. Cost $1.2. Sibling of ARCH #2.
- **A4 COCONUT HYBRID latent tokens**: 2412.06769, 2502.21074, 2310.02226. Cost $1.2.
- **A5 ROUTED MULTI-SLOT gated-delta (RIM)**: 1909.10893, 1907.05242, 2412.09764, 2405.06394. Cost $1; prior lowered after NK.
- **A6 WRITE TOKENS + DeltaProduct**: 2502.10297. Cost $1.5. Also relevant to N1 (eigenvalue sign range, 2411.12537).
- **A7 TITANS / TTT MLP head**: 2407.04620, 2501.00663, 2505.23735. Cost $1.2.
- Earlier candidates: ARCH_UPGRADES #2 decay-gated state, #3 frozen-trunk, #4 cached-cue, #5 discrete tied write (= K5 / R8 base).

## 10. Scale (S, M)
Published GSM8K numbers (2506.18582 Table 1):
- GPT-2: CoT 44.1, CODI 43.7, PCCoT 49.5.
- Llama-3.2-1B: CoT 61.6, CODI 55.6, PCCoT 53.4.
- So latent methods beat text CoT only at GPT-2 scale.

**Medium $0 read (post-hoc, descriptive; stored eval_ws pred_by_pass):** on TEST2 d 2, medium's pass-2 ':' emits **v_1** on 0.837 of items (lr 1e-4) and 1.000 (lr 3e-5); v_2 0.015. Small emits v_2 at 0.998.
- The pass-1 swap follow is 1.00 at 1e-4.
- Write gradient norms are comparable to small's, and the read is already at layer 12 of 24.
- So the readout echoes the memory's v_1, while S_2 holds v_2 aux-decodably.

Entries:
- **M1 ECHO-GATE** ($0.3): a pass-2 SWAP on d 3 + aux + echo rate, on p2m_s1b.
  - 2303.08112, 2406.12775, 2310.04625.
  - PASS if follow ≥ 0.90 and aux ≥ 0.95; PARK if < 0.5.
- **M2 COLON-CE** ($1.3): CE on the ':' logits at pass t (generalised --ask_p). 2309.05605.
- **M3 SHORTCUT-PRESSURE** ($4): d_max 3 and / or an S-only decoder. 2402.16837, 2411.16353, 2405.15071.
- **M4 READ-AT-COLON OFF** ($0.7): 2605.26797, 2605.06216.
- **Parked:** M5 read-layer fraction, M6 LR / LoRA (2405.09673), and M7 gradient starvation (contradicted by the reads).
- **S1 BASE READINESS LADDER** ($1): 2304.01373, 2412.15115, 2502.02737, 2502.17416.
- **S7 QWEN2.5-0.5B SWAP with an aux + SWAP gate** ($8): 2602.08783.
- **S8 CODI SCALE GAP on Qwen-0.5B** ($5).
- **S9 OURO-1.4B LoRA** ($5): 2510.25741.

## 11. Task generality toward GSM8K (T, N)
- **N1 SHUFFLED OBJECTS (BBH tracking_shuffled_objects)**: a holder chain through swaps; swaps that don't touch the object are holds.
  - 2210.09261 (random 22.5, Codex CoT 84.5), 2404.08819, 2411.12537.
  - Pilot: train d ≤ 4, test 5–10; ≥ 0.9 at d 8. Cost $4.
- **N2 BOXES entity tracking**: 2305.02363 (T5-base > 45% at 12 ops), 2402.14811. Cost $3.
- **N3 LEGO** (Z2, shuffled): 2206.04301. Cost $2.
- **T3 ProsQA**: forward reachability = the §260 loop.
  - 2412.06769 (Coconut 97.0 / CoT 77.5), 2505.12514.
  - Pilot: ≥ 0.97 with C1 gates. Cost $1.5.
- **T5 iGSM-lite DAGs**: 2407.20311, 2502.05252. Cost $2.5.
- **T6 MULTI-DIGIT VALUES**: 2405.14838, 2311.01460. Cost $2.5.
- **T4 PARAPHRASE FAMILIES**: 40 / 20 families per op. 2402.19255, 2410.05229. Cost $3; after the order fix.
- **N5 CLUTRR**: 2201.01787 (T5-small k10 56.8 / 85.5), 2112.00578 (Edge Transformer k10 92.1). Cost $4.
- **N6 MultiArith / SVAMP**: 2502.21074 (GPT-2 MultiArith CODI 92.8). Cost $3.
- **N4 TWO-HOP fact composition**: 2402.16837, 2406.12775. Cost $2.
- **N7 StepGame**: 2204.08292, 2401.03991. Cost $4.
- **N8 ProofWriter**: 2012.13048. Cost $5.
- **T10 SYMBOLIZED GSM8K-TRAIN TEMPLATES**: 2410.05229, 2312.09241. Cost $6; last rung.
- **E9 CODI-INTO-LOOP**: 2602.22441. Cost $8; not C4.

## Changelog
- 2026-11-03: created from 11 research passes (~95 entries; merges noted inline). §263 ORDER-FREE running.
