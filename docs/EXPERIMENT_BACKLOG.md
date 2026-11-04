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

## 1. Ranked queue (top of backlog; re-ranked 11-03 after wave 2 + Audit NORTH_STAR_GAP)

**Ranking rule change (Audit NORTH_STAR_GAP).** The cost-floored score let any $0 read outrank a $2 experiment by 8x before priors, which buried the big bets. The queue therefore runs in **lanes**, worked in parallel; the score orders entries within a lane.

Load-bearing restrictions, in order:
1. the plan in the cue;
2. per-pass simulator targets;
3. template / mod-11;
4. GPT-2 small with one lineage;
5. fixed K with prompt recompute.

Standing statistics rules (Audit 11-03):
- **Note 269-1 (ST2):** seed spread at d 6 is sigma 0.090, ~9x the item SE. A C2 claim needs >= 3 seeds compared on per-seed margins; > 500 items buys nothing at d 6.
- **Note 269-1 (FD1):** pass-2 formation is invisible at 2k (all runs at chance). Formation markers must read >= 4k.
- Wilson CIs and a NEAR flag on every binding cell;
- "one lineage" scope until X7 exists;
- C1 gates at d 6-7 when C1 and C2 are claimed together;
- a K'-curve for C2.

**Lane A: is bun the thing to polish? (registered; runs first)**

| # | ID | Experiment | $ | Status |
|---|---|---|---|---|
| A1 | §264 | SCORECARD-CHECK: ER / LFI / CY cue schedules; are C5 / C6 plan-clocked? (+ op_count; Theory predictions in Note 264-2) | ~0.5 | approved; amended 264-3 |
| A2 | §266 | SAME-DATA BASELINES: X3 text CoT (sentence x2 + value-only); X5 one-hop pipeline; C7 TEXT-SUBSTITUTE | ~4 | registered |
| A3 | X7 / B3 | one fresh FULL-LINEAGE seed (new askname + P-2 seed). Makes every "2/2" honest | ~5 | **§268 registered** |
| A4 | E0 | C3 RE-SCOPE + G-TIME: serial depth is only 1.2-1.6x better than value-only text; FLOPs 2.4-9.1x worse; bs64 slower | ~0.1 | to register (descriptive) |

**Lane B: restriction 2 (C4), in parallel with A**

| # | ID | Experiment | $ | Status |
|---|---|---|---|---|
| B1 | K1a | EXACT-EM-11, RESET-DM. Answer-only via the exact forward-DP marginal over Z_11 with no decoder in the loss. Stage A on OPS_CHAIN (the REG add_s DAG steps break Markov-ness; red team), d_max 2→3, with SWAP-V / MEM-ABLATE gates + CTRLs (noise-GRPO, em_hard, TOKEN-INJECT) | ~5 | **§267 registered** |
| B2 | K3 S0 | PREFIX-KL. **Audit ruling 11-03 (Note 266-1): NOT C4** (prefix answers = per-step targets by proxy + the solution graph); descriptive / mechanistic only | 0.6 | demoted |

**Lane C: restriction 1 (plan), after §263 reads**

| # | ID | Experiment | $ | Status |
|---|---|---|---|---|
| C1 | PLAN-PROBE | one-pass GPT-2 heads (next entity / parent / is-ancestor) on the insert_ho branch families + TEST-PERM. The shared falsifier for P1 / P2 / P3 | 0.15 | to register |
| C2 | §261 | TARGET-CUE pilot (approved; parked). From of_s1 if §263 passes; + H7 TC-HALT (the model's own halt) | 0.3 | parked |
| C3 | P1 / P2 | SELF-CUE or BACKWARD-PLAN, whichever PLAN-PROBE head is >= 0.9 on the branch families. Gates need TEST-PERM + branch families; free-running runner build | 0.4 | after C1 |

**Lane D: leave the template (restriction 3); the public small-benchmark target**

| # | ID | Experiment | $ | Status |
|---|---|---|---|---|
| D1 | PROSQA-OOD | port to ProsQA's structural OOD splits (Audit PROSQA_OOD_PORT.md; LIT_SUBLAT_PROSQA_OOD_2026-11-03.md).
  - **SubLAT 2610.40283** GPT-2 Table 2, test / 7h / 8h / DAG / Dense:
    - CoT-SFT 71.9 / 11.6 / 10.5 / 36.4 / 22.0;
    - Coconut 98.0 / 73.3 / 72.5 / 58.7 / 65.7;
    - CODI 82.5 / 69.9 / 70.1 / 59.3 / 65.8;
    - SubLAT 88.9 / 77.0 / 77.2 / 50.7 / 72.5.
  - **No generator, data or licence is released,** so the splits are regenerated from Coconut's Alg. 1 with our pre-stated knobs. Only own-run same-split baselines are comparable, so SubLAT's numbers are context only.
  - **Port:**
    - state = the gold-path node (§191 STEP-AUX form, 0.950);
    - cue = the question (plan-free, §261 form);
    - aux = text steps via the prosqa168s2 decoder;
    - reuse prosqa168s2.py, the Coconut harness + 3 trained seeds and the kept s191 models; ~750 new lines.
  - **Stages:**
    - Stage 0 ($0, a GO / REDESIGN gate): s191 A / B / Coconut on the regenerated splits.
    - Stage 1: ~$2 pilot.
    - Stage 2: ~$25 (3 seeds; own-run CoT / No-CoT / Coconut; FLOPs; zero-latent / noise arms). The cost basis is ~7 s/step on ProsQA.
  - **Main risk:** whether a one-token write can hold a frontier on DAG / Dense, where branching needs backtracking. | S0 $0 / ~2 / ~25 | Stage 0 after §263 / §264 read |
| D2 | N1 / B4 | SHUFFLED OBJECTS depth split (BBH tracking; step-index cue = no entity plan) with text CoT / pause / Coconut arms | ~3-4 | queued |
| D3 | NUM-100 | mod 11 → integers 0-99, still one token per value (GSM8K ladder N1) | 0.3 | queued |
| D4 | BRANCH-DAG | parallel branches + merge in mod 11 (GSM8K ladder T1). Recency vs named retrieval | 0.35 | after §263 |

**Lane E: polish inside the restrictions (registered; runs behind A-D unless a red team says otherwise)**
- **§265 DENSE-DISTRACT + D1a:** approved; amended 265-2. Its pilot reads after §264's op_count.
- **D12 ALPHA / SHARE READ ($0.1, bundle with a §264 host).** Then D11 RMS-MIX ($0.25), or D10 DG-ADD.
- **H5 HOLD-FROM-d+1 + RANDOM-REPEAT and H6 DONE-READOUT:** the C5 route if §264 reads PLAN-CLOCKED. H7 TC-HALT is in Lane C.
- **Scale:**
  - S11 S-ONLY DECODER on medium ($1.5; decides the recipe for larger bases);
  - S10 QWEN3-0.6B-Base port + readiness R1-R4 ($1.5);
  - M8 decoder text-dropout on small ($0.3).
- **C7:**
  - P2 VOCAB-DIRECTION EDIT ($0);
  - R6 TOKEN-WRITE / P3 EMBED-WRITE ($1-1.5), only if §266's TEXT-SUBSTITUTE fails.
- **Publication controls:**
  - B1 PAUSE-PASS / NO-WRITE ($2.5);
  - B2 HELD-OUT STRUCTURE eval ($0.3);
  - B6 checkpoint-faithfulness trajectory ($0.2).

**Struck / absorbed:**
- M1 ECHO-GATE: uninformative under Note 264-1's V1+TEXT.
- The C2 diagnostic bundle: absorbed into §264, except D3 (deferred) and H5 beam (killed if §264's d 8 max-prob AUC < 0.6).
- K1 as first drafted: superseded by K1a.
- E3 TOP-K refresh: k = 16 costs more than sentence CoT.

The earlier score table (11-03, first pass) is kept below for reference.

### 1a. First-pass score table (11-03, superseded by the lanes above)

| # | ID | Experiment | Targets | First stage, $ | Score | Depends on | Status |
|---|---|---|---|---|---|---|---|
| 1 | X1+X2 | CUE-CLOCK + FREEZE-vs-FIXED-POINT: are C5 / C6 plan-clocked? | C5, C6 scope | eval $0.2 | 12.6 | – | **§264 registered** |
| 2 | X5 | ONE-HOP-PIPELINE baseline + swap-at-t curve | C1 framing | eval $0.2 | 10.0 | askname ckpt | queued |
| 3 | X4+D3+D5+E1+H5 | C2 diagnostic bundle: first-error pass, length-matched d4, K′12/16, centroid snap, re-sweep, contrast decode, latent beam | C2 | eval $0.3 | 8.0 | – | queued |
| 4 | M1 | ECHO-GATE: medium's pass-2 write vs echo (re-gate §258) | scale, C1 | eval $0.3 | 2.0 | p2m_s1b model.pt | **demoted (Note 264-1: decoder computes the last hop from v_1 + text, follow_ca 1.00); M3 / M2 next for medium** |
| 5 | D1-S1 | DENSE-DISTRACT: 0–8 inserts decouple op-sentence count from depth | C2, C1 | 2k $0.2 | 6.4 | – | **§265 registered (+ D1a)** |
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
  - Prior: 2607.20519 (halting readouts vs trajectory), 2606.29983 (spurious loop-length correlation), 2310.02226 (pause tokens), 1603.08983 (ACT), 1807.03819 (Universal Transformers), 2606.18206 (Fixed-Point Reasoners, convergence halting), 2608.18222 (settle / drift; an answer-only freeze fails at depth; key for X2) (Literature 11-03). Cost $0.2. **Registered as §264.**
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
  - 2312.02179 (TRICE), 2512.20169, 1805.10872, 2212.12393, 2503.18866; closest method prior 2007.01919 (exact marginalisation of discrete latents via sparsity); 2310.04363 (amortised GFlowNet posterior over CoT); 1807.02322 (MAPO) (Literature 11-03).
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
- **D1a ATTENTION-FADING diagnostic (Audit lead L1, 11-03; $0-0.2 eval):** the ':' query's attention mass and entropy over op sentences, at a fixed pass, d 4 vs d 4 + k inserts vs d 8. Is softmax dispersal the mechanism behind Note 264-1's op-count degradation? 2501.19399 (Scalable-Softmax), 2410.01104 (softmax is not enough). Paired with D1.
- **D1b SSMAX LOGIT SCALE (if D1a = fading):** s·log n scaling of attention logits at the selecting layers, warm FT from bun. 2501.19399. Cost $0.6. The architectural fix for D1's symptom.
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
  - 2204.14198 (Flamingo: its tanh-gated, zero-init cross-attention), 2008.02217, 2203.08913, 2407.04841.
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
  - 2303.08112, 2406.12775, 2310.04625 (copy suppression: the v_1 echo may be a copy head, the opposite of a suppression head).
  - PASS if follow ≥ 0.90 and aux ≥ 0.95; PARK if < 0.5.
- **M2 COLON-CE** ($1.3): CE on the ':' logits at pass t (generalised --ask_p). 2309.05605.
- **M3 SHORTCUT-PRESSURE** ($4): d_max 3 and / or an S-only decoder. 2402.16837, 2411.16353, 2405.15071.
- **M8 DECODER TEXT-DROPOUT (Audit lead L2, 11-03):** mask the op sentences in the decoder pass's input with p ~0.5 in training, so the final hop must come from memory. A training-side alternative to M3's S-only decoder (Note 264-1: medium's decoder recomputes the last hop from v_1 + text). Test on small first (bun warm, 2k, NOT-DECODER and SWAP4 vs CTRL), then medium. Cost $0.3 small / $1.5 medium.
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

## 12. Wave-2 entries (11-03): 9 research agents + Audit's docs/research_audit_2026-11-03/

Audit's four new ids: SubLAT 2610.40283, SLT 2605.25745, halting-selection 2611.02333, adaptive loops 2603.08391.

### 12.1 C3 accounting (analytic; scripts in Theory's scratchpad c3flops/, cross-checked against Note 246-0 and REVIEW_CC)

**Per-item cost at K' = d, d 1→8:**

| | GFLOPs (d 1→8) | Serial layers (d 1 / 2 / 4 / 8) |
|---|---|---|
| bun | 15.3 → 219 | 24 / 30 / 42 / 66 |
| text CoT, value-only | 6.4 → 24.0 | 24 / 36 / 60 / 108 |
| text CoT, sentence | 8.4 → 33.1 | 120 / 180 / 300 / 540 |
| prefill only | 6.1 → 21.9 | — |

- **Ratios:**
  - bun / value-only text: 2.4x → 9.1x.
  - "~9x fewer serial layers" holds only against sentence CoT; vs value-only text it is 1.2-1.6x.
  - At K' 8, bun has more serial layers for d <= 4.
- **C5 halting:** the stability rule decodes every pass, which costs ~2x bun's FLOPs.
- **Latency model** (not measured): bun is slower at bs 64.

**Entries:**
- **E0 C3-RESCOPE + G-TIME:** measured wall-clock with random weights. ~$0.1. 2110.12894, 2211.05102.
- **E1 BUN-SHARED:** share blocks 0-5 over the prompt prefix and use a q-only read. -34% FLOPs, exact. $0 equivalence gates.
- **E2 X3+ arms:** value-only text (binding), sentence, majority vote n, pause m, Coconut-KV. Pareto curves per d. 2408.00724, 2408.03314, 2412.13171.
- **E3' ULT:** one upper-half latent row per pass, ~5% under value-only text at d 4. High C1 / C2 risk. $1.5.
- **E4 HALT-COST:** a 1-token probe stop instead of the decoder.

### 12.2 C4: EXACT-EM-11 design (memo)
- **K1a RESET-DM (O1):**
  - Structure:
    - The pass-t ':' output is an 11-way categorical.
    - S_t = write_seq(emb(v_t)) on the zero state: a Markov DeltaMem state with exactly 11 possible states.
    - The loss is the exact forward-DP marginal over v_1..v_{d-1}. There is no decoder in the loss, which kills the Note 264-1 decoder shortcut.
  - Cost: 1 + 11(d-1) upper-half forwards (d 2: 12, d 4: 34), batched by tiling 11x.
  - Gradient = posterior - prior. At the askname init, pass 1 cannot be pulled toward a, because the ops are bijections.
  - Guards: SWAP-V, MEM-ABLATE, pass-text-dropout.
  - Symmetry (value relabelling) is broken by the tied init, the d = t items, and write = emb(number).
  - Build: boxes/inject_em.py (--em 0 bitwise inject_aw), --em_hard, --pass_textdrop; inject_em_eval.py with part_swap_v and part_mem_ablate; an fp64 gate that the DP equals the explicit path sum at d 3.
  - **Red-team amendments:**
    - Stage A on OPS_CHAIN, because REG's add_s reads v_j with j <= t-2.
    - d_max 2→3, with K'0 <= chance + 0.1.
    - Scope: "C4 on a designed Z_11 alphabet, plan given".
  - Prior: 2312.02179 (TRICE), 2007.01919, 1805.10872, 2212.12393.
  - Stage A ~$2 + CTRLs $3.
- **K1b** EM-HARD (Viterbi) and EM-SAMPLED (RLOO) ablations. MAPO 1807.02322, 2411.04282, 2310.04363. $1 each.
- **K1c** PASS-TEXT-DROPOUT: drop the op sentences of S_<t from pass t. M8 moved into the passes. $1.
- **K1d** TOKEN-INJECT-11: the previous value as text, exact enumeration, no DeltaMem. Upper-bound CTRL. $1.

### 12.3 C2 / C6 memory design (decay memo)
- **Already in the code:** the delta heads' beta ∈ (0, 2), so negative eigenvalues are already present (2411.12537 is not a new lever).
- **Additive heads:** phi = elu + 1 > 0, so they can never cancel an old write, and they cannot be idempotent.
- **Caveat:** Note 264-1's same-pass op-count gap lies outside decay's mechanism, which lowers the EV of D10 / D11.
- **D12 ALPHA / SHARE READ** ($0.1 eval). Reads delta-head alpha / beta by pass type and the additive latest-write share by (d, t). Labels: GATED-USED / ALPHA-INERT; PREMISE-OK / PREMISE-FAIL; hold FREEZE / ECHO. It gates D10 / D11.
- **D10 DG-ADD:** lambda = exp(-c softplus(w u + b)), c = 0 projected >= 0 (bitwise bun at init). 2412.06464, 2405.21060, 2312.06635, 2307.08621, 2405.04517. S1 2k vs CTRL, $0.5.
- **D11 RMS-MIX READ:** oa' = oa + rho (r RMSnorm(oa) - oa), rho = 0 at init. 2210.10340, 2604.15259. $0.25; shares D10's CTRL.

### 12.4 C5 halting theory (memo; predictions committed as Note 264-2)
- **H3** ALPHA-BETA HOLD READ ($0; merges into D12).
- **H4** SCALE-FREE INCREMENT HALT: rho_t = ||dS_t|| / median of earlier passes < 0.3. Predicted to halt at d + 3 on bun.
- **H5** HOLD-FROM-d+1 + RANDOM-REPEAT (D2 extended): trains "already answered → null write" at every pass index. $0.4 / $1.4.
- **H6** DONE-READOUT HALT: the hold-pass aux target becomes a DONE token, and the halt is a model output. 2607.20519, 2107.05407. $0.3 / $1.2.
- **H7** TC-HALT on the §261 target-cue model: label HALT-OWN (khalt increasing in d, H-EXTEND halt-at-target >= 0.8). $0.3. Highest priority if §264 reads PLAN-CLOCKED.

### 12.5 C1 plan lane (red team #6-10)
- **REG items are DAGs:** add_s and *_v ops.
- **Plan-free gates need branch families + TEST-PERM.** On clean chain-order text, "next sentence's subject" solves the plan.
- **passes() cannot run free-running cues;** a new runner is needed.
- **PLAN-PROBE** ($0.15): a plain GPT-2 fine-tune, 1.5k steps, with three one-pass heads (next forward entity / parent of X / is-ancestor-of-S_d), evaluated on the insert_ho branch families + TEST-PERM. If one pass cannot exceed 0.9, the in-loop plan heads won't either.
- **Revised scores:** P1 2.9; K3 3.0 (needs a ruling); P2 2.4; P3 2.1 ("§261 first"); A1 1.2 / 2.0 / 1.0 by §263 branch. A1 notes: the causal mask encodes order (2203.16634); wpe[0] acts as a sink.

### 12.6 GSM8K ladder (one change per stage)
- **Tokenizer:** GPT-2 BPE 0-99 are single tokens. 4-digit numbers split inconsistently (" 1234" → " 12" + "34").
- **Licences:** GSM8K, TinyGSM and iGSM are MIT. GSM-Symbolic is CC BY-NC-ND and built from the TEST set (eval only). ASDiv is NC.
- **GPT-2 GSM8K-Aug SOTA:** PCCoT 49.48. Our pipeline: CODI 37.9 (§210).
- **Open slots:** no latent numbers exist for ASDiv / MAWPS / iGSM / GSM-Symbolic.

| Stage | Change | $ (pilot) | EV |
|---|---|---|---|
| NUM-100 | 0-99, one token; backward-uniform sampling | 0.3 | 8.0 |
| DIGIT-AUX-10K | per-digit 4×10 aux + a digit-split arm (2402.14903, 2405.17399, 2307.03381) | 0.7 | 2.3 |
| O1 | exact ÷ and 2-digit × | 0.5 / 1.5 | — |
| BRANCH-DAG | parallel branches + merge | 0.35 | 5.7 |
| IGSM-LOOP | med-pq, mod 23, "Define X" order as the cue (2407.20311) | 1 | 1.75 |
| L1 = T4 | paraphrase families | 3 | — |
| LLM-REWRITE | Claude rewrites + solver check (2309.12284) | 1.2 | 2.7 |
| SOCRATIC-CUE GSM8K PROBE | GSM8K-Aug items with 0-99 intermediates, socratic sub-questions as cues (2110.14168, 2509.20317). Direct triage | 2 | 1.5 |
| GSM8K-SYMBOLIZED-TRAIN | in-house templates from TRAIN only, 13-gram decontamination (2211.10435) | 3 | 0.4 |
| E PROTOCOL | same-data text + pause arm at every stage; 13-gram filter; GSM-Plus / SVAMP / GSM-Hard OOD (2405.00332) | — | — |
| SOCRATIC SELF-CUE | | 2 | — |

The full ladder is ~$45-70, so the GSM8K-scale stage needs its own approval.

### 12.7 Base model (<= 1B)
- **Recommendation:** Qwen3-0.6B-Base (Apache-2.0, GSM8K 59.6 at 4-shot, all 40 names single tokens).
  - Costs: a digit-split number readout (sequence-scored), QK-norm, head dim 128.
  - Second choice: Llama-3.2-1B, which has published CODI / PCCoT comparators but a gated licence and needs 48 GB. OLMo-2-1B is the Apache alternative.
  - Not recommended: GPT-2 large, Pythia, SmolLM2, Gemma-3-1B.
- **Any port needs a decoder-shortcut guard:** an S-only decoder or text-dropout, a pass-t SWAP gate, and a NOT-DECODER monitor.
- **S10** QWEN3 PORT + READINESS R1-R4: HF trunk adapter equivalence; R4 = DIRECT d 2 / 3 shortcut susceptibility. ~$1.5. 2505.09388, 2412.15115, 2608.11233.
- **S11** S-ONLY DECODER ON MEDIUM, warm from p2m_s1b. PASS if d 2 >= 0.9, pass-2 SWAP >= 0.9 and NOT-DECODER <= 0.1. ~$1.5. Decides the recipe for larger bases.
- **S12** QWEN3 P2 with the shortcut guard (~$4-5). **S13** a LLAMA / OLMo comparator (conditional).

### 12.8 Publication set (minimum publishable claim memo)
- **Headline H1** (defensible now, as a mechanism paper): "content selection, not depth budget, limits depth extrapolation; the causal battery holds on a pretrained LM". Scope stated up front.
  - Missing for it: a same-data text CoT, a pause control, seeds + CIs, held-out templates, a public task.
- **Headline H2:** H1 + a public depth split, ProsQA-OOD or shuffled objects. Fits within $30.
- **Entries:**
  - **B1 PAUSE-PASS / NO-WRITE control:** zero / constant / Coconut-feedback write. 2310.02226, 2404.15758. $2.5.
  - **B2 HELD-OUT STRUCTURE eval:** frames 6-7, unseen names, GSM-Symbolic perturbations, TEST-PERM. $0.3.
  - **B3 = X7.**
  - **B4 = N1** with baselines.
  - **B5 = E2.**
  - **B6 checkpoint-faithfulness trajectory:** 2607.06648, 2602.08783.
- **Related work to cite:**
  - 2606.29983: loop-count vs input-length correlation (independent support for §241 / §256).
  - 2608.18222 / 2606.18206: terminal fixed-point objectives. The hold penalty is the same idea, so cite them as prior.
  - 2604.06374: fine-tuned pretrained LMs collapse to shortcuts. Our causal battery is the counter-evidence.

### 12.9 C7 (Audit C7_EDITABLE)
- **Literature:** edits propagate only when the thought is tied to a token embedding (2505.15778 Soft Thinking, 2605.07315 LaTER, 2606.29712 DLR), or when the reader has no other source for the value.
- **P1 TEXT-SUBSTITUTE:** in §266.
- **P2 VOCAB-DIRECTION EDIT** ($0): h += a (wte[x'] - wte[x^]), compared with the probe / DAS at a matched flip rate.
- **P3 EMBED-WRITE** ($1-1.5): W_w LN(wte[decoded]), straight-through. Only if P1 fails.
- **P4 SOFT-WRITE.**

## 13. Wave 3 (11-03): 21 Fable research agents

Twenty themes not covered by waves 1-2, plus one late C4 agent. Every agent was read-only, computed nothing from stored outputs, and fetched every arXiv id it cites.

### 13.0 Findings that change framing

- **Expressivity theory.** Mod-11 programs are an affine-group word problem plus pointer chasing. A single 12-layer pass can express d up to about 64.
  - So "the loop is needed" is a learnability and extrapolation claim (C2), not an expressivity claim.
  - The loop is provably forced only for non-solvable value groups: S5 / shuffled objects (N1). N1 is therefore the "loop is necessary" task for the paper.
  - Prediction: §264's K' curve is a step at K' = d - 1, not a ramp.
- **Memory capacity.** Raw capacity is not the limit (6 delta heads x DH 64). Addressability is.
  - One ':' write per pass, plus the hold penalty, probably makes the keys value-driven rather than entity-driven.
  - The additive heads (phi > 0) are a single-value accumulator.
  - Retrieval by name (needed for DAG merges and GSM8K) is untested. MC1 / MC2 test it first.
- **History taxonomy (sections 0-268).** The recurring failure modes are:
  - shortcuts that pass the gate (~14 sections);
  - gates on accuracy rather than mechanism;
  - treating a seed lottery as a recipe (6 mechanism rescues failed, ~$40);
  - build bugs (~10);
  - uninformative controls;
  - gamed one-sided targets;
  - no learnability pilot before a new task.
  - Tried 3+ times without success: strict C4 (10 attempts), learned halting (5), seed rescue (6), paraphrase transfer (5), extra passes as a depth lever (4).
- **Process rules adopted for the registration template:**
  1. A shortcut-exclusion arm from the start.
  2. Gate on the mechanism at the claimed depth.
  3. Try a budget increase or LR re-warm before a redesign; redesign after three nulls at the same wall.
  4. Select across seeds; no mechanism-targeted seed rescues.
  5. A learnability pilot before any new task or base.
  6. Controls are exposed, and every new flag gets a p = 0 equivalence gate.
  7. Two-sided bars for targets on dynamics.
  8. Zero-init pathways get a usage monitor and a non-zero-init arm.
- **Scaling.** Medium's decoder shortcut is the general failure the literature predicts: latent models often skip the latent (2604.04902). The scaling question is the base's direct hop capacity vs d_max, not parameter count.
  - The predictor to test is SC2 SHORTCUT-PROPENSITY.
- **C3 / efficiency.**
  - Eval runs in pure fp32, with no tensor cores.
  - There is no torch.compile and no length bucketing.
  - Data generation is serial on the training thread.
  - Expected saving is 30-45% per cell (EF0 profile first).
- **Statistics.**
  - Seed variance, not item variance, is the binding uncertainty.
  - Compare recipes on continuous per-seed margins, never on pass counts.
  - A 0.02 bar is powered only on paired items.
  - Proposed: one stdlib theory/statlib.py.

### 13.1 Wave-3 shortlist: cheap, decision-changing items

Each costs <= $0.2, mostly as cloud eval parts on bun. Each needs a pre-statement and Audit's ack. Most can share the §264 host.

| ID | What it decides | $ |
|---|---|---|
| I5 PASS-7-AT-D8 TRIAGE | Is each late-pass error selection, memory or computation? This picks between D1b, a memory redesign and computation fixes | 0.2 |
| I2 READ-USE MAP | Which positions consume the block-6 read. Re-prices C3 cached designs (E3) and explains §263 | 0.1 |
| TH2 MIRROR-PAIR | Can a forward plan-free policy work at all (bound: prod 1/b), or only question-conditioned planning? Decides P1 vs P2 / P4 / TH5 | 0.15 |
| MC1 KEY-GEOMETRY | Are write keys name-, value- or constant-keyed? Gates multi-value / DAG work | 0.1 |
| RB3 PASS-TOKEN-OCCLUSION | The causal footprint of each pass; leakage > 20% rewords C1 | 0.15 |
| RB2 STATE-NOISE-RADIUS | sigma_50 per pass. Is the late-pass basin narrower? | 0.1 |
| CG1 SUB_S ZERO-SHOT / CG5 OPEN-VOCAB | Is the step factorised (operator x source) or a template lookup? Does selection work on multi-token or overlapping names? | 0.1 |
| TS3 TTA-VOTE / TS1 SC-LATENT | Are d 7-8 errors recoverable by voting (oracle@N), or systematic? | 0.2 |
| D12 + TH4 | Delta alpha / beta by pass type; does the additive hold write vanish or drift at O(t)? | 0.1 |
| FD1 FORMATION-CENSUS | Onset t* across all stored logs; does an early marker predict never-formers? ($0 read) | 0 |
| ST2 SEED-VARIANCE | sigma_seed vs item SE: sets the seeds-vs-items budget rule ($0 read) | 0 |
| EF0 STEP-PROFILE | Where a bun step's time goes; orders the speedups | 0.1 |
| AP6 NAR CEILING | Does a graph processor extrapolate to d 16 where the single-slot memory dies at d 8? Capacity vs algorithm ($0 CPU) | 0 |

### 13.2 Wave-3 training pilots worth registering (<= $1.5 first stage)

- **C4 lane:**
  - CA1 SG-WRITE and CA2 ONSET-FREEZE ($0.3 each; they localise the onset drift);
  - CA3 MULTI-LEAF and CA4 SHARED-STEP SIAMESE (C4-clean data and self-supervision);
  - AP1 HRM / TRM deep supervision ($0.3).
- **Plan lane:**
  - P9 CUE-FADE ($0.5);
  - H8 ANSWERABLE-NOW head ($0.25);
  - P8 CONFIDENCE-GAIN SELECTION (stage 1 $0.1);
  - TH5 DATAFLOW-ALL-WRITE ($0.4);
  - AP5 latent search ($0.15).
- **Lineage:**
  - LC5 BRIDGE2 2-stage recipe ($1.3);
  - LC4 STATE-DISTILL ($0.8; also the medium port);
  - HX1 SEED-FARM-2K ($2.4).
- **Public benchmarks:**
  - BM1 PRONTOQA-DEPTH ($2-3; pairs with ProsQA-OOD for headline H2);
  - BM2 WEB-OF-LIES ($1.5);
  - BM3 VT / VARS (a published 0.5B latent comparator).
- **Multi-task (north star):** MT1 JOINT-4 ($4.5) -> MT4 cross-task swap ($0.3). §174 was never launched; RELAY 2502.08482 calls the question open.
- **Scale:**
  - SC2 SHORTCUT-PROPENSITY INDEX ($1);
  - SC1 LOOP-vs-DEPTH ISO-SERIAL ($1.5);
  - SC3 TRUNCATED-MEDIUM ($1.5).
- **RL** (pass@64 pre-checks first):
  - RL2 RL-HALT ($1; the learned C5 if §264 reads PLAN-CLOCKED);
  - RL1 RL-DEPTH ($1.2) with the RL3 mechanism guard.
- **Hybrid:**
  - HY0 complementarity read ($0 on §266 outputs);
  - HY1 TPLV 2x2 (= P1 with a crossover gate).

### 13.3 Detail registry (condensed agent entries)

#### plan-free stopping (Fable)
H8 ANSWERABLE-NOW HEAD: 1-logit on ':' predicts whether decoder correct now (labels from own decoder, stop-grad); halt p>=0.5; from gb_s1 2k ($0.25 frozen / $0.4); GO halt-at-S_d>=0.8 H-EXTEND & Spearman(khalt,d)>0.8; park AUC<0.7 d5-6 | 2107.05407, 1603.08983, 2207.07061, 2402.06457 V-STaR
P7 GOAL-DISTANCE VALUE (TD V(S_t)=1+V(S_t+1), terminal from H8); halt + off-chain null write; $0 TD fit on stored gb dumps (pre-state) AUC>=0.8 -> 2k $0.4 | 1707.01495 HER, 2110.06169 IQL, 2201.08299, 2305.10601 ToT
P8 CONFIDENCE-GAIN SELECTION -> Q-SELECT critic: masked-sentence search picking the sentence raising decoder max-prob, STOP if none; distil to critic; stage1 $0.1 eval GO H-EXTEND & H-LIVE>=0.6 on gb | 2312.08935 Math-Shepherd, 2403.09629, 2305.10601, 2110.14168
P9 CUE-FADE: mask cue name with rising p 0->1 (late passes first) + self-distil to plan-cued EMA teacher; from bun_s1 4k $0.5 (+no-distil arm $0.4); GO generic cue d2-4>=0.9, H-EXTEND/H-LIVE>=0.6, d6>=0.8 | 2405.14838, 1511.03643, 2207.07061
P10 LATENT SUBGOAL TOKEN (VQ codebook M=16 in cue slot, STOP code) $0.45 | 1609.05140, 1711.00937
#### multi-task (Fable): §174 (mult+nested, no flag) registered 10-09 never launched; RELAY 2502.08482 says multi-task in one looped model is open. No SO/KIN generator exists.
MT1 JOINT-4 (WP + shuffled objects + ProsQA + kinship-lite; shared vs per-task cue; per-family aux gate; grad cosine) ~$4.5 | 1807.03819, 2209.11142 CLRS, 2502.08482
MT2 LEAVE-ONE-FAMILY-OUT (zero-shot + 1k adaptation sample-efficiency ratio vs bun) ~$5 | 2110.14056, 2604.07822
MT3 SEQUENTIAL-ADD (naive / replay / replay+memory freeze) ~$4 | 2007.07400, 2308.13320, 2001.06782 PCGrad
MT4 CROSS-TASK STATE SWAP + head-necessity matrix + task-id probe $0.3 eval (after MT1)
MT5 COMPOSED cross-family items $0.7
MT6 TASK-ID FLAG control $2.4 conditional
#### memory capacity (Fable): raw capacity not binding (delta 6x64 orthogonal = 384 pairs); limit is ADDRESSABILITY: one ':' write/pass, hold penalty likely makes keys VALUE-driven -> no retrieval by name. Additive heads (phi>0) = one-value accumulator. write_seq already accepts T>1 + mask.
MC1 KEY-GEOMETRY $0.1 eval (Gram of write keys by entity/value; NAME/VALUE/CONSTANT-KEYED) | 2102.11174, 2310.17191 binding IDs, 2501.12352
MC2 KEEP-k RECALL (MQAR for the loop; stage0 eval $0.1; stage1 FT j<d question $0.6) | 2312.04927 MQAR, 2402.18668, 2406.06484, 2504.18574 — precondition of BRANCH-DAG
MC3 ENTITY-KEYED MULTI-WRITE (pass-1 writes at each given sentence end; ':' write becomes by-name update) $1.3 | 2011.07831, 2412.06464, 2305.02363
MC4 HEAD-TYPE CAPACITY ABLATION (12 add vs 12 delta) $1.2 | 2008.02217, 2111.05498, 2402.01032
MC5 RECENCY vs NAME CUE (alpha decay as alternative explanation of op-count degradation) $0.4 | 2506.05233 MesaNet
#### statistics (Fable): unpaired 0.02 bar at p~0.95 needs n~1900/arm; paired (5% discordance) n~1000 ok. k/k seeds CP lower bounds 0.22 (2/2), 0.37 (3/3), 0.55 (5/5), 0.72 (9/9). Compare recipes on continuous per-seed margins (~6 seeds/arm), never pass counts.
ST1 STATLIB theory/statlib.py (stdlib): wilson, clopper_pearson, margin_se, paired_diff (McNemar), cluster_boot, seed_rate, n_items_for, conf_seq, sprt, e-values, holm/bh, conj_label NEAR-k | 2411.00640 Miller, 2108.13264 rliable, 1904.06979, 1606.04316
ST2 SEED-VARIANCE-COMPONENTS ($0 pre-stated read of s1/s2 per-item arrays) -> "met" = 5/5 seeds from 2 P-2 models | 1806.08295, 2103.03098, 2304.01315
ST3 SEQ-STOP confidence sequences / SPRT for pilot futility + sequential evals; $0 retrospective replay | 1810.08240, 1906.07801, 2402.14992 tinyBenchmarks
ST4 PAIRED-ARMS (sig join, common seeds; paired diff + McNemar) $0
ST5 MULTIPLICITY (NEAR-k conjunction labels; BH for grids; looks ledger; e-value seed pooling) $0
#### history taxonomy (Fable): failure modes: shortcut passes gate (~14 sections), gate on accuracy not mechanism, seed lottery (6 mechanism rescues failed ~$40), build bugs (~10), uninformative controls, gamed one-sided targets, post-hoc drift, no learnability pilot, budget mistaken for architecture (2-3x budget fixed 5 nulls). Worked: per-step supervision; break implicit counters; reader/copy from step 0; warm start; footholds; normalised reads; training-free rules > learned halts (learned halts failed 6x). 3+x failures: strict C4 (10x), learned halting (5), seed rescue (6), identical-input final-only (8), paraphrase transfer (5), extra passes as depth lever (4).
Process rules: shortcut-exclusion arm from start; gate on mechanism at claimed depth; budget/LR-rewarm before redesign, redesign after 3 same-wall nulls; no mechanism seed rescues (selection instead); learnability pilot before new task/base; exposed controls + delivery gates; two-sided bars for dynamics; zero-init pathways need usage monitor + non-zero-init arm.
HX1 SEED-FARM-2K (6-8 seeds x 2k, marker, successive halving) $2.4 | 1603.06560 Hyperband, 2002.10365
HX2 MARKOV-CONT (continuous reset-at-write; range principle on language loop) pilot 2k $0.2 | 2412.06464, 2411.12537
HX3 PIPELINE-DISTILL (X5 pipeline as deeper teacher, d5-9 labels) $0.8 | 1810.08575, 2203.14465, 2305.02301
HX4 GATE-USAGE ARM (zero-init channels: gate magnitude log + non-zero-init arm + DeltaMem dropout) template rule
#### C7 verbalizer (Fable): reader must see cue-only prompt (no op sentences); faithful-to-model test on bun's own errors; input-only control 2509.13316.
V1 STATE-SENTENCE VERBALIZER + content ladder (subject/value/op/operand) ~$0.5 | 2412.08686, 2512.15674, 2605.25903 UAV, 2509.13316, 2607.15495, 2401.06102
V2 FAITHFULNESS: counterfactual simulatability + claim-flip audit $0 | 2307.08678, 2607.20379, 2609.33702, 2502.14829
V3 READER-GRADIENT EDIT $0.3 | 2403.10949, 2602.08783, 2609.21662
V4 SELF-VERBALIZATION TRAINING $1.5 (low)
V5 VERBALIZER AS FAILURE DIAGNOSTIC (permuted / late-pass / idle passes) $0 on V1 dumps, EV 4
#### lineage compression (Fable): REG d<=4 from askname never forms pass 2 (§236 0/2); §258 BRIDGE (aw -> REG+distract 16k) = BRIDGE-OK (d6 0.949/d7 0.789). Stage restarts of warmup+cosine may matter.
LC1 ONE-RUN SCHEDULE (36k ramp; arm B SGDR restarts) pilot 0-20k $1.5 | 1608.03983, 1904.03626, 2412.06769
LC2 SELF-PACED DEPTH WEIGHTS (w_t = WS_{t-1}^2 + per-bin clipping) $0.8 | 1711.02257
LC3 SKIP-ASKNAME (one-hop co-training inside aw stage) $1.0 | 2410.05464, 2502.21074
LC4 STATE-DISTILL bun -> fresh student (& -> medium; progressive arm) $0.8 / medium $3 | 1503.02531, 1805.04770, 2410.05464
LC5 LEAVE-ONE-STAGE-OUT: BRIDGE2 aw -> REG+mix+hold 16k $1.3 (2-stage recipe) EV high
#### numbers (Fable): bun aux is sequence-scored (multi-token OK); 12 heads separable.
NR1 NUM-CODE-PROBE (Fourier vs digit code R2, error histogram, ablation use test) $0.2 after NUM-100 | 2406.03445, 2502.00873, 2410.11781, 2605.01148, 2301.05217
NR2 FONE-AUX (Fourier-feature aux T 10/100/1000) $0.7 | 2502.09741, 2006.10739
NR3 DIGIT-HEAD WRITE + per-digit swap gate $0.7 | 2405.17399, 2310.13121
NR4 R2L-SPACED VALUE FORMAT (prereq >99) $0.6 | 2307.03381, 2402.14903, 2410.21272
NR5 CARRY-STRATIFIED read + CARRY-SAVE arm (log carry count now) | 2510.00184
Not recommended: xVal/NALU scalar.
#### hybrid text/latent (Fable): bun is already a hybrid (text plan in cue, latent value); missing: interleaved runner (same as P1/P2).
HY0 COMPLEMENTARITY READ $0 on §266 outputs (oracle union latent vs text per item; dead lane if union - max < 0.03) | 2110.14168, 2505.13379 Thinkless
HY1 TPLV text-plan/latent-value 2x2 {clean, permuted} x {d<=4, d6-8} $0.6 (= P1 with crossover gate) | 2502.03275, 2509.20317, 2205.10625, 2210.03350
HY2 LATENT SUB-CALL as calculator (NUM-100 2-digit x) $1.5 | 2302.04761, 2211.10435, 2211.12588, 2505.18962
HY3 MODE ROUTER (Thinkless-style) $2 low | 2505.18454, 2505.16552
HY4 SOCRATIC SELF-CUE GSM8K-Aug $2
HY5 CONFIDENCE FALLBACK (bun unstable -> text CoT) $0.2
#### RL post-training (Fable): RLVR sharpens, not creates (2504.13837) -> pass@64 pre-checks; S-only decoder reward; RLOO/Dr.GRPO.
RL1 (K13) RL-DEPTH RLOO on d5-6 gold answers; pass@64 precheck $0.1; 1k steps $1.2 + random-reward CTRL | 2606.10184 Dropout-GRPO, 2607.19691 SLPO, 2604.17892, 2402.14740, 2402.03300, 2506.10947
RL2 (H8') RL-HALT STOP action, reward - lambda*passes $1 | 2502.04463, 2503.04697, 2505.13379, 2503.20783, 2607.20519
RL3 (X8) RL-MECH-GUARD every 200 steps (frozen readout, SWAP-V, KL; random-reward + 1%-mask arms) | 2511.08567, 2602.01599, 2505.11711, 2509.04259
RL4 (K14) RL-ROBUST on permuted/paraphrase items (pass@64 >= 0.3 precheck) $1.2
RL5 (P7') RL-SELF-CUE (name policy by answer reward, S-only decoder) $1.3 | 2604.22709
#### alternative paradigms (Fable)
AP1 (A8) DEEP-SUPERVISED RECURSION HRM/TRM (final-answer CE each pass, detach, inner micro-passes, q halt) from bun 2k $0.3 | 2506.21734, 2510.04871, 2211.09961, 2202.05826
AP2 (A9) IMPLICIT-PASS DEQ (JFB; $0 path-independence read first) | 1909.01377, 2103.12803, 2111.05177, 2609.26487
AP3 (A10) ENERGY-DESCENT WRITE (IRED/EBT) $0.4 | 2406.11179, 2206.15448, 2507.02092, 2605.12466
AP4 (A11) DENOISING over step slots (DoT/LLaDA; C3 serial ceiling) $0.3 | 2402.07754, 2502.09992, 2510.04573, 2502.01567
AP5 (A12) LATENT PLANNING BY SEARCH over op choices (beam 3, stability+logp verifier) $0.15 eval EV 6 | 1811.04551, 2301.04104
AP6 (A13) NAR PROCESSOR CEILING (CLRS MPNN on generator DAGs, $0 CPU; TransNAR stage 2) EV 8 | 2205.15659, 2406.09308
#### efficiency (Fable): eval pure fp32 no tensor cores; no compile/CUDA graphs/TF32; data gen serial on main thread; hold recomputes lower; far pass T_far 16-40 no-grad.
EF0 STEP-PROFILE $0.1 first
EF1 COMPILE-PASS (torch.compile + CUDA graphs; not bitwise -> new recipe flag) 1.2-1.5x | 2205.14135
EF2 BUCKET-PAD (sort by length within micro chunks; pad to x16) 1.1-1.4x $0.05 | 2107.02027
EF3 EVAL-BF16-SWEEP for descriptive parts only, fp32 re-check flips (eval-law change; Audit ruling) 3-5x | 1710.03740
EF4 PREFETCH-ITEMS background data gen 1.05-1.3x
EF5 HOLD-SHARED-LOWER ~5%
#### robustness (Fable): cue corruption exists (part_cue_wrong).
RB1 NOOP-CLAUSE (in-sentence irrelevant number / same-name other-unit) $0.1 eval + $0.6 FT | 2410.05229, 2302.00093, 2402.19255, 2109.05925
RB2 STATE-NOISE-RADIUS sigma_50 per pass (+ write-noise / Jacobian-norm contraction arms) $0.1 eval EV 9 | 2106.14342, 1805.01500
RB3 PASS-TOKEN-OCCLUSION FOOTPRINT (causal per-pass dependence; leakage >20% -> C1 "selection with leakage") $0.15 EV 11 | 2202.05262, 1711.02173, 2309.01686
RB4 INVARIANT-STATE CONSISTENCY on structural twins (R-Drop/SimCSE at state level) $1 | 2106.14448, 2104.08821, 2402.08939
RB5 ADV-DISTRACT (mined inserts, FreeLB-style in input space) $1.3 | 1909.11764
RB6 B2+ ROBUST-TABLE protocol (50 templates x 20 instantiations; perturbation rows w/ CIs vs text CoT & pause) $0.3
Prediction: bun CLAUSE-READ; late passes narrower basin; footprint spreads with op count.
#### scaling (Fable): post-hoc latent CoT loses above ~1B (2506.18582, 2606.31779); looping scales (Ouro, Huginn, MoR, Relaxed Recursive, PonderLM 2505.20674); loop laws: Parcae 2604.12946 (loops+data together), 2604.21106 (1 recurrence ~0.46 param doubling), 2609.40316; 2604.04902 latent models often skip latents (decoder shortcut general). Scaling question = base's direct hop capacity vs d_max.
SC1 (S14) LOOP-VS-DEPTH ISO-SERIAL: medium direct K=1 vs bun on d6-8 $1.5 | 2502.17416, 2604.21106, 2409.15647, 2310.19956
SC2 (S15) SHORTCUT-PROPENSITY INDEX: 2k direct FT per base, SPI(d) predicts write formation ~$1 EV 3
SC3 (S16) TRUNCATED-MEDIUM (12x1024) depth vs width of shortcut $1.5 | 2410.20672
SC4 (S17) PYTHIA SHAPE LADDER 160M/410M $2.1 | 2304.01373
SC5 (S18) PRETRAINING-DATA AXIS (Pythia ckpts 20k/70k/143k) $3
SC6 (S19) D_MAX x K LADDER on small (d_max 2,6; K=d_max+4) $1.6
#### test-time scaling (Fable): bun deterministic at eval; edit/on_state hooks exist.
TS1 SC-LATENT noise/MC-dropout rollouts vote@N/oracle@N $0.1 EV 4 | 2510.07745, 2505.11484, 2203.11171
TS2 NOISE-TRAIN-SC $1.2
TS3 TTA-VOTE over distractor re-renderings (permutations after §263) $0.1 EV 4.8 | 2402.08939
TS4 LRM-SELECT latent reward head best-of-N $0.3 | 2505.13308, 2606.17524
TS5 RE-SWEEP-TRAIN self-correction sweep $1.2 low | 2501.19393, 2409.12917, 2511.08577 Think-at-Hard
TS6 SELECTIVE SCALING fold-in
#### compositional (Fable): ops = {add,sub,mul}x{c,v} + {add,mul}x{s}; frames 0-5 train, 6-7 held, 8-11 DIV-train, 12-15 DIV-held.
CG1 SUB_S ZERO-SHOT (empty cell of operator x source grid) $0.1 eval EV 4.2 | 2305.18654, 2301.10884, 2311.12997, 2108.04378, 1908.08351
CG2 OP x CONSTRUCTION COGS split on DIV frames 8-11 $0.4 | 2010.05465 COGS, 1711.00350 SCAN, 2108.12284, 2210.03050, 2307.05596
CG3 NONCE-OP IN-CONTEXT DEFINITION (zork = +3; flipped-definition test) $0.6 EV 2.5 | 1906.05381, 2303.03846, 2205.05055, 2310.17191, 2210.03350
CG4 CONSTANT-PRIMITIVE HOLDOUT (SCAN-jump on Z_11; needs lineage rebuild) $0.6 | 2306.17844, 2210.10749, 2310.16028
CG5 OPEN-VOCABULARY ENTITIES (multi-token names, overlapping names, noun-phrase entities) $0 eval EV 5.4
#### mech interp (Fable): nothing in backlog does head-level circuits on bun; linear-attention state = exact per-head patchable.
I1 OP-SELECT HEADS (path patching at ':' and cue-name; pattern vs OV split; d4 vs d8) $0.2 | 2211.00593, 2304.05969, 2305.00586, 2402.11917, 2604.11791
I2 READ-USE MAP (which positions consume block-6 read; per memory head; EAP-IG) $0.1 — re-prices E3 / C3 | 2404.03646, 2404.05971, 2602.22719, 2602.00449
I3 VALUE CIRCUIT 7-11 (algorithm vs heuristics; pass-invariant faithfulness) $0.2 | 2305.15054, 2410.21272, 2406.11944, 2403.19647, 2310.08744, 2312.09230
I4 CROSS-PASS EAP (unrolled graph incl. 12 memory edges; mechanism figure) $0.2-0.5 | 2304.14997, 2310.10348, 2403.17806, 2406.16778, 2403.00824, 2604.05876
I5 PASS-7-AT-D8 TRIAGE (selection / memory / computation error census via patching) $0.2 EV high — run before D1b / memory redesign | 2402.16837, 2406.12775
#### public benchmark scout (Fable)
BM1 PRONTOQA-DEPTH (linear is-a chains; Apache-2.0 generator; no GPT-2 OOD-depth latent number; pairs with ProsQA for H2) $2-3 EV 2.0 | 2210.01240, 2305.15269, 2412.06769, 2610.40283
BM2 WEB-OF-LIES-DEPTH (BBH; Z2 chain keep/negate; first sub-1B number) $1.5 | 2210.09261, 2206.04301, 2602.08332 Thinking States
BM3 VT (RULER variable tracking) then VARS (Thinking States: 0.5B 33.8 vs CoT 6.8) $1.5 + $3 | 2602.08332, 2404.06654, 2404.08819
BM4 MODGRAPH-23 (Altabaa: latent recursion + step supervision + discrete bottleneck generalises 4x; fold into T5) $3 | 2510.14095, 2603.21676 (per-step supervision hurt there — counter-claim)
BM5 CONNECTIVITY-DIAM $1.5 | 2510.19753, 2406.04229 CLRS-Text
BM6 ADD-CHAIN / BBH MULTISTEP after NUM-100 $4 | 2409.15647
Rejected: Sudoku/Maze (HRM/TRM cost), bAbI, Dyck, logical deduction, date, flip-flop, list functions, CRUXEval.
#### formation dynamics (Fable): AdamW wd 0.01 all params, lr 1e-4, 100-step warmup, cosine; no wd/noise/attn-norm arm at P-2 ever. Plateau = partial solution (v_1 echo).
FD1 FORMATION-CENSUS + PROGRESS MEASURE ($0 pre-stated; onset t* across all stored logs; markers lead >=1k?) EV 8 | 2506.13688, 2604.20923 ILDR, 2306.13253, 2301.05217, 2002.06305
FD2 FORMATION-LOTTERY n=4 seeds at P-2 with early stop (rate + Wilson CI; bar for fixes) $3.2 | 1912.05671, 2607.20594
FD3 NEW-MODULE WEIGHT DECAY (0.3/1.0 on DeltaMem + write/read only) $2.4 | 2201.02177, 2210.01117, 2311.18817, 1802.01483
FD4 EUREKA-NORM (QK-LayerNorm / logit temperature on selecting layers during formation) $2.4 | 2310.12956, 2505.17863, 2404.07129, 2312.03002
FD5 NOISE-DOSE (bs 16 vs 256; fp32-softmax arm) $2.4 | 2306.04251, 2501.04697, 2410.23922, 1910.08475
FD6 COMPONENT-SHAPING (oracle memory emb(v_1) annealed 1->0 at P-2; formation tool) $1.2 | 2402.18361, 2310.17561, 2406.06467, 2402.11004
#### expressivity theory (Fable): mod-11 programs = affine AGL(1,11) word problem (solvable -> ACC0 subset TC0) + pointer chasing O(log d) layers -> one pass expresses d<=~64; "loop needed" is a learnability / length-generalisation claim (C2), not expressivity. Only non-solvable value groups (S5) force the loop (NC1-complete).
TH1 EXPRESSIVITY-MEMO (3 props: no forcing; register bound K+h; memory transition irrelevant) $0; prediction: K'-curve is a step at K'=d-1 (small) | 2207.00729, 2402.12875, 2310.07923, 2305.15408, 2210.10749, 2402.09268, 2503.03961, 2505.18948, 2502.17416, 2404.08819, 2411.12537, 2502.10297, 2301.13196
TH2 MIRROR-PAIR branch test (forward-only plan-free policy <= prod 1/b_t; question-conditioned can reach 1) $0.15 EV 16 — decides P1 vs P2/P4 | 2505.12514
TH3 SELECTION-MARGIN LAW (error linear in competing keys mentioning chain entities, not tokens; SSMax removes n-dependence) $0 on §265 outputs EV 11
TH4 HOLD FIXED-POINT predictions (delta contracts at |alpha(1-beta)|; additive grows O(t) unless writes vanish) rider on D12 EV 10.8
TH5 DATAFLOW-ALL-WRITE (write (entity,value) at every resolvable sentence; passes = DAG depth; plan-free; halt = question key resolved) $0.4 EV 6 | 2505.12514
TH6 NON-SOLVABLE VALUE GROUP (S5; N1 as the "loop is necessary" task) $0.6
#### C4 with accumulating memory (Fable): AO fails at one site (pass-t ':' code on d>t items drifts at onset).
CA1 (K13) SG-WRITE: detach ':' write source for t<d (answer grad reaches pass-t only via shared weights) — localises drift path vs weight interference; Note 251-10 harness 2k $0.3 | 2502.05171, 2506.21734, 2510.04871
CA2 (K14) ONSET-FREEZE (LP-FT two-timescale) + EMA write source $0.3-0.6 | 2202.10054, 2006.07733, 1703.01780
CA3 (K15) MULTI-LEAF QUESTIONS (same text, sibling leaf finals co-batched; ancestor never a target) $0.6 — Audit flag | 2407.20311
CA4 (K16) SHARED-STEP SIAMESE (S_1 consistency across pairs sharing S_0,S_1 sentences; VICReg variance) $0.5 | 2011.10566, 2105.04906, 2006.07733
CA5 (K17) IDENTITY-SUFFIX UPWEIGHT — needs Audit ruling $0.4
CA6 (K18) EM-ACCUM (§267 DP with accumulating state; partial enumeration + RLOO; DISTILL-TO-DM sub-arm needs ruling) ~$2 after §267 | 2402.14740

## 14. Audit literature sweep Oct-Nov 2026 (docs/research_audit_2026-11-03/wave2/LIT_SWEEP_OCT_NOV_2026.md) + REDTEAM_BUN: new leads
- **HOLD-RANDOM-PASS** (2610.19768): states form at loop k but are not retained beyond the trained range; supervising one random pass retains 99.8%.
  - A read on bun: pin S_t at the first-correct pass vs at t + 2.
  - Training form: hold supervision on one random pass per item.
  - Pairs with H5.
- **PAR prefix self-distillation** (2610.27628): the iteration-k target is the model's own state after k ops, plus a per-iteration norm re-anchor; 16x extrapolation.
  - **Audit ruling (Note 269-4): C4-ELIGIBLE, scoped 'plan given'.** Targets must be the model's own predictions and use only test-time information, with target-vs-truth accuracy logged.
  - This is now the strongest C4 route that keeps the accumulating state; a sibling of §267.
- **Overshoot control** (2610.31943, 2607.14427): looped-vs-stacked gains can be overshoot confounds.
  - Report d 6-8 at matched overshoot K' = d + c.
  - A training-free state-stability stop beats a trained router (supports H1 / H4).
- **Asymmetric Looped Transformers** (2610.28489): a C3 reference design for KV-cached latent passes (E1 / E3', design doc 03).
- **C5 reporting** (2610.32246, 2605.21642): report the batch-max passes, not only the mean (now in read_s269_redteam (b)).
- **Further references:**
  - LoopOPD 2610.10623: a self-teacher C4 route.
  - ELR 2610.12582: a pass-index clock, i.e. an X1 positive control.
  - 2610.05815: a Deep ProsQA split exists (D1).
- **Double-Take** (2610.13073): diagnoses §247's C7 null. A write whose key and value both come from the state is self-associative and cannot bind text to state; key-from-text with value-from-state lifted recall from 0.10 to 0.92.
  - Supports R6 / MC3 (design doc 01).
  - C7 ladder: capacity -> + text-keyed binding -> + change-sensitive write.
- **StateBench hierarchy** (2611.03094, Petty / Merrill / Sabharwal): REG is the aperiodic tier.
  - Tier-3 (soluble) items inside our template are a cheap eval; they pair with N1 (S5) for the 'what the loop computes' statement.
- **Looped Transformers as Optimizers** (2609.37379) and **Kronecker delta** (2610.26834): a delta-step write with learned decay / step and a bounded state norm without the hold penalty.
  - beta_t -> 0 gives a halt signal independent of the repeated cue, which addresses the §264 plan-clock worry. Pairs with D10 / D11.
- **Other references:** G-MemLLM 2602.00015 (gated slot bank on GPT-2 124M; a sanity bar for A3); DAMI 2610.41994 (gate-init timescales; may relate to seed basin variance).
- **Binding / entity-tracking arm of the sweep (Audit):**
  - **PERM-SNR** (2609.28442): a descriptive permutation-SNR read of pass-t latents, of_s* vs bun, so that 'order-free' is not misread as 'order-blind'. Runs on §263's landed models; descriptive (Note 263-3 to pre-state).
  - **PIN k vs k+4** (2610.19768): pin the pass-d latent to its pass-d vs pass-(d+3) value for the later passes. A large gap means the C6 'fixed point' is a post-cue freeze. Added to §269b.
  - **REPLACE-CONTENT arm** (2610.21182: readable latents are not evidence of reasoning). Swap pass-t latents with another same-depth item's.
    - bun's SWAP-MID already does this at d 4.
    - Make it standard in the ProsQA-OOD port's C1 panel.
  - **QUERY-HANDOFF patch** (2610.34639, 2609.39892, 2609.24635): patch the ':' queries from a cue run into a plan-free run. This tests whether the plan-free failure is query formation. It runs before P1 SELF-CUE (Lane C), and uses the free-running runner (design doc 06).
  - **DATA-ONLY re-format arm** (2610.39016): re-format facts to match the second-hop circuit; this closed the OOD gap at GPT-2 scale with no architecture change. A cheap control for MC3 / T-lane.
- **Sweep, final arms** (latent-CoT training; depth extrapolation):
  - **PAR NORM DRIFT** (2610.27628): log ||h_t|| per pass on bun at d <= 4 vs d 8. If the failing passes 7+ coincide with norm drift from pass 1, a per-pass RMS re-anchor is the lever (PAR ablation: 2x -> 16x). Pre-state it; the read can ride on §269 alphahold.
  - **STAIRCASE PROBE** (2610.24135): the decoded hop index per pass on d 8. A flat staircase after pass 6 means a stalled walk (hold interference), not a wrong read. Pairs with §269 triage.
  - **D1a scale choice:** the gap-count statistic of 2605.12697 picks the logit-scale exponent (log n vs log^2 n) before any SSMax arm; 2611.01472 gives a per-head distant-mass statistic.
  - **C4:** WIDER 2609.07406 (rank-collapse penalty on latents; targets onset drift) and SpaRLax 2610.27220 (entmax write + answer-only RL).
  - **Seeds:** 2610.26974 (two competing basins, init-steerable) suggests an early basin predictor for d 8 reliability, matching Note 269-1's sigma_seed.
  - **FEW LONG STRINGS** (2610.35715): add 1-5% higher-op-count items at the same depth as an X4 / D1 arm. This is a cheap sibling of §265.
- **Wave-4 note:** 2 of 20 design docs have landed (01 MC3, 14 paper outline). The required red-team sub-agents could not spawn under the 20-concurrent-agent limit, so the docs' red-team sections are self-reviews marked PENDING. An independent red team is required before any wave-4 doc is registered.
- **REDTEAM_BUN (Audit):**
  - bun was never evaluated on insert_ho; now in §269 part 6.
  - The C1 gates stop at d 4; queued in §269b.
  - C2 K'-probe at d 9-10 (pre-stated read).
  - Predicted end state if fears land: C1 at d 4 under a plan, C2 depth-only, C6 under a repeated cue, C5 partial. §264, §266 and §269 settle each.

## Changelog
- 2026-11-03 16:56: wave 3 merged (21 Fable agents: interpretability, formation dynamics, lineage compression, efficiency, statistics, memory capacity, compositional generalisation, verbalizer, test-time scaling, hybrid, expressivity theory, benchmark scouting, history taxonomy, robustness, alternative paradigms, multi-task, RL post-training, number representation, plan-free stopping, scaling, C4 with accumulating memory). Shortlist in §13.1; process rules in §13.0.
- 2026-11-03 16:34: wave 2 merged (9 agents + Audit NORTH_STAR_GAP / VALIDITY_STATS / C7_EDITABLE / EXTERNAL_SOTA). Queue re-ranked into lanes A-E. §264 / §265 amended pre-output (Notes 264-3 / 265-2); Theory's §264 predictions committed (Note 264-2); §266 SAME-DATA BASELINES registered. M1 struck; the C2 bundle absorbed into §264; K1 -> K1a (OPS_CHAIN); E3 top-k de-ranked.
- 2026-11-03: created from 11 research passes (~95 entries; merges noted inline). §263 ORDER-FREE running.
- 2026-11-03 16:17: Audit leads L1 (attention fading: D1a diagnostic + D1b SSMax) and L2 (M8 decoder text-dropout) added. Next registration: D1 + D1a (§265), then P1 / P2 after §263.
- 2026-11-03 16:15: Note 264-1 stored reads: X4a UNTRAINED-PASS (no first errors at passes 1-2; pass 7 reliability 0.91 at d 7 vs 0.59 / 0.80 at d 8, so later passes degrade with the op-sentence count, which favours D1 DENSE-DISTRACT); M1 V1+TEXT (medium's decoder ignores S's v_2; M1 demoted, M3 SHORTCUT-PRESSURE / M2 COLON-CE up).
- 2026-11-03: Literature spot-check (docs/LIT_BACKLOG_SPOTCHECK_2026-11-03.md): all 160 ids resolve; missed priors added to X1 / K1. X1 + X2 (+ op_count, X4a, M1 stored reads) registered as §264 SCORECARD-CHECK; X5, R5 and D3 deferred to a later cell.

## 15. Wave-4 red-team leads (Theory, 2026-11-04)

- **QB0 QUESTION-BLIND SUFFICIENCY (C1 plan-free lane; source 07_REDTEAM.md HIGH-1, cdb1fb46).**
  - Finding: in of's training law (insert_render.py:194-230), every insert is a sink, S_d is never read, and the question always asks the spine end. So the question-blind rule "follow the readers, skip sinks, stop at no-reader" is correct on 100% of training items.
  - This is a candidate mechanism for §260's plan-free collapse (H-EXTEND / H-LIVE / chain-live families fail, trained types ~1.0). It also means the P9 / H8 / TH5 / PLAN-PROBE designs get no training pressure toward question-conditioning.
  - Cheapest decisive step, $0 CPU: simulate the blind policy on the training stream and on each eval family, and report P(blind correct). Pre-state it before running.
  - Then the lever: `--retarget_p` re-points a share of training questions at an intermediate S_j or an insert's end, keeping the text fixed. The blind policy must be wrong on those items by construction. Any lane-07 arm should train with it.
- **NO-REPLICATE (09_REDTEAM F1):** compressed-lineage seeds that share aw_plaw2_s1 are one lineage. They cannot satisfy the >= 2-lineages rule.
- **PAPER (14_REDTEAM, Literature 82d38794):** headline = breaking positional slot counting (not content selection). Core model bun; of is the order-free extension. Claim the out-of-range accuracy, not the halting rule.
