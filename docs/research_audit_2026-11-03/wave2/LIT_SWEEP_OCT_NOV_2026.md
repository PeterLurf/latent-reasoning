# Literature sweep, arXiv Sept–Nov 2026 (ids 2609.*–2611.*) plus 2026 gaps — Audit wave 2

Date: 2026-11-03 17:20 EST (clock). Lead: Audit lit-sweep agent; arms: looped (sub-agent), halting (sub-agent), faithfulness + small-model benchmarks (lead). Four arms (latent-CoT training, depth extrapolation, variable binding, associative memory) were launched but had not reported when this file was written; see "Pending arms" at the end. Every id below was verified by fetching its arXiv abs page and grepped absent from the 931 ids cited in docs/LITERATURE.md, docs/LIT_*.md, docs/*/LIT_*.md, docs/EXPERIMENT_BACKLOG.md and docs/research_audit_2026-11-03/*.md (scratchpad known_ids.txt). The arXiv Atom API returned 429 part-way through every arm; a re-run of the 429'd queries may add a few entries.

## Top 5 leads (ranked)

1. **2610.19768 Learning to Compute Is Not Learning to Retain or Reuse in Looped Transformers** — looped models form the step-k state at loop k but trained-at-32 models keep only 22% at loop 64; supervising one random position per example retains 99.8%; first-appearance states are not reusable (pinning to loop k drops to 4–6%, loop k+4 harmless). This is our §264 X2 and the 251-8 onset-drift question in one controlled setting. Lead ($0 read on bun, then a hold-supervision change): pin S_t to its first-correct pass vs pass t+2 and measure downstream accuracy; if first-appearance states are unusable, switch the hold loss from "all passes after d" to "one random pass per item" and re-test d6–d8.
2. **2610.27628 PAR: Progress-Aligned Recurrence** — iteration-k target = state after k ops plus per-iteration norm re-anchoring gives 16x length extrapolation (0.5B, 32 → 512 moves); without norm anchoring < 2.2x; "prefix self-distillation" learns the alignment from shorter prefixes with final answers only. Lead: per-pass RMS re-anchor in the GPT-2 loop (D8-adjacent, $0.6) and prefix self-distillation as the C4 version of K3 PREFIX-KL / K9.
3. **2610.31943 Agreement rather than normalisation keeps depth reusable** + **2607.14427 Per-token fixed-point convergence** — (a) looped-vs-stacked at equal loop count is confounded by overshoot past the trained budget; refitting head + last loop layer recovers lost depth; (b) a training-free state-stability stop reaches depth-8 quality at 4.94 loops while a trained router saves nothing. Lead: score bun's d6–d8 with a matched-overshoot control and the head-refit repair before claiming C2 is loop-intrinsic; and run the state-stability halt vs the 3-unchanged rule as the X2 FREEZE-vs-FIXED-POINT read.
4. **2610.28489 Asymmetric Looped Transformers** (encode the prompt once to fixed KV, loop only the latent positions; beats full-loop at matched depth) — the unbuilt KV-LAT design behind the C3 row (246-0, 251-12). Lead: build the AsymLoop variant of the GPT-2 loop and replace the analytic 1.5–3x FLOP penalty with a measured one.
5. **2610.32246 Do Halting Heads Really Save Compute?** + **2605.21642 Ablate-to-Validate** — (a) halt heads trained on mixed iterations under-estimate; batch-max passes, not per-item mean, is the real cost; (b) a four-way token-replacement ladder (zero / random / first-repeat / oracle) separates "channel present" from "channel used". Leads: report batch-max pass count for the 3-unchanged rule before any C5 claim; add the first-repeat arm to the C1 scorecard.

Also high: 2610.10623 LoopOPD (terminal loop as self-teacher for intermediate loops — a C4 route for K7), 2610.12582 ELR (explicit pass-index conditioning = X1 CUE-CLOCK positive control), 2610.18127 (five-lens monitoring protocol for looped LMs; carry vs re-read loops), 2605.22870 Readout Shortcut (small-model text CoT = copy-last-number; the X3 comparator must corrupt the trailing number on both arms), 2610.05815 Plan Canvas (Deep ProsQA split exists — reuse for T3 PROSQA_OOD_PORT).

---

## Arm 1: looped / recurrent-depth transformers (sub-agent)


Scope: ids 2609.*-2611.* plus 2026 gaps. Every id below was verified by fetching https://arxiv.org/abs/<id> (title + date checked) and grepped against known_ids.txt (absent). Ranked by relevance to Latent (GPT-2-small, K=8 latent loop, REG word programs).

### 2610.19768 — Learning to Compute Is Not Learning to Retain or Reuse in Looped Transformers (Oct 2026)
- Claim: On in-context Mealy-machine execution with simulator targets and exact counterfactuals, looped models compute position k exactly at loop k (swapping the loop-k hidden vector redirects every later position, 237/237 pairs), but models trained to answer at loop 32 keep only 22% of answers at loop 64. Supervising one random position per example (not always the last) retains 99.8%; truncated-gradient "anti-overthinking" does not help. Even retained states are not reusable: pinning position k to its loop-k state drops accuracy to 4-6%, pinning to loop k+4 is harmless.
- Why it matters to us: C6 (hold stability past pass d; §264 X2 FREEZE-vs-FIXED-POINT), C1 (causal swap test is our latent-removal/edit bar), C4 (onset drift of the pass-t ':' code, 251-8..11 — "first appears" vs "reusable" is exactly that gap). Backlog: X2 FREEZE-vs-FIXED-POINT, D2 HOLD-INTERLEAVE, K3 PREFIX-KL.
- Lead: On a bun checkpoint, run their reuse test as a $0 read: pin the pass-t state of S_t to its first-correct pass vs. pass t+2 and measure downstream S_{t+1..d} accuracy; if first-appearance states are unusable, change the hold supervision from "all passes after d" to "one random pass per item" and re-test d6-d8 retention.
- Verified: abs page fetched yes; already in bib: no

### 2610.27628 — PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking (Oct 2026)
- Claim: Train the iteration-k prediction to match the state after the first k operations and rescale recurrent states to first-iteration norms; Qwen2.5-0.5B trained to 32 moves hits 95.5% full-state accuracy at 512 moves (16x). Removing norm anchoring cuts extrapolation to <2.2x; final-answer-only supervision to <=8x. A learned completion readout halts without a supplied depth; "prefix self-distillation" learns the progress alignment from shorter prefixes using final answers only.
- Why it matters to us: C2 (d6-d8 extrapolation; 264-1 pass-7+ decay), C4 (prefix self-distillation is an answer-only route to per-pass targets), C5 (completion readout = H2 SOURCE-REACHED HALT analogue). Backlog: K7 LATENT-STaR / SELF-LABEL DEPTH LADDER, K9 ANSWER-ONLY DEPTH CURRICULUM, K1a RESET-DM, H2.
- Lead: Add a per-pass RMS re-anchor (state norm reset to pass-1 norm) to the GPT-2 loop and re-run the d<=4 -> d8 extrapolation; separately, implement prefix self-distillation on REG: train pass t on the answer of the d=t prefix program only (no simulator values) and test whether the pass-t ':' code forms without onset drift.
- Verified: abs page fetched yes; already in bib: no

### 2610.24135 — Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks (Oct 2026)
- Claim: On H-hop permutation traversal (every intermediate hop known), a 4-layer block x8 iterations solves the task on all seeds while an untied model of equal effective depth never does unless given a lower LR or intermediate-layer supervision. Probing/intervention shows the looped model advances through intermediate nodes as a staircase scaled to the iteration budget; the untied model exposes them only in final layers.
- Why it matters to us: C1 (staircase = our "pass t holds S_t" claim, measured with interventions), and it gives a matched-depth untied control for "the loop does the work". Backlog: B1 PAUSE-PASS / NO-WRITE control, X6 PROBE TRANSFER to passes 5-7, B2 HELD-OUT STRUCTURE.
- Lead: Train an untied 12-layer GPT-2 control at equal effective depth (K=8 x 4-layer block vs. 8 distinct blocks) on REG d<=4 and compare the pass-t probe staircase; their "staircase scales to budget" prediction says pass-t codes should shift when K changes at train time — a cheap test of whether our S_t placement is cue-driven (X1) or budget-driven.
- Verified: abs page fetched yes; already in bib: no

### 2610.31943 — Agreement rather than normalisation keeps depth reusable in looped language models (Oct 2026)
- Claim: Removing loop-boundary normalisation drops Ouro-1.4B/2.6B to chance on a balanced binary task without state-norm blow-up (norm stays within 2.02x); in 19M from-scratch models (16 seeds), no-norm training was never worse and extrapolated better. Refitting head + last block layer fully recovers lost accuracy. Looped-vs-stacked comparisons at equal loop count are confounded by overshoot past the trained budget; at matched overshoot the loop advantage vanishes on one task.
- Why it matters to us: C2 (our d6-d8 numbers are overshoot past trained d<=4), C6 (seed-3 breaks; hold penalty vs. normalisation). Backlog: X4 LENGTH-NOT-DEPTH, X7 STATS-AND-SEEDS, D8 NORMED RECALL READ.
- Lead: Re-score bun's d6-d8 extrapolation with a matched-overshoot control (train K=8 model on d<=6 and compare d8 at equal overshoot) and try the cheap repair they report: refit only the readout head + last loop layer on d7-d8 items before claiming C2 is loop-intrinsic.
- Verified: abs page fetched yes; already in bib: no

### 2610.28489 — Asymmetric Looped Transformers: Fixed Global Representations for an Iterative Decoder (Oct 2026)
- Claim: Split the loop into a single-pass causal encoder that emits fixed per-layer Global KV caches and a weight-tied recurrent decoder that produces Local KV from its evolving state; one softmax attends over both. 60M-1B on FineWeb-Edu beats Full Loop at matched depth/tokens (1B: 49.58 -> 50.85 avg zero-shot).
- Why it matters to us: C3 — "prompt recompute per pass is architectural" (246-0, 251-12); this is the KV-LAT design we have not built. Backlog: C3 / efficiency, E0 C3-RESCOPE + G-TIME, A2 MULTI-SITE READ.
- Lead: Build the AsymLoop variant of the GPT-2 loop: encode the REG prompt once (frozen KV), loop only the latent/answer positions attending to prompt KV + their own evolving KV; measure FLOPs per pass and d<=4 accuracy against the current full-recompute loop to replace the analytic 1.5-3x FLOP penalty with a measured one.
- Verified: abs page fetched yes; already in bib: no

### 2610.32246 — Do Halting Heads Really Save Compute in Looped Language Models? (Oct 2026)
- Claim: Pretraining up to 1.3B from scratch: halting heads trained on a mix of iterations predict <half the iterations yet tokens run nearly all of them under the standard stop rule, because training never scores the exact stop iteration; scoring every iteration works only with a strong early-stop penalty. Reported mean-iteration savings overstate gains when batches wait for the slowest token. Proposes a pre-training bound on achievable savings.
- Why it matters to us: C5 (learned halt heads in-range only, §248; halt = +3 passes over d-1). Backlog: E4 HALT-COST, H1 IN-PASS HALT, C5 halting.
- Lead: Before any new halt head, run their pre-training bound on bun: compute the oracle per-item stop pass (first pass where the answer is final and stays) and report the real pass count under a batch-max rule, not the mean; if the bound is <1 pass saved over the 3-unchanged rule, park C5 halt heads.
- Verified: abs page fetched yes; already in bib: no

### 2610.10623 — Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models (Oct 2026, v2 23 Oct)
- Claim: LoopOPD uses the frozen terminal loop as teacher for an intermediate loop on the student's own rollouts (dense supervision, no external teacher or privileged info); D-LoopOPD refreshes the teacher as shared weights update, with conditions for a single update to improve both depths. Improves Ouro-Thinking on math and transfers to general reasoning/code.
- Why it matters to us: C4 (supervision from the model's own deeper pass instead of simulator values) and C2. Backlog: K7 LATENT-STaR / SELF-LABEL DEPTH LADDER, K3 PREFIX-KL, E9 CODI-INTO-LOOP.
- Lead: On an answer-only GPT-2 loop, add a KL from pass-K logits (frozen copy) onto pass-t logits for t<K on the model's own sampled REG items; test whether this dense self-teacher closes the 1/4 answer-only formation rate (§244/§245) without simulator values.
- Verified: abs page fetched yes; already in bib: no

### 2610.23036 — Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design (Oct 2026)
- Claim: Controlled from-scratch ablations of block placement, input injection and recurrent-state layout: prelude-recurrent-coda topology helps context extraction and reasoning; input injection mainly helps knowledge/context use; random init and split high/low states are mixed or negative. Huginn-style beats Ouro on 10/10 benchmarks at 730M/336B tokens and on MoE at 8B-total.
- Why it matters to us: E5 HUGINN RETROFIT and the loop's read/write design (A2, D11/D12 READ entries); companion to Part II (2610.06833, already in bib).
- Lead: Ablate input injection in the GPT-2 loop: compare the current per-pass prompt re-read against (a) prelude-embedding injection only and (b) no injection after pass 1, on REG d<=4 and d6-d8; their result predicts injection matters for retrieval (name lookup) more than for the chain computation, which bears directly on the "cue names S_t every pass" confound (260-5).
- Verified: abs page fetched yes; already in bib: no

### 2610.41720 — UpLoop: Training Looped Transformers via Upcycling at No Extra Pretraining Compute (Oct 2026)
- Claim: Start shallow, grow depth by copying learned layers, and spend the saved early compute on repeated passes through a shared middle block later; matches a dense baseline's parameters, tokens and FLOPs. 1.3B: 22-task average 22.99 -> 27.52; gains persist after math SFT.
- Why it matters to us: E5 HUGINN RETROFIT, S1 BASE READINESS LADDER, S7 QWEN2.5-0.5B SWAP (how to install a loop at no extra compute); C6 (growth schedule as a stability recipe).
- Lead: Replace the current "loop GPT-2's full stack from step 0" with an UpLoop schedule: train the first 30-40% of steps at K=1 on the pretrained layers, then switch to K=8 on the shared middle block; check whether the basin-entry / seed-3 breaks (§192/§205) disappear at matched FLOPs.
- Verified: abs page fetched yes; already in bib: no

### 2610.12582 — One Model to Loop Them All: Elastic Latent Recurrence for Language Models (Oct 2026)
- Claim: ELR removes the "one-loop tax": gated state update with learned step size, mixed loop counts per optimizer step, and conditioning each pass on its loop index and budget; late-pretraining mixed depth with zero-init conditioning beats mixed depth from scratch. 4B dense: +7.25 pts at 4 loops; keeping only last-loop KV retains most gains.
- Why it matters to us: C6/X1 CUE-CLOCK (loop-index conditioning is an explicit clock — a positive control for "is the fixed point cue-driven?"), C2 (mixed-depth training), C3 (last-loop-only KV).
- Lead: Add an explicit pass-index embedding to the GPT-2 loop as the X1 positive control: if accuracy/hold with a pass clock equals the cue-repeat model, the cue is acting as a clock; if the clocked model still needs the cue, the cue carries content, not timing.
- Verified: abs page fetched yes; already in bib: no

Also verified (abs fetched, not in bib), lower rank:
- 2609.21383 Prediction Dynamics in Depth-Recurrent LMs (Sep) — why intermediate answers match the final one while scores keep moving (Huginn/Ouro); bears on the 3-unchanged-answer halt rule (259-11/12) and X2.
- 2610.26030 Truncate or Not? scaling study (Oct) — ~600 runs: full BPTT through fewer loops beats truncated BPTT at matched compute; two fully-trained loops is compute-optimal. Relevant if we ever truncate K=8 backprop.
- 2609.21605 Trading Depth for Time in Recurrent Transformers (Sep, v2 Oct) — one latent thought token between vocab tokens recovers 89% of doubling depth; A4 COCONUT HYBRID.
- 2608.15062 Gated Recurrent Transformers (Aug, v4) — prelude/core x R/coda with noise-conditioned update gate; 3-layer model matches 12-layer GPT-2 Small at isoFLOPS; beats MoR. E4 MoD/MoR, A5 ROUTED MULTI-SLOT.
- 2610.25998 Internalizing Chain-of-thought into Looped Transformers (Oct) — theory: constant-size looped TMs interpolate CoT and latent thought, context and tokens reduced by factor L; C3 framing only.

Queries tried: arXiv Atom API: all:"looped transformer" (204 hits, 2 pages); all:"recurrent depth" AND transformer (36); "mixture of recursions" OR "looped language model" OR "looped LM" (59, 2 pages); "latent recurrence" OR "depth recurrence" OR "depth-recurrent" (46); "looped transformers" OR "loop transformer" OR "weight-tied" AND reasoning (214); "recurrent transformer" AND "latent reasoning" / "iterative refinement" hidden states; MoR-only and Huginn/Ouro-only queries (API 429'd; re-run via arxiv.org/search for "mixture of recursions" — only 2610.37903, 2610.08830, 2608.06727, 2607.00774 in window, none LM-core). WebSearch: "arxiv looped transformer 2026 latent reasoning"; site:arxiv.org "recurrent depth" language model October 2026; "mixture of recursions" arxiv 2026; "depth-recurrent" language model 2610/2611 Huginn; retrofitting loops onto pretrained LM 2026; "looped language models" halting / latent recurrence November 2026.

Candidates rejected as already-known: 2611.02333, 2610.14456, 2610.11570, 2610.11472, 2610.07591, 2610.06833, 2610.02185, 2610.00673, 2609.40316, 2609.36636, 2609.35748, 2609.35643, 2609.33144, 2609.19521, 2609.03379, 2609.01924, 2609.01117, 2608.11233, 2607.27656, 2607.25915, 2606.31779, 2605.26733, 2605.12466, 2604.07822, 2603.21676, 2602.11451, 2604.12946, 2606.10184, 2605.26797, 2604.01577.

Not reported (in window but off-topic for our loop: decoding speed, quantization, KV compression, vision/tabular/protein, safety): 2610.21594, 2610.38822, 2610.35556, 2609.34538, 2609.17184, 2609.23033, 2610.16368, 2610.05265, 2610.10381, 2610.07940, 2610.41825, 2610.34436, 2610.02383, 2610.34563, 2610.41819, 2611.02857, 2610.10625, 2610.02741, 2610.37903.

---

## Arm 2: adaptive halting / ponder (sub-agent)


### 2610.32246 — Do Halting Heads Really Save Compute in Looped Language Models? (Oct 2026)
- Claim: Halting heads trained on a mix of all iterations under-estimate the iterations tokens need, so under the standard stop rule tokens run almost every pass; scoring every iteration fixes it only with a strong early-stop penalty. Batched generation runs to the slowest token, so per-token mean savings overstate real cost; models up to 1.3B from scratch.
- Why it matters to us: C5; H7 TC-HALT, E4 HALT-COST; §248 "halt heads in-range only". Also an accounting rule: report batch-max passes, not per-item mean.
- Lead: on the d8 eval set log per-item halt pass AND batch-max pass for the "3 unchanged answers" rule; if batch-max ≈ 8 the C5 saving is nominal. Then try their fix for a learned head: supervise the continue/stop logit at every pass t with a +cost-per-pass penalty, not just at the chosen exit, and check halt pass vs d on held-out d5–d8.
- Verified: abs page fetched yes; already in bib: no

### 2607.14427 — Per-Token Fixed-Point Convergence in Depth-Recurrent Transformers (Jul 2026; missing from bib)
- Claim: 135M depth-recurrent LM on FineWeb-Edu; median token stabilises by loop 6, ~10% still updating at 8. A training-free "stop when the output stabilises" rule reaches depth-8 quality at 4.94 mean loops (-38%), while a linear router trained on convergence labels needed nearly full depth and saved nothing.
- Why it matters to us: X2 FREEZE-vs-FIXED-POINT, H1 IN-PASS HALT / SCALE-FREE CONVERGENCE EXIT, C5/C6. Independent evidence that a stability rule (ours) beats a trained halt head (§248), on a GPT-2-class model.
- Lead: $0 read on bun: per item, compute the pass at which ||h_t - h_{t-1}||/||h_t|| < tau (state rule) and the pass at which "3 unchanged answers" fires (answer rule). If state and answer halts coincide → fixed point; if answers freeze while the state keeps moving → FREEZE for X2. Repeat with the cue removed (X1 condition).
- Verified: abs page fetched yes; already in bib: no

### 2610.25646 — Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (Oct 2026)
- Claim: Runs FPRM's fixed-point iteration on the unit hypersphere (nGPT-style) so injection, damping and halting are geodesic operations with bounded values; proves fixed-point existence/convergence; 96.6% Sudoku-Extreme, 87.9% Maze-Hard, 59.1% Mini-ARC with better test-time compute allocation than FPRM (2606.18206, known).
- Why it matters to us: H1 SCALE-FREE CONVERGENCE EXIT (their halting is norm-free by construction), C6 stability under K'128, X2.
- Lead: replace the norm-based convergence criterion in H1 with the angular (cosine/geodesic) distance between consecutive normalised pass states; test on bun at K=8 and K'=128 whether the angular halt pass tracks d better than the raw-norm halt (which is sensitive to state growth, cf. perturbation-norm memory).
- Verified: abs page fetched yes; already in bib: no

### 2609.37351 — Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling (Sep 2026)
- Claim: Unconstrained recurrent latent models lose 22–62% accuracy when run K>=16 vs K<=4 ("deliberation drift cliff"). Splitting the latent flow into a conservative (stabilising) and rotational (expressive) part (DG-HHD) gives 58.7% peak and 35.3% retained at K=32; monotonic compute scaling on small-LM multi-hop.
- Why it matters to us: C6 (our K'128 hold), C2, X1 CUE-CLOCK / X2: a cliff is the signature of a loop that does not sit at a fixed point.
- Lead: accuracy-vs-K curve at K in {8,16,32,64,128} for bun with the cue present AND with the cue removed/frozen at pass d. A cliff only in the cue-removed condition = the C6 hold is cue-driven (FREEZE); no cliff in either = genuine attractor.
- Verified: abs page fetched yes; already in bib: no

### 2609.36653 — Scheduling Recursive Reasoning in Looped Transformers (Sep 2026)
- Claim: Decomposes the time-averaged terminal-loss dependence on step size into persistent progress vs centred fluctuation; TAPS tracks their balance online and adapts the update scale, reaching target quality in fewer loops without retraining; adding the principle to training gives up to 1.56x wall-clock at matched accuracy.
- Why it matters to us: H1 IN-PASS HALT (a halt signal that needs no answer readout), H6 DONE-READOUT alternative, C5.
- Lead: $0 read: over a window of 3 passes, split the per-pass state delta into mean (progress) and variance (fluctuation); define halt = progress-norm below fluctuation-norm. Compare the halt pass against d on d<=8 items and against the "3 unchanged answers" pass; if it fires at d+1 rather than d+3 it saves 2 passes for free.
- Verified: abs page fetched yes; already in bib: no

### 2610.25212 — Calibration-Free Early Stopping for Chain-of-Thought Reasoning via E-processes (Oct 2026)
- Claim: Treats stopping as sequential evidence accumulation: two e-processes on one per-chunk signal (position vs slow trend; signed velocity) with Ville's inequality giving anytime-valid false-alarm control, no calibration set; matches or beats calibrated stopping on the token–accuracy Pareto front.
- Why it matters to us: C5; direct replacement for the hand-set "3 unchanged answers" rule; the signal is text-CoT here but the construction is modality-agnostic.
- Lead: run an e-process on the per-pass answer-logit margin (or per-pass answer-digit agreement) of bun at alpha=0.05; measure halt pass vs d and accuracy at the halt. Success criterion: same accuracy as the 3-unchanged rule with mean halt pass <= d+1.
- Verified: abs page fetched yes; already in bib: no

### 2609.33540 — Reasoning on the Simplex: Geometric Fixed-Point Models (Sep 2026)
- Claim: Recurrent state is a product of simplices (categorical beliefs); the argmax is the answer at every step and a fixed point exists for any parameters (continuous self-map). ~7M model: 95.1% Sudoku-Extreme, 92.0% Maze-Hard, 100% S_5 at length 128, above FPRM; a 201M LM on FineWeb-Edu beats GPT-2 small on four zero-shot MC tasks.
- Why it matters to us: H6 DONE-READOUT, C5/C6: a state whose readout is the answer at every pass makes answer-stability halting a property of the state, not of a separate readout.
- Lead: our values are mod 11, so a per-pass belief over 11 classes is cheap: add a per-pass answer-belief readout loss (we already supervise streamed digits) and use belief entropy < eps as the halt signal; compare halt pass with the 3-unchanged rule on d5–d8.
- Verified: abs page fetched yes; already in bib: no

### 2610.14252 — Adaptive-Depth Is a Finishing Stage (Oct 2026)
- Claim: Fixed-first training: train a looped transformer at a fixed shallow depth for the first ~2/3 of the run and vary depth only in a short final phase. 200M: fixed depth 2 for 67% of training cuts modeled compute 45% with mean NLL across inference depths within 0.5%; 1B: 31–37% compute, 32–39% wall-clock, NLL within 0.8%.
- Why it matters to us: training-cost rule for H5 HOLD-FROM-d+1 + RANDOM-REPEAT and any random-depth / depth-curriculum arm (K9), C2.
- Lead: for the next H5 arm, train at fixed K=d for the first 16k of 24k steps and enable random-repeat / hold passes only in the last 8k; gate on the d6–d8 extension matching the all-the-way variant within the registered threshold. Saves roughly a third of each cloud run.
- Verified: abs page fetched yes; already in bib: no

### 2603.02023 — PonderLM-3: Adaptive Token-Wise Pondering with Differentiable Masking (Mar 2026; missing from bib)
- Claim: Self-supervised token-wise pondering where allocation is learned via a differentiable attention mask during pretraining, paired with a matching hard pruning rule at inference so train and inference halting agree; lower perplexity at equal FLOPs than recursive/adaptive baselines, on par with fixed-step PonderLM-2 at fewer FLOPs.
- Why it matters to us: E4 HALT-COST, C5 (strict-C4 compatible: halting trained from the LM loss only, no halt labels).
- Lead: implement the continue-mask as a straight-through per-pass gate in the loop, trained on answer CE + per-pass cost, with identical hard pruning at eval; the test is whether the learned halt pass tracks d on held-out d5–d8 (the §248 failure) when the train/inference rule is forced to agree.
- Verified: abs page fetched yes; already in bib: no

### 2602.08864 — Understanding Dynamic Compute Allocation in Recurrent Transformers (ANIRA) (Feb 2026; missing from bib)
- Claim: Complexity-controlled algorithmic/synthetic tasks with a per-token variable-compute recurrent Transformer; compute allocation matching task complexity emerges without difficulty labels, but early compute decisions rely on static structural cues while online halting tracks execution state; models fail to generalise to longer unseen inputs even when they spend more compute.
- Why it matters to us: X1 CUE-CLOCK is exactly their "static structural cue vs execution state" split; C2.
- Lead: fit a halt-pass probe on the pass-1 state only and a second on the running state at pass t. If the pass-1 probe predicts the halt pass as well as the running probe (on d<=4 and on d5–d8), the halt is clocked by the input (op-sentence count / cue), not by execution; report alongside X1.
- Verified: abs page fetched yes; already in bib: no

Also verified in-window but lower relevance (not ranked): 2610.06949 AdaLoop (learned-halt looped block for audio LMs, Oct); 2611.00983 segmentation controls where token-level adaptive compute goes (Oct 31); 2609.37432 Looped Actor (state-dependent loop count in RL, loop count rises with remaining pushes on Boxoban, Sep); 2610.23225 ReLaDrift (random-round training, one-round backprop, Oct); 2606.30852 LearnStop (cost-aware early exits for text reasoning: answer-run-stability competitive on MC and hard tasks, Jun; missing from bib).
Seen in listings but not verified (not reported): 2609.37379 Looped Transformers as Optimizers; 2604.21999 Universal Transformers Need Memory; 2603.01914 AdaPonderLM; 2605.13872; 2607.11116 (DEQ equilibrium = initialisation critique); 2604.04930 CoDE-Stop; 2605.05222.

Queries tried: WebSearch "adaptive computation time language model halting arXiv 2026"; "learned halting recurrent looped transformer arXiv October 2026"; "ponder latent reasoning early exit looped transformer 2026 arXiv"; "when to stop thinking latent reasoning halting overthinking arXiv 2026"; "answer stability stopping rule looped language model halting reward arXiv 2026"; "fixed point convergence stopping deep equilibrium language model reasoning 2026 arXiv"; "halting head generalize beyond training depth recurrent depth transformer extrapolation 2026"; "looped language model halting fixed point convergence exit latent reasoning October 2026". arXiv Atom API: all:"adaptive computation" AND transformer (50); all:halting AND (transformer OR "language model" OR recurrent) (50, two pages); all:"test-time compute" AND (recurrent OR looped OR "latent reasoning") (50); ("early exit" OR "adaptive depth" OR pondering) AND ("looped transformer" OR "recurrent depth" OR "latent reasoning") (50). Three further Atom queries (fixed point AND halting; overthinking/early stopping AND latent; halting AND reward) were refused with HTTP 429 and could not be read.

Candidates rejected as already-known: 2611.02333 (When Does Adaptive Halting Win?), 2609.03379 (RecurTrace), 2609.35748 (Adaptive Looped Transformers TTS), 2607.20519 (Diagnosing Learned Halting Gates), 2606.29983 (Learned Stochastic Stopping), 2607.20594 (Convergence Selection), 2606.18206 (FPRM), 2605.21488 (Equilibrium Reasoners), 2605.12466 (Attractor Models), 2604.07822 (Loop, Think, Generalize), 2606.24898 (Readout Blind Spot), 2608.11233 (Retrofitting Recurrent Depth), 2511.21581 (Learning When to Stop, 2025), 2509.24238 (Learning to Ponder, 2025), 2510.25741 (Ouro, 2025).

---

## Faithfulness and steering of latent reasoning (lead agent; arXiv API rate-limited mid-sweep, Atom queries: "latent reasoning" AND faithful/interpret/monitor/steer; "continuous thought(s)"/"soft thinking"; "chain of thought" AND faithfulness / monitorability)

### 2610.18127 — Monitoring latent reasoning in looped language models (Oct 2026)
- Claim: five-lens mech-interp protocol applied to eight looped models (Ouro, LoopUS, Parcae, Rec-OLMo, Huginn). Key split: "carry loops" (prompt lives only in the looped state) vs "re-read loops" (prompt re-supplied each iteration); asks whether extra iterations add capability and whether loops stay stable under prompt perturbation.
- Why it matters: C1 / C6 / C7. Our K=8 loop is a re-read loop with prompt recompute (C3 row); the carry-vs-re-read distinction is exactly the KV-LAT / E3 question, and their perturbation-stability lenses are a ready-made X2 FREEZE-vs-FIXED-POINT battery.
- Lead: port their five lenses to bun (per-pass logit lens, prompt-perturbation drift, iteration-ablation) as a $0 read; report whether bun's pass-7+ degradation (264-1) is a carry failure (state forgets the prompt) or a re-read failure (attention dispersal, D1a).
- Verified: abs fetched yes; in bib: no.

### 2605.21642 — Ablate-to-Validate: Are VLMs Really Using Continuous Thought Tokens? (May 2026)
- Claim: Token Replacement Test (zero / random / first-repeat / oracle substitutes for the latent tokens at fixed budget). Three published visual-thinking systems keep most of their gains with corrupted latents: a channel exists but is not used as a bottleneck.
- Why it matters: C1 — the same test the project already runs as latent-removal / SWAP, but with a four-way substitution ladder (zero, random, repeat-first, oracle) that separates "presence" from "content". Backlog X5 ONE-HOP-PIPELINE, R9.
- Lead: add the "first-repeat" arm (replay pass-1 state at every pass) to the C1 scorecard; a model that survives first-repeat is using the slot, not the trajectory. Cheap: one eval flag.
- Verified: yes; in bib: no.

### 2608.04928 — Does Out-of-Sight Equal Out-of-Mind in CoT Monitorability? (Aug 2026; Ferreira, Aziz, Titov)
- Claim: compares hint-reliance monitors across explicit CoT, weakly- and strongly-supervised latent CoT. Monitorability depends more on task properties and monitor access (probes, latent-to-text translation) than on reasoning mode.
- Why it matters: C7. Gives the two monitor families (activation probe vs latent→text translator) we call R1 TEXT-FREE READOUT and R4 CYCLE-CONSISTENCY, and a hint-reliance target that is a cheap faithfulness probe without edits.
- Lead: a hint-reliance C7 test for bun: inject a leaked answer digit into the prompt; measure whether the pass-t readouts (not just the answer) shift toward it. Faithful readouts should expose the reliance; $0 on stored checkpoints.
- Verified: yes; in bib: no.

### 2609.22197 — Dissecting Hierarchical Reasoning Models: A Mechanistic Study (Aug/Sep 2026)
- Claim: HRM on Sudoku/Maze/ARC: single-state recurrent transformers match HRM; linear probes decode task variables but ablating probe directions = random controls; SAE ablations do more but top features show no stable advantage over random size-matched sets. HRM = constraint-aware iterative refinement without a compact causal feature set.
- Why it matters: C7 / C1 methodology — a warning that "decodable" and "causal" diverge in recursive latent models; directly relevant to R2 DAS-EDIT's IIA bar and the §247 "readouts faithful, edits null" pattern.
- Lead: add a random-direction ablation control to every probe-based C1/C7 claim (same k as the DAS subspace); if bun's probe-direction ablation ≈ random, the value is in a distributed code and R3 codebook edits should target the write increment, not a probe direction.
- Verified: yes; in bib: no.

### 2610.28540 — Predictive Latents for Autoregressive LMs: Possibility, Gauge, and Obstruction (Oct 2026)
- Claim: theory for latent-iterating LMs (continuous thoughts, pause, recurrent depth). Three risks: token, teacher-forced latent, open-loop. A latent is identified only up to gauge; low next-latent loss does not imply token prediction and low token loss does not imply stable free-running rollout — null-space drift can grow linearly with horizon unless a rank/injectivity condition holds.
- Why it matters: C6 / C2 — a formal statement of the onset-drift and K′128 stability questions (259-4, 251-8); argues the hold penalty is a rank condition in disguise.
- Lead: measure the open-loop vs teacher-forced gap on bun directly (feed gold pass-t states vs own states for passes 5–8); if the gap grows linearly in passes, add an injectivity-style regulariser (readout-rank or whitening) rather than another hold penalty.
- Verified: yes; in bib: no.

### 2611.02672 — Chain of Faith: Policy-Internal Rewards for Reasoning Reliance (Nov 2026)
- Claim: four GRPO rewards from the policy's own signals (gradient sensitivity, confidence gap, progressive confidence gain, reasoning-prompt swap) to make answers depend on the generated reasoning; raises reliance on edited reasoning but accuracy/paraphrase robustness vary and high weights cause answer copying.
- Why it matters: C1 + C4 — a reward that pays for latent-removal sensitivity is the RL version of our C1 gate; also a Goodhart warning (copying) for K5 DISCRETE-GRPO.
- Lead: in K2/K5, add a "progressive confidence gain" term (Δ log p(answer) across passes, cf. D9 TRAJECTORY REWARD) and check for the copying failure with the SWAP gate before scoring.
- Verified: yes; in bib: no.

### 2605.18549 — Monitoring the Internal Monologue: Probe Trajectories Reveal Reasoning Dynamics (May 2026, v2 Nov 2026)
- Claim: run a concept probe at every reasoning token, extract volatility/trend/steady-state features of the probe trajectory; +21.7 AUROC where static probes are weakest; template-trained probes work almost as well as model-generated data; pooling needed.
- Why it matters: C5 / C6 — a per-pass probe trajectory (value-readout confidence over passes) is a halt signal candidate that is not the "3 unchanged answers" rule (H6 DONE-READOUT, X1 CUE-CLOCK).
- Lead: compute the readout-probability trajectory over passes 1–8 on d≤8 items and fit a tiny halt classifier on its trend/steady-state features; compare against the answer-stability rule on d7/d8 stopping precision.
- Verified: yes; in bib: no.

### 2608.02585 — GradCuit: Credit-Assigned Gradient Flow Enables Robust and Interpretable Test-Time Latent Reasoning (Aug 2026)
- Claim: optimise instance-specific latent states inserted at a chosen layer between prompt and continuation with frozen weights; +6.6 over CoT, more stable than LatentSeek; latent influence concentrates on connector tokens, early-middle layers best.
- Why it matters: C7 / A2 MULTI-SITE READ — evidence on which layer a latent should be injected to have causal effect; relevant to the "write ignores text" failure (247-2).
- Lead: a $0 layer sweep: inject the §247 text-derived edit at blocks 2/6/10 instead of the write site and measure edit-follow; if mid-layer injection is followed, the C7 fix is injection depth, not the encoder.
- Verified: yes; in bib: no.

### 2609.38717 — Soft Spatial Reasoning (Sep 2026)
- Claim: per-step softness controller (AdaptSoft) reads hidden state + predictive uncertainty to choose how soft the blended-embedding step is; trained with a gradient-alignment objective, no intermediate supervision.
- Why it matters: K10 DISCRETE→CONTINUOUS ANNEAL — a learned, per-step anneal instead of a schedule; C4-compatible (no step labels).
- Lead: replace K10's fixed anneal with an uncertainty-gated softness on the write (hard centroid when the ':' readout is confident, soft otherwise).
- Verified: yes; in bib: no.

Lower relevance, verified new: 2610.34173 (latent chemical reasoning: latents act as soft prompts — same shortcut as §207/§208), 2604.17121 (Mozer et al., position paper: state tracking needs recurrent dynamics; taxonomy of depth- vs step-recurrence), 2607.10296 SPARK (susceptibility-guided steering of hidden reasoning states, 7B+).
Rejected as already known: 2609.21662, 2606.01243, 2606.27378, 2607.06648, 2606.12689, 2604.23460, 2603.03475, 2604.04902, 2602.22441, 2602.00449, 2512.21711, 2609.01924.

---

## Small-model reasoning benchmarks (lead agent; Atom queries: ProsQA/ProntoQA/"shuffled objects"; "implicit chain of thought"/GSM8K+GPT-2/GSM8K+small LMs; web searches for Deep ProsQA, entity tracking, GPT-2 GSM8K 2026)

### 2611.00783 — A Tool in the Forward Pass: Frozen Tape Modules as Layers of a Small LM (Oct 31 2026)
- Claim: a learned writer maps hidden states to a short discrete "tape", a frozen module reduces it, a reader injects the result; 750/750 on BBH tracking-shuffled-objects with labelled tapes, ~1/10 the GPU time of a text-tool call at 7 objects. Writer trained from final answers only (with a copying prior) succeeds 5/6 runs on gemma-3-1b, 2/4 on 4B.
- Why it matters: N1 SHUFFLED OBJECTS comparator and a C4 data point: answer-only training of a discrete in-forward-pass state works but is seed-fragile, mirroring our 1/4 answer-only formation (§244/§245). Also an R8 FSQ / discrete tied write precedent.
- Lead: adopt their answer-only "copying prior" idea for K1/K5: initialise the write head as a copy of the operand token embedding (a hand-written prior) and train only the answer loss; compare seed success rate against K1's exact EM.
- Verified: yes; in bib: no.

### 2610.05815 — Plan Canvas: Fixed Reasoning Regions for Continuous Language Flows (Oct 2026)
- Claim: in continuous (diffusion/flow) language models, fix the trace/answer boundary with a fixed-capacity plan region and supervised padding; on Deep ProsQA (longer proofs) accuracy 73.0 → 87.0, valid-path share 30.8 → 59.1, gains largest on the longest proofs.
- Why it matters: T3 ProsQA OOD / C2 — "Deep ProsQA" is an existing longer-proof ProsQA split we should reuse for the PROSQA_OOD_PORT; also an external number for the table in EXTERNAL_SOTA.md.
- Lead: obtain the Deep ProsQA split from this paper and use it as the OOD depth axis for T3 instead of hand-rolling one; record 87.0 (flow model, fixed plan region) as the comparator.
- Verified: yes; in bib: no.

### 2610.09759 — Unrolled Flow Models for Reasoning (Oct 2026; Rusch group)
- Claim: flow LMs do not benefit from more integration steps unless trained through their own latent rollout on a random sub-interval and decoded only at the endpoint; ProsQA 97% and accuracy that improves with steps; sphere-projecting the latent stabilises long rollouts (Sudoku/Maze) and beats baselines with 3x parameters.
- Why it matters: C2 / C6 — training through own rollout + norm-projected state is the recipe for "more steps helps"; the sphere projection is a concrete alternative to our hold penalty (D8 NORMED RECALL READ, D3 CENTROID-SNAP).
- Lead: project bun's per-pass state onto a fixed-norm sphere before the write (one line) and retrain the K′ extension; test whether d8 and K′128 stability hold without the hold penalty.
- Verified: yes; in bib: no.

### 2605.22870 — The Readout Shortcut: Positional Number Copying Dominates Arithmetic CoT Readout in Small LMs (May 2026)
- Claim: 1–3B instruction-tuned models on GSM8K copy whichever number sits last before the answer delimiter: 95–96% match even on wrong items, gold-in-trace accounts for 89–92% of the teacher-forcing ceiling, wrong trailing number → near-zero accuracy; replicates on GSM-Symbolic; gating emerges only at 7–8B.
- Why it matters: X3 TEXT-CoT-D4 and C3 — the text-CoT comparator at GPT-2 scale will largely be a copy-last-number readout; our "value transport" critique of bun (NORTH_STAR_GAP (iii)) applies equally to text CoT, so the comparison must score last-step corruption on both.
- Lead: in X3, add the paper's trailing-number corruption test to the text-CoT arm and the equivalent last-pass state corruption to bun; report both so C3 compares computation, not transport.
- Verified: yes; in bib: no.

### 2609.07406 — Think Wider: Mitigating Latent Rank Collapse in Implicit CoT (Sep 2026)
- Claim: implicit-CoT latent states collapse onto one dominant direction; WIDER, a training-only spectral regulariser penalising projection on the shared direction, raises effective rank and improves matched implicit-CoT baselines without changing inference.
- Why it matters: C6 / C2 — a named failure mode for our pass trajectory; cheap to measure on bun (effective rank of S_1..S_8) and complementary to the Matrix-CODI rank-indifference result (2609.03090, known).
- Lead: $0 read: effective rank and dominant-direction energy of bun's pass states on d4 vs d8; if d8 failures coincide with collapse, add WIDER's penalty to the §265 successor.
- Verified: yes; in bib: no.

### 2607.02869 — Reward Granularity in RLVR: Process vs Outcome Rewards for Small LMs (Jul 2026)
- Claim: Qwen2.5-0.5B + GRPO on GSM8K: process-only 63.73% vs outcome-only 53.75%; hybrids improve with process weight except λ=0.1 which underperforms outcome-only.
- Why it matters: EXTERNAL_SOTA table (0.5B GSM8K text-CoT RL numbers) and K2/K5 design: a small outcome+process mix can be worse than either.
- Lead: when K5 adds any auxiliary per-pass term to the answer reward, sweep the weight at {0.9, 0.5} only; do not run a token 0.1 mix.
- Verified: yes; in bib: no.

### 2606.05315 — LoRi: Low-Rank Distillation for Implicit Reasoning (Jun 2026, v2 Oct 2026)
- Claim: teacher/student hidden-state reasoning trajectories are low-rank; aligning them in a shared low-rank tensor subspace (first- and second-order stats) beats prior iCoT distillation and approaches explicit CoT on multi-step math (LLaMA/Qwen scales).
- Why it matters: C4 (CoT-teacher targets are allowed) — a trajectory-level distillation target instead of per-step values; K7/K8 alternative.
- Lead: for the C4 ladder, distil bun's own pass trajectory (as teacher) into an answer-only student via subspace moments rather than per-pass values; this keeps the C4 wording since no simulator value is used.
- Verified: yes; in bib: no.

### 2611.01926 — TAC-CoT: Tool-Empowered Implicit CoT via Interleaving Explicit Tokens (Nov 1 2026)
- Claim: latent segments for reasoning, plain text only for tool calls/observations/answers; +8.8 accuracy and up to 6.8x fewer forward passes on agentic math/QA.
- Why it matters: C5 "when to write" — the explicit/latent interleave is the checkpointed design's external analogue.
- Lead: low priority; note as the current interleave reference for the write-gate design.
- Verified: yes; in bib: no.

### 2608.18083 — Entity tracking emerges in sub-billion parameter LMs and exceeds human performance in naturalistic narratives (Jun 2026, v2 Sep; EMNLP 2026)
- Claim: human-level entity tracking at ~410M parameters (Pythia/OLMo families), far below prior multi-billion code-model claims; humans degrade with narrative complexity, not length.
- Why it matters: N2 BOXES / N1 — a base-model capability floor: GPT-2 small (124M) is below the 410M threshold, GPT-2 medium (355M) near it; relevant to the M-line decision (base size).
- Lead: before N1/N2, run their naturalistic entity-tracking probe on GPT-2 small vs medium zero-shot to set the expected no-loop floor.
- Verified: yes; in bib: no.

Rejected as already known: 2609.03090, 2609.35643, 2606.07720, 2607.06648, 2602.01148, 2603.20219, 2609.09928, 2509.20317, 2510.00184, 2605.28600, 2604.04902, 2602.22441. Out of scope (not small-model or not our tasks): 2602.13738 OneLatent (visual), 2606.06920 (PEFT for sub-1B math; adjacent), 2605.22476 (structured-sparse attention for entity tracking; collapses when evolving properties > heads — worth a footnote for N2).
No 2026 paper found reporting new ProsQA/ProntoQA numbers for GPT-2 beyond those already in bib (2604.04902: GPT-2 Coconut 98.0 ProsQA; 2602.22441 shortcut warning).

---

## Pending arms (launched 17:05 EST, not reported at write time)

- continuous/latent CoT training (RL/EM for latents) — sub-agent running; interim: 2609.07406 WIDER, 2606.05315 LoRi, 2609.38717 AdaptSoft above.
- depth/length extrapolation — sub-agent running; interim: 2610.27628 PAR, 2610.31943 above.
- variable binding / entity tracking / multi-hop — sub-agent running; interim: 2608.18083, 2611.00783 above.
- associative memory / delta-rule / linear-attention memory — sub-agent running; interim: 2610.07907 Continuous Memory Machines (Oct 2026, verified in listing only, abs not fetched).

When the four reports land they should be appended under this heading; their scratchpad files are topic_latentcot.md, topic_extrap.md, topic_binding.md, topic_memory.md in the lead agent's scratchpad.


## Pending arm landed: associative memory / delta rule / TTT memory

# Literature sweep — associative memory / delta rule / TTT memory (2026-11-03)

Window: arXiv 2609.*–2611.* plus major 2601–2608 papers missing from the bibliography. All ids below were verified by fetching https://arxiv.org/abs/<id> (title + v1 date) and grepped against known_ids.txt (none present). Note: export.arxiv.org Atom API returned HTTP 429 for every query this session (rate-limited); the arxiv.org/search listing pages were used instead, plus WebSearch.

### 2609.39740 — LatentHarness: Learning Latent Actions for Memory and Reasoning via Counterfactual Policy Distillation (Sep 2026)
- Claim: A 1.4B model picks, at every internal step, one of three latent actions: think further, recall from a fast-weight memory that stores input evidence AND intermediate reasoning states, or exit and emit the token. The policy is trained by counterfactual distillation: branch one step on each action and score its effect on the emitted token. Beats the strongest baselines by 2.8-10% relative on six reasoning / long-context benchmarks, 5.9x faster. Work in progress.
- Why it matters to us: C5 (learned halting and when to write) and C1; A7 TITANS/TTT MLP head; H1 IN-PASS HALT; the recall action is a text->state path missing for C7.
- Lead: Add a per-pass 3-way latent action head (write / hold / exit) to the GPT-2 loop and train it with their one-step counterfactual signal: branch the pass under each action, measure the answer-logit delta, distil the argmax. This gives a halt that is a decision rather than the C6 hold, and a write that is gated by whether writing changes the answer. Cheap: K=8 passes x 3 branches at eval-only cost for the label.
- Verified: abs page fetched yes; already in bib: no

### 2610.36794 — Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training (Oct 2026)
- Claim: In-place TTT (MLP down-projection as fast weights) loses its gains on realistic long-context QA because every chunk is written with equal strength. Select-TTT scales each chunk's write learning rate by the probability mass the chunk's hidden states put on the query tokens under the frozen output head; no new parameters, no supervision; consistent gains on two backbones.
- Why it matters to us: C7 (the write ignores text, 247-2) and C5 (when to write); R6 TOKEN-WRITE; P4 SOFT-WRITE.
- Lead: Make the additive per-pass write text-conditioned without new parameters: multiply the pass-t write by a temperature-softmaxed score = LM-head probability that the current op-sentence's hidden states assign to the cue / S_t tokens. An edit to the op sentence then changes the write strength directly, which is exactly the path the §247 edit test found absent. Measure the C7 edit test and the §247 readout on bun with this one change (args diff: one flag).
- Verified: abs page fetched yes; already in bib: no

### 2609.37379 — Looped Transformers as Optimizers (Sep 2026)
- Claim: Treats the looped hidden state as a fast weight updated across depth: each loop transition = projection + local objective + optimizer step toward an implicit target. Existing loop designs have transition/projection mismatches; aligning them lowers loss, and OperLoop (explicit weight decay, adaptive step size, delta objective) beats looped and non-looped baselines at matched FLOPs.
- Why it matters to us: C6 (stable fixed point), C2; A6 WRITE TOKENS + DeltaProduct; D8 NORMED RECALL READ; H1 SCALE-FREE CONVERGENCE EXIT.
- Lead: Our per-pass write is s_{t+1} = s_t + f(x, s_t), a plain additive step. Re-parameterise it as a delta step s_{t+1} = (1 - lambda_t) s_t + beta_t (target_t - read(s_t)) with learned scalar decay lambda_t and step beta_t, i.e. write only the residual the read gets wrong. The adaptive step beta_t -> 0 at convergence gives a halt signal that does not depend on the repeated cue (X1/X2). Compare the hold-penalty C6 result (259-4) against this decay-only variant at K'128.
- Verified: abs page fetched yes; already in bib: no

### 2610.13073 — Double-Take: Improving Capacity, Binding, and Selectivity In Long-Context Within Bidirectional Memory (Oct 2026)
- Claim: In large-chunk TTT, single-step writes pile up interference (capacity) and self-associative writes cannot form bindings across positions. Multi-step unpinned updates + cross-position mixing + change-sensitive bidirectional writes fix this: capacity alone leaves end-to-end recall at chance (0.10); adding binding raises it to 0.92. Shown on genomic contexts to 1 Mb.
- Why it matters to us: C7 and C1; diagnoses the §247 failure mode: a write whose key and value both come from the state is self-associative and cannot bind text to state. D1 DENSE-DISTRACT; R6 TOKEN-WRITE.
- Lead: Split the write into key-from-text / value-from-state: key = projection of the current op-sentence tokens (cross-position), value = the loop state's update. A text edit then changes which address is written. Run their ablation ladder on REG d<=4: (a) capacity-only (more state dims), (b) + text-keyed binding, (c) + change-sensitive write; expect (a) to leave the C7 edit test null and (b) to move it.
- Verified: abs page fetched yes; already in bib: no

### 2609.07816 — Kalman Delta Networks: Uncertainty-aware Associative Memory (Sep 2026)
- Claim: Models the associative memory as a linear-Gaussian state-space system that tracks a memory estimate and its uncertainty; the write strength is a Kalman gain that weighs accumulated evidence against observation reliability. DeltaNet / Gated DeltaNet / KDA fall out as the isotropic-covariance special case. Diagonal and isotropic variants run as associative scans; gains at 750M and 1.3B.
- Why it matters to us: C6, C5 (when to write); A6 WRITE TOKENS + DeltaProduct; A5 ROUTED MULTI-SLOT gated-delta.
- Lead: Replace the fixed per-pass write gate with a per-channel Kalman gain driven by a tracked uncertainty vector (O(d) extra state). Once a slot has been written consistently its gain collapses, so later passes stop overwriting S_d, and the gain vector itself is a halt readout (sum of gains -> 0). Test against the cue-driven freeze question in §264 X1: if gains fall without the cue, the fixed point belongs to the loop.
- Verified: abs page fetched yes; already in bib: no

### 2610.02816 — Gated Slot Attention-2: Two-Sided Associative Memory Correction in Linear Attention (Oct 2026)
- Claim: Combines delta-rule value correction with slot-based two-stage key/value memory: Gated Oja Rule-2 (key side, separate erase and write controls) plus Gated Delta Rule-2 (value side) through shared latent slots; chunkwise parallel algorithm; beats strong linear-attention baselines.
- Why it matters to us: A5 ROUTED MULTI-SLOT gated-delta; A3 GATED CROSS-ATTENTION EPISODIC SLOTS; D8 NORMED RECALL READ.
- Lead: Give the latent loop a small bank of M shared slots (M = max depth, 4-8) with the GSA-2 two-sided rule: text keys address slots (Oja side keeps keys normalised), the pass-t state writes the slot value with separate erase and write gates. Read = query over slots. The erase gate gives a native "overwrite S_t with S_{t+1}" operation; the slot index is a cheap C1 probe (does pass t write slot t).
- Verified: abs page fetched yes; already in bib: no

### 2602.00015 — G-MemLLM: Gated Latent Memory Augmentation for Long-Context Reasoning in Large Language Models (Jan 2026; major 2026 paper missing from bib)
- Claim: Frozen LLM backbone + trainable latent memory bank of fixed learnable slots; a GRU-style gate decides per slot whether it is updated, preserved, or overwritten, to keep bridge entities across multi-hop chains. Evaluated from GPT-2 124M to Llama-3.1-8B on HotpotQA and ZsRE (+8.56 Answer-F1 for GPT-2 on HotpotQA).
- Why it matters to us: A3 GATED CROSS-ATTENTION EPISODIC SLOTS almost verbatim, at our base size; C7 (slots are written from text hidden states, so text edits reach the state); N4 TWO-HOP.
- Lead: Implement A3 as their slot bank on the frozen GPT-2 loop: slots updated by a GRU gate from the pass's text hidden states, read by cross-attention before the write. Their GPT-2 numbers give a sanity bar. Check the C7 edit test first: with a text-written slot bank the edit path exists by construction.
- Verified: abs page fetched yes; already in bib: no

### 2610.26834 — Kronecker Delta Rule for Long Context Linear Attention (Oct 2026)
- Claim: KronLSTM (Hochreiter group) revises an associative matrix memory along a single direction per update (Kronecker delta rule), which interferes less with stored associations than the full delta rule. Memory dynamics form a Markov chain whose expected memory norm stays bounded over the whole sequence, so no extra normalisation is needed. Competitive with / better than (Gated) DeltaNet, strongest on long context.
- Why it matters to us: C6 (bounded state norm without a hold penalty); D8 NORMED RECALL READ; D11 RMS-MIX READ.
- Lead: Our additive write has no norm bound (the hold penalty supplies one). Swap the write to a single-direction Kronecker-delta revision and drop the hold penalty; measure state norm across K'128 passes (C6) and whether the fixed point survives cue removal (X2). Their bounded-norm argument is the math backing the user asks for before an architecture change.
- Verified: abs page fetched yes; already in bib: no

### 2610.41994 — DAMI: Dual-Axis Memory Initialization for Enhanced Long-Context Retrieval in Gated Linear Attention (Oct 2026)
- Claim: In Gated DeltaNet the decay-gate parameters stay strongly correlated with their initial values after training, so initialisation decides memory timescale. DAMI assigns different timescales per head and a low-high-low input-sensitivity profile across layers, calibrated from a small activation sample; +5.3 to +16.7 points on RULER retrieval across GDN, KDA, GDN2, Mamba-2, RWKV-7 with no new parameters.
- Why it matters to us: D6 RANDOM-START MEMORY (how the memory starts matters more than expected); C6; also the basin-entry sensitivity we already see (seed and card flip the basin).
- Lead: Before training, calibrate the loop's write/decay gate so that different state channels have different effective timescales (some persist across all K passes, some reset each pass), using a few hundred activations. Compare basin entry rate over seeds against the current init; if gate init pins timescales, the seed-1 drift (basin-entry mechanism) may be an init artefact.
- Verified: abs page fetched yes; already in bib: no

### 2611.03094 — A Fine-Grained Hierarchy and Benchmark for State Tracking in Language Models (Nov 2026)
- Claim: Petty, Merrill, Sabharwal give a four-tier hierarchy of word problems (r-trivial immutable assignment; aperiodic mutable assignment; soluble self-referential assignment; general finite-state). Transformers' length-generalisable class excludes some aperiodic problems; GDN with positive eigenvalues expresses all aperiodic problems. StateBench: long-horizon benchmark; no pretrained hybrid or frontier model reliably solves even the soluble tier.
- Why it matters to us: B2 HELD-OUT STRUCTURE eval, N1 SHUFFLED OBJECTS, N2 BOXES; C2 depth extrapolation. Our REG word programs are mutable assignment (aperiodic tier) so the hierarchy places them precisely and gives a harder next tier (soluble: self-referential assignment) within the same template family.
- Lead: Port StateBench tier 2 (aperiodic) and tier 3 (soluble) items into the REG template at d<=4 and run the trained bun lineage eval-only; a tier-3 failure with tier-2 pass is a clean statement of what the loop computes, and it is a $0 read.
- Verified: abs page fetched yes; already in bib: no

## Also verified (in window, lower relevance; one line each)
- 2610.38578 Dual DeltaNet (Oct 2026): value-side delta rule reorganises where a value is stored (KDA extension). A6.
- 2609.32312 Delayed Supervision for Test-Time Language Models (Sep 2026): simulator-grounded questions asked after a delay on disposable branches; retain vs revise; +5.5 BABILong for LaCT. Training-signal idea for D4 CORRUPT-AND-RECOVER; note it uses simulator-grounded questions (C4 caveat).
- 2610.16268 Error Correction in Trained TTT Layers (Oct 2026): the prediction-derived component of the TTT inner gradient cancels at the released init; ablations on real checkpoints. Audit-style lesson for A7: check the write actually uses the prediction error.
- 2610.33025 Can Language Models Learn While They Think? (Oct 2026): gradient adaptation while reading captures style, not retrieval; storing facts via NTP memorises format; distilling own conclusions into a LoRA works. Negative result relevant to A7 TITANS/TTT at small scale.
- 2610.08553 DeltaTTT (Oct 2026): sequential TTT does not help nonlinear memories; layerwise local targets with a state-dependent delta rule do. A7 MLP head design.
- 2610.15185 DeltaNet-PC (Oct 2026): predictive-coding derivation of DeltaNet-style updates; perfect key-value recall without BPTT. A7.
- 2610.37645 MERLIN: Multi-State Linear Attention with second-order routing (Oct 2026): routes queries over per-segment memory states using key mean+variance. A5 ROUTED MULTI-SLOT.
- 2609.39034 Switching Linear Attention (Sep 2026): state update as online EM over a mixture of linear regressions; per-dimension component selection. A5.
- 2611.01909 Who Asks, Who Answers (Nov 2026): in GDN/Mamba hybrids the linear-attention state builds the retrieval query; transplanting it redirects retrieval 0.97-0.99. Mechanistic template for an R-bundle read (which part of our state carries the "which S_t" pointer; P5 POINTER HEAD).
- 2610.19210 Titan-Llama (Oct 2026): retrofit of Llama-3.2-1B with test-time-trained hierarchical memory. S-lane / A7 on a pretrained base.
- 2610.20582 Latent Compaction with TTT (Oct 2026): document-specific compact KV cache trained at test time with frozen weights. C3 cached-prompt redesign (251-12).
- 2610.41291 Agentic Soft-Token Compression (Oct 2026): training-free; uses GDN recurrent dynamics to decide what has been absorbed and carries it as soft tokens. A4 COCONUT HYBRID adjacent.
- 2610.18809 Depth Memory (Oct 2026): cross-layer mixing as recursive-least-squares associative memory over depth. Loop-depth read design.
- 2610.06516 Bayesian Cheatsheet (Oct 2026): HDP-GMM memory of reasoning behaviours adapted at test time. Verbal-strategy memory, not latent; low relevance.
- 2605.27686 Tensor Memory (May 2026, outside window): fixed-size recurrent 3D memory tensor with soft write and gated read. Mostly vision.

Queries tried: WebSearch: "arXiv 2026 delta rule linear attention gated delta reasoning"; "arXiv October 2026 test-time training memory language model Titans"; "arXiv 2026 associative memory transformer multi-step reasoning Hopfield"; "arXiv 2026 fast weights working memory latent reasoning language model"; "arXiv 2026 memory slots episodic working memory transformer multi-step reasoning write gating"; "arXiv 2026 DeltaProduct state tracking recurrent memory in-context binding"; "arxiv.org 2610 OR 2609 test-time memory language model recurrent reasoning associative recall"; "Learning State-Tracking from Code linear RNN arXiv 2026"; "recurrent memory transformer OR memory tokens multi-step reasoning latent state write gate". arXiv Atom API (export.arxiv.org): 10 query strings ("delta rule", "associative memory" AND transformer, "test-time training", "gated delta", "fast weights", "linear attention" AND memory, "working memory" AND reasoning AND transformer, "memory slots" AND reasoning, Titans AND memory AND "test-time", "state space" AND binding AND "in-context") — all returned HTTP 429 this session. arxiv.org/search listings (sorted by submitted date): "delta rule" attention; "associative memory" transformer; "test-time training" memory; DeltaNet OR DeltaProduct OR "gated delta"; "fast weight" OR "fast weights"; "working memory" reasoning transformer; "memory slots" OR "memory tokens" reasoning; "linear attention" memory recall; "memory tokens" reasoning; "episodic memory" "language model" reasoning; Hopfield transformer reasoning; "latent reasoning" memory recurrent.

Candidates rejected as already-known: 2609.16183 (Anatomy of Associative Recall), 2609.33093 (How Linear Attention Remembers), 2605.30343 (Reasoning in Memory), 2605.22791 (Gated DeltaNet-2), 2606.07720 (AGCLR), 2603.08391 (Adaptive Loops and Memory), 2602.14814 (State-Tracking from Code).
