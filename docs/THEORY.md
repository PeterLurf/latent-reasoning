# Theory: latent reasoning (condensed)

The Theory session's registrations, results and notes. **§0–§96 are in full, verbatim, in
[`docs/archive/THEORY_s0-s96.md`](archive/THEORY_s0-s96.md)** (moved 2026-10-03 at the user's request; every note and result is
filed under its own §; search "## N." or "Note N-"). §97 onward are below in full. New notes on archived sections are
appended at the end of this file.

**Research priority (the user, 2026-10-04 ~00:45): C1 > C6 > C4 = C2 > C3 > C5 > C7.** Favour designs where the state
can travel only through the latents (MASK, MARKOV), with removal and bypass controls on every arm. Accuracy with failed
removal is the headline. 5 seeds and long-rollout drift for any passing arm (C6). Train short, test long on every new
task (C2). C3 / C5 / C7 are reported where cheap, not designed for.

## Where things stand (2026-10-03)

* **North star:** one pretrained LM meeting C1–C7 (`docs/NORTH_STAR.md`).
* **Range principle (§26, §64, §66, §71).** Every length failure is a quantity that grows with the problem and leaves its
  trained range. Every working fix restarts from a written or snapped state with the history cut.
* **Latent restarts in GPT-2 (§86, §92).**
  - S5 carried in latent slots: exact to 2,048 from training at 128 (k = 4, 5/5), and certified at every length
    (k = 1, 0 / 14,400 errors).
  - Removal ≈ 0, at 0.29–0.45× the written state's compute. C2 + C3 + C6, scaffolded.
  - **C1 by donor patching (§104): slot-only serial use in 2/5 seeds**, with the slot and op tokens jointly in 3/5.
* **General latent thinker (§97).** No hand-built state: a text step target through the LM head, plus re-embedded text
  checkpoints.
  - S5 exact to 2,048 (3/3), at 0.56× the written state's compute (0.61× with restricted vocabulary).
  - Checkpoint edit IIA 1.0.
  - **C1 not supported (§104):** donor latents are ignored and the state is recomputed from visible ops (recipient-
    consistent up to 1.0). HYB-4 s5 keeps 1.000 under mean ablation (97-3). C6 met 5/5.
  - C6 seeds are running. **§98** (the same recipe on 4×4 / 5×5 multiplication, against ICoT-SI 0.998 / 0.996,
    Coconut and text CoT) is running.
* **C4 (§93, §95, §96).**
  - Thinning labels inside the LM fails (0/3, 0/9).
  - A free-table teacher trained on final answers (5 % short batches) certifies 4/5 and reproduces §92's per-op labels
    exactly. C4 for the pipeline, qualified by p ≤ 0.05 and oracle M; the LM hosts.
* **Free-table acquisition (§76, §82, §85, §87).** It needs short labels (≥ 3–10 %) or a short-label warm start. The ge3
  failure is optimisation (a symmetric saddle), not information.
* **Stable latent at S3 (§72b, §73):** a standardised readout gives 10/10. Pretraining buys operation discrimination, not
  paraphrase transfer (§74, §79).
* **In-depth composition (§88, §90, §91):** per-iteration targets, word diversity and enough steps give 5/5.
* **Fade to latent (§89, §94).** At 3× budget, both question types are at 1.000 (5/5).
  - **Closed** (the user, 2026-10-03): its working ingredient, the side step decoder, lives in §97.
* **GSM8K at GPT-2 small (§68, §78, §84):** written-state restarts do not generalise by step count. Closed; no GSM8K
  training without the user.
* **Toy-graph era (§1–§63).**
  - Mechanisms: pointer retrieval, a schedule-bound recurrence, count-keyed stops (Props 9–10).
  - The label-free ladder (Pilots K–M).
  - The checkpointed latent reasoner (§33–§51).
  - Full text is in the archive.
* **Directions memo** (2026-10-04, `docs/THEORY_DIRECTIONS_2026-10-04.md`): candidates for fast hand-offs, C4 without step
  labels, and latent chains without snaps. Top picks: a snap-strength dose-response, state-in-depth recurrence, filtered
  self-training by length.
* **Next (the user's plan):** §98 → the bench ladder (ProsQA-NA, a held-out task) → C4 by final-answer self-training
  (pending the user) → a general readout.
  - Queued: a routed snap for C5, with a non-confidence trigger (Literature 2af8ca18; §83 showed confidence fails out of
    range).

## Sections (one row per §; full text in the archive unless live)

| § | Question | Verdict / key number | Status |
|---|---|---|---|
| 0 | Why latent reasoners stop at the training depth | framing | theory |
| 1 | The latent state as a pointer moving along the chain | mechanism account | theory |
| 2 | Mechanism 1: the pointer is retrieved on a step-bound schedule | mechanism account | theory |
| 3 | A schedule-bound recurrence | theory sketch | theory |
| 4 | Mechanism 2: attenuation | measured in §14 | theory |
| 5 | Architectures suggested by the mechanism | proposals (tested in Pilots A–J) | design |
| 6 | Pilot A: architecture and schedule variants | scored in §15 | closed |
| 7 | Early log | — | archive |
| 8 | Toy simulation of the loop-gain account (Th3) | exploratory | closed |
| 9 | Pilot A interim; Pilot B registered | scored in §19 | closed |
| 10 | When the circuit is chosen (developmental data) | exploratory | closed |
| 11 | Pilot C: slack dose-response | competence and extrapolation by K (see archive) | closed |
| 12 | Training-free repair at inference | exploratory | closed |
| 13 | Public GPT-2 latent reasoners on GSM8K | latents carry work in step-supervised checkpoints | closed |
| 14 | Mechanism 2 measured: transfer curves of the local map | exploratory | closed |
| 15 | Pilot A scored | see archive | closed |
| 16 | Pilot B interim: anytime supervision gives a homogeneous front | interim | closed |
| 17 | Pilot D: anytime vs control, 12 + 12 seeds | PD1 supported; PD2, PD3 not met; PD4 met | closed |
| 18 | Formal sketch of the slack account | theory | theory |
| 19 | Pilot B scored | see archive | closed |
| 20 | Pilot E: depth wall or schedule end? | PE1 (depth wall) supported; PE2 reversed | closed |
| 21 | Out-of-sample test of the stage-3 retrieval marker | PO2 not met (AUC 0.51); early P7 AUC 0.92 | closed |
| 22 | Second-generation theory: four conditions for extrapolation | theory | theory |
| 23 | Pilot F: amplitude jitter, complete fix, selection window | J 1/8 competent; JAT 6/8 (5 extrapolate) | closed |
| 24 | Pilot G: monotone latent set | set unused (no verdict on A6); PG4 met | closed |
| 25 | Prop. 9: slack creates the clock | theory + predictions | theory |
| 26 | Re-anchored episodes (CTM) for C2 | re-anchoring fixes 31/34 | closed |
| 27 | Pilot H: repairing failures by fine-tuning | critical period; PH1–PH4 not met | closed |
| 28 | Pilot I: a clock that never runs out | PI1 narrowly not met; PI3 not met | closed |
| 29 | Pilot J: chunk-reset latent attention | PJ2 supported, PJ3 met; PJ1 not met | closed |
| 30 | Zero-cost follow-ups: renormalisation, randk, branch selection | RN1, RN2 not met | closed |
| 31 | Pilot K: label-free tight budgets | PK2–PK4 met; PK1 unresolved | closed |
| 32 | Pilot L: distilling re-anchored inference into one loop | PL1–PL4 met (fixes 19/20) | closed |
| 33 | The checkpointed latent reasoner (CLR) and C7 tests | design | design |
| 34 | Are silent thoughts editable? (T-C7c) | edit IIA 0.36; round trip 0.947 (E1, E5 narrowly not met) | closed |
| 35 | Prop. 10: confounded stop keys | theory + test | theory |
| 36 | Counter clamp at inference | CC2 met; CC1, CC3 not met | closed |
| 37 | Do Pilot L's students extrapolate past their depth? | X1–X3 hold | closed |
| 38 | Pilot M: second rung of the label-free ladder | PM1, PM4 met; PM2, PM3 not met | closed |
| 39 | Pilot T12: step-adversarial thoughts | PT2 not met; PT4 fails | closed |
| 40 | Prop. 10 across architectures (PROTOCOL9) | X9a–c hold | closed |
| 41 | Why a second training depth gives length generalisation | partial | closed |
| 42 | Predictions for PROTOCOL8's revised primary | predictions | closed |
| 43 | Where the deep frontier is read in 3–4-layer models | L1 not met (0.50); L2 met | closed |
| 44 | Set-valued checkpoints, zero-shot | S1, S2 not met: sets need training | closed |
| 45 | Pilot CLR-1: training set-valued checkpoints | PC3, PC4 met; set used; edit IIA 0.84–1.00 | closed |
| 46 | Where a written anchor helps most (hop sweep) | H1, H2 hold (sharp edge) | closed |
| 47 | Architecture review: CLR after §§34–46 | review | design |
| 48 | Multi-start beams across segments | BM1, BM2 not met | closed |
| 49 | A learned write head and closure halt (C5 v1–v2) | WH1, WH2 met | closed |
| 50 | A halt learned from final-answer reward (C5 v3) | LH1 met (9/10); LH2 not met | closed |
| 51 | Single-target answers from the halting segment | TH1 0/10; TH2 1/10 | closed |
| 52 | Frontier: new directions F1–F6 | proposals; verdicts in §53–§63 | design |
| 53 | F2: bidirectional latent reasoning, zero-shot | BD1 met; BD2, BD3 not met | closed |
| 54 | F1: annotated-input reasoning | AG1′ failed in both arms | closed |
| 55 | Is the anytime fall-back reward-neutral? | KT1 partial; KT2 met | closed |
| 56 | Are aged latent states interchangeable? | see archive | closed |
| 57 | Origin of the ProsQA adjacency cue | proximity heuristic 0.91 on prosqa_test | closed |
| 58 | F6: age-transplant training | falsified; F6-3 met | closed |
| 59 | ProsQA-NA: the cue neutralised | dataset | closed |
| 60 | Boundary memory, Phase A | gate A not passed (0/4) | closed |
| 61 | F1-S: a one-hop rule from shallow final answers | S1–S4 not met (1/3 seeds) | closed |
| 62 | Does depth-structured label noise install a count? | NC-4 met; others not met | closed |
| 63 | F4: a behavioural round-trip translator | not met (the translator did not train) | closed |
| 64 | Permutation tracking in the latent loop | S3 written-state restart 0.995 at N 9–16; A5 0/10 pass the gate | closed |
| 65 | Restart-distillation on permutation tracking | D2, D4 met; D1, D3 not met | closed |
| 66 | GPT-2 written-state chunked loop (boxes) | WC 1.000 at N 17–32; latents add nothing (P5 not met) | closed |
| 67 | Six-code latent, GPT-2 cell, exact marginal | pilot stop; table certifies, GPT-2 cell fails; 67b 1/5 | closed |
| 68 | GSM8K-Aug: our TC, Coconut, written-state arms | paused (no GSM8K training without the user) | paused |
| 69 | RL-style acquisition from outcome-only reward | spectral WFA recovers groups once length-2 pairs are seen (69-2) | closed |
| 70 | Toy acquisition round (A / B) | neither passes; line closes | closed |
| 71 | S5 written-state chunks, DeltaProduct protocol | WC₀₁ certified 5/5; 1.000 to 2,048 | closed |
| 72 | Where the GPT-2 cell loses acquisition | uninformative (lr design error); §72b: anisotropy, standardised 5/5 | closed |
| 73 | A stable latent: standardised-readout cell | certified S3 10/10 (L1–L3 met) | closed |
| 74 | Does the stable latent need a pretrained LM? | P1–P3 not met | closed |
| 75 | Scaling the stable latent: S4, A5 | gate fails | closed |
| 76 | Free-table automaton at S4 / A5 / S5; amortisation | needs short labels; GPT-2 hosts tables, does not compose | closed |
| 78 | GSM8K length generalisation by step count | no arm generalises (long ≈ 0.02) | closed |
| 79 | A latent automaton with natural-language paraphrases | P1 met (4/5 on trained phrasings); P2 not met | closed |
| 80 | A latent chunk machine (a code across k-op chunks) | P0 fails (unseen words 0.03 / 0.12) | closed |
| 82 | Acquisition ladder: subgroup footholds | "Neither": ge3 fails, std certifies, every group | closed |
| 83 | Self-gated restarts by confidence | U1–U3 fail (out-of-range overconfidence) | closed |
| 84 | Stop decision out of range: oracle stop, decoy padding | gate fails; long problems wrong from step 1 (84-1) | closed |
| 85 | Short-label threshold; oracle-initialised table | p 0.1 certifies 5/5 (A5); oracle init ε ≤ 2 → optimisation, not information | closed |
| 86 | Latent restarts (a latent per op, codebook snap) | gate fails at 8k steps (budget; see §92) | closed |
| 87 | Into the basin: warm start, annealing | warm25 5/5; annealing does not escape | closed |
| 88 | Loop-aligned composition, per-iteration targets | unseen 3-op words 1.000 (3/3); 2-op 0.061 | closed |
| 89 | Fade to latent (the user's idea) | F1 not met at 1× (0.615: "all" 1.000 ×5, "where" at chance); fixed at 3× (§94) | closed |
| 90 | Word diversity and loop-aligned composition | V1 met; V2 not met (a lottery) | closed |
| 91 | Making loop-aligned composition reliable | R met: LONG, WD-SP, CURR 5/5 | closed |
| 92 | §86 at 3× budget | met: S5 exact to 2,048 (5/5), k = 1 certified, removal ≈ 0, 0.29–0.45× | closed |
| 93 | Sparse chunk-end labels for latent restarts | not met (fix 0/3); dense 3/3 | closed |
| 94 | "Where" readout: budget vs inverse target | W1, W2 met: budget alone, 1.000 ×5 | closed |
| 95 | Foothold dose-response | no dose sufficient (0/9) | closed |
| 96 | Amortised self-labels from a final-answer teacher | P1, P2 met: 4/5 certified at p 0.05; labels 1.000; pipeline C4 (qualified) | closed |
| 97 | General latent thinker, step 1 | G4 met 3/3; C6 met 5/5; but mean ablation leaves HYB-4 s5 at 1.000 (fork bypass; C1 qualified) | closed (C1 → §104/§105) |
| 98 | Step 2a: multiplication vs ICoT-SI, Coconut, text CoT | not met at either size (GEN-M 0.163 / 0.005; TC 1.0; ICoT-SI 0.998 / 0.996); the k = 4 hand-off wall | closed |
| 99 | Speeding up the latent hand-off (k-curriculum, standardised slots, text-first, combo) | no arm accelerates (all censored; best partials TF-SLOT 0.37, STD 0.42) | closed |
| 100 | Shorter silent stretches: multiplication at k = 2 (GEN2, MASK2) | 4×4: M1, SOTA (1.000 vs 0.998), C3 (0.89×) met, C1 not; 5×5: not met (GEN2 0.09) | closed |
| 101 | Snap-strength dose-response (β_min 1 / 0.1 / 0.01 / 0) on S5 | registered | **open** |
| 102 | State in depth: gated slot-to-slot residual path (4×4, in §99's harness) | does not accelerate (all censored; test 0.01–0.17) | closed |
| 103 | Filtered self-training by length (C4 inside the LM, S5) | SELF not met: 0 / 0.9615 / 0.9585 at 512 from final answers only; pseudo-label precision ~0.98 caps it | closed |
| 104 | Serial causal use of slots: donor patching, blocking (§92, §97; eval only) | SERIAL not met: §92 2/5; §97 0/3 with donor latents ignored (state recomputed from visible ops) | closed |
| 105 | C1 for the headline: S5 GEN-4 with a masked fork + donor protocol | G4-M + MASK-DEC met; HANDOFF 0/3 (slots 2–3 latent-carried in 2/3; slot 1 recomputed from the visible start) | closed |
| 106 | C2 on a real task: bounded streaming multiplication (latent vs the written format) | written format + cut extrapolates 1.000 to b = 32 (WC-S-k2 3/3); masked latent arm fails even at k = 2 (106-4) | closed |
| 107 | Multiplication's blocker: carrier width (m latents) vs compute (pause), carriage forced | registered | **open** |
| 108 | Forced latent carriage on S5 (Markov segments, sink fix, TF bootstrap; donor-scored) | registered | **open** |
| 110 | Self-training v2: independent seeds, label-free gate, masked fork (C4 + C1) | design error (110-1): the restart gate needs non-identity restarts; s1–3 for the record | closing |
| 110b | Self-training v2 corrected: identity-start shorts plus random textual starts for length ≥ 3 | stopped after a clear null: loss at chance from random starts (110b-2) | closed |
| 111 | Reliability of the Markov hand-off: 2× budget vs DAgger-style own-state training; best-of-n | registered | **open** |
| 112 | Short-coverage dose-response for self-training (f = 0.5 / 0.25 / 0.1; never-labelled k = 1 accuracy) | registered | **open** |
| 113 | Auxiliary reader inside self-training (conditional on §111-READ) | registered (conditional) | **open** |

## Live sections

## 97. A general latent thinker, step 1: §92's latent restarts with the hand-built state removed (text step targets, re-embedded text checkpoints) (pre-registered 2026-10-03 12:01 EDT, after Literature's check 05bff3e3 and before any implementation or run; the user's decision via Audit, ~12:00)

**Why.** §92/92-1 meet C1 (use) + C2 + C3 + C6 on S5, but with task-specific structure:
* a 5 × 5 arrangement head for the per-op targets;
* a structured codebook for the restarts.

The user's plan ("generalized latent thinker") asks for one recipe with no hand-built state. §97 is step 1:
* the per-op target becomes the **text of the CoT step**;
* the restart becomes the **decoded text, re-embedded** by GPT-2.

**Main reading either way.** §97 and §92 differ only in these two interfaces. At matched budget, if §97 fails where §92
passed, the hand-built structure was doing the work.
* **What carries over from §89:** the step-decode side loss on the CoT text, i.e. fade_std_dec's load-bearing part.
* **What does not:** the α schedule. With one latent per op there is no per-token input to fade from, so α = 1 throughout,
  as in §92. A fade arm is a possible later variant.
* **Prior** (05bff3e3, `docs/LIT_S97_TEXT_SNAP_2026-10-03.md`). New by combination.
  - SELR 2608.13570: the same model decodes its own latents into steps; the closest to the fork decode.
  - SIM-CoT, PLaT 2601.21358 and LatentGuard decode latents into text but never feed the text back.
  - Thinking States 2602.08332 re-embeds generated text thoughts as a state, but its thoughts are all text and it keeps
    the KV cache.
  - 2510.14095 re-embeds discrete symbols every iteration, trained from scratch.
  - Delethink, InftyThink, PENCIL, Markovian Transformers and §71 carry text with a history cut, but have no latent
    slots.

**Model** (GPT-2 small, pretrained base, fine-tuned; otherwise exactly §92's recipe: 24,000 steps, bs 64, AdamW lr 1e-4,
wd 0.01, 100-step warmup + cosine, chunks cut at random positions from the S5=128 training sequences).
* **Chunk:**
  `[E(start text)] [op_1 tokens " perm p1 .. p5"] [slot_1] [op_2 tokens] [slot_2] ... [op_k tokens] [slot_k]`
  - The start text is a state written in TC's format (" > A B C D E": 6 tokens), embedded with GPT-2's own wte.
  - In training it is the true state before the chunk (identity at position 0), teacher-forced as in §92.
  - Slot j's input is the final hidden state at op j's last token (Coconut-style, as §92).
* **Per-op target (the only loss).** At every slot j, the KV cache is forked and the step text of the state after op j
  (" > X X X X X") is teacher-forced after the slot. CE is taken with GPT-2's **tied LM head over the full vocabulary**.
  - The fork is a side branch only. The chain continues from slot j without the step tokens.
  - There is no new head, and no task-specific vocabulary.
* **Snap (test) every k ops.** Greedy-decode the step text at slot k: 6 tokens, the length of TC's step for this task
  (the general form decodes to the next step delimiter " >"). Re-embed exactly those tokens with wte as the next chunk's
  start, with the history cut.
  - No validity check: a malformed checkpoint is carried as is and counts as wrong.
* **Per-position prediction** (for cumulative accuracy): the greedy-decoded step text at every slot, exact token match
  to the true state's text.

**Arms (seeds 1–3; seeds 4–5 are added if the k = 4 rule passes, for C6).**
* **GEN-4:** k = 4.
* **GEN-1:** k = 1.
* The comparison is §92 / 92-1 at the same budget: k = 4 seeds 1–5; k = 1 seeds 1–3.

**Measures** (§92's, generalised).
* Cumulative accuracy at 128 / 256 / 512 on §71's test set (2,000 × 512), and the 2,048 extension (first 500).
* **k = 1 certificate.** All 14,400 (state, op) inputs, with the start being the true state's text: the decoded step
  text equals the true next state's text. If it holds, the k = 1 machine is correct at every length.
* Per-slot decode accuracy. At k = 4, slots 1–3 are the silent latents' readability in text (C7 for latents).
* **C1 removal** at k = 4 on 10k fresh chunks: slot inputs 1..k−1 zeroed, and separately mean-replaced. Accuracy after
  the first ablated slot. This is also C7's input-only control.
  - **It is also the fork-bypass control** (Literature (b)). The fork sees the start checkpoint and ops 1..j, so it could
    re-derive the state without the latents. Decode accuracy at slots 2..k with the latents ablated but the ops
    visible measures that. High accuracy would mean bypass, and G4's removal bar excludes it.
* **Checkpoint validity rate:** the share of snapped texts that are a valid state (Literature (d)). Descriptive.
  Constrained decoding is not used, since it would reintroduce task structure.
* **wte drift** (Literature (c)): cosine between the initial and final wte rows of " >" and the five letters, which the
  tied head and the snaps share. No detaching in the main arms.
* **C7 edit at checkpoints** (descriptive): replace a decoded checkpoint with another valid state's text, then continue.
  The share of chunks whose next checkpoint equals the true composition from the edited state is reported as IIA.
* **C3:** FLOPs per 512-problem, counting every token processed including the snap decode and the start re-embedding.
  Reported against §71's retrained WC₀₄ (wc0k4r) and §92.

**Rules (fixed now; §92's).**
* **G4.** GEN-4 cumulative ≥ 0.99 at 512 in ≥ 2/3 seeds, with zero-removal ≤ 0.10 in those seeds.
* **G1.** The GEN-1 certificate holds in ≥ 2/3 seeds.
* **C3** (as §86's R3): median FLOPs ≤ 0.6× wc0k4r's.
* **C6:** if G4 passes, ≥ 4/5 seeds (1–5) at ≥ 0.99.

**Readings.**
* **G4 and G1.** §92's result does not depend on hand-built state: a generic text target and text checkpoint suffice.
  - Tags: C1 (use) + C2 + C7 (readable, editable text checkpoints; text-readable latents if per-slot decode ≈ 1), plus
    C3 if the FLOPs rule holds, and C6 at 4/5.
  - **Not C4** (per-step text targets).
  - This licenses step 2 (the same recipe on the bench ladder).
* **G1 only.** A per-op text restart works, but the latent hand-off over k − 1 silent steps needs the structured state.
* **Scope of G1 (Literature (a), fixed now).** At k = 1 a text snap follows every op, so no latent carries across ops.
  GEN-1 is essentially the written-state loop with one latent slot per op. **Only GEN-4 tests latent carriage**, and C1
  (use) is claimed only from GEN-4.
* **G4 fails, §92 passes.** The structure was doing the work.
  - **Attribution arm (pre-registered now, conditional; Literature (e)): HYB-4.** The text target (fork decode) with
    §92's structured codebook snap. The snap index comes from the decoded text's letters, and the true start code is
    used in training, as §92. k = 4, seeds 1–3, under G4's rule. It runs only if G4 fails.
  - HYB-4 passes: the text checkpoint is the problem. HYB-4 fails: the text target is.
  - No ladder step until this is read.

**Placement.** Vast 3090s, about 2–3 h per run (§92's chain plus the forks). 6 runs (plus HYB-4's 3 if G4 fails; seeds 4–5 if it passes): one proof host
(GEN-4 s1) first, then ≤ 3 hosts. About $4 for the 6. Projection to Audit. Implementation: the experiments session, from `boxes/latent86.py`, with
a smoke test of the fork loss, greedy snap and certificate before the panel.

**Note 97-1 (2026-10-03 12:01 EDT, before any implementation or run; Audit's review, on Literature 05bff3e3).**
* **Primary rule: G4 only.** GEN-1 (k = 1) is **descriptive**: its certificate and accuracies are reported, but G1 is not
  a verdict. A per-op text snap is essentially §71's written-state loop.
* **HYB-4 runs unconditionally:** text target (fork decode, tied LM head) + §92's structured codebook snap, k = 4, seeds
  1–3, under G4's rule.
* **Attribution (fixed now):**

  | §92 | HYB-4 | GEN-4 | Reading |
  |---|---|---|---|
  | pass | pass | pass | No hand-built state is needed. |
  | pass | pass | fail | The text checkpoint (snap) is the problem. |
  | pass | fail | fail | The text target (head) is the problem; the snap is untested. |
  | pass | fail | pass | Reported as found, with no reading fixed now. |
* **Already in §97:**
  - the decoder-bypass control: decode accuracy with slot inputs zero- and mean-ablated, ops visible;
  - checkpoint validity.
* **Cost.** 9 runs (GEN-4, GEN-1, HYB-4 × seeds 1–3), plus seeds 4–5 of any arm that passes G4. About $6. Projection to
  Audit before launch.

**§97 result (2026-10-03 17:35 EDT; runs by the experiments session on Vast; scored by Theory alone with
`theory/score_s97.py`, committed before any run (90034c85; key alignment before the panel);
`theory/results/score_s97.json`). G4 is met; the attribution reads "no hand-built state is needed".**

| arm (S5, 24k steps, seeds 1–3) | cumulative at 512 | 2,048 | zero-removal after slot 1 | mean-removal after slot 1 | FLOPs vs WC₀₄ (as metered / restricted vocab) | G4 |
|---|---|---|---|---|---|---|
| **GEN-4** (text target + text checkpoint) | **0.998 / 1.000 / 1.000** | **1.000 ×3** | ≤ 0.007 | 0.23 / 0.28 / 0.29 | **0.56× / 0.61×** | **3/3** |
| HYB-4 (text target + §92's codebook) | 1.000 ×3 | 1.000 ×3 | ≤ 0.003 | 0.15 / 0.50 / 0.31 | 0.49× / 0.53× | 3/3 |
| §92 (5 × 5 head + codebook) | 1.000 / 1.000 / 0.9995 | 1.000 ×3 | ≤ 0.003 | ≤ 0.003 | 0.39× / 0.45× | 3/3 |
| GEN-1 (descriptive) | 1.000 ×3 | 1.000 ×3 | — | — | — | certified 3/3 (0 / 14,400) |

* **G4 met (3/3), and C3 met as registered** (0.56× ≤ 0.6, as metered with full vocabulary on both sides). Under the
  restricted-vocabulary convention (92-3) it is 0.61×, just over the bar. Both are reported.
* **Attribution (97-1): §92 pass, HYB-4 pass, GEN-4 pass, so no hand-built state is needed.** §92's latent restarts
  work with the per-op target written as the CoT step's text, decoded by GPT-2's own LM head, and the checkpoint carried
  as that text re-embedded.
* **C7 (descriptive).**
  - Every checkpoint is plain text. Validity is ≥ 0.99999 (one malformed snap in 128k, GEN-4 s1).
  - **Checkpoint edit IIA is 1.000** in all 9 runs: replace a checkpoint with another state's text, and the next
    checkpoint is the true composition from the edited state.
  - The silent slots 1–4 decode to the correct step text at 1.000.
  - wte drift is negligible (min cosine ≥ 0.995).
* **C1 (use), with a qualifier from the bypass control (Literature (b)).**
  - Zeroing the latent slot inputs drops accuracy to ≤ 0.007, and long rollouts fail without the latents (mean-ablated
    512 accuracy ≈ 0).
  - But **mean-ablating them leaves the in-context text decoder partly able to recompute the state from the visible
    ops**:
    - GEN-4 per slot under mean ablation: slot 1 at 0.71–0.95; slots 2–4 at 0.21–0.30 (chance 0.008);
    - HYB-4: slots 2–4 at 0.37–0.64;
    - §92's restricted head: ≤ 0.009 everywhere.
  - So the latents carry the state (intact 1.0 vs ablated ≤ 0.30 from slot 2 on). The six-token text decode adds an
    in-pass recomputation route that §92's single-position head lacked.
  - C1 is claimed by the registered zero-removal rule, **qualified: partial recomputation under mean ablation**.
* **GEN-1 (descriptive, 97-1).** A per-op text snap is certified at every length (0 errors over 14,400). This is
  essentially the written-state loop.
* **Standing claim (§97).** One recipe with no task-specific state:
  - a text step target through the LM head, plus re-embedded text checkpoints with a history cut every 4 ops, plus one
    latent per op in between;
  - carries S5 exactly to 2,048 from training at 128 (3/3), at 0.56–0.61× the written state's compute.
  - **C1 (use, qualified) + C2 + C3 (as metered) + C7 for checkpoints** (readable, edit IIA 1.0); **not C4**.
  - C6 needs seeds 4–5 (97-1): next. This licenses step 2 of the user's plan (the bench ladder).

**Note 97-2 (2026-10-03 17:36 EDT; Audit's reproduction of §97).**
* **Reproduced:** GEN-4 0.998 / 1.0 / 1.0 at 512, 2,048 at 1.0 ×3; GEN-1 certified 3/3; edit IIA 1.0 in every run.
  Compute: GEN-4 0.556× wc0k4r as metered (≈ 0.608× restricted vocab); HYB-4 0.489×.
* **GEN-1's comparator is wc0k1** (k = 1 written state), not wc0k4r: 1.81e12 vs 2.42e12 FLOPs per 512-problem, so
  **0.75× as metered and 0.72× with both sides' vocabulary-logit terms removed** (2.37e11 each: 512 × 6 rows). Against
  wc0k4r, GEN-1 would read 1.10×, which is the wrong comparison.

**Note 97-3 (2026-10-04 01:31 EDT; C6 seeds 4–5 scored with `theory/score_s97.py`; Audit raw-read).**
* **C6 met (5/5, per 97-1).**
  - GEN-4 at 512: 0.998 / 1.0 / 1.0 / 1.0 / 1.0; at 2,048: 1.0 ×4, plus 0.998.
  - HYB-4: 1.0 at 512 and 2,048 in all 5.
  - Zero-removal is ≤ 0.008 in every seed. Checkpoint edit IIA is 1.0 and validity 1.0.
* **Headline under the user's C1 priority: mean ablation shows a fork bypass.** Mean-removal accuracy after the first
  ablated slot:

  | arm | s1 | s2 | s3 | s4 | s5 |
  |---|---|---|---|---|---|
  | GEN-4 | 0.23 | 0.28 | 0.29 | 0.26 | 0.33 |
  | HYB-4 | 0.15 | 0.50 | 0.31 | 0.57 | **1.000** |

  - In HYB-4 s5, replacing the slot inputs with their mean leaves **every state correct**. The text fork recomputes the
    whole chunk from the visible ops without the latents.
  - Zero ablation still drops it to 0.002, but zero is off-distribution.
* **Consequence.** For the text-fork recipe, C1 (use) rests only on the zero-ablation rule. Under in-distribution
  (mean) ablation the latents are not necessary in some seeds.
  - §97's C1 stays **qualified**, and this is now quantified per seed.
  - The decisive C1 test is a masked fork on S5 (§105, next) and §104's donor patching.

## 98. General latent thinker, step 2a: §97's recipe on multi-digit multiplication (bench mult4x4 / mult5x5), against ICoT-SI and Coconut (pre-registered 2026-10-03 17:39 EDT, after Literature's check e206d92d and before any implementation or run; the user's plan, step 2)

**Why.** §97 showed that one recipe with no hand-built state carries S5 exactly to 2,048: a text step target through
the LM head, re-embedded text checkpoints, and one latent per op. Step 2 of the user's plan puts the same recipe on the
bench ladder. This first rung is multi-digit multiplication: the published implicit-reasoning benchmark (Deng
2311.01460, 2405.14838), scored on the same 1,000 test items per task as the published numbers.
* **Prior** (e206d92d, `docs/LIT_S98_MULT_2026-10-03.md`).
  - Only ICoT-KD and ICoT-SI report on these items. No Coconut, CODI, SIM-CoT, PCCoT or CoLaR numbers exist, so COCO-M
    is a first measurement.
  - **2510.00184** ("Why Can't Transformers Learn Multiplication?"): an auxiliary running-sum prediction loss unlocks
    learning. It is the closest prior to the per-op S_j decode, and is cited.
  - No length-generalisation result exists for any implicit or latent method. ICoT-SI is per-size (11×11 at 0.74).
    Explicit baselines: Cho 2410.15787, Lee 2502.01612.

**Task and data.** The implicit-CoT repo's `{4,5}_by_{4,5}_mult` files: digits least-significant first,
`a * b || <steps> #### <product>`.
* Train on its train split; test on `bench/data/mult_{4x4,5x5}_test.txt` (the published test items).
* One task per run: 4×4 and 5×5 separately, as published.

**Recipe mapping** (the same code path as §97's `gen` arm; nothing task-specific beyond reading the data's own format).
* **Op j** (j = 1..n) = the multiplier digit b_j. Operand a is the problem's constant prefix and is visible in every
  chunk.
* **Step text j** = the dataset's own CoT segment j (segments split on " + "):
  - step 1 = "P_1";
  - step j > 1 = "+ P_j ( S_j )";
  - the last segment as the data has it, then the answer.
* **Chunk:** `[E(a and " *")] [E(start text)] [op_1] [slot_1] … [op_k] [slot_k]`.
  - The start text is the previous step's text: the true text in training; the greedy-decoded, re-embedded text at test,
    with the history cut. The first chunk has an empty start.
  - Slot inputs are as §92/§97.
* **Loss** (as §97). Fork at every slot and teacher-force the step text (CE through the tied LM head). After the last
  slot, fork and teacher-force " #### <product>". Nothing is fed back.
* k = 4 (§97). 4×4 is one chunk; 5×5 has one snap.
* **Answer:** greedy decode after the last slot. Exact match on digits (the bench scorer).
* **Budget:** §97's: 24k steps, bs 64, AdamW lr 1e-4. That is about 1.9 epochs of the 808k-problem train split, far
  below ICoT-SI's 200.
  - An M1 failure at this budget is read as **"not acquired within budget"** (as §86), not as a recipe failure.

**Arms (each task, seeds 1–3).**
* **GEN-M:** the recipe as above.
* **GEN-M-MASK (Audit (a)):** as GEN-M, but every side-branch fork (step text and answer) attends only to the slot's
  own position and its own decode tokens. The op tokens, the operand prefix, the start text and the earlier slots are
  masked out of the fork. The chain itself is unchanged. The decoder can then read the state only through the latent.
* **COCO-M (Audit (b)):** Coconut's stage curriculum on the same data and update budget, with n continuous thoughts.
  Removal is applied to its thoughts.
* **ICoT-SI (the SOTA bar):** the **released checkpoints**, evaluated by our harness on the same items.
  - Literature found the repo's 5×5 log at 0.996 (the README says 1.00); the bench's paper value is 0.95.
  - The bar is our evaluation of the released checkpoint. If it cannot be evaluated, the repo log's value is used
    (1.000 at 4×4; 0.996 at 5×5).
  - No retraining: its recipe is 200 epochs over 808k problems.
  - Its inference FLOPs and sequential passes are metered on the same items. No removal applies, since it emits no
    thoughts.
* **TC-M (text CoT at matched budget):** GPT-2 small fine-tuned on the full CoT text, with the same data, budget and
  seeds. It is **the C3 bar**, with FLOPs metered on the same items. The published value (100 / 100) is reported beside
  it.
* **No CoT:** published 29 / 1, the C1 floor.

**Measures.**
* Accuracy on the 1,000 test items.
* Per-slot step-text decode accuracy (exact).
* Zero and mean removal of slot inputs (answer accuracy), also on COCO-M.
* Checkpoint validity and edit IIA at the 5×5 snap.
* Inference FLOPs per problem (Meter; full vocabulary and restricted conventions, 92-3) for every arm, including text
  CoT and ICoT-SI on the same items.
* **Descriptive C2.** 4×4-trained GEN-M on 4×8 (an 8-digit multiplier, generated by the same rule), with snaps every 4
  ops.
  - The dataset's own checkpoint format is out of range here (Literature (a)): P_j carries j written leading zeros (a
    counter), and S_j grows to about n + j digits.
  - A bounded checkpoint (finalised low digits emitted, a positional shift) would be a hand-built format, so it is not
    used. The 4×8 result is descriptive and flagged as range-limited.
* Sequential passes are reported beside FLOPs (Literature (d)).

**Rules (fixed now; per task, seeds 1–3).**
* **M1 (the recipe solves the task):** GEN-M median accuracy ≥ 0.95.
* **M-SOTA:** GEN-M median accuracy ≥ ICoT-SI's published value on the same items (1.000 at 4×4; 0.950 at 5×5).
* **M-C1 (unqualified use):** GEN-M-MASK median accuracy ≥ 0.95, and in those seeds zero- and mean-removal answer
  accuracy ≤ No-CoT's published value + 0.05.
  - The slot-ablated S_j decode in the masked fork is reported per slot (Literature (c)). Operand a is always visible,
    so P_j is recomputable; **S_j is the only carried state.**
* **M-C3:** GEN-M median accuracy ≥ TC-M's − 0.01, at ≤ TC-M's median FLOPs on the same items (as metered; the
  restricted convention is also reported).
* COCO-M is reported under M1's bar, with removal. GEN-M − COCO-M is paired by seed (descriptive).

**Readings.**
* **M1 + M-SOTA at both sizes.** The general recipe matches or beats the best published implicit method on the standard
  multiplication benchmark: C3 if M-C3 holds, and C1 unqualified if M-C1 holds. Next rung: ProsQA-NA.
* **M1 only.** It solves the task but below ICoT-SI; report the gap.
* **M1 fails.** The recipe does not transfer from S5 to arithmetic as is. Report per-slot decode to locate where.
* **Tags:** C1 (use; unqualified only via M-C1) + C3 (vs text CoT) + C7 for checkpoints; C2 only descriptively here;
  not C4.

**Placement.** Vast 3090s: 24 runs (GEN-M, GEN-M-MASK, COCO-M, TC-M × 2 tasks × 3 seeds) plus the ICoT-SI checkpoint
evals. One proof host (GEN-M 4×4 s1) first, then ≤ 5. About $12–15. Projection to Audit. Implementation: the experiments session (a task adapter for gen97.py, the
fork mask, Coconut and TC from the GSM8K pipelines, and an ICoT-SI checkpoint loader).

**Note 98-1 (2026-10-03 17:41 EDT, before any §98 implementation or run; Audit's review of 3f6d85d3).**
* **M-C1's floor is pinned.** It is our own **NC-M**: GPT-2 small fine-tuned on question → answer, with no CoT. Same
  data, budget and seeds (1–3), per task, on the same 1,000 items.
  - The floor is **NC-M's median `acc_test` + 0.05**.
  - The published No-CoT row (29 / 1) is reported beside it but is not the floor. NC-M adds 6 cheap runs.
* **Training compute** is reported next to accuracy for every arm: epochs over the train split and training FLOPs
  (Meter).
  - ICoT-SI's released model: about 200 epochs (its recipe, Literature e206d92d). Ours: about 1.9.
* **Reading if M1 holds but M-SOTA misses:** "**below SOTA at about 1/100 of ICoT-SI's training**", never plain "below
  SOTA".
* **Automatic follow-up (part of §98; no new registration).** For each task where M1 holds and M-SOTA misses:
  - GEN-M runs at **4× budget** (96k steps, seeds 1–3) under the same rules;
  - so does GEN-M-MASK, if its median accuracy is < 0.95 at 24k.
  - Its reading, if it then meets the bar: "matches SOTA at about 1/25 of ICoT-SI's training".
  - Nothing else follows automatically.

**Note 98-2 (2026-10-03 17:45 EDT, before any §98 run; the experiments session's implementation check).**
* **The descriptive C2 measure (4×8) is dropped.** In the dataset's own CoT format the last step is "+ P_n" with **no
  running sum**. So a 4×4-trained model never writes S_4, and a restart after op 4 at 4×8 has no carried state: it fails
  by construction.
  - A 5×5 → 5×8 version fails the same way: no slot past op 5 is ever trained, the last-step format changes, and P_j's
    leading-zero counter grows (Literature (a)).
  - **Multiplication's CoT, in its published format, is not restartable past its trained length.** A C2 test here needs
    a restartable step format, which would be hand-built, so it is deferred.
  - §98 is now an in-distribution rung (M1, M-SOTA, M-C1, M-C3); its tags drop C2.
* **Implementation defaults recorded** (the experiments session; consistent with §97):
  - training chunks are aligned to the test-time chunks (ops 1–4, plus op 5 alone at 5×5 from the true step-4 text),
    with one chunk index drawn per batch;
  - the answer fork is on the final chunk only;
  - the loss is the mean over forks of per-fork token-summed CE;
  - step and answer decodes use the format's fixed token counts (as §97's 6);
  - removal ablates slot inputs 1..k−1 within each chunk (4×4: slots 1–3), and all thought inputs for COCO-M;
  - MASK gives each fork only its slot's cached K/V (all layers) plus its own tokens, with true positions;
  - NC-M: "a * b||" → " #### product". TC-M: "a * b||" → the full CoT + " #### product".

**Note 98-3 (2026-10-04 23:58 EDT, before any §98 scoring; Theory's diagnostic read, disclosed below).**
* **An implementation conflict voids the GEN-M and GEN-M-MASK runs for M1 and M-C1.**
  - In `boxes/mult98.py` `chunk()`, at the final chunk's last slot, the step-n fork (target "+ P_n", first token " +") and
    the answer fork (target " #### <product>", first token " ####") both start from the **identical** state. Both
    first-token predictions come from `logits(hslot)`. One hidden state is trained toward two different first tokens,
    and greedy decoding can produce at most one, so the last slot and the answer fail by construction.
  - The registration's "after the last slot, fork and teacher-force ' #### <product>'" was ambiguous; this is where it
    went wrong.
  - **Evidence:** GEN-M 4×4 s1–3 last-slot decode is 0.007 / 0.0 / 0.007 and answer accuracy 0.006, while slot 3 is
    0.92 / 0.95 / 0.37. The last slot's target (a × b_4, shifted) is as recomputable as slot 1's P_1, which decodes at
    1.0.
  - TC-M, NC-M, COCO-M and ICoT-SI do not use this path and are unaffected.
* **Fix (the dataset's own format):** at the final slot, **one fork** whose target is the last step text followed by
  " #### <product>" ("+ P_n #### product"), with CE over the concatenation and no separate answer fork.
  - At eval, greedy-decode len(step_n) + len(answer) tokens. The last slot's decode is the first len(step_n) tokens; the
    answer is the digits after " ####".
  - The same applies to MASK and to 5×5's final single-op chunk.
* **Reruns (part of §98, no new registration):** GEN-M and GEN-M-MASK, both tasks, seeds 1–3, §98's budget, on the fixed
  code. Rules, comparators and the 98-1 follow-up are unchanged.
  - The voided runs' slot curves and removal values are kept as descriptive.
  - They are renamed `*.void_forkconflict`, not deleted.
* **Disclosure.** Theory read the voided GEN-M 4×4 eval.json (per-slot decode and removal) as a descriptive diagnostic of
  the slot-3 wall, after §98's rules were fixed. That read surfaced the conflict.
  - Also descriptive: under mean ablation, slot 2 still decodes at 0.79–0.99 (slot 1 at 0.17–0.65). S_2 is largely
    recomputed from the visible operands, so on multiplication **the first true latent hand-off is at slot 3**.

**98-3 addendum (2026-10-04 23:59 EDT; Audit's review).**
* **Unaffected:** TC-M, NC-M, COCO-M and the ICoT-SI evals (no fork), and **§92/§97 on S5**. There the per-slot decode is
  itself the answer and there is no separate answer fork.
* **Guard test, required before any rerun:** a 50-step overfit smoke on a small fixed batch. It must reach exact decodes of
  **both** the last slot's step text and the answer (so a shared-first-token conflict fails it).
* **Rerun priority:** §98 → §100 → §99, with §99's relaunch on the local cards where it fits (4×4).

**Note 98-4 (2026-10-04 01:38 EDT; 98-3's guard test, run by the experiments session on a Vast 3090;
`theory/results/s98_smoke/guard98_3_output.txt`). The guard passes.**
* **Setup:** 8 fixed problems, lr 3e-4, fp32.

  | config | FIXED (final slot / answer) | OLD (final slot / answer) |
  |---|---|---|
  | GEN-M 4×4 | 1.000 / 1.000 | 1.000 / **0.000** |
  | GEN-M 5×5 | 1.000 / 1.000 | 0.125 / 0.875 |
  | MASK 4×4 | 1.000 / 1.000 | 0.5 / 0.5 |
  | MASK 5×5 | 1.000 / 1.000 at 400 steps | 0 / 0 |

* The OLD code shows the conflict exactly: the slot is learned and the answer never is.
* **Deviation from the addendum's "50 steps", disclosed:**
  - 50 steps at lr 1e-3 did not overfit even the fixed code (the test's optimisation, not the fix);
  - the guard used 150 steps at lr 3e-4;
  - MASK 5×5's slot-only fork emits 23 tokens and needed 400 steps (1.000 / 1.000; also at 800).
* The reruns proceed C1-first: MASK 4×4 / 5×5, MASK2, MARKOV with its comparators, §105; then §101; then the C3 parts.

**§98 result (2026-10-04 09:57 EDT; GEN-M / MASK reruns on fixed code (98-3), comparators, and ICoT-SI's released
checkpoints, all run by the experiments session; scored by Theory with `theory/score_s98.py`, committed before any run;
`theory/results/score_s98.json`; COCO-M's s3 runs are still local, descriptive). M1, M-SOTA, M-C1 and M-C3 are not met at
either size.**

| median accuracy | 4×4 | 5×5 |
|---|---|---|
| GEN-M (k = 4) | **0.163** | 0.005 |
| GEN-M-MASK | 0.034 | 0.0 |
| COCO-M (first Coconut measurement on these items) | ≈ 0.001 | 0.0 |
| TC-M (text CoT, matched budget) | 1.000 | 1.000 |
| ICoT-SI (released, our eval) | 0.998 | 0.996 |

* **Where it fails.** GEN-M 4×4's per-slot decode is [1, 1, 0.80–0.92, 1]; MASK 4×4's is [1, 0.90–1.0, 0.22–0.25, 1].
  - The loss sits at slot 3, the first true latent hand-off (98-3's diagnostic), and at the answer that depends on it.
  - Removal: zero 0, mean ≤ 0.015, so there is no bypass but also no carriage.
* **Reading (98-1).** **"Not acquired within budget"** (about 1.9 epochs against ICoT-SI's ~200), at k = 4. The 98-1
  4× follow-up is not triggered, since M1 is not held.
* §100 shows the same recipe at **k = 2** reaching 1.000 at 4×4 (SOTA parity, 0.89× text CoT FLOPs). The wall is
  several silent hand-offs of a growing sum per chunk (§107 / §106-k2 / notes 100-2, 100-3).

## 99. Speeding up the latent hand-off: k-curriculum, standardised slot inputs, a text-first stage, and their combination (pre-registered 2026-10-03 21:27 EDT, after Literature's check cfeea0b8 and before any implementation or run; the user's directive via Audit, ~21:20, to run in parallel with §98)

**Why.** §98's GEN-M training logs (training-time only, unscored) show the §86 staircase on multiplication. The first slots
are learned fast; the latent hand-off to later slots is slow. genm_5x5_s1 reads per-slot 1.0 / 1.0 / 0.49 / 0.06 / 0.0
at about 20k of 24k steps, with loss 1.6–2.8 against text CoT's ~1e-7.
* §99 asks which cheap change makes the hand-off converge faster.
* It is a separate comparison and leaves §98 untouched.
* **Prior** (cfeea0b8, `docs/LIT_S99_ACCEL_2026-10-03.md`). No published speed-up figures exist for latent hand-offs,
  so every arm is a new measurement.
  - Literature's ranking: a warm start (CoT init, keeping the side decoder) > a k-curriculum 1 → 4 > standardisation
    (frozen statistics, a σ floor).
  - Hidden-state distillation is not used: it risks a bypass (SCIT 2608.27265: CODI computes in the V-cache), geometric
    collapse (2606.20075), and a mismatch between a full-history teacher and our history cut.

**Task.** Bench mult 4×4 (§98's data and test items), §98's GEN-M recipe and code path (k = 4, text step target via a fork,
the answer fork after the last slot). Budget cap: §98's 24k steps, bs 64, lr 1e-4. Seeds 1–3 per arm.

**Arms.**
* **CTRL:** plain GEN-M, with §99's logging (a replication of §98's GEN-M 4×4 under early stopping).
* **K-CURR:** training chunk length 1 for steps 0–2k, 2 for 2k–6k, then 4 (test always k = 4).
  - Shorter chunks start from the true previous step text with the history cut, exactly as GEN-M's later chunks do.
* **STD:** each slot's input is g(h) = μ_E + σ_E · (h − μ_h) / σ_h (per coordinate). μ_h and σ_h are an EMA
  (momentum 0.99) over slot-input hidden states, frozen at the stopping step and used unchanged at test. μ_E and σ_E are
  GPT-2's token-embedding statistics (§89's fade_std g). σ_h is floored at 0.05 × its median over coordinates
  (Literature's σ floor).
* **TEXT-FIRST:** steps 0–3k train ordinary text CoT (TC-M's objective) on the same data; then GEN-M from those
  weights. The 3k text steps **count toward the budget and toward steps-to-threshold**.
* **COMBO:** K-CURR + STD + TEXT-FIRST (text CoT for 0–3k, then k = 1 for 3k–5k, k = 2 for 5k–9k, then k = 4, with STD
  throughout the latent phase).

**Logging and stopping.** Every 1k steps, on the repo's 1,000-item valid split (never the bench test):
* per-slot step-text decode accuracy;
* answer accuracy (k = 4 rollout).

A run **stops at the first log where every slot's decode is ≥ 0.95 and answer accuracy is ≥ 0.95**
(steps-to-threshold), or at 24k (censored).

At the stopping checkpoint, on the 1,000 bench test items:
* answer accuracy;
* zero- and mean-removal answer accuracy (slot inputs 1..k−1, as §98);
* per-slot decode;
* wall-clock and training FLOPs.

**Rule (fixed now).**
* An arm **accelerates** if its median steps-to-threshold is **≤ 0.5× CTRL's median**.
  - A censored run counts as 24k. If CTRL's median is censored, the bar is ≤ 12k.
* In addition, in the arm's threshold-reaching seeds, zero- and mean-removal answer accuracy must be **≤ the NC floor +
  0.05**, with the NC floor = §98's NC-M 4×4 median (note 98-1). This is the no-bypass condition.
* Reported for every arm: median steps-to-threshold, the per-seed curve, test accuracy at stop, removal, and FLOPs to
  threshold.

**Readings.**
* **An arm accelerates.** That ingredient speeds the latent hand-off without a bypass. It joins the recipe for the ladder's
  next rungs. COMBO vs its parts says whether the ingredients add.
* **None accelerates.** The hand-off's slowness is not fixed by these levers. Budget (§92) remains the known remedy.
* **An arm is fast but fails no-bypass.** The speed comes from recomputation, not latent carriage (§97's mean-ablation
  route). It is reported as such.

**Placement.** 15 runs (5 arms × 3 seeds), each capped at 24k steps, many stopping early. On local cards or Vast 3090s
(≤ 5 hosts, proof host first), about $4–8. Projection to Audit. Implementation: the experiments session, as flags on
`boxes/mult98.py` (`--kcurr`, `--std_slots`, `--text_first`, the valid-split early stop).

**Note 99-1 (2026-10-03 21:29 EDT, before any §99 implementation or run; Literature's addendum e19579b5 and Audit).**
* **New arm TF-SLOT** (teacher-forced slot inputs; scheduled sampling for slots; priors 1506.03099, PCCoT 2506.18582,
  C-MTP 2607.16972).
  - In training, at each slot position j < 4, a per-position coin with probability ρ(step) replaces slot j's latent with
    the **re-embedded gold step-j text**: the tokens, in context, with no history cut. Later slots therefore learn their
    transitions in parallel.
  - ρ anneals linearly from 1 at step 0 to **exactly 0 at step 12k (50 % of the cap)** and stays 0. That meets Audit's
    "0 before the last ≥ 25 %".
  - **ρ is logged at every 1k log** (Audit (c)).
* **New arm TEXTFIRST-MASK:** TEXT-FIRST with §98's masked fork (each fork attends only to its slot and its own tokens).
  It is TEXT-FIRST's bypass check, alongside slot removal (Literature's bypass table).
* **Threshold eligibility (fixed now; Audit (b)).** steps-to-threshold counts only logs taken in the arm's final
  configuration:
  - K-CURR from step 6k (k = 4);
  - COMBO from 9k;
  - TF-SLOT from 12k (ρ = 0);
  - TEXT-FIRST and TEXTFIRST-MASK from 3k (the latent phase);
  - CTRL and STD from step 0.
  - Earlier logs are recorded but cannot stop a run. **Every arm ends at k = 4** with no gold inputs, and is tested only
    at k = 4.
* **Not added:** up-weighting the later slots' loss, and a running-sum probe loss (2510.00184). Both are cheap but would
  dilute the panel; noted for later.
* **Panel and cost:** 7 arms × 3 seeds = 21 runs, about $6–10. The rule, readings and no-bypass condition are as §99. The
  scorer is updated in the same commit (arms `tfslot`, `textfirst_mask`).

**Note 99-2 (2026-10-04 23:58 EDT, before any §99 scoring).** §99's arms import mult98's GEN-M, so they carry 98-3's
final-fork conflict. Their stop criterion (answer ≥ 0.95) cannot be met. §99 runs on the fixed code, with the same
arms, seeds, rules and eligibility (99-1). Runs started on the old code are renamed `*.void_forkconflict`, kept, and not
scored.

**Note 99-3 (2026-10-04 00:01 EDT, before any fixed-code §99 run; Literature 0ca8a65a, `docs/LIT_S99_MARKOV_2026-10-03.md`).**
* **Why.** 98-3's diagnostic: on multiplication, slot 2's state is recomputed from the visible operands (mean-ablated
  latents still decode it at 0.79–0.99). So the first true latent hand-off is slot 3. Hypothesis: **visible-operand
  recomputation starves the latent of gradient to carry state.**
* **New arm MARKOV:** each slot step is a **fresh-cache segment**.
  - The segment for slot j is [operand prefix a, slot j−1's latent, op j]. For the chunk's first op it is [a, the chunk's
    start text, op 1].
  - Positions are re-indexed from 0 in every segment, identically in training and test (Literature (c); §86's
    segment-isolation construction).
  - S_j can reach slot j only through slot j−1's latent, so every slot from 2 on is a forced hand-off, and earlier ops
    are invisible.
  - The step and answer forks attend only to their own segment, so they cannot recompute from earlier ops either
    (Literature (a)). The final fork is 98-3's "+ P_n #### product".
  - **Bootstrap:** MARKOV uses K-CURR's chunk schedule (k = 1 for 0–2k, 2 for 2k–6k, then 4), and is eligible from 6k.
    Literature (b): our Pilot A Markovian step gave no gain and trained harder, and RMT needed a curriculum.
  - **MARKOV vs K-CURR differs only in the context restriction.** That isolates the starvation hypothesis.
* **Rule:** §99's (accelerates = median steps-to-threshold ≤ 0.5× CTRL's, with no-bypass removal), plus:
  - **descriptive H-starve:** MARKOV reaches threshold earlier than K-CURR (median steps), with per-slot curves;
  - if MARKOV is slower, or fails, the Pilot A prior holds here too.
  - **If MARKOV does not reach threshold by 24k, the reading is "not acquired within budget"** (pre-declared; Audit),
    not "the bottleneck cannot carry the state".
* **Fork scope, stated (Audit (1)).** MARKOV's forks see only their own segment [a, slot j−1, op j, slot j]. P_j is
  legitimately recomputable from a and b_j. S_j, the carried state, reaches the fork only through slot j−1, so there is
  no recompute route for the hand-off.
* MARKOV runs on 98-3's **fixed** code (one final fork), like all of §99 (99-2).
* **Panel:** 8 arms × 3 seeds = 24 runs. The scorer is updated (arm `markov`).

**§99 result (2026-10-04 13:35 EDT; all arms run on the fixed code (98-3) by the experiments session; scored by Theory alone
with `theory/score_s99.py`, committed before any run; `theory/results/score_s99.json`). No arm accelerates.**

| arm (4×4, seeds 1–3) | steps-to-threshold | test accuracy | removal (zero / mean) |
|---|---|---|---|
| CTRL | censored ×3 | 0.163 / 0.143 / 0.184 | 0 / ≤ 0.012 |
| K-CURR | censored ×3 | 0.048 / 0.118 / 0.074 | 0 / 0 |
| STD | censored ×3 | 0.031 / 0.029 / 0.421 | 0 / ≤ 0.007 |
| TEXT-FIRST | censored ×3 | 0.011 / 0.007 / 0.070 | 0 / ≤ 0.010 |
| TEXTFIRST-MASK | censored ×3 | 0.020 / 0.020 / 0.034 | 0 / ≤ 0.006 |
| TF-SLOT | censored ×3 | 0.049 / 0.247 / 0.369 | 0 / 0 |
| COMBO | censored ×3 | 0.011 / 0.014 / 0.004 | 0 / ≤ 0.001 |
| MARKOV (99-3) | censored ×3 | 0.047 / 0.041 / 0.020 | 0 / 0 |
| STATEPATH (§102) | censored ×3 | 0.170 / 0.032 / 0.013 | 0 / ≤ 0.007 |

* **Reading ("none accelerates"):** none of these levers makes multiplication's latent hand-off learnable within 24k at
  k = 4. Removal is at the floor everywhere, so there is no bypass, but also no carriage.
  - **H-starve (descriptive) is not supported:** MARKOV is not faster than K-CURR.
  - The best partial results are TF-SLOT (one seed 0.37) and STD s3 (0.42).
* **Implementation details recorded (the experiments session):**
  - TF-SLOT's coin is per position per batch, and replaced slots skip their fork CE;
  - the cosine LR runs over the full 24k cap;
  - STD's statistics come from the latent phase only;
  - removal acts on the fed input.
* **What followed:** §107 (width vs compute, running), §105 / §108 / §111 (forced carriage on S5), and Audit's
  no-read-basin diagnosis (111-1), which explains why chunk-start text lets the reader never learn.
* **Audit's reproduction and observation (faa08817).**
  - No eligible log in any run meets the threshold.
  - **Most levers ended below CTRL** at 24k (COMBO ≤ 0.014, TEXT-FIRST(-MASK) ≤ 0.07, MARKOV ≤ 0.047, K-CURR
    ≤ 0.118, against CTRL's 0.14–0.18). They slowed learning, consistent with 111-1's no-read basin: anything that
    delays pressure on the latent (text first, gold inputs) costs time.

## 100. Shorter silent stretches on multiplication: §98's recipe with a text snap every 2 steps (k = 2) (pre-registered 2026-10-03 23:54 EDT, after Literature's check 478cc193 and before any implementation or run; the user's request for an overnight plan, via Audit)

**Why.** On multiplication the first latent hand-off is learned and the second is the wall. §98's and §99's training logs
(raw, unscored): slots 1–2 ≈ 1.0 and slot 3 at 0.26–0.52 at 9k–20k steps.
* §100 shortens every silent stretch to one hand-off: a text snap every 2 steps.
* It asks whether the general recipe then solves multiplication, and whether it stays cheaper than text CoT at matched
  accuracy (C3), which is the real question at k = 2.
* It runs in parallel with §98 and §99 and does not depend on them. If §99 finds an accelerator, it can be combined
  later.
* **Prior** (478cc193, `docs/LIT_S100_K2_2026-10-03.md`).
  - No snap-frequency sweep has been published for arithmetic. §98 (k = 4) against §100 (k = 2) is the first.
  - Coconut's c thoughts per step and CoLaR's compression factor vary latents per step, not snap frequency.

**Task and recipe.** §98's data, test items, code path and budget (24k steps, bs 64, lr 1e-4), with **k = 2**.
* The test-time chunks are ops {1, 2}, {3, 4} and (5×5) {5}.
* At each chunk end the step text (which carries S_j at steps 2 and 4) is greedy-decoded and re-embedded as the next
  start, with the history cut.
* Training chunks are aligned to the test chunks, as 98-2.

**Arms (seeds 1–3, each task).**
* **GEN2:** GEN-M at k = 2.
* **MASK2:** GEN-M-MASK at k = 2 (each fork attends only to its slot and its own tokens).
* Comparators, from §98 on the same items: TC-M, NC-M, and the released ICoT-SI (0.998 / 0.996).

**Measures (§98's keys).**
* Test accuracy.
* Per-slot decode.
* Zero- and mean-removal answer accuracy (slot input 1 of each 2-op chunk).
* Checkpoint validity and edit IIA.
* FLOPs (as metered and restricted), sequential passes, and training compute.

**Rules (fixed now; per task, medians over seeds 1–3).**
* **M1-2:** GEN2 ≥ 0.95.
* **M-SOTA-2:** GEN2 ≥ the ICoT-SI bar (0.998 at 4×4, 0.996 at 5×5).
* **M-C3-2:** GEN2 ≥ TC-M − 0.01, at ≤ TC-M's median FLOPs (as metered; restricted reported).
* **M-C1-2:** MASK2 ≥ 0.95, and in those seeds zero- and mean-removal answer accuracy ≤ §98's NC-M median + 0.05.
* **Slot-1 removal decides the reading (Literature (1)).** At k = 2 the fork sees S_{j−1}, a, b_j and b_{j+1} in
  context, and two steps in one pass are within GPT-2's reach (ICoT-SI does a whole product in one pass).
  - If GEN2 keeps M1-level accuracy with the slot-1 latent zero- or mean-ablated, the result is **"text checkpoints every
    2 ops", not latent carriage**.
  - C1 is claimed only through M-C1-2.
* **C3 scope (Literature (2)).** "Cheaper" means cheaper than text CoT only. ICoT-SI writes no CoT and dominates on
  FLOPs at matched accuracy. Sequential passes are metered too.
* **Per-chunk accuracy** is reported, so a failure on 5×5's final single-op chunk is not confused with hand-off failure.
  Training chunks are aligned to test chunks, so single-op final chunks are trained.

**Readings.**
* **M1-2 + M-C3-2, with MASK2 and slot-1 removal showing latent use.** One latent hand-off between text checkpoints
  suffices on 4×4 / 5×5, more cheaply than text CoT. Without that latent-use evidence, it is a text-checkpoint result.
  - The snap interval k is the knob between latent work and written state.
  - Tags: C1 (unqualified if M-C1-2), C3, and C7 for checkpoints. Not C2 (in-distribution; 98-2). Not C4.
* **M1-2 only.** It solves the task, but the extra checkpoints cost as much as text CoT.
* **M1-2 fails.** Even single hand-offs between text checkpoints are not acquired in budget, so the obstacle is not
  hand-off depth.
* **Not run now:** a "train longer" reference (GEN-M k = 4 at 10× budget against ICoT-SI's ~200 epochs, about $20–40).
  It is deferred until §99 is scored.

**Placement.** 12 runs (2 arms × 2 tasks × 3 seeds), Vast (proof host first, ≤ 5) or local cards, about $8–12. Projection
to Audit. Run dirs `boxes/runs_mult/m100_{gen2,mask2}_{4x4,5x5}_s{1,2,3}`, with §98's eval.json keys. The scorer is
committed before any run.

**Note 100-1 (2026-10-04 23:58 EDT, before any §100 run).** §100 runs only on mult98 with 98-3's fix (one final fork
"+ P_n #### product").

**§100 result, 4×4 (2026-10-04 09:21 EDT; runs by the experiments session; scored by Theory with `theory/score_s100.py`,
committed before any run; 5×5 is pending).**

| 4×4 | value |
|---|---|
| GEN2 (k = 2) accuracy | 1.000 / 1.000 / 0.979 (median **1.000**) |
| ICoT-SI bar (released checkpoint, our eval) | 0.998 |
| TC-M (text CoT, matched budget) | 1.000 |
| GEN2 FLOPs / TC-M | **0.89×** as metered (0.997× restricted vocabulary) |
| MASK2 accuracy | 0.890 / 0.318 / 0.985 (median 0.89) |
| GEN2 mean-removal | 0.63 / 0.58 / 0.13 |

* **M1-2 met; M-SOTA-2 met** (1.000 ≥ 0.998); **M-C3-2 met** (≥ TC-M − 0.01 at 0.89× its FLOPs as metered; about
  equal with restricted vocabulary).
* **M-C1-2 not met** (MASK2 median 0.89 < 0.95).
* GEN2's mean-removal (up to 0.63) shows substantial recomputation from visible ops. No seed reaches the "text
  checkpoints only" flag (≥ 0.95).
* **Reading.** At 4×4, the general recipe with a text snap every 2 steps matches the best published implicit method
  (ICoT-SI) and costs ≤ text CoT. But its latents are partly bypassed, and the latent-only (masked) variant falls short.
  Tags: C3 (SOTA parity at 4×4), not C1.
  - MASK2's donor protocol (running, 106-4's question) says whether its 0.89 median is hand-off or recomputation.

**§100 result, 5×5 (2026-10-04 10:14 EDT; same scorer; §100 is now complete). Not met at 5×5.**
* GEN2: 0.100 / 0.093 / 0.091; MASK2: 0.060 / 0.016 / 0.002 (s2 and s3 evaluated `--eval_only` after eval OOMs).
  - TC-M is 1.000; ICoT-SI 0.996.
  - M1-2, M-SOTA-2, M-C3-2 and M-C1-2 are all not met.
* **Where it fails.** MASK2's per-slot decode is [1, 0.86, 0.77, 0.065, 1] / [1, 0.85, 0.75, 0.017, 1] /
  [1, 0.44, 0.08, 0.01, 1].
  - Chunk 1's hand-off (slot 2) is partly learned.
  - **Chunk 2's hand-off (slot 4) collapses**, as in notes 100-2 / 100-3: a chunk that starts from a visible checkpoint
    text, with a larger running sum at 5×5.
* **§100 overall.**
  - 4×4: k = 2 matches SOTA (1.000 vs 0.998) at ≤ text CoT's cost, with C1 not shown.
  - 5×5: the hand-off after a visible checkpoint fails.
  - Tags: C3 at 4×4 only.

**Note 100-2 (2026-10-04 09:27 EDT; descriptive; §104-style donor protocol on §100 MASK2 4×4, run by the experiments
session, `boxes/donor100.py`, `boxes/runs_mult/m100_mask2_4x4_s*/donor100.json`; n = 2000, masked decodes).**

| seed | intact S_2 / answer | chunk-1 slot-1 patch: donor / recipient | chunk-2 slot-3 patch: donor / recipient |
|---|---|---|---|
| s1 | 1.0 / 0.890 | 0.097 / 0.0095 | 0 / 0 |
| s2 | 1.0 / 0.288 | 0.166 / 0.010 | 0 / 0.002 |
| s3 | 1.0 / 0.981 | 0.109 / 0.010 | 0 / 0 |

(Chance overlap of the donor and recipient S_2 is 0.0095.)
* **Answers 106-4's caution.** Patching slot 1 drives S_2 from 1.0 to chance on the recipient side. So **S_2 is not
  recomputed from the visible operands; the slot-1 latent is causally necessary.** MASK2's success is not in-pass
  recomputation.
* **But the transplant is not clean:** only 10–17 % of S_2 decodes follow the donor, and at slot 3 the answer breaks
  entirely.
  - The likely confound is that **donors have a different multiplicand a**. The latent probably encodes the running sum
    relative to its own a (e.g. partial products), so it is incoherent in the recipient's context.
  - A **same-a donor variant** (donor problems share the recipient's a; only b differs) makes the transplant well
    defined. It has been requested, as descriptive.

**Note 100-3 (2026-10-04 09:31 EDT; descriptive; same-a donor variant requested in 100-2; B = (a_A, b_B), n = 2000,
`donor100_samea.json`).**

| seed | chunk-1 slot-1 patch: donor / recipient | chunk-2 slot-3 patch: donor / recipient |
|---|---|---|
| s1 | **0.740** / 0.112 | 0.008 / 0.002 |
| s2 | **0.791** / 0.112 | 0.003 / 0.025 |
| s3 | **0.999** / 0.112 | 0.008 / 0.008 |

(Recipient 0.112 equals the chance overlap: with the same a, S_2 coincides whenever b_1 matches.)
* **Chunk 1: the slot-1 latent carries the running sum, relative to a.** With a matched multiplicand, the transplant is
  clean (0.74–0.999), and the recipient is at chance. This is genuine latent carriage at multiplication's first hand-off
  (C1 signal, masked fork).
* **Chunk 2 breaks either way.** It starts from a visible checkpoint text (the recipient's S_2), and a donor latent
  conflicts with it. This is the same pattern as §105's slot 1: **the visible chunk start competes with the latent.**
  Markov segments (§108) remove that route.

## 101. How much discrete correction does a latent chain need? A snap-strength dose-response on S5 (pre-registered 2026-10-04 00:17 EDT, after Literature's checks a214bb6d and 5c2e080e and before any implementation or run; the user's approval of the directions memo, via Audit)

**Why.** Every success so far snaps to a discrete state. Every unsnapped chain drifts (directions memo, problem C). A snap
does two things:
* it **cuts the history**, which keeps context length in range;
* it **projects** onto a valid code, which corrects errors.

§101 keeps the cut and dials the projection down. That measures how much discrete correction a working latent chain
needs.

**Model.** §97's HYB-4 (text step target through the tied LM head; §92's structured codebook snap; k = 4; passed G4 3/3).
* The next chunk's start embedding is **s = (1 − β) · u + β · code_emb(decoded)**.
  - u = g(h), the last slot's output hidden state mapped into embedding space by §89's standardisation
    (μ_E + σ_E · (h − μ_h) / σ_h; an EMA frozen at the end of training).
  - decoded = the codebook index from the decoded step text, as in HYB-4.
* The history cut is kept at every β.
* **Training (Literature 5c2e080e).**
  - Sequences of a random **1–4 chunks** (4–16 ops) from a true start code, backpropagating through every u.
  - At each chunk boundary, β is a **per-chunk coin**: β = 1 with probability q, otherwise β_min. q anneals linearly
    from 1 to 0 over the first 60 % of updates, then 0, so training ends at β_min throughout.
  - The per-chunk coin forces chunks to read u from early on. It is the fix for the expected failure, posterior
    collapse onto the code (the §89 α = 1 pattern).
  - **Late noise:** Gaussian noise on u, σ = 0.05 × σ_E, in the last third of training.
  - Otherwise §97's recipe and budget (24k steps).

**Arms (seeds 1–3):** β_min ∈ {**1** (control, the same 2-chunk training), **0.1**, **0.01**, **0**}. 12 runs.

**Measures** (at β_min, all-position rollout).
* Cumulative accuracy at 512 and 2,048.
* Zero- and mean-removal (slot inputs 1..3; and u at boundaries).
* Per-chunk decode accuracy.
* Along the 2,048 rollout: ‖u‖ and the distance from u to the nearest code (drift).
* **The drift curve:** accuracy against length per β.
* **‖u‖ / ‖code_emb‖**, logged so the effective β matches the nominal one.
* FLOPs.

**Rule (fixed now, per arm).** An arm passes if cumulative ≥ 0.99 at 512 in ≥ 2/3 seeds, with zero-removal ≤ 0.10 in those
seeds (§92's rule).

**Primary reading:** the smallest β_min that passes.
* **β_min = 0 passes.** A continuous carry with a history cut and no discrete correction is exact on S5: the projection
  was not needed. Tags: C1 (use), C2, C6 (drift bounded) for "latent without discretisation". Not yet "without cuts".
* **Only 0.01 or 0.1.** A weak attractor suffices: "soft snap" is the honest design.
* **Only β = 1.** Discrete correction is load-bearing (Literature's flag that β → 0 removes the carrier). Directions C2,
  C3 and C6 are next.

**Placement.** About $8 on Vast (proof host first, ≤ 5). Projection to Audit.

**Note 101-1 (2026-10-04 00:22 EDT, before any §101 run; the user's question "self-discovered thought manifold in latent space
to guide back?"; Literature 6230b7f5, `docs/LIT_SELF_SNAP_2026-10-04.md`).**
* **New arm SELF-SNAP:** §101's soft snap s = (1 − β) · u + β · P(u), with **P(u) = the nearest centroid of a codebook
  learned online from the model's own u's**, replacing §92's given codebook at every snap.
  - M = 240 (2|G|, as §96).
  - EMA k-means updates (decay 0.99) with a dead-code reset (Łańcucki 2005.08520), and a small commitment term (0.05).
  - The codebook is initialised by k-means on u's at step 6k and used from then on. Before that, training uses
    single-chunk sequences only (Literature: start the codebook after the hand-off works).
  - The first chunk of a training sequence starts from the true start code. The first chunk at test starts from the
    identity's code (the problem's given initial state). Every later boundary uses learned centroids only.
  - **β_min = 0.1** with §101's per-chunk coin schedule. Seeds 1–3.
* **Premise gate ($0, fixed now; Literature (3)).** Run k-means (M = 240) on frozen §97 HYB-4 s1–3 last-slot u's over
  20k fresh chunks. Compute **purity**: the share of points whose cluster's majority true state equals their own.
  - SELF-SNAP runs only if the **median purity is ≥ 0.90**. Otherwise it is not run, and the purity is reported.
* **Rule:** §101's (cumulative ≥ 0.99 at 512 in ≥ 2/3 seeds, zero-removal ≤ 0.10).
  - **The comparison:** SELF-SNAP against the given-codebook β_min = 0.1 arm, paired by seed.
* **Descriptive:** code perplexity; centroid–state purity and NMI at the end of training (truth used by the scorer only);
  and FSQ (2309.15505) as a possible later no-collapse variant.
* **Scope (Literature).** "Self-discovered" applies to the clustering only. The latents are state-separable because of
  §97's per-op text targets, so this is not C4.
  - Readability (C7) of learned codes needs a post-hoc translator, unlike text checkpoints.
* **Prior:** State-Regularized RNNs 1901.08817 (from scratch); QBN 1811.12530; VQ-VAE; Token Assorted (codes, not drift
  correction). An online-learned snap in a pretrained LM's chain was not found, so it is new as scoped.

**Note 101-2 (2026-10-04 01:09 EDT; SELF-SNAP premise result, `theory/s101_premise.py`, `theory/results/s101_premise.json`).
The gate fails, so SELF-SNAP is not run (as pre-registered in 101-1).**
* **Result.** k-means (M = 240) on frozen §97 HYB-4 last-slot states u (standardised as §101's g), 20k fresh chunks:
  **purity 0.059 / 0.109 / 0.065 (median 0.065)**, NMI 0.20 / 0.37 / 0.23. All 240 clusters were used. The gate was
  ≥ 0.90.
* **Reading (descriptive).** The state is perfectly *readable* (the step-text decoder reads it at 1.0) but not
  *cluster-separable* in the raw hidden space. It occupies a low-variance subspace, dominated by other variation. A
  self-discovered manifold for snapping would have to be found in a learned projection (e.g. a small learned encoder
  trained only for clusterability / reconstruction), not by clustering raw latents.
  - That is a possible later design. It is not registered now.
* §101's other arms (β_min 1 / 0.1 / 0.01 / 0, given codebook) are unaffected.

## 102. State in depth: a gated residual path from slot j−1 to slot j on multiplication (pre-registered 2026-10-04 00:17 EDT, after Literature's checks a214bb6d and 5c2e080e and before any implementation or run; the user's approval via Audit)

**Why.** Latent hand-offs are learned slot by slot (the staircase; §92; §98's 4×4 slot 3). In our design, slot j gets its
state only by **attention retrieval** of slot j−1, which must be learned position by position. Depth recurrence with a
direct state path composes reliably (§88/§91).
* **Prior:** T²MLR 2607.15178 (a middle-layer state fed into the next token, retrofitted at 1.7B); Huginn / TRM
  input-injection recurrence.

**Arm STATEPATH**, in §99's harness on 4×4 multiplication with the fixed code (98-3).
* GEN-M, plus **h⁽⁴⁾_slot_j ← h⁽⁴⁾_slot_j + γ · W · h⁽⁹⁾_slot_{j−1}**.
  - Slot j−1's layer-9 hidden state is added to slot j's residual stream entering layer 5.
  - W is 768 × 768, initialised at identity; γ is a learned scalar gate initialised at 0.
* It applies within a chunk only (the first slot of a chunk has no previous slot).
* The normal attention path is unchanged. Seeds 1–3, §99's cap (24k), logging, valid-split early stop and eligibility
  (from step 0).
* **Control: §99's CTRL** (same code, seeds and harness).

**Rule (§99's, fixed now).** STATEPATH **accelerates** if its median steps-to-threshold is ≤ 0.5× CTRL's, with zero- and
mean-removal answer accuracy ≤ §98's NC-M 4×4 + 0.05 in its threshold-reaching seeds.
* Removal for STATEPATH also zeroes the injected path, so that the latent route is removed entirely.
* Descriptive (Literature 5c2e080e):
  - the per-slot curves (does the staircase flatten?);
  - the trajectory of the gate norm |γ| (it may stay ≈ 0 if attention already carries the state);
  - at test, separate ablations of the W path and of slot j's attention to slot j−1;
  - FLOPs and sequential passes, counting the extra 5-layer dependency.

**Readings.**
* **Accelerates.** Attention retrieval of the previous slot was the slow part, and a direct state path fixes it. It joins
  the recipe.
* **No.** The staircase is not about the retrieval route. The state's content or size is the next suspect (directions A3).

**Placement.** 3 runs on the local cards with §99 (about $0–3). It is scored with §99's scorer extended by the arm
`statepath`.

**§102 result (2026-10-04 09:07 EDT; statepath s1–3 run by the experiments session in §99's harness; scored by Theory with
`theory/score_s99.py` (arm `statepath`), committed before any run). STATEPATH does not accelerate.**
* All 3 runs reached the 24k cap without reaching threshold (CTRL is also censored, so the bar was ≤ 12k).
* Test accuracy: 0.170 / 0.032 / 0.013, against CTRL's 0.163 / 0.143 / 0.184. Removal: zero 0.0, mean ≤ 0.007.
* **Reading ("No").** A direct, gated residual state path from slot j−1 to slot j does not speed up multiplication's
  hand-off. The staircase is not about the attention-retrieval route. Consistent with §107 / 106-2: compute per op or the
  carrier's format, not routing.

## 103. Learning step states without step labels: filtered self-training by length (pre-registered 2026-10-04 00:17 EDT, after Literature's checks a214bb6d and 5c2e080e and before any implementation or run; the user's go-ahead via Audit)

**Why.** C4 inside the LM has failed by every label-thinning route (§93, §95) and by RL and annealing (§69, §87). §85 and
§96 show the barrier is **optimisation, not information**: a table teacher from final answers supplies exact per-op
labels.
* Self-training by length grows the competent range the way the std curriculum does for tables, with the model as its
  own teacher.
* **Prior:** STaR 2203.14465; Lee 2502.01612 and 2511.07378 (recursive self-training extends length, with a filter); our
  Pilot L/M (label-free ladder, toys).

**Task and model.** S5 under DeltaProduct's protocol, problems **from the identity**. §97's GEN-4 recipe (text step target
via a fork; text snap every 4 ops). Budget 24k steps total, seeds 1–3. **No ground-truth intermediate state is ever used.**

**Rounds** (fixed now).
* **R0 (steps 0–1.2k, 5 % of updates):** problems of length 1–2. Only the final answer is targeted: at the last slot,
  and, for length 2, at slot 2. Slot 1 gets no target.
* **R1–R4 (about 5.7k steps each):** problems of length ≤ 4, 8, 16, 32 (chunked at k = 4, uniform lengths).
  - Each round starts by **pseudo-labelling** a fresh pool of 50k problems with the current model's greedy step decodes.
  - A problem's intermediate targets are kept **only if** it passes three filters in order (Literature 5c2e080e):
    1. **Validity:** every decoded checkpoint parses as a step text.
    2. **Final answer:** the decoded final answer equals the true one.
    3. **Cross-seed vote:** at least 2 of the 3 seeds' round-(r−1) models produce identical intermediate decodes, and
       the majority decode is the target. This is a stand-in for 2502.01612's vote across independently trained
       models.
  - Self-consistency (the chain decode against a k = 1 restart decode) is reported but not required.
  - **Coupling disclosed:** the vote shares pseudo-labels across seeds, so C6 from §103 would need an independent
    replication.
  - Kept problems give per-slot targets. Discarded problems give the final-answer target only.
* The share of length ≤ 2 problems over all training is ≤ 5 % (p, as §96).
* **Adaptive length rule (fixed now; Literature (3)).** Before each round, the current model's accuracy at the new
  length and each filter's yield are recorded. If the kept share is < 20 %, an intermediate length is inserted (e.g.
  6, 12, 24) and the remaining budget is split evenly. The 24k total is unchanged.

**Measures.**
* Per round: the share filtered by each criterion, the kept share, accuracy at the new length before training, and the
  pseudo-label accuracy against truth. The last is measured by the scorer only; never used in training.
* At the end: cumulative accuracy at 512 and 2,048 (all-position decodes), zero- and mean-removal, per-slot decode, and
  checkpoint edit IIA.

**Rule (fixed now).**
* **SELF:** cumulative ≥ 0.99 at 512 in ≥ 2/3 seeds, with zero-removal ≤ 0.10 in those seeds.
* **Descriptive:** pseudo-label precision per round (truth only in the scorer).

**Readings.**
* **SELF met.** A pretrained LM learns S5 latent restarts from final answers only, with no teacher outside itself.
  **C4 inside the LM, qualified by p ≤ 0.05.** Round 0's length-1/2 answers are short labels at about 5 %, near §85's
  3–10 % threshold, plus §97's C1 (qualified) + C2 + C3 + C7 for checkpoints.
* **Fails with high pseudo-label precision but low coverage.** The filter is too strict; report the kept share.
* **Fails with low precision.** Self-generated errors lock in (Literature's noise flag; §62).

**Placement.** About $6–8 on Vast (one proof seed first). Projection to Audit. Implementation: the experiments session, from
`boxes/gen97.py`.

**§103 result (2026-10-04 08:20 EDT; run by the experiments session on Vast, 155ae67f; scored by Theory alone with
`theory/score_s103.py`, committed before any run; `theory/results/score_s103.json`). SELF is not met, but it is a near
miss for C4.**

| seed | cumulative at 128 / 256 / 512 | 2,048 | final-state acc | zero / mean removal |
|---|---|---|---|---|
| s1 | 0 / 0 / 0 | 0 | 0.006 | 0.003 / 0.27 |
| s2 | 0.980 / 0.973 / **0.9615** | 0.896 | 1.000 | 0.002 / **0.949** |
| s3 | 0.979 / 0.972 / **0.9585** | 0.842 | 0.996 | 0.003 / 0.19 |

* **SELF** (≥ 0.99 at 512 in ≥ 2/3): **not met**.
* **But:** trained **from final answers only** (short_fraction 0.050, no ground-truth intermediate state ever used), two
  of three seeds carry S5 to about 0.96 at 512 and 0.84–0.90 at 2,048. That is the closest in-LM C4 result in the
  project.
* **Rounds (shared by the cross-seed vote).**

  | round length | 4 | 3 | 4 | 8 | 6 | 5 | 6 | 8 | 16 | 32 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | kept share | 6e-5 | 1e-4 | 1e-4 | 0.003 | 0.005 | 0.007 | **0.54** | 0.67 | 0.998 | 0.999 |
  | pseudo-label precision | 0.0 | 0.5 | 1.0 | 0.24 | 0.33 | 0.32 | 0.978 | 0.983 | 0.984 | 0.983 |

  - The adaptive rule inserted lengths until, at length 6, the model's own checkpoints became reliable. From then on,
    precision held at about **0.98**.
  - **About 2 % wrong pseudo-labels were locked in**, which is consistent with the cap near 0.96 (Literature's noise
    flag; §62). The vote across seeds sharing a pool lets correlated errors through.
* **C1 caveat:** s2's mean-removal is 0.949, the fork bypass §104 found in §97. §103 used the unmasked recipe, so its C1
  is not claimed.
* **Reading.** C4 inside the LM nearly works by filtered self-training by length. The bottleneck is pseudo-label
  precision (about 0.98), not coverage (kept share up to 0.999).
  - A stricter filter (unanimity, plus a chain-vs-restart self-consistency check) and/or a final relabelling round is the
    natural next step. Not registered now; C1 work has priority.

## 104. Do the latent slots carry the state serially? Donor patching and step-specific blocking on §92 and §97 (pre-registered 2026-10-04 00:33 EDT, after Literature's check 5054f476 and before running on any checkpoint; C1-first under the user's priority order)

**Why.** C1's operational test asks for **step-specific blocking showing serial, hop-by-hop use**. Our removal tests so far
ablate slots 1..k−1 jointly. §97's text fork also partly recomputes the state under mean ablation. §104 tests each slot's
causal role directly, at no training cost.
* **Prior** (5054f476):
  - Geiger 2106.02997 (causal abstraction); DAS 2303.02536 (interchange intervention accuracy); ROME 2202.05262 (causal
    tracing).
  - Method cautions: Zhang & Nanda 2309.16042, Heimersheim & Nanda 2404.15255, multiple mediators (2606.27510).
  - Per-step latent-CoT interventions: Li 2602.08783, SCIT 2608.27265 (CODI computes in the V-cache); our §34/§45/§56.

**Models:** §92 k = 4, seeds 1–5 (5 × 5 head on the slot); §97 GEN-4 and HYB-4, seeds 1–3 (text fork).

**Setup.** Within one 4-op chunk (no snap; post-snap effects are not scored): 2,000 fresh recipient chunks A and donor
chunks B, with random starts and ops. Donors are patched at the same slot index (Literature (2)).
* **Primary (resample/donor patch).** For each slot j = 1..3:
  - **single:** slot j's input ← B's slot-j input;
  - **joint:** additionally, op j's token KV entries ← B's (the positions that produced slot j's input; Literature (4)).
  - Scored at the final slot:
    - **donor-consistent** = the decode equals B's state after j ops composed with A's ops j+1..4 (the slot is used);
    - **recipient-consistent** = the decode equals A's own true final state (recomputation / bypass).
  - Chance is 1/120.
* **Secondary:** zero and mean replacement of slot j's input; per-slot decode accuracy against A's truth.
* **Script:** `theory/s104_serial.py`, committed with this registration.

**Rules (fixed now).**
* **USE(j):** single-patch donor-consistent ≥ 0.90.
* **BYPASS(j):** single-patch recipient-consistent ≥ 0.10.
* **SERIAL** (per model): USE(j) for j = 1, 2, 3, with no BYPASS(j).
* **Per family:** SERIAL holds if ≥ 4/5 (§92) or ≥ 2/3 (§97 GEN-4; §97 HYB-4) of models are SERIAL.
* **Joint patch (descriptive):** if single fails but joint reaches USE, the state is carried jointly by the slot and the
  op-token residuals (multiple mediators).

**Predictions.**
* §92 is SERIAL (≥ 4/5): its head reads only the slot, and §92's joint mean removal is ≤ 0.03.
* §97 GEN-4 / HYB-4 show BYPASS at one or more slots, in line with their mean-ablation recompute (0.15–0.50).

**Readings.**
* **§92 SERIAL.** Unqualified, serial, hop-by-hop C1 for §92's latent restarts. That is the north star's C1 test, passed
  within a chunk.
* **§97 BYPASS at slot j.** The text fork recomputes from the visible ops at that slot. The recipient/donor split says
  where. This sets the bar for §98/§100's MASK arms and §99's MARKOV.
* **Joint-only USE.** The state is distributed across the slot and the op-token residuals. C1 holds for the latent
  positions jointly, not for the slot alone.

**Placement.** Eval only: about 20 GPU-minutes on a local card (as the experiments session offers), or the CPU at
Idle priority. $0.

**§104 result (2026-10-04 05:46 EDT; run by Theory on the local 4060, `theory/s104_serial.py` committed with the
registration; scored with `theory/score_s104.py`, written after the run printed §92's partial values (disclosed);
`theory/results/score_s104.json`). SERIAL is not met in any family. §97's latents are bypassed.**

| family | SERIAL (single-slot USE at all slots, no BYPASS) | joint patch donor-consistent at all slots | BYPASS (recipient-consistent ≥ 0.10) |
|---|---|---|---|
| §92 (5 × 5 head on the slot) | **2/5** (s1, s4: donor ≥ 0.97) | 3/5 (s1, s2, s4) | s3 slot 2 (0.29), s5 slot 1 (0.26) |
| §97 GEN-4 (text fork) | **0/3** (single donor ≤ 0.02) | 0/3 (≤ 0.31) | every slot, 0.21–**1.0** |
| §97 HYB-4 | **0/3** (single donor ≤ 0.27) | 0/3 (slot 1 only) | 0.05–**1.0** |

* **Headline (the user's C1 priority).** In §97's text-fork models, a natural-looking donor latent at slot j is
  **ignored**. The final decode equals the recipient's own true state (recipient-consistent up to 1.0), so the state is
  recomputed from the visible start text and ops in context.
  - Zero ablation broke them only because zeros are off-distribution.
  - **§97's C1 (use) is not supported by donor patching.** Its chunks behave like the written-state loop with latent
    tokens present but unused: the §66 WC = WC₀ pattern, now shown causally.
* **§92.** The state is in the slot alone in 2/5 seeds. In the other three it is carried jointly by the slot and the
  op-token residuals (multiple mediators, Literature (4)), with partial recomputation at one slot in s3 and s5.
  - Serial C1 at the slot level is **2/5**, short of ≥ 4/5.
* **Consequence.**
  - Our C1 claims so far (§92, §97) rest on zero ablation. Causal patching shows that only designs which remove the
    visible route can claim serial latent carriage.
  - That makes the C1-first queue decisive: §105 (masked fork, with §104's protocol as its HANDOFF rule), §99's MARKOV,
    and §107's forced-carriage arms.
  - An S5 MARKOV test (fresh segments; carriage forced; S5's per-op lookup is cheap compute) is the natural next
    registration.

## 105. C1 for the headline result: §97's S5 recipe with a masked fork, scored by donor patching (pre-registered 2026-10-04 01:34 EDT, after Literature's check 78a12af7 and before any implementation or run; Audit's steward pick, C1-first)

**Why.** §97's general recipe carries S5 exactly to 2,048 (C6: 5/5). But its C1 is qualified: under mean ablation the text
fork recomputes from the visible ops (note 97-3; HYB-4 s5 keeps 1.000).
* §105 removes that route: each side decode sees only its own slot.
* **Prior** (78a12af7): SIM-CoT's latent-conditioned auxiliary decoder; the input-only-control rationale (2509.13316);
  SCIT 2608.27265 and Li 2602.08783 for interventions. A masked fork as the main bypass control is new as a design.

**Model.** §97 GEN-4 (text step target, text snap every 4 ops, 24k steps, seeds 1–3; 4–5 added on a pass), with **§98's
fork mask**.
* Every step fork attends only to its own slot's cached K/V (all layers) and its own decode tokens.
* The chain is unchanged.

**Measures:** §97's (cumulative at 512 / 2,048, zero- and mean-removal, per-slot decode, validity, edit IIA, FLOPs), plus
**§104's donor protocol** run on the trained models (`theory/s104_serial.py`, models added).

**Rules (fixed now).**
* **G4-M:** cumulative ≥ 0.99 at 512 in ≥ 2/3 seeds, with zero-removal ≤ 0.10 (§97's G4).
* **MASK-DEC:** in those seeds, mean-removal accuracy after the first ablated slot ≤ 0.06 (chance 1/120 + 0.05).
* **HANDOFF:** §104's USE(j) for j = 1..3 (single-patch donor-consistent ≥ 0.90) with no BYPASS, in ≥ 2/3 models.
* **Wording (Literature).** A MASK pass shows only "decodable from the slot alone". The slot's own forward pass still
  sees the ops, so the hand-off claim needs HANDOFF too.
  - **Unqualified C1 for the headline result = G4-M + MASK-DEC + HANDOFF.**

**Readings.**
* **All three.** The general recipe's latents carry the S5 state hop by hop: C1 without qualification, plus §97's C2 /
  C6 (C6 at seeds 4–5) / C7 for checkpoints.
* **G4-M only.** The fork was the bypass route, but the slot recomputes in-pass. Report it, with the donor split.
* **G4-M fails.** The masked decoder cannot be trained at this budget.

**Placement.** About $3–4 (3 runs, then 2) on Vast. Projection to Audit.

**§105 result (2026-10-04 09:21 EDT; models trained by the experiments session; G4-M / MASK-DEC from eval.json; HANDOFF
from §104's donor protocol with the masked decode, run by Theory on the 4060 (`theory/s105_donor.py`); scored with
`theory/score_s105.py`, committed before any run). G4-M and MASK-DEC are met. HANDOFF is not met (0/3), so unqualified
C1 is not met.**

| model | cumulative 512 / 2,048 | mean-removal | single-slot donor-consistent (slots 1 / 2 / 3) | recipient-consistent (slot 1) | joint patch, slot 1 |
|---|---|---|---|---|---|
| s1 | 1.0 / 1.0 | 0.007 | 0.873 / **0.998 / 1.000** | 0.017 | 1.0 |
| s2 | 1.0 / 0.998 | 0.008 | 0.476 / **1.000 / 1.000** | 0.051 | 1.0 |
| s3 | 1.0 / 1.0 | 0.009 | 0.132 / 0.791 / 0.878 | 0.26 | 0.16 |

* **Reading.** The masked fork turns §97's fully bypassed latents (single donor ≤ 0.02 everywhere, §104) into **serial
  latent carriage at slots 2 and 3** in s1 and s2 (donor ≥ 0.998, recipient ≤ 0.012).
* **Slot 1 is the remaining visible route.** It is one op from the chunk's visible start text, so slot 2's forward pass
  can recompute S_1 from the start text plus op 1's tokens.
  - The joint patch (slot plus op-token KV) restores donor-consistency to 1.0 in s1 and s2.
  - s3 also recomputes at slot 1 (recipient 0.26).
* **Consequence.**
  - Masking the fork is necessary but not sufficient. The visible chunk start still offers a recompute route for the
    first hand-off.
  - §108's Markov segments remove exactly that route (no earlier ops or start text are visible past segment 1). §108 is
    the decisive test of unqualified C1 on S5.

## 106. C2 on a real task: bounded-state streaming multiplication, latent chains against the same format written out (pre-registered 2026-10-04 01:34 EDT, after Literature's check 78a12af7 and before any implementation or run; Audit's steward pick)

**Why.** The published multiplication CoT is not restartable: the running sum grows, and P_j carries a counter (98-2). So
§98 has no C2 test. Streaming long multiplication has a **bounded** state:
* at step j, T = C + a·b_j;
* **emit** d_j = T mod 10, which is final;
* carry C ← ⌊T / 10⌋, at most len(a) + 1 digits for any length of b.

That makes a train-short / test-long test possible on a real task.

**Prior** (78a12af7):
* Jelassi 2306.15400: 5×3 → 35×3, only with priming; relative PE alone fails.
* Cho 2410.15787; Lee 2502.01612; Turing Programs 2407.03310; Neural GPU 1511.08228 (bounded local state, long
  extrapolation).
* **Literature's key point:** a written-carry CoT *with a history cut* should already extrapolate. So C2 is credited to
  the format plus cut. The latents' contribution is C1 / C3 against the same format written out.

**Data (a generator; seeds fixed by the implementation).**
* Digits are least-significant first. a has 4 digits.
* **Train** b ∈ 1..5 digits. **Test** b = 4, 8, 16, 32 (1,000 items each), plus a = 5 at b = 8 (descriptive
  carry-width shift).
* Step text "emit d_j , carry C_j" (C_j written at fixed width len(a) + 1). A final **flush step** writes the remaining
  carry digits, and it is trained (Literature (b)).
* The product is **assembled outside the chunk** from the emitted digits and the flush (Literature (a)), so no emitted
  digits accumulate in context.

**Arms (seeds 1–3; §97's budget).**
* **GEN-S:** §97's recipe with k = 4 (latent per step; a text snap of "emit, carry" every 4 steps, history cut).
* **GEN-S-MASK:** as GEN-S, with §98's masked fork.
* **WC-S:** the same format **written out** with a history cut every 4 steps: text CoT per step, with the restart from the
  last written carry. This is Literature's required comparator.
* **TC-S:** the same format as one continuous text CoT, with no cut (the classic baseline).
* ICoT-SI: the released 4×4 checkpoint on our a = 4, b = 4 items only (in range, descriptive; it is per-size and cannot
  extrapolate).

**Rules (fixed now; exact product).**
* **C2-S** (per arm): ≥ 0.95 at b = 16 and ≥ 0.90 at b = 32 in ≥ 2/3 seeds (trained on b ≤ 5).
* **C1-S** (MASK): GEN-S-MASK meets C2-S, and in those seeds zero- and mean-removal of the slot inputs gives exact
  accuracy at b = 8 ≤ 0.05.
* **Attribution (descriptive, fixed now).**
  - If WC-S also meets C2-S, length generalisation belongs to the format plus cut.
  - The latents' contribution is then C1 (C1-S) and C3 (GEN-S FLOPs against WC-S at matched accuracy).
  - If only the latent arms meet C2-S, the latents add C2 beyond the written format.

**Readings.**
* **GEN-S-MASK meets C2-S and C1-S.** On a real task, latent chains with restartable checkpoints extrapolate from 5 to 32
  multiplier digits with the state carried in the latents. Tags: C1, C2 (with WC-S's attribution), C3 against WC-S.
* **Only WC-S / TC-S extrapolate.** The written format does it; the latents do not add length generalisation here.
* **TC-S fails while WC-S passes.** The range principle again: the cut, not the format alone, is what extrapolates.

**Placement.** 12 runs (4 arms × 3 seeds), about $8–12. One proof host first; launched after §98's fixed GEN-M reruns
show the recipe learns multiplication. Projection to Audit.

**Note 106-1 (2026-10-04 03:31 EDT, before any §106 run; Audit's 03:13 steward pass).**
* **Why move it earlier.** §99's fixed-code logs locate multiplication's C1 blocker at carrying a multi-digit running sum
  through one latent vector (slot 3). §106's streaming format carries only a bounded carry (≤ len(a) + 1 digits), and the
  emitted digit is final.
* **The pilot.** **GEN-S-MASK seeds 1–3 launch first** (one host, about $2–3), without waiting for §98's scoring.
  - These runs **are** §106's GEN-S-MASK arm: evaluated at every b, under §106's rules.
* **Gate for the rest of §106** (GEN-S, WC-S, TC-S): GEN-S-MASK's in-range accuracy (b = 4) ≥ 0.95 in ≥ 2/3 seeds.
  - If the gate fails, the latent arms are not run further. WC-S and TC-S may still run as a written-format C2 reference
    (descriptive).

**Note 106-2 (2026-10-04 06:45 EDT; the §106 pilot, GEN-S-MASK s1–3, run by the experiments session; descriptive under
106-1). The gate fails, so the latent arms are not run further.**
* **Result.**
  - Exact product at b = 4: **0.004 / 0.005 / 0.006**. 0 at b = 8 / 16 / 32 and a5b8.
  - Per-step decode at b = 4: 1.0 / ≈ 0.21 / ≈ 0.037 / ≈ 0.008.
  - Checkpoint validity 1.0; removal at b = 8 is 0.
  - s1's curve: step 1 reaches 1.0 by 20k; step 2 plateaus at about 0.23 from 10k on, with the loss at about 3.0.
* **Reading (descriptive).** The first emitted digit is learned. But **even a bounded carry (≤ 5 digits) is not carried
  past one latent hand-off** under the masked fork at k = 4.
  - So multiplication's C1 wall is **not** state growth (contrast §98's growing S_j).
  - It fits §107's compute-per-op reframing: each step needs carry + a · b_j, a multi-digit multiply-add, in one pass.
  - §100 MASK2 (one hand-off per chunk) and §107 (pause vs width) are the live tests.
* **WC-S / TC-S** (the written-format C2 reference) are **deferred** under the user's C1 priority. They would describe
  only the written format, not the latents. They are run only if a latent arm learns the streaming task.

**Note 106-3 (2026-10-04 06:46 EDT, before any k = 2 run; Audit's caution).**
* **Correction to 106-2's reading.** The pilot ran at k = 4, with several silent hand-offs per chunk. That **confounds**
  the bounded-carry format with the multi-hand-off wall already seen in §98 MASK 5×5 and §99. 106-2's "not state
  growth" is withdrawn pending this test. §100 MASK2 4×4 at k = 2 (one hand-off per chunk) learned in 2/3 (answers 0.89
  / 0.32 / 0.985), with removal at 0.
* **New arms (part of §106; seeds 1–3; train b ≤ 5, test b = 4 / 8 / 16 / 32):**
  - **GEN-S-MASK-k2:** the masked fork with a text snap every **2** steps (one latent hand-off per chunk).
  - **WC-S-k2:** the same format written out, with a history cut every 2 steps (the written comparator; Literature
    78a12af7).
* **Rules (§106's, unchanged):**
  - C2-S: ≥ 0.95 at b = 16 and ≥ 0.90 at b = 32 in ≥ 2/3.
  - C1-S for the MASK arm: C2-S plus zero/mean removal ≤ 0.05 at b = 8.
  - Attribution: if WC-S-k2 also meets C2-S, C2 belongs to format + cut, and the latents get C1 (and C3 vs WC-S).
* **Reading if GEN-S-MASK-k2 meets C2-S and C1-S:** latent hand-offs between restartable written carries extrapolate
  multiplication from 5 to 32 multiplier digits, with the carry passing through the latents. That would be C1 and C2 on
  a real task.
* About $4–6 (6 runs). It joins the C1 group after tfslot and §108. Scored with `theory/score_s106.py`, extended to
  the dirs `s106_gens_mask_k2_s*` and `s106_wcs_k2_s*`.

**Note 106-4 (2026-10-04 09:07 EDT; the 106-3 arms, run by the experiments session; scored by Theory with
`theory/score_s106.py`; GEN-S-MASK-k2 s2's re-evaluation is pending but cannot change the verdict).**
* **WC-S-k2** (the streaming format written out, history cut every 2 steps), trained on b ≤ 5: **1.000 at b = 4 / 8 / 16 /
  32 in 3/3** (one cell 0.999). C2-S is met: the written bounded-carry format plus the cut extrapolates multiplication
  from 5 to 32 multiplier digits perfectly.
* **GEN-S-MASK-k2** (masked fork, one latent hand-off per chunk): exact product at b = 4 is 0.003 / 0.007 (s1, s3).
  - Per-step decode at b = 4: 1.0 / 0.20–0.25 / 0.04–0.07 / ≤ 0.011, with checkpoint validity 1.0.
  - **Even one in-chunk hand-off of the carry fails.**
  - C2-S and C1-S are not met.
* **Attribution (fixed in 106-3).** C2 on multiplication belongs to **the written format plus the cut**. The latents add
  nothing here, so this is a written-state (range-principle) result, not a latent one.
* **Descriptive contrast with §100 MASK2 4×4 (dataset format, k = 2; answers 0.89 / 0.32 / 0.985).** There, S_2 is
  recomputable from the visible operands within the slot's own forward pass (§105's caution: the masked fork does not
  stop that). §100's partial success may be in-pass recomputation, not carriage. §104's donor protocol on the MASK2
  models would decide it.

## 107. Multiplication's C1 blocker: carrier width or compute per op? m latents per op vs pause positions, with carriage forced (pre-registered 2026-10-04 03:32 EDT, after Literature's check 640ffe59 and before any implementation or run; Audit's 03:13 steward pick)

**Why.** §99's fixed-code logs (raw, descriptive; Audit) locate the blocker at slot 3: the first step whose state, the
running sum S_3 = S_2 + a · b_3 · 10², must arrive through a latent. It reads CTRL 0.34–0.39 at 10k, and MARKOV (forced
carriage) 0.11–0.14 at 11k. S5's 120-way state was carried fine. Two explanations compete:
* **Carrier width:** one vector per op cannot carry a multi-digit number in a usable format.
* **Compute per op** (Literature's reframing): slot 3 must do a 4 × 1 product plus a carried add in one pass, where S5's op
  was a table lookup. Capacity is not the obvious limit (an 8-digit sum is about 27 bits in 768 dimensions; MUX
  2607.18264).
* **Prior** (640ffe59): Coconut's c thoughts per step (c = 2 on GSM8K; more helped), CoLaR, SuperThoughts; pause tokens
  2310.02226 / 2505.21024; 2510.00184 (a running-sum auxiliary unlocks multiplication).

**Arms** (§99's harness on 4×4 multiplication: fixed code, MARKOV's fresh-cache segments so carriage is forced, the K-CURR
bootstrap, 24k cap, the valid-split early stop, eligibility from 6k; seeds 1–3).
* **LAT-m (m = 2, 4).** Each op gets **m latent slots**.
  - Latent 1's input is the last hidden state at op j's token; latent i's input is the last hidden state of latent
    i − 1.
  - Segment j = [a, op j−1's **m** latents (or the chunk's start text), op j] → m new latents.
  - The step and final forks read from the last latent's position (own segment).
* **PAUSE-4.** One carried latent, plus **3 pause positions** before it within the segment. Each pause is a learned
  embedding shared across positions and is *not* carried to the next segment.
  - Segment j = [a, op j−1's latent, op j, pause × 3] → 1 latent.
  - It adds compute per op without widening the carrier.
* **m = 1: §99's MARKOV arm** (same harness and seeds), reused as the control.
* Re-indexing stays contiguous as segments grow (Literature (2)).

**Measures.** §99's (curve, steps-to-threshold, test accuracy, removal), plus:
* per-op decode on the test split;
* for LAT-m, the cosine between the m latents and a per-latent zero-removal effect (Literature (1): untargeted latents may
  collapse to copies);
* sequential passes and FLOPs, which scale with m.

**Rules (fixed now).**
* **CARRY-3** (per arm): test decode at op 3 ≥ 0.95 at the stop / cap checkpoint in ≥ 2/3 seeds.
* **NO-BYPASS:** in those seeds, zero- and mean-removal of every carried latent gives answer accuracy ≤ §98's NC-M 4×4
  median + 0.05.
* **Accelerates** (§99's rule against §99 CTRL) is reported.

**Readings (fixed now).**
* **LAT passes, PAUSE fails.** Carrier width is the blocker: a wider latent state carries the running sum.
* **PAUSE passes.** Compute per op is the blocker: extra in-step positions suffice and the single carrier is enough.
* **Both pass.** Compute was enough; width is not needed. LAT's latent cosine says whether the extra latents were used.
* **Neither.** Neither lever unblocks it at this budget. The explicit running-sum auxiliary (2510.00184) is next.
* **Tags:** C1 (carriage forced by MARKOV plus removal) for multiplication if NO-BYPASS holds. Not C4.

**Placement.** 9 runs (3 arms × 3 seeds), about $5, queued behind §101 / §103 / the §106 pilot as hosts free up.
Projection to Audit.

## 108. Forced latent carriage on S5: Markov segments, scored by donor patching (pre-registered 2026-10-04 05:48 EDT, after Literature's check de57be3f and before any implementation or run; C1-first)

**Why.** §104's donor patching shows our C1 claims rested on zero ablation.
* §97's latents are ignored: the state is recomputed from the visible start text and ops.
* §92's slot alone carries the state in only 2/5 seeds.
* Only designs that remove the visible route can claim serial latent carriage. On multiplication, forced carriage (§99
  MARKOV) is blocked by per-op compute (§107's reframing). S5's per-op update is a table lookup, so S5 is the most
  likely place for the first unqualified, serial C1 in a pretrained LM.
* **Prior** (de57be3f, `docs/LIT_S108_MARKOV_S5_2026-10-04.md`). New as scoped: no pretrained-LM S5 run under a
  latent-only carry was found. Nearest: T²MLR, LIFT, RMT, Markovian Transformers; from scratch, 2510.14095.
  - Unbottlenecked pretrained transformers use parallel-scan or shortcut mechanisms that fail out of distribution (Li
    2503.02854, Liu 2210.10749, 2602.14814). MARKOV removes them.
  - §71 WC₀₄'s in-pass composition is irrelevant here: a segment only needs read-latent + apply-one-op, which §92's
    slot 1 shows is easy. The open question is a self-consistent latent code.

**Model.** §97's GEN recipe (text step target through the tied LM head; text snap every k ops with a history cut), with
**Markov segments** (as §99 MARKOV).
* Each op step is a fresh-cache segment **[<|endoftext|>, slot j−1's latent (or the chunk's start text, for the first
  op), op j's tokens] → slot j**, with positions re-indexed from 0, identically in training and test.
  - **<|endoftext|> sits at position 0, GPT-2's attention sink** (massive activations; §72; 2402.17762 / 2410.10781), so
    the latent sits at position 1, never at the sink (Literature (1), the main fix).
* Step forks attend only to their own segment. S_j can reach slot j **only through slot j−1's latent**: earlier ops and
  slots are invisible.
* **Bootstrap (Literature (2)).**
  - K-CURR: k = 1 for the first 2k steps, 2 to 6k, then the arm's k.
  - **Teacher-forced segment inputs:** with probability ρ per segment (one coin per batch per position), the segment's
    input is the **gold previous step text** (as at a chunk start) instead of slot j−1's latent. ρ anneals linearly
    from 1 to **exactly 0 at 12k**, so all segments learn in parallel, and the last 50 % of training uses only the
    model's own latents.
  - Budget 24k steps, bs 64, lr 1e-4.
* **Segment-type balance (Literature (3)):** start-text segments and latent segments are both trained in every batch
  (the chunk structure gives a 1 : (k − 1) ratio). The ratio is reported.

**Arms (seeds 1–5; ≥ 5 because §104's §92 split looks like a per-seed solution choice; Audit).**
* **MK-4:** test snap every 4 ops.
* **MK-8:** test snap every 8 ops (a longer silent stretch; trained with k up to 8).

**Measures.**
* §97's: cumulative accuracy at 512 and 2,048, per-slot decode, checkpoint validity and edit IIA, FLOPs.
* **§104's donor protocol as the primary C1 measure** (from now on, per Audit): single-slot donor patching at every
  in-chunk slot, scored donor- vs recipient-consistent at the chunk's last slot.
* Zero and mean ablation, secondary.
* **Per-hand-off accuracy** (Literature): cumulative ≥ 0.99 at 512 needs about 0.9999 per hand-off.
* ρ is logged at every log, and the threshold is evaluated only once ρ = 0 (as 99-1).

**Rules (fixed now; per arm).**
* **ACC:** cumulative ≥ 0.99 at 512 in ≥ 4/5 seeds.
* **SERIAL** (per model): USE(j) (single donor-consistent ≥ 0.90) at every in-chunk slot, with no BYPASS
  (recipient-consistent ≥ 0.10).
* **C1-MK:** ACC, plus SERIAL in ≥ 4/5 seeds.
* **C6:** ACC at 2,048 (cumulative ≥ 0.99) in ≥ 4/5.

**Readings.**
* **C1-MK at MK-4 (and MK-8).** The first **unqualified, serial C1** in a pretrained LM: the S5 state is carried hop by
  hop through latents, with no visible route, exact to 512 (2,048), with readable text checkpoints. Tags: C1, C2
  (within training-length chunks, plus restarts), C6 (if ≥ 4/5 at 2,048), C7 for checkpoints. Not C4.
* **ACC fails.** Forced carriage is not acquired at this budget even for table-lookup steps. Budget (§92) and
  teacher-forced slots (§99 TF-SLOT) are the next levers.
* **ACC holds, SERIAL fails.** This should be impossible by construction, since the segments are fork-isolated. If seen,
  it means a leak in the segment construction, and it is reported as a bug. The donor protocol here is the sanity check
  (USE ≈ 1, BYPASS ≈ 0; Literature (4)).

**Placement.** 10 runs (2 arms × 5 seeds), about $6–8 on Vast (proof host MK-4 s1 first, ≤ 5 hosts). Projection to
Audit. Implementation: the experiments session (gen97 plus accel99's markov segment construction). Scorer committed
before any run.

## 110. Self-training by length, v2: independent seeds, label-free precision gate, masked fork (C4 + C1) (pre-registered 2026-10-04 08:24 EDT, after Literature's check 552c57a5 and before any implementation or run; Audit's request after §103)

**Why.** §103's filtered self-training (S5, final answers only, ≤ 5 % short) reached 0 / 0.9615 / 0.9585 at 512. It was
the closest in-LM C4 result.
* The cap was **pseudo-label precision ≈ 0.98**. The cross-seed vote on a shared pool lets correlated errors through.
* s2's mean-removal (0.949) shows the fork bypass, so there is no C1.
* §110 targets both: per-seed, label-free filtering with a precision gate, and the masked fork.
* **Prior** (552c57a5, `docs/LIT_S110_SELFTRAIN_2026-10-04.md`).
  - Multi-view agreement as a precision proxy: co-training and Dasgupta et al.'s agreement bounds, which hold only for
    conditionally independent views; Jiang 2106.13799; self-consistency filtering (2203.11171, Huang 2210.11610).
  - Iterated pseudo-labelling: noisy student 1911.04252, ReST 2308.08998, Lee 2502.01612, STaR 2203.14465.
  - Confirmation bias: Arazo 1908.02983.

**Task and model.** As §103: S5 under DeltaProduct's protocol, problems **from the identity**, **final answers only** (no
ground-truth intermediate state is ever used in training or in any training decision). §97's GEN-4 recipe with
**§105's masked fork** (each step fork sees only its slot's K/V and its own tokens). Budget 24k steps. **Seeds 1–5, each
with an independent pool** and no cross-seed vote.

**Rounds.**
* **R0:** as §103 (lengths 1–2, ≤ 5 % of updates; final-answer slots only).
* **R1…:** lengths ≤ 4, 8, 16, 32 (3..L_max), with §103's adaptive insertion.
* Per round, a fresh 50k pool is labelled with the seed's own greedy decodes. A problem's per-slot targets are kept only
  if, in order:
  1. **Validity:** every decoded checkpoint parses.
  2. **Final answer:** the decoded final answer is correct.
  3. **Self-consistency:** at every position, the chained decode equals the decode from a k = 1 restart at that
     position's previous checkpoint.
  - Otherwise only the final slot is targeted (as §103).
* **Label-free precision gate.** Among final-correct problems, the self-consistency rate must be ≥ 0.995. If it is not,
  an intermediate length is inserted and the pool is relabelled, **at most 2 insertions per round** (Literature (3)).
  - **The gate is a necessary condition, not an estimate.** Chain and restart share weights, so agreement
    over-estimates precision.
  - The agreement-vs-truth gap is logged per round. Truth-scored precision is computed by the scorer only and never
    used.
* **Final relabelling:** the last 4k updates relabel a fresh pool at the maximum length with the current model, and
  train on it.

**Measures.** §97's (cumulative at 512 / 2,048, removal, per-slot, validity, edit IIA), plus:
* §104's donor protocol on the trained models (primary C1);
* per round: the yield of each filter, the gate value, and the truth-scored precision (descriptive);
* short_fraction (≤ 0.05).
* The op distribution of kept vs all problems per round (selection bias), and the median and range over the 5 seeds.

**Rules (fixed now).**
* **SELF2 (C4):** cumulative ≥ 0.99 at 512 in ≥ 4/5 seeds.
* **C1 (§104 / §105):** in the SELF2 seeds, mean-removal ≤ 0.06, and donor USE (≥ 0.90) at every in-chunk slot with no
  BYPASS (≥ 0.10), in ≥ 4/5.
* **C6:** SELF2 at 2,048 (≥ 0.99) in ≥ 4/5.
* **Claim wording (fixed):** "C4 qualified by ≤ 5 % short problems (final answers only)".

**Readings.**
* **SELF2 + C1.** A pretrained LM learns S5 latent restarts from final answers alone, with its latents causally carrying
  the state: C4 (qualified) + C1 + C2 (restarts) + C6 if it holds at 2,048.
* **SELF2 without C1.** C4 holds, but the masked fork's slot still recomputes in-pass (§105's concern). Report the donor
  split.
* **Not SELF2, with high gate values.** Coverage or optimisation is the limit, not precision.
* **The gate never passes** (pre-registered outcome). Self-consistency is too strict, or the views disagree benignly,
  at this budget. Report the yields.
* **Conditional STUDENT arm (fixed now; Literature (2), confirmation bias).** If SELF2 fails but ≥ 3/5 seeds reach ≥ 0.90
  at 512:
  - each seed's final filtered relabelled pool trains a **fresh** GEN-4-MASK model from pretrained GPT-2, 24k steps,
    under the same rules;
  - this restarts from the original model, as STaR does. A STUDENT pass means the self-trained model's labels were good
    and its own training had locked in errors;
  - 5 runs, about $8.

**Placement.** 5 seeds, independent, about $8–10 on Vast (proof seed first). Projection to Audit. Implementation: the
experiments session (self103 with the masked fork, the self-consistency filter and gate, and independent pools).

**Note 110-1 (2026-10-04 09:28 EDT; the experiments session's round-1 report on seeds 1–3, disclosed; a design error,
Theory's).**
* **What happened.**
  - At round 1 (L = 4, then an inserted L = 3), the label-free gate is ≈ 0 in all three seeds: self-consistency among
    final-correct problems is 0.0017 / 0.0 / 0.0041, and kept problems are 0 / 0 / 2.
  - Yet the truth-scored rate of fully true chains among final-correct problems is 0.36 / 0.32 / 0.43.
  - The code check passes: the same filter keeps 120/120 on g105, which was trained from arbitrary starts.
* **Cause (design error).** The self-consistency view restarts at k = 1 from the model's own checkpoint text at a
  **non-identity** state. But every §110 training problem starts from the **identity** (as §103), so restart
  competence from other states is never trained early. The restart view fails even where the chain is right, and the
  gate cannot pass.
  - This is the pre-registered "gate never passes" outcome, but its cause is the design, not the method's idea.
* **Handling.**
  - Seeds 1–3 continue unchanged to completion, for the record, and are scored as registered. They now train on
    final-only targets in most rounds.
  - **Seeds 4–5 are not run** (held). The mechanism is identified and they would add nothing.
* **Corrected design to follow as §110b** (after a Literature check). Training problems start from **random given start
  states**, written as text. The start state is problem input, not an intermediate label, so supervision stays
  final-answer only.
  - The model then learns restarts from round 0, and the k = 1 restart becomes a trained second view.
  - Evaluation stays on the protocol's identity-start test set.

## 110b. Self-training by length, v2 corrected: random given start states (C4 + C1) (pre-registered 2026-10-04 09:29 EDT, after Literature's check e9b41709 and before any implementation or run; corrects §110's design error, note 110-1)

**Why (110-1).** §110's restart check needs restart competence from non-identity states, which identity-only training
never provides (its gate was ≈ 0). §110b gives every training problem a **random given start state, written as text**.
The start is problem input, so supervision stays final-answer only, and the k = 1 restart becomes a trained second view
from round 0. Otherwise §110 is unchanged.

**§110's motivation.** §103's filtered self-training (S5, final answers only, ≤ 5 % short) reached 0 / 0.9615 / 0.9585 at 512. It was
the closest in-LM C4 result.
* The cap was **pseudo-label precision ≈ 0.98**. The cross-seed vote on a shared pool lets correlated errors through.
* s2's mean-removal (0.949) shows the fork bypass, so there is no C1.
* §110 targets both: per-seed, label-free filtering with a precision gate, and the masked fork.
* **Prior** (e9b41709): ...

**Task and model (Literature e9b41709).** S5, final answers only (no ground-truth intermediate state is ever used in
training or in any training decision).
* **Short problems (R0, lengths 1–2, ≤ 5 % of updates) start from the identity**, exactly as §103. Their answers label only
  e · a and e · a · b, the same footholds as §103 (Literature: random-start shorts would label Cayley entries across the
  whole state space).
* **Problems of length ≥ 3 start from a random given state, drawn uniformly from the 120 states and written as text** in
  the same format as the identity start. Their final answers leak no step labels, and they train restarts from
  arbitrary states.
* The **test stays on DeltaProduct's identity-start protocol set**, with the identity start written as text in the same
  format, so it is in distribution.
* §97's GEN-4 recipe with **§105's masked fork**. Budget 24k steps. **Seeds 1–5, each with an independent pool**, no
  cross-seed vote. §97's GEN-4 recipe with
**§105's masked fork** (each step fork sees only its slot's K/V and its own tokens). Budget 24k steps. **Seeds 1–5, each
with an independent pool** and no cross-seed vote.

**Rounds.**
* **R0:** as §103 (lengths 1–2, ≤ 5 % of updates; final-answer slots only).
* **R1…:** lengths ≤ 4, 8, 16, 32 (3..L_max), with §103's adaptive insertion.
* Per round, a fresh 50k pool is labelled with the seed's own greedy decodes. A problem's per-slot targets are kept only
  if, in order:
  1. **Validity:** every decoded checkpoint parses.
  2. **Final answer:** the decoded final answer is correct.
  3. **Self-consistency:** at every position, the chained decode equals the decode from a k = 1 restart at that
     position's previous checkpoint.
  - Otherwise only the final slot is targeted (as §103).
* **Label-free precision gate.** Among final-correct problems, the self-consistency rate must be ≥ 0.995. If it is not,
  the round's length is halved (an intermediate length is inserted) and the pool is relabelled.
  - The truth-scored pseudo-label precision is logged by the scorer only, never used.
* **Final relabelling:** the last 4k updates relabel a fresh pool at the maximum length with the current model, and
  train on it.

**Measures.** §97's (cumulative at 512 / 2,048, removal, per-slot, validity, edit IIA), plus:
* §104's donor protocol on the trained models (primary C1);
* per round: the yield of each filter, the gate value, and the truth-scored precision (descriptive);
* short_fraction (≤ 0.05).

**Rules (fixed now).**
* **SELF2 (C4):** cumulative ≥ 0.99 at 512 in ≥ 4/5 seeds.
* **C1 (§104 / §105):** in the SELF2 seeds, mean-removal ≤ 0.06, and donor USE (≥ 0.90) at every in-chunk slot with no
  BYPASS (≥ 0.10), in ≥ 4/5.
* **C6:** SELF2 at 2,048 (≥ 0.99) in ≥ 4/5.
* **Claim wording (fixed; Audit, Literature):** "C4 qualified by ≤ 5 % identity-start short problems, with random textual
  starts (uniform over the 120 states) for problems of length ≥ 3. Every state appears as text in the inputs, so the
  state vocabulary is input-grounded."
* Reported: the fraction of distinct (state, op) transitions labelled by short problems (at most e's row plus 2-step
  compositions from e).

**Readings.**
* **SELF2 + C1.** A pretrained LM learns S5 latent restarts from final answers alone, with its latents causally carrying
  the state: C4 (qualified) + C1 + C2 (restarts) + C6 if it holds at 2,048.
* **SELF2 without C1.** C4 holds, but the masked fork's slot still recomputes in-pass (§105's concern). Report the donor
  split.
* **Not SELF2, with high gate values.** Coverage or optimisation is the limit, not precision.
* **The gate never passes.** Self-consistency is too strict at this budget. Report the yields.

**Placement.** 5 seeds, independent, about $8–10 on Vast (proof seed first). Projection to Audit. Implementation: the
experiments session (self103 with the masked fork, the self-consistency filter and gate, and independent pools).

**Note 110b-1 (2026-10-04 09:52 EDT; before any §110b result; on the experiments session's implementation fea7ca46).**
* **What a "short problem" labels in S5, stated explicitly.** An identity-start length-2 problem (e; a, b) has the final
  answer a · b, which is exactly the transition of state a under op b. Since a ranges over all 120 states, length-2
  problems **label Cayley-table entries from every state**. The R0 log's `short_transition_coverage` reads, in the
  smoke, 0.008 direct (e's row, from length-1 problems) and **0.79 including the 2-step pairs** (of 14,400).
* **Consequence for the C4 wording (§103, §110, §110b).** "≤ 5 % short problems" means ≤ 5 % of training problems are
  Cayley-table examples: §85's short-label fraction p, the same qualifier as §96's teacher.
  - Each result line carries the measured transition coverage next to p. Strict C4 would exclude them, as ge3 does,
    which §85 / §87 showed needs a basin-entry device.
  - Audit's earlier concern about random-start shorts applies in kind to identity-start length-2 problems too. The
    difference is only coverage (row e plus pairs vs every (s, a)).
* The test and rules are unchanged. The implementation (R0 identity-start shorts; uniform random textual starts for
  length ≥ 3; the test on identity starts) matches §110b.

## 111. Reliability of the Markov hand-off: twice the budget, and label-free best-of-n (pre-registered 2026-10-04 12:17 EDT, after Literature's check f41bb0bd and before any §111 run; C6 for §108's C1 design; Audit's steward pick)

**Why.** §108 MK-4 (S5, Markov segments, forced carriage) shows **unqualified serial C1 in one seed**: s1 is exact to
2,048, donor-consistent 1.0 at every slot. But it is a seed lottery.
* s3 carries serially, but its per-hand-off accuracy of 0.99845 compounds to 0.59 at 512.
* s2 is stuck at chance.
* C6 is now the binding constraint for this design. §91 found a composition lottery was a budget effect (4× steps gave
  5/5).
* **Prior** (f41bb0bd, `docs/LIT_S111_RELIABILITY_2026-10-04.md`).
  - **Budget fixes timing lotteries, not basin ones.** Timing: grokking (Power; Omnigrok; Nanda) and our §91. Basin:
    Zhang 2607.20594 (fixed by an operator-first curriculum), our §87 (a warm start), Kuo 2606.29983.
  - s2 (at chance) looks like a basin; s3 (1.5e-3 per hand-off) looks like timing or precision.
  - Compounding: DAgger 1011.0686 (training on the learner's own visited states cuts compounding from about O(T²ε) to
    about O(Tε)); scheduled sampling.
  - Best-of-n: Dodge 1909.03004 (report a curve over n).

**Arms (seeds 6–10 each, fresh; §108 MK-4 otherwise, including the sink fix).**
* **MK-4-2× (the timing lever; also a longer basin warm start):** 48k steps, with the schedule shape scaled ×2 (K-CURR
  stages at 4k / 12k; TF ρ → 0 at 24k; eligibility from then on).
* **MK-4-DAG (the precision / compounding lever; DAgger-style):** 24k steps as §108. After ρ = 0 (12k), half of the
  training sequences are **two chunks (8 ops)**: the second chunk starts from the model's **own** greedy-decoded
  checkpoint of chunk 1 (its own snap), and its segments receive the model's own latents. The model is trained on its
  own visited states, so compounding errors appear in training.

**Pre-registered selection analysis (a), no new runs.** Over every MK-4 run (§108 s1–5, and each §111 arm's s6–10),
select by a **training-data-only** criterion.
* **Primary criterion:** the per-hand-off error on 20k long rollouts (512 ops, snaps every 4) built from the
  **training-split** sequences (a P7 analogue; Literature (2): final loss would not separate s1 from s3).
* **Secondary:** final training loss.
* Report whether the selected run per group of 5 passes ACC at 512.
* Report each criterion's AUC for picking passing runs, and a best-of-n curve over n (Dodge).
* An honest recipe-level reading: "train n, keep the lowest-loss seed". It is reported as such, separately from per-seed
  reliability.

**Rules (fixed now).**
* **REL (C6):** MK-4-2× cumulative ≥ 0.99 at 512 in ≥ 4/5 seeds, with §104's donor SERIAL (USE ≥ 0.90 at every in-chunk
  slot, no BYPASS) in those seeds.
* **SEL:** the selection criterion picks a passing run in each group of 5 (§108's s1–5; §111's s6–10).
* Per-hand-off accuracy is reported; C6 at 2,048 (≥ 0.99) as §108.
* **REL is applied to each arm separately.** The comparison of 2× against DAG says whether the lottery is timing or
  compounding.
* **Descriptive numerics check:** §108 MK-4 s3 re-evaluated with an fp32 carry (Literature (3)).

**Readings.**
* **REL.** The Markov hand-off's lottery was budget, so unqualified serial C1 on S5 is reliable: C1 + C6, with C2 to
  2,048 if it holds.
* **Not REL but SEL.** Per-seed reliability is not reached, but a label-free selection recipe finds a working model.
* **Only MK-4-DAG passes REL.** Compounding from its own errors was the issue: training on its own states fixes it.
* **Only MK-4-2× passes.** It was timing or the length of the warm start.
* **Neither.** The lottery is not budget, and not exposure to its own states. Per-hand-off error correction (Audit's
  (b): soft snap plus noise between segments, informed by §101) is next.

**Placement.** 10 runs (5 at 2× length plus 5 DAG), about $14–16 on Vast. Proof host first. Projection to Audit.

**Note 111-1 (2026-10-04 12:20 EDT, before any §111 run; Audit's diagnosis of §108's failing seeds from the raw logs).**
* **Diagnosis (descriptive).**
  - The failing seeds' training loss follows a **null curve**, (k − 1)/k × (1 − ρ) × ln 120: every latent-fed slot is at
    chance. s2 follows it from 3k to 24k (3.60–3.69 against 3.59 after 12k), and s4 is on it so far.
  - **The latent is never read**, a total no-read basin; ρ's anneal only exposes it.
  - Escapers leave the curve early: s1's loss/null ratio is 0.37 at 3.6k; s3 escapes only after k = 4 starts (0.21 at
    8k).
  - **Mechanism:** during k = 1 the fork sees its own segment [EOT, start text, op 1, s_in], decodes state 1 from the
    visible text, and puts no pressure on s_in. Pressure arrives only at k ≥ 2, on the (1 − ρ) share. Writer and reader
    are stuck in a chicken-and-egg.
* **Amendments.**
  - **MK-4-2× is un-confounded (Audit (d)):** K-CURR and ρ as §108 (ρ → 0 at 12k); only the total budget doubles to
    48k. It isolates timing from warm-start length.
  - **New arm MK-4-READ (seeds 6–10):** §108 MK-4 plus an **auxiliary reader** at every slot from step 0 (k = 1
    included).
    - A fresh segment **[EOT, carried latent]**, exactly what the next segment sees before its op, decodes state j's
      step text (CE, weight 1) through the tied head.
    - The carried vector must then hold state j, and a reader of it is trained together with the writer, from the
      first step, with no text bypass.
    - Prior: SIM-CoT's latent-conditioned auxiliary decoder (78a12af7); slot-local auxiliary targets (042f94a2,
      2510.00184).
  - **Descriptive:** the loss/null ratio per seed at every log.
  - **The selection analysis adds Audit's early label-free criterion:** the ratio < 0.9 within 1k steps of k = 2
    starting. Its AUC is reported next to train_handoff_err and final loss.
* **Rules unchanged (REL per arm, SEL).** It is now 15 runs, about $22. The scorer is extended (arm `mk4_read`).

## 112. How small can the foothold be? A short-coverage dose-response for §110's self-training (pre-registered 2026-10-04 15:00 EDT, after Literature's check b1352c3e and before any implementation or run; Audit's steward idea)

**Why.** §110 s3, whose short problems label 71.5 % of S5's transitions, is correct on **all** 14,400 k = 1
transitions (110-5): the shorts acted as a foothold. §112 measures how small the foothold can be. Fewer labelled
transitions means closer to strict C4.
* **Prior** (b1352c3e): our §85 (dose-response for the free table: 0.1 → 5/5, 0.03 → 2/5) and §87 (basin entry); Wies
  2204.02892; Abbe 2306.16921. Contrasts: grokking's critical fraction, Lee 2606.07254, §76(B).

**Design.** §110's recipe unchanged (identity starts, masked fork, independent pools, validity → final →
self-consistency filters with the label-free gate, final relabel, 24k steps), except that R0's short problems are
restricted to a **fixed random subset T_f of transitions**.
* **T_f** is stratified by op: for every op b, a random fraction f of the 120 states s gives (s, b) ∈ T_f. It is fixed
  per seed, so no op is left without shorts (Literature (1)).
* A length-1 short (e; a) is allowed iff (e, a) ∈ T_f. A length-2 short (e; a, b) is allowed iff (e, a) ∈ T_f **and**
  (a, b) ∈ T_f (Literature (2)).
* The R0 short-problem count is held fixed (1,200 updates × 64; Literature (3)). Only which transitions they cover
  changes.
* **Doses:** f ∈ {0.5, 0.25, 0.1}, **5 seeds each** (s21–25). §110 itself is the f ≈ 0.715 reference (natural coverage).

**Measures.**
* **Primary:** k = 1 accuracy on transitions **never labelled by any short**. Split by whether the transition appears
  inside the self-trained pools' kept problems (Literature (4)).
* k = 1 accuracy on labelled transitions; the full certificate (all 14,400 correct); cumulative at 512 / 2,048;
  donor C1 (§104); the realised coverage.

**Rule (fixed now; per dose).**
* **FOOTHOLD-f:** ≥ 2/5 seeds with **cumulative ≥ 0.99 at 512** and **k = 1 accuracy ≥ 0.99 on never-labelled
  transitions**.
  - Existence at a dose is the dose criterion; reliability (C6) is separate, given §110's 1/3 seed lottery.
* **Primary reading:** the smallest f meeting FOOTHOLD-f. Certified seeds (all 14,400) are reported per dose.

**Readings.**
* **FOOTHOLD at f = 0.1.** A ~10 % short foothold suffices: the LM learns ~90 % of the transition table from long
  final answers. That is near-strict C4, with the qualifier stated as coverage.
* **Only at 0.5 or 0.25.** There is a coverage threshold; it is the measurement.
* **None.** §110's success needs near-natural coverage. Report the per-dose never-labelled accuracy.
* **Tags:** C4 (qualified by coverage f), C1 (donor), C2 (2,048), C6 per-dose seed rate.

**Placement.** 15 runs (3 doses × 5 seeds), about $8–10 on Vast. Proof seed first. Projection to Audit.

**Note 112-1 (2026-10-04 15:01 EDT, before any §112 run; Audit's arithmetic).**
* **The problem.** Every short starts at e. Stratifying T_f by op over all states puts (e, a) in T_f for only about f ·
  120 ops, so the reachable states shrink too.
  - Realised coverage ≈ **f² + f/120**: about 25 / 6.3 / 1.1 % for f = 0.5 / 0.25 / 0.1.
  - At low f, most first steps also lose their foothold, unlike §110.
* **Fix (Audit's option (b), adopted).**
  - **All 120 first-step transitions (e, a) are always in T_f**, so every first step keeps its foothold, as in §110.
  - The **inner transitions** (a, b), a ≠ e, are drawn stratified by op (for each b, a random fraction g of the
    non-identity states). g is set so the **realised coverage ≈ f**: g = (f · 14,400 − 120) / (14,400 − 120).
  - Length-1 shorts are always allowed. A length-2 short (e; a, b) is allowed iff (a, b) ∈ T_f.
* **The realised coverage is computed when T_f is built and logged** in args.json and eval.json (`coverage_target`,
  `coverage_realised`). The doses are reported by realised coverage.
* **Consistency:** §110's natural 71.5 % (a full row e plus randomly sampled inner pairs) corresponds to (b) at f ≈ 0.715.
* **Addendum (Audit).** The **reported dose and the "never labelled" set are defined by the R0 shorts actually drawn**
  (the 1,200 × 64 draws, as in `theory/s110_unlabelled.py`), not by |T_f|. §110's 71.5 % came from incomplete sampling,
  not a restriction. At f = 0.5, about 38k length-2 draws over about 7.1k transitions should nearly saturate T_f; at
  lower doses the pool size and mix decide it. Both `coverage_target` (|T_f|) and `coverage_drawn` are logged; the drawn
  value is the dose.
* The doses (f = 0.5 / 0.25 / 0.1), seeds, rules and scorer are otherwise unchanged.

## 113. The auxiliary reader inside self-training: §110 plus a reader on the model's own decodes (pre-registered 2026-10-04 15:00 EDT, after Literature's check b1352c3e and before any implementation or run; conditional on §111 MK-4-READ meeting REL)

**Why.** §110's seed failures (s1: a slot-1 bypass; s2: never acquires) look like 111-1's no-read family. If §111's
auxiliary reader makes §108's Markov hand-off reliable, the same reader should help §110.
* **Prior** (b1352c3e): SELR 2608.13570, ReLAT 2606.06252, 2606.20075, CODI, cycle consistency. New as scoped.

**Design.** §110's recipe, plus §111-READ's auxiliary reader at every slot from step 0.
* A fresh segment **[EOT, carried latent]** decodes the slot's state text.
* Its target is **the model's own filtered decode**: the per-slot pseudo-label of kept problems, and the final-answer
  slot of R0 shorts and of discarded problems.
* The **final-answer filter sits in front of the reader loss** (Literature: confirmation bias, 1908.02983). No ground-
  truth intermediate state is used, so C4 holds.
* Seeds 11–15.

**Rules.** §110's: SELF2 ≥ 4/5, donor C1 ≥ 4/5, C6 at 2,048. **REL here = SELF2 and C1 at ≥ 4/5.** Readability is not
use (Literature): C1 is scored with the donor protocol only.

**Readings.**
* **REL.** The reader turns §110's lottery into a reliable recipe: C4 (qualified) + C1 + C2 + C6 together.
* **Not REL.** The no-read account does not cover §110's failures.

**Placement.** 5 runs, about $2–4, **launched only if §111 MK-4-READ meets REL**. Otherwise it is not run (noted then).

## Notes on archived sections (append new notes here; the full sections are in the archive)

**Note 94-3 (§94; 2026-10-03 23:55 EDT; Step 0 probes on the 13 new §94 models, run by the experiments session, settings as
2e559168, `theory/results/s94_step0/f94_*.json`; descriptive, availability only per 94-1).**
* **STD-DEC-3× and STD-DEC-INV-3×, s6–10 (10 models):** letter → box and box → letter ≈ 1.000 / 1.000 at the final
  latent, for "where" and "all". The minimum all-4 over all positions is ≥ 0.998 in all 10.
* **FADE-STD-3× (no step decoder):**
  - s6: 1.000 / 1.000;
  - s7: 0.36 / 0.03;
  - s8: 0.25 / 0.01, chance-level availability on both question types.
* **Reading.** With the step decoder, the state is linearly available in every seed. Without it, availability is
  seed-dependent (1/3), and that tracks the no-decoder arm's behaviour (s6 is the one seed that answers both types).
  - This supports keeping the side step decoder in the §97 recipe.

**Note C4-coverage (2026-10-04 09:54 EDT; Audit's request after 110b-1; exact replays of the samplers,
`theory/c4_coverage.py`, `theory/results/c4_coverage.json`; descriptive).** This is the share of the 14,400 S5
(state, op) transitions labelled by short (length 1–2) problems in each C4-qualified result.
* **§96 teacher** (mixP 0.05, M 240, 16k updates × 256): **99.87 / 99.90 / 99.94 / 99.93 / 99.93 %** (seeds 2161–2165).
  Essentially the full Cayley table, from length-2 problems; row e is complete from length 1.
* **§103 R0** (identity-start lengths 1–2, 1,200 updates × 64): the 50k pool covers **78.2 %**. Each seed's R0 batches
  covered **71.1 / 71.1 / 71.3 %**.
* **§110b R0:** logged by its implementation (`short_transition_coverage`); 0.79 in the smoke.
* **Wording.** Every C4-qualified claim line carries its number:
  - "§96: C4 for the pipeline, qualified by p = 0.05 short problems covering ≈ 99.9 % of transitions, and oracle M";
  - "§103: ≤ 5 % short problems covering ≈ 71 % of transitions".
  - **Strict C4** (lengths ≥ 3 only, as ge3) is **unmet**. Per §85, that route needs an optimisation advance (a way into
    the basin), not self-training.

**Note 110b-2 (2026-10-04 11:35 EDT; §110b stopped after a clear null, Theory's decision; data kept and committed as
"aborted after a clear null").**
* **Result.** Seeds 1–3 (Vast) and s4 (local), through 13.2k of 24k steps, rounds L3 → L4 → L5 (inserted) → L6:
  - final-correct at every labelling is **0.0074–0.0091**, which is chance (1/120 = 0.0083);
  - the gate is ≤ 0.02, nothing is kept, and the truth-scored rate is 0;
  - **training loss is flat at 4.86–4.92 since R0 (ln 120 = 4.79)**: the final state is predicted at chance.
  - Label checks pass: the g105 control with random starts gives truth precision 1.0.
* **Reading.** From **random given starts**, final-answer-only training on problems of length ≥ 3 does not get off the
  ground at this budget. That is §85's strict-C4 barrier (ge3: no short labels per state) appearing in the LM.
  - §110 / §103's identity-start rounds progressed because identity-start problems are **prefixes of the 100k training
    rows** and can be memorised (§110 reached loss 1e-5 at L4; the experiments session's contrast). Random starts
    remove that by 120×.
  - **This qualifies §103's near-miss:** its bootstrap leaned on identity-start structure, and possibly on prefix
    memorisation, before per-op pseudo-labels took over at L ≥ 6.
* **Consequence.**
  - Self-training alone is not a route to strict C4 here (Audit's morning-summary point).
  - The in-LM route needs a way into the basin (§85 / §87: short-label warm start, oracle-like initialisation) or the
    amortised teacher (§96).
  - §110 (identity starts) s1–3 complete for the record.

**Note 110-3 (2026-10-04 14:31 EDT; §110 s1–3 finished, fetched e394f62d; scored with `theory/score_s110.py`; s4–5 now
running). It corrects 110-1.**
* **Correction to 110-1.** "The gate cannot pass" was wrong. Under identity-start training, **restart competence emerged
  from the final-only rounds at L ≥ 5**, which start from the model's own checkpoint texts.
  - s3's gate went from ≈ 0 (L4 / 3 / 4 / 8 / 6 / 5) to **1.0 at L6** (kept 0.754), then 1.0 with kept ≥ 0.9998 at L8 /
    16 / 32 and in the final round.
  - s1's gate rose more slowly (0.28 → 0.71–0.88 → 0.97). s2's stayed ≈ 0.
  - Seeds 4–5, held under 110-1, are now run to restore the registered panel. They also decide STUDENT's trigger.
* **Interim results (s1–3).**

  | seed | cumulative 512 / 2,048 | donor-consistent, slots 1 / 2 / 3 | recipient-consistent | mean-removal |
  |---|---|---|---|---|
  | **s3** | **0.999 / 0.994** | **0.962 / 0.983 / 0.968** | 0.012 / 0.011 / 0.009 | 0.011 |
  | s1 | 0.9505 / 0.826 | slot 1 recipient **0.9995** (bypass); slots 2–3 neither | — | 0.016 |
  | s2 | 0 / 0 | chance | — | — |

  - **s3 individually meets SELF2 and §110's donor C1 rule.** A pretrained LM learns S5 latent restarts **from final
    answers only** (≤ 5 % identity-start short problems, about 71–79 % transition coverage; note C4-coverage). Its
    latents carry the state at every in-chunk slot (masked fork, donor-consistent ≥ 0.96), exact to 2,048.
  - That is one seed: SELF2 and C1 need ≥ 4/5, and are at 1/3 now (at most 3/5).
  - Unlike §105 (supervised, masked fork), where slot 1 was recomputed from the chunk start (donor 0.13–0.87), s3 also
    carries slot 1 (0.962).
* **Reading (interim).** C4 (qualified) + C1 (masked fork, donor) + C2 to 2,048 are shown together **in one seed**. It is
  the same reliability picture as §108: the mechanism exists, and seeds lottery.

**Note 110-4 (2026-10-04 14:32 EDT; Audit's caveats on 110-3, for the write-up).**
* **Resumed runs, disclosed.** s1 and s3 were OOM-killed and resumed from `ckpt.pt` at step 4960 (after the RNG-device
  fix). `self110.py`'s checkpoint restores the model, the optimiser, the numpy pool and draw RNGs, the torch and CUDA RNG
  states, and the round state (step, rounds, pending lengths, remaining budget). They are therefore not uninterrupted
  runs, but the resume is state-complete.
* **s3's own short-problem coverage** (an exact replay of `self110.py`'s R0 sampler and draws): **71.5 %** of the 14,400
  transitions labelled by R0's short problems (s1 71.4 %, s2 71.9 %). s3's claim line: "C4 qualified by ≤ 5 %
  identity-start short problems covering 71.5 % of transitions".
* **The abrupt switch.** s3's gate went from 0 (L5) to 1.0 (L6), and accuracy at the new length went 0.67 → 0.75 →
  0.9998. Whether s4 / s5 show the same switch, and at what length, is logged when they land (`rounds`: gate,
  acc_new_length_before).
* **Filter audit (Audit, read in `self110.label_round`).** Keep = valid ∧ final == truth ∧ the restart self-consistency;
  the targets are the model's own decodes. Truth enters only through the final answer and logged statistics, so no
  intermediate label is used.

**Note 110-5 (2026-10-04 14:58 EDT; descriptive; Audit's IDEA 1 first step; `theory/s110_unlabelled.py` on CPU,
`theory/results/s110_unlabelled_s3.json`).**
* **Test.** §110 s3's model, with a k = 1 restart from each state's text (masked fork, greedy), on all 14,400 (state, op)
  transitions. They are split by whether s3's R0 short problems labelled them (an exact replay of `self110.py`'s
  sampler: 71.5 % labelled).
* **Result.**
  - Accuracy is **1.000 on labelled and 1.000 on unlabelled transitions**: 0 errors among the 4,105 transitions no short
    problem ever labelled.
  - The k = 1 restart system is a finite automaton over the 120 written states. Correct on every input, it is
    **certified at every length** (as §92-1's certificate), here **from final-answer supervision**.
* **Reading.** s3 learned the 28.5 % of transitions its shorts never labelled from the **final answers of longer problems
  alone**: it composes final-answer supervision into step knowledge. The short problems acted as a **foothold, not as
  labels**.
  - This is the first in-LM evidence pointing toward strict C4. It motivates the coverage dose-response (Audit's
    IDEA 1; registration next).

**110-5 wording corrected (2026-10-04 15:00 EDT; Literature b1352c3e).** The 4,105 transitions never labelled by a short
problem were **not unsupervised**. They occur inside long problems whose final answers, plus the filtered self-training
targets, constrain them.
* The correct statement: "**transitions learned from long-problem final answers, given a partial (71.5 %) short
  foothold**". That is a C4-direction result.
* It is **not** held-out-pair generalisation in the §76(B) / grokking sense, where held-out entries get no signal at all.
