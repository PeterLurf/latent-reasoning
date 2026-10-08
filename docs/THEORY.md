# Theory: latent reasoning (condensed)

The Theory session's registrations, results and notes. **§0–§96 are in full, verbatim, in
[`docs/archive/THEORY_s0-s96.md`](archive/THEORY_s0-s96.md)** (moved 2026-10-03 at the user's request; every note and result is
filed under its own §; search "## N." or "Note N-"). §97 onward are below in full. New notes on archived sections are
appended at the end of this file.

**Research priority (the user, 2026-10-04 ~00:45): C1 > C6 > C4 = C2 > C3 > C5 > C7.** Favour designs where the state
can travel only through the latents (MASK, MARKOV), with removal and bypass controls on every arm. Accuracy with failed
removal is the headline. 5 seeds and long-rollout drift for any passing arm (C6). Train short, test long on every new
task (C2). C3 / C5 / C7 are reported where cheap, not designed for.

## Where things stand (2026-10-05, afternoon)

* **North star:** one pretrained LM meeting C1–C7 (`docs/NORTH_STAR.md`).
* **Serial C1 + C6 on S5 (§108, §111).**
  - Markov segments (the latent is the only route) give unqualified serial C1 in every seed that learns.
  - The auxiliary reader removes the no-read basin: **5/5 exact at 512 with single-slot donor SERIAL, 4/5 at 2,048**
    (111-4).
  - Per-step supervised, so not C4. Twice the budget does not help (0/5, 111-3).
* **C4 by final-answer training through self-decoded restarts (§110, §112).**
  - §110 s3 (71.5 % short coverage) and §112 s21 (10 %) are certified on all 14,400 transitions.
  - Mechanism: the history cut turns long final-answer problems into short problems that start from the model's own
    checkpoint (112-2).
  - Under the masked fork the carrier is seed-dependent (112-3). Reliability is open.
* **No snaps at inference (§118, §115).** With a text-path anchor in training, 5/5 S5 seeds run fully latent and exact to
  2,048 with no snaps, serially (C1 + C2 + C6; text-anchored, not task-agnostic). Success tracks the hand-off's
  contraction ρ (pass ≤ 0.24, fail ≥ 0.35; 118-5).
* **Task-agnostic line (the user's directive; §120, §123, §125).** One harness (a Markov segment per item, an 8-d
  bottleneck, an answer decoder) on S5 and flip-flop.
  - The continuous bottleneck alone is exact to 2,048 on S5 (123-6). The learned discrete snap adds certification:
    flip-flop machines are certified at every length (minimal 2 states); S5's are open.
  - The reliability blocker is a soft-phase no-read basin (seed-specific; 123-3). §125 (a standardised carried code)
    tests the fix.
* **Multiplication:** the carrier holds the multiplier prefix, not the running sum (114-4, causal). Width, pauses,
  looped depth and anchoring did not fix it, so the line is paused by §122's stop rule. It returns only as a family in
  the task-agnostic harness.
* **C4:** final answers with a 10 % short foothold work in 2/5 (§112). Strict C4 fails (§116 NONE at chance). The
  reliability lottery (112-4) is tested by §124.
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
* **Looped / recurrent-depth proposals** (2026-10-04, `docs/THEORY_LOOPED_2026-10-04.md`): P1 looped depth per op (vs
  PAUSE), P2 redundant state coding, P3 straight-through Gumbel snaps (strict C4), P4 round-trip-triggered snapping
  (C5), P5 adaptive depth, P6 early triage.
* **Without hand-designed snaps** (2026-10-05, `docs/THEORY_AUTOSNAP_2026-10-05.md`; the user's directive): P-A FSQ
  digital recurrence (a learned discrete snap every step, certified by code enumeration), P-B continuous stabiliser,
  P-E learned what + when, P-C learned-notation text; one harness across S5, flip-flop, cycle navigation and
  multiplication, final answers primary.
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
| 101 | Snap-strength dose-response (β_min 1 / 0.1 / 0.01 / 0) on S5 | **smallest passing β_min = 0.01** (2/3; β 0 1/3): a weak attractor suffices, "soft snap" is the honest design; failing seeds collapse the carrier norm after chunk 1; mixed GPUs stated | closed |
| 102 | State in depth: gated slot-to-slot residual path (4×4, in §99's harness) | does not accelerate (all censored; test 0.01–0.17) | closed |
| 103 | Filtered self-training by length (C4 inside the LM, S5) | SELF not met: 0 / 0.9615 / 0.9585 at 512 from final answers only; pseudo-label precision ~0.98 caps it | closed |
| 104 | Serial causal use of slots: donor patching, blocking (§92, §97; eval only) | SERIAL not met: §92 2/5; §97 0/3 with donor latents ignored (state recomputed from visible ops) | closed |
| 105 | C1 for the headline: S5 GEN-4 with a masked fork + donor protocol | G4-M + MASK-DEC met; HANDOFF 0/3 (slots 2–3 latent-carried in 2/3; slot 1 recomputed from the visible start) | closed |
| 106 | C2 on a real task: bounded streaming multiplication (latent vs the written format) | written format + cut extrapolates 1.000 to b = 32 (WC-S-k2 3/3); masked latent arm fails even at k = 2 (106-4) | closed |
| 107 | Multiplication's blocker: carrier width (m latents) vs compute (pause), carriage forced | neither: CARRY-3 0/3 in LAT-2, LAT-4 and PAUSE-4; the carrier holds the multiplier prefix (114-4) | closed |
| 108 | Forced latent carriage on S5 (Markov segments, sink fix, TF bootstrap; donor-scored) | ACC not met (MK-4 2/5, MK-8 0/5); serial in all 5 learning seeds; MK-4 s1 1.0 at 512, 0.996 at 2,048 (unqualified serial C1, one seed); failures: no-read basin or precision | closed |
| 110 | Self-training v2: independent seeds, label-free gate, masked fork (C4 + C1) | SELF2 2/5, C1 1/5 (s3: C4 + C1 + C2, certified k = 1); STUDENT 3/5 (s1's student drops the slot-1 bypass) | closed |
| 110b | Self-training v2 corrected: identity-start shorts plus random textual starts for length ≥ 3 | stopped after a clear null: loss at chance from random starts (110b-2) | closed |
| 111 | Reliability of the Markov hand-off: 2× budget vs DAgger-style own-state training; best-of-n | **READ meets REL (5/5 serial at 512, 4/5 at 2,048)**; 2× 0/5, DAG 1/5: the lottery is the no-read basin | closed |
| 112 | Short-coverage dose-response for self-training (f = 0.5 / 0.25 / 0.1; never-labelled k = 1 accuracy) | **FOOTHOLD at every dose; smallest f = 0.1** (2/5, certified); acquisition 2–3/5 at every dose (start-reading basin, 112-4); C1 0/6 | closed |
| 113 | Auxiliary reader inside self-training (conditional on §111-READ) | not REL: SELF2 2/5, C1 0/5 (slot-1 bypass persists); acquisition 2/5, as 113-1 predicted | closed |
| 114 | Looped depth per op (LOOP-R vs LOOP-1 vs §107 PAUSE) for multiplication's hand-off | DEPTH not met: CARRY-3 0/3 in every arm, acc flat in r; the carrier holds the multiplier prefix (114-4) | closed |
| 115 | No-snap Markov chains, and a round-trip snap trigger (eval only) | **NOSNAP 0/3; TRIGGER 6/6** (first-error detection 1.0, accuracy ≥ fixed k 4 at fewer snaps; miss 0): C5 label-free | closed |
| 116 | Strict C4: final-answer training through self-decoded restarts (OPDEF: op definitions only; NONE: no shorts) | registered (conditional on §112 f = 0.1) | **open** |
| 117 | One model for C1 + C4 + C6: self-decoded restarts + Markov segments + reader (MK vs staged ST) | registered (conditional on §111 READ REL) | **open** |
| 118 | Fully latent at inference: text-path anchor vs longer silent stretches (S5, on READ) | **ANCHOR 5/5 no-snap exact to 2,048, serial (C1 + C2 + C6, text-anchored)**; READ control 2/5; LONG running | **open** (LONG) |
| 119 | Controlled C1 baselines: Coconut, CODI, SIM-CoT on our S5 protocol (donor + length) | **CONTROLLED-C1 not readable:** every baseline did not learn (Coconut 3/3, SIM-CoT 3/3, CODI 3/3 at chance; text CoT learns); no competent published comparator, so a competence (C3) finding, not a C1 one | closed |
| 120 | FSQ digital recurrence: a learned discrete snap every step, no template or schedule (S5 + flip-flop pilot) | pilot not met (S5 1/2 per arm: code collapse / loss basin); **flip-flop 4/4 in both arms, 2 seeds certified (Moore 2)**; extension deferred to §123 | closed (pilot) |
| 121 | Multiplication's carrier shortcut: text-path anchor + reader vs reader alone (4×4, Markov) | CARRY-3 0/3: the anchor binds to a dead text path (T1 = 0); READ lifts op 3 to 0.64 / 0.93 by recomputation, with no sum carrier; triggers §122 | closed |
| 122 | TEXT-MIX keeps the anchor's target competent (conditional on §121; stop rule for the multiplication line) | CARRY-3 0/3 (op 3 0.32–0.41) with a competent text path (t1 1.0); **STOP RULE fired**: the multiplication latent-carriage line pauses | closed |
| 123 | Attractor strength causal for no-snap? Dither dose-response (σ 0–0.5) + dithered FSQ (canonical codes), §120 harness | **causal question not reached**: dither moved neither ρ_ff (S5 s1 0.146–0.180; FF 0.050 → 0.087) nor accuracy nor the basin (S5 s2 plateaus at every σ); continuous 8-d carry exact to 2,048 (123-6); CANONICAL not met | closed |
| 124 | Does reading a given state fix self-training's lottery? §112 f = 0.1 + copy objective vs exposure control | **descriptive** (MECH-112 and NOT-112-4 both false): COPY 4/4 acquired, all at L5; EXPOSE 4/4 (L5–L8); BASE 2/4. COPY is the only arm with serial carriage (2/4: s26, s27 meet C4-qualified + C1 + C2) | closed |
| 125 | Readable bottleneck from step 0: standardised carried code vs the soft-phase no-read basin (task-agnostic, §120 harness) | **untested**: the EMA standardisation was not in force at 2k (carried norm 35); STD s2 plateaued; other runs held | closed (untested) |
| 126 | Is the bottleneck the cause of the soft-phase basin? RAW-H vs ANNEAL (stop rule → restart-on-plateau) | **BOTTLENECK-CAUSE** (RAW-H s1–3 escape the 2k flag; every bottlenecked s2 plateaus) **without ANNEAL-INHERITS** (s2 0 in both casts): the wide carrier enters but doesn't learn long chains (0.846 / 0 / 0); restart-on-plateau is the procedure; s4 descriptive | closed |
| 127 | Reliable entry without restarts? VAR / SKIP / STD2 / LR10 on §120's harness vs restart-on-plateau | **no arm adopted** (ENTRY-FIX false in all four): escaping the 2k flag is easy, learning afterwards fails (escape-without-learning dominates → §131); SKIP 2/4 exact vs plain 1/4 (descriptive); restart-on-plateau stays | closed |
| 128 | One recipe, five state-tracking families (S5, FF + held-out CYC / MOD / RDFA), restart-on-plateau; FAMILY-PASS = 3/4 exact + SERIAL (+ CLOSED) | **TASK-AGNOSTIC = true**: all five families pass; held-out CYC / MOD / RDFA 4/4 exact, SERIAL, CLOSED with 0 restarts; S5 3/4 (one escape without learning); fp32 certificate covers all visited codes | closed |
| 129 | Where does the entry basin live? Head-only re-draw at the 2k flag vs a CLOCK-ONLY warm restart (S5, 4 flagged seeds): LR / HEAD / BODY | **descriptive** (none of LR / HEAD / BODY): a fresh head escapes in 2/4 vs 0/4 for the warm restart, but no re-drawn seed learns; the trained body reads a fresh head (R²_state 0.4–0.96 by clock 700) | closed |
| 130 | Early label-free plateau marker (code variance at ~1k): fit on 4060 replays, frozen, tested cross-card (3070, primary) and same-card | **no PASS** (primary P 0.70 / R 0.78 cross-card, 0.47 / 0.88 same-card; 0.60 in-sample): both fail, so the marker is noisy, not card-specific: the early marker does not predict the 2k plateau; the collapse at ~650 is common and transient; the 2k flag fires after early pulls | closed |
| 131 | Second restart point at 4k: FLAG4 = loss(4k) ≥ 0.9·loss(2.6k) (floored), held-out precision / recall for "escape without learning" | **PASS**: 0/23 learners flagged, 8/11 escape-fails caught (recall 0.73); not deployable (8 < 10); 8-d carrier only | closed |
| 132 | Rules in context: per-problem permutation tables (explicit pairs) on §120's carrier; size ladder k4/m2 → k6/m3; PASS 3/4 exact + SERIAL on held-out tables | **rung 1 PASS 3/4** (slots 2–4 exact + SERIAL, certified; pilot 0.9865 = one unclosed table); rung 2 next | **open** |
| 133 | Can error correction make a wide latent carrier exact? CONT / FSQ / SOFT / NOISE / PIN at d 16–64 on S5, then an S5×4 capacity probe (FSQ-8 controls) | **stage 1: FSQ-64 / NOISE-64 / CONT-64 / SOFT-16 PASS**; FSQ-16 FAIL; SOFT-64 / PIN-64 dropped; stage 2 (S5×4) not yet approved | **open** (stage 2) |
| 134 | COPY-ANCHOR: label-free state-reading auxiliary (final-answer text autoencoded through the carrier) on §128's S5 recipe, no restarts; PASS 4/4 exact (plain 1/4) | **PASS on S5: 4/4 exact at 2,048 + SERIAL + ρ 0.14–0.22, no restarts** (plain 1/4); **guard FAILS on MOD** (exact + SERIAL, but not closed: 70k reachable codes); not adopted as default | closed |
| 135 | TEACHER-STATE DISTILLATION: a CoT model's hidden states (vs pretrained GPT-2 on the same text) as per-hand-off InfoNCE targets on §128's S5 carrier | **FAIL both arms (0/4)**: worse than plain; the aux pulls context into the code (135-9) | closed (stage B not run) |
| 136 | GROW (the user's idea): narrow → wide function-preserving latent growth at a loss threshold / plateau; GROW-P, COPY+GROW / STAGED / GROW-NOISE, GROW-4, NESTED-64 on S5 (S5×4 decides) | **stage 1: COPY+GROW / COPY+GROW-NOISE PASS 4/4** (growth safe; the new dims carry the state, R² 0.95–0.98); STAGED 3/4; GROW-P 1/4 (no rescue); GROW-4 0/4 (no entry at d 4); NESTED-64 does not enter (descriptive) | **open** (benefit: S5×4 with §133 stage 2) |
| 137 | Multiplication on fsq120, column-major: carried carry (FSQ-8), streamed product digits (C4 qualifier), restart slots; 4×4 → 5×5 → 4×8 | **4×4 DROP at proof** (test_acc 0.0; middle columns at chance; carry in h R² 0.69, in the code 0.24) → §140 | closed |
| 138 | FULL WIDTH (the user's trigger): §133's harness at d 768, no COPY; FULL-FSQ / FULL-NOISE / FULL-CONT (vs RAW-H) / FULL-HINGE; fails-to-improve → revisit; S5×4 as the benefit test | **all four dropped at proof** (ext 0.0; escape after the 2k flag to ≈ ln 120, like RAW-H on 3/4; the hinge never engaged) → rule (i): one revised method, §143 | closed (→ §143) |
| 139 | DISTILL-2 (user's go): a CoT teacher's state (probe or its text) as a warm-start fading to 0 by 4k, on multiplier-major multiplication; A = SOFT-16 + COPY (FINAL), B = A + DISTILL-2, C = constant weight | **stopped at proof** (139-1(c) copy gate; A′ / B s1 test_acc 0.0, B learns only step 1 at 0.59) | closed |
| 140 | DIGPAIR (§137's fallback; runs only on §137 4×4 DROP / FAIL): one digit pair per segment, carried column accumulator (FSQ-8), streamed digits; design-forced C1 | **4×4 DROP at proof** (0.002; columns 1–2 exact, first errors at the 3- and 4-pair columns; accumulator R² h 0.92 vs code 0.61) | closed |
| 141 | After §137's DROP: latent repeats per column (r4 + repeat index) × carrier (FSQ-8 / SOFT-16); r1×SOFT-16 gated on a nonlinear code probe; readings crossed with §140 | **all three cells DROP at proof** (r1s16 / r4f8 / r4s16 test 0.0; repeats change nothing, the column sum's R² ≤ 0.35 after any repeat; soft rounding slightly better) | closed |
| 142 | MULT-READ (the user's go): a per-step reader of the running sum (the teacher's S_j text) on NOISE-64, multiplier-major 4×4, 48k; then ICoT-style earliest-first removal to final-answer only (B-EARLY; B-LATE and CTRL descriptive); with the multiplication post-mortem | **DROP at proof** (A s1 0.002; S₁ 1.0, S₂ 0.35, S₃ 0.05; the gate never opened): not acquired within budget; the bounded-carrier multiplication line is parked | closed |
| 143 | FULL-GROW (§138's one revised method): §136 COPY+GROW-NOISE to 64, then 64 → 768 at ≥ 6k with μP lr on the new columns; e-space ρ; paired per seed with s136_cnoise; if it fails, full width stops | **s1 proof passes** (exact at 2,048 + SERIAL + e-space ρ 0.062; Jacobian 0.20; the new 704 dims encode the state in z, R² 0.85, but give only about 4 % of e (143-2)); slots 2–4 running | **open** |
| 144 | RUNNING-SUM (the user's "fix the multiplication"; diagnostic): sum mod 10⁴ with 2-digit items on fsq120, FSQ-8 / NOISE-64; can the carrier carry a growing number? (conditional magnitude ladder) | registered | **open** |
| 145 | MULTI-SLOT: K = 4 carried NOISE-64 slots (one write position each) on §140's digit-pair multiplication; CARRY1 control (same writes, 1 carried) if SLOT4 IMPROVES | **DROP at proof** (0.000; column 2 worse than §140, 0.31; one active slot, no digit specialisation): CARRY1 not run | closed |
| 146 | MID-PROBE ($0): layer-wise probes (0–12) of the accumulator on s140 / s137; MID-STACK SIGNAL = R² ≥ 0.5 at 100–200 in some layer ≤ 9 (top 0.05) → register a mid-stack hand-off | **NO SIGNAL** (best layer 4: 0.217 at 100–200; layers 1–12 flat; the top reproduces 140-3): the accumulator ≥ 100 is computed nowhere; B2 ranks last | closed |
| 147 | DELTA-STATE (program B1): a linear-recurrent memory (6 additive + 6 gated-delta heads) as GPT-2's only cross-segment channel, on §140's digit-pair 4×4; PASS = exact + SERIAL | **DROP at proof but IMPROVES** (test 0.743; columns 3–7 0.92–0.99 where every other design sits at about 0.1; accumulators ≥ 100 at 0.93; both head types needed; loss still falling): the first design past the middle-column wall → §151 | closed (→ §151) |
| 148 | LOOP-RETRO (program B3): prelude / weight-tied loop (blocks 4–7, damped) / coda retrofit for direct-answer 4×4; LOOP-4 vs PLAIN; C2 at r = 8 / 16 | **no improvement** (LOOP-4 0.002 = PLAIN 0.002; loops unused, acc flat in r): B3 stops at this budget; the direct baseline fails at the same middle columns as every latent design | closed |
| 149 | SOTA-PROBE (eval only): donor SERIAL (equation substitution), ablation and ρ on released GPT-2 Coconut / SIM-CoT / CODI, at release | **no model SERIAL**: SIM-CoT+Coconut PARTIAL (a third of swaps follow the donor), SIM-CoT+CODI BYPASSED (recomputed from the question); ρ marginal everywhere (0.37–0.50); the Lyapunov contrast not reproduced | closed |
| 150 | VOCAB-SLOT (program B4; the user's token-like working memory): 4 Gumbel-ST slots on GPT-2's vocabulary (FULL) or digits + blank (DIGIT, an abacus) as the only cross-segment channel, on §140's digit-pair 4×4 | registered | **open** |
| 151 | DELTA-LONG: §147's linear-recurrent memory at 48k, seeds 1–4 in parallel (after §147's IMPROVES: columns 3–7 0.92–0.99, loss still falling) | registered | **open** |
| 152 | DELTA-READ (the user's go, 23:00): DeltaMem as the only carrier on multiplier-major 4×4 + a reader of the teacher's S_j, then earliest-first removal; headline vs TC-M / ICoT-SI at matched compute (FLOPs, passes) with SERIAL | registered | **open** |

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
  - **H-starve (descriptive) is not supported:** MARKOV did not reach threshold either (both censored), and ended no
    higher than K-CURR at 24k.
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
    ≤ 0.118, against CTRL's 0.14–0.18). Every run is censored, so this is a level at 24k, not a speed claim. It is
    consistent with 111-1's no-read basin: anything that delays pressure on the latent (text first, gold inputs) leaves
    it lower at the cap.
  - **Wording (2026-10-04 23:09 EDT; Literature's project audit 2bf76954, via Audit):** §99's claims read "none reached threshold within
    24k", never "slower" or "faster".

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
* **Reading ("No").** With a direct, gated residual state path from slot j−1 to slot j, multiplication's hand-off still
  does not reach threshold within 24k (all censored; no speed claim). The staircase is not about the attention-retrieval route. Consistent with §107 / 106-2: compute per op or the
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

**§107 result (2026-10-05 02:37 EDT; 9 runs by the experiments session; scored by Theory alone with `theory/score_s107.py`, committed
before any run; `theory/results/score_s107.json`). Neither lever unblocks step 3.**

| arm (4×4, MARKOV, seeds 1–3) | op-3 decode | test answer |
|---|---|---|
| m = 1 (§99 MARKOV) | 0.253 / 0.273 / 0.236 | 0.047 / 0.041 / 0.020 |
| LAT-2 | 0.143 / 0.592 / 0.154 | 0.014 / 0.063 / 0.023 |
| LAT-4 | 0.235 / 0.316 / 0.375 | 0.042 / 0.053 / 0.023 |
| PAUSE-4 | 0.454 / 0.285 / 0.065 | 0.055 / 0.042 / 0.007 |

* **CARRY-3 is 0/3 in every arm**, so NO-BYPASS is moot. Every run reached the 24k cap without reaching threshold.
* LAT-4's latents are partly redundant (cosine 0.76–0.78), and removing any single latent leaves answers at 0.0–0.019.
* **Reading ("neither").** Neither carrier width nor extra positions per op unblocks multiplication's step-3 hand-off.
  - The registered next step was the running-sum auxiliary. 114-4 then explained why: the carrier holds the multiplier
    prefix, not the sum, so neither width nor compute can help.
  - §121 (text-path anchor + reader) targets the shortcut directly.

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

**§108 result (2026-10-04 20:59 EDT; 10 runs by the experiments session; scored by Theory alone with `theory/score_s108.py`, committed
before any run; `theory/results/score_s108.json`). ACC, C1-MK and C6 are not met in either arm. Every seed that
learns carries the state serially.**

| seed | MK-4: 512 / 2,048 | per-hand-off | donor min / recipient max | MK-8: 512 / 2,048 | per-hand-off | donor min / recipient max |
|---|---|---|---|---|---|---|
| s1 | **1.0 / 0.996** | 1.0 | 1.0 / 0.011 | 0.797 / 0.446 | 0.99941 | 0.996 / 0.011 |
| s2 | 0 / 0 | 0.008 | chance | 0 / 0 | 0.008 | chance |
| s3 | 0.59 / 0.12 | 0.99845 | 0.995 / 0.011 | 0.982 / 0.93 | 0.99995 | 0.999 / 0.010 |
| s4 | 0 / 0 | 0.008 | chance | 0 / 0 | 0.008 | chance |
| s5 | **0.994** / 0.982 | 0.99997 | 0.9995 / 0.011 | 0 / 0 | 0.674 | 0.072 / 0.011 |

* **ACC:** MK-4 2/5 (s1, s5), MK-8 0/5 (needs ≥ 4/5). **C6 at 2,048:** MK-4 1/5 (s1), MK-8 0/5.
* **SERIAL holds in all 5 seeds that learn** (MK-4 s1 / s3 / s5, MK-8 s1 / s3): donor ≥ 0.995 and recipient ≤ 0.011 at
  every in-chunk slot. That is the registered sanity check: no leak in the segment construction.
* **Two failure modes:**
  - **No-read basin** (MK-4 s2 / s4, MK-8 s2 / s4): per-hand-off at chance (111-1). MK-8 s5 is a partial reader.
  - **Precision:** MK-4 s3 and MK-8 s1 carry serially but err about once per 700–1,700 hand-offs, which compounds
    below 0.99 at 512.
* **Reading.** Forced carriage gives **unqualified serial C1 on S5 in single seeds**: MK-4 s1 reaches 0.996 at 2,048 with the
  state carried hop by hop through latents and no visible route. It is not reliable (C6). The registered next levers:
  §111 tests budget (2×: 0/5, 111-3), own-state training (DAG) and the auxiliary reader (READ: 3/3 so far).

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

**Note 111-2 (2026-10-04 16:05 EDT; §108 MK-4 s1–3 backfills by the experiments session; before any §111 result is read).**
* **`train_handoff_err`** (20k training rows, 128-op rollouts, a snap every 4):

  | seed | in-chunk error | cumulative at 128 |
  |---|---|---|
  | s1 | 1.6e-6 | 0.99985 |
  | s2 | 0.991 | 0 |
  | s3 | 1.36e-3 | 0.878 |

  - Snap hand-offs: 0 errors in all three.
  - **It ranks the seeds exactly as the test does**, supporting it as SEL's primary criterion.
  - final_train_loss: 0.0 / 3.607 / 0.0034.
* **fp32 re-evaluation of s3:** cumulative 0.875 / 0.770 / 0.59, pooled per-hand-off 0.99848, within noise of bf16.
  **Numerics are not the cause** of s3's residual error.
* **The registered early criterion (111-1) does not discriminate.** `early_escape` (the loss/null ratio < 0.9 in steps
  2,001–3,000) is True for all three; the window minima are s1 0.44, **s2 0.74**, s3 0.83.
  - At k = 2 with ρ ≈ 0.8, gold-fed segments are not exactly zero-loss early on, so the null is too small.
  - It stays reported as registered (non-discriminating).
* **Disclosed post-hoc secondary for §111's selection analysis**, designed after seeing these three seeds, so it is
  descriptive only: the **model-relative null**, the same batch with carried latents zeroed (as §114 uses).
  - The criterion is main loss / model-relative-null loss < 0.9 at the first log at or after k = 4 (6k).
  - Its hit table is reported next to `train_handoff_err`'s AUC, but it is not a pre-registered criterion.

**Note 111-3 (2026-10-04 20:57 EDT; MK-4-2× complete; scorer fix disclosed).**
* **MK-4-2× is 0/5 at 512, so REL is not met for 2×.**
  - s6, s7 and s9 are no-read: per-hand-off accuracy about 0.008, at chance.
  - s8 and s10 are partial readers: per-hand-off 0.738 and 0.123.
  - Doubling the budget does not rescue the lottery. That fits Literature's prior: budget fixes timing lotteries, not
    basins. DAG (s6–8: 0, 0.88, 0) and READ (s6–8: 3/3) are still open.
* **Scorer fix, made after the 2× results were read; no rule changed.**
  - `train_handoff_err` arrives as a dict, which crashed SEL. It is now read as `in_chunk_err`, the scalar 111-2 defines.
  - §108's `s111_backfill.json` values are merged in.
  - The fix cannot change 2×'s SEL: no 2× run passes ACC.
  - The pre-fix scorer is `theory/score_s111.py` at 4c4fc0b3. It crashed (TypeError in SEL) before writing any output,
    so there is no pre-fix score file; the crash is reproducible from that commit.

**§111 result (2026-10-05 00:35 EDT; 15 runs by the experiments session; scored by Theory alone with `theory/score_s111.py` (fix 111-3
disclosed); `theory/results/score_s111.json`). Only MK-4-READ meets REL.**

| arm (s6–10) | cumulative 512 | pooled per-hand-off | reading |
|---|---|---|---|
| **READ** | 0.9995 / 0.9945 / 1 / 1 / 1 | ≥ 0.99998 | **REL (5/5, SERIAL); 4/5 at 2,048** |
| 2× | 0 / 0 / 0 / 0 / 0 | 0.008 / 0.008 / 0.74 / 0.008 / 0.12 | no-read (3) or partial (2) |
| DAG | 0 / 0.8805 / 0 / 0 / 1.0 | 0.010 / 0.9997 / 0.105 / 0.919 / 1.0 | 1/5 (s10); s7 precision-limited |

* **Reading.** The Markov lottery is neither timing (2× is 0/5) nor exposure to the model's own states (DAG is 1/5).
  - It is the no-read basin, and the auxiliary reader removes it: writer and reader trained together from step 0.
  - That is the registered reading "neither 2× nor DAG", with the extra arm (111-1) supplying the fix.
* **SEL:** the lowest training hand-off error picks a passing run in §108 (s1), DAG (s10) and READ (s8). 2× has none to
  pick. AUC is 1.0 for both criteria, pooled over 20 runs.
* **early_escape (111-1) does not discriminate:** it fired in 11 of 12 failing runs and 8 of 8 passing ones (111-2).
* **Tags:** C1 (unqualified serial, S5) + C6 (5/5) + C2 to 2,048 (4/5), with per-step supervision (not C4). It
  triggered §113 and §117.

**Note 111-4 (2026-10-04 22:59 EDT; MK-4-READ complete, scored by Theory with `theory/score_s111.py`; DAG s9–10 still pending, so the §111
result follows when they land). MK-4-READ meets REL: 5/5.**

| READ seed | 128 / 256 / 512 | extension 1,024 / 2,048 | donor min / recipient max | pooled per-hand-off |
|---|---|---|---|---|
| s6 | 1.0 / 0.9995 / 0.9995 | 0.998 / 0.998 | 1.0 / 0.011 | 0.99998 |
| s7 | 0.999 / 0.997 / 0.9945 | 0.99 / 0.966 | 0.9995 / 0.011 | 0.99998 |
| s8 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 0.011 | 1.0 |
| s9 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 0.011 | 1.0 |
| s10 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 0.011 | 1.0 |

* **REL (C6) holds:** cumulative ≥ 0.99 at 512 in 5/5, with single-slot donor SERIAL in all 5. C6 at 2,048 (≥ 0.99) is
  4/5 (s7 0.966).
* **Reading.** Training the reader with the writer from step 0 removes the no-read basin: 0/5 READ seeds against 4/10 in
  §108 and 3/5 in 2×.
  - Unqualified serial C1 on S5 (Markov segments, no visible route) is now reliable: C1 + C6, with C2 to 2,048 in 4/5.
  - It is still per-step supervised (gold step text), so not C4.
* **SEL** picks a passing run in §108 (s1) and READ (s8). 2× and DAG have no passing run to pick so far.
* **Triggers.** §113 (READ inside §110) launches. §117 launches after Audit's truth-use grep, with its dose fixed in a
  note.

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

**Note 112-2 (2026-10-04 20:57 EDT; f = 0.1 s21, the first §112 run; descriptive, not the dose result).**
* **The seed meets FOOTHOLD-f's per-seed criterion** (1 of the 2 needed at f = 0.1).
  - coverage_drawn 0.10. k = 1 is 1.0 on the 12,960 never-labelled transitions and certified on all 14,400.
  - Cumulative 0.996 at 512 and 0.99 at 2,048.
* **C1: the slot alone fails, the joint patch passes.**
  - Single-slot donor-consistency is 0.12 / 0.12 / 0.34, with recipient-consistency 0.11 / 0.11 / 0.05: BYPASS at
    slots 1–2.
  - Mean-removal ≤ 0.055 at every slot, so the latents are necessary.
  - §104's joint patch (slot + op-token KV; `theory/s112_donor.py`, CPU, n = 100) gives **1.0 / 0.86 / 1.0**. The state
    is carried jointly by the slot and the op tokens, as §92 in 3/5. The n = 500 run, including §110 s1 / s3 / s5, is
    in 112-3.
* **Mechanism (descriptive): the acquisition was not driven by pseudo-labels.**
  - The kept share stayed ≈ 2·10⁻⁵ through rounds L3–L6: the self-consistency gate held the pools empty.
  - Never-labelled k = 1 jumped 0.072 → 0.999 during L6.
  - The likely driver: on lengths 5–6 (just past k = 4), the final answer is read from the model's own decoded
    checkpoint after the history cut, so final-answer targets train k = 1 / k = 2 transitions from self-decoded states.
  - The pools filled only after acquisition (L8, 0.36 → 0.999).

**Note 112-3 (2026-10-04 22:20 EDT; descriptive; 112-2's joint patch at n = 500, `theory/results/s112_donor.json`, CPU).** Donor- /
recipient-consistent per slot, single patch → joint (slot + op-token KV):

| model | slot 1 | slot 2 | slot 3 | reading |
|---|---|---|---|---|
| §112 s21 | 0.15 / 0.14 → **1.0** / 0.02 | 0.11 / 0.11 → **0.82** / 0.00 | 0.32 / 0.06 → **1.0** / 0.01 | carried jointly by the slot and the op tokens |
| §110 s3 | **0.96** / 0.02 → 1.0 / 0.02 | **0.99** / 0.01 → 1.0 / 0.00 | **0.98** / 0.01 → 1.0 / 0.01 | the slot alone (reproduces §110) |
| §110 s5 | 0.01 / 0.43 → 0.02 / 0.02 | 0.01 / 0.01 → 0.02 / 0.01 | 0.02 / 0.01 → 0.02 / 0.01 | neither: the state is spread over the prefix |
| §110 s1 | 0.02 / **1.0** → 0.02 / 0.02 | 0.02 / 0.09 → 0.02 / 0.01 | 0.00 / 0.09 → 0.02 / 0.01 | slot-1 bypass; joint gives chance |

* Under the masked fork, self-training lands on three different carriers: the slot alone (s3), the slot plus op tokens
  (s21), or a code spread over the whole prefix (s1, s5).
* Only Markov segments (§108) exclude the last two by construction. This is why the integrated candidate (self-training
  + Markov + READ) keeps single-slot donor SERIAL as its primary C1 rule (Audit (a)).

**§112 result (2026-10-05 03:57 EDT; 15 runs by the experiments session; scored by Theory alone with `theory/score_s112.py`, committed
before any run; `theory/results/score_s112.json`). FOOTHOLD is met at every dose; the smallest dose meeting it is f =
0.1 (coverage_drawn 0.10).**

| dose (coverage_drawn) | passing seeds (512) | certified k = 1 (never-labelled 1.0) | never acquired |
|---|---|---|---|
| f = 0.5 (0.47) | s23 0.9935, s25 0.9985 | s23, s24, s25 | s21, s22 |
| f = 0.25 (0.25) | s22 0.9995, s25 0.9995 | s22, s25 | s21, s23, s24 |
| **f = 0.1 (0.10)** | **s21 0.996, s23 0.9985** | s21, s22, s23 | s24, s25 (partial 0.39) |

* **Reading ("FOOTHOLD at f = 0.1").** About a 10 % short foothold suffices. Seeds learn the 90 % of S5's table they
  never see labelled from long final answers, certified at k = 1. That is C4 qualified by coverage, in 2/5 seeds.
* **The dose barely moves acquisition** (3/5, 2/5, 3/5). The lottery is 112-4's start-text no-read basin, not
  coverage. §116 NONE+COPY tests that.
* **C1 (single-slot donor SERIAL) is 0/6 in the passing seeds.**
  - Five show a slot-1 bypass (recipient 0.42–1.0) or the joint slot + op-token carrier (s21, 112-3).
  - f = 0.25 s22 comes closest (0.99 / 1.0 / 0.64).
  - Under the masked fork, self-training does not give serial latent use. That is §117's reason for Markov segments.
* **C6 at 2,048 (≥ 0.99):** f = 0.5 s25 (0.992), f = 0.25 s22 (0.998) and s25 (1.0), f = 0.1 s21 (0.99) and s23
  (0.996).
* **Tags:** C4 (qualified by 10 % short coverage), C2 (certified restarts), C6 per-dose seed rate 2/5. Not C1.

**Note 112-4 (2026-10-05 01:31 EDT; descriptive, $0, from eval.json `rounds`; Audit's question whether the never-acquiring seeds sit on a
no-read null). The seed lottery is a basin of a different kind: the start-text no-read basin.**
* Before acquisition, **every** seed (§112 f = 0.5 / 0.25 / 0.1, and §110 s1–5) has the same final-answer accuracy on
  each new round's pool (lengths 3..L): **0.50 / 0.67 / 0.50 / 0.33 at L = 4 / 5 / 6 / 8**.
  - So lengths 3–4 are solved exactly and every length ≥ 5 fails.
* **Lengths 3–4 are one chunk from the identity**, which the model solves from the ops alone. Lengths ≥ 5 restart after
  the cut from the model's own op-4 checkpoint.
  - That checkpoint is correct, since length 4 is solved, so the restarts' targets are consistent. Yet they stay at 0,
    and k = 1 restart accuracy stays at chance (0.008), until a seed acquires (abruptly, in L5–L8) or never does.
* **Reading.** Before the cut, the start is always the identity text, which carries no information. The model never
  learns to read a start state, the same chicken-and-egg as 111-1 but for the **start text**, not the latent.
  - Acquisition means learning to read a given state. The coverage dose barely moves it (f = 0.5: 3/5; 0.25: 2/5; §110
    natural: 3/5).
* **Prediction for §113, written before any §113 result (note 113-1):** §111's reader reads the slot latent, not the
  start text, so it should **not** raise the acquisition rate.
* **Candidate fix (not registered): reading footholds.**
  - Add **length-0 copy problems**: a random start state as text with no ops, whose answer is the start state.
  - They teach reading a given state while labelling **no** transition, so they are C4-neutral.
  - The same footholds may make §116 NONE (strict C4) acquirable.

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

**Note 113-1 (2026-10-05 01:31 EDT, before any §113 result; Theory's prediction from 112-4).** §110's and §112's never-acquiring
seeds fail to read the **start text** (restarts at chance while identity-start chunks are exact), not the slot latent.
§111's reader trains [EOT, slot latent] → state.
* **Predicted:** §113's acquisition rate stays near §110's (about 3/5 acquire). SELF2 is therefore unlikely at ≥ 4/5.
* Any C1 gain (fewer slot-1 bypasses) would come in the seeds that acquire.
* If §113 acquires in ≥ 4/5 anyway, 112-4's start-text account is incomplete, and that is reported as such.

**§113 result (2026-10-05 04:48 EDT; 5 runs by the experiments session; scored by Theory alone with `theory/score_s110.py` (s113),
committed before any run). REL is not met: SELF2 2/5, C1 0/5. The 113-1 prediction holds.**

| seed | 512 / 2,048 | donor / recipient (slots 1 / 2 / 3) | reader accuracy (slots 1–4) |
|---|---|---|---|
| s11 | **0.998 / 0.994** | 0.105 / 0.172; **0.992**; **0.990** | 1.0 throughout |
| s12 | 0 / 0 | chance | 0.72 / 0.12 / 0.05 / 0.03 |
| s13 | **0.9945** / 0.976 | 0.011 / **1.0** (slot-1 bypass); 0.011; **1.0** | about 1.0 |
| s14 | 0 / 0 | chance | about 0.08 |
| s15 | 0 / 0 | chance | about 0.1 |

* **Acquisition is 2/5, against §110's 3/5.** As predicted before any result (113-1), a reader of the slot latent does
  not open 112-4's start-text no-read basin. The three failing seeds never learn to read a start state (k = 1
  never-labelled 0.73 / 0.15 / 0.16).
* **C1 is not met in the acquirers.** s11's slots 2–3 are SERIAL, but slot 1 is neither (0.105 / 0.172). s13 bypasses
  slot 1 (recipient 1.0).
  - Under the masked fork, slot 1 can still recompute from the visible chunk start: the reader makes the latent readable,
    not used (Literature's caution, §113).
* `reader_target_precision` is logged in the round stats, not in eval.json; it is not used here.
* **Reading ("not REL").** The reader does not repair self-training's lottery or its slot-1 bypass. The two failure
  causes are separately addressed:
  - §116 NONE+COPY (start reading);
  - §117 (Markov segments, which remove the slot-1 route).

## 114. Looped depth per op: serial depth, not positions, for multiplication's latent hand-off (pre-registered 2026-10-04 15:54 EDT, after Literature's check c70d55e3 and before any implementation or run; the user's directive "register now", via Audit)

**Why.** Multiplication's latent hand-off fails at k = 4: §98 0.16, §99 none accelerates, §107 neither width (LAT) nor
extra token positions (PAUSE-4: 0.055 / 0.042 / 0.007) helps. Each op is a multi-digit multiply-add, and carries need
**serial depth** (Saunshi), which extra positions do not add. §114 adds depth per op by looping a middle block.
* **Prior** (c70d55e3, `docs/LIT_LOOPED_PROPOSALS_2026-10-04.md`).
  - Huginn 2502.05171, Ouro 2510.25741, UT, CoTFormer 2310.10845, Relaxed Recursive Transformers 2410.20672, MoR
    2507.10524.
  - Pretrained retrofits: Shapiro 2608.11233, McLeish 2511.07384, T²MLR 2607.15178.
  - Flags: untargeted iterations are slack (hence per-iteration targets); test-time depth scaling transfers modestly to
    retrofits; overthinking at large r.

**Harness** (§107's, so the two can be compared directly): 4×4 multiplication on the fixed code (98-3).
* Accel99's MARKOV segments: [a, *, op j−1's carried latent (or the chunk start text), op j] → slot j, fresh cache,
  forks on their own segment.
* The K-CURR bootstrap (k = 1 / 2 / 4 at 2k / 6k), eligible from 6k, a 24k cap, the valid-split early stop.

**The looped block.** GPT-2 layers 4–7 run **r times** over the segment's positions.
* Input injection: the layer-3 output is re-added before each iteration.
* Weights are tied across iterations, plus a per-(iteration, layer) LoRA (rank 8, zero-initialised), so r = 1 is GPT-2
  exactly.
* Layers 8–11 then run once on the final iteration.
* **Per-iteration targets:** each iteration's output, passed through layers 8–11 and the segment's fork, gets the step
  text's CE. The losses are averaged over iterations.

**Arms (seeds 1–3, paired with each other).**
* **LOOP-R:** r drawn uniformly from 1..8 per batch in training. The test uses **r = 8** (primary).
* **LOOP-1 (same-harness control; Audit (b)):** the same code path with r = 1 throughout. Depth is the only difference
  from LOOP-R.
* **Positions comparator:** §107 PAUSE-4 (existing, same harness; it failed CARRY-3).

**Measures.** §107's (per-op decode on the test split, test accuracy, removal of every carried latent, steps to
threshold), plus:
* **§104's donor protocol** on the carried latent at every in-chunk slot;
* the loss/null ratio at every log and `early_escape` (111-1);
* test-time **accuracy against r** ∈ {1, 2, 4, 8, 12, 16};
* FLOPs and sequential passes.

**Rules (fixed now).**
* **CARRY-3** (per arm): op-3 decode ≥ 0.95 in ≥ 2/3 seeds.
* **C1-donor:** in those seeds, §104 USE (single-patch donor-consistent ≥ 0.90) at every in-chunk slot, with no BYPASS
  (recipient < 0.10), and removal answer accuracy ≤ §98 NC-M 4×4 + 0.05.
* **DEPTH:** LOOP-R meets CARRY-3 + C1-donor while LOOP-1 does not, and PAUSE-4 did not.
* **No-read seeds (Audit (a), fixed now).** A seed with `early_escape` = false (loss/null ratio not < 0.9 within 1k steps
  of k = 2) is flagged **"no-read basin"**.
  - It **counts as a fail** in every rule above.
  - Results are **also reported among escaped seeds only**, as the conditional reading "among seeds that read their
    latent, does depth lift op 3?".
  - This keeps "loop depth didn't help" separate from "the latent was never read".

**Readings.**
* **DEPTH.** Serial depth per op, not extra positions, unlocks multiplication's latent hand-off: C1 on arithmetic with
  forced carriage. Accuracy against test-time r gives C2-in-depth and a C5 handle (P5 follows).
* **LOOP-R passes only among escaped seeds.** Depth works when the latent is read. Reliability waits on §111's fix.
* **Neither passes, even among escaped seeds.** Neither depth nor positions per op is the blocker at this budget. The
  running-sum auxiliary (§109-A's draft) is next.
* **Tags:** C1 (arithmetic, donor), C6 (descriptive), C5 handle (r); not C4.

**Placement.** 6 runs (2 arms × 3 seeds), about $6–9 (LOOP-R costs about 2.5–3× per step). Proof host first.
Projection to Audit. Implementation: the experiments session (accel99 plus a looped-block option in the GPT2KV forward).

**§114 result (2026-10-05 07:51 EDT; 6 runs by the experiments session; scored by Theory alone with `theory/score_s114.py`, committed
before any run). DEPTH is not met: looped depth does not unlock step 3.**

| arm (4×4, MARKOV, seeds 1–3) | op-3 decode | test | accuracy by test-time r |
|---|---|---|---|
| LOOP-R (r ∈ 1..8, input injection, tied + LoRA) | 0.342 / 0.195 / 0.362 | 0.055 / 0.026 / 0.062 | flat (r = 1 → 8: 0.056 → 0.055; it declines at r ≥ 12) |
| LOOP-1 (same harness, r = 1) | 0.174 / 0.165 / 0.245 | 0.012 / 0.019 / 0.031 | — |
| §107 PAUSE-4 (comparator) | 0.454 / 0.285 / 0.065 | 0.055 / 0.042 / 0.007 | — |

* **CARRY-3 is 0/3 in every arm.** No seed is in the no-read basin (the shuffled ratio shows the latent is read), so
  these are carriage failures, not reading failures.
* **Reading.** Serial depth per op does not help; the extra iterations change nothing (acc_by_r is flat).
  - 114-4 explains why: every carrier holds the multiplier prefix, not the running sum (causal donor test). No amount
    of compute inside a segment can supply a sum that is not carried.
  - The multiplication line moved to §121 (anchor) and §122 (anchor to a competent text path, with a stop rule).

**Note 114-1 (2026-10-04 15:56 EDT, before any §114 run; the experiments session's implementation choices, accepted).**
* **Per-iteration targets are an unbiased 2-sample estimator.** Each step takes CE at the final iteration r plus one
  uniform u ∈ {1..r−1}, with loss (CE_r + (r − 1) · CE_u) / r.
  - That is an unbiased estimate of the per-iteration average the registration specifies.
  - Its cost is about 3.5× per step, against about 8× for the literal all-iterations version. The literal version stays
    available as `--iter_targets all`.
  - LOOP-1 is unaffected.
* **Test r > 8** (`acc_by_r` at 12 and 16, descriptive) **reuses iteration 8's LoRA**, with the injection still applied.
* **LoRA:** rank 8 on c_attn and c_fc of layers 4–7, B zero-initialised (so r = 1 equals GPT-2), alpha/r = 1. Base
  weights stay trainable, as in accel99.
* **Injection:** iteration t ≥ 2's input is h_out(t − 1) + h3, with h3 the layer-3 output of the same positions.
* **The carried latent** is the final iteration's segment output. The donor patch replaces it at each in-chunk slot.
* Rules and arms are unchanged.

**Note 114-2 (2026-10-04 16:51 EDT, before any §114 result; motivated by 111-2 and the experiments session's null
analysis).**
* **The problem with the registered flag.** §114 flagged no-read seeds by 111-1's `early_escape`, which does not
  discriminate (111-2: all §108 seeds were True). The zero-latent null is out of distribution: a zeroed carried latent
  inflates even unread seeds' loss. §108 MK-4's final-checkpoint ratios (main / zero-null) were 6.8e-7, 1.8e-4, 0.27,
  0.31: separating, but not about 1 when unread.
* **The new flag.** A seed is **no-read** if **main loss / shuffled-null loss ≥ 0.9** at the first log at or after 6k
  (k = 4).
  - The shuffled null is the same batch with carried latents permuted across batch rows: in distribution, and about 1
    when the latent is not read.
  - The flag's use in §114's rules is unchanged: no-read seeds count as fails and are also reported separately.
* The same shuffled null replaces the zero null in §111's descriptive `early_escape_rel`, for the runs that get it
  natively.
* §114's l1 (the LOOP-R s1 proof) is restarted on the new code if it has not passed 6k. Otherwise its flag is computed
  as the final-checkpoint shuffled ratio, and labelled as such.

**Note 114-3 (2026-10-04 17:35 EDT, before any §114 result; calibration disclosed).**
* **Calibration data** (§108 MK-4 s1–4, `rel_ratio_final_shuffled` = main loss / loss with carried latents shuffled
  across the batch): **1e-6 / 0.742 / 9.7e-5 / 0.771**.
  - The readers (s1, s3) are ≤ 1e-4; the non-readers (s2, s4) are 0.74–0.77.
  - The non-readers' own latents still carry something the forks use: their loss is at chance (3.61), against 4.86
    with another row's latent. So no null sits at exactly 1 when unread.
* **114-2's flag (≥ 0.9 at the first log ≥ 6k) would misclassify.** The non-readers sit below 0.9, and a late escaper
  (s3 escaped only after k = 4) may still be near 0.7 at 6k.
* **The no-read flag (final):** a seed is **no-read if its final-checkpoint shuffled-null ratio is ≥ 0.5**. That sits
  between the calibration groups, with orders of magnitude of margin. It is used exactly as before: such seeds count as
  fails and are also reported separately.

**Note 114-4 (2026-10-05 02:10 EDT; descriptive, $0, CPU; the user's step-3 directive via Audit; `theory/mult_step3_diag.py`,
`theory/results/mult_step3_diag.json`; first model §99 MARKOV s1, n = 200; the other models follow). Multiplication's
step-3 wall is a shortcut: the latent carries the multiplier digits, not the running sum.**
* **Measurement.** per_slot_decode_acc is the test rollout: the chain runs on the model's own slot inputs, and forks are
  read, never fed back. The step texts are " P_1", " + P_j ( S_j )" for j = 2, 3, and " + P_4" (no sum). So steps 1 and
  4 need no carried state.
* **Step-3 anatomy.**
  - P_3 = a · b_3 is exact (1.0). S_3's digits 0–1, copied from S_2, are 1.0.
  - Digits 2 / 3 / 4 are 0.38 / 0.41 / 0.87.
  - **Errors do not track carries:** digit 3 is 0.40 with a carry-in against 0.41 without. Digit 2 is 0.38 although no
    carry can enter it. So this is not an addition-carry failure.
* **Linear probes on the carried slot-2 latent** (ridge, 1,500 synthetic problems, scored on 1,000 more):
  - **b_1 1.0, b_2 1.0** (b_3 0.08, b_4 0.12: future digits, at chance);
  - **S_2's digits 0.60 / 0.14 / 0.11 / 0.12 / 0.21 / 0.81** (chance about 0.10; the low and leading digits carry
    little entropy).
* **Reading.** The latent hands over b_1 and b_2, and each segment recomputes the sum from a (visible) and the multiplier
  prefix.
  - Step 2 then needs a 4×2 multiply in one pass (solved); step 3 a 4×3 multiply (fails).
  - A shortcut basin, not capacity: wider latents (§107 LAT), pauses (§107 PAUSE) and looped depth (§114 LOOP-R)
    cannot help, because nothing forces S_2 into the carrier.
* **Restart tests (T1 / T2)** are out of distribution here (K-CURR's text-start chunks ended at 6k): 0.0, not read.
* **All five models (2026-10-05 02:57 EDT, n = 200; Audit's generality check).**

  | model | step-3 decode | probe b_1 / b_2 | probe S_2 digits 2–4 | digit-3 accuracy (carry / no carry) |
  |---|---|---|---|---|
  | §99 MARKOV s1 | 0.24 | 1.0 / 1.0 | 0.11 / 0.12 / 0.21 | 0.40 / 0.41 |
  | §99 CTRL s1 (in context) | 0.815 | 1.0 / 1.0 | 0.12 / 0.11 / 0.13 | 0.90 / 0.88 |
  | §99 TF-SLOT s1 | 0.53 | 1.0 / 1.0 | 0.11 / 0.11 / 0.14 | 0.70 / 0.68 |
  | §107 PAUSE-4 s1 | 0.435 | 1.0 / 1.0 | 0.11 / 0.11 / 0.15 | 0.54 / 0.53 |
  | §100 GEN-2 s1 (k = 2) | from step-2 text: **1.0**, answer 1.0 | 1.0 / 1.0 | 0.12 / 0.11 / 0.14 | — |

  - **Prefix carriage is general:** every model's slot-2 latent holds b_1 / b_2 linearly at 1.0 and S_2's middle
    digits at chance. Errors are carry-independent everywhere.
  - §100's k = 2 model, restarted from the true step-2 text, computes step 3 and the answer at 1.0. The one-pass add
    S_2 + a·b_3·100 is easy when S_2 arrives as text; the failure is carrying S_2 in a latent.
  - (§100 evaluated as one k = 4 chunk is out of distribution: 0.)
* **Causal donor test (2026-10-05 03:09 EDT; `theory/mult_donor_diag.py`, n = 300, 233 SAMESUM donors found; Audit (a)).** Step-3 S_3
  is scored as mean agreement on digits 2–4 (chance about 0.1).

  | model | SAMEPREFIX: recipient / sum-donor | SAMESUM: recipient / prefix-donor | reading |
  |---|---|---|---|
  | §99 MARKOV s1 | **0.35** / 0.11 | 0.10 / **0.37** | **prefix carrier** (causal) |
  | §107 PAUSE-4 s1 | **0.56** / 0.10 | 0.11 / **0.59** | **prefix carrier** (causal) |
  | §99 CTRL s1 (in context) | 0.90 / 0.11 (99 % unchanged) | 0.90 / 0.10 (93 % unchanged) | latent ignored; recomputed from the visible ops |
  | §99 TF-SLOT s1 (in context) | 0.64 / 0.12 | 0.53 / 0.12 | mixed: partly read, partly recomputed |

  - Under Markov segments the step-3 output follows the **carried multiplier prefix**, never the running sum. 114-4
    holds causally, not only by probe.
  - The in-context models mostly bypass the slot, as §104 found for §97.
* **Matched design:** make the carrier hold the running sum. Use §111's reader on multiplication ([EOT, latent] → the
  step text, which contains S_j), registered next after Literature's check. Prior: 2510.00184 (a running-sum auxiliary
  unlocks multiplication).

## 115. Without snaps, and snapping only when needed: no-snap Markov chains and a round-trip snap trigger (eval only) (pre-registered 2026-10-04 15:55 EDT, after Literature's check c70d55e3 (P4) and before running on any checkpoint; the user's go-ahead for P4 via Audit)

**Why.** Every success so far snaps at hand-set points (C5 unmet), and length generalisation comes from the snap scaffold
(§106). §108's Markov segments make a new test possible at no cost.
* A segment sees only [EOT, the previous latent, op]. It is the same at every position, so **with snaps disabled** a
  working Markov model is a pure latent recurrence.
* (A) asks whether it stays exact without any discrete checkpoint (C2 + C6 for fully latent thinking).
* (B) asks whether a label-free trigger can decide when to snap (C5).

**Models.** §108 MK-4 seeds that escaped the no-read basin (s1, s3, and s5 if it escapes), plus §105 GEN-4-MASK s1–3
for (B).

**(A) No-snap chains.** Run the Markov recurrence with **no snaps at all** (the chunk start text only at position 0) on
§71's test set (2,000 × 512) and the 2,048 extension.
* Report cumulative accuracy against length, per-hand-off accuracy against position, and the latent norm and its
  distance to the snapped embedding along the chain (drift).
* **Rule NOSNAP (per model):** cumulative ≥ 0.99 at 512 with no snaps.

**(B) The round-trip trigger.** At each hand-off j, compare two paths for slot j+1:
* the **latent path**: [EOT, latent_j, op_{j+1}];
* the **text path**: [EOT, decoded text of state j, op_{j+1}].

Disagreement of their greedy decodes at slot j+1 is the label-free trigger. The policy runs latent and snaps (restarts
from the text path) only when the two disagree. There is no fixed k, apart from a cap at 64 hand-offs.
* Report: the **first-error detection rate** (the share of chains whose first wrong latent state is flagged at that
  hand-off, as in §83), the false-snap rate, accuracy at 512 / 2,048, and the snap count, against fixed k = 4.
* **Rule TRIGGER:** first-error detection ≥ 0.95, with accuracy ≥ fixed k = 4's at fewer snaps.

**Readings.**
* **NOSNAP holds.** The first fully latent chain without any discrete checkpoint that stays exact to 512 (2,048): C2 +
  C6 for latent thinking, on S5, in that seed. Snaps were needed in training, not at test.
* **NOSNAP fails, TRIGGER holds.** The latent drifts, but the model detects its own drift without labels: C5 for when to
  write.
* **Both fail.** The fixed snap interval remains a hand-set scaffold. Literature flags that a shared-weight round trip can
  miss consistent errors; the miss rate is reported.
* **Tags:** C2, C6, C5 (descriptive, per model; eval only).

**Placement.** Eval only, $0, on the local 3070 (idle). Implementation: the experiments session (mk108 / gen97 eval
modes: `--no_snap`, and an `--adaptive_snap disagreement` policy).

**Note 115-1 (2026-10-04 23:04 EDT; partial, descriptive; scorer format fix disclosed).**
* **Scorer fix after data was seen:** `fixed_k4.snaps_per_problem` is a number, not {mean, median}, which crashed the
  TRIGGER line. The scorer now accepts both. No rule changed.
* **So far, §108 MK-4 s1** (registered): NOSNAP fails (0 at 128; per-hand-off ≈ 0.2 past the 4-op horizon). TRIGGER
  holds: first-error detection 1.0, false-snap and miss rates 0, 1.0 at 512 / 2,048, but 99.9 snaps against fixed k = 4's
  127.
* **§111 READ s6** (an extra model, not in §115's registered set; descriptive only):
  - NOSNAP fails (0.002 at 128), but per-hand-off without snaps is **0.94 → 0.86** over positions 0–512, against s1's
    0.2.
  - **The trigger needs 19.7 snaps per 512 ops**, against 127 fixed (6.4× fewer). Detection is 1.0, false-snap and miss
    rates are 0, and accuracy is 1.0 at 512 / 2,048.
  - The reader makes the latent code far more stable across generations: an average silent stretch of about 26 ops,
    from training at 4.
* **Drift (Audit):** the free-running latent keeps its norm (ratio about 1.1), but its cosine to the snapped latent falls
  to about 0.01 by position 64. The no-snap failure is direction drift, not norm blow-up, which supports §118's cosine
  anchor.
* **Cost caveat (Audit; it goes into the §115 result):** the trigger runs the text path at every hand-off, so it saves
  checkpoints, not compute. A C5 design that saves compute needs a latent-only predictor of disagreement.
* The §115 result follows when MK-4 s3 / s5 and §105 s1–3 are in.

**Note 115-2 (2026-10-05 04:19 EDT; partial; READ models are outside §115's registered set (descriptive), and are §118's registered
control). One §111 READ seed runs fully latent to 2,048 with no snaps.**
* **§111 MK-4-READ s8.** The no-snap Markov recurrence (only the start text at position 0, then [EOT, carried latent,
  op] at every position, with no test-time text path, reader or decode feedback) gives **cumulative 1.0 at 128 / 256 /
  512 / 2,048** (2,000 × 512; 500 × 2,048).
  - Per-hand-off accuracy is 1.0 at every position. The latent norm ratio stays about 1.03, and the cosine to the
    snapped rollout about 0.95 after position 64.
  - Donor SERIAL is 1.0 (111-4). It is the first fully latent chain here that stays exact to 2,048 with no discrete
    checkpoint.
  - It is one seed: READ s6 / s7 drift (per-hand-off about 0.86 / 0.72); s9–10 are pending. It is per-step supervised,
    so not C4.
* **The trigger as a precision fix (registered models).** §108 MK-4 s3, which fixed k = 4 leaves at 0.585 at 512,
  reaches **1.0 with the trigger at 61 snaps** (against 127). s5 reaches 0.992 → 1.0 at 25 snaps; s1 1.0 at 99.9 snaps.
  §105 s2 reaches 1.0 at 112.6 snaps.
  - Snapping on self-disagreement removes the precision errors that fixed snapping lets through. That is C5 with a
    gain, though still using the text path (115-1's cost caveat).
* For §118: the READ control already has 1 seed passing NOSNAP-512 and NOSNAP-2048 (s8). ANCHOR and LONG are read
  against that rate.

**§115 result (2026-10-07 00:29 EDT; eval only, $0; the registered models are §108 MK-4 s1 / s3 / s5 for (A) and (B), and §105 GEN-4-MASK s1–3 for (B); scored by Theory
alone with `theory/score_s115.py`). NOSNAP fails, TRIGGER holds (6/6): the latent drifts, but the model detects its own drift without labels.**

| Model | (A) no-snap cumulative 128 / 512 / 2,048 | (B) trigger cumulative 512 | trigger snaps per problem (fixed k 4: 127) | first-error detection | false-snap / miss | fixed k 4 at 512 |
|---|---|---|---|---|---|---|
| §108 MK-4 s1 | 0 / 0 / 0 | 1.000 (2,048: 1.000) | 99.9 | 1.00 | 0 / 0 | 1.000 |
| §108 MK-4 s3 | 0 / 0 / 0 | 1.000 (2,048: 1.000) | 61.0 | 1.00 | 0 / 0 | 0.585 |
| §108 MK-4 s5 | 0 / 0 / 0 | 1.000 (2,048: 1.000) | 25.3 | 1.00 | 0 / 0 | 0.992 |
| §105 GEN-4-MASK s1 | — | 1.000 | 124.0 | 1.00 | 0 / 0 | 1.000 |
| §105 GEN-4-MASK s2 | — | 1.000 | 112.6 | 1.00 | 0 / 0 | 0.998 |
| §105 GEN-4-MASK s3 | — | 1.000 | 118.3 | 1.00 | 0 / 0 | 1.000 |

* **NOSNAP:** 0/3 registered Markov models. Without snaps the latent fails before 128 (per-hand-off decay as in 115-1).
* **TRIGGER:** **6/6.** First-error detection is 1.0 (≥ 0.95), and accuracy is at least fixed k = 4's at fewer snaps in every model.
  - Snapping on disagreement removes the precision errors that fixed snapping lets through (MK-4 s3: 0.585 → 1.000 at 61 snaps instead of 127).
  - Literature's worry, that a shared-weight round trip can miss consistent errors, did not occur here: miss rate 0 and false-snap rate 0 in all six.
* **Reading (as registered):** NOSNAP fails, TRIGGER holds. **C5 for when to write**, label-free.
  - Caveats: the trigger still runs the text path at every hand-off (115-1's cost caveat, about twice the compute).
  - For GEN-4-MASK the saving is small (113–124 snaps against 127).
* **Disclosed:**
  - g105_mask4_s1's trigger ran in row chunks of 100 under the local 6 GiB cap. The merged statistics use the same formulas as unchunked, but the
    bf16 batch composition differs from s2 / s3, which ran unchunked.
  - The GEN-4-MASK models use n 500 at 512 (the 114 layout); MK-4 uses 2,000 at 512 and 500 at 2,048.
* **Descriptive, outside the registered set:** §108 MK-4 s2 / s4 (no-read basin seeds) reach 1.0 with the trigger, but at about 507 snaps (every
  hand-off), so no saving. The READ / ANCHOR / LONG models are §118's (no-snap exact in ANCHOR 5/5, LONG s81–83, READ s8 / s10).

## 116. Strict C4: final-answer training through self-decoded restarts, from operator definitions only and from no short problems (pre-registered 2026-10-04 21:03 EDT, after Literature's check b0bca3c4 and before any implementation or run; launched only if §112 f = 0.1 meets FOOTHOLD-f)

**Why (112-2).** §112 f = 0.1 s21 is certified on all 14,400 k = 1 transitions, 90 % of them never labelled. Neither the
shorts nor pseudo-labels drove its acquisition.
* At the end of R0, k = 1 accuracy on labelled transitions was only 0.078.
* The gate kept the pools empty through rounds L3–L6 (kept share ≈ 2·10⁻⁵).
* Never-labelled k = 1 jumped 0.072 → 0.999 during L6, on final-answer targets alone.
* **Mechanism (Literature b0bca3c4).** Past the cut (L > k), the target is predicted from the model's **own decoded
  checkpoint** plus the remaining L − k ops. Each such problem is in effect a short problem from a self-generated start:
  the cut manufactures footholds, and the leap (Abbe 2302.11055) drops from L to L − k.
  - Where the checkpoint is wrong, the target is inconsistent (§62 noise). Learning is a self-reinforcing bootstrap.
* If the shorts are not the engine, the foothold may shrink to the operators' definitions, or vanish. No in-LM run has
  met strict C4 (lengths ≥ 3 only; §93, §95, §110b).
* **Prior** (b0bca3c4, `docs/LIT_S116_STRICT_C4_2026-10-04.md`): DAgger 1011.0686, TD bootstrapping, Delethink
  2510.06557, Markovian Transformers 2404.18988, Lee 2502.01612, 2511.07378, Zhang 2607.20594 (operator-first), §85,
  §110b. Acquisition of a non-solvable group's transitions through cut-induced self-started shorts: not found (new as
  scoped).

**Design.** §112's recipe unchanged except the short problems: §110 identity starts, masked fork, independent pools,
validity → final → self-consistency filters with the label-free gate, final relabel, 24k steps, lengths drawn uniformly
from 3..L_max per round.
* **OPDEF (seeds 31–35; 112-1's option (b) with g = 0).** T_f holds only the 120 first-step transitions (e, a).
  - R0 draws length-1 shorts (e; a), and length-2 shorts only where both transitions are in T_f (first op the
    identity), at §112's R0 count (1,200 × 64).
  - The round schedule is §112's, so OPDEF extends §112's dose curve. coverage_drawn ≈ 0.0083.
* **NONE (seeds 41–45; strict C4).** No R0 and no short problems at all; every training problem has length ≥ 3.
  - **Every round's L_max is ≥ k + 1 = 5** (Literature trap 1): round 1 is L_max = 6 (lengths 3..6), and §103's
    adaptive insertion never goes below L_max = 5.
  - So post-cut problems are present from the start. L ≤ 4 problems stay in the mix; they are the only targets for
    the op-4 checkpoint's decode.
  - The 1,200 R0 updates go to round 1 (24k total).
* **Logging, both arms; truth-scored and never used in training or any training decision.**
  - **Foothold yield per round** (Literature trap 2): on 5k fresh problems with L > k, the fraction whose decoded
    checkpoint at every cut is correct. Near 1/120 means a stall by construction.
  - **Accuracy by length per round** (trap 4): the per-length contribution.
* Eval keys as §112: k1_never_labelled_acc, k1_certified, per_round_split, coverage_drawn, single-patch donor,
  removal. Plus §104's joint patch (`theory/s112_donor.py`) on every seed meeting the per-seed criterion.

**Rules (fixed now; per arm).**
* **FOOTHOLD-OPDEF:** ≥ 2/5 seeds with cumulative ≥ 0.99 at 512 **and** k = 1 accuracy ≥ 0.99 on never-labelled
  transitions (§112's criterion).
* **STRICT-C4 (NONE):** ≥ 2/5 seeds with cumulative ≥ 0.99 at 512. Every transition is never-labelled; k1_certified is
  reported.
* C1, as §110 / §112: single-patch donor SERIAL (USE ≥ 0.90 at every in-chunk slot, no BYPASS ≥ 0.10), with the joint
  patch reported where single fails. C6: the per-arm seed rate, and ≥ 0.99 at 2,048. All 5 seeds are reported.
* **Claim wording (fixed; Literature (b), (3)).**
  - Method: "final-answer training through self-decoded restarts" (not "filtered self-training").
  - OPDEF: "C4 qualified by operator definitions (each op applied to the identity; 120 problems, no composition)". It
    also fixes the action convention.
  - NONE: "strict C4: final answers of problems of length ≥ 3 only".

**Readings.**
* **NONE passes.** The first strict-C4 result in a pretrained LM: S5 latent restarts learned from final answers alone
  (certified if k1_certified). Tags C4 + C2, plus C1 / C6 as scored.
* **OPDEF passes, NONE does not.** Operator definitions are the minimum foothold.
  - NONE's round schedule differs (L_max ≥ 5 from round 1), so this reading is qualified by that difference.
  - Foothold yield says whether NONE stalled by construction (yield ≈ 1/120) or bootstrapped too slowly.
* **Neither.** The shorts carried a needed signal that R0's labelled accuracy hides. §112's dose curve gives the
  threshold.
* **A k = 2 variant** (more footholds, tail length L mod 2) is the registered follow-up if NONE stalls with yield near
  chance. It is not run here.

**Placement.** 10 runs, about $12–15 on Vast, **launched only if §112 f = 0.1 meets FOOTHOLD-f (≥ 2/5)**; otherwise not
run (noted then). Implementation: the experiments session (self110: --cov with g = 0 for OPDEF; --no_r0 with an L_max
floor for NONE; foothold-yield and per-length logging). Scorer `theory/score_s116.py`, committed before any run.
Projection to Audit.

**Note 116-1 (2026-10-04 22:24 EDT, before any §116 implementation or run; Audit, Literature 3b9c8d61).**
* **Every in-run decision uses only final-answer quantities:** length insertion, filters, selection, stopping and any
  fallback. §116 has no in-run switch.
* Truth-scored intermediates are logged only: foothold_yield, truth_precision_kept and every k = 1 accuracy.
* The k = 2 follow-up is a between-registration diagnostic, not an in-run trigger. It would be a new registration
  that reads foothold_yield after §116 is scored.
* The implementation names every function where true intermediate states enter. Audit greps it before any run.

**Note 116-2 (2026-10-05 01:34 EDT, before any §116 implementation run or launch; a pre-registered amendment, Audit-approved about $6,
after Literature's check 15cf541a). New arm NONE+COPY: reading footholds against the start-text no-read basin (112-4).**
* **Arm NONE+COPY (seeds 41–45, the same seed numbers as NONE; dirs `s116_nonecopy_s4x`).** NONE's design exactly,
  except that **10 % of the rows of every batch**, from step 0 through the final relabel, are **length-0 copy problems**.
  - The input is a random start state (uniform over the 120) as text. The chunk has **zero ops and no marker token**
    (Literature (2)): the start text's last hidden state is fed as a slot, exactly as at an op position, and §105's
    masked fork decodes it.
  - The target is the start state's own text. No op is applied, so **no transition and no intermediate is ever
    labelled**. The budget is unchanged: the copy rows replace ordinary rows.
* **Logged (truth-scored, never used in any decision):**
  - the **stay rate** on real ops (k = 1 restarts whose output equals the input state), against 1/120;
  - accuracy on the L ≥ 5 share of each round's pool;
  - k = 1 restart accuracy per round.
* **Prediction (fixed now):** COPY lifts the L ≥ 5 pool accuracy off 0 and makes k = 1 restarts acquire in ≥ 4/5
  seeds, while NONE stays near chance in most seeds.
* **Rules (fixed now).**
  - **STRICT-C4-COPY:** ≥ 2/5 with cumulative ≥ 0.99 at 512 (NONE's criterion).
  - **MECH (the mechanism test):** NONE+COPY acquires (final k1_never_labelled_acc ≥ 0.99) in **≥ 4/5 seeds and in more
    seeds than NONE**.
* **Claim wording (fixed; Literature (1)).** A NONE+COPY pass is "C4: final answers plus a self-supervised copy objective
  (no intermediate or transition labels)". "Strict C4" stays for NONE alone; both are reported side by side.
* **Amendment (2026-10-05 01:36 EDT, before any run; Audit):** the rule STRICT-C4-COPY is renamed **C4-COPY** (the criterion is
  unchanged), so a pass can never be printed as strict. "Strict C4" is reserved for NONE.
* 116-1's truth-use rule is unchanged: copy problems use only the input. The launch condition (§112 FOOTHOLD-0.1) is
  unchanged. Scorer `theory/score_s116.py`, extended now.

**Note 116-3 (2026-10-05 01:36 EDT, before any §116 run; the experiments session's NONE+COPY implementation, accepted).**
* `self110 --no_r0 --copy_frac 0.1`: 6 of 64 rows per step are copy problems, from the same draw RNG, from step 0 through
  the final relabel.
* `Gen110.copy_loss` runs the KV chunk over the start text alone; its last hidden is the slot, and the masked fork
  decodes the start's own text. Loss = (58 · main + 6 · copy) / 64.
* Logged only:
  - stay_rate: the share of k = 1 restarts whose decode equals s, over the (s, op) pairs with δ(s, op) ≠ s;
  - acc_L5plus per round;
  - k1 accuracy per round.
* The copy path is in the truth-entry docstring (input only).

**Note 116-4 (2026-10-05 08:32 EDT; descriptive, written after 3/5 NONE+COPY seeds were seen (s41, s42, s45, all 0), before the §116
panel is scored; rules unchanged). NONE+COPY fails upstream of the mechanism it was meant to test.**
* In all three seeds, each round's pool accuracy is about **0.008 (chance) at every length, including 3–4**. Those are
  single-chunk identity-start problems that every §110 / §112 seed solved early with shorts (pool 0.50 at L = 4; 112-4).
  - The training loss stays at about 4.7–5.0, the per-answer null (ln 120 ≈ 4.79), for 24k steps.
* So without short problems the model never learns even a 3-op composition from final answers: the strict-C4 saddle
  (§85; Literature b0bca3c4 trap 1).
  - The start-text reading basin of 112-4 is never reached, so the copy rows have nothing to act on.
* **Reading for MECH (stated before the panel is scored):** a MECH failure with in-chunk pool accuracy at chance does
  **not** test 112-4; it is uninformative about it.
  - 112-4's mechanism can only be tested where in-chunk composition is learned, i.e. with a short foothold (§112's
    setting plus copy rows).
  - §117-COPY's trigger (MECH passes) will then not fire, and a direct test of 112-4 would need its own registration.
* **Addendum (2026-10-05 08:34 EDT; Theory's CPU check on the trained models): the copy objective itself was not learned.**
  - On 300 random states, the zero-op copy decode gives accuracy **0.0** in s41 (copy CE 5.58, above the 4.79 null) and in
    s45 (CE 27.1, no valid decode).
  - So NONE+COPY never delivered its intervention: neither the copy rows nor the main task trained.
  - A copy-only unit test (the experiments session) will tell an implementation fault from interference by the stuck main
    loss. Either way the arm is uninformative about 112-4.

**Note 116-5 (2026-10-05 08:45 EDT; the NONE+COPY fault, and what follows; written before the NONE / OPDEF arms are scored).**
* **Fault (the experiments session's diagnosis).** `copy_loss` sat under a `@torch.no_grad()` decorator that belonged to
  `read_greedy`. The copy CE entered the loss as a constant, with no gradient.
  - **Every NONE+COPY run (s41–45) was effectively a NONE replica.** The intervention was never delivered.
  - s41 / s42 / s45 are **voided** (logs kept, `*.void`, "intervention not delivered: no_grad bug"). s43–44 are stopped
    and voided the same way.
* **No automatic rerun (fixed now).**
  - NONE+COPY tests 112-4's start-reading account only if the in-chunk stage is reached. 116-4 shows the no-shorts
    setting never reaches it: pool accuracy at chance even at lengths 3–4.
  - **NONE+COPY is rerun (with the fix) only if ≥ 1 NONE seed shows in-chunk acquisition** (pool accuracy at L ≤ 4
    above 0.25 in some round). Otherwise it is not rerun.
  - The direct test of 112-4 is decided by note 117-5's rule (§112 f = 0.1 + copy rows).
* The fixed `copy_loss` is unit-tested copy-only before any future use.
* NONE and OPDEF are unaffected (no copy path) and are scored as registered.

**Note 116-6 (2026-10-05 09:27 EDT; NONE s42–45 scored; s41 pending). STRICT-C4 is decided: not met.**
* All four seeds sit at chance throughout: cumulative 0, k = 1 never-labelled 0.0083 = 1/120, and pool accuracy 0.0–0.009
  at every length in every round.
* STRICT-C4 needs ≥ 2/5, so it cannot pass whatever s41 does.
* The strict-C4 saddle holds in-LM with final answers only (§85, 116-4): there is no in-chunk acquisition, even at
  L ≤ 4.
* **116-5's NONE+COPY rerun condition is not met** by s42–45 (no in-chunk acquisition). It stays unmet unless s41 shows
  it.
* OPDEF (operator definitions, 0.83 %) is still running and is scored at panel end. It is now the arm that says whether
  any foothold below §112's 10 % works.

**Note 116-7 (2026-10-05 19:32 EDT; descriptive, OPDEF s33 / s34 / s35 of 5; FOOTHOLD-OPDEF is scored at panel end). With only the
operators' definitions as foothold, one seed learns S5's entire transition table.**
* **OPDEF s33** (local 4060): k = 1 never-labelled **1.0**, **certified on all 14,400** k = 1 transitions.
  - Only the 120 operator-definition problems (each op applied to the identity) were labelled: coverage 0.83 %.
  - Cumulative 0.9985 / 0.9925 / **0.987** at 128 / 256 / 512: just under FOOTHOLD's 0.99 bar. 112-5's
    post-acquisition-budget effect is a likely cause (descriptive).
* s34 and s35: never acquired (k = 1 never-labelled 0.04).
* **Reading (one seed, descriptive).** Final-answer training through self-decoded restarts can recover the whole S5 table
  from the operators' own definitions alone, with no transition composition labelled. That is "C4 qualified by operator
  definitions", in one seed.
  - Strict C4 (no definitions, NONE) failed in 4/4 at chance (116-6). The definitions are the minimum foothold seen.
* **Hardware:** s33 / s34 / s35 trained on three different local GPUs (4060 / 3070 / 2070). That is fine for a per-seed
  count (no within-seed comparison), but stated given the GPU sensitivity seen today (125-2).
* The remaining seeds (NONE s41, OPDEF s31 / s32) run on the local 2070 ($0).

## 117. One model for C1 + C4 + C6: final-answer training through self-decoded restarts under Markov segments, with the auxiliary reader (pre-registered 2026-10-04 22:24 EDT, after Literature's check f9ce7210 / 3b9c8d61 and before any implementation or run; launched only if §111 MK-4-READ meets REL)

**Why.** Three results point at one design.
* **§108:** Markov segments make the latent the only route. Every seed that learns is single-slot SERIAL (unqualified
  C1), but 4/10 seeds sit in the no-read basin.
* **§111 READ:** an auxiliary reader from step 0 removes the no-read basin (3/3 so far).
* **§112 / §116:** final-answer training through self-decoded restarts learns S5's table from a small foothold (112-2).
  - Under the masked fork the carrier is seed-dependent (112-3): the slot alone (§110 s3), the slot plus op tokens
    (§112 s21), or the whole prefix (§110 s1 / s5).
  - Markov segments exclude the last two by construction.
* No run so far has single-slot SERIAL C1 + C4 + C6 in one recipe.
* **Prior** (f9ce7210, `docs/LIT_S117_INTEGRATED_2026-10-04.md`): Markovian Transformers 2404.18988, RMT, Delethink
  2510.06557 (outcome-trained bottlenecks); SIM-CoT, SELR 2608.13570, ReLAT 2606.06252, 2606.20075, MUX
  (collapse-preventing readers); Arazo 1908.02983 (confirmation bias); Allchin 2607.11953 (warning). An outcome-trained
  carrier with a self-target reader, certified on a non-solvable group: not found (new as scoped).

**Common design (both arms).** §112's recipe (identity starts, final answers only, independent pools, validity → final
→ self-consistency filters with the label-free gate, final relabel, rounds 3..L_max, 24k steps), plus:
* **Markov segments (§108):** each op is a fresh-cache segment [EOT, slot j−1's latent (or the chunk's start text: the
  model's own decoded checkpoint after a cut), op j] → slot j. The step fork sees only its segment. Snaps every k = 4
  with a history cut.
* **No gold teacher forcing** (C4). §108's K-CURR runs on training k (1 → 2 → 4, at §108's step counts). At k = 1 every
  op restarts from the model's own decoded text.
  - Where a kept pseudo-label exists, a segment input may be that text, under §108's ρ schedule (the model's own
    filtered decode, never gold).
* **Auxiliary reader (§111 READ, as §113):** a fresh [EOT, carried latent] segment decodes the slot's state text (CE,
  weight 1) at every targeted slot from step 0.
  - Targets: the model's own filtered decodes (kept problems' per-slot pseudo-labels), and the true final answer at
    final slots, with the final-answer filter in front.
  - Final-slot reader updates continue in every round (Literature (c)).
* **Dose:** f = 0.1 (§112's T_f) if §112 meets FOOTHOLD-0.1, otherwise §110's natural shorts. It is fixed in a dated
  note before launch.
* **Every in-run decision uses only final-answer quantities** (Literature (d); Audit). That covers fallbacks, length
  insertion, the ST switch, filters, selection, stopping and reader targets.
  - Truth-scored intermediates are logged only: foothold_yield, truth_precision_kept, every k = 1 accuracy, and
    reader-vs-truth precision.
  - The implementation names every function where true intermediate states enter. Audit greps it before any run.

**Arms.**
* **MK (seeds 51–53, extended to 54–55):** Markov segments from step 0.
* **ST, staged bottleneck (seeds 61–63, extended to 64–65; Literature (b)).**
  - §112's masked fork (non-Markov, so the parallel in-pass shortcut can seed checkpoints) until a **label-free switch**:
    final-answer accuracy ≥ 0.5 on 5k fresh problems of length k + 1..k + 2, checked at each round's end, or at 12k at
    the latest.
  - Then Markov segments (with the reader) for **12k further steps**, plus the final relabel.
  - C1 is scored on the final model only. The switch step is logged.
* **Extension rule:** an arm with ALL in ≥ 2/3 of its first three seeds gets its two further seeds. Otherwise it stops
  at 3, reported.

**Measures.** §112's (cumulative 512 / 2,048, k1_never_labelled_acc, k1_certified, coverage_drawn, removal,
per_round_split) and §116's (foothold_yield, acc_by_length). §104's donor protocol (single and joint) at every in-chunk
slot; the shuffled-null no-read flag (114-3); per-hand-off accuracy; the **reader confirmation-bias measure**: the truth-scored precision of the reader's
self-targets on kept slots, per round (logged only).
Descriptive, where cheap: C3 (FLOPs vs the written state), C5 (§115's round-trip trigger), C7 (checkpoint edit IIA).

**Rules (fixed now; per arm).**
* **ALL (per seed):** cumulative ≥ 0.99 at 512, **and** single-slot donor SERIAL (USE ≥ 0.90 at every in-chunk slot,
  recipient < 0.10; Audit (a)), **and** k = 1 ≥ 0.99 on never-labelled transitions.
* **INTEGRATED (C1 + C4 + C6):** ALL in ≥ 4/5 seeds (extended arm). **EXIST:** ALL in ≥ 1 seed (an existence proof,
  reported separately from reliability).
* **C6 at length:** ALL plus ≥ 0.99 at 2,048, in ≥ 4/5.
* **Sanity:** an accuracy pass with single-slot SERIAL failing in a Markov final model means a leak, and is reported
  as a bug (as §108).
* **Claim wording (fixed):** "unqualified serial C1 + C4 qualified by <dose> identity-start shorts + C6 (seed rate)",
  with the method named "final-answer training through self-decoded restarts". ST adds "staged: masked fork first".

**Readings.**
* **INTEGRATED in MK.** One pretrained LM learns S5 latent restarts from final answers with a small foothold. Its
  latents carry the state hop by hop with no other route, reliably. C1 + C4 (qualified) + C6, with C2 by restarts.
* **INTEGRATED in ST only.** The same, but the bottleneck must come after acquisition: in-pass shortcuts seed the
  checkpoints, and Markov carriage is learned afterwards.
* **EXIST only.** The combination is possible; reliability is the gap. Report the failure modes (no-read flag, final
  accuracy at L > k, precision).
* **Neither.** Markov carriage and outcome-only acquisition do not combine at this budget. Report per-arm switch steps
  and no-read flags.

**Placement.** 6 runs first (3 per arm), up to 10, about $15–30 on Vast. **Launched only if §111 MK-4-READ meets REL**;
otherwise not run (noted then). Implementation: the experiments session (self110 + mk108's segment construction +
§113's reader + the ST switch). Scorer `theory/score_s117.py`, committed before any run. Projection to Audit.

**Note 117-1 (2026-10-04 22:25 EDT, before any §117 implementation or run; the experiments session's implementation readings, accepted).**
* **The reader is on from step 0 in both arms.** In ST before the switch it reads the masked-fork chunk's slot input,
  as §113.
* **K-CURR with self-training items.** Training chunks are cut at k_train (1 / 2 / 4 at §108's 2k / 6k); items are
  rebuilt when k_train changes. Labelling rollouts stay at k = 4.
  - Kept problems: k_train-chunks, each starting from the pseudo-label text at the cut.
  - Discarded problems: a final-only target on the last k_train-chunk, which starts from the model's own decode at that
    position in the labelling rollout. Problems whose decode there does not parse are dropped (a validity filter,
    label-free).
* **The ST switch** is checked at each round's end: final-answer accuracy on 5k fresh length-5–6 identity-start problems
  ≥ 0.5.
  - **The forced switch is at step 12,000 exactly**, mid-round if needed, as registered.
  - After the switch: 12k Markov steps plus the 4k final relabel. switch_step is logged.
* **No ρ-forcing** (the registration allowed it on kept pseudo-labels; it is not used).
* **Truth audit:** the experiments session lists every function where true intermediate states enter, in the
  implementation's docstring, for Audit's grep before launch.

**Note 117-2 (2026-10-04 23:02 EDT, before any §117 run; implementation choices accepted; Audit's truth-use grep passed at HEAD).**
* **ST's learning rate:** at the switch, the cosine horizon becomes switch_step + 16k (12k Markov steps + the 4k final
  relabel).
* **Empty-item fallback (label-free, logged as `fallback_items`).** It applies only when no item would be trainable
  (seen in the untrained CPU smoke, where every cut decode failed to parse).
  - A discarded problem then falls back to the nearest earlier cut with a parsing decode, or to the start, with the
    final-only target.
  - It uses validity only, never truth. A real model after R0 is not expected to hit it; the count is reported.
* The dose is still open. It is fixed in a further note when §112 FOOTHOLD-0.1 is decided.

## 118. Fully latent at inference: anchoring the carried latent to the text path, against longer silent stretches (pre-registered 2026-10-04 23:07 EDT, after Literature's check bd1db295 and before any implementation or run; C2 without snaps, the user's "full latent thinking without snaps")

**Why (115-1).** With the reader (§111 READ s6), the Markov latent code is far more stable across generations, but it
still drifts.
* Without snaps, per-hand-off accuracy is 0.94 → 0.86 over positions 0–512, against about 0.2 for §108 s1.
* The disagreement trigger needs 19.7 snaps per 512 ops, against 127 fixed.
* Snaps remain a test-time scaffold. §118 asks whether training can make the recurrence stationary, so a chain stays
  exact with **no discrete checkpoint at test**.
* **Prior** (bd1db295, `docs/LIT_S118_ANCHOR_2026-10-04.md`).
  - ANCHOR is professor forcing (1610.09038) in concept: free-running dynamics aligned to teacher-forced dynamics.
    Also CODI, KaVa, ICoT-KD, System-1.5, LIFT 2609.38149.
  - Drift priors: DEQ, Attractor Models, 2609.26487, nGPT, 2609.02049. SimSiam / BYOL for stop-gradient asymmetry.
  - Range: §41 (reach about 1.5–2× trained depth), §22 (no clock), §56 / §62.
  - A per-hand-off text-path anchor of a pretrained LM's carried latent, for no-snap stability, scored by drift and
    donor SERIAL: new as scoped, not as concept.

**Base.** §111 MK-4-READ exactly: S5, Markov segments with the sink fix, the TF bootstrap ending at 12k, the reader from
step 0, 24k steps, bs 64, lr 1e-4.

**Arms (3 seeds each first; extended to 5 when NOSNAP-512 holds in ≥ 2/3).**
* **ANCHOR (seeds 71–73, +74–75).** At every latent-fed segment j (training k ≤ 4), the latent path [EOT, latent_{j−1},
  op_j] gives carried latent a.
  - The text path [EOT, gold state text of j−1, op_j] gives b, with **stop-gradient** on b.
  - **L_anchor = (1 − cos(a, b)) + |log‖a‖ − log‖b‖|**, weight 1, added to the fork and reader losses (Literature (b)).
  - The latent code is pushed to be a function of the state alone (the generation-1 code): a soft snap without
    decoding.
* **LONG (seeds 81–83, +84–85).** No anchor. K-CURR continues past 4: k_train = 1 / 2 / 4 / 8 / 16 / 32 from steps 0 /
  2k / 6k / 10k / 14k / 18k. Training snaps every k_train; the TF bootstrap is as §111.
* **Control:** §111 READ s6–10, with the same no-snap evaluation (no new training).

**Measures.**
* **No-snap evaluation** (§115 (A)): the Markov recurrence from the identity text at position 0, then [EOT, carried
  latent, op] at every position.
  - **No test-time text path, reader or decode feedback** (Literature (d)). The per-step fork decode is read for scoring
    only.
  - Cumulative at 128 / 512 / 2,048, per-hand-off by position, and drift (norm ratio, cosine to the snapped rollout).
* Also measured:
  - §115 (B)'s trigger snap count;
  - the standard k = 4 snapped cumulative at 512 / 2,048;
  - §104's single-slot donor protocol;
  - reader_acc;
  - the **effective rank** of the carried latents (Literature (b));
  - the anchor cosine and log-norm gap at test, by position.

**Rules (fixed now; per arm).**
* **NOSNAP-512:** no-snap cumulative ≥ 0.99 at 512 in ≥ 4/5 (extended arm), with single-slot donor SERIAL (§104, at k =
  4 chunks) in those seeds. **EXIST:** ≥ 1 seed.
* **NOSNAP-2048 (C2 at length):** no-snap cumulative ≥ 0.99 at 2,048 in ≥ 4/5.
* **Guard:** the standard k = 4 snapped cumulative stays ≥ 0.99 at 512 in the passing seeds.
* **Claim wording (fixed):** "fully latent at inference, text-anchored in training: C1 + C2 + C6 with per-step state
  supervision (not C4)".

**Readings.**
* **ANCHOR passes NOSNAP-512 and -2048.** A pretrained LM's latent recurrence stays exact to 2,048 with no discrete
  checkpoint at test. Text is needed only as a training anchor. This is the first fully latent chain at length here.
* **LONG passes at 512 but fails at 2,048 (or beyond about 2× k_train).** The range-principle signature: longer
  training stretches move the boundary but do not remove it.
* **Neither.** Drift is not the only error: 0.99 at 512 needs about 0.99998 per hand-off. Snapping on the label-free
  trigger (§115 B) remains the route to C5.
* **Diagnostics:** effective-rank collapse or a shrinking norm under ANCHOR is reported as a pathology, not a pass.

**Placement.** 6 runs first (up to 10), about $9–15 on Vast. Not conditional (§111 READ met REL). Implementation: the
experiments session (mk108 `--arm read` plus `--anchor` and an extended `--kcurr`; snap115 (A)/(B) on every model).
Scorer `theory/score_s118.py`, committed before any run. Projection to Audit.

**§118 result, ANCHOR arm (2026-10-05 14:11 EDT; 5 runs by the experiments session; scored by Theory alone with `theory/score_s118.py`,
committed before any run; LONG still running). ANCHOR meets NOSNAP-512 and NOSNAP-2048: 5/5 seeds are fully latent at
inference and exact to 2,048 with no snaps.**

| seed | no-snap 128 / 512 / 2,048 | snapped 512 (guard) | donor min / recipient max | effective rank |
|---|---|---|---|---|
| s71 | 1.0 / 1.0 / 1.0 | 1.0 | 1.0 / 0.011 | 48 |
| s72 | 1.0 / 1.0 / 1.0 | 1.0 | 1.0 / 0.011 | 83 |
| s73 | 1.0 / 1.0 / 1.0 | 1.0 | 1.0 / 0.011 | 122 |
| s74 | 1.0 / 1.0 / 1.0 | 1.0 | 1.0 / 0.011 | 108 |
| s75 | 1.0 / 1.0 / 1.0 | 1.0 | 1.0 / 0.011 | 59 |

* **NOSNAP-512: 5/5** (needs ≥ 4/5) with single-slot SERIAL in all five. **NOSNAP-2048: 5/5.** The guard holds. The rate
  beats the READ control's 2/5 (118-2).
* **Mechanism (118-5):** all five ANCHOR seeds have ρ(ε 0.1) = 0.19–0.21 (basins 0.1–0.2), the strongest attractors
  measured. Every model with ρ ≤ 0.28 passes no-snap, and every model with ρ ≥ 0.38 fails.
* **Reading.** Tying the latent path's carried state to the stop-gradient text-path state makes the pretrained LM's
  latent recurrence strongly contractive and drift-free. On S5 a chain of 2,048 hand-offs stays exact with no discrete
  checkpoint at inference, in every seed.
* **Claim wording (fixed in §118):** "fully latent at inference, text-anchored in training: C1 + C2 + C6 with per-step
  state supervision (not C4)". It is **not task-agnostic**: the anchor uses the task's state text in training. §120 /
  §123 carry the task-agnostic line.
* **Tags:** C1 (serial, 5/5), C2 (2,048 with no snaps, from training chunks of 4), C6 (5/5), C7 (snapped checkpoints
  remain readable). Not C4.

**Note 118-1 (2026-10-04 23:14 EDT, before any §118 run; the experiments session's implementation, accepted).**
* **ANCHOR** applies only at latent-fed segments, not at gold-forced ones (TF bootstrap). The anchor latent is the
  stop-gradient text path [EOT, gold text of state j−1, op_j]. The loss is as registered.
* **LONG:** k_train 1 / 2 / 4 / 8 / 16 / 32 at 0 / 2k / 6k / 10k / 14k / 18k, as registered.
  - Micro-batches of bs / max(1, k // 8) with gradient accumulation: same effective batch, for memory.
  - TF bootstrap as §111.
* **Eval:** `latent_geometry` (effective rank over 2k carried latents; anchor cosine and log-norm gap by position
  against the text path from the true state, eval only), then snap115 (A)/(B) in-process → eval_s115.json.
  - No-snap uses no text path, reader or decode feedback.
* **Control:** snap115 on §111 READ s7–10 on a local GPU.

**Note 118-2 (2026-10-05 04:20 EDT, before any §118 result; Audit). The control already passes in one seed (§111 READ s8, 115-2), so a
single passing seed in ANCHOR or LONG shows nothing new.**
* **EXIST is read as uninformative** for the arms.
* **An arm's claim requires NOSNAP-512 in ≥ 4/5** (as registered), **and** a seed rate above the control's final rate
  over READ s6–10 (s6–8: 1/3; s9–10 pending).
* If the control itself reaches ≥ 4/5, the reader alone is the no-snap fix, and the arms are read only on NOSNAP-2048 and
  drift.

**Note 118-3 (2026-10-05 05:33 EDT; a descriptive diagnostic, $0, Audit's idea; predictions fixed before the full run, with a smoke peek
disclosed). Is the hand-off map an attractor, and does that explain why READ s8 runs without snaps while s6 / s7 drift?**
* **Script:** `theory/contraction_diag.py`. At in-range position T0 = 4 of a no-snap chain, the carried latent is
  perturbed by ε·‖z‖ in a random direction, and clean and perturbed chains run 16 more hand-offs on the same ops.
  - ρ: the median per-hand-off distance ratio, geometric mean over the first 8.
  - Decode agreement per hand-off.
  - Basin: the largest ε with agreement ≥ 0.99 after 8.
  - Models: READ s6–10, §108 MK-4 s1 / s3 / s5.
* **Smoke peek (20 chains, K = 4, disclosed):** s8 has ρ ≈ 0.22 and basin 0.5; s6 has ρ ≈ 0.58 and basin 0.03. Both
  contract in distance, but s6's decodes break at a far smaller perturbation.
* **Predictions (fixed now, for the full run at n = 500, K = 16).**
  - (a) Among READ s6–10, the seeds passing no-snap (s8, and any of s9 / s10) have the **largest basins (≥ 0.3)**. The
    drifting seeds have basins ≤ 0.1.
  - (b) ρ < 1 holds broadly. ρ alone does not separate success from drift: the decode margin (basin) does.
  - (c) §108 seeds (no reader) have smaller basins than every no-snap-passing READ seed.
* **Scoring split (Audit, fixed before the full run).** The split in (a) was chosen after seeing s6 and s8, so they are
  reported as the **calibration pair** only. (a)–(c) are confirmatory only on the held-out models: READ s7, s9, s10 and
  §108 s1 / s3 / s5.
* **Implication if (a) holds.** No-snap success needs a wide decode basin around the state code, not just contraction.
  - That favours objectives that **widen margins**: noise injection on the carried latent (P-B's denoising), or the
    discrete snap (§120), over plain norm control.
  - If (a) fails, drift is not a local-stability property, and P-B drops behind §120.

**Note 118-4 (2026-10-05 08:55 EDT; descriptive, the first three ANCHOR seeds; the verdict waits for the extension). ANCHOR removes the
drift: 3/3 seeds are fully latent and exact to 2,048 with no snaps.**

| seed | no-snap 128 / 512 / 2,048 | cosine to snapped (pos 448) | donor min / recipient max | effective rank | anchor cosine |
|---|---|---|---|---|---|
| s71 | 1.0 / 1.0 / 1.0 | 0.9998 | 1.0 / 0.011 | 48 | 0.9998 |
| s72 | 1.0 / 1.0 / 1.0 | 0.9998 | 1.0 / 0.011 | 83 | 0.9998 |
| s73 | 1.0 / 1.0 / 1.0 | 0.9982 | 1.0 / 0.011 | 122 | 0.998 |

* The snapped guard holds (1.0 at 512). The disagreement trigger only ever fires its forced cap (7 snaps per 512, no
  disagreement).
* **Against the READ control:** 1/4 so far (s8 passes; s6 / s7 / s9 drift; s10 pending).
* **Extension fires** (≥ 2/3 of the first three). Seeds 74–75 run. NOSNAP-512 needs ≥ 4/5 and a rate above the
  control's (118-2).
* **Scope:** "fully latent at inference, text-anchored in training" (the anchor uses the state's text in training;
  per-step supervised, not C4, not task-agnostic).

**Note 118-5 (2026-10-05 09:15 EDT; descriptive). 118-3's contraction diagnostic is scored on the held-out models, and the READ control
rate is final.**
* **READ control, no-snap** (§115 (A), serial): s6 0 / s7 0 / **s8 1.0** / s9 0.87 at 512 (0.55 at 2,048) / **s10 1.0
  (0.998 at 2,048)**. The control rate is **2/5**, so ANCHOR must reach ≥ 4/5 (118-2).
* **Held-out scoring of 118-3** (READ s7 / s9 / s10, §108 s1 / s3 / s5; s6 and s8 were the calibration pair).

  | model | ρ (ε 0.1) | basin | no-snap 512 | prediction (a) |
  |---|---|---|---|---|
  | READ s10 | 0.25 | 0.5 | 1.0 | ✓ (pass, basin ≥ 0.3) |
  | READ s7 | 0.75 | 0.0 | 0 | ✓ (drift, basin ≤ 0.1) |
  | READ s9 | 0.38 | 0.3 | 0.87 | ✗ (drifts with basin 0.3) |
  | §108 s1 / s3 / s5 | 0.99 / 0.87 / 0.82 | 0.0 | 0 | (c) ✓ |

  - (a) holds in 2 of 3 held-out READ seeds.
  - **(b) is refuted:** ρ separates as well as basin does.
  - (c) holds.
* **Reading.** No-snap success is graded with attractor strength. ρ ≈ 0.25–0.28 with basin 0.5 passes (s8, s10); ρ 0.38
  with basin 0.3 drifts mildly (s9); ρ ≥ 0.66 drifts (s6, s7, §108).
  - The boundary lies between ρ 0.28 and 0.38. Within READ this is correlational; §123 tests it causally.
  - **§118 ANCHOR s71–73 (2026-10-05 10:14 EDT):** ρ = 0.191 / 0.188 / 0.194 at ε 0.1, the strongest attractors measured, with basins
    0.2 / 0.1 / 0.1, all no-snap exact.
  - So **ρ, not basin, tracks no-snap success** across both READ and ANCHOR: every model with ρ ≤ 0.28 passes, every
    model with ρ ≥ 0.38 fails. The anchor works by making the hand-off strongly contractive.
  - **Floor check (Audit / claude-13; contraction_diag ran in fp32, so the bf16 issue of 123-5 does not apply).** In ANCHOR
    the distance falls below 5e-5 by k = 4–5, so the late ratios are noise and the k = 1..8 geometric means are upper
    bounds.
    - ρ over k = 1..5 at ε 0.1: ANCHOR 0.09–0.18; READ s8 / s10 0.23 / 0.24; s9 0.35; s6 / s7 0.59 / 0.64; §108 0.73–0.88.
    - The ordering and the boundary (pass ≤ 0.24, fail ≥ 0.35 on k = 1..5) are unchanged.

## 119. Controlled baselines for C1: published latent-CoT methods on our S5 protocol, scored with the donor and length protocols (pre-registered 2026-10-04 23:14 EDT, after Literature's checks 2bf76954 (project audit, missing comparison 2) and 1a473f82, and before any implementation or run; Audit's guards)

**Why.** Our C1 claim compares against the literature only on paper: no latent-CoT paper shows serial,
donor-consistent use. §119 makes it a controlled claim. Published methods are trained on our S5 data and scored with
§104's donor protocol and our length protocol.
* **Prior** (1a473f82, `docs/LIT_S119_BASELINES_2026-10-04.md`): Coconut 2412.06769, CODI 2502.21074, SIM-CoT
  2509.20317. Thinking States 2602.08332 is an optional fourth arm. Related: KaVa, CoLaR, PCCoT, System-1.5, LIFT, SELR,
  MUX.
* Literature (3): §105 MASK ≈ "SIM-CoT with a tied, in-context decoder, one latent per op, plus snaps". §97 GEN differs
  decisively, because its decoder sees the context.

**Methods** (GPT-2 small, the official recipes adapted to S5; Literature (1)).
* **Coconut.** The question is visible and thoughts attend to the full prefix. Stage s replaces s steps with s·c
  thoughts, with CE on the remaining text and the answer, and an optimiser reset per stage. c = 1, 4 stages per chunk,
  from the base model.
* **CODI.** One shared model, α = β = γ = 1. An L1 loss (stop-gradient teacher) on the hidden state of a fixed
  pre-answer token, across all layers, per-layer std-normalised. 4 thoughts, an MLP + LN projection, LoRA r128 / α32 as
  published. Optionally, a full fine-tuning variant (reported separately).
* **SIM-CoT.** Coconut with c = 2, initialised from the trained S5 Coconut checkpoint. A **separate** decoder copy
  (φ ≠ θ), conditioned on z_k only, generates step k's state text (λ_step from the official code defaults, recorded).
  The decoder is removed at inference.

**Hosts (Literature (2)).**
* **Chunked (primary).** §97 / §108's harness: the chunk start state as text, 4 ops, the chunk's final state as text,
  restarts from the decoded answer every 4 ops at test (cumulative 512 / 2,048). The only host that reaches 512 / 2,048.
* **Native (descriptive).** One sequence with no restarts, trained up to 16 ops. Length curve 4–64 (context-limited),
  with the donor protocol in range.

**Fairness.** Equal small lr sweeps per method (3 values, chosen on a training-split final-answer criterion).
Trainable parameters and updates are reported. 3 seeds per method on the chunked host, 1 per method native.

**Guards (Audit, fixed now).**
* **(a) Competence gate.** A chunked seed is scored for SERIAL / BYPASS only if its in-chunk final-state accuracy
  (§104's intact accuracy at the chunk's last slot, random starts) is ≥ 0.90. Otherwise it is reported as **"did not
  learn"**, not "not serial".
  - A method counts toward CONTROLLED-C1 only with ≥ 2 competent seeds; otherwise "inconclusive".
* **(b) Faithfulness.** Each method's port must first reproduce one published number with our implementation.
  - Coconut: our ProsQA run of the official recipe gives 97.6, against the published 97.0 (bench, 0ef4a78).
  - CODI and SIM-CoT: the released GPT-2 checkpoints' published GSM8K numbers, through our inference port; or a
    published training curve if there is no checkpoint (to within ±2 points).
  - A method failing (b) is reported as "port unverified" and excluded from CONTROLLED-C1.
* **(c) Wording.** READ's serial use includes the Markov bottleneck by design. §108 without the reader was already
  SERIAL in every seed that learned. The comparison shows that existing methods do not give serial latent use on this
  protocol. It does not show that the reader alone causes it.

**Scoring.**
* **Coconut and SIM-CoT:** §104 single-slot donor at latent positions 1–3. For SIM-CoT, slot j means op j's 2 thoughts
  patched jointly. SERIAL = USE ≥ 0.90 at every slot, recipient < 0.10; BYPASS = recipient ≥ 0.10.
* **CODI:** its thoughts are not aligned to ops, so per-step SERIAL is undefined. It gets a **block donor** test
  (every thought patched): donor- vs recipient-consistent final answer. Descriptive, excluded from CONTROLLED-C1's
  per-step claim.
* Removal (zero / mean) and the length curves are descriptive.

**Rule (fixed now).** **CONTROLLED-C1:** among Coconut and SIM-CoT (each faithful, with ≥ 2 competent seeds), **neither
is SERIAL in ≥ 2/3 of its competent seeds**, while §111 READ is SERIAL in 5/5 (111-4).

**Readings.**
* **CONTROLLED-C1 holds.** Published latent-CoT methods, trained on the same data and scored the same way, do not carry
  the S5 state serially through their latents (bypass or neither). Serial use on this protocol needs a forced
  bottleneck (§108) and is made reliable by the reader (§111). The literature comparison becomes a controlled one.
* **A baseline is SERIAL.** Our C1 is not unique to the bottleneck design. Report it and re-scope the claim.
* **Inconclusive** (a method did not learn or its port is unverified). Report it; no claim either way for that
  method.

**Placement.** About 9 chunked + 3 native runs + lr sweeps + faithfulness evals, about $20–30 on Vast. **Queued behind
§117 / §118.** Thinking States is an optional fourth arm if the budget allows (noted then). Implementation: the
experiments session. Scorer `theory/score_s119.py`, committed before any run. Projection to Audit.

## 120. A learned discrete snap at every step, with no template and no schedule: FSQ digital recurrence on S5 and flip-flop (pilot) (pre-registered 2026-10-05 00:38 EDT, after Literature's checks b284f980 / 46fe5ca0 and before any implementation or run; the user's no-snap directive, P-A in `docs/THEORY_AUTOSNAP_2026-10-05.md`; spend approved by Audit)

**Why.** Every success so far snaps to a task-specific state text on a hand-set schedule. The user wants neither.
* Without snaps the latent drifts in direction (115-1).
* **The carried latent is not state-clustered even in range** (`theory/autosnap_premise.py`). On §111 READ s6, a
  per-state centroid codebook fitted on in-range latents classifies held-out in-range latents at 3–5 % (chance 0.8 %),
  while the reader decodes the state at about 1.0. A post-hoc snap cannot work; the discretisation must shape the code
  in training.
* P-A makes every hand-off a learned, task-agnostic discrete snap. If each step is exact on the reachable codes, the
  chain is a learned finite-state machine: exact at any length, and certifiable by enumeration.
* **Prior** (46fe5ca0): FSQ 2309.15505; closest SR-RNN 1901.08817 and QBN 1811.12530 (from scratch or post hoc). VQ in
  latent CoT quantises text spans, not the carried state. New as scoped.

**One harness (the same code and hyperparameters for both families).**
* **Markov segment per input item** (the dataset's own item unit): [EOT, carried state embedding, item tokens] →
  last hidden h (fresh cache, positions from 0, as §108).
* **The FSQ snap (fixed now).**
  - z = W_down(LN(h)), d = 8; z_b = tanh(z) · (L − 1) / 2 with L = 8 levels (half-integer offset as FSQ for even L);
    q = round(z_b).
  - Straight-through: q_st = z_b + stop-grad(q − z_b). The carried state embedding is W_up(q_st) (8 → 768) plus a
    learned bias.
  - Code space L^d = 8⁸ ≈ 1.7·10⁷, against minimal automata of 120 (S5) and 2 (flip-flop).
  - **Rounding schedule:** soft (q = z_b) for steps 0–2k; then each hand-off is hard with probability rising linearly
    0 → 1 over 2k–4k; hard from 4k on. Test is always hard.
* **Start:** the start state as text at position 0, the only text input (S5: a random given start arrangement, as
  §110b; flip-flop: none).
* **Answer decoder:** a fresh segment [EOT, the final carried state embedding] → the answer text through the tied head
  (S5: the 6-token state text; flip-flop: the bit token).
* **Training:** lengths drawn uniformly from 1..L_max with L_max = 2 / 4 / 8 / 16 / 32 from steps 0 / 2k / 5k / 9k /
  14k. Full backpropagation through the chain. 24k steps, bs 64, AdamW lr 1e-4 (FSQ projections included), warmup 100
  + cosine.

**Families and long test lengths (fixed now).**
* **S5** (group): DeltaProduct's protocol test set (2,000 × 512, identity start as text), scored as cumulative accuracy
  at **512**. Descriptive: 2,048 (500 sequences).
* **Flip-flop** (2306.00946; non-group, with reset). Instructions w0 / w1 / i0 / i1 / r with p_ignore = 0.8 (p_w = p_r =
  0.1); the first instruction is a write and the last a read. The answer is the bit last written.
  - Test: 2,000 sequences of length **512** at p_ignore = 0.8, scored as accuracy at every read.
  - Descriptive: sparse 0.98 and dense 0.1, at 512.

**Arms (2 seeds each per family; 8 runs).**
* **FINAL (primary):** only the final answer is supervised (S5: the final state; flip-flop: the last read's bit).
* **TRACE:** plus the answer decoder applied at every position where the dataset has a native label (S5: the state after
  every op; flip-flop: every read's bit), CE weight 1.
  - Pre-registered wording: **"task-agnostic architecture, per-task supervision"**. The native trace is supervision,
    not a snap template (Literature 46fe5ca0 (e)4); it is never fed back.

**Measures.**
* Long-length accuracy (above), and accuracy against length (16 … 512).
* **Code usage:** perplexity of codes over test chains, and the number of distinct codes.
* **Codes per state:** distinct codes visited per true state on test chains.
* **CERT (reachable-table certificate).**
  - BFS from every start code (S5: the 120 start texts; flip-flop: the empty start) over every item (S5: 120 ops;
    flip-flop: 5 instructions), until closure or a cap of 10⁶ codes.
  - Decode every reachable code with the answer decoder. CERT holds if every code decodes to a valid state and, for
    every reachable (code, item), decode(next) = δ(decode(code), item).
  - Moore minimisation gives the number of distinct behaviours, against 120 / 2.
* **C1:** single-slot donor patching of the carried code at positions 2, 4 and 6 of length-8 chains (§104 rules: USE ≥
  0.90, recipient < 0.10).
* Code drift: none by construction. The number of hand-offs where the rounded code changes under ±0.5 noise on z_b
  (the margin) is reported.

**Rules (fixed now).**
* **PILOT-PASS (per family × arm):** long-length accuracy ≥ 0.99 in **both** seeds.
* **P-A pilot reading:** FINAL passes PILOT-PASS in **both** families, the directive's minimum of ≥ 2 families incl. ≥ 1
  non-group, with the same code and hyperparameters. A TRACE-only pass is reported as "per-task supervision".
* **CERT** is reported per seed. A passing seed with CERT is "certified at every length".
* **Extension:** a passing family × arm gets seeds 3–5, plus cycle navigation (Delétang 2207.02098; train lengths 1–40,
  test 41–500 at **500**) and 4×4 multiplication. That is a new note before launch, with spend posted to Audit.

**Readings.**
* **FINAL passes in both families (with CERT).** A pretrained LM learns, from final answers alone, a discrete latent
  recurrence that is exact at 512 with no text snaps and no schedule, on a group and a non-group task, with one design.
  The snap is learned and task-agnostic. Tags: C2 (length), C5 (no hand-set schedule: every step snaps to a learned
  code), C1 (donor), C7 (codes per state); C6 at extension.
* **Only TRACE passes.** The discrete recurrence works but needs per-step supervision on these tasks.
* **Fails with high in-range accuracy.** The codes generalise poorly in length: report codes per state and margin.
  P-B (continuous stabiliser) or a larger d / L is next.
* **Fails in range.** The straight-through estimator or the no-read basin blocks learning. Report code usage and the
  shuffled-null ratio.

**Placement.** 8 runs, about $10 on Vast, approved by Audit. It goes ahead of §119 (credit floor). Implementation: the
experiments session (a new module with both families, the FSQ snap and the CERT enumerator). Scorer
`theory/score_s120.py`, committed before any run. Projection to Audit.

**Note 120-1 (2026-10-05 00:39 EDT, before any §120 run; Audit).** FINAL's curriculum draws lengths 1..L_max, so length-1 and length-2
problems make up about 50 % of early updates and about 1/L_max later. On S5 they start from **random** arrangements, so
their final answers label the transition table directly, at far more than §110's ≤ 5 %.
* **C4 is not claimed for §120.** A FINAL pass reads "final-answer supervision with a short-problem curriculum (C4
  qualified: length-1–2 problems label transitions)".
* The directive concerns snaps, not C4, so the design is unchanged.
* **Logged:** coverage_drawn, the share of the 14,400 S5 transitions (s, a) labelled by length-1 and length-2 training
  problems, as §112. For flip-flop, the same share over its (state, instruction) pairs.

**Note 120-2 (2026-10-05 00:51 EDT, before any §120 run; the experiments session's implementation in `boxes/fsq120.py`, accepted).**
* **FSQ:** z_b = tanh(z + atanh(0.5 / 3.5)) · 3.5 − 0.5, so q ∈ {−4, …, 3}: FSQ's even-L form with 8 levels.
  - In the 2k–4k ramp, one Bernoulli(p_hard) coin per hand-off, batch-wide.
  - The flip-flop start is W_up(0) + bias.
* **Items:**
  - S5: an op's 6 tokens, trained from uniform random start texts and tested from the identity text.
  - Flip-flop: 2-token items (" write 0/1", " ignore 0/1", " read."), with the answer " 0" / " 1". Its automaton has 3
    states (none, last bit 0, last bit 1).
* **long_acc (flip-flop)** is the mean read accuracy. long_acc_seq (every read right) is reported too.
* **CERT** starts from all first-step codes (S5: 120 start texts × 120 ops; flip-flop: the empty start × 5 items) and
  runs BFS over every item, capped at 10⁶.
  - It holds iff every code decodes validly, the transition law holds on every pair, and nothing was capped.
  - n_moore comes from partition refinement on (decode, successors).
* **donor:** random-start length-8 chains, single-code patch at positions 2 / 4 / 6.
* **margin_flip_rate:** 500 test chains × the first 128 hand-offs, with U(−0.5, 0.5) noise on z_b.
* **coverage_drawn** (120-1): (s, x₁) from every drawn length-1 and length-2 problem, plus (δ(s, x₁), x₂) from length-2
  problems.

## 121. Breaking multiplication's carrier shortcut: a text-path anchor on the carried latent, against the reader alone (pre-registered 2026-10-05 02:15 EDT, after Literature's check c9651bff and before any implementation or run; the user's step-3 directive via Audit; spend pre-approved by Audit)

**Why (114-4).** On 4×4 multiplication under Markov segments, the carried slot-2 latent holds the **multiplier prefix**,
not the running sum.
* Linear probes read b_1 / b_2 at 1.0, while S_2's middle digits sit at chance.
* Each segment recomputes a × prefix: 4×2 at step 2 works, 4×3 at step 3 fails (0.24). Step-3 errors do not track
  carries.
* A causal donor smoke (`theory/mult_donor_diag.py`, n = 40) agrees. Same prefix with a different sum leaves the step-3
  output recipient-like (0.15, intact 0.225) with no sum-donor answers (0.0); same sum with a different prefix never
  stays recipient-consistent (0.0).
* So width, pauses and looped depth could not help (§107, §114): nothing forces the sum into the carrier.
* **Prior** (c9651bff, `docs/LIT_S121_CARRY_2026-10-05.md`).
  - Input-prefix carriage in latent multiplication chains: not documented (new as a diagnosis). 2510.00184's ICoT
    model recomputes pairwise partial products, and a running-sum auxiliary unlocks it. Liang & Pan 2602.00449 (late
    fusion).
  - **Literature's design point:** the reader alone may not force the sum. A latent carrying (a, b_1, b_2) lets
    [EOT, latent] decode S_2 in one pass. The lever is computation: tie the latent to a state code that cannot contain
    the prefix.

**Base.** §99 MARKOV on 4×4, exactly: Markov segments [prefix a, slot j−1's latent (or the chunk start text), op j] →
slot j; K-CURR (k = 1 to 2k, 2 to 6k, then 4); 24k cap; eligibility from 6k; the valid-split early stop; bs 64, lr 1e-4.

**Arms.**
* **ANCHOR+READ (primary; seeds 1–3).**
  - The reader (§111): at every slot whose latent is carried, a fresh segment [EOT, carried latent] decodes the
    dataset's step-j text (CE, weight 1), from step 0.
  - **The anchor (§118's form):** at every latent-fed segment j, the text path [prefix a, gold step j−1 text, op j]
    (the chunk-start format, which contains S_{j−1} and no prefix code) gives carried latent b, with stop-gradient.
    L_anchor = (1 − cos(a, b)) + |log‖a‖ − log‖b‖|, weight 1, where a is the latent path's carried latent.
* **READ only (seeds 1–2; it tests Literature's prediction):** the reader without the anchor.
* **Control:** §99 MARKOV s1–3 (existing).

**Measures.**
* §99's: per-slot test decode, test answer accuracy, removal.
* §104's random-donor protocol at slots 1–3.
* **MECH, the shortcut-blocking donor test** (`theory/mult_donor_diag.py`, n = 300, Audit (a)):
  - SAMEPREFIX donors (same b_1 b_2, different a, so a different S_2);
  - SAMESUM donors (a_D · m_D = a_R · m_R, with a different prefix);
  - S_3 scored as recipient-, sum-donor- or prefix-donor-consistent.
* **Probes on the slot-2 latent** (descriptive; Literature): S_2 digits up, b-prefix down, and an **a-probe** (a
  decodable while S is not is the input-carriage signature).
* The anchor cosine at test.

**Rules (fixed now).**
* **CARRY-3 (ANCHOR+READ):** op-3 test decode ≥ 0.95 in ≥ 2/3 seeds.
* **ACC:** test answer accuracy ≥ 0.95 in ≥ 2/3.
* **C1:** §104 donor SERIAL (USE ≥ 0.90 at slots 1–3, recipient < 0.10) in the passing seeds.
* **MECH:** in the passing seeds, SAMEPREFIX → sum-donor-consistent ≥ 0.80 **and** SAMESUM → recipient-consistent ≥ 0.80
  (the sum-carrier signature).
* READ-only is reported descriptively against the same thresholds.
* **Claim wording (fixed):** "C1 on arithmetic with per-step text supervision (not C4)".

**Readings.**
* **ANCHOR+READ passes (CARRY-3 + MECH), READ only does not.** The carrier shortcut is broken by tying the latent to the
  text-path state code, not by readability. One mechanism (§118's anchor) serves no-snap stability and the arithmetic
  hand-off.
* **Both pass.** The reader alone forces the sum (Literature's trap did not bite).
* **Neither.** The anchor does not transfer the text-path code under K-CURR. Finer step granularity (one partial-product
  digit per item) is next.

**Placement.** 5 runs, about $6–7 on Vast (Audit pre-approved about $4 for 3 seeds; the 2 READ-only seeds are posted to
Audit as about $2.5 extra), proof seed first (ANCHOR+READ s1). Implementation: the experiments session (accel99 markov
+ `--reader` + `--anchor`). Scorer `theory/score_s121.py`, committed before any run.

**§121 result (2026-10-05 07:02 EDT; 5 runs by the experiments session; scored by Theory alone with `theory/score_s121.py`, committed
before any run; T1, probes and MECH from `theory/mult_step3_diag.py` / `theory/mult_donor_diag.py`). CARRY-3 is not met:
the anchor did not break the shortcut. §122 triggers.**

| run | op-3 decode | test | anchor cosine | T1 (step 3 from the true step-2 text) | S_2 probe, digits 2–4 |
|---|---|---|---|---|---|
| ANCHOR+READ s1 | 0.231 | 0.031 | 0.98–0.99 | **0.0** | 0.13 / 0.12 / 0.18 |
| ANCHOR+READ s2 | 0.869 | 0.203 | 0.98–0.99 | **0.0** | 0.13 / 0.13 / 0.17 |
| ANCHOR+READ s3 | 0.274 | 0.044 | 0.98–0.99 | **0.0** | 0.10 / 0.12 / 0.15 |
| READ s1 (control) | 0.643 | 0.078 | — | about 0 | at chance |
| READ s2 (control) | 0.927 | 0.219 | — | 0.0 | at chance |

* **CARRY-3: 0/3** (primary); ACC 0/3. READ-only reaches 0.95 in neither seed. So C1 and MECH are moot.
* **Reading ("neither").** The anchor binds (cosine about 0.98–0.99), but to a text path that no longer computes. T1 is
  0.0 in every seed: text-start segments are untrained after K-CURR's k ≤ 2 phase (121-2).
  - In every model the carried slot-2 code stays the multiplier prefix or an entangled code (121-3), never S_2.
  - The reader raises op 3 (to 0.93 in one seed) by better recomputation, not by carrying the sum.
* **§122's trigger is met** (CARRY-3 fails, and T1 ≤ 0.10 in all failing seeds). TEXT-MIX keeps the text path
  competent. §122 is the last multiplication patch before its stop rule.

**Note 121-1 (2026-10-05 02:30 EDT, before any §121 result; the experiments session's implementation 903b74f1, accepted).**
* accel99 MARKOV with K-CURR as §99.
* **Reader:** at every carried slot (j < c − 1), a fresh segment [EOT, carried latent] decodes step j's text (CE, weight
  1).
* **Anchor:** at every latent-fed segment, (1 − cos) + |log-norm gap| against the stop-gradient text path [a, gold step
  j−1 text, op j].
* **eval.json:** §99's keys, plus donor slot1–3 (random donors, n = 2,000), reader_acc per carried slot, and the anchor
  cosine and log-norm gap by position against the true text path. model.pt is kept for the MECH donor test and the
  probes.
* **Order:** proof seed ANCHOR+READ s1 first. Then ANCHOR+READ s2–3 and READ s1–2, after Audit's gate (a sane 1k-step
  log with all three losses, peak memory, an eval-path probe).

**Note 117-3 (2026-10-05 02:59 EDT, before any §117 run; the dose, as 117-2 required). §112 meets FOOTHOLD-0.1 locally**
(`theory/score_s112.py`): the passing seeds are [21, 23] (cumulative 512 0.996, 0.9985; never-labelled k = 1 1.0, certified).
* **§117's dose is fixed at f = 0.1**: §112's T_f (112-1 option (b), with the drawn coverage logged).
* The claim wording becomes "C4 qualified by 10 % identity-start short coverage".
* The §116 launch condition is also met. §116 launches (OPDEF, NONE, NONE+COPY; 116-1 to 116-3).
* The §112 result follows when f = 0.1 s24 lands.

**Note 117-4 (2026-10-05 03:30 EDT, before any §117 or §116 result; a conditional follow-up fixed in advance, Audit's proposal; no
spend until it triggers). §117-COPY.**
* **Trigger:** §116's MECH passes (NONE+COPY acquires in ≥ 4/5, and in more seeds than NONE; 116-2).
* **Design:** §117's recipe at f = 0.1, plus 116-2's copy rows. 10 % of every batch's rows are length-0 copy problems:
  a random start text and zero ops, the slot read from the start text, decoded to the start state. Self-supervised;
  no transition labels.
  - Under Markov segments the copy slot is the carried state of the zero-op segment [EOT, start text], read by the
    reader / fork.
* **Arm:** the §117 arm with more ALL seeds among its first three (MK if tied), seeds 56–58 (MK) or 66–68 (ST).
* **Rules:** §117's ALL / EXIST / INTEGRATED and C6 rules unchanged, with the wording "C4 qualified by 10 % short
  coverage, plus a self-supervised copy objective".
* **Placement:** 3 runs, about $2–3. Literature's copy check (15cf541a) applies; a one-line confirm is requested at
  trigger time.
* §121 is not included: it is step-supervised with text-start K-CURR chunks, so 112-4's start-reading basin does not
  arise there.

**Note 121-2 (2026-10-05 04:39 EDT; descriptive, ANCHOR+READ s1 only, before the panel is scored; rules unchanged). The anchor bound,
but to an incompetent target.**
* **s1.** Op-3 decode 0.231 and test 0.031 at the 24k cap.
  - The anchor did bind: test cosine to the true text path 0.98–0.99 at positions 1–3, log-norm gap about 0.03.
* **The anchor's target has stopped working.** Restarted from the **true** step-2 text (T1, `theory/mult_step3_diag.py`),
  step 3 is 0.0 and even P_3 is right only 0.125.
  - In 4×4 at k = 4 every test chunk starts from no text. Text-start segments are trained only in K-CURR's k ≤ 2 phase
    (≤ 6k).
  - So by 24k the text path [a, step j−1 text, op j] no longer computes the step, and the latent is tied to an
    incompetent code.
* **The carrier is unchanged.** Slot-2 probes: b_1 / b_2 1.0, S_2's middle digits 0.12–0.19 (chance).
  - The causal donor test is now mixed: SAMEPREFIX recipient 0.23 vs sum-donor 0.18; SAMESUM recipient 0.19 vs
    prefix-donor 0.22, against MARKOV's clean prefix signature.
* **Reading (provisional, one seed).** The anchor transfers whatever code the text path writes. The text path must itself
  stay competent (trained throughout) for the code to carry S_j.
  - §100's k = 2 model shows the text-start step is learnable in one pass (T1 = 1.0, 114-4).
  - Experiments' reading ("the failure is the segment's 4×3 computation") does not fit T1 here: with the true S_2 text
    available, a competent segment needs only a 4×1 multiply-add.

## 122. The anchor needs a competent target: TEXT-MIX keeps the text path trained (conditional; the last multiplication patch before the stop rule) (pre-registered 2026-10-05 04:40 EDT, after Literature's check f7b7a467 and before any implementation or run; Audit-approved, conditional)

**Why (121-2).** In §121 ANCHOR+READ s1 the anchor bound (cosine 0.98–0.99 to the text path), but its target was
incompetent.
* Restarted from the **true** step-2 text, step 3 is 0.0 (P_3 0.125). Text-start segments are trained only in K-CURR's
  k ≤ 2 phase, so by 24k the text path no longer computes, and the latent is tied to a dead code.
* §100's k = 2 model shows the text-start step is learnable in one pass (T1 = 1.0, 114-4).
* **Prior** (f7b7a467, `docs/LIT_S122_TEXTMIX_2026-10-05.md`): CODI 2502.21074 is the direct precedent (the shared
  teacher keeps its own CE throughout, γ = 1). Professor forcing 1610.09038; BYOL / SimSiam (a target detached from
  task learning drifts); ICoT-KD.

**Trigger (fixed now).** §122 runs only if §121 ANCHOR+READ fails CARRY-3 **and** T1 (step 3 from the true step-2 text)
is ≤ 0.10 in its failing seeds. Otherwise it is not run (noted then).

**Design.** §121 exactly, plus **TEXT-MIX:** 25 % of every batch's rows, throughout training, are k = 1 chunks from
the **true** previous step text at a uniformly drawn position j ∈ {2, 3, 4}, with the text path's own CE on step j's text.
* **TEXTMIX+ANCHOR+READ (primary; seeds 1–3).**
* **TEXTMIX+READ (control, no anchor; seeds 1–2; Literature (3)).** A pass is credited to the anchor only if this
  control fails.
* **Logged every 1k steps:** T1 (step 3 from the true step-2 text) and the text path's step-j accuracy, so the anchor's
  target is shown competent.

**Rules (fixed now).**
* §121's CARRY-3 and ACC (≥ 2/3 in the primary arm), C1 donor SERIAL, and MECH (the SAMEPREFIX / SAMESUM sum-carrier
  signature ≥ 0.80) in the passing seeds.
* **Attribution:** "the anchor breaks the shortcut" requires the primary arm to pass and the control to fail CARRY-3
  (op 3 < 0.95 in both seeds). If both pass, TEXT-MIX alone suffices.
* **STOP RULE (Audit):** if the primary arm fails CARRY-3, **the latent-carriage line on multiplication pauses**, with
  no further patch. Multiplication then proceeds only through the task-agnostic route (§120's FSQ extension, which plans
  4×4 as a family).
* **Scope, fixed now.**
  - A pass shows that anchoring to a competent text-path code breaks the carrier shortcut: C1 on arithmetic with
    per-step supervision (TEXT-MIX feeds gold previous states), on 4×4 at k = 4.
  - It does **not** show length generalisation (5×5, streaming), C4, or no-snap carriage on arithmetic.

**Placement.** 5 runs, about $6–7 on Vast (Audit approved about $4 for the primary, plus the control on Literature's
advice), proof seed first, **only on the trigger**. Implementation: the experiments session (accel99 `--textmix 0.25`).
Scorer `theory/score_s122.py`, committed before any run.

**§122 result (2026-10-05 14:15 EDT; 5 runs by the experiments session; scored by Theory alone with `theory/score_s122.py`, committed
before any run). CARRY-3 is not met (0/3); the STOP RULE fires (122-4).**

| arm | op-3 decode (s1 / s2 / s3) | test | t1 (step 3 from the true step-2 text) |
|---|---|---|---|
| TEXTMIX+ANCHOR+READ | 0.362 / 0.408 / 0.324 | 0.047 / 0.053 / 0.051 | 1.0 / 1.0 / 1.0 |
| TEXTMIX+READ (control) | 0.246 / 0.344 | 0.016 / 0.047 | 1.0 / 1.0 |

* TEXT-MIX made the anchor's target competent everywhere (t1 = 1.0), so the question was tested. The anchor does not
  break the carrier shortcut; attribution is moot.
* **Reading** (121-2, 121-3, 122-2, 122-3).
  - In every Markov multiplication model the carried code is the multiplier prefix, or an entangled code, never the
    running sum.
  - The anchor binds the shared component, not the state: centred binding is weak (0.74, own-match 24 %). The single-step
    text path's latent is never consumed downstream, so it need not carry the sum.
* **The multiplication latent-carriage line pauses (stop rule).** Multiplication continues only through the task-agnostic
  route (§120's extension and §123).

**Note 122-1 (2026-10-05 05:01 EDT, before any §122 run; the experiments session's implementation 94beeb42, accepted; not launched).**
* `accel99 --textmix 0.25` with `--arm99 anchorread` (primary) or `read` (control).
  - 25 % of every latent-phase batch's rows are k = 1 chunks from the true step j−1 text, j uniform in {2, 3, 4}.
  - CE on step j's text, plus the answer at j = 4, row-weighted with the main chunk.
  - Neither the reader nor the anchor fires on those rows (no carried latent).
* text_acc {2, 3, 4} and t1 = text_acc["3"] are logged every 1k steps on the valid split, and in eval.json on test.
  Logging only.
* If triggered: proof seed first, with §121's GPU memory and eval probe.

**Note 120-3 (2026-10-05 05:42 EDT, before any §120 result is read; CERT scheduling, Audit's request).** CERT's BFS on b1 ran over 2 h and
blocked the queue, so it now runs post hoc.
* The main eval uses `--no_cert`. A separate `--cert_only` job writes `<run>/eval_cert.json` under a 45-minute limit.
* Status "done" carries holds true / false as registered.
* "inconclusive_cap" (10⁶ codes) or "inconclusive_time" mean **no certification claim** for that seed, the same as
  120-2's "capped ⇒ does not hold". They are not a training failure, and PILOT-PASS (accuracy) is unaffected.
* A long BFS is itself informative: many reachable codes per state. n_reachable and codes per state are reported.
* `theory/score_s120.py` reads eval_cert.json.

**Note 121-3 (2026-10-05 06:19 EDT; descriptive, the READ-only arm, before the panel is scored). The reader improves step 3 but does not
make a sum carrier.**
* **Step 3:** READ s1 0.643, s2 0.927, against §99 MARKOV's 0.24–0.27 and ANCHOR+READ s1's 0.231. Test answers 0.078 /
  0.219. The answer still needs S_3 carried into slot 4.
* **Probes on the slot-2 latent** (both seeds): b_1 / b_2 1.0, S_2's middle digits 0.11–0.20 (chance). Linearly, still
  the multiplier prefix.
* **Causal donor test** (S_3 digits 2–4).
  - s1 is the prefix-carrier signature: SAMEPREFIX recipient 0.30 vs sum-donor 0.14; SAMESUM prefix-donor 0.28 vs
    recipient 0.11.
  - s2 is **neither**: intact agreement 0.97, but every patched hypothesis is 0.20–0.26. The carried code is
    problem-specific and entangled; it transplants as neither S_2 nor the prefix.
* **T1** (step 3 from the true step-2 text) is about 0 in both: the text path is dead here too (121-2).
* **Reading (provisional).** The reader made the segment better at recomputing from the carried code (s1's digit 2 is
  0.815, against MARKOV's 0.38). It did not install S_2 as the carried quantity.
  - Literature's prediction (c9651bff: the reader can be satisfied without forcing S) holds for the carrier, even though
    accuracy rose.
  - C1 on arithmetic in §121's sense (MECH) is not shown by READ-only.

**Note 120-4 (2026-10-05 06:22 EDT; descriptive, the first §120 cell, before the pilot is scored). S5 FINAL s1: a learned discrete snap
at every step, with no text snaps and no schedule, is exact to 512 and serial.**
* **Accuracy:** long_acc **1.0** (cumulative at 512, 2,000 test chains; 1.0 at every length from 16 to 512). 2,048: **0.998**
  (n = 500).
* **C1:** single-code donor patching at positions 2 / 4 / 6 is donor-consistent **1.0 / 1.0 / 1.0**, with recipient
  equal to chance overlap. Serial use of the discrete carrier.
* **Code:** 3,374 distinct codes on test chains, **about 28 per state**, perplexity 812.
  - margin_flip_rate is 0.91: under U(−0.5, 0.5) noise on z_b the code changes at most hand-offs, so exactness does
    not come from centred codes.
  - It comes from a redundant code: many codes per state, each mapped correctly.
* **CERT** (post hoc, 120-3) is running. The 2,048 errors (0.998) suggest a few reachable code transitions are wrong, so
  CERT may not hold.
* **Scope:**
  - coverage_drawn is 1.0 (120-1: the short-problem curriculum labels every transition), so no C4.
  - One seed of one family. The pilot reading needs both seeds and both families (flip-flop, the non-group family).

**Note 120-5 (2026-10-05 07:30 EDT; descriptive, S5 FINAL s1; Audit's questions). The learned code machine is open, not closed: it
wanders slowly through new, correctly decoded codes.**
* **CERT start set:** all first-step codes from the 120 start texts × 120 ops (14,400 pairs, deduplicated), as 120-2.
  The first post-hoc CERT reached **537,873 reachable codes in 45 min without closure** (inconclusive_time, so no
  certification claim). A layered rerun gives the growth by BFS depth (to follow).
* **Distinct codes visited** on the 2,048-step extension rollouts (n = 500, cumulative 0.996):

  | by position | 16 | 64 | 128 | 256 | 512 | 1,024 | 2,048 |
  |---|---|---|---|---|---|---|---|
  | distinct codes | 1,422 | 2,120 | 2,376 | 2,629 | 2,864 | 3,109 | 3,342 |
  | codes per true state | 11.9 | 17.7 | 19.8 | 21.9 | 25.4 | 29.4 | 34.2 |

  - After 128, about 240 new codes per doubling: roughly logarithmic, unsaturated.
* **Layered CERT (rerun on a shared GPU, inconclusive_time):** the 14,400 first-step pairs collapse to **734 codes**
  (about 6 per state). The frontier keeps growing: 2,591 at depth 2, 3,079 at depth 3, ≥ 4,737 at depth 4 (partial).
  No closure, consistent with the rollout growth.
* **Reading.** Each hand-off is snapped, so there is no continuous drift. But the code is **not canonical**: a state's
  code depends on the path that reached it, and long chains keep visiting new (state, code) pairs.
  - Almost all are mapped correctly (0.996 at 2,048), but correctness on unvisited codes is not guaranteed. That is the
    gap between "exact at 512" and "certified at any length".
  - A closed machine would need canonical codes (codes per state → 1), which this objective does not ask for.

## 123. Is attractor strength causal for chains without snaps? A dither dose-response on §120's harness, and dithered FSQ for canonical codes (pre-registered 2026-10-05 07:46 EDT, after Literature's check 597edee9 and before any implementation or run; Audit's P-B proposal; launched after §120's pilot reads)

**Why.**
* **118-3:** no-snap quality grades with attractor strength across trained models. The per-hand-off perturbation ratio
  ρ is about 0.25 in the READ seeds that run without snaps, 0.38 with mild drift, 0.66–0.75 for drift, and 0.77–1.0 for
  §108.
* **120-5:** §120's FSQ machine is exact to 512 but open, with non-canonical codes (codes per state growing with
  length; 537k reachable codes).
* A correlation across seeds is not causation. §123 **intervenes** on attractor strength with a task-agnostic objective,
  denoising dither on the carried state, in §120's harness.
* **Prior** (597edee9, `docs/LIT_S123_DITHER_2026-10-05.md`): Ballé 1611.01704 and Agustsson & Theis 2006.09952 (uniform
  noise as a quantisation proxy; universal quantisation); 2510.14095; Rifai 1104.4153 (contractive AE); Stable Recurrent
  Models 1805.10369; Attractor Models / DEQ; 2609.26487 (perturbation-measured stability, the closest to ρ). New as
  scoped: the dose-response framing.

**Harness.** §120's exactly: a Markov segment per item, z_b = tanh(W_down LN(h) + c) · 3.5 − 0.5 (d = 8), carried
embedding W_up(·), the answer decoder, the length curriculum, 24k steps, FINAL supervision only. Families: S5 (512) and
flip-flop (512, p_ignore 0.8).

**Arms.**
* **Continuous dose-response (Audit's layout).** z_b is carried **without rounding**. At every training hand-off after
  2k, additive dither U(−σ, σ) is applied per coordinate; test is clean.
  - **S5: σ ∈ {0, 0.1, 0.25, 0.5} × seeds 1–2** (8 runs). σ = 0 is the 8-dim continuous no-noise control (Literature
    (c): compare within the harness, not with READ).
  - **Flip-flop: σ = 0 (seed 1), plus σ* (seed 1).** σ* is the σ > 0 with the lowest mean S5 ρ, chosen by that rule once
    the S5 cells read (2 runs).
* **FSQ+DITHER, 2 seeds per family.** §120's hard rounding, plus pre-rounding noise U(−0.25, 0.25) on z_b at every
  training hand-off after 4k. 4 runs.

**Measures.**
* No-snap long accuracy (the outcome), and accuracy against length.
* **ρ and basin** (contraction_diag ported to fsq120; Literature (c)): perturbation δ = ε · std_d · u_d, with std_d the
  per-coordinate std of carried z_b on clean test chains and u Gaussian.
  - **ρ := the geometric mean over 8 hand-offs of the median distance ratio at ε = 0.1** (fixed now).
  - Basin := the largest ε ∈ {0.03, 0.1, 0.3, 1.0} with decode agreement ≥ 0.99 after 8 hand-offs.
* **Implicit quantisation:** the share of test-time z_b coordinates within 0.1 of a grid point (q − 0.5 offsets as FSQ).
* Codes per state against length and the layered CERT (120-5's measures), plus Moore minimisation, for FSQ+DITHER.

**Rules (fixed now).**
* **CAUSAL (attractor strength causes no-snap success):** all of (i)–(iii).
  - (i) The S5 mean ρ falls from σ = 0 to σ = 0.5, and the flip-flop ρ at σ* is below its σ = 0 control.
  - (ii) Every continuous cell with ρ < 0.3 has long accuracy ≥ 0.99.
  - (iii) Every continuous cell with ρ > 0.6 has long accuracy < 0.99.
* **NOT CAUSAL:** ρ falls with σ (i) but some cell with ρ < 0.3 fails, or long accuracy does not track ρ. P-B is then
  dropped as a stabiliser.
* **The continuous-stabiliser reading** additionally requires that the passing cells are **not** implicitly quantised
  (grid share < 0.5). Otherwise the reading is "attractors via implicit quantisation".
* **CANONICAL (FSQ+DITHER):** codes per state at 2,048 below §120 FINAL's (S5 s1: 34.2) in both seeds of a family,
  **and** CERT done with holds = true in ≥ 1 seed (certified at every length).
* **Task-agnostic claims** need both families, with the same code and hyperparameters.

**Readings.**
* **CAUSAL.** Attractor strength is a cause of no-snap success, and a task-agnostic denoising objective supplies it.
* **CANONICAL.** Dithered FSQ gives a closed, certified learned machine: fully discrete-latent reasoning, exact at every
  length, with no template and no schedule.
* **Neither.** Stability needs something these objectives do not provide. §120's every-step snap stands as the route.

**Placement.** 14 runs (S5 8, flip-flop 2, FSQ+DITHER 4), about $7–10 on Vast (within Audit's about $12), **launched
after §120's pilot reads**; flip-flop σ* after the S5 cells read. Implementation:
the experiments session (fsq120 `--dither σ` / `--no_round`, `--fsq_dither 0.25`, and the ported contraction eval).
Scorer `theory/score_s123.py`, committed before any run.

**Note 123-1 (2026-10-05 08:02 EDT, before any §123 run; the experiments session's implementation, accepted; not launched until §120's
pilot reads).**
* `--no_round` (W_up(z_b), unrounded at train and test), `--dither σ` (training hand-offs after 2k) and
  `--fsq_dither σ` (before rounding, after 4k), as registered.
* **ρ / basin** as registered: 500 chains from the training distribution, T0 = 4, δ = ε · std_d · u (std_d over
  positions of clean carried z_b), ρ at ε = 0.1 as the geometric mean over k = 1..8. In FSQ arms the perturbation goes
  before rounding.
* **grid_share:** the share of test z_b within 0.1 of an integer (z_b includes FSQ's −0.5 offset).
* **Rule for FSQ arms (fixed now).** Rounding can absorb a perturbation exactly (distance 0), which makes the ratio
  meaningless. So in FSQ arms ρ is reported as:
  - (a) the **exact-recovery share** (chains with distance 0 at every k = 1..8);
  - (b) ρ over the chains with nonzero distance at k − 1.
  - ρ enters CAUSAL only for the continuous cells, where this does not arise.

**Note 117-5 (2026-10-05 08:32 EDT, before any §117 result; a decision rule fixed in advance, Audit). Whether to run a direct test of
112-4 (§112's f = 0.1 setting plus 10 % copy rows) depends on §117's acquisition rate.**
* **Acquisition** = final k = 1 never-labelled accuracy ≥ 0.99 (as §116's MECH).
* It is counted over §117's first wave: MK s51–53 and ST s61–63, 6 seeds.
* **≥ 5/6 acquire:** Markov segments (with the reader) solved the lottery, and the copy test is dropped.
* **≤ 3/6 acquire:** the lottery persists. §112+COPY is registered after a Literature check: 5 seeds, about $6, rule
  acquisition ≥ 4/5 against §112 f = 0.1's 3/5.
* **4/6:** decided after §117's extension seeds if an arm extends; otherwise §112+COPY runs.

**Note 120-6 (2026-10-05 08:55 EDT; descriptive, S5 FINAL s2). The FSQ chain has a seed lottery: code collapse.**
* s2 uses **9 codes in all** (perplexity 2.8; far fewer codes than the 120 states). Accuracy is 0 at
  every length.
* So S5 FINAL fails PILOT-PASS (1/2: s1 1.0, s2 0). The pilot reading needs both seeds in both families, so FINAL cannot
  pass the pilot.
* Flip-flop and TRACE cells are pending. The collapse is a training-time basin (FSQ's usual codebook-usage failure);
  §123's FSQ+DITHER arm and code-usage monitoring bear on it.

**Note 117-6 (2026-10-05 08:55 EDT; descriptive, MK s52–53 of the first wave). Markov from step 0 does not acquire in-chunk
composition from final answers.**
* Both seeds score 0 at 512, with never-labelled k = 1 at chance.
* Pool accuracy is 0.008–0.039 at lengths 3–4 in every round: even single-chunk problems are never solved, against
  §112's 0.50 at L = 4 with the masked fork.
* That is Literature's predicted risk (b0bca3c4 / f9ce7210 (b)): without the in-pass parallel shortcut that seeded the
  checkpoints in §112, the composition is not learned from outcomes. It is the ST arm's reason to exist.
* For 117-5's rule: 0/2 acquired so far (MK s51 and ST s61–63 pending).

**Note 120-7 (2026-10-05 10:14 EDT; amendment after data, disclosed). Flip-flop CERT was ill-posed as registered.**
* 120-2's CERT for flip-flop runs BFS from the empty start over all 5 items, including ignore and read before any
  write. The true state there is "no write yet", which has no answer text (the decoder emits a bit). The task never
  produces those prefixes, since every sequence starts with a write.
* So a perfect model fails the registered CERT by construction. FF TRACE s2 shows it: 47 reachable codes, 18 violations,
  all codes decode validly as bits.
* **Corrected definition (applied to every flip-flop cell; both reported):** CERT over the task's input language. The
  BFS starts from the empty start with **write items only** (w0, w1) as the first item, then all 5 items.
  - It holds iff every reachable code decodes validly and the transition law holds on every reachable pair.
  - S5's CERT is unchanged (every S5 prefix is valid).
* **FF TRACE s2, descriptive:** long_acc 1.0 (p_ignore 0.8, 0.98 and 0.1 all 1.0), donor 1.0 at positions 2 / 4 / 6. Its
  code machine **closes**: 7 codes on test (3.5 per state), saturated from position 16 to 2,048; registered-CERT BFS
  closes at depth 8 with 47 codes (Moore 20). The corrected CERT is to follow.
* **Donor rule for flip-flop (the same amendment, disclosed).** With binary answers, the donor's and recipient's answers
  coincide by chance (chance_overlap 0.54–0.71). So SERIAL's "recipient < 0.10" is read as **recipient − chance_overlap
  < 0.10**, §104's intent; for S5, chance_overlap is about 0.008, so nothing changes there.
  - FF FINAL s2 and TRACE s2: donor 1.0 at positions 2 / 4 / 6, recipient exactly at chance overlap, so SERIAL.
* **FF TRACE s2 is certified (2026-10-05 10:15 EDT).** The corrected CERT (task language) is done and **holds**: 7 reachable codes, all
  decoding validly, the transition law exact on every reachable (code, item) pair.
  - **Moore minimisation gives 2 states: exactly flip-flop's minimal automaton after the first write.**
  - So a pretrained LM with a learned, task-agnostic discrete snap at every step is a learned finite-state machine,
    exact at every length by enumeration, discovering the minimal automaton.
  - Scope: TRACE ("task-agnostic architecture, per-task supervision"), one seed. The pilot reading needs FINAL in both
    families.

**Note 120-8 (2026-10-05 10:15 EDT; Audit). The headline wording keeps the registered verdicts, and the corrected definitions are
pre-registered from now on.**
* **FF TRACE s2 headline, fixed:**
  - "registered CERT: fails (unreachable read/ignore-before-write prefixes); task-language CERT: holds (7 codes, Moore 2)";
  - "SERIAL (registered): fails on chance overlap; corrected (recipient − chance overlap): holds".
* **Pre-registered from this note on** (before the remaining §120 flip-flop cells — FF FINAL s1, FF FINAL s2's CERT, FF
  TRACE s1 — and every §123 flip-flop cell land):
  - flip-flop CERT is the task-language CERT (first item a write), with the registered CERT reported beside it;
  - SERIAL everywhere is donor ≥ 0.90 and recipient − chance_overlap < 0.10. Both verdicts are still printed for
    flip-flop.

**§120 result (2026-10-05 11:38 EDT; 8 runs by the experiments session; scored by Theory alone with `theory/score_s120.py`, committed before
any run; amendments 120-3, 120-7 and 120-8 disclosed). The P-A pilot reading is not met: FINAL passes in flip-flop but
not in S5. Flip-flop passes in both arms, with two seeds certified at every length.**

| family × arm | s1 | s2 | PILOT-PASS |
|---|---|---|---|
| S5 FINAL | **1.0** (2,048: 0.998; 28 codes per state) | 0 (code collapse: 9 codes) | ✗ |
| S5 TRACE | 0 (a failure basin, loss 3.23 in two realisations) | **1.0** (2,048: 1.0; 18 codes per state) | ✗ |
| FF FINAL | **1.0** (10 codes per state) | **1.0**: task CERT **holds** (36 reachable, **Moore 2**) | **✓** |
| FF TRACE | **0.995** (all reads right in 0.885 of sequences; 268 codes per state) | **1.0**: task CERT **holds** (7 reachable, **Moore 2**) | **✓** |

* **Every passing seed is serial** (single-code donor 1.0). For flip-flop that is the corrected rule (recipient − chance
  < 0.10); the registered rule fails there on chance overlap (120-8).
* **Certification:**
  - **FF FINAL s2 and FF TRACE s2 are certified at every length on the task language.** Their learned code machines
    minimise to flip-flop's 2-state automaton.
  - The registered CERT fails in both, on unreachable read-before-write prefixes (120-7).
  - S5 FINAL s1's machine is open (120-5), so it is not certified.
  - FF FINAL s2's growth saturates at 35 codes on 2,048-step rollouts.
  - FF TRACE s1 is **not** certified (task CERT: 1,024 reachable, Moore 870, 685 violations; its code set does not
    saturate), consistent with its 0.885 all-reads-right rate. FF is certified in 2/4 cells. The other cells' CERT is
    pending.
  - **S5 machines stay open.** TRACE s2 (accuracy 1.0 at 2,048) visits 1,156 codes by position 16 and 2,176 by 2,048,
    still growing; CERT is inconclusive_time (344,527 reachable). TRACE s1 (the failed seed) saturates at about 100 codes
    but is wrong (Moore 224, 39,641 violations).
* **Reading.**
  - **(a) A learned, task-agnostic discrete snap at every step works where the state space is small.** Flip-flop passes
    4/4 seeds in both arms, from final answers too, with certified minimal machines in 2 seeds.
  - **(b) On S5 (120 states) it is a seed lottery: 2/4 S5 seeds fail in range.** S5 FINAL s2 collapses its codes; S5
    TRACE s1 sits in a loss basin. Where S5 succeeds, the code is redundant and open (17–34 codes per state, growing
    with length).
  - So the directive's minimum (≥ 2 families with the same design) is met for flip-flop only: "fails in range" in S5.
* **Scope:** coverage_drawn is 1.0 (S5) and 0.27 (flip-flop), from the short curriculum (120-1), so no C4. §123's
  FSQ+DITHER arm targets S5's reliability and canonical codes.
* **Extension (registered for passing family × arm cells):** deferred until §123's FSQ+DITHER reads. If dithering fixes
  S5's lottery, the extension (seeds 3–5, cycle navigation, multiplication) uses it. A note will precede that launch.

**Note 117-7 (2026-10-05 12:00 EDT; descriptive, 4 of 6 first-wave seeds in; MK s51 and ST s61 pending). 117-5's rule fires.**
* MK s52 / s53 and ST s62 / s63 all score 0 at 512, with never-labelled k = 1 at 0.008–0.041.
  - In ST the label-free switch never fired: final-answer accuracy at lengths 5–6 never reached 0.5 under the masked
    fork. So the forced switch came at 12,000, and nothing was acquired after it.
* **Acquisition is 0/4, so the first wave is at most 2/6.** By 117-5 (≤ 3/6): the lottery persists, and the direct test
  of 112-4 (§112's f = 0.1 recipe plus 10 % copy rows, with the fixed copy path) is registered after a Literature check.
* §117's extension cannot fire (ALL ≤ 1/3 per arm). The §117 result follows when s51 / s61 land.

## 124. Does reading a given state fix self-training's lottery? §112 f = 0.1 plus a self-supervised copy objective, against an exposure control (pre-registered 2026-10-05 12:03 EDT, after Literature's check b5078932 and before any run; triggered by note 117-5's rule (117-7); Audit's matched design)

**Why (112-4, 116-4 to 116-6, 117-7).** Self-training's acquisition is a lottery: §112 f = 0.1 acquires in 3/5, §110 in
3/5, §117's first wave in 0/4.
* 112-4's account: every seed solves single-chunk identity-start problems, but post-cut lengths need **reading a given
  start state**, which identity-start training never exercises. The start-text no-read basin.
* §116 never tested it: NONE has no in-chunk stage, and NONE+COPY's copy path had a no_grad bug. The fixed path is
  unit-tested (copy accuracy 1.0 within 100 steps).
* **Prior** (b5078932, `docs/LIT_S124_COPY_2026-10-05.md`; also 15cf541a): Dai & Le 1511.01432, NTM copy, He 1901.05534
  (reconstruction forces encoder use), Ruiz & Gu 2507.02782 (state coverage), Abbe's leap.
  - Our §110b: random textual starts without a reading objective stayed at chance, so the active ingredient should be
    the read, not exposure. New as scoped.

**Scope (Audit).** §124 tests 112-4 in the **masked-fork** setting only. §117's Markov seeds never learn even
single-chunk composition (117-6 / 117-7): that is §85's saddle again, not 112-4's basin.

**Design (matched and concurrent; Audit, Literature (2)).** §112's f = 0.1 recipe exactly (masked fork, identity
starts, T_f with all (e, a) plus inner g, independent pools, validity → final → self-consistency filters with the
label-free gate, final relabel, 24k steps). **Three arms on the same new seeds 26–29, launched together:**
* **BASE:** §112 f = 0.1 unchanged (no extra rows), as a concurrent baseline on the same code.
* **COPY:** plus 10 % of every batch's rows, from step 0 through the final relabel, as copy problems.
  - Each is a random start state (uniform over 120) as text, with zero ops. The start text's last hidden is fed as a
    slot, and the masked fork decodes the start's own text.
  - Self-supervised: no transition or intermediate labels.
* **TEXT-EXPOSE (Literature (2)):** the same 10 % random-start rows trained only with plain next-token LM loss on the
  start text. No slot readout, nothing labelled.
* Extra rows replace ordinary rows (the budget is unchanged). copy_loss is logged separately, along with the stay rate
  on real ops (truth, logged only).

**Measures.** §112's: cumulative 512 / 2,048, k1_never_labelled_acc, k1_certified, per_round_split, coverage_drawn,
single-patch donor. Plus:
* the **acquisition round**: the first round whose never-labelled k = 1 exceeds 0.5;
* the round at which L ≥ 5 pool accuracy first exceeds 0.5;
* copy accuracy at the end.

**Rules (fixed now; at n = 4 per arm the counts are descriptive unless the difference is large).**
* **Primary readout:** the acquisition count per arm (final never-labelled k = 1 ≥ 0.99) and the timing signature (the
  acquisition round).
* **MECH-112 (supports 112-4):** COPY acquires in **4/4**, **and** BASE in ≤ 2/4, **and** TEXT-EXPOSE in ≤ 2/4.
* **NOT 112-4:** COPY acquires in no more seeds than BASE, and no earlier (median acquisition round).
* Otherwise the result is **descriptive**: counts and rounds per arm are reported, with no mechanism claim.
* FOOTHOLD (as §112) is reported per arm.
* **Claim wording (fixed):** "C4 qualified by f = 0.1 short-problem coverage (identity starts) plus a self-supervised copy
  objective (no intermediate or transition labels)".

**Readings.**
* **MECH-112.** Self-training's lottery is the start-reading basin, and a label-free copy objective removes it: a
  reliable C4-qualified recipe (C6 for C4).
* **NOT 112-4.** Something other than start reading limits acquisition. Report per-round splits; the account is
  revised.

**Placement.** 12 runs (3 arms × seeds 26–29), about $10–12 on Vast, **after §123 in the queue** (Audit). Implementation:
self110 `--cov 0.1` (BASE), plus `--copy_frac 0.1` (the fixed copy path) or `--expose_frac 0.1` (LM loss on the start
text). Scorer
`theory/score_s124.py`, committed before any run.

**Note 124-1 (2026-10-05 12:41 EDT, before any §124 run; the experiments session's implementation, accepted).**
* `--expose_frac`: `Gen110.expose_loss` is plain next-token LM CE on [EOT] + the start text, with no slot, fork or state
  decode. The rows replace ordinary rows, like copy rows.
* copy_loss and expose_loss are logged separately; stay_rate is in per_round_split.
* eval.json carries copy_acc for every arm (masked-fork greedy decode of the copy path, n = 1,000), BASE included as the
  untrained baseline.
* The gradient-flow check passes for both new losses (it guards against 116-5's no_grad fault).
* **"Matched seeds"** (Audit) means the same initialisation and the same labelling pools. The batch streams diverge
  after step 1, because the extra rows draw from the same RNG.
* Queues c1–c6 (P = 2): proof host c1 (EXPOSE s26–27), then EXPOSE s28–29, COPY s26–29, BASE s26–29. They run after
  §123's b7 / b8.

**Note 123-2 (2026-10-05 13:08 EDT, before any §123 cell is scored; Audit / claude-13's scorer check).**
* **CANONICAL for flip-flop uses the task-language CERT** (`cert_task`, as 120-8 pre-registered), with the registered
  CERT printed beside it. The scorer had read the registered BFS, which fails by construction on unreachable prefixes.
* **The codes-per-state baseline is per family:** "below §120 FINAL's".
  - S5: 34.2 (FINAL s1 at 2,048).
  - Flip-flop: **13.75**, the mean of FF FINAL s1's 10.0 (eval.json, all positions; it has no growth file) and s2's
    17.5 (eval_growth at 2,048). Flip-flop's code set saturates by position 16. Corrected from 14.0 (Audit), before any
    §123 cell. The registration's parenthetical gave only the S5 number.
* **ρ's distance metric is kept as registered:** the median Euclidean distance ratio in z_b. The perturbation is
  per-coordinate normalised (δ = ε · std_d · u), the distance is not.
  - A std-normalised ρ is reported beside it as descriptive. Over 8 hand-offs the geometric-mean ratio approaches the
    dominant contraction rate in either metric.

**Note 122-2 (2026-10-05 13:14 EDT; descriptive, the primary arm's s1; s2–3 pending; rules and stop rule unchanged). A competent anchor
target is not enough: the text path's latent never had to carry the sum.**
* **s1:** op 3 0.362, test 0.047. The text path **is** competent: T1 = 1.0, P_3 1.0, text_acc 1.0 at steps 2–4. The
  anchor binds (cosine 0.986–0.990).
* **The carrier is unchanged.** Slot-2 probes: b_1 / b_2 1.0, S_2's middle digits 0.12–0.21 (chance). The donor test
  is mixed (SAMEPREFIX recipient 0.23 vs sum-donor 0.18; SAMESUM recipient 0.22 vs prefix-donor 0.18).
* **Why (structural).** TEXT-MIX's k = 1 chunks train the text path to decode step j, but the text path's own **carried
  latent is never consumed by a following segment**.
  - Its fork reads it together with the visible segment [a, S_{j−1} text, b_j], so nothing forces that latent to hold
    S_j.
  - The anchor ties the chain's latent to a code with no obligation to carry the sum.
  - On S5 (§118) the text-start segment's latent **is** consumed by the next in-chunk segment (k = 4 chunks), so the
    anchored code carries the state. That is why the anchor works there.
* **Implication (recorded; not proposed as a run, per the stop rule).** An anchor transfers carriage only from a
  teacher path whose carried state is itself read downstream. The general lesson for future designs: anchor to a
  text-started chain of at least two hand-offs, not to a single text-started step.

**Note 120-9 (2026-10-05 13:19 EDT; descriptive, after data, disclosed). A behavioural certificate for flip-flop, whose outputs exist
only at reads.**
* **FF FINAL s1:** perfect reads (long_acc and long_acc_seq 1.0; 2,048-step accuracy 1.0; 20 codes, saturated). But the
  task-language CERT fails: 20 reachable codes, Moore 4, 13 violations.
  - FINAL supervises only reads, so a code reached after a write or ignore need not decode to the true bit. Only its
    read successor must.
  - The decode-every-code CERT (120-7 / 120-8) therefore demands outputs that the task never observes.
* **cert_task_reads (descriptive, reported beside the registered and task-language CERTs; no rule changes).** BFS over
  the product of the code machine and the true flip-flop automaton.
  - It starts from (start code, no-write state) and follows only valid sequences (first item a write).
  - It checks decode(next(c, read)) = the true last-written bit **only on read transitions**.
  - It holds iff every reachable read is correct, which certifies correct reads at every length. It is the right
    notion for an output-at-reads task.
  - S5's CERT is unchanged: every S5 position is a potential final answer, so every code must decode.
* The §120 and §123 verdicts stay on the pre-registered CERT (120-8). cert_task_reads is reported as a description.
* **Results on all four §120 flip-flop cells** (4060, bf16):

  | cell | cert_task_reads | cert_task (holds / Moore / violations) | registered |
  |---|---|---|---|
  | FF FINAL s1 | **holds** (20 pairs, 0 violations) | false / 4 / 13 | false / 4 / 16 |
  | FF FINAL s2 | **holds** (35 pairs) | **true** / 2 / 0 | false / 2 / 3 |
  | FF TRACE s1 | false (1,339 pairs, 521 violations) | false / 847 / 663 | false / 841 / 668 |
  | FF TRACE s2 | **holds** (7 pairs) | **true** / 2 / 0 | false / 20 / 18 |

  - Exact on every reachable read in **3/4** cells (descriptive). Certified under the verdict's cert_task in 2/4.
  - FINAL s1's cert_task failures are all non-read outputs.
* **Numerics caveat.** Certificates are computed in bf16 (the eval dtype), so they certify the bf16 model as deployed.
  - The certified cells give identical results on two GPUs.
  - The uncertified TRACE s1 shifts slightly across cards (Moore 869 → 841), which is cross-card bf16 nondeterminism on a
    large inexact machine.

**Note 122-3 (2026-10-05 14:06 EDT; descriptive, $0; Audit / claude-13's anchor-floor question; `theory/anchor_floor.py`, n = 400, hand-off
j = 2). The anchor binds state-specifically on S5, and only weakly on multiplication.**

| model | raw cos(z, own b) | raw cos(b_i, b_k) (floor) | centred cos(z, own b) | centred cross | nearest centred b is own |
|---|---|---|---|---|---|
| §122 a122 s1 (4×4) | 0.987 | 0.786 | 0.737 | 0.024 | 0.235 |
| §118 anchor s71 (S5) | 0.9998 | 0.848 | 0.971 | −0.001 | 0.83 |

* The raw floor between different problems' text codes is 0.79–0.85, not about 0.98. A raw anchor cosine of 0.987 is
  well above it.
* Centred, the multiplication binding is real but weak (0.74, own-match 24 %), against S5's 0.97 / 83 %. Together with
  122-2 (the text path's latent never has to carry the sum), the anchor transfers a loosely bound code that does not
  hold S_2.
* **Design fact for future anchors:** report centred anchor cosines and the own-match rate, not raw cosines. A future
  anchor loss could use centred or standardised codes.

**Note 112-5 (2026-10-05 14:06 EDT; Audit / claude-13's analysis of 32 self-training runs, descriptive; a $1 check pre-stated). FOOTHOLD
counts acquisition timing as well as acquisition.**
* **The label-free gate tracks acquisition:** it crosses 0.5 in the same round as never-labelled k = 1 in 8 of 10
  acquirers and lags one round in 2, never leading. A gate rule also needs kept_share ≥ 0.1 (§116 NONE s43 showed gate
  0.95 on 0.8 % kept).
* **Nothing predicts acquisition before about 11k** (gate ≤ 0.007; every other signal interleaves), consistent with
  112-4. Early-abort rules cannot work.
* **Timing decides FOOTHOLD.**
  - All 8 seeds acquiring by 14.9k pass.
  - Of the 5 acquiring at ≥ 16.6k, 3 fail, two with k = 1 = 1.0 on every never-labelled transition (f = 0.1 s22 0.9645;
    f = 0.5 s24 0.968), with < 7.5k steps left after acquisition against ≥ 9k for every passer.
  - So by k = 1, acquisition is f = 0.1 3/5, f = 0.5 4/5, §110 3/5.
* **Pre-stated $1 check (descriptive).** Resume f = 0.1 s22, f = 0.5 s24 and §110 s1 from their saved model and final pool
  for one more labelling round plus the 4k relabel.
  - Prediction: cumulative ≥ 0.99 at 512 in both k = 1 = 1.0 seeds (s22, s24).
  - If confirmed, `--post_acq_steps 9k` (triggered label-free by gate ≥ 0.5 with kept ≥ 0.1) becomes a candidate for
    future self-training registrations.
  - **Disclosed exception (Audit / claude-13, verified): §124 EXPOSE s27.** It acquired at L5 (step 10,976, the earliest
    in any self-training run: never-labelled 0.998 by L5's end), yet ended at cumulative 0.9795 at 512 and 0.934 at 2,048.
    - It has a **slot-1 bypass** (donor 0.017 / recipient 0.804 at slot 1; slots 2–3 neither, about 0.4 / 0.05).
    - So post-acquisition budget is sufficient in the data so far but not necessary-and-sufficient: a carrier with a
      bypass can stay below 0.99 despite early acquisition.
    - The resume results are read with each run's donor profile beside them.
  - **First resume (f = 0.5 s24, local 4060): confirms.** 0.968 → **0.996** at 512 (0.982 at 2,048), never-labelled k = 1
    still 1.0, certified.
  - **Resume details, before the runs** (the experiments session):
    - One extra L_max round of 4k steps, then the 4k final relabel.
    - Fresh AdamW, warmup 100 plus cosine from 1e-4 over 8k, and fresh pools.
    - The labelled set for the k = 1 split is T_f (the source's drawn matrix was not saved; flagged drawn_proxy). For §110
      s1, never-labelled means all transitions.
  - **Three-way reading, fixed before the runs** (claude-13, via Audit; the resume restarts AdamW on a converged model).
  - **Resume learning rate (decided before the runs; replaces "the source's last-round LR"): peak 3e-5, cosine to 0
    over the 8k steps** (`--resume_lr 3e-5`).
    - Reason: the check simulates the passers' post-acquisition budget. On the 24k cosine from 1e-4, the latest passer
      acquired at about 14.9k, where the LR was about 3e-5, and then decayed to about 0.
    - 1e-4 risks an LR shock; the source's tail LR (about 7e-6) would barely move the model and could not test the
      hypothesis.
    - Cumulative ≥ 0.99 with k = 1 = 1.0: **confirms** (the budget was the limit).
    - Cumulative < 0.99 with k = 1 < 1.0: **an optimiser / LR shock**, not a refutation.
    - Cumulative < 0.99 with k = 1 = 1.0: **refutes** (more budget does not close the gap).

**Note 124-2 (2026-10-05 14:06 EDT, before any §124 run; no rule change).** 112-5 shows FOOTHOLD also counts the steps left after
acquisition. So an arm that acquires earlier can gain FOOTHOLD from budget alone. §124's mechanism signal is the
**acquisition round** (and the count), as registered. FOOTHOLD per arm is reported but not read as mechanism.

**Note 123-3 (2026-10-05 14:08 EDT, before any §123 cell lands; pre-data amendment, disclosed; Audit / claude-13). A per-cell
no-read-plateau flag, so a basin seed cannot fake contraction.**
* **Finding.** Both failed §120 S5 seeds were decided in the soft phase (before any rounding or dither): loss(2k) /
  loss(1k) was 0.99 (FINAL s2) and 0.96 (TRACE s1), against 0.78 / 0.40 for the passers.
  - 120-6's "code collapse" is the symptom; a no-read basin (§108 / §111) is the cause.
  - A basin seed ignores its carried state, so perturbations do not propagate. ρ is then near 0 with long accuracy 0,
    which would fail CAUSAL (ii) as an artefact.
* **Plateau flag (label-free: the training loss only).** plateau ⇔ loss(2k) ≥ 0.9 · loss(1k) **and** loss(2k) ≥
  0.25 · loss(100). The second clause excludes cells already near zero, e.g. flip-flop, which learns within 1k.
  - Calibrated on §120's 8 cells (disclosed): it flags exactly the two failed S5 seeds.
  - The final loss is reported per cell.
* **Rule changes (fixed now).**
  - CAUSAL (i)–(iii) and σ* are evaluated over **non-plateau** cells only. A σ level with no non-plateau cell makes (i)
    unevaluable, and CAUSAL is then "not evaluable".
  - CANONICAL counts non-plateau seeds only; "both seeds" becomes "all non-plateau seeds, at least one".
  - The plateau count is reported per arm. **Check:** it should not depend on σ (the dither starts after 2k).
  - The registered verdicts are printed beside, as 120-8.
* **Predictions (claude-13).** (P1) Continuous S5 cells split into plateau and not, at any σ. (P2) FSQ+DITHER S5
  plateaus at about §120's rate, and its CANONICAL is readable only on non-plateau seeds.
* **Flagged as engineering, not run:** restart-on-plateau at 2k. **Mechanism pointer:** an objective that makes the
  bottleneck read in the first 2k (a reader on z_b) is the task-agnostic fix, not noise after 4k.

**Note 122-4 (2026-10-05 14:13 EDT; decided before s3 lands, from local files, reproduced by Audit). §122's STOP RULE fires: the
multiplication latent-carriage line pauses.**
* Primary TEXTMIX+ANCHOR+READ: op 3 is **0.362 (s1), 0.408 (s2)**, test 0.047 / 0.053. CARRY-3 (≥ 2/3 at ≥ 0.95) cannot
  be met whatever s3 does.
* Control TEXTMIX+READ: op 3 0.246 / 0.344.
* These are valid tests: t1 = 1.0 in all four, so the anchor's target is competent. The donor test is "neither" at every
  slot.
* **Reading** (121-2 / 121-3 / 122-2 / 122-3):
  - In every Markov multiplication model, the carried slot-2 code holds the multiplier prefix or an entangled code,
    never the running sum.
  - Anchoring to a competent single-step text path binds the shared component (raw cosine 0.99) but only weakly the
    state (centred 0.74, own-match 24 %). The text path's own latent is never read downstream, so it need not carry the
    sum.
* **Per the stop rule there is no further patch.** Multiplication proceeds only through the task-agnostic route
  (§120's extension and §123), with 4×4 as a family.
* The lesson stands for any future design: anchor to a text-started chain of at least two hand-offs, and report centred
  binding. The formal §122 result follows when s3 lands.

**Note 123-4 (2026-10-05 15:15 EDT; descriptive, the first two §123 cells; rules as 123-2 / 123-3). S5 FSQ+DITHER: one seed learns
with fewer codes, one plateaus.**
* **s1: not plateau.** Accuracy 1.0 at 512 and 2,048; donor 1.0 at positions 2 / 4 / 6.
  - **21.0 codes per state at 2,048, against §120 FINAL s1's 34.2.** The dither reduces code redundancy.
  - Codes on the extension rollouts still grow (1,196 at 16 → 2,519 at 2,048), so the machine is likely still open.
    CERT and growth are pending.
* **s2: plateau by 123-3's flag.** It is decided in the soft phase before the dither starts: 2 codes, accuracy 0, the
  same seed's basin as §120 FINAL s2.
  - Its ρ ≈ 0 and basin 1.0 come from a constant-output machine, the artefact 123-3 guards against.
* **Prediction P2 holds so far:** FSQ+DITHER S5 plateaus at §120's rate, on the same seed. The basin is set early, by
  seed, before any dither acts. So CANONICAL for S5 rests on s1 alone (codes per state 21 < 34.2 ✓; CERT pending).

**Note 123-5 (2026-10-05 15:31 EDT, before any S5 continuous cell lands; a pre-data amendment, disclosed; Audit / claude-13). ρ for
CAUSAL is computed in fp32.**
* **The problem.** The 8-dimensional carried z_b is evaluated under bf16 autocast. After 3–4 hand-offs the perturbed and
  clean chains become bit-identical (distance at one ulp, then 0), and the geometric mean floors those zeros at 1e-6.
  - So a contracting continuous cell reports ρ ~ 1e-4 to 1e-5 whatever its true rate. Example: the FF σ = 0 control's
    rho_k = [0.052, 0.251, 0.183, 0, 0, …], registered ρ 8.4e-5.
  - CAUSAL (i) would then compare floors, and (ii) / (iii) would be met trivially. 118-3's 768-dimensional diagnostic
    never hit this.
* **Amendment.** **CAUSAL's ρ (and σ\*) is ρ_fp32**: the same contraction eval re-run post hoc in fp32 from model.pt
  (`--contraction_only --fp32` → `eval_contraction_fp32.json`).
  - The registered (bf16) ρ and rho_nonzero are printed beside it.
  - A cell without ρ_fp32 makes CAUSAL "incomplete".
  - rho_nonzero is not used, because dropping merged hand-offs biases it upward.

## 125. A readable bottleneck from step 0: standardising the carried code against the soft-phase no-read basin (task-agnostic) (pre-registered 2026-10-05 15:34 EDT, after Literature's check 1974cd64 and before any implementation or run; the user's task-agnostic directive (~15:30, via Audit); spend pre-approved)

**Why.** The task-agnostic design (§120's every-step learned snap) is unreliable on S5 because seeds fall into a
**soft-phase no-read basin** before any rounding or dither acts.
* §120 FINAL s2 and TRACE s1, and §123 FSQ+DITHER s2 (the same seed as §120 FINAL s2), all sit at the null loss by 2k
  (123-3).
* A reader alone does not prevent it: §120 TRACE decodes at every step and s1 still plateaued.
* **Diagnosis at initialisation ($0).** The carried embedding W_up(z_b) has row norm about **29**, against about **4** for
  GPT-2 token embeddings and about 2.9 for the start text.
  - Its input-dependent part is small (per-coordinate std across inputs 0.14, i.e. norm about 3.9).
  - So GPT-2 receives a large, nearly constant vector with the state signal buried in it, a plausible trigger for §111's
    writer–reader chicken-and-egg.
* **Prior** (1974cd64, `docs/LIT_S125_STD_2026-10-05.md`): Lester 2104.08691 and prefix-tuning 2101.00190 (scale and
  init sensitivity of injected vectors); HRPO (raw hidden injection fails, a normalised projection trains); CODI (MLP +
  LN); our §72b / §73 / §89 (standardisation, with centring load-bearing); posterior collapse (He 1901.05534). New as
  scoped.

**Design.** §120's harness exactly (FINAL supervision, d = 8, L = 8, the soft → hard schedule, the length curriculum,
24k steps, bs 64), plus, in the STD arm, a **standardised carried embedding**:
* e′ = μ_E + sd_E ⊙ (e − μ_e) / sd_e, where μ_E and sd_E are GPT-2's token-embedding per-coordinate mean and std, and
  μ_e and sd_e are per-coordinate EMA statistics (momentum 0.99) of the carried embeddings in each update, from step 1
  (as §72b / fade89). The statistics are frozen at the end of training and used unchanged at test.
  - Literature (3)'s EMA lag at the soft → hard shift lasts about 100 steps around 4k; the statistics are frozen 20k
    steps later, so test sees hard-phase statistics.
  - The same e′ is used everywhere a carried code enters GPT-2: segments and the answer decoder.
* It is task-agnostic: no task information, the same for every family. It draws no random numbers, so a seed fixes the
  initialisation **and** the data order: STD at seed k is a within-seed comparison with §120 at seed k (Audit (a)).

**Arms and seeds.**
* **STD:** S5 seeds 1–4 and flip-flop seeds 1–2.
* **CONTROL (concurrent):** §120 FINAL S5 seeds 3–4. Seeds 1–2 already exist: s1 learned, s2 plateaued, and s2 plateaued
  again in §123.
* 8 runs. A next-latent-prediction arm (Literature (2); NextLat 2511.05963) is the noted follow-up if STD is
  insufficient. Reconstruction of h is not run (Literature: it keeps nuisance PCs).

**Measures.** §120's: long accuracy, acc_by_length, 2,048, donor, codes, growth, and CERT (task-language and reads-only
for flip-flop). Plus:
* the 123-3 **plateau flag** per seed, and the final loss;
* **the carried-embedding row norm and per-coordinate std across inputs, at initialisation and at 2k, in every arm**
  (control included), so the mechanism is measured (Audit (c)).

**Rules (fixed now).**
* **PLATEAU-FIX:** STD's S5 plateau count ≤ 1/4, **and** seed 2 (the known basin seed) is not plateaued.
* **ACC:** S5 long accuracy ≥ 0.99 in ≥ 3/4 STD seeds.
* **FF guard (Audit (b)):** flip-flop long accuracy ≥ 0.99 in 2/2, single-code SERIAL (recipient − chance < 0.10), and
  CERT reported.
* The CONTROL plateau count is reported as the concurrent baseline for seeds 3–4.
* Code statistics (codes per state, growth, CERT) are descriptive.

**Readings.**
* **PLATEAU-FIX + ACC.** Matching the injected code's statistics to the LM's own embeddings removes the soft-phase
  basin. The task-agnostic every-step snap becomes reliable on S5 (C6 for P-A), and the extension to more families uses
  STD.
* **PLATEAU-FIX without ACC.** The basin is removed, but another failure limits S5. Report it.
* **Neither.** The scale mismatch is not the cause. The next-latent arm is next.

**Placement.** 8 runs, about $4–5 on Vast (Audit pre-approved), proof seed first (STD S5 s2, the basin seed).
Implementation: the experiments session (fsq120 `--std`, plus the init / 2k carried-norm logging). Scorer
`theory/score_s125.py`, committed before any run.

**Note 123-6 (2026-10-05 15:45 EDT; descriptive, §123 cells landed so far, before CAUSAL is scored; Audit's framing). The continuous 8-d
bottleneck alone is exact: snapping was not what made §120 work.**
* **S5 σ = 0 s1** (continuous z_b, no rounding, no dither, FINAL): long accuracy 1.0 at 16–512, **1.0 at 2,048** (n =
  500), and single-code donor 1.0 at positions 2 / 4 / 6.
  - grid_share is 0.21, about uniform (no implicit quantisation), and it does not plateau.
  - So **a task-agnostic continuous carry is exact to 2,048 on S5**: no text, no template, no schedule, no rounding.
* **S5 σ = 0 s2 plateaus** (123-3 flag; accuracy 0): the same seed-2 basin as §120 FINAL s2 and §123 FSQ+DITHER s2. The
  basin is shared across every arm at that seed and is decided before rounding or dither.
* **Flip-flop:** σ = 0 s1 1.0; FSQ+DITHER s1 1.0 (15 codes per state) and s2 1.0 (4 codes per state).
* **Reading.**
  - The task-agnostic line is best stated as **"continuous bottleneck + reliable entry"**, not "discrete snap".
  - The reliability blocker is the soft-phase no-read basin, which §125 tests.
  - Discretisation's distinct value is **enumerability** (CERT), and it requires canonical codes (closed on flip-flop,
    open on S5).
  - CAUSAL waits for every dose cell and its fp32 ρ (123-5).

**§125 result (2026-10-05 18:09 EDT; the proof seed only, by design: the remaining runs were held once PLATEAU-FIX was decided (125-4),
with Audit's agreement, disclosed). PLATEAU-FIX is untested: the standardisation was not in force at 2k, where the
plateau is decided (correction 2026-10-05 18:16 EDT, Audit / claude-13).**
* **STD S5 s2** (RTX 3090, the within-seed twin of §120 FINAL s2): plateaued throughout. Final loss 4.80, long accuracy 0,
  1 code.
* **carried_norm:** row norm 3.94 at init (wte 3.96: matched, as designed), but **35.2 at 2k** (per-coordinate std 0.012
  → 0.184).
  - Under the EMA, a few coordinates with tiny variance are amplified by the division, so the standardised code does
    not stay at embedding scale once training moves the encoder.
* STD S5 s1 / s3 / s4, the control s3–4 and FF STD were **not run**: ACC and the FF guard are untested.
* **Reading (corrected 2026-10-05 18:16 EDT).** The intervention was **not delivered** at 2k.
  - The EMA-0.99 mean lags a fast-moving raw embedding (|μ| about 44), and dividing by a tiny sd (median 0.054,
    amplification up to 48×) blows the lag up. GPT-2 again received a norm-35, nearly constant vector.
  - So §125 does **not** falsify the scale-mismatch account. It is untested.
  - Whether to rerun with correctly delivered standardisation (per-update batch statistics with stop-gradient, as §72b,
    or an absolute floor) is an explicit decision, taken after §126's RAW-H result.

**Note 125-1 (2026-10-05 15:51 EDT, before any §125 run; the experiments session's implementation d93e105b, accepted).**
* `--std` sits inside `embed_code`, so every path where a code enters GPT-2 uses e′: segments, the answer decoder, the
  flip-flop empty start, eval, CERT, donor and contraction.
  - Buffers are registered only under `--std`, so older models still load.
  - The EMA is updated after each optimiser step, with the sd floor, and frozen at the end. It makes no random draws.
* **Mechanism verified at initialisation** (CPU smoke, S5 s2): the carried-embedding row norm is **27.0 (control) → 3.93
  (STD)**, against GPT-2's wte 3.96.
* Gradient flow passes for the main path, W_up and W_down.
* Every S5 run uses `--micro 2` (gradient accumulation, so each step sees the same batch). Data order matches §120 at
  the same seed; numerics differ slightly.
* Proof host: STD S5 s2 (the basin seed). Then STD S5 s1 / s3 / s4, CTRL S5 s3–4 and STD FF s1–2.

**Note 125-2 (2026-10-05 15:57 EDT, before any §125 result; Audit / claude-13). The GPU type is fixed.**
* Training trajectories are seed-and-GPU-dependent. §123 S5 seed 2 stayed on the ln-120 plateau bit-for-bit through 2k on
  three RTX 3090-class cards, but left it on an RTX 4090.
* **Every §125 run (STD and CTRL) trains on an RTX 3090 or 3090 Ti,** the class of §120 FINAL s2 (the seed-2 control).
  A run on another GPU type is reported but excluded from PLATEAU-FIX.
* The **training** GPU is logged per run (eval.json's `gpu` key records the evaluating card).

**Note 123-7 (2026-10-05 15:57 EDT; descriptive, before CAUSAL is scored; Audit / claude-13). A second failure mode, and a hardware
caveat.**
* **S5 σ = 0.5 s2** (b9, RTX 4090): loss 5.00 / 4.78 / 4.08 at 1.0 / 1.5 / 2.0k, so it left the plateau before any dither
  and 123-3 does not flag it.
  - It jumped to 4.66 when the dither started at 2.1k and ended in the **~3.2 basin** (final 3.20, accuracy 0), like §120
    TRACE s1.
  - Registered ρ 5e-4, rho_nonzero 0.52. The fp32 ρ (123-5) decides its role in (ii) / (iii).
  - **It stays in CAUSAL:** "does σ 0.5 cause failure" is what (iii) tests, so excluding it would be circular.
* **The two basins are reported separately:** the chance plateau (ln 120, flagged by 123-3) and the ~3.2 basin (left the
  plateau, then stuck). A "final loss > 1.0" flag is post hoc and descriptive only.
* **Training GPUs (the experiments session's deploy logs).**
  - §120: S5 FINAL s1 / s2 on 3090 Ti; S5 TRACE s1 on 4090 (its first attempt, which reached the same ~3.2 basin by 14k,
    on 3090 Ti, so that basin is not hardware); S5 TRACE s2 on 3090.
  - §120 flip-flop: FINAL s1 / s2 on 3090; TRACE s1 on 3090, TRACE s2 on 4090.
  - §123: every cell on 3090 except b9's σ 0.5 pair (4090; rerun on 3090, 123-8).
  - From now on, `train_gpu` is logged in args.json and eval.json.
* **Hardware caveat.** Trajectories depend on seed **and** GPU type. σ 0.5 (RTX 4090) and σ 0 (RTX 3090) at seed 2 are
  matched by seed but not by trajectory. The training GPU is listed per cell in the result.

**Note 123-8 (2026-10-05 16:00 EDT, before the reruns and before CAUSAL is scored; Audit-approved about $1). The σ = 0.5 pair is rerun
on RTX 3090s.**
* Both b9 cells (RTX 4090) deviate from their 3090 twins in the pre-dither window, in opposite directions:
  - σ 0.5 s1 is a plateau seed, while 3090 seed 1 left the plateau by 2k (3.90);
  - σ 0.5 s2 left the plateau, while 3090 seed 2 stayed on it.
* So the σ = 0.5 pair is hardware-confounded.
* **CAUSAL reads σ = 0.5 from new RTX 3090 reruns of S5 σ 0.5 s1–s2** (dirs `s123_s5_dither050_s{1,2}_3090`). The b9 4090
  cells are reported descriptively, as a GPU-variance observation.
* Every §123 cell entering CAUSAL then trains on the 3090 class. If a cell outside the class is used, its GPU is stated.

**Note 125-3 (2026-10-05 16:00 EDT, before any §125 result; Audit / claude-13's pre-launch review).**
* Post-hoc jobs (`--growth_only`, `--cert_only`, `--contraction_only`) on STD models must carry `--std`. A strict load
  fails loudly without it.
* The `check_fsq120_std` gradient-flow PASS is committed before the §125 verdict.
* **Wording:** the carried code is standardised **from the second update**. Step 1's forward uses μ = 0 / sd = 1, because
  the EMA initialises after the first optimiser step. This is harmless.
* b13 (STD S5 s2) must be an RTX 3090 / 3090 Ti (125-2). Otherwise its seed-2 reading is excluded.

**Note 125-4 (2026-10-05 16:24 EDT; PLATEAU-FIX decided before the panel, rules unchanged; descriptive analysis). Standardisation does not
rescue the basin seed.**
* **STD S5 s2** (RTX 3090, the within-seed twin of §120 FINAL s2) has loss 4.984 at 1k and **5.000 at 2k**, so 123-3's
  flag marks it a plateau. PLATEAU-FIX requires seed 2 not plateaued, so **PLATEAU-FIX cannot be met.**
  - b13 runs to completion (does anything change after 2k?). The remaining §125 runs (b14–b16, b18) are held pending it,
    to save about $3.
* **Correction (2026-10-05 18:16 EDT):** "PLATEAU-FIX cannot be met" stands as a rule outcome, but the reading "scale mismatch falsified"
  is withdrawn. The standardisation was not in force at 2k (§125 result). The initialisation analysis below is
  unaffected.
* **Initialisation analysis ($0, CPU, seeds 1–4, 256 random (state, op) segments).**
  - **No tanh saturation:** mean tanh′ is 0.76–0.81, and 0 % of coordinates have tanh′ < 0.1.
  - In **every** seed the bottleneck input barely depends on the input: z's per-coordinate std across inputs is 0.03–0.04,
    against a shared level of about 0.5, and z_b's is 0.08–0.10 (the level spacing is 1).
  - Seed 1 (which escapes) and seed 2 (which plateaus) are indistinguishable at init. The basin is set by training
    dynamics and data order, not by initial geometry or scale. STD amplifies the decoder-side signal, but that is not
    enough.
* **Candidate (not registered; Audit to weigh against fix loops): bottleneck annealing from a carrier known to escape.**
  - §111 READ, which carries the raw 768-d hidden state with a reader, escaped in 5/5.
  - Start with e = std(h) and blend to W_up(z_b) over 2k–6k (task-agnostic), so the decoder learns to read a trained
    carrier before the bottleneck takes over.
  - Literature's next-latent auxiliary (2511.05963) is the other option.

## 126. Is the bottleneck the cause of the soft-phase basin? Raw-h carriage against an annealed bottleneck, with a stop rule (pre-registered 2026-10-05 16:27 EDT, after Literature's check d30785f9 and before any implementation or run; Audit's decisive-test framing; the user's task-agnostic directive)

**Why.** In the task-agnostic harness (§120), S5 seed 2 plateaus at the ln-120 null by 2k in every arm tried: §120
FINAL, §123 FSQ+DITHER and σ = 0, and §125 STD.
* §125's standardisation, as implemented, was not in force at 2k, so it is untested (correction 2026-10-05 18:16 EDT); and no tanh saturates at
  init.
* Seeds 1 and 2 are indistinguishable at init (125-4).
* §111 READ, which carries the raw h with a reader and per-step supervision, never plateaued (5/5).
* Two candidate causes: the 8-d bottleneck, or final-answer supervision without a reader.
* **Prior** (d30785f9, `docs/LIT_S126_ANNEAL_2026-10-05.md`):
  - QBN 1811.12530 is the wide-then-narrow precedent for recurrent latents (continuous first, then quantised bottlenecks
    distilled to reproduce the hidden states). KL annealing (Bowman) runs the same direction; Burgess 1804.03599 the
    opposite.
  - §89's α → 1 collapse is the warning. §108 (raw latent, per-step targets, no reader) hit the basin in 2/4, so RAW-H
    may plateau.

**Harness.** §120's exactly (FINAL supervision: final answers only; a Markov segment per item; the answer decoder; the
length curriculum; 24k steps; bs 64). S5 seeds 1–4, **RTX 3090 / 3090 Ti only** (125-2).

**Arms (sequential; Audit / Literature (3)1).**
* **Stage 1, RAW-H (the decisive control), S5 s1–4, proof seed s2.** No bottleneck: the carried vector is the segment's last hidden h, standardised to
  GPT-2's embedding statistics (§72b / §73 / 125's map, EMA statistics frozen at the end). It separates bottleneck from
  supervision.
* **Stage 2, ANNEAL, run only if RAW-H s2 is not plateaued, and only on the RAW-H seeds that escaped.** e = (1 − λ) ·
  std(h) + λ · W_up(z_b), with λ = 0 until 2k, then rising linearly to 1 at 6k.
  - During the ramp a QBN-style distillation loss trains the bottleneck to reproduce the carrier: ‖W_up(z_b) − sg(std(h))‖²
    / ‖sg(std(h))‖², weight 1 (Literature (3)2).
  - FSQ's soft → hard rounding is shifted to 6k–8k. Test uses λ = 1, hard.
  - The std(h) statistics are frozen at 2k and re-estimated at the FSQ switch (Literature (3)4).
* ANNEAL equals RAW-H until 2k, where the plateau is decided, so it cannot answer the plateau question. It tests only
  whether narrowing to the 8-d bottleneck survives after escape. Literature notes the α → 1 collapse risk, and that QBN
  used distillation.
* **If RAW-H s2 plateaus:** ANNEAL is skipped and the STOP RULE fires as written.

**Measures.** The 123-3 plateau flag per seed (at 2k), final loss, long accuracy (S5 cumulative at 512 with no snaps;
2,048), single-code / single-carrier donor, and carried_norm at init / 2k. For ANNEAL: accuracy at λ = 1, codes per
state, and growth.

**Rules (fixed now).**
* **BOTTLENECK-CAUSE:** RAW-H's seed 2 is **not** plateaued **and** RAW-H's plateau count ≤ 1/4. Then the bottleneck
  causes the basin.
* **ANNEAL-INHERITS (Stage 2 only):** ANNEAL reaches long accuracy ≥ 0.99 at λ = 1 in every seed it runs on. The narrow
  code inherits the carrier.
* **STOP RULE (Audit, fixed now):** if RAW-H does not rescue seed 2 and its plateau count is not below §120's S5 rate
  (2/4), the task-agnostic line **stops trying to remove the basin**.
  - It adopts label-free restart-on-plateau at 2k (re-seed the run when 123-3's flag fires) as a disclosed engineering
    reliability procedure: best-of-n with early triage (Dodge 2002.06305). Restarts and **cost per accepted seed** are
    reported.
  - It extends the families (cycle navigation, multiplication via items) with that procedure. There is no fourth basin
    fix.

**Readings.**
* **BOTTLENECK-CAUSE + ANNEAL-INHERITS.** The 8-d bottleneck causes the basin, and entering through a wide carrier fixes
  it. ANNEAL joins the harness.
* **BOTTLENECK-CAUSE without ANNEAL-INHERITS.** The bottleneck causes it, but a narrow code cannot take over. Restart-on-
  plateau, as the stop rule.
* **RAW-H plateaus on seed 2.** The basin comes from final-answer supervision without a reader, not the bottleneck. Per
  the stop rule: restart-on-plateau. (Literature: the mechanistic fix would be a reader or COPY, noted and not run.)

**Placement.** Stage 1: 4 runs, about $2.5. Stage 2: ≤ 4 runs, about $2.5, only on the trigger. RTX 3090-class only,
proof seed first (RAW-H s2).
Implementation: the experiments session (fsq120 `--raw_h` and `--anneal`). Scorer `theory/score_s126.py`, committed
before any run.

**Note 126-1 (2026-10-05 16:27 EDT, before any §126 run; a consistency fix).** The STOP RULE's trigger is **RAW-H seed 2 plateaued**,
matching the readings and Audit's sequential plan ("if RAW-H s2 also plateaus … the stop rule fires").
* The plateau-count clause (≥ 2/4) is descriptive.
* As registered, a seed-2 plateau with a count of 1/4 would have met neither BOTTLENECK-CAUSE nor the STOP RULE. Now it
  fires the STOP RULE.

**Note 123-9 (2026-10-05 16:39 EDT; defined after the σ = 0 cells landed and before any dose cell (σ 0.1 / 0.25 / 0.5-3090); disclosed;
Audit / claude-13). ρ_ff, the floor-free fp32 ρ. This is the last metric amendment before the §123 verdict.**
* **The problem.** In the 8-d bottleneck the fp32 floor arrives by hand-off 4–5, after which rho_k is noise over noise
  (0.45–0.85), inflating the k = 1..8 geometric mean.
  - S5 σ 0 s1's median distance runs 0.46 → 0.06 → 0.0099 → 0.0015 → 0.0002 → 0.
* **ρ_ff := the geometric mean of fp32 rho_k at ε = 0.1 over hand-offs whose preceding median distance exceeds 1e-3.**
  At least 2 such hand-offs are required; otherwise the cell has no ρ and CAUSAL is "incomplete".
  - Examples: S5 σ 0 s1 **0.146** (fp32-registered 0.191), FF σ 0 s1 0.050 (0.172).
* CAUSAL (i)–(iii) and σ* read ρ_ff. The fp32 k = 1..8 and bf16 values are printed beside.
* **The thresholds (0.3 / 0.6) are unchanged.** Note the scale: ρ_ff sits lower than §118's fp32 k = 1..8 values (themselves
  floor-inflated upper bounds), so the thresholds bind more cells.
* **The 123-3 plateau exclusion is still what protects (ii):** a basin seed has ρ < 0.3 at accuracy 0 in any precision.
* **Cap (Audit):** no further metric amendments for §123. If ρ_ff misbehaves on the dose cells, it is reported as a
  measurement limitation.

**Note 126-2 (2026-10-05 17:00 EDT, before any §126 run; ANNEAL implementation 535f5a98, accepted; not launched).**
* The distillation runs only during the ramp (2k < step ≤ 6k), so ANNEAL equals RAW-H **in definition** through the
  window where the plateau is decided (before 2k).
  - **In practice it is not bit-identical** (Audit). ANNEAL s2 is 2.55 / 1.35 at 1k / 2k, against RAW-H s2's 3.42 / 1.65.
  - The causes: the bf16 cast of the mixed carrier (126-5), and a different host / driver.
  - Not causes: both arms build the same ln / down / up modules (same init RNG); data comes from a numpy RNG seeded by the
    seed (same order); and W_up and W_down get zero gradient at λ = 0 (the branch is multiplied by 0) and draw no dither
    before 6k.
  - Both escaped by 2k, so the verdict is unaffected.
* std(h) statistics are re-estimated once at 6k (the FSQ switch), then frozen. After λ = 1 they no longer enter the carrier.

**Note 124-3 (2026-10-05 17:00 EDT, before §124 is scored). Matched units are per seed and per GPU type.**
* Trajectories are seed-and-GPU-dependent (125-2), so each seed's trio (BASE, COPY, EXPOSE) must train on one GPU type.
* Seeds 28–29 run as trios on single RTX 4090 hosts (Audit's within-seed rule).
* For seeds 26–27, COPY and BASE must run on the same GPU type as EXPOSE (c1). A seed whose trio spans GPU types is
  reported but excluded from MECH-112.
* The training GPU is listed per run.

**Note 126-3 (2026-10-05 17:24 EDT, before any §126 run; Audit's supply fallback).**
* **Trigger:** if no clean RTX 3090 / 3090 Ti rents within about 1 h, Stage 1's proof runs as a **matched pair on one
  clean 24 GB host of any class**: RAW-H s2 plus a fresh §120 FINAL s2 control (same flags, P = 2, about +$0.6).
* **Decisive readout: RAW-H s2 against the same-host control s2.**
  - Control plateaus (as on 3090-class): RAW-H s2 decides. Plateau fires the STOP RULE (126-1); escape is
    BOTTLENECK-CAUSE's seed-2 condition.
  - Control escapes on that host: the host has changed the trajectory, the test is **uninformative** there, and it is
    retried on another host.
* RAW-H s1 / s3 / s4 (b20) then run on the same host type. Their plateau count enters BOTTLENECK-CAUSE only with
  same-host controls; otherwise it is descriptive.
* The training GPU is logged and reported.

**Note 126-4 (2026-10-05 17:45 EDT; the decisive seed-2 readout as reported from the host, RTX 3090; the formal score follows from the
fetched files). Without the bottleneck, seed 2 does not plateau: the 8-d bottleneck causes the basin at that seed.**

| step | RAW-H s2 (3090) | §120 FINAL s2 (3090 Ti) | §125 STD s2 (3090) |
|---|---|---|---|
| 1k | **3.423** | 4.98 | 4.984 |
| 2k | **1.648** | (4.98 at 1.5k) | 5.000 |

* RAW-H s2 descends faster than any bottlenecked seed (§120 FINAL s1 was 4.76 at 1.5k). By 123-3's flag it is not a
  plateau.
* **The STOP RULE does not fire.** Stage 2 (ANNEAL) is triggered for seed 2 and deploys on a 3090.
  - RAW-H s1 / s3 / s4 (b20) complete BOTTLENECK-CAUSE's count clause.
  - Long accuracy, final loss and donor are reported beside the flag when b19 lands.
* **Provisional (Audit's reproduction from log.jsonl).** Loss is 8.09 at step 100, 3.42 at 1k and 1.65 at 2k (ratio
  0.48, so no plateau). It then bumps to 3.06 at 2.5k and 2.57 at 3k, the L_max step-up at 2k that §120 cells also show.
  - The escape at 2k is established. Whether seed 2 finishes exact is open until b19's long accuracy and final loss land.
    "The bottleneck causes the basin" is provisional until then.

**Note 126-5 (2026-10-05 18:09 EDT; descriptive). ANNEAL s2 also escapes before 2k, but is not bit-identical to RAW-H there.**
* ANNEAL s2 (RTX 3090, driver 560): loss 2.546 at 1k and 1.351 at 2k, against RAW-H s2's 3.423 / 1.648 (RTX 3090, driver
  550).
* **Cause (the experiments session):** at λ = 0, ANNEAL's carrier is cast to W_up's bf16 output dtype under autocast,
  while RAW-H carries std(h) in fp32. The driver difference may add to it.
* Both seed-2 runs escape, so the seed-2 plateau readout (126-4) holds in either form. No rerun.
* ANNEAL's own question, whether the bottleneck holds at λ = 1, is read from its final accuracy.

**Note 126-6 (2026-10-05 18:16 EDT; Audit / claude-13). Before RAW-H is read, check its carried norm.** RAW-H uses the same EMA standard-
isation map, on the 768-d h.
* The carried_norm at init and at 2k is reported beside RAW-H's plateau flag and verdict.
* If the map blew up at 2k as in §125, RAW-H's escape was made with a carrier of uncontrolled scale. The verdict then reads
  "no 8-d bottleneck, carrier as delivered" rather than "standardised h". The bottleneck-vs-no-bottleneck contrast stands
  either way.
* Any RAW-H plateau would equally be on an arm whose map was not in force.

**Note 123-10 (2026-10-05 18:29 EDT, before the σ = 0.5 RTX 3090 cells land; Audit / claude-13). A reading pre-stated for the dose
response.**
* **Matched 3090 cells so far.** Seed 1 has ρ_ff 0.146 / 0.180 / 0.149 at σ 0 / 0.1 / 0.25, long accuracy 1.0 at every dose.
  Seed 2 plateaus at every σ (plateau count 1/2 per level).
* **Pre-stated:** the dose response reads **"flat within single-seed noise"** unless σ 0.5's ρ_ff (seed 1, if not
  plateaued) differs from 0.146 by more than the spread already seen (0.146–0.180).
* In the result: CAUSAL (i) rests on one non-plateau seed per level; σ* for flip-flop (currently 0.25, by 0.149 against
  0.180) is a single-seed pick; (ii) is checked on the non-plateau cells; and (iii) has no cell (no non-plateau cell
  with ρ_ff > 0.6).
* So far, dither changes neither which seed escapes nor how strongly the escaping seed contracts.

**Note 126-7 (2026-10-05 19:20 EDT; descriptive, RAW-H s2 final; the §126 verdict waits for s1 / s3 / s4 and ANNEAL). Without the
bottleneck the basin seed escapes, but the raw carrier drifts with length.**
* **RAW-H s2** (RTX 3090):
  - Accuracy 1.0 at 16 / 32 / 64, 0.9945 at 128, 0.93 at 256, **0.846 at 512**, **0.732 at 2,048**. Final loss 1.6e-5.
  - Donor 1.0 at positions 2 / 4 / 6 (recipient at chance): serial.
  - **The map was in force (126-6):** carried_norm row 2.18 → 3.91 at 2k (wte 3.96), with no blow-up. So this is a verdict
    on standardised h.
* **The emerging split in the task-agnostic harness:**
  - **768-d raw carrier (standardised): reliable entry** (seed 2 escapes), but **drift with length** (bf16 ρ 0.695;
    the fp32 value is pending). This is the §111 READ / §108 pattern without an anchor.
  - **8-d bottleneck: no drift once entered** (ρ_ff about 0.15; exact to 2,048 in every non-plateau seed), but **entry is a
    seed lottery** (seed 2 plateaus in every arm).
* **RAW-H s2 fp32 contraction (Audit / claude-13; verified, with a correction):** ρ_ff = **0.794** under 123-9's
  definition (the floor is never reached, so all 16 hand-offs count; rho_k 0.12 → 0.95). The fp32 k = 1..8 value is 0.653.
  Either way it sits against 0.146–0.180 in every exact bottlenecked seed.
  - So **the 8-d bottleneck supplies both the basin and the attractor**. This reproduces 118-5's ρ ↔ exactness link in fp32,
    without artefacts. It is a point outside §123 that fits (iii): ρ > 0.6 gives long accuracy < 0.99.
  - §126 wording: "RAW-H s2 escapes (valid; carrier standardised at 2k) but is not exact (0.846 at 512 / 0.732 at 2,048,
    ρ_ff 0.794)".
* **ANNEAL-INHERITS is now the load-bearing question.** ANNEAL s2's final long accuracy and ρ_ff are compared with RAW-H's
  0.846 / 0.794 and the bottlenecked s1's 1.0 / about 0.15.
* ANNEAL (enter wide, then narrow) aims to combine the two. So far it does not: s2's loss jumps to 5.43 when λ reaches 1
  at 6k, with a distillation residual of 0.49, and only partly recovers. The final verdict is pending.

**Note 123-11 (2026-10-05 19:23 EDT, before flip-flop's σ* cell lands; Audit). The reading is fixed: the manipulation check failed, so §123
does not reach the causal question.**
* **The matched S5 dose panel is complete (RTX 3090).** Seed 1's ρ_ff at ε 0.1 is **0.146 / 0.180 / 0.149 / 0.156** at σ 0 /
  0.1 / 0.25 / 0.5, with long accuracy 1.0 at every dose. Seed 2 plateaus at every σ.
  - 123-10's pre-stated reading applies: **flat within single-seed noise** (σ 0.5 is inside the 0.146–0.180 spread).
* **σ\* = 0.25** by rule (lowest mean ρ_ff over non-plateau σ > 0 cells: 0.149 against 0.156 / 0.180). It is a
  single-seed pick, and flip-flop's σ\* cell (s123_ff_dither025_s1) is the 10th cell.
* **Wording, fixed now:** "On this harness dither changes neither contraction, nor accuracy, nor the basin; the causal
  question is not reached."
  - CAUSAL (i) fails because the manipulation did not move ρ, not because ρ is non-causal.
  - §123 is **uninformative** about causality, not evidence against it. 118-5's correlation stands untouched.
  - The registered "Neither" reading ("stability needs something these objectives do not provide") **overstates** it and
    is not used.
* **The section's findings are descriptive:**
  - the continuous 8-d carry is exact to 2,048 (123-6);
  - the seed-2 basin appears at every σ;
  - FSQ+DITHER codes are smaller (21 against 34 codes per state at 2,048) but open on S5.

**Note 126-8 (2026-10-05 20:35 EDT; ANNEAL s2 final, RTX 3090, pre-126-5 cast). ANNEAL-INHERITS cannot be met: the narrow code does not
take over the carrier.**
* **ANNEAL s2:** long accuracy 0 at every length 16–512, 2,048 at 0, final loss 3.18. 5,462 codes (359 per state), donor
  near chance (0.04).
  - It escaped like RAW-H to 2k (1.35). The loss jumped to 5.43 when λ reached 1 at 6k (distillation residual 0.49) and
    never recovered.
* ANNEAL-INHERITS needs ≥ 0.99 in every ANNEAL seed, so **it is decided false**. No further ANNEAL seeds are run on the
  remaining escaped RAW-H seeds: they cannot change it, and the run is saved.
* **Pending:** RAW-H s1 / s3 / s4, for BOTTLENECK-CAUSE's count.
  - If they escape, §126 reads "BOTTLENECK-CAUSE without ANNEAL-INHERITS": the bottleneck causes the basin, but
    narrowing a working wide carrier into it fails.
  - Per §126's reading, restart-on-plateau is then adopted as the reliability procedure.

**Note 126-9 (2026-10-05 20:46 EDT; descriptive, RAW-H s3; before §126 is scored). Escaping the 2k plateau is not learning: the raw
carrier fails at long chains in a second way.**
* **RAW-H s3** (RTX 3090, run alone): loss 2.54 at 1k and **1.83 at 2k**, so 123-3's flag says not plateaued.
  - It then degrades as L_max grows: 3.20 at 3k (L 4), 3.76 at 6k (L 8), 4.20 at 10k (L 16), 4.60 at 16k (L 32), 4.42
    final.
  - **Long accuracy 0 at every length**, 2,048 at 0, donor at chance, bf16 ρ 0.74.
* RAW-H so far: **no-plateau in 2/2 by the 2k flag** (s2, s3). But **long accuracy is 0.846 (s2) and 0 (s3)**.
* **Reading caution for the §126 result.** BOTTLENECK-CAUSE is an entry criterion (the 2k plateau flag), and RAW-H may meet
  it formally while producing no exact model.
  - The result will state both: "the 8-d bottleneck causes the **2k entry** basin; without it, entry is reliable, but the
    raw carrier does not learn long chains reliably (s2 drifts, s3 fails)".
  - Each seed's long accuracy sits beside its flag.

**Note 126-10 (2026-10-05 20:53 EDT, before the rerun; Audit). Could the bf16 cast have caused ANNEAL s2's collapse? A fixed-cast rerun
settles it.**
* **Not the cause at λ = 1:** from 6k on, the carrier is W_up(z_b) alone, computed exactly as in every working
  bottlenecked arm (§120 / §123, bf16 autocast). The fix (024150c0) changes only λ < 1. The collapse step itself is
  identical in old and fixed code.
* **Possibly a contributor during the ramp** (2k–6k, 0 < λ < 1): the old code bf16-rounded the mixed carrier (relative error
  about 0.4 %).
  - That is small against the 0.49 distillation residual at the switch. The residual is a capacity mismatch, since 8 dims
    cannot reproduce std(h).
  - But a different ramp trajectory could leave a different state at 6k, so the cast cannot be excluded by argument
    alone.
* **Decision (credit is no longer binding):** one rerun of **ANNEAL s2 with the fixed cast** (fp32 carrier while λ < 1;
  RTX 3090; dir `s126_s5_anneal_s2_fp32cast`). Reading, fixed now:
  - **Fails** (long accuracy < 0.99 at λ = 1): ANNEAL-INHERITS = false stands on a delivered arm.
  - **Succeeds** (≥ 0.99): the first ANNEAL s2 is reclassified as undelivered, and ANNEAL-INHERITS is re-scored on the
    rerun. ANNEAL then also runs on the other escaped RAW-H seeds.
* §126's verdict waits for this rerun and for RAW-H s1 / s4.

**Note 126-11 (2026-10-05 21:18 EDT; erratum, wording only). "ANNEAL-INHERITS" was corrupted by the doc-writing script.**
* The script substituted the timestamp for the placeholder "TS", which also matched the end of "ANNEAL-INHERITS". The rule
  name became "ANNEAL-INHERI<timestamp>" in 8 places: §126's rules and readings, and notes 126-8 and 126-10.
* All 8 are restored to "ANNEAL-INHERITS". No rule, number or reading changes (`git show` on this commit shows the diff).
* Theory's patch scripts now use a placeholder that can't occur in text.

**§123 result (2026-10-05 21:19 EDT; scored by `theory/score_s123.py`, all 10 dose cells and 4 FSQ+DITHER runs, ρ_ff under 123-9).
The causal question is not reached: dither moved neither contraction, nor accuracy, nor the basin.**
* **Wording, as fixed by 123-11:** "On this harness dither changes neither contraction, nor accuracy, nor the basin; the
  causal question is not reached."
* **CAUSAL = false, by (i) on both families.** (ii) and (iii) hold on the non-plateau cells, (iii) vacuously.
  - S5 seed 1: ρ_ff 0.146 / 0.180 / 0.149 / 0.156 at σ 0 / 0.1 / 0.25 / 0.5, long accuracy 1.0 at every dose.
  - S5 seed 2: plateaued at every σ.
  - Flip-flop s1, ρ_ff **0.050 at σ 0 against 0.087 at σ\* = 0.25**: the ρ rose, and both cells are exact (1.0; p_ignore 0.98 / 0.1 also
    1.0).
  - (i) fails because the manipulation did not lower ρ, so §123 is **uninformative about causality**. 118-5's correlation
    stands, and 126-7's RAW-H point (ρ_ff 0.794, not exact) fits it.
* **The registered all-cells reading is printed beside.** (ii) is false there only because of a plateaued cell (S5 σ 0.5 s2:
  ρ_ff 0.019, accuracy 0). **A collapsed code is contractive without information**, which is why 123-3 excludes
  plateaued cells.
* **CANONICAL is not met:**
  - S5: s1 has 20.9 codes per state at 2,048 against 34.2, but s2 plateaued.
  - Flip-flop: 15.5 / 4.0 against 13.75. s1's task-language CERT holds; the registered CERT does not.
* **Descriptive findings kept:**
  - the continuous 8-d carry is exact to 2,048 in every non-plateau seed, with no implicit quantisation (grid share 0.15–0.24);
  - the seed-2 entry basin appears at every σ, on S5 only.

## 127. Reliable entry without restarts? Four task-agnostic entry fixes on §120's harness, against restart-on-plateau (pre-registered 2026-10-05 21:25 EDT, after Literature's checks 86df4b3a / ba531055 / e925cf3e and before any implementation or run; the user's "do more research" (~21:20, via Audit); Audit's compact parallel panel)

**Why.**
* On §120's harness the 8-d bottleneck is both the attractor and the entry basin.
  - It is the attractor: every non-plateau seed is exact to 2,048, ρ_ff 0.146–0.180 (§123 result).
  - It is the basin: S5 seed 2 plateaus by 2k in every bottlenecked arm on 3090-class cards (§120, §123 at every σ,
    FSQ+DITHER, §125).
  - Without it, seed 2 escapes, but the wide carrier does not hold (126-7, 126-9). Narrowing it fails (126-8).
* **Diagnosis: posterior collapse** (He 1901.05534).
  - At init, z_b's per-coordinate std across inputs is 0.03 (125-4).
  - claude-13's early replay (seed 1, steps 3–6, fp32): z_b encodes the current op (R² 0.95), far more than the start or
    next state (0.32 / 0.23).
  - With uniform S5 starts, the op alone says nothing about the state. A code that holds the op is useless to the decoder,
    which learns to ignore it.
* **Relation to §126.** §126's registered readings point to restart-on-plateau, and §127 does not amend them.
  Restart-on-plateau is the baseline each arm must beat, and the line falls back to it if no arm is adopted.
* **Prior** (Literature 86df4b3a / ba531055 / e925cf3e):
  - skip connections from the latent into the decoder (Dieng 1807.04863);
  - VICReg 2105.04906, and its use on latent dynamics (PLDM 2502.14819);
  - aggressive encoder training, and a higher LR for new modules (He 1901.05534; LoRA+ 2402.12354);
  - batch standardisation with an absolute floor (Batch Renorm 1702.03275; §72b).
  - New as scoped.

**Harness.** §120 FINAL exactly: S5; FSQ d = 8, L = 8; soft → hard rounding over 2k–4k; L_max 2 until 2k; 24k steps; bs 64;
lr 1e-4.
* RTX 3090 / 3090 Ti only (125-2).
* Each arm adds one change and draws nothing from §120's random streams (new buffers start at zero; B's maps use their
  own generator). So seed k shares §120 seed k's initialisation and data order.

**Arms (one change each).**
* **A. VAR: item-residual variance on z_b (VICReg-style).**
  - Residual: r = z_b − m[x, j], where x is the item the segment read and j the segment's position in the chain. m is a
    stop-gradient EMA (momentum 0.99) of z_b per (item, position); a pair's first sighting initialises its row.
    Position is included because it varies before 2k (Literature e925cf3e (1)).
  - Loss term, over all hand-offs of each forward: w(step) · [mean_d max(0, 1 − sqrt(Var(r_d) + 1e-4)) + 0.04 · (1/8)
    Σ_{d≠d′} Cov(r)²_{dd′}]. w = 1 to 2k, then linear to 0 at 6k.
  - **Why the residual, not Literature's time-wise variance** (ba531055 (b)): before 2k, chains have only 1–2 hand-offs, so
    a within-chain variance has at most 2 samples. Batch-wise and time-wise variance are both met by item identity (R² 0.95
    at init). The residual can only be supplied by what the segment brought in: the start text or the carried code.
  - It uses only the input items, so it is task-agnostic.
* **B. SKIP: skip-to-decoder (Dieng).** Applies at every answer-decoder position after EOT: the carried slot, and every
  teacher-forced or generated answer position.
  - S_ℓ(q_st) is added to the residual stream entering GPT-2 blocks ℓ = 6..11, and S_f(q_st) to the final hidden before
    ln_f. Each S is 8 → 768, with no bias, initialised N(0, 0.02²) from a dedicated generator (seed 12700 + the run's
    seed), so the global random stream is unchanged.
  - Not zero-initialised: zero init gives q_st no gradient at step 0 (Literature e925cf3e (3)).
  - Segments are unchanged. The skip is part of the decoder at test (greedy decoding, donor, CERT).
* **C. STD2: §125-v2, standardisation delivered.**
  - Everywhere a carried embedding enters GPT-2: e′ = μ_E + sd_E ⊙ (e − mean(e)) / max(sd(e), 0.5 · sd_E). The statistics
    come from the current forward's hand-offs, with stop-gradient.
  - At test: an EMA (0.99) of the same statistics, frozen at the end of training.
  - **Manipulation check** (§125's failure): the carried row norm at 2k must be within 0.5×–2× of wte's (3.96). Outside
    that, C is scored "not delivered", not failed.
* **D. LR10.** The new modules (LN, W_down, W_up) train at 10× GPT-2's LR (1e-3), with the same warmup and cosine, AdamW
  wd 0.01, and the global clip unchanged.
* **Not run: a predictive aux on z** (CPC / NextLat). On S5, a code that holds only the op makes the next code exactly
  predictable from the next op, so the trap is built in. Without the next op, uniform ops make the next state independent
  of the current one. Literature agrees (e925cf3e (2)), and notes it would carry signal on flip-flop.

**Stages.**
* **Stage 1:** seed 2 in every arm, in parallel. Seed 2 is the basin seed, plateaued in at least 7 bottlenecked
  3090-class runs. A run whose 2k flag fires may be stopped at 2k; its log is kept, and it is scored plateaued.
* **Stage 2, only for arms whose seed 2 escapes:** S5 seeds 1, 3 and 4, and flip-flop seeds 1–2, in that arm.
  - Plus one shared concurrent control: §120 FINAL S5 seeds 3–4 (run once, if any arm reaches Stage 2).
* **Drop rule (fixed now):** an arm whose seed 2 plateaus is dropped, with no follow-up patch or variant.
  - If no arm escapes, restart-on-plateau is the reliability procedure, and the line moves to the family extension
    (`docs/THEORY_ENTRY_2026-10-05.md`).

**Measures.**
* The 123-3 plateau flag at 2k; final loss; long accuracy (S5 cumulative at 512, no snaps) and at 2,048.
* ρ_ff (fp32, 123-9); donor SERIAL (chance-corrected); codes per state.
* The carried row norm and per-coordinate std at init and at 2k, in every arm.
* **Probe** (claude-13 / Audit; every arm and the control; at init and 2k).
  - Data: a fixed batch with its own RNG, 1,024 chains of length 2, in eval mode.
  - R²_item is the share of z_b's variance explained by the item read; R²_state is the share explained by the state
    after it.
  - Method: per-group means are fitted on half the rows and R² is scored on the other half. Groups unseen in the fit are
    predicted by the global mean.

**Rules (fixed now).**
* **ESCAPE(arm):** seed 2 is not plateaued.
* **ENTRY-FIX(arm):** all of
  - ESCAPE;
  - S5 plateau count ≤ 1/4 over seeds 1–4;
  - long accuracy ≥ 0.99 in at least 3/4 seeds;
  - ρ_ff ≤ 0.28 (118-5's pass band) in every seed with long accuracy ≥ 0.99, so the attractor survives.
* **FF guard:** flip-flop long accuracy ≥ 0.99 in 2/2.
* **ADOPT:** ENTRY-FIX and the FF guard. If several arms are adopted, the harness default is the one with the lowest cost
  per accepted seed; ties go to the fewest new terms (D, C, B, A).
* **Cost (descriptive, n = 4):** cost per accepted seed = (1 + (p / (1 − p)) / 12) / a run-equivalents.
  - p is the S5 plateau rate, and a is the share of escaped seeds with long accuracy ≥ 0.99. A plateaued attempt costs
    about 1/12 of a run.
  - Reported for each arm and for restart-on-plateau on the control (§120 FINAL S5 s1–4; s1 trained on a 3070, and its
    GPU is stated).

**Readings.**
* **An arm is adopted.** A task-agnostic entry fix replaces restarts, and the family extension uses it.
  - Mechanism, if the probe shows it: "the fix moves the code off the item onto the state". That is, R²_state exceeds
    R²_item by 2k where the control's does not.
* **ESCAPE without ENTRY-FIX.** Entry is fixed, but something later fails (the attractor is lost, or other seeds plateau).
  It is reported, and restart-on-plateau stays.
* **No arm escapes.** These four task-agnostic levers do not move the basin. Restart-on-plateau is the procedure, and the
  line moves to the family extension.

**Placement.**
* Stage 1: 4 runs, about $1–4 (stopped at 2k if plateaued).
* Stage 2: at most 5 runs per escaped arm, plus 2 controls. At most about 26 runs in all, about $20–26. RTX 3090-class only.
* Implementation: the experiments session (fsq120 `--var_res`, `--skip_dec`, `--std2`, `--lr_new 10`, and the R² probe).
* Scorer: `theory/score_s127.py`, committed before any run.
* Dirs: `boxes/runs_fsq/s127_s5_{var,skip,std2,lr10}_s{1..4}`, `s127_ff_{arm}_s{1,2}`, `s127_s5_ctrl_s{3,4}`.

**Note 127-1 (2026-10-05 21:27 EDT, before any implementation or run; Audit, after two undelivered interventions today, §116's copy no_grad and §125's std at 2k). Delivery checks per arm, and the undelivered rule.**
* **Logging.** Every §127 run writes lines {"step": s, "mcheck": {...}} to log.jsonl at s = 1, 500 and 2000, on a fixed
  probe batch. These lines carry no "loss" key, so 123-3's flag is unchanged.
  - Also {"step": 0 / 2000, "probe": {...}}: 127's R² probe, plus the descriptive z-ablation Δloss.
  - So a run stopped at 2k still carries its checks.
* **Delivery checks** (they test that the intervention acts, not that it succeeds):
  - **SKIP:** the gradient of the answer loss with respect to q_st through the skip maps alone (q_st fed to them as a
    leaf) is nonzero at steps 1, 500 and 2000 (`skip_grad_q` > 1e-8).
  - **VAR:**
    - (i) the VAR term's gradient into W_down is nonzero at step 1 (`var_grad_down` > 1e-8);
    - (ii) the item-and-position residual std of z_b on the probe batch (mean over dims) is ≥ γ / 2 = 0.5 at 2000
      (`resid_std`). This is the dose check: the term must actually spread the residual.
  - **STD2:** the carried row norm is within 0.5×–2× of wte's at steps 1, 500 and 2000 (`row_norm`, `wte_row_norm`). This
    replaces the 2k-only check in §127.
  - **LR10:** the new-module / GPT-2 LR ratio is 10 (±1 %) at steps 1, 500 and 2000 (`lr_ratio`).
* **Descriptive, not a delivery check:** the answer-loss change at 2k when the carried code (slot and skip) is replaced by
  the batch-mean code.
  - That is the outcome itself: a plateaued model ignores its code by definition.
* **Undelivered rule (Audit).** An arm that fails a delivery check reads **"undelivered"**: its seed-2 plateau is not read
  as a failure of the idea.
  - It gets **one** rerun with a fix fixed now:
    - a code fix, if the cause is a bug;
    - for VAR's dose check (ii), w = 10 in place of 1.
  - If the rerun also fails its check, the arm is "not deliverable as specified" and is dropped. No further reruns.
* **Before Stage 1:** Experiments' gradflow check shows the following at step 1, and a failure blocks launch:
  - VAR's and SKIP's gradients reach W_down;
  - the group LRs (LR10);
  - the std map's row norm (STD2).

**Note 119-1 (2026-10-05 21:28 EDT, before any §119 run is read; Audit's guard (b); the experiments session's port spec, read from the released
code). The ports are fixed as below, with two calls.**
* **Shared.**
  - Chunk: the start text plus op_1..op_4 texts. Steps are the state texts after ops 1–3, and the answer is the state
    after op 4, then EOS.
  - Test: a snap every 4 ops with a history cut (gen97).
  - Budget: 24k updates at bs 64 per method.
  - LR sweep: 3 values ×3 apart around each published LR, on seed 1. The criterion is train_split_final_acc (5,000
    training chunks). Seeds 2–3 run at the chosen LR.
* **Coconut** (code-confirmed):
  - `<bot>`, n = c·s thoughts (the previous position's ln_f hidden), `<eot>`; stage s replaces steps 1..s.
  - AdamW wd 0.01, constant LR, no clipping, optimiser reset per stage.
  - Stages 0–2 at 3k steps each, stage 3 for the remaining 15k.
* **SIM-CoT** (code-confirmed):
  - c = 2, initialised from the same seed's Coconut.
  - The step decoder φ is a separate GPT-2 from the **pretrained base**. It sees only step k's thoughts, then step k's
    text through θ's wte.
  - λ_step = 1 (`loss += 1.0 * loss_explain_all / c_thought_num`). φ is unused at test.
* **CODI** (code-confirmed):
  - LoRA r 128 / α 32, dropout 0.1, on attention and MLP projections; base frozen. `<bot>` / `<eot>` frozen at the wte
    mean. prj = Linear-GELU-Linear-LN. 4 latents.
  - Loss: student + teacher + KD, weights 1. AdamW wd 0.1, 3 % warmup + cosine, clip 2.0.
* **Faithfulness (b) is met by all three:** Coconut ProsQA 97.6 vs 97.0; SIM-CoT GSM8K 44.8 vs 44.8; CODI 42.5 vs 43.7.
  Trainable parameters are reported (CODI about 20M; Coconut 124M; SIM-CoT 249M with φ).
* **Call 1: CODI's KD loss is SmoothL1, as in the released code.** §119's "L1" followed the paper's text. The released
  code is the method that reproduced the published number (b), so it is used, and disclosed.
* **Call 2: chunk-end cumulative.**
  - For the baselines, in-chunk states are latent, so cumulative 512 / 2,048 counts **chunk ends only**. This is weaker
    than READ's all-positions cumulative.
  - The length curves are descriptive (§119), and CONTROLLED-C1 rests on the donor test.
  - Wherever the two are compared, READ's own chunk-end cumulative is reported beside its all-positions number, so the
    comparison is like for like.
* **Donor:**
  - Coconut: slot j = thought j.
  - SIM-CoT: slot j = thoughts 2j−1 and 2j, jointly.
  - CODI: a block patch of all 4 latents, with the donor target B's final state.
  - Removal (zero / mean) is descriptive.

**Note 124-4 (2026-10-05 21:55 EDT; MECH-112 decided by count, before §124 is complete; Audit / claude-13, reproduced by `theory/score_s124.py`). MECH-112
is false: TEXT-EXPOSE has already acquired in 3 seeds.**
* **EXPOSE acquires in s26 (L6), s27 (L5) and s29 (L8).** "TEXT-EXPOSE in ≤ 2/4" cannot hold whatever s28 does, so
  MECH-112 is false.
  - s29 acquires late: k = 1 at 1.0, cumulative 512 0.80, with the slot-1 bypass.
* **Still open:** NOT-112-4 versus the descriptive reading, on COPY / BASE s26–28 and EXPOSE s28.
  - So far there is one complete trio (s29): COPY acquires at L5, and BASE does not.
* **The result will report,** within the registered readouts (count and acquisition round; FOOTHOLD per arm, not read as
  mechanism, 124-2):
  - by count, whether COPY and plain exposure to random start texts both lift acquisition above BASE. If they do,
    "reading a given state" and "seeing non-identity start text" are not separable at this n;
  - by round, whether COPY acquires earlier than EXPOSE;
  - descriptively, each arm's donor profile (slot carriage against the slot-1 bypass).

**Note 112-6 (2026-10-05 22:08 EDT; the result of 112-5's pre-stated check, all three resumes in, read by Theory from the raw eval.json files; Audit /
claude-13 reproduced s22 / s24). The budget was the limit. One extra round confirms 3/3, through a bypassing carrier.**

| Resume (`--resume_lr 3e-5`) | GPU | 512: source → resume | 2,048: source → resume | k = 1 (never-labelled) | Slot 1 donor / recipient | Slots 2–3 donor |
|---|---|---|---|---|---|---|
| f = 0.1 s22 | 3070 | 0.9645 → **0.995** | 0.858 → **0.984** | 1.0 (12,960), certified | 0.009 / 0.716 | ≤ 0.007 |
| f = 0.5 s24 | 4060 | 0.968 → **0.996** | 0.886 → **0.982** | 1.0 (7,200), certified | 0.0095 / 0.602 | ≤ 0.01 |
| §110 s1 | 4060 | 0.9505 → **0.994** | 0.826 → **0.986** | 1.0 (all 14,400), certified | 0.0105 / 1.0 | ≤ 0.013 |

* **The three-way reading is "confirms" in every resume:** cumulative ≥ 0.99 at 512 with k = 1 at 1.0. The pre-stated
  prediction (s22, s24) holds 2/2, and §110 s1 agrees.
* **Candidate default (as 112-5 stated):** `--post_acq_steps 9k`, triggered label-free when the gate reaches 0.5 with
  kept_share ≥ 0.1, for future self-training registrations.
* **Caveats:**
  - n = 3, on mixed GPUs.
  - A resume is not the same as in-run budget: AdamW and the pools are fresh, at 3e-5 cosine.
  - §124 EXPOSE s27 is the disclosed exception: it acquired early and still ended at 0.9795. So the budget is sufficient
    here but not guaranteed.
* **No C1.** All three resumed models carry no state in the slots (donor ≤ 0.013 at every slot), and slot 1 is a bypass
  (recipient 0.60–1.0).
  - The extra budget fixes C6 / C2 (cumulative 512 and 2,048) through a bypassing carrier.
  - Slot carriage in self-training, so far, comes only with §124's COPY (s29: donor 0.88 / 0.75 / 0.52).

**Note 127-2 (2026-10-05 22:31 EDT, Stage 1 deploying, before any §127 result is read; implementation details from the experiments session, 0baed523;
no rule change). The delivery checks measure what GPT-2 sees in training.**
* **Training semantics.** 127-1's checks run deterministically: eval mode, no RNG draws.
  - Codes are soft unless p_hard = 1. Every check step (1 / 500 / 2000) is in the soft phase.
  - STD2 uses the hand-off's batch statistics.
  - Why: plain eval mode hard-rounds codes and gives STD2 its test-time EMA. In the smoke that gave a row norm 2.0–4.3×
    wte's, a carrier GPT-2 never sees in training. On training semantics it is 0.83×.
  - The plain-eval values are logged beside the checks as descriptive (`row_norm_eval`, `resid_std_eval`).
  - The R² probe stays in plain eval mode, as registered.
* **STD2's statistics are per hand-off**, from that hand-off's batch rows. §127 said "the current forward's hand-offs", but a
  sequential chain cannot standardise hand-off t with statistics from later hand-offs.
  - The test EMA is updated once per optimiser step, from that forward's per-hand-off average.
  - The flip-flop start is not a hand-off: it uses the EMA once one exists.
  - The train / test gap (Batch Renorm 1702.03275's issue) is reported descriptively at the end: row_norm in training
    semantics against plain eval.
* **The gradient checks** (skip_grad_q, var_grad_down) use the first 256 of the 1,024 probe chains, for memory. The other
  keys use all 1,024.
* **Stopped runs.** A run stopped at 2k is drained as `<dir>.void_partial` and renamed back to its run dir, so the scorer
  reads its log. This is disclosed in the drain commit.
* **Training details:** `--micro 2` acts only at L ≥ 32 (after 14k), so it cannot touch Stage 1's 2k reading. `--no_cert`:
  CERT is post-hoc.

**§126 result (2026-10-05 22:38 EDT; scored by `theory/score_s126.py` and by count; RAW-H s4 pending, descriptive, since it cannot change either rule).
The 8-d bottleneck causes the 2k entry basin. Without it entry is reliable, but the wide carrier does not learn long chains,
and narrowing it into the bottleneck fails.**

| Arm, seed (RTX 3090) | 2k plateau flag | Long accuracy (512) | 2,048 | Final loss | Row norm at 2k (wte 3.95) |
|---|---|---|---|---|---|
| RAW-H s1 | no | 0 | 0 | 3.11 | 4.06 |
| RAW-H s2 | no | 0.846 | 0.732 | 1.6e-5 | 3.91 |
| RAW-H s3 | no | 0 | 0 | 4.42 | 3.81 |
| RAW-H s4 | no | 0 | | 4.07 | |
| ANNEAL s2 (bf16 cast) | no | 0 | 0 | 3.18 | |
| ANNEAL s2 (fixed cast, 126-10) | no | 0 | 0 | 2.46 | 3.91 |

* **BOTTLENECK-CAUSE = true (decided).** RAW-H's seed 2 is not plateaued, and the plateau count is 0 over s1–s3, so ≤ 1/4
  holds whatever s4 does. Every bottlenecked seed 2 on the 3090 class plateaued: 7 runs.
  - The map was in force (row norm about 4 at 2k).
* **ANNEAL-INHERITS = false (decided, 126-8; delivered by 126-10).** Both casts collapse when λ reaches 1 at 6k (loss 5.43 /
  5.31) and never recover (264–359 codes per state, donor at chance).
  - The cast changed the trajectory from step 1, but not the outcome.
* **The STOP RULE did not fire** (RAW-H seed 2 escaped).
* **Registered reading: "BOTTLENECK-CAUSE without ANNEAL-INHERITS":** the bottleneck causes the basin, but a narrow code
  cannot take over from the wide carrier. Restart-on-plateau is the reliability procedure.
* **126-9's wording, with per-seed accuracy beside the flags:** escaping the 2k flag is not learning.
  - The raw carrier drifts (s2: 0.846 at 512, ρ_ff 0.794) or fails as L_max grows (s1, s3: loss rises after 4k to 3.1–4.4,
    accuracy 0).
  - So on this harness the 8-d bottleneck supplies **both** the entry basin and the attractor (ρ_ff 0.146–0.180, exact, in
    every bottlenecked seed that enters).
* **What follows:**
  - §127 (registered after this reading's direction was known) tests task-agnostic entry fixes on the bottleneck itself,
    with restart-on-plateau as the baseline to beat. §126 does not change §127.
  - fp32 contraction for RAW-H s1 and the ANNEAL rerun is descriptive. RAW-H s4 is added when it lands.
  - **RAW-H s4 landed (2026-10-05 23:00 EDT):** not plateaued at 2k, long accuracy 0, final loss 4.07 (bf16 ρ 0.715). The count is 0/4
    plateaued, and long accuracy is 0.846 / 0 / 0 / 0. Nothing changes.

**Note 127-3 (2026-10-05 23:07 EDT; Stage 1 b27 read on-host by the experiments session; before any rerun and before any Stage 2 run). VAR is
undelivered and gets one rerun with its estimator corrected. SKIP escapes, but its run hit NaN at L16, so it gets one rerun
with a non-finite guard.**
* **VAR s2 (RTX 3090): plateaued.** Loss 8.84 / 5.02 / 4.98 at 100 / 1k / 2k; stopped at 2k.
  - **Delivery check (i):** var_grad_down was NaN at steps 1 / 500 / 2000. The bf16 gradient *check* is unstable on b27's GPU
    (118.7 on one call, non-finite on another; fp32 gives 0.131). Training stayed finite.
  - **Delivery check (ii) fails:** resid_std (pooled within-(item, position) std on the probe) is **0.117** at 2k, below 0.5.
  - **The cause is the estimator, not the dose.** The residual against the EMA rows reached 1.10 by step 500 (≥ γ). So the
    hinge was satisfied by **the EMA lagging a moving z_b**, which differs by item. Item identity re-entered through the
    lag, and the within-group spread never moved.
  - The probe agrees: R²_item 0.47 → 0.42, R²_state −0.15 at 2k.
* **Deviation from 127-1's pre-fixed rerun, with its reason.** 127-1 fixed w = 10 for a dose failure. Here the hinge was
  already met, so a larger weight multiplies a term that is already about zero: w = 10 cannot address this cause.
* **VAR's one rerun (`s127_s5_var_s2_rerun`):** the residual uses **the forward's own group means**. r = z_b − mean over
  the forward's rows with the same (item, position).
  - Variance and covariance are pooled within groups of ≥ 2 rows. Singleton rows contribute nothing, and a forward with
    no such group skips the term.
  - Weight w = 1 and the schedule are as registered. This is the estimator the delivery check (ii) already measures.
  - If the rerun fails a delivery check or plateaus, VAR is dropped (127-1). No further variant.
* **SKIP s2 (RTX 3090): ESCAPE.** Loss 8.25 / 4.86 / **3.81** at 100 / 1k / 2k (ratio 0.78).
  - Delivered: skip_grad_q 0.0086 / 0.0014 / 0.0032.
  - **Probe at 2k: R²_state 0.28 against R²_item −0.08** (init 0.47 / −0.15 the other way round). The code moved off the
    item onto the state, the hypothesised mechanism (abl_dloss 0.68).
  - For comparison, claude-13's replay of the learning seed 1 has R²_next 0.93 and R²_op 0.24 at 2k.
  - **After 2k the loss returned to about the plateau:** 4.94 at 3k (L 4), about 4.5 at L 8.
  - **Then training went NaN from the switch to L_max 16** (step 9,001). §120 FINAL without the skip never did.
* **SKIP reading.** A NaN makes the run's final accuracy undefined, so it is treated as a numerics defect under 127-1's
  undelivered rule.
  - **One rerun of s2 (`s127_s5_skip_s2_rerun`) with a non-finite guard:** a step whose loss or gradient norm is not
    finite skips its update, and is logged as {"step": s, "nonfinite": 1} with the first non-finite tensor's name.
  - The skip maps' norms are logged every 1k (descriptive).
  - **If more than 1 % of steps are skipped, SKIP is "not deliverable as specified"** and is dropped.
* **Stage 2 opens for SKIP, as registered** (ESCAPE is true): S5 s1 / s3 / s4, FF s1–2, and the shared §120 FINAL control
  s3–4.
  - All of them run with the same guard. On finite steps it is a no-op, so finite runs are unchanged.
  - From now on the guard is in every §127 run, including b28's STD2 / LR10 redeploy and the controls.
  - The non-finite count is reported per run.
* **Gradient checks are computed with autocast off (fp32)** in every run from now on: the same question, asked stably.
  The plain bf16 value is logged beside it (descriptive).
* The scorer counts "nonfinite" log lines. A run above 1 % of its logged steps fails delivery.

**Note 127-4 (2026-10-05 23:31 EDT, before SKIP's rerun or any SKIP Stage 2 run; Literature f12b031b, Audit). SKIP reruns as registered. A
skip-ablated readout is added as descriptive.**
* **No design change to SKIP.** Literature's stability lead (LayerScale 2103.17239 / ReZero 2003.04887 scale, a norm on the
  injected term, a post-entry decay of the skip) is recorded and **not run**.
  - Each would be a new SKIP variant after seeing data, the loop 127-3 closed: one rerun, the non-finite guard only.
  - If SKIP is "not deliverable as specified" (> 1 % skipped steps) or fails ENTRY-FIX, a bounded skip is a candidate for
    a new registration, not a §127 arm.
* **Where the skip acts.** The skip maps feed q_st only into the **answer decoder**. Segments still hand state forward
  through W_up(q) alone.
  - So the registered donor patch at positions 2 / 4 / 6 already tests serial carriage on the slot carrier: the patched
    code reaches the end only through later segments' W_up path.
* **Added (descriptive, every SKIP run with an eval.json):** with the skip maps zeroed at test (the decoder reads the slot
  embedding alone):
  - long accuracy at 512;
  - the donor profile;
  - abl_dloss at 2k (already logged).
  - A large gap between skip-on and skip-off accuracy means the decoder relies on the skip path. For SKIP, C1 is read on
    the skip-off donor beside the registered one.
* No rule changes.

## 128. One recipe, five state-tracking families: §120's FSQ carrier with restart-on-plateau, on S5, flip-flop and three held-out families (pre-registered 2026-10-05 23:38 EDT, after Literature's check 4d50870d / 032eebf5 and before any implementation or run; the user's task-agnostic directive; §126's reading (restart-on-plateau); cloud spend within the user's "keep cloud running", via Audit)

**Why.**
* The user asked for a task-agnostic method. On §120's harness one fixed recipe already gives, in every seed that enters:
  - exact S5 chains to 2,048 with a contractive code (ρ_ff 0.146–0.180, §123 result);
  - certified flip-flop (§120, Moore 2).
* §126 fixed label-free restart-on-plateau as the entry procedure. §127 tests whether an entry fix can replace it.
* §128 asks whether **the same code and hyperparameters**, with restarts, work on families the recipe was never tuned on.
  That is the step from "works on S5" to "a method".
* **Prior** (Literature 4d50870d / 032eebf5, `docs/LIT_S128_FAMILIES_2026-10-06.md`):
  - cross-task state tracking from scratch: Delétang 2207.02098, Ruoss 2305.16843, MLP-LDRU 2605.26035, DeltaProduct / Grazzi,
    Liu 2210.10749 (shortcuts to automata), Merrill;
  - in pretrained LMs, only Li 2503.02854 / 2602.14814, with no certificate and no cross-family result;
  - held-out random automata: Allchin 2607.11953.
  - New as scoped. The families cover the Krohn–Rhodes primitives: S5 (non-solvable group), CYC (abelian group), FF (reset), MOD
    (group action + reset), and a random automaton.

**Recipe (one code version, fixed at launch, identical for every family).**
* §120 FINAL fsq120: d = 8, L = 8; soft → hard rounding over 2k–4k; L_max 2 / 4 / 8 / 16 / 32 (this curriculum is not Delétang's
  U(1, 40)); 24k steps; bs 64; lr 1e-4.
* Plus 127-3's non-finite guard, a no-op on finite steps. RTX 3090 / 3090 Ti only (125-2).
* **Entry: restart-on-plateau.** Seed slot k ∈ 1..4 tries seeds k, k+100, k+200, k+300 in order. An attempt whose 123-3 flag fires
  is stopped at 2k (log kept), and the next seed runs. The slot's accepted run is its first non-plateau attempt.
* If §127 adopts an entry fix (ADOPT true), a follow-up reruns §128 with it; §128 is not changed.

**Families** (non-S5 families start from the code-zero start, as flip-flop; answers are one digit token).
* **S5** and **flip-flop (FF):** exactly §120's.
* **CYC (held out), Delétang's Cycle Navigation:** a 5-cycle, start 0. Items " stay" / " right" / " left" (0 / +1 / −1, one token
  each). Answer: the position " 0".." 4".
* **MOD (held out), Delétang's Modular Arithmetic (Simple) with an implicit leading 0:** a left-to-right fold mod 5 from 0.
  Items " plus d" / " minus d" / " times d", d ∈ 0..4 (15 items, two tokens each). Answer: the value " 0".." 4".
  - " times 0" (1 in 15 items) resets the state.
* **RDFA (held out), a seeded random automaton:** 8 states, 4 inputs " a" / " b" / " c" / " d", start 0, answer the state
  " 0".." 7".
  - The table is the first draw of `np.random.default_rng(12800).integers(0, 8, (8, 4))` that is strongly connected, not a group
    (some input non-bijective), and has no constant input. Rows are states 0–7, columns inputs a–d:
    [[4, 2, 0, 7], [3, 3, 2, 4], [3, 6, 5, 3], [4, 5, 2, 6], [7, 1, 0, 3], [4, 0, 5, 3], [5, 4, 7, 5], [7, 2, 6, 3]]
  - The first draw qualified. Input image sizes are 4 / 7 / 5 / 5, and the shortest synchronising word has length 3.
  - Random strings synchronise in a median of 9 steps (p90 18), so long chains depend mainly on a short suffix, as with MOD's
    resets.
* Nothing in the recipe was tuned on CYC, MOD or RDFA.

**Measures.**
* S5 / CYC / MOD / RDFA: cumulative all-positions accuracy at 512 (no snaps). FF: per-read accuracy at 512, as §120.
* **Delétang's own score** (descriptive, so published numbers are comparable): final-output per-sequence accuracy on 2,000
  sequences with lengths U(41, 500). Delétang's chance is 20 % for the 5-class tasks; RNN / LSTM score 100, Transformer 61.9
  (CYC) / 24.2 (MOD).
* 2,048 (descriptive); for CYC also 8,192 (descriptive).
* Donor SERIAL at positions 2 / 4 / 6: donor ≥ 0.90 and recipient − chance overlap < 0.10.
* Task-language CERT (BFS from the start code over every item; every reachable code decodes validly and obeys the transition
  law) with its Moore-minimised size. ρ_ff (fp32, 123-9); codes per state; §127's R² probe at 0 and 2k.
* MOD / RDFA: the distribution of the effective suffix (steps since the state was last determined regardless of the past), and
  accuracy binned by it.
* Restarts per slot, and cost per exact accepted seed.

**Rules (fixed now).**
* **FAMILY-PASS(f):** at least 3/4 slots have an accepted run that is exact (≥ 0.99 at 512) and SERIAL.
  - **For CYC, MOD and RDFA it must also be CLOSED:** CERT holds, not capped or timed out, with Moore-minimised size equal to
    the minimal state count (5 / 5 / 8).
  - Literature (c): an unreduced counter code is exact at 512 on CYC without being the automaton.
  - For S5 and FF, CERT is descriptive, as in §120 / §123.
  - A slot whose four attempts all plateau fails. A run with more than 1 % skipped non-finite steps fails.
* **TASK-AGNOSTIC:** FAMILY-PASS in all five families. RDFA counts toward it.
* **Scope of a pass:** regular-language state tracking. Tasks needing a stack or tape are outside it.

**Readings.**
* **TASK-AGNOSTIC.** One fixed recipe with a label-free restart rule gives exact, serial, closed discrete latent state tracking in
  a pretrained LM on five regular families, three of them held out. C1 + C2 + C6 (with restarts); C4 as final-answer supervision.
* **The tuned families pass, a held-out one fails.** The recipe is fitted to S5 / FF. Report which property breaks it: exact but
  not closed (a counter), not serial, or not exact.
* **A held-out family fails by entry** (slots exhaust their restarts). Entry, not the attractor, limits generality.

**Placement.**
* 20 slots: at least 20 full runs, plus about 2k-step restarts. About $13–19 on Vast 3090s.
* Implementation: the experiments session (fsq120 `--family cyc / mod / rdfa`, the restart driver, the Delétang eval and the
  effective-suffix bins). Scorer: `theory/score_s128.py`, committed before any run.
* Dirs: `boxes/runs_fsq/s128_{s5,ff,cyc,mod,rdfa}_s{seed}`.
* Multiplication is excluded: about 10¹² states, against 8⁸ ≈ 1.7·10⁷ codes, and widening d would change every family's
  harness. The AGL(1, 5) variant of MOD (no resets) is noted, not run.

**Note 128-1 (2026-10-05 23:43 EDT, before any §128 run; Audit / claude-13; no rule change). What CLOSED means, and a check of the plateau flag on
the held-out families.**
* **CLOSED is not "canonical."** The Moore clause passes a code that is redundant but correct: several codes for one state
  are Moore-equivalent and minimise to it, as with FF FINAL s2's 36 → 2.
  - A counter code fails through the CERT cap or timeout: its reachable codes keep growing.
  - So CLOSED reads "a closed machine whose minimisation is the task automaton". Codes per state is reported beside it.
* **The plateau flag on 5- and 8-state families.** 123-3 was calibrated on S5's ln 120 null. Its two clauses are ratios
  (loss(2k) / loss(1k), loss(2k) / loss(100)), but they have been validated only on S5 and FF.
  - **The registered restart rule is unchanged.**
  - **Added check (descriptive):** in each held-out family (CYC, MOD, RDFA), the first attempt whose flag fires is **not
    stopped**. It trains to 24k beside the replacement seed, which starts as the rule says.
  - Its final long accuracy is reported as the flag's false-positive check for that family. At most 3 extra runs, about
    $2–3.
  - The slot still takes the next seed, per the rule.
  - Every attempt's 2k loss is also reported against its family's null (ln 5 ≈ 1.61 for CYC and MOD; ln 8 ≈ 2.08 for RDFA).
* If a check run ends exact, the flag misfires on that family. This is reported in the result, and the cost per accepted
  seed is given both ways. The registered verdict still uses the registered rule.

**Note 127-5 (2026-10-06 00:12 EDT; descriptive, the seed-1 basin replay (bf16, RTX 3090; `theory/results/replay_basin_s1/`), read by Theory from
replay.jsonl after claude-13's account; a prediction for seed 2's replay, fixed before b28's replay lands). How seed 1 leaves the
basin, and what seed 2 should show.**
* **Parity:** loss 4.951 / 3.802 at 1k / 2k, against §120 FINAL s1's 4.978 / 3.896.
* **Seed 1's order of events** (R² of z_b on the op read, the chain's start state and the next state; dmean = the decoder's
  loss gain from the carrier, L1_dmean):
  - **0–1k: the decoder ignores the code** (dmean ≤ 0.009). The loss is flat at about 5.0.
    - Meanwhile the code drifts off the item: R²_op 0.95 → 0.59 (minimum before 1k; 0.64 at 1k); R²_start 0.32 → peak 0.68
      at 800, back to 0.42 at 1k;
      R²_next flat at 0.22–0.24.
    - The gradient into W_down is 0.8–4.4, against GPT-2's 62–154.
  - **1.0–1.2k: the decoder starts reading** (dmean 0.006 → 0.34 at 1,200). In the same window the code re-targets to the
    next state (R²_next 0.23 → 0.54), its spread grows (zb_std 0.14 → 0.70), and R²_op falls to 0.30.
  - **1.2–2k: consolidation.** R²_next reaches 0.93 and the effective rank 2.8. The W_down gradient grows only after the
    decoder pulls (23.7 at 2k).
* **Reading (one seed, so a hypothesis):** the chicken-and-egg breaks on the decoder side, after the code has drifted off
  the item. The drift precedes the pull, but the R² values are noisy step to step (R²_op rebounds to 0.67 at 1,100).
* **Prediction for seed 2's replay, fixed now** (Audit / claude-13). R²s are compared as 3-point running means over the
  logged steps (every 50):
  - **(P1)** R²_start stays below R²_op at every logged step from 0 to 2k: the code never drifts off the item.
  - **(P2)** dmean stays below 0.05 throughout: the decoder never pulls.
  - Parity first: seed 2's replay loss at 1k / 2k must match §120 FINAL s2's (4.98 / 4.95) within 0.1, or the replay is
    not read.
* **Readings:**
  - **P1 and P2 hold.** The basin seed's code stays on the item, and the seeds differ in the drift. This matches 127-3's
    VAR probe (R²_item 0.42, R²_state −0.15 at 2k) and SKIP's change (R²_state 0.28). The registered VAR rerun (residuals
    against the forward's own group means) is the test of pushing the early code off the item. **No new arm.**
  - **P1 fails, P2 holds.** Seed 2's code drifts too, but the decoder never pulls, so the drift is not sufficient. The
    trigger is on the decoder side, SKIP's direction.
  - **P2 fails.** Seed 2's decoder reads its code by 2k, against the plateau. Then the replay is not faithful to §120 s2.

**Note 119-2 (2026-10-06 00:15 EDT; Coconut's seed-1 LR sweep read on-host by the experiments session; before Coconut seeds 2–3 or any SIM-CoT run).
Coconut did not learn at any of the three LRs, and §119 continues exactly as registered.**
* **Coconut s1** (chunked host, RTX A5000), at lr 1e-4 / 3e-4 / 3e-5:
  - intact accuracy 0.0085 / 0.0075 / 0.0070, and train_split_final_acc 0.0076 / 0.0092 / 0.0102. Chance is 1/120 = 0.0083.
  - Cumulative 0 at every length; donor and removal at chance.
  - Stage-3 loss 0.70–0.72, which is about ln 120 / 7: the answer is at chance per token.
  - The port does learn text CoT: stage 0 reaches loss 1e-4 at lr 3e-5. Latent composition fails by stage 3.
  - **Guard (a): "did not learn", in all three.**
* **Decision: run as registered.**
  - The LR picked by the registered criterion is 3e-5 (0.0102). All three values are at chance, so the pick is
    noise-level; disclosed.
  - Coconut seeds 2–3 at 3e-5 (about $2).
  - SIM-CoT as registered: its own 3-value LR sweep on s1, initialised from Coconut s1 at 3e-5, then seeds 2–3 from Coconut
    s2–3.
  - No amendment to Coconut's curriculum or c. A longer schedule or c = 2 would be a new design chosen after data.
* **Pre-stated expectations:**
  - Coconut s2–3 also "did not learn". Coconut is then **inconclusive** for CONTROLLED-C1 (0 competent seeds), as guard (a)
    reads.
  - **Confirmed (03:47):** Coconut s2 / s3 (RTX 3090, driver 550.163) have intact accuracy 0.0085 / 0.0115 and cumulative 0, so they
    did not learn: 3/3. `score_s119.py`: Coconut is inconclusive for CONTROLLED-C1.
  - SIM-CoT is the informative arm. Its published claim is that step supervision stabilises latents where Coconut
    collapses (2509.20317). Whether it becomes competent from a Coconut start that has not learned is reported as is.
* **Wording:** "Coconut (c = 1, the published stage schedule scaled to 24k updates) did not learn 4-op S5 composition
  through latent thoughts on the chunked host". This is not "Coconut is not serial".

**Note 128-2 (2026-10-06 00:20 EDT, before any §128 run; wording only, Audit / claude-13). What "median 9, p90 18" measures for RDFA.**
* It is **the length of the shortest prefix of a uniformly random input string that maps all 8 states to one state**: the
  number of steps after which the state no longer depends on the start. 4,000 strings give median 9, p90 18.
* For i.i.d. uniform inputs, this has the same distribution as the **effective suffix** at a position (the shortest
  synchronising suffix ending there). Measured on 2,830 positions of a 20,000-step string: median 9, p90 18, the same.
* Earlier, claude-13 reported 12 / 26 for this statistic. It retracted that at 00:22: a scripting error fed each state its own
  independent input instead of one common word. Corrected, it gives median 9 / p90 18, and its effective_suffix agrees.
* The registered measure (effective-suffix distribution and accuracy binned by it) is unchanged.

**Note 127-6 (2026-10-06 00:50 EDT; descriptive, Stage 1 b28 read on-host by the experiments session; the reading of a driver-class check fixed before
it runs; no rule change). STD2 and LR10 escape on seed 2 too, but on a newer driver. The same-host plain replay decides
whether that counts.**
* **b28** (RTX 3090, driver 595.71), seed 2, still training:
  - **STD2:** loss 8.19 / 4.41 / **3.66** at 100 / 1k / 2k (ratio 0.83, no flag).
    - Delivered: row norm in training semantics 3.27 / 3.23 / 3.81 (0.82–0.96× wte); plain eval 8.3 / 11.8 / 6.2.
    - Probe at 2k: R²_item −0.04, R²_state −0.15.
    - After 2k: 4.53 (4k), 4.83 (8k), 4.86 (11.5k).
  - **LR10:** loss 8.16 / 4.97 / **4.01** (ratio 0.81, no flag).
    - Delivered: LR ratio 10.0.
    - Probe at 2k: **R²_state 0.25, R²_item −0.05.**
    - After 2k: 4.72 (4k), 4.69 (8k), 4.61 (13k).
* **The pattern so far:** SKIP (b27, driver 565), STD2 and LR10 all pass seed 2's 2k flag. Each then drifts back to about
  4.5–4.9 as L_max grows: "escape is not learning", as RAW-H (126-9). VAR plateaued (undelivered, 127-3).
* **Confound.** Every seed-2 basin so far (§120, §123, §125; 7 runs) trained on 3090s with drivers 550–565. b28 runs 595.71,
  and GPU class is known to change which seed plateaus (125-2).
* **The check:** `replay_basin` seed 2, plain §120 FINAL s2 with no §127 change, on the same b28 host. Reading, fixed now:
  - **The replay plateaus** (loss at 1k / 2k within 0.1 of 4.98 / 4.95, the flag fires): seed 2 is still the basin seed on
    595.71. STD2's and LR10's ESCAPE stand, as does any §127 run on that driver.
  - **The replay escapes** (no flag): seed 2 is not a basin seed on 595.71. ESCAPE for every §127 arm trained on that
    driver is uninformative: STD2, LR10, the VAR and SKIP reruns.
    - SKIP's original escape (b27, driver 565) stands.
    - The Stage 2 matched unit with its own concurrent control (b30 / b31, 595.71) governs the arms' reading, through
      plateau counts against the control's.
* **Order:** Stage 2 for STD2 / LR10 starts after this replay reads, not before. Stage 2 is still "only for arms whose seed 2
  escapes"; only the timing waits.
* **From now on,** every §127 / §128 run records its driver (`train_driver`, from nvidia-smi) beside train_gpu. §128 reports
  the driver per attempt. Its slots stay valid across drivers, which adds to the robustness claim.
* **Addendum (2026-10-06 01:02 EDT, before any rerun reads).** The VAR and SKIP reruns run on b27's own host (b35, driver 565.77), the machine of
  Stage 1, not on 595. So they are outside the confound whatever the replay shows. SKIP's Stage 2 matched unit (b30 / b31) and
  b28 are on 595.71. Drivers per §128 host: e31 595.71, e33 595.91, e34 595.58 (3090 Ti), e30 595.71, CYC redeploy 565.57.
* **Corroboration (2026-10-06 01:57 EDT; Audit / claude-13, read by Theory from the log).** Plain seed 2 under §128's S5 recipe (§120 FINAL plus the guard,
  a no-op on finite steps), on a second 595.71 host (first e30, RTX 3090, `s128_s5_s2`), plateaus with the same digits as on
  550–565: 8.261 / 4.984 / 4.953 at 100 / 1k / 2k, flagged at 2k (probe R²_state −0.20, R²_item −0.07). So the STD2 / LR10
  escapes on b28 are not a driver artefact on this evidence. b28's same-host replay remains the registered decider.

**Note 119-3 (2026-10-06 01:39 EDT, before any SIM-CoT run, the CODI rerun or Coconut s2–3 read; Audit's steward pass). The all-inconclusive case
is pre-stated, and SIM-CoT's initialisation is kept as registered.**
* **SIM-CoT's initialisation is unchanged:** the same seed's trained Coconut checkpoint, as §119 registered and the
  released pipeline does. Reasons:
  - SIM-CoT's curriculum restarts at stage 0, where all steps are text. On this host the port relearns that stage to loss
    1e-4 within its 3k steps (Coconut s1 at lr 3e-5, 119-2). So Coconut's stage-3 failure need not carry over.
  - Starting from the base model instead would depart from the released recipe, which is the comparison §119 promises.
* **Reported to show the start was not degraded** (descriptive): SIM-CoT's loss at the end of its stage 0 against
  Coconut's (1e-4). If SIM-CoT fails to relearn stage 0 (loss above 0.1 at the end of stage 0), its seed reads "did not
  learn (degraded start)". A base-model SIM-CoT then becomes a candidate for a new registration, not a §119 arm.
* **The all-inconclusive case, pre-stated.** If no method reaches ≥ 2 competent seeds, CONTROLLED-C1 is **not readable**.
  The result then says:
  - "Under their released recipes, at our matched budget (24k updates, bs 64, a 3-value LR sweep per method) on the
    chunked S5 host, Coconut, SIM-CoT and CODI do not learn 4-op S5 composition through latent thoughts (in-chunk
    final-state accuracy below 0.90 in every seed). Text chain-of-thought does learn (stage 0)."
  - "There is no competent published comparator for our serial C1 on this protocol, so the C1 claim stays uncontrolled by
    baselines."
  - "This is a competence finding (C3: latent methods below text at matched compute on this task), not evidence about
    their serial use."
  - Wording limit: "at this budget on this host", never "cannot".
  - No budget extension or recipe change is run under §119. Either would be a new registration.
* **Mixed case:** a method with ≥ 2 competent seeds is scored as registered. The others are reported as above.

**Note 124-5 (2026-10-06 02:43 EDT; descriptive, 8 of 12 §124 runs in (s28 trio on an A10, s29 trio on a 4090, EXPOSE s26–27 on 3090s), read by
Theory from the eval.json files; Audit / claude-13 reproduced; no rule change). Carrier type varies within every arm. COPY's
consistent signature so far is timing.**

| Arm | Acquired (round) | Cumulative 512 | Slots 1 / 2 / 3, donor (recipient) |
|---|---|---|---|
| BASE s28 | L6 | 0.998 | 0.011 (1.0) / 0.007 / 0.007: slot-1 bypass |
| BASE s29 | no | 0 | chance |
| COPY s28 | L5 | 0.9495 | 0.065 (0.28) / 0.044 (0.34) / 0.076 (0.18): mixed |
| COPY s29 | L5 | 0.999 | 0.883 / 0.747 / 0.516 (≤ 0.02): slot-carried |
| EXPOSE s26 | L6 | 0.9985 | 0.011 (1.0): slot-1 bypass |
| EXPOSE s27 | L5 | 0.9795 | 0.017 (0.80) / 0.389 / 0.411: bypass + partial |
| EXPOSE s28 | L6 | 0.993 | 0.618 / 0.701 / 0.751 (≤ 0.024): slot-carried |
| EXPOSE s29 | L8 | 0.8005 | 0.011 (0.999): slot-1 bypass |

* **Acquisition:** EXPOSE 4/4, COPY 2/2 (both at L5), BASE 1/2. FOOTHOLD (descriptive, 124-2): EXPOSE 2/4, COPY 1/2,
  BASE 1/2.
* **Correction to 124-4 / 112-6.** Seed 29 suggested that COPY gives slot carriage. Seed 28 reverses it: COPY s28 is mixed,
  and EXPOSE s28 is slot-carried.
  - So 112-6's line "slot carriage in self-training, so far, comes only with §124's COPY" is **withdrawn**. Slot carriage
    appears in 1 of 2 COPY seeds and 1 of 4 EXPOSE seeds.
  - "COPY → C1" is not claimed unless COPY s26–27 change the picture.
* **COPY's one consistent signature so far is timing:** L5 in both seeds, and the sharpest jump. That is §124's registered
  readout (acquisition round), read when COPY / BASE s26–27 land.
* **Not run:** resuming the k = 1 = 1.0 FOOTHOLD misses (112-5's lever). §124's question is acquisition count and round,
  and FOOTHOLD is not read as mechanism (124-2), so resumes would not inform it.

**Note 127-7 (2026-10-06 03:12 EDT; descriptive, the seed-2 basin replay (bf16, b28 RTX 3090, driver 595.71; `theory/results/replay_basin_s2/`) read by
Theory from replay.jsonl; Audit reproduced; 127-5's predictions scored as written). The driver confound is closed. P1 fails, and P2
fails on the letter at one step. Both seeds make the same first pull attempt; seed 2's fails and its code collapses.**
* **Parity holds:** loss 4.99 / 4.953 at 1k / 2k, against §120 FINAL s2's 4.98 / 4.95. Plain seed 2 plateaus on the same host
  and driver where STD2 and LR10 escaped.
  - **127-6's confound is closed:** STD2's and LR10's ESCAPEs stand. **Stage 2 now runs for both, as registered.**
* **127-5's predictions, as written:**
  - **P1 fails.** R²_start ≥ R²_op (3-point running means) at steps 450–600. Seed 2's code also drifts off the item early:
    R²_op 0.96 → 0.44, R²_start 0.31 → 0.62 by 400–600. It then reverts: at 700, R²_op 0.82 and R²_start 0.45.
  - **P2 fails on the letter at one logged step:** dmean 0.060 at 650. Otherwise it is ≤ 0.027, and exactly 0 from 700 to
    2k.
  - 127-5's reading for a P2 failure ("the decoder reads its code by 2k; the replay is not faithful") does **not** describe
    this. Parity holds, and dmean is 0 at 2k. So no registered reading applies cleanly, and the outcome is reported as
    observed.
* **What separates the seeds (n = 1 each, descriptive):**
  - **Seed 2 at 650: one pull attempt.** dmean 0.060, code spread across inputs (zb_std_in) 0.725, W_down gradient 10.3.
    - It fails. By 700 the code's spread collapses to 0.030 (0.021 at 2k), and dmean stays at 0.
  - **Seed 1's first attempt at 1,050 looks the same:** dmean 0.059, spread 0.698. It also dips, to 0.015 / 0.37 at 1,100.
    - But the next attempts stick: dmean 0.34 and spread 0.70 at 1,200; R²_next 0.93 by 2k.
  - **Reading (a hypothesis):** the basin is a **collapse of the code's variance after a failed first pull**, a measured
    instance of posterior collapse. It is not a failure to leave the item.
* **Consequences, no new arm:**
  - **(a)** The VAR rerun (in-forward group-mean residual, w = 1 from step 1 to 2k) is active through 650–700. It is the
    live test of whether keeping the code's within-group spread up prevents this collapse. Its delivery checks at 500 and
    2,000 are read with that in mind.
  - **(b)** The code spread at 700–1,000 (0.02–0.03 in the basin seed, against 0.05–0.14 before it) is a candidate
    label-free early plateau marker, about 1.3k steps before 123-3's flag. **Descriptive only.** A restart rule built on it
    needs its own registration and a check on §128's attempts.
* **LR10 s2 final** (Audit reproduced): escape without learning.
  - Loss 4.34 at 3k, 4.72 at 4k, final 4.585. Long accuracy 0 at every length, 2,048 at 0, donor at chance; 440 codes, 23.2
    per state.
  - The carrier is inflated (row norm 47.2 at 2k).
  - The same pattern as SKIP s2 and RAW-H.
  - **STD2 s2 final (03:22):** the same. Long accuracy 0 at every length, final loss 4.81, 23.8 codes per state,
    donor at chance, 0 non-finite steps. Stage 2 for STD2 / LR10 is deploying on four RTX 3090s with driver 595.71 (b36–b39),
    matching SKIP's control hosts.
* **Pre-stated for Stage 2:** STD2 and LR10 are unlikely to meet ENTRY-FIX (3/4 exact), given seed 2's drift-back. They run
  because the rule says so.

## 129. Where does the entry basin live? Re-drawing only the bottleneck head at the 2k plateau flag, with GPT-2's body kept (pre-registered 2026-10-06 03:47 EDT, after Literature's check 3fd1f1fe and before any implementation or run; Audit's steward idea A (03:45), on note 127-7)

**Why.**
* In §120's harness the entry basin is a seed property on 3090-class cards: S5 seed 2 plateaus in every bottlenecked arm.
* The seed-2 replay (127-7) times it. The first decoder pull, at step 650, fails. The code's variance then collapses
  (zb_std_in 0.73 → 0.03 at 700), and the decoder never pulls again. Seed 1's first pull looks the same, but its next one
  holds.
* §127's entry fixes that let seed 2 escape the 2k flag did not make it learn (127-7). Restart-on-plateau (§126) is the
  working procedure.
* §129 asks where the basin is stored once the flag fires:
  - in the bottleneck head's state (its init and its weights after the collapse); or
  - in GPT-2's body, which by 2k has learned to ignore the carrier.
* **Prior** (3fd1f1fe): `docs/LIT_S129_HEAD_REDRAW_2026-10-06.md`: primacy bias / resets 2205.07802, ReDo 2302.12902, shrink-and-perturb 1910.08475,
  fortuitous forgetting 2202.00155, RIFLE 2007.03349 (periodic head re-init in fine-tuning), dead-code re-init in VQ (2005.08520,
  Jukebox). New as scoped: a head-only re-draw of an injected-latent bottleneck as a head-versus-body diagnostic.

**Design.**
* §120 FINAL S5 exactly, plus 127-3's non-finite guard. RTX 3090 / 3090 Ti, with train_driver recorded.
* **Phase 1.** Train to 2k. If 123-3's flag fires, **re-draw the head**: ln, W_down and W_up get a fresh init from a dedicated
  generator (seed 12900 + the run's seed) and fresh Adam moments.
  - **GPT-2's body weights and its Adam state are kept.** Literature (b): "keep" is a fair single test. After a no-read regime, the body's
    second moments for code-carrying rows are small, which makes their steps larger, and stale first moments decay in about 10 steps.
    Shrink-and-perturb on the body is noted, not run.
* **Phase 2.**
  - The schedule clock restarts: LR warmup and cosine, the soft phase (p_hard 0 to 2k), and L_max 2 to 2k.
  - The 123-3 flag is applied again at clock 2k. A re-plateaued seed stops there.
  - Otherwise it trains 24k phase-2 steps to the usual eval.
* **CLOCK-ONLY control (Literature (c), Audit).** Restarting the clock is an LR warm restart, which can break a plateau on
  its own (SGDR-like).
  - Every PANEL seed saves its 2k checkpoint: model, both optimizers' state, and RNG.
  - From that checkpoint a second continuation restarts the schedule clock **without** re-drawing the head (dir
    `s129_s5_s{seed}_clock`).
  - Same flag at clock 2k, then 24k if it escapes.
* **Seeds,** in a fixed order: 2, 1, 3, 4, 101–104, 201–204, and so on.
  - A seed whose phase-1 flag does not fire stops at 2k and is reported.
  - **PANEL** = the first 4 seeds whose phase-1 flag fires.
* **Logging:** zb_std_in on a fixed probe batch every 50 steps over 0–2k in both phases (descriptive), plus the §127 probe at 0 /
  2k of each phase.

**Rules (fixed now), checked in this order.** ESC = the phase-2 flag does not fire. PASS = ESC and long accuracy ≥ 0.99 (S5
cumulative at 512).
* **LR:** CLOCK-ONLY escapes in at least 3/4. The warm restart alone breaks the plateau. REDRAW is reported beside it.
* **HEAD:** REDRAW passes in at least 3/4, **and** REDRAW escapes in at least 2 more seeds than CLOCK-ONLY (Audit).
* **BODY:** REDRAW's phase-2 flag fires in all 4.
* Otherwise, descriptive. Fresh seeds escape at about 0.5, so 1–2/4 REDRAW passes are consistent with a head-init lottery at
  the fresh-seed rate.

**Readings.**
* **LR.** Restarting the LR schedule from a plateaued 2k state is enough to enter, so the plateau is an optimisation-schedule
  basin. A clock-restart rule (no re-draw) is then a candidate procedure. Whether the escaped CLOCK-ONLY seeds also learn is
  reported (escape ≠ learning, 127-7).
* **HEAD.** The basin is held in the head's state. A re-drawn head enters on a body trained 2k steps without reading it,
  where a warm restart alone does not. The head re-draw is a restart that keeps the body; its success rate is compared with
  fresh seeds' 1 − p.
* **BODY (Audit's wording, fixed now).** "By 2k the body ignores a fresh head."
  - This does **not** show the basin was in the body at 650: the body had 1.3k more steps to adapt after the collapse.
  - An earlier-trigger arm would be a separate registration, not a reinterpretation of §129.
* **Otherwise.** Not distinguishable from a head-init lottery. Reported as such.

**Placement.**
* About 8 phase-1 runs of 2k steps (to find 4 flagged seeds at p ≈ 0.5).
* Then 4 REDRAW plus 4 CLOCK-ONLY continuations of 2k steps, and at most 8 continuations to 24k.
* About $4–8 on one RTX 3090 host (Audit-approved).
* Implementation: the experiments session (fsq120 `--redraw_head`, `--save_at_flag`, `--resume_2k <ckpt> --clock_only`; logs carry
  "phase": 2 and "clock" after the trigger).
* Scorer: `theory/score_s129.py`, committed before any run.
* Dirs: `boxes/runs_fsq/s129_s5_s{seed}` (REDRAW) and `s129_s5_s{seed}_clock`.

**Note 127-8 (2026-10-06 04:19 EDT; Stage 2, scored by `theory/score_s127.py`; Audit reproduced). STD2 and LR10 are decided: ENTRY-FIX = false,
ADOPT = false. SKIP is the only adoptable arm.**
* **Seed 4 plateaus in both arms** (RTX 3090, driver 595.71; both delivered):
  - **LR10 s4:** 8.49 / 5.015 / 4.789 at 100 / 1k / 2k; LR ratio 10 at every check.
  - **STD2 s4:** 8.40 / 4.982 / 4.948; row norm 0.53–0.94× wte.
* With seed 2 at long accuracy 0 (127-7), at most 2/4 seeds can be exact in each arm. **ENTRY-FIX and ADOPT are false**
  whatever s1 / s3 do. The scorer now decides this as soon as two seeds can no longer be exact.
* **Remaining cells (cost):**
  - STD2 / LR10 S5 s1 / s3 are already running and **continue**. They are descriptive: does "escape without learning"
    (127-7) hold beyond seed 2?
  - The STD2 / LR10 flip-flop cells (b39, not started) are **not run**. They could only feed a guard on a decided arm.
* **SKIP:** ADOPT still needs its s2 rerun not to plateau (b35, guard only), s1 / s3 / s4 exact (b30 / b31) and the FF guard.
  - Given s2's drift-back and every escaped seed so far at 0, it is unlikely. That is stated, not used.

**Note 127-9 (2026-10-06 04:23 EDT; the VAR s2 rerun (b35, RTX 3090, driver 565.77), read by Theory from the raw files; Audit reproduced).
VAR is delivered and escapes, but does not learn. Stopping the variance collapse fills the code with something other than the
state.**
* **Delivered** (in-forward group-mean residual, w = 1; 127-3):
  - resid_std 0.015 → **1.10** (500) → **1.16** (2k); check (ii) passes.
  - var_grad_down 0.185 / 5.23 / 0.55 at steps 1 / 500 / 2000, finite in fp32 and bf16. 0 non-finite steps.
* **ESCAPE:** loss 8.94 / 4.85 / 3.92 at 100 / 1k / 2k (ratio 0.81).
* **Then it fails:** final loss 3.22 (the ~3.2 basin, 123-7, not the ln 120 null). Long accuracy 0; 2,048 at 0; 2,444 codes, 126
  per state; donor 0.04 at every position, recipient ≤ 0.005.
* **Probe at 2k:** R²_state −0.05, R²_item −0.05. The decoder reads the carrier (abl_dloss +0.48).
  - Not a driver artefact: plain seed 2 plateaus on 550–565 (127-6), and this host runs 565.77.
* **Reading (descriptive):** preventing 127-7's collapse of the code's variance is not sufficient. The variance is filled, and
  read, with content that is not the state.
  - claude-13 / Audit: in seed 1's replay, content (R²_start 0.68) came before spread. VAR forces spread first.
* **VAR Stage 2, as registered, sequenced for cost.** Seed 2 is not exact, so ENTRY-FIX needs s1, s3 and s4 all exact.
  - They run **one at a time, behind §129**, in the order s4, s3, s1. The first that plateaus or ends below 0.99 decides
    ENTRY-FIX = false, and the rest are not run.
  - The flip-flop cells run only if all three are exact.
  - The rule is unchanged; only the order and stopping are cost choices.
* **The record so far:** every arm that escaped seed 2's flag ended at long accuracy 0: SKIP (NaN), STD2, LR10, VAR, and RAW-H
  s1 / s3 / s4.

**Note 129-1 (2026-10-06 04:23 EDT, before any §129 run; claude-13 / Audit; descriptive, no rule change).** In both continuations (REDRAW and
CLOCK-ONLY), add the R² probe at clock 700 and 1,200 as well as 0 / 2k. Report R²_item, R²_state (next state) and R²_start (the
chain's start state).
* Seed 1's replay had state content before spread and before the decoder's pull: R²_start 0.68 by 800, the pull at 1,050.
* A REDRAW pass whose head shows no state content before its pull would not follow that order, and is reported as such.

**Note 129-2 (2026-10-06 04:40 EDT, before any phase-2 result; disclosure, no rule change). The first seed-2 pair crashed at load and was voided.
Seed 2 reruns on the same host.**
* **What happened** (b40, RTX 3090, driver 595.71):
  - The host gradflow passed under bf16. Phase 1 of seed 2 flagged at 2k, at parity: 8.26 / 4.98 / 4.95.
  - Both continuations crashed when restoring the checkpoint. The saved CPU RNG state was moved to the GPU by `map_location`,
    and `set_rng_state` rejected it.
  - No phase-2 step ran, so **no phase-2 outcome was seen**. The driver then deleted that checkpoint.
* **Kept as void** (aborted unread): `s129_s5_s2.void_rngbug` (phase-1 log, flag fired), `s129_s5_s2_clock.void_rngbug`,
  and `s129_s5_s1.void_killed` (phase 1 halted at step 1,300 by the restart; no flag decision).
* **Fixed** (5c16f261 / 621da899: RNG states moved to the CPU; an end-to-end GPU resume smoke passed). Restarted with the fixed
  code, same host and order: seed 2's phase 1 reruns under the plain name, then its pair, then seeds 1, 3, 4, ….
  - A rerun of the same seed on the same host can differ at the bit level. The registered seed order and PANEL rule are
    unchanged.
  - The voided phase 1 is reported in the result as seed 2's first attempt (flagged; the continuations crashed unread).
* Checked in the code: the REDRAW continuation appends to the phase-1 log (mode "a"). CLOCK-ONLY writes its own dir.
  129-1's probes (clock 700 / 1,200, with r2_start) are in.
* **For the result** (Audit's pre-launch review): "keep the body" includes the body's small Adam second moments
  for the slot-reading rows, so its first phase-2 steps there are large (Literature (b)). That is part of what the arm tests.

**Note 116-8 (2026-10-06 04:43 EDT; OPDEF s31 scored by `theory/score_s116.py`, Audit reproduced; before s32 lands). FOOTHOLD-OPDEF is decided
false. s32 is descriptive, and the GPU per seed is stated.**
* **OPDEF s31** (local RTX 3070): never acquired. k = 1 never-labelled 0.0083, cumulative 512 at 0, donor at chance.
* **FOOTHOLD-OPDEF needs cumulative ≥ 0.99 at 512 *and* k = 1 ≥ 0.99, in ≥ 2/5 seeds.**
  - s33 acquired the whole table (k = 1 1.0, certified) but ended at **0.987** at 512, under the bar (116-7). So no
    scored seed passes: s31, s33, s34 and s35 do not.
  - Only s32 remains, so the count can reach at most 1/5. **FOOTHOLD-OPDEF = false**, whatever s32 does.
  - Correction to the 04:55 tally, which counted s33 as a FOOTHOLD seed.
* STRICT-C4 (NONE) was already decided false (116-6, 0/4 at chance).
* **s32 (local RTX 4060) is descriptive:** an acquisition count, with the card stated per seed.
  - Acquisition so far: s33 (4060) yes; s31 (3070), s34 (3070), s35 (2070) no.
  - If s32 (4060) acquires, it is reported as "2/2 on the 4060 vs 0/3 on other cards", a GPU-class confound (125-2), not a clean
    2/5.
  - **s32 landed (05:16):** RTX 4060, never acquired (cumulative 0, k = 1 at chance). OPDEF ends **1/5 acquired** (s33). Since
    s32 shares s33's card, the card confound does not arise.
* **Wording, per Literature (ba531055 (d)):** operator definitions **suffice** as a C4 foothold in 1/5 seeds so far
  (s33: the whole table learned and certified, but not FOOTHOLD by 512 accuracy). "Minimum foothold" (116-7) is replaced by
  "sufficient in k/n seeds", since minimality is untested.

**Note 127-10 (2026-10-06 05:20 EDT, before any SKIP Stage 2 S5 cell is read; Audit / claude-13; a clarification of 127-1). A solved task cannot
fail the gradient delivery check.**
* **The case:** SKIP's FF s1 has long accuracy 1.0, is SERIAL, and has R²_state 0.95. Its skip_grad_q is 0.0023 at step 1, then 9e-10 /
  3e-9 at 500 / 2000, below 127-1's 1e-8.
  - The cause: FF's training loss is about 0 by then, so no gradient reaches any parameter. That is not an intervention failing to
    act.
* **Rule (applies to every SKIP cell, S5 and FF):**
  - skip_grad_q > 1e-8 is required at step 1.
  - At steps 500 and 2,000 it is required only while the training loss at that step (nearest logged) is above 1e-3.
  - At a step where the loss is ≤ 1e-3, the check is recorded as "no gradient (loss ≈ 0)", not as a failure.
  - The FF guard stays a rule on accuracy, as registered. FF delivery is reported descriptively.
* **No effect on S5 cells:** S5 losses at 500 and 2,000 are about 5 in every run so far. The rule is fixed before any SKIP S5
  Stage 2 cell is read. `score_s127.py` implements it (tested: FF solved early → delivered; S5 unsolved with a tiny gradient →
  not delivered).

## 130. An early, label-free plateau marker: fitted on one GPU class, tested on another, from $0 replays of §120's first 2k steps (pre-registered 2026-10-06 05:22 EDT, after Literature's check 722b9f4c and before any §130 replay; Audit's steward idea for the idle local cards; C6: the cost of seed reliability)

**Why.**
* Restart-on-plateau (§126) is the working entry procedure, and §128 uses it. Each plateaued attempt costs 2k steps before the
  123-3 flag can fire.
* The two 3090 replays (127-5, 127-7) show an earlier signal. In the plateau seed, the code's spread across inputs (zb_std_in)
  collapses to about 0.02–0.03 by 700, after a failed first pull at 650. In the escaping seed it stays at 0.05–0.14 and then
  grows when the decoder pulls.
* A marker that fires by about 1k would halve each restart's cost, but it is n = 1 per seed. §130 fits a marker on one card
  class and tests it on another, before any rule uses it.
* **Prior** (722b9f4c): `docs/LIT_S130_MARKER_2026-10-06.md`: seed triage from loss curves (Dodge 2002.06305; Hyperband 1603.06560 /
  successive halving; Domhan's learning-curve extrapolation). **zb_std_in is the "active units" statistic of Burda et al.
  (IWAE 1509.00519)**; VICReg uses per-dimension std as its collapse signal. New as scoped: an early representation-variance
  marker for a no-read basin, validated prospectively across GPU classes.

**Replays.**
* `scratch/replay_basin.py` (reviewed, 0baed523): fsq120 FINAL S5's first 2k steps, the same init and data RNG, bf16 autocast;
  fp32 probes every 50 steps. One card class per block. A gradflow smoke runs on each card first.
* **FIT:** local RTX 4060, seeds 3–10, extended with 35–42 if it has fewer than 3 flagged or 3 non-flagged seeds.
* **TEST-CROSS (primary):** local RTX 3070, seeds 11–18, then 27–34, then 43–58.
* **TEST-SAME (secondary):** local RTX 4060, seeds 19–26, then 59–82. It runs after FIT, so a cross-card failure can be read as
  "card" rather than "overfit" (Literature (b)).
* **Each TEST block runs until it has ≥ 8 flagged and ≥ 8 non-flagged seeds** (Literature (c)), or its seed list is used up.
* **Blinding:** every TEST replay, on either card, stays unread until the freeze note.
* Outputs: `theory/results/replay130/{card}_s{seed}.jsonl`.

**Definitions** (from each replay's own log).
* FLAG: 123-3's flag on the replay loss (nearest logged steps to 100 / 1k / 2k).
* PULL: the first probe with L1_dmean > 0.05.
* COLLAPSE: zb_std_in < 0.04 at two consecutive probes at or after 600.
* marker(θ, window): zb_std_in < θ at two consecutive probes inside the window.

**Rules (fixed now).**
* **FIT** (FIT seeds only): choose θ ∈ {0.03, 0.04, 0.05, 0.06, 0.08} and window ∈ {600–1000, 700–1000, 700–1200, 600–1500}
  that maximise min(precision, recall) against FLAG. Ties go to the earlier window end, then the smaller θ.
  - The neighbouring cells' scores are reported: a flat optimum is more trustworthy.
  - The choice is **frozen in a dated note (130-1) before any TEST replay is read.**
* **PASS (primary, TEST-CROSS):** the frozen marker's precision and recall against FLAG are both ≥ 0.9, with ≥ 8 flagged and ≥ 8
  non-flagged seeds. Otherwise the result is incomplete, with its counts.
  - TEST-SAME is scored the same way, as secondary.
  - Two-sided 95 % Clopper–Pearson intervals are reported (8/8 gives a lower bound of 0.63).
* **Descriptive:**
  - **H-127-7:** every flagged seed shows COLLAPSE, and every non-flagged seed reaches L1_dmean ≥ 0.1 by 2k. Counts per block.
  - The escape rate per card class.
  - The frozen marker on §129's 3090 phase-1 logs.

**Readings.**
* **PASS (cross-card).** A label-free marker at about 1k predicts the 2k plateau across GPU classes. An early restart rule built
  on it becomes a candidate for registration, halving each restart's cost. §130 itself changes no procedure.
* **TEST-SAME passes, CROSS fails.** The marker holds on its own card class but not across classes, consistent with the GPU
  sensitivity of the basin (125-2). A per-card threshold is not adopted from this.
* **Both fail.** The marker is noisy or overfit. Restart-on-plateau keeps the 2k flag.
* **H-127-7 fails in some seeds.** The collapse-after-failed-pull account (127-7) does not generalise. Counterexamples are
  reported.

**Placement.** About 40–60 replays of 2k steps (about 20–40 min each) on the two idle local cards, running roughly 8–16 h; $0.
* Implementation: the experiments session (a queue around replay_basin.py per card).
* Scorer: `theory/score_s130.py`, committed before any replay. Run `fit` first, then `test θ lo hi` after note 130-1.

**Note 130-0 (2026-10-06 05:30 EDT, before any TEST replay finishes; Audit; a clarification of §130's blinding). What "unread" means for TEST
seeds.**
* **Blinded until note 130-1:** for every TEST seed (TEST-CROSS on the 3070, TEST-SAME on the 4060), zb_std_in, the marker values,
  and anything derived from them: PULL and COLLAPSE steps, R² probes, L1_dmean, effective rank.
* **Not blinded:** each TEST seed's 2k loss flag (123-3 on the replay's own loss). §130's stopping rule needs it: each TEST block
  runs until it has ≥ 8 flagged and ≥ 8 non-flagged seeds.
  - The marker is chosen on FIT seeds only, deterministically (max min(P, R), fixed ties), and frozen before TEST is scored.
  - A stopping rule on label counts cannot inform θ or the window, or bias precision or recall.
* This narrows §130's "every TEST replay stays unread" to the quantities above. The rules are unchanged.
* **Addendum (2026-10-06 06:13 EDT, before the freeze; a disclosed deviation). FIT is extended with 4060 seeds 35–42, one fixed block,
  with no further extension.**
  - §130's trigger (fewer than 3 flagged or 3 non-flagged FIT seeds) is not met.
  - But FIT s3–s9 show the 2k flag firing on late learners (s5 and s7 pull at 1,050 / 850, loss still falling). s9 collapses
    from 600 to 1,000, recovers at 1,100–1,300, and escapes. At n = 8 the grid's choice hinges on single seeds.
  - The extension touches only the fitting set. TEST stays blind, and the freeze (130-1) uses all 16 FIT seeds.
  - It also fixes before the freeze: the secondary BASIN label (FLAG and no PULL by 2k), with its own FIT cell scored on TEST
    descriptively, and per-seed collapse and recovery steps.
  - Descriptive check runs follow TEST-SAME: full fsq120 FINAL S5 24k for 4060 s5, s6 and s7.
    - Pre-stated: if s5 and s7 end exact and s6 does not, the 2k flag has false positives on this card class.

**Note 127-11 (2026-10-06 05:37 EDT; descriptive, SKIP Stage 2 s4 read by Theory from the raw files; Audit reproduced; before s1 / s3, ρ_ff
and the control s4 are read). The first entry-fix cell that escapes and learns, and its trained chain runs on the slot carrier
alone.**
* **SKIP s4** (b30 group, RTX 3090; driver to be recorded from nvidia-smi, since the host predates train_driver):
  - Loss 8.54 / 4.72 / 3.32 at 100 / 1k / 2k (no flag), final 2e-5, 0 non-finite steps.
  - Delivered: skip_grad_q 0.0078 / 0.0030 / 0.0063.
  - **Long accuracy 1.0 at 16–512, 0.998 at 2,048.** Donor 1.0 at positions 2 / 4 / 6 with recipient = chance: SERIAL.
  - **Skip-off (127-4; skip maps zeroed at test): 1.0 at 16 / 64 / 512, still SERIAL.** The skip paths are not needed at test,
    so the skip acted as a training-time aid.
* **What this does and does not show (Audit):**
  - Whether s4 is a basin seed on the plain recipe is answered by the concurrent control s4 (`s127_s5_ctrl_s4`, same host
    group). LR10 and STD2 s4 plateaued on 595.71, but under other changes.
  - SKIP's ENTRY-FIX still needs s1 and s3 exact (s2 is not), ρ_ff ≤ 0.28 in the exact seeds (fp32 post-hoc pending), and
    the FF guard (s1 passed on accuracy, 127-10; s2 pending).
  - ADOPT is then read on cost per accepted seed against the control's.
* **The record of escaped seeds, updated:** 1 of 6 escaped arm-seeds learned (SKIP s4). The others (SKIP / STD2 / LR10 / VAR s2,
  and RAW-H) did not.

**Note 127-12 (2026-10-06 05:41 EDT, before SKIP s1 / s3 and the controls are read; Audit / claude-13; a scoring clarification). ENTRY-FIX's ρ
clause on rounded codes.**
* **The issue.** With FSQ rounding, a perturbed chain snaps back exactly within one or two hand-offs. 123-9's ρ_ff then has fewer
  than 2 usable hand-offs and is **undefined**, and the scorer would hold ENTRY-FIX at "incomplete (ρ_ff)" for ever. Read as about
  0, the clause would be met by construction.
  - SKIP s4 (fp32): ρ_k 0.183, then 0; median distance 0.74 → 0.14 → 0; exact recovery 0.154.
* **The clause now reads ρ_ff when it is defined.** When it is undefined because the ε 0.1 median distance reaches ≤ 1e-3 (exact
  recovery by rounding), it reads **the first-hand-off ratio ρ₁**. ρ₁ is the informative number (123-1), and it is not met by
  construction.
* Disclosure: SKIP s4's ρ₁ (0.183) was seen before this note. It passes under either reading, so the choice cannot have been
  made for it.
  - The exact §123 seeds' band, for comparison, is 0.146–0.180.
* Exact-recovery share and ρ₁ are reported beside the clause for every exact seed.

**§101 result (2026-10-06 05:47 EDT; scored by `theory/score_s101.py`; the decisive b0.01 s3 is in). The smallest passing β_min is 0.01: a weak
attractor suffices.**

| β_min | Cumulative 512 (s1 / s2 / s3) | 2,048 | Passing seeds | Arm | GPUs |
|---|---|---|---|---|---|
| 1 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 3/3 | PASS | A5000 |
| 0.1 | 0.9995 / 0.0005 / 1.0 | 1.0 / 0 / 1.0 | 2/3 | PASS | A5000 |
| **0.01** | 0 / 0.9995 / 1.0 | 0 / 0.996 / 1.0 | **2/3** | **PASS** | RTX 3090 |
| 0 | 0 / 0 / 0.9965 | 0 / 0 / 0.988 | 1/3 | not met | A5000 (s1, s3), 3090 (s2) |

* Zero-removal after the first ablated slot is ≤ 0.007 in every cell, so the latents are used wherever a seed passes.
* **Registered reading, "only 0.01 or 0.1":** a weak attractor suffices, and "soft snap" is the honest design. β_min = 0 (no
  discrete correction) is exact in only 1/3 seeds.
* **Descriptive.** Every failing seed shows a carrier-norm drop after the first chunk, held for the rest of the chain:
  b0.01 s1 4.96 → 3.31; b0 s1 4.88 → 3.09; b0 s2 4.58 → 4.19; b0.1 s2 to 3.82.
  - Passing seeds hold about 3.96 throughout (u_code_norm_ratio 0.94–1.0, against 0.77–0.88 in the failing seeds that
    collapsed).
* **Caveat:** the arms trained on different GPU classes (A5000 against RTX 3090). Seed counts are per arm, so there is no
  within-seed comparison across β, but the GPU sensitivity of the basins (125-2) is stated.

**Note 124-6 (2026-10-06 05:54 EDT; descriptive, COPY s27 (RTX 3090) read from eval.json; Audit reproduced; COPY s26 and BASE s26–27 pending).
The first self-training seed to meet C4 (qualified), C1 and C2 together.**
* **COPY s27:**
  - k = 1 never-labelled 1.0, certified on all 14,400 transitions.
  - Cumulative 1.0 at 128–512, and 1.0 at 1,024 / 2,048.
  - **Donor 1.0 at slots 1 / 2 / 3, recipients 0.0105 / 0.0095 / 0.0075:** single-slot SERIAL at every slot, with no slot-1
    bypass. Zero / mean removal 0.0, copy accuracy 1.0. Acquired at L5.
  - So final-answer self-training (C4 qualified by the f = 0.1 foothold) gives serial latent use (C1) and length
    generalisation (C2), in one seed.
* **COPY's registered readout, acquisition round, is L5 in 3/3 seeds so far** (s27, s28, s29). EXPOSE's is L5–L8.
* **Carrier type still varies within the arm** (124-5): s27 SERIAL, s29 strongly slot-carried (0.88 / 0.75 / 0.52), s28 mixed.
  - No arm-level "COPY → C1" claim. If s26 is also slot-carried, "COPY gives slot carriage in k/4" is a reasonable descriptive
    line, under §124's registered readings.

**Note 124-7 (2026-10-06 06:14 EDT; descriptive, the COPY arm complete (s26 on an RTX 3090); Audit reproduced; BASE s26–27 pending). COPY acquires at
L5 in 4/4 seeds, and gives single-slot SERIAL carriage in 2/4. EXPOSE gives it in 0/4.**
* **COPY s26:** cumulative 0.9995 at 512; k = 1 never-labelled 1.0, certified. Donor 0.979 / 1.0 / 1.0 at slots 1 / 2 / 3, recipients
  ≤ 0.0105: SERIAL.
* **Registered readout, acquisition round:** COPY L5 in **4/4** (s26–s29); EXPOSE L5–L8 (L6 / L5 / L6 / L8); BASE 1/2 so far
  (L6).
* **Carrier type (descriptive, single-slot SERIAL = donor ≥ 0.90 at every slot, recipient − chance < 0.10):**
  - COPY 2/4 (s26, s27); s29 is slot-carried but under 0.90 at slot 3 (0.88 / 0.75 / 0.52); s28 mixed.
  - EXPOSE 0/4 (s28 slot-carried at 0.62–0.75; three seeds with a slot-1 bypass).
  - BASE 0/1 acquirers so far (s28 bypass).
* **Wording, within §124's registered readings:** "COPY makes acquisition early and uniform (L5 in 4/4), and gives serial slot
  carriage in 2/4 seeds against 0/4 for EXPOSE."
  - The second part is descriptive: §124 registered acquisition count and round, not C1. MECH-112 stays false (124-4).
  - NOT-112-4 versus the descriptive reading is decided when BASE s26–27 land.

**Note 127-13 (2026-10-06 06:34 EDT; the concurrent control s4 and the SKIP FF guard, read from the raw files; Audit reproduced; a descriptive
check fixed before it runs). SKIP s4 escaped and became exact where the plain recipe flagged at 2k. Whether s4 is a basin seed is
open.**
* **ctrl s4** (plain §120 FINAL plus the guard; b31, RTX 3090, driver 595.71.05; the same host group as SKIP s4): flagged at 2k
  and stopped. Loss 8.55 / 4.922 / 4.539.
* **The flag is borderline** (Audit):
  - the 2k/1k ratio is 0.922 against the 0.9 cut;
  - 4.54 is already under the ln 120 null (4.79);
  - the carried embedding is not collapsed at 2k (per-coordinate std 0.53).
  - That is the profile of §130's 4060 late learners (FIT s5 / s7: flagged, but pulling at 850–1,050).
* **Wording, fixed now:** "SKIP s4 escaped and became exact where the plain recipe flagged at 2k". It is **not** "SKIP rescued a
  basin seed" unless the check below shows s4 to be one.
* **Descriptive check** (`s127_s5_ctrl_s4_full`): plain S5 s4 on an RTX 3090 (driver 595.71), with no plateau stop, to 24k;
  about $0.4. Reading, fixed now:
  - **It ends at the null** (long accuracy 0, loss near 4.8): s4 is a basin seed, and "SKIP rescued a basin seed" is earned
    (n = 1).
  - **It ends exact:** the 2k flag gave a false positive on a 3090 too. That is reported against restart-on-plateau's cost
    and 128-1's check rate.
  - **Anything else:** reported as observed.
* **SKIP's FF guard is met: FF s1 / s2 long accuracy 1.0**, 0 non-finite steps (delivery read per 127-10).
* SKIP's ENTRY-FIX still needs s1 and s3 exact (b30). The ρ clause reads ρ₁ (127-12). ADOPT is read on cost against the
  control.
* **Outcome (2026-10-06 09:28 EDT; Audit reproduced): the unstopped plain s4 run (`s127_s5_ctrl_s4_full`, RTX 3090, driver 595.71) ends at the
  null:** loss 8.551 / 4.922 / 4.539 at 100 / 1k / 2k (identical to the stopped ctrl s4), 4.59 at 4k, 4.69 at 24k; long accuracy 0,
  2,048 at 0, donor at chance. By the reading fixed above, **seed 4 is a basin seed on this card, and "SKIP rescued a basin seed" is
  earned (n = 1).** The borderline 2k flag (0.922) was a true positive here.

**§124 result (2026-10-06 06:43 EDT; scored by `theory/score_s124.py`, all 12 runs; each seed's three arms on one card type (s26–27
RTX 3090, s28 A10, s29 RTX 4090); Audit reproduced). Descriptive: both COPY and plain exposure lift acquisition. COPY is uniform
and earliest, and is the only arm with serial slot carriage.**

| Arm | Acquired (never-labelled k = 1 ≥ 0.99) | Round (s26 / s27 / s28 / s29) | FOOTHOLD (descriptive) | Single-slot SERIAL (descriptive) |
|---|---|---|---|---|
| BASE | 2/4 | L6 / – / L6 / – | 2/4 | 0/2 acquirers (both slot-1 bypass) |
| COPY | **4/4** | **L5 / L5 / L5 / L5** | 3/4 | **2/4** (s26, s27) |
| TEXT-EXPOSE | 4/4 | L6 / L5 / L6 / L8 | 2/4 | 0/4 |

* **Rules:** MECH-112 is false (124-4: EXPOSE acquired in 4/4, not ≤ 2/4). NOT-112-4 is false (COPY acquires in more seeds than
  BASE, 4 against 2, and earlier). So the result is **descriptive**, as registered.
* **Reading:**
  - **By count**, the copy objective and plain LM exposure to random start texts both lift acquisition above BASE. So
    "reading a given state" is not separable from "seeing non-identity start text" at this n.
  - **By timing**, COPY is the only arm with a uniform signature: L5 in every seed, the sharpest jump.
  - **Descriptive, beyond the registered readouts:** COPY gives single-slot SERIAL carriage in 2/4 seeds (s27 at every slot,
    with no bypass). EXPOSE gives it in 0/4, and BASE's two acquirers both bypass slot 1.
* **Claim wording (fixed in §124):** "C4 qualified by f = 0.1 short-problem coverage (identity starts) plus a self-supervised
  copy objective (no intermediate or transition labels)". COPY s26 / s27 meet it together with C1 (serial) and C2 (exact to
  2,048): the first self-training seeds to do so (124-6, 124-7).

**Note 119-4 (2026-10-06 06:44 EDT; SIM-CoT's s1 LR sweep read on-host by the experiments session; before SIM-CoT s2–3). SIM-CoT did not learn at
any LR. The tied criterion is broken by the published LR, and seeds 2–3 run as registered.**
* **SIM-CoT s1** (A5000, driver 595.71.05; initialised from Coconut s1 at 3e-5): intact 0.006 / 0.007 / 0.007; train-split
  final accuracy **0.0104 at all three LRs**, an exact tie (constant output); cumulative 0.
  - Step decoder φ's CE is about 4.95 per row from stage 1 on: the marginal (ln 120 plus EOS). So the thoughts carry no step
    state.
  - Guard (a): "did not learn", at all three LRs.
  - Stage-0 relearning (119-3's degraded-start check) is read from the stage-0 loss in the logs and reported.
* **Tie-break, fixed now:** the swept value nearest the published LR, **1e-4** (the middle of {3e-5, 1e-4, 3e-4}).
  - **SIM-CoT s2 / s3 run at 1e-4** from Coconut s2 / s3, as registered (3 seeds per method), as for Coconut (119-2).
  - Expected: "did not learn" in 3/3, so SIM-CoT is **inconclusive** for CONTROLLED-C1. 119-3's all-inconclusive wording then
    applies to Coconut and SIM-CoT.
* **119-3's degraded-start check fires (06:45):** SIM-CoT s1's loss at the end of stage 0 (step 3,000) is 0.195 / 0.796 / 0.794
  at lr 3e-5 / 1e-4 / 3e-4, all above 0.1. Coconut s1 reached 1e-4 at 3e-5. So s1 reads **"did not learn (degraded start)"**.
  - As pre-stated, a SIM-CoT started from the base model is a candidate for a new registration, not a §119 arm.
  - s2 / s3 run as registered (d6, RTX 3090, driver 570.211). Their stage-0 loss is read with the same check.
* **A collapsed predictor, not noisy chance** (Audit): at lr 1e-4 and 3e-4 SIM-CoT s1 outputs **one constant final answer** for all
  2,000 test sequences (top share 1.0). At 3e-5 it gives 2 distinct answers (top share 0.986). That is why the 1e-4 and 3e-4 cells match
  to the digit. It is consistent with the degraded stage-0 start.
* **Logging bug, disclosed.** In these logs, stage ≥ 1 lines have `"step"` overwritten by the decoder CE (a key collision); line i
  is step 100·i. Training and eval are unaffected. Fixed for later runs (`step_ce`).

**Note 130-1 (2026-10-06 07:05 EDT; THE FREEZE, from all 16 FIT replays (4060 s3–10, s35–42); `python theory/score_s130.py fit`, then
`theory/results/score_s130_frozen.json`; before any TEST replay is read). Four cells are frozen. The registered marker fails
its bar even in-sample.**
* **Frozen cells (4; the multiplicity is stated):**

| Cell | Role | θ | Window | FIT precision | FIT recall | Neighbours' min(P, R) |
|---|---|---|---|---|---|---|
| dip, FLAG | **primary (registered)** | 0.08 | 700–1200 | 0.60 | 1.0 | 0.55 / 0.55 / 0.55 / 0.55 |
| dip, BASIN | secondary | 0.03 | 700–1000 | 0.33 | 0.5 | 0.25–0.33 |
| persist, FLAG | secondary | 0.08 | 700–1000 | 0.67 | 0.33 | 0.17 |
| persist, BASIN | secondary | 0.03 | 700–1000 | 0.50 | 0.5 | 0–0.5 |

* **Tie rule as used:** max min(P, R); ties go to the earlier window end, then the smaller θ, then the earlier window start.
* **The primary optimum sits at the θ grid's upper edge** (0.08, the largest value tried), on a near-flat row (0.55–0.56 next door).
  A larger θ was not tried, and is not tried now: the grid was fixed in §130.
* **What FIT shows (descriptive, 4060):**
  - FLAG fires in 6/16 (s5, s6, s7, s37, s41, s42), so the escape rate is 10/16. BASIN (flag and no pull by 2k) holds in
    only 2/16 (s6, s37).
  - **The code-variance collapse at about 650 is common and mostly transient.** It occurs in 7/16 seeds, and in all 7 the
    spread recovers (at 700–1,700).
    - Two of the 7 never pull (s6, s37): the basin. One flags after a late pull (s42, pull at 1,400). Four escape (s9, s10, s36, s38).
  - **H-127-7 fails both ways:** 3/6 flagged seeds collapse; every non-flagged seed pulls (10/10, dmean_max ≥ 1.97).
* **Pre-stated:** the primary cell is at min(P, R) = 0.60 in-sample, so a TEST PASS (≥ 0.9 / ≥ 0.9) is very unlikely. TEST is
  scored as registered (`python theory/score_s130.py test` reads the frozen file).
  - The expected reading is "no PASS: on these cards the early code-variance marker does not predict the 2k plateau".
  - Restart-on-plateau keeps the 2k flag.
* **The 2k flag's own reliability** is the next question: FIT's s5 / s7 / s41 / s42 flag after a pull. The 24k check runs of s5,
  s6 and s7 (130-0) answer it on the 4060, and §127's ctrl-s4 full run answers it on a 3090.

## 131. A second restart point at 4k: flagging "escape without learning" from the loss alone (pre-registered 2026-10-06 07:10 EDT, after Literature's check 2d097e1c; Audit / claude-13's observation; $0, log-only; C6: the cost of seed reliability)

**Why.**
* §127's main failure mode is "escape without learning". Every entry fix that cleared the 2k flag on seed 2 (SKIP, STD2, LR10,
  VAR, plus RAW-H) still ended with long accuracy 0, and each such run costs a full 24k steps.
* claude-13's observation: at 2.1k (L_max 2 → 4) every S5 loss jumps 0.4–0.8. Learners descend again by 4k (ratio
  loss(4k) / loss(2.6k) of 0.67–0.85). Escaped-but-failed runs stay flat or rise.
* A label-free check at 4k would cut such a run's cost from 24k to about 4k steps.
* **Prior** (2d097e1c): `docs/LIT_S131_FLAG4_2026-10-06.md`: learning progress after a task change, used to schedule tasks rather than to stop runs
  (Graves 1704.03003; TSCL 1707.00183); run triage from loss curves (Dodge 2002.06305, Hyperband, Domhan). Stage-conditioned
  triage is new as scoped.

**Rule (fixed now).**
* **FLAG4 = loss(4k) ≥ 0.9 · loss(2.6k) AND loss(4k) ≥ 0.25 · loss(100),** on the nearest logged steps. The second clause is
  123-3's floor, so a solved task near zero loss cannot flag.
* **Target:** the run does not learn (final long accuracy < 0.99).
* **Single points, not windows.** Literature (b) suggested windowed means (3.8–4k against 2.4–2.6k). In-sample, windows lowered recall
  from 9/13 to 7/13, because slowly rising runs average under the threshold. On the 13 held-out runs seen at registration the two
  forms agree (0 flagged). The single-point rule is registered, and the windowed form is reported beside it.
* **Scope:** the held-out set is pre-filtered to runs that passed the 2k flag, so precision holds among those runs.
  - The ~3.2 basin (late plateaus at L_max 8–16) is out of scope: a later rule.
* **In-sample** (where the rule was derived; descriptive): the 30 fsq120 S5 runs on disk with ≥ 4k logged steps when it was
  derived, listed in `theory/score_s131.py`.
  - Learners 8, flagged 0. Escaped-and-failed 13, flagged 9. **Corrected in 131-1:** 12 distinct trajectories, 8 flagged (0.667).
  - The 4 misses are a named class: ANNEAL s2 ×2 (0.87 / 0.75), RAW-H s2 (0.85; the 0.846 partial learner), and the VAR s2 rerun
    (0.88; it ends in the ~3.2 basin).
* **Held-out:** every other fsq120 run (§127, §128 any family, s130chk) with ≥ 4k logged steps, an eval.json and no 2k stop.
  - §129's continuations (schedule clock restarted) are reported separately, not scored.
  - **Disclosure:** the scorer was run once before this registration, after the rule was fixed. It saw 13 held-out runs then on
    disk, all exact §128 runs (MOD, CYC, RDFA), and none was flagged.
  - Those 13 are reported as "seen at registration". Runs landing afterwards are "prospective". The PASS is computed on both
    together and on the prospective subset alone.

**PASS (fixed now).**
* (i) No held-out run that ends exact is flagged, with ≥ 5 exact held-out runs.
* (ii) At least 2/3 of the held-out runs that escape the 2k flag and end below 0.99 are flagged, with ≥ 3 such runs.
* Otherwise the result is incomplete, with its counts.
* Precision and recall are reported with two-sided 95 % Clopper–Pearson intervals.
* **"Deployable"** as a restart rule needs ≥ 10 flagged held-out runs with 0 false positives (Literature (c); the 95 % lower
  bound is then about 0.74). A 5/5 PASS has a lower bound of about 0.55.

**Readings.**
* **PASS.** A label-free 4k check catches most escape-without-learning runs and never kills a learner on held-out data. A
  two-point restart rule (2k and 4k) becomes a candidate procedure. Its adoption, with a cost comparison, is a separate
  registration.
* **(i) fails.** The 4k check kills learners, so it is not usable as a restart rule.
* **(ii) fails.** The 4k check misses too many failures. The ~3.2 basin and partial learners are reported as classes.

**Held-out runs added for ground truth (Audit; local RTX 3070, $0).** Full fsq120 FINAL S5, 24k, no plateau stop, for four
§130 TEST-CROSS seeds (`s130chk_s5_s{17,30,34,12}_3070`). Their 2k status is known from the replays:
* s17 and s30 were flagged after an early pull (s17: loss 3.79 → 3.66, far under the null);
* s34 is BASIN (persistent collapse);
* s12 was a marker false positive and escaped.
* **Pre-stated:** if s17 and s30 end exact, the 2k flag gives false positives on the 3070 too (the 4060 counterpart is
  s130chk_s5_s{5,6,7}). This is reported against restart-on-plateau's cost and 128-1's check rate.
* Each run reaching 4k also counts as held-out for FLAG4.

**Placement.** $0, log-only, over the runs already scheduled, plus the four 3070 check runs (about 3–4 h each, sequential).
Scorer: `theory/score_s131.py`, committed with this registration.

**§130 result (2026-10-06 07:12 EDT; scored by `theory/score_s130.py test` from the frozen cells (130-1); TEST-CROSS complete, 17 replays on the
RTX 3070 (9 flagged, 8 not); Audit reproduced two ways). No PASS: the early code-variance marker does not predict the 2k plateau
across cards. Restart-on-plateau keeps the 2k flag.**

| Frozen cell | Role | TEST-CROSS precision (95 % CI) | Recall (95 % CI) | Verdict |
|---|---|---|---|---|
| dip, FLAG (θ 0.08, 700–1200) | **primary** | 0.70 (0.35–0.93) | 0.78 (0.40–0.97) | **PASS = false** |
| dip, BASIN (θ 0.03, 700–1000) | secondary | 0.50 | 0.67 | descriptive (BASIN n = 3) |
| persist, FLAG (θ 0.08, 700–1000) | secondary | 1.0 | 0.33 | descriptive |
| persist, BASIN (θ 0.03, 700–1000) | secondary | 1.0 | 0.33 | descriptive (BASIN n = 3) |

* As pre-stated in 130-1: the primary cell reached only 0.60 in-sample.
* **Both FLAG misses (s17, s30) pulled early but were still flagged.** s17's loss goes 3.79 → 3.66, far under the null. The ratio
  flag has no family-null floor at the 1k end, which is Literature's §131 point seen here at 2k.
* **Descriptive:**
  - H-127-7 fails on both cards: flagged seeds collapse in 3/6 (4060) and 6/9 (3070), and every non-flagged seed pulls.
  - Escape rate by the 2k flag: 10/16 on the 4060 (FIT), 8/17 on the 3070.
  - The collapse at about 650 is common and mostly transient on both cards.
* **TEST-SAME (secondary, 4060)** has 1 seed so far. It is added when complete, and cannot change the primary verdict.
* **TEST-SAME complete (2026-10-06 10:42 EDT; 24 seeds on the RTX 4060, 8 flagged).** The primary cell gets precision 0.47 (CI 0.21–0.73) and recall 0.88
  (0.47–1.0): **PASS = false here too.**
  - persist | FLAG: 0.75 / 0.38. BASIN occurs in 0/24 seeds (no flagged seed failed to pull).
  - Both the cross-card and same-card tests fail, so the registered reading is **"Both fail: the marker is noisy or overfit"**, not
    card-specific.
* **What follows:** the 2k flag's own reliability is in question. It fires after early pulls, on loss well under the null.
* **Qualification (2026-10-06 09:38 EDT; Audit).** "The 2k flag fires on late learners" (130-0 addendum, 130-1) rests on the replays alone: a pull and
  falling loss by 2k, with no learning outcome. **This wording is held** until the seven check runs reach 24k (4060 s5, s6, s7;
  3070 s17, s30, s34, s12).
  - Early sign the other way: 4060 s5's full run is heading to the null (4.72 at 2k, 4.93 at 2.6k, 5.01 at 4k; FLAG4 1.017).
  - 3070 s17 is undecided (2k flag true, FLAG4 0.744, 3.81 at 7.9k).
  - §127's ctrl s4 (borderline, 0.922) was a true positive (127-13 outcome).
  - The registered §130 verdict (no PASS for the marker) is unaffected.
  - Ground truth: §131's 24k check runs on the 4060 (s5, s6, s7) and 3070 (s17, s30, s34, s12), and §127's ctrl-s4 full run on a
    3090.
  - §131 tests the 4k flag held-out.

**Note 131-1 (2026-10-06 07:16 EDT, before any held-out run passes 4k; Audit / claude-13's pre-launch review; no rule change). Four scoring
clarifications.**
* **(a) Criterion (i) outside S5.** FF / CYC / MOD / RDFA sit near zero loss by 2.6k, so the floor stops FLAG4 from firing on them,
  and "0 false positives" there is nearly vacuous.
  - The scorer also reports (i) for S5 runs alone. The reading is worded "0 false positives among N S5 learners".
  - The 13 held-out runs seen at registration are all non-S5.
* **(b) In-sample double count.** SKIP s2 and its rerun follow the same trajectory to 9k (ratio 1.057 for both), so they count
  once. The in-sample tally is **8/12 escaped-and-failed flagged** (0.667, exactly the bar) and 0/8 learners, not 9/13.
* **(c)** Held-out excludes `*.void*` dirs.
* **(d) Runs flagged at 2k that ran on** (ctrl s4 full, the s130chk runs): if one ends exact, it counts in (i) as registered,
  which is conservative. A FLAG4 hit there would be a false positive that a deployed 2k + 4k rule never reaches.
  - The scorer also reports (i) for learners **not** flagged at 2k, the population a two-point rule acts on.
  - A failure of (i) confined to 2k-flagged runs is read that way.

**§127 result (2026-10-06 07:45 EDT; scored by `theory/score_s127.py`; all four arms decided; RTX 3090 class, Stage 2 on driver 595.71; Audit
reproduced). No entry fix is adopted. Escaping the 2k flag is easy; learning afterwards is the failure. Restart-on-plateau
stays.**

| Arm (S5 seeds 1 / 2 / 3 / 4) | Long accuracy at 512 (or plateau at 2k) | Exact | ENTRY-FIX | ADOPT |
|---|---|---|---|---|
| VAR (item-residual variance) | – / 0 (rerun) / – / plateau | 0/2 run | false | false |
| SKIP (skip-to-decoder) | **0.999** / 0 / 0 / **1.0** | **2/4** | false | false |
| STD2 (standardisation) | 0 / 0 / 0 / plateau | 0/4 | false | false |
| LR10 (10× new-module LR) | plateau / 0 / 0 / plateau | 0/4 | false | false |
| Plain control (§120 FINAL) | 1.0 (3070) / plateau / 0 / plateau (borderline) | 1/4 | – | – |

* **Registered reading, "ESCAPE without ENTRY-FIX":** entry is fixed for some seeds, but something later fails.
  Restart-on-plateau stays the procedure.
  - VAR's Stage 2 stopped at its first failure (s4 plateaued), as 127-9 sequenced. STD2 and LR10 were decided at s4 (127-8).
* **The dominant failure is escape without learning.** It hits STD2 3/3 escaped seeds, LR10 2/2, SKIP 2/4, VAR 1/1, and the plain
  control s3. §131 registers a label-free 4k flag for it (2/2 caught held-out so far).
* **SKIP, descriptive:**
  - 2/4 exact against the plain control's 1/4 on the same host class. Its cost per exact seed is 2.0 run-equivalents, against
    the control's 2.17.
  - It never broke a learner (plain s1 learns, and so does SKIP s1).
  - Its trained chains run on the slot carrier alone (skip-off 0.999 / 1.0, SERIAL).
  - ρ₁ (127-12) is 0.183 in s4 (s1 pending). FF guard 2/2.
* **Seed 4 (127-13, decided):** SKIP s4 escaped and became exact, while the unstopped plain s4 ends at the null (4.69 at 24k,
  accuracy 0). **SKIP rescued a basin seed (n = 1).**
* **Mechanism notes kept (descriptive):**
  - the code-variance collapse at about 650 (127-7) is common and mostly transient across cards (§130);
  - stopping the collapse (VAR) fills the code with non-state content (127-9);
  - LR10 inflates the carrier (row norm 47 at 2k).

**Note 128-3 (2026-10-06 07:46 EDT, before any §128 S5 or FF slot is accepted; Audit's forecast; no rule change). How an S5 failure is read.**
* **The risk.** On the RTX 3090, plain §120 FINAL S5 is exact in 1 of 4 seeds in §127's control: s2 plateaus, s4 flags
  (borderline), and s3 escapes the 2k flag and ends at 0.
  - §128's restart rule acts only at 2k. It accepts an escaped seed like s3, so that slot then fails.
  - At that rate FAMILY-PASS(S5), and with it TASK-AGNOSTIC, is at real risk.
* **Reading, fixed now:** if FAMILY-PASS(S5) fails because accepted runs escaped the 2k flag and ended below 0.99, the result says
  **"S5 fails by escape-without-learning, not by entry"**, and TASK-AGNOSTIC is false as registered.
  - If slots exhaust their restarts instead, it is "fails by entry".
  - The same applies to FF.
* **Follow-up (a separate registration, not a change to §128):** §128's recipe plus §131's 4k restart point. It is registered
  only if §131 passes held-out (0 false positives among S5 learners, and ≥ 2/3 recall).
* The held-out families' result (MOD and CYC pass, RDFA pending) stands on its own either way. It is reported as "the recipe
  learns three held-out regular families without restarts", whatever S5 does.

**§131 result (2026-10-06 22:38 EDT; scored by Theory alone with `theory/score_s131.py` at the registered endpoint: 131-2, 131-3, 131-4, 131-5, 131-6, all inputs in
(s131x a39d4e65); Audit-reproduced independently). PASS on held-out and on prospective runs. Not deployable (8 flagged held-out runs, below 10).**
* **(i) No learner is flagged.** Held-out exact runs: 23, of which 0 are flagged (S5 learners 0/5; learners not flagged at 2k 0/23). Prospective:
  10 exact, 0 flagged. Precision 1.0 (95 % CI 0.63–1.0).
* **(ii) Most escape-without-learning runs are caught.** Held-out escaped-and-failed runs: 11, of which **8 are flagged**. Recall **0.73** (0.39–0.94),
  at least 2/3.
  - The prospective set is the same 11 (none was seen at registration).
* **The three misses, by class:**
  - s130chk_s5_s17_a4000: FLAG4 0.779, a late re-entry at the L_max 8 step (131-7).
  - s131x_s5_s402_2080ti: FLAG4 0.754, ending at 3.21, the ~3.2 basin.
  - s131x_s5_s401_2080ti: FLAG4 0.387, a **near-learner**. Its loss is 0.009 at 24k, with long_acc 0.987 and ext_2048 0.976. It counts as failed only
    by the 0.99 threshold.
* **Not deployable:** "deployable" as a restart rule needs ≥ 10 flagged held-out runs with 0 false positives (Literature (c)), and 8 are flagged.
* **Disclosed:**
  - The endpoint was extended twice after an interim reading (131-3, 131-5 / 131-6). Both were the conservative direction and fixed before
    the added runs existed.
  - s128_s5_s1 counted once, as its in-sample twin (131-8).
  - 131-4's "one host per card class" was not met (each run read on its own flags).
  - The s131x runs trained in fp32 on RTX 2080 Ti / 595.71.05.
* **Scope:** FLAG4 and its thresholds are calibrated on the 8-d FSQ carrier. Outside the registered set, a 64-d FSQ learner is flagged (133-5:
  s133_fsq64_s201, 0.930, then exact), and a 4060 prototype run is flagged at 0.906 while still descending (unresolved). Any use as a restart
  trigger is per carrier width.
* **Reading (as registered):** a label-free 4k check catches most escape-without-learning runs and never killed an 8-d learner on held-out data.
  As a restart rule it is not yet deployable. On S5, §134's copy anchor makes restarts unnecessary (4/4 without restarts).

## 132. Rules in context: can the same latent carrier track state when each problem gives its own transition table? A size ladder on §120's recipe (pre-registered 2026-10-06 07:48 EDT, after Literature's check 54a30677 and before any implementation or run; Audit's steward idea; the bridge to ProsQA-NA)

**Why.**
* §128 shows that one fixed recipe learns held-out regular families with no restarts: MOD and CYC pass, and RDFA's slots are
  exact. In every one the automaton is stored in the weights.
* ProsQA-NA, multiplication and anything GSM8K-like give their rules in the prompt. The open question for C1 / C2 at scale is
  whether the latent carrier tracks state when the transition table changes per problem.
* **Prior** (54a30677, `docs/LIT_S132_INCONTEXT_TABLE_2026-10-06.md`):
  - in-context language learning (Akyürek 2401.12973: regular languages induced from example strings);
  - induction heads 2209.11895; Pointer Value Retrieval 2107.12580 (indirection failures);
  - entity tracking with operations in text (2305.02363, 2402.14811).
  - New as scoped: a latent carrier executing a per-problem in-context table, certified and serial.

**Harness** (§120's recipe exactly: FSQ d = 8, L = 8, FINAL supervision, L_max 2 → 32, 24k steps, bs 64, lr 1e-4, the 127-3 guard;
RTX 3090 / 3090 Ti; restart-on-plateau slots as §128).
* **Table:** each problem draws m ops (" a" " b" …), each a uniformly random **permutation** of k states. Permutations never
  synchronise, so the state depends on the whole history.
* **Table text, as explicit pairs** (Literature (b)1: a content-addressed lookup, not a positional two-hop):
  " a0>3 a1>5 … b0>2 …", 4 tokens per pair, in canonical order. The row format is a later, harder variant, not run here.
* **Segment = [EOT, table text, carried code, item].** The table is a visible prefix in every segment, and **the state lives
  only in the code** (114-4's rule). Start state 0. The answer is the state digit (" 0" …).
* **Held-out tables are literal** (Audit): test tables are hashed and rejected from every training draw, and the overlap count
  is logged (it must be 0).
* **OOD set (descriptive; Audit):** training never draws a table whose op a is an involution (a∘a = identity). A separate test
  set uses only such tables.

**Size ladder (fixed now; Literature (c), Audit).**
* **Rung 1: k = 4 states, m = 2 ops** (table 32 tokens; 576 possible tables).
  - Held-out = 68 non-involution tables, split off before training. Training uses the remaining 268. OOD = the 240 tables with
    an involution a.
  - Dirs `s132_k4_s{seed}`.
* **Rung 2: k = 6, m = 3** (table 72 tokens). Held-out = 2,000 random non-involution tables, hash-excluded. OOD = 500
  involution-a tables. Dirs `s132_k6_s{seed}`.
* **Moving up:** pilot = rung 1, slot 1. If its accepted run is exact, slots 2–4 run. **Rung 2 runs only if rung 1 is PASS.**
* **What each failure means, fixed now, with no redesign under §132:**
  - **Rung-1 ENTRY-FAIL:** entry is the limit for in-context tables at this scale.
  - **Rung-1 NOT-LEARNED:** a per-step in-context lookup is not learned at this scale and budget.
  - **Rung 2 fails after a rung-1 PASS:** a size limit between 4/2 and 6/3.
  - A longer budget, the row format, or a bigger model would each be a new registration.

**Rules (fixed now; per rung).**
* **PASS:** ≥ 3/4 slots have an accepted run that is exact (cumulative ≥ 0.99 at 512 on held-out tables) and SERIAL. SERIAL uses
  the same-table donor at positions 2 / 4 / 6: donor ≥ 0.90, recipient − chance < 0.10.
* **ENTRY-FAIL:** ≥ 2 slots exhaust their four attempts.
* **NOT-LEARNED:** ≥ 2 accepted runs below 0.99.
* **§132's verdict is rung 2's.** A rung-1 failure stops the ladder and is the verdict.

**Measures (descriptive unless stated).**
* 2,048.
* **Different-table donor:** a code from a problem with a different table, read against the recipient's table. Does the code hold
  the absolute state number?
* **Table swap** mid-chain at test (from position 256).
* OOD accuracy.
* CERT on 20 sampled held-out tables (Moore = k).
* **Op-history probe** (Literature (b)3, Audit): R² of the code on the last 1–3 ops against on the state, at 2k and at the end.
  An 8-d code can hold a short op history, so a short-length success must be attributed correctly; PASS at 512 cannot be met
  that way.
* The plateau flag's 2k and 4k ratios (§131) are logged.

**Readings.**
* **PASS.** The latent carrier tracks state from in-context rules, the bridge to ProsQA-NA (a graph given in context).
* **NOT-LEARNED / ENTRY-FAIL.** As fixed in the ladder.

**Placement.**
* Rung 1: about $1.5 per run (segments about 35 tokens). Rung 2: about $3 per run (about 75 tokens). With restarts, at most about
  $20; Audit's earlier estimate was about $8, so this is flagged.
* Memory check at L 32 with the longest table on a 3090 first, then a CUDA-bf16 gradflow smoke.
* Implementation: the experiments session (fsq120 `--family table --k --m`). Scorer: `theory/score_s132.py`, committed before any
  run.

**Note 132-1 (2026-10-06 07:53 EDT, before any §132 implementation or run; Audit / claude-13's review). MEMORISED can arise at rung 1, plus two
scoring checks and a naming rule.**
* **MEMORISED (rung 1).** k = 4 / m = 2 draws from a finite pool of 268 training tables, each seen about 5,700 times over 24k × 64
  problems. So a weight-stored table lookup is available, unlike rung 2's 6!³ pool.
  - eval.json reports **train-table accuracy** (2,000 sequences on training tables) beside held-out accuracy, in both rungs.
  - **MEMORISED = ≥ 2 accepted runs with train-table accuracy ≥ 0.99 and held-out accuracy < 0.5.** It reads as weight-stored lookup,
    not in-context use.
  - Precedence: PASS, ENTRY-FAIL, MEMORISED, NOT-LEARNED.
  - Like any rung-1 result other than PASS, it **stops the ladder** (Audit's condition (1)). Whether rung 2's unmemorisable pool
    is then worth running is a new decision, not part of §132.
* **Scorer checks:** a run whose args k / m do not match its rung (s132_k4_* must be k 4 / m 2), or whose eval reports a nonzero
  held-out overlap, is an **error** and is not a valid attempt. The held-out set size is printed.
* **Smoke and memory-check runs use non-slot names** (e.g. `s132_k4_smoke`), never `s132_k4_s{seed}`. A smoke args.json inside a slot
  dir would hide that slot's later attempts.

**Note 132-2 (2026-10-06 07:59 EDT, before the pilot; Audit / claude-13's harness review of `boxes/table132.py`, which matches §132 on
everything PASS depends on; no rule change). Unreachable states in rung-1 tables.**
* **(A) 26 % of the 576 k4 / m2 tables do not reach all 4 states from start 0** (36 reach only 0, 36 reach 2, 78 reach 3).
  - Held-out 68: 10 intransitive tables, 4 of which never leave state 0. So about 6 % of held-out problems are trivial. That
    cannot carry a run to 0.99, and PASS is unchanged.
  - OOD 240: 42 % intransitive, 24 never leave 0, so ood_acc is softer than it looks.
  - **(i)** cert_sample reports `moore_eq_reachable` (n_moore against the table's reachable-from-0 count), not "Moore = k". A perfect
    machine on an intransitive table would otherwise count as a miss.
  - **(ii)** eval.json reports **long_acc_transitive** (held-out tables that reach all k states) beside the registered long_acc,
    descriptively.
* **(B)** The scorer's smoke filter also skips any args with mem_check set.
* The bf16 smoke runs under a non-slot name before the pilot. The fsq120 gradflow carries over: there is no new loss term.

**Note 131-2 (2026-10-06 08:20 EDT; interim score, and the end of held-out collection fixed so there is no optional stopping; no rule change).
The PASS conditions are met on the held-out runs so far. The verdict is read once the scheduled runs are in.**
* **Interim** (`score_s131.py`; SKIP s2 / rerun and s128_s5_s3 ≡ s127_s5_ctrl_s3 each counted once):
  - Held-out exact 18, flagged 0. Of these, 4 are S5 learners, flagged 0; the 14 non-S5 runs are near-vacuous (131-1 (a)).
  - Escaped-and-failed: 4 distinct, 4 flagged (LR10 s3 0.988, STD2 s3 0.994, ctrl s3 / s128 S5 s3, SKIP s3).
  - Precision 1.0 (CI 0.40–1.0), recall 1.0 (0.40–1.0). The prospective subset gives the same: 5 exact, 4 / 4 flagged.
* **§131 did not fix where held-out collection ends, so it is fixed now, before more lands.** Disclosure (Audit): the endpoint was fixed
  *after* an interim reading that already met PASS (18 exact / 0 flagged; 4 / 4 escaped-and-failed flagged). It extends collection
  rather than stopping at that reading, the conservative direction. The verdict is read once all of
  these have landed (or are void):
  - §128 FF slots 1–4 and RDFA s3 / s4;
  - §127 ctrl s4 full (b42);
  - the s130chk check runs: 4060 s5, s6, s7 and 3070 s17, s30, s34, s12.
  - Runs landing after that are reported but not scored.
* **Not yet deployable** either way (Literature (c)): 4 flagged so far, against the ≥ 10 flagged with 0 false positives needed
  for a restart rule.

**Note 119-5 (2026-10-06 08:36 EDT; CODI's s1 LR sweep, rerun with the autocast fix, read on-host by the experiments session; before CODI s2–3).
CODI did not learn at any LR. Its teacher learns written-out CoT and its latent student stays at chance. Seeds 2–3 run as
registered.**
* **CODI s1** (d3, RTX A5000, driver 595.84): intact 0.0075 / 0.007 / 0.0075, train split 0.011 / 0.0104 / 0.0106 at lr 1e-3 / 3e-3 /
  1e-2, cumulative 0; block donor and removal at chance.
  - Guard (a): "did not learn", at every LR.
* **The fix is delivered: the LoRA trains.** At 1e-3 the teacher (written-out CoT through the same model) reaches CE 0.000 by 6k.
  - The student (4 latents) stays at about 0.44 per token, chance over the answer span.
  - KD stays about 0.01: the hidden state at ":" can be matched without the latents carrying the S5 state.
* **LR, by the registered criterion: 1e-3** (train split 0.011, the highest, though all three are noise-level; disclosed).
  CODI s2 / s3 run at 1e-3, as registered (3 seeds per method), as for Coconut (119-2) and SIM-CoT (119-4).
* **So far**, all three published methods read "did not learn" on this host: Coconut 3/3, SIM-CoT s1 (s2–3 running), CODI s1.
  - Text CoT learns in every port (Coconut stage 0, CODI's teacher).
  - If CODI s2–3 and SIM-CoT s2–3 also fail, 119-3's all-inconclusive wording applies.

**§129 result (2026-10-06 08:50 EDT; scored by `theory/score_s129.py`; b40, RTX 3090, driver 595.71.05; host gradflow PASS under bf16; PANEL = s2, s4,
s202, s301, the first 4 flagged in the fixed order, with 10 seeds unflagged; Audit reproduced). Descriptive: a fresh head escapes
where a warm restart does not, but no re-drawn seed learns.**

| PANEL seed | REDRAW: phase-2 flag / long accuracy | CLOCK-ONLY: phase-2 flag |
|---|---|---|
| s2 | re-plateaus | re-plateaus |
| s4 | **escapes** / 0 (donor about 0.01) | re-plateaus |
| s202 | **escapes** / 0 (donor about 0.14) | re-plateaus |
| s301 | re-plateaus | re-plateaus |

* **Registered rules:**
  - LR (CLOCK-ONLY escapes ≥ 3/4): no, 0/4.
  - HEAD (REDRAW passes ≥ 3/4 and escapes ≥ 2 more than CLOCK): no, 0 passes.
  - BODY (REDRAW re-plateaus 4/4): no, 2/4.
  - **Reading: descriptive** (129's "otherwise").
* **What it shows (descriptive, n = 4):**
  - **The warm restart alone never breaks the plateau (0/4). A fresh head does in 2/4.** The plateau is partly held in the
    head's state.
  - **But no re-drawn seed learns long chains.** Both escapes end at 0: escape without learning again, as in §127.
  - **The trained body does not ignore a fresh head** (129-1's probes). Re-drawn heads pick up state content fast: R²_state about
    0.41 at clock 700 in s4 and s202, and 0.96 in s301. The decoder reads them (abl_dloss 0.3–15).
  - **s301:** its code holds the state (R²_state 0.96) and the decoder relies on it, yet its loss rose from 3.36 to 3.73 and the
    clock-2k flag fired. That is another instance of the flag firing on a run whose code carries the state (§130; §131's check
    runs).
* **§131, descriptive (phase-2 runs are not scored there):** at clock 4k / 2.6k, s4's ratio is 0.80 (no FLAG4, fails) and
  s202's is 1.01 (FLAG4, fails).
* **Disclosures:** 129-2's voided first seed-2 attempt (phase 1 flagged; continuations crashed unread) and the killed s1 stay
  void. Phase-2 checkpoints were deleted after both continuations (listed in the driver summary). Audit's line (129-2): "keep
  the body" includes its small Adam second moments.

**Note 128-4 (2026-10-06 09:27 EDT; descriptive, before §128's result is written; Audit's check against fsq120.certificate(); no rule change).
CLOSED covers the certificate's own forward pass. A few codes the eval visits are uncertified.**
* **RDFA s4's CERT is closed** (110 reachable, Moore 8), so RDFA is CLOSED in 4/4. Every certified MOD / CYC / RDFA run is closed
  with Moore = the minimal state count.
* **Caveat.** For non-S5 families, the CERT's BFS starts from the same start code the eval chains use. Yet in 5 of 12 certified
  runs the eval visits a few more distinct codes than the BFS reaches:
  - cyc s1 89 / 88; cyc s2 40 / 39; mod s1 387 / 375; rdfa s2 83 / 80; rdfa s4 123 / 110.
  - With deterministic rounding every visited code should be reachable. The likely cause is bf16 batch sensitivity: the eval runs at
    batch 250 / 500 and the CERT at 2,048, so a few borderline codes round differently (the same class as the bf16 contraction
    artefact, 123-5).
  - So "CERT holds" certifies the BFS set under the CERT's own forward pass. 1–13 codes the eval visits are correct (long accuracy
    1.0) but uncertified.
* **Checks (descriptive):**
  - **(a)** CERT re-run in fp32 (`--cert_only` with autocast off) on mod s1 and rdfa s4, then on all 12 if cheap, against an fp32
    eval's code set.
  - **(b)** `n_uncovered` = |eval codes − BFS set| reported per run.
  - If the excess vanishes in fp32, it is a bf16 artefact, and §128's result says so.
* **Wording for the §128 result, fixed now:** CLOSED is stated as "closed under the certificate's forward pass; n_uncovered eval
  codes are correct but uncertified". The fp32 outcome goes beside it.
* **First fp32 outcome (2026-10-06 10:41 EDT; rdfa s4, local RTX 4060, 2,000 × 512; Audit reproduced): the gap is a bf16 artefact.**
  - bf16: eval 119 codes, BFS 119, n_uncovered 3; holds, Moore 8.
  - **fp32: eval 123, BFS 125, n_uncovered 0;** holds, Moore 8, not capped.
  - The bf16 BFS set is device-dependent (110 on the original 3090 Ti, 119 on the 4060). A mod s1 bf16 re-run gives 377 eval codes
    against 387 originally.
* **CLOSED stays as registered** (the CERT as run, which holds in all 12 runs). The fp32 certificate is reported beside it as the
  stronger statement, "the certificate covers every code the eval visits". The definition is not switched after seeing data.
  - The fp32 pass runs on all 12 certified runs at low priority on the local 4060, after §130's replays. The §128 result carries an
    n_uncovered table (bf16 and fp32) and fills it as runs land.
  - **Scope, revised (2026-10-06 10:46 EDT; Audit):** fp32 runs only where bf16 shows a gap (cyc s1, cyc s2, mod s1, rdfa s2; rdfa s4 done), on one
    Vast RTX 3090, so the local check runs are not slowed. In the other runs every bf16 eval code is already in the BFS set. The table
    is n_uncovered in bf16 for every run, and in fp32 wherever bf16 shows a gap.
* **Complete (2026-10-06 11:35 EDT; 5 runs on one RTX 3090, `boxes/runs_fsq/s128cc3090_*`, plus 2 local 4060 cross-checks; Audit reproduced).**

| Run | bf16 n_uncovered (3090) | fp32 n_uncovered | fp32 CERT |
|---|---|---|---|
| cyc s1 | 1 | **0** | holds, Moore 5 |
| cyc s2 | 8 | **0** | holds, Moore 5 |
| mod s1 | 14 (4060: 12) | **0** (4060: 0) | holds, Moore 5 |
| rdfa s2 | 1 | **0** | holds, Moore 8 |
| rdfa s4 | 11 (4060: 3) | **0** (4060: 0) | holds, Moore 8 |

  - **In fp32 the certificate covers every code the eval visits, in 7/7 passes,** with an unchanged minimal machine. fp32 code sets
    reproduce across cards (mod s1 382 / 386 and rdfa s4 123 / 125, on both the 4060 and the 3090).
  - bf16 counts are host-dependent even within the 3090 class (mod s1 BFS 375 originally, 379 here).
  - **Wording:** "CLOSED (as registered); in fp32 the certificate covers every visited code and reproduces across GPUs". The bf16
    uncovered codes are a rounding artefact.

**Note 119-6 (2026-10-06 09:39 EDT; SIM-CoT s2 / s3 in, scored by `theory/score_s119.py`; CODI s2 / s3 pending). SIM-CoT did not learn in 3/3; all
three seeds had a degraded start.**
* **SIM-CoT s2 / s3** (lr 1e-4 from Coconut s2 / s3; RTX 3090, driver 570.211, bf16):
  - intact 0.014 / 0.007, train split 0.0106 / 0.0104, cumulative 0;
  - stage-0 CE at 3k 0.193 / 0.759, both > 0.1, so **degraded start** (119-3) in 3/3 with s1.
* **Descriptive:** in s3, the step decoder φ partly reads step state from the thoughts (final step CE 3.33, against about 4.9 at
  chance), but the answer path stays at chance. That is partial state in the latents without use.
* **Tally:** Coconut 3/3 and SIM-CoT 3/3 "did not learn"; CODI s1 "did not learn", with s2 / s3 running (all bf16 on 3090 / A5000).
  CONTROLLED-C1 is so far not readable. 119-3's all-inconclusive wording is expected once CODI lands.

**Note 132-3 (2026-10-06 09:51 EDT; disclosure, before the pilot's result is read). The per-term gradflow check ran after slot 1 started, not
before.**
* §132's order put the CUDA-bf16 gradflow check before the pilot. Only the bf16 GPU smoke (training runs) preceded slot 1.
* The explicit check ran afterwards on the pilot host (b43, RTX 3090, driver 595.71) and **passes**
  (`theory/results/gradflow_check_s132_host.json`):
  - k4 / m2: main-loss gradient 81.9, reaching W_down 48.6, W_up 7.6, LN 0.16;
  - k6 / m3: 97.7, reaching W_down 56.0, W_up 4.7, LN 0.73.
* So the pilot ran on a delivered code path. The ordering slip is recorded. The pilot is not rerun.

**Note 131-3 (2026-10-06 11:50 EDT, before the runs exist; Audit's suggestion; an extension of 131-2's endpoint, disclosed). Four more held-out S5
runs on the idle local RTX 2070.**
* **Runs:** `s131x_s5_s{401,402,403,404}_2070`, plain fsq120 §120 FINAL S5, 24k, **no plateau stop**, run sequentially, with the guard,
  train_driver and a memory check at L 32 first (`--micro` as needed).
  - The 2070 (Turing) has no bf16, so they train in **fp32**: a different numerics class from every other held-out run, stated per
    run.
* **They join §131's held-out set and its endpoint.** The verdict is read once they land too.
  - Extending the endpoint after 131-2's interim reading is the conservative direction: each added run is another chance for a
    learner to be flagged. It is fixed before any of them runs.
* **Why:** criterion (i) as registered (≥ 5 exact held-out runs) is already met (18), but only 4 of those are S5 learners. The
  non-S5 ones are near-vacuous for (i) (131-1 (a)). These runs add S5 learners for (i), and possibly escaped-and-failed runs for (ii).

**Note 132-4 (2026-10-06 11:58 EDT; the rung-1 pilot, read by Theory from eval.json; Audit reproduced; a disclosed amendment to the pilot gate,
fixed before any further §132 run). The pilot is near-exact, in context, SERIAL and not memorised. Slots 2–4 open.**
* **Pilot `s132_k4_s1`** (RTX 3090, driver 595.71.05; accepted on its first attempt):
  - **long accuracy 0.9865** on held-out tables (just under 0.99), 0.988 at 2,048; by length 0.9985 / 0.998 / 0.994 / 0.9915 / 0.9875 /
    0.9865 at 16–512, a slow per-step error, not a collapse.
  - **train_acc 1.0; held-out overlap 0.** Not MEMORISED.
  - **SERIAL:** same-table donor 1.0 at positions 2 / 4 / 6, recipient = chance.
  - **The code holds the absolute state:** a different-table donor is read correctly under the recipient's table (0.985 / 0.9885 /
    0.99).
  - Table swap at 256: 0.981 over the next 8 steps, cumulative 0.958.
  - **No op-history shortcut:** at the end, R² on the last 1–3 ops is about 0.01, on the state 0.41.
  - cert_sample: 19/20 held-out tables closed with Moore = reachable states. One is open (56 codes, Moore 33, for 4 states).
  - **OOD (involution-a tables) 0.18.** The in-context lookup generalises within the training distribution of table shapes, not
    to involution-a tables.
* **The gap in §132.** The pilot gate opened slots 2–4 only for an exact pilot, so rung 1 would stop at "incomplete (1/4)" with no
  registered outcome.
* **Amendment (disclosed; fixed before any further run; Audit proposed it):** the gate was a cost gate. Slots 2–4 open when the
  pilot is accepted with long accuracy ≥ 0.95. The rules are unchanged.
  - The pilot counts as a non-exact slot, so **PASS now needs s2, s3 and s4 all exact.** NOT-LEARNED needs one more accepted run
    below 0.99.
  - About $6 more, inside the $20 ceiling.
* **Wording, fixed now, for a NOT-LEARNED made of near-exact runs** (accepted runs at 0.95–0.99 that are SERIAL, not memorised, and
  carry state across tables): "NOT-LEARNED (registered label): near-exact in-context tracking, with residual errors concentrated on
  tables whose code machine is open". The per-table breakdown comes from the kept model (descriptive).
* Rung 2 still opens only on a rung-1 PASS.
* **Reading of the per-table breakdown, fixed before it runs (2026-10-06 12:04 EDT; Audit / claude-13's review of `scratch/table132_breakdown.py`):**
  - The exact replay may miss 0.9865 by a few chains: bf16 forwards differ between 3090 hosts (128-4), and the training host is gone.
    |long_acc_replay − 0.9865| × 2,000 is read as host-dependent flips, not a scorer error.
  - **The fresh 68-table sample (100 problems × 512 per table) is the primary measure** of where the residual sits. The replay's
    per-table attribution is the breakdown host's forward only.
* **Probe change, disclosed (2026-10-06 14:36 EDT):** probe_ophist now runs in 256-row chunks on identical draws (the 1,024-row pass ran out of memory
  on a local card). Rung-1 slots 2–4 use the old single pass. Values from later runs can differ in the last digits (bf16 batch-size
  kernels), and they are compared at that resolution. The probe is descriptive.

**§128 result (2026-10-06 12:23 EDT; scored by `theory/score_s128.py`, all 20 slots; RTX 3090 / 3090 Ti, drivers 565 / 595; Audit reproduced).
TASK-AGNOSTIC = true. One fixed recipe gives exact, serial latent state tracking on five regular families, three held out, with
the held-out families closed.**

| Family | FAMILY-PASS | Slots exact + SERIAL (+ CLOSED) | Restarts | Delétang score | Notes |
|---|---|---|---|---|---|
| S5 | **true** | 3/4 (s1 0.9985, s102 0.9995, s104 1.0) | 2 (slots 2, 4) | 0.9995–1.0 | slot 3 accepted, then escaped without learning (0) |
| FF | **true** | 4/4 (1.0) | 0 | – | |
| CYC (held out) | **true** | 4/4, CLOSED (Moore 5) | 0 | 1.0 (Transformer 61.9) | 1.0 at 2,048 and 8,192 |
| MOD (held out) | **true** | 4/4, CLOSED (Moore 5) | 0 | 1.0 (Transformer 24.2) | 1.0 at 2,048 |
| RDFA (held out) | **true** | 4/4, CLOSED (Moore 8) | 0 | 1.0 | 1.0 at 2,048 |

* **Registered reading, TASK-AGNOSTIC:** one fixed recipe gives exact, serial, discrete latent state tracking in a pretrained LM on
  five regular families, three of them held out. The recipe is GPT-2 small, an 8-d FSQ carrier, FINAL supervision, and the
  label-free 2k restart, with no per-task template or schedule. It covers C1 + C2 + C6 (with restarts), and C4 as final-answer
  supervision.
  - The held-out families needed **no restarts** (0 in 12 slots). The recipe's only restarts (2 in 20 slots) were on S5.
* **Caveats, stated with the claim:**
  - **S5 is the weak family:** exactly 3/4. One accepted run escaped the 2k flag and never learned (128-3's failure mode,
    occurring once). §131's 4k flag is the registered tool for it.
  - **CLOSED is as registered** (the CERT holds, with Moore = the minimal state count). The bf16 certificate is host-dependent:
    1–14 eval codes per run fall outside the bf16 BFS set. **In fp32 the certificate covers every code the eval visits in 7/7
    passes, and reproduces across GPUs** (128-4).
  - **Donor power varies by family.** S5's chance overlap is 0.01, so its SERIAL test is strong. In FF / MOD / RDFA the overlap is
    0.3–0.7, because resets and synchronisation make many chains uninformative. The donor is followed in every informative chain,
    but fewer chains test it.
  - MOD and RDFA are suffix-dominated (MOD's ×0 resets; RDFA synchronises in a median of 9 steps). CLOSED is what keeps them
    non-trivial.
* **Scope, and what this does not show:**
  - It covers regular-language state tracking only: no stack- or tape-bound task.
  - It uses GPT-2 small, not a model at pretraining scale.
  - Each automaton is stored in the weights. Whether the same carrier tracks state from rules given in context is §132 (rung 1
    near-exact so far, 132-4).
  - S5's 2k-restart rule leaves escape-without-learning uncaught; §131 is open.
* **FF certificates (descriptive; 2026-10-06 13:27 EDT; post-hoc on the local RTX 2070, so fp32; Audit reproduced).**
  - The generic CERT fails in 4/4: it reads from an undefined start before any write, the ill-posed case known from §120 (120-7).
  - The task-language cert_task holds in s3 / s4 (Moore 2) and fails in s1 / s2.
  - The reads-only certificate (120-9) holds in 4/4.
  - The CERT output did not record device or dtype; it is labelled here from the run log. --cert_only is asked to record both.
* **S5 certificates (descriptive; 2026-10-06 14:15 EDT; wording fixed before the cloud pass on b61 is read).** The local 2070 fp32 CERT on S5 s1 hit
  its 45-min limit with **173,486 reachable codes** (from 14,400 start pairs, still growing), against 3,042 codes visited by the eval.
  - So S5's code machine, explored from every (state, item) start, is far larger than the part the eval chains use. It is not a
    clean 120-state machine off the eval distribution.
  - If b61's fp32 pass also times out, the S5 row reads **"CERT inconclusive by time, 173k+ reachable"**, nothing implying a closed
    machine. FAMILY-PASS(S5) is unaffected: CERT is descriptive for S5.
* **Contraction (descriptive; 2026-10-06 13:30 EDT; fp32, local RTX 3070 and earlier runs). The first-hand-off ratio ρ₁ (127-12) is the informative
  number: the overall ρ is vacuous (about 1e-6) under hard rounding.**
  - **Every one of the 19 learning runs contracts, ρ₁ 0.03–0.27, inside 118-5's pass band (≤ 0.28):**
    - FF 0.051 / 0.116 / 0.078 / 0.032;
    - MOD 0.152 / 0.163 / 0.117 / 0.103;
    - CYC 0.181 / 0.129 / 0.118 / 0.168;
    - RDFA 0.216 / 0.149 / 0.269 / 0.185;
    - S5 0.201 / 0.147 / 0.127.
    - **RDFA s3 (0.269) is close to the bound.** (Corrected from an s1-only range of 0.03–0.22; Audit.)
  - Exact-recovery share 0.084–0.266. Basin 0.1 (S5) to 0.3–1.0 (FF / MOD / CYC / RDFA).
  - **S5 s3, which never learned, shows ρ₁ 0, exact recovery 0.56 and basin 1.0.** A collapsed code is contractive without
    information, as in §123's plateaued cell.

**Note 131-4 (2026-10-06 12:40 EDT, before any rerun is read; the user's directive via Audit). Two check runs crashed (out of memory at 14k), and
all seven are rerun with checkpoints on larger cloud cards.**
* **Crashes:** `s130chk_s5_s5_4060` and `s130chk_s5_s17_3070` ran out of memory at about 14k (the L = 32 stage does not fit 8 GB at
  micro 4). With no checkpoint they cannot resume.
  - Their partial logs are kept (`*.void_oom`) and are **descriptive only**. At 14k, s5_4060 was at 4.78 (the null) and s17_3070 at
    3.43 (the ~3.2 basin). Neither enters §131's counts.
* **Reruns (user):** checkpointing (`--ckpt_every` / `--resume`, with a resume-parity smoke first). All seven s130chk runs restart
  from scratch on 16 GB Vast cards, one host per card class, driver recorded:
  - 4060 seeds (s5, s6, s7) on an **RTX 4060 Ti 16 GB**: the same Ada architecture, but a different die (AD106, against the desktop
    4060's AD107);
  - 3070 seeds (s17, s30, s34, s12) on an **RTX A4000 16 GB**: the same GA104 chip.
  - Directory names carry the new card (e.g. `s130chk_s5_s5_4060ti`, `s130chk_s5_s17_a4000`).
* **Card identity, as read:** each rerun is read on **its own** 2k flag and its own 24k outcome. That is all §131 and §130's held
  question ("does the 2k flag fire on runs that learn?") need, so strict identity is not required.
  - "This replay seed was a late learner" does need the same trajectory. So each rerun's loss at 1k and 2k is compared with its
    replay's (parity: the flag outcome matches, and the losses agree within 0.1). The result says, per seed, whether the rerun
    reproduces its replay.
* §131's endpoint (131-2 / 131-3) now names the reruns in place of the crashed and unfinished local runs.

**Note 131-5 (2026-10-06 12:43 EDT, before any s131x run is read; the user's directive via Audit: full training runs go to cloud cards of ≥ 16 GB
with checkpoints; local 8 GB cards run short tests only). The s131x held-out runs move off the local RTX 2070.**
* **Where they run:** the four held-out S5 runs of 131-3 (seeds 401–404) move to a cloud card with checkpointing, chosen in this order,
  fixed now:
  - (a) **a cloud Turing card with ≥ 16 GB (Titan RTX)**, keeping fp32 numerics (no bf16 on Turing), as 131-3 intended. Dirs
    `s131x_s5_s{seed}_titanrtx`.
  - (b) if none is available, **an RTX 3090 in the default bf16**, the card class of most §131 held-out runs. Dirs
    `s131x_s5_s{seed}_3090`.
    - The fp32 angle is then dropped. It was incidental: the rule is loss-only, and these runs exist to add S5 learners.
* Any 2070 partial is kept as descriptive only (`*.void_moved`) and does not enter §131's counts.
* §131's endpoint names the moved runs in place of the 2070 ones. Seeds and settings are otherwise as in 131-3.

**Note 131-6 (2026-10-06 12:55 EDT, before any s131x run is read). The s131x card, and the root cause of the 14k crashes.**
* **Card:** 131-5's option (a) is widened from "Titan RTX" to **"a cloud Turing card with ≥ 16 GB"**. The four s131x seeds (401–404) run
  on **RTX 2080 Ti 22 GB** hosts (TU102; no bf16, so fp32 as 131-3 intended), with checkpointing. Dirs `s131x_s5_s{seed}_2080ti`.
  - It is a different die from the 2070 (TU106) in the same architecture. Each run is read on its own flags and outcome (131-4).
  - The local 2070's s401 (old code) will run out of memory at 14k. Its partial is descriptive only (`*.void_moved` / `*.void_oom`)
    and never counted.
* **Root cause of the out-of-memory crashes (the experiments session):** fsq120 applied `--micro` only at L ≥ 32, while training draws
  L ~ U(1, L_max). After 14k, every batch with L of 17–31 ran at the full 64 rows.
  - Fixed in 105545b8: micro now applies at L > 16. It is gradient accumulation, so memory only, and a no-op at micro 1.
  - Runs with micro > 1 after this fix differ from earlier runs at the bit level for L of 17–31 (summation order). Each run is read
    on its own.

**§119 result (2026-10-06 14:02 EDT; scored by `theory/score_s119.py`; all nine seeds in; every port met faithfulness guard (b), 119-1). CONTROLLED-C1
is not readable: no published method learned the task. 119-3's pre-stated wording applies.**

| Method | Seeds | Intact final-state accuracy (chance 1/120 = 0.0083) | Guard (a) | Notes |
|---|---|---|---|---|
| Coconut (c = 1) | 3 | 0.007 / 0.0085 / 0.0115 | did not learn, 3/3 | text CoT (stage 0) learns, to loss 1e-4 |
| SIM-CoT (c = 2, from Coconut) | 3 | 0.007 / 0.014 / 0.007 | did not learn, 3/3 | degraded start in 3/3 (119-3); s1 a constant predictor; s3's step decoder partly reads state |
| CODI (4 latents, LoRA) | 3 | 0.0075 / 0.0065 / 0.0085 | did not learn, 3/3 | teacher (text CoT) CE 0.000; latent student at chance |

* **Wording (fixed in 119-3):** "Under their released recipes, at our matched budget (24k updates, bs 64, a 3-value LR sweep per method)
  on the chunked S5 host, Coconut, SIM-CoT and CODI do not learn 4-op S5 composition through latent thoughts. In-chunk final-state
  accuracy is below 0.90 in every seed (at chance). Text chain-of-thought does learn (Coconut stage 0; CODI's teacher)."
  - "There is no competent published comparator for our serial C1 on this protocol, so the C1 claim stays uncontrolled by
    baselines."
  - "This is a competence finding (C3: latent methods below text at matched compute on this task), not evidence about their serial
    use."
  - Limit: "at this budget on this host", never "cannot".
* **Disclosures:** each LR was picked by the registered criterion at noise level (119-2 / 119-4 / 119-5). CODI's first sweep was
  void (an autocast-cache bug, fixed in c1caef75). The step-key logging bug was fixed. CODI s2 / s3 ran bf16 on an RTX 3090, and s1
  on an A5000.
* **For contrast (not part of §119's rule):** our own method's serial C1 holds on this task family, e.g. §111 READ (5/5) and §128's S5 /
  CYC / MOD / RDFA (exact and SERIAL). No published baseline reaches competence here to compare against.

**§132 rung-1 result (2026-10-06 17:20 EDT; 4 accepted runs on RTX 3090 / 595.71.05 by the experiments session, each accepted on its first attempt with no
restart; scored by Theory alone with `theory/score_s132.py`; helper-read and Audit-reproduced). Rung 1 (k 4, m 2) is PASS (3/4), so rung 2
(k 6, m 3) opens.**
* **Slots 2 / 3 / 4: exact and SERIAL.**
  - long_acc 1.0 and ext_2048 1.0 on 68 held-out tables, train_acc 1.0, heldout_overlap 0 (not MEMORISED).
  - Donor 1.0 with recipient − chance 0.000 at positions 2 / 4 / 6.
  - cert_sample holds in 20/20, with Moore = reachable states.
  - Solved (first loss ≤ 0.01) at 7,700 / 3,800 / 3,000 steps.
  - The end probe holds the state (R²_state 0.93 / 0.84 / 0.88) with no op history (≤ 0.003). The per-family code is a state code,
    as in §128.
* **Slot 1, the pilot: near-exact, 0.9865, and not a pass.** The per-table breakdown (`s132bd_k4_s1/eval_breakdown.json`, a replay that
  matches the eval) puts **all 27 eval failures on one held-out table**, i 63 = [[2, 1, 3, 0], [0, 3, 2, 1]], with 4 reachable states.
  - That table has sample_acc 0.0, a median first error at position 106.5, and a CERT that does not hold (n_moore 33 against 4
    states): an open, drifting code for one table.
  - The other 67 tables sample at 1.0, with CERT holding and Moore = reachable.
* **Descriptive:** ood_acc (involution tables, not a rule) is 0.074–0.39.
  - Slot 3's files are read from the fetched run; the drain commit is pending.
* **Next (registered):** rung 2 (k 6, m 3, 2,000 held-out non-involution tables), with a memory check at L 32 and a CUDA-bf16 gradflow
  smoke first, then Audit's budget approval (about $3 per run). §132's verdict is rung 2's.
  - **Risk, stated now:** the local k6 prototype (s132proto_k6_s1, 2k) stopped on the plateau flag with a code holding the last three
    ops verbatim (R² 0.96–0.99, state 0.0). That is the op-history shortcut as a plateau code.

## 133. Can error correction make a wide latent carrier exact? Width × correction on §128's S5 recipe, then a capacity probe a narrow code cannot hold (pre-registered 2026-10-06 14:52 EDT, after Literature's check 7dbdb426 and before any implementation or run; the user's question "can we fix error propagation to maintain latent expressiveness", via Audit)

**Why.**
* The 8-d FSQ code is exact and contractive (§128: every learner has ρ₁ 0.03–0.27), but its capacity is capped at 8⁸ ≈ 1.7·10⁷ codes
  (24 bits).
* Every wide carrier tried so far drifts: RAW-H 768-d, ρ_ff 0.5–0.8, 0/4 exact (§126). §119's continuous published methods did not
  learn at all.
* §133 asks whether a task-agnostic error-correction mechanism keeps a **wide** carrier exact, and whether the width then buys
  capacity the 8-d code lacks.
* **Prior** (7dbdb426, `docs/LIT_S133_WIDE_CARRIER_2026-10-06.md`):
  - soft-to-hard VQ 1704.00648; FSQ 2309.15505; the dither proxy for quantisation (1611.01704 / 2006.09952);
  - Lipschitz RNNs 2006.12070; Stable Recurrent Models 1805.10369; Jacobian regularisation 1908.02729; contractive autoencoders
    (Rifai); nGPT.
  - New as scoped.

**Harness.** §128's recipe exactly: §120 FINAL fsq120 on S5, the length curriculum, 24k steps, bs 64, the 127-3 guard, and
restart-on-plateau slots k, k+100, k+200, k+300.
* **Cloud RTX 3090 only, with checkpoints** (the user's rule), driver recorded.
* Only the carrier changes. z_b has width d, and how it is corrected at each hand-off is the arm.

**Stage 1 arms (one change each).**
* **CONT-64:** z_b of width 64, never rounded. The control: does a wide code drift?
* **FSQ-16 / FSQ-64:** hard FSQ (8 levels) at width 16 / 64. Wide but discrete.
* **SOFT-16 / SOFT-64:** s = (1 − β)·z_b + β·round(z_b) with **β = 0.1** at every hand-off, train and test. A weak attractor toward the
  FSQ grid, with no learned codebook; §101 found β 0.01 enough on HYB-4. Grid share is reported (Literature (b)1).
* **NOISE-64:** continuous width 64, plus training dither U(±0.25·std_d) at every hand-off after 2k (§123's dither at width 64).
* **PIN-64:** continuous width 64, plus a parameter-free LayerNorm on z_b before W_up (a norm pin). Angular drift is reported
  (Literature (b)4).
* **JAC (contraction penalty): not run** (Literature (b)2, Audit). S5's carry is an isometry on its states, so a global ‖J‖ bound
  merges states (stability means forgetting, 1805.10369). An anisotropic, off-data penalty would be a separate registration.
* **Proof seed per arm: S5 slot 1 (seed 1),** a learner under FSQ-8 on the 3090 (§128). **If slot 1's accepted run is not exact at
  2,048, the arm is dropped, with no further seeds and no variants.**

**Stage 1 rule (fixed now).**
* **PASS(arm):** at least 3/4 slots whose accepted run is **exact at 2,048** (ext_2048 ≥ 0.99), **SERIAL** (donor ≥ 0.90,
  recipient − chance < 0.10), and **contractive** (ρ ≤ 0.28 by 127-12's clause: ρ_ff, or ρ₁ when rounding recovers exactly).
* CERT is **not** part of PASS: it may not close at width 64 (Literature (b)3).
  - Reported where feasible: a time-capped fp32 CERT plus n_uncovered (128-4), with "inconclusive by time" an allowed outcome.

**Stage 2: capacity (only after Audit re-approves; arms = the stage-1 passers plus two FSQ-8 controls).**
* **Tasks** (all registers queried at the end; final-answer only):
  - **S5×2:** two independent S5 registers. Items are (register, op), and the state space is 120² = 14,400. The in-capacity control.
  - **S5×4:** four registers, 120⁴ ≈ 2.07·10⁸ states (27.6 bits), more than FSQ-8's 24 bits. The capacity probe.
  - The answer is all registers' states, so every register gets gradient on every problem (Literature (b)5).
* **Controls:**
  - **FSQ-8** (8 levels, 24 bits) on S5×2 and S5×4;
  - **FSQ-8 at 16 levels** (32 bits, 16⁸ ≈ 4.3·10⁹ ≥ 120⁴) on S5×4. It separates "number of codes" from "width".
* **CAP-PASS(arm, task):** long accuracy ≥ 0.99 at 512 in at least 3/4 slots.
* **Reading a capacity limit, fixed now (Audit):** FSQ-8's S5×4 failure is read as **capacity** only if all three hold:
  - (i) FSQ-8 passes S5×2, so the task is learnable at this size;
  - (ii) FSQ-8 enters on S5×4 (no exhausted slots; accepted runs past the 2k flag) and ends below 0.99;
  - (iii) a wide arm, or FSQ-8 at 16 levels, passes S5×4.
  - Otherwise it is "difficulty" or "entry", and no capacity claim is made.
  - Codes per state and per-register probe R² are reported beside it.

**Readings.**
* **A wide arm PASSES stage 1 and CAP-PASSES S5×4, where FSQ-8 fails by capacity.** Error correction keeps a wide latent exact, and the
  width buys capacity the 8-d code lacks. The answer to the user's question is yes, with that mechanism.
* **Only FSQ-W passes.** Width helps when the code is discrete; the soft or continuous mechanisms did not keep a wide carrier exact.
* **No wide arm passes.** These task-agnostic corrections do not keep a wide carrier exact on S5. The 8-d discrete code stays the
  exact carrier, and its capacity cap is stated.

**Placement.**
* Stage 1: 7 proof runs, about $6–7. Completing the passing arms: up to about $3 each.
* Stage 2: about 4 runs per cell (S5×2: FSQ-8; S5×4: FSQ-8, FSQ-8-L16 and the passers), about $15–20.
* About $35 in all. Audit approves each stage against credit.
* Implementation: the experiments session (fsq120 `--d_q`, `--levels`, `--soft_beta`, `--carrier_ln`, and a `--family s5x{2,4}`).
  Gradflow under CUDA bf16 for every arm before any run.
* Scorer: `theory/score_s133.py`, committed before any run.

**Note 133-1 (2026-10-06 14:53 EDT, before any §133 implementation or run; Audit / claude-13's scorer review, which crossed with the registration).
Which contraction number each arm is held to, plus scoring checks.**
* **(a) The ρ clause per arm type, fixed now.**
  - Rounded arms (FSQ-16, FSQ-64, and stage 2's FSQ-8 controls) use 127-12's clause: ρ_ff, or ρ₁ when rounding recovers exactly.
  - Continuous and soft arms (CONT-64, SOFT-16, SOFT-64, NOISE-64, PIN-64) never collapse the perturbation at test, so hand-offs at the
    fp32 floor would enter ρ_ff and could fail a genuinely contractive carrier on numerics. They are held to **the geometric mean of
    ρ_k over the first 3 hand-offs.**
  - The full ρ_k row is reported beside every value.
  - **The 0.28 bound is carried over from d = 8** (118-5). It is stated as such, not re-derived for wide codes.
* **(b) Args checks.** A run whose args do not match its arm is an error row and an invalid slot. Arms are checked on d_q, levels,
  no_round, soft_beta, dither and carrier_ln. The smoke filter also skips mem_check args.
* **(c)** A warning prints when an arm's accepted runs span more than one GPU name (125-2).
* **(d)** JAC was not run and counts for nothing: no arm, no verdict, no evidence either way.

**Note 133-2 (2026-10-06 14:55 EDT, before any §133 proof result exists; Audit / claude-13). The proof-slot near-miss rule, a mechanical capacity
reading, and a refinement of 133-1 (a).**
* **(1) Near miss at the proof slot (the §132 lesson; FSQ-8's own s1 is 0.998).** An arm is **dropped only if its proof seed is below 0.95
  at 2,048.**
  - Between 0.95 and 0.99, the proof counts as a non-exact slot and slots 2–4 run, so PASS then needs all three exact, as in 132-4.
    Each step still needs Audit's re-approval.
* **(2) The capacity reading is computed by the scorer from the files, not judged by hand:**
  - **CAPACITY** = FSQ-8 passes S5×2, **and** FSQ-8 enters S5×4 (no exhausted slot; accepted runs past 2k) and fails it, **and** a wide arm
    or FSQ-8 at 16 levels passes S5×4.
  - Otherwise: "difficulty" (FSQ-8 fails S5×2, or nothing passes S5×4), "entry" (FSQ-8 does not enter S5×4), or "no capacity limit
    shown".
* **(3) Refinement of 133-1 (a)** (helper: a 64-d contractive carrier reaches the floor in about 3 hand-offs, and the hand-off that
  lands there carries a ratio of 0.5–0.6). The continuous and soft arms' ρ is the geomean over the first 3 hand-offs, **excluding
  any whose preceding median distance is ≤ 1e-2** (as 123-9 defines it: this drops floor-to-floor ratios; a hand-off landing
  on the floor from above stays in and only understates contraction). If none qualify, it is ρ₁.
  - Tested: a carrier at 0.10 with a floor-landing third step scores 0.10, not about 0.18. RAW-H-style drift (distances 5.4 → 0.7
    → 0.4) stays separable.

**§133 stage-1 result (2026-10-07 07:44 EDT; cloud RTX 3090s, several drivers, recorded per slot; restart slots as §128; scored by Theory alone with `theory/score_s133.py`; the
interim notes 133-5 – 133-10). Four arms PASS stage 1 (three at d 64); FSQ-16 FAILS; SOFT-64 and PIN-64 were dropped at proof.**

| Arm | Width / correction | Slots exact + SERIAL + ρ ≤ 0.28 | ρ (fp32; 127-12 for FSQ, first-three form for continuous) | Verdict |
|---|---|---|---|---|
| FSQ-64 | 64, hard FSQ | 4/4 (s201, s2, s3, s104) | 0.24 / 0.19 / 0.21 / 0.15 | **PASS** |
| NOISE-64 | 64, continuous + dither | 4/4 (s201, s2, s3, s104) | 0.12 / 0.19 / 0.11 / 0.10 | **PASS** |
| CONT-64 | 64, continuous, no correction | 3/3 decided (s201, s2, s3; slot 4 pending, descriptive) | 0.22 / 0.16 / 0.14 | **PASS** |
| SOFT-16 | 16, soft pull β 0.1 | 4/4 | 0.13 / 0.11 / 0.11 / 0.11 | **PASS** |
| FSQ-16 | 16, hard FSQ | 2/4 (s2, s3 escape and fail) | — | **FAIL** |
| SOFT-64 | 64, soft pull β 0.1 | proof 0.646 at 2,048 (a constant 2·10⁻⁴ per-step error) | — | dropped |
| PIN-64 | 64, LayerNorm pin | proof 0.0 (F2b-transition) | — | dropped |

* **Reading (stage 1, as registered):** wide carriers stay exact at 2,048, SERIAL and contractive on S5.
  - This holds with hard rounding (FSQ-64), with dither (NOISE-64) and **even uncorrected** (CONT-64).
  - The earlier picture, "wide carriers drift", came from RAW-H's raw 768-d hidden state. The learned tanh-bounded code at d 64 does not drift.
* **Entry costs restarts at d 64:** every 64-d arm flagged seeds 1 and 101 (133-5). The accepted runs needed 0–2 restarts per slot, and the width changed the basin.
* **Capacity is not yet shown:** an 8-d code suffices on S5. Stage 2 (S5×4: CAP-PASS where FSQ-8 fails by capacity) needs Audit's re-approval and is proposed
  separately.
* **The user's trigger fires** ("if 64 wide latents pass, try full width"): §138 (768-d) is drafted from `docs/THEORY_FULLWIDTH_PLAN_2026-10-07.md` and goes to
  Literature and Audit before registration.

## 134. COPY-ANCHOR: a label-free state-reading auxiliary from step 0, against entry failure and escape without learning (pre-registered 2026-10-06 15:01 EDT, after Literature's check 4da2f672 / e0909d0b / ebc2a039 and before any implementation or run; the user's "invent and synthesize a stable latent method" (via Audit), candidate 1 of `docs/THEORY_STABLE_LATENT_2026-10-06.md`)

**Why.**
* §128's task-agnostic recipe needs restarts on S5. Its dominant failure is **escape without learning** (F2): the run passes the 2k
  flag, then never composes, and the code is read by the decoder but does not hold the state (127-9; §127).
* The strongest entry aids we have both make the model **read a given state**:
  - §124 COPY: acquisition at L5 in 4/4, serial in 2/4 against 0/4;
  - §111 READ: reliable hand-off 5/5, but with step labels.
* **Prior** (4da2f672 / e0909d0b / ebc2a039): autoencoding / copy auxiliaries against posterior collapse (Dai & Le 1511.01432; Bowman;
  He 1901.05534; Z-Forcing); NTM copy; LP-FT 2202.10054 (interface warm-up). New as scoped (a state-copy anchor through a quantised
  recurrent carrier of a pretrained LM).

**Design.** §128's S5 recipe exactly (fsq120 FINAL, FSQ d 8 / L 8, the length curriculum, 24k, bs 64, the 127-3 guard; cloud RTX 3090 /
driver 595.71 class, checkpoints), plus **copy rows**:
* A fraction **f = 0.1** of each batch's rows are copy rows. Each takes **the final-answer text of another problem in the batch**: a
  final-answer-only target, so task-agnostic and C4-compatible. On S5 it is a uniformly random state text.
* It runs that text through the carrier: [EOT, answer text] → segment → last hidden → quantised code → embedding.
* It decodes **from the code alone**, in a fresh [EOT, code] answer-decoder segment (as every answer is decoded; Audit / Literature:
  no bypass through the input text).
* The CE on the same text is added to the main loss with weight 1, from step 0 to the end.
* **Gradflow under CUDA bf16 before any run:** the copy loss reaches W_down and W_up (the §116 lesson).
* **No restarts and no plateau stop:** each seed is one 24k run, so the test is "stable to train without restarts".

**Seeds.** S5 seeds 1–4 on the RTX 3090 / 595.71 class, where the plain recipe gives:
* s1 exact;
* s2 plateau (§120 / §128);
* s3 escaped and failed (§128 slot 3; §127 ctrl s3);
* s4 plateau, a basin seed (127-13).
* So **plain is 1/4 without restarts.**

**Rules (fixed now).**
* **PASS:** all four seeds exact at 2,048 (≥ 0.99), SERIAL (donor ≥ 0.90, recipient − chance < 0.10) and contractive (127-12's ρ clause ≤
  0.28).
* **IMPROVES** (descriptive): 3/4. **FAIL:** ≤ 2/4.
* **Guard (only after PASS; 1 seed each):** FF / CYC / MOD / RDFA with copy rows must stay exact and SERIAL, and closed where §128
  required (CERT holds, Moore = minimal), or the copy anchor is not task-agnostic.

**Mechanism (logged in every run; Audit).**
* The probe at 2k and at the end: R²_state, R²_item, and op-history R² (last 1–3 ops).
* **Dialect probe** (Literature ebc2a039 (i)): for the same state, the FSQ-code identity rate (and cosine) between the **copy code** (from
  [EOT, text]) and the **transition code** (from [EOT, code, item]), at 2k and at the end, with copy accuracy logged over training.
  - Copy fixes the text ↔ code interface; composition must still come from the final-answer loss. If the two writers use different
    code regions ("two dialects"), the copy anchor does not transfer.
  - No copy → one-op → answer rows are added: random-start one-op problems would label transitions, which is more than FINAL.
* **f stays constant** (Literature (ii)): an interface anchor like §111's reader. Removing it after entry is the ANNEAL / α → 1 collapse
  pattern.
* **FF caveat:** FF's final answer is the read bit, so copy anchors only 2 codes there. The FF guard is expected to show little.
* **Pre-stated:** if F2 is the short-chain op-history shortcut (Literature 4da2f672), COPY seeds show R²_state above op-history R² at 2k,
  where plain escape-without-learning seeds show the reverse. The local probe of s128_s5_s3 against the learners measures the plain
  side.

**Readings.**
* **PASS (+ guard).** A label-free state-reading anchor makes the recipe stable to train on S5 without restarts. §128's recipe plus COPY
  becomes the default, and candidate 2 (the round-trip attractor) builds on it.
* **IMPROVES.** COPY helps but does not remove the failure. The mechanism probe says which failure survives.
* **FAIL.** Reading a copy of the state is not enough on S5. The failure lies in composition itself, and the probe reports what the
  code holds.
* **IMPROVES / FAIL with high copy accuracy but low dialect agreement:** "two dialects": the anchor fixed the interface, not the
  transition path. The next step is a registration that ties the two writers (e.g. the round-trip attractor, candidate 2).

**Placement.**
* Main: 4 runs, about $3.5. Guard: 4 runs, about $2.5. About $6 in all, staged (Audit).
* Implementation: the experiments session (fsq120 `--copy_frac 0.1`, copy rows as above; dirs `s134_copy_s5_s{1..4}`,
  `s134_copy_{ff,cyc,mod,rdfa}_s1`).
* Scorer: `theory/score_s134.py`, committed before any run.

**Note 133-3 (2026-10-06 15:03 EDT, before any §133 run; questions from the experiments session's implementation, 81e37687). Two clarifications,
no rule change.**
* **NOISE-64's dither is §123's, absolute:** U(±0.25) per coordinate on z_b at every training hand-off after 2k (`--dither 0.25`).
  §133's text said "U(±0.25·std_d)", which contradicts its own "§123's dither". That was a writing error, corrected here before any run.
* **SOFT-16 / SOFT-64 run at β = 0.1 from step 0,** train and test, with no p_hard ramp. This is intended: the soft snap is a constant
  weak pull. round() is straight-through as in FSQ, and the recorded codes are round(z_b).
* As implemented: §128's micro rule (micro 2 only at L ≥ 32, `--micro_from 32`); angular_drift at positions 1 / 16 / 64 / 256 / 512 for
  every arm; grid_share as §123.

**Note 134-1 (2026-10-06 15:03 EDT, before any §134 implementation or run; Audit). A second dialect test.** Beside the code-identity rate:
* **Copy-code donor injection:** in a transition chain, replace the carried code at position j with **the copy-written code of the same
  true state** (from [EOT, that state's answer text]), then score the chain's final answer.
* If the copy and transition writers share a dialect, accuracy matches the intact chain. If they split, it drops. Descriptive.
* A split means the copy loss trained an interface, not the carrier.

**Note 134-2 (2026-10-06 15:06 EDT, before any §134 run; descriptive predictions from the helper's mining; no rule change).** Escape without learning
shows only in the 2.6k → 4k window: loss(4k) / loss(2.6k) is 0.71 for learners against 0.997 for escape-fails (0/12 against 14/18 at 0.9).
Two predictions, fixed now:
* **(a)** If COPY fixes F2, every COPY seed's loss(4k) / loss(2.6k) is below 0.9, including s3 (plain 0.99) and s2 / s4 (plain plateau).
* **(b)** Its 4.5k loss is **≤ 3.0**. The statistic is the loss at the logged step nearest 4,500 (not the running minimum). It applies to
  hard-rounded carriers only (FSQ-8, as in §134), since continuous arms overlap both groups (dither010 s1 exact at 3.31; RAW-H / ANNEAL
  failures at 1.96–2.83). On that statistic, hard-rounded learners reach 0.37–2.75 (n 8) and failures 3.45 and up (n 11) (helper's
  corrected run lists, with Audit's recomputation agreeing). (Corrected 2026-10-06 15:09 EDT, before any §134 run; Audit's recomputation from raw logs: hard-rounded S5
  learners reach 0.37–2.75 (n 8) and failures 3.42 and up (n 16), so the gap is 2.75–3.42. The earlier "≤ 2.5" would have mis-scored 4 of
  the 8 known learners). The 4k / 2.6k ratio < 0.9 in (a) is the cleaner of the two.
* Both are read from log.jsonl, and are descriptive beside §134's PASS rule.

**Note 133-4 (2026-10-06 15:09 EDT, before any §133 result; the Audit helper's harness review of 81e37687; disclosure, no rule change).** SOFT's
forward pass is the registered mix, (1 − β)·z_b + β·round(z_b). Its gradient is straight-through with unit slope: round() passes the
gradient unchanged, as in FSQ. Everything else matches §133 / 133-1 / 133-3.

**Note 134-3 (2026-10-06 15:09 EDT, before any §134 run; the helper's review and post-hoc probes; no rule change).**
* **The mechanism prediction moves to the end-of-training probe,** where it can discriminate.
  - At 2k, plain escape-fails already show R²_state 0.25–0.44, with the current op gone (n 6). So "op-history above state at 2k" does not
    describe them.
  - The post-hoc probe of the kept escape-fail (s128 S5 s3) shows a **reversion:** R²_state 0.04 / −0.06 / −0.06 at L 2 / 4 / 8 by 24k
    (0.44 at 2k), with the current op back in the code at L 8 (0.83). The learners hold the state at every length (R² 0.98–1.00).
  - **Pre-stated now:** COPY seeds hold the state at the end at L 2 / 4 / 8 (R²_state ≥ 0.9) with no op reversion. A COPY seed that still
    escapes and fails is read against this.
* **On S5, op-history is read from last1 only.** The last2 / last3 groups (14,400 / 1.7M) are unreadable at n 4,096 (coverage 0.13 / 0.00),
  and are reported with their coverage.
* **Scorer** (`score_s134.py`, updated now):
  - smoke filter plus an args check (d_q 8, levels 8, family, copy_frac 0.1); a mismatch is an error row and an invalid run;
  - the dialect and copy-donor fields are printed;
  - a guard run without a CERT file reads "incomplete", not ok.

**Note 134-4 (2026-10-06 15:23 EDT, before any §134 run; the Audit helper's harness review of b693aa53, endorsed by Audit). Loss logging, copy weight and ramp pins.**
* **(a) "loss" logs the main chain only.** The harness logged the total (main mean + copy CE) as "loss". That makes the 123-3 flag
  reading, 134-2(b)'s 2.6k → 4k ratio and the 4.5k ≤ 3.0 statistic incomparable with every plain comparator.
  - **Fixed now:** each training log record has "loss" = the mean CE over the main rows only, as in the plain recipe.
  - "copy_loss" is logged separately, and **"loss_total"** (main + copy) is logged beside them. The 127-3 guard reads the total.
  - A §134 run whose log has copy_loss but no loss_total is from the old harness: an **error row (invalid)**, and its loss-based
    readings are not made.
* **(b) Copy weight, fixed now:** **L = CE_main + 1.0 · CE_copy.** Each term is the mean over its own rows: 58 main rows and 6 copy
  rows at bs 64.
  - The two terms have **equal total weight**, so a copy row's gradient weight is about 9.7× a main row's. This is the standard
    auxiliary-loss convention (λ = 1 on the mean, as for §118's anchor). Fast anchoring at entry is the point of the arm, and the copy CE
    falls quickly, so its gradient shrinks after entry whatever its weight.
  - The per-row-normalised alternative (copy weight 6/58 ≈ 0.10 on the mean) is a weaker anchor. It is not registered: it is a
    possible ablation after a PASS.
  - The weight is the same for every seed and guard run.
* **(b′) Disclosed:** copy rows replace main rows, so the main task sees 58 rows per step against 64 in the plain comparator (9 %
  fewer examples, a slightly noisier main gradient). That is as registered ("a fraction f of each batch's rows"). The direction of its
  effect on entry is not known, and it is stated beside the result.
* **(c) Ramp pins:** score_s134's args check now also requires **p_hard_start 2000 and p_hard_end 4000** (§120's ramp, recorded as
  p_hard_ramp), so a candidate-3 ramp setting cannot enter a §134 directory unnoticed. stop_plateau false is already required.

**Note 128-5 (2026-10-06 15:38 EDT; descriptive, extends 128-4's fp32 table; Audit and the helper's growth reading). fp32 rollouts find codes beyond two bf16 certificates.**
* The fp32 growth rollouts (2070, 500 × 2,048; eval_growth.json, all at acc 1.0) visit more distinct codes than the bf16 BFS set
  reaches in **cyc s3 (62 against 57)** and **rdfa s3 (154 against 151)**. That is 128-4's bf16 artefact class, in runs where the bf16
  eval had shown no gap.
* **Queued (cloud RTX 3090, gen-b61, after its S5 certificates; ~$0):** fp32 cert_cover for cyc s3 and rdfa s3. Also cyc s4 and mod
  s2 / s3 / s4: their fp32 growth counts are at or below their bf16 BFS counts, but only an fp32 certificate makes the bounded-growth
  reading certified.
* CLOSED is unchanged (the certificate as run, 128-4). The fp32 n_uncovered column is filled for these six as they land.

**Note 131-7 (2026-10-06 15:39 EDT; interim, descriptive; the endpoint is not reached; helper's reading, Audit-reproduced). The first prospective FLAG4 miss: a late re-entry.**
* **s130chk_s5_s17_a4000** (RTX A4000, driver 560.35.03, micro 1). Loss 8.343 / 4.502 / 3.666 at 100 / 1k / 2k: **no 2k flag**
  (ratio 0.814). Then 3.930 at 2.6k and 3.063 at 4k, so **FLAG4 false** (0.779). Then 3.453 at 4.5k and 4.098 after the L_max 8
  step (5.1k), ending at 4.105 at 24k. long_acc 0, ext_2048 0, n_codes 9,646, donor at chance.
  - So the run escapes at 2k, descends to about 3.0 by 4k, and **re-enters failure at the L_max 8 step**. A 4k check cannot see this
    by construction. It is a miss class distinct from the ~3.2 basin and the partial learner.
* **Parity (131-4):** it does not reproduce its 3070 replay. Both the replay and the crashed 3070 partial flagged at 2k (ratios 0.964
  / 0.966). |Δloss(1k)| = 0.709 with opposite flag outcomes, so seed 17's 2k outcome depends on the host within GA104 (the 125-2
  pattern). It is read on its own flags, and the "late learner" question for s17 stays unanswered.
* **score_s131 now:** held-out escaped-and-failed recall is 4/5 (all held-out and prospective); precision is 1.0 (0 of 22 / 9 exact
  flagged). (ii) still holds at ≥ 2/3. The verdict waits for the endpoint.
* **134-2(b):** statistic A at 4.5k is 3.453, on the failure side of ≤ 3.0, as predicted. The exact runs' maximum is 2.746.

**Note 131-8 (2026-10-06 16:09 EDT; interim, before the endpoint; the helper's and Audit's record-identity sweeps, re-run by Theory). A duplicate removed, cloud reruns read, and host disclosure.**
* **Duplicate (131-1 (b)'s rule, applied as written):** s128_s5_s1 (held-out) is record-for-record s120_s5_final_s1 (in-sample) for
  140 / 140 loss records to 14k. It diverges only from 14.1k, where the old micro rule engaged. So it is **counted once, as in-sample**
  (score_s131 SAME_TRAJECTORY).
  - This removes an unflagged held-out S5 learner, which is the conservative direction for (i).
  - Theory's sweep of every in-sample / held-out S5 pair finds no other held-out duplicate through 4.5k. In-sample,
    s123_s5_fsqdither_s1 equals s120_s5_final_s1 through 4k (the dither starts after 4k), so the two share one FLAG4 value. The
    in-sample figures are descriptive and are left as they are.
  - Runs sharing only the first 2k (same seed, recipe and card class) are expected, and their FLAG4 windows differ.
* **Cloud reruns read so far (131-4):**
  - s12_a4000 is **exact** (FLAG4 0.513; ext_2048 0.996).
  - s7_4060ti escaped and failed, and is **caught** (FLAG4 0.953).
  - s17_a4000 is the miss (131-7).
  - s6_4060ti is flagged at 2k (a plateau, not escaped), with long_acc 0 at 24k although the end probe reads the state (R²_state
    0.79): the first 24k answer to §130's "late learner" question on a flagged seed is no.
* **Parity:** within 0.1 at 1k and 2k, with a matching flag, in 1 of the 4 reruns so far (s6). Flags match in 2 / 4.
* **Disclosed:** 131-4's "one host per card class" was not met. The A4000 seeds ran on two hosts (drivers 560.35.03 / 595.71.05),
  and the 4060 Ti seeds on two (595.84 / 595.71.05). Each run is read on its own flags, as 131-4 allows. s7's non-parity is on a
  595.71.05 host, so the driver does not explain it.
* **score_s131 now:** held-out and prospective escaped-and-failed are flagged 5 / 6; exact runs flagged 0 (22 held-out, 9 prospective;
  the S5 learners 0 / 4). The verdict waits for the endpoint (the remaining s130chk and s131x reruns).

**Note 128-6 (2026-10-06 16:59 EDT; descriptive; completes 128-5, Audit-approved local run; helper-read, Theory-verified from the files). Every fp32 certificate covers the eval.**
* **Six fp32 cover certificates**, local RTX 2070 (Turing, fp32-native), driver 591.86, under the 6 GiB per-process cap; 1–2 s each, none
  capped or timed out. Dirs `boxes/runs_fsq/s128cc2070_*`.

| Run | bf16 reachable | fp32 reachable | fp32 eval distinct | fp32 n_uncovered | fp32 CERT |
|---|---|---|---|---|---|
| cyc s3 | 57 | 62 | 62 | **0** | holds, Moore 5 |
| cyc s4 | 171 | 135 | 135 | **0** | holds, Moore 5 |
| rdfa s3 | 151 | 154 | 154 | **0** | holds, Moore 8 |
| mod s2 | 407 | 423 | 393 | **0** | holds, Moore 5 |
| mod s3 | 226 | 229 | 214 | **0** | holds, Moore 5 |
| mod s4 | 511 | 505 | 465 | **0** | holds, Moore 5 |

* **With 128-4's five** (cyc s1 / s2, mod s1, rdfa s2 / s4 on the 3090), **11 of the 12 CLOSED runs** have an fp32 certificate that
  holds with Moore minimal and **n_uncovered 0**. The 13th and 14th certificates are 128-4's local 4060 cross-checks, also 0.
  - rdfa s1 is bf16-only. Its bf16 eval showed no gap, and its fp32 growth count (90 at 2,048) is below its bf16 reachable count (98).
* **bf16 certificates err both ways:** under by 3–32 (cyc s3, rdfa s3, mod s1 / s2 / s3) and over by 6 and 36 (mod s4, cyc s4). That
  is 128-4's artefact class in full.
* The fp32 sets equal the visited set in cyc s3 / s4 and rdfa s3. In MOD they are strict supersets (slack 4–40), so they bound MOD's
  still-rising fp32 growth.
* CLOSED is unchanged (the certificate as run, 128-4). The stronger statement holds for 11 of 12: **the fp32 certificate covers every
  code the eval visits.**

**§134 result (2026-10-06 22:17 EDT; 4 runs on RTX 3090 / 595.71.05 by the experiments session, one 24k run per seed with no restart and no plateau stop;
scored by Theory alone with `theory/score_s134.py`, committed before any run; Audit-reproduced). COPY-ANCHOR is PASS on S5: 4/4 seeds exact at
2,048, SERIAL and contractive, against 1/4 for the plain recipe on the same seeds and card class.**

| Seed | Plain recipe (same seed, 3090) | long_acc | ext_2048 | SERIAL | fp32 ρ (127-12) | 4k / 2.6k | loss 4.5k | first loss ≤ 0.01 |
|---|---|---|---|---|---|---|---|---|
| s1 | exact | 1.000 | 0.998 | yes | 0.138 | 0.62 | 1.06 | 14.4k |
| s2 | plateau | 0.999 | 0.994 | yes | 0.142 | 0.32 | 0.60 | 11.0k |
| s3 | escape without learning | 1.000 | 0.998 | yes | 0.184 | 0.41 | 0.92 | 12.8k |
| s4 | basin (ctrl s4 full null) | 0.998 | 0.998 | yes | 0.220 | 0.80 | 1.58 | 19.0k |

* **Mechanism** (134-1 / 134-3; all four seeds):
  - The end code holds the state at L 2 / 4 / 8 (R²_state 0.995–0.998), with no op reversion (last-op R² < 0). The prefix accuracy at 1 / 2 / 4 /
    8 is 1.0, and copy_acc is 1.0.
  - **Dialects:** the copy-written and transition-written codes are never identical (identity_rate 0.0) but close (cos 0.85–0.90).
  - A copy-written code injected at position 2 / 4 / 8 is continued correctly (0.980–1.000; intact 1.0). At 2k the same injection is at
    chance (0.006–0.013) while copy_acc is already 0.88–0.98. The anchor first fixes the decoder's reading; the transition writer then moves
    into the copy region.
  - Codes per state 14–29, as for the plain learners.
* **134-2(b)'s descriptive predictions hold in 4/4:** the 4k / 2.6k ratio is < 0.9, and the loss at 4.5k is ≤ 3.0.
* **Disclosed:**
  - the main task saw 58 rows per step, against 64 in the plain comparator (134-4 b′);
  - s2's fp32 contraction was computed on the local RTX 3070 (fp32 is device-independent here, 128-4); the others on their hosts;
  - all four trained with micro 2 beyond L 16.
* **Reading (as registered):** PASS on S5. **A label-free state-reading anchor (final-answer texts only, so C4-compatible) makes §128's recipe
  stable to train on S5 without restarts.** It turns all three failure types (plateau, escape without learning, basin) into exact learners.
  - §134 registered "PASS (+ guard)" for adoption as the default. **The guard stage (FF / CYC / MOD / RDFA, one seed each, ~$2.5) is approved and
    runs next.** The "default recipe" reading waits for it.

**§134 guard result (2026-10-07 04:48 EDT; FF / CYC / MOD / RDFA, one seed each with COPY rows, RTX 3090 / 595.71.05, trained --no_cert, fp32 cover certificates post hoc as 128-6;
scored by Theory with score_s134). The guard FAILS on MOD: COPY keeps every family exact and SERIAL, but MOD's carrier is no longer closed.**

| Family | long_acc | ext_2048 | SERIAL | fp32 CERT | reachable codes (fp32 BFS) | Moore classes | §128 plain (same family) |
|---|---|---|---|---|---|---|---|
| FF | 1.0 | n/a | yes | n/a (not required) | — | — | — |
| CYC | 1.0 | 1.0 | yes | **holds** | 38 | 5 | holds |
| RDFA | 1.0 | 1.0 | yes | **holds** | 218 | 8 | holds |
| MOD | 1.0 | 1.0 | yes | **does not hold** (185,903 violations) | **70,012** | 34,725 | holds (s1: 386, Moore 5) |

* **MOD with COPY.**
  - Random test chains visit only 560–572 codes, and all are correct (long_acc 1.0, ext 1.0).
  - The exhaustive BFS over every op sequence from the start explodes from depth 10 (frontier 83 → 13,990 at depth 16) into a transient region of about 70k
    codes, and dies out by depth 32. Along those paths the decoded answers are inconsistent with a 5-state machine.
  - So particular op sequences of about 10–20 steps, which random chains almost never sample, lead the MOD + COPY carrier astray. Plain §128 MOD is closed
    on all of them.
* **Reading (as registered):** the guard requires closure where §128 required it. MOD fails, so **the copy anchor is not task-agnostic in §134's sense.**
  - **§128's recipe + COPY is not adopted as the default.**
  - What stands: §134's S5 result (4/4 exact without restarts, a fix for entry on S5), and COPY's exactness and SERIAL on all five families. The cost
    is closure on MOD.
  - Descriptive: COPY's final-answer copy rows add a second writer whose code region the transition writer moves into (134-5). On MOD this seems to
    leave the transition map defined only near the visited codes. That is a hypothesis, not tested.
* **Consequence for §139 (COPY base):** multiplication has no certificate, so a closure loss there would not show in §139's rules. That limitation is
  stated beside §139's result.

## 135. TEACHER-STATE DISTILLATION: a text-CoT model's hidden states, not its text, as per-hand-off targets for the latent carrier (pre-registered 2026-10-06 17:07 EDT, after Literature's check 0bb148e9 and before any implementation or run; the user's idea "natural language CoT is not really a good training artifact for latent reasoning; can we distill a CoT model into a latent model?" (via Audit), candidate 4 of `docs/THEORY_STABLE_LATENT_2026-10-06.md`)

**Why.**
* The user's point: a CoT model's **internal state**, not its text, should be the training artifact for a latent reasoner.
* Our latent carrier is exact once entered (§128). It fails at entry on S5, and the failures are mostly **transition-side** (synthesis:
  F2a / F2b-transition / dialect in 7 of 10 failures). It also hits the k = 4 wall on multiplication (§98 GEN-M 0.163 / 0.005, where
  text CoT is 1.0).
* **Prior** (0bb148e9, `docs/LIT_COT_DISTILL_2026-10-06.md`): ICoT-KD 2311.01460 (hidden states across layers); KaVa 2510.02312 (stepwise
  KV distillation, robust to natural-language traces); CODI 2502.21074 (one token); SIM-CoT; Distill S2→S1 2407.06023; System-1.5
  2505.18962; LIFT 2609.38149; geometric imitation collapses latents, contrastive targets preserve them (2606.20075).
  - New as scoped: a per-hand-off target through a discrete Markov carrier, scored by certificate, donor and length.

**Teacher (frozen; gate first).**
* GPT-2 small, pretrained, trained on S5 text CoT in TC's format: the start state text, then each op followed by the state text
  " > A B C D E", then the answer. L ~ U(1, 32), 8k steps, bs 32, lr 1e-4, cloud RTX 3090. Dir `boxes/runs_fsq/s135_teacher_s5`.
* **Gate A1:** greedy final-answer accuracy ≥ 0.99 at L = 32 on 1,000 held-out problems.
* **Gate A2 (teacher probe; Literature (d)):** a linear 120-way state probe on the hidden state at **the last token of each step's
  state text**, at every layer, on held-out problems. The layer with the highest probe accuracy is chosen (ties → the later layer),
  fixed before any student run. It needs probe accuracy ≥ 0.95. The curve is reported.
  - The probe uses state labels on the teacher side only. That is disclosed, and it is part of why §135 is not C4.
* If A1 or A2 fails, no student run happens.

**Student.** §128's S5 recipe exactly (fsq120 FINAL, FSQ d 8 / L 8, the length curriculum, p_hard 2000 → 4000, 24k, bs 64, the 127-3
guard; cloud RTX 3090 / 595.71 class, checkpoints), with **no restarts and no plateau stop**, as in §134. Plus one auxiliary loss:
* At every hand-off j of a training chain: s_j = normalise(P_s(e_j)), where e_j = W_up(code_j) is the carried embedding.
  t_j = normalise(P_t(sg(target_j))). P_s and P_t are learned linear maps 768 → 256 into one shared space: student → shared,
  teacher → shared, **one P_s and one P_t shared across all hand-offs**.
* **InfoNCE** (τ 0.1): s_j against t_j, with every other (row, hand-off) target in the batch as a negative. Negatives with
  cos(target, target_j) > 0.99 are masked: a label-free near-duplicate mask (Literature (b) / (c): a contrastive target, so the
  8-d code does not spend capacity on the teacher's token or format nuisance, and no regression residual leaks into the encoder).
* **Weight 1, constant from step 0** (Literature 4; §126). Inference never uses the teacher.
* **Arms (one change: the target's source):**
  - **TSD:** target_j = the teacher's hidden state at the chosen layer, at the last token of step j's state text, for the same problem
    (no_grad).
  - **TEXT (control, the same per-step information as text; the same step boundary: step j's gold state text in TC's format):** target_j = the student's own text-path embedding of the gold state text
    j, W_up(copy_write(state text j)) (§134's writer, sg). This is §118 ANCHOR's information, in InfoNCE form.
* **Gradflow under CUDA bf16 before any run** (the Audit helper; the §116 lesson):
  - the auxiliary loss must give a nonzero gradient into the **carrier's encoder**: W_down and the GPT-2 trunk that writes z_b,
    through the FSQ straight-through path. Reaching W_up, P_s and P_t alone would leave the code's content unchanged;
  - **no** gradient may reach the teacher.

**Seeds.** S5 seeds 1–4 on the 3090 / 595.71 class, where the plain recipe without restarts is 1/4 (§134's comparator).

**Rules (fixed now; per arm).**
* **PASS:** all four seeds exact at 2,048 (ext_2048 ≥ 0.99), SERIAL (donor ≥ 0.90, recipient − chance < 0.10) and contractive (127-12's
  ρ clause ≤ 0.28). **IMPROVES** (descriptive): 3/4. **FAIL:** ≤ 2/4.
* **The user's claim on S5 (teacher internals against step text):**
  - TSD PASS and TEXT not: the teacher's hidden state helps entry beyond the step text;
  - both PASS: S5 does not separate them (expected: per-step supervision solves S5, §111 / §118);
  - TEXT PASS and TSD not: the text target is better here.
* **Stage B (multiplication; registered after stage A reads, with Audit's approval).** Teachers are §98's kept TC 4×4 models
  (`boxes/runs_mult/m98_tc_4x4_s*`, acc_test 1.0, $0), in §98's latent harness. The intended rule is exact ≥ 0.99 on §98's 4×4 test
  items in ≥ 3/4 seeds with SERIAL, against ICoT-SI 0.998 and GEN-M 0.163. B decides the user's claim where latent methods fail.
* **Claim wording (fixed):** "latent at inference, distilled from a CoT model's hidden states: C1 / C6 / C2 with per-step supervision
  by proxy (not C4)".

**Scope by failure class (the Audit helper).** TSD and TEXT supervise the **write** side, the code's content at each hand-off. That
bears directly on F2a and F2b-transition, and on F2b-compounding through a cleaner target. F2b-dialect (state present and read but
mapped wrong) is not addressed by a write-side target alone; there the answer CE carries the reader. So the prefix profile
(positions 1 and 2) is reported per seed beside PASS.

**Mechanism (descriptive, every run).** Prefix accuracy at 1 / 2 / 4 / 8, the end probe (R²_state by L, abl_dloss), the fp32 ρ₁, and
the InfoNCE top-1 retrieval accuracy over training (the student's code picks out its own teacher step).

**Placement.** Teacher: 1 run (about 1 h, about $0.3). Students: 8 runs (about $7). About $7.5 in all, cloud only, with Audit's approval.
Implementation: the experiments session (fsq120 `--tsd {teacher,text}`, `--tsd_teacher <dir>`, `--tsd_layer`, `--tsd_w 1.0`,
`--tsd_tau 0.1`). Dirs `s135_{tsd,text}_s5_s{1..4}`. Scorer: `theory/score_s135.py`, committed before any run.

**Note 135-1 (2026-10-06 17:11 EDT, before any §135 implementation or run; the Audit helper's read of f1899abd). The control becomes stationary, targets are centred, and the scorer's details are named.**
* **(c) The control changes from TEXT to BASE.** TEXT's target, W_up(copy_write(state text)), runs through the student's own trunk,
  which the answer CE keeps moving. TSD's target is frozen. A TSD-over-TEXT result would therefore mix target stationarity with the
  information source.
  - **BASE (replaces TEXT):** target_j = the **pretrained, not fine-tuned, GPT-2's** hidden state at the same chosen layer and the
    same token (the last token of step j's state text), on **the same CoT text** the teacher reads. It is frozen, with no_grad.
  - TSD and BASE now differ only in the encoder's weights: CoT-fine-tuned against pretrained. A TSD-over-BASE result reads as "the
    CoT model's internals help, beyond a generic encoding of the same text". That is the user's claim.
  - Dirs `s135_base_s5_s{1..4}`, args `--tsd base`. §118 ANCHOR (the student's own text path, 5/5 in its harness) remains the
    known positive reference; it is not rerun.
  - Same budget: two arms × 4 seeds.
* **(d) Centring and the mask.** GPT-2 hidden states are anisotropic, so a raw-cosine mask could remove almost no negatives at one
  layer and most of the batch at another.
  - Both arms' targets are centred, with a mean fixed at gate-A2 time from the held-out problems (the teacher's mean for TSD,
    pretrained GPT-2's for BASE), before the mask and before P_t.
  - The mask is centred cos > 0.99. The masked share of negatives is logged per batch and reported as `s135.masked_share`.
    - **Expected:** about the same-state share of pairs, about 1/120 (under 0.01).
    - A share above 0.05 in either arm is reported as a mask failure (descriptive).
* **(a) Image pin:** §135 runs must come from the current fsq120 (args carry p_hard_start / p_hard_end and the version fields). Args
  missing those keys mean an old image, which is an error row in score_s135.
* **(b) ρ:** the ρ in PASS is 127-12's clause as implemented in score_s134: the geometric mean of ρ_k over the hand-offs with
  dist_med > 1e-3 and ρ > 0, else ρ₁. It is not 133-1's first-three form, since §135's carrier is rounded.
* score_s135.py is updated to match: arms tsd / base, the masked share, and the claim wording.

**Note 135-2 (2026-10-06 17:13 EDT, before the teacher run is read; the Audit helper's read of boxes/teacher135.py, 547ba6e1 / 6f78d8a2). Teacher details fixed, and one equivalence check.**
* **Teacher schedule (unregistered until now, fixed here):** AdamW, lr 1e-4, warmup 100, cosine decay. CE on the state texts and the answer
  only; ops are inputs.
* **Gate A2 and the centring means are computed at L 8** (1,024 held-out problems, seed 13502), while students train to L_max 32. The
  layer and the means are therefore chosen at L 8.
  - "Probe accuracy" is the held-out-half accuracy of a linear probe fitted with 300 Adam steps (lr 1e-2, standardised features), not
    a closed-form value.
* **Equivalence check (before gate A2 is read):** `hidden_layers` re-implements GPT-2's forward and is the only path to the targets and
  the probe. A CPU fp32 smoke must assert that hidden_layers(ids)[-1] equals the model's own final hidden state on one batch (to about
  1e-4). A mismatch voids the gates until it is fixed.

**Note 128-7 (2026-10-06 17:14 EDT; descriptive; completes 128-6).** rdfa s1's fp32 cover certificate (local 2070, `s128cc2070_rdfa_s1`) holds:
90 reachable, Moore 8, eval distinct 90, **n_uncovered 0**. Its bf16 certificate had 98, an over-count by 8. **All 12 CLOSED runs now
carry an fp32 certificate that covers every code the eval visits** (14 certificates, counting 128-4's two 4060 duplicates).
* The visited set equals the fp32-reachable set where code growth saturates (cyc s3 / s4, rdfa s1 / s3). It is a strict subset where
  growth is still open (MOD s1–s4).

**Note 135-3 (2026-10-06 17:18 EDT, before any §135 student run; disclosure, no rule change).**
* The equivalence check passes exactly (`theory/results/teacher135_equiv.json`: max |Δ| 0.0 in CPU fp32).
* CPU gradflow passes: the auxiliary term reaches W_down and the trunk at p_hard 0 and 1, and no gradient reaches the target model.
  CUDA bf16 is run before launch.
* **Disclosed:** InfoNCE negatives are drawn **within the micro-batch**, so the negative set shrinks wherever `--micro` > 1 applies
  (L > 16). The registration's "every other target in the batch" holds at micro 1. The micro factor per run is recorded (micro_gate)
  and reported beside the result.

**Note 135-4 (2026-10-06 17:19 EDT, before any §135 student run; the Audit helper's read of 5387d85e). The duplicate mask uses gold-state identity, and CUDA gradflow is filed.**
* **The risk.** The centred-cosine mask removes same-state negatives only if the target model's hidden state for a repeated state is
  context-invariant to within cos 0.99. It probably is not: the hidden state of a state text depends on its position and history, and
  more so for BASE's pretrained GPT-2.
  - Unmasked same-state targets are false negatives (about 1/120 of pairs). The InfoNCE gradient then pushes the student's code to
    **separate same-state chains by their history**: anti-Markov, the F4 open-growth pattern. The two arms would also be masked to
    different degrees.
* **Fixed now, for both arms:** a negative is masked when its **gold intermediate state equals the positive's** (lab_i == lab_j).
  - This uses per-step state identity, which is inside §135's declared scope ("per-step supervision by proxy, not C4"; the teacher
    and its probe already use these labels). The arms stay identical in masking.
  - The centred-cosine share is still logged, as descriptive (`s135.masked_share_cos`), beside the label-mask share
    (`s135.masked_share`, expected about 1/120).
* **Gradflow under CUDA bf16** (registered before any run): the json is committed under theory/results before the first student
  starts. It gives the norms on W_down, the layer-3 attention weight, W_up, P_s and P_t at p_hard 0 and 1, and the target model's no-grad
  check.

**Note 128-8 (2026-10-06 17:42 EDT; descriptive; S5 fp32 certificates; the experiments session disclosed the confound, the helper read the files, Theory verified).**
* `s128cert_s5_{s1,s102,s104}` (cloud RTX 3090 b61, driver 570.211.01, cap 10⁶, 2,700 s) all read **inconclusive by time**.
  - **Confound:** two queue runners ran each certificate concurrently on the one GPU, at about half throughput. The files parse as
    single complete JSONs.
  - The clean single-runner point is the local 2070's s1 pass, also inconclusive (173,486 partial). The 3090 s1 agrees with it to 1 %.
* **Depth-complete lower bounds** on the fp32 BFS-reachable code set: s1 **164,748** (depth 6), s102 **157,618** (depth 5), s104 **99,853**
  (depth 5). Partials are 175,535 / 349,249 / 380,394, with frontiers still multiplying at the cut.
* So the S5 learners' reachable code sets (over all start states and op sequences) are at least about 10⁵, against 2.3–2.9 × 10³ distinct
  codes visited by 500 eval rollouts of 2,048 steps. The S5 carrier spreads each state over many codes and stays exact (§128 S5 learners:
  ext_2048 1.0).
* S5 certification is out of reach at this budget, by construction. Further S5 certificates are not run. FAMILY-PASS is unaffected
  (CERT was not required for S5).

**Note 135-5 (2026-10-06 19:46 EDT, before any §135 student run; read by Theory with score_s135's teacher gate; helper-read). The teacher passes both gates.**
* `s135_teacher_s5`: cloud RTX 3090, driver 590.48.01, torch 2.5.1+cu124; 8k steps, bs 32, lr 1e-4, L ~ U(1, 32), seed 1.
* **A1:** acc_L32 **1.000** on 1,000 held-out problems at L 32, with every state text and the answer generated (step_acc_L32 1.000).
* **A2:** held-out probe accuracy by layer, at L 8 with n 1,024, read at the last token of each step's state text: 0.039 (embeddings, ≈ 1/24,
  as the last letter alone allows), 0.80, 0.97, 0.9998, then 1.000 at layers 4–7 and 10.
  - The registered tie-break (the later layer) fixes **chosen_layer 10**. It was fixed before any data, and both arms use it. The
    earliest saturated layer is 4; that is noted, not a change.
* **Centring means** (layer 10, L 8, seed 13502): TSD norm 144.6, BASE norm 127.9, cosine between the means 0.57.
* The students may start (after §133 / §134 launch, per Audit). The teacher's json files are committed before the first student.

**Note 131-9 (2026-10-06 19:53 EDT; descriptive; findings from the Audit helper's archive (session closed) that no committed note held; Theory re-read each from the raw files).**
* **The one "flagged and learning" candidate is unreproduced.** The stopped local 3070 partial of s30 (`s130chk_s5_s30_3070.void_stopped`,
  driver 591.86, micro 4) flagged at 2k (4.644 → 4.501), then fell to 3.40 at 4k, 2.24 at 5k, 1.53 at 8k and **0.429 at 9k**, inside the
  exact runs' 9k range. It was stopped at 10.7k.
  - Its A4000 rerun is a 4-code null to 24k (131-8). The void partial enters no count. So §130's "late learner" wording has no
    reproduced example.
* **s6_4060ti's late re-organisation** (2k-flagged): loss 4.99 / 4.80 / 4.81 at 2k / 9k / 14k, then 3.51 at 20k and 3.36 at 24k. The end
  code reads the state (R² 0.79, abl 6.05) with long_acc 0: the F2b-dialect profile.
* **The 2.1k jump scales with depth at 2k:** s12_a4000 rose 1.939 → 3.673 (+1.73) at the L_max 4 step, against +0.4–0.5 in runs that sat at
  3.8–4.1 at 2k.
* **Replay-side pull rule** (§130 data, descriptive): dmax at 2k below 2.0 agrees with the 2k flag in 19 of 20 replay_x replays. The miss is
  3070_s517: dmax 1.84, unflagged at ratio 0.874.
* **§132:** slot 4's BFS-reachable code sets are smaller than the pilot's on the same sampled tables (13 / 12 / 8 / 12 / 16 / 16 against 38 / 22
  / 16 / 42 / 28 / 17). The passing slots solved at 3,000–7,700 steps, against the pilot's 19,900.

**Note 133-5 (2026-10-06 19:54 EDT; interim, descriptive; the first two stage-1 proof cells, read by Theory from the raw files; Audit-reproduced). FSQ-64 and NOISE-64 are exact at 2,048 on their proof slot; ρ waits for fp32.**
* **Slot 1 of both arms:** seeds 1 and 101 flagged at 2k, and seed 201 was accepted (cloud RTX 3090; FSQ-64 on driver 570.195.03, NOISE-64 on
  595.84; torch 2.5.1+cu124, micro 2).
  - FSQ-64: long_acc 0.998, ext_2048 0.998, SERIAL (donor 1.0, recipient − chance 0.000), n_codes 437,698 (codes_per_state 3,650: almost
    one code per visit, yet exact), grid_share 0.188.
  - NOISE-64: 0.998 / 0.998, SERIAL, grid_share 0.184.
  - **Both clear the proof criterion** (≥ 0.95 at 2,048, 133-2), so slots 2–4 run (Audit approved about $6).
* **ρ (for PASS) waits for eval_contraction_fp32.json.** The in-eval bf16 numbers: FSQ-64 ρ_k 0.176 / 0.388 / 0 (127-12's clause ≈ 0.26).
  NOISE-64 ρ_k 0.078 / 0.386 / 0.756, with the distance flattening at about 0.04 (rho_k3 ≈ 0.28, at the bound). bf16 is not the registered
  measure.
* **Scorer note:** the two flagged attempts' run dirs were not fetched. score_s133 reads their "plateau" status from the host's slot record
  (`s133_slots/{arm}_k1/eval.json`) when the raw log is missing, and marks it as such. The raw logs are requested.
* **Descriptive:**
  - (a) **Width changes the entry basin.** The two arms are identical to 2k (both unrounded and undithered there; the losses agree to
    three decimals). Seed 1 flags at d 64 but learns at d 8 on the 3090 (§128). The entry plateau depends on the code width, not on
    rounding.
  - (b) **The first FLAG4 false positive on record.** FSQ-64 s201 is flagged by FLAG4 (4k / 2.6k = 0.930), and its 4.5k loss is 3.52 (above
    134-2(b)'s 3.0). It then learns: 0.008 at 14k, exact.
    - s133_ runs are outside §131's registered set, so §131 is unaffected.
    - FLAG4 and the 4.5k cut are calibrated on the 8-d carrier. Any use as a restart trigger must be per carrier width.

**Note 133-6 (2026-10-06 20:07 EDT; interim; completes 133-5's proof slots with the registered fp32 ρ; Theory with score_s133).** Both proof slots are **contractive**: slot 1 passes in FSQ-64 and in NOISE-64.
* **FSQ-64 s201** (rounded; 127-12's clause on detail.eps 0.1): ρ_k 0.166 / 0.344 / 0, giving **0.239 ≤ 0.28**.
* **NOISE-64 s201** (continuous; 133-1 (a) / 133-2 rho_k3): ρ_k 0.068 / 0.163 / 0.170, giving **0.124**.
  - In fp32 the perturbation keeps shrinking geometrically, by about 0.17 per hand-off: dist_med 1.66 → 0.11 → 0.018 → 0.003 → 5e-4 →
    1e-4.
  - **133-5's bf16 "flattening at about 0.04" was a bf16 floor**, not the carrier.
  - A 64-d **continuous** carrier trained with dither is an attractor with genuine contraction (descriptive, one slot).
* Both slots are exact at 2,048, SERIAL and contractive, so 1 of 4 slots passes in each arm. PASS(arm) needs 3/4. Slots 2–4 are running.
* Fields named: the files' top-level "rho" (2.2e-5 for FSQ-64) is the vacuous hard-rounded summary, not the registered measure.

**Note 134-5 (2026-10-06 20:31 EDT; interim, 3 of 4 seeds; scored by Theory with score_s134; Audit-reproduced). COPY-ANCHOR turns all three plain-failure seeds into exact learners, with no restarts.**
* **s2** (plain: plateau), **s3** (plain: escape without learning), **s4** (plain: basin, the ctrl s4 full null). All on RTX 3090 /
  595.71.05, one 24k run each, no restart or plateau stop.
  - long_acc 0.999 / 1.000 / 0.998, ext_2048 **0.994 / 0.998 / 0.998**.
  - SERIAL (donor ≥ 0.9995, recipient − chance 0). acc_by_length 1 / 2 / 4 / 8 = 1.0.
  - fp32 ρ (127-12's clause): s3 **0.184**, s4 **0.220**; s2 pending.
  - Each run first reaches loss ≤ 0.01 at 11.0k / 12.8k / 19.0k.
* **134-2(b)'s predictions hold for all three:** the 4k / 2.6k ratio is 0.32 / 0.41 / 0.80 (< 0.9), and the loss at 4.5k is 0.60 / 0.92 / 1.58
  (≤ 3.0).
* **134-3's mechanism prediction holds:** at the end the code holds the state at L 2 / 4 / 8 (R²_state 0.995–0.996), with no op reversion (last-op
  R² < 0).
* **Dialect and copy-donor (134-1):**
  - **At 2k** the decoder already reads copy-written codes (copy_acc 0.88–0.98). But the two writers are far apart: dialect cos 0.22–0.49,
    and a copy-written code injected into a transition chain is continued at **chance** (copy-donor 0.006–0.013).
  - **At the end** the codes are still never identical (identity_rate 0.0 at every position), but they are close (cos 0.85–0.90). A
    copy-written code injected at position 2 / 4 / 8 **is continued correctly**: s2 0.999 / 1.000 / 0.998, s3 0.997 / 1.000 / 0.999,
    s4 0.989 / 0.989 / 0.980, with intact chains 1.0.
  - So there are two distinct but **functionally interchangeable** dialects (many codes per state, §128 / 128-8). The transition writer
    moved into the region the copy anchor defined. The "two dialects" failure did not occur.
* **Pending for the verdict** (PASS = 4/4 exact + SERIAL + ρ ≤ 0.28): s1 (training) and s2's fp32 ρ. The guard stage needs Audit's approval
  after a PASS.

**Note 133-7 (2026-10-06 21:10 EDT; interim; three more stage-1 proof slots, scored by Theory with score_s133; Audit-read). FSQ-16 and SOFT-16 clear the proof; SOFT-64 is dropped at proof.**
* **FSQ-16 s1** (first attempt; driver 550.144.03): long_acc 0.999, ext_2048 **0.996**, SERIAL. fp32 ρ_k 0.249 / 0.112 / 0, giving 127-12's
  clause **0.167**. The slot passes.
* **SOFT-16 s1** (first attempt; 590.48.01): long_acc 1.000 (1.0 at every length to 512), ext_2048 **0.998**, SERIAL. fp32 ρ is pending, so the
  slot is not yet decided.
* **SOFT-64** (slot 1: s1 flagged, s101 accepted; 595.84): 0.902 at 512 and **0.646 at 2,048**, SERIAL. **Dropped at proof** (133-2: s1 < 0.95
  at 2,048), with no further seeds and no variants.
  - **Mechanism (descriptive): a constant per-step error rate, not compounding drift.**
    - Its fp32 ρ_k (0.157 / 0.358 / 0.360 / 0.361 / 0.32) is below 1 throughout: perturbations shrink about 0.36 per hand-off.
      rho_k3 is 0.27, inside the bound.
    - acc_by_length fits a constant per-hand-off survival of 0.99980 (fit at 512): predicted 0.987 / 0.975 / 0.950 / 0.662 at 64 / 128 /
      256 / 2,048, observed 0.986 / 0.971 / 0.946 / 0.646.
    - So about 2·10⁻⁴ of hand-offs land in the wrong basin, whatever the chain length. Exactness at 2,048 needs a per-step survival of
      at least about 0.999995.
    - **ρ in band does not imply exactness:** the weak soft pull (β 0.1) contracts, but its basins are not separated finely enough at
      d 64.
* **So far:** FSQ-16, FSQ-64 and NOISE-64 each have their proof slot passing. SOFT-16 is exact, with ρ pending. SOFT-64 is dropped.
  CONT-64 and PIN-64 have not landed. Slots 2–4 are approved for the arms that clear their proof.

**Note 133-8 (2026-10-06 21:29 EDT; interim; two more stage-1 proof slots, scored by Theory with score_s133). The CONT-64 control is exact on its proof slot; PIN-64 is dropped.**
* The flagged attempts now have their raw args and logs under plain names, so the 2k flags are scored from the logs. The slot-file fallback
  (133-5) is no longer used for them.
* **CONT-64** (continuous 64-d, no correction; slot 1: s1 and s101 flagged, s201 accepted; driver 610.43):
  - long_acc 0.997, acc_by_length 0.9995 from 4 to 128, ext_2048 **0.990** (the exact bound itself), SERIAL (donor 1.0, recipient − chance 0).
  - fp32 rho_k3 = geomean(0.114, 0.301, 0.308) = **0.219**, so the slot passes. Later hand-offs keep shrinking at about 0.31 (dist down to 2e-4
    by hand-off 8).
  - **So, on this slot, a plain 64-d continuous carrier on §128's recipe does not drift**, unlike RAW-H at 768-d (§126). Width, not
    correction alone, may be the variable. The arm verdict needs slots 2–4.
* **PIN-64** (LayerNorm pin; slot 1: s1 flagged, s101 accepted; driver 550.144.03): long_acc 0.0, ext_2048 0.0, not SERIAL. **Dropped at
  proof** (133-2).
  - Descriptive: acc_by_length 1.0 at position 1 but **0.055 at position 2**. The F2b-transition profile: the code written from START
    is read, but the transition cannot consume a pinned carried code. FLAG4 0.967 (caught). Loss 3.22 at 24k.
* **Stage-1 tally so far (proof slots):**
  - passing: FSQ-16 0.167, FSQ-64 0.239, NOISE-64 0.124, CONT-64 0.219;
  - SOFT-16 exact, with ρ pending;
  - dropped: SOFT-64 (constant per-step error, 133-7) and PIN-64 (transition).
  - Slots 2–4 are running for the passing arms.

**Note 135-6 (2026-10-06 21:33 EDT; before any rerun and before the cause is known; a ruling stated in advance). s135_base_s5_s2 broke at step 14,002 on b78.**
* **What happened** (the experiments session; log and args committed in 4e31863a): healthy to 14k (loss 4.41, retrieval top-1 0.71, masked share
  0.0083).
  - From 14,002, the first step at L_max 32 and with micro-batching, the nonfinite-gradient guard fired on 3,198 of about 3,400 steps.
  - The logged label-mask share rose to about 1.5·10⁸, and a device-side index assert appeared inside teacher_targets at 17,425. No
    eval.json.
  - The other seven students had no nonfinite steps; three of them were already past 14k.
* **Evidence:** the masked share is computed from **integer gold labels only** (tsd_loss135: (lab_i == lab_j) & ~eye summed over N(N − 1), at
  most 2 by construction). A value of 10⁸ cannot arise from training dynamics, which points to a device or CUDA-context fault.
  teacher_targets' indices are data-only as well.
* **Ruling, fixed now:**
  - (1) **Host fault** (the CPU repro across 14k is clean and b78 fails a deterministic GPU check): the run is void (kept as
    `.void_hostfault`, read for nothing) and rerun from scratch on a different 3090 host, as for 131-4's crashed runs. "No restarts"
    governs plateau restarts, not host faults.
  - (2) **Harness bug** at the L 32 / micro switch: it is fixed, and every affected student resumes from its last checkpoint before
    14k. Students already past 14k are read as they are if the bug cannot have touched them; otherwise they rerun.
  - (3) **If the rerun on a healthy host also breaks** at the switch, BASE s2 is scored as a failure.

**Note 135-7 (2026-10-06 21:46 EDT; applies 135-6).** **The b78 host fault is confirmed** (`s135_base_s5_s2.void_hostfault/hostcheck_b78.txt`, 6e0d0de0).
* A 16 GiB pattern test fails on every pass, with 108–165 bad words per 256 MiB block, always in blocks 27–56 (about 6.75–14 GiB). Blocks 0–26
  are clean.
* The low-address share test (×1000) and the fp32 matmul checks are exact against CPU. Resuming the step-16000 checkpoint on that card gives
  nonfinite gradients at once.
* That fits the L_max 32 allocations first reaching the faulty region at 14,002.
* Under 135-6 (1), the run is **void** (kept, read for nothing) and **BASE s2 is rerun from scratch** on another 3090 (gen-b89, 595.71).
  - That holds provided the experiments session's CPU repro across the 14k switch is clean; that repro is pending.
* New deploys run a host memory check before their queue and refuse to start on a failure.

**§135 stage-A result (2026-10-07 00:52 EDT; 8 student runs on RTX 3090 / 595.71.05, no restarts; the BASE s2 attempt on b78 is void (host fault, 135-6–135-8) and was
rerun on b89; scored by Theory alone with `theory/score_s135.py`; post-mortem 135-9 / 135-10). FAIL in both arms (0/4 each). The user's claim on S5:
neither arm passes.**
* **TSD** (teacher hidden state): 0/4 exact.
  - acc at prefix 1 is 0.05–0.54, so the runs are wrong from the first step.
  - ρ 0.15–0.35. The main loss ends at 3.2–4.7 (chance 4.79).
  - codes_per_state 1,957–2,557.
* **BASE** (pretrained GPT-2 on the same text): 0/4 exact.
  - acc at prefix 1 is 0.95–1.0 and at prefix 2 0.08–0.62: compounding.
  - ρ 0.48–0.78, non-contracting. The main loss ends at 1.6–4.6.
  - codes_per_state 3,049–6,870.
* **Both are worse than no distillation** on the same seeds (plain s1 exact, 1/4 overall), and far below §134 COPY (4/4).
* **Mechanism (135-9):** the weight-1 InfoNCE has a floor and never vanishes (its minimum is at 2k, then it rises to about 2 by 24k). Its
  context-specific positive pulls history into the code: 2,000–6,900 codes per state, against 14–29 in exact runs. That breaks the Markov carrier.
* **Consequences (as registered):** stage B (multiplication) is not run. A revision would be a new registration, after the user's decision (Audit
  is relaying the post-mortem).

## 136. GROW: start the latent narrow, widen it once the loss reaches a threshold or a plateau (pre-registered 2026-10-06 22:16 EDT, after Literature's check 15ef99be / 1b5b7ea9 and Audit's review, before any implementation or run; the user's idea "restrict the latent size in early training, then gradually widen it once a loss threshold or plateau is reached", with "a few arms with unique modifications that hope to improve latent training" (via Audit))

**Why.**
* **Entry got worse with width:** at d 64 every §133 arm flagged seeds 1 and 101, while d 8 / d 16 entered seed 1.
* **Once entered, width is safe:** FSQ-64, NOISE-64 and CONT-64 are exact and contractive on their proof slots (133-6 / 133-8).
* **Capacity needs width:** FSQ-8 has 8⁸ codes, and §133 stage 2 tests S5×4.
* §126 ANNEAL (wide → narrow) failed 0/4. **Narrow → wide is the untested direction.**
* **Prior** (`docs/LIT_S136_GROW_2026-10-06.md`, new as scoped):
  - Net2Net 1511.05641, GradMax 2201.05125, NeST 1711.02017, DEN 1708.01547, Self-Expanding NNs 2307.04526;
  - ProGAN 1710.10196 (fade-in); RVQ / SoundStream 2107.03312;
  - Burgess 1804.03599 (controlled capacity increase, the closest concept);
  - nested dropout 1402.0915, Matryoshka 2205.13147, MatQuant 2502.06786.
  - Growth mechanisms are known; their use on an LM's latent carrier, scored by exactness, SERIAL and capacity, is not.
* **Scope of stage 1 (Literature, Audit):** on S5 an 8-d code already suffices, so new dimensions get no capacity pressure. S5 tests **entry,
  rescue and no harm**. **S5×4 (stage 2) decides the benefit.**

**Base.** §128's S5 recipe (fsq120 FINAL, FSQ with 8 levels, the length curriculum, p_hard 2000 → 4000, 24k, bs 64, the 127-3 guard; cloud RTX
3090, checkpoints, host memory check), with **no restarts and no plateau stop** (as §134).

**The growth operation (every widening, d → d′).**
* W_down gains d′ − d rows with fresh default initialisation. W_up gains d′ − d columns at **exactly 0** (zero only the outgoing side; zero on
  both sides is a dead saddle, Literature).
* **Fade-in:** the new W_up columns are scaled by α(t) = min(1, (t − t_switch) / 500) (ProGAN).
* **Optimiser:** the new parameters form their own Adam group with fresh moments and a linear learning-rate warm-up over 300 steps. Old
  parameters keep their state.
* **Asserted at a forced switch before launch:**
  - max |Δ logits| = 0 on a fixed batch (function preserved);
  - at least 90 % of the new dimensions take at least 2 FSQ levels on that batch (not inert);
  - gradflow under CUDA bf16: nonzero gradient into the new W_up columns from the first step after the switch, and into the new W_down
    rows from the next.
* Each widening is logged as {"step", "grow": {"from", "to"}}. A pre-switch checkpoint is kept until its fp32 contraction is computed on the
  host (ρ before / after, descriptive), then deleted.

**Triggers (label-free; fixed now).**
* **THRESHOLD:** the first logged step t ≥ 2,000 whose main-chain loss is ≤ 2.5. Exact S5 runs reach 0.37–2.75 by 4.5k; failures stay ≥ 3.4.
* **PLATEAU (at a non-zero loss; DEN-style):** the first logged step t ≥ 4,000 (after the rounding ramp) where the mean logged loss over
  (t − 500, t] is ≥ 0.5 and ≥ 0.98 × the mean over (t − 1,000, t − 500].
* Each run reports the would-be step of the trigger it does not use.

**Arms (S5 seeds 1–4, RTX 3090 class, one 24k run each).**
* **GROW-P** (the user's plateau version, plain base): d 8 → 64 once, at PLATEAU. On seeds where plain d 8 stalls (s2–s4 on the 3090), this
  tests whether **widening rescues a stuck narrow run**. Where d 8 learns, PLATEAU should not fire.
* **COPY+GROW** (the threshold version, on §134's base, which enters 4/4): d 8 → 64 once, at THRESHOLD. This tests **no harm** from growth on
  entered runs.
* **COPY+STAGED** (the user's "gradually"): d 8 → 16 at THRESHOLD, then doubling (→ 32 → 64), each time at least 1,000 steps have passed
  since the last widening and the loss is ≤ 2.5.
* **COPY+GROW-NOISE:** COPY+GROW, with the 56 new dimensions continuous (never rounded) and dither-trained, U(±0.25) at training hand-offs
  (NOISE-64's stabiliser). The first 8 stay hard FSQ.
* **GROW-4** (the narrower start; plain base): d **4** → 64 once, at THRESHOLD. **Entry at d 4 is pre-registered against d 8's 1/4** on these
  seeds (Literature, Audit).
* **NESTED-64** (one run, seed 1; the no-trigger comparator): d 64 from step 0. Each training chain draws a prefix width from
  {4, 8, 16, 32, 64}, and code dimensions beyond it are zeroed at every hand-off (training only; the test uses all 64). Accuracy with only
  the first 8 dimensions at test is reported.
* COPY arms use §134's copy rows from step 0 exactly (f 0.1; CE_main + 1.0 CE_copy, 134-4).

**Comparators (existing; not rerun).**
* Plain FSQ-8 on these seeds and card class, no restarts: 1/4 (s1; s2 plateau, s3 escape-fail, s4 basin).
* COPY-8 (§134): 4/4.
* Fixed d 64 (§133): entry flagged on s1 / s101.

**Rules (fixed now; per arm, 4 seeds).**
* **PASS:** all four seeds exact at 2,048 (ext_2048 ≥ 0.99), SERIAL (donor ≥ 0.90, recipient − chance < 0.10) and contractive (ρ ≤ 0.28:
  127-12's clause as in score_s134; 133-1's first-three form for COPY+GROW-NOISE, whose new dimensions never round). **IMPROVES**
  (descriptive): 3/4. **FAIL:** ≤ 2/4.
* NESTED-64 (one run) is descriptive.

**Readings.**
* **GROW-P PASS** (plain 1/4): widening at a plateau rescues stuck narrow runs. That is a new entry fix by the user's mechanism.
* **COPY+GROW / COPY+STAGED / COPY+GROW-NOISE PASS** (COPY-8 4/4): growth is **safe** on entered runs. Otherwise, the widening harmed a settled
  carrier; the switch spikes and ρ before / after say how.
* **GROW-4:** entry at d 4 (THRESHOLD fired) in ≥ 3/4 reads "a narrower start enters more reliably". PASS reads as for GROW-P.
* **Grown runs exact but the new dimensions unused:** expected on S5, as stated. Stage 2 decides.

**Descriptive (every run).**
* The trigger steps, the switch spike (max loss over the 1,000 steps after a widening, minus the loss at the switch) and ρ before / after.
* Dimension use at the end: the variance share in the new dimensions, probe R²_state from the new dimensions alone, effective rank.
* Codes per state, prefix accuracy at 1 / 2 / 4 / 8, and the end probe.

**Stage 2 (a separate registration and approval; shared with §133 stage 2).** S5×4 with the PLATEAU trigger at a non-zero loss: the GROW arms
that are safe on S5, against FSQ-8, FSQ-8-L16 and the fixed-width §133 passers.

**Placement.** Stage 1: 5 arms × 4 seeds + 1 nested run = 21 runs, about $10.5 (Audit approved $8; the extra $2.5 is requested). They deploy as
slots free, under 16 hosts and $4/h, after §133's slots 2–4 and §135's students. Implementation: the experiments session (fsq120 `--grow
8:64`, `--grow_trigger 2.5|plateau`, `--grow_staged`, `--grow_noise 0.25`, `--nested`; the current width saved in the checkpoint and resume
path). Smoke_plain plus the asserts above before launch. Scorer: `theory/score_s136.py`, committed before any run.

**Note 135-8 (2026-10-06 23:02 EDT; closes 135-6 / 135-7).** The CPU repro across the L_max 32 / micro switch is clean
(`s135_base_s5_s2.void_hostfault/repro_cpu_switch_log.jsonl`): 0 nonfinite steps through 260 steps past the switch, and the label-mask share
stays in its expected range throughout.
* **135-6's ruling (1) holds: a host fault, not the harness.**
* `s135_base_s5_s2.void_hostfault` is read for nothing. BASE s2 is the rerun on gen-b89 (host check PASS).

**Note 135-9 (2026-10-06 23:50 EDT; post-mortem before any new arm; 6 of 8 students read from the raw files, TSD s1 and the BASE s2 rerun still training; Audit read
the same files; the user flagged "distillation loss very bad"). Stage A fails in both arms, and both are worse than no distillation.**
* **The verdict is already determined.** Each arm has 3 failing seeds of 4, so at most 1/4 can pass. Both arms read **FAIL** by the registered rule
  (≤ 2/4). The formal score_s135 run follows when the last two land.

| Run | ext_2048 | acc at 1 / 2 | fp32 ρ | codes per state | main loss 2k → 24k | aux InfoNCE 2k → 24k |
|---|---|---|---|---|---|---|
| BASE s1 | 0 | 0.95 / 0.13 | 0.62 | 6,870 | 2.50 → 3.15 | 0.80 → 2.20 |
| BASE s3 | 0 | 1.00 / 0.62 | 0.48 | 3,049 | 2.07 → 1.63 | 0.73 → 2.05 |
| BASE s4 | 0 | 1.00 / 0.37 | 0.56 | 3,568 | 2.40 → 1.76 | 0.70 → 1.93 |
| TSD s2 | 0 | 0.13 / 0.002 | 0.18 | 1,995 | 3.97 → 4.71 | 0.61 → 1.98 |
| TSD s3 | 0 | 0.54 / 0.04 | 0.24 | 2,557 | 3.61 → 4.74 | 0.67 → 2.15 |
| TSD s4 | 0 | 0.05 / 0.002 | 0.15 | 1,957 | 3.53 → 3.18 | 0.68 → 1.76 |

* **Comparators on the same seeds and card class:**
  - plain s1 is exact (codes per state 29), while BASE s1 fails;
  - §134 COPY is 4/4 exact (14–29 codes per state).
* **Mechanism, as read from the data:**
  - (a) **The auxiliary helps entry, then fights the task.** At 2k, BASE's main loss (2.1–2.5) is as low as COPY's (1.2–2.2), and the InfoNCE
    is at its minimum (0.6–0.8, retrieval 0.87–0.92). From the rounding ramp on, the InfoNCE climbs at every stage (to 1.8–2.2, retrieval
    0.57–0.87), and the main loss stalls.
    - Unlike COPY's CE, which reaches ≈ 0 by 2k so its gradient vanishes, **the InfoNCE has an irreducible floor** (τ 0.1, a context-specific
      positive), so a weight-1 gradient persists for 22k steps.
  - (b) **It pushes context into the code.** The positive target is a hidden state that carries position and history. 135-4's mask removed
    same-state *negatives*, but the pull toward the context-specific *positive* remains. Codes per state are 2,000–6,900, against 14–29 in
    exact runs, so the code is history-dependent, not Markov.
    - BASE ends F2b-compounding: right at step 1, then decaying, non-contracting (ρ 0.48–0.62).
    - TSD, whose richer target dominates more, is wrong from step 1, with a contractive map to a wrong attractor (F2a-like).
  - (c) masked_share is 0.0083 in every run, so the mask worked as designed and is not the cause.
* **Design choices that contributed (Theory's, disclosed):**
  - constant weight 1 (135 / 135-1; Audit's draft had an anneal);
  - a contrastive target with a floor, where §118 ANCHOR used an achievable target (the student's own text path);
  - the full hidden state rather than a state subspace (Literature 0bb148e9 (b) offered both, and InfoNCE was chosen).
* **Consequences:**
  - Stage B (multiplication) is **not run** as registered: its premise was a working stage-A method.
  - **The user's claim (teacher internals against generic text encodings) is not readable from S5:** both arms sit below plain.
  - Any revised design is a new registration, put to the user via Audit first, not a quick fix. The candidate directions are listed in the
    synthesis.

**Note 135-10 (2026-10-06 23:51 EDT; field naming for 135-9, after Audit's reproduction).** 135-9's "codes per state" is eval.json's **`codes_per_state`**: distinct codes
per state over the 2,000 × 512 test chains. It reads BASE 6,870 / 3,049 / 3,568 and TSD 1,995 / 2,557 / 1,957, against plain s1 29.2 and COPY s1 14.0.
* Audit quoted the other field, **`code_growth_ext["2048"].per_state_mean`** (500 chains of 2,048): BASE 6,901 / 3,289 / 4,041 and TSD 1,754 / 2,458 /
  2,058, against plain 29.5 and COPY 16.5.
* Both fields give the same reading: the §135 codes carry 60–470 times more codes per state than exact runs. The prose range "2,000–6,900" in
  135-9 should read 1,957–6,870 (`codes_per_state`), or 1,754–6,901 (`code_growth_ext`).

**Note 133-9 (2026-10-07 01:19 EDT; interim; scored by Theory with score_s133). SOFT-16 reads stage-1 PASS. FSQ-16 is decided by its slot 2.**
* **SOFT-16: PASS** (≥ 3/4 slots).
  - slot 1 s1: ext 0.998, ρ 0.134 (fp32 rho_k3, now computed);
  - slot 2 s102 (s2 flagged at 2k): ext 1.000, ρ 0.113;
  - slot 4 s4: ext 1.000, ρ 0.111.
  - All are SERIAL. Slot 3 is still running and is descriptive for the verdict.
  - A weak soft pull (β 0.1) toward the FSQ grid keeps a **16-d** carrier exact and contractive, where it did not at d 64 (SOFT-64 dropped, 133-7).
* **FSQ-16:** slot 1 passes (0.996, ρ 0.167), slot 4 passes (s4: 1.000, ρ 0.176), and slot 3 fails (s3 unflagged at 2k, ext 0.0). **Slot 2 (running)
  decides PASS or FAIL.**
* **64-d arms** (FSQ-64, NOISE-64, CONT-64): only slot 1 is decided (all pass). Slots 2–4 are running.

**§136 result (2026-10-07 12:35 EDT; stage 1, RTX 3090 class, one 24k run per seed, no restarts; scored by Theory with score_s136; NESTED-64 descriptive, below).
COPY+GROW and COPY+GROW-NOISE PASS 4/4. COPY+STAGED reads IMPROVES (3/4). GROW-P FAILS 1/4 and GROW-4 FAILS 0/4.**

| Arm | Verdict | ext_2048 by seed | fp32 ρ at the end | Widening |
|---|---|---|---|---|
| COPY+GROW | **PASS 4/4** | 0.998 / 1.0 / 0.998 / 0.992 | 0.097–0.142 | 8 → 64 at 2k (THRESHOLD) |
| COPY+GROW-NOISE | **PASS 4/4** | 1.0 / 0.99 / 1.0 / 0.998 | 0.072–0.110 | 8 → 64 at 2k, new dimensions continuous |
| COPY+STAGED | IMPROVES 3/4 | 1.0 / 0.998 / **0.976** / 0.992 | 0.081–0.103 | 8 → 16 at 2k, → 32 at 3.0–3.9k, → 64 at 4.0–4.9k |
| GROW-P | FAIL 1/4 | 0.99 / 0 / 0 / 0 | s1 0.137 | 8 → 64 at 4.7k (s1, 136-1) and at 4.0k (s2–s4) |
| GROW-4 | FAIL 0/4 | 0 / 0 / 0 / 0 | — | never: THRESHOLD did not fire at d 4 |

* **Readings (as registered):**
  - **GROW-P (plain d 8 is 1/4 on these seeds):** widening at a plateau does **not** rescue stuck narrow runs. s2–s4 widened at 4.0k and stayed at ext 0,
    with the state unreadable from the new dimensions (R² −0.19 to 0.08). On s1, widening at a false plateau did no harm (136-1).
  - **COPY+GROW and COPY+GROW-NOISE (COPY-8 is 4/4):** growth is **safe** on entered runs. The 2k widening costs a loss spike of +1.39 to +2.07 nats, which
    recovers.
  - **COPY+STAGED:** s3 ends at 0.976 (SERIAL, ρ 0.10), so the arm does not pass. Its later widenings fell inside the rounding ramp, where the pre-switch ρ
    was 0.43–0.96.
  - **GROW-4:** no seed entered at d 4 (0/4, against d 8's 1/4). A narrower start enters **less** reliably.
* **Not as expected: the new dimensions are used.** In the grown COPY runs they hold 0.78–0.86 of the code variance, and the state is readable from them
  alone at R² 0.95–0.98. On S5 the code moves into the new width without capacity pressure.
* **The PLATEAU trigger** would also have fired on every COPY learner (would-be steps 4.5–5.2k). Its 2 % tolerance fires on learners in general (136-1).
* **Disclosed:** COPY+GROW, COPY+GROW-NOISE and COPY+STAGED are identical up to their 2k widening (same seed, base and card; pre-switch ρ 0.123 / 0.166 /
  0.169 / 0.146 in all three). They are three continuations of one entry per seed, not independent entries.
* **The user's idea, on S5:** widening a narrow latent at a loss threshold is safe, and the code uses the new width. Widening at a plateau does not rescue
  failed runs, and a narrower start hurts entry. **The benefit is decided at S5×4** (§133 stage 2), where COPY+GROW and COPY+GROW-NOISE join the proposal.
* **NESTED-64 (one run, seed 1, descriptive; landed 14:09):** it never enters. ext_2048 0.0, acc at length 1 0.0095 (chance), final loss 4.81 (≈ ln 120),
  effective rank 1.17 (a collapsed code), acc with the first 8 dimensions 0.0. That matches fixed d 64 on seed 1, flagged at 2k in §133. Random prefix widths do
  not fix entry at d 64.

## 137. Multiplication on the task-agnostic carrier, column by column: a carried carry, streamed product digits (pre-registered 2026-10-07 01:34 EDT, after Literature's check c5687f0c and before any implementation or run; the user's "start multiplication" (via Audit), plan `docs/THEORY_MULT_PLAN_2026-10-07.md` v2)

**Why.**
* The user's rule: multiplication returns only through the task-agnostic harness (fsq120, §128's recipe), not §98's task-specific one.
* §98's latent arm hit the k = 4 hand-off wall (GEN-M 0.163 / 0.005 at 4×4 / 5×5). Text CoT is 1.0, and ICoT-SI's released checkpoints reach
  0.998 / 0.996 on our harness.
* **Ordering, after Literature (`docs/LIT_MULT_PLAN_2026-10-07.md`):** output-column-major (Neural GPU 1511.08228; position coupling 2410.15787).
  - **The carried state is the carry only:** it is bounded (≤ 2 digits, about 6 bits, plus the emitted digit) at any length, so it **fits FSQ-8**.
    The multiplier-major alternative carries S_j (26.6 / 33.2 bits) and stays in the plan as the capacity stress arm.
  - It is the only ordering whose state does not grow with length, so it can be tested on 4×8 (C2).
* **New as scoped:** no published latent result does final-answer-content-only multiplication through a carried recurrent code (Literature).

**Task and data.** The bench files (digits least-significant first): train on `bench/data/mult_train/{4x4,5x5}_train.txt`, using only (a, b, product).
The CoT text is **never used**. Test on the **1,000 published items** (`bench/data/mult_{4x4,5x5}_test.txt`), the same items as §98's comparators.

**The family (one adapter in fsq120; `--family mult_col --task 4x4|5x5`).**
* **Column k = 1 … 2n:** segment k = [EOT, carried code, a's digit tokens, b's digit tokens, a column-index token k].
  - The operands are in context every segment, as in §132's in-context tables.
* **The code after segment k** is decoded by the answer decoder into **product digit p_k** (one token).
* The **start** is a fixed code (as for FF / CYC / MOD / RDFA), so the carry is 0.
* **The prediction** is the 2n emitted digits. The test is exact match on all of them (the bench scorer).
* **C4 qualifier (disclosed, as 110b-1):** the per-step targets are **final-answer digits delivered per step**, aligned to the column index.
  - There are no carry labels and no intermediate state of the bench CoT.
  - Strict final-only supervision would make the code hold the whole product, which is the multiplier-major capacity arm.
* **Curriculum** (§120's idea over columns): chains of L columns with L ~ U(1, L_max), whose targets are the product's low L digits. L_max is
  2 / 4 / 8 at 0 / 2k / 5k for 4×4, and 2 / 4 / 8 / 10 at 0 / 2k / 5k / 9k for 5×5.

**Recipe.** §128's (FSQ d 8 / L 8, p_hard 2000 → 4000, 24k steps, bs 64, AdamW lr 1e-4, the 127-3 guard), with **restart-on-plateau slots** (seeds
k, k + 100, k + 200, k + 300; 123-3's 2k flag). Cloud RTX 3090, checkpoints, the host memory check, versions recorded.
* **No COPY:** its targets (whole products) are not the carried quantity, and 8-digit products exceed 24 bits.
* **Before any run:** gradflow under CUDA bf16 (the digit CE at column k ≥ 2 reaches W_down / W_up through the carried code) and smoke_plain.

**Rules (fixed now; per size).**
* **M-PASS:** at least 3/4 slots whose accepted run has **test_acc ≥ 0.99** on the 1,000 items, is **SERIAL** and is **contractive** (ρ ≤ 0.28,
  127-12's clause).
  - **SERIAL:** inject the code of a donor problem after column k (k = 2 / 4 / 6) into a recipient chain. The following emitted digits must follow
    **the recipient's columns with the donor's carry** (donor ≥ 0.90), not the recipient's own carry (recipient − chance < 0.10). Chance is the
    rate at which the two carries give the same digits; the ground truth is computable exactly.
* **FAIL:** at least 2 decided slots not passing.
* **DROP at proof:** slot 1's accepted run below 0.95 stops that size (no further seeds).
* **Order:** 4×4 proof → 4×4 slots 2–4 → 5×5 (four slots) only after 4×4 PASS → 4×8 (descriptive; below).

**Descriptive.**
* **M-SOTA:** the median test_acc against ICoT-SI (0.998 / 0.996), TC-M (1.0) and GEN-M (0.163 / 0.005) on the same items. Our distinguishing
  claim is C1 (SERIAL).
* **acc_by_column:** where errors start.
* **Shortcut probe** (Literature / Audit): R² of the **true carry** read from code_k, against R² of the **context-only column sum mod 10** (what a
  model could compute without the carried code). A model that ignores the code fails exactly when a carry matters, and SERIAL shows it.
* **4×8 (C2):** the 4×4 models evaluated on 1,000 generated 4×8 items (8-digit b, 12 columns). Column indices above 8 are unseen in training, and
  that limitation is stated.

**Readings.**
* **4×4 (and 5×5) PASS:** a pretrained LM's task-agnostic latent carrier does multi-digit multiplication exactly, from final-answer digits, with a
  causally used carry: where §98's latent arm hit its wall.
* **DROP or FAIL:** the per-column step (up to n products plus a carry, in one segment) is too hard at this budget. The registered fallback is
  digit-pair ops (n² easier steps, the same carry), as a separate registration.
* **4×8 exact:** length generalisation with a bounded state (C2 on a natural task).

**Placement.**
* 4×4: up to 4 slots × at most 4 attempts, about $0.2–0.4 per run (2n segments of about 12 tokens).
* 5×5: the same. About $2–4 in all (approved by the user / Audit).
* Implementation: the experiments session (the family adapter: multi-token context, streamed digit decoding, the donor ground truth, the
  shortcut probe, the 4×8 generator). Dirs `s137_col_{4x4,5x5}_s{seed}`.
* Scorer: `theory/score_s137.py`, committed before any run.
* The full-width plan (`docs/THEORY_FULLWIDTH_PLAN_2026-10-07.md`) becomes §138 when it triggers.

**Note 137-1 (2026-10-07 01:37 EDT, before any §137 implementation or run; Audit's review of 9c9b1899, APPROVED with four additions).**
* **(1) Leakage.** Every generated item (synthetic 4×4 / 5×5 if any, and the 4×8 descriptive set) is deduplicated against the 1,000 test items by
  (a, b), with commuted pairs included. The bench train split is also checked against the test items. The generator logs the counts removed.
* **(2) The bypass reading, stated now.** Each segment sees all of a and b plus k, so carry_k and p_k can be computed **in-pass**, without the
  carried code. ICoT-SI's 0.998 / 0.996 shows that GPT-2 small can learn in-pass multiplication.
  - A run with test_acc ≥ 0.99 but **SERIAL failing** reads **"IN-PASS (ICoT-like), not a latent carry"**. It counts as not passing for M-PASS,
    and it is reported as that class, not as a capacity failure.
  - The shortcut probe gains a column: R² of the true carry from **h_k before W_down**, beside code_k and the context-only column sum. score_s137
    prints all three.
* **(3) The 2k flag on digit CE.** The 123-3 flag was calibrated on S5 losses. On per-digit CE (chance ln 10 ≈ 2.30, with the early columns easy)
  its ratios may misfire. It stays in force as registered.
  - Its decision on every attempt is logged.
  - A **misfire** (the flag fires while loss(2k) is already below 1.0 nat per digit) on the proof slot voids the proof as a harness issue, not a
    method failure. The proof is then rerun with restarts off, disclosed.
* **(4) SERIAL chance** is computed **per donor–recipient pair**, as the share of later columns whose digit is the same under both carries. It is
  reported per position (2 / 4 / 6), since the carry range grows with k at 5×5.
* **C-tags from now on: the user approved the C4 rewording** (Audit, NORTH_STAR). C4 = trainable from supervision that exists at scale: final
  answers or rewards, or a text-CoT model's outputs. No hand-built intermediate states. Strict C4 (final answers and rewards only) is kept as the
  stretch goal for surpassing the teacher.
  - **§137's streamed product digits qualify under C4.** Simulator states (S5's per-step state labels, §111 / §118) do not.

**Note 137-2 (2026-10-07 01:55 EDT, before any §137 result; the experiments session's implementation, Audit-approved). As implemented.**
* **Curriculum:** **column prefixes of full bench items**. A chain of L ~ U(1, L_max) columns has the product's low L digits as targets. **No
  generated or synthetic training items** (no shorter multipliers).
* **Dedup (137-1 (1)):** commuted (a, b) pairs against the 1,000 test items. **17 bench train items removed at 4×4.** The ext_4x8 descriptive set had
  0 to remove.
* **Gates before launch:** CUDA bf16 gradflow PASS (`theory/results/gradflow_s137_cuda.json`), GPU smoke PASS, peak memory 4.47 GiB (4×4) / 5.78
  GiB (5×5 eval at bs 250). 5×5 evaluations stay on 3090s, since 5.78 GiB is near the local cap.
* The 4×4 proof (slot 1: seeds 1 / 101 / 201 / 301, 123-3's 2k flag) deploys on one RTX 3090.

**Note 133-10 (2026-10-07 01:56 EDT; the 16-d arms scored by Theory with score_s133; Audit-read; and a correction).**
* **SOFT-16: PASS, 4/4 slots.** Exact at 2,048, SERIAL, with fp32 rho_k3 0.134 / 0.113 / **0.105** (slot 3, s3) / 0.111. One restart, in slot 2 (s2 flagged at 2k,
  then s102).
* **FSQ-16: FAIL, 2/4.** Slots 1 and 4 pass. Slots 2 (s2) and 3 (s3) are accepted runs, not flagged at 2k, that end at ext 0.0: escape without learning.
* **Descriptive:**
  - seed 3 escapes and fails under FSQ-8 (§128 / §127) and under FSQ-16, but **learns under SOFT-16**;
  - the soft carrier (β 0.1 at every hand-off, train and test) never goes through §120's hard-rounding ramp.
  - At d 16 this is consistent with candidate 3's rounding hypothesis, while the d-8 ramp prototype found F2 mostly without rounding. Different
    widths, n small, so no claim.
* **Correction to 133-9:** the 64-d arms' slots 2–4 were **never queued**: only slot-1 queues existed. 133-9's "running" was wrong. The 9 runs
  (FSQ-64, NOISE-64, CONT-64 × slots 2–4; about $4.5, approved under 133-5) are queued now. §138's trigger cannot fire before they land.

**Note 136-1 (2026-10-07 01:59 EDT; before any §136 result is read; Audit's interim raw-log read of b95; a reading fixed now, no rule change).** GROW-P's PLATEAU
trigger fired on seed 1, the plain learner on this card class.
* s136_growp_s5_s1 widened 8 → 64 at **step 4,700**. The firing is correct as registered: mean over (4.2k, 4.7k] = 2.475, which is ≥ 0.98 × mean over
  (3.7k, 4.2k] = 2.461, and ≥ 0.5.
* But the run was still descending noisily (3.82 at 3.5k, then 2.13–2.82 over 4.1k–4.7k). A 2 % tolerance over 500-step windows fires on a
  **slow learner**. After the widening the loss went 2.34 → 4.19 at the L_max 8 step (5.1k). It is not decided at 11.3k.
* **How s1 will be read, fixed now:**
  - **s1 ends exact:** widening at an early, false plateau did no harm.
  - **s1 ends unexact:** GROW-P fails on s1 by its registered rule. The cause is named **"the plateau trigger fired on a still-learning run (a false
    plateau) and widened an unsettled carrier"**. It is **not** read as "growth harms a settled carrier", which the THRESHOLD-triggered COPY arms
    test.
  - The trigger is part of GROW-P's method, so the arm's verdict stands either way. The false-plateau rate on learners is reported as a property of
    the trigger.

**§137 result (2026-10-07 05:21 EDT; the 4×4 proof, s137_col_4x4_s1, RTX 3090 / 595.71.05, unflagged at 2k; scored by Theory with score_s137). DROP at proof: test_acc 0.000.
4×4 stops there (no further slots, no 5×5). §140 (DIGPAIR) triggers.**
* **Where it fails** (acc_by_column, per column): **1.000 / 0.326 / 0.124 / 0.103 / 0.077 / 0.104 / 0.105 / 0.953**. Cumulative: 1.0 / 0.33 / 0.06 / 0.01 / 0 / 0 / 0 / 0.
  - Column 1 is solved (a₁b₁ mod 10, computable from the context alone) and the top digit is easy. The carry-dependent middle columns are near chance.
  - The per-digit CE rose with L_max: 0.43 at L 2, 0.89 at L 4, then about 1.30 from 6k to 24k. train_acc is 0.000 too, so this is not overfitting.
* **Probe (137-1 (2)):**
  - the true carry is linearly readable from **h before W_down at R² 0.69**, but from the **code only at 0.24**;
  - the context-only column sum is readable at column 1 (R² 0.78) and not at all from column 2 on (≤ 0.15).
  - Read: the segment does not compute the middle columns' digit-product sums from the in-context operands. Part of the carry reaches the segment's last
    hidden state but not the 8-d code. Descriptive, n = 1.
* **SERIAL:** not met (donor 0.24–0.36, about equal to recipient). As expected for a run that never learned the carry.
* **Fits the pre-registered reading (DROP or FAIL):** the per-column step (up to n products plus a carry, in one segment) is too hard at this budget. Audit's
  interim read located the load in-pass. The registered fallback is §140 (digit-pair ops: one product per step, the code carrying the accumulator).
* **What 4×4 = 0 does not show:** the latent carry itself was never tested, because the transition was never learned. §140 tests that, with design-forced C1.

## 139. DISTILL-2: a CoT teacher's state as a fading warm-start for the latent carrier, on multiplier-major multiplication (pre-registered 2026-10-07 02:10 EDT, after Literature's checks 0bb148e9 / ff34ece2 and Audit's notes, before any implementation or run; the user's "go on distill" (via Audit), plan `docs/THEORY_DISTILL_PLAN_2026-10-07.md`)

**Why.**
* §135 failed: a weight-1 InfoNCE to a teacher's full hidden state never vanished and pulled context into the code (135-9).
* The user still asks whether a CoT model can be distilled into a latent model.
* DISTILL-2 fixes the three causes:
  - a **state-only, achievable** target;
  - a **warm-start that fades to exactly 0**;
  - **COPY's base** for entry.
* It is tested on a task where final-answer-only latent training may hit a wall: **multiplier-major multiplication**, where the carried state is the
  running sum S_j.
* **Prior** (ff34ece2, `docs/LIT_DISTILL2_2026-10-07.md`): KD soft targets 1503.02531, FitNets 1412.6550, Concept Bottleneck Models 2007.04612;
  ICoT-KD, KaVa, CODI. Distilling a teacher's *probe-decoded* state into a discrete carrier is new as scoped.

**Task (fsq120 `--family mult_row --task 4x4`).**
* Segment j = [EOT, carried code, a's digits, b_j, position token j], for j = 1 … L.
* **Final answer only:** the product a × (b_1 … b_L) = S_L, decoded from the final code.
* Train on the bench 4×4 train split (only a, b and the product; commuted test pairs removed, 137-1), and score on the 1,000 published test items.
* **Curriculum:** L ~ U(1, L_max), with L_max = 1 / 2 / 3 / 4 at steps 0 / 1k / 2k / 3k. It is compressed so that the full length is reached **before**
  the fade ends (Literature (a)); see below.

**Carrier and base (arm A).**
* **SOFT-16** (d 16, 8 levels, β 0.1 at every hand-off, train and test), §133's only stage-1 PASS (4/4). Its FSQ grid holds 8¹⁶ codes (48 bits) and the
  soft mix is continuous, so 4×4's S_j < 10⁸ (26.6 bits) fits.
* **§134 copy rows** (f 0.1; CE_main + 1.0 CE_copy, 134-4). Their targets are other problems' products, numbers of the same kind as S_j.
* 24k steps, bs 64, lr 1e-4, the 127-3 guard, **no restarts, no plateau stop**. Cloud RTX 3090, checkpoints, the host check.
* **A is final-answer content only (strict C4).**

**DISTILL-2 (arm B = A + one change).**
* **Teacher:** §98's kept TC-M 4×4 GPT-2 (`boxes/runs_mult/m98_tc_4x4_s1`, acc_test 1.0), frozen.
  - For each training problem it **generates its own CoT** (greedy).
  - At the last token of each S_j text in that generated CoT (P_1 for j = 1), its hidden state at a probe-chosen layer is read.
* **Teacher gate (before any student):**
  - per-layer, per-digit linear probes (10-way for each digit of S_j, padded to 2n digits) are **fit on the teacher's own decoded S_j text** (C4
    as reworded: a text-CoT model's outputs, not simulator labels), on 4,096 training problems;
  - the chosen layer is the argmax of held-out exact-match over all S_j digits (ties go to the later layer), and it needs ≥ 0.95.
  - Also reported: the probe's agreement with the teacher's decoded S_j, and with the **gold** S_j.
* **Target mode, fixed now (Literature (2)):**
  - **KL** to the probe's per-digit distribution if the probe agrees with the teacher's decoded S_j at ≥ 0.99 per digit (held-out);
  - otherwise **CE on the teacher's own generated S_j text**.
  - Both are C4 as reworded. The mode is recorded as s139.target_mode.
* **Student:** a linear readout head on e_j = W_up(code_j) predicts S_j's digits. The loss is averaged over digits and hand-offs, and reaches the
  carrier's encoder through the code.
* **Weight:** 1 until step 3,000 (when the curriculum reaches L 4), then linear to **exactly 0 at 4,000**. This keeps the user-approved "faded by
  4k", and every hand-off length is covered before the fade (Literature (a); Audit (1)).
* **Arm C (constant-weight control; Literature):** B with the weight held at 1 for all 24k. It is READ-like, with teacher targets.
* **Gradflow under CUDA bf16 before any run:**
  - the auxiliary loss reaches W_down and the trunk through the code (nonzero) before 3k;
  - its gradient is **exactly 0** from step 4,000 in B;
  - no gradient reaches the teacher.

**Rules (fixed now; per arm, seeds 1–4).**
* **PASS:** at least 3/4 seeds with test_acc ≥ 0.99, SERIAL and contractive (133-1's first-three form, ρ ≤ 0.28, since SOFT is continuous). **FAIL:**
  otherwise.
  - **SERIAL:** inject a donor's code after step 2. The answer must equal the donor's S_2 continued with the recipient's a and remaining digits (donor
    ≥ 0.90; recipient − chance < 0.10). The ground truth is exact.
* **Readings, fixed now (Audit (2)):**
  - **B passes where A does not** (or B has the higher exact count): **a CoT teacher's state can stand in for gold step labels**, as supervision that
    exists at scale (C4 as reworded).
    - Because TC-M is about perfect at 4×4, its probe-decoded state ≈ its text ≈ the gold S_j. This **does not** show that hidden-state
      distillation adds value beyond the teacher's text.
    - The claim wording carries the measured probe agreement with gold S_j. If it is about 1.0, the claim says so.
  - **A passes:** final-answer content plus COPY already suffices on 4×4. B is then read on steps-to-exact and the seed count (descriptive).
  - **B worse than A:** the warm-start harms, as §135 did.
  - **C against B (descriptive):** whether keeping the teacher signal on longer helps or hurts.
* **Descriptive (Literature (b); Audit (3)):**
  - after the fade, the frozen readout's S_j accuracy on the code, beside a **b-prefix probe** (the shortcut);
  - steps to first exact; acc_by_length; codes per state.

**Order and placement.**
* §139 runs **after §137's 4×4 proof reads and the §134 guard reads** (COPY's no-harm gate).
* Teacher probe first ($0, local).
* Then a 1-host proof (A s1 and B s1).
* Then seeds 2–4 for A / B / C: 12 runs, about $3–4 (Audit approves the extra for C).
* 5×5 comes only after the 4×4 reading, with its own approval.
* Implementation: the experiments session (fsq120 `--family mult_row`, `--distill`, `--distill_full 3000`, `--distill_end 4000|0`, `--distill_teacher
  <dir>`, `--distill_layer`). Dirs `s139_{a,b,c}_4x4_s{1..4}`; probe `s139_teacher_probe/`. Scorer: `theory/score_s139.py`, committed before any run.

**Note 139-1 (2026-10-07 02:10 EDT, before any §139 implementation or run; Audit's approval of 683318e2, including arm C, total about $3.7).**
* **(a)** The compressed curriculum (L_max 1 / 2 / 3 / 4 at 0 / 1k / 2k / 3k) applies to **A, B and C alike**. A therefore does not use §98 GEN-M's schedule,
  so A against GEN-M is descriptive only. **The decisive comparison is B against A on the same schedule.**
* **(b) Remaining readings, fixed now:**
  - both PASS: multiplier-major 4×4 does not separate them (steps-to-exact and seed counts are descriptive);
  - both FAIL: DISTILL-2 does not rescue final-answer-only multiplier-major at d 16;
  - C better than B: the fade costs accuracy;
  - B better than C: the fade matters.
* **(c) SOFT-16 + COPY has never been run together** (COPY was proven on FSQ-8). The 1-host proof logs copy_acc and copy_loss. **If copy_acc < 0.9 at 4k in
  A s1, the stage stops and is reported before seeds 2–4**, so that an A failure can be told apart from the combination breaking.
* **(d) The multiplier-major adapter is new code**, so it gets §137's gates:
  - dedup including commuted pairs;
  - the exact S_j ground truth tested against integer arithmetic;
  - gradflow: the aux nonzero before 3k, exactly 0 from 4k in B, nonzero throughout in C, and 0 into the teacher;
  - a memory check;
  - Audit's diff review.

**Note 139-2 (2026-10-07 02:28 EDT, before any §139 student run; the teacher-probe gate read; a registration conflict resolved and disclosed).**
* **The teacher probe** (`boxes/runs_fsq/s139_teacher_probe/teacher_probe.json`; 4,096 deduplicated train problems; RTX 4060 bf16):
  - **The teacher itself is perfect:** format valid 1.000, its decoded S_j equals gold 1.000.
  - **But its S_j is not linearly decodable** at the last token of the S_j text. The best held-out exact match over all 8 digits is **0.174** (layer 8), the
    per-digit agreement with the teacher's text is 0.814, and the gate (≥ 0.95) **fails**.
* **The conflict in 139's text.** The registered target-mode rule falls back to **CE on the teacher's own generated S_j text** when the probe agrees
  < 0.99 per digit. Any such probe has all-8-digit exactness around 0.99⁸ ≈ 0.92 or less, so it also fails the 0.95 gate. Read literally, the fallback
  could never run.
  - Its purpose (Literature ff34ece2 (2)) was exactly this case. **Resolution:** the probe gate governs the **probe (KL) mode only**. In CE mode the
    teacher gate is the validity of the teacher's own text (format valid ≥ 0.99; here 1.000).
  - **B and C proceed in CE mode** (target_mode "ce"): the teacher's generated S_j digits, which here equal gold.
  - The readout position is **not** changed to rescue the probe; that would be a fix after seeing the result.
  - score_s139's teacher gate is updated to match.
* **Descriptive, and relevant to the user's premise** (diagnostics are pending: per-j exactness, per-digit position, fit-set exactness):
  - a perfect text-CoT teacher **keeps its running sum in its text**. It is not a linearly readable compact state at that position.
  - Here "the CoT model's internals" give no better target than its outputs. §139 therefore tests whether **a CoT teacher's outputs** can replace
    gold step labels as a fading warm-start, as 139 pre-stated (C4 as reworded).

**Note 139-3 (2026-10-07 02:39 EDT, before any §139 run; Audit's review of boxes/multrow.py, code approved). What the curriculum labels, and a strict-C4 arm.**
* **(1) A is not strict C4.** The curriculum draws L ~ U(1, L_max) over multiplier prefixes, and the FINAL CE on S_L = a × (b_1 … b_L) supervises S_1, S_2 and S_3
  as the final answers of shortened problems. That is **110b-1's qualifier**, exactly as S5's short problems label transitions.
  - A is renamed **"FINAL with prefix problems (110b-1-qualified)"**. It counts under C4 as reworded (final answers of real, shorter problems), not as
    answers-only.
* **(2) B adds no new information over A here,** because the teacher's text equals gold.
  - B gives the same S_j labels at **every hand-off of one chain**, where A sees them only across different draws.
  - **B better than A** therefore reads "per-hand-off targets within the chain help entry" (cf. §111 / §118), with the teacher only as the label source. "A
    teacher replaces gold labels" cannot be told apart from that here.
  - The wording of 139's B-better-than-A reading is revised accordingly, before any run.
* **(3) New arm A′ (strict C4):** A with **L fixed at 4** (no prefix problems, no curriculum). Its only supervision is the full product and COPY's final-answer
  texts. 4 seeds, about $1.2 (Audit: "your call"); taken.
  - A′ s1 joins the 1-host proof beside A s1 and B s1.
  - **Readings:**
    - **A′ PASS:** strict-C4 multiplier-major multiplication on the latent carrier, the user's stretch goal on this task;
    - **A′ FAIL with A PASS:** the prefix problems (110b-1 labels) are what make it learnable.
  - Dirs `s139_ap_4x4_s{1..4}`, args `--fixed_L 4`. score_s139 is updated.

## 140. DIGPAIR: multiplication one digit pair at a time, §137's registered fallback (pre-registered 2026-10-07 03:40 EDT, **before §137's 4×4 proof reads**; runs only if §137 4×4 reads DROP at proof or FAIL; Literature named the ordering in c5687f0c; Audit's steward draft)

**Why.**
* §137 (column-major) asks each segment to sum up to n digit-pair products from the context in one pass. That is ICoT-style in-pass work.
* Audit's interim read of the 4×4 proof: the per-column CE rises with L_max (0.43 at L 2, then 0.84–0.89 at L 4, then 1.19–1.37 at L 8, to 13.4k).
* §137's readings registered this fallback for DROP or FAIL: digit-pair ops, n² easier steps, the same carry.

**The family (fsq120 `--family mult_pair --task 4x4|5x5`).**
* **One segment per digit pair (i, j)**, in output-column order (columns k = 1 … 2n − 1, pairs with i + j − 1 = k, ascending i), then **one flush
  segment**: n² + 1 steps (17 / 26).
* **Segment** = [EOT, carried code, a_i, b_j, index tokens i and j]. **Only the two digits are visible.**
* **The carried code holds the running column accumulator:** carry + products so far, < 400 (about 9 bits), which fits FSQ-8.
* **Emission:** at the last pair of column k the answer decoder reads the code and emits p_k. The flush segment emits p_{2n}. These are streamed
  final-answer digits, §137's disclosed C4 qualifier (C4 as reworded). There are no accumulator labels.
* **In-pass work** is one 1-digit product plus an add. It is length-general: 4×8 is 32 pairs.
* **Unlike §137, C1 is design-forced here:** a segment cannot see the other digits, so the code must carry the accumulator. SERIAL shows a correct
  latent accumulator, not an unforced choice. That is the trade-off for learnability.

**Recipe, rules and placement: §137's exactly** (137-1 / 137-2 apply).
* Bench train split, using (a, b, product) only, deduplicated including commuted pairs. Test on the 1,000 published items.
* Curriculum over columns: chains covering the first c columns' pairs, targets = the low c digits, c ~ U(1, c_max), with c_max 1 / 2 / 4 / 8 at 0 / 2k
  / 5k / 9k (4×4).
* §128's FSQ-8 with restart slots, 24k steps.
* **M-PASS:** ≥ 3/4 slots with test_acc ≥ 0.99, SERIAL and ρ ≤ 0.28.
  - SERIAL: inject a donor's code after pair 4 / 8 / 12. The later digits must follow the recipient's pairs with the donor's accumulator (exact
    ground truth).
  - FAIL: ≥ 2 decided slots not passing. DROP at proof: slot 1 below 0.95.
  - 5×5 only after 4×4 PASS. 4×8 is descriptive (j indices above 4 unseen).
* **Descriptive:** M-SOTA against ICoT-SI / TC-M / GEN-M on the same items; acc_by_column; an accumulator probe (code_k, h_k before W_down).
* **Readings:**
  - **PASS:** a pretrained LM's latent carrier multiplies exactly when each step's in-pass work is small (design-forced C1, C4 as reworded).
    Together with §137's failure, that locates the barrier in the per-segment in-pass load, not the latent carry.
  - **DROP or FAIL:** even one-digit steps with a carried accumulator are not learned at this budget.
* **Placement:** a 1-host proof (4×4 slot 1, about $0.3–1; steps are many but short), then slots 2–4 and 5×5 on PASS. The adapter is new code, so §137's
  gates apply (dedup, exact ground truth against integer arithmetic, gradflow, memory, Audit's diff review). Dirs `s140_pair_{4x4,5x5}_s{seed}`.
  Scorer: `theory/score_s140.py`, committed now.

**Note 140-1 (2026-10-07 03:42 EDT, before any §140 implementation or run; Literature acf778d8, `docs/LIT_S140_DIGPAIR_2026-10-07.md`: new as scoped).**
* **Index-free segments (replacing the (i, j) index tokens).** Absolute indices would make j up to 7 unseen when testing 4×8 after 4×4 training.
  - Segment = [EOT, carried code, a_i, b_j, a **column-boundary flag**]. The flag marks the first pair of a new column (the code is then read as the old
    accumulator, carried as ⌊acc / 10⌋ before the new product is added).
  - The flush segment = [EOT, code, flush flag].
  - The harness reads the code at each column's last pair to emit p_k = acc mod 10. **The op is then identical at every position.**
  - There is no index-token variant.
* **Range, stated:** the accumulator is at most 81 · min(|a|, |b|) + carry. That is < 400 for 4 × n at any n (good for 4×8), but it grows for n × n. Length
  generalisation therefore holds along one operand only.
* **Reported:** per-step accuracy and the first-error position. 4×8 has 33 steps, so it needs a per-step accuracy of roughly 0.997 or more.
* Emitted digits are temporally split final-answer supervision, as stated in §140. C1 is design-forced, also as stated.

**Note 140-2 (2026-10-07 03:52 EDT, before any §140 run; the experiments session's adapter, boxes/multpair.py; CPU gates PASS).**
* **Flag tokens confirmed** (single GPT-2 tokens, distinct from the digits): " ;" = the first pair of a new column (take acc // 10), " ," = the same column, " =" =
  the flush. Recorded as args pair_flags "column_start" and flag_tokens.
* **5×5 curriculum, fixed now** (it was missing; used only after a 4×4 PASS): c_max = 1 / 2 / 4 / 8 / 10 at 0 / 2k / 5k / 9k / 12k, where 10 covers 5×5's 9 product
  columns plus the flush, as 8 does for 4×4's 7 + 1.
* **Semantics, as implemented:**
  - acc = carry-in + the column's products so far;
  - p_k = acc mod 10 at the column's last pair; the next column starts from acc // 10; the flush emits the final carry;
  - the ground truth equals integer arithmetic on 4×4, 5×5, 4×8, 1×1 and 3×6 (3,000 each; maximum acc 296);
  - the donor continuation is self-consistent.
* **eval.json** carries per_step_given_correct_prefix and first_error_position (140-1), in ext_4x8 too.

**Note 134-6 (2026-10-07 04:49 EDT; descriptive follow-up to the guard, before these runs exist; Audit's caution: the guard's MOD failure is n = 1).**
* The plain side is already established: **§128's plain MOD is fp32-closed in 4/4 seeds** (128-6: 386 / 423 / 229 / 505 reachable, Moore 5, n_uncovered 0, all on
  the 3090 class).
* Proposed, about $1, Audit to approve: **s134_copy_mod_s2 and s134_copy_mod_s3** (the guard's recipe, the 3090 class, --no_cert plus the post-hoc fp32 cover
  certificate).
* **Readings, fixed now:**
  - both fail closure: COPY causes the loss of closure on MOD (3/3 against plain's 0/4);
  - both close: s1's failure is seed-specific.
  - One of each is reported as mixed.
* §134's verdict is unchanged either way (the registered guard rule decided it). This bears only on the mechanism.

**Note 134-7 (2026-10-07 04:50 EDT; before the 134-6 runs exist; Audit's addition).**
* **The mixed outcome, stated now:** if exactly 1 of s2 / s3 fails closure, the reading is "COPY raises the risk of an open transient region on MOD (2/3 against
  plain's 0/4), not a certainty". COPY remains non-default.
* **A "closes" reading needs an uncapped fp32 certificate** (capped false, not timed out). A capped or timed-out certificate reads as inconclusive for that seed.
* Audit approved the two runs (about $1, descriptive), deployed after §137 / §139 / §140.

**Note 137-3 (2026-10-07 05:21 EDT; a wording fix to §137's result, after Audit's reproduction).** The context-only column-sum probe per column is 0.78 / 0.15 / 0.01 / −0.03 / … The result's
"not at all from column 2 on (≤ 0.15)" should read: **0.15 at column 2, and about 0 from column 3 on.** The reading is unchanged.

**§140 result (2026-10-07 08:38 EDT; the 4×4 proof, s140_pair_4x4_s1, RTX 3090 / 595.91.07, unflagged at 2k; scored by Theory with score_s140). DROP at proof: test_acc 0.002
(train 0.001). 4×4 stops; §140 is closed.**
* **Per column:** 1.0 / **1.0** / 0.315 / 0.132 / 0.104 / 0.24 / 0.50 / 0.936. Per step given a correct prefix: 1.0 / 1.0 / 0.32 / 0.18 / 0.13 / 0.43 / 0.67 / 1.0.
  - **The first error is at column 3 in 685 / 1000 chains, and at column 4 in 259.**
  - The loss was **≈ 0 while only columns 1–2 were trained** (1k–2k), then rose as columns 3+ entered the curriculum: 0.35 at 5k, 0.54 at 9k, 0.87 at 24k.
* **Probe:** the column accumulator is readable from **h at R² 0.92**, from the **code at 0.61**. The context-only column sum is ≈ 0 (as designed: only one pair is
  visible).
* **SERIAL:** not met (donor about recipient, against chance 0.46–0.67). ext_4x8 is 0.0.
* **Reading (as registered: DROP or FAIL means "even one-digit steps with a carried accumulator are not learned at this budget"):**
  - one product per step fixes column 2 (§137 had 0.326);
  - **columns that accumulate 3–4 products fail**, the first errors fall at column 3, and the accumulator is only partly held by the code (0.61 against h's 0.92).
  - So the barrier is not only the per-segment in-pass load. **Carrying a multi-step accumulator** (sums up to about 300) **through the 8-d code across 3–4 steps**
    is not learned here.
  - Descriptive, n = 1. With §137 / §141-r1s16, the result is now "columns of ≥ 3 products fail in every design so far". §141's r4 cells are the last open test.

## 141. After §137's DROP: latent repeats per column, and a wider carrier (pre-registered 2026-10-07 05:41 EDT, after Literature's check 6bf78643 and before any implementation or run; Audit's steward ideas; the user's "revisit methods if fail to improve")

**Why.**
* §137's column-major 4×4 dropped at proof: test_acc 0.000, with the carry-dependent middle columns at chance (§137 result).
* The carry was readable from h before W_down at R² 0.69, but from the 8-d code at R² 0.24 (linear).
* Two latent-side questions follow, crossed with §140 (digit pairs, pair-only visibility):
  - **Repeats:** do extra latent steps without new input buy the per-column compute?
  - **Width:** was the 8-d code the bottleneck?
* **Prior** (6bf78643, `docs/LIT_S137_FOLLOWUP_2026-10-07.md`, new as scoped): the closest is Huginn 2502.05171 (the same injected input each recurrence
  through a carried state). Also UT / ACT / PonderNet, Ouro 2510.25741, looped transformers (2409.15647, 2502.17416), and pause tokens (§107's PAUSE-4 did
  not unblock). Repeats carried through a discrete code are new.
  - Literature's prediction is **r4 > SOFT-16**, since capacity is not binding (the column state is about 9 bits).

**Cells (§137's family, recipe and rules otherwise unchanged; 4×4; restart slots).**
* **r4f8:** r = 4 segments per column. Each is [EOT, carried code, a's digits, b's digits, column token k, **repeat-index token q = 1 … 4**], chained through the
  code, and the digit p_k is decoded only after q = 4. FSQ-8.
  - The repeat index avoids a fixed point after one repeat (Literature). r is fixed at 4, so the index never leaves the trained range.
* **r4s16:** r4f8 with **SOFT-16** (d 16, β 0.1). FSQ rounding can erase sub-level increments between repeats; the soft code does not (Literature).
* **r1s16:** §137 unchanged but with SOFT-16. **It runs only if a nonlinear code probe on §137's model shows the carry missing from the code:**
  - the probe maps each distinct code_k to its majority true carry (a lookup), fit and scored on held-out halves, on s137_col_4x4_s1;
  - the cell runs if that accuracy is < 0.9. Otherwise the code holds the carry nonlinearly, width is not the bottleneck, and the cell is not run.
  - Output: `s137_col_4x4_s1/eval_probe_percol.json` (carry_code_nonlinear_acc).
* **Rules (§137's):**
  - **M-PASS:** ≥ 3/4 slots at test_acc ≥ 0.99, SERIAL (a donor code injected after column 2 / 4 / 6, at that column's last repeat) and ρ ≤ 0.28 (127-12's clause
    for FSQ-8, 133-1's first-three form for SOFT-16);
  - **FAIL:** ≥ 2 decided slots not passing;
  - **DROP at proof:** slot 1 below 0.95 stops the cell.
  - The C4 qualifier (streamed digits) and the IN-PASS reading (137-1 (2)) carry over.
* **Descriptive:**
  - **per-repeat probes:** R² of the column's partial product sum and of the carry from code after each repeat, plus ‖Δcode‖ per repeat. A sum rising across
    repeats is the mechanism;
  - acc_by_column; 4×8.

**Readings (fixed now; crossed with §140).**
* **r4 PASS:** extra latent steps without new input buy the per-column compute. That is direct evidence that latent recurrence adds computation (C3), and r is a
  later C5 handle.
  - **With §140 PASS:** one product per step suffices whether the step sees only the pair (§140, forced C1) or the whole column (r4, unforced C1).
  - **With §140 FAIL:** whole-column visibility plus repeats works where pair-only steps do not.
* **r4 DROP or FAIL:**
  - **with §140 PASS:** the input must be decomposed (pair-only steps); repeats over the whole column do not build sums;
  - **with §140 FAIL:** multiplication is not reached by this harness at this budget, and the post-mortem reads the probes.
* **r4s16 against r4f8:** whether FSQ rounding blocks the per-repeat accumulation.
* **r1s16 (if it runs):** PASS means the 8-d code was the bottleneck; DROP means the in-pass load reading stands alone.

**Placement.**
* Proof slot per cell: r4 costs 4× the segments, about $1.2 each; r1s16 about $0.3. About $2.7 for the proofs; slots 2–4 only for cells that clear the proof.
* After §140's proof starts. The adapter extends §137's multcol (`--repeats 4 --repeat_index`, SOFT-16 flags) with §137's gates (gradflow through the repeats,
  memory, smoke_plain, Audit's diff review).
* The nonlinear probe runs first, locally, $0. Dirs `s141_{r4f8,r4s16,r1s16}_4x4_s{seed}`. Scorer: `theory/score_s141.py`, committed before any run.

**Note 141-1 (2026-10-07 05:42 EDT, before any §141 implementation or run; Audit's approval of 9e45b92b).**
* **The mechanism reading, fixed now.** Every repeat sees the whole column, so an r4 PASS could come from extra serial depth (the 4th repeat doing the whole sum)
  rather than accumulation across repeats.
  - "**Accumulation across repeats**" is claimed only if the per-repeat probe shows the R² of the column's partial product sum **rising from repeat 1 to repeat 4**,
    with ‖Δcode‖ > 0 at every repeat.
  - Otherwise the reading is "**extra latent depth suffices**": still relevant to C3, but not stepwise.
* **Implementation requirements:**
  - the repeat-index tokens are single GPT-2 tokens, disjoint from the digit and column tokens;
  - r4's SERIAL donor is injected **at the column's last repeat**, so the donor's carry is complete.

**Note 141-2 (2026-10-07 06:11 EDT, before any §141 run; Audit's code review of multcol R = 4). Which contraction unit the ρ rule uses.**
* **The rule uses ρ per column (per input item), measured at the item boundaries only.**
  - Perturb the code at a column's last repeat by ε, run the next column's R repeats, and compare at that column's last repeat.
  - Then 127-12's clause (FSQ-8) or 133-1's first-three form (SOFT-16) over those column hand-offs, with ρ ≤ 0.28.
* **Why the column, not the step:** every earlier ρ rule (S5 per op, §128 per item, §137 per column) is per input item, and the 0.28 bound was calibrated per
  item.
  - Within a column, the repeat steps carry a partial sum by design, so a perturbation can legitimately persist there (per-step ρ near 1). A per-step rule would be
    a different and stricter quantity (per-column ≈ the product of the per-step values).
  - At R = 1 the two coincide, so r1s16 is unaffected.
* **Reported beside it, descriptive:** the per-step ρ, as the harness computes it now (every repeat counted as a hand-off).
* The adapter computes the per-column form for the rule (T0 and K counted in columns). score_s141 reads `eval_contraction_fp32.json`. That file must be the
  per-column form, with the per-step form in `eval_contraction_fp32_step.json`.
* This differs from Audit's suggestion (the per-step form as the rule). The reason is above; Audit may object before any run.

**Note 141-3 (2026-10-07 06:30 EDT; before any §141 result; the r1s16 gate read, plus §137's per-column probe, descriptive).**
* **The r1s16 gate opens.** The nonlinear code → carry lookup on s137_col_4x4_s1 (held-out half) is **0.327** pooled, below 0.9. Per column: 0.946 / 0.166 / 0.110 / 0.094 /
  0.092 / 0.174 / 0.326 / 0.706. **r1s16 runs.**
* **§137's per-column probe** (`eval_probe_percol_cv.json`: ridge with λ chosen by 5-fold CV; v1's fixed λ overfit h's 768 features). R² from h / code:
  - **carry-out:** 1.000 / 0.222 (column 1), 0.356 / 0.094 (column 2), negative or unstable from h at columns 3–5 (code 0.03–0.28), 0.902 / 0.409 (column 6), 0.877 / 0.431
    (column 7);
  - **carry-in at column 8:** 0.977 / 0.922;
  - the **context-only column sum** follows the carry: about 1.0 from h at columns 1, 6 and 7, and none at 3–5.
  - Digit accuracy from h or code is about chance at columns 3–6.
* **Reading (descriptive; it fits both parts of §141's design):** two failures stack.
  - **(a)** Where the in-pass load is highest (columns 3–5: 3–4 products) the segment's h does not encode the column sum at all. It does where there are only 1–2
    products (columns 1, 6, 7). That is the in-pass load §140 and r4 address.
  - **(b)** Where h does hold it (columns 6–7, R² about 0.9), the FSQ-8 code keeps only about 0.4 linearly (0.17–0.33 by lookup). That is a W_down / code loss, which
    the SOFT-16 cells address.
  - n = 1, one failed model.

## 138. FULL WIDTH: the latent carrier at the residual width (768-d) (pre-registered 2026-10-07 07:49 EDT, after Literature's checks db3df361 / d4f0d18b (the hinge) and before any implementation or run; the user's directive "if throughout the night 64 wide latents pass, try full width latents. Revisit methods if fail to improve of course" (via Audit); the trigger fired with §133's stage-1 result)

**Why.**
* §133 stage 1: FSQ-64, NOISE-64 and **CONT-64** (uncorrected) are exact at 2,048, SERIAL and contractive at d 64. A learned tanh-bounded code of that width does not
  drift.
* §126's RAW-H, which carried the **raw, unbounded 768-d hidden state**, drifted (0.85 at 512, 0.73 at 2,048; ρ_ff 0.5–0.8).
* §138 asks whether the carrier can be as wide as the model's residual stream, with no bottleneck in width.

**Design: §133's harness exactly, with d = 768** (§128's S5 recipe, restart-on-plateau slots k, k + 100, k + 200, k + 300, 24k, bs 64, the 127-3 guard, cloud RTX 3090,
checkpoints, the host check).
* **What differs from RAW-H:** a learned projection with a bounded code, z_b = tanh(W_down(LN(h)) + shift) · HALF − OFF at width 768, then W_up. RAW-H carried h itself.
  - With COPY dropped, FULL-CONT against RAW-H isolates the **bounded learned code** as the one design change. Any further differences between §126's harness and
    §133's (the restart slots) are listed beside the result.
  - Literature db3df361: bounded projections appear only as engineering choices (CODI, SoftCoT 2502.12134, HRPO, Soft Thinking 2505.15778, nGPT 2410.01131), never
    isolated as a stability variable at matched width. FSQ at 768 has no precedent (FSQ / LFQ / BSQ stay at a few to tens of dimensions). New as scoped.
* **Changed from the plan (`docs/THEORY_FULLWIDTH_PLAN_2026-10-07.md`), before registration:** **no COPY rows**.
  - §133's passes used restart slots, not COPY, so width stays the only change.
  - §134's guard showed COPY is not task-agnostic (MOD closure).
  - Entry is handled by the restart slots, as at d 64.
* **GROW-768 is not included:** §136 is not yet scored. It would be a separate note after §136 reads.

**Arms.**
* **FULL-FSQ:** d 768, hard FSQ with 8 levels.
* **FULL-NOISE:** d 768, continuous, with dither U(±0.25) at training hand-offs after 2k.
* **FULL-CONT:** d 768, continuous, no correction. This is the direct contrast with RAW-H.
* **FULL-HINGE** (candidate 6; Literature d4f0d18b): FULL-CONT plus a **local contraction hinge**.
  - **Base carrier: CONT** (FULL-CONT). From 2k, the pre-quantisation z of a carried code (continuous here) is perturbed by δ, with ‖δ‖ = ε.
  - **ε reference (Audit):** state centroids are the mean e_j per true state over a fixed probe batch at 2k. ε = **0.15 × the median, over states, of each centroid's
    distance to the nearest other centroid**. It is measured once at 2k and then fixed.
  - Both are pushed through one segment, and **max(0, ρ̂ − 0.3)** is penalised, with ρ̂ = ‖Δz_{j+1}‖ / ‖δ‖, weight 1.
  - **δ is drawn half along a random direction and half along the difference between the state's centroid and its nearest other centroid** (Literature (3d)).
  - **Hinged hand-offs are subsampled** (1 in 4 per batch) to cap the extra forward cost.
  - It is local, not the global penalty rejected for §133.
* **Proof first:** slot 1 per arm. An arm whose slot 1 is accepted below 0.95 at 2,048 is dropped (as 133-2).

**Rules (fixed now; §133's stage-1 rule).**
* **PASS(arm):** ≥ 3/4 slots exact at 2,048, SERIAL and contractive (ρ ≤ 0.28: 127-12's clause for FULL-FSQ, 133-1's first-three form for the continuous arms).
  - **At 768, exact recovery after a random perturbation of all coordinates becomes rare**, so 127-12's ρ₁ branch will seldom apply. FULL-FSQ is then read on the
    clause's geometric-mean branch over non-zero hand-offs (ρ_ff / rho_nonzero), stated now (Literature (3a)).
  - contraction() and the tanh before rounding run in fp32 (Literature (3e)).
* **The user's "fails to improve → revisit", fixed now:**
  - **(i) Stability:** if no full-width arm passes, full width is not adopted. A post-mortem (ρ, prefix 1 / 2, codes per state, dimension use) follows, and at most one
    revised method, as a fresh registration.
  - **(ii) Benefit:** S5 cannot show gain. A passing full-width arm is compared with the passing 64-d arm on a capacity task (S5×4, §133 stage 2, or multiplier-major
    multiplication). Improves = it passes where 64-d fails, or a higher exact rate at an equal budget. Otherwise 64-d stays the operating width ("stable but no gain").
    - **That benchmark is §133 stage 2 (S5×4), not yet approved.** Its proposal includes the full-width passers, so one capacity run serves both (Audit).
* **Capacity task and budget for rule (ii), named now (Literature (3f)):** §133 stage 2, **S5×4** (CAP-PASS = long_acc ≥ 0.99 at 512 in ≥ 3/4 slots), with the 64-d
  passers, the full-width passers and the FSQ-8 / FSQ-8-L16 controls. About $15–25, proposed to Audit separately.
* **HINGE readings:**
  - HINGE PASS where FULL-CONT fails: the local hinge removes full-width drift;
  - both PASS: not needed on S5 at 768;
  - HINGE FAIL where FULL-CONT passes: the hinge harms.

**Descriptive (Audit).** At d 768, codes per state means nothing for the continuous arms, and the FSQ grid is 8⁷⁶⁸. In their place:
* the **effective rank** of the carried code;
* **descriptive ρ along the state-separating directions** (perturbations along centroid differences, plus power iteration on the hand-off Jacobian, 1705.10941),
  reported and not gated. A random-δ ρ at 768 is a bulk average and could read contractive while a few directions expand (Literature (3b));
* **FULL-FSQ: codes per state and the dimensions actually used** (Literature (2)), plus the initial code diversity at step 0 and at 2k (Literature (3c));
* **within-state spread over between-state distance** (the mean distance of codes to their state centroid, over the median distance between centroids);
* angular drift by position, grid share (FULL-FSQ), prefix 1 / 2 / 4 / 8, and dimension use (the variance share beyond the first 64 dimensions).
* CERT is not part of PASS (as §133). An fp32 certificate is attempted for FULL-FSQ only, with "inconclusive by time" allowed.

**Placement.**
* Proofs: 4 runs, about $2. Full slots for the arms that clear their proof: up to about $6 more. Audit approves each stage against the credit.
* Implementation: the experiments session (fsq120 `--d_q 768`; `--hinge --hinge_eps_frac 0.15`). Gradflow under CUDA bf16: the hinge term reaches W_down and the trunk,
  and its ρ̂ is computed on the pre-quantisation z. **Memory is checked at L 32 on a 3090 with the micro rule** (W_down / W_up 768 × 768, plus the hinge's extra
  forward).
* Dirs `s138_{full_fsq,full_noise,full_cont,full_hinge}_s{seed}`. Scorer: `theory/score_s138.py`, committed before any run.

**Note 138-1 (2026-10-07 07:49 EDT, before any §138 run; Audit on Literature db3df361).**
* **(1) The ρ reading at d 768:** the rounded arm (FULL-FSQ) uses 127-12's geometric-mean branch over non-zero hand-offs (**rho_nonzero / ρ_ff**). exact_recovery_share
  collapses with d, so the ρ₁ fallback would be degenerate. Already in §138, restated here as the binding reading.
* **(2) The centroid-direction ρ** (and the power-iteration estimate) is **descriptive only**, never gated.
* **(3) The actual differences between FULL-CONT and §126 RAW-H, listed explicitly:**
  - **(a) The carried vector:** a learned W_down projection to a **tanh-bounded 768-d code**, then W_up, against RAW-H's **standardised raw last hidden h** (std(h), no
    learned projection, no bound).
  - **(b) The harness:** §133's **restart-on-plateau slots** (§128) against RAW-H's single run per seed (§120's harness, before §128's slots).
  - **(c) The same:** the S5 FINAL recipe, the length curriculum, 24k steps, bs 64, the RTX 3090 class.
  - There is **no COPY in either.**
  - A FULL-CONT PASS where RAW-H drifted is therefore read as "the bounded learned code (with restarts for entry)". RAW-H entered 4/4 without restarts, so (b) bears on
    entry, not on drift.

**Note 141-4 (2026-10-07 08:30 EDT; interim, one cell decided; scored by Theory with score_s141).** **r1s16 (§137's column task with SOFT-16) is dropped at proof:** s1 test_acc 0.000
(train_acc 0.000).
* Per column: 1.0 / **0.663** / 0.19 / 0.121 / 0.099 / 0.104 / 0.193 / 0.966, against FSQ-8 r1's 1.0 / 0.326 / 0.124 / 0.103 / 0.077 / 0.104 / 0.105 / 0.953.
* fp32 rho_k3 is 0.090.
* **Reading (descriptive, consistent with 141-3's stacked failures):** the wider code roughly doubles column 2 (2 digit products) but leaves columns 3–7 (3–4 products)
  at about chance. Width helps where the in-pass load is small, not where it is large.
* The decisive cells are r4f8 and r4s16 (repeats), still running. The 2×2's width-only cell reads DROP.

**Note 139-4 (2026-10-07 08:38 EDT; the 1-host proof on b126; 139-1 (c)'s stop rule).** Audit's raw read: **A s1 stopped at 4k by the copy gate** (copy_acc 0.0, copy_loss 8.5). By 139-1 (c)
**the stage stops before seeds 2–4**, and this is the report.
* In the same proof, A′ s1 finished 24k with copy_acc 1.0 but a final loss of 10.94 (it looks failed), and B s1 was at 22.9k with a loss of 6.50.
* COPY learns in A′ and B but not in A, so A's log is to be checked for divergence or nonfinite steps before reading anything into it.
* The three proof runs are scored when their files land (descriptive; no seeds 2–4).

**Note 139-5 (2026-10-07 08:41 EDT; corrects 139-4's "COPY learns in A′ and B but not in A", from the experiments session's read of b126's logs).**
* **There is no divergence in A:** 0 nonfinite steps.
* **The copy loss falls slowly and alike in all three arms:** A 16.2 → 8.5, B 15.6 → 7.1, A′ 16.0 → 9.2 (500 → 4k).
* **copy_acc at 4k is about 0 everywhere:** A 0.0, B 0.0, A′ 0.004. It is 0.02–0.03 at 5k.
* **So the gate tripped on slow copy learning of 8-digit products through SOFT-16, not on a fault of A.** Only A carried the 4k gate.
  - The 0.9-at-4k threshold, set from FSQ-8 / S5's copy (one of 120 state texts), was unreachable for this copy target.
  - **The stop stands as registered** (seeds 2–4 are not run). A′ and B's proof runs (24k, 0 nonfinite) are read descriptively when their evals land.

**Note 138-2 (2026-10-07 09:20 EDT, before any FULL-HINGE run; Audit's code review: HINGE on hold, the other three arms deploying). FULL-HINGE is made label-free.**
* **The issue:** the implemented hinge drew its δ direction from the gold S5 state at every training hand-off, and built its ε reference from gold-label centroids. That is
  per-step simulator state during training: not C4-clean under the revised wording, and a second difference from FULL-CONT besides the hinge.
* **Fixed now (Audit's option (a), plus a label-free ε):**
  - **δ is a random direction only** (the centroid-difference half is dropped);
  - **ε = 0.15 × the median, over the probe batch's hand-off codes at 2k, of each code's distance to its nearest distinct code** (distance > 10⁻³). The nearest distinct
    code tracks the inter-state spacing without labels.
  - Everything else is as §138: the CONT base, from 2k, max(0, ρ̂ − 0.3) on the pre-quantisation z, 1 in 4 hand-offs, weight 1.
* **Labels remain only in the descriptive measures** (the centroid-direction ρ, the spread ratios), which never feed training.
* **FULL-HINGE now differs from FULL-CONT only by the label-free hinge.** A HINGE PASS where FULL-CONT fails reads "a label-free local stabiliser removes full-width drift".
* Literature's concern that random δ can miss state-separating directions stays descriptive (the directional ρ is reported).

**Note 138-3 (2026-10-07 09:21 EDT, before any FULL-HINGE run; Audit on 138-2). The label-free ε reference is corrected.**
* 138-2's "nearest distinct code" fails on the CONT carrier: same-state codes are not identical, so a code's nearest distinct code is almost always a **same-state** neighbour.
  That measures within-state spread, not inter-state spacing.
* **Fixed now (Audit's option (i)):** **ε = 0.1 × the median distance between random pairs of hand-off codes** on the fixed probe batch at 2k. That is a global scale; with
  120 states, most pairs are cross-state. It is measured once, label-free.
  - 0.1 of a typical cross-state distance is roughly 0.15 of the nearest-centroid spacing, the low end of Literature's 0.1–0.25 range.
* **Logged (descriptive, labels used only here):** ε, ε / the measured within-state spread, and ε / the median nearest-centroid distance.
* **Reading:**
  - if ε is several times the within-state spread and well below the inter-state spacing, the hinge probes contraction "locally at the inter-state scale";
  - if ε is at or below the within-state spread, a HINGE result is read as an infinitesimal-Jacobian constraint, and stated so.

**Note 138-4 (2026-10-07 09:30 EDT, before any FULL-HINGE run; the implementation 1d76fc3c).** 138-3's "hand-off codes" means the **pre-quantisation z_b**, the space the hinge perturbs (not
e_j = W_up(code)). ε = 0.1 × the median distance between 20k random pairs of z_b on the probe batch at 2k, as implemented. The ratios (ε / within-state spread, ε / nearest-
centroid distance) are logged in the same space. At init these are 0.13 / 2.09; that is uninformative, since the states are not yet separated.

**Note 140-3 (2026-10-07 09:51 EDT; descriptive; `s140_pair_4x4_s1/eval_probe_acc.json` and the local 5k prototypes). Where the digit-pair accumulator breaks.**
* **By accumulator magnitude:** R² code / h is **0.49 / 0.87 below 100**, then **−4.1 / 0.05 for 100–200**, and negative above 200.
  - Accumulator values of 100 or more are **not represented even in h**. The failure is learning to represent the larger sums, which occur in the 3–4-pair columns, not
    only the 8-d code dropping what h has.
* **By step within the column:** code 0.51 / 0.51 / 0.69 / 0.73, h 0.96 / 0.84 / 0.81 / 0.75 (1st–4th pair). The flush step is 0.95 / 0.98.
* **Local 4×4 prototypes to 5k** (seeds 501 / 502 on the 4060 / 3070; descriptive), per-digit loss at c_max 4 (3k–5k):
  - column r1 (proto137): 0.83–0.98;
  - **column r4 with repeats (proto141): 0.79–0.90, about the same as r1;**
  - digit pair (proto140): 0.29–0.58, where §140's cloud proof later rose to 0.87 once the 3+-pair columns entered.
  - Repeats show no early advantage. §141's cloud r4 proofs decide.

**Note 134-8 (2026-10-07 11:07 EDT; the 134-6 follow-up read by Theory from the raw files; Audit-checked). Both extra MOD + COPY seeds close, so s1's failure is seed-specific.**
* s134_copy_mod_s2 (driver 570.181): fp32 CERT holds, **236 reachable**, Moore 5, 0 violations, uncapped; long_acc 1.0, ext 1.0, copy_acc 1.0.
* s134_copy_mod_s3 (595.71.05): holds, **238 reachable**, Moore 5, 0 violations, uncapped; 1.0 / 1.0 / 1.0.
* **Reading (as pre-stated in 134-6 / 134-7, "both close"):** s1's loss of closure is seed-specific. COPY closes MOD in 2/3 seeds against plain's 4/4, so the evidence that COPY
  causes open transients is **weak** (one seed).
  - **§134's registered guard verdict stands as decided** (not adopted as the default). A later default decision could revisit it with more seeds.

**Note 139-6 (2026-10-07 11:18 EDT; descriptive: the 1-host proof runs after the 139-4 / 139-5 stop; scored with score_s139). No arm learns multiplier-major 4×4.**
* **A s1:** stopped at 4k by its copy gate (139-5: slow 8-digit copy, no fault). Not read further.
* **A′ s1** (strict C4: L fixed at 4, final answer plus COPY only): test_acc 0.000, train_acc 0.000, accuracy by length 0 / 0 / 0 / 0, fp32 ρ (rho_k3) 0.56. COPY itself
  was learned by the end (copy_acc 0.996).
* **B s1** (A plus the teacher-text warm start, faded to 0 by 4k; CE mode, 139-2): test_acc 0.000.
  - Accuracy by length is **0.586 / 0.017 / 0.001 / 0**: the first step (a × b₁, a 4 × 1 product) is partly learned, and nothing beyond it.
  - The frozen readout's S_j accuracy at the end is 0.03. ρ 0.65.
* **Reading (descriptive, n = 1 per arm):** on multiplier-major multiplication, neither final-answer training (A′) nor a teacher-text warm start that fades by 4k (B)
  gets past the first hand-off.
  - This matches §137 / §140 / §141-r1s16: every design so far fails once a step must carry or accumulate a multi-digit number.
  - The user's DISTILL-2 question has no positive reading on this task. **§139 is closed at the proof stage.**

**Note 141-5 (2026-10-07 14:56 EDT; interim; scored by Theory with score_s141). r4f8 is dropped at proof: s1 test_acc 0.000 (train 0.000).**
* **Per column:** 1.0 / 0.371 / 0.137 / 0.099 / 0.073 / 0.104 / 0.100 / 0.813, against §137's r1 FSQ-8 at 1.0 / 0.326 / 0.124 / 0.103 / 0.077 / 0.104 / 0.105 / 0.953.
  Four latent repeats per column change nothing where a column holds 3–4 products.
* **The per-digit loss follows §137's:** 0.45 at 2k, 0.93 at 3k (L 4), about 1.25 from 9k to 24k.
* **Per-repeat probe (141-1; descriptive for a DROP):** after repeats 1–4, the column's partial-product sum is readable from the code at R² 0.13 / 0.20 / 0.28 / 0.21, the
  carry at 0.22 / 0.23 / 0.27 / 0.21, with ‖Δcode‖ 4.9 / 2.0 / 1.5 / 3.0. The repeats move the code and lift the sum slightly through repeat 3, then lose it at repeat 4.
  Nothing approaches a readable sum.
* **SERIAL** is not met (row-level donor 0.000–0.001). 4×8: 0.0.
* **Reading (as registered: r4 DROP with §140 DROP):** multiplication is not reached by this harness at this budget. Extra latent steps without new input do not buy the
  per-column compute. r4s16 (b131, about 19:00) completes §141. §142's post-mortem table shows it as pending.

**§138 result (2026-10-07 15:18 EDT; proof slot 1 per arm, cloud RTX 3090s, drivers recorded per run; scored by Theory with score_s138). All four full-width arms are dropped at proof
(ext_2048 0.0 on slot 1, an accepted run unflagged at 2k). Rule (i) applies: full width is not adopted; post-mortem below; one revised method follows as §143 (below).**

| Arm (d 768, s1) | ext_2048 | acc at length 1 / 2 | fp32 ρ (rule form) | ρ along state directions / Jacobian top | eff rank | spread / between | final loss |
|---|---|---|---|---|---|---|---|
| FULL-FSQ | 0.0 | 0.0095 / 0.0 | 0.088 | 0.21 / 0.67 | 2.8 | 17.9 | 4.82 |
| FULL-NOISE | 0.0 | 0.0485 / 0.001 | 0.070 | 0.20 / 0.66 | 4.8 | 13.8 | 4.78 |
| FULL-CONT | 0.0 | 0.0685 / 0.0025 | 0.207 | 0.43 / **1.14** | 3.5 | 21.1 | 4.64 |
| FULL-HINGE | 0.0 | 0.0415 / 0.001 | 0.042 | 0.13 / 0.41 | 2.3 | 22.6 | 4.79 |

* **Post-mortem (rule (i)'s measures):**
  - **One path, then escape.** All arms are identical to 2k (same seed and base; the FSQ ramp and the dither start at 2k): loss 3.87 at 2k, not flagged by 123-3. Each then
    climbs back to ≈ ln 120 (4.6–4.8) as L_max grows (L 4 at 2k, 8 at 5k) and stays there to 24k. Accuracy at length 1 ends at chance or near it: an F2a-like
    contractive map to a state-blind code, not drift.
  - **ρ is in band but the code holds no state:** fp32 ρ 0.04–0.21 by the rule, while the within-state spread is 14–23 times the distance between state centroids.
    FULL-CONT's hand-off Jacobian expands along its top direction (1.14), Literature's (3b) case.
  - **Dimension use:** effective rank 2.3–4.8, with 92 % of the variance beyond the first 64 dimensions. FULL-FSQ uses all 768 dimensions and 8,132 codes per state
    (`codes_per_state`): every chain gets its own code.
  - **The carried norm doubles by 2k** (25 → 53; CONT-64: 26 → 21).
  - **The hinge never engaged.**
    - ε at 2k is 2.79: 0.155 of the within-state spread (18.0) and 2.75 times the median nearest-centroid distance (1.01). The states were not separated at 2k.
    - The median ρ̂ was 0.047–0.05 throughout, far below the 0.3 margin, with an active share of 1.6 % at 2.1k and 0.008 % at 24k.
    - That is Literature's random-direction warning (3b): the bulk contracts, so a random-δ hinge is inert.
  - **The same failure at 768 without the learned map:** §126 RAW-H (raw h) cleared the 2k flag 4/4 at lower loss (1.65–2.06), then escaped on s1 / s3 / s4 (ext 0.0) and
    learned only on s2 (0.73 at 2,048, drifting). Of 8 decided 768-d runs, 1 learned. At d 64 the same harness's accepted runs learn (§133 stage 1).
  - **Read:** at full width the failure is **learning from scratch** (escape after the 2k flag), common to the raw and the learned bounded carriers, not instability of a learned
    code. Every proof is seed 1 (one path to 2k), so this is n = 1 per arm under the registered proof rule; RAW-H's 3/4 escapes are the independent support.
* **The one revised method (rule (i)), §143 FULL-GROW:**
  - enter at d 8 with COPY, and grow to 64 exactly as §136's 4/4 passer;
  - then widen 64 → 768 with a μP-scaled learning rate.
  - It removes exactly the failing step: learning at 768 from scratch.
* The benefit question (rule (ii)) is moot unless §143 passes. S5×4 proceeds with the 64-d passers.
* **Scorer fix, disclosed:** `score_s138.py` still expected `hinge_eps_frac` 0.15 (§138's first text) and flagged the run as an args error. 138-3 had set 0.1 before
  any HINGE run, and the run used 0.1. The expected value now follows 138-3. No threshold or rule changed.

## 142. MULT-READ: a per-step reader of the running sum on a wide continuous carrier, then ICoT-style removal to final-answer only (pre-registered 2026-10-07 14:56 EDT, after Literature's checks bb1f4eeb / 602541ab and Audit's review, before any implementation or run; the user's "yes go ahead" (via Audit); registered with the multiplication post-mortem once §141's r4f8 cell read, r4s16 pending (Audit's ruling))

**Why: the multiplication post-mortem (§98, §100, §137, §139, §140, §141).**

| § / arm | Design | Carrier | Per-step S targets | 4×4 test_acc | Where it fails |
|---|---|---|---|---|---|
| 98 GEN-M | multiplier-major, k = 4 in-context slots | 768-d slots | text fork that also sees the context | 0.163 | slot 3 decode 0.80–0.92 |
| 98 GEN-M-MASK | the same; the fork reads the slot only | 768-d slots | from the slot alone | 0.034 | slot 3 decode 0.22–0.25 |
| 100 GEN, k = 2 | text re-embedded every 2 ops | 768-d slots | text fork | **1.000** | none (a hybrid: text checkpoints) |
| 137 | column-major, carried carry | FSQ-8 | none | 0.000 | columns 2–7 near chance; carry R² h 0.69, code 0.24 |
| 141 r1s16 | the same | SOFT-16 | none | 0.000 | column 2 0.663; columns 3–7 at chance |
| 141 r4f8 | 4 latent repeats per column | FSQ-8 | none | 0.000 | columns as §137 (0.37 / 0.14 / 0.10 / 0.07 …); the sum's R² ≤ 0.28 after any repeat (141-5) |
| 141 r4s16 | the same | SOFT-16 | none | pending (about 19:00) | an addendum when it lands |
| 140 | one digit pair per step, carried accumulator | FSQ-8 | none | 0.002 | first error at column 3 (685 / 1,000); sums ≥ 100 absent even from h |
| 139 A′ | multiplier-major, final answer at L 4 | SOFT-16 + COPY | none | 0.000 | nothing but COPY learned |
| 139 B | the same + the teacher's S_j, faded by 4k | SOFT-16 + COPY | a linear readout, 0–4k | 0.000 | S_1 0.586; the readout CE stayed 1.0–1.2 nats per digit while on |

* **The common failure:** every latent design fails at the first step that must build or hold a multi-digit running quantity in one vector. Per-step supervision has
  only worked with text checkpoints (§100, k = 2).
* **The teacher does not hold S_j in a vector either:** TC-M's S_j is linearly decodable from its own hidden state at only 0.174 exact (139-2). Text CoT spreads each
  sum over one pass per digit; a latent carrier must compute and compress it in one segment pass.
* **What §142 changes, against the nearest arms:**
  - **GEN-M-MASK** (the closest prior arm, 0.034): a Markov, bounded, dithered 64-d code between separate segment passes, instead of in-context slots; a length
    curriculum; twice the budget; S_j as the only target.
  - **§139 B:** the reader is the shared answer decoder (nonlinear), not a linear readout; it stays on for the whole run instead of fading by 4k; NOISE-64 instead
    of SOFT-16; no COPY.
  - **Already in the curriculum:** shorter problems supervise S_L at hand-off L in other rows (139-3's prefix labels). The reader makes that dense: every hand-off
    of every row.
* **Prior** (bb1f4eeb / 602541ab, `docs/LIT_MULT_READ_REMOVAL_2026-10-07.md`): the running-sum auxiliary loss (2510.00184), here decoded from a Markov code; removal
  as in ICoT-SI 2405.14838 (earliest-first, removal smoothing, an optimiser reset) and Coconut. New as scoped.
* **Expectation, stated now:** given GEN-M-MASK and §139 B, a DROP at proof is the likelier outcome. **If A drops at proof, multiplication on the task-agnostic
  carrier is parked** (no further fix arms). The project's multiplication result then stays §100's hybrid (1.000 with text checkpoints every 2 ops).

**Task.** fsq120 `--family mult_row --task 4x4` (boxes/multrow.py): segment j = [EOT, carried code, a's digits, b_j, position j]; S_j = a × (b_1 … b_j).
* The bench 4×4 train split (commuted test pairs removed); the 1,000 published test items.
* Curriculum: L ~ U(1, L_max), with L_max 1 / 2 / 3 / 4 at 0 / 2k / 5k / 9k (§139 used 0 / 1k / 2k / 3k).

**Carrier.** NOISE-64 (`--d_q 64 --no_round --dither 0.25`: continuous, dither U(±0.25) at training hand-offs after 2k, clean at test; §133 stage 1 PASS 4/4).
No COPY (139-5: 8-digit copy learns slowly). No restarts.

**The reader.**
* At every hand-off j < L, the shared answer decoder decodes S_j from code_j alone, in the answer format (8 digits, least-significant first). Nothing decoded is
  fed back.
* **Targets:** the TC-M teacher's (`m98_tc_4x4_s1`) own greedy S_j, precomputed once for the train split and cached (label_source "teacher_generated"; format
  valid 1.000 and equal to gold on 4,096 train problems, 139-2). C4 as reworded. A row with invalid teacher text gets no reader term at that hand-off.
* **Loss:** CE_final + the mean over hand-offs j < L of CE_j. Every term is over the same 8 answer-format tokens, so no S_j dominates (Literature).

**Arms.**
* **A (READ):** the reader for the whole run, **48k steps** (about 3.8 epochs: twice §98's budget, still far below ICoT-SI's; Audit (2)).
  - **Removal gate:** once L_max is 4, greedy final-answer exact match on a fixed 512-item train subset at L 4 is checked every 1k steps. The first check ≥ 0.95
    saves `gate.pt` and `s142.gate_step`.
* **B-EARLY (the primary removal arm; ICoT's order):** forked from A's `gate.pt` (same seed), 20k steps.
  - The reader targets go **S_1 at +0, S_2 at +4k, S_3 at +8k**; final answer only from +8k to +20k.
  - Each removal resets AdamW and re-warms the LR over 200 steps (1e-4, cosine over the fork). 10% of rows preview the next removal (ICoT-SI's smoothing).
  - **Every fork row is a 4×4 problem** (`--fixed_L 4`, no prefix problems). Otherwise the curriculum's shorter problems would still supervise a removed S_j.
    After the last removal the only target is the 4×4 product, as in ICoT-SI's final stage.
* **B-LATE (descriptive; Audit's original order):** the same fork, removing S_3, then S_2, then S_1.
* **CTRL (descriptive, s1 only, after A s1's proof):** A without the reader (final answer only), with the same carrier, curriculum and budget. It shows whether the
  reader, rather than the carrier or the budget, made A learn.
* A seed whose A run never opens the gate gets no B runs ("not reached").

**Rules (fixed now; seeds 1–4).**
* **PASS(arm):** ≥ 3/4 seeds with test_acc ≥ 0.99 on the 1,000 items, **SERIAL** and **contractive** (133-1's first-three form, rho_k3 ≤ 0.28, fp32 on the host).
  - **SERIAL:** inject a donor's code after segment 2. Donor ≥ 0.90 and recipient − chance < 0.10, against donor S_2 + a_rec × (b_rec,3 b_rec,4) × 100 (exact).
  - B arms are read on the fork's final (final-answer-only) model. A "not reached" seed counts as not passing.
  - FAIL: ≥ 2 decided seeds not passing. **DROP at proof: A s1 test_acc < 0.95 stops the section** (no B, CTRL or further seeds).
* **Readings:**
  - **A PASS:** a per-step reader of the running sum makes multiplication learnable through a Markov latent code (C4 as reworded).
    - **C1:** b_1 … b_(j−1) are not visible later, so the code must carry the prefix. SERIAL shows it carries the running sum, not the b-prefix shortcut.
  - **A DROP or FAIL:** "not acquired within budget" (about 3.8 epochs against ICoT-SI's ~200), not impossibility. Multiplication on the carrier is then parked.
  - **B-EARLY PASS:** exact multiplication survives ICoT-style removal to final-answer only. **Not strict C4** (trained through CoT). It is comparable to ICoT-SI's
    protocol, with a latent carrier and C1 checked by SERIAL.
  - **B-EARLY FAIL with A PASS:** removing the per-step targets loses the skill; retention says which S_j goes first.
  - **B-LATE against B-EARLY** (descriptive): whether the order matters.
  - **Exact without SERIAL:** in-pass or shortcut, not a carried S_j (137-1 (2)).
* **Descriptive:**
  - SERIAL before (the gate model) and after removal;
  - under the donor swap, the **b-prefix-consistent** rate (the named shortcut: the code carries the donor's b-prefix and the step recomputes with the recipient's
    a), and the operand-consistent rate;
  - retention of every removed S_j (reader accuracy per hand-off at each stage);
  - ridge probes of S_j and the b-prefix on code_j;
  - 4×8 (positions 5–8 unseen).

**Placement.**
* **Proof:** A s1 on one cloud 3090 with checkpoints, about $1 (48k steps at about 0.4 s; the teacher precompute about 10 min).
* **After a proof ≥ 0.95:** A s2–s4, B-EARLY s1–s4, B-LATE s1 and CTRL s1, about $6–7 more, with Audit's approval against the credit. That exceeds the approved
  $3–5 because the proof runs twice as long (Audit (2)).
* 5×5 only after a 4×4 PASS, with a new approval.
* **Implementation (the experiments session):** multrow gains:
  - `--no_round --dither` (NOISE-64), `--reader`, the teacher-target cache, `--lmax_steps`;
  - the gate (`gate.pt`, `s142.gate_step`);
  - the fork (`--fork_from --fork_steps --removal early|late --removal_every 4000 --rewarm 200`);
  - stage evals (`s142.fork_start`, `s142.stages`) and the donor's b-prefix / operand rates.
* **Gates:**
  - gradflow under CUDA bf16 (every reader term reaches W_down and the trunk through the code; a removed target gets exactly 0 gradient);
  - memory at L 4 with the reader;
  - smoke_plain;
  - Audit's diff review.
* Dirs `s142_{a,bearly,blate,ctrl}_4x4_s{seed}`. Scorer: `theory/score_s142.py`, committed with this registration.

**Note 142-1 (2026-10-07 14:59 EDT, before any §142 run; the experiments session's implementation questions). Two implementation details fixed.**
* **4×8 (descriptive) is `ext_4x8_low8`.** With an 8-digit multiplier, S_5 … S_8 reach 12 digits, beyond the 8-token answer format. The measure is exact match of
  **S_j mod 10⁸** at j = 5–8, on 1,000 random 4×8 items, with positions 5–8 unseen in training.
  - The low 8 digits of S_j depend only on S_(j−1) mod 10⁸ and a × b_j × 10^(j−1) mod 10⁸, so a carrier that tracks them suffices. It remains a fair length-extrapolation
    probe.
* **Teacher targets are precomputed once** (as registered) into a cache: the teacher's greedy S_j digits and validity for the deduplicated train split.
  - The cache is keyed by the split's sha and the teacher's model.pt sha; multrow refuses on a mismatch.
  - args record label_source "teacher_generated" and the cache sha.
  - The mismatch and invalid shares are computed over the whole cache.
  - **139-2's CE-mode gate applies:** teacher format validity ≥ 0.99 on the cache, or the section stops and reports before any run.

**§142 result (2026-10-07 18:52 EDT; the A s1 proof, 48k steps, RTX 3090 / 565.57.01; scored by Theory with score_s142). DROP at proof: test_acc 0.002. The section stops (no B forks,
CTRL or further seeds). Reading as registered: "not acquired within budget".**
* **By step** (the reader on 4-step test chains): S₁ **1.0** / S₂ 0.352 / S₃ 0.052 / S₄ 0.002. The 4 × 1 first step is learned exactly; nothing past the second hand-off.
  - Against §139 B's 0.586 / 0.017 / 0.001, the reader on NOISE-64 helps the first two steps only.
* **The removal gate never opened** (gate_step none), so B-EARLY / B-LATE did not run, as registered.
* **The loss was still falling** (4.7 at 30k → 3.66 at 48k), consistent with "not acquired within budget": about 3.8 epochs against ICoT-SI's ~200.
* **SERIAL is untestable**, since the model fails at the later steps itself: donor / recipient / b-prefix / operand rates all 0.0 (chance 0.001).
  - The log-magnitude probes do not separate S_j from the b-prefix (R² 0.98 / 0.90 / 0.78 against 0.98 / 0.91 / 0.82), because log S_j = log a + log b-prefix.
* fp32 ρ (rho_k3) 0.581: not contractive.
* **As registered (§142, 142-2): the bounded single-vector carrier's multiplication line is parked.** It is the closest repeat of §121's READ, at twice the budget.
  - The user-directed §144 / §145 and the program's new architectures (§147, §150) continue as registered.
  - §148's cross-design finding applies: the middle-column wall appears in every format at this budget.

## 143. FULL-GROW: full width by staged growth from an entered 64-d carrier (§138's one revised method; pre-registered 2026-10-07 15:18 EDT, after Literature's checks 9dc25047 / 5b22bd7f and Audit's review, before any implementation or run)

**Why (§138's post-mortem, above).**
* Every 768-d carrier tried fails at **learning from scratch**: it escapes after the 2k flag. That covers §138's four proofs and §126 RAW-H on 3 of 4 seeds.
* §136 shows that growth from an entered narrow code is safe at 64 (COPY+GROW-NOISE 4/4), and that the new dimensions carry the state.
* FULL-GROW removes the failing step: the carrier enters at d 8 (with COPY), grows to 64 exactly as §136's passer, and only then to 768.
* **Prior** (9dc25047, `docs/LIT_S143_FULLGROW_2026-10-07.md`):
  - there is no ~100× single-step growth; practice is about 2× per stage (Net2Net, bert2BERT, LiGO, MSG 2305.02869, FLM-101B 2309.03852, ProGAN);
  - zero-init outgoing weights are fine (ReZero 2003.04887, GradMax); the pitfall is Adam's step at a large fan-in (μP 2203.03466);
  - new as scoped.

**Design: §136's COPY+GROW-NOISE exactly, plus one second widening.** The base is fsq120 S5 FINAL, FSQ with 8 levels, COPY rows f 0.1 from step 0, the length curriculum,
p_hard 2000 → 4000, 24k, bs 64, no restarts and no plateau stop, RTX 3090 class, checkpoints and the host check.
* **Switch 1 (8 → 64):** at THRESHOLD (the first logged step ≥ 2,000 with loss ≤ 2.5). The 56 new dimensions are continuous with dither U(±0.25).
  - Each seed is therefore **identical to `s136_cnoise_s5_s{seed}` (PASS 4/4) up to switch 2**, on the same 3090 / driver. The scorer checks the logged losses for
    identity. A mismatch makes the comparison unpaired, and it is stated. The readings use each run's own pre-switch-2 state either way.
* **Switch 2 (64 → 768):** at the first logged step ≥ 6,000 (after the rounding ramp and the L_max 8 step) that is ≥ 1,000 steps after switch 1 and has loss ≤ 2.5.
  §136's GROW-NOISE runs would fire at 6.0–6.1k.
  - The 704 new dimensions are continuous with dither U(±0.25).
  - W_up's new columns start at 0 with a 500-step fade-in. The new parameters form their own Adam group with a 300-step warm-up.
  - **μP-style learning rate:** the 704 new W_up columns train at **64/768 × the base lr**. The old 64 keep the base lr, so the pairing holds, and W_down is unscaled
    (its fan-in is still 768). Under Adam the output change per step grows with fan-in (Literature 9dc25047 (b), 5b22bd7f (1)). The per-group effective step size is
    logged.
  - **The state right before switch 2 is measured on the host:** accuracy at 512, fp32 ρ, directional ρ, the Jacobian top and the carried norm. The checkpoint is then
    deleted.
* A seed whose second trigger never fires stays at 64 and does not count as full width.

**Rules (fixed now; §138's bar).**
* **PASS:** ≥ 3/4 seeds exact at 2,048 (ext_2048 ≥ 0.99), SERIAL, and contractive by the **e-space** first-three form ≤ 0.28.
  - **The e-space ρ:** fp32 contraction() perturbs all 768 coordinates of z_b as usual, but measures every distance as ‖Δe‖, with e = W_up(z) the carried embedding.
  - **Why e-space:** while W_up_new ≈ 0 a z-space ratio is deflated by perturbation that does not propagate (Literature 5b22bd7f; Audit). Once the new dimensions are
    used, their perturbations count.
  - The z-space rho_k3, the 64-block e-space ρ and the Jacobian top are reported beside it.
  - §136's 4/4 count is reported beside the verdict.
* **DROP at proof:** s1 below 0.95 at 2,048 stops the section.
* **The user's rule, explicit:** if §143 fails or drops, **full width stops**. 64-d stays the operating width, with no further fixes.
* **Readings per seed (fixed now; against the same seed's state right before switch 2):**
  - **LOCATED at the 64 → 768 step:** exact before switch 2 (accuracy at 512 ≥ 0.99), not exact at the end.
  - **INHERITS FULL-CONT's expansion:** a failing seed whose Jacobian top ρ at the end is > 1, or whose **carried-embedding norm ‖e‖ = ‖W_up z‖** at the end is ≥ 2× its
    value right before switch 2. The code's own norm would double by construction (704 new bounded dimensions), while ‖e‖ is preserved at the switch (Literature
    5b22bd7f).
  - **AVOIDS it:** a passing seed (so the e-space rule holds) with the Jacobian top ρ ≤ 1 at the end. Both are fp32 (contraction() and the power iteration).
  - **A pass reads "stable at 768", not a benefit.** The benefit is S5×4's question.
  - **The switches:** the loss spike (max over the 1,000 steps after, minus the loss at the switch) and the recovery.
    - Switch 1 is expected to repeat §136 (+1.4 to +2.1 nats, recovered).
    - A failure before switch 2 reads as §136 not replicating, not as a full-width failure.
* **PASS reading:** a pretrained LM's latent carrier reaches the residual width exact, SERIAL and contractive by staged growth. It joins the S5×4 proposal as the full-width arm
  (§138 rule (ii)).

**Descriptive.**
* §138's measures: effective rank, variance beyond the first 64 dimensions, spread over between, directional ρ, Jacobian power, carried norm.
* §136's measures: switch spikes and new-dimension use.
* Literature (c):
  - ‖W_up_new‖ and the new block's share of W_up's output;
  - ρ split by block (FSQ-8, the 56, the 704), and the 64-block e-space ρ (`s143.rho_block64_e`);
  - R²_state from the 704-dimension block alone.

**Placement.**
* **Cost:** proof s1 about $0.3–0.4, then s2–s4 about $1; about $1.2–1.5 in all.
* **Implementation (the experiments session):**
  - fsq120 `--grow2 768 --grow2_min_step 6000 --grow2_lr_scale 0.0833`;
  - the switch-2 pre-eval in grow.json (`pre_eval`: acc_512, rho_k3, the directional ρ, the Jacobian top, and the carried norm ‖e‖);
  - per-group step logging;
  - **the e-space distances in `eval_contraction_fp32.json`** (`rho_k_e`, `dist_med_k_e`, beside the z-space `rho_k` / `dist_med_k`);
  - the block-split ρ, and in eval.json `s143.{block704_var_share, block704_r2_state, w_up_new_norm, rho_by_block, rho_block64_e}` plus `carried_norm.end` (‖e‖).
* **Gates:**
  - function preservation at a forced switch 2 (max |Δ logits| = 0);
  - gradflow under CUDA bf16 into the new W_up columns from the first step after switch 2;
  - memory at L 32 with d 768 plus COPY rows;
  - smoke_plain;
  - Audit's diff review.
* Dirs `s143_fullgrow_s5_s{1..4}`. Scorer: `theory/score_s143.py`, committed with this registration.

**Note 133-11 (2026-10-07 15:21 EDT, before any stage-2 implementation or run; Audit approved the proofs (about $2.5), and the completions wait for the credit). Stage 2's lean cell
list, fixed now.**
* **Cells (5):**
  - S5×2 FSQ-8: the in-capacity control (condition (i));
  - S5×4 FSQ-8: the probe (24 bits < 27.6);
  - S5×4 FSQ-8-L16: 32 bits, separating codes from width;
  - S5×4 NOISE-64: the stage-1 passer standing for the 64-d carriers;
  - S5×4 COPY+GROW-NOISE with the PLATEAU trigger (§136 stage 2, note 136-2).
  - FSQ-64, CONT-64 and SOFT-16 are not run (budget). FULL-GROW joins only if §143 passes (a dated note then).
* **Harness:** §133's (the S5 recipe, restart slots k / k + 100 / k + 200 / k + 300, 24k, RTX 3090, checkpoints, the host check) with `--family s5x2 | s5x4`.
  - k independent S5 registers; items are (register token, op); the answer is all k states (Literature (b)5); FINAL only.
  - COPY+GROW-NOISE follows §136 (COPY from step 0, no restarts).
* **Rules unchanged:** CAP-PASS(cell) = long_acc ≥ 0.99 at 512 in ≥ 3/4 slots. CAPACITY is read mechanically as 133-2 (2), with COPY+GROW-NOISE counted among the wide arms.
* **Proofs first** (slot 1 per cell):
  - A cell whose proof is accepted with long_acc < 0.95 at 512 is dropped, **except S5×4 FSQ-8**: its failure is the probe's expected outcome, so it completes to establish
    CAP-FAIL with entry.
  - **Proof-level reading** (descriptive, n = 1 per cell): 133-2 (2)'s logic on slot 1 alone.
  - Completion (slots 2–4, about $7.5) needs Audit's approval against the live credit.
* **Scorer (`score_s133.py`)** gains:
  - the COPY+GROW-NOISE cell (no restarts);
  - the family check (s5x2 / s5x4);
  - the proof rule and the proof-level reading;
  - a fix to the never-run stage-2 reading, made before any stage-2 data: an **incomplete** cell had read as a failure ("difficulty"). The reading now waits.

**Note 136-2 (2026-10-07 15:21 EDT, before any stage-2 run; §136's stage 2, registered here as the paragraph §136 called for; it shares §133 stage 2's cells and budget). COPY+GROW-NOISE
on S5×4, with the PLATEAU trigger.**
* **The arm:** §136's COPY+GROW-NOISE (COPY rows from step 0, FSQ d 8, 56 new continuous dimensions with dither U(±0.25), §136's growth operation), triggered by
  **PLATEAU** as §136's stage-2 text registered.
  - PLATEAU, as §136 defines it: the first logged step t ≥ 4,000 whose mean loss over (t − 500, t] is ≥ 0.5 and ≥ 0.98 × the mean over (t − 1,000, t − 500].
  - On S5×4 the d-8 code is short of capacity (24 bits < 27.6). A plateau at a non-zero loss is then the user's intended use: widen when the narrow code stalls.
  - §136 found that PLATEAU also fires on still-learning runs (would-be 4.5–5.2k on S5). An early widening is safe there (§136), so this is disclosed, not corrected.
* Seeds 1–4, no restarts (as §136). Dirs `s133cap4_cgrownoise_s{seed}`.
* **Readings:**
  - **CAP-PASS where S5×4 FSQ-8 fails by capacity:** widening at a plateau supplies the capacity the narrow code lacks (the user's GROW idea, with a benefit).
  - **Against NOISE-64** (fixed width from step 0): if both pass, growth costs nothing; if GROW passes where NOISE-64 fails, a narrow-first entry helps at capacity.
  - **CAP-FAIL:** no benefit shown.

**§143 result (2026-10-08 00:01 EDT; s1, s2 and s4, RTX 3090 / 595.71; s3 pending; scored by Theory with score_s143). PASS: all three decided seeds pass, so the ≥ 3/4 rule is met.**
* **Every passing seed:** ext_2048 1.0, SERIAL, e-space ρ 0.062 / 0.043 / 0.057 (≤ 0.28), and **AVOIDS** FULL-CONT's expansion.
  - At the end the Jacobian top is 0.20 / 0.15 / 0.21.
  - ‖e‖ grows 1.17× / 1.16× / 1.12× across switch 2 (the bar is 2×).
* **Switch spikes:** switch 1 +2.37 / +1.51 / +1.42 (§136: +1.4 to +2.1); switch 2 +0.67 / +0.58 / +0.61. All recovered.
* **Pairing:** s2 and s4 match s136_cnoise exactly up to switch 2 (max loss difference 0.0). s1 is unpaired (1.59; driver).
* **Note 143-2's 704-block ablation:**
  - s2 **WIDTH UNUSED** (ext_2048 1.0 and SERIAL with the new block zeroed);
  - s4 **WIDTH USED** (0.926, SERIAL);
  - s1 cannot be tested (its model was not kept); s3 is pending.
  - The new block's share of e is 4–7 % (`s143.w_up_new_out_share` 0.043 in s1, 0.069 in s4).
* **Reading (registered):** a pretrained LM's latent carrier reaches the residual width (768) exact, SERIAL and contractive by staged growth.
  - That is "stable at 768", not a benefit.
  - Whether the new dimensions are needed differs by seed (one of the two testable seeds).
  - It joins the S5×4 proposal as the full-width arm (note 133-12).
* **s3 (00:41): PASS** (ext_2048 0.998, SERIAL, ρ 0.049); 704-block ablation **UNUSED** (0.994). §143 is **4 / 4**. Width is USED in 1 of the 3 testable
  seeds (s4).

## 144. RUNNING-SUM: can the task-agnostic carrier carry a growing multi-digit number? (a diagnostic for multiplication; pre-registered 2026-10-07 15:40 EDT, after Literature's check d8ccceb5 and Audit's draft, before any implementation or run; the user's "do another research run, we gotta figure out how to fix the multiplication" (via Audit))

**Why.**
* Every multiplication design on the carrier fails where a step must hold or build a multi-digit running quantity (§142's post-mortem).
* **Note 140-3:** §140's accumulator values ≥ 100 are not represented even in h.
* §144 removes multiplication's structure (in-pass products, column bookkeeping) and asks only whether the carrier can carry and update a 4-digit number.
* **Prior** (d8ccceb5, `docs/LIT_RUNSUM_MULTISLOT_2026-10-07.md`):
  - LMs code numbers digit-wise (2410.11781), with Fourier features where addition is a rotation (2406.03445), as a helix (2502.00873), or by heuristics (2410.21272);
  - carry cascades are the hard case (2310.13121), and significance must be explicit (Abacus 2405.17399);
  - single units don't extrapolate magnitudes (NALU 1808.00508);
  - a discrete latent carrier holding a growing multi-digit number is new as scoped.

**The family (`--family sum4`, fsq120's harness).**
* **State:** s ∈ {0 … 9,999}.
* **Start:** the 4-digit text of a uniform random s0, as S5's random start. Every state then occurs in training, so no length test needs unseen magnitudes.
* **Items:** x ∈ {0 … 99}, written as two digit tokens, least-significant first.
* **Transition:** s ← (s + x) mod 10⁴.
* **Segment:** [EOT, carried code, x's two digits].
* **FINAL answer:** s as four digit tokens, least-significant first. Shorter chains' answers are partial sums, as in every fsq120 family.
* **Recipe:** §133's harness exactly: §128's S5 recipe (L_max 2 → 32 by 14k), restart slots k / k + 100 / k + 200 / k + 300, 24k, bs 64, RTX 3090, checkpoints, the
  host check.

**Arms (seeds: slots 1–4).**
* **SUM-F8:** FSQ-8, §140's carrier.
* **SUM-N64:** NOISE-64 (`--d_q 64 --no_round --dither 0.25`), §142's carrier.
* **The magnitude ladder (conditional; Literature (3)):** an arm whose proof drops at 10⁴ is rerun at **mod 10² and mod 10³**, slot 1, to locate where carriage breaks
  (dirs `s144_{arm}_m{2,3}_s1`; items 0–99 at both moduli). mod 10² sits close to §128's closed CYC / MOD families.

**Rules (fixed now; §128's FAMILY-PASS form).**
* **PASS(arm):** ≥ 3/4 slots exact at 512 (long_acc ≥ 0.99), SERIAL (a donor's code after positions 2 / 4 / 6; donor ≥ 0.90 and recipient − chance < 0.10, exact ground
  truth) and contractive (ρ ≤ 0.28: 127-12's clause for FSQ-8, the first-three form for NOISE-64).
* **DROP at proof:** slot 1 accepted with long_acc < 0.95 at 512 stops the arm. FAIL: ≥ 2 decided slots not passing.
* **Readings:**
  - **Either arm PASSES:** the carrier carries and updates a 4-digit number exactly. Number carriage is not the barrier, and multiplication's failures point to the in-pass
    products and column bookkeeping, or to the missing per-step supervision that §142 adds.
  - **Both FAIL, with the per-digit profile below losing the high digits:** number carriage itself is the barrier, which every single-vector design inherits. §145's multi-slot
    layout is the response.
  - **SUM-F8 FAILS and SUM-N64 PASSES:** the 8-d rounded code is the bottleneck for numbers.
* **Descriptive:**
  - per-digit accuracy (units / tens / hundreds / thousands) by length;
  - probe R² of each digit and of the value from the code and from h before W_down (as 140-3);
  - **which code is used** (Literature (2)): digit-wise R² against clock R² (cos / sin(2πs / T) for T = 10, 100, 1,000, 10⁴). A digit-wise code must resolve carries in
    one pass; a clock code is carry-free but needs a phase-to-digit decoder;
  - **start-decode accuracy** (the start text decoded at length 0), so that a start-encoding failure is not read as a carriage failure (Literature (5));
  - accuracy at 2,048;
  - codes per state (SUM-F8).
  - **CERT** is not part of PASS (about 10⁶ transitions); it runs time-capped, with "inconclusive by time" allowed.

**Placement.**
* **Proofs:** slot 1 per arm on cloud 3090s, about $0.3–0.4 each. Local 5k prototypes ($0, descriptive) may run first as filler.
* The conditional ladder costs about $0.3 per run. Only an arm that drops at 10⁴ triggers it.
* Slots 2–4 only for arms whose proof ≥ 0.95, with Audit's approval against the credit.
* **Implementation (the experiments session):** the `sum4` family in fsq120 (start text, item tokens, answer format, labels).
* **Gates:** exact ground truth against integer arithmetic, gradflow, memory at L 32, smoke_plain, Audit's diff review.
* Dirs `s144_{f8,n64}_s{seed}`. Scorer: `theory/score_s144.py`, committed with this registration.

**§144 result (2026-10-08 01:44 EDT; RTX 3090; scored by Theory with score_s144). Both arms DROP at proof. F8 never enters; N64 enters but carries no exact digit.**
* **F8:** slot 1 is exhausted (no entry by the 2k flag in 4 attempts). Run with no stop to 24k (note 144-1), it keeps one code, with loss 9.40, about chance.
* **N64 slot 1** (sum mod 10⁴, random start): long_acc 0.000 at 512.
  - The thousands digit is 0.98 at L = 1 and falls to 0.10 at L = 64. The three low digits are at chance from L = 1.
  - It copies the start's top digit as a coarse magnitude (code R² 0.83 / 0.82 for digits 2 / 3, about 0 for digits 0 / 1). It never holds the low digits.
* **The ladder:** F8 and N64 at mod 100 and mod 1000 all have long_acc 0.0, with final losses at chance (N64: 4.61 ≈ ln 100; 7.16 ≈ ln 1000) and every real digit at chance.
  A smaller modulus does not help.
* **The L = 1 harness check:** PARTIAL.
  - s601 learns one of the two digits of (start + x) mod 100 by 5k in both arms (loss 2.44 / 2.48, ≈ ln 10). s602 learns neither.
  - So even one exact addition into the code is slow at this budget.
* **Reading (registered):** number carriage itself is the barrier.
  - A single-vector carrier, FSQ-8 or NOISE-64, does not hold an exact multi-digit number across even one addition at this budget. At most it carries the start's top
    digit as a magnitude.
  - Every single-vector design inherits this (L2). §145's slots and §147's matrix memory are the responses.
* **Against 147-4:** fsq120's carrier is computed by the whole network from the previous code, so it is genuinely recurrent. This is a failure of a recurrent
  single-vector state, not of an input-only buffer.
* **N64 run with no stop** (note 144-1): it does not enter within 24k. s1's loss is 9.42 and every digit is at chance. The slot-1 run that did enter (s301) carried the top two
  digits, so N64's entry, like F8's, depends on the seed.
* **Correction (2026-10-08 01:53 EDT, Audit's reproduction of s144_n64_s301):** two wording errors; the verdict is unchanged.
  - "The three low digits at chance from L = 1" is wrong. The hundreds digit is 0.666 / 0.463 / 0.218 at L = 1 / 2 / 4, matching its code R² 0.83. **N64 keeps the top
    two digits as a coarse magnitude**, not only the top one.
  - "7.16 ≈ ln 1000" is 0.25 nats above chance (6.908; F8 m3 sits at 6.908).

## 145. MULTI-SLOT: carry K = 4 latent vectors per hand-off, one write position each (§140's digit-pair multiplication; pre-registered 2026-10-07 15:40 EDT, after Literature's check d8ccceb5 and Audit's draft, before any implementation or run; the user's "we gotta figure out how to fix the multiplication" (via Audit))

**Why.**
* Every carrier so far hands one vector to the next step. A number is several digits.
* TC-M, which multiplies perfectly in text, does not hold S_j linearly in one hidden state (0.174 exact, 139-2). It spreads each sum over one position per digit.
* §140 (one product per step, 1 × FSQ-8) failed from column 3: accumulators ≥ 100 are absent even from h (140-3).
* **§145 gives the latent a digit-like layout without tokens:** K = 4 carried slots, written at K positions in one pass.
* **Prior** (d8ccceb5):
  - recurrent memory of several vectors: Memory Transformer 2006.11527, RMT 2207.06881, ARMT 2407.04841, Block-Recurrent Transformer 2203.07852;
  - several latent thoughts: Coconut 2412.06769, PCCoT 2506.18582; pause tokens 2310.02226 (§107's PAUSE-4 did not help);
  - specialisation by competition: Slot Attention 2006.15055, RIMs 1909.10893;
  - digit-position specialisation of carried slots: not found (Abacus 2405.17399 is the nearest, in token space). New as scoped.

**Design: §140 exactly** (mult_pair 4×4, one segment per digit pair in output-column order plus a flush; the column flag; streamed product digits; the column
curriculum; restart slots; 24k), except for the carrier:
* **Segment:** [EOT, slot₁ … slot_K (carried), a_i, b_j, column flag, w₁ … w_K]. The flush segment is [EOT, slot₁ … slot_K, " =", w₁ … w_K].
  - w₁ … w_K are K write positions: learned embeddings, initialised from four distinct unused tokens.
* **Each write position's last hidden state** goes through its own W_down_k → NOISE-64 (continuous; dither U(±0.25) at training hand-offs after 2k) → its own W_up_k.
  It becomes slot_k of the next segment.
* **The start** is K zero codes (as fsq120's non-S5 start).
* **The decoder** reads [EOT, slot₁ … slot_K] and emits p_k at a column's last pair, as §140.
* **The slots are told apart** by their positions, their write embeddings and their own maps. The decoder sees only the K slots (the GEN-M-MASK lesson).
* **All slots are NOISE-64, our stage-1 passer.** Chains are 17 hand-offs, so long-range drift without rounding is not the risk here (Literature (5)).
* **Arms:**
  - **SLOT4:** K = 4 write positions, 4 carried slots.
  - **CARRY1 (the control; Literature (1)):** the same 4 write positions and compute, but **only slot 1 is carried** (slots 2–4 are written and dropped). It runs **only if
    SLOT4's proof IMPROVES** (below).
    - It holds NOISE-64 and the extra write compute fixed, so it isolates the multi-vector state from both.

**Rules (fixed now; §140's).**
* **M-PASS:** ≥ 3/4 slots with test_acc ≥ 0.99 on the 1,000 items, SERIAL, and contractive.
  - **SERIAL:** all K slots of a donor injected after pair 4 / 8 / 12; the later digits must follow the recipient's pairs with the donor's accumulator (exact).
    - It is read at the **row level** (all later digits), since digit-level chance is 0.46–0.67 here (§140).
  - Contractive: the first-three form, ρ ≤ 0.28, on the concatenated K × 64 code.
  - DROP at proof: slot 1 below 0.95. FAIL: ≥ 2 decided slots not passing.
* **IMPROVES (the user's falsifiable claim, descriptive):** per-column accuracy at columns 3–7 ≥ 0.9 on SLOT4's proof, where §140's 1 × FSQ-8 had 0.10–0.50.
* **Readings:**
  - **SLOT4 PASS:** a pretrained LM multiplies exactly through a multi-slot latent carrier (design-forced C1, C4 as reworded: streamed digits, as §140).
  - **SLOT4 IMPROVES, CARRY1 not:** the multi-vector state is what represents the larger accumulators.
  - **Both IMPROVE:** NOISE-64 plus the extra write compute was enough; the carried slots are not shown to matter.
  - **SLOT4 DROPS:** spreading the carrier over K positions does not fix the accumulator at this budget.
* **Descriptive:**
  - **specialisation (Literature (3)):** a per-slot × per-digit probe matrix (R² of the accumulator's units / tens / hundreds from each slot), per-slot donor swaps
    (C1 per slot), and per-slot ablation (zeroing one slot);
  - the whole accumulator's R² from all slots (against §140's code 0.61 and h 0.92);
  - accuracy by accumulator magnitude (< 100, 100–199, ≥ 200);
  - acc_by_column, ext_4x8, the slot-symmetry check (the pairwise cosine of the K slots).

**Placement.**
* **Cost:** SLOT4's proof about $0.5 (longer segments); CARRY1's conditional proof about $0.5. Slots 2–4 only after a proof ≥ 0.95, with Audit's approval.
* **Implementation (the experiments session):** multpair `--slots K --carry_slots C` (C = K for SLOT4, C = 1 for CARRY1), with write embeddings, per-slot W_down /
  W_up / NOISE, the K-slot decoder input, the K-slot donor injection and contraction, and the per-slot swaps and ablations.
* **Gates:**
  - gradflow under CUDA bf16 into every W_down_k / W_up_k and the write embeddings;
  - exact ground truth as §140;
  - memory at the longest chain;
  - smoke_plain;
  - Audit's diff review.
* Dirs `s145_{slot4,carry1}_4x4_s{seed}`. Scorer: `theory/score_s145.py`, committed with this registration.

**Note 145-1 (2026-10-07 15:52 EDT, before any §145 run; found by the experiments session's gradflow check). CARRY1 carries the slot read at the last write position.**
* **The flaw:** GPT-2 attends causally, and the segment ends [..., w₁, w₂, w₃, w₄]. A slot read at w₁ cannot see w₂–w₄. As registered, CARRY1 (slot 1 carried) gave exactly zero
  gradient into w₂–w₄ and their maps: one write position plus inert compute, not "the same write compute".
* **Fixed now (`--carry_from last`):** CARRY1 carries **the slot read at w₄**, through w₄'s own maps, so w₁–w₃ are compute that feeds the one carried vector. The decoder reads
  that one slot.
* This is the registered intent: the same 4 write positions and compute, one carried vector. SLOT4 is unchanged (every w_k and map gets gradient), and so are the rules
  and readings.

**Note 142-2 (2026-10-07 16:40 EDT, before any §142 result; Theory's disclosure of internal priors missed at registration). §121 / §122 are the closest internal priors and predict a DROP.**
* **§121's READ arm is §142's mechanism on §99's harness.** At every carried slot, [EOT, carried latent] decoded the step-j text from step 0.
  - It lifted op 3 to 0.64 / 0.93 **by recomputation**. The carried latent held the multiplier prefix, with S_2's digits at chance (114-4).
  - §122 (TEXT-MIX + anchor + reader) failed too, and its **stop rule paused the multiplication latent-carriage line** (2026-10-05).
  - The user's later directives ("start multiplication", "yes go ahead") reopened multiplication on fsq120 (§137–§142).
* **What differs in §142:** the bounded 64-d carrier on fsq120's segment format, twice the budget (48k), the L curriculum, S_j-only targets, and the removal forks.
  - Literature's §121 design point applies unchanged: a code holding (a, b₁ … b_j) lets the reader decode S_j by recomputation, so a reader alone need not force the sum.
* **Expectation, sharpened:** a DROP at proof, or exactness without SERIAL (the b-prefix shortcut, which §142's SERIAL and b-prefix rate detect). The approved proof (about
  $1) proceeds. No rule changes.

**Note 145-2 (2026-10-07 16:40 EDT, before any §145 run; disclosure of an internal prior missed at registration). §107's LAT-4 carried m latents per op and failed; §145 differs where it
matters.**
* **§107 LAT-m** (m = 2, 4; §99's MARKOV harness, multiplier-major): CARRY-3 0/3, test ≤ 0.063. LAT-4's latents were partly redundant (cosine 0.76–0.78).
  - 114-4 explained it: the carrier held the multiplier prefix, so no carrier change could help (§114's looped depth failed the same way).
* **How §145 differs:**
  - the digit-pair ordering removes the prefix shortcut: only a_i and b_j are visible, so the accumulator must be carried (design-forced C1);
  - there, §140's failure is the accumulator's representation (140-3), which §145 targets;
  - the K slots are written in parallel at K positions of one pass, not chained;
  - each slot is a bounded NOISE-64 code.
* §107's redundancy is the named risk. The slot cosine and the per-slot ablation report it.

**§145 result (2026-10-07 19:36 EDT; SLOT4's proof slot, RTX 3090; scored by Theory with score_s145). DROP at proof: test_acc 0.000. No improvement, so CARRY1 does not run (as registered).**
* **Per column:** 1.0 / **0.306** / 0.118 / 0.103 / 0.076 / 0.104 / 0.211 / 0.951. It is **worse** than §140's 1 × FSQ-8 at column 2 (1.0) and the same elsewhere.
  - The loss is ≈ 0 while only columns 1–2 train, then 0.60 at 3k and 1.38 at 24k, as §140.
* **The slots do not specialise:** per-slot R² for the accumulator's units / tens / hundreds is about 0.13 / 0.29–0.36 / 0.10 in every slot.
  - The whole accumulator's R² from all four slots is 0.54 (§140: code 0.61, h 0.92).
  - By magnitude, accuracy is < 100: 0.39, 100–199: 0.10, ≥ 200: 0.14.
* **One active slot (Literature's pitfall):** zeroing slot 4 (the last write position, which attends to the others) collapses every column, even column 1 (0.11). Zeroing
  slots 1–3 leaves column 1 at 0.90–0.99.
  - The slots are not copies (pairwise cosine 0.005, against §107 LAT-4's 0.76–0.78). The model funnels what it uses into the last slot.
* **SERIAL is untestable** (row-level donor 0.000–0.001; the model fails these columns itself).
* **Reading (as registered): spreading the carrier over K positions does not fix the accumulator at this budget.** It fits §148's cross-design wall.

## 146. MID-PROBE: is the accumulator readable mid-stack where the top layer loses it? (the $0 first test of the research program's mid-stack hand-off bet; pre-registered 2026-10-07 16:43 EDT, before the probe is run; the user's "wide research sweep… major new architecture changes" (via Audit, 16:40))

**Why.** Every carrier so far hands off the **top** layer's last hidden state (after ln_f) into the next segment's input embeddings.
* 140-3: on §140's digit-pair model (forced carriage), accumulators of 100–200 are **not** linearly readable from that top state (R² 0.05; > 200 negative).
* Mid-layer recurrence (T²MLR 2607.15178, to be verified by Literature) hands off a middle layer instead. If the accumulator exists mid-stack and is lost by the top, the
  hand-off point is a fix. If it exists nowhere, it is not.

**Probe (descriptive; `theory/probe_midstack.py`, CPU, $0).**
* On `s140_pair_4x4_s1` (primary) and `s137_col_4x4_s1`, with the model's own hard chains over the 1,000 test items, the residual stream at the segment's last position
  after each block ℓ = 0 … 12 (pre ln_f) is recorded at every pair step.
* **For each ℓ:** 140-3's method exactly. One ridge fit over all pair steps (λ by 5-fold CV on the fit half; fit on half the problems, scored on the other half) of the
  true accumulator. Held-out R² is reported within the bins < 100, 100–200 and > 200.
* **Also, descriptive:** the same per layer for §137's carry-out and column sum (per column).

**Decision rule (fixed now; on s140).**
* **MID-STACK SIGNAL:** some layer ℓ ≤ 9 reaches held-out **R² ≥ 0.5 in the 100–200 bin**, against the top layer's 0.05.
  - The accumulator is then computed mid-stack and lost by the top. A mid-stack hand-off (layer ℓ_out → an early layer of the next segment) is registered next on the
    digit-pair task, after Literature's check.
* **NO SIGNAL:** no layer reaches 0.5 in that bin. The accumulator ≥ 100 is not computed anywhere in this model, so the hand-off point is not the fix (on this evidence),
  and the mid-stack bet ranks below the others.
* n = 1 model per task, as stated. A model trained with a mid-stack hand-off could differ, so NO SIGNAL lowers the bet's rank but does not falsify it.

**§146 result (2026-10-07 16:52 EDT; s140_pair_4x4_s1 on CPU in fp32; the layer mirror matches the model exactly (max difference 0.0); scored by Theory with `theory/probe_midstack.py`;
`theory/results/probe_midstack_s140_pair_4x4_s1.json`). NO SIGNAL: no layer holds the accumulator ≥ 100.**
* **Held-out R² of the true accumulator by layer** (bins < 100 / 100–200):
  - layer 0 (embeddings): 0.11 / −10.1;
  - **layers 1–12: flat at 0.85–0.88 / 0.10–0.22** (best ≤ 9: layer 4, 0.217);
  - the top state after ln_f: 0.87 / 0.054, which reproduces 140-3's 0.05;
  - > 200 (n = 9) is negative at every layer.
* **Reading (as registered):** the accumulator above 100 is not computed anywhere in the stack, so on this evidence the hand-off point is not the fix.
  - **The mid-stack bet (B2) ranks below B1 and B3.** §147 keeps ℓ_m = 6.
* The small-accumulator information is already present at layer 1 and stays flat. What a step can use is what the carried code delivers, plus the current product; no layer
  adds the missing magnitude. L2 is a learning failure, not a storage one (Literature 75f51298).
* The descriptive s137 leg is not run, since the decision rests on s140 by registration.

## 147. DELTA-STATE: a linear-recurrent associative memory as the only cross-step channel of GPT-2 (the research program's first bet, B1; pre-registered 2026-10-07 16:49 EDT, after Literature's checks 87cd2af5 / 75f51298 and Audit's sweep, before any implementation or run; the user's "major new architecture changes… don't get stuck in the minuscule improvements loop" (via Audit))

**Why (the program note `docs/THEORY_PROGRAM_2026-10-07.md`, lessons L1–L6).**
* Bounded single-vector hand-offs carry finite state (L3) but not a growing accumulator, even in h (L2; 140-3).
* A **linear additive memory** accumulates without any in-pass addition: per-column sums sit as values under a column key, carry-save, with no squashing.
* A **delta-rule** memory (erase-then-write; β ∈ (0, 2), so eigenvalues can go negative) tracks finite state (Grazzi 2411.12537, DeltaProduct 2502.10297).
* **Prior** (75f51298, `docs/LIT_BIG_BETS_2026-10-07.md`): the architecture is not new.
  - Infini-attention 2404.07143 (segment-local attention plus a linear memory with a delta variant, retrofitted at 1B / 8B).
  - ARMT 2407.04841 (an associative memory on pretrained GPT-2).
  - Both are evaluated on retrieval only. **New in use:** arithmetic and state tracking, with the memory as the sole cross-step channel and C1 by design.
  - **Literature's key flag, adopted:** a delta write *replaces* the value at a key, so accumulating would still need the in-pass old + product (140-3's failure). Hence an
    **additive, no-decay path** for sums beside the delta path.

**Architecture (on the §140 digit-pair harness; no carried code token).**
* **The memory branch** sits after block ℓ_m: x ← x + W_o · [additive heads ‖ delta heads](LN(x)), 12 heads with d_k = d_v = 64.
  - **6 additive heads:** S ← S + φ(k) vᵀ, read φ(q)ᵀ S, with no decay and no erase (linear attention; φ = elu + 1).
  - **6 gated-delta heads:** S ← α S (I − β k kᵀ) + β v kᵀ, with β = 2σ(·) ∈ (0, 2), α = σ(·) initialised near 1, and k L2-normalised.
* **Installation (L4):**
  - q / k / v are initialised from block ℓ_m's own attention projections;
  - W_o starts at 0 with a 500-step fade-in, as §136's growth;
  - the new parameters get their own Adam group with a 300-step warm-up. The pretrained function is unchanged at step 0.
* **ℓ_m = 6** unless §146 reads MID-STACK SIGNAL, in which case ℓ_m = the best layer (Literature: the probe locates where to put B1).
* **Segments:** [EOT, a_i, b_j, column flag] per digit pair in output-column order, then the flush [EOT, " ="].
  - Each segment is a fresh pass. Attention sees only the segment; **the memory's state (all 12 heads) is carried to the next segment and is the only cross-segment
    channel** (C1 by design).
* **Emission:** at a column's last pair, the LM head at the segment's last position emits p_k (streamed, least-significant first; C4 as reworded, as §140). Carry
  resolution happens in the readout (Literature).
* §140's recipe otherwise: column curriculum, 24k, bs 64, restart slots via the 2k flag, RTX 3090, checkpoints.

**Rules (fixed now).**
* **PASS:** ≥ 3/4 slots with test_acc ≥ 0.99 on the 1,000 items and **SERIAL**.
  - SERIAL: the donor's **whole memory state** swapped in at hand-off after pair 4 / 8 / 12. The later digits must follow the recipient's pairs with the donor's
    accumulator (row level, exact).
  - DROP at proof: slot 1 below 0.95.
* **Contraction is not part of PASS, stated now.** An exact integrator has eigenvalue 1 on its additive subspace by design, so ρ ≤ 0.28 cannot apply. Stability (C6) is read
  on 4×8 (33 steps) instead.
* **IMPROVES (descriptive):** columns 3–7 all ≥ 0.9 on the proof (§140's FSQ-8: 0.10–0.50).
* **Readings:**
  - **PASS:** a linear-recurrent memory as the sole cross-step channel makes a pretrained LM multiply exactly. C1 by design and by SERIAL; C4 as reworded.
  - **IMPROVES without PASS:** the additive substrate fixes the accumulator, but the readout's carry resolution is short at this budget.
  - **DROP:** a linear memory does not fix in-pass digit products or readout at this budget.
* **Descriptive:**
  - accuracy by column and by accumulator magnitude; ext_4x8;
  - per-head-type ablation (additive heads zeroed, delta heads zeroed);
  - the per-column readout of the additive state: R² of the column sum from φ(q_col)ᵀ S;
  - ρ by head type.

**Placement.**
* **Cost:** proof (slot 1) about $0.4; slots 2–4 on a proof ≥ 0.95 (approved by the credit rule).
* **Implementation (the experiments session):** a `DeltaMem` branch in boxes (pure PyTorch, sequential over the short segments) and a `--family mult_pair --carrier
  deltamem` path in multpair.
* **Gates:**
  - function preservation at step 0 (max |Δ logits| = 0);
  - gradflow under CUDA bf16 into q / k / v / W_o / α / β after the fade-in starts;
  - exact ground truth;
  - memory at the longest chain;
  - smoke_plain;
  - Audit's diff review.
* Dirs `s147_delta_4x4_s{seed}`. Scorer: `theory/score_s147.py`, committed with this registration.

**§147 result (2026-10-07 22:10 EDT; the s1 proof, 24k, RTX 3090 / 595.71.05; scored by Theory with score_s147). DROP at proof by the registered rule (test_acc 0.743 < 0.95), but IMPROVES:
the first design to break the middle-column wall.**
* **Per column:** 1.0 / 1.0 / **0.994 / 0.948 / 0.916 / 0.931 / 0.937** / 0.983.
  - Every other multiplication design sits at 0.08–0.41 in columns 3–6: §140, §141, §145, §148, and the direct baseline.
  - By accumulator magnitude: < 100 0.97, 100–199 **0.93**, ≥ 200 0.93. §140 had 0.10–0.14 at ≥ 100, absent even from h (140-3).
* **The mechanism, as designed:**
  - **The additive state holds the column sums:** readout R² by column 1.00 / 0.91 / 0.87 / 0.87 / 0.85 / 0.81 / 0.91, and 0.48 for the flush carry. Interference grows only
    mildly with the column index (note 147-1 (2)).
  - **Both head types are needed:** zeroing the additive heads (columns 2–7 ≤ 0.25) or the delta heads (≤ 0.16) collapses it. The group gains (147-2) let the delta heads train.
  - α moved from 0.993 in some heads (one to 0.919).
  - The additive ρ_k is 1.0 exactly: an integrator, as stated.
* **SERIAL is not met at this accuracy** (row level, at positions 4 / 8 / 12):
  - donor-consistent 0.17 / 0.21 / 0.40;
  - recipient-consistent ≤ 0.011, at chance (0.01–0.02).
  - The memory is the channel (the recipient's answer never survives the swap), but a swapped state transfers only in part, as expected at 0.74 exactness.
* **The loss was still falling at 24k:** ≈ 0 at c_max 1, 0.46 at 3k, 0.026 at 5k, 0.14 at 9k, 0.22 at 12k, 0.06 at 20k, 0.052 at 24k.
* **4×8 (descriptive):** columns 1–4 at 1.0 / 1.0 / 0.99 / 0.94, then column 5 at **0.0**, where a column first holds four products, and later columns near chance. Length
  generalisation fails at the first out-of-range column structure.
* **Reading (as registered): IMPROVES without PASS.** The additive linear memory fixes the accumulator that every single-vector carrier failed to represent (L2). Exactness
  falls short at this budget, with the loss still falling.
  - **The registered rule stops §147.** A longer-budget follow-up is a separate registration (§151).

## 148. LOOP-RETRO: recurrent depth retrofitted into GPT-2 for direct-answer multiplication (the program's depth bet, B3; pre-registered 2026-10-07 16:49 EDT, after Literature's checks 87cd2af5 / 75f51298, before any implementation or run; the user's "major new architecture changes" (via Audit))

**Why.**
* §137's column-major steps failed in-pass. Depth adds serial compute per token without any cross-step latent.
* **Prior** (75f51298): the retrofit is not new (2608.11233, McLeish 2511.07384, Relaxed Recursive 2410.20672, MoR 2507.10524, Huginn 2502.05171; Saunshi 2502.17416).
  - GPT-2 small with no CoT on 4×4: 0.29 (ICoT-SI's table, long training). Our §98 harness: 0.003.
  - ICoT-SI reaches 1.00 with no loops, so the C3 bar is high. **B3's distinctive claims are C2 (loops beyond training help) and C5 (stop at convergence).**
  - A removal ↔ loop coupled schedule is new: a second stage.

**Design.**
* **Task:** direct-answer 4×4, "a * b =" → the product's 8 digits (bench format, least-significant first), no CoT. Bench train split; the 1,000 test items.
* **Architecture:** GPT-2 small as prelude (blocks 0–3), loop (blocks 4–7, weight-tied, run r times) and coda (blocks 8–11).
  - **Input injection:** each iteration's input is h_r + P(prelude output), with P a zero-initialised linear map.
  - **Damped update:** h_{r+1} = h_r + η (Loop(h_r + P(e)) − h_r), with η = 0.25 for r > 1 ("Right direction, wrong step" 2609.16665).
  - At r = 1 and η = 1 it is plain GPT-2, so it is installed function-preservingly.
  - **The r curriculum:** r = 1 to 2k, then r ~ U{1 … 4} per batch to 24k.
  - **Deep supervision (TRM):** CE on the answer after each loop group, at the final r plus one uniform earlier r (114-1's unbiased two-sample form).
* **Arms (proof s1 each):** LOOP-4 (above) and PLAIN (r = 1 always, the same data and budget: the direct-answer baseline).

**Rules (fixed now).**
* **PASS(LOOP-4):** ≥ 3/4 seeds with test_acc ≥ 0.99 at r = 4.
* **IMPROVES (descriptive):** LOOP-4 ≥ PLAIN + 0.2 on the proof.
* **C2 (descriptive, then a rule at the full stage):** accuracy at test r = 8 and 16 ≥ at r = 4 (more loops help beyond training).
* **Per-loop monotone progress** (Literature: all inputs are visible every loop, §114's L1 risk): answer accuracy and CE after each iteration, r = 1 … 16.
* **C5 (descriptive):** halting when ‖h_{r+1} − h_r‖ / ‖h_r‖ < 10⁻³; accuracy and mean r used.
* **Readings:**
  - **IMPROVES:** looped depth buys implicit computation at this budget. Stage 2 (CoT removal coupled to added loops) follows as a new registration.
  - **No improvement:** depth alone does not buy direct multiplication at this budget. B3 stops.
* **Descriptive:** C1 as loop-state swaps (the donor's h after iteration r swapped into the recipient's loop: does the answer follow the donor?); ρ of the loop map.

**Placement.**
* **Cost:** proofs about $0.5 (LOOP-4) + about $0.25 (PLAIN).
* **Implementation (the experiments session):** a `--family mult_direct` adapter with a looped GPT-2 wrapper (prelude / loop / coda, P, η, the r curriculum, deep supervision).
* **Gates:** r = 1 equivalence with GPT-2 (max |Δ logits| = 0), gradflow, memory at r = 4, smoke_plain, Audit's diff review.
* Dirs `s148_{loop4,plain}_4x4_s{seed}`. Scorer: `theory/score_s148.py`, committed with this registration.

**Note 147-1 (2026-10-07 16:50 EDT, before any §147 / §148 run; Audit's formal review, approved). Three pre-run points.**
* **(1) The 2k flag's misfire rule (137-1 (3)) applies to §147's restart slots.** Digit-pair losses are near 0 early (s140: 1.7·10⁻⁶ at 1k), where the ratio test is
  numerically noisy. A flag with loss(2k) < 1.0 is a **misfire**: logged, not a stop, and the run is accepted.
  - score_s147 and the restart driver both apply it. §148 runs one run per seed, so it has no restarts.
* **(2) §147: interference from the never-decaying additive heads.** Every column's writes stay in S, so later columns read through growing interference.
  - Descriptive: the per-column readout R² of the column sum against the column index (`s147.colsum_readout_r2_by_column`).
  - It locates a DROP: accumulation is fine but interference grows, against a failed readout.
* **(3) §148: "no improvement" refutes loops only at this budget.** PLAIN's own number is reported beside ICoT-SI's 0.29 (long training, no CoT). Our §98 harness gave
  0.003.
* **Order (Audit):** §147's proof runs in parallel with §142–§145; §148 follows.

**Note 147-2 (2026-10-07 17:30 EDT, before any §147 run; Audit's code review of 44b6843d). Per-group read gains, so the delta heads can train.**
* **The issue:** the additive read φ(q)ᵀS is unnormalised and grows with the number of writes, which is the point of an exact sum. It dominates the shared W_o.
  - Measured on CUDA: W_o's gradient into the delta-head columns is about 6,000× smaller than into the additive columns (0.0025 against 15.0), and the α / β gate gradient is
    about 3·10⁻⁴.
  - The delta heads (B1's finite-state half) might never train. A PASS or DROP would then test the additive half alone.
* **Fixed now (Audit's option (b)):** a learnable scalar gain per head group on its read before W_o: x ← x + W_o · [g_add · read_add ‖ g_delta · read_delta].
  - **Initialisation:** g_delta = 1, and g_add = the median of ‖read_delta‖ / ‖read_add‖ over a probe batch's hand-offs at step 0. This is function-preserving, since W_o = 0.
  - **A scalar keeps the additive sums exact** up to a learned scale. A LayerNorm would destroy their magnitude.
  - The gains train in the new parameters' Adam group.
* **Gate before launch:** a CUDA gradflow recheck with the delta-column W_o and gate gradients within 10× of the additive ones.
* **A reading added (descriptive):** if zeroing the delta heads changes test_acc by < 0.01 at the end, B1 as tested reads "additive-only".
* args record `mem_group_gain: true`.
* **Amendment (2026-10-07 17:40 EDT, before any run; Audit on the implementation):** both gains are log-parametrised, g = exp(s), with s initialised at log(the initial ratio) (s_delta = 0).
  A raw g_add near 2·10⁻⁴ under Adam (lr about 10⁻⁴) would move about 50 % per step and could change sign. exp(s) is identical at step 0, takes relative steps and stays positive.

**Note 147-3 (2026-10-07 17:52 EDT, before any §147 run; the CUDA gradflow recheck after 147-2). The decay gate α is exempt from the 10× bar; β must meet it.**
* **The recheck:** W_o delta / additive 0.70 (passes). The α / β gate rows are 0.080 under bf16 (CPU 0.18): β rows 0.129, α rows 0.012.
  - α's bias +5 (α ≈ 0.993) sits where σ′ ≈ 0.0066.
* **Why α is exempt:**
  - Near α ≈ 1 the forgetting rate 1 − α ≈ 0.007 is small, and any parametrisation's raw gradient scales with it. exp(−softplus(u)) behaves the same: at large bias,
    σ(b) ≈ 1 − e^(−b) is already a log-rate form.
  - Under Adam the logit still moves about lr per step (about 2.4 over 24k steps with a consistent sign), which spans α from about 0.93 to 0.9994.
  - Lowering the bias to +3 (α 0.953) would force about 96 % forgetting over a chain and cripple the delta heads' memory.
* **Decision (the experiments session's option (i)):** the gradflow gate applies to W_o (both groups) and β. α is exempt.
* **Descriptive:** α per head at the end and at each checkpoint (`s147.alpha_by_head`), so whether decay was learned is visible.

**§148 result (2026-10-07 18:26 EDT; the two proof runs, RTX 3090 / 595.71.05; scored by Theory with score_s148). No improvement: LOOP-4 0.002 = PLAIN 0.002. B3 stops at this
budget (147-1 (3)).**
* **Per column (PLAIN / LOOP-4):** 1.0 / 1.0 / 0.41 / 0.13 / 0.08 / 0.10 / 0.66 / 0.96 against 1.0 / 1.0 / 0.39 / 0.13 / 0.08 / 0.11 / 0.67 / 0.97. They are identical, and
  the final losses are 1.011 / 1.018.
* **The loops go unused:**
  - accuracy is flat over test r = 1–16 (0.000–0.006);
  - CE is 1.018 at r = 1–4 and rises past r = 8, so C2 is not shown;
  - convergence halting never fires (mean r = 16);
  - the loop map barely contracts (ρ 0.96 at ε 0.1);
  - a loop-state swap gives donor-consistent 0.002, so no causal loop state.
* PLAIN's 0.002 sits with our §98 harness's 0.003, against ICoT-SI's 0.29 for no-CoT GPT-2 with long training. The budget binds direct answers.
* **A cross-design finding (descriptive):** the direct-answer model fails at exactly the columns every latent design failed at: the 3–4-product columns 3–6.
  - §137: 0.33 / 0.12 / 0.10 / 0.08. §140: 0.32 / 0.13 / 0.10 / 0.24. §141 r4f8: 0.37 / 0.14 / 0.10 / 0.07.
  - At this budget GPT-2 small does not learn multi-product column sums **in any format we have tried**, with or without a latent carrier, so the carrier is not what fails
    there.
  - Text CoT (TC-M 1.000) succeeds by writing the partial products and running sums as tokens.
* **For the program:** designs that keep text CoT's per-step decomposition (§142's reader, supervised states) are the ones that remove this wall at this budget. FINAL-only
  designs (§147, §150) face it, with their registered readings unchanged.

## 149. SOTA-PROBE: our donor SERIAL, ablation and contraction on released GPT-2 latent reasoners, at release (eval only; pre-registered 2026-10-07 17:54 EDT, after Literature's notes 478cac66 / 029e8e4a and Audit's GO, before any run; the user's request for SOTA shortcomings (via Literature and Audit))

**Why.** No published latent method is scored by donor patching or contraction. 2607.09698 reads Coconut as an expanding system and CODI as a stable attractor
(Lyapunov). §149 brings our C1 / C6 measurements to the released models.
* Training is not involved; the user's GSM8K pause covers our training (Audit).
* "At release" is one point in training (2607.06648: responsiveness can fall while accuracy rises).

**Models and data** (029e8e4a, `docs/LIT_S149_SOTA_PROBE_FACTS_2026-10-07.md`; our GSM8K_REPRO harness loads all four):
* **coconut:** `connordilgren/gpt2-gsm8k-coconut` ckpt 33, 6 thoughts (raw last hidden fed back).
* **simcot_coconut:** `internlm/SIM_COT-GPT2-Coconut` ckpt 28, 10 thoughts.
* **simcot_codi:** `internlm/SIM_COT-GPT2-CODI`, 6 latents via prj.
* **codi:** `zen-E/CODI-gpt2`, 6 latents via prj.
* **Test:** `bench/data/gsm8k_aug/test.txt` (1,319 items; equation chains `<<expr=val>>`).

**Rules and readings (fixed now; per model).**
* **SANITY GATE:** test accuracy within ±2 points of our bench harness (33.1 / 44.8 / 39.4 / 42.5). A model failing it is not read.
* **Alignment, thought ↔ equation step t (t ≤ 3):**
  - coconut: pair (2t − 1, 2t) (a training-history prior);
  - simcot_coconut: pair t (documented in code);
  - simcot_codi: latent t − 1 (code);
  - codi: probe-defined (the latent whose ridge probe best reads step t's value), so its SERIAL is **descriptive**.
* **SERIAL(t):**
  - **Pairs:** the **same equation template** (operators and dependency graph). The recipient is correct unpatched, step t is referenced downstream, the donor's step-t value
    differs, and the substituted answer differs from the recipient's own. The excluded share is reported. All-template pairs are descriptive.
  - **The swap:** the aligned thought(s) from the donor's run go into the recipient's run.
  - **Expected:** the recipient's equation chain re-evaluated with step t := the donor's value (an exact substitution; Literature flag (a)).
  - **SERIAL(t):** donor-consistent ≥ 0.90 and recipient-consistent < 0.10, read only with n ≥ 30 pairs.
  - **BYPASS(t):** recipient-consistent ≥ 0.5: the step is recomputed from the visible question (L1).
* **C1 at release:**
  - CAUSAL = SERIAL at every read step;
  - BYPASSED = BYPASS at any step;
  - PARTIAL otherwise.
* **Ablation:** zero and mean replacement per aligned thought, and all thoughts removed; accuracy by step count (2, 3, 4+).
* **RHO:**
  - thought t perturbed by ε = 0.1 × its norm (raw-hidden models in the fed-back space, CODI-style after prj); the deviation ratios of later thoughts; the first-three
    geometric mean;
  - EXPANDING > 1, CONTRACTIVE ≤ 0.28 (our bound, carried over), MARGINAL otherwise;
  - the Jacobian top (power iteration) is descriptive.
  - 2607.09698's prediction is tested descriptively: Coconut EXPANDING, CODI CONTRACTIVE.
* **Our comparators are qualitative** (different tasks): the S5 carriers are SERIAL with fp32 ρ 0.07–0.26 (§128 / §133). GSM8K's steps form a DAG with the question visible
  (L1), so BYPASS is the expected failure mode, and the test is whether released latents carry step values causally at all.

**Placement.**
* Eval only, on a local GPU (short) or a brief cloud eval, under $0.5.
* **Implementation (the experiments session):** the patch hooks per model format (raw-hidden or prj), the template matcher and substitution evaluator over the `<<…>>`
  chains, and the perturbation / power-iteration hooks.
* **Gates:** the sanity gate reproduced first, and unit tests of the substitution evaluator on hand-checked chains.
* Dirs `s149_{coconut,simcot_coconut,simcot_codi,codi}`. Scorer: `theory/score_s149.py`, committed with this registration.

**Note 149-1 (2026-10-07 17:55 EDT, before any §149 run; Audit's formal review, approved). Power and multiplicity.**
* **(1) UNDERPOWERED:** any (model, t) cell with fewer than 30 pairs reads UNDERPOWERED, distinct from PARTIAL. n is reported for every cell.
* **(2) Which cells decide:** only the code-documented alignments feed a per-model C1 verdict: **simcot_coconut and simcot_codi, t = 1–3.**
  - **CAUSAL:** every read cell is SERIAL, with at least 2 cells read.
  - **BYPASSED:** at least 2 BYPASS cells.
  - PARTIAL otherwise.
  - coconut (a training-history alignment), codi (probe-defined) and every ablation cell are descriptive. No single cell can decide a verdict by chance.

**§149 result (2026-10-07 20:43 EDT; four full evals, n = 1,319 each; probe 612fee03, and 85c1b96b for coconut / codi, memory path only; local 4060 / 2070; scored by Theory with
score_s149). All four pass the sanity gate. No released GPT-2 latent reasoner carries its step values serially at release.**

| Model | Acc (gate) | Swap at t = 1: donor / recipient (n) | t = 2 | t = 3 | C1 at release | ρ (first-3) / Jacobian top eig | Zeroing thought 1 / all |
|---|---|---|---|---|---|---|---|
| simcot_coconut | 0.448 (44.8) | 0.35 / 0.07 (299) PARTIAL | 0.35 / 0.09 (114) PARTIAL | n 26 UNDERPOWERED | **PARTIAL** | 0.39 / 0.21 | 0.30 / — |
| simcot_codi | 0.393 (39.4) | 0.11 / 0.55 (284) BYPASS | 0.07 / 0.74 (107) BYPASS | n 20 UNDERPOWERED | **BYPASSED** | 0.50 / 0.37 | 0.39 / 0.25 |
| coconut (descriptive) | 0.331 (33.1) | 0.30 / 0.10 (251) PARTIAL | 0.00 / 0.59 (85) BYPASS | n 14 | — | 0.47 / 0.24 | 0.15 / 0.03 |
| codi (descriptive) | 0.425 (42.5) | 0.17 / 0.55 (300) BYPASS | 0.01 / 0.16 (106) PARTIAL | n 18 | — | 0.37 / 0.18 | 0.41 / 0.22 |

* **C1 at release (note 149-1's verdict cells):**
  - **SIM-CoT+Coconut is PARTIAL.** About a third of same-template swaps move the answer to the donor's step value exactly, and the recipient's own answer rarely survives
    (0.07–0.09). The thoughts carry step values in part.
  - **SIM-CoT+CODI is BYPASSED.** The recipient's answer survives the swap in 55–74 % of pairs: the step is recomputed from the visible question (L1).
* **No cell reaches SERIAL** (best donor-consistent 0.35). Step 3 is underpowered for every model (14–26 same-template pairs).
* **Ablation:** Coconut depends most on its thoughts (zeroing pair 1: 0.33 → 0.15; all: 0.03). The CODI family hardly moves under single-latent ablation (all zero: 0.22–0.25).
* **ρ:**
  - Every model reads MARGINAL by our finite-perturbation ρ (0.37–0.50: above our 0.28 bound, below 1), with later hand-off ratios near or above 1.
  - **2607.09698's contrast (Coconut expanding, CODI contracting) is not reproduced** at this resolution. CODI is slightly more contractive (0.37 against 0.47).
  - The Jacobian's top eigenvalue is 0.18–0.37. It is not the spectral norm, so it can understate expansion.
* **Our comparators, qualitatively** (different tasks): the S5 carriers are SERIAL by design, with fp32 ρ 0.07–0.26 (§128 / §133). SOTA latents on GSM8K are partially
  causal at best, and BYPASS (L1) is their common failure.

## 150. VOCAB-SLOT: a working memory of K token-shaped slots between passes (program B4; the user's "working memory… that mimics discretised tokens in thought"; pre-registered 2026-10-07 18:05 EDT, after Literature's check c583bb23 and Audit's draft, before any implementation or run)

**Why.**
* **The user's idea, made concrete:** the state between passes is K slots, each a token on GPT-2's own vocabulary. It is soft during training and a real token at
  inference.
* **What it draws on:**
  - L3: snapping to a discrete set corrects errors (FSQ).
  - L2: numbers live across positions, and GPT-2 already represents digits as tokens.
  - §145: the multi-slot layout.
* **Prior** (c583bb23, `docs/LIT_VOCAB_SLOT_2026-10-07.md`; all 10 leads verify):
  - text as the sole channel: Markovian Transformers 2404.18988, Delethink 2510.06557, our §100;
  - Gumbel straight-through messages over a vocabulary: emergent communication 1705.11192;
  - a frozen LM vocabulary as a codebook: LQAE 2302.00902, SPAE 2306.17842;
  - soft tokens: Soft Thinking 2505.15778, 2509.19170 (train soft, infer hard).
  - **K parallel slots on GPT-2's own vocabulary simplex as the sole state, trained on final answers only, is new as scoped.**
* **Literature's prediction, stated now:** token states help arithmetic when supervised as numbers (scratchpads, §100), not when emergent. The in-pass computation (140-3) is
  unchanged by the carrier's shape. A DROP is plausible.

**Design: §140's digit-pair 4×4 harness with §145's K = 4 write positions.**
* **Segments** as §145: [EOT, slot₁ … slot₄, a_i, b_j, column flag, w₁ … w₄], and the flush.
* **Slot write:**
  - logits_k = h(w_k) · wteᵀ over the slot vocabulary, using GPT-2's own head;
  - **Gumbel-softmax** with temperature τ (1 → 0.3 over 2k–6k);
  - soft slot_k = Σ p · wte.
  - **Hardening:** straight-through (a one-hot forward, a soft backward) on a fraction p_hard of hand-offs, ramping 0 → 1 over **6k–8k**, after the read path forms
    (Literature (5); the ramp prototype's late-ramp benefit).
  - At test: hard argmax with no noise. **The slot path uses a detached wte**, so it never moves the vocabulary (Literature (6)).
* **The slots are the only cross-segment channel.** The decoder reads [EOT, slot₁ … slot₄] and emits p_k at a column's last pair, as §140 (streamed digits; C4 as
  reworded).
* **Start:** four blank-token slots. Restart slots as §140, with 147-1's misfire rule.
* **Arms (Literature (1)):**
  - **FULL:** the whole GPT-2 vocabulary.
  - **DIGIT:** the 10 digit tokens plus one blank token. Four slots then form an **abacus register**. This arm tests directly whether token-shaped number states help, and
    makes the C7 test meaningful.

**Rules (fixed now; per arm).**
* **PASS:** ≥ 3/4 slots with test_acc ≥ 0.99 at **hard** inference, **SERIAL**, and **contractive**.
  - SERIAL: all 4 slot tokens of a donor swapped in after pair 4 / 8 / 12; row level, exact.
  - Contractive: 127-12's clause on the hard-token carrier, with the perturbation on h(w_k) before the head.
  - DROP at proof: slot 1 below 0.95. FAIL: ≥ 2 decided slots not passing.
* **IMPROVES (descriptive):** columns 3–7 all ≥ 0.9 on the proof.
* **Readings:**
  - **DIGIT PASS:** an abacus of digit tokens, never supervised, carries multiplication exactly. A token-shaped working memory works for numbers.
  - **FULL PASS:** model-chosen tokens suffice.
  - **DIGIT IMPROVES, FULL not:** the restricted digit manifold is what helps.
  - **Both DROP:** token-shaped slots do not fix in-pass computation at this budget (Literature's prediction).
  - **Hard vs soft at test:** hard ≥ soft means snapping corrects (C6); soft > hard means hardening loses information.
* **C7, fixed now (Literature):** what can be claimed is "the state is K vocabulary tokens", certifiable over tokens, with single-token swaps causal. Semantics are claimed
  only where the tokens match task quantities on held-out items: the share of steps whose DIGIT slots spell the accumulator's digits. It is not "human-readable reasoning".
* **Descriptive:**
  - per-slot token histograms and distinct tokens per slot (collapse);
  - slot entropy and the distance to the nearest wte row in the soft phase (off-manifold);
  - a per-slot × per-digit probe;
  - accuracy by accumulator magnitude; ext_4x8.

**Placement.**
* **Cost:** 2 proofs, about $0.5 each. Slots 2–4 by the credit rule on a proof ≥ 0.95.
* **Implementation (the experiments session):** multpair `--slots 4 --slot_space vocab|digits --slot_gumbel --slot_tau 1:0.3:2000:6000 --slot_hard 6000:8000`, with the
  wte detached on the slot path (args `slot_gumbel`, `slot_wte_detached`).
* **Gates:**
  - gradflow under CUDA bf16 into w_k and the trunk through the soft path;
  - exact ground truth;
  - memory;
  - smoke_plain;
  - Audit's diff review.
* Dirs `s150_{full,digit}_4x4_s{seed}`. Scorer: `theory/score_s150.py`, committed with this registration.

**Note 150-1 (2026-10-07 18:05 EDT, before any §150 run; Audit's formal review, approved). The tied head, and robustness on hard slots.**
* **(1) The whole slot path uses a detached wte, the write logits included:** logits_k = ln_f(h(w_k)) · wte.detach()ᵀ, and the slot mixture uses wte.detach().
  - The slot loss never moves the vocabulary. The trunk, ln_f and the write embeddings carry the learning.
  - wte still trains through the rest of the model, as in §140. Its drift ‖wte − wte₀‖ / ‖wte₀‖ is logged (`s150.wte_drift`).
* **(2) On hard slots ρ will mostly read 0** (127-12's ρ₁ branch: a perturbation matters only if it flips an argmax), as with FSQ.
  - **Descriptive robustness:** the per-slot logit margin to the runner-up token at the end, as the median and the 5th percentile (`s150.slot_margin`), the analogue of
    the FSQ grid margin.

**Note 148-1 (2026-10-07 18:30 EDT; a wording fix to the §148 result, after Audit's reproduction).** The loop-state swap (donor-consistent 0.002) is **untestable** here: the model is at 0.002
itself. It is not evidence of "no causal loop state". Everything else in the result stands as Audit reproduced it (loop ρ_k 0.988 → 1.000 at ε 0.01).

**§141 result (2026-10-07 19:03 EDT; all three cells' proof slots, RTX 3090s; scored by Theory with score_s141). Every cell is dropped at proof (test_acc 0.000), so §141 is closed.**
* **r4s16** (b131, 24k): per column 1.0 / **0.495** / 0.154 / 0.109 / 0.075 / 0.100 / 0.090 / 0.901, against r4f8's 1.0 / 0.371 / 0.137 / … (141-5) and r1s16's 1.0 / 0.663 / 0.19
  / … (141-4).
  - **Per-repeat probe:** the column sum's R² 0.17 / 0.22 / 0.35 / 0.28, with ‖Δcode‖ 9.5 / 8.0 / 7.3 / 9.1. This is r4f8's shape (rising through repeat 3, lost at repeat 4)
    at a slightly higher level.
  - SERIAL is untestable (row-level donor 0.000–0.001; the model fails these columns itself).
* **Reading (as registered: r4 DROP with §140 DROP):** multiplication is not reached by this harness at this budget. Extra latent steps without new input do not buy the
  per-column compute.
  - **r4s16 against r4f8:** soft rounding helps slightly (column 2 0.50 against 0.37; probe peak 0.35 against 0.28), so FSQ rounding is not the main block.
* **This completes §142's post-mortem table** (the r4s16 row: 0.000, with columns 3–7 at chance), as that table promised.

**Note 150-2 (2026-10-07 19:40 EDT, before any §150 result; Audit / the experiments session, 5affcd31).** The arms differ in slot-head precision at s1.
* DIGIT s1 (b145, multpair 8efc841e) runs its slot logits under bf16 autocast, with the 11-row gather.
* FULL s1 (b144, 2bc2adad) runs its slot head in fp32 with autocast off, after a fix to a full-vocabulary memory bug (a wte[V] copy per hand-off). Its effective batch
  stays 64, with --micro chosen on the host.
* Seeds 2–4 use the fp32 slot head for both arms. A DIGIT-vs-FULL difference at s1 alone is read with this caveat.

**Note 145-3 (2026-10-07 19:41 EDT; a wording fix to the §145 result, after Audit's reproduction).** "The model funnels what it uses into the last slot" is too strong.
* Zeroing slot 1 / 2 / 3 leaves columns 2–7 near their chance-level values. It drops **column 8 (the top digit)** from 0.951 to 0.412 / 0.236 / 0.304, and column 1 slightly
  (0.92 / 0.90 / 0.99).
* Zeroing slot 4 collapses everything (column 1 0.11, column 8 0.001).
* **Corrected reading:** slot 4 carries the state, slots 1–3 contribute to the top digit, and none specialise by digit position.

**Note 144-1 (2026-10-07 19:51 EDT, before SUM-N64's proof reads; Audit's raw read of SUM-F8's proof). The 2k stop and a no-stop follow-up.**
* **SUM-F8's proof slot is exhausted.** All four attempts (s1 / s101 / s201 / s301) were flagged at 2k by 123-3. Losses ℓ100 / ℓ1k / ℓ2k are about 10.3 / 9.7 / 9.6,
  against chance 4 · ln 10 ≈ 9.21: no entry.
  - It reads as **DROP at proof** (the arm stops), the registered intent for a failed proof slot. The wording is **"no entry within 2k in 4 attempts"**, not "cannot carry a
    running sum".
* **Why the wording matters:** the 2k flag was calibrated on S5's entry timescale.
  - Local sum4 prototypes without the flag (stopped at 5k) enter later on some seeds: proto144_f8_s502 goes 7.2 at 1k → 5.1 at 5k, and proto144_n64_s501 7.2 → 5.0.
  - The misfire rule (ℓ2k < 1.0) does not cover this case.
* **Follow-ups for a dropped arm, fixed now:**
  - **(i) The ladder, as registered:** mod 10² / 10³, slot 1, the same harness.
  - **(ii) NOFLAG:** one 24k run at mod 10⁴ with **no plateau stop** (seed 1; dir `s144_{arm}_noflag_s1`), read descriptively as exact at 512 or not.
    - "It carries a 4-digit running sum when given time to enter", or "it does not within 24k". This answers the diagnostic directly.
  - About $0.35 per run.
* score_s144 reads an exhausted proof slot as dropped at proof, and prints the ladder and NOFLAG rows.

**Note 144-2 (2026-10-07 19:56 EDT, before any ladder or NOFLAG run; Audit approved 144-1). The ladder rungs also run without the plateau stop.**
* With the 2k flag, a rung could drop by "no entry within 2k" and confound slow entry with a magnitude limit. So the mod 10² / 10³ rungs run as single 24k runs with no plateau stop (seed 1), like NOFLAG.
* **For a dropped arm, the ladder is three no-stop runs:** mod 10², 10³ and 10⁴ (the last is NOFLAG), read descriptively as long_acc at 512 by modulus.
* About $0.35 each, so about $2.1 if both arms drop.
* The local 8k no-stop prototypes (3070, S 509–512) give an early, descriptive read.

**Note 143-1 (2026-10-07 21:21 EDT; interim; s1's proof, RTX 3090 / 595.71.05; scored by Theory with score_s143). The proof passes: the first full-width (768-d) carrier that is exact,
SERIAL and contractive. Slots 2–4 run (the credit rule).**
* **Exactness:** ext_2048 1.0; long_acc at 512 0.999; accuracy 1.0 at every length to 64.
* **SERIAL:** donor-consistent 0.9995 at positions 2 / 4 / 6, recipient = chance.
* **Contraction:** e-space ρ (the rule) 0.062; z-space 0.117; the 64-block e-space ρ 0.060; by block FSQ-8 / 56 / 704: 0.10 / 0.10 / 0.13.
* **AVOIDS FULL-CONT's expansion:** the Jacobian's top eigenvalue at the end is 0.20 (0.49 right before switch 2). The carried-embedding norm goes 45 → 53, well under 2×.
* **The new width is used:**
  - the 704 block holds 51 % of the code variance and reads the state alone at R² 0.85;
  - effective rank 10.7, and spread over between 0.54 (FULL-CONT: 21), so the states are separated.
* **Switches:** switch 1 at 2k (spike +2.37), switch 2 at 6k (spike +0.67).
  - The run was not exact before switch 2 (accuracy at 512 0.0 at 6k, loss 0.57). It became exact by 16k–20k, after the widening.
  - **Unpaired:** the losses differ from s136_cnoise_s5_s1's (max difference 1.59), so the comparison is unpaired, as pre-stated. The reading uses s1's own state.
* Slots 2–4 decide PASS (≥ 3/4).

**Note 143-2 (2026-10-07 21:22 EDT, before slots 2–4 read; Audit's reproduction of 143-1). A qualification, and the 704-block ablation.**
* **The qualification:** the new 704 columns contribute only **about 4 % of the carried embedding** e = W_up z (`s143.w_up_new_out_share` 0.043, ‖W_up_new‖ 0.28 under the μP
  lr).
  - The new block encodes the state in z (variance share 0.51, R² 0.85), but what the next segment receives is about 96 % the original 64 dimensions' contribution.
  - So 143-1's "the new width is used" reads **"encoded in z, but about 4 % of e"**. "Stable at 768" stands.
* **The 704-block ablation (descriptive, fixed now; slots 2–4, and s1 if its model is kept):** zero the new block's W_up output at test (e = W_up_old · z_old), then measure
  ext_2048 and SERIAL.
  - **WIDTH UNUSED:** ext_2048 stays ≥ 0.99 and SERIAL holds. The 768-d carrier is present and stable but works through its 64 original dimensions, which is a different
    claim from §136's 64-d result.
  - **WIDTH USED:** ext_2048 falls below 0.99 or SERIAL fails.
  - eval.json field: `s143.ablate704` = {ext_2048, donor}.
* §143's registered PASS rule is unchanged. This only qualifies what a PASS means.

**Note 150-3 (2026-10-07 21:54 EDT; a correction found by Audit, before §150's result). multpair's column curriculum (CMAX, 4×4) is c_max 1 / 2 / 4 / 8 at 0 / 2k / 5k / 9k.**
* **Until 2k only column 1 is trained,** and column 1 is computable from the first pair, so it needs no carriage. Three statements are corrected:
  - **§140's result:** "≈ 0 while only columns 1–2 were trained (1k–2k)" reads **"while only column 1 was trained (to 2k)"**. Column 2 enters at 2k.
  - **§145's result:** "≈ 0 while only columns 1–2 train" reads **"while only column 1 trains"**.
  - **§150 DIGIT s1** (my message to Audit, the user summary and the backlog block's premise): its 0.016 at 1–2k reflects column 1 only, **not** soft slots carrying column 2.
    - The rise to 0.61 at 3k coincides with c_max 2 at 2k, not only with τ annealing.
    - **"The annealing / hardening schedule collapsed the soft working memory" is withdrawn as the cause.** Collapse to constant tokens describes the end state only.
* **The soft-only prototypes** (τ fixed at 1, no hardening) agree:
  - proto150_digitsoft_s501_4060: 0.073 / 0.001 / 0.691 / 0.503 / 0.556 at 1k–5k;
  - s503_2070: 0.0 / 0.0 / 0.534 / 0.622 at 1k–4k.
  - The soft memory fails as soon as column 2 enters, at about §140's level.
  - Their registered reading (the 6k–8k mean, ≤ 0.1 CARRIES, ≥ 0.3 FAILS) stands as fixed. These numbers point to FAILS, pending 6k–8k.
  - The block's other premise, "c_max is 4 from 5k", is right.

**§150 result (2026-10-08 00:01 EDT; FULL and DIGIT proof slots, RTX 3090; scored by Theory with score_s150). Both DROP at proof: test_acc 0.000, hard and soft at test.**
* **Per column (FULL / DIGIT):** column 1 is 0.889 / 0.901; it is computable in-pass with no carry. Columns 2–8 are at chance (0.08–0.25). §140's FSQ-8 code had columns 1–2
  at 1.0.
* **The slots collapse to constant tokens.**
  - FULL: all four slots hold " secretary" on about 16,000 of 17,000 hand-offs (entropy ≤ 0.002).
  - DIGIT: slots 1 / 2 hold "2" / "6" on 14,122 / 16,012 (entropy ≤ 0.022).
  - DIGIT's slots spell the accumulator on 1.2 % of steps.
* **Loss (FULL / DIGIT):** 0.08 / 0.02 at 1k (column 1 only), 0.64 / 0.61 at 3k (column 2 enters at 2k), 1.05 / 1.02 at 6k, 1.71 / 1.72 at 24k.
* **Soft-only DIGIT protos** (τ fixed at 1, never hardened): **FAILS** in all three finished runs (6k–8k mean loss 1.09 / 1.57 / 1.11; s504 running). The failure is there
  while soft.
* **Hard = soft = 0.000** at test, so C6 gets no reading.
* **Reading (registered, "Both DROP"):** token-shaped slots do not fix in-pass computation at this budget (Literature's prediction).
  - Descriptively, the failure comes earlier than the middle columns. The token channel does not carry even one carry digit (column 2 at chance, below §140's FSQ-8 code),
    soft or hard.
* **The program:** B4 stops at this design (the single-vector deprioritisation). The user's token-like working memory moves to B1's matrix memory, with a reader that decodes
  it into digit tokens (§152 / §153).

## 151. DELTA-LONG: §147's linear-recurrent memory with twice the budget, four seeds (pre-registered 2026-10-07 22:11 EDT, after §147's IMPROVES and before any run; the user's "we gotta figure out how to fix the multiplication")

**Why.**
* §147 s1 broke the middle-column wall: columns 3–7 at 0.92–0.99, where every other design sits at about 0.1, and accumulators ≥ 100 at 0.93.
* It dropped at proof by rule (0.743), with **the loss still falling at 24k** (0.06 → 0.052).
* §151 asks whether the same design reaches exactness with twice the budget, and how reliable it is across seeds (C6).
* The design is unchanged from §147, so Literature's checks (75f51298, 87cd2af5) apply.

**Design: §147 exactly** (notes 147-1 / 147-2 / 147-3: 6 additive + 6 gated-delta heads after block 6, log-parametrised group gains, α exempt, W_o fade-in), except:
* **48k steps** (the cosine over 48k), fresh runs.
* **Seeds 1–4 in parallel** (no proof-first stage: §147 s1 is the proof that it moves), RTX 3090 class.

**Rules (fixed now; §147's).**
* **PASS:** ≥ 3/4 seeds with test_acc ≥ 0.99 on the 1,000 items and **SERIAL** (row level, the whole memory state swapped after pair 4 / 8 / 12: donor ≥ 0.90, recipient − chance < 0.10).
* No contraction term (an integrator by design). FAIL: ≥ 2 seeds not passing.
* **IMPROVES** (descriptive): columns 3–7 all ≥ 0.9.
* **Readings:**
  - **PASS:** a linear-recurrent memory as the sole cross-step channel makes a pretrained LM multiply 4×4 exactly (C1 by design and by SERIAL; C4 as reworded: streamed final
    digits). It is the first exact latent multiplication on this carrier line.
  - **IMPROVES without PASS:** budget-limited or readout-limited. The per-column errors say where.
  - **Neither:** §147 s1 was a lucky seed. The seed rate says how often.
* **Descriptive:**
  - §147's measures: per-head-type ablation, colsum readout by column, accuracy by magnitude, α by head, the read-norm ratio;
  - **ext_4x8** (§147 s1 failed at column 5, the first four-product column);
  - learning curves against §147 s1 at 24k.
* **5×5 only after a 4×4 PASS**, as a new note.

**Placement.**
* 4 runs × about 7.5 h on a 3090, about $1.5 each, about $6 in all (the credit rule; Audit approves).
* Implementation: the §147 code as is, `--steps 48000`.
* Dirs `s151_delta_4x4_s{1..4}`. Scorer: `theory/score_s151.py` (§147's scorer with 48k and the new dirs), committed with this registration.

**Note 151-1 (2026-10-07 22:13 EDT, before any §151 run; Audit's formal review, approved). What "48k" changes, and SERIAL by checkpoint.**
* **(1) The schedule:** the learning rate warms up over 100 steps and follows the cosine over **48k** (fsq120's lr_at over `steps`). The column curriculum's boundaries are
  **unchanged** (c_max 1 / 2 / 4 / 8 at 0 / 2k / 5k / 9k).
  - So §151 spends **39k steps at c_max 8** (§147: 15k), at a higher mid-run learning rate. §151 against §147 is read as "a 48k schedule", not "steps only".
* **(2) SERIAL by checkpoint** (Literature 2607.06648: faithfulness can move during training): the row-level donor protocol at **24k / 36k / 48k**
  (`s151.serial_by_ckpt`), descriptive.
  - A PASS or a SERIAL failure can then be located in training. Only the 48k model's SERIAL enters the rule.

**Note 144-3 (2026-10-07 22:31 EDT; interim, descriptive; before §144's result; scored by Theory with score_s144, after a scorer fix).**
* **The scorer fix:** flagged attempts are fetched as `.void_partial`, so score_s144 now reads them from the host's slot records (`s144_slots/{arm}_k1`), as score_s133 does.
* **Proof reads:**
  - **SUM-F8 is dropped at proof** (slot 1 exhausted: no entry within 2k in 4 attempts).
  - **SUM-N64 is dropped at proof:** slot 1 was accepted on its fourth attempt (s301), with s1 / s101 / s201 flagged; long_acc 0.0, final loss 4.67.
* **N64 carries an analog magnitude, not a number.**
  - **Per-digit accuracy** (units / tens / hundreds / thousands): 0.10 / 0.10 / 0.67 / 0.98 at L 1, 0 / 0 / 0.05 / 0.82 at L 8, 0 / 0 / 0 / 0.60 at L 16, and 0 at L 512.
    The low digits are never learned. The high digits are tracked at first and decay with length.
  - **Probes:** the code reads the value at R² 0.82 (h 0.94) and the coarse clock phase at R² **0.985 / 0.964** (T = 1,000 / 10⁴), but the fine phases (T = 10 / 100) at about
    0.
  - **Read:** the continuous carrier holds the running sum as an approximate magnitude with no digit precision, and the error accumulates over steps.
* **The registered conditional reading does not apply as written.** "Both FAIL, with the per-digit profile losing the **high** digits" does not describe this profile, which
  loses the **low** digits first (precision), then the high ones by drift.
  - The scorer's mechanical "number carriage itself is the barrier" line is therefore qualified: carriage fails **for exact digits**, while an analog magnitude is carried.
* **The L = 1 check so far** (PARTIAL: one digit learned in F8 s601 and N64 s601, F8 s602 at chance): even one carried 2-digit addition is only partly learned.
* §144's result follows once F8's NOFLAG, N64's registered follow-ups (144-1 / 144-2, for a dropped arm) and the last L = 1 run land.

**§151 result (2026-10-08 02:16 EDT; slots 1–2 at 48k, RTX 3090 / 595.71; slots 3–4 pending, descriptive; scored by Theory with score_s151). DROP at proof: slot 1 test_acc
0.837 < 0.95. Slot 2 is 0.889.**
* **Doubling the budget helps but does not reach exactness.** §147 s1 was 0.743 at 24k. At 48k:
  - slot 1: 0.837, per column 1.0 / 1.0 / 0.999 / 0.96 / 0.937 / 0.95 / 0.99 / 0.986;
  - slot 2: 0.889, per column 1.0 / 1.0 / 1.0 / 0.983 / 0.962 / 0.971 / 0.992 / 0.975.
  - IMPROVES holds (columns 3–7 ≥ 0.9). The loss is still falling at 48k (slot 2: 0.072 / 0.037 / 0.015 at 24k / 32k / 48k).
* **Not SERIAL.** Row-level donor 0.08 / 0.08 / 0.17 (slot 1) and 0.53 / 0.50 / 0.68 (slot 2) at pairs 4 / 8 / 12. Recipient ≤ 0.016.
* **No length generalisation.** ext_4x8 is 0.000 / 0.001. Column 5 is again 0.0 at 4×8 (966 of 1,000 first errors there), so §154's pair-count schedule signature persists at
  48k.
* **Reading (registered, in note 147-4's wording):** an input-driven linear memory improves with budget but does not make multiplication exact at 48k. Carry resolution
  at read time, or storage precision (note 152-5), falls short.

## 152. DELTA-READ: the linear-recurrent memory with a CoT-derived reader of the running sum, then ICoT-style removal; against text CoT at matched compute (pre-registered 2026-10-07 23:01 EDT, after Literature's check 03059976 and Audit's draft, before any implementation or run; the user's "sure" (23:00, via Audit) and the user's operative standard: the latent model uses latent space more effectively than text)

**Why.**
* §147 broke the middle-column wall with a linear additive memory as the only cross-step channel: columns 3–7 at 0.92–0.99, accumulators ≥ 100 at 0.93, exactness 0.743
  at 24k. §151 tests more budget.
* **§152 adds what made text CoT work: per-step targets taken from the CoT,** then removes them as ICoT-SI does.
* **The user's standard (23:00):**
  - strict C4 is not required, so CoT-derived per-step supervision and CoT-then-removal are allowed;
  - **the headline is C3 + C1:** at least text-CoT accuracy at less compute, with SERIAL.
* **Multiplier-major ordering** (§142's), because its per-step state S_j is in the CoT text, as the teacher's own S_j (139-2, the §142 cache).
  - It is also the cheap order: 4 segment passes plus an 8-digit decode, about 12 sequential passes, against TC-M's 57.
* **Prior** (03059976, `docs/LIT_S152_DELTA_READ_2026-10-07.md`):
  - no linear-memory carrier for multi-digit arithmetic is known; state tracking covers parity, counting and permutations;
  - the additive state is linear attention's Σ v kᵀ (fast weights 2102.11174, 2212.07677), applied to digit products under column keys;
  - reader precedents: RMT / ARMT read-only queries, LM2, 2510.00184's running-sum auxiliary, and SIM-CoT (a text reader);
  - new as scoped.

**Design.**
* **The carrier:** §147's DeltaMem exactly (6 additive no-decay heads plus 6 gated-delta heads after block 6; log-parametrised group gains; α exempt; W_o fade-in; notes 147-1 to
  147-3), as the **only** cross-segment channel.
  - Segment j = [EOT, a's digits, b_j, position j], with no carried code token. Attention stays within the segment.
  - The additive heads can hold S_j in carry-save form (a_i · b_j written under column keys), so no in-pass multi-digit addition is needed.
* **Reading the memory:** a side-branch pass [EOT, " ="] reads the memory through the DeltaMem branch and writes nothing back. It emits 8 digits, least-significant first,
  through the LM head.
  - The final answer is decoded this way after the last segment.
* **Training:** §142's recipe (the bench 4×4 train split, commuted test pairs removed; L ~ U(1, L_max) with L_max 1 / 2 / 3 / 4 at 0 / 2k / 5k / 9k), 48k steps (cosine over
  48k), bs 64.
* **Arms (seeds 1–4; A s1 and CTRL s1 first, in parallel with §151):**
  - **A (READ):** at every hand-off j < L, the side-branch reader decodes **S_j** (the teacher's text, cached; label_source "teacher_generated"). Loss = CE_final + the mean
    over hand-offs of CE_j (each over 8 tokens).
    - **Gate:** once L_max is 4, greedy final exact on 512 train items at L 4, checked every 1k. The first value ≥ 0.95 saves `gate.pt`.
  - **B-EARLY (ICoT's order):** forked from A's gate.pt (same seed), 20k steps, **fixed_L 4**.
    - S_1's reader target is removed at +0, S_2 at +4k, S_3 at +8k; final answer only from +8k to +20k.
    - **At each removal the LR re-warms over 200 steps, and AdamW is NOT reset.** This is deliberate (Literature (2)): the inputs do not change at removal, only the loss, so
      ICoT-SI's reason for a reset is absent, and a reset would disturb the settled gate / gain moments. The loss jump is logged.
    - 10 % of rows preview the next removal. Removed S_j stay decoded unweighted (retention).
  - **CTRL:** A without the reader (FINAL only), the same carrier and budget, s1. It is §147's design transposed to multiplier-major.
  - A seed whose A run never opens the gate gets no B run.

**Rules (fixed now).**
* **PASS(arm):** ≥ 3/4 seeds with test_acc ≥ 0.99 on the 1,000 items and **SERIAL**.
  - **SERIAL:** the whole memory state swapped from a donor after segment 2. Donors are chosen with **a_donor ≠ a_rec and a different b-prefix** (Literature (4)).
    - Each answer is classified as **(i) carry-save** = donor S_2 + a_rec × (b_rec,3 b_rec,4) × 100, **(ii) b-prefix** = a_rec × (b_donor,1 b_donor,2 b_rec,3 b_rec,4), or
      **(iii) recomputation** = the recipient's own product. All three shares are reported (row level).
    - SERIAL = (i) ≥ 0.90 and (iii) − chance < 0.10.
  - No contraction term: the additive part is an integrator by design.
  - DROP at proof: A s1 below 0.95.
  - B is read on its final, final-answer-only model.
* **The headline (C3 + C1, the user's standard; fixed now).** On the same 1,000 4×4 items, Meter FLOPs per item and sequential passes, against **TC-M** (1.000; 1.55·10¹⁰
  FLOPs; 57 passes) and **ICoT-SI** (0.998; 4.93·10⁹; 13 passes):
  - **LATENT BEATS TEXT** for a passing arm whose median test_acc ≥ TC-M − 0.01, with **fewer FLOPs and fewer sequential passes than TC-M**, and SERIAL.
  - **The wording, fixed now (Literature (3)):** "matches text CoT at about 0.6× FLOPs and about 0.2× sequential passes, with SERIAL. ICoT-SI is cheaper still at 4×4,
    but has no causally testable carrier."
    - A C3 win over ICoT-SI needs a regime where ICoT-SI degrades (5×5+ at a matched budget).
    - Training compute is reported separately (48k × 64 against ICoT-SI's up to 200 epochs × 808k).
* **Readings:**
  - **B-EARLY PASS with LATENT BEATS TEXT:** a pretrained LM multiplies exactly through a latent memory that carries the computation (SERIAL), from final answers after
    CoT-guided training, at less compute than writing the CoT. This is the user's headline.
  - **A PASS, B FAIL:** the reader is needed at inference-free training time, and removal loses the skill. Retention (the removed S_j's decode accuracy) says which step goes
    first.
  - **CTRL PASS:** the multiplier-major DeltaMem needs no reader.
  - **A DROP:** not acquired within budget.
  - **Exact without SERIAL:** in-pass or shortcut (the b-prefix: a is visible every step). The donor's b-prefix-consistent rate names it.
* **Descriptive:**
  - SERIAL by checkpoint (row level, 24k / 36k / 48k; and at each B stage);
  - the (i) / (ii) / (iii) shares at every checkpoint;
  - a held-out b-prefix probe on the memory after removal;
  - the additive state's norm per segment (Literature (2); needed before any length or 5×5 claim);
  - per-head-type ablation; the additive state's column-sum readout;
  - ext_4x8_low8; α by head.

**Placement.**
* **Proof:** A s1 plus CTRL s1, about $1.5 each, in parallel with §151 (the user's call: credit about $146).
* **Then:** B-EARLY on gated seeds (about $0.6 each) and A s2–s4 (about $1.5 each), by the credit rule. About $9 in all.
* 5×5 after a 4×4 PASS, with the same headline against TC-M 5×5 (2.33·10¹⁰; 87) and ICoT-SI (0.996; 5.76·10⁹; 15).
* **Implementation (the experiments session):** multrow `--carrier deltamem` (§147's branch in the multiplier-major segment), the side-branch reader / decoder pass that
  writes nothing back, §142's reader / gate / fork machinery, the memory-state donor swap, and the Meter on inference.
* **Gates:** step-0 function preservation, gradflow (CUDA bf16) into every DeltaMem parameter from the reader and the final CE, a check that the reader writes nothing back,
  memory at L 4, smoke_plain, and Audit's diff review.
* Dirs `s152_{a,bearly,ctrl}_4x4_s{seed}`. Scorer: `theory/score_s152.py`, committed with this registration.

**Note 152-1 (2026-10-07 23:03 EDT, before any §152 run; Audit's formal review, approved). The layout confound, one change to the segment order, and a partial-product probe.**
* **(1) The confound, stated now.** §152 changes two things against §147's success: the reader, and the layout (digit-pair → multiplier-major).
  - Multiplier-major asks every segment for the 4 × 1 product a × b_j, the in-pass load §137 / §142 / §148 failed on.
  - **Readings, fixed now:**
    - **A and CTRL both DROP:** the multiplier-major layout's in-pass a × b_j is the block, not "DeltaMem + reader fails".
    - **CTRL DROPS, A PASSES:** the CoT reader is what makes multiplier-major learnable.
* **(2) The segment order changes to [EOT, b_j, position j, a's digits (least-significant first)]** (§142's was [a's digits, b_j, j]).
  - Under causal attention, each a_i position then sees b_j and j.
  - Each position can write its own one-digit product a_i · b_j under the column key i + j − 1, in carry-save form with no in-pass multi-digit carry. That is the same
    per-product load as digit-pair, where §147 succeeded.
  - It narrows the confound in (1) without removing it: the model still has to learn four parallel writes per segment.
  - It applies to all three arms.
* **(3) An in-pass partial-product probe** (descriptive; `s152.pp_probe`): ridge / 10-way linear probes of P_j = a × b_j's digits from h at the segment's positions (held-out
  half), per segment. A DROP can then be located at the product step rather than at accumulation.

**§152 result (2026-10-08 01:51 EDT; A s1 and CTRL s1, RTX 3090 / 595.71, 48k; scored by Theory with score_s152). DROP at proof: A s1 test_acc 0.331 < 0.95. CTRL s1 is 0.012.**
* **By chain length (rows of b, j = 1–4):** A 1.0 / 1.0 / 0.999 / **0.331**; CTRL 1.0 / 1.0 / 0.27 / 0.012. A's reader decodes S_1..S_4 at 1.0 / 1.0 / 0.999 / 0.331.
* **A was still learning row 4.**
  - Train-exact at L 4 (the gate) rose from 0.14 at 41k to 0.36 at 48k; the gate (0.95) never opened, so B-EARLY is not run.
  - Loss 0.164 at 24k, 0.075 at 32k, 0.043 at 48k.
* **Note 152-1's locating probe:**
  - Every per-position product a_i · b_j forms in-pass at ≥ 0.98 in both arms (`pp_probe.aibj_at_ai`).
  - The additive state holds the carry-save column sums (A: R² 0.90–0.99 by column) and the b-prefix (R² ≈ 1.0).
  - Both head types are needed. The additive gain fell to about 4·10⁻⁴ against state norms in the thousands.
* **SERIAL:** not at this accuracy. Row-level donor 0.208 at the end (0.018 / 0.09 at 24k / 36k), recipient 0.0, b-prefix 0.002.
* **Compute:** 7.10·10⁹ FLOPs (0.46× TC-M), 12 sequential passes (0.21×). With accuracy far below TC-M's 1.000, there is **no LATENT BEATS TEXT**.
* **Extrapolation (note 152-3):** A is 1.0 / 1.0 / 0.999 / 0.315 at j = 1–4 and 0.0 at j = 5–8. TC-M and ICoT-SI are 1.0 / 0.916 at j = 4, 0.017 / 0.041 at j = 5 and
  0.0 at j = 6–8 (`theory/results/ext_baselines_s152.json`). Nothing separates the methods on length.
* **Reading.**
  - **The registered rule:** DROP at proof, not acquired within budget. §152 stops, and §153's gate (A s1 ≥ 0.95) stays closed.
  - **Note 152-1 (1)'s mechanical reading for both arms below 0.95, "the layout's in-pass a × b_j is the block", is contradicted by 152-1 (3)'s own locating probe.** The
    products form (≥ 0.98), and A is exact through three rows. The failure sits at the fourth row: read-time carry resolution of the full 8-digit product, which A was still
    learning at 48k.
  - **Note 152-4's prediction (A ≥ 0.95) fails at this budget.** The reader's effect goes the predicted way and is large: A far above CTRL at L = 3 (0.999 against 0.27)
    and L = 4 (0.331 against 0.012). But it is not sufficient within 48k.
  - **Under 147-4's mechanism this is expected:** the memory accumulates the carry-save sums correctly, and only the nonlinear carry resolution at read time remains. The
    reader trains that resolution at every j, and the 4-row resolution comes last.
* **Conditional follow-up, pre-registered now:** if §155 R or SHIFT-R passes, §152 A / CTRL and §153 VISIBLE / PURE re-run on DeltaMem-R (`--mem_write top`), with the same
  rules.

## 153. THINK-PASS: item-agnostic latent passes over the DeltaMem (program B5; §152's recipe without the designer's per-pass input; pre-registered 2026-10-07 23:40 EDT, after Literature's check b7b2d52c and Audit's draft review, before any implementation or run)

**Why.**
* Every harness so far hands each pass a designer-chosen input (a digit pair, a row b_j). A general latent reasoner reads the problem and decides what each pass computes.
* **B5 is due** ("after B1 / B4 read"): B1 (DeltaMem) IMPROVES at §147; B4 (token slots) fails (§150 DIGIT, the soft-only protos).
* §153 keeps everything from §152 and changes only **what each pass sees**: DeltaMem as the only cross-pass channel, the read-only reader, teacher S_j targets,
  earliest-first removal, SERIAL, the compute headline.
* **Prior** (b7b2d52c, `docs/LIT_S153_THINK_PASS_2026-10-07.md`): new as scoped.
  - The pieces are known. ARMT / Infini-attention carry memory across *input* segments, Huginn / looped models carry the residual with input re-injected, and SIM-CoT
    decodes steps without removal. Retrofit 2608.11233 runs step-per-loop supervision then outcome-only annealing on a residual state (our A → B arc).
  - **No latent passes with a matrix memory as the only cross-pass channel, per-pass step decoding, then removal.**
  - VISIBLE's "selects its own step input" is covered by input-injection looped models. It is framed as **item-agnostic passes**, not a new capability.

**Design: 4×4, the teacher's multiplier-major S_j; K = L passes (L ~ U(1, L_max) as §152, b's first L digits in the problem), then §152's read-only reader / decoder pass.**
* **VISIBLE:** pass j = [EOT, a's digits, " ×", b's digits, step token j]. The whole problem every pass; the model picks b_j itself.
* **PURE:** pass 0 = [EOT, a's digits, " ×", b's digits] (writes only); passes j = [EOT, step token j, 4 register positions].
  - The problem is never visible again. The memory must hold the operands (multi-query associative recall; 8 keys, within the range of Zoology 2312.04927 / Based
    2402.18668) and the accumulator.
  - C1 holds by design, and L1 cannot apply.
* **Per input:** A (reader on S_j; gate at 0.95), then B-EARLY (removal, no AdamW reset) on gated seeds; CTRL (final only) s1. Everything else is §152's (48k, the
  curriculum, the teacher cache, bs 64).

**Rules (fixed now).**
* **PASS(input, arm):** ≥ 3/4 seeds with test_acc ≥ 0.99 and SERIAL. DROP at proof: A s1 < 0.95.
  - **VISIBLE's SERIAL is §152's swap:** the whole memory from a donor after pass 2, donors with a_donor ≠ a_rec and a different b-prefix. Carry-save ≥ 0.90 and
    recompute − chance < 0.10.
  - **PURE's SERIAL is a pass-skip test** (C1's step-specific blocking). In PURE the memory also holds the operands, so a whole-memory swap moves the problem with it:
    every memory-carried solution gives the donor's own product, and the swap cannot discriminate.
    - Pass 2 and pass 3 are each skipped, with the memory passed through unchanged.
    - **Row-skip-consistent** = S_4 minus that row's a × b_j × 10^(j−1), the serial result. **Full product** = recomputation from carried operands.
    - SERIAL = row-skip-consistent ≥ 0.90 and full product − chance < 0.10 at both skips (chance = rows with b_j = 0). The swap is reported descriptively.
* **Headline:** LATENT BEATS TEXT against TC-M (fewer FLOPs and passes, accuracy ≥ TC-M − 0.01, SERIAL), with ICoT-SI beside it.
* **Predictions, stated now (Literature):**
  - **(i)** VISIBLE's recompute share is higher than §152 A's (L1: the recipient's operands stay visible).
  - **(ii)** PURE's ext_4x5 (K = 5) fails (< 0.5) at the untrained step key j = 5 (the clock-out-of-range concern, Prop. 9 / §35). A failure there reads as the step-token
    design, not the memory.
* **Readings:**
  - **VISIBLE PASS:** item-agnostic passes carry the computation.
  - **PURE PASS:** the memory alone holds problem and state across passes (the strongest C1 form).
  - **PURE fails, VISIBLE passes:** operand recall is the block. The pass-1 and per-pass recall probes say where.
  - **Both fail and §152 passes:** the designer's per-pass input was doing the work.
* **Descriptive:**
  - VISIBLE: attention mass on b_j's position per pass (does pass j select digit j?); the pass-skip shares.
  - PURE (Audit): **operand recall at pass 1**, held-out linear probes of a's and b's 8 digits from (a) the memory state after pass 0 and (b) the branch's reads at pass 1. A
    recall failure is then separated from a later overwrite.
  - PURE (Literature): **b_j recall at every pass** (delta-head overwrite) and **per-head roles** (operand / accumulator, by ablation; additive crosstalk).
  - Memory norm per pass (Literature (d)).
  - C2 / C6: 4×5 with K = 5, and no-op passes K′ = 6 / 8 on 4×4, with the memory change per extra pass logged (an attractor test).
  - C7: the reader's per-pass text, paired with the donor split (decodable is not used; Literature).

**Stage 2, pre-registered now (conditional; Audit): REPEAT, if PURE passes at 4×4 (A or B-EARLY).**
* PURE with **one repeated think token** in place of the step token, so the memory keeps its own pointer. Same arms, rules and SERIAL (pass-skip).
* **C2 (fixed now):** ext_4x5 test_acc ≥ 0.9 at K = 5 in every passing seed. **No-op stability:** 4×4 test_acc ≥ 0.99 at K′ = 6 and 8.
* Literature's alternative, if REPEAT fails to enter: relative step tokens ("the next row") in place of absolute indices.

**Placement.**
* **Gate:** §152 A s1 ≥ 0.95 (the shared recipe works). Then A s1 + CTRL s1 per input, about $1.5 each, **in parallel with §152's B-EARLY and A s2–s4** (Audit: the arms
  answer different questions; credit about $144). Further seeds by the credit rule.
* **Implementation (Experiments):** a pass-content flag on §152's code (`--pass_input visible|pure|repeat`, `--think_token` for REPEAT), the pass-skip evaluation
  (`skip.pass2` / `skip.pass3`: row_skip_consistent, full_product, chance_overlap), `ext_4x5` and the probes. Gates as §152.
* Dirs `s153_{visible,pure,repeat}_{a,bearly,ctrl}_4x4_s{seed}`. Scorer: `theory/score_s153.py`, committed with this registration.

**Note 153-1 (2026-10-07 23:42 EDT, before any §153 run; Audit approved the pass-skip rule, with two clarifications).**
* **(1) The denominator.** Items with b_j = 0 are excluded from both shares, since their row-skip and full-product values coincide; their count is reported
  (`n_bj0_excluded`).
  - SERIAL (PURE / REPEAT) is therefore **row_skip_consistent ≥ 0.90 and full_product < 0.10, over items with b_j ≠ 0**, at both skips. There is no chance term; this
    replaces "− chance".
* **(2) What a skip is.** The step-j pass is not run at all, and the memory goes unchanged to pass j + 1, which keeps its own step token j + 1. Later passes are not
  renumbered.

**Note 133-12 (2026-10-08 00:01 EDT; after §143's PASS, before any stage-2 run). FULL-GROW joins §133 stage 2 as the S5×4 full-width cell (§143: "FULL-GROW joins only if §143 passes
(a dated note then)").**
* **Cell cap4|fullgrow:** §143's recipe on `--family s5x4`, with **note 136-2's PLATEAU trigger** for switch 1.
  - §143's loss threshold (2.5) may never fire where the narrow code stalls; note 136-2 chose PLATEAU for the same reason.
  - Switch 2 to 768 is as §143: at step ≥ 6,000 and ≥ 1,000 steps after switch 1, with lr × 64/768 on the new columns.
  - COPY rows and no restarts (as cgrownoise). Dirs `s133cap4_fullgrow_s{seed}`.
* **Rules unchanged:** CAP-PASS = long_acc ≥ 0.99 at 512 in ≥ 3/4 slots; the proof rule. The cell counts among the wide arms in 133-2's capacity reading.
* **Reading added (fixed now), against cgrownoise (the 64-d growth arm):**
  - FULL-GROW passes where cgrownoise fails: the 768 width supplies capacity (a benefit of full width, the question §143 left open).
  - Both pass: full width costs nothing at capacity.
* score_s133 gains the cell.

**Note 152-2 (2026-10-08 00:03 EDT, before any §152 run). The headline's ratios come from the run.** The fixed wording's "about 0.6× FLOPs / about 0.2× sequential passes" were design
estimates.
* Experiments' implementation smoke meters 7.10·10⁹ FLOPs (0.46× TC-M) and 12 sequential passes (0.21×). The DeltaMem branch is counted.
* The result quotes the ratios from each run's own `compute` block in that wording's place. The rule (fewer FLOPs and fewer passes than TC-M) is unchanged.

**Note 153-2 (2026-10-08 00:06 EDT, before any §153 run; from Experiments' implementation). Two definitions made explicit.**
* **ext_4x5.test_acc is S_5 mod 10⁸ exact.** The reader emits 8 digits, and a 4 × 5 product can have 9, so this is the same convention as ext_4x8_low8.
  - Reported beside it: the exact rate on items with S_5 < 10⁸, and their share.
  - Prediction (ii) and stage 2's C2 bar (≥ 0.9) read this field.
* **Tokens:** the times token is " ×" (one GPT-2 token, 13958) and PURE's register token is " _". Both are recorded in args.json (`times_token`, `register_token`).

**Note 152-3 (2026-10-08 00:14 EDT, before any §152 result; descriptive). The baselines' extrapolation on ext_4x8_low8's own items.**
* §152 reports ext_4x8_low8: 4-digit a × 8-digit b, with the low 8 digits of S_j exact at every j, where j = 5..8 lie past training. Text CoT and implicit CoT have not been
  measured on the same items.
* `theory/eval_ext_baselines.py` covers both, on the prefix problems a × (b_1..b_j), j = 4..8, from the same generator and seed. It scores the low 8 digits exact (and the
  full product), on CPU at Idle priority.
  - **TC-M:** free greedy CoT; the answer is taken after "####" and truncated to 4 + j digits.
  - **ICoT-SI:** the released checkpoint, in icotsi98's format.
* **Smoke (20 items):** both baselines are exact at j = 4 and at 0 at j = 5. Both ignore b_5: ICoT-SI returns its j = 4 answer, and TC-M writes four partial products.
* **Reading (descriptive):** §152 extrapolating where both baselines stay at 0 would separate the methods on C2 and C3. The latent passes take more steps for longer inputs; a
  fixed text recipe and a single implicit pass do not.
  - **Caveat, stated now:** §152 is handed b_j at each pass (the designer's segmentation). Its extrapolation tests the unseen position tokens j > 4, not item-agnostic
    generalisation, which is §153's question.

**Note 152-4 (2026-10-08 00:33 EDT, before any §152 result; the user's research pass via Audit). A prediction from 2510.00184, stated now.**
* **The paper:** Bai, Pres, Deng et al. (2510.00184) reverse-engineer ICoT's multiplier. An attention DAG caches pairwise partial products.
  - Standard fine-tuning falls into a local optimum that lacks the long-range dependencies middle-digit carries need. That is our middle-column wall (§137–§148, PLAIN
    included).
  - An auxiliary loss predicting the running sum through a linear probe removes it.
* **§152's reader** decodes the teacher's running sum S_j from the memory: a decoder version of that inductive bias.
* **Predictions:**
  - **A enters** (A s1 ≥ 0.95).
  - **CTRL** (final only) is partial and below A. 2510.00184 predicts the local optimum, but §147's additive memory makes the running sum native (columns 3–7 at 0.92–0.99).
  - **If A drops:** 2510.00184's mechanism does not transfer to a memory-carried running sum at this budget, and note 152-1's layout readings apply.
* **C4:** 2510.00184's targets are computed running sums, hand-built and outside the current C4 wording. §152's come from the CoT text, which is allowed. Whether a computed
  running-sum probe loss is acceptable is the user's call (asked by Audit).

## 154. SHIFT-MEM: local state and relative keys for the DeltaMem's length generalisation (C2) (the user's research pass, 2026-10-08 00:30 via Audit; pre-registered 2026-10-08 00:41 EDT, after Literature's check c369c459 and Audit's draft review, before any implementation or run)

**Why.**
* **§147's memory does not extrapolate.** ext_4x8 is 0.0. Columns 1–4 are fine (1.0 / 1.0 / 0.989 / 0.941), but **column 5 is exactly 0.000** and 931 of 1,000
  first errors fall there.
  - Exactly 0 is not chance (0.1): it is a learned 4×4 schedule.
  - With no column index in the input, the model must track where it is from the pair count. Pair #14 starts column 6 in 4×4 and is still column 5 in 4×8.
* In 4×n the accumulator is **bounded**: a column holds at most 4 products (≤ 4·81 + carry). So 4×8 needs no new magnitudes, only addressing (Literature: "4×n →
  4×(n+k) along the multiplier").
* **Prior** (c369c459, `docs/LIT_ARITH_MEMORY_LG_2026-10-08.md`):
  - Absolute column keys learned for columns 0..7 give no key for column 8+.
  - The only recurrent length-generalising arithmetic (Neural GPU 1511.08228, Grid LSTM) uses **local / relative state**. Position coupling 2405.20671 and Abacus
    2405.17399 are attention-only.
  - The two routes Literature names, **(a) relative / rotary keys** and **(b) a shift-register memory**, are both **untested for memory carriers: new if tried**.

**Design: §147's digit-pair 4×4 DeltaMem exactly** (multpair `--carrier deltamem`; 6 additive + 6 gated-delta heads after block 6; log gains; fade-in; 24k; bs 64;
restart slots with 147-1's misfire rule; the memory is the only cross-segment channel). Two arms change only how the additive store is addressed:
* **SHIFT (route b, a one-slot shift register):** at every new-column segment (flag " ;") and at the flush, the harness **zeroes the additive heads' state** before the
  segment runs. The gated-delta heads are untouched.
  - The additive store then holds only the current column. That column's sum is the read at every pair.
  - The carry crosses the column boundary only through the delta heads (a bounded value, replaced each pass).
  - Every pass is then the same local computation:
    - add this pair's product;
    - at the column's first pair, add the carry read from the delta heads;
    - replace the delta heads' carry with ⌊sum / 10⌋;
    - emit the sum mod 10 (read at the column's last pair).
  - No column count is needed.
  - **Scaffold disclosure (Audit):** the reset is hand-wired. The harness applies it at the known column boundary, which is already an input token (" ;"). So a SHIFT
    pass reads **"a memory with a task-structured reset"**, not a learned one. The learnable successor is stage 2 below.
* **ROTARY (route a, relative keys):** the additive heads' feature vectors are rotated by the pair's column index c, RoPE-style (2104.09864):
  - k̃ = R(θ·c) φ(k) and q̃ = R(θ·(c + δ_h)) φ(q), with δ_h = 0 for three additive heads (read the current column) and −1 for three (read the previous column, for the carry).
  - Read weights then depend only on the column offset, so column k + 1 is read by the same operation as column k. Nothing is reset.

**Rules (fixed now).**
* **PASS(arm)** (§147's rule): ≥ 3/4 slots with test_acc ≥ 0.99 at 4×4 and SERIAL (row-level donor after pairs 4 / 8 / 12, as §151).
* **LENGTH-GENERAL(arm)** (C2): ≥ 3/4 accepted runs with **ext_4x8 test_acc ≥ 0.9** (all 12 digits exact on 4×8; §147 gets 0.0).
* **Proof:** slot 1 per arm. An arm stops if slot 1 has 4×4 test_acc < 0.95 **and** ext_4x8 < 0.5. Otherwise slots 2–4 complete.
* **IMPROVES** (descriptive): 4×4 columns 3–7 all ≥ 0.95.
* **Readings:**
  - **SHIFT LENGTH-GENERAL:** a memory carrier generalises in length when its active state is local. That is the Neural GPU principle in a pretrained LM's memory (new).
  - **ROTARY LENGTH-GENERAL:** relative keys suffice, with no reset.
  - **Neither:** the per-column computation or the emission does not transfer. The 4×8 per-column profile and the carry probe locate it.
  - **SHIFT PASS at 4×4 where §147 dropped (0.743):** cross-column interference in the additive store was §147's ceiling.
  - **SHIFT passes 4×4 but fails ext_4x8 (Audit):** the reset is not sufficient for addressing; the carry hand-off through the delta heads is still count-based. The
    carry probe by column locates it.
* **Descriptive:**
  - per-column accuracy at 4×4 / 4×8, and ext_4x16 (columns to 20);
  - R² of the column sum from the additive read, per segment;
  - **the carry probe by column:** held-out R² of each column's carry-in from the delta heads at the column's first pair, at 4×4 and at 4×8;
  - α by head.
  - **A C7 edit test (SHIFT):** at a column's last pair, add δ·u to the additive read, for δ ∈ {−2, −1, +1, +2} (small, reported by δ; Audit). Here u = w / ‖w‖² for the held-out ridge probe w of the
    column sum, so the decoded sum moves by exactly δ. Report the share of rows whose emitted digit becomes (s + δ) mod 10, against the unedited rate. Decoded, edited and
    causally obeyed, it is the program's first C7-style test.

**Stage 2, pre-registered now (conditional on SHIFT LENGTH-GENERAL; Audit): LEARNED-RESET.**
* SHIFT with the hand-wired reset replaced by a **learnable erase gate on the additive store**, in GDN-2's form (2605.22791). It is computed from each segment's residual,
  so the flag token can drive it, and initialised to reset on new-column segments.
* **Same rules.** A pass reads "the reset is kept when learnable". A further task-agnostic variant, initialised closed (as §147), would test whether the reset is
  discovered.

**Placement.**
* **SHIFT s1 first** (about $0.8, a 3090 at 24k). ROTARY s1 when implemented (it needs more code). Slots 2–4 by the rule.
* The fixed 24k budget compares directly with §147 (0.743, ext 0.0), so the runs do not wait on §151.
* **Implementation (Experiments):** multpair `--add_reset column` (SHIFT) and `--add_rotary column --rotary_read_offsets 0,0,0,-1,-1,-1` (ROTARY), ext_4x16, the
  probes and the edit test.
* **Gates:** step-0 function preservation; gradflow into every DeltaMem parameter; smoke_plain; Audit's diff review.
* Dirs `s154_{shift,rotary}_4x4_s{seed}`. Scorer: `theory/score_s154.py`, committed with this registration.

**Note 154-1 (2026-10-08 00:57 EDT, before any §154 run; descriptive only). The C7 edit test made precise.**
* **One edited column per rollout.** For each column k = 2..7 and δ ∈ {−2, −1, +1, +2}, the edit is applied only at column k's emission step, on the score half.
  - Editing every emission step at once would change the downstream carries and leave "(ĉ_k + δ) mod 10" undefined after the first edited column.
* **The probe's target is the accumulator** ĉ_k (column sum plus carry-in), not the emitted digit. So u = w / ‖w‖² moves the decoded accumulator by exactly δ.
* **Reported per (k, δ) and pooled:**
  - `shifted_share`: column k's digit = (ĉ_k + δ) mod 10;
  - `unedited_share`: the same rows unedited, digit = ĉ_k mod 10;
  - **`carry_follow_share`** (new): column k + 1's digit = (ĉ_{k+1} − ⌊ĉ_k/10⌋ + ⌊(ĉ_k + δ)/10⌋) mod 10, i.e. the next column uses the edited carry.
* No PASS or LENGTH-GENERAL rule changes.

**Note 147-4 (2026-10-08 01:10 EDT, before any §151 / §152 / §154 result; Experiments' structural observation, Audit's request). DeltaMem's writes are input-only, so its claims are
restated.**
* **The fact.** DeltaMem's keys, values and gates come from LN(x) at block-6 *entry*. x there is blocks 0–5 over the current segment's own tokens (attention inside the
  segment). The branch's read enters after that point and feeds only blocks 6–11 at the same position. Each segment is a fresh pass.
  - So the state after every step is **a fixed function of the input tokens alone**: an additive sum and an order-dependent delta recursion of input-derived features.
  - No read, no computed result and no carry is ever written back. All nonlinear use of the memory happens at read time.
* **§147 restated.**
  - *"The latent carries the running computation"* is withdrawn. What §147 shows: **a linear memory of input-derived pair features breaks the middle-column wall, with
    read-time carry resolution.**
  - Each pass forms its pair's product in-pass, the memory sums those products across passes (carry-save), and blocks 6–11 resolve column sums and carries when they read.
  - The whole-memory swap shows the answer follows the donor's *stored* sums. That is necessarily true of such a buffer, so it is not evidence of serial use of computed
    intermediates.
  - **C1 in its north-star sense (hop-by-hop use of computed intermediates) cannot be met by this design.** The SERIAL numbers stand as measured, under the narrower name
    "the stored sums are used".
* **The registered §151–§154 readings, reworded now (rules unchanged):**
  - §151 PASS reads *"exact multiplication through an input-driven linear memory with read-time carry resolution"*, not "C1 by design".
  - §152's headline reads *"matches text CoT at less compute through an input-driven carry-save memory (the passes' products accumulate; the reader resolves the carries)"*.
    The C3 compute comparison stands as measured. "A latent memory that carries the computation" is withdrawn.
  - §153 VISIBLE reads the same way. PURE and REPEAT move (note 153-3).
  - §154: see note 154-2.
* **The fix**, a write-after-read memory where the state is genuinely recurrent, S_{t+1} = f(S_t, input_t), is registered separately (§155).

**Note 154-2 (2026-10-08 01:10 EDT, before any §154 result). SHIFT under input-only writes (147-4), and the C7 test.**
* **The registered carry path cannot be implemented.** "Replace the delta heads' carry with ⌊sum / 10⌋" fails because ⌊sum / 10⌋ exists only after the read, and nothing
  after the read is written.
  - Under the reset, the carry can come only from the delta heads' input-driven recursion. For example, stored products under a flag-driven decay α(" ;") → 0.1, a decimal
    shift on stored inputs.
  - That is possible in principle, but α stayed near-frozen in §147 (147-3).
* **SHIFT s1 runs as built** (the reset is unchanged). Its readings are amended:
  - **PASS or LENGTH-GENERAL:** the delta heads' input-driven recursion carries the carry across a per-column reset.
  - **Failure on carry-dependent digits (the predicted outcome):** input-only writes cannot carry a computed carry across a reset, and §155 is the fix.
  - **Descriptive (added):** per-column accuracy split by carry-in = 0 against ≠ 0 (the signature); α on new-column against same-column segments.
* **C7 (Audit):**
  - **The C7 test is the state edit.** At column k's emission step, before the read and before that step's writes, perturb the additive state along the direction whose read
    moves ĉ_k by δ (δ ∈ {−2, −1, +1, +2}). Score `c7_state_edit.shifted_share`: the emitted digit = (ĉ_k + δ) mod 10.
  - **Pre-stated:** under SHIFT the next column's carry does *not* follow. The reset zeroes the edited store, and nothing computed is written.
    `c7_state_edit.carry_follow_share` is expected at the unedited rate. The propagating test belongs to §155.
  - **The read edit is descriptive only.** It is local to the emission position, and `carry_follow` is read only under the state edit.
  - A delta-head carry edit (δ·u′ along the carry probe, after column k's last pair) is also descriptive.
* ROTARY is deferred to §155's architecture: its offset −1 read would give the previous column's stored products, not its carry-in.

**Note 153-3 (2026-10-08 01:10 EDT, before any §153 run). PURE and REPEAT cannot work with input-only writes (147-4).**
* PURE's pass j sees [EOT, step j, " _" × 4], identical across items. Its writes are therefore item-independent, and passes 1..K store nothing about the operands held in
  memory.
  - The decode would have to compute the whole product from pass 0's memory in one go. That is the direct-answer wall (§148: 0.002), so PURE would fail by construction and
    uninformatively. REPEAT likewise.
* **PURE and REPEAT move to §155's write-after-read memory.** They are additionally gated on §155's proof, with the rules and readings unchanged.
* **VISIBLE is unaffected** (it sees the problem and the step each pass, so its writes can form that row's products, as §152's do). It stays gated on §152 A s1, with 147-4's
  wording.

## 155. DELTA-R: a write-after-read memory, so the state can hold computed results (note 147-4's fix; the user's research pass; pre-registered 2026-10-08 01:19 EDT, after Literature's check c55a1f15 and Audit's draft review, before any implementation or run)

**Why.**
* Note 147-4: DeltaMem writes from block-6 *entry*, so its state is a fixed function of the input tokens.
  - No pass can store what it computed. Serial use of computed intermediates (C1), state edits that propagate (C7) and passes that compute on memory-held operands
    (§153 PURE) are impossible by construction.
* **The fix:** compute the writes (k, v, α, β) from the residual *after* the read has been processed, at the top of the stack. Then S_{t+1} = f(S_t, input_t) is
  genuinely recurrent: whatever pass t computes from S_t can be written.
* **Prior** (c55a1f15, `docs/LIT_S155_DELTA_R_2026-10-08.md`):
  - The principle is known: write-from-the-top recurrence in RMT, Memformer 2010.06891, T²MLR (the same topology with a vector state) and the Feedback Transformer.
  - ARMT and Infini-attention are recurrent only because they keep a memory at every layer. 147-4 is the single-memory-layer degenerate case.
  - **A matrix additive + gated-delta memory read mid-stack, written from the top, as a pretrained LM's only cross-pass channel and tested by state edits, is new as scoped.**

**Design: DeltaMem-R.**
* **Read as now:** q from LN(x) at block-6 entry reads the pass-start state S_t, and x ← x + fade·W_o·read.
* **Writes:** k, v, α, β from u = (1 − γ)·LN_6(h_6) + γ·LN_w(h_top) at every token of the pass. h_6 is the residual entering block 6, as now, and h_top the residual
  after block 11 (pre-ln_f). The writes are applied sequentially as now, giving S_{t+1}. Reads within a pass see only S_t.
  - **γ, the function-preserving migration (Literature):** γ = 0 until 2k, then linear to 1 at 6k.
    - At γ = 0 the writes are exactly §147's, so training starts from §147's working input-only solution, not an empty memory (otherwise it is a re-entry problem).
    - Recurrence then enters gradually.
  - **Bounded writes:** LN_w normalises h_top, so v is bounded for bounded weights. The delta heads' k is L2-normalised as now. This guards the read → top → write loop in the
    no-decay additive heads.
* **Initialisation:** LN_w comes from block 11's ln_1. The write projections are §147's (block 6's c_attn slices), shared across the mix. The gates are as §147
  (α ≈ 0.993, β = 1). W_o = 0 with the 500-step fade-in, so the pretrained function is unchanged at step 0.
  - Literature's small-β initialisation is replaced by the migration: the loop only forms as γ grows, by which time β has trained.
  - Full BPTT with multpair's global gradient clipping at 1.0. Truncated BPTT is only a fallback (it costs accuracy, 2604.21106).
* **Everything else is §147's digit-pair 4×4 recipe:** 6 additive + 6 gated-delta heads, 147-2 gains, 24k, bs 64, restart slots with 147-1's misfire rule.
* **Arms:**
  - **R:** DeltaMem-R, no reset.
  - **SHIFT-R:** R plus §154's per-column reset of the additive store. The network can now compute ⌊ĉ_k / 10⌋ from its read and write it into the delta heads, which is
    §154's intended design.
  - **Fallback, fixed now (Audit): writes from block 9's output.** It runs for an arm only if that arm's top-write slot 1 shows non-finite steps, a gradient blow-up (the
    write-gradient norm above 100× its 1k-step median), or no entry by the 2k flag (exhausted).
    - Top-residual features are tuned for the unembedding, and BPTT through 6 blocks × 17 passes is deep, so the fallback is fixed now, not chosen after the fact.
* **Recurrence forms only as W_o and γ grow.** At step 0 (W_o = 0, γ = 0) the state is input-only.

**Rules (fixed now).**
* **PASS** (§147's rule): ≥ 3/4 slots with test_acc ≥ 0.99 at 4×4 and SERIAL (row-level swap after pairs 4 / 8 / 12).
* **LENGTH-GENERAL** (§154's rule): ≥ 3/4 accepted runs with ext_4x8 ≥ 0.9.
* **RECURRENT USE** (SHIFT-R; the C1 test this design makes possible).
  - **The edit:** at column k + 1's first pair (k = 2..5, δ = ±1), make a rank-1 edit of the delta heads' state. It is the minimum-norm edit that moves the held-out carry
    probe (column k + 1's carry-in, read at that pair's query) by δ.
  - **What is scored:** **propagation_share** is the share of rows where column k + 2's digit follows the edited chain, i.e. (ĉ_{k+2} − r_{k+1} + ⌊(ĉ_{k+1} + δ) / 10⌋)
    mod 10. It is counted only on rows where the edit changes column k + 2's carry.
  - **Under input-only writes this cannot happen.** In §154 SHIFT, column k + 2's carry is never recomputed from the edited state, so SHIFT is the null comparator and is
    scored the same way.
    - **Its expected null value (Audit):** propagation_share = the unedited-next rate, i.e. the share of those rows whose unedited column k + 2 digit already equals the
      edited-chain value. §154 is scored against exactly that.
  - **The carry direction** is fitted on held-out items. **Nulls (Literature):**
    - a random rank-1 edit of equal norm, scored the same way;
    - §154 SHIFT (no recurrence).
  - **RECURRENT USE = propagation_share ≥ 0.8 in ≥ 3/4 accepted SHIFT-R runs, with the random-edit null below 0.3.**
* **Readings:**
  - **SHIFT-R PASS + LENGTH-GENERAL + RECURRENT USE:** a pretrained LM's memory carries computed intermediates step by step, as a local recurrent process that
    generalises in length. That is C1 in the north-star sense, with C2 and C7 edits.
  - **R ≥ §147 at 4×4:** storing computed results helps even without the reset.
  - **SHIFT-R fails where SHIFT also fails:** the reset is the obstacle, not the write path.
  - **A failed linear edit** reads "not linearly editable", not "not carried" (Literature (c); onion representations, 2408.10920).
  - **Instability** (non-finite losses, or loss blow-ups after the fade-in): the deep recurrence through blocks 6–11 needs stabilising. That is a registered caution, not a fix
    loop.
* **Descriptive:**
  - **recurrence strength per 1k window (Audit):**
    - the share of the write's variance explained by the read: R² of the write features from the read, against from the input alone;
    - ρ = ‖∂S_{t+1}/∂S_t‖ by a JVP on a fixed batch. That ratio is meaningful only now; earlier DeltaMem ρ was input-driven.
    - A failure then reads as "recurrence never formed" or "formed but did not help".
  - per pass: memory norm and the write-norm ratio (top / block 6, while γ < 1);
  - per-column accuracy (by carry-in 0 / ≠ 0); the carry probe by column; colsum R²; §154's state-edit C7 test (the emitted digit and the next column's
  carry); α by head; gradient norm through the recurrence by segment.

**Placement.**
* R s1 + SHIFT-R s1, about $0.8–1 each (24k, a 3090). Slots 2–4 by the rules.
* §153 PURE / REPEAT run on DeltaMem-R after R's proof (note 153-3).
* **Implementation (Experiments):** deltamem.py gets a write source option (default unchanged, so §147 / §151 / §152 / §154 are unaffected): `read(h6, S)` at block 6
  and `write(h_top, S)` after block 11. Then multpair `--mem_write top` and the edits.
* **Gates:** step-0 function preservation; gradflow showing nonzero ∂loss / ∂(write parameters) **through at least two hand-offs** (BPTT flows, not a one-step gradient;
  CUDA bf16); smoke_plain; memory; Audit's diff review.
* Dirs `s155_{r,shiftr}_4x4_s{seed}`. Scorer: `theory/score_s155.py`, committed with this registration.

**Note 152-5 (2026-10-08 01:53 EDT; Audit's request; descriptive, outside §152's registered verdict). One 96k run of A s1, and the exact-storage test.**
* **The run:** `s152_a96k_4x4_s1`, §152 A's arguments with `--steps 96000` (cosine over 96k), run beside §155.
  - **§152's registered verdict stays DROP whatever it shows.**
  - **B-EARLY opens only if the 96k run reaches test_acc ≥ 0.95 and its gate opens** (forked from its gate.pt; §152 B's rules).
* **Exact storage (Audit):** R² 0.90–0.97 on integer column sums up to about 300 is not exact storage. So "stored, resolved at read time" is not yet separated from "stored
  imprecisely".
  - For A s1 and CTRL s1 at 48k (local models; CPU) and for the 96k run, report the **rounded exact-match accuracy of the held-out column-sum readout, per column**
    (`colsum_exact_by_column`): linear ridge and kernel ridge on the additive state, fit on half, scored on the other half.
  - **Reading, fixed now:** if exact-match is well below 1 (< 0.9) at columns 4–7, 147-4's "read-time carry resolution" weakens to **"storage precision or read-time
    resolution"**. §155 is then read against that.

## 156. SEP-SLOT: a prediction slot per answer digit for carry resolution (the user's research pass via Audit; pre-registered 2026-10-08 02:01 EDT, after Literature's check d27dae0b and Audit's draft, before any implementation or run)

**Why.**
* Two of our failures are carry resolution while digits are emitted:
  - §148's plain GPT-2 direct answer: 0.002, per column 1.0 / 1.0 / 0.41 / 0.13 / 0.08 / 0.10 / 0.66 / 0.96;
  - §152's reader at the fourth row: A s1 0.331.
* In our formats each answer digit is one BPE token (" 8"). The position that emits a digit is also where its carry must be resolved, and there is no separate slot.
* **Prior** (d27dae0b, `docs/LIT_SEPARATOR_CARRY_2026-10-08.md`; 2610.06605):
  - In small transformers trained from scratch on 4×4 with no CoT, a separator token before each answer digit lifts exact match from 1 % to 89 %.
  - The separator position holds the carry-in as a ring (angle) code, and patching the slot transfers the carry. That is up to 84 %, for one middle column of their best
    model; the other columns assemble the carry at the neighbouring slot first.
  - From pretrained GPT-2, a smaller gain is expected.
* **Pure formatting, so C4 is unaffected.** The slots are extra computation positions, counted in FLOPs.

**Design.**
* **The separator:** "|" (GPT-2 id 91), a forced input placed before each answer digit: | d₁ | d₂ … | d₈.
  - **LSB first, as §148** ("the product's 8 digits LSB first"); §152's reader is also LSB first.
  - Separators are **forced in eval exactly as in training, identically in all arms.** They get no loss and are never generated, so accuracy counts only digit positions.
  - The digit after each separator is predicted at the separator's position.
* **(a) On §148's PLAIN direct-answer 4×4** (loop148 `--loops 1`; the same recipe, LR, 24k budget and GPU class as §148, with the GPU recorded per cell). **Two seeds per
  arm (Audit), s1 / s2:**
  - **SEP:** separators interleaved;
  - **PAUSE:** the same 8 "|" tokens placed before the answer, not interleaved. This is the compute-matched control (Literature (2)): a slot per digit against
    more positions.
  - The comparator is §148 PLAIN (0.002, RTX 3090). If an arm runs on another GPU class, PLAIN is rerun on that class (`s156_plain_4x4_s1`) and serves as the comparator.
* **(b) On §152 A's reader decode:** **A-SEP**, 48k, otherwise §152 A's recipe. The comparator is §152 A s1 (0.331).

**Rules (fixed now; descriptive verdicts; two seeds per arm in (a), one gate seed in (b)).**
* **(a)** An arm reads **≥ 0.5 only if both seeds' test exact is ≥ 0.5**, and reads **< 0.5 only if both seeds are below**. Mixed seeds read "seed-dependent (no
  verdict)".
  - **SEP ≥ 0.5, PAUSE < 0.5:** a per-digit slot, not more positions, makes in-pass carry resolution learnable in a pretrained LM. The middle-column wall was partly a
    missing-slot artefact.
  - **SEP and PAUSE both ≥ 0.5:** extra positions, not their placement.
  - **Both < 0.5, but SEP − PAUSE ≥ 0.2 in both seed pairs (Audit):** placement helps but is not sufficient at this budget. Descriptive; no follow-up is implied.
  - **SEP < 0.5 otherwise:** at this budget, the wall is not about position count.
* **(b)** **A-SEP ≥ 0.95:** slots close §152's 4-row resolution. A-SEP s2–s4 and its B-EARLY then follow under §152's rules as a new arm.
* **Diagnostics:**
  - per-column accuracy (do the middle columns, the paper's last learned, move?);
  - a held-out ring-angle carry probe (cos / sin of 2π · carry / C) at each separator slot;
  - **slot patching** as a readout-level C1 check: one middle column's slot residual swapped from a donor with a different carry-in, scored as the share whose emitted digit
    follows the donor's carry;
  - FLOPs and sequential passes with the extra positions (Meter).

**Placement.**
* SEP s1 / s2 + PAUSE s1 / s2 (about $0.15 each, 39 min on a 3090) and A-SEP s1 (about $0.4, the gate for (b)), beside §155.
* **Implementation (Experiments):** loop148 and multrow get `--answer_sep interleave|pause --sep_token "|"`.
* **Gates:**
  - a smoke assertion that the tokenised sequence is exactly [… "|", " d", "|", " d", …], with "|" = 91 and no merge of " d|" in any context;
  - an assertion that PAUSE's total sequence length equals SEP's;
  - gradflow_check on the new format path;
  - smoke_plain on both harnesses without the flag (unchanged);
  - Audit's diff review.
* Dirs `s156_{sep,pause}_4x4_s{1,2}` and `s156_asep_4x4_s1` (plus `s156_plain_4x4_s1` only if the GPU class differs). Scorer: `theory/score_s156.py`, committed with this registration.

**Note 155-1 (2026-10-08 02:04 EDT, before any §155 run; Audit's diff review e3fed0e6). Three clarifications of §155's rules.**
* **(1) The fallback trigger, registered meaning.**
  - "Write-gradient blow-up" = the gradient norm over the **write rows only** (LN_w, the k / v rows of the write projections, the gate) above 100× **its median over the
    preceding 1k steps**.
  - It is scored **only after write_mix_start (2k)**. Before then the W_o fade-in and γ's start move write gradients from about 0, which is not instability.
  - Non-finite steps and slot exhaustion are unchanged.
* **(2) Recurrence strength uses rho_loop.**
  - ρ = ‖∂S_{t+1}/∂S_t‖ is about 1 by construction (the additive carry-through; α ≈ 0.993), so it cannot separate "formed" from "not formed".
  - **rho_loop = ‖J v − J v|_{fade = 0}‖** on a fixed random direction v isolates the read → network → write path, and the registered reading uses it.
  - **"Recurrence never formed"** = rho_loop ≈ 0 with write_var_from_read_r2 ≈ write_var_from_input_r2. ρ is reported beside it.
* **(3) R against §147 changes two things:** the write source, and the read lag (R reads only the pass-start state; §147's read at token t sees that pass's earlier writes).
  - **A descriptive arm R-γ0 is added:** γ held at 0, so writes come from block 6 as in §147 while reads use the pass-start state. About $0.8; dir `s155_rg0_4x4_s1`.
  - **R > R-γ0:** the write source helps.
  - **R-γ0 < §147:** the read lag costs.
  - "R ≥ §147" is read only through that pair.

**Note 152-5 result (2026-10-08 03:33 EDT; descriptive; A s1 and CTRL s1 at 48k, CPU; `theory/results/colsum_exact_s152*.json`, `read_exact_s152.json`).**
* **The registered test** (s152's probe, 2,000 pooled rows): exact ≤ 0.19 per column for A and CTRL. By the fixed reading, 147-4 weakens to **"storage precision OR read-time
  resolution"**.
* **That test was limited by its power, not the store.** With λ at its grid floor and 8,000 train items (final state, λ ≤ 1e-7), A's linear readout of the column sums is exact
  1.0 / 0.985 / 0.944 / 0.857 / 0.723 / 0.805 / 0.866 (R² ≥ 0.9997). The state holds the sums almost exactly.
* **What the model reads (Audit's model-internal probe).** The reader is teacher-forced, with a ridge on the 768-d captures, fit on 4,000 train items and scored on test.
  - The read output's linear readout of the accumulator ĉ_k is exact only 0.11–0.35 (A) / 0.05–0.34 (CTRL) at columns 1–7 (R² ≥ 0.95). It falls further through
    blocks 6–11.
  - Yet the **teacher-forced digit accuracy** is:
    - A: 1.0 / 1.0 / 1.0 / 1.0 / 0.999 / **0.338** / 0.998 / 1.0;
    - CTRL: 1.0 / 1.0 / 1.0 / 0.999 / **0.094** / **0.197** / 1.0 / 1.0.
* **Reading.**
  - The model does not carry the accumulator as a linearly readable integer at any layer, yet emits most digits exactly. It computes digits from a non-integer code.
    **Linear integer-closeness of the read is therefore not a valid diagnostic for this model.**
  - **§152's failure is localised.** A fails one column (6) and CTRL two (5 and 6): the columns with the largest carry-ins. The reader fixed column 5.
  - "Read-time carry resolution" narrows to **"carry resolution at columns 5–6, still being learned at 48k"** (A's gate was rising at the end). The 96k run (b165) tests the
    budget.

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

**§110 result (2026-10-04 18:30 EDT; 5 seeds, run by the experiments session (s1–3 e394f62d, s4–5 de1c71bc); scored by
Theory alone with `theory/score_s110.py`, committed before any run; `theory/results/score_s110.json`). SELF2, C1 and C6
are not met. The STUDENT trigger fires.**

| seed | cumulative 512 / 2,048 | donor / recipient (slots 1 / 2 / 3) | reading |
|---|---|---|---|
| s1 | 0.9505 / 0.826 | 0.011 / **1.0**; 0.009 / 0.083; 0.012 / 0.069 | slot-1 bypass |
| s2 | 0 / 0 | chance | never acquired (gate ≈ 0) |
| **s3** | **0.999 / 0.994** | **0.962 / 0.983 / 0.968** donor; recipient ≈ 0.01 | **C4 (qualified) + C1 + C2** |
| s4 | 0 / 0 | chance | never acquired (gate ≈ 0) |
| s5 | **0.9915** / 0.954 | 0.013 / **0.432**; slots 2–3 neither | partial bypass |

* **SELF2 is 2/5** (s3, s5; needs ≥ 4/5). **C1 is 1/5** (s3). **C6 at 2,048 is 1/5** (s3). Median at 512 is 0.9505,
  range 0–0.999.
* **The STUDENT arm triggers** (SELF2 fails; 3/5 seeds reach ≥ 0.90 at 512). As registered, every seed's final pool
  trains a fresh GEN-4-MASK model, 5 students.
  - s2 and s4 kept almost nothing, so their students get essentially final-only targets, and STUDENT can reach at most
    3/5.
  - The informative comparison is student against teacher on s1, s3 and s5.
* **Reading.** Self-training by length with a masked fork gives, in **one seed (s3)**, C4 (≤ 5 % identity-start shorts
  covering 71.5 % of transitions) + C1 (donor, every slot, including slot 1) + C2 to 2,048. Its k = 1 system is
  certified on all 14,400 transitions (110-5).
  - Reliability is the gap: two seeds never acquire, and two acquire with a slot-1 bypass.
  - That is the same no-read and bypass family as §108. §113 (READ inside self-training) is conditional on §111.
* **Audit's reproduction (77db3092).** The panel reproduces; gate maxima are s2 0.0021 and s4 0.0023, against ≥ 0.999 for s1 /
  s3 / s5.
  - **s5's slots 2–3 are neither donor- nor recipient-consistent** (0.007 / 0.011 and 0.007 / 0.005), yet mean-removal
    is 0.0077. The latents are necessary but do not transplant across problems: the state is encoded jointly with
    row-specific context (multiple mediators, as §104 / §97).
  - s5's STUDENT may show whether that encoding is an artefact of the training path.


**§110 STUDENT result (2026-10-05 03:45 EDT; 5 students by the experiments session; scored by Theory alone with
`theory/score_s110.py`, committed before any run). SELF2 is not met (3/5); C1 is 2/5; C6 at 2,048 is 1/5.**

| student (teacher) | 512 / 2,048 | donor / recipient (slots 1 / 2 / 3) | against the teacher |
|---|---|---|---|
| s1 (slot-1 bypass) | 0.999 / 0.982 | **1.0 / 0.011**; 1.0; 0.999 | **bypass gone: SERIAL** |
| s2 (never acquired) | 0 / 0 | chance | near-empty pool: never acquires |
| s3 (C4 + C1) | **1.0 / 1.0** | 1.0; 1.0; 0.999 | reproduced, exact to 2,048 |
| s4 (never acquired) | 0 / 0 | chance | near-empty pool: never acquires |
| s5 (partial bypass) | 0.9955 / 0.988 | 0.055 / **0.475**; 0.948; 0.995 | slot-1 bypass persists |

* **Reading.** The STUDENT pass reading ("the labels were good; the teacher's own training had locked in errors") does
  not apply: 3/5, needing ≥ 4/5. The two empty-pool seeds carry 112-4's start-reading basin into their students.
* Descriptive: retraining from s1's labels removes the teacher's slot-1 bypass (the student is SERIAL at every slot). The
  bypass came from the training path, not the labels. s5's does not go away.
* **§110 is closed.** §113 (the reader inside self-training) and §116 / §117 carry the line forward.