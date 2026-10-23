# DIGEST of THEORY §0–§96 (source: docs/archive/THEORY_s0-s96.md), written 2026-10-23 (Fri Oct 23 13:28 EDT 2026)

Scope: Theory-stream history 2026-09-29 .. 2026-10-03. Verdict words are the sections' own; numbers are copied from result blocks. "unclear" = text gives no verdict. §77 and §81 have no section of their own (only notes inside §76/§79/§80). Early toy setting = 2-layer Coconut on graph reachability (K latent steps, train depth <= 6); later = GPT-2 with finite-state tasks (S3/S4/A5/S5 group products) and boxes.

| § | title | question | verdict | key numbers | lesson / why | C |
|---|---|---|---|---|---|---|
| 0-1 | Question; thought is a pointer | Why do some latent reasoners stop at training depth? | descriptive | thought = sharp pointer (logit excess 4-10), moves one hop/step | failure begins at first untrained step K+1 | C2 |
| 2 | Mech 1: retrieval on step schedule | Is the pointer copied from earlier thoughts by step lag? | descriptive; causal | blocking retrieval past K rescues 17/23; 7-hop 24->41/59 | step-indexed retrieval = implicit clock | C2 |
| 3-5 | Schedule-bound theory; attenuation; architectures | What explains stopping; what to build? | theory (untested then) | pointer fades ~x0.8/hop; A1-A3 proposed | local map must be restoring | C2,C6 |
| 6,15 | Pilot A: Markov (no latent-latent attn) | Does removing retrieval route extrapolate? | PA1 unresolved, PA2 not supported, PA3 fails | C 3/6 extrapolate, far8-10 mean 0.526 | Markov removes one implementation, not the reason (slack) | C2 |
| 7-8 | Log; toy loop-gain sim | Does a restoring map extend reach? | descriptive | restoring reach > analog in toy | supports gain account | C6 |
| 9,19 | Pilot B: unity/anytime/restore arms | Do tight budgets, anytime, restore help? | PB1 supported (n=4), PB3 met, PB2 half, PB4 met on n=1 | AT 3/4 extrapolate; MR 0.709 far8-10 | restore module hurts learnability (MR 1/4, ATR 2/4) | C2,C6 |
| 10 | When circuit is chosen | When does retrieval vs local circuit win? | descriptive | stage 3 acquires first advance | marker later failed out of sample (§21) | C2 |
| 11 | Pilot C: slack dose-response | Does more K-slack lower extrapolation? | PC1 not supported | K5 3/4 = K7 3/4; K6 3/6; retrieval share 0.06/0.14/0.26 | dose-response form of slack account failed | C2 |
| 12 | Training-free repair at inference | Can block + gain boost fix failures? | positive (toy only) | 7-hop 24 -> 41 (block) -> 46 (alpha 1.3) | amplitude fix saturates ~46/59 | C2 |
| 13 | Public GPT-2 latent reasoners (CODI, SIM-CoT) | Do trained latent budgets extrapolate on GSM8K? | negative | CODI 0.425 at 6 latents, 0.401 at 12; SIM-CoT 0.393 | trained budget is a horizon; toy repair does not transfer | C2 |
| 14 | Transfer curves of local map | Does a one-step rule predict extrapolation? | out-of-sample rule failed | 8/15 correct (chance); in-sample 18/23 | P7 stays the reliable predictor | C6 |
| 16-17 | Pilot D: anytime supervision, 12+12 | Does anytime supervision give extrapolation? | PD1 supported; PD2, PD3 not met | AT 8/10 vs C 3/9 extrapolate, Fisher p 0.055; far8-10 0.543 vs 0.526 | gain only at d7; pointer falls back after K | C2 |
| 18 | Formal sketch of slack account | Why is continuation past K unidentified? | theory (Prop 1) | - | slack >= hops -> any visit-each-depth-once schedule fits | C2 |
| 20 | Pilot E: depth wall or schedule end? | Do anytime runs cross trained depth for larger K? | PE1 "DW supported"; PE3, PE4 not met, PE2 reversed | 0/12 anytime runs in E8+E3 cross | learned fall-back, not capacity limit | C2 |
| 21 | Out-of-sample stage-3 retrieval marker (addendum O) | Does retrieval share predict failure out of sample? | PO1 unresolved, PO2 not met | AUC 0.51 (in-sample 0.78) vs early-P7 0.92 | marker downgraded; failures use fade/restart too | C2 |
| 22 | Four conditions C1-C4; Props 4-8 | General theory of step extrapolation | theory | identification, homogeneity, stability, idempotence | Prop 8: one-pass-solvable problems bypass latents | C1,C2 |
| 23 | Pilot F: amplitude jitter, selection window | Does making map restoring add reach? | PF5 not met; PF4 not supported | J alone 1/8 competent; JAT 5/6 extrapolate, d8 0.568 vs AT 0.584 | restoring map gives stability, not continued advance | C6 |
| 24 | Pilot G: monotone latent set (A6) | Does a gated set memory help? | no verdict; PG4 met, PG5 unresolved | set ablation matters in 3/11 G, 0/13 GL | gate from 0 never opened; A6 untested | C2,C6 |
| 25 | Prop 9: slack creates the clock | Why does pointer reach only demanded depth? | theory + LE tests | p[K-1]~3 for K=6,8 alike | fractional speed needs a clock | C2 |
| 26 | Re-anchored episodes (CTM) | Does restarting the loop from a written node fix failures? | RA1-RA4 met | re2 7-hop 55/59 (31/34 failures rescued); single 24 | local operator generalises; counts reset by restart | C2,C7 |
| 27 | Pilot H: fine-tune repair | Can fine-tuning repair trained failures? | PH1-PH4 not met | FT-AT 1/18, FT-C 0/20 | critical period: brake is kept | C2 |
| 28 | Pilot I: periodic clock | Does a phased clock never run out? | PI1 narrowly not met (0.57); PI3 not met | pointer keeps advancing 4/7 vs 0/4 | removes part of stop key at learnability cost | C2 |
| 29 | Pilot J: chunk-reset attention | Does cutting latent history every 3 steps help? | PJ2 supported, PJ3 met, PJ1 not met | d8 median 0.604 vs 0.503; pair 6/8 | best from-scratch toy d8; foreshadows restarts | C2 |
| 30 | Zero-cost: renorm, randk, branches | Is amplitude restoration the gap to re-anchoring? | RN1, RN2 not met; RR1 not supported | renorm+block 45/59 vs 55/59 re-anchor | gap is not amplitude; F3a dropped | C2 |
| 31 | Pilot K: label-free tight budgets | Does random-tight budget match anytime? | PK1 unresolved, PK2-4 met | all 4 competent RT extrapolate | consistent with anytime, small n | C2,C4 |
| 32 | Pilot L: distil re-anchoring into one loop | Can a loop learn re-anchored behaviour label-free? | PL1-PL4 met | L-T8 19/20 extrapolate vs FT-C 0/20; d8 0.638 | strongest fix: label-free distillation of restarts | C2,C4 |
| 33 | Toward north star: CLR design, C7 tests | What to build and test? | design | - | checkpointed latent reasoner proposed | all |
| 34 | T-C7c: edit/round-trip of silent thoughts | Are silent thoughts causally editable? | E2-E4, E6 met; E1, E5 narrowly not | IIA 0.36; answers +0.83/-0.59 | pointer is causal but linear map imperfect | C7,C1 |
| 35 | Prop 10: confounded stop keys; P10e | Which count keys the fall-back? | "step key" | skip: peak depth +2; 13/17 models | fall-back keyed to counts, not hop limit | C2 |
| 36 | Counter clamp (inference) | Can clamping the count rescue failures? | CC2 met; CC1, CC3 not | freeze3 9/34 vs 0/34 control | works a little at in-range cost | C2 |
| 37 | Do L students extrapolate past training depth? | Depth reach of distilled loops | X1-X3 hold | GT students 17/20 one loop, d13-18 median 0.74; T8 2/20; self 0/20 | second deeper tight budget -> ~2x depth | C2,C4 |
| 38 | Pilot M: second label-free rung | Can ladder climb again label-free? | PM1 supported; PM2 narrow no; PM3 not met; M-2 filter: M2a supported | M-2 12/20 reach d13-18; 9/11 pairs p 0.033 | agreement filter contains error avalanche | C2,C4 |
| 39 | Pilot T12: step-adversarial thoughts | Does removing step info help? | PT4 fails | 2/8 competent | removing the count breaks learning | C2 |
| 40 | Prop 10 across architectures | Does mechanism hold in 3-4 layer/512 wide? | X9a-c hold | anytime p[K-1] 5 vs 2; passes 8 in 1/16; count decoded 25/25 | brake removed, fall-back kept | C2 |
| 41 | Second training depth: why length generalisation | Content key or moved count? | partial | train to 9 -> ~13-14 hops; 12 -> ~22; 18 -> flat to 23 | reach graded by trained depth (hypothesis) | C2 |
| 42 | Predictions for PROTOCOL8 | Does anytime help GPT-2 Coconut on ProsQA? | "Falsified, more strongly than the stated falsifier" | anytime 0.50 vs 0.56 Coconut; no latent seed competent at d11-12 | toy slack account does not transfer to GPT-2 ProsQA | C2 |
| 43 | Where is deep frontier read (layers) | Is the sweep in one layer? | L1 not met (4/8); L2 met (3/16) | - | half layer choice, half redundancy | C1 |
| 44 | Set-valued checkpoints zero-shot | Can one episode restart from several anchors? | S1, S2 not met | set3 - top1 -0.006 | sets need training; separate episodes work | C2 |
| 45 | CLR-1: train set-valued checkpoints | Does training sets help? | PC3, PC4 met; PC1 not supported (5/10); PC2 met on pooled 10/20 | edit IIA 0.84 (single 1.00) | checkpoint is causal channel; set no better than single | C7,C2 |
| 46 | Hop sweep of restart | Where does a written anchor help? | H1, H2 hold | 0.91-0.95 up to 6 hops, 0.51 at 7 | one restart covers exactly its trained horizon | C2 |
| 47 | Architecture review CLR | Best design after §§34-46? | descriptive | single-anchor checkpointed, label-free; d19-24 ~0.65 | bound every count by construction | C2,C4,C7 |
| 48 | Multi-start beams | Do beams help? | BM1, BM2 not met | B3 0.510 vs B1 0.517 at d19-24 | extra episodes add false positives | C2 |
| 49 | Learned write head + closure halt | Can a halt be label-free? | WH1, WH2 met | agreement 0.85; halts 3/4/5 segs for 12/18/24 | C5 v1-v2 works | C5 |
| 50 | Halt learned from final-answer reward | Does reward teach depth-tracking halt? | LH1 met 9/10; LH2 not met 0/10 | stops at 2 segs always | max-combined answers accumulate false positives | C5 |
| 51 | Single-target aggregator + learned halt | Does fixing the aggregator help? | TH1 not met 0/10; TH2 1/10 | stops at segment 1 | halt line closed | C5 |
| 52 | Frontier: new directions F1-F5 | What new designs? | idea list (not registered) | - | see Open threads | all |
| 53 | F2 bidirectional zero-shot | Meet-in-the-middle extends reach? | BD1 met; BD2, BD3 not met | 0.712 vs 0.501 fwd; FP 0.417 | recall up, precision down; parked | C2 |
| 54/54b | F1 annotated-input on GPT-2 | Learn per-token labels from final answers? | "AG1' failed in both arms" | soft/stoch 0/3; per-pass-hint 1.000 all 3 | per-pass targets learn, final answers don't | C7,C5,C4 |
| 55 | Kill test: fall-back reward-neutral? | Is fall-back free? | KT1 partial at boundary; KT2 met | kill/pos ratio 0.30; -0.4 pt acc | commit-gate buys no reach; D3 parked | C5,C6 |
| 56 | Entry-state normal form (interchange) | Are aged latents interchangeable? | NF1 normal form (against expectation); NF2, NF3 not met | GT students G 0.09 vs T8 0.84 | path independence tracks one-loop reach (post hoc) | C2,C7 |
| 57 | Origin of ProsQA cue | Why is Coconut good on ProsQA? | descriptive | proximity heuristic 0.91 | negative option always hangs off another person | C3 |
| 58 | F6 age-transplant training | Can shallow data teach age-invariance? | "Falsified by its own rule" | aged dG -0.04, no reach (6/10 all arms) | path independence is not cause of reach | C2 |
| 59 | ProsQA-NA | Neutralise cue w/o changing answers | descriptive | proximity 0.91 -> 0.50; prosqa_na residual 0.62 | p8_na clean; prosqa_na keeps second-order cue | C3 |
| 60/60b | Boundary memory Phase A | Learn Boolean-table merger? | Gate A not passed 0/4 (both) | identity baseline 0.877; best seed 0.88/0.84 | tables not algebraically valid; line stops | C1 |
| 61 | F1-S shallow final answers | Learn label rule from shallow answers? | S1-S4 not met (1/3); then 1/15 overall | s3 1.000 vs others chance | success was a numerics draw | C4,C7 |
| 62 | Depth-structured label noise | Does noise install age code? | hypothesis not supported | uniform G 0.67 vs clean 0.09; seam step 0.008->0.243 | self-labels must be near-clean | C4 |
| 63 | F4 round-trip translator | Make silent toy thoughts readable? | F4-1..5 all not met | RT 0.916; edit IIA 0.013 | translator never trained; uninformative | C7 |
| 64/64b | Permutation tracking (S3, A5) | Do count-stop and restart reappear off graphs? | A5 no verdict (0/10, also curriculum); S3 ST1 met, ST2 not | S3 9/10 gated, 0.997; A5 loss ln 5 | restarts fix S3 length; A5 never learned | C2,C6 |
| 65 | Restart-distillation on permutation | Can near-clean teacher give one loop reach? | D1 not met, D2 met | RD 0.658 (N 9-16), 0.06 (17-24) | limit is continuous state, not labels | C2,C4 |
| 66 | GPT-2 WC chunked loop on state tracking | Scaffolded test of C2/C3/C4 | gate passed; P2-4 met; P5 not; P1 uninformative | WC4 1.00 at N17-32, TC 0.14; no-latent WC0_4 also 1.00 | scaffold (short context + written state) does work, not latents | C2,C3 |
| 67/67b | C1 Design 1: six-code categorical latent | Can GPT-2 cell acquire automaton by marginalisation? | C1-1 not met | D1b certified 1/5; TAB 5/5; GPT-2 stuck at partial automaton | shared-cell acquisition with pretrained LM fails | C1,C4 |
| 68 | GSM8K step 2 (own arms) | TC/Coconut/written-state on GSM8K-Aug | paused by user, not run | refs: CoT 41.6, Coconut 33.1 (Dilgren) | "beat SOTA on smaller tasks first" | C3 |
| 69 | RL for discrete-latent acquisition | Do RL updates fix acquisition? | "Neither" - no method passes | table 0-1/10 (bon-4 5/10); MLP 0/10 | spectral baseline: data not the limit | C1,C4 |
| 70 | Operator-conditioned likelihood; tempered posterior | Do A or B certify? | neither passes; line closes | A mul 10/10 vs add 0/10 (descriptive) | multiplicative cell wins at matched size | C1 |
| 71 | S5 4x length, WC loop (DeltaProduct protocol) | Extrapolate with written-state chunking? | S1 met; S2 met with caveat | 1.000 at 512, 2,048 (16x); k=2 tail 2/5 certified | scaffolded, not C1; k=1 certifies 5/5 | C2,C6 |
| 72/72b | Where GPT-2 cell loses acquisition | Frozen vs random vs tuned vs toy | 72 uninformative (lr 1e-2 design error); 72b clear | PF 0/5, RF 5/5, PF-std 5/5, PT 1/5 | pretrained anisotropy (cosine 0.99) blocks head | C1,C4 |
| 73 | Stable latent: standardised readout, 10 seeds | Is a stable latent possible in GPT-2? | L1, L2, L3 all met | PF-std 10/10 certified; PT-std 10/10 | centring removes shared direction | C1,C6 |
| 74 | Paraphrase transfer to unseen phrasings | Does stable latent need pretraining? | P1, P2, P3 not met | 0/5 certified (parity-flip curriculum artefact) | curriculum x generator artefact | C1 |
| 75 | Scaling to S4, A5 | Does std readout scale? | gate fails; panel not run | PF-std 0.486 at N128 (S4) | blocks of states merged, no bijection | C1 |
| 76 | Free-table automaton S4/A5/S5; amortise | Does free table acquire; does LM learn from it? | (A) "std passes but ge3 fails"; (B) both arms fail | std 10/10 each; ge3 0/10; held-out rows 0.04-0.17 | needs short labels; hosting works, composition absent | C1,C4 |
| 78 | GSM8K length generalisation by step count | Does written-state chunking beat text CoT? | L2 not met, L3 met | TC long 0.022 vs WS-pad 0.018 | no gain over CoT on long problems | C2,C3 |
| 79 | Latent automaton needing pretrained LM (paraphrases) | Held-out phrasings? | P1 met; P2 fails (note 79-6) | PF-std held-out certificates 0/5; BOW 0.889 H1 | pretraining adds op discrimination, not paraphrase agreement | C1 |
| 80 | Latent chunk machine | Does GPT-2 compose unseen words? | P0 fails; stops | unseen-word 0.030 (k=2), 0.118 (k=3) | memorises words | C1,C2 |
| 82 | Acquisition ladder / footholds | Do groups help or trap? | "Neither" | ge3 0/5 all groups; std 3/3 | short-label need generic across groups | C4 |
| 83 | Self-gated restarts | Can solver pick chunk boundaries? | U1-U3 fail ("U2 fails, out-of-range overconfidence") | cumulative 0.000; miss rate 0.317 | confidence is out of range when unsure | C5,C6 |
| 84 | Oracle-stop diagnostic; decoy padding | Is halting the §78 bottleneck? | gate FAILS; closes | ORACLE long 0.030 vs own stop 0.015 | bottleneck is per-step computation past 2, not halting | C5,C3 |
| 85 | Short-label threshold; oracle init | Information or optimisation? | descriptive | p 0.1: 5/5 certified; 0.001: 0/5 | short labels at >= ~3-10% are needed | C4 |
| 86 | Latent restarts (one latent per op, snapped) | Carry per-op state in latents? | gate fails at 8k budget | slot1 1.000, slot2 0.112 | latent hand-off within chunk not acquired in budget | C1,C2 |
| 87 | Short-label warm start; annealing | Reach basin without short labels? | warm25 5/5; warm12, LTA, DA-EM 0/5 | warm25 first certified 10-15k | brief early short-label exposure suffices | C4 |
| 88 | Loop-aligned composition | Looped block with per-iteration targets composes? | k=3 1.000 3/3; k=2 0.061 | LOOP-end 0.159 | per-iteration targets are key | C1,C2 |
| 89 | Fade to latent (user idea) | Fade text CoT into hidden states? | F1-F3 arms scored; fade_std_dec best | N5-8 0.615; "all" 1.000 x5; "where" 1.000 in 1 seed | side step decoder is working ingredient | C1,C3,C4 |
| 90 | Word diversity and composition | Is §88's k=2 failure data? | mixed / seed-variable | k2-400: 0.113/0.976/0.167; k2-100 0.548 | outcome bimodal by seed; budget issue (see §91) | C1 |
| 91 | Make composition reliable | Longer train, WD, curriculum? | "Prediction R met" | LONG, WD-SP, CURR all 5/5 (BASE 0/5) | underfit; 16k steps fixes | C1,C2 |
| 92 | §86 at 3x budget | Does latent hand-off work with budget? | gate passes; R1-R3 met | cum. 1.000 to 2,048; ablation <= 0.03; 0.39x FLOPs | latents carry the state; k=1 certified (0/14,400 errors) | C1,C2,C3,C6 |
| 93 | Sparse labels (chunk-end only) | Can LM learn from thinned labels? | P-S4, P-S8 not met | SPARSE 0/3 (one 0.9725) | thinning labels inside LM fails | C4 |
| 94 | "Where" readout reversal | Budget vs inverse target? | W1, W2 met | STD-DEC-3x 1.000 x5 both | failure was budget lottery | C1 |
| 95 | Foothold dose-response | How many per-op footholds? | "No dose is sufficient" | f up to 0.5: 0/9; slot 1 only | footholds don't teach hand-off | C4 |
| 96 | Amortised self-labels | Free-table teacher gives §92 labels? | P1, P2 met | mixP 0.05 certified 4/5; agreement 1.000 over 12.8M | C4 for pipeline (p<=0.05, oracle M); LM hosts only | C4 |

## What worked
- Pointer view of toy latents; causal inference-time block of late retrieval (§2, §12): 17/23 retrieval seeds rescued, 7-hop 24 -> 41 -> 46/59 with gain boost.
- Re-anchored / restart episodes from a written node (§26): 55/59 7-hop (31/34 failures); the central C2 mechanism, later reused for GPT-2 state tracking (§71, §92).
- Label-free distillation of restarts into one loop (§32 Pilot L, 19/20), second deeper rung (§37, §38 M-2 with agreement filter 12/20 at d13-18).
- Anytime (minimum-progress) supervision: 8/10 vs 3/9 (§17), modest and weak across architectures (§40, PROTOCOL9).
- Chunk-reset attention (§29): highest from-scratch toy d8 (0.604).
- Written-state chunked loop at 16x training length, certified (§71, §66 on state tracking); latent hand-off restart at 3x budget (§92, §92-1: 1.000 to 2,048, 0.29-0.39x FLOPs; k=1 certified).
- Standardised/centred readout over frozen GPT-2 (§72b, §73): 10/10 stable S3 latent; multiplicative cell 10/10 vs additive 0/10 (§70).
- Short labels: ~3-10% short batches or a 25% warm start give certified free tables (§85, §87, §96); then self-labels reproduce teacher labels exactly (§96).
- Loop-aligned per-iteration targets with enough steps/diversity (§88, §91 LONG/WD-SP/CURR 5/5); side step decoder in fade (§89, §94 5/5).
- Edit of the written checkpoint is causal (§45 IIA 0.84/1.00); silent pointer edits steer answers (§34).

## What didn't (dead ends)
- Markov-only architecture (§6/§15), slack dose-response K (§11), retrieval-share marker (§21), one-step transfer rule (§14): failed out of sample.
- Inference-time repair for public CODI/SIM-CoT (§13); fine-tuning repair of brakes (§27, 1/18); amplitude renorm (§30); jitter alone (§23, 1/8); set memory A6 never opened (§24); removing step info (§39, T12 2/8).
- Sets in one episode (§44, §45), beams (§48), learned halt from reward (§50, §51), self-gated restarts (§83 overconfidence).
- Anytime/random-tight on GPT-2 ProsQA (§42): below Coconut; models stop using latents.
- F1 on GPT-2 from final answers (§54b, §61: 1/15 luck); F4 translator (§63); F6 age transplant (§58); boundary memory (§60 identity collapse); F2 precision (§53); depth-structured noise (§62).
- A5 never learned (§64b); RL for discrete latents (§69); additive cells (§70); fine-tuned/pretrained cell without standardisation (§67b, §72b); paraphrase transfer (§74, §79); scaling std readout (§75); no-short-label free tables (§76 ge3, §82); held-out transition rows (§76B); word-generalisation with plain text (§80).
- Thinning labels in the LM (§93 chunk-end, §95 footholds up to f=0.5 0/9); latent hand-off at 8k steps (§86, solved by budget in §92).
- GSM8K length generalisation (§78): written-state no better than text CoT; bottleneck is per-step computation, not halting (§84).

## Recurring traps / methodology lessons
- Timestamp/order of registration estimated, not read from clock (Timestamp correction); marker claim "frozen before analysis" was inaccurate (§14).
- Out-of-sample checks flipped in-sample results (§14, §21); small-n pilots (n=1-4) cannot carry verdicts (§19 PB4, §61).
- Competent-run gating: guard-exhausted runs count as not competent (§15 rule); learnability changes (J 1/8, T12 2/8) can mask effects.
- Same recipe can succeed or flood by numerics draw (§61 vs §61b 1/15).
- Design errors voided results: lr 1e-2 from one-hot toy (§72); labels creating identity map (§60); parity-flip curriculum (§74); transcription error "33 of 34" -> 31 (§26).
- Cue/shortcut checks: ProsQA proximity 0.91 (§57); no-latent control equals latent arm (§66) -> scaffold, not latents; bypass gate (note 64-2).
- Report "uninformative" when the learner did not train (§63, §72).
- Max-combined answers accumulate false positives (§48, §50); backward episodes over-mark (§53).
- Budget lotteries: §94 "where" failure was budget; §86 latent hand-off was budget (§92).

## Open threads worth revisiting
- F1 (annotated-input discrete state), F3 self-synthesis of deeper data, F5 verify-then-extend (§52) mostly untried; F7 path doubling.
- Trained backward direction for F2 if sets precise (§53); A6 set with non-zero-gate init (§24).
- Function-vector / mid-layer readouts for paraphrase agreement ("Lead 4 reopened slightly", §79-6).
- Wide-readout §77 never registered (§76 notes).
- Segment-isolating attention / segmented Coconut with entity checkpoints for GPT-2 (§47 addendum); CLUTRR / StepGame domain.
- GSM8K step 2 (§68 paused): train only WS against matched public CoT/Coconut once a smaller-task SOTA is beaten.
- C4 for latent restarts at scale: p <= 0.05 short-label teacher plus final-answer self-training (§96); C7 readout of fade/restart latents.
