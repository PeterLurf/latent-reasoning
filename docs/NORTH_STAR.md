# North star — what "solved latent reasoning" means for this project

Set by the user on 2026-09-29: *"this entire project is based on trying to solve these problems"*. The goal is a model
that satisfies all seven conditions below at once, in a pretrained language model, reliably across training seeds.
This document defines each condition operationally, records where the project stands, and names the next steps. It
states goals, not predictions: every test that bears on a condition is still pre-registered in the usual PROTOCOL /
THEORY files before its data exist.

## The seven conditions

| # | Condition | Operational test (what would count as meeting it) |
|---|---|---|
| C1 | **The thoughts do the work** | Removing or blocking the latent steps drops accuracy to chance on problems that exceed one forward pass (addendum G/J style: K′ = 0 and K′ = 1 far below K), and step-specific blocking shows serial, hop-by-hop use |
| C2 | **More thinking helps, beyond training** | Accuracy rises with latent steps past the trained budget (extrapolation at K′ > K), in every competent seed, not a lucky subset (the P7 / seed-lottery problem) |
| C3 | **Competitive with text chain-of-thought** | At matched compute, equal or better accuracy than written CoT on tasks that need many steps (maths, planning, long graph search) |
| C4 | **Trainable without step labels** | Reaches C1–C3 from final-answer supervision or RL alone, starting from a pretrained LM; no ground-truth intermediate states |
| C5 | **Decides how long to think, and when to write** | Learned halting and learned externalisation: spends more steps on harder inputs; emits a discrete checkpoint when it helps |
| C6 | **Stable** | No drift or collapse over long latent runs; low seed-to-seed variance in the outcome |
| C7 | **Faithfully readable in human language** | A translation of each thought (or each checkpoint) into language that passes two causal tests: **round trip** — decode thought → text, re-encode text → thought, substitute, and behaviour is unchanged; **edit** — change the text (e.g. "node 37 is unreachable"), re-encode, substitute, and behaviour changes exactly as the edit predicts. Readable is not enough: Part I showed probes can decode what they cannot causally use |

**Audit requirements for any C7 claim** (from the Literature survey, `docs/LIT_NORTHSTAR_2026-09-29.md`, 0734c84):
(a) **behavioural substitution**: the re-encoded thought replaces the original and the model's answers are compared;
reconstruction quality (fraction of variance explained) alone does not count. (b) **Edit test scored by
interchange-intervention accuracy** (DAS, 2303.02536), plus the share of answers that change, at the level of the
claim (e.g. which node ids are in the frontier). (c) **Input-only control**: the same decoder applied to the input
without the thought must not match its performance, since verbalizers can be right from the input alone (2509.13316).
(d) **Restart / hidden-channel tests** (LIT_NORTHSTAR §5, 49ff87e): only restarts that drop text or carry non-text state
are informative, since restarting a standard transformer from its full text is identical by construction. Report
per-instance answer agreement and KL, not just aggregate accuracy (Delethink and InftyThink report only aggregates).
The closest trained precedent for "answer from its own text only" is Markovian Transformers (2404.18988); hold any
"first" claim to it.

## Where the field stands (Literature survey, 2026-09-29)

- No published work passes a behavioural round-trip and predicted-edit test on latent reasoning states. The closest
  activation-to-text decoder (NLA, 2026) reports reconstruction and steering, not behavioural substitution.
- Ouro (2510.25741, 1.4–2.6B) claims C3/C4/C5 and a correlational form of C7, but degrades beyond its trained depth
  (C2 not met). Huginn, LOTUS, SIM-CoT and TRM each cover two or three conditions.
- The pretrained-LM precedent for C2/C3 is a pre-registered looped retrofit of Qwen-0.5B (2608.11233): ~1.44–1.5× depth extrapolation, beating a scratchpad past its horizon, but with intermediate-state supervision before outcome-only annealing and no matched-compute claim, so it does not show C3 + C4 jointly. Any PROTOCOL8/8b claim is held to it (LIT_TASKS_ARCH_2026-09-30).
- Nobody claims C2 in every seed, or a causal C7. No published design combines a latent loop, a set-valued readable
  checkpoint and a learned restart. Any "first" claim here should be scoped to exactly that.

**ProsQA caveat (2026-09-30, THEORY §57, Audit-confirmed):** a person-agnostic proximity heuristic, with no reasoning from the root, scores 0.91 on prosqa_test and 0.90 on our deep p8 sets, because the generator hangs the negative option off another person. Any ProsQA claim needs a shortcut-neutral control (tie subset or other-person insertion). Not documented in prior work (LIT_PROSQA_SHORTCUT_2026-09-30), so it also bears on published ProsQA results. On the shortcut-free test (bench, Audit-recomputed) our reproduction of published Coconut falls from 0.974 to 0.738, and 0.732 with zero thoughts, so its thoughts add < 1 point; replications reach 0.64–0.73.

## Where the project stands (toy 2-layer model unless marked)

| # | Status | Evidence |
|---|---|---|
| C1 | Toy: met (delayed format; blocking the frontier cuts deep nodes). GPT-2 on ProsQA: **not met** — one-pass conduit (G, J), no training-depth wall (Q) | PROTOCOL3–5; PROTOCOL6 D/G/J/N/P/Q; PROTOCOL7 T1b/T2b |
| C2 | Toy: understood, not solved. Anytime supervision gains ~1 hop (Pilot D); across architectures it does not extend reach, and its learnability gain holds only in the wider model (PROTOCOL9, 64 pairs); the fall-back is keyed to an iteration count (P10e). Restart-based fixes work: re-anchoring 31/34 (§26), and distilling it into one loop fixes 19/20 (Pilot L); a second label-free rung reaches 13–18 hops in 12/20 without eroding the first when self-labels are agreement-filtered (Pilot M, M-2). Chaining restarts without extra training: each segment covers exactly 6 hops (§46), six single-anchor segments reach d19–24 at 0.65 (descriptive), and multi-anchor beams add nothing (§48). A public boundary-response-table architecture (user proposal) failed Phase A: no seed learns composition to 0.95, and the count returns as fading table confidence (§60b). An inference-time count clamp rescues only 9/34 (§36). Outside graph search (permutation tracking on S3, §64): one loop also fails past its trained length (0.99 at ≤ 8 ops, 0.03 at 13–16), though not by stopping at the 8-op state, and written-state restarts fix it (0.995, 9/9 runs; edit IIA 0.97); A5 did not learn even with curriculum (0/10, §64b). In Coconut GPT-2, untrained restarts from written checkpoints do not help, even with the oracle anchor (PROTOCOL8b pilot): checkpoint use must be trained | THEORY §§17, 26, 32, 35, 36, 38, 46, 48, 64; PROTOCOL9 |
| C3 | GPT-2 on deep ProsQA (PROTOCOL8): best latent arm 0.56 vs text CoT 0.76 at depths 7–10, not competitive; no latent seed competent (0/12), CoT 3/4. On the shortcut-free set (amendment 3) every latent arm is at chance (coco 0.564 → 0.508, at 0.486, rt 0.501), while CoT holds (0.762 → 0.767): the latent arms' above-chance accuracy was the ProsQA cue (Audit-reproduced). GPT-2 permutation tracking (§66): chunking with a written state and history cut is perfect at 17–32 ops (1.000, 5/5 seeds) where text CoT trained on ≤ 8 falls to chance, at fewer FLOPs, but the same chunks without latent thoughts do exactly as well, so the gain is the scaffold, not latent computation (C1 not shown) | PROTOCOL8 |
| C4 | Toy base models use a lookup scaffold. Label-free fine-tuning (distilling re-anchored inference) fixes 19/20 failures, but on deep graphs from the evaluation generator (Pilot L). Random tight budgets with final-answer labels only match anytime in-range, but only 4/8 runs learn (Pilot K, suggestive). GPT-2: a per-entity discrete label channel trained on final answers only (shallow-question curriculum, disjoint graphs) solves depths 1–14 perfectly in 1/3 seeds; the other 2 learn only depths 1–2 (§61), so it is learnable but seed-unreliable. Distillation needs near-clean self-labels: any label noise (uniform or depth-structured) leaves students age-causal and, if structured, at chance beyond 12 hops (§62). | THEORY §§31, 32, 61, 62 |
| C5 | Not tested; re-anchoring (§26) is a hand-built version of "write a checkpoint every K steps". Untrained models ignore written checkpoints (GPT-2 pilot, toy set checkpoints §44), so checkpoint use must be trained first; in GPT-2, checkpoint fine-tuning reduces the harm of written checkpoints but correct ones still do not help (oracle gain −0.04, PROTOCOL8b step 1). Chaining segments with self-written checkpoints gives no gain at d11–14 (PROTOCOL10). Toy: a label-free learned write head matches the attention anchor (0.85) and a fixed closure-halt rule stops later on longer chains (3/4/5 segments at 12/18/24 hops, 8/10 models; §49); a halt learned from final-answer reward beats fixed rules on reward but stops at a constant ~2 segments, not tracking depth (§50) | THEORY §§26, 44, 49, 50; PROTOCOL8B_PILOT |
| C6 | Partly characterised: outcome set at full-depth entry (E1), fading per hop (Mechanism 2); not solved | PROTOCOL6 A/E, THEORY §§4, 14 |
| C7 | Toy: the silent pointer is causally editable at node-id level: answers follow edits (IIA 0.36). Round trip 0.947, just under 0.95. The vocabulary is node ids, not language, and the encoder is fitted on the model's own thoughts. Naive probe directions were inert (Part I). Training makes the model use a written checkpoint channel (edit IIA 0.84 set / 1.00 single, §45); that channel is forced by the history cut, so it is not C7 for silent steps (0.36, §34). A trained behavioural translator (thought ↔ node set) fails all five F4 tests: round trip 0.916 barely beats input-only 0.864, and edit IIA is 0.01; the hard straight-through translator itself did not train (flat losses, decoded sets ~28 nodes at precision 0.13), so this is uninformative about readability, not a negative (§63). | THEORY §§34, 45, 63; Part I |

## The candidate design this points to

**Checkpointed latent reasoning**: think silently for up to K steps, then write a discrete, human-readable checkpoint
(the current frontier: which nodes are reached, at what depth), and restart the latent loop from it. Re-anchoring
suggests this can serve C2 (no step count outside the trained range), C6 (amplitude restored at each checkpoint) and
C7 (checkpoints are readable by construction), and a learned checkpoint policy is C5. Open questions:
1. Are the silent stretches between checkpoints themselves faithful (C7 for latent steps, not only checkpoints)?
2. Can the checkpoint carry a *set* (the frontier holds several branches; the model's own anchor is right in ~1/W of
   graphs) without losing the parallelism that makes latent reasoning worth having?
3. Can it be learned end-to-end without step labels (C4), e.g. by distillation (Pilot L) or RL on final answers?
4. Does it transfer to a pretrained LM on problems deeper than one pass (PROTOCOL8's format)?

## Roles (see the memory note "latent-sessions")

- **Theory** — design the checkpointed architecture and the C7 round-trip / edit tests on the toy model; keep the
  mechanism account current against C2/C6.
- **Experiments** — confirmatory tests at scale; after PROTOCOL8, carry the checkpoint design and the C7 tests to a
  pretrained model.
- **Literature** — prior work on faithful chain-of-thought and CoT monitorability, latent-state decoding and
  translation (e.g. logit/tuned lens, activation-to-text decoders), adaptive computation and learned halting, and
  hybrid latent/discrete reasoning.
- **Audit** — hold every claim of progress on a condition to its operational test above; keep this status table
  current as results are scored.
