# North star — what "solved latent reasoning" means for this project

Set by the user on 2026-09-29: *"this entire project is based on trying to solve these problems"*. The goal is a model
that meets all seven conditions below at once, in a pretrained language model, reliably across seeds. This page defines
the conditions, gives the current status (one line per result, details in the cited THEORY § / PROTOCOL), and names the
next steps. Every test is still pre-registered in THEORY / PROTOCOL files before its data exist.
Condensed by Audit on 2026-10-03 (history before the S5 line kept as one-line summaries; full text in git history).

## The seven conditions

**Research priority (user, 2026-10-04): C1 > C6 > C4 = C2 > C3 > C5 > C7.** Order new work and GPU time by this; C5/C7 are
reported where cheap but not designed for.

| # | Condition | Operational test (what would count as meeting it) |
|---|---|---|
| C1 | **The thoughts do the work** | Removing or blocking the latent steps drops accuracy to chance on problems that exceed one forward pass (addendum G/J style: K′ = 0 and K′ = 1 far below K), and step-specific blocking shows serial, hop-by-hop use |
| C2 | **More thinking helps, beyond training** | Accuracy rises with latent steps past the trained budget (extrapolation at K′ > K), in every competent seed, not a lucky subset (the P7 / seed-lottery problem) |
| C3 | **Competitive with text chain-of-thought** | At matched compute, equal or better accuracy than written CoT on tasks that need many steps (maths, planning, long graph search) |
| C4 | **Trainable without step labels** | Reaches C1–C3 from final-answer supervision or RL alone, starting from a pretrained LM; no ground-truth intermediate states |
| C5 | **Decides how long to think, and when to write** | Learned halting and learned externalisation: spends more steps on harder inputs; emits a discrete checkpoint when it helps |
| C6 | **Stable** | No drift or collapse over long latent runs; low seed-to-seed variance in the outcome |
| C7 | **Faithfully readable in human language** | A translation of each thought (or checkpoint) into language that passes two causal tests: **round trip** (decode → text → re-encode → substitute; behaviour unchanged) and **edit** (change the text, re-encode, substitute; behaviour changes as the edit predicts). Decodable is not enough: probes can read what the model does not causally use |

**Audit rules for a C7 claim** (LIT_NORTHSTAR, 0734c84 / 49ff87e): (a) behavioural substitution, not reconstruction
quality; (b) edits scored by interchange-intervention accuracy (DAS 2303.02536); (c) an input-only control, since
verbalizers can be right from the input alone (2509.13316); (d) only restarts that drop text or carry non-text state are
informative. Nearest precedent for "answer from its own text only": Markovian Transformers 2404.18988.

## Field and benchmark notes

- No published work certifies latent reasoning or passes a behavioural round-trip/edit test; few run removal tests
  (LIT_SEMI_DISCRETE a1cde61). Ouro 2510.25741 claims C3/C4/C5 but degrades past its trained depth; the closest
  pretrained C2/C3 precedent is a looped Qwen-0.5B retrofit (2608.11233, ~1.5× depth, step-supervised first).
- Closest priors to our design: 2510.14095 (supervised latent recurrence with discrete re-embedding, from scratch) and
  Thinking States 2602.08332 (text thoughts re-embedded as state); full map in LIT_S92_NOVELTY eaea620.
- **ProsQA shortcut** (§57): a proximity heuristic with no reasoning scores 0.91, so ProsQA claims need the
  shortcut-free set; there, published Coconut falls 0.974 → 0.738 (0.732 with zero thoughts).
- Bench bars on our items: ICoT-SI released checkpoints 0.998 (4×4) and 0.996 (5×5; paper table 0.95).

## Where the project stands (2026-10-03)

**Headline.** On S5 state tracking a pretrained GPT-2 carries the state in latents, exact far past training length and
certified, cheaper than writing the state (C2 + C3 + C6, scaffolded); serial causal use of the latents (C1) is shown in only 2/5
§92 models and not in §97, whose latents are bypassed (§104). It works
without hand-built state (§97), and its step labels can come from a table trained on final answers plus ≤ 5 % short problems, but those shorts cover
≈ 99.9 % of the S5 transition table (§96), so this is not meaningful C4. Not yet shown on natural tasks (maths, planning); C5 untouched. All lines below are Audit-reproduced.

| # | Status | Key results (newest first) | Where |
|---|---|---|---|
| C1 | **Not met (causal test)** | **§104 donor patching (the operational serial test):** §92 SERIAL in 2/5 models (slot-only donor use ≥ 0.97 in s1, s4; joint slot + op patch in 3/5); §97 GEN-4 0/3 and HYB-4 0/3: natural donor latents are ignored (donor ≤ 0.27) and the decode is the recipient's own recomputed state (recipient-consistent up to 1.000), so earlier zero-ablation C1 evidence was off-distribution. Masked-route designs are the C1 path: **§105 (S5, masked fork)** keeps exactness (1.000 at 512 ×3, ≥ 0.998 at 2,048) and closes the bypass (mean-removal 0.007–0.009), but donor HANDOFF is 0/3: slots 2–3 carry the state serially in 2/3 seeds (donor ≥ 0.998), while slot 1 is recomputed from the visible chunk start (donor 0.13–0.87); §108's Markov segments remove that route (Audit-reproduced); **§108 (S5, Markov segments: state reaches each slot only through the previous latent), scored: s1 is the first model with serial, unqualified C1 (serial use under a forced bottleneck: enforced by Markov segments, not emergent; S5 steps are lookups)** — 1.000 at 512, 0.996 at 2,048, donor-consistent 1.000 at every slot, recipient ≤ 0.011, zero-removal 0.004. ACC not met (MK-4 2/5: s1, s5 0.994; MK-8 0/5, best 0.982); SERIAL holds in all 5 seeds that learned (donor ≥ 0.995, recipient ≤ 0.011); failures are the no-read basin (MK-4 s2, s4; MK-8 s2, s4, s5) or precision (MK-4 s3 0.59, MK-8 s1 0.80) (Audit-reproduced). **§111: an auxiliary reader fixes the basin, 3/3 serial and exact so far (C6 row)**; §100 MASK2 4×4 removal → 0.0 (scoring pending). Earlier: §97 (no hand-built state): zero-removal ≤ 0.007; qualified: mean-removal leaves 0.23–0.33 (GEN-4) and 0.15–1.000 (HYB-4; s5 fully correct), so the text fork bypasses the latents under mean ablation and C1 rests on zero-ablation only (§97 C6 scoring, note 97-3); §105 (masked fork) and §104 (donor patching) test this. §92: zero-removal ≤ 0.003, mean ≤ 0.03. §94: the user's fade recipe at 3× budget answers "all" and "where" 1.000 in 5/5, removal at the no-CoT floor. Earlier: toy model met; GPT-2 Coconut-style latents on ProsQA were a one-pass conduit; published GSM8K latents (SIM-CoT 0.448 vs 0.051 with 0 thoughts; CODI 0.425 vs 0.221) carry some work | THEORY §§89, 92, 94, 97; GSM8K_REPRO |
| C2 | **Met as length generalisation on S5** | §92/§97: trained at 128, exact at 512 and 2,048; k = 1 certified (all 14,400 (state, op) inputs) so exact at every length. Restarts from a compact state with the history cut are what extends range (§64–§66, §71). Not on natural language: GSM8K 1–2 → 5+ steps fails for written state and text CoT alike (≈ 0.02; §78, §84 oracle 0.030). Fade recipe alone does not extend (N 9–16 ≈ 0.2). Bounded-carry multiplication (§106 pilot, masked fork): b = 4 accuracy 0.004–0.006, gate failed, so the latent arms stop; per-step decode collapses after the first digit (Audit-reproduced). At k = 2 the written bounded-carry chain with a history cut (WC-S-k2) is 1.000 at b = 4/8/16/32 from training at b ≤ 5 (3/3), while the masked latent version stays at ≤ 0.007: C2 on multiplication belongs to the written format plus cut, not the latents (§106-k2, Audit-reproduced) | §§64–66, 71, 78, 84, 92, 97 |
| C3 | **Met on S5 (scaffolded); open on maths** | §92: equal exactness to the written state at 0.29–0.45× its FLOPs; §97 0.56× as metered (0.61× restricted-vocab). §89: fade beats Coconut stages by +0.40 at matched compute. **§100 (k = 2 latent restarts, 4×4 multiplication): 1.000 / 1.000 / 0.979, matching ICoT-SI's 0.998 on the same items, at 0.89× text CoT's FLOPs (ICoT-SI, which writes no chain, is cheaper still); not C1 (mean-removal 0.13–0.63)**; at 5×5 it fails (GEN2 0.09–0.10, MASK2 ≤ 0.06 vs ICoT-SI 0.996) (Audit-reproduced). §98 at k = 4 fails all rules: GEN-M 0.16 (4×4) / 0.005 (5×5), MASK 0.03 / 0.0, Coconut ≈ 0 (partial panel), text CoT 1.0 / 1.0, no-CoT 0.003 / 0.0 (Audit-reproduced). §99 (k = 4, 4×4: 8 speed-up levers incl. K-CURR, standardised slots, text-first, Markov segments, §102's state path): none reaches threshold, all 27 runs censored at 24k, best test 0.42 (STD s3) vs CTRL 0.14–0.18, removal ≤ 0.012 (Audit-reproduced). ProsQA-NA: latent arms at chance, text CoT 0.77 | §§71, 89, 92, 97, 98; PROTOCOL8 |
| C4 | **Not met (strict); qualified results only** | **Qualifier (note 110b-1):** in S5 an identity-start length-2 problem's answer a·b is the transition (a, b) → a·b, so the ≤ 5 % "short problems" label Cayley transitions from every state (measured: §96 teacher data 99.9 %, §103 trained R0 71 %, §110b smoke 79 % of the 14,400 (state, op) pairs; theory/results/c4_coverage.json); C4 claims below are qualified by that coverage, and strict C4 (lengths ≥ 3 only) remains unmet. §96: an S5 table trained by exact marginal likelihood on final answers (≤ 5 % short problems, oracle state count) certifies 4/5 and reproduces §92's per-op labels exactly (12.8M positions), so §92 follows; the LM itself acquires nothing. Closest in-LM attempt: filtered self-training by length from final answers only plus ≤ 5 % short (Cayley-label) problems carries S5 to 0.96 / 0.96 at 512 (0.90 / 0.84 at 2,048) in 2/3 seeds, 0 in the third; pseudo-label precision ~0.98 caps it, and the unmasked fork bypasses (mean-removal 0.95 in s2), so SELF (≥ 0.99) is not met and there is no C1 claim (§103, Audit-reproduced). **§110 (scored, 5 seeds): C4 (≤ 5 % identity-start shorts, 71.5 % coverage) + C1 (donor ≥ 0.96 at every slot, recipient ≤ 0.012, mean-removal 0.011) + C2 (0.994 at 2,048) jointly in 1/5 seeds (s3; OOM-resumed at 4,960, state-complete, note 110-4; its k = 1 restart is exact on all 14,400 transitions, including the 4,105 no short labelled, learned from long final answers given a partial foothold, not held-out-pair generalisation, note 110-5); not reliable: SELF2 2/5 (s3 0.999, s5 0.9915), C1 1/5, C6 at 2,048 1/5, median 0.9505; s1 slot-1 bypass, s5 partial bypass (recipient 0.43 at slot 1), s2/s4 never acquired (gate ≈ 0); STUDENT arm triggered (Audit-reproduced).** **§112 interim (foothold dose): with shorts covering only 10 % of transitions, s21 reaches 0.996 at 512 / 0.99 at 2,048 and its k = 1 restart is exact on all 12,960 never-labelled transitions (learned from long final answers); latents are necessary (mean-removal 0.025) but single-slot donor fails (0.12–0.34), the state is carried jointly with op-token K/V (joint patch 1.0 / 0.82 / 1.0 at n = 500, note 112-3; §110 s3 is slot-carried, s1/s5 spread over the prefix (joint at chance): the masked fork lands on three different carriers, hence Markov segments + single-slot SERIAL for the integrated design); FOOTHOLD needs ≥ 2/5 per dose (Audit-reproduced).** With random textual starts for length ≥ 3 (§110b) the same training stays at chance (loss ≈ ln 120 through 13.2k, kept share ≈ 0; stopped after a clear null, note 110b-2), so §103's near-miss leaned on identity-start structure; self-training is not a route to strict C4 here (Audit-reproduced). In-LM routes fail: RL (§69), annealing (§87), chunk-end-only labels (§93, 0/6), footholds up to f = 0.5 (§95, 0/9). Short problems work as a warm start only (§87: ¼ then removed, 5/5); the table's failure without them is optimisation, not information (§85) | §§69, 85, 87, 93, 95, 96 |
| C5 | **Not met** | **§115 partial (note 115-1): a label-free round-trip trigger (snap only when the latent path and the text path disagree at the next slot) catches 100 % of first errors with 0 false snaps and keeps 1.0 at 512 / 2,048 at 99.9 snaps vs 127 (§108 MK-4 s1, TRIGGER met) and 19.7 vs 127 on READ s6 (descriptive); it computes the text path at every hand-off, so it saves checkpoints, not compute; with no snaps at all the chain fails (0 at 128; the free-running latent's cosine to the snapped one falls to ~0.01 by step 64) (Audit-reproduced).** Confidence-triggered restarts fail out of range (§83: misses 32 % of first errors; adaptive 0.00 vs fixed k = 4 1.00). Restart points are hand-set. Earlier: learned halts stop at a constant ~2 segments (§50); untrained models ignore written checkpoints | §§50, 83 |
| C6 | **Met on S5** | §92: k = 4 exact in 5/5, k = 1 certified 3/3; §97 5/5 for both GEN-4 and HYB-4 (≥ 0.99 at 512, ≥ 0.998 at 2,048). §91: loop composition 15/15 once trained 4× longer; the earlier "seed lotteries" in §90 / §89 were budget-limited, but budget is not the general fix: §111's 2× budget arm is 0/5 on the Markov hand-off, where an auxiliary reader is 5/5 (no-read basin, not budget). **Not yet for the Markov hand-off (§108, the C1 design): 1/4 seeds succeed (s1); s2 and s4 never read the latent (loss on the chance-for-latent-slots curve), s3 learned late (0.59 at 512); §111 scored (note 111-4): an auxiliary reader trained with the writer from k = 1 (MK-4-READ) meets REL — 5/5 at ≥ 0.99 at 512 (0.9995 / 0.9945 / 1 / 1 / 1) with single-slot SERIAL in all 5 (donor ≥ 0.9995, recipient ≤ 0.0105), 4/5 at ≥ 0.99 at 2,048 (s7 0.966); 2× budget 0/5, DAgger 1/5 (s10 1.0; s7 0.88): §111 complete, only READ meets REL (a21c09a2); the no-read basin is fixed by the reader, not by budget (Audit-reproduced)** (Audit) | §§91, 92, 97, 108 |
| C7 | **Partial** | §97: checkpoints are text by construction (validity ≥ 0.99999), silent slots decode to the correct text 1.0, checkpoint-edit IIA 1.000 (round-trip test not yet scored). §94 probes: state linearly available on every latent. Earlier: toy pointer editable (IIA 0.36); the F4 translator did not train (uninformative) | §§63, 94, 97 |

**Other verified lessons** (one line each):
- Per-step targets are load-bearing: every success had them; snapping prevents drift (2609.02049 explains collapse).
- GPT-2 anisotropy blocks latent acquisition until features are standardised (§72b/§73: 0/5 → 10/10).
- In-pass composition of seen steps into unseen sequences needs per-iteration targets, word diversity and budget (§88, §90, §91).
- Pretraining helps operation discrimination, not paraphrase transfer (§79).
- Closed lines: GSM8K length split at GPT-2 small (§78/§84); fade line (closed positively, §89/§94; its side decoder lives on in §97); sparse-label thinning (§93/§95).

## Current design and open questions

**Design (realised in §92/§97):** latent slots, one per step, trained only through a side decoder that predicts the
step's text (never fed back); every k steps the decoded text is re-embedded as a checkpoint and the history is cut.
Open:
1. **C3 on real tasks:** §98 (multiplication), then shortcut-free ProsQA and a held-out task.
2. **C1 without qualification:** a side decoder that cannot see the op tokens (§98 MASK arm).
3. **C4 in the LM:** final-answer-only self-training by length (proposed, awaiting the user).
4. **C5:** a learned or validity-triggered restart (not confidence, §83).
5. **Scale:** a 0.5–1.5B model, then GSM8K, only once 1–3 work (user approval before large spend).

## Roles

Theory designs and registers; Experiments runs and scores runs it owns; Literature checks priors before each
registration; Audit reproduces every scored verdict and keeps this page current (memory note "latent-sessions").
