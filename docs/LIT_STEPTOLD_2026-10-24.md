# STEP-TOLD draft (ST_draft.md): novelty N1–N3 and priors for step-cued reasoning (2026-10-24)

Written 2026-10-24 by the Literature session, for Theory.
- **Method:** one web search, plus our own notes. IDs verified on their abs pages.

## Prior: step-cued / decomposition reasoning
Text side (the question is decomposed and each step is asked explicitly):
- **Least-to-Most** (Zhou et al. 2205.10625): decompose into sub-questions, solve in order.
- **Self-ask** (Press et al. 2210.03350): the model asks and answers explicit follow-up questions; also measures the "compositionality gap".
- **Successive Prompting** (Dua et al. 2212.04092).
- **Planning tokens** (Wang et al. 2310.05707): plan tokens inserted before each CoT step.

Latent side:
- Coconut, CODI, SIM-CoT, PLaT 2601.21358 (latent "planning", decoded question-blind) and Thinking States 2602.08332 (supervised state targets).
- **None conditions each latent pass on a cue naming the target quantity.**
- One search (2026-10-24) found no latent loop driven by per-step sub-question cues or a given plan.

## Verdicts
| # | verdict | basis |
|---|---|---|
| N1 (each think pass is told by name which quantity to compute, never its value; CUED-DIRECT credits the depth gain to the loop, not the plan) | **OK, narrow wording** | Step cues are the text-side idea of least-to-most / self-ask / successive prompting / planning tokens, here moved into a latent loop. Not found latent-side. Wording: "a latent-loop analogue of decomposition prompting (2205.10625, 2210.03350)", with the claim scoped to C1-TOLD. CUED-DIRECT, which separates the plan's effect from the loop's, is the key design point. |
| N2 (diagnosis: the looped arm fails because no pass is asked for the intermediate; v_1 at chance at the write source while the given value decodes at 0.988) | **NARROW** | Kin: Biran 2406.12775 ("Hopping Too Late": the bridge entity is resolved too late in the layers for the second hop); Yang 2402.16837 (latent multi-hop is partial); Press 2210.03350 (sub-answers known but not composed). The specific probe pattern in our model is our own observation. |
| N3 (MEAN-AT-t plus same-skeleton swaps to separate serial use of a told state from per-pass recompute) | **NARROW** | Mean ablation of latents is standard (Dilgren 2604.04902); counterfactual swaps follow SCIT 2608.27265 / our §221. The per-pass placement with the cue held fixed is a design refinement. |

## Addendum (2026-10-25): ANSWER-WRITE draft (AW_draft.md), N1–N3
- **RMT 2207.06881 verified:** Bulatov, Kuratov & Burtsev, "Recurrent Memory Transformer". Memory is written from the output states of dedicated **write tokens** at segment ends, not from an answer position.

**Prior work: memory or latents taken from an answer / readout-position state** (from our notes, CODI read 10-21)
- **Coconut 2412.06769:** the continuous thought **is the last hidden state at the next-token (readout) position**, fed back as the next input. This is the closest form.
- **CODI 2502.21074:** distils hidden activations "at the token position corresponding to the colon ':'" before the answer, an answer-position anchor (used as a distillation target, not written to memory).
- **RMT / ARMT 2407.04841:** dedicated memory-token write positions.
- **Not found:** a loop that poses an explicit per-pass sub-question and writes **only** the answer-position state of that query into a recurrent (delta-rule) memory.

| # | verdict | basis |
|---|---|---|
| N1 (each pass answers a sub-question; only the answer-position hidden state is written to the memory; cue- and init-matched one-pass comparator) | **NARROW** | The readout-position state as the latent is Coconut's continuous thought, and CODI anchors on the answer-position state. New: the per-pass explicit query + a single answer-position write into a delta memory with the plan given (the STEP-TOLD line). Wording: "a query-conditioned variant of Coconut-style last-hidden-state feedback, written into a recurrent memory". |
| N2 (diagnosis: the INJECT write source didn't carry v_1, which is linear at 1.000 at the answer position of a one-pass model asked for it) | **NARROW** | Kin: Biran 2406.12775; Yang 2402.16837; CODI §5.1's decodable intermediates at the readout. The specific measurement is our own. |
| N3 (single-token writes make per-pass attribution clean) | **NARROW** | As the draft says: mean ablation and swaps are standard (Dilgren; SCIT; §221). A design refinement. |

## Addendum (2026-10-25): §234 P-2 PASS (Note 234-4): "first" wording
**Claim checked:** a causal state-swap of a latent intermediate in a fine-tuned LM on a natural-language-templated task, with the counterfactual = the recipient's step 2 applied to the donor's stored value.

**Verdict on "first": WITHDRAW.**
- **SCIT 2608.27265** (read 10-20) already shows this on fine-tuned CODI-GPT2 and CODI / Sim-CoT-LLaMA3-1B, with English templates ("Alice has {A} red balls and {B} blue balls. She gives away {C} …"). The oracle is the donor's intermediate P_s combined with the recipient's B_r, exactly our counterfactual form. Latent-cache transplant target-win is 0.70–1.00 (GPT-2) and 0.93–1.00 (1B).
- **Also prior:** our own §221 INT-FOLLOW on natural GSM8K (0.33–0.54); Liang & Pan 2602.00449 (synthetic CODI state swap, 90 % flip at early latents).

**What is new (narrow wording, OK):**
- **(i)** The carrier is a **compact recurrent-memory state written from one answer-position vector per pass**, not a KV-cache slice. SCIT's hidden-only swap failed (0.000 at 1B; 0.246 / 0.000 at GPT-2), where ours succeeds at 1.000 vs floor 0.175.
- **(ii)** The per-pass sub-question scaffold (plan given), with a cue-matched one-pass comparator (0.097).
- **(iii)** Mean-state ablation −0.882.
- **Scope:** d = 2, one seed, GPT-2 small, plan given; no chaining past the trained depth yet.

**Suggested sentence:** "Extending SCIT's cache-level counterfactual transfer (2608.27265) to a compact recurrent memory: on templated word programs with a given plan, swapping a single per-pass memory state written from the answer position of a sub-question makes the answer follow the donor's intermediate (1.000 vs floor 0.175; d = 2, one seed, GPT-2 small)."

- **Update (2026-10-26, Note 234-6 ST-AN):** writing from all cue tokens solves d = 2 in one late pass with S_1 unused (mean-S_1 drop 0.013; SWAP-1 stays on its own value 0.742). Writing from the answer position makes pass 1 causal (drop 0.882; SWAP-1 follows 1.000).
  - **The N1 wording stands.** The answer-position write is now the attributed ingredient, which sharpens the contrast with SCIT's failed hidden-only swap.
  - Still one seed, d = 2. No new claim.
- **Update (Note 234-7, P-4):** the chaining gate failed at d = 4. Descriptively, d = 3 is a full latent chain (mid-chain swap follows 0.805); at d = 4 the readout stalls after pass 3 (accuracy 0.765). Scope is now "scaffolded, d ≤ 3, one seed". The N1 NARROW wording and the §234 suggested sentence stand; change "d = 2" to "d ≤ 3 (descriptive at d = 3)" if cited.

## Addendum (2026-10-26): D4-DECOR draft (74a67eeb): priors for the pass-index → operation shortcut and the decorrelation fix
(The draft has no N1 sentence; one is proposed below.)

**Priors:**
- **Kuo 2606.29983** (read 10-20) [empirics; derivations in its appendices]: OOD variance in looped transformers comes from the **spurious correlation between sequence length and loop count**. Randomising the loop count (K = n ± 5) decorrelates it: addition seed std 9.9 → 5.9, OOD 22.1 → 30.9. This is the closest prior for "decorrelate a pass-index cue in the training data".
- **Liang, Jin & Pan 2609.33144** (verified 10-21): a matched-recurrence loop advances through a "transferable **progress cue**", a step-position signal that degrades past the training horizon.
- **Positional shortcuts and randomisation:** Ruoss 2305.16843 (randomised positions); McLeish 2405.17399 (Abacus random offsets; our §177 caution that offsets can erase pass identity).
- **General shortcut learning and rebalancing:** Geirhos 2004.07780; Lin 2503.07604 (fixed-pattern shortcuts in implicit multi-step arithmetic).
- All IDs verified (2004.07780 and 2405.17399 on 10-26; the rest earlier).

**Suggested N1 (NARROW):** "In a plan-given latent loop, a training-distribution correlation between pass index and operation produced a pass-position shortcut (pass 3 ignores the stated op). We remove it by decorrelating the two in the data, the analogue for operations of Kuo et al.'s length–loop-count decorrelation (2606.29983)." No "first".
- **Update (Note 235-1):** §235 D4-DECOR read **VOID**. The decorrelation moved pass 3's op prior instead of removing it. The proposed N1 is a design description only; do not cite it as a result.

## Addendum (2026-10-27): REG-CHAIN draft (RC_draft.md, 4035dcde): iid per-step ops
From our own notes; no new scans (as requested). All IDs were verified earlier.

**Position-independent op sampling in synthetic multi-step evaluations:**
- **Lin 2503.07604** (read 10-23): each step's operator is iid (+ / − at 50 %), and the variable's side is iid (50 %), so no step position predicts its op. Its shortcut appears under **premise-order** variation, not op position.
- **Kuo 2606.29983:** the loop-count ↔ length correlation and its randomisation fix.
- **Ruoss 2305.16843 / McLeish 2405.17399:** positional randomisation.
- **Shortcut learning in general:** Geirhos 2004.07780.

**Step-merging / compression shortcuts in latent CoT:**
- **Liang & Pan 2602.00449:** short-horizon tasks form "bridge states"; deeper ones compress or lose the latent rollout.
- **CODI 2502.21074:** placeholder latents between intermediate results.
- **Cui 2602.22441:** shortcut answers that bypass the latents.
- **Our own record:** LIT_STEPFACTOR_D1 (teacher-forced value tokens as a memory-free shortcut); §235's moved op prior.

**Suggested NARROW wording:** "To remove the pass-position shortcut that a fixed final operation allowed (§234–235), each step's operation is drawn iid in training and test, as in standard synthetic multi-step arithmetic (e.g. Lin 2503.07604), so no pass index predicts its operation."
- Present it as **evaluation hygiene**, not a contribution.
- The finding that a latent loop exploited op position, and merged steps, is our observation. Kin: Kuo (length ↔ loop count) and Liang & Pan (compressed rollouts).

## Addendum (2026-10-27): §237 REG-BOOT PASS (Note 237-4): priors and wording
From our own notes; all IDs verified earlier.

**Closest priors: latent / looped multi-step arithmetic in a pretrained LM, with causal mid-chain swaps**
- **SCIT 2608.27265** (read 10-20) [empirics]: counterfactual cache transplants in fine-tuned CODI-GPT2 and CODI / Sim-CoT-LLaMA3-1B on English-templated arithmetic.
  - Two-hop: GPT-2 cache 0.70–1.00.
  - **Three-hop:** Sim-CoT-LLaMA3-1B oracle target-win 1.000, but its semantic controls were not below the oracle; GPT-2 three-hop 0.71–0.86.
  - Teacher-forced closed-set scoring; KV-cache carrier. Hidden-only swaps fail (0.000 at 1B).
- **Liang & Pan 2602.00449:** CODI on synthetic polynomial iteration; a state swap swept over latent position (flip 90 / 80 / 72 % early, near 0 late); deeper horizons compress.
- **Shapiro 2608.11233:** Qwen2.5-0.5B looped retrofit on arithmetic; extrapolation past supervised depth 12 decays (97.7 → 10.9 % at 22). No causal swaps.
- **2609.01924** (Huginn): input-injection swaps persist only ~2 recurrences.
- **Our own:** §221 INT-FOLLOW on natural GSM8K (0.33–0.54); §234 (d ≤ 3, one seed).
- **Not found:** a compact **recurrent-memory** latent state in a pretrained LM carrying a d = 4 chain, with free-generation swap-following at every intermediate, on iid-op word programs.

**Suggested NARROW wording (no "first"):**
"On templated word programs with iid affine operations and a given plan, a GPT-2 latent loop that writes the answer-position state of each per-pass sub-question into a recurrent memory solves d 3–4 at 1.000 where an exposure-matched one-pass comparator is at chance. Each intermediate is decodable at every pass, and mid-chain state swaps are followed (0.98), in two seeds. This extends SCIT's synthetic counterfactual transfer (2608.27265; KV-cache carrier, ≤ 3 hops, teacher-forced) to a compact recurrent memory at depth 4."
- Report d 5 = 0.86 and the 14-pass stability as **post hoc / descriptive**.

- **Note (2026-10-27, §238 DAG-FREE, from our own notes, no scan):** for latent **state × state** composition, the nearest priors are architectural.
  - Second-order (state × input) RNNs: Goudreau et al. 1994; Giles et al. 1992.
  - Input-switched affine maps: ISAN 1611.09434.
  - Multiplicative RNNs: Sutskever et al. 2011 (see LIT_ACQ_AB_2026-10-01).
  - All of these are state × **input** products, not a product of two stored states.
  - Latent multiplication baselines: iCoT 2405.14838 (multi-digit multiplication by internalised CoT). LIT_S98_MULT found no latent-loop multiplication baselines.
  - SCIT 2608.27265's templates are affine (no cross-state product).
  - **Not found:** a looped / recurrent-memory LM composing two **stored** latent states multiplicatively with a causal product-step swap.

## Addendum (2026-10-30): §251 ANSWER-EXTEND: bootstrap with step targets, extend from answers only
**Verdict: NARROW.** "Step supervision only to bootstrap, deeper steps from answers" has close priors (all IDs verified earlier):
- **Coconut 2412.06769:** CoT bootstrap, then latents with no direct targets, trained by downstream loss.
- **iCoT 2405.14838:** the CoT is internalised progressively until the answer is alone.
- **ATC 2609.19717:** continuous thoughts learned with **no intermediate supervision** through a task-accuracy-gated difficulty curriculum (99.9 on shortcut-controlled reachability).
- **Latent-GRPO 2604.27998 and Abstract-CoT 2604.22709:** a latent SFT warm start, then answer-reward RL.
- **Lee 2502.01612:** a bootstrap, then extension by self-labels.

**New:** in a recurrent-memory loop, depth extension past the supervised range from answer loss alone, with a keep-targets (AEX) vs drop-targets (AEX0) ablation. Our own §245 (answer-only formation 1 / 4 seeds) is the internal motivation.
- **Outcome (Note 251-3):** AEX (targets kept on d ≤ 2) learned d 3–4 from answers (d 4 = 1.000 in 2 / 2; full causal bar 1 / 2, mid-swap 0.87 vs 0.46). AEX0 failed 0 / 2: one seed forgot the seeded skill, the other never extended.
  - **Is an anchor already reported?** Partly, as replay:
    - Lee 2502.01612 always keeps the gold D_0 in every self-training round's mix;
    - iCoT 2405.14838 fails if CoT is removed too fast (Δ 16), and needs removal smoothing / optimizer reset;
    - Coconut without its curriculum drops to 14.4 on GSM8K;
    - forgetting of the seeded skill is the standard continual-learning failure that replay prevents.
  - **Counterexample to "needed":** ATC 2609.19717 learns latents with no intermediate targets at all, from scratch, with a competence-gated curriculum.
  - **Suggested wording:** "In our recurrent-memory loop (2 seeds), answer-only depth extension (+2) succeeded only when step targets were kept on the seeded depths, which act as a replay anchor (cf. Lee 2502.01612's retained D_0); without them, one seed forgot the seeded skill and the other never extended. A competence-gated curriculum (ATC 2609.19717) is the untested alternative."

## Addendum (2026-10-30): ATC 2609.19717's exact mechanism (read in its HTML; for a C4 design)
1. **Gate:** advance when held-out **answer accuracy at the current difficulty > τ = 0.9 for m = 5 consecutive evaluations**.
   - **Rehearsal:** each stage mixes λ = 0.1 of examples from earlier difficulties.
   - **Backtracking:** on regression, the curriculum returns to the earliest failed difficulty ("rehearsal is preventive, backtracking corrective"); the default uses both.
2. **Difficulty = both.** At stage s, a depth-d instance becomes a question of depth min(s + 1, d) and gets min(s, d − 1) continuous thoughts. So the latent count is tied to stage and depth (d − 1 thoughts eventually; at most 16, one per stage).
3. **Architecture:** Coconut-style continuous thoughts (hidden states fed back as tokens), **not** a recurrent memory. Truncated-backprop variants (b = 2 / 5) are studied.
4. **Init:** **from scratch.** A two-layer GPT-2 (8 heads, width 768) on reachability. A 1-layer / 1-head model on parity, for the theorem's setting.
5. **Loss:** **answer CE only**, with no intermediate or reconstruction term. One augmentation: the two candidate answers are hidden on ρ = 0.75 of examples, to stop shortcutting to a candidate. AdamW at constant lr 3e-5.
6. **Seeds:**
   - Parity: 20 seeds per condition.
   - Reachability: the intervention probes report ± 1 sd over n = 20 seeds.
   - **No explicit failure count found in the parts read.** Table 1 gives ATC 99.9 % (51 epochs to 0.95) vs Coconut-gated 99.9 (132), Coconut-fixed 97.6 (468), and no-curriculum 47.8.

**Note 251-6 pointer (nearest-template op substitution):**
- McCoy, Pavlick & Linzen 1902.01007 (NLI models adopt lexical-overlap heuristics);
- Kim, Linzen & Smolensky 2212.10769 (lexical exposure);
- GSM-Symbolic 2410.05229 (template sensitivity).
- Not found for latent CoT specifically.
- **ANSWER-AT-CUE (2026-10-30):** the closest priors for depth-matched answer supervision are Fan 2409.15647 (loss at the given T(n)) and **ATC 2609.19717** (answer CE after exactly min(s, d − 1) thoughts per item, i.e. depth-matched). Chen 2603.21676 applies final-answer CE at **every** step (not depth-matched); TRM / HRM use deep supervision. Not found: depth-matched answer CE at a recurrent-memory write source, used to stop intermediate-code drift. **NARROW**, cite Fan + ATC.
