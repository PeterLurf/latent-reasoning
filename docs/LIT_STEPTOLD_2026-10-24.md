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
