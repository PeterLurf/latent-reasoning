# Novelty pass on STEP-TARGET-FACTOR (§204) D1: "under teacher forcing, the string aux target's value tokens are memory-free" (2026-10-14)

Literature session, for Theory (the request) and Audit.
- **Sources:** SIM-CoT 2509.20317, CODI 2502.21074 and PCCoT 2506.18582 searched in full text (pdftotext) for teacher forcing, shortcut and leakage. Abstracts for the rest.
- **IDs:** verified.

## Verdict: D1 as stated is NOT FOUND in the CODI / SIM-CoT / PCCoT literature. Its general principle is PARTLY KNOWN.
- **SIM-CoT** describes its step decoder as teacher-forced on the step string, p(y_{k,t} | z_k, y_{k,<t}) (§3.4). It **does not discuss** that the result tokens after "a =" are predictable from the teacher-forced prefix alone, so the latent never has to hold the result. There's no "shortcut" or "leak" discussion.
- **CODI** names a *different* in-context shortcut: it drops the final CoT step because "the model might directly copy the result from the last CoT step to the token responsible for generating the exact answer", which would spoil the distillation target (§3; ablation "w/ CoT last step" 31.7 vs 43.7). That's a shortcut on the teacher's answer token, not on per-step value tokens.
- **PCCoT:** no discussion (it inherits CODI).
- **RiM 2605.30343** masks readouts from *other* steps ("without information leakage"). Within a step string, though, its targets are still teacher-forced, so D1 applies to RiM too, and RiM doesn't discuss it.
- **The general principle is known: Bachmann & Nagarajan 2403.06963**, "The pitfalls of next-token prediction" (DERIVATION + EMPIRICAL).
  - Teacher forcing can fail to learn the intended predictor because later tokens become predictable from the revealed prefix (their "Clever Hans" mechanism).
  - Their fix is "teacherless" multi-token prediction.
  - D1 is a specific instance: the revealed operands make the value token locally computable, so no pressure reaches the latent.
  - **Cite it as the parent.**
- Recent latent-CoT causal / mechanistic studies, 2602.00449 (CODI on sequential tasks; probes and patching) and 2602.08783 (Coconut / CODI step-wise do-interventions), report how intermediate states are used. They don't make the target-design point.

## Defensible claim line
"To our knowledge, the latent-CoT step-supervision literature (SIM-CoT, CODI, PCCoT, RiM) does not note that teacher-forced step strings leave result tokens predictable from the in-context operands, so they apply no pressure on the latent to hold results. We show this formally (D1) and measure it (value top-1 0.880 with zero memory vs 0.894). It's an instance of the teacher-forcing 'Clever Hans' failure (Bachmann & Nagarajan 2403.06963)."

**Tag:** D1 itself is PROVED (an information argument), with in-house DATA. The parent is a DERIVATION + EMPIRICAL.

## Implication worth stating
D1 predicts that **SIM-CoT's step decoder mainly trains operand / plan identity, not results**, on the string format. That's a cheap, testable reading on the public SIM-CoT checkpoint we already bench: decode the step tokens from z_k, with vs without z_k, on the value positions.

## Addendum (2026-10-14): the SIM-CoT checkpoint test (THEORY note 204-2, 5b14cc44)
- **Setup:** internlm SIM_COT-GPT2-Coconut checkpoint_28, using its own step decoder, on Aug valid 500, stages 1–4. Teacher-forced top-1 under three latent conditions: TRUE / MEAN-latent / ZERO-latent.

  | token type | TRUE | MEAN | ZERO |
  |---|---|---|---|
  | value | 0.939 | 0.915 | 0.447 |
  | operand q | 0.692 | 0.091 | 0.029 |
  | operand r | 0.530 | 0.053 | 0.024 |
  | format | 0.963 | 0.999 | 0.499 |

- **Reading:** under the in-distribution MEAN replacement, the decoder reads results from the in-context operands and needs z_k only for operand / plan identity, as D1 predicts.
- **The ZERO collapse is off-distribution:** it even breaks the item-free format tokens.
- **Prefer MEAN (or resample) over ZERO as the ablation baseline.** Zero ablation is a known off-distribution corruption: Zhang & Nanda 2309.16042 ("Towards Best Practices of Activation Patching") show the corruption method can change the conclusions. Mean ablation is the IOI-circuit convention (Wang et al. 2211.00593). Both are EMPIRICAL.
- **By its pre-stated letter the test reads MIXED.** Report it as: "replacing the step latents with their mean leaves value-token accuracy at 0.915 (vs 0.939) while operand accuracy falls to 0.05–0.09; ZERO, an off-distribution baseline, collapses all token types (value 0.447, format 0.499)."
