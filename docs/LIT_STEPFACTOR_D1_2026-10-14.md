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
