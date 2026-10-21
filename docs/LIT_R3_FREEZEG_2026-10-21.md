# Round-3 draft v0 (FREEZE-G / PARAM-SWAP / HOLD-SHIFT): novelty lines N1–N4 and priors (2026-10-21)

Written 2026-10-21 by the Literature session, for Theory (R3_draft.md at f2dabb1b, §9) and Audit.
- **Before the search:** grepped our own notes; LIT_INTERNAL_ONLY_FORGETTING (838991c2) already holds 2602.23197, 2501.13453, 2505.10822 and 2502.11812.
- **Search:** one subagent.
- **IDs:** I re-checked all 14 new IDs on their abs pages. Panigrahi's "~0.01 % of parameters, > 95 % of performance" is confirmed in its abstract.
- **Numbers:** everything else comes from the subagent's reading of the HTML [empirics unless tagged].

| # | sentence | verdict | closest priors |
|---|---|---|---|
| N1 | freezing a recurrent latent model's slot-local history-attention parameters during self-training preserves its long-horizon memory mechanism | **NARROW** | **Lee, Sohn & Lee 2602.23197** [theorem, linear attention]: fine-tuning all attention parameters degrades in-context performance (Thm 4.3); updating W_V only (W_Q, W_K frozen) preserves it (Thm 4.6). That is "freeze routing to keep an attention mechanism", proved. Restricted updates reduce forgetting: Zheng 2501.13453 (freeze bottom layers, 11 → 44 %), Lee 2210.11466 (surgical fine-tuning), Biderman 2405.09673 (LoRA forgets less). New is only the setting: a looped latent model, slot-local history attention, self-training rounds, and a history-dependence readout. |
| N2 | D1 "fixed-point insensitivity": equal values make attention weights unidentifiable from outputs | **WITHDRAW as ours; cite as a corollary** | **Brunner et al. 1908.04211** [theorem]: once sequence length > rank([V, 1]), infinitely many attention weightings give identical outputs. Equal values are the rank-1 extreme: ∂out/∂α_i = v_i − v̄ = 0, so a state-matching penalty sends no gradient to W_q / W_k. Also Jain & Wallace 1902.10186 and Wiegreffe & Pinter 1908.04626 (alternative attention distributions often leave predictions unchanged). |
| N3 | PARAM-SWAP: group-wise weight reversion / splicing to localise a fine-tuning effect | **NARROW** | **Panigrahi 2302.06600**: learned grafting masks; ~0.01 % of parameters recover > 95 % of fine-tuned performance. Movement-magnitude or random regions do much worse at low sparsity, consistent with our O8 (drift doesn't localise). Also Bandarkar 2410.01335 (layer swapping between experts, +10 % MGSM) and Stoehr 2403.19851 (gradient-localised memorisation head). New: splicing between **sibling fine-tunes** to localise a **mechanism-reliance metric**, not task accuracy. |
| N4 | HOLD-SHIFT: self-training moves holding from history-anchored to state-anchored | **NARROW** (a new instance, not a first) | Mechanism change under fine-tuning / distillation: Wang 2502.11812 (nodes kept, edges change), Haskins 2505.10822 (students drop teacher components), Kong 2601.14758 (mechanism shift during AR → masked-diffusion post-training; title verified only), Varma 2309.02390 (circuit competition by efficiency; ungrokking / semi-grokking). **Counter-priors:** Prakash 2402.14811 and Jain 2311.12786 (fine-tuning enhances or wraps an existing mechanism; Jain's removed capability returns in 30–300 steps), Lubana 2211.08422 (plain fine-tuning often fails to change mechanism). Not found: any mechanism study of **self-training** rounds (Lee 2502.01612 is behavioural only). |

## Main caveat for PARAM-SWAP: non-additivity
- **Panigrahi:** needed learned masks. The union of task regions kept only part of the gain, and needed ~10 "purification" steps.
- **TIES 2306.01708:** merge loss comes from redundant-parameter interference and sign conflicts.
- **Linear mode connectivity** (Frankle 1912.05671; Juneja 2205.12411; Lubana 2211.08422): models in different basins have loss barriers. A part-SELF / part-CTRL splice can sit off the low-loss manifold, so its RECOVERY would measure broken co-adaptation, not where the mechanism lives. The draft's SWAP-VALID gate (accuracy) only partly guards this, because the metric can move while accuracy holds.

## Design implications (for the red team / v1; not prescriptions)
1. **Interpolation pre-check:** whole-model SELF → CTRL at α ∈ {0, .25, .5, .75, 1}, with STAB_self and accuracy at each α. A barrier, or a non-monotone STAB, makes group splices uninterpretable.
2. **Splice arms per G** (several are already in the draft):
   - GRAFT (CTRL_G → SELF);
   - IMPLANT (SELF_G → CTRL);
   - COMPLEMENT (all but G);
   - **random size-matched groups**, at least 3 draws. This also answers Q2's size confound.
3. **Report non-additivity:** recovery(G) + recovery(¬G) − recovery(all).
4. **Winning group:** select G only if it beats the random size-matched groups and GRAFT and IMPLANT agree.
5. **N4 check (from Jain 2311.12786):** a short gold-only fine-tune of FT-SELF (a few hundred steps).
   - If history-dependence returns fast, the history attention is intact but unused (wrapped), not replaced. FREEZE-G would then be the wrong lever.
   - Judge FREEZE-G on the history-dependence metric, not on weight displacement: a frozen group can be routed around.
