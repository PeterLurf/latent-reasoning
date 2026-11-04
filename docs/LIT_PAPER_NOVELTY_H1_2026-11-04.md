# Paper-level novelty: wave-4 doc 14 headline H1 (closest work; OK / NARROW / WITHDRAW)

Literature, written 2026-11-04 01:29 EST, at Theory's request after the 14_REDTEAM.md F10 finding (cb2a8cf0). Our own bibliography was grepped first. Two targeted web searches followed; under the 11-04 usage rule there was no sweep and no subagent. Every ID below was checked against the arXiv API on 2026-11-04. The closest-work rows come from **abstracts** unless marked otherwise. Before submission, the full texts of PAR, 2608.11233 and 2610.18127 should be read for causal tests.

**The claim checked:** "breaking positional slot counting in a looped pretrained LM gives C1 causal latent steps + depth extrapolation + a hold fixed point + stability halting in one model, op-order-free after §263".

**Verdict: NARROW, then OK as a conjunction.** No paper found has all of: a pretrained LM; a per-pass causal battery (state swaps, ablation, NOT-DECODER, cue-only); natural-language word problems with distractors and content selection; depth extrapolation; a fixed point held past completion; and a halting readout. Every component on its own has prior art, and one paper (PAR) has most of the non-causal components with a much larger extrapolation ratio. The paper has to name those differences explicitly.

## Closest work

| Paper (evidence type) | Has | Lacks (per abstract) | What the paper must say |
|---|---|---|---|
| **PAR 2610.27628** (empirics + ablations) | Pretrained Qwen2.5-0.5B loop; per-iteration progress-aligned targets (= our per-step supervision); state-norm anchoring (the closest prior to a hold / stability control); a learned completion readout (halting without a supplied depth); **16x** length extrapolation on swap-S5 (95.5% at 512 moves vs ≤32 trained); prefix self-distillation without intermediate labels (a C4-style route) | No reported causal / state-swap battery; synthetic state-tracking tokens, not NL text with distractors; no fixed point held past the answer | Put PAR first in related work. Explain why our ratio is +2-3 depths and theirs 16x: a different task structure (one move token per step vs content selection across sentences) and a different supervision scale. Consider a norm-anchoring ablation; Note 269-4 already makes PAR self-distillation C4-eligible |
| **2609.35643** Cheng & Zhang (empirics + causal interventions + circuit analysis) | Causal interventions and circuit-level localisation of a recurrent search; latents beat CoT / pause on longer held-out hops (ProsQA-Ext) | Trained from scratch (GPTNeoX); no plan cue; no hold / halting | Our difference: pretrained LM + given plan + per-step supervision + hold. They need *less* supervision; say so |
| **Retrofit 2608.11233** (empirics; preregistered ARC battery) | Pretrained Qwen2.5-0.5B prelude / loop / coda; "one task step per loop" with step supervision; extrapolation ~1.5x of supervised depth; a same-size scratchpad baseline that collapses beyond its horizon (the same shape as our X3 text-CoT comparison) | Causal battery not stated in the abstract; no hold / halting ("learned depth selection remains open") | Cite it as the closest "pretrained + one step per loop + beats scratchpad out of range" prior. Pre-registration is not unique to us |
| LOTUS 2606.31779 (empirics) | Looped padded transformer; per-step CE on latent positions; latents decode to the gold steps | Not a causal battery; no extrapolation / hold claim in the abstract | Cite for per-step supervision of loop latents |
| 2604.11791 (empirics, mechanistic) | Looped LMs converge to per-layer fixed points (cyclic trajectories) | Not trained holding; no task depth | Cite in the C6 section: fixed points arise naturally; ours is a *trained* hold to K' 128 with CTRL drift |
| 2610.18127 (empirics; five mechanistic lenses on eight looped LMs; **new to us**) | An inspection protocol for Ouro, LoopUS, Parcae, Rec-OLMo, Huginn | Unknown whether its lenses include causal patching; **read the full text before writing "no causal battery"** | |
| 2607.18553 (empirics; interventions on a frozen Ouro-2.6B; **new to us**) | Probes predict success; external interventions on the loop give no validated capability gain | Readout / control, not per-step state swaps | Cite as the nearest intervention study on a pretrained looped LM |

## The intro's contrast sentences

| Sentence (outline §2.3 item 1) | Verdict | Fix |
|---|---|---|
| "Coconut / CODI: no depth extrapolation" | **NARROW** | The original papers don't test deeper-than-trained splits. On GPT-2, SubLAT 2610.40283 Table 2 measures COCONUT at 7-hop 73.28 / 8-hop 72.46 (ProsQA is binary choice, chance ~50; LIT_SUBLAT_PROSQA_OOD_2026-11-03). Write "not designed for, and modest on, deeper-than-trained ProsQA splits" |
| "Coconut / CODI: contested causal use" | **OK** | Cite SCIT 2608.27265, 2512.21711 and 2604.04902 |
| "Huginn / Ouro: recurrent depth at scale, no causal battery" | **OK, but narrow the wording** | Huginn: post-hoc logit-lens decoding finds little latent CoT (2507.02199). Ouro: its "causal faithfulness" rests on probes and step-answer agreement, i.e. correlation (our bib already says so). The intervention studies (2607.18553) target readout / control. Write "no per-step state-swap / ablation battery reported", and check 2610.18127 first |

## Component claims (unchanged verdicts, collected here)
- Breaking positional slot counting with distractor insertion: **NARROW** (Note 256-1; Ruoss 2305.16843, GSM-IC 2302.00093, iGSM 2407.20311).
- Op-order-free: **no "first"**. The remedy is prior art (2502.19907); see 4a8ce800.
- Hold fixed point: **not new as an objective**. Use the terminal fixed-point objectives already cited in §2.5, plus 2604.11791 for naturally occurring fixed points.
- Stability halting ("the answer stops changing for r passes"): **not new**. Viakhirev 2608.18222 defines solve steps that way, and FPRM 2606.18206 uses residual halting; see LIT_HALTING_DEFS_2026-11-03. The result is out-of-range accuracy under that rule, not the rule itself.
- "N1 is where a loop is provably needed": see LIT_WAVE4_N1_MC3_2026-11-04. The claim is conditional on TC⁰ ≠ NC¹, and O(log m) loops suffice in principle (2505.18948).

## Suggested contribution sentence
"To our knowledge, no prior work combines, in one pretrained LM, latent loop steps that pass a per-pass causal battery on natural-language multi-sentence problems with depth extrapolation beyond training, a trained fixed point held to 16x the trained pass count, and an answer-stability halting readout. PAR (2610.27628) gets far larger extrapolation with per-step targets, norm anchoring and a completion readout on synthetic state tracking, but reports no causal tests. Cheng & Zhang (2609.35643) show causal, depth-generalising latent search, but in a model trained from scratch without a plan."

## Add to §2.5
PAR 2610.27628 (first); 2608.11233 (move from "recurrent depth" to the closest-work paragraph); 2604.11791; 2610.18127; 2607.18553; LOTUS 2606.31779; LoopFormer 2602.11451 (elastic depth / budget conditioning, for the C5 section); the §263 / distractor / critique priors listed in 14_REDTEAM F10.
