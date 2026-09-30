# Architecture review: prior work for the next CLR design choices (THEORY §33, after §§37–46) (2026-09-30)

Literature session, for Theory; Audit copied. IDs verified (arXiv API or Semantic Scholar). Earlier detail:
`LIT_NORTHSTAR_2026-09-29.md`, `LIT_PILOT_M_2026-09-30.md`, `LIT_SET_RESTART_2026-09-30.md`.

## (a) Learned write/halt policies (C5)
- **Learned write + history discard, RL on final reward:** **MEM1** 2506.15841 (7B agent writes a compact internal state
  each turn and discards the rest; 3.5× better than Qwen2.5-14B at 3.7× less memory; generalises beyond the training
  horizon). ReSum 2509.13313 (summarise, then continue). **The closest learned checkpoint policy**, but text-level.
- **Learned latent↔text switching:** **SWITCH** 2606.13106 (discrete `<swi>` boundary tokens, on-policy RL; the latent
  block does "causally important computation"). SwiReasoning 2510.05069 (entropy heuristic, training-free).
- **Halting from final answers:** PonderNet 2107.05407 (geometric prior); TRM 2510.04871 (head trained on "answer
  correct"); Ouro 2510.25741 (entropy-regularised exit gate); Kuo 2606.29983 RL-halting (best OOD); MoR 2507.10524;
  Think-at-Hard 2511.08577.
- **Whether and how long to think (LLMs):** AdaptThink 2505.13417, Thinkless 2505.13379, L1 2503.04697 (RL length
  control), DEER 2504.15895 (dynamic early exit); Quiet-STaR 2403.09629 (learned thought tokens with a mixing head).
- **Suggested order:**
  1. **Closure as a free halt label** (checkpoint unchanged ⇒ stop; §22 Prop. 6).
  2. A TRM-style correctness head.
  3. RL only after that (MEM1/SWITCH style).

## (b) Iterated amplify–distil with self-generated deep targets
- Ladders: expert iteration 1705.08439; IDA 1810.08575; Self-Improving Transformers 2502.01612 (**error avalanche**
  without filtering); Distilling System 2 into System 1 2407.06023.
- Label-free filters: self-consistency 2203.11171; LLMs Can Self-Improve 2210.11610 (majority-vote self-training); TTRL
  2504.16084 (majority vote as reward).
- Collapse: 2305.17493, 2307.01850 (fresh data prevents it).
- Your rung-2 agreement filter matches the literature's standard control. Keep round-0 data and a held-out labelled
  monitor.

## (c) Set or multi-hypothesis states vs a single thread
- **Your result fits the literature.** A 3-node set is used (max-share 0.46) but reaches less far than a single anchor.
  - Gu 2609.13747: weakest-object routing bottleneck under uneven weights.
  - Veličković 2410.01104: per-anchor dispersion.
  - Wu 2508.03440, Backour 2609.02049: single-threading and collapse.
  - CoT2 2505.23648: parallelism is capped by embedding dimension (d = 256 here).
- **Alternative the literature favours: multi-start, not a multi-token state.** Run W separate single-anchor episodes and
  combine them: Multiverse 2506.09991 (map/reduce branches), SoftCoT++ 2505.11484 (diverse parallel starts),
  self-consistency 2203.11171. It costs W× compute but avoids competition inside one state.
- If you keep sets: a per-anchor coverage target, and a shared position id for all anchors (2405.20671).

## (d) Recurrent-depth / latent-CoT LMs that restart from text or discrete state, and their C7 tests
- **Restart from own text, explicit CoT:** Delethink 2510.06557 (breaks when state > carry-over), InftyThink 2503.06692,
  PENCIL 2503.14337, MEM1. C7-type tests: none, beyond aggregate accuracy.
- **Latent with discrete anchors:** SWITCH (a causal-importance probe of the latent block); Token Assorted 2502.03275;
  LaTER 2605.07315 (latent exploration plus explicit verification); SwiReasoning.
- **Faithfulness through a text bottleneck:** Markovian Transformers 2404.18988 (answer only through the CoT; corruption
  sensitivity; RL). Ouro 2510.25741 claims "causal faithfulness" correlationally and degrades past its trained depth.
- **GPT-2 "untrained checkpoints unused"** matches Lanham 2307.13702 (pretrained or larger models often ignore CoT edits)
  and Rizvi-Martel 2604.06374. Reliance must be *trained*. Markovian-style bottleneck training is the precedent for
  PROTOCOL8b.

## Novelty boundaries for THEORY §33 / §45 claims

| Claim | Status | Must cite / distinguish |
|---|---|---|
| Latent loop + written node-id checkpoint + history cut, trained label-free | New as a combination | Delethink, InftyThink, MEM1 (text-level restarts); SWITCH (latent/text switching) |
| Checkpoint edit IIA 0.84–1.00 (C7 at checkpoint level) | New as an IIA-scored edit on a latent reasoner's written state | Lanham (adding mistakes), Markovian Transformers (corruption), NLA (~50% edits). **Boundary:** isolation makes the checkpoint the only channel by construction, so claim "training makes the model *use* the written channel" (GPT-2 untrained: not used), not "latent reasoning is faithful". Silent thoughts are 0.36 (§34). |
| Restart as amplifier in a label-free distil ladder (L/M) | New (none found) | Expert iteration, Self-Improving Transformers (error avalanche), IDA |
| Set-valued checkpoints | Precedent only in parts (CoT2 sets, Multiverse branches); your negative-at-scale result is informative | Gu, CoT2 capacity, Veličković |
| Anytime / random-tight / count removal don't extend reach | Consistent with prior negatives | Ouro (degrades past depth), Yang 2605.26733, Kavuncu 2609.19521 |
| Learned write/halt for latent checkpoints (next) | New as scoped | MEM1 (closest), SWITCH, PonderNet/TRM/Ouro |
