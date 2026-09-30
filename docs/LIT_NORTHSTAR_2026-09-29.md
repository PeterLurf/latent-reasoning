# Literature for the north star (C1–C7) — faithful readability, latent decoding, hybrid reasoning, halting (2026-09-29)

From the Literature session. It answers Audit's survey request (four parts, priority order) and Theory's five prior-art
questions for the checkpointed latent reasoner (docs/NORTH_STAR.md, THEORY §33). Every arXiv ID was checked, through
the arXiv API or Semantic Scholar. "(full text)" marks details read from the paper or its web version; the rest are from
abstracts.

## Headline for C7

**Nobody has published the behavioural round trip (thought → text → thought, substitute, behaviour unchanged) together
with the predicted-edit test (edit text → behaviour changes as a reference model predicts) for latent reasoning
states.** The pieces exist separately:
- **Natural Language Autoencoders** (NLA; Fraser-Taliente et al., Anthropic, May 2026, transformer-circuits.pub/2026/nla,
  full text) train a verbalizer (activation → text) and a reconstructor (text → activation) jointly with RL. They report
  reconstruction as fraction of variance explained (0.6–0.8 on Claude Haiku 3.5 / 4.5 and Opus 4.6), *not* as unchanged
  behaviour after substitution. They do run an **edit test**: edit the explanation ("rabbit" → "mouse"), reconstruct
  both, use the difference as a steering vector. It shifts the completion "only roughly 50% of the time". They list
  confabulation and possible private codes as limitations.
- **The critique:** *Train the Model, Not the Reader* 2607.20379 (full text). On released Qwen-2.5-7B NLAs,
  reconstruction is high (r̃ = 0.84), but a claim-flip audit finds only **~2% of salient specific claims are
  reconstruction-dependent**. "Reconstruction survives paraphrase but collapses when content words are masked": the
  gist carries the score, not the specifics. Their RECAP (auxiliary heads that keep designated content decodable)
  certifies what is *stored*, not what is *used* ("patching the tap barely moves behavior, ~0.003").
- *Understanding Confabulation and Rethinking Reconstruction in Activation Explanations* 2609.33702 (Sept 2026): as NLAs
  improve, they add unsupported details. Flow-NLA models the distribution of activations compatible with a text.
- *Do Activation Verbalization Methods Convey Privileged Information?* 2509.13316 (full text): zero-shot prompting on the
  *input alone* matches Patchscopes/LIT. Verbalizers often report their own knowledge, not the target's.

**Implication for our tests.**
1. A round trip measured by reconstruction (or by unchanged behaviour) can pass on *gist* alone. Our round trip must be
   paired with a **claim-flip / edit test at the level of individual node ids** (2607.20379's method), which our edit
   test already is.
2. Add the **input-only control** from 2509.13316: can the decoder produce the "checkpoint" from the graph and step
   index alone, without the thought? If yes, readability says nothing about the state.
3. The toy has an advantage the LLM work lacks: a **ground-truth reference model** (BFS on the edited frontier). So the
   edit test can use **interchange-intervention accuracy** (IIA; Geiger et al. 2303.02536, Causal Abstraction
   2301.04709): the share of edits after which the model's output equals the reference model's counterfactual output.
   That is the standard metric for "the edit changes behaviour as predicted" (see §1).

---

## 1. Faithfulness of chain-of-thought and monitorability — how it is operationalised

**Definitions.** Jacovi & Goldberg 2004.03685 (faithfulness means reflecting the true reasoning process; it is separate
from plausibility).

**Intervention tests on written CoT (closest to our round-trip and edit tests):**

| Work | Intervention | Our analogue |
|---|---|---|
| **Lanham et al. 2307.13702** | Early answering (truncate CoT), **adding mistakes** (does the answer follow the corruption?), **paraphrasing** (reworded CoT, answer unchanged?), filler tokens. Larger models are *less* faithful on most tasks. | **Paraphrase = behavioural round trip** for text; **adding mistakes = edit test**. Scored as % answer change or area over the truncation curve. |
| **Tutek et al. 2502.14829** (FUR) | *Unlearn* the information in one reasoning step from the weights; faithful if the answer changes | Parametric edit test |
| **Paul et al. 2402.13950** (FRODO) | Causal mediation: indirect effect of the reasoning on the answer | Mediation analogue of substitution |
| **Matton et al. 2504.14150** | Counterfactual edits of input *concepts*; compare which concepts the explanation says mattered with which actually did | "Edit changes behaviour as predicted", at concept level |
| **Atanasova et al. 2305.18029** | Counterfactual input editor inserts words that flip the prediction; checks whether the explanation mentions them | Edit sensitivity |
| **Chen et al. 2307.08678** | *Counterfactual simulatability*: from an explanation, can one predict the model's output on counterfactual inputs (simulation precision / generality)? | The prediction side of our edit test |
| **Parcalabescu & Frank 2311.07466** | CC-SHAP: self-consistency of input contributions to answer vs explanation | Weaker (correlational) |
| **Turpin et al. 2305.04388**; **Chen et al. 2505.05410** | Biasing features or hints that change answers but go unmentioned (Claude 3.7 / R1 verbalise hints only a minority of the time) | Hidden-channel test (a latent influence not in the text) |
| **Arcuschin et al. 2503.08679** | Unfaithfulness "in the wild" (post-hoc rationalisation) | — |

**Hidden channels and steganography:** Roger & Greenblatt 2310.18512 (**paraphrasing as a defence** against encoded
reasoning bounds the hidden capacity; close to our "hidden-channel" test); Skaf et al. 2506.01926 (steganographic CoT
emerges under process supervision and generalises); Baker et al. 2503.11926 (optimising against CoT monitors induces
obfuscation); Emmons et al. 2507.05246 (when CoT is *necessary*, models struggle to evade monitors, the argument for
necessity as a precondition for monitorability); Korbak et al. 2507.11473 (monitorability as "fragile").
Filler-token hidden computation: Pfau 2404.15758; Brauer et al. 2607.03502 (decodable); Baherwani et al. 2607.22925.

**Standard metric for "the edit changes behaviour as predicted".** There is no single CoT standard. Lanham reports
"% answers changed" under corruption, which checks *sensitivity*, not *correct direction*. The metric that checks the
*predicted* change is **interchange-intervention accuracy** from causal abstraction, used in DAS (2303.02536). Report
both: sensitivity (did it change?) and **IIA** (did it change to what BFS on the edited frontier predicts?).

## 2. Decoding latent states into language — and whether any is causally validated

| Method | What it decodes | Causal validation? |
|---|---|---|
| Logit lens (nostalgebraist 2020); **Tuned lens** 2303.08112 | Intermediate residual → next-token distribution | Partial: tuned-lens features are causally relevant ("causal basis") — read-out only, no round trip |
| **Patchscopes** 2401.06102 | Patch a hidden state into a target prompt; the model describes it | No round trip; 2509.13316 shows much of it can be matched from input alone |
| **SelfIE** 2403.10949 | The model interprets its own embeddings in text | **Edit-type**: "supervised control" edits open-ended concepts via its interpretations; "reinforcement control" erases knowledge |
| **LatentQA / LIT** 2412.08686 | A decoder LLM answers questions about activations | **Edit-type**: steers via gradients of a language-described loss through the decoder (e.g. Golden-Gate-style steering) |
| **Predictive Concept Decoders** 2512.15712 | Activations → sparse concept list → answers questions | No reported intervention validation |
| **NLA** (2026, above); Flow-NLA 2609.33702; RECAP 2607.20379; Universal Activation Verbalizer 2605.25903; AVPO 2609.34033 | Activation ↔ text autoencoding | **Reconstruction round trip** (FVE), **claim-flip audit** (2607.20379), **edit-as-steering** (NLA, ~50%). No behavioural-substitution round trip reported |
| **vec2text** 2310.06816 | Text embeddings → text via iterative re-embedding | A *representational* round trip (embed(decoded) ≈ target): the method is itself a round trip, but for sentence embeddings, not reasoning states |
| **Introspection**: Lindsey 2601.01828 (Anthropic); Binder et al. 2410.13787 | Inject a concept vector, and the model reports it (~20% detection in Opus 4/4.1) | **Causal by construction** (injection → report), narrow and fragile |
| Gurnee et al. 2607.15495 | Verbalizable representations form a mid-depth "global workspace" | Causal potency of verbalizable directions |
| Latent-CoT decoding: Coconut 2412.06769 (LM-head decoding of thoughts); SIM-CoT 2509.20317 (auxiliary decoder per latent); Dilgren & Wiegreffe 2604.04902 (gold traces decoded 65–93% when latents are needed) | Continuous thoughts → tokens/steps | Readability only; none substitute re-encoded text |

**Answer to Theory's Q3 ("has anyone done decode → re-encode → substitute?"):** for general activations, **NLA** comes
closest (a text bottleneck plus reconstruction, and an edit test through steering), but it reports reconstruction
variance, not unchanged behaviour. For latent-reasoning thoughts, **no one**. Write "first behavioural round-trip and
predicted-edit test for latent reasoning states", scoped to that.

## 3. Hybrid latent/discrete reasoning, learned externalisation, adaptive computation

**Hybrids (Theory Q1).**

| Work | Latent ↔ text interplay | Restarts from its own emitted text? | Emits a set? |
|---|---|---|---|
| Coconut 2412.06769 | Curriculum replaces text steps by thoughts; `uniform_prob` mixes stages | No | Distribution only (decodable) |
| **Token Assorted** 2502.03275 | Mixes discrete latent tokens (VQ-VAE codes) with text in one trace | No (interleaves) | No |
| **SoftCoT** 2502.12134 / **SoftCoT++** 2505.11484 | Assistant model generates soft thoughts for the main LLM; ++ uses *multiple diverse* soft-thought starts | No | ++: several parallel soft starts |
| **LightThinker** 2502.15589 | Compresses verbose thought steps into gist tokens and **discards** the text | Restarts from a *latent* summary (the opposite direction) | No |
| **PENCIL** 2503.14337 | Reduction rules erase intermediate thoughts | **Continues from reduced text** | No |
| **InftyThink** 2503.06692 | Segments with summaries carried forward | **Yes: restarts from its own written summary** | No |
| **Markovian Thinker / Delethink** 2510.06557 | Fixed-size chunks; context reset with a short text carry-over | **Yes: restarts from own text** | No |
| **SwiReasoning** 2510.05069 (ICLR 2026) | Training-free switching between latent and explicit by entropy trends | Latent ↔ explicit alternation | No |
| **SWITCH** 2606.13106 | *Learned* switching between latent recurrence and text via on-policy RL | Alternation | No |
| **CoT2** 2505.23648 | Continuous tokens as mixtures; samples and composes **K discrete tokens per step** | — | **Yes: explicit K-token sets** |
| Soft Thinking 2505.15778; MUX 2607.18264 | Probability-weighted concept tokens; multiplexed spans | — | Weighted sets (superposition) |

**Closest to our design:** InftyThink and Delethink restart from their own emitted text, but the text is prose and the
loop is explicit CoT, not latent. CoT2 and Soft Thinking emit sets, but never restart a latent loop from them. **No work
combines a latent loop, a discrete set-valued checkpoint, a restart with the history cut, and a learned checkpoint
policy.**

**Learned halting / when-to-write from final answers only (Theory Q4).**
- **PonderNet** 2107.05407 (halting distribution with a geometric prior, answer loss only); ACT 1603.08983.
- **TRM** 2510.04871 (full text): a halting head trained on "is the current answer correct", i.e. from final answers.
- **Kuo et al.** 2606.29983: RL-halting, the best OOD result (45%).
- **Ouro / LoopLM** 2510.25741 (full text): an exit gate trained with an entropy-regularised expected loss (Stage I),
  then a greedy marginal-improvement signal (Stage II). No step labels.
- **Mixture-of-Recursions** 2507.10524 (learned per-token routing); Huginn 2502.05171 (KL-threshold exit, zero-shot);
  **Think-at-Hard** 2511.08577 (a neural decider for extra latent iterations).
- **When to write (externalise):** **AdaptThink** 2505.13417 and **Thinkless** 2505.13379 (RL learns *whether* to produce
  text reasoning); **SWITCH** 2606.13106 (RL switch between latent and text).
- None learns *when to emit a checkpoint and restart*. The nearest are SWITCH (switching) and Delethink (fixed chunk
  length).

**Set-valued intermediate states and parallel starts (Theory Q5).** CoT2 2505.23648 (K discrete tokens per step; the
parallelism limit is set by embedding dimension); SoftCoT++ 2505.11484 (diverse parallel soft starts); LaDiR 2510.04573
(latent diffusion with repulsive diversity guidance across a batch of latent trajectories); *Next Thoughts Are
Distributions* 2609.33271 and **Latent Thought Flow** 2606.16222 (sample next thoughts from multi-mode distributions;
GFlowNet over latent trajectories); Gu et al. 2609.13747 (capacity of superposed alternatives); Coconut's own analysis
(spread, then commit). **Cost/parallelism evidence:** CoT2 (parallelism ∝ embedding dimension); Gu (cumulative vs
frontier capacity); our per-edge budget law (Finding 3). A set checkpoint of W node ids has cost ∝ W tokens but
restores every branch's amplitude, which the literature does not quantify for latent loops.

## 4. Does any published system claim several of the seven conditions together?

| System | Scale | Conditions claimed (our labels) | Missing |
|---|---|---|---|
| **Ouro / LoopLM** 2510.25741 | 1.4B / 2.6B, pretrained | C3-ish (matches 4B/8B dense models), C4 (pretraining, no step labels), C5 (learned exit), **C7 claimed** ("reasoning traces more aligned with final outputs … greater causal faithfulness") | **C2 fails**: "performance peaks at the trained depth (T = 4) and then degrades". C7 is claimed via intermediate-prediction alignment, not a round trip or edit test |
| Huginn 2502.05171 | 3.5B | C4 (pretraining), C5 (adaptive exit), C2 partly (more recurrence helps up to a point) | C3 vs CoT, C7 |
| LOTUS 2606.31779 | up to 3B | C3 (first latent method to match explicit CoT at 3B) | C2, C7 |
| SIM-CoT 2509.20317; CODI 2502.21074 | GPT-2 to 8B | C3 at GPT-2 (SIM-CoT +2.1 over CoT), partial C7 (per-latent decoder) | C4 (needs CoT teacher), C2 (our §13: horizon at trained budget) |
| SwiReasoning 2510.05069 | LLMs | C3 (Pareto-superior), C5 heuristic, C7 partial (explicit segments) | Learned policy, C2 |
| TRM 2510.04871 / HRM 2506.21734 | 7–27M, not LMs | C2-ish (more recursion helps), C4 (no step labels), C5 (halting) | LM, C3 vs CoT, C7 |

**Verdict:** the most comprehensive claim is **Ouro** (C3–C5 plus a C7 claim at 1.4–2.6B). Its own tables show C2 failing
beyond the trained depth, and its faithfulness evidence is correlational. No system claims C2 in every seed, and none
tests C7 causally. The north-star combination, especially C2-in-every-seed plus causal C7, is open.

---

## Summary for Theory's five questions (ranked)

1. **Hybrid restart-from-own-text:** InftyThink 2503.06692 > Delethink 2510.06557 > PENCIL 2503.14337 (all explicit);
   latent hybrids Token Assorted 2502.03275, SWITCH 2606.13106, SwiReasoning 2510.05069, LightThinker 2502.15589.
   **Set emission:** CoT2 2505.23648, Soft Thinking 2505.15778. None does latent loop + set checkpoint + restart.
2. **Causal CoT faithfulness:** Lanham 2307.13702 (paraphrase = round trip; adding mistakes = edit) > Matton 2504.14150 >
   Tutek 2502.14829 > Chen 2307.08678 (counterfactual simulatability). Hidden channel: Roger & Greenblatt 2310.18512,
   Turpin 2305.04388. **Metric:** IIA (2303.02536) plus sensitivity.
3. **Decode → re-encode:** NLA (transformer-circuits 2026) is the only activation↔text autoencoder with an edit test
   (~50%). Critiques: 2607.20379 (~2% of claims grounded; add a claim-flip audit), 2509.13316 (add an input-only
   control), 2609.33702. No behavioural substitution anywhere.
4. **Learned halting and when to write:** Ouro 2510.25741, TRM 2510.04871, PonderNet 2107.05407, Kuo RL-halting; when to
   write: AdaptThink 2505.13417, Thinkless 2505.13379, SWITCH 2606.13106.
5. **Set-valued states / parallel starts:** CoT2 2505.23648, SoftCoT++ 2505.11484, Latent Thought Flow 2606.16222,
   2609.33271, LaDiR 2510.04573; capacity from Gu 2609.13747.
