# Looped / recurrent-depth latent reasoning: priors and flags for P1–P4, plus leads for the two walls (2026-10-04)

Literature session, for Theory (user directive via Audit); Audit copied. IDs verified (arXiv API 2026-10-04; earlier
notes). One line per item.

**The walls:** (1) the no-read basin (a seed lottery); (2) per-hand-off precision on rich states (the multiplication
running sum).

## P1: looped middle block per op (layers 4–7, r iterations, input injection, random r in training)
- **Priors:**
  - Huginn 2502.05171 (random-r training, test-time depth scaling); Ouro 2510.25741 (looped LMs at scale; degrades
    beyond trained depth); Universal Transformers 1807.03819; CoTFormer 2310.10845.
  - Relaxed Recursive Transformers 2410.20672 (converting pretrained LMs into layer-tied recursive models with
    layer-wise LoRA).
  - Mixture-of-Recursions 2507.10524 (token-level adaptive recursion depth).
  - **Retrofits:** Shapiro 2608.11233 (Qwen-0.5B looped retrofit, about 1.44–1.5× depth extrapolation); McLeish et al.
    2511.07384 (a recurrence curriculum preserves performance; deeper recurrence helps on maths); T²MLR 2607.15178.
- **Does test-time depth scaling transfer to retrofits?** Modestly: about 1.5× (Shapiro). McLeish reports gains with
  more recurrence on maths. Kohli 2604.07822 warns that overthinking degrades deep recurrence. Kuo 2606.29983:
  randomised stopping stabilises it across seeds.
- **Flags:**
  - (a) §107's PAUSE (more *positions*) didn't help. P1 adds *serial depth at one position*, which is different:
    multi-digit carries need serial depth (cf. Saunshi 2502.17416, looping ≈ implicit CoT). Pre-register P1 vs PAUSE
    to separate depth from width.
  - (b) Untargeted iterations are slack (our Prop. 4/9). Random r helps (Huginn); per-iteration targets as in §88 help
    more.
  - (c) Layer-tying a pretrained middle block needs a LoRA relaxation (2410.20672) or a recurrence curriculum
    (2511.07384) to keep GPT-2 intact.

## P2: halting by fixed-point residual ‖z_{i+1} − z_i‖ < ε (C5 handle)
- **Priors:**
  - ACT 1603.08983, PonderNet 2107.05407, DEQ 1909.01377, Attractor Models 2605.12466. Huginn uses a
    successive-step divergence exit.
  - **Direct support: "When Recursive Models Finish Computing" 2609.26487.** Latent motion drops sharply after
    completion, but the Jacobian keeps expanding directions (anisotropic stability). A residual threshold can fire
    early in flat directions, so calibrate ε in range and test out of range (our §83 lesson).

## P3: differentiable soft-text snap (mixture of token embeddings of the predicted step text)
- **Priors:** Soft Thinking 2505.15778, SoftCoT 2502.12134, CoT2 2505.23648, MUX 2607.18264 (lossless superposed
  spans).
- **Against:**
  - mixture collapse 2609.02049 (feedback amplifies or contracts);
  - 2604.06374 (superposition unused in fine-tuned LMs);
  - Wu 2508.03440 (top-token dominance).
- **Flag: use a straight-through Gumbel snap** (a hard forward, soft backward; Gumbel-softmax 1611.01144; SofT-GRPO
  2511.06411). It keeps the forward state discrete (error-correcting) while passing gradients. A fully soft forward
  pass invites the collapse modes above.

## P4: snap triggered by round-trip discrepancy ‖latent − Enc(Dec(latent))‖ > τ
- **Prior:** ReLAT 2606.06252 (reconstruction loss as an input-anchored fidelity check); SELR 2608.13570. Our §83
  showed token confidence fails out of range (miss 0.317), which motivates a non-confidence trigger.
- **Flag:** Enc and Dec share weights, so the discrepancy can be small exactly when both are wrong in a consistent way
  (the correlated-error caveat). Calibrate τ in range, test out of range, and report the first-error detection rate as
  §83 did.

## Other leads for the walls
**Wall 1, the no-read basin:**
- **Early triage from Audit's "no-read curve" signature.** Kill and restart seeds that sit on the
  (k−1)/k·(1−ρ)·ln 120 curve at the first checkpoint. Dodge 2002.06305 shows bad fine-tuning seeds are identifiable
  early and stopping them saves compute. This is our P7 idea applied here.
- **Teacher-forced segment inputs** (scheduled sampling 1506.03099) and **DAgger** 1011.0686 (§111 DAG arm).
- **Operator-first curriculum:** Zhang 2607.20594.

**Wall 2, per-step precision:**
- Error-correction noise on carried states: 2510.14095.
- A norm constraint on the state: nGPT 2410.01131 (hypersphere; faster, more stable training).
- A scalar running-sum regression auxiliary: 2510.00184 (§109-A).
- **Redundant state coding:** carry the state in 2–3 latents and decode by vote or average. The engineering form of
  error-correcting codes; I found no latent-CoT prior, so it would be new.
- Self-supervised latent-dynamics consistency: Next-Latent Prediction 2511.05963 (belief-state latents).
- Teacher-forced parallel state training: LIFT 2609.38149.
