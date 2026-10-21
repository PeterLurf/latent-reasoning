# Pre-scout for IDLE-HORIZON (Note 216-14): idle-pass drift and contraction / fixed-point levers (2026-10-21)

Written 2026-10-21 by the Literature session, for Theory. It matters only if **H_DRIFT** reads.
- **Our own notes already held:** LIT_DEEP_PASS_2026-10-11 (2606.18206, 2609.33144), LIT_ADD_NORM (STARS), LIT_CAND5_6 (local contraction hinge vs global Lipschitz), LIT_FIVE_LEVERS (path independence).
- **Search:** one subagent.
- **IDs:** every ID re-checked on its abs page.
- **Checked by me:** Viakhirev 2608.18222's dose-response and safety numbers, in its HTML.

## (a) Levers that make extra passes harmless
| ID | recipe | evidence | tag |
|---|---|---|---|
| **Viakhirev 2608.18222** ("Think Shallow, Solve Deep", 2026-08) | terminal fixed-point penalty ~ ‖f(z_{T−1}) − z_{T−1}‖, headline weight 0.05; trained at 16 steps, tested to 128 | **Dose-response (read):** weight 0 drifts (λ_max +0.04; depth damage 0.87–0.93); weight 0.005 already settles (damage gone, 0 / 3 unsafe); 0.02–0.2 also lifts "conversion" (0.60–0.97, saturating ≈ 0.88). **Across 164 competent runs, none of the 17 settling runs loses > 0.004 EM to depth**, against 30 % (marginal) and 22 % (drifting) losing > 0.1. An answer-only penalty does **not** give safety; the latent attractor is needed. | proposition (answer frozen once the remaining path is below the decoder margin) + empirics; small synthetic models |
| **STARS 2605.26733** | (1−λ)·L_SFT + λ·mean‖Jv‖², v from one power-iteration JVP, λ 0.1, loops ~ log-normal(1.7, 0.4) clipped to [1, 16] | Ouro-1.4B, 5-benchmark mean at loops 2 / 4 / 8: 66.9 / 74.2 / 65.6 vs Ouro 64.5 / 67.4 / 55.1 | empirics |
| **Bai 2106.14342** (DEQ Jacobian regularisation) | γ‖J_f(z*)‖_F² / d by Hutchinson (1–2 samples), applied on a Bernoulli(p) fraction of steps; e.g. WT103 γ 1.6 → 2.5, p 0.35 | stabilises equilibrium models; WT103 ppl 24.0 → 24.5 at half the NFE | empirics |
| **Anil 2211.09961** (path independence) | input injection; mixed init (half zero, half Gaussian); randomised depth U[3, 63] | path-independent models keep 100 % to 500 iterations (AA 1.00 vs 0.32, 87 %) | empirics |
| **Movahedi 2606.18206** (Fixed-Point Reasoners) | learnable residual scales α1 / α2 (init 0.75 / 0.25) with α2·λ_f < 1; deep supervision; TBPTT 4–6; damped inference | trained at 12–24 iterations, run far longer | theorem (sufficient condition) + empirics |
| InfiLoop 2610.11570 / Huang 2610.06833 | running-average carry with learned decay; orthogonal injection with ‖Q_qΛ‖ < 1 | TRM 62.2 → 58.1 % vs InfiLoop 90.9 % at 4× layers; ≤ 2 % perplexity cost at R 64 after training at R 5 (with TBPTT b 2) | empirics (Lemma assumes κ-contraction) |

## (b) Drift measured over extra iterations [empirics]
- **Shapiro 2608.11233** (Qwen-0.5B retrofit, supervised depth 12): 97.7 % at 12, then 85.9 / 70.3 / 46.1 / 10.9 % at 16 / 18 / 20 / 22. Single seed.
- **Ouro-1.4B**, loops 4 → 8: 67.4 → 55.1 (from STARS).
- **Viakhirev 2608.18222**, generic recurrence: prefix-sum 1.00 at depth 16 → 0.65 at 128.
- **Kolawole 2609.39277:** Huginn settles (late steps shrink ~0.93× per step); Ouro does not.
- **Liang, Jin & Pan 2609.33144:** past the training horizon, "limited self-correction allows local errors to persist and compound" (both loop types).

## Caution, and what fits our case
- **Over-contraction removes the computation.** Global Lipschitz < 1 conflicts with an isometric carry (LIT_CAND5_6). Viakhirev's own dose curve shows the window. The subagent also reports collapse to a constant answer on Huginn when the fixed-point loss was used alone; I did not read that passage myself.
- **Apply it only on idle passes**, at the smallest weight that settles. Viakhirev 0.005–0.05 is the evidenced range on small models. The lever is then a **local** fixed-point penalty on idle-pass slot states (after the answer is computed), not a global Jacobian bound.
- Viakhirev's answer-only control implies a penalty on the **latent state**, not only on the read answer.
- **Goodhart check for the derivation:** a fixed-point target on internal dynamics is a one-sided dynamics target, the trap in our 10-14 memory. Tie it to the held answer: penalise drift of the state only while the decoded answer is the computed one.
- **Overlap with our own levers:** Anil's randomised depth / mixed init is close to 216-9's small-n K′24 rollout mix. STARS-style log-normal loop sampling is our RAND-K family (LIT_RAND_K_2026-10-09).
