# General improvements to latent training and to our methodology (2026-10-13)

**Request.** The user asked: "Research general improvements to latent training that is not only specific to our architecture. But would also practically improve our methodologies too."

**Sources.** Three Opus subagents (G1 latent-training methods, G2 optimisation and stability, G3 experimental methodology). Their full notes, with verified arXiv IDs, abstract / full-text tags and a math basis per item, are in `theory/genlat199/`. Theory checked the arithmetic and the key harness facts; Audit checked G3's math.

**Status.** Nothing below is adopted. Literature's verification pass is docs/LIT_GENLAT_VERIFY_2026-10-13.md (980374f3): one error corrected (Muon), one scope note added (ZClip); the rest confirmed with evidence tags. Each change that alters training or how results are judged needs its own registration with a derivation (standing rule, 2026-10-13). Process changes go to the user via Audit. 

## General principles (architecture-agnostic)
1. **A seed's outcome is a binary random variable.**
   - "3 / 3" at a base rate of 0.6–0.67 is false 22–30% of the time (p³).
   - A one-seed rescue of a failed seed reads positive with probability p(1 − ρ) ≈ 0.4–0.6 under no effect, because same-seed outcomes correlate weakly across recipes (one-bit changes re-draw the basin; Summers & Dinneen 2103.04514) [G3 #1, #3; G2 #1].
2. **Bias beats noise.** In a noisy branch selection, P(good basin) = Φ(μ_Z/σ_Z) (docs/THEORY_SEEDSTAB Part II).
   - Levers that only change noise or time (batch size, longer windows, symmetric amplifiers such as Grokfast) scale as √ and need 15–42×.
   - Tilts need 4–7×.
   - Classify every proposed lever this way before spending [G2; Literature: Kondepudi-type law].
3. **The decision is early, and the eventual loser often leads.** The failed seeds had the lowest 8k losses (post hoc, p 0.033 / 0.012).
   - Never prune runs by early loss rank (successive halving).
   - Early detectors must be frozen and pre-registered [G3 #2, #5].
4. **Quality depends on when a run entered the basin.**
   - Between-run DEPTH SD is 0.086 raw but 0.029 within entry strata (eval noise 0.008).
   - Report quality within entry strata, and entry time as survival data [G3 #4].

## Ranked recommendations

| # | Item | Kind (theory) | Cost | Status |
|---|---|---|---|---|
| 1 | **Same-init linear mode connectivity** between s185 s1 / s2 / s3: a barrier on the s3 path means separate basins | diagnostic | $0 (local) | **registered: note 199-M2** (bd749ffe) |
| 2 | **Operating-characteristic block per seed-rate claim** (p0, p1, α at both base rates, power, design, fresh seeds, E[$], max $; a pinned script). Cheapest real claim: waves of 3 to 6/6 (α 0.047 / 0.088, power 0.53 at 0.9, max about $13.8 at $2.3 per cell) | method (exact binomial) | $0 | to the user via Audit |
| 3 | **Conservative censoring:** a frozen-detector flag or no entry by 30k counts as a failure, so α stays valid with an unvalidated detector | method | $0 | to the user |
| 4 | **Placebo (null-reroll) control** before any same-seed rescue: 2–3 inert one-bit replays of s3 measure ρ | method | about $2–4.5 | needs a registration |
| 5 | **Free logging:** per-group gradient and update norms, branch gradient-noise scale, the delta retention-bias excess over the pure-decay path (Δ_α), EMA shadow weights | diagnostic, logging only | $0 | to add to cells **after** §199-A's (pairing) |
| 6 | **No weight decay on biases, gains and gates.** wd 0.01 currently hits the retention logits. All 5 failed models sit on the pure-decay path, all 10 converged models above it (post hoc; a consequence readout). Decay pushes α toward forgetting. | small tilt, known sign | free | needs a registration |
| 7 | **Re-warm s3's final model** (constant LR, 10k): escape (Part II: P ≤ 0.12) vs drive threshold (Part I §5.4: enters) | discriminator | about $0.8 | needs a registration |
| 8 | **Relative clipping (ZClip-style).** Clip binds on every step at 2–8k, and per-step weights vary about ±28%. CLIP-1: gn vs aux_ce +0.67 at 1–12k, so high-aux batches are down-weighted in the decision window. **Scope (Literature):** with the clip binding on 100% of steps, a z-score threshold would mostly stop clipping, which changes the effective step size, not just spike handling. The registration must treat it as a step-size change. | small tilt plus a step-size change | free | needs a registration |
| 9 | **Annealing stage-switch rules** (continuous schedule, look-ahead tail, optimiser reset decided by measurement) for §197's fresh-seed redesign | escape lever | — | folds into §197 |
| 10 | **A correctness-trained halt head on a frozen model** (TRM / Ouro Stage II) for C5 | eval-side | $0–1 | later |
| 11 | **Check the trunk's LM ability** (perplexity) before natural-language transfer (GSM8K) | hygiene | $0 | before any GSM8K step |
| 12 | **A small-scale proxy for the lottery** (20+ seeds per arm), validated by decision accuracy before it gates spend | method | small | later |

## Conflicts and items ranked out
* **The sign of DETACH-PASS** (TRM / HRM one-step gradients) is disputed:
  - G1: a positive tilt (it removes the common-mode cross-pass drive).
  - G2: a negative tilt (keep full BPTT).
  - Audit: it removes credit for what earlier passes store, and TRM / HRM detach for refinement, not composition.
  - **Parked** until a derivation, or G1's $0 CPU gradient decomposition on s3, settles the sign.
* **Ranked out:**
  - Muon. No selection mechanism, and no consistent advantage when fine-tuning AdamW-pretrained models. Corrected per Literature 980374f3: Moonlight Table 6 is mixed (MMLU 55.3 vs 52.0, GSM8K −2.5); the 50.2 figure belongs to a different cell;
  - batch size (a √-lever);
  - precision and determinism work (neutral; record the card class);
  - random-initial-state training and STARS. Both push toward contraction, which our data contradicts: every converged model is path-dependent.
