# Whole-project prior-art and novelty audit (2026-10-04)

Literature session, at the user's request via Audit ("run another literature check on everything we have done").
Claims are as stated in `docs/NORTH_STAR.md` (2026-10-03 table). arXiv IDs verified (arXiv API, Semantic Scholar or
abstract pages; LIT_INDEX). Novelty is **new**, **partly known** or **known**.

## Per finding: (a) closest prior · (b) novelty · (c) overclaim or missing qualifier · (d) what a reviewer will demand

**1. Latent restarts (text snap + history cut) give S5 length generalisation: train at 128, exact to 2,048
(§92/§97; range principle).**
- (a) Supervised latent recurrence with discrete re-embedding, 4× OOD from scratch (2510.14095). Thinking States
  2602.08332. Text carry-over with a cut: Delethink 2510.06557, PENCIL 2503.14337, InftyThink 2503.06692, Markovian
  Transformers 2404.18988. Native S5 extrapolation: DeltaProduct 2502.10297, Howe 2609.18966, PD-SSM 2509.22284.
  Pretrained LMs fail S5 past the training length (Li 2503.02854; 2602.14814).
- (b) **Partly known.** The mechanism is anticipated; the pretrained-LM, certified S5 instance is new
  (LIT_S92_NOVELTY).
- (c) The length generalisation belongs to the **snap + cut scaffold**: §71's written-state loop already reaches 2,048.
  §104 shows §97's latents are bypassed. So "carries the state in latents" must be per seed (C1 only where donor
  SERIAL holds), and the latent contribution to item 1 is C3 (compute), not C2.
- (d) A **native recurrent baseline reproduced on the same protocol** (DeltaProduct n_h = 4, GRU, Howe −b; §71's
  comparator was not reproduced), and same-format text with a cut at matched FLOPs.

**2. Markov segments + single-slot donor patching as the C1 test; serial, unqualified latent use in a pretrained LM
(§104/§108).**
- (a) Interchange / IIA: Geiger 2106.02997, DAS 2303.02536. Patching practice: 2309.16042. Latent-CoT interventions,
  all finding jumps, bypass or cache carriers:
  - Li 2602.08783;
  - SCIT 2608.27265 (CODI computes in the value cache);
  - Zhang 2512.21711;
  - Aswal 2606.12689;
  - Liang & Pan 2602.00449;
  - Chang 2606.01243.

  (Sequential activation patching 2608.22332 is explicit CoT only.) Bottleneck priors: our Pilot A, Markovian
  Transformers, RMT 2207.06881.
- (b) **The result is new** (no latent-CoT paper shows donor-consistent serial use at every step in a pretrained LM).
  **The method is partly known** (IIA plus bottleneck).
- (c) Under Markov segments the latent is the *only* route, so use is **enforced by the architecture**, not emergent.
  State it as "serial causal use under a forced bottleneck". S5 is a per-op lookup, the easy case. Before READ it
  held in 1/4 seeds.
- (d) The **same donor protocol on unbottlenecked latent baselines trained on our S5 data** (Coconut, CODI, SIM-CoT)
  to show they bypass. And a second task.

**3. The no-read basin (failing seeds sit on the "latent slots at chance" loss curve); a reader trained with the
writer from step 0 fixes it (§111 READ 5/5); 2× budget 0/5 and DAgger 0/3 don't.**
- (a) Posterior collapse and its decoder-side fixes: Bowman 1511.06349, free bits 1606.04934, lagging inference
  1901.05534. An auxiliary decoder against latent collapse: **SIM-CoT 2509.20317** (gold step targets). Also MUX
  2607.18264, 2606.20075. Seed lotteries fixed by curricula: Zhang 2607.20594. Early triage of bad seeds: Dodge
  2002.06305.
- (b) **Partly known.** The fix is SIM-CoT-like. The analytic loss-curve diagnosis of a no-read basin is new.
- (c) The reader uses **gold per-step state targets**, so it is per-step supervision (not C4). "Fixes the basin"
  rests on 5 seeds. DAgger s9–10 were pending at the time of the claim.
- (d) A **SIM-CoT-style decoder vs our slot reader** ablation, ≥ 10 seeds, and the reader with self-labels (§113/§117).

**4. Final-answer-only self-training through self-decoded restarts: exact S5 with a 10% foothold, never-labelled
transitions learned; the mechanism is the history cut (§110/§112; LIT_S116).**
- (a) STaR 2203.14465, ReST 2308.08998, Lee 2502.01612, **2511.07378** (recursive self-training provably extends
  length), Delethink, Markovian Transformers. Bootstrapping from own states: DAgger 1011.0686. Leap complexity
  2302.11055; Wies 2204.02892.
- (b) **New as scoped:** cut-induced footholds turning long final-answer problems into self-started shorts. The
  concept is bootstrapping (DAgger/TD).
- (c) "Never-labelled" transitions occur inside long problems constrained by final answers (note 110-5 wording). It is
  **coverage-qualified** (10–71.5%) and **1/5 to 1–2 seeds**. §110b's random-start null shows dependence on
  identity-start structure. Not strict C4.
- (d) **Strict C4 (§116 NONE)**, ≥ 5 seeds, and a **non-group task**.

**5. Carrier heterogeneity: the masked fork lands on slot-only, slot + op-token or prefix carriers across seeds
(note 112-3).**
- (a) Seed-dependent mechanisms: Li, Guo & Andreas 2503.02854 (AA vs PAA per run); Zhang 2607.20594; our toy (retrieval
  vs local). Off-latent carriers: SCIT 2608.27265. Multiple mediators: 2606.27510.
- (b) **Partly known** (seed-specific mechanisms are known; the carrier taxonomy for latent CoT is new).
- (c) Fine as descriptive. Don't generalise beyond the seeds measured.
- (d) Carrier localisation (single vs joint patch) across many seeds, as a distribution.

**6. §115: no-snap chains fail at every length; a round-trip disagreement trigger catches 100% of first errors at
about 20 snaps per 512 ops (unscored).**
- (a) Drift and collapse: 2609.02049, 2604.06374, Coconut / Ouro beyond-budget degradation. Routing by confidence:
  ThinkRouter 2602.11683. Reconstruction checks: ReLAT 2606.06252. Our §83 (confidence fails out of range).
- (b) **New as scoped** (the trigger). The drift itself is known.
- (c) **Unscored**, so no claim yet. Chain and re-encode share weights (correlated errors). "100%" needs a CI and an
  out-of-range test.
- (d) Pre-registered out-of-range scoring against a confidence trigger and a fixed k at a matched snap count.

**7. Multiplication: the multi-hand-off wall (§98/§100); pause tokens don't help (§107); 4×4 parity with ICoT-SI at
0.89× text-CoT FLOPs (§100).**
- (a) ICoT-SI 2405.14838 (released 0.998 / 0.996). **2510.00184** (multiplication fails for lack of long-range
  dependencies; a running-sum auxiliary fixes it). Cho 2410.15787, Lee 2502.01612, Jelassi 2306.15400, Neural GPU
  1511.08228. Pause: 2310.02226, 2505.21024; latent vs pause on depth: Cheng & Zhang 2609.35643.
- (b) **Partly known.** The wall is consistent with 2510.00184.
- (c) **"0.89× FLOPs" is against text CoT. ICoT-SI writes no CoT and is cheaper; state it.** §100 is not C1
  (mean-removal 0.13–0.63). 5×5 fails.
- (d) The 2510.00184 running-sum auxiliary on our setup (§109-A), and FLOPs against ICoT-SI.

**8. Negative results: speed-up levers that delay latent pressure slow learning (§99); confidence-triggered restarts
fail out of range (§83).**
- (a) §99: posterior collapse under a strong token path (Bowman); Coconut's curriculum. §83: OOD overconfidence (Guo
  1706.04599, Hein 1812.05720, Ovadia 1906.02530), CALM/LTT calibration.
- (b) §99 **partly known**. §83 **new as a measurement** (LM confidence under length extrapolation, first-error
  detection).
- (c) §99: all 27 runs censored at 24k, so "slow learning" is a censored comparison, not a measured slowdown. Word it
  as "none reached threshold within 24k".
- (d) §99: a longer budget or survival analysis on the censored runs. §83: more seeds and models.

## 2025–2026 work that already does items 2, 3, 4 or 6
- **Item 2:** no one shows serial donor-consistent latent use per step. Latent-CoT patching papers (2602.08783,
  2608.27265, 2512.21711, 2606.12689) find bypass or non-serial influence, which *supports* our §104 bypass finding.
- **Item 3:** SIM-CoT is the published auxiliary decoder against latent collapse (gold targets). SELR 2608.13570 and
  LatentGuard 2608.03838 are readers/decoders, not for the basin.
- **Item 4:** 2511.07378 proves recursive self-training extends length (on CoT). Delethink trains continuation from its
  own carried text under RL. No latent, final-answer, cut-footholded group acquisition.
- **Item 6:** ThinkRouter (confidence routing) and ReLAT (reconstruction check). No round-trip-triggered snap.

## Contradictions
- **Internal, worth fixing:** NORTH_STAR's C6 row says the earlier seed lotteries (§89/§90) "were budget". §111 shows
  2× budget 0/5 for the no-read basin, and Zhang 2607.20594 fixes a lottery by curriculum. Reword: "some lotteries were
  budget (timing); the no-read basin is not (§111)".
- **External:** none of our claims is contradicted by a known paper. Li 2503.02854 and 2602.14814 (pretrained
  transformers fail S5 out of range) and Aswal / SCIT (latent bypass) are consistent with our results and support the
  scaffold and bypass readings.

## The 3 most important missing comparisons
1. **Native recurrent S5 baselines reproduced on our exact protocol with FLOPs:** DeltaProduct n_h = 4, a GRU,
   Howe's −b Householder RNN. §71's comparator was never reproduced.
2. **Unbottlenecked latent-CoT methods trained on our S5 data and scored with our donor and length protocol:** Coconut,
   CODI, SIM-CoT. This shows Markov + reader are necessary, and SIM-CoT is the direct reader baseline.
3. **Matched-compute written-state baselines for every C2/C3 claim:** same-format text with a history cut (as §106-k2
   did), and ICoT-SI FLOPs on multiplication. Plus a **second, non-group task** (shortcut-free ProsQA or a held-out
   automaton) for C1/C4 generality.
