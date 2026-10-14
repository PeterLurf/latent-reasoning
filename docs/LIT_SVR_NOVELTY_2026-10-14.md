# Novelty check for SVR (shallow pass-free value readout), SEEDSTAB2 draft rev 3 (17c06299) (2026-10-14)

Literature session, for Audit (the request) and Theory.
- **IDs:** verified via arXiv abstract pages.
- **Claims:** CLRS / Generalist / Rodionov / "Cosine Misleads" checked in full text (pdftotext); the rest from abstracts.
- **Draft read:** §0, §4.1, and §8's prior-art rows.

## Verdict per component

| component | verdict | closest prior (tag) |
|---|---|---|
| (a) An auxiliary loss decoding **each intermediate value from each recurrent step's state** | **KNOWN** | **CLRS hints, 2205.15659, and the Generalist learner, 2209.11142** [full text]: per-step "hints" (intermediate algorithm states) supervised at every processor step through a **linear decoder for each hint**, shared across steps. Also DSN 1409.5185 (companion objectives on hidden layers), Concept Bottleneck Models 2007.04612 (supervised intermediate concepts), Trinh 1803.00144 (RNN auxiliary losses). All EMPIRICAL. **CLRS / Generalist are missing from the draft and are the closest precedent.** |
| (b) A **shared, step-index-free** readout | **KNOWN** | CLRS / Generalist decoders are shared over steps and take no step index. |
| (c) The **training-only, never fed back** constraint (the memory stays the only cross-pass channel) | **Distinguishing** | In the Generalist learner, "the hint predictions at each time step are fed back as inputs for the next step" (teacher forcing was removed, and its own predictions are re-encoded). SVR's readout never re-enters the computation. |
| (d) **Selectivity-constrained form**: linear, bias-free cosine head, stop-gradient query, per-head stop-gradient scaling, designed so the readout **cannot compute** (Lemma L) and any reported value must already be in the state | **NEW as a loss design** (not found) | The parents are probe-selectivity diagnostics (Hewitt & Liang 1909.03368; Voita & Titov 2003.12298), as the draft says. I found no paper that turns probe selectivity into a constraint on an *auxiliary training head* so the head can't do the work. |
| (e) Applied to a **pretrained LM's recurrent memory under identical input per pass** | **NEW in setting** | Neighbours: SIM-CoT 2509.20317 (a *deep* step decoder: the contrast the draft draws), CODI 2502.21074 (all-layer L1 to a teacher), iCoT-KD 2311.01460 (a learned emulator of teacher states). None uses a capacity-limited linear readout. |
| (f) Aimed at **seed-robust basin entry** (a tilt in a selection law) | **NEW as an aim** (not found) | CLRS-style hints are motivated by OOD / size generalisation. The Generalist paper uses teacher forcing and other stabilisers for *training stability*, not basin selection. |

## Prior evidence the registration should address (it bears on MET-NOT-USED)
1. **Rodionov & Prokhorenkova 2306.13411** (CLRS, EMPIRICAL): a model trained with hint supervision "demonstrates the alignment to the original algorithm trajectory, [but] could not capture the exact dependency between the hint sequence and the final result". No-hint models are competitive. That's exactly rev 3's MET-NOT-USED failure, observed in the closest precedent.
2. **"Cosine Misleads" 2606.05753** (EMPIRICAL): latents supervised by an auxiliary alignment loss "are largely bypassed". Corrupting them moves accuracy by at most 4 points, and alignment correlates *negatively* with accuracy (r = −0.94). Their PRISM pairs a linear probe with a corruption test, the same shape as rev 3's §4.3.
3. **"Observable Patterns Are Not Explanations" 2606.12689** (EMPIRICAL): in Coconut / CODI, decodable arithmetic and BFS-like patterns also appear in controls and do not always causally affect behaviour; use is graded.
- **Together** they make rev 3's causal tests (subspace ablation, donor swap, credit only for ENTERED-AND-USED) the right design. They also raise the prior on MET-NOT-USED, so cite them under §5's Goodhart analysis and §9's risks.

## Defensible claim line
"SVR's ingredients are known: per-step intermediate-value supervision through shared linear decoders (CLRS hints) and deep supervision. What is new is a **selectivity-constrained** form (linear, bias-free, pass-free, stop-gradient query, never fed back) on a pretrained LM's recurrent memory. It is derived so that the readout cannot compute the value (Lemma L), and it is used as a basin-selection tilt, with registered causal-use tests."

**Avoid:**
- "first per-step value supervision" or "first probe-as-loss": CLRS / DSN / concept bottlenecks have both;
- any wording that treats decodability as use: the three papers above.

## Suggested additions to §8's prior-art rows
- KNOWN: CLRS 2205.15659; Generalist 2209.11142 (linear per-hint decoders; predictions fed back).
- RISK EVIDENCE: Rodionov 2306.13411; 2606.05753; 2606.12689.
- RELATED: Concept Bottleneck 2007.04612.
