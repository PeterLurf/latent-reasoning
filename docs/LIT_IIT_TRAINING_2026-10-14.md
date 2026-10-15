# Interchange-intervention training (IIT) as a training objective for looped / recurrent latent reasoning: ID check and prior art (for DAS-USED / the next method) (2026-10-14)

Literature session, for Theory (the request) and Audit.
- **IDs:** verified via arXiv abstract pages.
- **Sources:** IIT 2112.00826 and InterpBench 2407.14494 in full text (pdftotext); the rest from abstracts.

## (1) 2112.00826: VERIFIED
- **Paper:** Geiger, Wu, Lu, Rozner, Kreiss, Icard, Goodman & Potts, "Inducing Causal Structure for Interpretable Neural Networks" (ICML 2022, PMLR v162).
- **The method:** align high-level causal variables with neural representations, then train the network to match the causal model's **counterfactual** output when the aligned representation is set to its value on a source input.
- **The guarantee:** "guarantees that the target causal model is a causal abstraction of the neural model when its loss is zero" (abstract). **THEOREM-level, under exact zero loss**; in practice it's a soft objective.
- **Tasks:** MNIST-PVR (ResNet), **ReaSCAN with a CNN-Bi-LSTM** (a recurrent encoder-decoder; IIT aligns variables with LSTM-side representations), and MQNLI (BERT).
- **On ReaSCAN's systematic-generalisation splits,** "only models that use IIT are able to consistently get traction" (e.g. novel-length splits where standard training sits at about 0).

## (2) Prior art for IIT-style training of recurrent / looped / latent-CoT models
| line | status | notes |
|---|---|---|
| IIT on a **recurrent** network | **KNOWN** (2112.00826, the CNN-Bi-LSTM on ReaSCAN) | An encoder-decoder over a sequence, not a weight-tied loop over passes. |
| IIT follow-ups | KNOWN | Causal distillation 2112.02505 (interchange interventions between teacher and student, on BERT); type-level IIT for characters in subword LMs 2212.09897; Strict IIT (SIIT) in InterpBench 2407.14494; an MLP application 2503.14442 (glucose). All EMPIRICAL. |
| IIT on **looped / weight-tied** transformers, **latent CoT** (Coconut / CODI / PCCoT / LOTUS) or **CLRS-style algorithmic reasoners** | **NOT FOUND** | The latent-CoT causal work is analysis only: do-interventions in 2602.08783, patching in 2602.00449, causal-geometric low-rank directions in 2606.12689. |
| IIT on the **whole recurrent state** (coordinate-free) at a pass boundary, with the counterfactual from a known serial program | **NOT FOUND** | The closest is the IIT framework itself, where alignments can be whole layers. Your coordinate-free argument (DAS coordinates don't transfer across runs under the per-head GL(64) × O(64) symmetry) is a sound reason to prefer it. |

## Gaming evidence and design implications
1. **IIT doesn't constrain un-aligned components** (InterpBench 2407.14494, full text):
   - "IIT never intervenes on low-level nodes that are not aligned with any node in the high-level model. This can lead to scenarios in which the nodes that are not intervened during training end up performing non-trivial computations that affect the low-level model's output."
   - Their fix, **SIIT**, also intervenes on non-circuit nodes and trains them to have no effect. Their ablations find "SIIT correctly generates transformers with known circuits, even when IIT does not".
   - **For us:** with the whole memory as the aligned variable, the un-aligned path is the **within-pass trunk acting on the re-presented input**. A plain IIT loss lets the trunk compute part of the answer directly.
   - **The SIIT analogue:** swap or resample the *input-side* features while holding the memory, and require no output change where the program says the memory alone determines it.
2. **Precomputation can satisfy a whole-state swap if the pairs are badly chosen (DERIVED).**
   - If the state after pass 2 already encodes the *final* answer (one-pass dominance), swapping in the donor's state yields the donor's answer.
   - That matches the counterfactual target **only** when the base and donor share all post-swap operations.
   - **Design rule:** choose (base, donor) pairs whose **post-swap operations differ**, so the program's counterfactual (donor's intermediate, continued with the base's remaining steps) differs from both originals. Then precomputation fails the loss, and only a state carrying the *intermediate* value and a later pass that *uses* it can meet it.
   - Report the share of pairs where the counterfactual ≠ the donor's own answer.
3. **Presence ≠ use still applies,** but in reverse: IIT directly trains *use*, so the C7 risk moves to **gaming via un-aligned paths** (point 1). Keep §203-style random-subspace and clean-run checks as evaluation.
4. **IIT's own track record is EMPIRICAL:** gains in systematic generalisation in its three settings. There's no evidence yet about seed reliability. Using IIT to make basin entry reliable would be new.

## Defensible claim line
"Interchange intervention training (Geiger et al. 2112.00826) is established, including on a recurrent encoder (CNN-Bi-LSTM). To our knowledge it has not been applied to looped / weight-tied latent-reasoning models, or with the whole recurrent state as the aligned variable, as a training objective to make intermediate states causally used. We adopt SIIT's lesson (2407.14494) by also constraining the non-memory path."
