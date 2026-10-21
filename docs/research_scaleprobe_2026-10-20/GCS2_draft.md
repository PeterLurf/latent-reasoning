# GSM-CACHE-SWAP-2 (successor to §220): the same battery behind a fidelity gate (DRAFT, 2026-10-21, not registered)

**Why a successor.** §220 closed at stage 0, UNREAD in both GPT-2 models (§220 stage 0 close, c9ddcd0a). The reason was structural, not a harness fault:
- §220's G-R compared our pad-free bs-1 path against the official test.py semantics at virtual bs 128. Those semantics attend unmasked left pads, so they are batch-dependent:
  - official-vs-official agreement is 0.87 (CODI20) and 0.66 (zen);
  - for zen they lower accuracy from the published 43.7 to 0.247.
- Our bs-1 path reproduces both checkpoints' established accuracies exactly: CODI20 0.3791 against 210-4's 0.379, and zen 0.4245640636846095, bitwise equal to §209 G1′.
- So the padded run is a degraded reference, not the reference computation. Audit ruled no second gate amendment; this successor is the route instead.

**What is inherited unchanged** (GCS_v1 at 3a88762e plus Addendum 1, Notes 220-1 / 220-2):
- models, stages, pins, items, bs 1, the parser;
- the exact-length pairs (rng 22020, ≤ 3 uses per donor);
- every arm: INTACT, MEAN-Z, STATE-SWAP, CACHE-SWAP, LEAK-SEG, SELF-SWAP, the descriptive rows;
- the permutation floors, readings, label precedence, outcome → action table, OC, censoring, wording;
- the stage-1 trigger (CARRIER-PORTS in either GPT-2 model).

**The one change: G-R → G-FID (fidelity), replacing G-R in full:**
- **(a) Port fidelity at bs 1.** For each model with an official code path (zen-gpt2; stage 1 CODI-1B), the official zhenyi4/codi test.py run at **batch size 1** has no pads, so its semantics are well defined. Our port's bs-1 INTACT must agree with it per item at ≥ 0.97 (parsed predictions, U10) and |Δacc| ≤ 0.01.
  - For CODI20, trained by our own codi_port with no official checkpoint, the port is the training code. (a) is replaced by: the bs-1 INTACT per-item predictions equal those of a second, independent bs-1 run of codi_port.evaluate (bitwise), which checks determinism.
- **(b) The established-accuracy gates,** as §220: CODI20 |acc − 0.379| ≤ 0.025; zen |acc − 0.4246| ≤ 0.025; CODI-1B 0.556 ± 0.03 as a flag.
- **The padded official-semantics runs** (virtual bs 128, and the 220-3 shuffles) are reported descriptively. They are never a gate.

**Data.**
- The stage-0 swap-arm data already exist in the committed eval_gcs.json files. They are byte-identical to what this design would compute (same script e5ea9325 path, bs 1, same seeds). **No one has read them.** They are sealed (sha 7c3822cf…).
- This design is registered and frozen **before** the seal is lifted. Its readings are then computed from those files plus the new G-FID runs. The new runs are only the official test.py at bs 1 for zen and the codi_port determinism re-run for CODI20 (about 15 min each, $0).
- So no arm value informs this design, which is Audit's condition for a successor.

**OC and priors:** as §220. P(pass | no effect) ≈ 0; P(pass | floor + 0.35) 0.98–0.99. G-FID adds a known-answer gate with P(fail | faithful port) ≈ 0, since bs-1 is deterministic and has no pads.

**Open for Audit:**
- (1) Is reusing the sealed committed data acceptable, given the freeze precedes the unseal? The alternative is a byte-identical re-run, which costs about 2 h of card time and changes nothing.
- (2) Is CODI20's G-FID(a) as a determinism check enough, given it has no official code path?
- (3) Should §220's appendix wording ("pad-free bs-1 harness, gate-failed") still apply once this successor reads the same values as registered readings?
