# Reliable entry into a contractive bottleneck: research memo (2026-10-05)

A memo for the user ("do more research", ~21:20 via Audit), under the task-agnostic directive (no per-task design) and
C1 > C6 > C4 = C2.
* Literature's checks: 86df4b3a (remedy map), ba531055 (this memo).
* Audit's framing: one compact panel instead of fix-by-fix. It is registered as **§127** in docs/THEORY.md.

## What the evidence says (§120, §123, §125, §126)
* **The 8-d bottleneck is the attractor.** Every bottlenecked seed that learns is exact to 2,048, with ρ_ff 0.146–0.180.
  There is no text, template or schedule (§123 result).
* **The bottleneck is also the entry basin.** S5 seed 2 sits on the ln-120 plateau by 2k in every bottlenecked arm on
  3090-class cards.
* **A wide carrier enters but does not hold.** RAW-H escapes the 2k flag, but drifts (s2: 0.846 at 512, ρ_ff 0.794) or
  fails (s3: 0) (126-7, 126-9).
* **What did not help:**
  - narrowing a working wide carrier (ANNEAL s2: 0; 126-8);
  - noise (§123);
  - §125's standardisation, which was never in force at 2k.
* **The symptom is posterior collapse** (He 1901.05534): the decoder ignores a near-constant code, so the encoder gets
  no signal.
  - At init, z_b's per-coordinate std across inputs is 0.03 (125-4).
  - claude-13's early replay (seed 1, steps 3–6): **z_b encodes the current op** (R² 0.95), not the start or next state
    (0.32 / 0.23). Its effective rank is about 2.
  - With uniform S5 starts, the op alone says nothing about the state, so a code holding only the op is useless to the
    decoder.

## Proposals (one panel, §127; restart-on-plateau is the baseline)
1. **Skip-to-decoder** (Dieng 1807.04863; Literature's top gradient source). The code also feeds GPT-2's upper blocks and
   the final hidden, at answer positions, through zero-init maps. It gives a short code → loss path from step 0.
2. **Item-residual variance (VICReg 2105.04906 / PLDM 2502.14819).** A hinge on the code's variance after removing each
   item's mean (a stop-gradient EMA per item). This variance can only come from what the segment brought in (start or
   carried code), so the op-identity trap can't satisfy it.
   - Time-wise variance (PLDM) is degenerate here: chains have ≤ 2 hand-offs before 2k, and it too is met by op identity.
3. **§125-v2:** standardisation actually delivered (per-forward batch statistics with stop-gradient, an absolute floor of
   0.5·sd_E). Its manipulation check is the row norm at 2k ≈ 4.
4. **10× LR on the new modules** (He 1901.05534; LoRA+ 2402.12354).

**Not run:** a predictive aux on z (CPC / NextLat). On S5 a code holding only the op makes the next code predictable from
the next op, so the trap is built in.

**Baseline: restart-on-plateau** (re-seed when the label-free 2k flag fires).
* Its cost per accepted seed is (1 + (p / (1 − p)) / 12) / a, with p the plateau rate and a the share of escaped seeds that
  end exact. A fix must beat it.
* **Drop rule:** an arm whose seed 2 plateaus is dropped, with no follow-up patch.
* If no arm escapes, restart-on-plateau is the procedure, and the line moves to the family extension.

## Strict C4 (angle 3)
* **Wording, per Literature (d):** "operator definitions (each op applied to the identity, no composition) suffice as a
  C4 foothold in 1/3 seeds (OPDEF s33 certified); with no foothold, training stays at chance (NONE 0/4)."
* Minimality is untested; it would need half the ops' definitions, or generators only.
* No new C4 registration now. OPDEF's reliability (1/3) is the next C4 lever, through whichever entry procedure §127
  adopts.

## Family extension (angle 4): design note
* **Harness rule:** each segment sees only [EOT, carried code, item]. No visible prefix can stand in for the state
  (114-4's lesson).
* **Families:** S5, flip-flop, cycle navigation (Delétang; train 1–40, test 500), and multiplication with a's and b's
  digits as items.
* **Code size:** multiplication needs about 10¹² states, so d = 16, L = 8 (2.8·10¹⁴) for every family.
* It runs with §127's adopted procedure (or restarts).
