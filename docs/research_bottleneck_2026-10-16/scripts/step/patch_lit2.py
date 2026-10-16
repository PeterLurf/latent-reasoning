import sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, ("NOT UNIQUE/FOUND", a[:70])
    s = s.replace(a, b)


# --- summary
rep("""   - Together they remove the training signal that rewards recomputation and add a signal that penalises it at every handle height, from step 0.
4. **Plan.** $0 diagnostics on existing checkpoints and on §211's finals, then one or two nested cells (about $0.8–2 each).""",
"""   - Together they remove the training signal that rewards recomputation and add a signal that penalises it at every handle height, from step 0.
   - A second cell adds **SELF-SNAP**, LIT2's lead from Altabaa 2510.14095. It snaps the row-space part of each stored version to a canonical code of its decode, straight-through, so the step's input code is pass-free.
4. **Plan.** $0 diagnostics on existing checkpoints and on §211's finals, then two nested cells (about $0.8–2.2 each).""")

# --- prior art block in 1.2
rep("""  - **The diffuse gather.** Off-self mass is 0.80–0.94 in N2/N3 (211-9, 211-12). The pass input at slot j mixes positions, so core depth is spent reassembling E_{t−1}.
""",
"""  - **The diffuse gather.** Off-self mass is 0.80–0.94 in N2/N3 (211-9, 211-12). The pass input at slot j mixes positions, so core depth is spent reassembling E_{t−1}.

**Prior art (LIT2, scratchpad research2/LIT2.md; IDs verified there).**
- **Per-loop targets work on this task.** RELAY 2502.08482 trains per-loop CoT-round targets on Feng-style arithmetic, ≤ 15 → 25 operators: 0.98, against 0.77 unsupervised. It is from scratch with an oracle loop count. This is R2's premise.
- **At depth, the lever was a discrete snap plus corruption, not the soft target.** Altabaa 2510.14095 (mod-23 DAGs, N ≤ 32 → 128): snap + corruption 100%, snap alone 94.8%, continuous latent supervision 54.8%. That is a measured S2 failure (§3.3) in a related setting.
- **Ground-truth teacher forcing hurts OOD.** CLRS 2209.11142 ("feeding ground-truth hints … hurts generalisation") and one-step denoising (2604.18839). This fits N3's failure to recover (211-12) and argues for own-state training.
- **Full cross-iteration history hurts depth extrapolation.** CoTFormer 2609.19521 scores 50.1% at depth 20, against 99.4% for recent-1; see also Bohde 2403.04929. VB's bank is this pattern, which is a C2 risk (§7).
""")

# --- 2.1 alternatives
rep("""- MCT (N3) forces corrected inputs into a bank that a diffuse gather mixes (211-12 (c)).""",
"""- MCT (N3) forces corrected inputs into a bank that a diffuse gather mixes (211-12 (c)). It is also ground-truth teacher forcing, which hurt OOD in CLRS 2209.11142 and DRM 2604.18839.""")

# --- replace old 2.3
a0 = "### 2.3 Secondary mechanisms (registered only after the stated trigger)"
a1 = "**Rejected: per-operator MoE (c).** The premise fails (E6)."
i0 = s.index(a0); i1 = s.index(a1) + len(a1)
new23 = """### 2.3 SELF-SNAP: a canonical row-space code for stored versions (cell S2; LIT2 lead (i))

**Stored entry after pass t:**

  S_t = V_t + α(step) · sg(Δ_t)

- Δ_t is the row-space edit (LN coordinates, pinv of R's weight, ≤ 3 iterations, VBankN.mct machinery) that sets R's logits at every real slot to the canonical point L*(dec V_t) = μ_s·(onehot − 1/20), with μ_s = 4 nats.
- **Forward:** the bank, gather and carry read S_t. R2 / COMMUTE and R1 read the computed V_t (as in MCT's pin (e)).
- **Backward:** identity through sg(Δ) (straight-through).
- **Ramp:** α goes 0 → 1 over steps 4k–8k, after R has trained. Per-slot ‖Δ‖/‖V‖ is clipped at 0.05, and the clip rate is logged.
- **OPX under SNAP** ("snap corruption", Altabaa B.4) is the same sampling as §2.2. The operand slots' snap targets in versions s_k..t−1 become a′ / b′ instead of dec, and the target is COMMUTE's (follow).
- **Not proposed: full snap** (S_t = emb(dec V_t)). The inter-pass state would then be a 20-class slot sequence, a discrete scratchpad. That undercuts the user's standard of using latent space better than text (NORTH_STAR C4). The row-space snap keeps 748 of 768 dimensions continuous.
- **Parameters reached:** as S1. SNAP adds no parameters. Its edits are sg; R's weights shape Δ only through the forward pass.

**Corruption semantics.**
- Altabaa keeps the true target, so the model learns to *correct* corruptions.
- In VB a reduced operand's own operands are VOIDed. Correcting a corrupted computed value therefore needs the bank history (the version in which its operands were alive) or a deep recompute.
- The history evidence above (CoTFormer, Bohde) favours the Markov "follow" semantics.
- A "correct" variant (corrupt only the newest copy, target true) is a descriptive alternative and is not proposed.

### 2.4 Secondary mechanisms (only after the stated trigger)

- **OFFSET** (stationarity, coverage).
  - With p 0.5, o ~ U{1..4} forced copies of V_0 are appended as versions 1..o, with HOLD targets, before reduction 1. This decorrelates pass index, bank size and age from progress (§3.3 coverage).
  - **Trigger:** COUNT-KEYED, PASS-KEYED, or DEPTH < 0.5 with STEP-COMPOSES.
  - STALL then becomes trained-for, and NO-CLOCK becomes descriptive.
- **Bansal start** (2202.05826, LIT2 lead (ii)).
  - t₀ ~ U{0..n_i−1} passes run under no_grad, with versions and bank K/V detached. Passes t₀+1..K_i then run with gradient and losses.
  - [DERIVED] No gradient reaches what the prefix writes. Cross-pass hidden channels (null-space information a later pass learns to read, the S2 violation) are then not reinforced from the prefix, and the step is trained from its own state at a random depth.
  - **Trigger:** PASS-KEYED. **Cost:** the prefix's writes are trained only by undetached batches.
- **Capacity** (design question (b)).
  - **Trigger:** H-CAP (low STEP-CAP in every hh class, rising with VOID gap).
  - Probe first with a fixed 2× core per pass, not learned halting: ACT / PonderNet collapse to one loop (RecurTrace 2609.03379).
  - If 2× helps, ELASTIC-RECODE: L = −log Σ_{monotone a: a(t) − a(t−1) ∈ {0, 1}, a(K_i) = n_i} Π_t q_t(E_{a(t)}).
    - Hold passes come out of the existing slack K_i − n_i ∈ {0..4}, so the loop cannot collapse below n_i reductions.
    - R1's window must stay fixed. Weighting R1 by the alignment posterior would create a dodge-the-loss incentive [DERIVED].
- **History / gather.**
  - **Trigger:** H-GATHER (D2(iv)), or DILUTION at n 9–16.
  - Options: SLOT-LOCAL GATHER (U_t[j] = Σ_s α_{j,s} N(V_s[j])), or a descriptive bank = {V_0, V_{t−1}} arm (the CoTFormer recent-1 analogue).
  - The second narrows the pass-global memory, so it is the user's call.
- **Rejected: per-operator routing (c).**
  - The premise fails (E6).
  - Learned routing ≈ frozen-random routing on modular addition (Smithline 2605.09403).
  - Any future routing cell needs a frozen-random-route control."""
s = s[:i0] + new23 + s[i1:]

# --- 3.1 SNAP lemmas after X1
rep("""- *Transfer* to natural versions is not implied; that is the EDIT-ONLY risk, measured by natural-donor VALUE-SWAP [CONJECTURE].""",
"""- *Transfer* to natural versions is not implied; that is the EDIT-ONLY risk, measured by natural-donor VALUE-SWAP [CONJECTURE].

**Lemma SN1 (canonical row-space code)** [PROVED up to the LN fixed-point tolerance, gated as W1].
- At α = 1, every stored entry's R-logit vector is L*(dec V_t), a function of decoded content alone. A reader of the bank through R's row space therefore sees a pass-free, history-free code: S2 restricted to the row space.
- Snap-corrupted entries are identical to natural ones in that subspace. G-EDIT then needs the null space.

**Lemma SN2 (edit size)** [PROVED]. ‖Δ_t[j]‖ ≤ σ_LN·‖L* − L(V_t[j])‖ / σ_min(W_R).
- R is a fresh head with small σ_min at init, so snapping before R is trained would apply large off-manifold edits. Hence the 4k–8k ramp and the clip.
- MCT-type edits on trained R measured ‖Δ‖/‖V‖ ≈ 1e-3 (211-3).

**Not guaranteed** [CONJECTURE]. The null space still drifts, and the step can read it; Altabaa's full snap had no null space. How much of 100 vs 54.8 needs the full snap is unknown.""")

# --- Goodhart
rep("""| G6 counters / clocks (bank size, ALiBi age, VOID count) | Unaffected; both "reduce until done" and count-keyed rules meet COMMUTE. STALL measures it; OFFSET closes the pass / bank part if triggered (content VOID count remains). |
| G8 sabotage / timed release | No negative-likelihood term; targets are sg; Lemma C2. |""",
"""| G6 counters / clocks (bank size, ALiBi age, VOID count) | Unaffected; both "reduce until done" and count-keyed rules meet COMMUTE. STALL measures it; OFFSET closes the pass / bank part if triggered. SELF-SNAP erases row-space clocks only (SN1). |
| G8 sabotage / timed release | No negative-likelihood term; targets are sg (C2). |""")
rep("""| **New: G-SKIP** (early terminal decode) | Closed by the skeleton fallback (C2); `commute_fb` is logged. |""",
"""| **New: G-SKIP / G-FALLBACK** (early terminal decode, or wrong skeletons kept so targets stay free-running) | Skip is closed by the skeleton fallback (C2). Skeleton errors are paid at their own pass. COMMUTE-INACTIVE if `commute_fb` > 0.2 over the last 8k. |
| **New: G-NULL** (S2: information moves to the null space, which SELF-SNAP leaves continuous) | Possible. Descriptive: linear probes of pass index and of the values from the null-space part of stored entries. |""")
rep("""| **New: G-FALLBACK** (keep skeletons wrong so targets stay free-running) | No incentive (skeleton errors are paid at their own pass). Labelled COMMUTE-INACTIVE if `commute_fb` > 0.2 over the last 8k. |
""", "")

# --- labels
rep("""- **COMMUTE-INACTIVE.**""", """- **COMMUTE-INACTIVE.**
- **SNAP-CLIPPED** (S2): clip rate > 0.1 over the last 8k, i.e. SNAP was not exercised as designed.""")

# --- gates
rep("""   - OPX edit gate (W1 on slots l / r, every other slot unchanged ≥ 0.999).""",
"""   - OPX edit gate (W1 on slots l / r, every other slot unchanged ≥ 0.999).
   - SNAP gate (SN1): logits within 0.05 nats of L*(dec) after ≤ 3 iterations on a trained-R checkpoint (a §211 final). Straight-through gradient check: ∂L/∂V_t equals ∂L/∂S_t.""")

# --- stage 1 table
rep("""| S2 (optional) | N2 + COMMUTE only | Is removing the bad tilt enough without OPX? (descriptive separation) | ≈ $0.7–1.7 |

- **Family projection:** about $1.5–3.7. **Cap $4.5**, with a per-cell guard of $2.5.""",
"""| **S2** | S1 + SELF-SNAP (α ramp 4k–8k; OPX as snap corruption) | Does a canonical row-space code add stationarity (STEP-COMPOSES with PASS-KEYED absent; DEPTH descriptive)? | ≈ S1 + 10% (N3's MCT path cost +9% in G-TIME): **$0.9–2.2** |
| S3 (optional) | N2 + COMMUTE only | Is removing the bad tilt enough without OPX? (descriptive separation) | ≈ $0.7–1.7 |

- **Family projection (S1 + S2):** about $1.7–4.2. **Cap $5**, with a per-cell guard of $2.5.
- S1 vs S2 at one seed each is descriptive.""")

# --- what follows
rep("""- **S1 SCR-C1:** seeds 2–3 (the rate claim, about $2–4). Then **S3 = S1 + OFFSET** (O 4; ≤ 25% more passes, about $1–2.5) for C2, with a pre-registered **DEPTH-STATIONARY** criterion: on n 9–12 exact-input events, hazard by t at fixed hh flat to ≤ 0.01 per pass. Then GSM8K transfer, as a separate registration.""",
"""- **S1 or S2 reads SCR-C1:** seeds 2–3 of the better cell (the rate claim, about $2–4).
  - Then that cell + OFFSET (about $1–2.5) for C2, with a pre-registered **DEPTH-STATIONARY** criterion: on n 9–12 exact-input events, hazard by t at fixed hh flat to ≤ 0.01 per pass.
  - GSM8K transfer comes after, as a separate registration.""")

# --- risks
rep("""4. **Anneal confound.** S1 inherits N2's anneal cost (ANNEAL-COSTLY, 211-11) and its diffuse gather.
   - OPX edits every version of the slot, so cross-version mixing at slot j stays consistent. Cross-position mixing (off-self 0.8) may dilute the edits, as in 211-12 (c). D2(iv) reads this.
   - If N1 reads clearly better, the base question goes to the user, since it touches the attention-memory constraint.""",
"""4. **Anneal confound.** S1 / S2 inherit N2's anneal cost (211-11) and its diffuse gather. Cross-position mixing (off-self 0.8) may dilute the edits, as in 211-12 (c); D2(iv) reads this. If N1 reads clearly better, the base is the user's call, since it touches the constraint.
6. **History hurts depth** (CoTFormer, Bohde). Even a cell that meets SCR-C1 may fail DEPTH because the bank holds the full history. *Reading:* DILUTION, or DEPTH < 0.5 with STEP-COMPOSES. The descriptive {V_0, V_{t−1}} arm then needs the user's view on the constraint.
7. **SNAP shifts the manifold** (SN2) or is not exercised (SNAP-CLIPPED). *Falsifier for SNAP as a stationarity lever:* S2 meets STEP-COMPOSES but reads PASS-KEYED as S1 does, with G-NULL probes carrying the pass index.""")

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
