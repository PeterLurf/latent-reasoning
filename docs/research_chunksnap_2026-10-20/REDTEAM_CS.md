# REDTEAM_CS: red team of CHUNK-SNAP (CS_draft.md at 8cd29919)

Written 2026-10-20 ~22:50 EDT by a red-team subagent for Theory. The work was read-only: the draft, NORTH_STAR, THEORY (§201 / 201-7 / T4 result, 212-7 / 212-9, 213-35..39, §216 result, 216-5..11, §217), boxes/vbank_diag.py (part_rs, long_step, part_chainlabel(_fresh), part_steppos), boxes/vbank.py (rollout, rerender, batch_x), boxes/nested170.py, and the committed eval JSONs of the five named models (eval_rs, eval_chainlabel(_fresh), eval_deep, eval_steppos). Nothing under lvb_* or the sealed set was read. No GPU or scorer runs.

**Verdict.** Do not register as drafted.
- The draft's premise, "nothing between k = 1 and k = ∞ has been measured", is wrong. Committed **eval_rs** already holds the first-chunk surface e(m, k) for four of the five models at m ≤ 16, plus two-chunk restarts. That data already rules out both pure hypotheses (F1).
- The primary reading nearly passes on DEEP with k = 1 alone (F2).
- The mechanism reading is confounded by token position and by decodability (F3).
- A slim descriptive version is worth ~1 card-hour after SCALE-PROBE (F17).

---

## BLOCKERS

**F1. BLOCKER: e(m, k) is already measured (eval_rs), and it already rules out both pure H_pass and pure H_size. The §5 priors ignore it.**
- **The object is the same.** part_rs decodes argmax R(V_t) on the uninterrupted rollout at t = n − r, and scores it exact iff it equals cot_steps[t].
  - V_t is causal in t, so this is exactly the draft's first-chunk e(n, t): a t-pass rollout on a fresh size-n input, decoded and compared with reduce_t.
  - RESTART(n, r) is a two-chunk CHUNK: a checkpoint at t, then an r-pass chunk on the re-rooted item.
- **Committed exact_rate = e(m, k)**, 1,000 items per cell, 3070, fp32, bs 32:

| model (support) | k | e at m = (support − 2 … support + 6) |
|---|---|---|
| D3-T s1 (8) | 4 | m 6 1.000, 8 1.000, 10 0.962, 12 0.927 |
| | 6 | m 8 1.000, 10 0.606, 12 0.552, 14 0.536 |
| | 8 | m 10 0.046, 12 0.021, 14 0.014, 16 0.010 |
| FT-SELF (12) | 6 | m 12 0.996, 14 0.968 |
| | 8 | m 10 1.000, 12 0.989, 14 0.863, 16 0.695 |
| | 10 | m 12 0.993, 14 0.590, 16 0.298 |
| R1-s2 (12) | 8 | m 12 0.991, 14 0.868, 16 0.739 |
| R2-s1 (16) | 8–14 | all m ≤ 16 in support: 0.973–1.000 |

- **Pure H_size (D3: survival keyed to live size m − i) is falsified.**
  - D3-T at m 14: e(14, 8) / e(14, 6) = 0.014 / 0.536 = 0.026. Passes 7–8 run at live sizes 8 and 7, which are *in support*, yet they fail 97 % of the time.
  - FT-SELF at m 16: e(16, 10) / e(16, 8) = 0.43.
  - So survival depends on the pass index and/or the chunk's starting size or layout, not on the live size alone.
- **Pure H_pass (e ≥ 0.99 for all m whenever k ≤ k*) is falsified at fixed k in every model with m > support.**
  - D3-T k = 4: 1.000 → 0.962 → 0.927 from m 8 to 12.
  - FT-SELF k = 8: 0.989 → 0.695 from m 12 to 16.
- **What is measured is H_mixed:** a step at the support edge, then a gentle slope in m, with a steep, start-size-dependent fall in k.
- **R2-s1 at k = 1 is already size-keyed past support** (committed fresh chain, rng 21337).
  - precision_strict n 16 → 24 is 0.991 → 0.890, so the mean per-step survival over m 17–24 is ≈ 0.987.
  - In support (n 9 → 16) it is ≈ 0.9989, so the error rate is ≈ 12× higher past support.
  - D5's "per-pass survival at k = 1 is size-robust to ~2× support" overstates this. D3-T long_step at m 15–16 is 0.982 / 0.975.
- **Consequences:**
  - P(H_pass at k = 4) ≈ 0.55 and P(H_pass at k = 8) ≈ 0.40 are unsupported.
  - Scaled by support (n50 / support ≈ 1.28 is constant across rounds, 216-9), the D3-T analogue of R2-s1 at k = 8 on FAR (k / support 0.5, m / support ≥ 1.56) is e(12, 4) = 0.927. That is already below the SIZE-FREE-CHUNK bar.
  - **CHUNK-C2-BASE (D3-T s1, k = 4, ≥ 0.90 at n 17–24) is a known fail.** The best case at n 17 is ≈ e(17, 4) · e(13, 4) · e(9, 4) ≲ 0.93 · 0.9 · 0.97 ≈ 0.8, and its own k = 1 chain is 0.78–0.89 at n 17–20.
- **Fix:**
  - Cite eval_rs in §0 and D5, and delete "nothing in between has been measured".
  - Re-derive the priors from the table above. My estimates: P(SIZE-FREE-CHUNK, R2-s1, k = 4) ≈ 0.10–0.15; at k = 8 ≈ 0.05; P(CHUNK-C2) ≈ 0.05–0.10.
  - Add the hypothesis the data point to, **H_start**: survival keyed to (pass index, chunk-start size / slot layout). VB keeps the original slot layout with VOIDs inside a chunk (vbank.rerender drops VOIDs only at decode), while a restart compacts it.
  - Re-scope the novel part to what eval_rs cannot give: R2-s1 at m 17–40, multi-chunk chains, and the FAR sizes.
  - Add a gate: CHUNK's first-chunk e(n, t) on IN n 9–16 must reproduce eval_rs exact_rate at the same (n, t) within sampling (500 is a subset of the 1,000).

**F2. BLOCKER: CHUNK-C2 is nearly passed by k = 1 on DEEP, and it is not a C2 claim.**
- **R2-s1's committed k = 1 chain** (fresh items; FRESH-MATCHES-TEST holds within 0.02) reads precision_strict n 17–24 = 0.992 / 0.976 / 0.973 / 0.954 / 0.944 / 0.905 / **0.893 / 0.890**.
  - Only n 23–24 miss the 0.90 bar, by 0.007–0.010, under one SE at 500 items (0.014).
  - The DEEP half therefore cannot discriminate. Any k that removes some k = 1 decode losses passes it whatever the mechanism. The claim rides entirely on FAR.
- **"Solves n 17–40 with re-rooted checkpoints" is the k = 1 labeller's capability** (213-35 / 213-38 / LABELLER-KEPT-2). What CHUNK can add is only that (k − 1)/k of the writes can be skipped at no loss.
- **Not C2.** NORTH_STAR C2 means accuracy *rises with latent steps* past the trained budget. CHUNK has no budget axis: every item gets exactly n latent passes, and k only changes how often the model writes.
  - "C2 (hybrid)" is a C5-with-fixed-k reach claim plus an efficiency claim.
- **Fix:**
  - Rename the reading CHUNK-REACH, never "C2".
  - Make the primary **paired and relative**: CHUNK(k) − CHUNK(1) on the same items per FAR n (McNemar), with the absolute bars kept as a conjunct. Pre-state the null "latent passes add nothing over writing every step" (CHUNK(k) ≤ CHUNK(1)) and its OC.
  - The wording must say the k = 1 chain's reach is the baseline.

**F3. BLOCKER: SIZE-FREE-CHUNK cannot attribute a failure to operator count. Position and decodability are confounded with m.**
- **Position.**
  - Rendered length T is collinear with m.
  - FAR T is ≈ 85–135, while the trained T is ≤ 61 (T_train, §216) and the largest T any chain has been read at is ≈ 81 (n 24).
  - D3-T-family tables train wpe, so rows past the training lengths are untrained.
  - R2-s1's own eval_steppos shows absolute position is load-bearing: **POS_OFFSET 8 → exact 0.000 at every m 1–8**, keep0 ≤ 0.027.
  - So a FAR drop may be positional, not size-keyed. NOT-SIZE-KEYED-2 compared T ≤ 61 vs > 61 at n 21 only, inside the tested range.
- **Decodability.**
  - e is a *decode* exact rate, and the decode is a trained reader (READY-RECODE targets). Off-range it can fail while the latent is right.
  - D3-T at n 9: latent answer 0.923, but e(9, 7) = 0.501, with 36 % of the decodes at t = 7 unparseable.
  - So "SIZE-FREE-CHUNK false" can mean "the reader is size-keyed", not "per-pass latent accuracy is size-keyed". This is the decode-faithfulness-out-of-range lesson.
- **Tree shape.** At fixed m, FAR trees have deeper bracket nesting and later leftmost-redex positions than anything trained. That is a second source of positional OOD.
- **Fix:**
  - Report e(m, k) by (m, T) and by bracket depth (long_step already has a by-T mode).
  - Read SIZE-vs-LENGTH by regression on the overlap: bracket count varies T at fixed m, as in part_steppos B. Pre-state that T > 81 is an untested positional region.
  - Split every inexact chunk into unparseable / ops-mismatch / parseable-wrong (copy, compute, locate; see F11).
  - Add **DECODE-LOST** at m ≤ the latent horizon: P(chunk-1 decode inexact | the uninterrupted latent answer is correct).
  - Change the TRUE wording to "the first k passes are size- and length-robust to m 40". "The latent limit is a pass-horizon" does not follow (see F6). FALSE then reads "size-, length- or decode-keyed (unresolved)" unless the splits resolve it.

## MAJOR

**F4. MAJOR: the EDIT arm passes by construction. It is not a C7 edit test.**
- By D1, after a checkpoint the model sees *only* the text. So EDIT-FOLLOW is just CHUNK(4) precision on the edited size-(n − 4) problem, ≈ 0.99 at n 13–16.
  - "≥ 0.9 follow would be the first behavioural edit row on VB" is guaranteed.
  - Audit C7 rule (d) says exactly this: only restarts that drop text or carry non-text state are informative, and CHUNK carries text only.
  - It is also identical to "edit the input instead". When the edited leaf is a copied original leaf, it equals editing the input and re-running.
- **Value-neutral edits.** In Z11, an edit under a "· 0" ancestor leaves the root value unchanged, so FOLLOW and OWN coincide on those items.
- **Rule (c)'s input-only control** does not bear on EDIT-FOLLOW. It bears on whether the checkpoint text comes from the latent passes; see F15 (STALL-in-chunk).
- **Minimal fix:**
  - (i) Rename the arm TEXT-SUFFICIENCY and treat it as a D1 implementation check: per-item label agreement between "continue from edited checkpoint" and "fresh CHUNK(4) on the same text" must be 1.000. It is an assertion, not a reading.
  - (ii) Restrict to value-changing edits and report the neutral share.
  - (iii) Split edits on copied leaves from edits on computed values.
  - (iv) Drop every C7 claim. A real C7 edit needs the non-text state carried: substitute an encoding of the edited text into the *ongoing* rollout (the rollout edit / ver_sub hooks exist), in V_k's slot-aligned layout. That needs its own design and derivation. Until then, VALUE-SWAP (§217) is VB's interchange-intervention row.

**F5. MAJOR: the "round trip by construction" (C7) and the C3 framing are wrong.**
- **Round trip.** C7's round trip asks that behaviour be *unchanged* after decode → re-encode → substitute.
  - Wherever CHUNK beats the latent rollout, behaviour changed. part_rs already shows RESTART − INTACT up to +0.87 on D3-T, and 212-9 shows +0.83 on N1.
  - In support both are ≈ 1.0, so agreement there is trivial.
  - **Fix:** report agreement of CHUNK(k) with the latent label, split by whether the latent is correct, and make no C7 claim.
- **C3.**
  - TS CTX-BOUND is a property of *accumulated* text CoT, which writes ≈ 1.5 n² rows in context.
  - CHUNK is Markov. The matched text comparator is the Markov text chain, which is CHUNK(1), the same model writing every step. It is context-feasible at every n.
  - So CTX-BOUND gives CHUNK no advantage over text.
- **"≥ 75 % silent".**
  - It is true for reductions only if the final leaf is excluded. At k 4, n 17, counting the final leaf gives 71 %.
  - In tokens, each checkpoint rewrites the whole remainder, so written tokens grow as O(n² / k). At n 40, k 4 that is ≈ 550 tokens, more than a compact "a op b = c" CoT (≈ 200).
  - **Fix:** define the silent share; put written tokens next to it in the CHUNK wording; compare CHUNK(k) tokens with CHUNK(1)'s and with s170_cot's context rows; never call it a C3 claim.

**F6. MAJOR: the first chunk does not identify the latent rollout's limit (Q3).**
- **What the first chunk is.** Chunk 1 equals passes 1..k of the latent rollout, the same computation as eval_rs.
- **How later chunks differ.** They start fresh on a *compacted* input. The latent rollout's late passes run on the original slot layout, with VOIDs, a bank of t entries and history.
  - So first-chunk e speaks to the first k passes at size m. It says nothing about why the latent fails at n ≥ 21 (pass ≈ 18+).
  - RS already shows that from the same exact state, restart gives 1.000 while intact continuation gives 0.09–0.36 (D3-T / FT-SELF). That makes the latent limit history- or layout-keyed, which first chunks never exercise.
- **Re-rooted items are not generator draws** (A1): the left side is collapsed and the brackets and leftmost-redex position differ.
- **Fix:**
  - Report re-rooted e_rr(m, k) per cell: chunk j ≥ 2 exactness given an exact arrival, by m. Test A1 per cell, not only through the MARKOV-PRODUCT aggregate.
  - Reword SIZE-FREE-CHUNK TRUE as in F3.

**F7. MAJOR: §201 Prop 8 predicts *bounded* reach, and the draft cites it as the motivation without reconciling.**
- Prop 8 (PROVED, given two horizons): a checkpoint at pass k needs n − k ≤ H_W, where the write is reliable. "More checkpoints do not extend" reach past H_L + H_W.
- In VB, H_W is the decode of a size-(m − k) remainder. The FAR n 40 test is mainly a test of H_W ≥ 36, and R2-s1's k = 1 data (≈ 0.987 per step at m 18–24) already suggest H_W is size-keyed.
- §0's H_pass ("accuracy rises with the number of chunks without bound") also contradicts D2. The product is non-increasing in the number of chunks; "flat in n" is the best case.
- **Fix:** add a D-item that states Prop 8's prediction under a size-keyed write and the condition under which CHUNK escapes it (a size-free write). Fix the H_pass wording.

**F8. MAJOR: the OC is mis-specified, and one number is wrong.**
- **P(pass | H_size) ≈ 0.03 is not derived.**
  - Under H_size, k = 1 is not the "best case". D3 itself says k changes only the decode factor, so k = 4 removes up to 3/4 of the decode losses.
  - DEEP then plausibly passes (F2). FAR passes at n 25–28 under a flat 0.987 per step (≈ 0.88 / 0.85), and n 32 is borderline.
  - So the conjunction is decided by n 36–40 and by position. It is not near zero by construction.
  - The meaningful null is CHUNK(k) = CHUNK(1) (F2).
- **P(pass | effect) ≈ 0.75 ignores** the positional OOD (POS_OFFSET 8 → 0) and the measured decode drift.
- **Arithmetic.** At true 0.92 per DEEP n with 500 items, SE is 0.0121 and the bar is 1.65 SE away, so P(each) ≈ 0.95 and P(all 8) ≈ **0.66**, not 0.85. (0.98 per n would need p ≈ 0.925.)
- **Multiplicity.** "k = 8 or k = 4" is two looks. State the OC of the disjunction.
- **Fix:** recompute the OC from eval_rs-based priors (F1), with H_start and the positional term, and add the realised $ ($0) line.

**F9. MAJOR: the gates are partly vacuous, undefined, or misdescribed.**
- **Ceiling gate.** R2-s1's committed test chain at n 9–12 is 0.996–1.000, and FT-SELF's is 0.998–1.000. A ±0.01 gate at the ceiling cannot detect card drift where it matters (n ≥ 20): the control is not exposed.
  - Gate R2-s1 on its committed **fresh n 17–24** rows (rng 21337 is reproducible), where chains fail 1–11 %.
- **Misstated provenance.** "For R2-s1, FT-SELF and R1-s2, whose committed chain rows are on fresh items" is wrong: all three also have test-set eval_chainlabel.json (3070, bs 32).
- **D3-T s2 has no eval_chainlabel, no eval_rs and no eval_deep.** D3-T s2 and R1-s2 both lack eval_deep.
  - So D3-T s2's gate is undefined, and the k = ∞ DEEP reference for both is missing, which reads pending forever under the 215-7 rule.
  - Gate D3-T s2 on its fresh rows. Run the latent reference fresh on the 4060 for every model; that is cheap and also fixes the cross-card comparison in CHUNK-BEATS-LATENT.
- **G-C1 "bitwise labels" will not hold.** Step 4's ops-count FAIL differs from part_chainlabel, which has no ops check: a chain with a 0- and a 2-reduction step can still finish in n steps.
  - Compare the per-item precision_strict set instead, at **eval_bs 32**, the committed setting.
- **Add the eval_rs reproduction gate** (F1).

**F10. MAJOR: FAR touches the sealed n 25–30 range (Q2).**
- Item-level exclusion is fine. But 216-5 seals n 25–30 as a round-3 extrapolation set, and the draft reads R2-s1 (the round-3 parent) at n 25 and 28, including a latent K′ = n + 2 reading.
  - That is a distribution-level preview of exactly what a round-3 design would tune on.
- The exclusion against pools and against test n ≤ 24 is vacuous anyway: a different n gives a different operator count, so an exact rendered match is impossible. Only the sealed set can collide.
- **Fix:** FAR at **n ∈ {31, 34, 37, 40}** keeps four points and avoids the sealed n range entirely. Or keep 25 / 28 only with an explicit Audit ruling, plus a pre-stated line that any round-3 OC treats these readings as known.

**F11. MAJOR: the rows that would set the next design are missing.** All of them are free from the same rollouts, because rollout returns every V_i.
- **Per-pass decodes inside every chunk** (R(V_i), i = 1..k_eff): the first inexact pass index within the chunk. This gives s(m, i) directly, which is H_mixed's surface and the pass-index-within-chunk question. Store canonical and non-canonical classes separately; D3-T shows off-trajectory but correct latents.
- **Error taxonomy per failing chunk:** unparseable / ops-mismatch / copy error (the unreduced region differs) / compute error (wrong reduced value) / locate error (wrong redex or order), plus the token position of the first difference. Positions past T_train are the positional signature. Literature's lesson from He 2512.02677 and NRS 2402.17407 is that errors compound in locate and copy.
- **Adaptive-trigger proxies (C5 headroom):**
  - ORACLE-k: the best per-item boundary given gold, an upper bound.
  - CONF-TRIGGER: write at the last pass whose decode parses with ops = ops(cur) − i and min per-slot max-prob ≥ τ (τ fixed on IN n ≤ 8). Label-free.
  - The AUROC of decode confidence for chunk exactness.
  - Without these, the draft cannot say whether a learned trigger, its stated next design, has headroom over fixed k.
- **CHUNK(k) − CHUNK(1), paired, per n** (F2).
- **Per-item storage** of every chain: labels, per-chunk strings, per-pass classes, confidence (store-items rule). The share gates (silent share, CHUNK bars) must be recomputable by Audit.

## MINOR

**F12. MINOR: the SIZE-FREE-CHUNK baseline is biased (Q4).**
- The in-support mean over m 4..support includes m < k cells, where k_eff = m < k, so it is not at fixed k. That inflates the baseline.
- min-of-5 FAR cells against a mean adds ≈ 1 SE of downward bias.
- **Fix:** use cells with k_eff = k only, plus a slope test of e on m (and on T, F3) with a bootstrap CI, pre-stated. Undefined when k > support.

**F13. MINOR: the MARKOV-PRODUCT tolerance of 0.02 is below its sampling SD at small k.**
- At k 1, n 24 the product of 24 factors from 500-item cells has SE ≈ 0.02, plus the chain's own 0.014.
- **Fix:** a bootstrap CI over items (cells and chain resampled jointly), plus the per-cell A1 check (F6).

**F14. MINOR: the latent reference at FAR runs K′ = n + 2 ≤ 42, past the LENGTH-FREE range (24).**
- "Every model is LENGTH-FREE to K′ 24" is also false for R1-s2 (a_24(1) = 0.921, FT-DIVERGED).
- **Fix:** report K′ = min(n + 2, 24) beside n + 2. Exclude R1-s2 from any k*-vs-support trend; descriptive evaluation of the censored cell is otherwise fine.

**F15. MINOR: C1 inside chunks at FAR sizes is untested.** §217's band is n 9–18.
- **Fix:** add STALL-in-chunk (V_t := V_{t−1} at one pass of chunk 1). The decode should then equal reduce_{k−1}. This also serves as rule (c)'s control that the checkpoint comes from the passes, not the input.

**F16. MINOR: smaller fixes.**
- (a) Step 5's parse → render canonicalises the decode, a parser-side repair. State it, and count chains it rescues (precision can then exceed chain_exact).
- (b) D1 is PROVED only given batch-composition invariance [MEASURED: BS-CHECK, 0 flips at T ≤ 81]. Add a BS-CHECK at FAR lengths (bs 16 vs 8, n 40, k 4).
- (c) G-C2's stub is perfect only. Add an error-injecting stub (failing at a known pass and size) to test the FAIL accounting and the per-pass index fields.
- (d) The cap ⌈n/k⌉ + 4 never binds under the ops check. Harmless.
- (e) The cost is overstated: committed sec_per_item (0.015–0.025 s for a full chain at bs 32) puts it at ≈ 1 h per model.

## Ranking against the alternatives (Q5; north-star value, C1 first)

1. **SCALE-PROBE** ($0, local, 1B bf16).
   - C1 on language at scale, the project's open gap. No published question-swap / donor test exists at ≥ 1B (LIT_SCALEPROBE).
   - It decides whether the language line moves to a 1B base. Highest value per card-hour.
2. **MUX matched replication** (~$9). A single-change C1 method test (per-latent local loss → portable state?) on GPT-2 GSM8K.
   - **Run it only after SCALE-PROBE's $0 MUX-1B contrast reads.** If MUX-1B shows donor follow where CODI-1B does not, the $9 replication is a sharp test; if not, drop it.
3. **KL-DRIFT** (216-10 / 216-11, $0, < 30 min per model). A C6 instance that picks the round-3 lever. It is already pre-stated and OC'd, so it does not compete: run it.
4. **CHUNK-SNAP as drafted.**
   - Its primary is non-discriminating on DEEP and mislabelled C2. Its mechanism reading is confounded, and is largely answered already by eval_rs (H_mixed / H_start).
   - Its stated payoff (a learned write trigger) is C5, priority 6 of 7.
   - **Worth keeping only as a slim $0 descriptive row** after SCALE-PROBE frees the 4060:
     - R2-s1 only, with D3-T / FT-SELF taken from committed eval_rs;
     - e(m, k) with per-pass decodes at m 17–40 (FAR n 31–40), stratified by T;
     - CHUNK(1) vs CHUNK(4 / 8), paired;
     - error taxonomy and CONF-TRIGGER / ORACLE-k;
     - no C2 / C3 / C7 claims.

**F17 (summary of the scope fix).** Register the slim version above, or nothing. If kept, the questions it can still answer are:
- (a) Does R2-s1's first-k-pass accuracy extend past m 16 like D3-T / FT-SELF's? The predicted answer is a support-edge step.
- (b) Does skipping writes cost accuracy beyond support? Predicted: yes; best k = 1.
- (c) Is there trigger headroom (ORACLE-k − fixed k)?

Only (c) is genuinely open.
