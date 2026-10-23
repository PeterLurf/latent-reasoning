# REVIEW_LAT: red team of LAT_v1_draft.md (ad2a3be4) (2026-10-23)

Reviewer: Theory's red-team subagent. Read-only on the repo, apart from this file and `review_scripts_lat/` (CPU, 2 threads, $0). No blinded file was read.
- **Inputs:** LAT_v1_draft.md, lat_checks.py / .json, OW_draft.md, REVIEW_OW.md; THEORY §184, §207, Notes 226-1 / 226-2 / 227-1 / 227-2 (grep only); boxes/nested170.py (InjectN, aux_steps, AUX_W, train_items), boxes/deltamem.py (wo bias), boxes/ow_words.py, boxes/codi_ps.py (draw_step, CurGate); s184_reweight_s1/s2 args.json, log.jsonl, eval.json (`s170.acc_by_n_kprime`); scratch/gate_s226/pilot_p0d.json, scratch/gate_s227/pilot_text.json; NORTH_STAR C1 / C4.
- **Checks:** `review_scripts_lat/rv_lat.py` → `rv_lat.json`: (A) exposure and trends, (B) back-references by depth, (C) R2 under a per-depth K′ model, (D) R3 under a carried-share mixture and the LEAK-SEG false-fail, (E) d = 3 pairs at t = 1, (F) a FLOP cross-check of s/step.

## Verdict: GO to v1 with fixes (fixes 1–6 are binding before Audit; not STOP)
- The core holds: §184 is the only recipe with a C1 record, the DIRECT-first band is right, the union target plus a matched floor is the right kind of serial reading, and lat_checks reproduces (exposure replay, binomial bounds, kept shares, floors).
- **One rule would fail the design's own positive case.** R2 pools K′1 over a band that starts at d = 2. §184's own C1 models solve n = 2 at K′1 (0.971 / 0.976), so the draft's R2 fails a §184-like serial model at P = 0.05 (issue 1).
- **"Single change of surface" is not true.** The step target loses most of §184's information content (issue 2), and the pilot never exercises a memory back-reference (issue 3).
- No new red team is needed if fixes 1–6 are applied as written and the OC is re-run with rv_lat's scenarios added.

## Answers to the brief, one line each
1. **Single change?** No. There are at least seven other changes, the step target's content being the largest (issue 2).
2. **Exposure / G-W0:** the numbers reproduce (E 6,000 ± 72, 0.1 % bound 5,776; 5,016 = 2 × 2,508). "Exposure-adequate" for P0d is not established. The trend is the evidence, not the count (issue 7).
3. **R3:** no non-S channel can pass it, since STATE-SWAP is single-channel. But a 38 % carried share passes (issue 6), and LEAK-SEG-H guards the wrong thing and can fail a strong carrier (issue 5).
4. **K′0:** it is not a one-pass reference. In §184, K′0 at n = 2 reads 0.15 / 0.17 where §170 DIRECT reads ≥ 0.90. R2 must change (issues 1 and 4).
5. **Cost / futility:** the FLOP cross-check gives 1.15–1.41 s/step cached, not 1.01. The futility stop needs a trend clause (issues 8 and 10).
6. **Pilot:** yes, extend it to U{1..4} with the cloud batch law. d = 2 has 0 % references to a computed state; every d ≥ 3 item has one (issue 3).
7. **Scoping:** C1-only is right. Five phrases need rewording (issue 13).
8. **Simpler:** defer TEXT to Stage 2; run DIRECT in parallel with the pilot; drop the LEAK-SEG / CACHE-SWAP engineering from the binding path (issues 11 and 14).

## Issues, ranked by severity

**1. [HIGH] R2 fails the §184 serial pattern, because K′1 is two sequential reads and B includes d = 2.**
- **What K′ = 1 means in INJECT:** one think pass, then the decoder pass, which re-reads the question and reads S₁. That is two hops of compute.
- **§184 measured this** (`s170.acc_by_n_kprime`): K′1 at n = 2 is **0.971 (s1) / 0.976 (s2)**, K′4 solves n ≤ 5, and K′0 at n = 2 is 0.145 / 0.168. A C1 model solves n ≤ K′ + 1.
- On OW, P0d implies n\* = 1, so B = {2..8}. Then pooled K′1 ≈ (0.97 + 6 × 0.11) / 7 ≈ 0.23, against a bar of DIRECT + 0.10 ≈ 0.22.
- **rv_lat C:** P(R2) under the draft rule is **0.048** for a hop-per-pass serial model, and 1.000 under the fix.
  - The draft's OC misses this because it gives K′0 and K′1 one shared p at every depth.
  - The predicted P(R2 | R1) = 0.85 is therefore wrong under the draft's own rule for the most likely positive mechanism.
- **Fix:**
  - R2 = acc(8) − max(K′0 on B, K′1 on B ∩ {d ≥ 3}) ≥ 0.30, **and** K′0 ≤ DIRECT + 0.10 on B, **and** K′1 ≤ DIRECT + 0.10 on B ∩ {d ≥ 3}. Each comparison is made against DIRECT on the same depths.
  - Descriptive: the per-depth K′ onset (the smallest K′ with acc ≥ 0.5). The hop-per-pass prediction is onset ≈ d − 1, the §184 signature.
  - Re-run the OC with per-depth K′ arrays.

**2. [HIGH] The port changes more than the surface. "A null then speaks about language" is overclaimed.**

Changes vs s184_reweight_s2 besides the token adapters:

| # | change | §184 | LAT-OW |
|---|---|---|---|
| a | **step-target content** | aux_steps = e_{t+1}, the whole partly reduced expression: every pending operand plus the new value (≈ 16 tokens per pass at n = 8) | "⟨S_t⟩ has ⟨v_t⟩ ⟨u⟩." (one value). Back-referenced S_j (j ≤ t − 2, needed by mul_s / add_s) get **no** memory supervision |
| b | computation per hop | bracket-local reduction, operator in place | name-chained premise lookup in a TOPO-shuffled text with 0–2 distractors, givens, 6 frames (some inverted, "A has c fewer than B") |
| c | think length | batch-max 29 tokens | 36 / 72 / 127 at d 1 / 4 / 8: 4–5× more positions write into the same DeltaMem state per pass |
| d | batch law | mixed n per batch (train_items) | one d per optimiser step (A3); interacts with AUX_W (issue 9) |
| e | cache_lower | off (not in args.json) | on |
| f | exclusion / item space | excl_survive n = 1: 0.11 (tiny n = 1 space) | large item space at every d |
| g | decoder suffix / answer | " =" → token | " The answer is:" → " 0".." 10" |
| — | d ~ U{1..8} from step 0 | same as §184 (n_max_train 8, no anneal) | **new for this task**: §226 / §227 both used CurGate, and §227 TEXT passed only under CurGate |

- **Fix:**
  - (i) Delete "the only changes" and "a null then speaks about language". State a NO-GAIN as "§184's args with a single-value step target, one d per step, on TOPO word programs".
  - (ii) Add (a)–(g) as numbered deviations with A-tags.
  - (iii) Pre-name one fallback for (a), and only for a pilot fail at d 3–4 (issue 3): the **cumulative** step target R_step[1..t] (all values so far; ≈ 5t tokens per pass; ≈ +140 aux tokens at d = 8). It is the closest functional analogue of e_{t+1}'s "everything still needed". This is the no-loop rule's single allowed fallback, and it is stated now.

**3. [HIGH] The U{1, 2} pilot never exercises the capacity that B needs.**
- **rv_lat B (TOPO, frames 0–5, 4,000 items per d):**
  - d = 2 has a back-reference on 15 % of items, always to S₀, which is stated in the text. **0 % reference a computed state, and 0 % contain mul_s.**
  - d ≥ 3 has a reference to a computed S_j (j ≥ 1) and a mul_s on **100 %** of items.
  - So LP3 tests two chained steps, but never a value that must be held in memory and multiplied later. B needs ≥ 2 depths ≥ 4.
- **Fix: P-LAT4 / P-DIR4** with d ~ U{1..4}, one d per step (the cloud law), 16k steps. E(d) = 4,000 (0.1 % bound ≈ 3,860) = 1.6× TEXT's d = 2 need, the same as the draft's pilot.
  - LP0–LP3 unchanged.
  - **LP4 (pre-stated, decides the fallback):** d = 3 and d = 4 TEST2 at K′ 8, each ≥ P-DIR4 + 0.15.
    - LP3 pass with LP4 fail → **D2-ONLY**: the single fallback of issue 2 (iii) gets one pilot, never a retune.
    - LP3 fail → LAT-NOT-LEARNABLE-D2, as drafted.
  - **Cost:** ≈ 1.25× per step × 2× steps ≈ 5–8 h on the 4060 (too long for "local short"), or ≈ 4.5 h on a cloud 3090 ≈ $1.0–1.3 (recommended; auto-approvable).
  - §184's mixed-batch setting does not belong in the pilot unless the cloud cell adopts it: the pilot must use the cloud's law to predict it.

**4. [MED-HIGH] K′0 is a zero-memory, off-distribution read, not a one-pass reference.**
- **Evidence:** §184 K′0 at n = 2 is 0.145 / 0.168, while §170's one-pass DIRECT solves n ≤ 2 at ≥ 0.90. A trunk co-trained with the memory collapses without it.
- In §184 the additive state's norm grows ≈ 2k per pass (a pass counter, §184 memnorm), so S = 0 lies far outside the trained range. The b_o point is real but second-order.
- **Consequence:** "≤ DIRECT + 0.10" is met almost automatically, so R2 tests only one direction (whether the trunk computes alone).
- **Fix:**
  - add **MEAN-S₈** (S replaced by the per-d test-set mean of S₈, which is in range and carries no item information) as a binding co-condition: acc(MEAN-S₈) ≤ DIRECT + 0.10 on B;
  - keep K′0-NOBIAS descriptive;
  - word R2 as "zero-memory and mean-state reads fall to the one-pass level", not "removing the passes".

**5. [MED-HIGH] LEAK-SEG-H guards a channel that STATE-SWAP does not have, and it can fail a true carrier.**
- In §221, LEAK-SEG asked whether CACHE-SWAP's latent-position K/V followed because those rows carried copies of the donor's question rows.
- Under INJECT's STATE-SWAP, nothing from the donor reaches the recipient except S, as the draft itself says. So R3 has no leak channel to guard.
- **What LEAK-SEG-H actually measures:** how strongly passes 1..t write S from the final-span contextual rows. Those rows see the whole donor question causally.
  - A carrier that summarises at the last tokens, which is common in GPT-2, can read LEAK-SEG 0.30 while carrying perfectly.
  - **rv_lat D:** a strong carrier (follow 0.75) with leak 0.25 / 0.30 passes R3 at P = **0.50 / 0.00**.
- **Fix:**
  - LEAK-SEG-H becomes descriptive; SELF-SWAP bitwise stays binding.
  - The real ambiguity under a single-channel swap is *what* S_t holds:
    - (i) step values;
    - (ii) a buffer of the donor's premises 1..t, which later passes compute through ("deferred"). This also outputs cf(a), since cf(a) is a function of premises ≤ t.
  - Separate (i) from (ii) by timing. Pre-state a sub-label **STATE-TIMED** = BUDGET-FRONT b\* ≤ d − t + 1 on ≥ 0.5 of followed pairs at d ∈ {7, 8} (no slack for re-computation from scratch); otherwise **DEFERRED**.
  - Only STATE-TIMED licenses "the state after pass t carries the donor's progress". C1-LANG itself does not need it.

**6. [MED] R3 passes when most swaps are ignored.**
- **The algebra:** on own-mode items the output is own, which lies outside U and outside every floor target. So follow = f and floor ≈ 0.17 f, and the 0.30 bar needs f ≥ 0.36.
- **rv_lat D:** P(R3) = 0.88 at f = 0.38 (own 0.62), 0.99 at f = 0.40, 1.00 at f = 0.45. The sub-label OWN (≥ 0.50) is only reachable under DEEP-NOT-SERIAL, so a C1-LANG with 60 % own is possible.
- NORTH_STAR C1 asks for "serial, hop-by-hop use".
- **Fix:**
  - add **follow_U ≥ 0.50** to R3 (P = 0.51 at f = 0.50, 1.00 at f = 0.60);
  - report R3 per depth, with d = 8 (zero slack: K − t = d − t) shown separately;
  - an OWN share concentrated at d 4–5 (6 / 5 spare passes) with following at d 7–8 is a slack story, not §207's scaffold. Pre-state that reading (open question 1).

**7. [MED] P0d's d = 2 fail is not shown to be "exposure-adequate".**
- **Why TEXT's 2,037 is not a requirement:**
  - TEXT supervises every step sentence (denser than answer-only DIRECT), so its count is a lower bound on what DIRECT needs, not an adequacy bar.
  - TEXT's d = 2 crossing at 10k coincided with d = 1's own jump (decode 0.695 → 0.915 at 10k after 8k flat at ≈ 0.7). That was a global transition, not a d = 2 exposure threshold.
- **The real evidence is the trend:**
  - P0d's implied d = 2 is 0.065 / 0.052 / 0.107 / 0.084 / 0.144 / 0.062 over six bins, E(2) 0 → 3,367: flat;
  - TEXT rose 0.10 → 0.29 within 514 d = 2 steps.
- One-pass implicit composition is known to arrive late, after long plateaus (grokking reports; Literature to cite and verify, e.g. Wang et al. "Grokked transformers are implicit reasoners").
- **Fix:**
  - word P0d as "did not learn d = 2 in 12k (E(2) = 3,367; flat over six bins; ≥ 1.3× serial TEXT's d = 2 count, a yardstick, not adequacy)";
  - make DIRECT-STILL-RISING binding per depth: a band depth whose DIRECT train acc rises > 0.10 between the last two 6k windows leaves B;
  - keep 5,016 as a stated convention;
  - word the claim "did not learn at 6,000 steps per depth (48k)".

**8. [MED] The futility stop is plausible, but it lacks a trend clause and has no symmetric "still rising" rule.**
- **§184's own curves** (pooled answer CE): near chance until 8–10k, basin at 14–18k (s1 0.56 at 16k; s2 0.48 at 18k).
  - **s1 sat at 0.45 from 18k to 40k and dropped to 0.09 only at 42–44k.**
  - OW sequences are 4–5× longer.
- A slow learner can therefore be mid-transition at 24k, or still entering its deep depths at 48k.
- **Fix:**
  - stop at 24k only if d = 2 train acc (last 1k bin) < 0.50 **and** it rose < 0.05 over the last 4 bins. Otherwise check once more at 32k, with the same rule and no further extension;
  - add **LAT-STILL-RISING:** band-depth train acc up > 0.10 over the last 6k ⇒ NO-GAIN is worded "at 48k, still rising" and is not a closure. Any extension is a new Note;
  - lat_checks' "cloud_U1..8_at_16k_futility_check" is stale (the draft says 24k; E(2) at 24k = 3,000, 0.1 % bound 2,842).

**9. [MED] A3: one d per step makes the aux / answer ratio swing 6× from step to step.**
- **The swing:** a step's aux weight is mean_{t ≤ d} AUX_W[t] = 0.368 (d = 1), 0.475 (d = 2), 0.726 (d = 4) and **2.31 (d = 8)**. In §184's mixed batches, the per-item mean averaged this to a constant.
- Deep steps are aux-dominated and are the ones the 1.0 clip binds on (w₈ = 8). Adam's moments see a d-alternating gradient mix.
- Mixed batches would restore §184's law, but padding to the batch max (≈ 150 vs a mean of 79) costs ≈ +60 % of LAT. Per-d micro-buckets make the launch-bound loop slower.
- **Fix:**
  - keep one d per step for cost, but list it as deviation (d) of issue 2;
  - log clip rate and pre-clip gn by d;
  - name A3 in NO-GAIN's wording;
  - if F2 shows padded mixed batches within the cap, prefer them (the single-factor port).

**10. [MED] The cost uses the faster of two §184 hosts and a K + 2 pass count.**
- **FLOP model** (6 × non-embedding params × tokens; decoder and aux run the full trunk, as in nested170): §184 implies 14.2 effective TFLOPs. Projected LAT-OW cached = **1.15 s/step (s2's host) / 1.41 s/step (s1's host)**, vs the draft's 1.01; uncached 1.79 / 2.18.
- At s1's host: 18.8 h ≈ **$4.6 with eval, ≈ $8.7 at 2× host variance**, above the LAT share of $7.5. Stage 1 can then exceed $12.
- (The draft's work model counts K + 1 think passes plus the decoder in both numerator and denominator. That is harmless for the ratio, but should be fixed.)
- **Fix:**
  - set the LAT share to $9 and keep F2 × 1.3 admission;
  - moving TEXT to Stage 2 (issue 11) keeps Stage 1 ≤ $12, and the family stays ≤ $22.

**11. [MED-LOW] TEXT at U{1..8} from step 0 is an untested recipe, and C3 is descriptive.**
- §227 passed under CurGate. Its d = 1 decode took 10k steps even with a d = 1-only stage.
- **Fix:** run TEXT only in Stage 2 (on C1-LANG), where C3 matters, under §227's CurGate (exposure difference disclosed, C3 only). This saves $2.5 and the ow_text_eval build in the ≈ 90 % non-C1 outcomes.

**12. [LOW] cache_lower gates.**
- §184 ran uncached, so "port-kernel identity" must be uncached-bitwise against nested170 `--arm inject`, followed by the fp64 cached = uncached gate on OW shapes.
- Add a 200-step bf16 loss-trajectory comparison of cached vs uncached on OW, as a descriptive drift magnitude (the backward accumulation order differs).
- Also replicate `s170_init.json`'s g_add init path.

**13. [LOW] Wording of the C1 sentence (C1-only scoping is right).**
- "depths that one pass cannot learn at equal exposure" → "depths a one-pass model trained on answers alone did not learn at equal per-depth exposure (6,000 steps per depth, 48k)".
- "Removing the latent passes" → "zero-memory, one-pass-budget and mean-state reads".
- "the answer follows the donor-state counterfactual" → "on ≥ ½ of kept pairs, ≥ 0.30 above a skeleton-matched floor".
- Disclose in the sentence that LAT also receives generator step-state targets, which DIRECT does not. R1 compares recipes; R2 / R3 carry the causal claim. That disclosure also keeps C4 honest.
- The prediction product is ≈ 0.08, not 0.10.

**14. [LOW] Small additions.**
- **d = 3 is not degenerate at t = 1:** 1,642 / 3,000 kept (0.547; rv_lat E). Add d = 3 at t = 1, so the "≥ 2 band depths ≥ 4" pair condition rarely binds.
- **Sequencing (open question 10):** launch cloud DIRECT in parallel with P-LAT4. Both gate LAT, so the parallel launch costs nothing extra and saves ≈ 6 h.
- CACHE-SWAP(K) and LEAK-SEG-H stay as descriptive eval parts. Do not let their engineering block the binding path.

## Mechanism-null OC (rv_lat; adds to lat_checks E)

| truth | draft rule | with fixes 1, 4, 6 |
|---|---|---|
| hop-per-pass serial (§184 pattern: K′1 solves d = 2) | P(R2) **0.05** → C1-LANG ≈ 0.05 | P(R2) 1.00 |
| strong carrier with final-span-sensitive writes (LEAK-SEG 0.30) | P(R3) **0.00** | P(R3) 1.00 (LEAK-SEG descriptive) |
| partial carrier, 38 % of swaps followed (62 % own) | P(R3) 0.88 | 0.00 |
| partial carrier, 60 % followed | 1.00 | 1.00 |
| two hops per pass (K′1 solves d ≤ 3) | R2 0.00 | 0.00 (correct: one-pass-budget reads are not removed) |
| deferred premise buffer (outputs cf(a)) | passes R3 | passes R3; sub-label DEFERRED via BUDGET-FRONT (issue 5) |

- Other mechanism nulls (pause, relay, §207 scaffold, floor collapse) remain 0, as in lat_checks.

## Novelty (Literature's addendum in LIT_ONEPASS_DEPTH_THRESHOLD, folded in)
- **N1: NARROW.** A combination only; no "new" or "first".
- **N2: NARROW.** The cf(a) ∪ cf_b union and the skeleton-matched floor extend SCIT 2608.27265's oracle counterfactual. Present them as design choices.
- **N3 (a):** no exposure-matched one-pass vs serial comparison was found. Call it a design property, not a contribution.
- **N3 (b):** no delta-rule / recurrent-memory latent loop tested for causal serial use on templated word problems was found. AGCLR 2606.07720 is a single gated vector with no memory ablation. Word it as "we did not find", never "first".
- **Literature to verify:** the grokking / implicit-composition citation in issue 7.

## Files
- `review_scripts_lat/rv_lat.py` → `rv_lat.json` (CPU, 2 threads, seeds 23101–23105, review-only).
