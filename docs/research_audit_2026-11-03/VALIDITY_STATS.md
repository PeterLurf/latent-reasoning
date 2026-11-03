# Validity + statistics audit of §251–§265 (2026-11-03, Audit research subagent; read-only)

Scope: docs/NORTH_STAR.md, DIGEST rows 251–260, scorers score_ds / bu / c5c / gbf / of / dd (+ their helpers score_hx / lh / ao), the eval module boxes/inject_aw_eval.py, run args and result JSONs. No new data.

## 1. Statistical power

**Item noise (n = 1,000 per depth, binomial SE):** p 0.50 → SE 0.016 (95 % ±0.031); p 0.90 → 0.009 (±0.019); p 0.96 → 0.006; p 0.98 → 0.004. A difference of two independent cells near 0.95 has SE ≈ 0.010. So:

- Bars with margin ≥ 0.05 (C2 retention `d6 >= ref - 0.05`, score_bu.py:45, score_of.py:42; AMBITION `+0.10`, score_ds.py:47; NOT-DECODER `<= 0.20` vs chance 0.09, score_ds.py:44) are 5–6 SE wide: safe.
- **0.02 bars are ~2 SE** when the two accuracies come from different draws: C6 hold `acc[128] >= acc[8] - 0.02` (score_hx.py:52), C5 `acc_halt >= acc_k8 - 0.02` (score_hx.py:35). They are paired on the same items (discordant-pair SE ≈ 0.005–0.008), so a 0.028 miss (bun_s2, r = 2, d 6, Note 259-8) is real but tiny; the r = 2 → r = 3 decision turned on a ~3-SE effect.
- **Sub-cells are smaller than 1,000:** `min(by_pos) >= 0.80` (score_ds.py:41, score_gbf.py:38) splits 1,000 items over positions (~330–500 each, SE ≈ 0.02 at 0.8); SWAP uses 1,000 pairs (inject_aw_eval.py:197). A true 0.82 model fails by_pos ~15 % of the time.
- **§251 s2 SWAP4 0.461 vs bar 0.50** (Note 251-3, THEORY.md:23141): Wilson 95 % [0.430, 0.492]; 2.5 SE below the bar, but the bar is a convention, and the first-match label (score_ds.py:50-52 pattern; score_ae) renamed the seed EXTEND-D3-ONLY on that single gate while 3/4 d-4 gates passed.

**Seed noise.** 2/2 gives a one-sided 95 % lower bound on the per-seed pass probability of 0.05^(1/2) = **0.22**; it cannot separate a 50 % lottery (P(2/2) = 0.25) from reliability. §255 showed this empirically: §251's 2/2 d-4 accuracy became 2/4, the full causal bar 1/4 (Note 255-1, THEORY.md:23463). For a lower bound ≥ 0.5 you need 5/5; ≥ 0.7 needs 9/9. SEED-VAR `|s1 - s2| <= 0.05` (score_bu.py:66-73) is not a variance estimate with n = 2.

**Worse: the seeds are not independent.** Every run in §251–§265 (aex_s1–s4, rcb_plaw4_s1/s2, dis, dch, bun, gb) has `init_model = aw_plaw2_s1` and `init_trunk = s233_askname_s1` (boxes/runs_fsq/rcb_plaw4_s2/args.json, aex_s3/args.json, bun_s2/args.json). "Seed" varies only the last fine-tune RNG; one acquisition event (the P-2 seed) underlies every "2/2". All train and eval on RTX 3090 (args train_gpu), so no cross-card check either; memory notes 4090 vs 3090 flips basins.

**Recommendation.** (a) Every eval cell writes Wilson 95 % intervals and the scorer records `margin_se = (value - bar) / SE`, labelling |margin| < 2 as NEAR rather than a clean pass/fail. (b) "Reproducible" / C-status "met" needs ≥ 4 fine-tune seeds from ≥ 2 independent P-2 seed models, reported as k/n with Clopper–Pearson; 2/2 is "replicated once". (c) Seed-level summaries as min / median, not |s1 − s2|.

## 2. Recurring confounds

1. **One lineage** (above). Also the comparator for C2 retention is the same lineage's parent (dch_sK → bun_sK, score_bu.py:24), so lineage-specific quirks cancel in differences but inflate absolute claims.
2. **Repeated-cue measurement = training objective.** The hold penalty is trained on passes where the cue repeats S_d, and C6 / C5 are measured on exactly those passes (§264 registration, THEORY.md:19873-19876; Theory's own OC puts P(C5 PLAN-CLOCKED) ≈ 0.7). Trained readings are artefacts until §264 lands.
3. **Eval overlap at d 1–4.** TEST2 is excluded from training by program signature (`sig = " ".join(syms)`, boxes/inject_ow.py:77), good; but the C5 FRESH draw (Note 259-11 scope) is *not* training-excluded at d 1–4, and nobody has counted the d 1–2 stratum size vs 1,000 draws (memory: small strata).
4. **Many descriptive looks per section.** §251 alone has 16 numbered notes; §264 pre-states a 3 × 3 contrast-decode grid. Pre-state/ack discipline is good, but no section reports how many reads it made, so a lead's effective p-value is unknown.
5. **Post-hoc rule choice.** C5's r was registered as {1, 2} (score_c5c.py:35), r = 3 was chosen after seeing s2's r = 2 miss (Note 259-9, labelled post-hoc), then confirmed on a fresh *item* draw of the *same two models* (score_c5c.py:89; Note 259-12). The fresh draw removes item noise, not the fact that r was tuned to these models' flicker rate. Mitigation already present: gb_s1 / gb_s2 give r = 3 C5-EXTRAP descriptively (score_gbf.json c5_r3_desc), a genuine out-of-sample check that should be cited as such.
6. **Bar drift between sections.** C1 d 1–4 bar is 0.98 in ds / bu / of / dd but 0.95 in gbf (score_gbf.py:36); C2 is absolute 0.80 in gbf (score_gbf.py:39) and relative (ref − 0.05) elsewhere; §265's pilot gate averages over seeds (score_dd.py:53-54) rather than per-seed. Each is defensible, none is tabulated.
7. **Conjunction labels.** BUNDLE needs ~14 sub-gates per seed (score_bu.py:41-49). At 2-SE margins a spurious single-gate fail is likely, and the first-match naming (score_ds.py:50-52) turns one near-miss into a differently named category (§251 s2).

## 3. Construct validity of the C1–C7 operationalisations

- **C1 "thoughts do the work"** = NOT-DECODER (d 4 acc at K′2 ≤ 0.20), SWAP-mid at t = 2 (follow_U ≥ 0.50 and ≥ floor + 0.30), LP5-D4 (mean-ablation drop ≥ 0.30), CUE-ONLY (score_ds.py:44-45; inject_aw_eval.py:259-266, 511-527). SWAP-mid is a proper hop-level interchange; NOT-DECODER only shows two passes are insufficient (a sentence-slot counter also fails at K′2). LP5 requires a drop of 0.30, i.e. up to 0.70 accuracy survives without the latent, whereas NORTH_STAR row C1 says "drops to chance". `follow_U` pools two outcomes (donor final OR cf_a), inflating the follow rate. **All C1 gates run at d 4 only; no causal test exists at d 5–8**, so "C1 + C2 in one model" (§259) is C1 at d 4 plus accuracy at d 6. The carried state is a value mod 11 (≈ 3.5 bits), which should be stated in every C1 claim.
- **C2 "more thinking helps beyond training"** is scored as depth generalisation (d 6 at fixed K = 8); K′ > K is "untested" in §259's own scope line (THEORY.md:19788). The NORTH_STAR definition (accuracy rises with K′ past K) is only touched descriptively (§256 "flat to K′14"; §264 STD d 7–8 at K′ 12/16/24). So the C2 label currently measures a different construct.
- **C4**: aux targets on d ≤ 2 are hand-built per-pass states; "partial" is right, and the rate is 1/4 (§255).
- **C5 "decides how long to think"** = an external training-free rule on the output stream (score_c5c.py:20-24), halting at d − 1 + 3 passes. Nothing in the model decides; khalt tracks d because the cue schedule stops changing at d (§264's premise). "Spends more steps on harder inputs" is satisfied by construction. Status "met, scoped" overstates; "an external stopping rule works given C6" is accurate.
- **C6 "stable"** = hold to K′128 on d 1–6 under a repeated cue, plus a one-sided 1.25× norm guard vs control (score_lh.py:67-74) and LP0-W. Measured at the trained objective (confound 2); "low seed variance" rests on n = 2 from one lineage. Only 16 × K passes; fine as scoped if the §264 LF / CY schedules are made binding.
- **C3, C7** are not scored in §251–§265; the digest's C-status table should say so rather than carry older rows forward.

## 4. Ranked cheap fixes

1. **Independent replicates.** Train one more P-2 seed model (new `aw_plaw2` seed, ~$1) and re-run the bun recipe from it (2 seeds, ~$4). Would have changed every "2/2" in §255–§260 from "replicated within one lineage" to a real rate.
2. **≥ 4 seeds and k/n with Clopper–Pearson before any C-status "met"**; 2/2 = "replicated once". Changes §259 C6 SEED-VAR, §259/§260 C5 "met", §257 CONTENT-SELECTOR 2/2.
3. **Wilson CIs + `margin_se` in every scorer; NEAR label within 2 SE.** Changes §251 s2 (SWAP4 NEAR, label stays EXTEND with a flag), §259 s2 r = 2 (NEAR), by_pos sub-cells in §256/§257.
4. **Run SWAP-mid and LP5 at the claimed depth (d 6, and d 7) for any C2 claim** (eval-only, ~$0.3). Changes §256/§257/§259 wording from "C1 + C2" to "C1 at d 4; C2 accuracy only" until it passes.
5. **Make the §264 ER / LF / CY schedules binding for C5 and C6** (already registered; ~$0.5). Changes §259 C5 to "pending" and C6 to "pending scope".
6. **Add a binding K′-curve gate for C2** (d 6–8 at K′ 8 / 12 / 16 on new items; §264 collects it descriptively). Changes §256's C2 label to "depth generalisation" and tests the registered C2 construct for the first time.
7. **Post-hoc rule confirmations use new models**: cite gb_s1/s2 r = 3 EXTRAP (score_gbf.json) as the out-of-sample confirmation and require the same from of_s*/dd_s*. Changes Note 259-12's wording, not its verdict.
8. **LP5 to chance**: report `mean_at_t` accuracy itself and add a secondary gate `<= chance + 0.10`; otherwise state "latents necessary for ≥ 0.30 of accuracy". Changes the C1 sentence in §251–§260.
9. **Bar table**: one doc row per section listing C1/C2/C6 bars and which are absolute vs relative; freeze the C1 bar at 0.98 (gbf uses 0.95). Changes §260's PLAN-FREE-C1 1/2 to a NEAR check on gb_s1's d 1–4.
10. **Stratum / overlap check + looks ledger** ($0): count distinct d 1–2 programs vs 1,000 TEST2 draws; exclude training items from the next FRESH draw by sig; each closing note states how many descriptive reads it made. Affects §259's IN-RANGE C5 and the RETAIN gates only.
