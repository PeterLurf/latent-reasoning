# REVIEW_R3: red team of R3_draft.md v0 (FREEZE-G / PARAM-SWAP / Stage B)

Reviewer: red-team subagent for Theory, 2026-10-21. Not committed. $0: CPU only, read-only on committed files.
Read: R3_draft.md, its three scripts and JSONs, THEORY Notes 216-5 / 216-8 / 216-9 / 216-13 and the KEEPS-SELF-STAB rows, vbank_diag.part_hist24 / part_d3hist / part_stabhist docstrings, slva.gather, vbank_eval.ablate_keep, LIT_R3_FREEZEG_2026-10-21 (d4a043ae).
Not opened: any lvb_direct_s1 file, the s130 TEST replays, s162 outputs, the sealed n 25–30 items.
Check scripts (review_scripts/, each writes a JSON beside itself): rv_hist24_recheck.py, rv_route_split.py, rv_group_sizes.py, rv_oc_recheck.py.

## Verdict: **REVISE** (not GO to v1)
The mechanism question is good and cheap, but the registered target is mis-specified. The committed hist24 arms show that most of the "STAB_self loss" is **an added backup route** (core / coda bank history) plus **a short-horizon state hold**. SLVA's own history stays *sufficient* in every s1 model. A lever whose pass condition is "STAB_self goes back up" is therefore a lever *against redundancy*. Run the $0 route and K′-horizon checks (#1, #2) first, then decide whether a lever is needed at all. Drop Stage B from this family (#9).

## Reproduced (all match the draft)
- STAB_self from hist24 (field `self_ablate`, the score_s213.hist24 formula): D3-T s1 0.7818, FT-CTRL 0.6963, FT-SELF 0.4892, R2-s1 0.4372, D3-T s2 0.6840, R1-s2 0.4892.
- eval_vb `acc_k['24']` at n 1–2: R1-s2 0.921 / 0.973; D3-T s2 0.994 / 0.987.
- D6 panels from R2's eval_deep `acc_n2`: mean a(17..20) 0.8745, mean a(21..24) 0.2083, n50 20.54 → 24.54; H_prop 0.998 / 0.987 / 0.932 / 0.821 and 0.659 / 0.423 / 0.269 / 0.180.
- §5 closed form: 0.0056 / 0.496 at σ 0.04, and the full table.
- Logsplit ratios (gn_preclip self / plain) 21–644×. γ differences ≤ 0.05 logits at age 22. Guard sum $6.90.

## Issues, ranked by severity

**1. CRITICAL: the target measures SLVA's *necessity*, and what the self rounds changed is redundancy.** [MEASURED rv_route_split.json]
- In hist24, `self_ablate` masks SLVA's own-slot versions only, `other_ablate` masks the core / coda bank only, and `bank_ablate_slva` masks all readers (slva.py gather; part_d3hist doc).
- Masked acc_k24, mean over n 3–6:

| model | SLVA alone (`other_ablate`) | bank + state (`self_ablate`) | state only (`bank_ablate_slva`) | BANK backup = self − bank |
|---|---|---|---|---|
| D3-T s1 | 1.000 | 0.218 | 0.204 | 0.014 |
| FT-CTRL | 1.000 | 0.304 | 0.262 | 0.042 |
| FT-SELF | 1.000 | 0.511 | 0.352 | 0.159 |
| R2-s1 | 0.995 | 0.561 | 0.384 | 0.177 |
| D3-T s2 | 0.907 | 0.306 | 0.266 | 0.040 |
| R1-s2 | 0.937 | 0.508 | 0.314 | 0.194 |

- **The self-specific "loss"** (FT-SELF − FT-CTRL, 0.207) splits into 0.117 (56 %) from the bank route and 0.090 from the state hold. In s2 (R1-s2 − D3-T s2, 0.202) it is 0.154 (76 %) bank and 0.048 state. Per item, R2-s1 has 17.8 % of items rescued by the bank when SLVA's history is masked; R1-s2 has 19.8 %.
- **SLVA's history route is not lost.** On its own it still holds 0.995–1.000 in every s1 model, and in s2 its sufficiency *rose* (0.907 → 0.937).
- So "MEM-TRADED" (by 0.011 under an absolute 0.5 bar) is mostly the model gaining backup routes. HOLD-SHIFT (§2.1) is half right: the state hold grew, but the larger part is a move from one *history reader* to another.
- Stage A's pass condition (ΔSTAB ≥ +0.08) rewards *removing* these backups. That is a one-sided dynamics target wearing a constraint's clothes (the CSR lesson). OVERSHOOT caps the target but does not fix its direction.
- **Fix:**
  - State the hypothesis as route redistribution.
  - Register SUFF_SLVA (`other_ablate`), BANK (`self_ablate − bank_ablate_slva`) and STATE (`bank_ablate_slva`) as two-sided descriptive rows.
  - Move the primary to output stability at long idle (#2), and keep STAB_self descriptive. If a lever is still wanted, its target is the failing route, not "more dependence on SLVA".

**2. CRITICAL: the actual C6 harm is unmeasured where it lives.** This is the more interesting direction, and it is untested.
- O3's harm is at n 1–2, with 22–23 idle passes at K′ 24. hist24 covers n 3–8 only, and no eval reads n ≤ 8 past K′ 24 (eval_vb `acc_k` keys 0–16 and 24; eval_deep covers n 13–24 only). [MEASURED rv_hist24_recheck.json]
- **The state hold decays with idle passes in the self lineages.** OLS slope of `self_ablate` acc_k24 on (24 − n), n 3–8: D3-T −0.011, FT-CTRL +0.001, FT-SELF −0.049, R2 −0.060, R1-s2 −0.076. At fixed K′ 24, n and idle count are collinear, so this cannot yet be called an idle-pass effect.
- **$0 test, before any lever:** an (n × K′) grid, n 1–8 × K′ {24, 32, 48, 64}, with arms intact / `self_ablate` / `other_ablate` / `bank_ablate_slva`, on the six committed models. It is local fp32 and uses part_hist24's rollout with extra coda_rows (needs a K′ parameter, a G-D review, and intact at K′ 24 bitwise equal to eval_vb). Store per-item answers.
- **Pre-state the outcomes:**
  - H_RED (benign redundancy): intact a(K′) ≥ 0.99 out to K′ 64 in FT-SELF and R2. Then there is no stability problem to fix in s1, and round 3 needs no lever.
  - H_DRIFT: intact a(K′) at n 1–3 decays with K′ faster in the self lineages, faster where STATE is larger, and R1-s2's n 1 errors are items where the state route wins. Then the lever should act on the state hold: make the idle pass a contraction or fixed point. That is a step-free stability route that needs no history at all, which is the stronger C6 direction.
  - Either way, read the masked arms at fixed n across K′ to separate n from idle count.

**3. HIGH: the PARAM-SWAP selection rule is unsound under redundancy and co-adaptation.**
- **(a) Damage reads as recovery.** STAB = 1 − masked acc, so any splice that damages the backup or state routes raises STAB. SWAP-VALID gates on *intact* accuracy, which is the most redundant and least sensitive signal. It is the wrong gate.
- **(b) The splice is maximally off-manifold.** SELF and CTRL displacements are nearly orthogonal in every matrix group: cos 0.13–0.16 (r3_param_drift `cos_self_ctrl_displacement`), and rel‖SELF − CTRL‖ ≈ 1.5 % against ≈ 1.1 % for each model vs base.
- **(c) Non-additivity** (Literature: Panigrahi needed learned masks; TIES interference; LMC barriers).
- **Fix** (Literature's checklist folded in, plus a damage control):
  - First, a whole-model SELF → CTRL interpolation at α ∈ {0, .25, .5, .75, 1} on all three masked arms and intact. A barrier or a non-monotone curve voids group splices.
  - For each G, run GRAFT (CTRL_G → SELF), IMPLANT (SELF_G → CTRL), the COMPLEMENT, ≥ 3 random size-matched element masks, and a REVERT-TO-BASE (D3-T_G) arm. Report rec(G) + rec(¬G) − rec(all).
  - SWAP-VALID must also require STATE acc ≥ min(body, donor) − 0.02 and `dup_foreign` at chance, so damage cannot pass.
  - Select G only if GRAFT and IMPLANT agree within 0.15 and both beat the random-mask 95th percentile.
- **Better $0 attribution: activation / path patching on the masked forward.** Run FT-SELF's `self_ablate` rollout and patch in CTRL's same-input outputs per component and per pass: each core layer's bank-read attention output, the coda bank read, SLVA's U_t. Do it in both directions. Route-split already names the candidate, the core / coda bank read. If patching CTRL's bank-read outputs drops SELF's masked acc to CTRL's level, the backup is localised there. Patching activations avoids weight co-adaptation. It needs a hooked eval, a G-D review and an intact-path bitwise gate.

**4. HIGH: the OC null is too favourable, and σ_rep's REPLAY would likely measure nothing.** [DERIVED rv_oc_recheck.json]
- **The right null is a generic-freeze effect δ_g, not δ = 0.** Any freeze slows self-item learning, and less self learning means higher STAB (FT-CTRL is the extreme, 0.207). ABSORB (min acc n 9–12 ≥ 0.95, twins at 1.000) still passes for a slowed cell.

| σ_rep \ δ_g | 0 | 0.02 | 0.04 | 0.06 |
|---|---|---|---|---|
| 0.04: P(both) | 0.006 | 0.019 | 0.052 | 0.118 |
| 0.06: P(both) | 0.027 | 0.052 | 0.091 | 0.148 |

  The likelihood ratio at σ 0.04 falls from ≈ 90 to ≈ 10 at δ_g 0.04.
- **P(Stage B launch | null)** = P(s1 SEED-PASS), because §6 lets MECH-PARTIAL with s1 launch B. That is 0.075 at σ 0.04, δ_g 0, and 0.23 at δ_g 0.04, not 0.006.
- **REPLAY:** same seed and same card class reproduce a run exactly (project lesson), so a $0.5 same-recipe 3090 REPLAY would most likely be bitwise identical and give σ_host = 0. A different card class changes the basin, which is the wrong noise.
- **Fix:**
  - Check determinism with a 200-step replay (≈ $0.02).
  - Then run one **SHAM-FREEZE** cell first: s1, a random element mask of |G| parameters, everything else identical (≈ $0.55). Its D_sham measures δ_g plus trajectory chaos in one draw.
  - If |D_sham| ≥ 0.08, stop Stage A. Otherwise register SEED-PASS as D_FRZ − D_sham ≥ 0.08 and redo the OC with three cells.
  - The existing two-seed points (base spread 0.098, R1 spread 0.000) give σ ≈ 0.05 with 2 df. Use that as the central value, not 0.04.

**5. HIGH: there is no size-matched control, and D4 is circular.** [MEASURED rv_group_sizes.json]
- Parameter shares: prelude 33.8 % (42.5 M), core 22.6 %, coda 11.3 %, slva 0.94 % (1.18 M), of 125.6 M.
- D4 rests on A4 (a STAB change scales with displacement). The draft's own O8 / D3 shows that displacement does not track function, so D4 cannot certify a pass as group-specific.
- **Fix:** admit a candidate G only together with a sham at the same parameter count. For prelude or coda that is mandatory: freezing a third of the plasticity is a generic slow-down. Otherwise restrict the candidate set to slva and bank_bias.

**6. HIGH: freezing does not freeze behaviour, and the grown route is outside slva anyway.**
- SLVA's query comes from N(V_{t−1}) and its keys from core states (slva.gather). γ moved ≤ 0.05 logits, yet on self batches the off-newest copy mass rose within R2 from 0.417 to 0.455 (r3_logsplit `slva_offnewest_copy`).
- The bank backup (#1) lives in the core / coda bank read, so under route-split FREEZE-slva is predicted to *fail*. A fail would not show H_DIFF.
- **Fix:**
  - Make Q5 a registered descriptive row: gather_log newest mass at idle passes, K′ 24, n 3–6, FRZ vs twin.
  - Word a fail as "not carried by slva's weights", never "spread / co-adapted".

**7. MEDIUM: the s2 comparator is a censored cell.**
- R1-s2 is FT-DIVERGED, a rule-b censor under which "every R1-s2 reading fails" (216-8). Using its STAB 0.489 as a twin needs an explicit Audit ruling.
- A3 is unsupported. Weak support: D3-T s2's BANK backup (0.040) equals FT-CTRL s1's (0.042).
- The draft's "intact acc_k24 ≥ 0.997 at n 3–8 everywhere" is false for D3-T s2: 0.972 at n 3, 0.990 at n 4 (eval_vb).
- **Fix:** get the ruling, or drop s2 from the primary and keep it descriptive.

**8. MEDIUM: there are hidden second changes in the FRZ recipe.**
- The twin's args.json has `auto_micro: True`. Its self_micro 2 came from a memory probe (micro 1 OOM). Freezing frees Adam state and gradient memory, so the probe can pick another micro, which changes bf16 accumulation.
- The global grad-norm clip now excludes G (slva ≈ 6 % of the norm).
- The run must stay on the twin's GPU class (RTX 3090), since card class changes the basin.
- **Fix:**
  - Pin micro 1 / self_micro 2 and turn auto_micro off, with ARGS-DIFF listing these as equal-valued.
  - Disclose the clip change.
  - Add to K4 that the step-1 loss and grad norm on non-G params are bitwise the twin's (the forward is identical at step 1).

**9. MEDIUM: Stage B adds little given Stage A.**
- LR ≈ 2.4. DBS-3 passes under both panels. The suggested N50-3 ≥ 24.5 also passes H_shift (24.54), so neither discriminates.
- MEM-HELD-3 is again a necessity target (#1).
- B spends the one-shot sealed n 25–30 on a lever cell. And the lever acts on R2-s1, whose backup (0.177) already exists.
- B's real question, whether round 3 still adds depth (n50 / support ≈ 1.29), needs no lever.
- **Fix:**
  - Remove B from this family.
  - If #2 reads H_RED, register **R3-PLAIN alone** as the depth test, with the K′-grid rows and n50 read against the two panels, under the user's OK.
  - If #2 reads H_DRIFT, design the lever against the failing route first.

**10. MEDIUM: problem → fix loop risk.**
- The "problem" is a 0.011 miss on an absolute bar of a necessity statistic. The pipeline (select G, freeze, apply in B) is a patch chain on that statistic.
- **Fix:** answer the mechanism question (#1–#3 patching, #2 horizon) at $0 first. Fund a lever only against a measured output harm.

**11. LOW: censoring and wording.**
- Never-reached and censor rules are otherwise complete.
- Add: a FRZ cell whose STATE acc falls below its twin's by more than 0.05 reads ROUTE-DAMAGE, never MEM kept. That is the training-side analogue of 3(a).
- O9 compares off-newest mass across batch types (self 0.37–0.46 vs plain 0.27–0.28). Self items are n 9–12, so that comparison is confounded by n. Word it within type only.
- D3's "slva 1.45 % vs 1.37 %": state the denominator. Slva is 0.94 % of all parameters, so 1.37 % must be its share of gradient-receiving parameters.

**12. LOW: Literature fold-in (d4a043ae).**
- **N2:** withdraw as ours. Re-tag D1 as [PROVEN, a corollary of Brunner 1908.04211].
- **N1 / N3 / N4:** NARROW; cite 2602.23197, Panigrahi 2302.06600 and Jain 2311.12786.
- **LIT pre-checks (1)–(4):** adopt them inside #3.
- **LIT (5), a short gold-only fine-tune of FT-SELF:** it should *not* replace selection as the first step.
  - FT-CTRL shows that gold-only FT from D3-T lowers STAB too (0.782 → 0.696), so "dependence does not return" is predicted under both readings. Only a fast return would be informative.
  - Route-split already answers Jain's "wrapped" question at $0 for sufficiency: SLVA's history is still sufficient (`other_ablate` ≈ 1.0).
  - (5) is a short local-GPU test, not CPU. It is an optional add-on after #2.

## Order of work for v1 ($0 unless noted)
1. Route-split (done, rv_route_split.json). Rewrite §2 / §4 around SUFF / BANK / STATE.
2. The (n × K′ ≤ 64) four-arm grid on the six models, pre-stated per #2 (local, after a G-D review).
3. Only if H_DRIFT: interpolation α, then patching (#3), then a lever with a sham-freeze cell first (#4) and a size-matched control (#5).
4. Stage B: only as R3-PLAIN, decided by step 2, with the user's OK.
