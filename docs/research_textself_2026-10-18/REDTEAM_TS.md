# REDTEAM_TS: adversarial review of the TEXT-SELF draft (proposed §218)

Clock 2026-10-18 13:20 EDT start (`date`); repo HEAD 681d7f92 (the draft was written at 598e75de, 12:04–13:17). Opus red-team subagent for Theory. **$0: read-only under D:/Latent**; CPU checks ≤ 2 threads, idle priority, `python -B`; scripts and JSON in research6/rt_ts/.
Target: D:/Latent/docs/research_textself_2026-10-18/TS_draft.md (652 lines, read in full).
Also read: THEORY §170 (registration, TEXT-COT format, interims 08:40 / 09:16), §215 (+ 215-1…215-8, pair result), §216 (+ 216-1…216-9, s1 result), §217 (+ 217-1, 217-2); SELFDEPTH_v1 §0–§3; SD2_v1 §2.1–§2.2, §4.3 (grep), §6 (grep); LIT_FAITHFULNESS_SELFIMPROVE_2026-10-18 (full), LIT_ITERATED_SELFTRAIN_2026-10-17 (grep); boxes/nested170.py (main, loss_items textcot, train_items, cot_ids, cot_answers, mem_check, the training loop, the end-of-training eval); boxes/vbank.py l.33–58 (recode_targets, gate_rerender), l.744–775 (draw_step, rng_s); boxes/core.py GPT2KV.forward; theory/score_s216.py args_diff / args_ref_check; theory/score_s213.py META; s170_cot_s1 args.json / args_eval.json / mem_check.json; the draft's own scripts and JSON (research6/scripts/ts_*.py / .json).
Severity: **BLOCKER** = cannot be registered as written (a rule fails by construction or a reading is uninterpretable); **MAJOR** = a reading's meaning, OC or rule compliance changes, fix before v1; **MINOR** = wording, disclosure, cost, cheap gate.

Status: COMPLETE (1 BLOCKER, 6 MAJOR, 14 MINOR). Order: findings (§1), matching table (§2), OC re-shading (§3), cost / card / cut order (§4), one-line recommendations (§5), verification log (§6).

---

## 1. Findings

### BLOCKER

**B1. ARGS-DIFF censors every text cell by construction.**
- The draft's §4.6 ARGS-DIFF compares each FT cell with `s170_cot_s1/args.json` and allows only {init_from, steps, lr, self_pool, p_self, self_micro, out, args_ref, mem_check_self, mem_check_steps, stop_at, keep_ckpt_steps, draws_log_steps}; "anything else → ARGS_BAD (censor)".
- The base args.json is from 2026-10-09. nested170.py has since gained **30 argparse keys that the base file lacks** (n_slots, anneal, probe_loss_every, mem_layer, cache_lower, anneal_gate, csr, iit, iit_tau, iit_nmax, grokfast, gf_alpha, gf_lamb, cot_eval_bs, eval_only, aux_step, aux_fade, kmatch_impl, add_norm, aux_nopass, aux_reweight, aux_hold, kprimes, path_indep, excl_min_n, scratch, warmup, k_window, fast_add, keep_ckpt_steps), plus two `vars(a)` / info fields written into every args.json (`excl_survive_by_n`, `hostprobe_sps`) [VERIFIED: argparse keys regex over nested170.py vs the base args.json]. S213.META covers none of them except keep_ckpt_steps.
- score_s216.args_diff (the template) counts a key absent from the reference as `unexpected_new_keys` unless allowed (l.374). So **every s218 cell reads ARGS_BAD** (P(pass | anything) = 0 for every comparison row), or G-T5 halts the launch and the rule is amended after registration (the 215-5 / 216-6 pattern, now for ~32 keys).
- **Minimal fix:** register a `TEXT_NEUTRAL_NEW` map (each post-10-09 key at its off default: aux_step 0.0, warmup 100, excl_min_n 1, k_window "", fast_add false, …; `excl_survive_by_n` / `hostprobe_sps` to META), as score_s216's BASE_NEUTRAL_NEW; and a G-T12 fixture run on **a real smoke-launched args.json** (G-T4's `--stop_at 2` output) before registration, not only on synthetic dicts. If the trainer lives in a new module (m10), its own new keys join the allowed set.

### MAJOR

**M1. The s2 plan is stale: the latent s2 lineage is dead, so the latent comparator is s1 only.**
- Note 216-8 (12:41, during the draft's writing): R1-s2 is FT-DIVERGED (a_24(1) 0.921), a rule-b censor; R2-s2 (b315) was cancelled. The draft's §2 still lists the partner as "in progress" and §6 quotes POOLED rows (P(PARITY | no diff) 0.82; headline PARITY 0.26).
- Under the draft's own rule (POOLED over "the lineages with a readable round r"), latent s2 has **no readable round 1 or 2**: R1-s2's n50 15.33 is a censored cell's number (descriptive only, 216-9). A text s2 lineage therefore pairs with nothing; POOLED becomes 2 text vs 1 latent lineage, σ_eff = s·√0.75 = 0.87 s, not s/√2 = 0.71 s [DERIVED rt_oc.json]: P(PARITY | no diff) 0.70 at s 0.35 (draft 0.82), and at the measured s ≈ 0.2 the gain over one lineage is only 0.915 → 0.956.
- Survivorship: the latent comparator is the lineage that survived a latent-only stability censor (K′24 at n 1); the text arm has no such censor. Not a measured bias (s1 and s2 latent n50 agree within 0.13–0.20), but it belongs in the claim line.
- **Minimal fix:** s1 only (no text s2 spend); delete POOLED / UNPAIRED rows and their OC; state "latent comparator = s1, the surviving lineage; s2 censored at R1 (216-8)". Optional $0: re-evaluate `s170_cot_s2` at fp32 on the eval card as a descriptive text round-0 noise point.

**M2. The trainer's built-in end-of-training eval is an unregistered bf16 read of T1's extrapolation range, and it hangs the CPU smokes.**
- After the loop, nested170.main saves model.pt and **always** runs `cot_answers(TE)` over test_set(1000, 16) with `ac()` = bf16 autocast on the training card, then writes `eval.json` with acc_by_n n 1–16 [VERIFIED nested170.py l.1166–1180, l.1287]. Only `--time_steps` and `--eval_only` / `--mem_check` return early.
- For `s218_t1_s1` this is a **bf16, cloud-card read of n 13–16 (T1's past-support range) that lands with the drain before eval_text.json**, outside the eval contract and before G-T19's ordering applies to it. A second, dtype-different reading of the same items is the BF16-VS-FP32 fork the design exists to avoid.
- It also costs ≈ 0.2–0.3 h of cloud time per cell (16,000 generations) that §8 does not list, and with the new `--stop_at` the G-T4 / G-T7 CPU smokes would fall through into 16,000 CPU generations.
- **Minimal fix:** T1 edit adds `--no_final_eval` (default off → bitwise G-T7), set in every s218 launch and in the allowed set; or, if kept, the file is renamed `eval_train_bf16.json`, declared descriptive, and not opened before eval_text.json and the G-T19 note. `--stop_at` returns before the eval.

**M3. ROUND-1-ONLY reads LATENT-AHEAD-1 under "text self-training did nothing" (including a trainer bug).**
- The only uncensored way to fail the round-2 condition is ¬T-ABSORB. Then Δ_1 ≈ n50_T(1) − 15.46 with n50_T(1) ≈ 10.4–12, so **LATENT-AHEAD-1 passes with P ≈ 1 under no text effect** (P ≈ 0.09 of all outcomes by the draft's structural priors).
- The latent ABSORB rested on a reviewed, previously exercised recipe; the text trainer path (init_from, self pool, compose_T, per-type micro) is new code. A ¬T-ABSORB result cannot separate "text self-training cannot absorb a +4 window in 12k steps" from an implementation fault.
- **Minimal fix:** ¬T-ABSORB reads **TEXT-NOT-ABSORBED** (no PARITY-1 / TEXT-AHEAD-1 / LATENT-AHEAD-1 label; Δ_1 reported descriptively), with OC P(pass | no effect) ≈ 1 stated. ROUND-1-ONLY's three labels apply only to TEXT-LABEL-STALL (T1 absorbed, T2 unlabellable).

**M4. Eval card: the recommended Plan A evaluates the text arm on a different card class from the latent comparators, and the stated reason for Plan A is gone.**
- Project rule (and the brief): one GPU class and the comparators' eval card class; the latent n50s are 3070 fp32. Plan A puts base / T1 / T2 eval_text on cloud 3090s; the draft's only support is "[ASSUMED for the cross-card case]" from the latent's single-token BS-CHECK.
- My estimate agrees the fp32, tf32-off numeric effect is small (a flip needs a top-2 logit margin ≲ 1e-5 somewhere in a ~400-token chain; order 1e-3–1e-2 of frontier items, ≤ 0.005 in a(n)) — but it is unmeasured, and the bias table below shows the PARITY reading loses 0.06 per 0.15 n of systematic offset. A rule-compliant plan now costs less: **§216 s2 stopped (216-8), so the 3070's "≈ 19–22 h §216 queue" that motivated Plan A no longer exists** (it is finishing R1-s2's record parts and the pool_r1 restore, 8c87f71a).
- **Minimal fix:** Plan B′ — training only on cloud 3090s; pools, gates and **all three eval_text runs on one local card class at fp32, tf32 off, cot_eval_bs 32**. Reference card = the 3070. The idle 2070 may take the text evals only through a pre-stated **CARD-GATE**: base eval_text on both cards, per-n |Δa| ≤ 0.005 at n 1–16, |Δn50| ≤ 0.02, item flips listed; else 3070 only. Labeller card (TF-GREEDY) = the gate card (as §215 / §216: pool and gates on one class).

**M5. What the readings can claim with one lineage per arm (wording), and the GPT-2 position confound.**
- Measured lineage noise is small: three committed seed pairs (latent base 10.31 / 10.51, text base 10.44 / 10.60, latent R1 15.46 / 15.33) give σ_arm ≈ 0.12, **σ_Δ ≈ 0.17** for one lineage per arm [DERIVED rt_oc.json] (3 dof: 95 % upper ≈ 3.7×). So a LATENT-AHEAD of the additive size (Δ ≈ −1 / −2) is ~6–12 σ: robust as an instance. PARITY and any |Δ_2| < 1 are within reach of protocol biases of 0.15–0.3 n (M4, M6, m1): a +0.30 bias gives P(TEXT-AHEAD | no difference) 0.24 and P(PARITY | N-PROP) 0.23 at s 0.2.
- LATENT-AHEAD cannot by itself separate "text-CoT self-training extrapolates additively" from "GPT-2's learned absolute positions past the trained rows fail" (text reads rows to 767 at n 20 after training ≤ 527; the latent reads ≤ 83). FAIL-POS measures this but is only descriptive.
- **Minimal fix:** pre-state the claim line: *"one lineage each (latent = the surviving s1 lineage); GPT-2 small with learned absolute positions (1,024 rows); Arithmetic(n, 11); two rounds"*; the FAIL-POS share of first deviations at rows > T_train,r on the same line as any LATENT-AHEAD; |Δ_2| ∈ (0.5, 1.0] worded "marginal (within plausible protocol bias)".

**M6. The statistic (levels) does not match the question ("move by as much"), and the round-0 offset is carried into every Δ.**
- Δ_r = n50_T(r) − n50_L(r) includes Δ_0 (+0.13 on the bf16 curve; fp32 unknown). Under a shared proportional law the offset scales with support (+0.20 / +0.26), eating half of the ±0.5 band at round 2; the draft's §0 question is about the **gain**.
- A pure gain statistic Δ_r − Δ_0 is not better in general: it adds round-0 noise unless lineage effects persist (rt_oc: PARITY | proportional 0.75–0.85 at s 0.2 for correlation 0–0.5, vs 0.70–0.90 in levels; additive detection unchanged at 1.0).
- **Minimal fix:** keep levels as registered, plus a pre-stated **BASE-PARITY** condition on the fp32 re-eval (|Δ_0| ≤ 0.25); if it fails, the comparison reads on Δ_r − Δ_0 (decided now, before the re-eval), and both are always reported.

### MINOR

**m1. Gate π is defined on all values for text but on the label for the latent.** The draft sets π^E ≡ π (all values correct) and applies GATE's π_pool ≥ 0.995 / π_n ≥ 0.99 to it; the latent's π is label-correct and only π^E_pool ≥ 0.99 is on states. The text gate is therefore stricter (all-values ≥ 0.995 pooled, ≥ 0.99 per n). Fix: π_T := final value = gold, π^E_T := all values = gold, the gate and LABEL-NOISY-T on exactly the latent's terms. (Small effect: a wrong intermediate value usually changes the label.)

**m2. P(TEXT-LABEL-STALL) is over-stated; the stall is mostly a labeller-format artefact, and it should be worded so.** [DERIVED rt_reach.json]
- The base's REROOT keep is a function of the **input length** L0, not n: within n 16, keep 0.734 (L0 ≤ median) vs 0.505 (above); pooled logistic: keep 0.90 at L0 ≈ 42, 0.70 at 47.7, 0.50 at 51.4, against a trained maximum of **31** (200k training-law draws; p99.9 29), i.e. the single step reaches ≈ +11 tokens / ×1.35 past its trained copy length.
- T1 trains copies to L0 ≈ 45 (n 12 max on 25k draws). Under either reach law T1's projected REROOT keep at n 13–16 is **0.997 / 0.995 / 0.989 / 0.978** (additive) or 0.996 / 0.994 / 0.990 / 0.984 (ratio); with half the transfer 0.987 / 0.974 / 0.949 / 0.903. Only zero transfer (the base curve: 0.67 at n 16) stalls. So **P(GATE-T2 fails | T-ABSORB) ≈ 0.05–0.10**, not 0.15 (the stall row ≈ 0.06 of outcomes, both rounds readable 0.74 instead of 0.68).
- Why it is an artefact: the latent labeller decodes the next state per slot (VOIDs over the span, every other slot copies itself at the same position: vbank.recode_targets / rerender), so its "copy" is position-aligned and never fails (CV ≡ KEEP). The text labeller must re-emit the shortened expression token by token. The ~6-operation labelling-reach gap at equal training (§1.2a) is that output format, not reasoning depth.
- Fix: keep the matched gate and the base fallback (no text-only bridge, no lowered bar — both unmatched, and pre-state "no T1.5 bridge"); word TEXT-LABEL-STALL as "the text format's single-step labeller did not reach the round-2 window under the latent-calibrated gate (keep / π per n)"; give it an OC line; never fold it into LATENT-AHEAD.

**m3. D1 applies to both labellers under REROOT; state it.** Both re-rooted labellers see the current expression with the handle's operands visible (latent: x = [EOT] + render(cur) + [" ="], K = 1, SELFDEPTH_v1 §2.1). So REROOT gives neither arm extra label information; the D1 asymmetry is test-time only (the latent must carry values across passes over the original input). §1.4 frames D1 as text-only; reword. The step target is the same object in both arms: rerender(E_t) == cot_steps(root)[t] (vbank.gate_rerender) and the text target is cot_steps[1:] (nested170.cot_ids), brackets of the handle dropped in the same step in both [VERIFIED code].

**m4. Literature's filter vs the draft's filter.** Literature (LIT_FAITHFULNESS, last bullet) recommends a final-answer / step-count validity filter in both arms; the draft uses KEEP-T ∧ CV-T, i.e. the latent's full per-step form filter, which binds for text (frame failures) and not for the latent (CV ≡ KEEP). This is a defensible "same filter", but "Literature's recommendation made explicit" is inaccurate: say "the latent's filter applied verbatim; stricter than Literature's minimum".

**m5. G-T16 tests the wrong protocol, and the 90 / 90 evidence is thin.** ts_label_check's `gen_check` runs only under `proto == "full"`; the 90 agreements include 12 kept items. REROOT single steps were never compared with KV-cache generation. Fix: G-T16 compares TF-GREEDY REROOT with plain KV-cache greedy generation **of each re-rooted single step**, on ≥ 200 kept items per n at the gate card.

**m6. Eval cost is under-counted (MEMO-GAP-T and the context term).** [DERIVED rt_reach.json] eval_text over n 1–24 = 332,844 sequential decode steps (the draft's 293k–336k: OK). MEMO-GAP-T (2 × 2,000 items per window n) adds **119,658 steps for T1 (0.36×) and 323,788 for T2 (0.97× a full eval_text)**; §8 lists 1.5–2.7 h for "eval_text + eval_textself". The latency model is context-free (`b_sec_per_ctx_tok` 0.0) although GPT2KV concatenates the whole KV cache every step (core.py l.141: ≈ 2.4 MB × T per step at fp32 bs 32, ~+7–11 ms/step at T ≈ 700 on a 448 GB/s card). Fix: MEMO-GAP-T at 1,000 per set (SE of the gap ≈ 0.006 at a 0.98, the 0.02 bar ≈ 3 SE) or budget it; G-T18 times the longest n-24 batch and the projection carries a context term. Realistic: T2 evals ≈ 2.1–4.0 h, per lineage ≈ +1–2 h.

**m7. Pool spare 512 records (0.13 %).** The latent pools kept 13–16 % spare. Any cursor off-by-one or a resume fault voids a lineage. Fix: ≥ 1 % spare (97,000 per n), labelling cost +1 %.

**m8. Per-type micro changes the objective.** textcot's loss is a token mean per micro-chunk and the step averages micro means ((lk / micro).backward(), l.1068–1069), so with variable-length chains a self micro of 2 vs 4 reweights tokens, and G-T9 picks micro per host. vbank does the same ((L / mic).backward(), l.781–790), so it is matched in kind; the effect is larger for text (chunk token counts vary with n²). Fix: fix micro per round in the registration (T1 2, T2 4; G-T9 only confirms it fits), and say so.

**m9. No text FT-CTRL; T-SELF's wording attributes to self items.** The comparison is whole recipe vs whole recipe, so Δ needs no control; but "extended … by fine-tuning on its own chain" implies attribution that the latent got from FT-CTRL (§215 pair). Fix: wording "(not attributed: no text FT-CTRL)", or a text FT-CTRL (12k plain steps ≈ 0.9 h ≈ $0.2 + one eval).

**m10. Shared module.** vbank.py imports nested170 (recode_targets uses N), so §217's runner and R1-s2's record parts import it now. Fix: put the text FT trainer in a new module (as §217 did with vbank_causal.py) importing nested170 read-only; G-T7 then reduces to "nested170.py unchanged".

**m11. OC-rule gaps.** ROUND-1-ONLY's PARITY-1 / TEXT-AHEAD-1 / LATENT-AHEAD-1, TEXT-LABEL-STALL and CROSSED carry no P(pass | no effect) / P(pass | effect). Add them (rt_oc gives the numbers).

**m12. Sentinels in Δ.** n50 "> 24" / "< 1" are defined per cell but Δ_r is not defined for them. Fix: "> 24" ⇒ Δ_r > 24 − n50_L(r) (TEXT-AHEAD side), "< 1" ⇒ LATENT-AHEAD side; fixture.

**m13. Small items.** (a) Adaptive controllers (214-5 rule): state "no loss-reading controller on the textcot path [VERIFIED: AnnealGate / KW / CSR are inject-only]; clip 1.0 reads mixed batches, as the latent's". (b) The latent K′ = n + 2 is an evaluator-supplied, shape-derived compute budget; text's C_struct + 50 is too — say both are shape-informed; report the latent's fixed-K′ (e.g. K′ 26 from eval_deep) n50 descriptively. (c) Re-grep seeds at registration HEAD (my rt scripts used 23996–23999 for shape-only statistics; never shown to a model). (d) The cut order is unstated (see §3). (e) "T1 trains copies up to 43 tokens" — 25k draws give max 45 at n 12 (p99 41).

**m14. Overflow (Q3) is fair and immaterial.** Only n 24 overflows (29 / 1,000). Under PARITY or LATENT-AHEAD the round-2 crossing sits at n ≈ 18–21, where every item fits; under TEXT-AHEAD a crossing at n 23–24 moves by ≤ ~0.1 n, M_far by ≤ 0.007. Counting overflow as failure is the right intrinsic-cost rule. CTX-BOUND is a GPT-2 (1,024-row) property, not a text-CoT property — word it so.

---

## 2. Is the comparison matched? (the brief's question 1, element by element)

| element | matched? | where an outcome difference could come from the protocol |
|---|---|---|
| Base competence | yes at round 0 (n50 10.44 vs 10.31, bf16 text curve) | Δ_0 carried into every level Δ (M6); fp32 re-eval may move it |
| Labeller type | yes (own single step, fresh root each step, K = 1 / one emitted step) | output format: latent per-slot aligned copy vs text shifted re-emission → labelling reach, selection, stall (m2) |
| Label information | yes: both see the operands under REROOT (m3) | none at labelling; D1 is test-time only |
| Step granularity | yes: both = nested170.cot_steps (vbank.gate_rerender; cot_ids) | — |
| Filter | yes: KEEP ∧ CV in both, values never checked | the same rule removes ~0 % latent items but 3–30 % text items (KEPT-VS-DRAWN) |
| Gate | rule yes; **π terms not** (m1) | text gate slightly stricter |
| Mixture / windows / p_self / steps / lr / schedule / wd / clip | yes | — |
| Exposure | yes, exact (rng_s replay incl. 64 discarded integers(0, 5) per self step [VERIFIED vbank.py l.755, l.764]) | supervised tokens per item differ (format) |
| Micro accumulation | yes in kind (both mean of micro means) | per-host micro choice (m8) |
| Plain stream | each arm's own base law, replayed | — |
| Training card / dtype | yes (3090, bf16 autocast) | — |
| Eval protocol | answer accuracy in both; shape-derived compute budgets in both (K′ n + 2 / C_struct + 50) | trainer's extra bf16 eval (M2) |
| Eval card / dtype | dtype yes (fp32); **card no in Plan A** (M4) | ≤ 0.005 in a(n) expected; unmeasured |
| Censoring | asymmetric: latent has K′24 / LENGTH-FREE etc.; text only FT-DIVERGED-T | comparator is the surviving latent lineage (M1) |
| Control | latent had FT-CTRL; text has none | attribution wording only (m9) |
| Positions | latent rows ≤ 83; text rows to 767 at n 20, 1,085 at n 24 | GPT-2 learned-APE confound in LATENT-AHEAD (M5) |

**The comparator is s1 only** (216-8). **With one lineage per arm the readings claim an instance**: measured σ_Δ ≈ 0.17 makes an additive-size gap (≈ −1 / −2) robust; PARITY and |Δ_2| < 1 are bias-sensitive (M5).

---

## 3. OC re-shading [DERIVED rt_ts/rt_oc.json; panel means from the draft's ts_panels.json; priors as the draft]

**Noise.** σ_Δ (one lineage per arm) ≈ **0.17** from three committed seed pairs (|Δ| 0.20 / 0.16 / 0.13); the draft's s = 0.20 row is the measured one, 0.35 is a conservative alternative (3-dof upper bound ≈ 0.6).

| truth | s | P(PARITY) | P(TEXT-AHEAD) | P(LATENT-AHEAD) | P(CROSSED) |
|---|---|---|---|---|---|
| no difference | 0.17 / 0.20 / 0.35 | 0.96 / 0.92 / 0.61 | 0.02 / 0.04 / 0.15 | 0.02 / 0.04 / 0.16 | 0.00 / 0.01 / 0.08 |
| no difference + bias +0.15 / +0.30 | 0.20 | 0.86 / 0.68 | 0.11 / **0.24** | 0.01 / 0.00 | 0.02 / 0.08 |
| −1 / −1 | 0.20 / 0.35 | 0.00 / 0.04 | 0 | 0.96 / 0.85 | 0.04 / 0.12 |
| additive panels (N-SHIFT, POS-ADD) | 0.20–0.35 | 0.00 | 0 | **≥ 0.998** | ≤ 0.002 |
| POS-PROP / N-PROP | 0.20 | 0.90 / 0.70 | — | — | — |
| N-PROP + bias +0.30 | 0.20 | 0.23 | — | — | — |
| band ±0.75, no difference / −1 / −1 | 0.20 | 0.99 / 0.05 | — | — / 0.81 | — |
| POOLED, 2 text vs 1 latent, no difference | 0.20 / 0.35 | 0.96 / **0.70** (draft 0.82) | | | |

**Shaded headline** (s 0.20, band ±0.5, given both rounds readable): **LATENT-AHEAD 0.56, PARITY 0.29, TEXT-AHEAD 0.15, CROSSED 0.01** (draft 0.59 / 0.19 / 0.18 / 0.03 at s 0.35). P(both rounds readable) **0.74** (draft 0.68; P(GATE-T2 | T-ABSORB) 0.93 from m2). Unconditional: LATENT-AHEAD 0.41, PARITY 0.21, TEXT-AHEAD 0.11, CROSSED 0.01; TEXT-NOT-ABSORBED 0.09; TEXT-LABEL-STALL 0.06; censored ≈ 0.10. LATENT-AHEAD's 0.56 is prior-driven (additive 0.45); the discriminating power is the robust part.

**Readings that pass under no effect:** PARITY (by design, under "no difference": 0.92 at s 0.2; specificity P(PARITY | 1-n gap) ≤ 0.035); LATENT-AHEAD-1 under "text FT did nothing" (≈ 1, M3 → TEXT-NOT-ABSORBED); TEXT-LABEL-STALL under zero copy-reach transfer (≈ 1; the base curve gives keep 0.67 at n 16); TEXT-AHEAD under a +0.3 protocol bias (0.24). T-ABSORB / T-DBS / T-DBS-2: ≈ 0, as drafted. ROUND-1-ONLY at s 0.2 (for the stall branch): PARITY-1 | no difference ≈ 0.99, LATENT-AHEAD-1 | N-SHIFT ≈ 0.99.

---

## 4. Cost, caps, card, GUARD_STOP, cut order (the brief's question 6)

- **Plan A's per-lineage hours are low by ≈ 1.5–3.5 h** (m6: MEMO-GAP-T 0.36× / 0.97× of an eval_text; the KV-cat context term; M2: the trainer's own bf16 final eval ≈ 0.2–0.3 h per cell): ≈ 11.7–22.6 h, **$2.0–5.9 per lineage**, so the $5.5 admission can refuse on a slow host. A 12–23 h single-host chain also carries host-loss / IP-change risk; the stage-boundary mirrors cover it.
- **Recommended Plan B′ (s1 only):** cloud = T1 and T2 training only (two deploys, as §215 / §216): 4.4–8.4 h ≈ **$0.75–2.2**; per-cell admission $1.0 (T1) / $1.45 (T2) by G-TIME, guards $1.2 / $1.7, **family cap $3.0**, GUARD_STOP = dead (rule b), steps never reduced. Local fp32 ≈ 11–14 h per lineage (base eval 2.2–2.9 h with the context term; GATE-T1 + pool ≈ 1.0–1.3 h; T1 evals ≈ 2.7–3.5 h with MEMO-GAP at 1,000 per set; GATE-T2 + pool ≈ 1.6–2.1 h; T2 evals ≈ 3.4–4.4 h), on the 3070 or on the 2070 after CARD-GATE. Labelling and gates on the same class as their pool.
- **Cut order (pre-state):** (1) no s2 text lineage (now); (2) MEMO-GAP-T to 1,000 per set; (3) descriptive parts (LABELLER-FULL, LABELLER-KEPT-T, BF16-VS-FP32, DEPTH-PAIRED, TOKEN-COST) after the registered parts, on the idle card; never steps, gates, eval_text or LABEL-AUDIT-T.

---

## 5. One-line recommendations on the draft's open questions (and the brief's list)

1. **Re-rooted vs whole-chain:** REROOT primary (matched; FULL keeps 0.05 / 0.00 at n 11 / 12); FULL descriptive at the gates. Confirm.
2. **TF-GREEDY:** confirm, with G-T16 re-specified on REROOT single steps and ≥ 200 kept items per n (m5); fallback = generation for the failing window only.
3. **Card plan:** Plan B′ — train on cloud 3090s; pools, gates and all eval_text on one local class at fp32 (3070 reference); **the idle 2070 only after a pre-stated CARD-GATE on the base** (per-n |Δa| ≤ 0.005, |Δn50| ≤ 0.02), else 3070 (M4).
4. **Run s2:** no — the latent partner is censored (216-8), POOLED is 2-vs-1 (σ_eff 0.87 s) and gains ≤ 0.04 in P(PARITY | 0) at the measured noise; optionally re-eval `s170_cot_s2` at fp32 ($0) as a text noise point (M1).
5. **PARITY band:** keep ±0.5 (measured σ_Δ ≈ 0.17 gives 0.92 / 0.002 for no-gap / 1-n gap); ±0.75 drops P(LATENT-AHEAD | −1 / −1) 0.96 → 0.81. Control bias instead (M4, M6) and word |Δ_2| ≤ 1 as marginal (M5).
6. **Metric:** free-generation answer accuracy as registered (matched to the latent's answer-only a(n)); STRICT descriptive.
7. **fp32 re-eval:** yes; the committed bf16 curve descriptive only; and disable or seal the trainer's bf16 final eval (M2).
8. **CTX-BOUND:** yes, a pre-stated end, worded as GPT-2's 1,024-row limit (m14).
9. **Mechanism without STOP-LOC:** acceptable; name the merged label T-SHIFT/MIXED.
10. **Coin replay:** keep (exact G-T3 at zero cost).
11. **Latent fixed numbers:** fine (BS-CHECK 0 / 8,000 flips); add the latent's fixed-K′ n50 descriptively (m13b).
12. **T2 stall rule:** keep the matched gate and base fallback, no bridge, no lowered bar; P ≈ 0.06; gate π on the latent's label / all-values terms (m1, m2).
13. **LABELLER-REACH:** add as a descriptive row, worded as the output format's reach (per-slot vs re-emitted copy), not depth (m2).
14. **Brief-specific:** comparator = s1 only; ¬T-ABSORB → TEXT-NOT-ABSORBED (M3); ARGS-DIFF neutral map (B1); BASE-PARITY (M6).

---

## 6. Verification log

- **rt_ts/rt_reach.py → rt_reach.json** (CPU, idle, 54.5 s; generator statistics and the draft's per-item flags only, no model run): rebuilt the draft's 2,400 precondition items (rng 23180 stream, exclusion as ts_label_check) and joined ts_label_check_s1 / _s1_deep per-item REROOT flags (per-n keep reproduced exactly at n 9–20); keep by L0 and within-n median split; pooled logistic (a 11.90, b −0.232 per token); training-law L0 maxima (200k draws, n 1–8: max 31, p99.9 29; 25k per n at 9–16); T1 keep projections; decode-step counts under cot_answers' batching (eval_text 332,844; MEMO-GAP T1 119,658, T2 323,788). Shape-only rngs 23997–23999 (never shown to a model).
- **rt_ts/rt_oc.py → rt_oc.json** (CPU arithmetic, 400k draws, rng 23996): measured σ from three seed pairs; four-way rule at s 0.17 / 0.20 / 0.35, bands ±0.5 / ±0.75, biases 0 / ±0.15 / ±0.30; POOLED 2-vs-1; base-corrected Δ; structural shading.
- **Code read:** nested170.py (argparse keys vs the base args.json: 30 absent + excl_survive_by_n / hostprobe_sps; loss_items textcot token-mean CE; cot_answers gold budget and bf16 `ac()`; the unconditional end-of-training eval l.1162–1180 / l.1287; mem_check's own rng 0; train_items), vbank.py (recode_targets / rerender / gate_rerender; draw_step's rng_s coin and 64 integers(0, 5); per-type micro l.781–790; `import boxes.nested170 as N`), core.py GPT2KV (`wpe[p0:p0+T]`, per-step torch.cat of the cache), score_s216.args_diff / args_ref_check, score_s213.META.
- **Repo state:** HEAD 681d7f92 (+ d143849c, 98b4c757 after it, §217 only); 216-8 / 216-9 postdate the draft's input list; the draft's seeds 23180–23580 still have 0 hits outside TS_draft.md.
- Nothing written under D:/Latent; no GPU used.

*Red team written 2026-10-18 13:20–13:44 EDT.*
