# PROOFS_TS: proof-and-arithmetic check of TS_draft.md (TEXT-SELF, proposed §218)

Clock: started 2026-10-18 13:20 EDT (`date`). Opus checker for Theory. $0, CPU only (≤ 2 threads, Idle priority, `python -B`), no repo edits.
Target: D:/Latent/docs/research_textself_2026-10-18/TS_draft.md and its scripts/ (read only).
Scripts: research6/proofs_ts/pc_*.py, JSON beside each. Fresh items drawn here: **rng 23187** (n 9–20, model-read) and shape-only rng 23188 / 23189 (never shown to a model). 23183–23189 had 0 grep hits in docs/THEORY.md, docs/research_*/*.md, boxes/*.py, theory/*.py, scratch/*.{py,sh}. **23187's syms must join every TEXT-SELF exclusion set** (they were read by s170_cot_s1).
Marks: PASS (recomputed, agrees), CORRECTED (right value + minimal fix text), FAIL (claim wrong as stated), NOTE (not a tagged claim, but it bears on a reading).


## Summary (for Theory)

Finished 2026-10-18 15:02 EDT. CPU only, about 1.5 h at ≤ 2 threads and Idle priority; nothing written under D:/Latent. Checked ≈ 54 tagged claims / numbers: **≈ 37 PASS, 17 CORRECTED, 0 FAIL**.
**Material (change feasibility, a gate or a reading):**
1. **s2 pool exhaustion (P4).** The seed-2 coin stream takes 6,076 self steps = 388,864 items, more than a 384,000-record pool, so the s2 lineage would be voided. Size the s2 pools ≥ 388,864 (e.g. 97,500 per n; 65,000 / 32,500). The s1 replay is exact: 5,992 / 383,488, with coins 12,000 / 12,000 and every U value matching, in R1 and in R2.
2. **TF-GREEDY exactness (P1).** The proof is PASS. The draft's 90 / 90 covered only the FULL protocol; the registered REROOT path was untested. Now measured: REROOT 1,500 / 1,500 items (169 rejections) and FULL 600 / 600, all agreeing with independent KV-cache generation, min margin ≥ 1e-3. G-T16 must specify REROOT generation, not the eval loop.
3. **Noise model (P5).** The measured between-lineage n50 spreads (0.20, 0.13, 0.16) give ŝ = 0.17 (3 df), half of the central s 0.35. At ŝ, P(PARITY | no difference) = 0.96, not 0.61. The OC arithmetic itself is exact.
4. **Cost (P6).** MEMO-GAP-T's 16k / 32k free generations and TF padding (+28 %) were uncosted: **$1.9–5.6 per lineage** (draft $1.7–5.0), family $3.8–11.2 under the $12 cap. The $5.5 admission rule binds at the slow / dear corner.
**Minor:** 148 → 149; "6 operations" → 5; n 25 fit 0.52–0.55; copy / row frontiers 45 / 59 and 313 / 531; 290k–332k steps; memory slope 0.082 GiB per row (micro 2 / 4 unchanged); G-T18 6.0 GiB local cap; Plan B $0.7–2.0 and 10–13 h of 3070; generation at bs ≥ 256 is only ≈ 1.5–3× slower than TF; the D1 wording (a value check is equally available to the latent filter); FULL kept chains impure (π 0.96 at n 10–11, 0.80 at n 11).
**Replicated on fresh items (rng 23187; add these syms to every exclusion set):** REROOT keep 1.000 / 1.000 / 0.995 / 1.000 (n 9–12), 0.66 (n 16), 0.11 (n 20); 0 kept-wrong at n 9–12; FULL 0.56 / 0.075 / 0.00. Filter form-only and equal to the latent's: PASS. Context overflow (29 / 1,000 at n 24 only): PASS.

---

## P4. Coin-stream replay and pool sizes (§1.5, §3.2) [pc_streams.py → pc_streams.json]

Replay of rng_s = default_rng(seed + 21800) with vbank.draw_step's consumption (one `random()` per step; on a coin < 0.5, 64 scalar `integers(0, self_kextra + 1)`; self_kextra = 4 in all three latent FT cells' args.json), compared with each cell's draws_log.json (12,000 records each):

| latent cell | seed | replay self steps | logged | coin agree | U agree (self steps) | self items |
|---|---|---|---|---|---|---|
| s215_ftself_s1 (R1) | 1 | 5,992 | 5,992 (log.jsonl 5,992) | 12,000 / 12,000 | 5,992 / 5,992 | 383,488 |
| s216_r2_s1 (R2) | 1 | 5,992 | 5,992 | 12,000 / 12,000 | 5,992 / 5,992 | 383,488 |
| s216_r1_s2 (R1-s2) | 2 | **6,076** | 6,076 | 12,000 / 12,000 | 6,076 / 6,076 | **388,864** |

- **S1-T stream identity [PROVED by stream identity]: PASS** for s1. The replay reproduces the coins *and* every logged U value, so the consumption model (random + 64 bounded draws) is exact, including the bitgen-level uint32 buffering. A vectorised `integers(0, 5, size=64)` gives the same coins (checked). Bound matters: the replay must call `integers(0, 5)` (a bound of 1 would consume nothing in numpy).
- **5,992 self steps, 383,488 items, spare 512 = 8 steps: PASS (s1 only).**
- **Hypergeometric per-n consumption: PASS** (T1 95,872 SD 9.79; T2 63,914.7 SD 8.43 / 31,957.3 SD 6.25).
- Q10's alternative (coin only, no U draws): 6,051 self steps for seed 1 (law 6,000 ± 54.8): PASS for "5,992 ± 55".
- **CORRECTED (material for the s2 lineage).** The s2 coin stream (seed 2, the R1-s2 / R2-s2 stream the draft adopts in §2) takes **6,076 self steps = 388,864 self items**, more than the 384,000-record pool. With §1.5's sizing the s2 text pool is exhausted at step ≈ 11,9xx, and "exhaustion = void" censors the cell. Fix text for §1.5 "Pool sizing": *"Consumption is deterministic per lineage: s1 383,488 (5,992 steps), s2 388,864 (6,076 steps; replayed from s216_r1_s2/draws_log.json, 12,000 / 12,000 coins). Pools: s1 96,000 per n (T1) and 64,000 / 32,000 (T2) = 384,000; s2 97,500 per n (T1, 390,000) and 65,000 / 32,500 (T2, 390,000)."* (The latent pools had 440,000 / 444,000 records, so the latent arm never faced this.) §2's "G-T3 checks it against s216_r1_s2/draws_log.json once that file exists": it exists now and the replay matches it.

## P3. Context overflow and token lengths (§1.7, §3.1, §3.3) [pc_tokens.py → pc_tokens.json]

Independent shape-only length computation (placeholder values), asserted equal to len(N.cot_ids) and len(syms) + 2 on all 24,000 test items.
- **Feasible iff P + C ≤ 1025: PASS.** The prompt fills rows 0..P−1; the C-th (EOT) token is predicted at row P + C − 2 ≤ 1023.
- **Test set: only n 24 overflows, 29 / 1,000; n ≤ 23 all fit; max 1,085: PASS.** Means / maxima at n 16–24 reproduce the §1.7 table exactly (435.5 / 519 … 942.0 / 1,085). Seven n-24 items sit exactly at P + C = 1025.
- **Eval budget min(max C + 50, 1025 − P): PASS, with a note for E1.** The cap binds in 4 batches at n 23 and 24 at n 24. In those batches, if any row does not terminate, cot_answers' loop feeds its last generated token at row 1024. That does not crash: `wpe[1024:1025]` is empty, the forward returns a zero-length h, and the loop ends without reading it. The E1 edit should keep that property (or break before the last feed), and G-T18 should include a cap-bound batch with a non-terminating row.
- **Round-3 context bound: PASS, n 25 CORRECTED to a range.** On fresh shapes (rng 23188, 2,000 per n) I get 0.956 / **0.520** / 0.078 / 0.002 / 0.000 at n 24–28, against the draft's 0.959 / 0.551 / 0.086 / 0.003 / 0.000 (rng 23192). The n-25 values differ by 2.0 binomial SE; pooled ≈ 0.535 ± 0.011. Fix: "n 25 ≈ 0.52–0.55 fit". CTX-BOUND is unaffected.
- **Structure: PASS.** C̄ / n² = 1.489 / 1.506 / 1.507 (n 8 / 16 / 24); P̄(16) = 49.96; value share 12 / 217.7 = 5.5 % (n 12) and 16 / 385.5 = 4.15 % (n 16). Copy reach: L̄_0 39.0 → 48.0 (n 13 → 16); trained max L_0 31 is the theoretical maximum at n 8 (7 bracket pairs) and is attained (31 over 384,000 n-8 shapes); trained rows 147 = the max P + C over 384,000 n-8 shapes (PASS).
- **CORRECTED (minor) frontier values.** "T1 trains copies up to 43" and "T2 … copies ≤ 57, rows ≤ 527": those are test-set or 12,800-draw maxima. Over pool-sized draws (96,000 n-12, 64,000 n-16) the maxima are **L_0 45 / rows 313** (T1) and **L_0 59 / rows 531** (T2); the 64 longest n-16 shapes average 517 rows. Fix: "copies ≤ 45 (T1) / ≤ 59 (T2); rows ≈ 313 / 531 (pool_meta's T_max_by_n is the registered value)". "≈ 1.2–1.35 × the 147 trained rows" → **1.16–1.35 ×** (170 / 147 = 1.16).
- Test-time cost table (§3.3): PASS (386 / 606 / 868 sequential steps; 436 / 668 / 942 vs ≈ 899 / 1,365 / 1,921 latent token-positions; 21–33× steps, ≈ 0.48–0.49 of the positions).

## P5. Base and latent n50, bars, OC, noise model (§0, §1.1, §4.3, §6) [pc_oc.py → pc_oc.json]

- **n50s from the committed files: PASS.** Text base s1 (bf16, RTX 3070) n50 **10.438**, n90 9.33; s2 10.597 / 9.60. Latent D3-T s1 10.306; FT-SELF s1 15.460; R2-s1 20.539 (eval_deep acc_n2, n 13–24); D3-T s2 10.507; R1-s2 15.328. n50 / support: latent 1.288 / 1.288 / 1.284; text base 1.305. Δ_0 (bf16) = +0.13. PASS.
- **Bars: PASS.** Text bf16 instance MG_T 0.665, SB_T 0.435. The latent instance reproduces SD2's 0.63 / 0.40 / 0.785 / 0.60 from the latent curves (M2_shift 0.71725, M2_prop 0.85156).
- **N-SHIFT / N-PROP panels, independently re-derived: PASS** (Δ −1.023 / −2.101 and +0.196 / +0.336; M 0.515 / 0.749 / 0.888).
- **POS-PROP / POS-ADD, independently re-fitted on a 5× finer grid [pc_panels.py]: PASS.** x0 1.31 / w 0.076 and x0 45.6 / w 11.2 (draft 1.31 / 0.075, 45.5 / 11.0); SSE 0.00198 for both ("the base alone cannot split them": PASS). Δ +0.097 / −0.031 and −0.977 / −1.996 (draft +0.10 / −0.03, −0.98 / −2.00). With the pool-size row maxima (313 / 531): +0.14 / +0.05 and −0.94 / −1.94, so no reading changes.
- **Comparison OC by exact quadrature: PASS.** Every OC cell in §0, §6 and its POOLED paragraph reproduces to ±0.005, with no Monte-Carlo error. Examples: P(PARITY | 0 / 0) **0.915 / 0.608 / 0.385** at s 0.2 / 0.35 / 0.5; P(PARITY | −1 / −1) 0.002 / 0.035 / 0.072; band edge 0.312 / 0.501 / 0.019 / 0.168; additive LATENT-AHEAD ≥ 0.982; prior-weighted headline (s 0.35) 0.195 / 0.184 / 0.590 / 0.032; POOLED 0.987 / 0.821 / 0.601 and LATENT-AHEAD 0.994 / 0.923 / 0.841, headline PARITY 0.256 / LATENT-AHEAD 0.569.
- **Structural products: PASS.** P(both rounds readable) = 0.95 · 0.99 · 0.90 · 0.85 · 0.95 · 0.99 = **0.677**; T-SELF 0.846; T-SELF-2 0.546; rest = ¬T-ABSORB 0.094 + LABEL-STALL 0.127 + censored 0.102. Per-cell T-DBS / T-DBS-2 OC bands match ts_oc.json.
- **NOTE (bears on the OC reading and Q4 / Q5): the noise model vs the measured spreads.** Three measured between-lineage n50 differences are each one draw of Δ under no difference (equal lineage SDs): latent r0 +0.201, latent r1 −0.132, text r0 (bf16) +0.159. Their RMS gives **ŝ = 0.167 (3 df; 95 % χ² CI 0.094–0.62)**. The draft's central s 0.35 is 2.1 × ŝ, inside the CI but on the conservative side, and the data show no round-to-round growth (r1 0.13 < r0 0.20). At ŝ: P(PARITY | no difference) **0.96** (one lineage), P(PARITY | −1 / −1) 0.000, P(LATENT-AHEAD | −1 / −1) 0.98, headline PARITY 0.30 / LATENT-AHEAD 0.555 / TEXT-AHEAD 0.14. Sensitivity at s 0.35: no growth (k 1, ρ 0.5) gives 0.74; random walk (ρ 0.707) gives 0.64. So §6's "PARITY is a weak reading at one lineage (0.61)" holds only at s 0.35. Fix text: *"P(PARITY | true parity) is 0.96 at the measured in-house spread (ŝ 0.17, 3 df) and 0.61 at the conservative s 0.35; the text FT lineage noise is unmeasured."* Widening the band (Q5) is not needed at ŝ. The case for s2 rests on the unmeasured text-FT noise, not on the base-level spread.

## P6. Cost and memory (§1.2, §3.7, §8) [pc_cost.py → pc_cost.json; pc_tokens.json]

- **TF-GREEDY token-positions per item: PASS unpadded** (364 / 572 vs 365 / 575). **CORRECTED, padded:** label_reroot right-pads both chunks to the batch maximum. At label batch 128 (sorted by prompt length, as the code does) that gives **471 (T1) / 729 (T2)** positions per item, +29 % / +27 %. Pool hours become ≈ 0.75–1.3 (T1) and 1.15–2.0 (T2). Fix: "0.75–1.3 h per T1 pool, 1.15–2.0 h per T2 pool (padding included)", or sort by (prefix, suffix) length.
- **CORRECTED: eval_textself's free generation is not costed.** MEMO-GAP-T generates 2,000 pool + 2,000 gate items per window n: 16,000 at n 9–12 (T1) and 32,000 at n 9–16 (T2). On ts_cost's own latency model (fp32, 3070) that is 92k–117k sequential steps, **0.35–0.61 h (T1)**, and 265k–315k steps, **1.17–1.82 h (T2)**, about one more eval_text. Eval stages become **T1 1.45–2.98 h, T2 2.02–4.19 h** (draft: 1.5–2.7 each). Per lineage (Plan A): **≈ 10.7–20.9 h → $1.8–5.4**, or **$1.9–5.6** with the padding above (draft $1.73–4.97). Family s1 + s2 **≈ $3.8–11.2**: still under the $12 cap and the $30 rule, but the **$5.5 admission projection now binds** at the slow / $0.26 corner. Fix text, §8 rows "T1 / T2 eval_text + eval_textself": **1.5–3.0 / 2.0–4.2 h**; per lineage **11–21.5 h, $1.9–5.6**; family **$3.8–11.2 (cap $12)**.
- **Plan B CORRECTED:** cloud part (1.4–2.9 + 2.2–4.3 + 0.4–0.6 h) × $0.17–0.26 = **$0.7–2.0** (draft $0.6–1.9). Local 3070 time: the pools at the 3070 rate are 2.6 h (3.3 h padded), not 1.6–2.6, and eval_textself is missing. Local total ≈ **10–13 h per lineage** (draft 8–11).
- **"TF-GREEDY is what makes the matched exposure affordable": CORRECTED (overstated).** The 8–20 h generation figure assumes KV-cache generation at bs 32, which is latency-bound (12–16 ms per decode step). REROOT prompts are ≤ 60 rows, so generation at bs 256–512 fits in < 3 GB of KV cache and would cut that by about 4–6× (≈ 2–4 h per pool) [ASSUMED latency scaling]. TF-GREEDY is still ≈ 1.5–3× faster, and the G-T16 fallback is cheaper than "slow; Theory re-plans" suggests.
- **Training memory (§3.7): CORRECTED (minor), conclusion unchanged.** The 12.99 GiB mem_check point is a backward with no optimizer step, so it excludes AdamW's ≈ 0.93 GiB of moments. The slope is (12.99 − 0.93) / 147 = 0.082 GiB per row (not 0.0754). With pool-max rows: T1 (313) micro 2 → **14.7 GiB**; T2 (531) micro 2 → 23.7 GiB > 21.0, micro 4 → **12.8 GiB**. So the choice is still **T1 micro 2, T2 micro 4**.
- **Eval memory: PASS.** KV fp32 = 73.7 KB per token per item; at bs 32 and 1,024 rows that is 2.42 GB. With the old and new caches both live across a step, the peak is ≈ 4.8 GB + 0.5 GB weights ≈ 5.1 GiB. That is under 8 GB, and under the local 6.0 GiB memcap (Plan B). Fix G-T18: "≤ 6.0 GiB (the local LATENT_MEM_CAP_GB) if an 8 GB card is used" (draft says 7).
- **Training-time model (0.59 / 1.03 s per self step; 1.43 / 2.16 h): PASS** as arithmetic (forms ASSUMED; G-T10 decides).
- **NOTE (Q3):** the local RTX 2070 was at 100 % utilisation with 3.1 GB used at 13:21 EDT, not idle.

## P7. Filter: form-only, and equal to the latent arm's (§1.3, §1.4)

Read: boxes/selfdepth.py (check_step, label_chains, compose_E) against §1.3 and ts_label_check.py.
- **Form-only: PASS.** The text filter's only value test is VALUE-SLOT (any of ' 0'…' 10'). label_reroot / label_full take v as the model's argmax and never call N.apply (gold_values is called only for scoring, after labelling). The latent check_step's value test is `dec[hi] in VALC` (any value class). Neither compares the final answer with gold.
- **Equal to the latent filter, up to representation: PASS.** Each latent per-step flag has a text counterpart (FRAME/"other_changed" ↔ FRAME-T exact copy; SPAN ↔ SPAN-T; ORDER = the leftmost-ready handle's slots ↔ ORDER-T; PARSE / CANON ↔ PARSE-T / CANON-T; KEEP's "exactly n steps" in both).
- **Uniqueness and kept ⟺ s_k = c_k(v_k) [PROVED]: PASS.** A node is bracketed iff it has an operator whose precedence is below its parent's, or it is an equal-precedence right child. Reducing the handle removes only its own brackets, and every other bracket decision is unchanged. c_k(v) is therefore e_{k−1} with the handle's span replaced by one token, canonical by construction. Values are single tokens (asserted in nested170).
- compose_T = cot_ids(gold) iff every v_k equals the gold chain's step value; π^E ≡ π; S2-T(a) identity: PASS.
- **Precise asymmetries (fix text for §1.4):**
  1. *Terminator:* TERM-T (' =' iff an operator remains, EOT iff a single number) is a text-only form requirement; the latent frame has no terminator. Both arms reject a chain of ≠ n steps.
  2. *Copy mechanics:* the latent step is one parallel per-slot decode of a fixed-length frame (VOID + value). The text step re-emits L_k − 1 copy tokens autoregressively. This is why text REROOT fails at step 1 on long inputs (the copy-length reach) and D3-T does not.
  3. *D1 is not a labeller asymmetry.* Under REROOT both labellers read the handle's operands from a fresh root: the text prompt e_{k−1}, and the latent x = [EOT] + render(cur) + [' =']` with one pass, K = 1. A value check (apply(op, a, b) on the parsed handle) is equally available to, and equally a gold check in, **both** filters, because check_step also holds cur_tree. The D1 asymmetry is in the trained target and at test time: text targets and generation keep every operand in context, while the latent's E_self / K′ = n + 2 rollout must hold values in slots across passes. Fix §1.4: *"…would be a gold step-value check; it is equally available to the latent filter (check_step holds the parsed handle), and both arms forgo it."*
  4. *Consequence for the §1.4 [CONJECTURE]:* "π_text ≥ π_latent because a local read vs a held value" is mis-motivated, since both labels are local reads. Any π difference comes from the decoders (autoregressive copy vs per-slot), not from D1.
  5. *Test metric:* text a_T(n) is lenient (any chain ending in the right number counts). Latent a(n) reads an answer head. Both keep a chance floor, as the draft states (STRICT is reported beside).

## P8. Other tagged numbers (spot-recomputed)

- §1.2a "**148** of the 165 REROOT failures are at step 1": **CORRECTED → 149** (1 + 4 + 5 + 9 + 23 + 41 + 66 = 149 of 0 + 1 + 5 + 6 + 9 + 25 + 45 + 74 = 165; ts_label_check_s1.json fail_step_hist).
- §1.2a "the latent single step reaches about **6** operations further … (D3-T at n 20 vs text at n 15)": **CORRECTED → about 5** (D3-T 0.781 at n 20, text 0.775 at n 15; 20 − 15 = 5; SD2_v1 §2.1 table checked).
- §1.2a per-n table, π_pool 787 / 788 = 0.9987, CP one-sided lower bounds 0.976–0.985, FULL 3 / 317 vs REROOT 1 / 1,435 kept-wrong, n 17–20 deep keep: PASS against the JSONs. Note: GATE's π_n ≥ 0.99 is a point-estimate count rule (SD2_v1 c_n), so the 200-item preview *passes* it (1.000 / 1.000 / 0.995 / 1.000); the CP sentence is a comment, not a gate failure.
- §1.6 / §8 "293k–(293k + 50 × batches)" sequential steps: **CORRECTED → 290k–332k** (ts_cost.json steps_lo 289,796 / steps_hi 331,880); hours unchanged.
- §3.4(a): ε ≤ 3 / (1,000 × 95.28) = 3.15e-5 → a(11) ≥ (1 − 3.15e-5)^181.4 = 0.9943 vs measured 0.234: PASS. §3.4(d) 36,000 value tokens at n 1–8: PASS. Chance floor 0.100 / 0.088 / 0.082 / 0.040 (n 13–16): PASS.
- §2 sampling SE of n50 ≈ 0.016 / (0.3–0.5 per n) = 0.03–0.05 n: PASS.
- §1.2 generation estimate 12,250 batches × 190 / 300 steps × 12–16 (20) ms = 7.8–10.3 h / 12.3–20.4 h: PASS as arithmetic (see P6 for why it is the wrong comparator).
- §3.7 KV 73.7 KB per token per item, 2.42 GB at bs 32 × 1,024: PASS.
- §8 Plan A table sums (10.2–19.1 h, $1.73–4.97, family $3.5–10.0): PASS as arithmetic of the listed rows (see P6 for the missing rows).

## P1. TF-GREEDY: teacher-forced verification vs free greedy generation (§1.2) [pc_reroot.py → pc_reroot.json; code read]

**Proof check (exact arithmetic): PASS.** Let x be the step prompt and c = pre + [v] + suf + [term] the unique form-valid candidate (P7). Greedy decoding emits y_i = argmax p(· | x, y_<i). TF-GREEDY feeds c and tests argmax p(· | x, c_<i) = c_i at every fixed slot and ∈ NUMS at the value slot. Suppose the tests pass. Then by induction y_<i = c_<i at every i, so greedy emits exactly c. Suppose the first failure is at slot i. Then y_<i = c_<i and y_i ≠ c_i. Every fixed slot is independent of v (leaf rendering and bracket decisions do not depend on values), so no candidate c(v') is emitted and both protocols reject. The two-chunk computation is exact:
- Chunk 1 (prompt + prefix) reads the argmax at rows P − 1 … P − 1 + |pre|.
- Chunk 2 ([v] + suffix, positions continuing from len(chunk 1), right-pad keys masked by kvalid) checks suffix + terminator.
- `fwd` re-implements core.GPT2KV.forward op for op (same ln / gelu-tanh / SDPA / tied head).
- Under causal masking the logits at a row depend only on the prefix, so parallel vs cached computation is identical in exact arithmetic.

Edge cases, each checked against the code:
- **Ties:** both paths take `argmax` (first maximal index) of the same vector, so ties break identically in exact arithmetic. In fp32, flips are possible only at near-ties (measured below).
- **EOT / ' =' inside a step:** the step body contains neither token, so an early terminator is a deviation at a fixed slot and rejects in both paths. The last step needs EOT (tgt = suf + [EOT] with empty pre / suf), the others ' ='.
- **Length caps:** the REROOT budget is irrelevant, because a valid step is shorter than its prompt. The step cap n + 4 is vacuous under CV-T (each kept step removes exactly one operator). The eval budget C + 50 ≥ C always admits a form-valid chain.
- **Parse:** cur ← parse(c_k(v)) equals the shape-reduced tree, because render is canonical and the parser inverts it (the latent uses the same _parse_tree).
- **Batching:** cot_answers' equal-prompt-length batches have no padding, so rows never interact. TF's right-padding is masked.

**Measured exactness on fresh items (CPU fp32): PASS, and it covers what the draft did not.** The draft's 90 / 90 is the **FULL** protocol only (gen_check runs in the `proto == "full"` branch), and 78 of those 90 were TF-rejected items whose check is only "generation did not yield a complete valid chain". The registered labeller's code path (label_reroot, two-chunk batched) was never compared with generation. Here, independent per-step KV-cache generation (core.GPT2KV, stop at the first ' =' / EOT, validity by brute force over v, no candidate() / _render_pos) gives:

| protocol | items | TF vs GEN: keep and every value identical | TF-rejected items | same failing step | min top-2 logit margin (GEN) |
|---|---|---|---|---|---|
| REROOT n 9 / 10 / 11 / 12 | 200 each | 200 / 200 each | 0 / 0 / 1 / 0 | 1 / 1 | 7.55 / 6.49 / 2.99 / 0.40 |
| REROOT n 16 | 200 | 200 / 200 | 68 | 68 / 68 | 0.014 |
| REROOT n 20 | 100 | 100 / 100 | 89 | 89 / 89 | 0.016 |
| FULL n 10 / 11 / 12 | 200 each | 200 / 200 each (112 / 15 / 0 kept) | 88 / 185 / 200 | — | 0.010 / 0.005 / 0.0012 |

On the draft's own items (rng 23180, n 11 / 12; pc_reroot_draft.py → pc_reroot_draft.json), the draft's TF code reproduces its per-item keep and failing step on 400 / 400. Generation agrees on keep and values on 400 / 400, including all 11 TF rejections (same failing step; margins ≥ 1.2) and the n-11 kept-wrong chain (identical wrong values).

**1,700 / 1,700 fresh items agree, and 400 / 400 draft items do too** (REROOT 1,100, with 158 rejections and 942 kept chains; FULL 600). No item had a margin < 1e-3. Near-ties exist in FULL's long contexts (≈ 1e-3), not in REROOT's short prompts at n ≤ 12 (≥ 0.40).

Fix text, §1.2 fp32 caveat: *"…equal KV-cache greedy generation on 90 / 90 FULL-protocol items (draft) and on 1,100 / 1,100 REROOT + 600 / 600 FULL items (PROOFS_TS, fresh rng 23187, incl. 158 REROOT rejections)."*
**CORRECTED, G-T16 wording:** "plain KV-cache greedy generation (the eval loop)" is FULL-chain generation, and against the REROOT labeller it would disagree on most items at n ≥ 10 (FULL keep 0.07 vs REROOT 0.995 at n 11). Fix: *"REROOT generation: per step a fresh prompt [EOT] + e_{k−1} + [' =']; greedy KV-cache decode to the first ' =' / EOT, validity as CV-T; and FULL generation for LABELLER-FULL."* The GPU check (fp32, tf32 off) is still needed. CPU margins say flips need |Δlogit| < 1e-3, which is not seen at n ≤ 20 on REROOT.

## P2. REROOT / FULL keep and purity, fresh items (rng 23187) [pc_reroot.json; CIs Wilson 95 %]

| n | 9 | 10 | 11 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|
| REROOT keep, fresh | 1.000 (200) | 1.000 | 0.995 | 1.000 | 0.660 [0.59, 0.72] | 0.11 (100) [0.06, 0.19] |
| REROOT keep, draft | 1.000 | 0.995 | 0.975 [0.94, 0.99] | 0.970 [0.94, 0.99] | 0.630 [0.56, 0.69] | 0.14 [0.09, 0.22] |
| REROOT kept-wrong, fresh | 0 / 200 | 0 / 200 | 0 / 199 | 0 / 200 | 1 / 132 | 0 / 11 |
| FULL keep, fresh | — | 0.560 | 0.075 | 0.000 | — | — |
| FULL keep, draft | — | 0.540 | 0.050 | 0.000 | — | — |
| FULL kept-wrong, fresh | — | 3 / 112 | **4 / 15** | — | — | — |

- **REROOT n 9–12 ≥ 0.97 and round 1 labellable: PASS (replicated, stronger).** Fresh keep is 799 / 800. The draft's set had 788 / 800, a significant difference (Fisher two-sided p = 0.003). Pooled: 0.992 [0.986, 0.995]. The gate conclusion is unaffected either way.
- The fresh vs draft difference at n 11–12 is sampling, not code: the draft's counts reproduce exactly on its items (P1), and generation confirms both sets.
- **NOTE (§5.3 Goodhart row "its few failures are first steps on the longest inputs"): partly CORRECTED.** At n 12 the draft's failing inputs are longer than average (L_0 37–41, mean 39.7 vs 36.1). At n 11 they are not (L_0 29–35, mean 31.4 vs the n-11 mean 32.9). Round-1 selection is ≤ 3 %, but not uniformly toward shorter inputs. Fix: *"…its few failures are first steps (on longer-than-average inputs from n 12 on)…"*. KEPT-VS-DRAWN reports it anyway.
- **n 16 0.63 → pooled 0.645 [0.60, 0.69]: PASS.** "The base cannot label n 16 under keep_n ≥ 0.70" holds; the pooled upper bound is 0.69. n 20: 0.14 / 0.11, pooled 0.125 [0.09, 0.18], PASS. Failures are at step 1 in 67 / 68 (n 16) and 87 / 89 (n 20): the copy-length reach is replicated.
- **REROOT purity "1 wrong of 788": PASS (replicated).** Fresh: 0 / 799 kept at n 9–12, and 1 / 942 over all fresh REROOT kept chains (the one at n 16). Pooled n 9–12: 1 / 1,587, π 0.9994.
- **FULL keep 0.54 / 0.05 / 0.00: PASS** (pooled 0.55 [0.50, 0.60] / 0.0625 [0.04, 0.09] / 0.000 [0, 0.01]). **NOTE, strengthens the draft's FULL-is-descriptive reading:** FULL's kept chains are impure. Fresh π is 0.973 (n 10) and 0.733 (n 11). Pooled with the draft: 10 wrong of 245 kept at n 10–11 (π 0.959 [0.93, 0.98]), and 5 / 25 at n 11 (π 0.80). So FULL fails GATE's π_n ≥ 0.99 too, not only keep_n. Fix §1.2a reading 2: *"…against 10 of 245 for FULL at n 10–11 (π 0.96; 5 of 25 at n 11)."*
