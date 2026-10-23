# LAT-OW v1 (draft): a C1 test of the §184 latent recipe on English-templated word programs (2026-10-23 10:02 EDT)

**Status:** draft for Theory → red team → Literature (§9) → Audit (cap ≤ $30). Not registered. Nothing built or run except the $0 CPU checks in this folder (`lat_checks.py` → `lat_checks.json`).
- **Follows:** Note 226-2 (one pass fails d = 2), Note 227-2 (TEXT crosses d 2–5), REVIEW_OW (every latent-design fix adopted, §12), Audit's 10-23 addition (the §221 / PS2 swap battery; decodability descriptive only).
- **Task / recipe fixed by §227:** TOPO generator (boxes/ow_words.py, `order_mode topo`, `dag on`, mod 11, v1.2 layout), train frames 0–5, full fine-tuning, GPT-2 small.

## 0. Claim, scoped
- **C1 only:** "On English-templated modular arithmetic (TOPO word programs, mod 11), GPT-2 small trained with §184's latent recipe answers depths that one pass cannot learn at equal exposure. Removing the latent passes drops accuracy to the one-pass level, and a mid-chain swap of the latent state makes the answer follow the donor-state counterfactual."
  - One seed (s1); a second seed only on C1-LANG (§7, Stage 2). C6 not claimed.
  - Never GSM8K, never "language" unqualified, never "first".
- **C4: not assessed.** Generator step sentences are simulator states (NORTH_STAR C4; REVIEW_OW #10).
- **C2, C3, C7: descriptive.** C2 = d 9–12 at K′ > 8. C3 = §225-style comparator vs TEXT (§4 R-C3). C7 = step values decoded from the state (decodable ≠ used).

## 1. Which latent architecture: the §184 port (INJECT-OW), one arm
**Decision:** one latent arm, a port of §184's recipe (INJECT + read-only step-aux decoder + AUX-REWEIGHT, full FT) to OW tokens. The codi_ps CODI latent loop is not run.

| | §184 port (INJECT-OW) | codi_ps CODI loop + step decoder |
|---|---|---|
| C1 record | **met the C1 conjunction on mod-11 nested at 2 / 2 seeds** (DEEP 0.974 / 0.9997; K′0 / 1 0.094 / 0.112; SERIAL-NEST max 0.849 / 0.943 vs null 0.057 / 0.050) | none: §223 M1 ≈ DIRECT (no band there); §221 CODI-GPT2 on GSM8K CONTEXT-BOUND (cache-swap follow 0.033 vs floor 0.008) |
| language record | §207 / §208: on GSM8K, memory = scaffold, accuracy = one pass. GSM8K had no measured band. | §209 / §210: GSM8K accuracy, no serial evidence |
| cross-pass channel | the DeltaMem state S only; each pass re-reads the question with a fresh KV cache → STATE-SWAP is a clean single-channel carrier test | z inputs + latent-row KV (two channels; §221's GRANULARITY axis) |
| code | new module (adapters + OW eval); §184 kernels imported unchanged | flags exist; needs `--recode_target step` + an OW port of ps_eval |
| cost | higher (K passes re-read the question): projected 1.0–1.8 s/step on a 3090 (§6) | M1 3.16 s/step on the 4060 (223-9) |

- **Why the port.** [DERIVED] It makes the question single-factor: "does the only recipe with a C1 record survive an English surface on a task with a measured band?" A null then speaks about language, not about an untested architecture.
- **Why not both.** Two arms double LAT spend and build for an attribution question (which architecture), not the C1 question. If INJECT-OW reads NO-GAIN or fails the pilot, a CODI arm is a separate registration (named here, not pre-committed; no-loop rule).
- **Known threat carried over:** §207's question-recompute mode (memory as scaffold). The task's band (one pass fails d = 2) removes the GSM8K excuse; R2 and R3 detect the mode if it appears (§3, §5 OC).

## 2. Cells (three; one seed; one GPU class: RTX 3090)
| cell | module / flags | role |
|---|---|---|
| **DIRECT** | inject_ow `--arm direct` | one-pass, same harness, same d-sequence; fixes the band (G-W0) |
| **LAT** | inject_ow `--arm inject` + §184's args | the C1 test cell |
| **TEXT** | codi_ps `--direct --text_cot --full_ft --lr 1e-4 --d_stages 8 --excl_v2` | C3 comparator (descriptive); §227's recipe |

- **§184 args kept exactly** (s184_reweight_s2/args.json): think_k 8, steps 48,000, bs 64, lr 1e-4, warmup 100 + cosine, clip as §184, aux_step 1.0, aux_fade 48000,48000 (NOFADE), aux_reweight (AUX_W table), mem_write top, write_mix 2000 → 6000, fast_add, bf16 autocast (fp32 false), path_indep.
- **Task adapters (the only changes):**
  - think input = [EOT] + enc(" " + text) (the question ends "…have?"); answer decoder = think input + enc(" The answer is:") → the number token (" 0".." 10", single tokens);
  - aux after pass t ≤ d = [EOT, " Step", pass_tok(t), " :"] + enc(" " + R_step[t]) + [EOT] (no question; read-only; dropped at test). No target for t > d (NOFADE, no hold).
  - **d per optimiser step** (one d per batch) from rng = default_rng(seed + 22331), d ~ U{1..8} from step 0 (§184 had no curriculum). Deviation from §184's mixed-n batches, for exposure identity (D2) and no cross-d padding.
  - stream OW.Stream(22930, excl, frames 0–5, topo, dag on); excl = TEST2 ∪ PAIRS2 ∪ floor donors ∪ the §226 / §227 sets.
- **`--cache_lower`** (nested170's, exact in exact arithmetic, gate_cache_lower_n170) on for every LAT run, after an fp64 equivalence gate on OW shapes. No LAT cell is compared with an uncached LAT cell.
- **Single-change diffs:** LAT vs DIRECT = {arm, think_k, aux_*, mem_*, write_mix_*, fast_add, cache_lower}. TEXT is another harness (codi_ps); C3 only, disclosed.

## 3. G-W0: an exposure-matched band (from DIRECT, before any LAT / TEXT reading)
- **Exposure E_c(d)** = optimiser steps whose batch has depth d in cell c (logged `exposure_by_d.steps`).
- **Measured, $0 (lat_checks A; replay of codi_ps's step-draw law reproduces both logged totals exactly):**
  - TEXT (§227) reached d = 2 ≥ 0.80 at **E(2) = 2,037**, ≥ 0.95 at **E(2) = 2,508** (5,000 global steps past introduction). Once d = 2 was learned, d 3 / 4 / 5 needed only E = 326 / 492 / 181.
  - P0d DIRECT stopped with **E(2) = 3,367** (1.34× TEXT's d = 2 need) but **E(3) = 326**. So 226-2's d = 2 fail was already exposure-adequate; d ≥ 3 was untested.
- **The match (D2):** DIRECT and LAT draw the identical d-sequence (same seed and law) → E_DIR(d) = E_LAT(d) at every d, by construction. Under U{1..8} × 48k: E(d) = 6,000, ≥ 5,776 at the 0.1 % binomial bound → **≥ 2.3× TEXT's d = 2 need, 48k global steps past introduction at every d**.
- **Eligibility:** depth d may enter the band only if E_DIR(d) ≥ E_LAT(d) and E_DIR(d) ≥ **5,016** (2 × 2,508). Under the shared sequence both hold at d 1–8; the rule binds only on an engineering fault (G-EXP).
- **223-6 guards:** DIRECT TEST2 acc at d = 1 ≥ 0.80, and entry_d1 ≠ None (entry_d1 = the first 100-step window of d = 1 steps with mean number CE < 0.5). Else **NO-ENTRY**: every reading UNREADABLE, no LAT launch.
- **n\*** = the largest d in 1..8 with DIRECT acc > 0.50. **B** = {d ∈ n\*+1..8 : DIRECT acc ≤ 0.50, eligible}. B needs **≥ 3 depths, ≥ 2 of them ≥ 4** (d = 3 has no informative pairs). Fixed in a dated Note from DIRECT alone (as 170-3).
  - B invalid → **ONE-PASS-SUFFICES-OW**; no LAT / TEXT cloud spend.
  - Descriptive flag **DIRECT-STILL-RISING** if DIRECT's band-depth train acc rises > 0.10 between its last two 6k windows; R1 is then worded "at 48k".
- **Shortcut panel (TOPO, frames 0–5; lat_checks D):** every single-feature rule ≤ 0.127 at d ≥ 2 (gate ≤ 0.25). At d = 1, prod mod 11 = 0.389 is the task (a single mul).
- **Test sets (REVIEW_OW #2):** TEST2 = 1,000 per d, d 1..12, frames 0–5, seed 22960. Frames 6–7 = a descriptive transfer set (seed 22961). Pairs from seed 22970 / 22971, floor donors 22975, permutation rng 22980 (all unused in boxes/*.py and theory/*.py, grep 10-23).

## 4. Readings (LAT; pooled over B; TEST2; K′ = 8 = trained K unless stated)
- **R1 DEEP-GAIN:** acc_LAT − acc_DIR ≥ 0.30 **and** acc_LAT ≥ 0.60.
- **R2 REMOVE (to the one-pass level; REVIEW #13):** acc(8) − max(acc(K′0), acc(K′1)) ≥ 0.30 **and** max(acc(K′0), acc(K′1)) ≤ acc_DIR + 0.10. Reported vs chance too (answer max share 0.10–0.13).
  - [DERIVED, code-read of boxes/deltamem.py read_fixed] K′ = 0 is the same-harness single pass, **not** bitwise the plain trunk: zero memory gives zero reads (φ(q)ᵀ0 = 0, 0·q = 0), but `wo` has a trainable bias (zero-initialised), so the branch adds the learned constant fade·b_o at block 6. Descriptive: ‖b_o‖ and a K′0-NOBIAS arm (b_o zeroed).
- **R3 SERIAL (the §221 / PS2 battery adapted; Audit 10-23):**
  - **Pairs:** redraw donors on the recipient's skeleton (same names, unit, order, **exact token length**: 0 mismatches), t = ⌈d/2⌉, kept iff own, donor-final, cf(a) distinct **and** cf_b ∉ {own, donor-final}. Kept share 0.53–0.63 at d 4–8 → 1,587–1,887 of 3,000; cap 1,000 per d; **N ≥ 2,000** (≥ 2 band depths ≥ 4). d = 3 kept 0 / 3,000 (degenerate).
  - **Primary arm STATE-SWAP(t)** (= PS2's PREFIX-SWAP at the state granularity): run the donor t passes on its own question; replace the recipient's S after pass t by the donor's; passes t+1..8 on the recipient's question; decode with the recipient's question.
  - **follow_U** = P(pred ∈ {cf(a), cf_b}).
  - **Floor** = max(matched floor, permutation floor):
    - matched = the same swap with 3 further redraw donors on the recipient's skeleton, scored against donor 1's {cf(a), cf_b};
    - permutation = §221's: 1,000 within-d re-pairings, P(pred ∈ U of another pair);
    - ideal-carrier values (lat_checks C): matched 0.15–0.19, permutation 0.13–0.15.
  - **R3 = follow_U − floor ≥ 0.30 ∧ LEAK-SEG follow_U ≤ 0.25 ∧ SELF-SWAP bitwise.**
- **Battery map (§221 / PS2 → LAT-OW):**

| arm | LAT-OW definition | status |
|---|---|---|
| PREFIX-SWAP / STATE-SWAP(t) | memory S after pass t from the donor | **primary (R3)** |
| CACHE-SWAP | INJECT keeps no latent-position KV across passes, so mid-chain CACHE ≡ STATE [DERIVED]. Defined at the decoder: CACHE-SWAP(K) = the donor's S_8 read by the recipient's decoder | descriptive: donor-final follow; **SCAFFOLD** flag if own ≥ 0.70 (answer recomputed from the question, §207) |
| LEAK-SEG | token-level splice is vacuous: the last 7 question tokens are identical in 100 % of kept pairs (measured). Splice the donor's **lower-block (0–5) rows** at those 7 positions into the recipient's passes 1..t (memory untouched) | **guard ≤ 0.25**; exposure asserts: spliced rows ≠ recipient rows, and the changed-output share is reported |
| SELF-SWAP | donor = recipient for STATE-, CACHE-, LEAK-SEG; plus donor-own re-run | **bitwise identity gate** |
| permutation floor | §221 method on the union target | part of the floor |
| exact-length pairs | redraw keeps token length (0 mismatches over 8,872 kept pairs, d 4–8) | **pair rule** |
| MARKER-SWAP | no marker token feeds the state (no `<bot>` output) | N/A, stated |
| BUDGET-FRONT | swap at t, then b ∈ 0..8−t passes; b\* = first b with pred ∈ U | descriptive |

  - Descriptive: STATE-SWAP at every t in 1..d−1; donor-final and own shares under each arm; cf(a) vs cf_b split (where back-references are read from); MEAN-S (S_t replaced by its test-set mean at each t).
  - Skip-pass is uninformative here [PROVEN, §184 note]: under identical-input INJECT, skipping pass t = running K−1 passes. Step-specific evidence is the swap.
- **Exposure check:** E_LAT(d) / E_DIR(d) > 2 at any band depth → R1 **CONFOUNDED**, so no C1-LANG. Identity is expected (G-EXP). Basin-entry steps (per-d first 1k bin with train acc ≥ 0.80) are reported per cell.
- **R-C3 (descriptive, §225's method):**
  - (S) sequential steps: LAT K′ ∈ {0..12} vs TEXT generated tokens;
  - (F) block-position proxy: LAT F = 6·L·(K′ + 2) + 12·s (blocks 0–5 once over the question L; blocks 6–11 per pass; decoder = upper blocks over L + suffix s, lower blocks over s only, since the prefix rows are shared causally); TEXT F = 12·(P + ntok);
  - B\*(d, 0.5 / 0.9); labels C3-MATCHED-X / TEXT-AHEAD-X / MIXED-X as 225; one lineage each;
  - censoring as §225: a LAT item counts from the K′ where it becomes and stays correct (cap 12); TEXT is correct at budget b only if it terminated within b.
- **Labels, in precedence:**
  - UNREADABLE (§5);
  - NO-ENTRY (G-W0 guards, or the LAT futility stop);
  - ONE-PASS-SUFFICES-OW;
  - CONFOUNDED (exposure);
  - NO-GAIN (¬R1);
  - **C1-LANG** (R1 ∧ R2 ∧ R3);
  - DEEP-NOT-SERIAL (R1 ∧ R2 ∧ ¬R3), with sub-labels CARRIER-LEAK (follow bar met, LEAK-SEG > 0.25), OWN (own ≥ 0.50 under the swap: pause / recompute), RELAY (donor-final ≥ 0.50);
  - DEEP-UNREMOVED (R1 ∧ ¬R2).
  - Every label worded "one seed, English-templated modular arithmetic, TOPO layout, GPT-2 small".

## 5. Derivation, OC, censoring
- **A1 [MEASURED]:** a band exists in codi_ps. P0d d = 2 ≈ 0.06 (pooled d ≤ 2 0.53 with d = 1 0.998); TEXT d 2–5 ≥ 0.97.
- **A2 [ASSUMED]:** §184's mechanism transfers from token expressions to English premises. The pilot (§7) tests d = 2; d ≥ 3 is untested before cloud.
- **A3 [ASSUMED]:** one-d-per-step batches do not change §184's dynamics materially.
- **D1 [PROVEN]:** in Z_11, add / sub / mul-by-b (b ≥ 2) are bijections, so no rule that ignores S_0 or any one step's operand beats chance. Measured: max single-feature rule ≤ 0.127 at d ≥ 2.
- **D2 [DERIVED]:** identical seed + d law ⇒ identical d-sequences in DIRECT and LAT ⇒ E equal at every d. U{1..8}: E ~ Bin(48,000, 1/8) = 6,000 ± 72. TEXT (codi_ps, `--d_stages 8` = one stage = U{1..8}, same rng law, no K draw for direct cells) also draws the identical d-sequence, provided it runs with a fixed `--micro` (auto_micro pre-draws 256 items, which shifts the item stream but not the d-sequence).
- **D3 [DERIVED] swap outcomes on kept pairs:**
  - (a) serial carrier, back-references read from memory → **cf(a)**;
  - (a′) serial carrier, back-references re-derived from the visible question → **cf_b** (donor S_t exists only in the swapped state, so this is still serial use);
  - (b) relay / precompute → **donor-final**;
  - (c) unused, or pause-like extra compute → **own**.
  - The kept rule makes U disjoint from {own, donor-final}, so only (a) / (a′) pass R3.
  - d = 3 is excluded [PROVEN: the forced mul_s at t = 3 takes j = 1, so cf(a) = donor-final].
- **D4 [DERIVED]:** the floor takes the max because the next-pair donor (OW_draft) can collapse to own (excluded), deflating the floor. The OC shows a matched-only floor lets a weak carrier pass at 0.047; the max rule gives 0.
- **D5 (Goodhart):** the aux is a decodability target (it sees no question). A model that decodes S_t but answers from the question fails R2 or R3, never C1-LANG. Decode readings stay descriptive (§203 lesson).
- **Predictions [ASSUMED]:**

| event | P |
|---|---|
| pilot P-LAT2 passes LP0–LP3 | 0.55 |
| P-DIR2 d = 2 < 0.50 (one pass fails in this harness too) | 0.90 |
| band valid (n\* ≤ 3, ≥ 3 depths), given the pilots | 0.85 |
| LAT survives the 24k futility check, given the pilot | 0.75 |
| R1, given B and the futility check | 0.55 |
| R2, given R1 | 0.85 |
| R3, given R1 ∧ R2 | 0.55 |
| **C1-LANG overall** | **≈ 0.10** |

- **OC** (lat_checks E; 20,000 reps; design effect 1.3; minimum |B| = 3 × 1,000 items, 2,000 pairs; floor 0.17 from C):

| truth | P(R1) | P(R2) | P(R3) | P(C1-LANG) |
|---|---|---|---|---|
| no effect (LAT = DIRECT 0.15) | 0 | 0 | 0 | **0** |
| strong serial, cf(a) carrier | 1 | 1 | 1 | **1.000** |
| strong serial, cf_b carrier | 1 | 1 | 1 (0.000 if scored on cf(a) only) | **1.000** |
| marginal (LAT 0.64, K′0/1 0.22, follow 0.52) | 1.000 | 0.991 | 0.999 | **0.990** |
| at the bars | 0.376 | 0.335 | 0.376 | 0.094 |
| partial removal (K′0/1 0.45) | 1 | 0 | 1 | 0 (DEEP-UNREMOVED) |
| pause / extra compute (swap → own) | 1 | 1 | 0 | **0** (DEEP-NOT-SERIAL 1.0) |
| relay / precompute (swap → donor-final) | 1 | 1 | 0 | **0** |
| question-recompute scaffold (§207; K′0/1 0.70) | 1 | 0 | 0 | **0** |
| weak carrier + floor collapse (follow 0.30) | 1 | 1 | 0 (0.047 with the matched floor alone) | **0** |
| carrier-leak (LEAK-SEG 0.40) | 1 | 1 | 0 | **0** |

  - Mechanism nulls are systematic, not sampling: each gives 0 because its output lands outside U or fails a guard.
- **Censoring (rule b):**
  - dead / diverged / contaminated / ARGS-DIFF / G-EXP failure → every reading of that cell fails (UNREADABLE);
  - the LAT futility stop → NO-ENTRY, all LAT readings fail (reported, never dropped);
  - every depth is trained from step 0, so there are no unreached depths, and items are scored as measured;
  - non-kept pairs are excluded with the share reported; < 1,000 kept pairs → R3 UNREADABLE (possible only by fault: each band depth ≥ 4 gives ≥ 1,587);
  - TEXT unparseable / non-terminating / over budget / context overflow → wrong;
  - a missing file reads pending (215-7); a pilot crash is resumed, not failed.

## 6. Cost (RTX 3090 at $0.22/h; measured inputs; F2 re-measures before admission)
- **LAT rate** (lat_checks B): s184_s2 measured 0.730 s/step on a 3090 at nested lengths (batch-max think 29 tokens). OW TOPO think length is 36 / 72 / 127 tokens at d 1 / 4 / 8. A token-linear work model [ASSUMED] gives a ratio of 2.41 → **1.76 s/step uncached, 1.01 with cache_lower.**
  - Host variance up to 2× on this launch-bound loop (house record).
- **DIRECT:** one decoder pass; P0d ran 0.31 s/step on the 4060 at d ≤ 3 → ≈ 0.3–0.45 s/step at mean think length 79.
- **TEXT:** §227 measured 0.40–0.52 s/step on the 4060 at d_max 1–6 → ≤ 0.6 s/step at U{1..8}.

| cell | train 48k | eval | $ (central – 2× host) |
|---|---|---|---|
| DIRECT | 4–6 h | 0.5 h (acc) | 1.0–1.5 |
| LAT (cache_lower) | 13.5 h | 2 h (K′ sweep, battery, floors ×3, C3 grid) | 3.4–6.4 |
| TEXT | ≤ 8 h | 1 h (greedy, 12k items) | 1.5–2.0 |
| **Stage 1** | | | **≈ 6–10** |
| Stage 2 (on C1-LANG only): LAT s2 + DIRECT s2 | | | 4.4–7.9 |

- **Family cap $20:** Stage 1 $12 (DIRECT $2, LAT $7.5, TEXT $2.5), Stage 2 $8. ≤ $30, so Audit can auto-approve.
- **Admission and guards:** F2 projection × 1.3 ≤ the cell share, else no launch and an over-cap Note (re-scope, never a silent step cut). Guard marks to the final step; off-host full-state mirrors with a tested resume; drain waiters armed at launch.
- **LAT futility stop (pre-stated):** at 24k (E(2) ≈ 3,000 ≥ 1.2× TEXT's need), LAT d = 2 train acc (last 1k bin) < 0.50 → stop, NO-ENTRY. This saves ≈ 50 % of LAT on the likeliest failure.
- Early closes: pilot fail $0; ONE-PASS-SUFFICES-OW ≈ $1.5.

## 7. $0 learnability pilot (local RTX 4060, registered kernel, checkpoints / resume) and sequence
- **P-DIR2:** inject_ow `--arm direct`, d ~ U{1, 2}, 8k steps, full FT lr 1e-4 (≈ 40 min).
- **P-LAT2:** inject_ow `--arm inject` with §184's args, d ~ U{1, 2}, 8k steps, cache_lower. Projected 0.89 s/step on a 3090 → ≈ 2.5–4 h on the 4060 (F2 measures).
  - The same d-sequence as P-DIR2; E(2) = 4,000 (≥ 3,861 at 0.1 %) = 1.6× TEXT's d = 2 need.
  - AUX_W is used as §184 (passes 1–2 only; disclosed).
- **Gates (P-LAT2):**
  - **LP0:** after the first window, no 100-step mean gn_preclip > 10, no non-finite row.
  - **LP1 (the language is read):** d = 1 train acc in the last 1k bin ≥ 0.90.
  - **LP2 (the state carries step 1):** aux value-token acc at pass 1, last 1k bin ≥ 0.80 (a readiness check, not a reading).
  - **LP3 (binding, composition):** TEST2 d = 2 acc at stop (1,000 items, K′ 8) ≥ 0.70 **and** ≥ P-DIR2's d = 2 + 0.30.
  - Descriptive: d = 2 at K′ 0 / 1; per-pass aux acc; wrap split.
- **P-DIR2 check:** d = 2 ≥ 0.80 → **ONE-PASS-IN-HARNESS**: no cloud, Note (the band would be harness-specific). Otherwise descriptive.
- **Outcomes:**
  - LP0–LP3 pass → args-freeze Note → cloud DIRECT → G-W0 Note → cloud LAT + TEXT (parallel hosts) → evals → score_lat.
  - LP3 fails → **LAT-NOT-LEARNABLE-D2**; stop with no cloud spend. A CODI-arm registration may be proposed separately.
  - LP1 fails → harness fault, or language not read under INJECT. Diagnose once (a G-D'd fix), never retune on the pilot.
- **Why U{1, 2}:** d = 2 is the hurdle. TEXT needed E(2) ≈ 2.5k, then d ≥ 3 in ≤ 500 each. Short sequences keep the pilot short-local. The pilot does not test U{1..8} mixing (risk 2).

## 8. Code changes and gates
1. **boxes/inject_ow.py (NEW).** Imports nested170 / deltamem / s5 / fsq120 read-only.
   - Class InjectOW(InjectN) overrides only think_inputs / dec_logits / aux_logits for token lists (adapters in §2).
   - Arms direct / inject; d per step (rng seed + 22331); TEST2 / PAIRS2 / floor-donor construction via OW.Stream with frames 0–5 (no generator edit; ow_words sha unchanged).
   - Logs: exposure_by_d, per-d 1k-bin acc / CE, per-pass aux value acc, per-1k (d, item-sig) hash, gn_preclip, entry_d1.
   - Full-state checkpoints; `--stop_at`; `--args_only` / `--args_ref`; `--mem_check` at d 8 / think length 154 / K 8.
2. **boxes/inject_ow_eval.py (NEW).** Parts:
   - acc (d 1..12, K′ 8);
   - kprime {0, 1, 2, 4, 8, 12, d + 2};
   - mean_s;
   - swaps (STATE-SWAP at ⌈d/2⌉ and every t; matched floor × 3; permutation floor; LEAK-SEG-H; CACHE-SWAP(K); SELF; donor-own);
   - budget_front;
   - decode (aux from S_t, descriptive);
   - c3 (per item L, s, answers at each K′).
   - Every per-item output is stored (store-items rule); fp32 eval, on host.
3. **boxes/codi_ps.py:** `--excl_v2` (default off) adds TEST2 / PAIRS2 / floor-donor signatures to excl.
4. **boxes/ow_text_eval.py (NEW, small):** loads a codi_ps `--text_cot` run; greedy-decodes TEST2 with codi_ps.decode_eval's code path; stores per-item tokens, ntok, answer, parsed, terminated (for R-C3).
5. **theory/score_lat.py (NEW):** G-W0, R1–R3, labels, R-C3, exposure. Fixtures (synthetic carriers a / a′ / relay / own / leak) before any output; field-path check against each module (scorer field-path rule).

- **Gates (all before any pilot):**
  - Audit G-D on 1–5;
  - **flags-off bitwise:** codi_ps with `--excl_v2` off = the §227 commit, ≥ 50 steps, same card / seed, for the DIRECT / TEXT / latent paths;
  - **port-kernel identity:** inject_ow's write / think / dec / aux on a nested batch rendered to token lists = nested170 `--arm inject` with §184's args, bitwise, 50 steps fp32 (only the adapters differ);
  - cache_lower fp64 equivalence on OW shapes;
  - per-term gradflow (answer CE, aux CE: nonzero grad on the memory branch and trunk; no detach; AUX_W asserts);
  - plain-path smoke (arm direct, construct + 1 step, no new flags);
  - eval known-answer stubs (SELF bitwise; synthetic carriers → expected labels; LEAK-SEG exposure asserts);
  - **G-EXP:** identical d-hash in DIRECT / LAT / TEXT logs (binding); item-hash identity descriptive;
  - F2 memory / timing at the deepest shape on the 4060 and the 3090;
  - frozen args + single-change diffs.

## 9. Novelty sentences for Literature (no "new" / "first" before the check)
- **N1:** a latent-recipe C1 test on English-templated multi-step word programs (paraphrase frames, names, distractors, mod 11): one-pass vs latent at identical per-depth exposure, removal to the one-pass level, and a mid-chain state swap scored on the donor-state counterfactual.
  - Expected narrow. Cite: Lin 2503.07604; Feng 2305.15408; CODI 2502.21074; Coconut; SIM-CoT 2509.20317; Deng 2405.14838; internal §184 / §221 / §223 / §227.
- **N2:** the union counterfactual target {cf(a), cf_b}, with a skeleton-matched redraw floor, as the guard separating serial state from relay, precompute and pause compute when the question stays visible.
  - Cite: SCIT 2608.27265; Liang & Pan; internal §221 CARRIER / §223 cf(a).
- **N3 (fact check, not novelty):**
  - is exposure-matched one-pass vs serial (equal per-depth steps) used in Lin or Physics-of-LMs 2.1 (2407.20311)?
  - has a recurrent-memory latent loop (DeltaNet-style state) been tested for causal serial use on templated word problems?

## 10. Open questions for the red team
1. **§207's recompute mode:** INJECT re-reads the question every pass. Can a model pass R1 / R2 while R3 reads OWN, and is DEEP-NOT-SERIAL then the right label (vs a capacity story)?
2. **Uniform d from step 0** (§184's law) vs a curriculum on language. The pilot tests U{1, 2} only. Is the 24k futility rule fair to a slow U{1..8} learner?
3. **One d per batch** vs §184's mixed-n batches: is A3 plausible? Does it interact with AUX_W (per-item mean)?
4. **K = 8 fixed with d ≤ 8:** zero slack at d = 8 (as §184). Should K be 10, given the deeper OW sentences, at the cost of the single-factor port?
5. **LEAK-SEG-H** (lower-block rows at the final-question span): is it exposed and a fair §221 analogue? Alternative: splice the donor's value-token rows.
6. **Floor = max(matched × 3, permutation):** conservative enough or too conservative (union size 1.5)?
7. **R2's one-pass level = DIRECT + 0.10:** right tolerance, given K′ 0 is off-distribution for LAT?
8. **The aux target carries name + unit** (R_step), not just the value. Does that matter for the swap (names equal in pairs) or for decode readings?
9. **TEXT in codi_ps** (another harness, HF model, BOS / `<bot>` prompt): acceptable for a descriptive C3, or should TEXT move into inject_ow?
10. **Sequencing:** sequential DIRECT → G-W0 → LAT saves ≈ $0.9 in expectation but costs ≈ 6 h of wall time. Is a sealed parallel launch preferable?
11. **cache_lower:** is the fp64 gate on OW shapes enough, given the "never compare cached vs uncached" rule (LAT s1 / s2 both cached)?
12. **P0d's d = 2 was already exposure-adequate** (E 3,367 > 2,508). Does that change how 226-2 / 227-2 should be cited?

## 11. Main risks
1. **Language INJECT may not compose** (no language C1 record; §207 scaffold mode). Mitigation: the $0 pilot LP3, then the futility stop.
2. **U{1..8} mixing untested on language;** the pilot covers d ≤ 2 only.
3. **Cost uncertainty:** the token-linear model plus host variance gives LAT $3.4–6.4. F2 admission rule.
4. **Engineering:** a new module plus a new eval with swap hooks. Port-kernel identity, G-D, stubs, the field-path rule. No module edits while runs are live (lazy-import skew).
5. **Wrap semantics:** most statements are false in ordinary English (REVIEW #7). The wording is fixed: "English-templated modular arithmetic".
6. **One seed;** Stage 2 adds s2 only on C1-LANG.

## 12. REVIEW_OW fixes adopted
- **#1:** R3 on cf(a) ∪ cf_b, with the kept rule.
- **#2:** frames 0–5 primary (TEST2); frames 6–7 descriptive.
- **#3:** the cheap check comes first (pilots, then DIRECT / G-W0, then LAT).
- **#4 / #5:** superseded by §226 / §227 (full FT).
- **#6:** the architecture choice (§1).
- **#7:** wording.
- **#8:** no CurGate in LAT / DIRECT (uniform d); TEXT's single stage is disclosed.
- **#9:** matched redraw floor × 3, max with the permutation floor.
- **#10:** C1 only.
- **#11:** the mechanism-null OC; N_PAIRS is the realised minimum of 2,000.
- **#13:** R2 to the one-pass level.
- **#14:** binding d = 2 test readout (LP3).
- **#15:** measured-rate projection, the over-cap rule and the futility stop.
- **Censoring add:** exposure check and per-d exposure logging.

## Files (this folder; $0, CPU, ≤ 2 threads)
- `lat_checks.py` → `lat_checks.json`: (A) exposure replay (validated against the logged totals); (B) lengths and cost; (C) pairs and floors; (D) shortcut panel; (E) OC.
