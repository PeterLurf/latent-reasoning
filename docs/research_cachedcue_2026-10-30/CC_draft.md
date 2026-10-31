# CACHED-CUE draft: a latent loop that is cheaper than text CoT in FLOPs (C3) (2026-10-30)

**Status:** draft for red team + Literature. Theory's choice. The answer-only C4 line is parked (Note 251-11), and C3 is the next open condition with a mechanism-level fix.

## 1. Why (Note 246-0; arithmetic in scratchpad ccflops.py, to be committed as theory/c3_flops_cc.py)
- **Today's loop loses on FLOPs.**
  - Each pass reruns blocks 6–11 over the whole prompt, because DeltaMem is read at every prompt position (inject_aw.py:47-51).
  - The per-pass cue is ~11 tokens ("How many <u> does <S_t> have? The answer is:"), against ~5 tokens per text-CoT step ("Kim has 8 eggs.").
  - Executed GFLOPs at K′ = d − 1, LAT vs TEXT: d 2 18.6 vs 12.0; d 4 57.5 vs 19.1.
- **Two changes make the loop cheaper than TEXT by construction.** GFLOPs, analytic, REG TEST2 token counts, D 768, full-vocabulary head per readout:

  | d | prompt P | TEXT full chain | CC, long cue (11 tok) | CC, short cue (2 tok) | shared prefill |
  |---|---|---|---|---|---|
  | 2 | 44 | 11.3 | 11.6 | **8.7** | 7.5 |
  | 3 | 58 | 14.9 | 16.0 | **11.6** | 9.9 |
  | 4 | 71 | 18.5 | 20.4 | **14.3** | 12.2 |
  | 6 | 99 | 25.8 | 29.3 | **20.0** | 17.0 |
  | 8 | 126 | 33.1 | 38.2 | **25.6** | 21.8 |

  - **(a) Cached prompt:** DeltaMem is read and written only at cue positions. The prompt runs once through all 12 blocks with no memory, and its KV is cached. Each pass runs only the cue tokens, attending to the cached prompt KV and reading S at block 6.
  - **(b) Short cue:** " <S_t> has" (2 tokens: name + " has"). The readout is the LM head at " has" predicting v_t, the text-CoT sentence prefix ("Kim has [8]"). S is written from the " has" state, as AW writes from the ':' state.
  - The decoder is the final cue " <S_d> has" after K′ passes, reading S.
- **Short cue vs TEXT (beyond the shared prefill):** 2.1 vs 6.3 G at d 4, 3.8 vs 11.3 G at d 8, about 3× less generation compute.
- **Dominance argument:** if CC reaches ≥ 0.98 at K′ = d − 1 with FLOPs below TEXT's full chain, then at CC's budget TEXT cannot finish its chain, while CC is at ceiling. So C3-F holds without a TEXT arm.
  - The TEXT prompt is counted plan-free (shorter), which is conservative.
  - C3-S (serial layers) already favours LAT.

## 2. Arm
- **CC:** cc_s1 / cc_s2, new module boxes/inject_cc.py (inject_aw untouched).
  - GPT-2 trunk + DeltaMem at block 6 on cue positions only; prompt KV cached per item.
  - Init s233_askname_s1 (trunk) + a fresh DeltaMem with fade-in, as in P-2.
  - REG law, d ~ U{1..4}, K 8 (passes after d repeat the final cue = hold), 16k, RTX 3090.
  - Per-pass aux = CE at each pass's " has" position on v_t. This is a hand-built target, allowed here: C3 does not require C4. Disclosed.
  - hold_inc as §240 (the C6 recipe).
- **Comparators:**
  - rcb_plaw4_s1 / s2 (long cue, uncached; the same law / d / steps): accuracy and C1 gates reference;
  - CD-AN rcb_pcd4_s1 (one pass);
  - no TEXT arm (dominance, §1).

## 3. Readings (REG TEST2, per seed, binding)
- **ACC:** d 1–4 ≥ 0.98 at K′ = d − 1 and at K′ 8.
- **C1 gates (§237 / §251):**
  - d 4 at K′ 2 ≤ 0.20;
  - SWAP-MID d 4 t 2 ≥ 0.50 and ≥ floor + 0.30;
  - LP5-D4 ≥ 0.30;
  - CUE-ONLY guard;
  - STABLE.
- **C3-F (analytic, per item, from token counts):** F_CC(K′ = d − 1) < F_TEXT(full chain) for every d 1–4 item. Verified once with FlopCounterMode on each path.
- **Labels:**
  - C3-MET (ACC + C1 + C3-F, both seeds);
  - CHEAP-BUT-WEAK (C3-F holds but ACC or C1 fails);
  - FAIL.
- **Descriptive:**
  - d 5–8;
  - HOLD to K′ 64;
  - WS;
  - the learning curve vs rcb.

## 4. OC (sketch)
- P(C3-MET per seed | cue-position reads suffice to chain) ≈ 0.5; P(C3-MET | the loop needs memory at prompt positions) ≈ 0.05.
- **Risks:**
  - The step needs S at prompt positions to look up the op.
    - Counter: attention from the cue to the prompt KV carries the text; S only needs v_{t−1}.
  - The askname init is narrow to its cued format (Note 250-4), so the " has" cue may learn slower.

## 5. Cost / build
- 2 × 16k; per-pass cost is ~10× lower than inject_aw, so likely < $0.6 per run. Guard $2.
- **Build (large):** inject_cc.py, the prompt-KV cache path, cue-position DeltaMem read / write, eval parts (acc by K′, swap_kd, lp5h, cue_only, ws).
- **G-D:**
  - fp64 cache equivalence (cached vs uncached prompt, the same outputs);
  - padding invariance;
  - gradflow to DeltaMem and trunk;
  - a FLOP check;
  - an args diff;
  - a full-scale data smoke.
