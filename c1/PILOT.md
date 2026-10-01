# THEORY §67 pilot log (seed 99, not a panel seed; every configuration tried, with its outcome)

Started 2026-10-01 18:4x EDT. Common to all: 2000 updates, bs 512, AdamW (wd 0), fp32, curriculum lengths
{1}, {1,2}, {1,2,4}, {1,2,4,8} in 4 equal stages; hard-execution eval at N 8–128 (2,000 per depth).
Audit's conditions (18:5x):
* log every configuration;
* freeze ONE init rule before the panel and apply it to every GPT-2 arm (d1, reset, FP); TAB keeps its default;
* at most 4 further attempts after configuration 6, then stop and report "GPT-2 cell fails rule acquisition".

| # | arm | configuration | outcome |
|---|---|---|---|
| 0 | d1 | lr 1e-4 (all params), default heads, codes = shared mean + 0.02 noise | eval OOM (10,000-row null batch); voided `pil_d1.void_evaloom`; training plateaued at loss 0.786 (L = 1), acc ≈ 0.18–0.30 |
| 1 | tab | lr 1e-2 | **learns**: certified (0 transition errors), 1.000 at N 8–128 by update 1,500 |
| 2 | d1 | lr 1e-4, lr_new 1e-3 | loss 0.786 at L = 1 → 1.49; acc 0.15–0.18 (chance) |
| 3 | d1 | lr 1e-4, lr_new 1e-2 | loss → ln 6; acc chance |
| 4 | d1 | zero-initialised heads | loss exactly ln 6 throughout (symmetric saddle: identical codes + zero heads) |
| 5 | d1 | small heads (initial logit std ≈ 0.5) + distinct codes (N(0, 0.14²)) | acc 0.32–0.34 at every N 8–128 (no decay), transition errors 26 → 15; not certified |
| 6 | d1 | small heads only | acc 0.16–0.18 by update 1,500 (chance) |

**Diagnosis.**
* GPT-2's last hidden state has norm ~200, with one outlier dimension (|h| ≈ 194, the "massive activation"). Op-to-op
  differences are ~10.
* Default heads therefore start at logit std ~5 and saturate: configuration 2's trained cell maps 4 of 6 ops from code 0
  to one code with p = 1.0.
* With nearly identical codes, the decoder cannot tell codes apart, a symmetric saddle (configuration 4).
* Configuration 5's flat 1/3 at every depth is consistent with a stable automaton that tracks only the sign of the
  permutation (S3 → Z2): it then picks among 3.

**Remaining attempts (at most 4):**
* 7: configuration 5 with 6,000 updates.
* 8: LayerNorm on the cell output before both heads (Audit) + distinct codes, default heads.
* 9 and 10: reserved.
