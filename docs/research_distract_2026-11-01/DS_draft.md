# DISTRACT draft (§256): does training with irrelevant op sentences turn the slot counter into a step selector, and does that buy depth past training? (C1 scope, C2) (2026-11-01)

**Status:** draft for red team. Theory's choice on Audit's suggestion. Pilot GO (Note 255-8).

## 1. Why
- **The C1 loops count op-sentence slots** (Note 255-4). One extra op sentence before the chain collapses them (hop 1 ~0.1); inside the chain, everything from the insert on collapses; after it, the final read is disrupted.
- **The early-hop losses past the trained depth match the extra-op-sentence effect** (255-2 / 255-4). So the C2 failure may be a step-selection failure, not a depth limit.
- **Pilot** (2k from rcb_plaw4_s1, 0-2 add_c inserts per item, chain order kept), on held-out items: clean 0.992; BEFORE k2 hop 1 0.743; INSIDE k2 acc 0.986; final acc 0.93-0.99 in every insert condition (step 0 0.08-0.27).
- Unlike permutation (§253 / §254), the chain's order is kept, and the write survives.
- **Questions:**
  - (a) Does it hold for insert types never trained (held-out manipulation)?
  - (b) Are the C1 causal gates kept?
  - (c) Does it extend depth past 4 (C2)?

## 2. Arms (RTX 3090; micro 2 pinned; 16k final; rcb_plaw4's args + --distract_p 1)
| Arm | Runs | Init |
|---|---|---|
| DIS | dis_s1 / dis_s2 | rcb_plaw4_s1 / s2 |

- **Comparator:** rcb_plaw4_s1 / s2 (step 0; INSERT and d 5-8 known).
- **Attribution question for the red team:** is a CTRL arm (rcb + 16k more training without inserts) needed? Credit $17.94 vs Audit's $15 stop line.

## 3. Readings (per seed; binding)
**HELD-OUT INSERT** (new eval sets, seeds disjoint):
- inserts with ops never used in training inserts (sub_c, mul_c, add_v templates, frames 0-5);
- k = 3 (more than trained);
- positions BEFORE / INSIDE / AFTER.
- Bar: final acc K' 8 ≥ 0.90 in each held-out condition, and BEFORE hop 1 ≥ 0.60.

**RETAIN (TEST2):** d 1-4 ≥ 0.98; SWAP4 ≥ 0.50 and ≥ floor + 0.30; LP5-D4 ≥ 0.30.

**C2 (TEST2 d 5-8, K' 8):** d 6 ≥ 0.65 and d 7 ≥ 0.40.
- rcb: d 6 0.35 / 0.37, d 7 0.12 / 0.10.

**Guards:** CUE-ONLY; STABLE.

**Labels:**
- SELECTOR+C2 (all);
- SELECTOR (HELD-OUT + RETAIN, C2 fails);
- TRAINED-INSERT-ONLY (in-distribution inserts pass, held-out fail);
- NOT-CAUSAL (RETAIN causal gates fail);
- FAIL.

**Descriptive:**
- in-distribution INSERT;
- TEST-PERM;
- WS per pass on inserted items;
- hop[k] (the decoder's next-value reading);
- d 9-12.

## 4. OC (per seed)
- P(SELECTOR | the inserts teach name / value selection) ≈ 0.6; P(SELECTOR | the pilot learned an insert-type filter) ≈ 0.1.
- P(C2 bar | selector) ≈ 0.4.
- **Censoring:** VOID-RERUN once; no extension.

## 5. Cost
- 2 × 16k at ~$0.9 (insert sequences are longer) ≈ $1.8; guard $2.5. Credit $17.94 → ≥ $15.4.
- **Build:**
  - held-out insert eval sets (ops / k = 3) in the INSERT driver;
  - the d 5-8 TEST2 eval exists;
  - host sha asserts as in Note 255-6 (local RAM).
