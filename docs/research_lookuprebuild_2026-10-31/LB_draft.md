# LOOKUP-REBUILD draft: a C1 loop that finds its steps by name (rebuild from the name-lookup init) (2026-10-31)

**Status:** draft for red team. Theory's choice after Note 252-5.

## 1. Why
- **The C1 loops are position-bound** (Note 251-16): TEST-PERM d 2–4 is 0.15–0.20, and pass 1 is 0.10 although the cue names S_1.
- **The cause is the init's training data** (Note 252-4): s233_askname's one-hop classes always had the asked sentence first.
- **Retraining the init with permuted texts** (s252_askperm_s1, Note 252-5) gives one-hop name lookup: PERM 0.988, CHAIN 1.000, D1 1.000.
- **Fine-tuning a positional loop on permuted text did not re-route it** in 2k (§252 pilot: pass 1 stays at 0.10).
- **Question:** does a loop built from the lookup init, along §237's path with permuted training text, chain by name? (C1 scope: plan given, text in any order.)

## 2. Stages (as §237's path; REG / TOPO laws as there; perm_p 0.5 throughout; RTX 3090; micro pinned)
**Stage 1, P2P** (p2p_s1; one seed, as aw_plaw2_s1 was):
- aw_plaw2_s1's args with `--init_trunk s252_askperm_s1/model.pt --perm_p 0.5`; d ≤ 2; aux on; 12k.
- **Gate to Stage 2:**
  - TEST2 d 2 ≥ 0.90;
  - TEST-PERM d 2 ≥ 0.80, with the law-matched TEST-PERM renderer;
  - WS d 2 pass 1 on TEST-PERM ≥ 0.80.
- If it fails: STOP (no Stage 2), label NO-SEED.

**Stage 2, LBR** (lbr_s1 / lbr_s2):
- rcb_plaw4_s{1,2}'s args with `--init_model p2p_s1 --perm_p 0.5`; REG; d ~ U{1..4}; aux on; 16k final.

**Args diff vs the existing path:** init (s252_askperm) and perm_p.
- Disclosed: two factors differ from rcb (init + permuted training).
- The §252 NP pilot (rcb init + perm, 2k) is the partial control for "perm training alone". A full NP 16k control is optional, if budget allows after the result.

## 3. Readings (Stage 2, per seed; binding)
**On TEST-PERM:**
- ACC d 2–4 ≥ 0.90 (K′ 8);
- NOT-DECODER d 4 K′ 2 ≤ 0.20;
- LP5-D4 ≥ 0.30.

**On TEST2 (RETAIN, 252-1):**
- d 1–4 ≥ 0.98;
- SWAP4 ≥ 0.50 and ≥ floor + 0.30;
- LP5-D4 ≥ 0.30.

**Guards:** CUE-ONLY (named); STABLE.

**Labels:**
- VOID / UNSTABLE;
- NOT-FORMED (TEST2 d 2 < 0.90);
- **LOOKUP-C1** (all binding);
- ORDER-ONLY (RETAIN passes, TEST-PERM fails);
- PARTIAL (TEST-PERM d 2 ≥ 0.90, d 3–4 fail).
- Per arm: PASS / SPLIT / FAIL.

**Descriptive:**
- pass-1 WS on TEST-PERM;
- a probe of S_t for spine identity and pass index (does the counter weaken?);
- the add_s-ambiguity stratum;
- TEST-PERM by number of inversions;
- d 5–8.

## 4. OC (per seed)
- P(LOOKUP-C1 | lookup survives loop training) ≈ 0.5; P(LOOKUP-C1 | a positional route re-emerges) ≈ 0.05.
- P(Stage 1 gate | lookup survives) ≈ 0.8.
- **Censoring:** VOID-RERUN once on crash or unfinished run; Stage 2 only after the Stage 1 gate; no extensions.

## 5. Cost
- Stage 1, 12k at d ≤ 2: ~$0.5–0.7. Stage 2, 2 × 16k: ~$1.6–2.3.
- Total ≈ $2.5–3; guard $4. Credit $25.7.
- **Build:**
  - --perm_p on the TOPO law (Stage 1) as well as REG;
  - a TEST-PERM renderer for the TOPO law (Stage 1 gate);
  - eval parts as §252 (test_perm acc / ws / lp5).
