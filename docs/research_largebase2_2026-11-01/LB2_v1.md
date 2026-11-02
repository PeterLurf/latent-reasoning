# LARGER-BASE DISTRACT v1 (§258): the §256 recipe on GPT-2 medium (then large), staged, with a small-base bridge (2026-11-01)

**Status:** v1 after REVIEW_LB2.md (GO-with-fixes, all adopted). Supersedes LB2_draft.md. Stage-gate scorer theory/score_lb2.py, committed before the first gate read.

## 1. Why, and base choice
- §256: C2-partial on GPT-2 small. Does the recipe hold, and improve, with scale? This is a single-factor scale test.
- **Same tokenizer, data and code path:** GPT-2 medium (302M non-embedding, 24 × 1024, 16 heads) is primary; GPT-2 large (708M, 36 × 1280, 20 heads) follows only if medium IMPROVES. Qwen2.5-0.5B is a later follow-up (RoPE / GQA / SwiGLU port, digit tokenisation).
- **Cost model (non-embedding params; the loop is launch-bound):** small → medium ≈ 3.5× FLOPs, small → large ≈ 8.3×.
  - The repo's lb_cost.json puts GPT-2 large at ~10 s/step on 3090-class cards (older inject). Large S3 would be ~$10-16, too much for a first pass.
  - Medium S3 ≈ 2.5-4 s/step on a 24-48 GB card ≈ $4-6.
- **Earlier large-base attempts (§228 / §229 / §231)** predate the askname / AW recipe. The init here is trained, not assumed eager.

## 2. Stages (pilot, 1 seed; host sha asserts; host-built sets; off-host checkpoints; no C: temp tarballs)
- **B BRIDGE (small, RTX 3090, ~$1.1):** aw_plaw2_s1 → REG d ≤ 4 + --distract_p 1, 16k, in one stage (rcb and §256 merged).
  - It tests whether inserts during d 3-4 formation block the write (Note 254-1's perm effect).
  - It is the matched small comparator for S3.
  - If B fails §256's C2 gates (d 6 ≥ 0.90) or RETAIN, S3 is run in two stages (REG d ≤ 4 16k, then + distract 16k).
- **S0 INSTALL (~$1):**
  - `--base gpt2-medium` (HF keys with `transformer.` stripped; H from config); DeltaMem at layer L/2 = 12 with heads H/2 + H/2 = 8 + 8 (DH 64); the hh_* 768 hard-codes generalised.
  - Gates:
    - flags-off **bitwise equivalence on GPT-2 small** (these paths serve every model);
    - a plain-path smoke;
    - step-0 max |Δlogits| = 0 at medium.
  - **G-TIME and mem_check at the S3 shape** (REG d 4 + 2 inserts, micro to fit; --mem_budget raised).
  - **Projection rule:** if S1-S3 is projected above $14, stop and re-plan.
- **S1 ASKNAME-M:** the s233 recipe exactly (askname classes, no perm; dis's lineage used askname, and Note 254-3 shows chain training erases perm lookup anyway). 8k.
  - **LR bracket** {3e-5, 1e-4}; the winner by TEST2 d 1 and P1-ASK+NAME carries forward.
  - Gate: P1-ASK+NAME ≥ 0.98 per class and TEST2 d 1 ≥ 0.98.
- **S2 P2-M:** aw_plaw2's args (d ≤ 2, aux, chain order), 12k, at S1's LR.
  - Gate (eval fields): eval_acc by_d['2'].acc ≥ 0.90 AND eval_ws by_d['2']['2'] ≥ 0.90.
  - Formation rule: read at 6k and 12k; one +6k extension under the guard if WS pass 2 is still < 0.90 but rising; else NO-SEED, STOP.
- **S3 DIS-M:** from S2, REG d ≤ 4 + --distract_p 1, 16k (or two stages per B).
  - **Binding:**
    - RETAIN (TEST2 d 1-4 ≥ 0.98; NOT-DECODER d 4 K' 2 ≤ 0.20; SWAP4; LP5-D4; CUE-ONLY; STABLE);
    - **HOLDS:** d 6 ≥ 0.90 AND d 7 ≥ 0.80 AND eval_ws_deep d 6 passes 1-6 ≥ 0.90.
  - **IMPROVES (pre-stated, vs B and dis_s1 / s2):** d 8 ≥ 0.70 (small 0.53 / 0.38) OR d 7 ≥ 0.95.
  - **Labels:** MEDIUM-C2-IMPROVES / MEDIUM-C2-HOLDS / MEDIUM-C1-ONLY / FAIL. Each is labelled PILOT (1 seed).
  - **Seed-2 rule:** if S3 passes HOLDS, run seed 2 of S2-S3 before any scale claim.
- **Large:** a separate registration, only if S3 IMPROVES.
- **Text-CoT reference:** not run here. C2 is internal (NORTH_STAR). Same-base text-CoT (R) is a separate design if a "latent beats text at depth" claim is wanted.

## 3. OC
- P(B passes) ≈ 0.6; P(S1) ≈ 0.9; P(S2 mechanism gate) ≈ 0.7; P(HOLDS | S2) ≈ 0.6; P(IMPROVES | HOLDS) ≈ 0.35.
- **Censoring:** VOID-RERUN once per stage.
- **Disclosed confounds:** the GPU class (A6000 / L40S vs 3090) and a single seed.

## 4. Cost
- B $1.1 + S0 $1 + S1 2 × $0.7 + S2 ~$2.5 (+ extension ~$1.2) + S3 ~$5 (+$5 if two stages) ≈ $11-17; guard **$20**.
