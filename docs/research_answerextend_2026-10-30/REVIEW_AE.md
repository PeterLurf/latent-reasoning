# REVIEW_AE: red team of AE_draft.md (Opus subagent, 2026-10-30). Verdict: GO-with-fixes (all adopted in AE_v1.md)

1. **No hold targets exist.** Aux applies only to passes t ≤ d (inject_aw.py:271; inject_ow.py:94 / 97 assert), so `--aux_d_max` is a clean single change. Batches hold one d per step. Adopted, stated in v1 §2.
2. **BLOCKER (crash).** Under the flag a d 3–4 step has no aux item, and `torch.stack([])` at inject_aw.py:283 raises. Fix: aux := 0 when no item qualifies. Gates: flag on is bitwise equal to CTRL until the first d ≥ 3 step; a smoke through a d 3 step; gradflow with a nonzero answer-CE gradient on the write and read params at d 4. Adopted (v1 §2, §3 G2–G3).
3. **The gradient path is real.** The decoder reads S_K, with no detach, hold_inc 0, and the fade restarting as in CTRL. Noted in v1 §2.
4. **BLOCKER (vacuous d 3).** d 3 gates can pass via passes 1–2 plus the decoder's last step: CTRL d 3 is 1.000 at K′ 2. Make CAUSAL d 4-centred:
   - SWAP-MID d 4 t 2 ≥ 0.50 and ≥ floor + 0.30;
   - d 4 acc at K′ 2 ≤ 0.20;
   - LP5 drop at d 4 ≥ 0.30;
   - a new label, EXTEND-D3-ONLY.

   Adopted (v1 §4). Verified locally: rcb_plaw4_s1 / s2 d 3 at K′ 2 = 1.000; d 4 at K′ 2 = 0.008 / 0.012.
5. **WS is descriptive,** because the ':' readout is shaped by the decoder CE equally in both arms. Adopted.
6. **Scorer and protocol.** Write score_ae.py. 16k is final, with no extension. Take the learning curve from the log's by-d acc. Re-run CTRL's eval under the current eval module (sha 21ae80 vs bd515f). RTX 3090 only. CD-AN rcb_pcd4_s1 d 3 / d 4 = 0.085 / 0.090. Adopted.
7. **Add an arm AEX0** (`--aux_step 1 --aux_fade 0,0` from the same init) to test "targets only in the seed". Keep the same init with 2 data orders. Adopted. lam_at gives 0 for every step ≥ 1 (inject_aw.py:247; the loop starts at step 1).
8. **Per-depth OC.** Cost ~$1.0–1.2 per run; with AEX0 ≈ $3.5–4.5; guard $4.5. Adopted (v1 §5–§6).

**Audit's G-D points (pre-v1):** default-off bitwise vs §237's module; gradflow on mixed batches; step-0 REG d 3 / d 4 = 0.028 / 0.096. In v1 §3.
