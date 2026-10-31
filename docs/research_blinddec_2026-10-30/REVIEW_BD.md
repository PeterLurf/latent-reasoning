# REVIEW_BD: red team of BD_draft.md (Opus subagent, 2026-10-30). Verdict: NO-GO as drafted. Theory: WITHDRAWN (Note 251-9)

1. **BLOCKER:** the premise is wrong.
   - In the failing AO seeds the shortcut never formed either: d 2 is at chance at every K′, and s3 / s4 ~0.2 equals K′ 0.
   - The failure is the onset break of pass 1's d 2 write: ws_readout["2"]["1"] at 1k is 0.22 / 0.09 / 0.16 / 0.27, and d 1 stays at 1.0.
   - The only AO formation (ao_s2_ext) used the shortcut (K′ 1 0.88). A blind decoder removes the only route that ever worked.
2. **BLOCKER:** two changes. Under a blind decoder, d 1 items become a blind S → v_1 target on pass 1, i.e. pass-1 aux in disguise. The aux prompt is already blind (inject_ow.py:92-103).
3. **BLOCKER (gates):**
   - SWAP-MID d 2 t 1 does not exist at d_max 2 (part_swap_kd loops over d 3..d_max), and swap_1 has no floor.
   - SWAP-K / STATE-USE, MEAN-STATE and CUE-ONLY are vacuous under a blind decoder.
   - LP5 is implied by acc_kprime["2"].
   - ACC2 at K 8 allows a delayed chain.
4. **BLOCKER (build):** the flag would not reach eval automatically. There are ~14 tok_aw call sites across training and eval, and the tok_aw assert at inject_aw.py:70 would fire.
5. **Minor:** the AO base rate at 12k is 0/2, and the only formation came at 18k with a re-warm. BLD-AUX only checks wiring (OC ~0.95).
6. **Minor:** the OC is ungrounded. The 0/2 outcome has a likelihood ratio of only ~3, and the censoring rules are missing.
7. **Cheaper first:**
   - (a) a $0 probe of whether S_1 stores v_1 on d 2 items in the AO models;
   - (b) a local 2k pilot reading ws d 2 pass 1 at 1k / 2k.
8. **Minor:** pin --micro 1 (P-2 peak 14.4 GiB); G-D bitwise default-off plus a plain-path smoke.
