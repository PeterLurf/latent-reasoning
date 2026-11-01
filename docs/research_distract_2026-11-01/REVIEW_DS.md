# REVIEW_DS: red team of DS_draft.md (Opus subagent, 2026-11-01; no model.pt loaded, CPU-only reads). Verdict: GO-with-fixes. All adopted in DS_v1.md

1. **BLOCKER:** the held-out set did not test the filter hypothesis.
   - Every training insert is add_c, with a given-name source, the given before S_1, and a dead-end subject (insert_render.py:34-44).
   - Families added:
     - H-CHAINSRC (an op reading a chain name, the key test);
     - H-GIVPOS (the given placed elsewhere);
     - H-EXTEND (stopping by name);
     - H-LIVE (a live insert chain);
     - H-OP and K3.
   - Binding: H-OP, H-CHAINSRC, H-GIVPOS.
2. **BLOCKER (reading):** the hop drop is a decoder change. Passes are intact (aux 0.997-0.967; WS ≥ 0.99). At K' 2 the decoder now partly applies the named sentence's op to the state (0.53 vs 0.13 at step 0). Drop hop[k] as a mechanism reading. Add per-pass WS for d 5-8 (WS items only cover d ≤ d_max) and the full eval set (the pilot ran 'acc' only).
3. **BLOCKER (C2):**
   - Passes 5+ are never trained to compute: d 6 needs pass 5, and rcb_s2 had d 6 hop 1 0.79 yet only 0.37 accuracy.
   - Fix:
     - d 5 ≥ 0.80 retention;
     - the d 6 ambition kept with "≥ rcb + 0.10";
     - SELECTION-CREDIT = WS passes 1-4 at d 6 ≥ 0.90;
     - P(C2 | selector) lowered to 0.2-0.25.
4. **No CTRL arm needed.** Inserts are at chance at step 0. Plain-FT d 6 drift is −0.33 to +0.03 (df_plaw4, lh_ctrl), and those DF-law runs cannot serve as direct controls.
5. **Labels, cost and RAM:**
   - a precedence order and C2-ONLY added;
   - cost ~$1.0-1.3 per cell, cell cap $1.4, guard $3;
   - STABLE: watch clip_frac;
   - host-built sets with committed shas; INS_TEXTS extended.
6. **Novelty (unverified, from memory):** Shi et al. 2023 GSM-IC; GSM-Plus; Physics of LMs 2.1 (unnecessary parameters, explicit CoT); Jia & Liang 2017 (AddSent robustness failing under AddSentMod, a precedent for the filter risk). New here: selection inside a latent loop with per-pass causal readouts. Modest.
