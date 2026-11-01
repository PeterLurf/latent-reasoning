# DISTRACT v1 (§256): does insert training turn the slot counter into a step selector, and does that buy depth past training? (C1 scope, C2) (2026-11-01)

**Status:** v1 after REVIEW_DS.md (GO-with-fixes, all adopted). Supersedes DS_draft.md. Scorer: theory/score_ds.py (fixtures 8/8), committed before any output.

## 1. Why
- The C1 loops count op-sentence slots (Note 255-4). The pilot (Note 255-8; 2k from rcb_plaw4_s1, 0-2 add_c inserts) restored the final answer under every in-distribution insert condition (0.93-0.99).
- **Every training insert shares surface features** (insert_render.py): add_c; source Y is a given name, never a chain name; Y's given sits right before S_1; subject X is a dead end.
  - So a surface filter ("skip ops whose source is a fresh given / whose subject is never read") could explain the pilot. The held-out families below are built to break that filter.
- **The pilot's clean hop 1 / 2 drop is a decoder change.** Passes stay intact (aux 0.997-0.967; WS clean d 4 ≥ 0.99). At K' 2 the decoder now partly applies the op of the sentence the question names (0.53), rather than the next slot's op.

## 2. Arms (RTX 3090; micro 2 pinned; 16k final; rcb_plaw4's args + --distract_p 1; full eval set incl. swap_kd, lp5h, ws, cue_only)
| Arm | Runs | Init |
|---|---|---|
| DIS | dis_s1 / dis_s2 | rcb_plaw4_s1 / s2 |

- **No CTRL arm (REVIEW_DS #4):**
  - inserts are at chance at step 0, and plain training has no route to insert robustness;
  - plain-FT drift at d 6 so far is −0.33 to +0.03, so a C2 gain counts only if ≥ rcb + 0.10 and backed by the WS mechanism.

## 3. Readings (per seed; binding)
**HELD-OUT INSERT** (host-built, set shas committed from a clean build; texts disjoint from training; frames 0-5 only). Each binding family needs acc K' 8 ≥ 0.90 and min over positions ≥ 0.80:
- **H-OP:** sub_c / mul_c / add_v inserts.
- **H-CHAINSRC:** "X has c more than S_j" after S_j (j = 1..d−1), with no new given. Pass j+1 then sees two sentences reading S_j, and only the cue S_t disambiguates.
- **H-GIVPOS:** the new given at the text start, right before its op, or after S_d.
- **Descriptive:** H-EXTEND (X1 reads S_d, X2 reads X1: looks like a d+2 chain; tests stopping by name), H-LIVE (a two-step insert chain), K3 (3 inserts).

**RETAIN (TEST2):** d 1-4 ≥ 0.98; SWAP4 ≥ 0.50 and ≥ floor + 0.30; LP5-D4 ≥ 0.30.

**C2 (TEST2, K' 8):**
- d 5 ≥ 0.80 (retention);
- **AMBITION:** d 6 ≥ 0.65 and ≥ rcb same seed + 0.10 (rcb 0.35 / 0.37);
- **SELECTION-CREDIT:** WS passes 1-4 at d 6 ≥ 0.90 each (new host eval eval_ws_deep, d 5-8, also run eval-only on rcb_plaw4_s1 / s2).

**Guards:** CUE-ONLY; STABLE. The pilot had clip_frac 0.37-0.56 vs rcb's 0.21, so watch it.

**Labels (first match):**
- VOID / UNSTABLE;
- NOT-CAUSAL;
- SELECTOR+C2 (HELD-OUT + d 5 + AMBITION + SELECTION-CREDIT);
- SELECTOR;
- C2-ONLY;
- TRAINED-INSERT-ONLY (in-distribution acc ≥ 0.90 in every condition);
- FAIL.
- Per arm: "SELECTOR k/2, C2 k/2".

**Descriptive:**
- TEST-PERM, pre-read now: a rise means free lookup; no rise is consistent with an in-order scan matching names;
- WS d 5-8 per pass, and P(WS pass 5 = v_5 | pass 4 right) at d 6;
- acc at K' = d−1 and the plateau over K' 8-14;
- the decoder-op reading at K' 2.

## 4. OC (per seed)
- P(SELECTOR | inserts teach name / value selection) ≈ 0.6; P(SELECTOR | the pilot learned a surface filter) ≈ 0.1.
- P(SELECTOR+C2 | selector) ≈ 0.2-0.25. Passes 5+ are never trained to compute; at d ≤ 4 they hold.
- **Censoring:** VOID-RERUN once; no extension; 16k final.

## 5. Cost and RAM
- The pilot ran 1.09 s/step, so 16k is about 4.8 h plus the full and new evals: ~$1.0-1.3 per cell. Cell cap $1.4, guard $3.
- Credit $17.94 → ~$15.5 expected, ~$14.9 worst case (Audit's stop line is $15).
- **RAM safeguards:**
  - host sha asserts on init, module, eval and render scripts against the git blobs (Note 255-6);
  - held-out and WS-deep sets built on the host, with their set shas asserted against shas committed from a clean build;
  - INS_TEXTS disjointness extended to every new set.
