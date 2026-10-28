# NORM-COUNTER v1: is the memory's norm the loop's step counter? ($0 causal test, 2026-10-28)

**Status:** Theory's choice under the user's standing rules (the C2 line, after §241 STEPS). Eval-only, $0, local. Readings are pre-stated before any output.

## 1. Why
- §241: d 6 fails from pass 5 (s1) / 4 (s2) at any text length.
- There is no pass embedding, so the loop can only "know" how many steps it has taken from S. The additive state's norm grows by a near-constant amount per write (REVIEW_OF: ~550 per step; 239-2: ||S|| linear in passes before §240's hold penalty).
- **H-NORM:** the loop reads ||S|| as a step counter, learned on d <= 4, so "norm of ~4 steps" means stop computing.
- **H-OTHER:** the counter is in S's direction or content (e.g. the number of separable keys), not its scale.

## 2. Interventions (on lh_fh_s1 / lh_fh_s2, TEST2-DF items, K' 8, fp32, one card / eval_bs per model)
Every rescale multiplies all heads of S uniformly, S <- S * (n_ref / ||S||), with n_ref from the same item's intact run. Direction is kept; only scale changes.
- **DOWN (d 6, 1,000 items):** after pass 4, rescale S_4 to the item's ||S_2||, then run passes 5..8 on their own cues.
- **UP (d 4, 1,000 items):** after pass 2, rescale S_2 to the item's ||S_4||, then run passes 3..8.
- **HARMLESS (d 4):** after pass 3, rescale S_3 to ||S_1||, then run passes 4..8. This checks whether a scale change by itself breaks the read.
- **Intact** baselines for d 4 / d 6 come from the same call.
- **Part `normctr`** → eval_normctr.json {arm: {acc, ws {t}, n, mean_scale}} for arms intact_d4, intact_d6, down_d6, up_d4, harm_d4.

## 3. Readings (per model; pre-stated)
- **HARMLESS:** acc(harm_d4) >= 0.90, else INCONCLUSIVE.
- **DOWN:** acc(down_d6) - acc(intact_d6) >= +0.15 and WS(pass 5) up by >= +0.15.
- **UP:** acc(intact_d4) - acc(up_d4) >= 0.15.
- **Labels:**
  - NORM-COUNTER (HARMLESS and DOWN and UP);
  - PARTIAL-DOWN / PARTIAL-UP (HARMLESS and exactly one);
  - NOT-NORM (HARMLESS, neither);
  - INCONCLUSIVE.
- Overall: the shared label, else MIXED.
- **Pre-named next:**
  - NORM-COUNTER or PARTIAL-DOWN → a scale-invariant read (normalise S before the read, or a norm-matched training augmentation), as a training lever for C2;
  - NOT-NORM → the counter is in S's content, and the next step is the state-addressing question (e.g. probes of key separability by step).

## 4. Cost
$0. Build: the eval part plus a flags-off gate (the intact arms reproduce eval_acc's d 4 / d 6 at K' 8). Audit G-D.
