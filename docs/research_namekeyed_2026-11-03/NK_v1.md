# NAME-KEYED MEMORY v1 (§262): name-bound value transport in the loop, tested on the passes it can affect (C1 scope) (2026-11-03)

**Status:** v1 after REVIEW_NK.md (GO-with-fixes; 4 blockers, all adopted). Supersedes NK_draft.md. Pilot gate theory/score_nk_pilot.py (fixtures 6/6), committed before output. Literature: NARROW (EntNet-style name-keyed memory inside a latent reasoning loop; also check Entities-as-Experts / Mention Memory).

## 1. What changed from the draft, and why
**REVIEW_NK's $0 read** (dch_s* test_perm items):
- pass 1 is purely positional (1.000 when S_1 stays in op-slot 1, 0.31-0.38 when moved; d 2 always moved → 0.10);
- passes >= 2 are order-bound even after a correct previous pass (S_t moved: d 3 pass 2 0.27 / 0.20, d 4 pass 4 0.25 / 0.19; unmoved 0.78-0.91).

So step **selection** fails at every pass, and NK addresses only value **transport**. v1 therefore:
- (a) gates on the passes NK can affect, with a conditional, causal measure;
- (b) adds the one-hop name-lookup co-training that retrained pass-1 lookup (Note 252-5: 0.988) as a **shared** change in both arms;
- (c) scopes the claim: NK supplies values by name; selection must still be learned.

## 2. Mechanism (fixes adopted)
- **Packed state:** NK state blocks are packed into S as extra blocks (the §185 pack_state precedent), so every consumer of S sees them: swap_kd, the lp5h mean-at-t, the hold penalty / norm guard, halt_inc and the ||S|| readouts. read_fixed's existing [:H_ADD+H_DEL] slice is untouched (audited in G-D).
- **Tied keys:** one projection gives k = q = normalize(W_n · e(name)).
  - One head, DH 64. With 40 names (nested_words.py:33) the learned keys can be exactly orthogonal.
  - [DER] own match ~1, so crosstalk can go to 0.
  - [ASM] an unwritten name reads crosstalk, not 0 (disclosed).
- **Name tokens:** the global set of 40 single-token ids (incl. inserted side names). The write key is the cue's name token, asserted to equal spine[min(t, d)] (the plan is still given).
- **Value:** block-11 h at ':' (pass_aw), through ln_w. Delta-rule write (erase-then-write; idempotent across hold passes).
- **Read:** at block 6, at name-token positions, + W_on · S_n q_p. W_on = 0 at init with **no fade buffer** (a zero fade would kill W_on's gradient), so the model is function-preserving.
- **Ids threaded** through read_fixed, fwd_read (decoder and aux; the aux's teacher-forced R_step[t] starts with S_t's name), ws_readout, eval passes and read_mask. NK asserts ids on every call.
- **Text:** perm-then-insert (perm_p 0.5 applied before the §257 inserts; inserts placed relative to the S labels).
  - This removes the `assert not (distract_p and perm_p)` conflict (inject_aw.py:278).
  - Both disjointness asserts are kept (INS_TEXTS, TP_TEXTS).
- **Shared co-training:** askperm-style one-hop items (ST_v1 classes i-iv, permuted with p 0.5) mixed at 25% of items in both arms.

## 3. Arms (from bun_s1 / s2; the bun recipe; RTX 3090; micro pinned; host sha asserts)
| Arm | Runs | Change vs bun |
|---|---|---|
| NK | pilot_nk_s1, then nk_s1 / nk_s2 | + NK heads + perm-then-insert + one-hop co-training |
| CTRL | pilot_nkc_s1, then nkc_s1 / nkc_s2 | + perm-then-insert + one-hop co-training (no heads) |

## 4. Pilot gate (seed 1; --steps 8000 --stop_at 2000; host eval at 2k → eval_nk_pilot.json)
- **Measure:** cond_ws_perm[d][t] = P(pass t correct | pass t − 1 correct) on permuted d 3-4 items, t >= 2. NK also runs with its read ablated (W_on r → 0).
- **Rule (score_nk_pilot.py), pooled over d 3-4, t >= 2:**
  - GO if NK >= 0.60, NK >= CTRL + 0.20, and NK − NK-ablated >= 0.20 (the binding is causally used);
  - PARK if NK < 0.30 or the ablation drop < 0.05;
  - else NOTE.
- Pass-1 permuted WS is reported for both arms (it shows the effect of the co-training), not gated.

## 5. Full stage (on GO; scorer committed before launch)
- 4 cells × 8k.
- **Binding:**
  - ORDER-FREE: NK TEST-PERM d 2-4 >= 0.85 and >= CTRL + 0.20;
  - NK ablation drop >= 0.20 on TEST-PERM;
  - the name-swap test (swap slot(name): only readers of that name follow) follow >= 0.5 and >= floor + 0.3;
  - RETAIN: TEST2 d 1-4 >= 0.98, NOT-DECODER (weaker evidence here: the decoder can copy S_d's slot; disclosed), SWAP4, LP5-D4;
  - C2: d 6 / d 7 >= dch − 0.05 and WS d 6 passes 5-6 >= 0.90;
  - C6: K' 64 / 128 hold at d 1-6, with the ||S|| ratio incl. NK blocks.
- **Descriptive:** C5 r = 3; the insert families; CTRL vs bun.

## 6. OC
- P(GO) ~0.15 (REVIEW_NK: selection is the binding failure; NK helps only transport).
- P(full ORDER-FREE | GO) ~0.4.
- The co-training effect on pass 1 is measured in both arms either way.
- **Censoring:** VOID-RERUN once.

## 7. Cost
- Pilot 2 × ~$0.35; full ~$6-8. Guard $15.
- **Build (large):**
  - the NK module packed into S;
  - ids threading;
  - perm-then-insert;
  - one-hop co-training items;
  - eval_nk_pilot (conditional WS, ablation).
- **G-D:**
  - flags-off bitwise;
  - bitwise step-0 with heads (fp64 incl. hold_step norms);
  - gradflow into W_n / W_vn / W_on;
  - the ids assert;
  - disjointness;
  - a slice audit.
