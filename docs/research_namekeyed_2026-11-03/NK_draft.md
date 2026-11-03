# NAME-KEYED MEMORY draft (§262): binding slots keyed by entity names (C1 scope: order-free; C2) (2026-11-03)

**Status:** draft for red team. Audit G-D'd in principle (docs/ARCH_UPGRADES_2026-11-03.md, candidate 1). Literature: NARROW ("EntNet-style name-keyed memory inside a latent reasoning loop"; EntNet 1612.03969, KV-MemNN 1606.03126, MemN2N 1503.08895, NTM 1410.5401, binding IDs 2310.17191).

## 1. Why
**Every C1 loop is order-bound** (TEST-PERM ~0.1; Notes 251-16, 253-4, 254-3). Three training-only routes failed:
- perm FT on a positional loop (§252);
- a lookup init, where perm blocks the write and chain order re-teaches position (§253 / §254);
- plan-free cues, where the loop follows forward reachability (§260).

**Diagnosis:** the additive DeltaMem read is keyed by the cue / position context, not by the entity being read.

**Change:** add memory heads whose keys are entity names, so that a sentence receives each entity's stored value at that entity's name tokens, wherever the sentence sits.

## 2. Mechanism (exact)
**Name tokens:**
- REG spine / given / distractor names are single GPT-2 tokens (4800/4800 spine names, REVIEW_CC #3); the build asserts this per item.
- e(n) = GPT-2 wte row of name token n (frozen copy at init).

**New heads NK** (h_n = 2, DH = 64) beside the existing DeltaMem (6 add + 6 delta, unchanged). They hold a separate state S_n ∈ R^{h_n × 64 × 64}.

**Write at pass t:**
- key k_t = normalize(W_kn · e(name(S_t)));
- value u_t = W_vn · LN(h_':', pass t);
- delta rule S_n ← S_n (I − k_t k_tᵀ) + u_t k_tᵀ (erase-then-write per key, so hold passes rewrite the same key and the operation is idempotent).
- **The key comes from the cue's entity: the plan is still given.**

**Read at block 6, every token position p:**
- q_p = normalize(W_qn · e(token_p)) if token_p is a name token of the item, else 0;
- r_p = S_n q_p;
- the residual gets + W_on · r_p (W_on = 0 at init, so the model is function-preserving: bun is bitwise unchanged at step 0 under a det wrap / fp64).
- Givens' values are stated in the text and need no slot; a read of an unwritten name returns ~0.

**Decoder pass:** reads S_n the same way (the final question names S_d).

## 3. Derivation (assumptions tagged)
- [ASM] W_kn maps distinct name embeddings to near-orthogonal unit keys in 64-d. Random unit vectors have |cos| ~ 1/sqrt(64) = 0.125; learning can only reduce this for co-occurring names.
- [ASM] m ≤ 9 written entities per item (d ≤ 8 spine, givens not written).
- [DER] the read at name n returns u_n + Σ_{i≠n} <k_n, k_i> u_i (exactly u_n if the keys are orthogonal; the delta rule removes stale same-key content). Interference norm ≈ 0.125 · sqrt(m − 1) · ||u||, so SNR ≥ 1 / (0.125 · sqrt 8) ≈ 2.8 for d ≤ 8, independent of statement order and pass index.
- [DER] reads are triggered by token identity, so any permutation of the sentences leaves every read value unchanged. Order-free accuracy then follows if the step's computation (op on read values) is position-free; Note 252-5 shows one-hop name lookup is learnable (0.988).
- [ASM] the existing additive heads still carry a pass counter (Note 251-16), so C2 / C6 may still use it.

**Predictions:**
- (i) TEST-PERM d 2-4 rises toward TEST2 (≥ 0.85 vs ~0.1);
- (ii) RETAIN C1 / C2 / C6;
- (iii) a slot swap by name changes only answers that read that name (descriptive).

**Falsifier:** permuted pass-2 WS < 0.3 with the heads means the binding is not used.

## 4. Arms (from bun_s1 / s2, the SOTA recipe: hold penalty + the §257 insert mix; RTX 3090; micro pinned)
| Arm | Runs | Change |
|---|---|---|
| NK | nk_s1 / nk_s2 | + NK heads, + --perm_p 0.5 |
| CTRL | nkc_s1 / nkc_s2 | + --perm_p 0.5 only (no heads), so a TEST-PERM gain is attributed to the architecture, not to perm FT (§254-3: perm FT alone erodes the write) |

- **Pilot (seed 1, 2k of an 8k schedule):** pilot_nk_s1, pilot_nkc_s1.
  - Gate on the log's ws_readout_perm['2']['2'] at 2k: GO if NK ≥ 0.50 and ≥ CTRL + 0.20; PARK if NK < 0.30.
- **Full (on GO):** 4 cells × 8k.

## 5. Readings (full; per seed; binding)
- **ORDER-FREE:** NK TEST-PERM d 2-4 ≥ 0.85 and ≥ CTRL + 0.20 (same seed).
- **RETAIN:** TEST2 d 1-4 ≥ 0.98; NOT-DECODER d 4 K' 2 ≤ 0.20; SWAP4; LP5-D4.
- **C2:** d 6 ≥ bun − 0.10.
- **C6:** K' 64 / 128 ≥ K' 8 − 0.02 at d 1-6.
- **Descriptive:** C5 r = 3 halt; H-CHAINSRC-OP / H-LIVE / H-EXTEND; the NK read gate norm ||W_on r|| share; the name-swap test.
- **Labels:** ORDER-FREE-C1 (all) / named failing criteria. Per arm k/2.

## 6. OC
- P(GO) ~0.45 (binding used) vs ~0.05 (heads ignored).
- P(ORDER-FREE | GO) ~0.5.
- **Censoring:** VOID-RERUN once.

## 7. Cost
- Pilot 2 × ~$0.45; full 4 × ~$1.4 + evals. Total ≈ $7-8; guard $15 (Audit).
- **Build:** the NK module (new file), a flags-off bitwise gate, a step-0 bitwise-with-heads gate, gradflow into W_kn / W_vn / W_qn / W_on, the name-token-single assert, and host sha asserts.
