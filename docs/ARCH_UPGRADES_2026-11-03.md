# Architecture upgrades for the SOTA loop: a research pass (2026-11-03)

**Requested:** by the user via Audit. **Author:** Theory.
**Status:** ranked candidates. Nothing is registered; Audit G-Ds the top pick.

**Tags:**
- [EMP] = measured in this repo (the Note is cited);
- [DER] = follows from the stated assumptions;
- [ASM] = assumption;
- [LIT] = prior work (top 2 checked by Literature).

## The SOTA and its gaps
**SOTA = bun_s1 / s2 (§259):**
- **Architecture:** GPT-2 small. A DeltaMem state S (6 additive + 6 gated-delta heads, DH 64) is read at block 6 at every prompt position. It is written from the ':' answer-position state of each pass's cue. A decoder pass reads S_K.
- **Training:** per-pass value aux, the §257 distractor mix, and the hold-increment penalty.
- **Scorecard:** C1 met, scoped (plan given, chain order); C2 partial (d 6 0.97, d 7 0.86, d 8 ~0.5); C6 met; C5 met, scoped (training-free answer-stability halt); C3 split; C4 not in model; C7 not met.

**Gaps that training alone has not closed:**
- **Order-bound:** TEST-PERM ~0.1, every lineage [EMP 251-16, 253-4, 254-3].
- **Needs the per-step names:** plan-free cues lose stopping and live-distractor rejection [EMP 260-5].
- **Horizon:** per-hop reliability ~0.9 past d 6; untrained passes write increments small relative to ||S|| [EMP 256-1, 259-6].
- **Answer-only writes drift at onset** [EMP 251-9].
- **Prompt recompute per pass** [EMP 246-0].
- **The write ignores text** [EMP 247-2].

## Ranked candidates (by north-star priority × prior / cost; C1 > C6 > C4 = C2 > C3 > C5 > C7)

### 1. NAME-KEYED MEMORY (binding slots): a content-addressed write and read keyed by the entity's name (targets C1 scope: order-free, plan-light; C2)
**Mechanism:** add k_n key-value heads to DeltaMem.
- **Write:** at pass t, the key is the name-token embedding of the entity the pass computes, k_t = W_k · e(name(S_t)) (L2-normalised); the value is v_t = W_v · h_':'.
- **Read:** at every token position p, the query is formed from that token's own name embedding (q_p = W_q · e(token_p) when the token is an entity name; 0 otherwise).
- **Effect:** a sentence "Kim has 3 more than Ian" receives Ian's stored value exactly at the 'Ian' tokens, wherever the sentence sits.

**Derivation:**
- [ASM] distinct names have near-orthogonal keys (random 64-d unit vectors: |<k_i, k_j>| ~ 1/sqrt(64) = 0.125).
- [DER] the read at name n returns v_n + Σ_{i≠n} <k_n, k_i> v_i. With m stored entities, the interference norm is ~ sqrt(m-1) · 0.125 · ||v||. For m ≤ 9 (d ≤ 8 chains), SNR ≥ 1 / (0.125 · sqrt 8) ≈ 2.8, independent of statement order and of the pass index.
- [DER] because the read is triggered by the token's identity, not by its position, permuting the sentences leaves every read unchanged. TEST-PERM accuracy should then match TEST2, provided the step's computation (the op on the read value) is position-free, which the askname-perm result supports [EMP 252-5: one-hop lookup by name 0.988].
- [DER] with values bound to names, a target-only cue can stop at S_d by checking whether S_d's slot is filled (the binding is directly testable, cf. §261).

**Predictions:**
- (i) TEST-PERM d 2-4 ≥ 0.9 (vs ~0.1);
- (ii) H-CHAINSRC-OP / H-LIVE kept;
- (iii) the swap-by-name test: swapping slot(Ian) changes only answers that read Ian.

**Breaks / risks:**
- Unseen names: keys come from embeddings, so this is OK.
- Repeated names across a problem: rare in REG.
- The DF law's cross-state products read two slots (fine).
- Additive interference grows with m; the delta-rule variant (erase-then-write per key) removes stale values.
- The current additive state also carries a pass counter [EMP 251-16]. Removing slot counting may cost C2 unless the counter is kept in the other heads.

**Cheapest falsifying pilot:**
- **S0 ($0):** CPU gradflow and an equivalence gate with the name heads at W_o = 0.
- **S1 (~$1, 2k pilot):** from bun_s1 + name heads, train on 0.5 permuted text; read TEST-PERM d 2 pass-1 / pass-2 WS at 2k. GO if pass-2 WS on permuted items ≥ 0.5 (bun: ~0.1).
- **S2:** 2 seeds × 8k (~$2).

**Nearest prior work [LIT, to check]:**
- key-value memory networks (Sukhbaatar 1503.08895; Miller 1606.03126);
- binding IDs in LMs (Feng & Steinhardt 2310.17191);
- Neural Turing Machines / DNC content addressing;
- recurrent "entity memory" (EntNet, Henaff 1612.03969).
- The new element would be the name-keyed DeltaMem write inside a latent reasoning loop.

### 2. DECAY-GATED / NORMALISED STATE: a bounded, depth-stationary memory (targets C2's horizon; supports C6)
**Mechanism:** replace the 6 unbounded additive heads by gated heads, S ← α S + φ(k) vᵀ with a learned α ∈ (0, 1) (init 0.999), or RMS-normalise S before each read.

**Derivation:**
- [EMP] ||S|| grows with passes, and untrained passes' increments fall below in-range scale relative to ||S|| [259-6].
- [DER] with additive writes, the signal of the latest write in a read is ~1 term against t−1 interfering terms. Read SNR falls like 1/sqrt(t−1), so per-hop reliability decays with depth (observed ~0.9 per hop past d 6 [EMP 256-1]).
- [DER] with decay α, the state is a geometric sum with effective length 1/(1−α). After ~1/(1−α) passes its distribution is stationary, so the read SNR stops depending on t. REG needs only v_{t−1} plus the text, so a short memory suffices.
- Predicted: per-hop reliability is flat in t, so the horizon is limited only by the pass budget K. d 8-10 should rise toward d 6's level.

**Breaks / risks:**
- The DF law's older-state products need longer memory (choose α by law).
- The hold penalty assumes S_{d+2} scale.
- Re-calibrate g_add.
- The gated-delta heads already decay, which is a partial control.

**Cheapest falsifying pilot:** an 8k FT from bun_s1 with α learned on the additive heads (~$1). Read d 5-10 and the per-hop WS slope vs bun.
- PASS if d 8 ≥ 0.75 (bun 0.47 / 0.59) with d 1-4 kept.

**Prior work:** RetNet decay (2307.08621), Gated DeltaNet (2412.06464), Mamba selective decay (2312.00752), and "length generalisation of recurrent states".

### 3. FROZEN-TRUNK ADAPTER LOOP: train only the memory on answers (targets C4)
**Mechanism:** freeze the GPT-2 trunk at the askname init (which already answers single-step questions at ':' [EMP G-AW0 1.000]). Train only DeltaMem's q / k / v / W_o / gains with the answer loss; no per-pass targets.

**Derivation:**
- [EMP] answer-only training fails because the seeded ':' code on d > t items drifts at onset (AW-SOURCE-DRIFT, u1|v1 0.11-0.25) [251-9].
- [DER] with the trunk frozen, the write source h_':' is a fixed function of (text, cue, read), so drift is impossible by construction.
- The loop forms iff the memory can inject v_{t−1} so that the frozen askname computation treats it as a stated given (class (iv) in the askname training: "S_1 has v_1 u" stated → S_2 computed [EMP ST_v1 1.000]). That is a pure read-path learning problem.
- [ASM] the injection at block 6 can emulate a stated given.

**Predictions:** formation (pass-2 WS ≥ 0.9) in ≥ 2/3 seeds without per-pass targets (vs AO 1/4).

**Breaks / risks:**
- The frozen trunk cannot adapt to new ops or phrasings.
- The decoder pass is frozen too, so a small trainable head may be needed.
- C2 / C6 then rest on the memory alone.

**Cheapest falsifying pilot:** a 2k local / cloud pilot from askname (~$0.3); read pass-2 WS at 1k / 2k vs AO's 0.1-0.3. Then 3 seeds × 12k (~$2).

**Prior work:** adapters and prefix tuning (frozen backbone); "memory tokens" with a frozen LM (RMT, Bulatov 2207.06881); Coconut needs full FT.

### 4. CACHED-CUE PASSES: the prompt KV computed once; passes process only the cue (targets C3)
**Mechanism:** reads restricted to cue rows; prompt KV cached across passes; a short cue.

**Derivation [EMP 251-12]:** per-item GFLOPs at d 4: 13.98 vs 18.73 for sentence text-CoT, but 13.47 for value-only text. So this reaches parity at best.
- [EMP 251-13] masking prompt reads in a trained loop keeps 1-2 hops only, so a warm FT is required.

**Predictions:** d ≤ 4 retained after an S1 FT; FLOPs parity vs compressed text-CoT; latency stays ~9× better.

**Breaks:** cue-row reads must carry all hops; C1 content selection reads the prompt.

**Pilot:** the REVIEW_CC S1 warm FT from bun (~$1.5).

**Prior work:** LLoCoT 2610.11472, LOTUS 2606.31779, KV-LAT (REVIEW_KV), pause tokens.

**Rank reason:** C3 is 5th priority, and the ceiling is parity.

### 5. DISCRETE TIED WRITE: write the embedding of the printed value (targets C7)
**Mechanism:** S_t's write value = W · e(argmax readout_t), the token embedding of the value the pass prints (straight-through gradient).

**Derivation [DER]:**
- the state is then a function of the printed token, so editing the printed value edits the state identically (faithful by construction);
- round-trip agreement = 1;
- the loss of non-value information is irrelevant for value-only chains.

**Predictions:** C7 edit-follow ≥ 0.9 (vs 0 [EMP 247-2]), and C1 retained if argmax accuracy stays ~1.

**Breaks:** straight-through training noise; it may lose the distributed hold signal (C5 / C6); a C2 hop error becomes a hard error.

**Pilot:** an 8k FT from bun (~$1) + the §247 edit eval.

**Prior work:** discrete CoT vs Coconut; VQ bottlenecks (van den Oord 1711.00937); "faithful CoT by construction".

**Rank reason:** C7 is last priority.

## Recommendation
**Top pick: (1) NAME-KEYED MEMORY.** It attacks C1's two remaining scope limits (order and plan) by an architectural change, where three training-only attempts failed (§252 perm, §253 / §254 lookup init, §260 plan-free). Its pilot is cheap.
**Second: (2) DECAY-GATED STATE** (C2 horizon; cheap; well-founded).
Literature is asked to check novelty for (1) and (2) only.
