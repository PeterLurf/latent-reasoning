# LANG-VB: VERSION-BANK / SLVA + self-teaching on language (ProsQA-SYM deep pilot, then GSM8K) — DRAFT v0

**Status.** Draft by an Opus research-design subagent for the Theory session, started 2026-10-18 12:03 EDT (`date`).
Not a registration. $0 to write: no GPU, no repo edits. User-approved to START THE DESIGN only (12:02 EDT 10-18, "yes" via Audit);
any spend needs registration, Audit's review, and the user's rule (one design, one total cap <= $25 under the $30 standing rule).
Binding additions from Audit (12:0x 10-18, relayed by Theory): (1) Stage A uses the shortcut-NEUTRALISED ProsQA (§59 p8_na construction
or a balanced regeneration at the new depths) and re-runs the §57/§59 shallow-heuristic panel at every depth band; a band with any shallow
heuristic >= 0.6 is not used for readings (gate). (2) Stages A + B are ONE design with one total cap (<= $25); not split; if B's estimate
exceeds the remainder, it is flagged for the user's decision.

**Tags:** [PROVED], [PROVED UNDER X], [DERIVED script], [MEASURED file], [ASSUMED], [CONJECTURE].

(Written incrementally. All sections were complete at 12:38 EDT 10-18. This is v0 for Theory; it has not yet been proof-checked or red-teamed.)

## 0. One screen
| item | LANG-VB v0 |
|---|---|
| **Question** | Does VERSION-BANK / SLVA (D3-T's recipe), ported to language through a **slot subset**, learn serial latent computation that passes the causal C1 tests: on deep ProsQA graph search (Stage A), with self-teaching past the gold depth, and on GSM8K against CODI-20 (Stage B)? |
| **Key $0 finding first** | p8_na is clean only at the option level. Its chain **anchors keep the §57 cue (proximity 0.896–0.926 at depth ≥ 6)**, and a double-neutralised p8 keeps anchor parents' in-degree at 0.61–0.67 [MEASURED scripts/heur_panel.py, nn_regen.py]. So Stage A uses a **new exchangeable generator, ProsQA-SYM**, gated by a level-wise panel per depth band. Every person-agnostic heuristic has expectation ½ [PROVED]. A CPU prototype passes the panel at every D 3–16 (max 0.518–0.552) with 0 invariant failures [MEASURED scripts/sym_proto.py]. §191's 0.950 may partly use the anchor cue: a $0 check (A0-1) is proposed |
| **Architecture** | Prelude 0–5 over the whole question → V_0 (all rows in the bank, as β⁰ entries). Versions are written only at slots: **concept slots** (ProsQA) / **12 quantity slots** (GSM8K). Core 6–9 looped, with SLVA's slot-local version read as pass input (D3-T). Question-free coda 10–11 with a bank-free decoder path. Linear recode heads. No pass token. A new module subclasses slva.SLVA; vbank / slva are not edited |
| **Targets** | ProsQA: gold **REACH** (dist(R, c) ≤ t) on D 3–8. Then **self** D 9–12 labelled by the model's own chain, kept by the label-free **SOUND** verifier (root-anchored, every new mark has an in-edge from the previous marks, an option reached); **SOUND-ONLY R2** ⇒ **zero supervised noise** [PROVED via Lemma S-L + SYM's invariant]. GSM8K: GSM8K-Aug equations → **DECOMP** binary-op programs (op, two provenance pointers, digits); one op per pass; C4-legal distillation |
| **How it resists one-pass precompute** | Bounded core over the previous version (L1-L depth lemma [PROVED]); the question is read only as layer-6 V_0; R2 pins the readable schedule (over-reach is penalised). The open routes (precompute-and-reveal, recompute from V_0) are **not** closed by loss. They are *detected* by PROBE(1), FUTURE-DECODE, VALUE-SWAP (incl. the slot-only arm) and STALL, with PASS1-PRECOMPUTE-GSM / ONE-PASS-CHAIN-L pre-stated |
| **Cells** | A0 ($0 gates incl. A0-1, LEVEL-PANEL) → **A1** LANG-SLVA on SYM D 3–8 (24k) → **A2** self round 1 (FT 12k, D 9–12) [+ A2-CTRL recommended, A-TXT optional] → B0 ($0) → **B1** LANG-SLVA-GSM at CODI-20's exposure (120k; interim GO at 24k). B2 (GSM8K self) deferred |
| **Primary readings** | **VB-C1-L** = DEEP ∧ PROBE-GAIN ≥ 0.35 ∧ band ON(D) slope ≥ 0.6 ∧ PERSON-SWAP follow ≥ 0.80 ∧ STALL ≥ 0.9 (D 5–8). **SD-C2-L** (D 13–16 ≥ 0.80, untouched) + CAUSAL-DEEP-L + MEM-TRADED-L. **VB-C1-GSM** (m ≥ 3) = PROBE-GAIN ≥ 0.10 ∧ class-A swap follow ≥ 0.6 / own ≤ 0.3 ∧ STALL ∧ ¬ONE-PASS-CHAIN; accuracy vs CODI-20 (0.3791) paired on the same 1,319 items |
| **OC headline** | P(VB-C1-L \| no serial use) ≤ 0.02, \| serial ≈ 0.5, prior ≈ 0.2. P(SD-C2-L) ≈ 0.05 (\| H_shift ≈ 0.02). P(VB-C1-GSM) ≈ 0.08 (\| one-pass ≤ 0.01). P(B1 ≥ CODI-20 − 0.02) ≈ 0.12. **P(≥ 1 decisive C1 positive) ≈ 0.25.** The likeliest GSM8K outcome is the §208 failure, detected cheaply |
| **Cost** | Required A1 + A2 + B1: **$5.3–10.6 (central about $6)**; optionals +$1.6–2.5. **One design, family cap $25** (user's $30 rule). Nothing projects above the remainder. MEM-TRADED-L after A2 routes any further round to the user |
| **Theory must decide** | D-1 SYM + D-1b A0-1 to Audit; D-3 concept slots; D-5 labeller selection; D-6 fixed-K′ headline; D-8 defer B2; D-9 matched 120k + interim look; D-10 optionals; D-11 binary-answer bars; D-12 REACH as supervised (not C4); D-14 PROOFS / REDTEAM before registration |

## 1. Prior record and what it rules out

### 1.1 The nested VB / SLVA / self-depth line (the thing being ported)
| result | number | source |
|---|---|---|
| §211 N2 PURE-ANNEAL reads VB-C1 ("supervised serial latent computation in a pass-global attention memory"), one seed | DEEP 0.9998; PROBE(0)/(1) 0.119/0.132 → PROBE-GAIN 0.868; median onset 2/3/4/6/6/8 at n 3–8 (slope on height 1.29); VALUE-SWAP (ii) follow 1.000 / own 0.000 (4,183 pairs); STALL 0.997 | [MEASURED THEORY §211 N2 result; s211_vb_pa_s1/eval_vb.json] |
| N1 CARRY is also VB-C1 (through the carry only); N3 (+MCT) is COUNT-KEYED (STALL 0.499) | — | [MEASURED §211 N1 / N3 results] |
| §213 SLVA (D3-T): the PA cliff removed (LENGTH-FREE; STEP-GENERAL to m 16), VB-C1 kept, but RNN-IN-FUNCTION (HISTORY-USED drop 0.000); DEPTH n 9–12 0.480 | swap NEWEST / BANK / OLD 1.000 / 0 / 0 | [MEASURED §213 D3-T result] |
| §215 / §216 self-depth by the re-rooted single-step chain (KEEP ∧ COMPOSE-VALID): support 8 → 12 → 16; n50 / support 1.29 / 1.29 / 1.28; literal C2 not met (LIT-C2-2 0.018) | ABSORB 1.000 (R1), 0.997 (R2); M2 (n 17–20) 0.8745 | [MEASURED §215 FT-SELF; §216 s1 round-2] |
| The memory cost of self items (MEM-TRADED) | STAB_self 0.782 → 0.489 → 0.437; FT-CTRL 0.696 | [MEASURED §215 pair; §216 s1] |
| §217 in range: no causal erosion across rounds (n 4–8) | follow_ii 1.000, STALL 1.000 in D3-T, FT-SELF, R2-s1 | [MEASURED §217 table] |

**What made it work on nested, and what language lacks.**
- (a) *Natural slots.* Every token of a T ≤ 34 expression is a slot, and a reduction changes one contiguous span.
- (b) *Exact per-pass R2 targets.* E_t is a deterministic function of the input: one Feng reduction per pass.
- (c) *A structural verifier for self labels.* COMPOSE-VALID checks FRAME / SPAN / ORDER / CANON from the parsed input and never computes a value (boxes/selfdepth.py check_step).
- (d) *An observable depth axis.* n = the operator count, read off the input.

Language has none of (a)–(d) exactly. §2–§4 say what replaces each, and what is lost.

### 1.2 ProsQA-deep record
| result | number | source |
|---|---|---|
| DIRECT GPT-2 on p8_na: no single-pass regime (D* = 3); memorises | NA s1, depths 3–10: 0.538 / 0.476 / 0.504 / 0.549 / 0.508 / 0.473 / 0.496 / 0.478 | [MEASURED §168 note 168-5] |
| p8 depth = a 3–4-hop branching base + m UNBRANCHED chain hops ("serial lookups", not deeper search) | — | [MEASURED prosqa/make_long.py; note 168-6] |
| STEP-AUX (DeltaMem INJECT, K 10) without relabelling memorises | train 1.000 / test 0.490 | [MEASURED §168 STEP-AUX s1] |
| §191 relabelling (per-draw names into ENT17 / CON38, premise shuffle, option swap): NAME-MEM | A (STEP-AUX) 0.950: d3–10 ≥ 0.955, d11–14 0.909 / 0.840 / 0.729 / 0.500 (K′ 10); K′ 0 / 1 at chance (0.45–0.54) at every depth. B (TEXT-COT) 0.938 | [MEASURED s191_stepaux_relabel_s1/eval.json; §191 results] |
| §191 A had no donor, serial or clock test | only removal K′ 0 / 1, PATH-INDEP (answer_agree 0.614), STEP-RECOVER (0.53 / 0.25 / ≤ 0.15) | [MEASURED eval.json s168s2.*] |
| acc_ctl is measured on p8_ctl, built from standard (cue-carrying) p8 | proximity 0.897 on p8_ctl | [MEASURED by the ProsQA summary subagent with prosqa_na.heuristic] |

**New $0 finding for this draft (binding addition 1 makes it decisive).**

p8_na neutralises the §57 proximity cue at the *option* level only. The options c_m / d_m sit at the ends of two unbranched m-hop chains. The chains' *anchors* are the original ProsQA target and negative, and the anchors keep the §57 cue:

| p8_na_test depth | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| option-level proximity (§59) | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| **anchor-level proximity** | 0.500 | 0.674 | **0.896** | 0.896 | 0.896 | 0.896 | 0.897 | 0.902 | 0.905 | 0.904 | 0.926 |
| anchor-level mentions / parents | 0.572 | 0.628 | 0.652 | 0.652 | 0.652 | 0.652 | 0.651 | 0.656 | 0.661 | 0.684 | 0.735 |

- **Sources.** [MEASURED scripts/heur_panel.py → scripts/heur_p8na_test.json]. On train (heur_p8na_train.json), anchor proximity is 0.897–0.945 at depth ≥ 6.
- **Validation.** The panel code reproduces §59's table: prosqa_test option level 0.911 / 0.662 / 0.662 / 0.590 / 0.554 / 0.515, and p8_na_test 0.50 everywhere. The implementation is therefore checked against the registered numbers.
- **Mechanism.** NA's distance equalisation adds "<other person> is a <option or a parent of it>" [DERIVED from theory/prosqa_na.py neutralise()].
  - For m = 1 the parent is the anchor, so the anchor is neutralised too (depth 4 reads 0.500).
  - For m ≥ 2 the parent is a chain node, so the anchors are untouched.
- **What a model can do with it.** Chase each option's unbranched chain back m hops, then apply the proximity cue at the anchors. That gives about 0.90 with no search from the root. Because the chase is serial lookups, K′ 0 / 1 would still be at chance.
- **Consequences.**
  - (i) p8_na fails a level-wise panel at every depth ≥ 5. Under Audit's gate it cannot carry LANG-VB's readings.
  - (ii) §191's NAME-MEM 0.950, with K′ 0 / 1 at chance, is *also* consistent with "chain-chase + anchor cue". That is not shown, but it is not excluded. A $0 eval of the kept §191 models on anchor-neutral items (A0-1, §8) separates the two. **This is for Theory and Audit; by itself it changes no registered reading.**
- **The double-neutralised p8 ("NN") also fails the gate.**
  - NN applies §59 at the base, extends, then applies §59 at the options. It removes the anchor proximity cue.
  - But parents' in-degree at the anchors stays at **0.606–0.668 in every band** [MEASURED scripts/nn_regen.py → scripts/nn_nn_test.json]. This is §59's own prosqa_na residual (0.62).
  - So NN fails the 0.6 gate, which is why §3.1 builds a new exchangeable generator (ProsQA-SYM).

### 1.3 GSM8K latent record
| result | number | source |
|---|---|---|
| §200: latent (DeltaMem INJECT step-aux) below text CoT at 4 epochs | A 0.147 vs C 0.321; K′1 0.125 | [MEASURED §200 result] |
| §204: factored per-pass targets (value, 17-class op, two pointers) are met by a one-pass chain | VALUE-LEAK from M_1 0.126 vs VALUE-EXACT from M_t 0.140; VALUE-EXACT by pass 0.338 / 0.157 / 0.108 | [MEASURED §204 F-P] |
| §207 GSM-SWAP: every latent model RECOMPUTES-FROM-INPUT (memory = scaffold) | OWN-KEPT(2) 0.80–0.88; INPUT-FOLLOW(1) 0.87–0.93; CARRIED(1) ≤ 0.005 | [MEASURED §207 table] |
| §208 QUESTION-ONCE: PASS1-PRECOMPUTE (latent accuracy is one forward pass) | K′ 0 / 1 / 2 / 4 / 10 = 0.014 / 0.143 / 0.140 / 0.143 / 0.138; ONSET 1 in every stratum | [MEASURED §208] |
| §209 / §210 CODI at our budget: the shortfall was budget, not architecture | CODI4 0.189 → **CODI-20 0.3791** (RTX 3090, 59,822 steps, about $9.28). m ≤ 2: 0.621 (n 422); m ≥ 3: 0.265 (n 879). 0 latents 0.171. K-sweep at K 0–6: 0.171 / 0.151 / 0.125 / 0.287 / 0.293 / 0.372 / 0.379 | [MEASURED gsm_codi_20ep_s1/eval.json; §210] |
| CODI-20's latents are content-bearing but context-bound | DONOR-SWAP follow 0.040 / own 0.084 / neither 0.876 | [MEASURED note 210-4] |
| Published bars (GPT-2 small, GSM8K-Aug) | CODI 43.7; PCCoT 49.48; CoT-SFT 44.1; Coconut 34.1 | [Literature notes LIT_GSM8K_AUG_BARS, LIT_PCCOT_PROBE] |

**GSM8K structure for the slot design** [MEASURED scripts/gsm_stats.py]:
- **Test m** (the gold "<<" count), for m = 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8: 18 / 65 / 357 / 364 / 290 / 138 / 57 / 21 / 9. So m ≥ 3 has 879 items, m ≥ 4 has 515 and m ≥ 6 has 87.
- **Train m** (first 60k lines), for m = 1 / 2 / 3 / 4 / 5 / 6 / 7+: 16% / 38% / 27% / 12% / 4.5% / 1.5% / 0.9%.
- **Operators:** 61% of test items (70% of train) have only single-operator equations.
- **Values:** p99 is 6 integer digits (test max 7); 2.0% are non-integer on test (10% on train); negatives are < 0.3%.
- **Questions:** p50 53 / p90 86 / max 182 GPT-2 tokens (test), with 3.4 numeric literals per question.

### 1.4 What the record rules out (design constraints)
1. **Re-reading the question every pass at full depth (INJECT)** leads to recompute-from-input (§207). LANG-VB keeps VB's split: the question is encoded once (prelude), and afterwards it is read only as layer-6 bank entries V_0 (§2).
2. **Per-pass targets alone do not force serial computation** (§204 was met by a one-pass chain). The targets must sit on a bounded core whose input is the previous version, and the readings must be able to see precompute: PROBE(1), FUTURE-DECODE and the slot-only splice (§5).
3. **Trained-coda and trained-R readings are artefacts** (project memory: trained readings). Every decisive reading is either a frozen-model causal test (VALUE-SWAP, STALL) or a fresh post-hoc probe (PROBE(t)).
4. **ProsQA claims need anchor- and option-level cue control at every depth** (§1.2), and per-draw relabelling (§191 NAME-MEM).
5. **Budget dominates GSM8K.** §209 vs §210 went 0.189 → 0.379 from 4 to 20 epochs. A GSM8K comparison with CODI-20 is fair only at matched exposure (7.66M examples).
6. **MCT teacher forcing** cost the clock test (N3), so it is not used.
7. **Self items trade the history channel** (MEM-TRADED twice). That is expected again, and it is stated in the OC and routed to the user.

## 2. Architecture

### 2.1 LANG-SLVA: one skeleton for both tasks (D3-T's architecture over a slot subset)
**Principle.** Every part is SLVA / VERSION-BANK as registered in §211 / §213 and used by D3-T. The one structural change is that **versions are written only at a slot subset S of the token positions**. The question itself is encoded once, and later passes read it only through V_0 bank entries.

**Prelude.** x = [EOT] + question tokens, plus Q placeholder tokens for GSM8K (§2.3). Blocks 0–5 run causally over all T tokens → H_6 (B, T, D).
- V_0 := H_6[S]: the slot rows, (B, |S|, D).
- The bank's version-0 entries are **all T rows of H_6**: the premises / wording live at non-slot tokens, and every pass must be able to read them.
- The bias on version 0 is a constant β⁰ with no age term, as in §211 pin 2.

**Core, pass t = 1..K** (blocks 6–9, shared). Queries are the current version's |S| slot rows. Keys / values are:
- [version 0: all T rows] ∪ [versions 1..t−1: the slot rows] ∪ [the current version's slots, causal in position order].
- Bank K / V come from block ℓ's own ln_1 + c_attn (`vbank.bank_kv`, unchanged), with the bias β + γ·(t−1−s) for s ≥ 1, calibrated per head at step 0 (99th-percentile bank mass ≤ 0.02, ratio test), as in §211.

**Pass input (SLVA, unchanged math).**
- U_t[j]_h = Σ_{s<t} α^h_{j,s} · N(V_s)[j]_h.
- α^h_{j,s} = softmax_s( ⟨W^h_q LN_q(N(V_{t−1})[j]), W^h_k LN_k(V_s[j])⟩/8 + β^h_0·1[s = 0] + γ^h·(t−1−s)·1[s ≥ 1] − c(step)·1[s ≠ t−1] ).
- V_0[j] = H_6 at slot j's token. The init is from block 6, and c follows `vbank.c_at(step, 1000, 8000)`: exact CARRY to 1k, then 20 → 0 by 8k.
- N is the per-slot RMS reset against the slot's own V_0 row.

**Write.** V_t = the core output at the slots, appended. Versions are never overwritten. There is **no pass token or pass position**: slot position ids are fixed across passes.

**Coda after pass t (question-free).**
- Blocks 10–11 run over [V_t slots | decoder tokens], with bank versions 1..t−1 (slots only). **V_0 is excluded** (211-1 pin 4), so after the core the answer can be computed only from versions.
- The decoder string goes through blocks 0–9 **on its own**: no question keys, no bank, positions continuing after T. Only then does it join the slots at block 10.
  - This replaces the VB draft's "prelude-encoded among themselves" line, so that the pretrained layer-10 weights see layer-10-depth inputs [ASSUMED better at init; G-L1b measures drift].
  - Theory decision D-2 (§10).
- K′ = 0 is the coda on V_0's slot rows.

**Recode heads R** (LN + Linear per slot, linear so they cannot compute):
- ProsQA: R_P 768 → 2 classes {OUT, IN}.
- GSM8K: R_G, a set of linear heads per quantity slot (§2.3).

**Losses: READY-RECODE, unchanged in form.**
- **R1:** readiness-window answer CE, mean over t ∈ [τ_i, K_i], K_i = min(τ_i + U{0..4}, K_max). τ_i = depth D_i (ProsQA) or binary-step count m_i (GSM8K).
- **R2:** per (item, t), ½ the mean CE over changed slots + ½ over unchanged slots, IN / value classes weighted ×2, HOLD after τ_i.
- **L** = R1 + 1.0·R2. No MCT (R3), no PAUSE, no STALL-AUG.

**Not changed.**
- GPT-2 small (`s5.load_gpt2`).
- `vbank.bank_kv` and `vbank.block_bank` (the bias tensor is (B, H, |S|, T + (t−1)·|S|)).
- SLVA's parameters and init (`slva_wq/bq/wk/bk/ln_q/ln_k/beta0/gamma`), the c schedule, and the β / β⁰ calibration.
- lr 1e-4 cosine, wd 0.01, bs 64, kextra 4, rec_w 1.0, the trained position table (D3-T, not D3-F).
- Off-host checkpoint mirror + resume; eval in local fp32 at the original eval_bs.

### 2.2 Code plan (Experiments implements; Audit's diff review; nothing written here)
| file | content | touches existing code? |
|---|---|---|
| boxes/langvb.py (new) | `LangSLVA(slva.SLVA)`: `rollout_ls(x, pad, slot_idx, slot_pad, K, …)`, the eval hooks from vbank (`edit`, `ver_keep`, `v0_off_after`, `dup_newest`, `input_edit`, `slva_view`); `bias_row_ls` (version-0 width T, later widths \|S\|); `coda_dec`; R_P / R_G; `lvb_loss`; reuses vbank's CurGate and c_at | **no.** Subclass only. vbank.py and slva.py are imported read-only |
| boxes/prosqa_sym.py (new) | the SYM generator (§3.1), per-draw relabelling into NONCE-300, render, gold REACH targets (BFS), the SOUND verifier, the LEVEL-PANEL | no |
| boxes/gsm_prog.py (new) | DECOMP: GSM8K-Aug equation strings → binary-op programs with provenance (§3.2); per-pass targets; the ARITH / PROV verifier; the class-A donor maker (reuses gsm8k_data.parse_recomputable / draw_literal read-only) | no |
| boxes/langvb_eval.py (new) | the parts acc, probe, vswap, stall, hist24-L, future-decode, gsmswap-L (§5), each merged into eval_lvb.json as it finishes, with per-item outputs | no |

**Plain-path smoke:** D3-T (`make_model(g, "slva")`) is constructed and run 1 step, plus nested `vbank_eval` acc on 64 items bitwise against eval_vb. This proves the nested paths are unchanged.

### 2.3 Slots per task
**ProsQA (scheme L-P, recommended: concept slots).**
- One slot per distinct concept, at the last BPE token of its **first** mention (concept names are multi-token: 3 tokens for 30 of CON38's 38 names).
- The two option concepts are ordinary slots.
- |S| is about 50 at D 8 (prototype mean 49.6, max 75). It is ≤ 76 at D 16 under the per-world cap of 2D + 6 (§3.1).
- **Alternative L-P′ (mention slots):** one slot per mention, |S| ≈ 100 at D 8, about 2.5× the attention work [DERIVED scripts/cost_model.py: 70.7M vs 28.4M entries per item; est. 1.17 vs 0.82 s/step, ASSUMED speed law]. The cost is not worth it unless a Stage 0 smoke shows L-P cannot learn the binding.

**GSM8K (scheme L-G).**
- Q = 12 **quantity slots**: placeholder tokens appended after the question. Each uses one existing GPT-2 token (" _"), so there is no new embedding; position ids are T_q+1..T_q+12.
- Their V_0 rows see the whole question (causal prelude).
- Question literals are **not** slots: they stay readable in V_0 every pass. A quantity slot's version holds one intermediate result.
- R_G heads per slot, all linear:
  - op (6 classes: VOID + − × ÷ OTHER);
  - ptrA and ptrB (VOID, lit_1..lit_14 by order of occurrence, q_1..q_11, CONST);
  - sign (2);
  - 7 integer digit places + 2 fractional places, least significant first (11 classes each: 0–9 and PAD).
- Values are rounded to 2 decimals. Items with a value outside ±10^7, or not exact at 2 decimals, keep R1 but **mask that slot's value terms** (no item dropped; §3.3 same-filter).
- The schedule is one binary operation per pass. So q_t is written at pass t, and slots k > t hold VOID at pass t.

### 2.4 Why these choices
- **Slot subset (forced by memory).** Full-T versions cost O(K²T²) attention: at T 450, K 12 that is about 760M saved attention entries per item, roughly 190 GB for bs 64 in fp32 [DERIVED: 48·T²·K(K+1)/2]. Concept slots need 28.4M per item (7.3 GB at bs 64 fp32; 3.7 GB bf16) [DERIVED scripts/cost_model.json].
- **SLVA, not PA.** Language slots sit at positions up to about 760, and D3-T is the VB cell with no length cliff (LENGTH-FREE; STEP-GENERAL to m 16) [MEASURED §213]. N2's content-free position gather broke at j ≥ 33.
- **All question rows in V_0.** This keeps recompute-from-V_0 open, exactly as Addendum 1 of the VB draft found for nested. Only the bounded core limits it (§4.2). The design does not pretend otherwise; VALUE-SWAP's own share and PROBE(1) / FUTURE-DECODE read it.
- **Question-free coda.** The coda cannot re-read the question, so it cannot re-solve after the passes (the G3 route), as in VB.
- **One binary operation per pass on GSM8K.** This gives a uniform step, the nested analogue (one reduction per pass), and makes the arithmetic verifier exact (§3.2). The price is more passes: m_bin ≥ m.

## 3. Targets and labellers per stage

### 3.1 Stage A data: ProsQA-SYM, an exchangeable generator (the "balanced regeneration" of binding addition 1)
**Why a new generator.**
- p8_na fails the level-wise panel at depth ≥ 5: anchor proximity 0.86–0.93.
- NN fails it in every band: anchor parents' in-degree 0.61–0.67 (§1.2).
- Coconut ships no ProsQA generator (LIT_PROSQA_FRESH, per the ProsQA summary).
- p8's depth is unbranched chains (168-6). SYM puts branching at every depth.

**SYM(D, rng)**, for D ∈ 3..16 (the spec; Experiments implements in boxes/prosqa_sym.py):
1. **Persons.** The root R (asked), the foil P, and 0–2 extra persons. Names come from §191's ENT17, drawn per item.
2. **Two worlds from the same random process G(D)**, W_R from R and W_P from P:
   - a path p → a_1 → … → a_D; a_D is the world's option (o_R = target, o_P = negative);
   - each a_k (k < D) gets Binomial(2, ½) distractor children, each with a random distractor subtree of depth ≤ min(3, D − k) that does not reach the option;
   - within-world extra edges u → a_k are allowed only if depth(u) + 1 ≥ k, so the option's BFS depth stays exactly D (several shortest paths are allowed);
   - the person has r ~ U{1..4} direct statements: one is a_1, the rest point to distractors. So multi-direct roots are in distribution, which the re-rooted labeller needs (§3.1 self).
3. **Cross-world edges.** c ~ U{0..2} in each direction, between distractor nodes only. Targets are never ancestors of the other world's option, so R cannot reach o_P and P cannot reach o_R. The law is the same in both directions.
4. **Extra persons** get disjoint 2–4-node trees that never reach either option.
5. **Names.** Concepts are drawn without replacement per item from **NONCE-300**: 300 frozen ProsQA-style pseudo-words of 2–3 GPT-2 tokens, sha recorded. CON38 is too small: depth 16 needs about 76 concepts.
6. **Text.**
   - "R is a X." / "Every A is a B." for every edge, in a uniformly shuffled order.
   - The question is "Is R a o1 or o2?", with the option order a fair coin.
   - The answer string is "### o_R." (§2.1 coda; no person name).
   - World size about 2D + 6 concepts, so about 450 tokens at D 8 and 740 at D 16 (< 1024) [DERIVED from the size law; verified by G-L0].
7. **Invariants asserted per item:**
   - dist(R, o_R) = dist(P, o_P) = D;
   - o_P ∉ Reach(R); o_R ∉ Reach(P);
   - every concept is mentioned at least once;
   - T ≤ 1000.

**Exchangeability [PROVED].**
- Let σ swap (R, W_R, o_R) ↔ (P, W_P, o_P). The generator's law of (graph, shuffled text) is σ-invariant, apart from which person the question names: W_R and W_P are i.i.d. from G(D), the cross-edge law is symmetric, and the sentence order is uniform.
- So any heuristic h that ignores the asked person's identity picks the target and the negative with equal probability: E[acc(h)] = ½ exactly.
- All six §59 heuristics, at every chain or path level, are person-agnostic, so all have expectation ½ at every D. The empirical LEVEL-PANEL (G-L0) then tests only the implementation, within sampling error.
- **Not covered:** heuristics that use the asked person, i.e. partial search. These are reported as **BIDIR(r)**, which decides by a bidirectional BFS with r hops from R and r hops back from each option, guessing on a tie. It is exact once 2r ≥ D, and its accuracy at 2r < D measures how much a shallow root-aware search gets from SYM's cross edges.
- BIDIR(1) and BIDIR(2) are reported per D, descriptively. **C1 bands are D ≥ 5:** more than DIRECT's D* = 3 and more than BIDIR(2)'s exact range of 4. Fixed now.

**Prototype check [MEASURED scripts/sym_proto.py → scripts/sym_proto.json; 1,000 items per D, CPU, a feasibility prototype, not the implementation].**

| D | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| invariant failures | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| max §59 heuristic over all path levels (direction-free) | 0.520 | 0.529 | 0.518 | 0.552 | 0.540 | 0.539 | 0.534 | 0.542 | 0.539 |
| BIDIR(1) / BIDIR(2) | 0.50 / 1.00 | 0.50 / 1.00 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| concepts mean (max) | 20 (34) | 26 (45) | 32 (49) | 38 (59) | 50 (75) | 62 (90) | 73 (103) | 85 (115) | 97 (131) |
| words mean (max) | 119 (192) | 153 (276) | 189 (303) | 225 (349) | 295 (452) | 367 (556) | 437 (679) | 508 (719) | 579 (834) |

- Every band passes the 0.6 gate with a margin of ≥ 0.05. The max over 6·D cells at n = 1,000 is at the expected 0.5 + about 3 SE, consistent with E = ½.
- BIDIR(2) is exact only at D ≤ 4, as predicted: cross edges enter non-ancestor leaves only, so they add no shortcut.
- **Size law needs a cap.** At D 16 the prototype reaches about 580 words, roughly 930 GPT-2 tokens at about 1.6 tokens per word [ASSUMED], max above 1024. The implementation must cap per-world concepts at 2D + 6 by thinning distractor subtrees for D > 8. G-L0 then measures T with the GPT-2 tokenizer (invariant T ≤ 1000).

**Gold per-pass targets (A1): REACH.**
- E_t(c) = IN iff dist(R, c) ≤ t (BFS on the item's edges), for every concept slot c; E_0 = all OUT.
- Changed slots at pass t are the concepts at distance exactly t. A pass with none is HOLD (unchanged half only).
- This is the exact one-hop-per-pass analogue of E_t. It is a **simulator state**: it appears in no text CoT. So A1 is never C4 (§6).
- **Considered and not adopted: PATH targets** (the gold path node at hop t, ProsQA's CoT). They would be C4-legal, but the forward choice of a_t needs the remaining D − t hops (look-ahead to the target). The per-pass step would then not be a local map, and CAP-aff could not hold for it [DERIVED].
- **A1 training law:**
  - D ~ U{3..D_max}, with curriculum D_max 4 → 6 → 8;
  - CurGate (vbank.CurGate, unchanged) on **plain items' answer CE only**, thr 0.35 (binary CE; chance ln 2), need 2, cap 6k;
  - final stage to 24k;
  - per-draw fresh items (seed 22100 + step stream), relabelling, premise shuffle, option coin;
  - K_i = min(D_i + U{0..4}, 12).

**Self labeller and verifier (A2, round 1, self window D 9–12).** The labeller is fixed in advance (A1's model.pt). It is chosen between two pre-stated chains on **GATE items only** (seed 22400, 2,000 per D 9–12), never on pool or test items:
- **L-ROLL.** The model's own rollout, K′ ≤ 16. M_t = {c : argmax R_P(V_t)[c] = IN}.
- **L-RR (frontier re-root, the §215 chain's analogue).**
  - Each step re-renders the item with the asked person's direct statements replaced by "R is a f." for f ∈ the current frontier, at the same sentence slots, extra ones at random boundaries.
  - It runs K = 2 and takes F_{t+1} = (pass-2 marks) \ (pass-1 marks) \ M_t.
  - Depth lives in the chain. The rollout is only ever 2 passes on a fresh input.
- **Selection rule.** Take the chain with the higher kept yield at D 11–12. Ties go to L-RR (it is the §215-proven pattern).
- **Verifier SOUND** (label-free; it reads only the item's own statements, parsed from its text):
  - **P1 ROOT-ANCHORED:** M_1 ⊆ the asked person's direct concepts.
  - **P2 SOUND:** every c ∈ M_t \ M_{t−1} has a statement "Every a is a c." with a ∈ M_{t−1}, or "R is a c." (late marking of a direct concept is allowed).
  - **P3 MONOTONE:** M_{t−1} ⊆ M_t.
  - **P4 TERMINAL:** there is a first pass s ≤ 16 with an option in M_s.
  - KEEP = P1 ∧ P2 ∧ P3 ∧ P4. The label is the option in M_s, and s is the self stop.
  - The verifier **never computes a BFS layer**. Completeness (M_t = R_t) is not checked. Checking it would make the verifier the simulator.
- **Lemma S-L [PROVED]** (by induction on t, from P1–P3):
  - every c ∈ M_t has a statement chain from R of length ≤ t, so M_t ⊆ R_t;
  - with SYM's invariant o_P ∉ R_t for all t, every kept label is o_R (**π_label = 1, φ = 0 by construction**) and s ≥ D.
- **Self targets: SOUND-ONLY R2.**
  - R2 terms only on slots in M_s: IN from the pass at which each was first marked. Changed half = newly marked at t; unchanged half = marked before t.
  - **No OUT targets** on self items, and no R2 terms on unmarked slots.
  - R1 window [s, K_i], with K_i = min(s + U{0..4}, 16).
- **Corollary [PROVED from S-L].** Every self target equals the gold target of the same (item, slot, pass): a concept marked at t has dist ≤ t, so its gold E_t is IN, and reach is monotone. The self R1 window [s, K_i] lies inside the gold window, since s ≥ D. **So round-1 supervised label noise is exactly 0.**
  - The ½ / ½ changed / unchanged *weighting* can differ from gold for late marks (dist < t): a late mark is "changed" in self, "unchanged" in gold. That reweights terms; it never flips a target.
  - What is distorted is coverage: omitted slots, no OUT terms, a later R1 start. The selection is biased toward items the model partly solves.
- **Pool.**
  - Seed 22300, D 9–12 (D is a generator parameter; **never stored** in a pool record and never read by the trainer; disclosed).
  - Built until 60,000 kept per D, or a draw cap of 400,000 per D, after which it is exhausted = no A2.
  - Record keys exactly {text_ids, slot_idx, stmts_parsed, marks_by_pass, stop_s, label}, with a key-set assert and a grep (the §215 G-S0 pattern).
  - **Fresh-once** (pool cursor; self_reuse = 0 asserted).
- **A2 training law.** FT from A1 at lr 3e-5, 12k × bs 64, c = 0. Each step a coin p_self 0.5 picks a self batch, otherwise a plain gold D 3–8 batch from the same stream law as A1. Self K from rng_s; plain batches identical to the control's at the same step index (the §215 STREAMS identity).

### 3.2 Stage B data: GSM8K-Aug with DECOMP programs
**Items** are exactly CODI-20's: bench/data/gsm8k_aug/train.txt with CODI's filters (≤ 1000 tokens, answer starts with a digit), 382,859 items [MEASURED §209-2 / codi_port.load_items].
- **R_ν (per-draw name relabelling, §200; label-preserving) is on.** R_x (number resampling) is off, so that answers and targets are CODI's (decision D-7).

**DECOMP** (boxes/gsm_prog.py):
- Each equation "LHS=RHS" is parsed (Python ast; + − × ÷ only) into a binary tree and evaluated in post-order. So every step is one binary operation, and an equation with j operators gives j steps.
- **Operand provenance** for each leaf, by value:
  - an earlier step result (the most recent with that value);
  - a question literal (unique by value);
  - CONST (a number word or constant).
  - §200's ambiguity rules apply: a value matching both an earlier result and a question literal, or a non-unique literal, gives an ambiguous pointer. **Ambiguous pointer terms are masked**, never guessed.
- **Step values.**
  - Internal sub-steps: the exact value.
  - An equation's final step: the stated RHS when |exact − stated| ≤ the stated rounding. Otherwise that step's value terms are masked (the 0.31% "wrong" equations).
- **Coverage** [MEASURED scripts/gsm_mbin.py]: DECOMP parses 98.6% of train items (first 60k) and 93.9% of test items. On test, m_bin ≤ 12 covers 100% of the parsed items; the cumulative share is 0.259 at m_bin ≤ 2, 0.496 at ≤ 3 and 0.846 at ≤ 5.

**Per-pass targets (B1, gold, all m).**
- Step k goes to quantity slot k (k ≤ 12).
- E_t:
  - slots k ≤ t hold (op_k, ptrA_k, ptrB_k, sign_k, digits_k);
  - slots k > t are VOID in every head.
- The changed slot at pass t is slot t; unchanged = every other slot (copies of earlier results, and the VOID future). Value digits are weighted ×2.
- R1 is the decoder string "The answer is: N" (CODI's answer format) over t ∈ [m_bin, K_i].
- **K_i ~ U{m_bin, …, 16}** (not m_bin + U{0..4}). A fixed, non-oracle eval K′ = 14 is then in distribution, which matches CODI's fixed 6 latents (decision D-6). Unparsed items get R1 only, with K_i ~ U{4..16}.
- **Schedule.** The fixed curriculum is m_bin ≤ 2 to 4k, ≤ 5 to 12k, then all. **No adaptive controller**, so the 214-5 CurGate trap cannot occur.

**GSM8K self-teaching (B2): the design, and why it is not recommended for this spend.**
- **Depth axis = m.**
  - A B2 base would get gold per-pass targets only for m ≤ 3, and answers only for m ≥ 4.
  - Its round would self-label m 4–5 per-pass states with its own rollout.
  - **Verifier** (label-free except the gold final answer; outcome supervision is C4-legal):
    - ARITH-VALID: each decoded step's value = op(decoded operand values) exactly at 2 decimals; operands are question literals parsed from the text, or earlier decoded slots;
    - PROV-VALID: pointers point to existing literals or earlier non-VOID slots;
    - CONTIGUOUS: no VOID gap;
    - ANSWER-MATCH: the last value equals the gold answer.
- **Noise:** π_prog = P / (P + φ(1 − P)) (the §215 S2(a) identity), with φ = P(ANSWER-MATCH | a non-equivalent program).
  - φ > 0 on GSM8K: small numbers coincide. It is [ASSUMED 0.01–0.05], to be measured on a gold gate.
  - The "self beyond support" test would be m ≥ 6: **87 test items**, which gives an SE of about 0.05 at accuracy 0.3 and cannot separate H_shift from growth.
- **Not recommended.**
  - B2 needs its own base cell, about $3–7 (B1 has seen gold m ≥ 4 states).
  - Its depth test is under-powered.
  - It answers a C4-parity question, not C2.
- **Recommended instead:** carry B2 as a pre-stated follow-up whose trigger is B1 reading VB-C1-GSM. Theory decision D-8.

### 3.3 Contamination, exclusion and the same-filter rule
- **ProsQA-SYM.**
  - Training draws, the self pool (22300), GATE (22400) and test (22200; 1,000 per D for D 3..16) use disjoint seeds.
  - Every test and GATE item is excluded from training and pool streams by two signatures, with collisions logged: (a) the sha of the name-free canonical graph (generator-internal ids), and (b) the sha of the rendered text before relabelling. A hit is redrawn.
  - **D 13–16 test items are the untouched extrapolation set.**
  - Gold (BFS / D / target index) is used only in eval, GATE-1-L and LABEL-AUDIT-L, which run as separate processes.
- **GSM8K.**
  - Train = Aug train; test = the 1,319 GSM8K test items (CODI's file). They are disjoint by construction.
  - Class-A donors and resampled recipients (§5) are made from test items for eval only, and are never trained on.
- **Same-filter.**
  - ProsQA: every VB cell, and the optional A-TXT text-CoT cell, read the same SYM streams and the same test files.
  - GSM8K: B1 trains on CODI-20's 382,859 items, with no item dropped (unparsed parts are only masked). Every reading, CODI-20's included, is computed on the identical item subset (full 1,319; by m; DECOMP-parsed; class-A eligible), using CODI-20's per-item eval_items.json.

## 4. Mathematics

### 4.1 Depth lemma for the slot-subset bank (L1-L) [PROVED; the VB L1 proof carries over]
**Statement.** Any function of V_t is computed by a layered network of depth ≤ P + d·t (P = 6, d = 4). The answer after K′ passes has depth ≤ P + d·K′ + 2 from the question.

**Proof.**
- Every edge in the computation graph goes:
  - from H_6 (depth P) into a core layer;
  - from core layer ℓ of pass s into layer ℓ+1 of pass s, or into the bank read of a later pass;
  - from a stored version into SLVA's input of a later pass.
- So any path from a question token to V_t crosses at most t blocks of 4 core layers after the prelude.
- Reads of V_0 by later passes start at depth P, so they add no depth (L(0) = 0, Addendum 1 of the VB draft).
- The decoder string's blocks 0–9 have no question input, so they lie on no question-dependent path.
- Restricting versions to a slot subset only removes edges. ∎

### 4.2 Serial use is *not* forced by any theorem here; what is measured instead
- **[PROVED, trivially]** ProsQA-SYM reachability (≤ 82 concepts) and GSM8K arithmetic (≤ 12 binary steps, bounded digits) are finite functions. A wide enough network computes them in one pass. So no unconditional lower bound on passes applies at these sizes, as the VB draft already noted for n ≤ 16.
- **[ASSUMED] CAP-aff-L.** A chain of m passes of the *trained* model closes ≤ c₀ + (m − 1)·c hops (ProsQA) or binary steps (GSM8K), with c ≈ 1 and c₀ ≈ 1–3.
  - ProsQA evidence for a low c₀: DIRECT GPT-2 is at chance from depth 3 (D* = 3) [MEASURED 168-5]; §191 A's K′ 0 / 1 are at chance at every depth [MEASURED].
  - GSM8K evidence: one-pass accuracy is 0.14–0.17 (§208; CODI 0-latent 0.171) [MEASURED].
  - Pointer doubling (hops 2^t per pass) is representable (a 768-d slot can hold a ≤ 82-bit reach set), so c could exceed 1. R2 pins the *decodable* schedule to one hop per pass (§4.4), but not hidden computation.
- **Hence the decisive evidence for serial use is measured inside the model:**
  - PROBE(t) onset growing with D or m;
  - PROBE(1) at chance in the deep band;
  - VALUE-SWAP follow;
  - FUTURE-DECODE at chance (GSM8K).
- **C1 bands** are fixed before output: ProsQA D ≥ 5 (§3.1); GSM8K m ≥ 3 (gold "<<" count; n 879; CODI-20 scores 0.265 there).

### 4.3 Support identity and H_shift on ProsQA [PROVED, analogue of SELFDEPTH S0]
- **(a)** In A1, every R2 / R1 target has its last changed pass ≤ 8 (D ≤ 8, REACH saturates by D in the target-relevant sense). In A2, self items add stops s ∈ [D, 16].
- **(b)** For any t_d ≥ s_max, "expand while t ≤ t_d", a pass-count clock, matches every training target. A reach-saturation rule ("expand until an option is IN") matches them too.
  - So **depth beyond the training stops cannot be guaranteed by loss**: H_shift is irreducible, exactly as SELFDEPTH S0(b).
  - STALL separates the two rules in range. DEPTH-BEYOND-SELF-L (D 13–16, untouched) separates them out of range.
- **(c)** s ≥ D (Lemma S-L). A2's *effective* support is by self stop, not by D. A late stop (s > D) is legal and is reported (STOP-LAG = mean s − D on GATE items).

### 4.4 What R2 pins, and the routes it leaves open
- **Over-reach is penalised in decodable form [DERIVED].** Marking a concept at distance t + 1 at pass t is a CE error on an OUT target (ProsQA). Writing q_{t+1} at pass t is an error on a VOID target (GSM8K). R2 therefore pins the *readable* schedule to one hop / one binary step per pass.
- **Precompute-and-reveal is a zero-loss route [PROVED, given capacity].**
  - Pass 1 computes every q_k in a subspace R_G cannot read; pass t then copies q_t into the readable subspace of slot t, keyed on content ("the first VOID slot").
  - This meets R1 and R2 exactly, and it involves no clock, so STALL cannot see it.
  - It is the §204 F-P failure ("a one-pass chain") in VB form.
  - **Detectors [DERIVED]:**
    - (i) PROBE(1): a fresh 2-block coda reads the answer after pass 1. It is high under precompute, about one-pass level under serial computation.
    - (ii) FUTURE-DECODE(t, k): a fresh linear probe for q_k's digits from all slot rows of V_t, k ≥ t + 2. Under serial computation it sits at the V_0 baseline FD(0, k); under precompute FD(1, k) ≈ FD(k, k).
    - (iii) VALUE-SWAP arm (i), slot t only. Under serial use the later steps read the spliced q_t (follow). Under reveal they read their own precomputed q_{>t} (own).
- **Recompute from V_0 stays open** (V_0 is in every core layer's bank).
  - Under CAP-aff-L a pass can recompute at most c₀ levels from V_0 [PROVED UNDER CAP-aff-L, as Addendum 1].
  - VALUE-SWAP's own share reads it. On ProsQA, recompute at t ≥ 3 needs more than one hop per pass (§4.6).

### 4.5 Noise under verifier filtering
- **ProsQA [PROVED].**
  - **Lemma S-L:** P1 ∧ P2 ∧ P3 ⇒ M_t ⊆ R_t for all t (induction: a mark added at t has an in-edge from M_{t−1} ⊆ R_{t−1}).
  - With SYM's invariant o_P ∉ R_t, a kept chain's label is o_R: **π_label = 1, φ = 0**; and s ≥ D.
  - **Corollary:** under SOUND-ONLY R2, every self target (IN at passes ≥ first mark; R1 from s ≥ D) equals the gold target at the same (slot, pass). Only the changed / unchanged weighting of late marks differs. **The supervised label noise is 0 in every round, as long as each round's labeller is verified by SOUND on SYM.** Lee's avalanche (label accuracy < 0.9 ⇒ collapse, LIT_ITERATED_SELFTRAIN) cannot be triggered by labels.
  - **What can go wrong instead [DERIVED]:**
    - (a) *Coverage bias:* kept items are those the model partly solves. Read by keep-yield by D, and by MEMO-GAP-L (pool vs fresh accuracy).
    - (b) *Missing OUT terms:* over-marking on deep items costs nothing on self items (Goodhart G7). Read by FALSE-IN (spurious IN rate vs gold BFS on GATE items, after A2).
    - (c) *Late R1 start:* a stop at s > D teaches "answer later" at deep D (STOP-LAG).
- **GSM8K (B2 only) [PROVED identity; ASSUMED φ].**
  - π_prog = P / (P + φ(1 − P)).
  - A gold gate measures π_prog, counting a kept program as correct if it equals a gold program up to commutativity and reassociation of equal values.
  - B2's gate needs π_prog ≥ 0.95, which Lee's threshold sits below.
  - Noise compounds per round as SELFDEPTH S2(b): ē_{q+1} ≤ ē_q(1 + κφ n_q / P_q).

### 4.6 PERSON-SWAP identifiability on ProsQA [PROVED]
- **Construction.** The donor is the recipient with the question's person token R replaced by P. Everything else is identical, including names, statements and order.
- **(a)** The prelude is causal and the question is the text's last sentence. So V_0 rows at every premise token, **every concept slot included**, are bitwise equal in donor and recipient; only the question rows differ.
- **(b)** Under the one-hop-per-pass schedule the donor's V_t encodes R_t(P). Splicing it into the recipient at pass t, a pass map that expands the stored frontier reaches o_P at pass D: dist(P, o_P) = D, and the frontier is at P-depth t. So y_hyb = o_P (follow) is well defined.
- **(c)** Re-deriving from the recipient's question needs D passes. The passes left are K′ − t = D + 2 − t, which is enough only for t ≤ 2. **So at t ≥ 3, an "own" answer requires more than one hop per pass from V_0 (over-reach) or a non-serial route.** Decisive t ∈ {3..D−1}; t = 2 is descriptive.
- **(d)** The candidate answer is binary, so follow + own = 1. Random disruption gives about 0.5, so the bar is follow ≥ 0.8 (§7).
- **VALUE-SWAP (iv) (V_0 masked after t) is not informative on ProsQA.** The edges live only in V_0, so masking it stops any continuation by construction. It is reported descriptively. On GSM8K, (iv) is informative only for "pure-chain" items, whose later steps use no question literal.

### 4.7 Why VB-GSM might avoid §207 / §208 where §204 F-P did not, and why it may not [CONJECTURE, with the record]
| difference from F-P | expected effect | evidence against |
|---|---|---|
| R is **linear per slot**. F-P's per-pass value was read by an aux *decoder* that could compute from pointers | the slot itself must hold the value | F-P's VALUE-EXACT at pass 1 was only 0.338: the single step was hard for GPT-2 small at 4 epochs |
| **Bounded core** fed the previous version, with the question only as layer-6 V_0 rows. F-P re-read the question at full depth every pass (INJECT) | recompute costs depth (L1-L) | pass 1 (10 core-path layers) is still GPT-2-deep. m_bin ≤ 2 precompute is expected and legal (VB draft §9) |
| **One binary op per pass** (DECOMP). F-P used whole equations, 30% of them multi-op | a uniform, learnable step; an exact verifier | more passes: mean m_bin 3.80 on the parsed test items (4,706 / 1,238) vs mean m 3.25 [DERIVED from scripts/gsm_mbin.py's distribution] |
| **Matched exposure** (20 epochs, 7.66M examples). F-P ran 4 epochs | §209 → §210 showed budget is decisive for CODI | none; this one is fair |

**Honest prior.** The §208 failure mode (PASS1-PRECOMPUTE, latent accuracy = one pass) is the single likeliest outcome for B1 (P ≈ 0.45, §4.8). B1 is designed to *detect* it cheaply: interim look at 24k, §9.

### 4.8 Predictions [CONJECTURE, before any output]
| cell | P(entry) | P(DEEP / ABSORB) | P(decisive C1) | other |
|---|---|---|---|---|
| A1 (LANG-SLVA on SYM D 3–8) | 0.6 | DEEP 0.4 | **VB-C1-L 0.2** | PROBE(1) ≤ 0.6 at D ≥ 6: 0.8 (if DEEP). DEPTH-L (D 9–12) central 0.55. COUNT-KEYED 0.15 |
| A2 (self round 1, D 9–12) | — | ABSORB 0.5 (0.65 given A1 DEEP) | **SD-C2-L 0.05** (H_shift 0.02 / H_grow 0.3 / H_full 0.7 given ABSORB) | MEM-TRADED-L 0.5; CAUSAL-DEEP-L given ABSORB 0.75 |
| B1 (LANG-SLVA-GSM, 20-epoch matched) | 0.75 | — | **VB-C1-GSM 0.08** | acc ≥ 0.359 (CODI-20 − 0.02): 0.12; BEATS (≥ +0.03): 0.04; ONE-PASS-CHAIN-L 0.45; STEP-CAP ≥ 0.8: 0.3 |

**Basis.**
- Nested VB gave 2 of 3 cells VB-C1 in range [MEASURED].
- But language adds multi-token name binding over T ≈ 450, and SYM puts branching at every depth. §168's history is memorisation unless relabelled.
- GSM8K: §204 / §208.

## 5. Readings and scorer field contracts

**General rules.**
- Every reading runs on the frozen model in local fp32, at the eval_bs that G-EVMEM pins (the "original eval_bs" for every re-check). All cells of a stage use **one local card**, recorded in every file.
- **Per-item outputs are stored behind every share or threshold** (items / pairs arrays).
- Only depth bands that pass the LEVEL-PANEL gate enter a ProsQA reading. A reading whose whole range fails the gate is UNREADABLE (= not met).
- Each part is merged into the eval file as it finishes, so a late crash keeps the earlier parts.

### 5.1 ProsQA: eval_lvb.json (A1, A2, and A2-CTRL if run); SYM test, 1,000 items per D, D 3..16
| part | fields | definition |
|---|---|---|
| panel | `panel {D: {n, max_heuristic, max_which, gate_pass, bidir: {r: acc}}}` | LEVEL-PANEL + BIDIR on the test files (G-L0); copied, not recomputed |
| acc | `acc_d2 {D}` (K′ = D + 2); `acc_k {K′: {D}}` K′ 0..20; `acc_fixed12 {D ≤ 8}`; `items {D: {K′: [0/1]}}`; `margins` | candidate accuracy: log p("### o_R.") > log p("### o_P.") from the coda after K′ passes |
| probe | `probe {t: {acc_by_D, items, train_ce_final}}`, t ∈ {0,1,2,3,4,6,8,10,12} | a fresh 2-block coda copy (from the trained coda; LM head tied and frozen) reads V_t + bank → candidate CE. 20k fresh gold-support items (D 3–8, seed 22500), 2k steps, bs 64, AdamW 1e-4. Model frozen |
| vswap | `vswap {n_pairs, follow_ii, own_ii, follow_i, own_i, by_D, by_t, t2_descriptive, pairs: [...]}` | PERSON-SWAP (§4.6). Recipients: test D 4–8, t ~ U{3..D−1} (rng 22610), t = 2 kept separately. Clean = recipient and donor each correct at K′ = D + 2. (ii) the stored entry V_t := donor's V_t (bank and SLVA source); (i) only the slots whose REACH decode differs between donor and recipient at t |
| stall | `stall {recovery, by_D, items}` | at t = ⌈D/2⌉ the stored entry is V_{t−1}. recovery = pooled acc(D 4–8, K′ = D + 1, stalled) / pooled acc(D 4–8, K′ = D, intact) |
| hist24 | the §213-21 contract (intact / bank_ablate_slva / self_ablate / dup_newest at K′ 24, D 3–8) → STAB_self, STAB_bank, dup_newest | the memory channel, KEEPS-SELF-STAB as SELFDEPTH_v1 §4.3 |
| causal_deep | the §217 contract at D 9..16 (vswap + stall unchanged except the D range; competent band B = {D : a(D) ≥ 0.95}) | C1 at the new depths |
| labeller (A2) | `gate1 {D: {n, kept, yield, omit_rate, false_in, stop_lag}}`, `label_audit` (5,000 consumed records vs gold) | Lemma S-L check: π_label must be 1.000. A single wrong label = a code bug, which voids A2 |

**Decisive readings: A1, comparator-free, D 4–8 within the C1 bands.**
- **DEEP-L:** acc_d2 ≥ 0.90 at every D in 4–8.
- **PROBE-GAIN-L:** pooled ACC(K′ = D + 2) − max(PROBE(0), PROBE(1)), over D 5..8 (the C1 bands in support), ≥ **0.35**. Chance is 0.5, so the maximum possible gain is 0.5.
  - PROBE-VALID: PROBE(12) ≥ ACC − 0.05.
  - **ONESHOT-L** = PROBE(2) ≥ 0.8 at D = 8 (falsifies the premise).
- **SERIAL-ONSET-L.** This is a **band-level** onset, because a per-item onset is coin-biased with binary answers.
  - ON(D) = the smallest t in the probe set with PROBE(t′) ≥ 0.90 at depth D for every t′ ≥ t; never reached = failure.
  - Required: the OLS slope of ON(D) on D (D 4–8) ≥ 0.6, and ON(8) − ON(4) ≥ 3.
- **SERIAL-STEP-L:** pooled follow_ii ≥ **0.80**, and ≥ 0.60 at every D 5–8, with ≥ 100 clean pairs pooled; otherwise UNREADABLE.
- **NO-CLOCK-L:** STALL recovery ≥ 0.9.
- **VB-C1-L** = DEEP-L ∧ PROBE-GAIN-L ∧ SERIAL-ONSET-L ∧ SERIAL-STEP-L ∧ NO-CLOCK-L.

**Labels.**
- ONESHOT-L.
- NO-ENTRY: plain answer CE never ≤ 0.35 in two consecutive 1k bins to the end.
- LOSES-ONE-PASS: ¬DEEP-L ∧ PROBE-GAIN-L < 0.15.
- COUNT-KEYED: STALL < 0.9, or DEEP with REACH(0.8) ≤ 8 and DEPTH-L < 0.5.
- RECOMPUTE-L: own_ii > 0.3 at t ≥ 3.
- RNN-IN-FUNCTION / HISTORY-USED: the §213 swap triple and bank ablation, descriptive.

**Depth readings.**
- **DEPTH-L** (A1, descriptive, C2 proxy before self) = mean acc_d2 over D 9–12.
- **A2 claims** (the SELFDEPTH_v1 forms):
  - ABSORB-L = min acc_d2 over D 9–12 ≥ 0.95.
  - SD-SELF-L = ABSORB-L ∧ VB-C1-L (re-read on A2) ∧ K′24-STABLE (min acc_k['24'] over D 3–8 ≥ 0.97).
  - **SD-C2-L = SD-SELF-L ∧ DEPTH-BEYOND-SELF-L** (mean acc_d2 over D 13–16 ≥ 0.80; the untouched set).
  - MARGIN-GROWS-L, the H_shift panel: the bar is A1's own acc_d2 curve translated by +4, computed by the scorer from A1's file before A2 output (the §216 §4.3 formula pattern).
  - LIT-C2-L (literal C2): mean over D 15–16 of [acc_k[K′ = 18] − acc_k[K′ = 16]] ≥ 0.10.
- **Kept readings (must not break, A2):** VB-C1-L at D 4–8, LENGTH (acc_d2 at D 3–8 ≥ 0.99), K′24-STABLE.
- **MEM:** KEEPS-SELF-STAB-L (STAB_self ≥ 0.5), else **MEM-TRADED-L**, stated beside any positive. **CAUSAL-DEEP-L** (§217 rule) is stated beside every A2 positive.

### 5.2 GSM8K: eval_lvb_gsm.json (B1), all 1,319 test items, CODI's eval protocol
Protocol: greedy, max_new 40, answer = the last number after "The answer is:", correct iff |pred − gold| < 1e-4.

| part | fields | definition |
|---|---|---|
| acc | `acc_fixed14`, `items [{ok, pred_txt (full, not cut to 80 chars)}]`, `by_m {m}`, `by_mbin`, `acc_k {K′}` K′ 0..16, `acc_oracle` (K′ = m_bin + 2; descriptive), `acc_halt` (content halt: the first t whose new slot decodes VOID, +1; descriptive) | the **headline is acc_fixed14** (non-oracle, as CODI's fixed 6) |
| codi_pair | `codi_pair {delta, se_paired, n, by_m, by_parse}` vs gsm_codi_20ep_s1/eval_items.json | the identical item list, asserted by sha |
| probe | `probe {t: {acc_by_m, items}}`, t ∈ {0,1,2,3,4,6,8,12} | a fresh coda copy (2 blocks + decoder path) on 20k train items, 2k steps |
| future | `future {t: {k: {acc, n}}}` for t ∈ {0,1,2,4}, k ≥ t + 2 | a fresh linear probe (concatenated slot rows of V_t → q_k's digits, exact match), trained on 20k train items, read on DECOMP-parsed test items with m_bin ≥ k |
| vswap | `vswap {n_pairs, follow_ii, own_ii, neither_ii, follow_i, own_i, follow_iv, own_iv (pure-chain only), by_m, pairs}` | class A: step t ∈ {2..m_bin−1} with both operands question literals used by no other step. The donor redraws them by §200's draw_literal law, all downstream values recomputed exactly, y_donor ≠ y. Recipients: test items + 5 R_x variants per item (rng 22720; eval only). Clean = both correct at K′ 14. (ii) whole V_t; (i) slot t only |
| stall | `stall {recovery, by_m, items}` | t = ⌈m_bin/2⌉ (m_bin ≥ 3); stalled at K′ = m_bin + 1 / intact at K′ = m_bin |
| stepcap | `stepcap {by_t, by_op, by_digits}` | slot t's exact decode at pass t on items whose slots < t decode exactly (R_G, descriptive mechanism) |
| gsmswap | §207's STATE-SWAP / INPUT-SWAP arms in VB form (V_0 rows replaced after k) | the §207 label rule applied, for comparability; descriptive |

**Decisive readings (B1; C1 band m ≥ 3).**
- **PROBE-GAIN-GSM:** acc_fixed14(m ≥ 3) − max(PROBE(0), PROBE(1))(m ≥ 3) ≥ **0.10**. With n 879 the SE is about 0.02.
- **SERIAL-STEP-GSM:** follow_ii ≥ 0.6 ∧ own_ii ≤ 0.3, on ≥ 100 clean pairs; otherwise UNREADABLE.
- **NO-CLOCK-GSM:** recovery ≥ 0.9.
- **NOT-ONE-PASS-CHAIN:** ¬(FD(1, k) ≥ 0.8·FD(k, k) pooled over k ≥ 3, with FD(k, k) ≥ 0.3).
- **VB-C1-GSM** = PROBE-GAIN-GSM ∧ SERIAL-STEP-GSM ∧ NO-CLOCK-GSM ∧ NOT-ONE-PASS-CHAIN.

**Pre-stated failure readings: the §207 / §208 detectors.**
- **PASS1-PRECOMPUTE-GSM** = PROBE-GAIN-GSM < 0.05 ∨ acc_k[14](m ≥ 3) − acc_k[1](m ≥ 3) < 0.03.
- **ONE-PASS-CHAIN-L** = the negation of NOT-ONE-PASS-CHAIN.
- **RECOMPUTES-FROM-INPUT-L** = own_ii ≥ 0.5.
- **STEP-NOT-LEARNED** = stepcap pooled < 0.5.

**Accuracy label (C3-type, one seed), Δ = acc_fixed14 − 0.3791 (paired SE about 0.014):**
- **BEATS-CODI20:** Δ ≥ +0.03.
- **MATCHES:** |Δ| < 0.03.
- **BELOW:** Δ ≤ −0.03.

### 5.3 Scorer
- **theory/score_lvb.py** (Theory writes it). It is fixture-checked before any output on synthetic panels:
  - VB-C1-L; ONESHOT-L; COUNT-KEYED; RECOMPUTE-L; NO-ENTRY; a dead run; a panel-failed band; < 100 pairs;
  - an A2 SD-SELF / SD-C2 / MEM-TRADED / CONTAMINATED / a π_label < 1 bug;
  - B1 VB-C1-GSM / PASS1-PRECOMPUTE-GSM / ONE-PASS-CHAIN-L / BEATS / BELOW;
  - pending.
- It returns **pending** while any registered file is missing (the 215-7 rule).
- It reads only the field paths named above. Before any data exist it is greped against langvb_eval.py and run on a synthetic eval (the scorer field-path check).

## 6. Claims, wording, censoring, Goodhart

### 6.1 Claims and pre-stated wording
| claim | wording if met | never worded as |
|---|---|---|
| VB-C1-L (A1) | "supervised serial latent graph search in a pass-global attention memory, on natural-language premises (ProsQA-SYM, shortcut-exchangeable by construction, LEVEL-PANEL-gated), D 5–8, one seed" | C4 (REACH targets are simulator states, in no text CoT); C2; "unsupervised discovery" |
| DEPTH-L (A1) | descriptive: "depth past the gold support before self-training" | C2 |
| SD-SELF-L (A2) | "latent search depth extended from 8 to 12 by training on the model's own SOUND-verified chains (labels gold-equal by SYM's construction, π = 1; no gold state beyond D 8); in support" | C2, C4 |
| SD-C2-L (A2) | "depth extrapolation past the self-labelled range (D 13–16, untouched SYM items; C2 proxy; one lineage: an instance)", with LIT-C2-L, MEM-TRADED-L and CAUSAL-DEEP-L on the same line | literal C2 unless LIT-C2-L holds |
| VB-C1-GSM (B1) | "CoT-distilled serial latent arithmetic in a pass-global attention memory on GSM8K (m ≥ 3; per-pass targets from GSM8K-Aug's equations, C4-legal as distillation)" | C2; "beyond text" |
| BEATS / MATCHES-CODI20 (B1) | "at matched exposure (7.66M examples), same items, CODI's eval protocol, one seed each" | C3 (no matched text-CoT arm here; §200 C was 4 epochs) |
| PASS1-PRECOMPUTE-GSM / ONE-PASS-CHAIN-L | "the §208 / §204 failure recurs in VERSION-BANK form" | — |

Two further wording rules:
- **The constraint check** (the user's pass-global attention memory) follows §213. RNN-IN-FUNCTION (SLVA one-hot on the newest version) is reported, and **is not credited as attention-memory use**. HISTORY-USED is reported beside.
- **The §1.2 finding** (the p8_na anchor cue) is reported as a data fact. Whether §191's models use it is A0-1's reading, never assumed.

### 6.2 Censoring (rule b: each fails every reading of that cell; never-reached events are failures)
- **Dead run.** The log is short of the registered steps; a guard stop counts.
- **NO-ENTRY.**
  - ProsQA: plain answer CE never ≤ 0.35 in two consecutive 1k bins.
  - GSM8K: Aug-valid accuracy (500 items, every 2k steps) ≤ 0.15 at every check to the end, i.e. never above §208's one-pass level.
- **CONTAMINATED.** Any pool/test/gate signature hit, any gold key in a pool record, self_reuse > 0, or pool exhaustion.
- **ARGS-DIFF.** A mismatch against the reference args, beyond the pre-listed keys.
- **LABEL-BUG (A2).** Any wrong label in GATE-1-L or LABEL-AUDIT-L. Lemma S-L makes π_label < 1 impossible, so a wrong label is a code fault: A2 is voided and re-gated.
- **UNREADABLE (= not met).**
  - < 100 clean swap pairs;
  - < 500 STALL items;
  - PROBE-INVALID;
  - a panel-failed range;
  - an eval-card or eval_bs mismatch;
  - eval_lvb.json items_sha mismatch.
- **Never-reached events.** A never-reached onset is a failure; a never-halting acc_halt item counts as wrong; a never-crossed n50-type crossing reads "> 16", never missing.

### 6.3 Goodhart routes
| route | status | read by |
|---|---|---|
| G1 pass-1 precompute (global) | legal at shallow depth; bounded at depth by L1-L + CAP-aff-L [ASSUMED] | PROBE(1) / (2), ONESHOT-L, PROBE-GAIN |
| **G1′ precompute-and-reveal (GSM8K)** | **open: zero-loss if capacity allows [PROVED]** | FUTURE-DECODE (ONE-PASS-CHAIN-L), PROBE(1), VALUE-SWAP (i) |
| G2 recompute from V_0 | open (V_0 in every core layer's bank); bounded by depth | VALUE-SWAP own share; on ProsQA at t ≥ 3 "own" needs over-reach (§4.6) |
| G3 coda solving | closed for question access (no V_0 in the coda; the decoder string is question-free and bank-free in blocks 0–9). It can read the option slots' IN bits, which is the intended read | PROBE vs ACC; the coda is never a decisive reader |
| G4 data cues | p8_na anchor cue (§1.2) → SYM exchangeability [PROVED] + LEVEL-PANEL gate + per-draw relabelling (NAME-MEM) | the panel by band; MEMO-GAP-L |
| G5 pass clock / count keying (incl. GSM8K's "slot t at pass t" schedule) | open in principle (S0-L(b)); no pass token | STALL (NO-CLOCK-L / -GSM); DEPTH-BEYOND |
| G6 over-reach (> 1 hop per pass, hidden) | decodable over-reach penalised (R2 OUT / VOID); hidden not | ON(D) slope; PROBE(t) above schedule |
| G7 over-marking on self items (no OUT terms) | open in A2 | FALSE-IN on GATE items after A2; ABSORB; kept readings |
| G8 sabotage / timed release | no incentive: every term is −log q(fixed target) [PROVED, VB §5.6] | — |
| G9 history traded for depth (MEM) | expected (2 of 2 nested rounds) | KEEPS-SELF-STAB-L → MEM-TRADED-L |
| G10 adaptive-controller contamination (214-5) | CurGate reads **plain gold items' answer CE only** (an assert in the trainer: self / augmented batch types never enter `GATE.update`); GSM8K uses a fixed schedule | G-L7 gate; log `plain_answer_ce` |
| G11 length as a depth proxy (SYM text grows with D) | it cannot pick the option (both are depth D), but could time a stop | STALL; DEPTH-BEYOND; the descriptive text-length-matched split (D 13–16 items with T ≤ the train max) |
| G12 name memorisation | per-draw relabelling into NONCE-300 / ENT17; fresh items per draw | MEMO-GAP-L; the train-fit vs test gap |

## 7. OC per reading
One seed per cell, so a rate claim needs ≥ 3 seeds and the user's OK. Sampling SEs are at the registered n. The binding uncertainty is the model-behaviour priors [ASSUMED], not sampling. Realised spend at drafting: **$0**.

| reading | n (expected) | P(pass \| no effect) | P(pass \| effect) | what "no effect" means |
|---|---|---|---|---|
| PROBE-GAIN-L ≥ 0.35 | 4,000 items (D 5–8) | ≈ 0 (gain ≈ 0 ± 0.01) | ≈ 0.8 (nested 3 / 3 cells ≥ 0.87) | pass-1 precompute (PROBE(1) ≈ ACC) |
| SERIAL-ONSET-L (slope ≥ 0.6, gap ≥ 3) | 8 probe points × 5 D | ≤ 0.05 (a flat ON(D)) | ≈ 0.7 | depth not tied to passes |
| SERIAL-STEP-L follow ≥ 0.80 | ≥ 1,000 clean pairs if DEEP | ≈ 0 under recompute (own ≈ 1); ≈ 0 under random disruption (follow 0.5 ± 0.016) | ≈ 0.85 (nested follow_ii 0.95–1.00) | recompute / no stored-state use |
| NO-CLOCK-L recovery ≥ 0.9 | ≥ 2,000 STALL items | ≤ 0.05 (count-keyed N3: 0.499) | ≈ 0.9 (N1 / N2 / D3-T 0.997–1.000) | pass-count clock |
| **VB-C1-L** | — | **≤ 0.02** | **≈ 0.5** (the component passes are positively correlated; the independent product is 0.43) | no serial use |
| DEEP-L | 5,000 | — | — | competence, not a test |
| ABSORB-L (A2) | 4,000 | ≈ 0.05 (A1's D 9–12 at about 0.55 central) | ≈ 0.75 (nested R1 / R2: 1.000 / 0.997) | self items not absorbed |
| **SD-C2-L** (D 13–16 ≥ 0.80) | 4,000 | ≈ 0.02 under H_shift (sampling SE 0.008; lineage σ 0.06–0.12) | H_grow ≈ 0.3; H_full ≈ 0.7 | exact translation by +4 |
| KEEPS-SELF-STAB-L | — | — | P(keeps ≥ 0.5) ≈ 0.5 (nested: 0 / 2 rounds kept) | — |
| PROBE-GAIN-GSM ≥ 0.10 (m ≥ 3) | 879 | ≈ 0 (SE about 0.02) | ≈ 0.85 | one-pass accuracy (§208) |
| SERIAL-STEP-GSM | ≥ 100 pairs (yield about 10% of candidates; recipients incl. 5 R_x variants per test item) | ≈ 0.01 under recompute (§207 CARRIED ≤ 0.005) | ≈ 0.6 (numeric answers; "neither" possible) | memory as scaffold |
| NOT-ONE-PASS-CHAIN | ~500 items with m_bin ≥ 3 per (t, k) | ≈ 0.05 under precompute | ≈ 0.8 | §204's one-pass chain |
| **VB-C1-GSM** | — | **≤ 0.01** | **≈ 0.4** | one-pass / recompute |
| BEATS-CODI20 (Δ ≥ +0.03) | 1,319 paired | ≈ 0.02 (paired SE ≈ 0.014) | ≈ 0.98 if the true Δ is +0.06 | equal accuracy |

**Headline OC (priors shaded by the record).**
- P(VB-C1-L) ≈ 0.2 (P(serial use learned) ≈ 0.35 × 0.5).
- P(SD-C2-L) ≈ 0.05; P(VB-C1-GSM) ≈ 0.08; P(B1 ≥ CODI-20 − 0.02) ≈ 0.12.
- **P(at least one decisive C1 positive in the design) ≈ 0.25.**
- A positive is strong evidence. A null is weak, and the most likely null is informative: PASS1-PRECOMPUTE-GSM or ONE-PASS-CHAIN-L on GSM8K, and NO-ENTRY / LOSES-ONE-PASS on SYM.

## 8. Gates
All gates are $0 and local. Each writes its own gate JSON as it finishes (gates are incremental). All are green before any spend, and Audit's diff review comes before any launch.

### 8.1 Stage A0 (ProsQA)
| id | gate / row | pass rule |
|---|---|---|
| **A0-1** (descriptive; for Theory / Audit) | ANCHOR-CUE-RELIANCE of the kept §191 models (s191_stepaux_relabel_s1, s191_textcot_relabel_s1; model.pt kept). Re-eval on p8_na_test with per-item outputs, at the original dtype and bs, split by anchor proximity (pro 2,630 / tie 511 / anti 78 items at depth ≥ 5 [MEASURED scripts/anticue.py]) and on NN items rendered in CON38 at D ≤ 8 | **CUE-RELIANT** iff acc(tie) ≤ acc(pro) − 0.15 ∧ acc(tie) ≤ 0.75. Pre-stated now, before any output |
| **A0-2** | the SYM generator invariants on 100k draws (dist = D, o_P ∉ Reach(R), o_R ∉ Reach(P), T ≤ 1000) + **LEVEL-PANEL** on the test / gate / pool samples, n ≥ 1,000 per band | 0 invariant failures. **Band usable iff every heuristic at every level < 0.6** (binding addition 1). With E = ½ exactly (§3.1), the max over about 100 heuristic × level cells is about 0.55 at n 1,000 [DERIVED: 0.5 + 3.2 SE] |
| A0-3 | G-L1 identity | (a) all-positions slots, K = 1, bank masked == GPT-2, fp64 ≤ 1e-9; (b) LangSLVA with S = all positions on nested inputs == D3-T's SLVA rollout, bitwise fp32 |
| A0-4 | G-L1b init-drift smoke, 2k steps local (memory-capped) | the drift falls below its init value; trained-range CE falls (the 211-6 rule) |
| A0-5 | path-specific gradflow (scratch/gradflow_check.py pattern) | nonzero grad from R1, R2-changed, R2-unchanged and each R_G head; ∂L/∂bank K/V ≠ 0; ∂L_ans/∂V_1 ≠ 0 with the SLVA input stop-gradiented |
| A0-6 | REACH re-render (an independent BFS) on 100k items; S-L fuzz (random sound chains ⊆ R_t) on 100k | 0 diffs; 0 violations |
| A0-7 | G-L7 CurGate reference-only | a unit test feeding self / augmented batches: `GATE.update` receives plain gold answer CE only (asserted in the trainer) |
| A0-8 | mem_check at the largest shapes (A1: D 8, max T, K 12; A2: self D 12, K 16; eval: D 16, K′ 20) → **G-EVMEM** pins eval_bs per part | peak ≤ budget (24 GB host; 8 GB local eval) |
| A0-9 | smoke at full data scale: the generator / pool builder at full size (timed), the loader over the full stream | completes; the projected pool time is logged |
| A0-10 | plain-path smoke | D3-T constructs + 1 step; nested vbank_eval acc on 64 items bitwise = eval_vb |
| A0-11 | off-host mirror + **tested resume** | kill a smoke at step 300 and resume; losses at 301–400 bitwise = an uninterrupted run (same card) |
| A0-12 | ARGS-DIFF printed in the launch log vs D3-T's args.json | only the pre-listed new keys differ |
| A0-13 | G-TIME on forced worst-case batches | admission (§9) |
| A0-14 | Audit's diff review: decorators by ast, return shapes, the bank cache's gradient, the eval hooks | approved |
| **A0-15** (A2 only, before pool use) | **GATE-1-L** on 2,000 GATE items per D 9–12, with gold: π_label, yield, omit_rate, false_in, stop_lag; labeller selection L-ROLL vs L-RR by §3.1's rule | π_label = 1.000 (any miss = LABEL-BUG, a code fault); yield ≥ 0.2 at every D, else no A2 (a pre-stated C2-negative outcome) |
| A0-16 | A2 GO rule | A1 reads DEEP-L ∧ NO-CLOCK-L (VB-C1-L not required: self-teaching tests depth), and A0-15 passes |

### 8.2 Stage B0 (GSM8K)
| id | gate / row | pass rule |
|---|---|---|
| B0-1 | DECOMP over all 384,622 train lines: coverage, provenance ambiguity, per-term mask rates, the m_bin table | logged. A coverage of < 0.95 items with ≥ 1 targeted step → Theory reviews before launch |
| B0-2 | CODI-20 re-score on the eval card from trainable.pt with codi_probe | per item == gsm_codi_20ep_s1/eval_items.json (acc 0.3791). Required for the paired reading |
| B0-3 | B-STEP smoke (local, 2k steps, m_bin ≤ 2, memory-capped) | descriptive learnability preview; read by no rule |
| B0-4 | the A0-3 / 4 / 5 / 8 / 10 / 11 / 12 / 13 / 14 analogues for the GSM8K path (R_G heads, coda_dec) | as A0 |
| **B0-5** | **B1 interim look (pre-stated)** at 24k steps, from Aug valid (500) only, never test | **GO to 120k iff** valid acc_fixed14 ≥ 0.17 ∧ (valid acc_k14 − acc_k1) ≥ 0.03. Else stop: B1 is scored at 24k as **B1-24k**, with every reading as registered and the accuracy label against **CODI4 (0.189)**, the exposure-matched comparator |

### 8.3 Launch conditions (every cloud cell)
- RTX 3090 only (one GPU class).
- Off-host checkpoint mirror armed at launch, with the resume tested (A0-11).
- Drain waiters armed.
- Per-cell guard with GUARD_STOP.
- No module edit while any deploy is in flight.
- Scorer fixtures done.
- Checkpoints deleted at train end (model.pt kept until every registered reading is read).

## 9. Cost, caps, admission
**Speed law** [ASSUMED]: s/step = max(0.42, 0.42·(0.4 + 0.6·work ratio)), anchored on D3-T's measured 0.42 s/step ($1.12 / 48k at about $0.20/h). Work ratios come from scripts/cost_model.json [DERIVED]. G-TIME replaces every projection, and the realised $ replaces G-TIME.

| cell | steps | est. s/step | projected $ (range) | admission (G-TIME, forced worst case) | guard | status |
|---|---|---|---|---|---|---|
| A1 LANG-SLVA on SYM (D 3–8) | 24k × 64 | 0.82 (prelude-dominated, T ≈ 450) | **1.2** (0.8–2.5) | ≤ $2.5 | $2.8 | required |
| A2 self round 1 (D 9–12 pool; FT 12k) | 12k × 64 | about 1.0 (½ self D 12-class at 1.15, ½ plain at 0.82) | **0.75** (0.5–1.5) | ≤ $1.5 | $1.7 | on A0-16 |
| A2-CTRL (p_self 0, the same stream) | 12k × 64 | 0.82 | 0.55 (0.4–1.1) | ≤ $1.1 | $1.2 | optional (attribution) |
| A-TXT (TEXT-COT on SYM, §191 B recipe) | 12k × 64 | 1.35 (§191 B measured: 16,180 s / 12k) | 1.0 (0.8–1.4) | ≤ $1.3 | $1.4 | optional (C3 on SYM) |
| B1 LANG-SLVA-GSM (matched exposure 7.66M) | 120k × 64 | 0.45–0.9 (launch-bound; mean K about 10 under K_i ~ U{m_bin..16}) | **3.3–6.6** (central 4) | ≤ $8 | $9 | required; **interim stop at 24k ($0.7–1.3) if B0-5 fails** |
| Stage A0 / B0, labelling, evals | — | local | $0 | — | — | — |
| bad hosts + egress | — | — | about +10% | — | — | — |

**Totals.**
- **Required (A1 + A2 + B1): projected $5.3–10.6, central about $6.**
- With both optional cells: +$1.6–2.5.
- **One design, one family cap: $25** (GUARD_STOP on the family). The per-cell guards sum to $16.1 with the optionals, ≤ $25.
- B1's projection plus the realised A spend fits the cap with a margin of more than $10. **No item currently projects above the remainder.**
- The rule stands anyway: if B1's G-TIME projects above $25 minus realised spend, there is no launch, and it goes to Audit and the user. Nothing is dropped silently, and steps are never reduced to fit.
- **Under the user's $30 rule** this is one task ≤ $30, so it is Audit-approvable. MEM-TRADED-L after A2 (§6) still routes any **further** self round to the user regardless of cap, per SELFDEPTH_v1 §9.
- **Credit check.** Flag if account credit is < $30 at launch (memory rule). The last recorded credit was about $98 (CURRENT_TASKS, 10-09), which is stale: Experiments should re-read it.

## 10. Open questions for Theory; recommended Stage A cell list

### 10.1 Decisions for Theory (with my recommendation)
| # | decision | recommendation | why |
|---|---|---|---|
| D-1 | Stage A substrate: SYM (new) / NN / p8_na | **SYM** | p8_na fails the level panel at depth ≥ 5 (anchor proximity 0.86–0.93) and NN fails everywhere (anchor parents' in-degree 0.61–0.67) [MEASURED]. SYM is exchangeable by construction [PROVED] and branches at every depth (it fixes 168-6) |
| D-1b | Send §1.2's anchor-cue finding to Audit and run A0-1 on the §191 models ($0) | **yes** | §191's NAME-MEM 0.950 is not distinguishable from chain-chase + anchor cue without it |
| D-2 | Decoder string path: blocks 0–9 alone, or the VB draft's prelude-only | blocks 0–9 alone | layer-10 weights see layer-10-depth inputs [ASSUMED]; G-L1b reads the drift |
| D-3 | Slots: concept (first-mention last token) or mention | **concept** | 2.5× cheaper attention [DERIVED]; a cleaner set-valued state |
| D-4 | Current-version attention among slots: causal (position order) or bidirectional | causal | closer to pretrained; cross-slot flow goes through the bank with a one-pass delay, which BFS tolerates |
| D-5 | Self labeller: pre-select L-RR, or choose L-ROLL / L-RR on GATE | choose on GATE by the pre-stated rule | nested showed the rollout is trajectory-bound and the re-rooted step is not; the ProsQA analogue is untested |
| D-6 | GSM8K K_i law and headline | K_i ~ U{m_bin..16}; **headline acc at fixed K′ 14** | non-oracle like CODI's fixed 6; makes K′ 14 in distribution |
| D-7 | GSM8K augmentation | R_ν on, R_x off | label-preserving; keeps CODI's answers and targets (one variable vs CODI-20) |
| D-8 | B2 (GSM8K self-teaching) | **defer**; trigger = B1 VB-C1-GSM | it needs its own base ($3–7); m ≥ 6 has n = 87 (SE about 0.05) |
| D-9 | B1 budget | matched 120k with the B0-5 interim look | §209 → §210: budget decides; the interim look caps a precompute failure at about $1 |
| D-10 | Optional cells | A2-CTRL **yes** ($0.55; §215's CTRL attributed 0.086 of the STAB_self loss to FT alone); A-TXT if a C3 statement on SYM is wanted | — |
| D-11 | Binary-answer bars | PROBE-GAIN-L ≥ 0.35; SERIAL-STEP-L follow ≥ 0.80; a band-level onset | chance 0.5 (nested's bars assume chance 1/11) |
| D-12 | REACH targets are simulator states (C4-illegal) | accept for A1 as "supervised", like nested R2 | the C4-legal PATH targets make the step non-local (§3.1) [DERIVED]; GSM8K's B1 targets are C4-legal |
| D-13 | B-GO condition | B1 launches when A1 has landed and is ¬dead ∧ ¬NO-ENTRY (the harness works on language); A1's C1 result does not gate B | different task; Theory may choose stricter |
| D-14 | Math backing before registration | **PROOFS_LANG + REDTEAM_LANG** by Opus subagents (math-backing rule) on §3.1 exchangeability, Lemma S-L + corollary, §4.3, §4.4 (reveal route), §4.6 (PERSON-SWAP), and the OC | required by the rule; not yet done |

### 10.2 Open questions (no answer in the record)
1. **Per-hop capacity on language (CAP-aff-L)** is unknown. Can 4 core layers do one REACH hop over multi-token names at T ≈ 450? The binding needs two chained lookups per hop (§2.3).
   - A0-4 (the 2k smoke) gives a descriptive preview.
   - The decisive answer is A1's ON(D).
2. **SLVA's CARRY collapse** (Prop 6 of D3; D3-T is RNN-IN-FUNCTION) is expected again. HISTORY-USED on language is therefore likely false, and the attention-memory claim needs it. Reported, not designed for.
3. **Positions up to about 760.** GPT-2's wpe covers 1024. D3-T's trained table was length-free to T 55 only, so long-position behaviour on SYM is new. NO-KNEE-type descriptive rows by T band are added to the acc part.
4. **SOUND-ONLY R2's missing OUT terms** (G7) could teach over-marking. FALSE-IN is descriptive. If it exceeds 0.05 after A2, the round-2 design must add a soundness-preserving OUT signal. Not designed now.
5. **GSM8K's step capacity.** One multi-digit binary op from linear digit slots in 4 layers is untested. F-P's per-pass values were only 0.338 at pass 1. STEP-NOT-LEARNED is the likeliest B1 failure after PASS1-PRECOMPUTE.
6. **Whether the GSM8K slot schedule ("slot t at pass t") is a clock.** STALL reads it. A content-keyed "write the first VOID slot" is the intended rule.

### 10.3 Recommended Stage A cell list (in order)
1. **A0** ($0, local; §8.1):
   - A0-1, §191 cue reliance: run first, with the result to Theory / Audit;
   - A0-2, SYM + LEVEL-PANEL: every band must pass before anything else is built on SYM;
   - A0-3…A0-14, the harness gates.
2. **A1 `lvb_sym_s1`.**
   - LANG-SLVA (D3-T recipe), concept slots, gold REACH targets on SYM D 3–8, per-draw relabelling.
   - CurGate 4 → 6 → 8 on plain answer CE only (thr 0.35, cap 6k), K_i = min(D + U{0..4}, 12).
   - 24k × bs 64, lr 1e-4 cosine, c_sched 1000,8000; RTX 3090, seed 1. Projected $1.2, guard $2.8.
   - Readings: VB-C1-L (D 5–8 for PROBE-GAIN-L), DEPTH-L, MEM.
3. **A2 `lvb_self_s1`** (on A0-16 + A0-15).
   - FT 12k from A1 at lr 3e-5, c 0, p_self 0.5.
   - Self pool D 9–12 from the GATE-selected labeller: SOUND verifier, SOUND-ONLY R2, fresh-once.
   - Projected $0.75, guard $1.7.
   - Readings: SD-SELF-L / SD-C2-L / LIT-C2-L / CAUSAL-DEEP-L / MEM-TRADED-L.
4. **A2-CTRL `lvb_ctrl_s1`** (recommended optional). The same stream with p_self 0. Projected $0.55.
5. *(optional)* **A-TXT `lvb_txt_s1`.** TEXT-COT on the same SYM stream (the §191 B recipe), for C3 on SYM. Projected $1.0.

Then Stage B, inside the same design and cap: B0 → **B1 `lvb_gsm_s1`** (projected $3.3–6.6, guard $9, interim look at 24k).

## 11. Verification log (this draft; all $0, CPU, OMP_NUM_THREADS=1, python -B, repo read-only)
| script (research7/scripts/) | output | checks |
|---|---|---|
| heur_panel.py | heur_prosqa_test.json, heur_p8na_test.json, heur_p8na_train.json | reproduces §59's table (prosqa_test 0.911 / 0.662 / 0.662 / 0.590 / 0.554 / 0.515; p8_na option level 0.50); finds the anchor-level cue (0.896–0.945 at depth ≥ 6) |
| nn_regen.py | nn_p8na_test.json, nn_nn_test.json | the p8_na order reproduces the anchor cue; double neutralisation leaves anchor parents' in-degree at 0.606–0.668 |
| anticue.py | stdout | the p8_na_test anchor-proximity strata for A0-1 (m ≥ 2: pro 2,630 / tie 511 / anti 78) |
| sym_proto.py | sym_proto.json | the SYM prototype: invariants, LEVEL-PANEL, BIDIR, size law |
| gsm_stats.py, gsm_mbin.py | stdout | GSM8K m / m_bin distributions, DECOMP coverage, values, question lengths |
| cost_model.py | cost_model.json | work ratios and attention memory vs D3-T (the speed law is ASSUMED) |

Not verified here, and needed before registration:
- the GPT-2 token counts of SYM at D 16 (tokenizer at G-L0);
- the PROOFS / REDTEAM pass (D-14);
- the arXiv IDs quoted from the LIT notes, which were not re-checked by this subagent;
- the §57 "Coconut 0.974 → 0.738" primary line, which the ProsQA summary subagent could not locate. It is not used by this design.
