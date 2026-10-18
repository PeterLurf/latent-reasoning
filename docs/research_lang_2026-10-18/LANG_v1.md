# LANG-VB v1: VERSION-BANK / SLVA + self-teaching on language (ProsQA-SYM v1 serial reachability, then GSM8K): registration-ready text

Clock 2026-10-18 13:25 EDT (`date`). Opus design-revision subagent for Theory. $0.

**Status: registration-ready text, NOT registered.** Audit reviews it. It is one design with one family cap of **$25**, under the user's
standing $30 rule (Audit may approve a single design with cap ≤ $30). Theory will tell the user the shaded odds (§0, §6) before registration.
The user said "yes" to starting the design (12:02 EDT 10-18, via Audit); that is not approval of spend.

Inputs read: docs/research_lang_2026-10-18/LANG_draft.md (v0, 12:38) + scripts/; PROOFS_LANG.md (13:10) + proofs_lang/;
research7/REDTEAM_LANG.md (13:15) + rt_lang/; THEORY.md notes 191-2, 191-3, 214-5, 215-7, the §216 s1 round-2 result, 216-8, 216-9, §217
(+ 217-1, 217-2); the frozen SD2_v1.md (structural model); boxes/vbank.py (CurGate l.545–575, draw_step l.744–770, the CurGate exclusion
l.812, rollout l.262–345), boxes/slva.py (gather) and boxes/prosqa168s2.py (ENT17 / CON38), read only; THEORY notes 203-6 / 203-8 and
§207's INPUT-SWAP. Theory's decisions L1–L7 (relayed 10-18), Audit's slot-local addition (A-1) and Audit's CONSUMPTION-FRONT addition (A-2,
with Literature's ID verification, addendum 85f6a055), all relayed during this revision, are applied and listed in §1.
$0 checks: research7/v1_checks/ (JSON beside each script; CPU ≤ 2 threads, idle priority, `python -B`, no file written under D:/Latent).
Tags: [PROVED], [PROVED UNDER X], [DERIVED script] (exact arithmetic on measured inputs), [MEASURED file], [ASSUMED], [CONJECTURE].

(Written incrementally, section by section; all sections complete at 13:58 EDT 10-18; Audit's A-2 (CONSUMPTION-FRONT) integrated 14:02–14:11.)

## 0. One screen

| item | LANG-VB v1 (proposed §218) |
|---|---|
| **Question** | Does VERSION-BANK / SLVA (D3-T's recipe, ported to language through concept / quantity slots) learn **serial** latent computation that passes causal C1 tests on natural-language reachability (Stage A, ProsQA-SYM v1), extend its depth by self-teaching (A2), and, only if Stage A reads VB-C1-L, on GSM8K against CODI-20 (Stage B)? |
| **Plainly** | **The expected value is mainly mechanism information** (serial vs precompute-and-carry vs recompute vs one-pass, read by certified detectors). **A null is the likeliest outcome** (P ≈ 0.94). A positive is strong evidence (false share ≈ 0.01: CONSUMPTION-FRONT is decode-free) |
| **Data (Stage A)** | **ProsQA-SYM v1**: two i.i.d. worlds; cross edges only between the persons' direct subtrees (leaf targets); 4 ancestral decoys per world (5 at D ≤ 4); one-token persons (PERSON-40); NONCE-300 two-token names with distinct final tokens; leading-space rendering; T ≤ 1,000. Person-agnostic accuracy ½ exactly [PROVED]; uBIDIR ½ and UDIST 1 exactly, REACH saturates at D [PROVED]; gated single-parent chases **0.540–0.555** at every D 3–16, person-agnostic panel ≤ 0.526, 0 / 75,000 invariant failures [MEASURED, n 5,000 per D]; T mean 433–859, \|S\| 45–97 |
| **Cells / stages** | A0 ($0: A0-1 §191 cue contribution, G-L0 SYM gate, harness gates) → **DIRECT-SYM ∥ A1** (24k each, RTX 3090) → [A2 GO: DEEP ∧ NO-CLOCK ∧ GATE-1-L] labelling (local bf16) → **A2 ∥ A2-CTRL** (12k FT; self pool 106k kept per D 9–12; seed 1 consumes exactly 383,488) → [**B GO = VB-C1-L**] B0 → **B1** (120k, probe-based interim GO at 24k). A-TXT optional; B2 deferred |
| **Primary readings** | **VB-C1-L** = DEEP (≥ 0.90, D 4–8) ∧ PROBE-GAIN ≥ 0.35 over the C1 band (DIRECT-SYM's measured boundary; PLANT-certified) ∧ ON(D) slope ≥ 0.6 ∧ **SERIAL-STEP: PREMISE-SWAP × (ii′) bank splice ≥ 0.80 and slot-local (i′) ≥ 0.80** (exposed 1,000 items per D; (i′) at D 5–8, D 4 excluded and listed) ∧ **CONSUMPTION-FRONT: input switched to a one-fact donor from pass t; crossover t\* ∈ {j − 1, j} on ≥ 0.75 of clean pairs ∧ within-D Spearman(t\*, j) ≥ 0.60** (j = 3..D−1; 39–60 % of items exposed per j; last hop descriptive) ∧ STALL ≥ 0.9 ∧ **NO-PRECOMPUTE: FD-MIRROR AUROC ≤ 0.65 at k ≥ 3** (null ½ exactly). **SD-C2-L** = SD-SELF ∧ D 13–16 ≥ 0.80 ∧ MARGIN-GROWS (+4-translated A1 + 0.10) ∧ BEATS-CTRL (+0.10 over the required A2-CTRL). **VB-C1-GSM** = PROBE-GAIN ≥ 0.10 (PROBE-VALID-GSM) ∧ (ii′) and slot-t (i) follow ≥ 0.6 / own ≤ 0.3 (95 exposed items × 21 R_x recipients) ∧ CF-GSM (37 exposed steps with j ≥ 3 × 41 recipients) ∧ STALL ∧ NOT-ONE-PASS-CHAIN (linear and MLP, FD(k, k) ≥ 0.3). Accuracy vs CODI-20 paired; BEATS = Δ ≥ 0.03 ∧ McNemar p ≤ 0.05 |
| **OC headline** (RT's shaded "with fixes", recomputed exactly, then L1–L4 and A-2) | P(VB-C1-L) **0.063** (0.075 without CONSUMPTION-FRONT; RT-exact 0.086); P(· \| no serial use) **0.004** (0.036 without CF); P(· \| serial use) 0.40. P(SD-C2-L) **0.005** (\| H_shift 0.02). P(B1 launched) 0.063; P(VB-C1-GSM \| launched) **≤ 0.10**, \| launched ∧ no serial use **≤ 0.008** (0.027 if B1 ran ungated). BEATS-CODI20 \| equal accuracy **0.016 item-level, 0.12 with seed variance** (a single-seed comparison speaks about two models on 1,319 items, not about recipes). **P(≥ 1 decisive C1) 0.064** (0.076 without CF; RT-exact 0.087; RT's 0.095 is the expected count) |
| **Cost** | All required cells run: projected **$10.4–17.1** (central $13.3); expected realised ≈ **$7** (Stage A base DIRECT-SYM + A1 ≈ $5.6). **One design, family cap $25**; guards sum $24.2; cut order **B1 → A-TXT → host labelling**; required A cells never cut, never shortened |
| **Approval** | Audit's review; under the user's $30 rule (cap ≤ $25); Theory tells the user the shaded odds before registration. **MEM-TRADED-L in any cell → every later stage needs the user's explicit OK** (SELFDEPTH_v1 §9) |
| **Open (§9)** | DECOMP's C4 status (Audit: Stage B is not C4-legal as a whole as written); the ALL-ANCHORS power hole (proved; RESET-t descriptive); the cost-aware labeller tie rule; PLANT strength; eval cards; DIRECT-SYM in parallel with A1; CF's derived window / thresholds vs Audit's proposed ±1 / 0.8 / 0.8 |

## 1. Change table (every review finding, with source)

The draft's design stands unless a row below changes it: LANG-SLVA (D3-T's recipe over a slot subset), V_0 = all T prelude rows in the
bank, the question-free coda with a bank-free decoder path (D-2), concept slots (D-3), causal current-version attention (D-4), REACH R2
targets (D-12), the SOUND verifier and SOUND-ONLY R2, DECOMP binary-op programs on GSM8K, B2 deferred (D-8), matched 120k for B1 (D-9).

### 1.1 Theory's decisions (L1–L7) and Audit's addition (A-1)

| # | decision | applied in |
|---|---|---|
| L1 (B1) | **One-pass precompute-and-carry (RT F1-1).** (a) PLANT positive controls for every PROBE(t ≤ 2): **PROBE-VALID** = the probe recovers an independent planted code of known size (linear and XOR) from V_t, else PROBE-INVALID (censor); (b) **FUTURE-DECODE-L** (per-slot, linear + 1-hidden-layer MLP, on σ-mirrored slot classes so that its null is exactly ½) made **decisive** (NO-PRECOMPUTE-L); (c) the **slot-local splice (i′)** (the analogue of nested VALUE-SWAP arm (i)) a **co-primary** arm. On nested arithmetic arm (i) is not open to this hole because it already reads follow_i 0.9959 / 0.9993 / 0.9995 (D3-T / FT-SELF / R2-s1, class A, 4,184 pairs; class B 0.9990 / 0.9990 / 0.9981, 4,837 pairs) [MEASURED eval_vb.json of s213_d3t_s1, s215_ftself_s1, s216_r2_s1]: a nested model that carried a hidden answer outside the reduced slot would read own on (i). (d) GSM8K: **PROBE-VALID-GSM** (PROBE(14) ≥ acc_fixed14 − 0.03, plus PLANT on slot-t digits), an **MLP FD probe** beside the linear one, and **FD(k, k)** in the field contract (FD(k,k) < 0.3 → UNREADABLE, never a vacuous pass) | §2.5, §3.5, §4.1, §4.2 |
| L1 (B2) | **§4.6(c)'s false [PROVED] PERSON-SWAP argument (RT F1-2, PROOFS §4) replaced.** Decisive arm = **PREMISE-SWAP × (ii′)**: the donor is the recipient with the two persons' names exchanged in the premises (question unchanged), and the donor's V_1..V_t are spliced (bank and SLVA source). The **bank-swap (history-read) arm** is the (ii′) splice itself, reported beside arm (ii) (V_t only) so that "(ii) own ∧ (ii′) follow" reads HISTORY-READ. Re-proved in §3.6 with precise tags: follow is [PROVED UNDER S-NAME] (serial state indexed by person names, incl. ALL-PERSONS); own is [PROVED] for recompute-from-V_0; **a new counter-model ALL-ANCHORS (serial, anchor-tagged) reads own** [PROVED], so "own" is never worded "recompute" alone (§1.4 N1) | §3.6, §4.1, §5.2 |
| L1 (B3) | **A2 pool (RT F3-1).** Coin per step, a self step = 64 records (vbank.py l.755–766): seed 1's demand is exactly **5,992 self steps = 383,488 records** [DERIVED v1_pool.py, a replay of rng_s asserted equal to §216's measured cursor]. Pool **106,000 kept per D** (424,000; spare 10,128 per D at seed 1), **draw cap 530,000 per D**, and an exhaustion assert before launch (G-L20) | §2.4, §7, v1_pool.json |
| L2 | **RT MAJORs accepted.** SYM decoys prototyped and **gated** (single-parent chases < 0.6 at every D; §2.2); the undirected-search check in the panel (uBIDIR = ½ exactly); cross edges only between the persons' extra direct-statement subtrees (fixes the BIDIR-U leak and SOUND's hold-after-τ hole); distinct final name tokens; single-token persons; **DIRECT-SYM required**; L-RR distribution fix (chunked re-roots ≤ 4) + **MAXIMAL SOUND SUB-CHAIN** (SOUND-CLOSURE) with selection-bias reporting; a **probe-based interim GO** for B1 (never the trained coda at K′ 1); C4 qualifier on DECOMP (**Stage B is not C4-legal as a whole**; sent to Audit, §9); **SD-C2-L requires MARGIN-GROWS-L and beats a REQUIRED A2-CTRL**; CurGate / NO-ENTRY on a **per-item two-candidate answer CE of plain reference items only** (214-5); cost re-estimated at the measured tokens (2-token NONCE names: T 633 mean at D 8), the T²K memory model with **micro-batching**, and the labelling GPU-hours planned (§8) | §2, §4, §7, §8 |
| L3 | Every PROOFS_LANG CORRECTED / FAIL item applied (§1.3): SYM's σ-conditions (independent per-world thinning, role-symmetric contamination signature, 1-token names), directed BIDIR dropped (uninformative), Lemma S-L's scope (SOUND never over-reaches, accepts incomplete chains), FD(k, k), L1-L's SLVA sublayer, the GSM8K C4 qualifier, P(SD-C2-L) ≈ 0.02 by the draft's chain, P(≥ 1) 0.20–0.26 pre-shading, cost $5.4–10.6 baseline (superseded by §8) | §1.3 |
| L4 | D-1…D-14 per RT's one-liners: D-1 SYM + decoys + distinct tokens + single-token persons + cap from D 9; **D-1b A0-1 registered here** as a $0 descriptive arm (pro / tie / anti, worded per 191-3: it measures the anchor cue's CONTRIBUTION and cannot explain §191); D-3 concept slots on identifying tokens; D-5 L-RR fix + sound sub-chain; D-6 every K′ < m_bin reading dropped from decisions; D-7 "one variable" struck; D-8 B2 deferred; D-9 120k + the probe-based GO; D-10 **A2-CTRL and DIRECT-SYM both required**, A-TXT optional; D-11 bars kept on the new swap (0.80 on PREMISE-SWAP × (ii′) and on (i′); band onset; probe grid + t 5, 7); D-12 DECOMP's C4 status to Audit; **D-13 launch B1 only if VB-C1-L reads**; D-14 §3.6 re-proof + scorer fixtures for the counter-models (precompute-and-carry; person-tracking serial; plus ALL-ANCHORS) | §2, §4, §5, §7 |
| L5 | OC headline = RT's shaded "with fixes" column, **recomputed exactly** (v1_oc.py asserts RT's oc_shade.json), then L1–L4's effects applied with every added factor stated (§6). §0 says plainly that the expected value is mainly mechanism information and that a null is the likeliest outcome | §0, §6 |
| L6 | MEM-TRADED-L in any cell → every later stage (A2 after A1, B1, any further self round) needs **the user's explicit OK** (SELFDEPTH_v1 §9). Off-host checkpoints, GUARD_STOP, one GPU class (RTX 3090) for every training cell, the eval card class fixed per stage, smoke at full data scale, per-item stores, the 215-7 pending rule in score_s218 | §5.2, §7, §8 |
| L7 | Family cap **$25** with per-cell admission bounds and guards that sum ≤ $25; cut order if the re-estimate exceeds $25 (B before A optionals; never split the design) | §8 |
| **A-1** (Audit, relayed 10-18) | The single-slot / handle-local splice is a **co-primary from the start**, with its **exposed-item count** stated per depth (items whose handle has a valid single-slot donor) plus clean-pair counts; a reading with too few exposed items is **excluded and listed, never passed** (controls-must-be-exposed rule). Nested values cited above (L1 B1) | §4.1 (exposure table), §4.2, §5.2 |
| **A-2** (Audit's research pass, relayed by Theory; IDs verified by Literature, addendum 85f6a055) | **CONSUMPTION-FRONT** as a co-primary for blocker 1, with a derivation and an OC: one-fact donor at hop j, input switched from pass t, crossover vs j. Precondition checked (every pass re-reads the prelude output: yes, §3.6 PS-6). Pass rule **derived and calibrated**: window {j − 1, j} (the serial set), j 3..D−1, share ≥ 0.75, within-D Spearman ≥ 0.60 (Audit's proposed ±1 / 0.8 / 0.8 calibrated beside it and kept as a reported row). Exposure per (D, j) stated (39–60 %). Prior arms 203-6 / §207 INPUT-SWAP cited. Framed as an **extension of Liang & Pan 2602.00449** (single-input-token corruption × latent-position patching in CODI; 2602.08783, ICML 2026, does state do-interventions only). Literature's last-hop late-fusion prior pre-stated (j = D descriptive; the rule runs over j < last). Stage B: **CF-GSM**, the late-fusion detector | §2.5, §2.7, §3.6 PS-6, §4.1, §4.2, §6 (e) |

### 1.2 Red team (REDTEAM_LANG.md)

| # | finding | v1 | verdict |
|---|---|---|---|
| F1-1 (B) | precompute-and-carry passes VB-C1 | L1 (B1): PLANT, FD-L decisive, (i′) co-primary, PROBE-VALID-GSM, FD-MLP, FD(k, k); probes trained 10k steps on ≥ 200k fresh items with the curve stored (RT F1-1e) | accepted |
| F1-2 (B) | §4.6(c) false; ALL-PERSONS reads own | L1 (B2): PREMISE-SWAP × (ii′); PERSON-SWAP kept as a descriptive mechanism probe ("PERSON own ∧ PREMISE follow" = the ALL-PERSONS signature) | accepted, extended (ALL-ANCHORS, §3.6) |
| F3-1 (B) | pool 240k vs demand 384k | L1 (B3) | accepted (exact replay) |
| F1-3 | CAP-aff-L unsupported on fresh data | DIRECT-SYM (required); C1 bands from D*_fresh (§2.6); CAP-aff-L kept [ASSUMED] with PROOFS' caveat (its cited evidence measures no one-pass capacity) | accepted |
| F1-4 | PROBE(0) = ½ by construction | stated and proved (§3.5); PROBE-GAIN-L reads PROBE(1); PROBE(2) enters ONESHOT-L | accepted |
| F1-5 | trained-coda K′ 1 readings in two detectors | PASS1-PRECOMPUTE-GSM's acc_k14 − acc_k1 clause dropped; B0-5 GO on a fresh PROBE(1) + FD-MLP from the mirrored 24k checkpoint (§2.7) | accepted |
| F1-6 | keep causal current-version attention | kept (D-4); slot-diagonal noted as a stricter variant, not used (§9) | accepted |
| F2-1 | backward pointer chase (H-ROOT 1.000) | decoys **re-tuned**: RT's law reads 0.61–0.66 (and 0.58–0.69 across six laws in my re-runs, because the draft's within-world extra edges give R-leading alternative parents); v1 law = 4 decoys per world (5 at D ≤ 4) on the top path levels, round-robin, chain 1–2 nodes, the last chain node with the path nodes' child law, extra-edge probability 0.3 → 0.1 → **the gated chase set reads at most 0.540–0.555 at every D 3–16** (n 5,000 per D) [MEASURED v1_sym.json]; the gated set adds LA(1) (a chase that avoids non-asked persons) | accepted, re-tuned |
| F2-2 | one-pass boundary unmeasured | DIRECT-SYM, D*_fresh rule pre-stated, empty band = not met (§2.6) | accepted |
| F2-3 | "chance" for search readings | with the direct-subtree kernel UDIST = 1.000 and uBIDIR = 0.500 at every D [MEASURED]: any full search (directed or not) is exact, any person-agnostic rule is ½; wording uses both (§5.1) | accepted (re-measured) |
| F2-4 | every CON38 slot on "us" | NONCE-300 v1: 300 two-token names with pairwise-distinct final tokens, disjoint from every non-final token, person and template token [MEASURED v1_names.json] | accepted |
| F2-5 | 2-token person names | PERSON-40, all single-token; plus the leading-space rule (§1.4 N3) | accepted, extended |
| F2-6 / F3-3 | L-RR re-roots OOD (layers > 4) | chunked re-roots (≤ 4 "R is a f." per rollout; ≈ D + 0.5 rollouts per chain at D 9–12 [MEASURED]); the frontier size per step stored in GATE-1-L | accepted (chunking, not r ~ U{1..8}: it keeps A1's input law and T) |
| F3-2 | reject-on-any-false-mark | SOUND-CLOSURE (maximal sound sub-chain); kept-vs-drawn reporting (T, concepts, cross edges, decoys, frontier width by D) and FALSE-IN on GATE (§2.4) | accepted |
| F3-4 | the labeller's reach is thin; P(A2 runs) ≈ 0.15 | stated in the OC (§6) as the modal A2 outcome (no A2 = a pre-stated C2-negative) | accepted |
| F3-5 / F6-4 | 15–37 GPU-h labelling unplanned | re-derived at v1's T and chunked L-RR: 424k kept needs L-RR 23–30 local fp32 GPU-h (13–17 bf16) at yield 0.5 and 58–75 fp32 (33–43 bf16) at yield 0.2; L-ROLL 4.6–7.8× cheaper (2.5–6.2 bf16 GPU-h) [DERIVED v1_pool.json]; plan and decision rule in §8 (two local cards in bf16, else a 3090 host in bf16 under its own admission) | accepted, re-estimated |
| F3-6 | store both chains' KEEP decisions | in GATE-1-L's per-item store (§4.1) | accepted |
| F4-1 | DECOMP C4 status | Theory's L2 qualifier; Audit rules before registration; strict variant (mask calculator-only sub-step values) pre-stated as Audit's option (§9) | accepted |
| F4-2 | interim GO does not catch §208 | probe-based GO (§2.7) | accepted |
| F4-3 | GSM8K detectors | L1 (B1 d); slot-t-only arm co-primary; class-A exposure measured: **95 base test items** (94 at m ≥ 3) have an exposed class-A step [DERIVED v1_gsm_exposure.json], so recipients use 20 R_x variants per exposed item (§4.2) | accepted, extended |
| F4-4 | CODI-20 comparator | "matched examples", a system comparison; BEATS needs Δ ≥ 0.03 ∧ a paired exact McNemar p ≤ 0.05 on the 1,319 items; wording "one seed each" with what it can / cannot claim (§5.1, §6) | accepted |
| F4-5 | gate B1 on VB-C1-L | D-13 per L4 | accepted |
| F4-6 | B1-24k vs CODI4; m_bin > 12; K′ 14 | B1-24k descriptive only; mask steps k > 12 (171 train items) and pointers to literals > 14 (18 items); assert K′ 14 ≥ m_bin on test | accepted |
| F5-1 | SD-C2-L under H_shift | SD-C2-L = SD-SELF-L ∧ DEPTH-BEYOND-SELF-L ∧ MARGIN-GROWS-L ∧ BEATS-CTRL-L; A2-CTRL required; ABSORB-L's no-effect row likewise read against A2-CTRL (descriptive) | accepted |
| F5-2 | per-token CE passes at chance | the gate CE is the two-candidate CE (−log softmax over the two candidates' summed log-probs, chance ln 2), per item, plain gold items only; an init fixture asserts 0.693 ± 0.05; logged `plain_cand_ce` | accepted |
| F5-3 | LIT-C2-L ≈ 0 under success | LIT-C2-L reported as UNINFORMATIVE on SYM (D ≤ 16 fits K′ 16); never a C2 negative | accepted |
| F5-4 | probe grid | t ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12} | accepted |
| F6-1 | T is 1.2–1.5× | v1 names are 2 tokens: T mean 434 (D 3) / 633 (D 8) / 722 (D 12) / 860 (D 16) with the leading space, max ≤ 1,000 by rejection [MEASURED v1_sym.json]; D 13–16 stays in the untouched set | accepted (re-measured) |
| F6-2 | bs 64 does not fit | per-batch-type micro-batching pinned by G-L8 (projected micro 4 for A1 / A2 / A2-CTRL at the padded batch shapes, 2 for DIRECT-SYM, 1 for B1; §3.9) | accepted |
| F6-3 | admissions too tight | admissions re-set from the v1 projection (§8) | accepted |
| §5.3 (a)–(f) | censoring holes | (a) L1 B3; (b) PLANT; (c) not admitted = UNREAD, never dead / pending; (d) empty C1 band = not met; (e) A0-1 adds acc(anti) and the NN-rendered contrast; (f) GSM8K pair projection from B0-3 + 20 R_x variants | accepted |
| §5.4 | wording risks | "person-agnostic cues excluded by construction; single-parent chases gated", never "no shortcut"; "serial latent reachability", never "graph search"; RECOMPUTE-L never on (ii) or PERSON-SWAP | accepted |

### 1.3 Proofs (PROOFS_LANG.md; PASS rows unchanged and not listed)

| # | finding | v1 |
|---|---|---|
| 1 | p8_na also fails at depth 3 (option-level parents' in-degree 0.63–0.65); "0.86–0.93" are regeneration values; NN residual ≈ 0.63 (3-step bases) / ≈ 0.58 (4-step) | stated in §2.2's rationale; D-1 unaffected |
| 1c | §191 A beats the person-agnostic cue ceiling at every depth 3–10 | A0-1 worded per 191-3 (§2.8) |
| 2a | σ-conditions: per-world thinning; role-symmetric contamination signature; 1-token names | spec items S-σ1…S-σ3 (§2.2) |
| 2b (FAIL) | directed BIDIR ≡ ½ / 1; undirected leak 0.60–0.63 | directed BIDIR dropped; uBIDIR(r), 2r < D, in the gated panel (= ½ exactly under the direct-subtree kernel) |
| 2c | margin "≥ 0.05" → "≥ 0.048"; tokens 1.8 per word; \|S\| ≤ 4D + 20 | re-measured on v1 (§2.2); with decoys \|S\| ≤ 4D + 52 at D ≥ 9 (§3.3f), measured max 110 |
| 3 (FAIL) | zero-noise corollary vs HOLD-after-τ | fix (b): direct-subtree kernel with leaf targets ⇒ REACH saturates at D at **every** D 3–16 (beyond-D share 0.000 [MEASURED]); HOLD exact; corollary [PROVED] (§3.3) |
| 3 | SOUND parses the original item, never the re-render | spec (§2.4) |
| 4 | L1-L: + t SLVA attention sublayers | §3.1 |
| 4 | CAP-aff-L evidence measures no one-pass capacity | §3.2; DIRECT-SYM |
| 4 | detectors one-sided; undetectable combination | §3.5 states it; CONSUMPTION-FRONT (A-2, decode-free) now fails it; without CF the no-effect rows would be conditional on PLANT-certified decodability (§6) |
| 4 | FD(k, k) missing; vacuous pass | §4.2 |
| 4 (FAIL) | §4.6(c) history read | (ii′) + §3.6 |
| 5 | mean m on the parsed subset 3.30; CODI "m ≤ 2" = m 1–2; C4 wording; m_bin > 12; literals > 14; CONST 12.9 % | §2.7, §5.1 |
| 6 | P(SD-C2-L) ≈ 0.02; P(≥ 1) 0.20–0.26; A1 $1.30 at measured T; B1 ≥ $4.1; total $5.4–10.6 | superseded by §6 / §8 (the RT-shaded, v1-recomputed values) |

### 1.4 New in v1 (not in any review)

| # | item | why |
|---|---|---|
| N1 | **ALL-ANCHORS counter-model**: a serial one-hop-per-pass state tagged by anchor concepts (the objects of person statements), not by person names, reads **own** under PREMISE-SWAP × (ii′) and under (i′) [PROVED, §3.6]. Added: a descriptive **RESET-t** arm (V_1..V_t replaced by the version-0 slot rows) that separates it from recompute-from-V_0 (drop vs no drop); RECOMPUTE-L := own > 0.3 ∧ RESET recovery ≥ 0.9. CONSUMPTION-FRONT (A-2) reads ALL-ANCHORS as serial (tags propagate through the current input's edges, §3.6 PS-6), so the hole is a power loss of SERIAL-STEP-L only | the re-proof L1 (B2) asked for; a false negative of SERIAL-STEP-L, never a false positive |
| N2 | **FD-MIRROR**: FUTURE-DECODE-L compares slots at distance t + k from R with slots at distance t + k from P, so any person-agnostic feature reads AUROC ½ exactly (σ) [PROVED] | RT's "AUROC ≤ 0.65 among slots with dist > t + 1" has an unknown null (person-agnostic depth features separate both worlds' depth-(t + k) nodes from the rest) |
| N3 | **Leading-space rendering**: the prelude input is [EOT] + " " + text. Without it PREMISE-SWAP shifts positions in 2.3–3.0 % of items (11 of the 40 person names are 2 tokens without a leading space) [MEASURED v1_sym exposure, 300-item re-check: 9 / 300 → 0 / 300] | slot-position identity is the (i′) / (ii′) exposure condition |
| N4 | Decoy law re-tuned, LA(1) gated, LA(2..4) descriptive (a bounded-lookahead backward chase is exact at lookahead ≥ chain length + 1: wording "serial latent reachability") | F2-1 as drafted did not reach the gate |
| N5 | PERSON-40 (decoy persons need up to 14 names per item) | ENT17 has 15 one-token names |
| N6 | GSM8K class-A exposure is 95 base test items → 20 R_x variants per exposed item; SERIAL-STEP-GSM pair projection from B0-3 | otherwise UNREADABLE by construction at m ≥ 3 accuracies ≈ 0.27 |
| N7 | DIRECT-SYM costs about as much as A1 (12 blocks over the padded T vs A1's prelude + core), so $2.1–3.2 (central $2.6), not RT's $0.4–0.8 [DERIVED v1_mem.json] | §8 |
| N8 | With B1 gated on VB-C1-L, P(≥ 1 decisive C1) = P(VB-C1-L) + P(SD-C2-L ∧ ¬VB-C1-L on A1) ≈ P(VB-C1-L); RT's 0.095 is the expected **count** of decisive positives | §6 |
| N9 | Cost-aware labeller tie rule: L-RR is chosen over L-ROLL only if its kept yield at D 11–12 exceeds L-ROLL's by ≥ 0.05 (L-RR costs 4.6–7.8× L-ROLL per drawn item) | §2.4; Theory to confirm (§9) |
| N10 | Eval card per stage: every Stage A reading on the RTX 4060 (fp32, one card for A1 / A2 / A2-CTRL / DIRECT-SYM), and Stage B on the RTX 4060 too, with CODI-20's B0-2 re-score on that card; pre-stated whole-stage fallback to the RTX 3070 (§4) | the same-card rule; the 3070 is hot and was lost 10-17 |

## 2. Design (registration text)

### 2.1 Architecture: LANG-SLVA (D3-T's recipe over a slot subset; draft §2.1 unchanged except where marked v1)
- **Prelude.** x = [EOT] + " " + text (**v1: leading space**, N3) [+ 12 quantity placeholders " _" for GSM8K]. Blocks 0–5 over all T
  tokens → H_6. V_0 := H_6 at the slot positions; the bank's version-0 entries are **all T rows** of H_6 (β⁰ constant, no age term).
- **Core, pass t = 1..K** (blocks 6–9, shared): queries = the current version's |S| slot rows; keys / values = [version 0: all T rows] ∪
  [versions 1..t−1: slot rows] ∪ [current slots, causal in position order (D-4)], from each block's own ln_1 + c_attn (`vbank.bank_kv`),
  bias β + γ·(t−1−s), calibrated per head at step 0 (99th-percentile bank mass ≤ 0.02).
- **Pass input (SLVA, unchanged):** U_t[j] = Σ_{s<t} α_{j,s} N(V_s)[j], α from the SLVA query on N(V_{t−1})[j]; c(step) = `vbank.c_at(step, 1000, 8000)`.
- **Write:** V_t = core output at the slots, appended, never overwritten; no pass token; slot position ids fixed across passes.
- **Coda after pass t (question-free):** blocks 10–11 over [V_t slots | decoder tokens] with bank versions 1..t−1 (V_0 excluded); the
  decoder string runs blocks 0–9 alone (no question keys, no bank), positions T+1.. (D-2; assert T + L_dec ≤ 1,024).
- **Recode heads:** ProsQA R_P (LN + Linear, 768 → {OUT, IN}) per concept slot; GSM8K R_G (op 6, ptrA / ptrB 27, sign 2, 9 digit places × 11).
- **Slots.** ProsQA: one per distinct concept, at the **last BPE token of its first mention** (v1: NONCE-300 names are 2 tokens with
  pairwise-distinct final tokens, so the slot token identifies the concept, RT F2-4). Person names are not slots (their rows are in V_0).
  GSM8K: 12 quantity slots (draft §2.3; steps k > 12 masked, pointer terms to literals > 14 masked).
- **Losses (READY-RECODE):** L = R1 + 1.0·R2. R1 = the decoder string's token CE over the readiness window t ∈ [τ_i, K_i]. R2 per
  (item, t): ½ mean CE over changed slots + ½ over unchanged slots, IN / value classes ×2, HOLD after τ_i. No MCT, no PAUSE.
- **v1 gate CE (RT F5-2).** `cand_ce(item, t)` = −log softmax([Σ log p("### o_R."), Σ log p("### o_P.")])[o_R] from the coda after
  pass t (chance ln 2). `plain_cand_ce` = the mean over **plain gold items** of the per-item mean over its readiness window. CurGate and
  NO-ENTRY read **only** `plain_cand_ce` (214-5); self / augmented batch types never call `GATE.update` (asserted in the trainer, G-L7).
- **Micro-batching (v1, RT F6-2):** `--micro` (plain) and `--self_micro` (self), each the smallest of {1, 2, 4, 8} with peak ≤ 21.0 GiB
  at the forced worst shape (G-L8); projected micro 4 for A1 / A2 / A2-CTRL, 2 for DIRECT-SYM, 1 for B1 [DERIVED v1_mem.json].
- Not changed: GPT-2 small; `vbank.bank_kv` / `block_bank`; SLVA parameters and init; lr 1e-4 cosine, wd 0.01, bs 64, kextra 4, rec_w 1.0,
  trained position table; fp32 evals.

### 2.2 Stage A data: ProsQA-SYM v1 (the spec for boxes/prosqa_sym.py)
**Generator SYM(D, rng), D ∈ 3..16.**
1. **Two worlds W_R, W_P, independently from the same law G(D):** path a_1 → … → a_D (a_D = the world's option); each a_k (k < D) gets
   Binomial(2, ½) distractor children with subtrees of depth ≤ min(3, D − k) (branching Binomial(2, 0.3)); within-world extra edges
   u → a_k with depth(u) + 1 ≥ k, **probability 0.1 per distractor node** (draft 0.3; v1 N4); r ~ U{1..4} direct statements of the
   person: a_1 plus r − 1 fresh direct nodes with 0–2 children each (the **direct subtrees**).
2. **Per-world cap (D ≥ 9):** ≤ 2D + 6 non-decoy concepts, by removing whole distractor subtrees uniformly at random, **in each world
   independently by the same rule** (S-σ1, PROOFS 2a(i)).
3. **Ancestral decoys (RT F2-1, v1 law):** per world, 4 decoys (5 at D ≤ 4), attached round-robin to path levels a_D, a_{D−1}, …, a_2 (then
   again from a_D); each decoy = a fresh decoy person Y → b_1 (→ b_2), chain length U{1..2}, edge b_last → a_k, and b_last gets Binomial(2, ½)
   leaf children (the path nodes' child law). Decoys never change any distance from R or P.
4. **Cross edges (PROOFS 2b / §3 fix (b)):** c ~ U{0..2} in each direction, **source** = a node of the source person's direct subtrees,
   **target** = a leaf of the other person's direct subtrees. Nothing else joins the worlds.
5. **Extra persons:** 0–2, each with a disjoint 2–4-node tree reaching no option.
6. **Names:** persons from **PERSON-40** (40 single-token names, ENT17's 15 one-token names first; S-σ3), concepts from **NONCE-300**
   (300 two-token pseudo-words, distinct final tokens), drawn without replacement per item (per-draw relabelling) [v1_names.json; sha
   recorded in the module].
7. **Text:** "<Person> is a <c>." / "Every <a> is a <b>." for every edge, uniformly shuffled; question "Is <R> a <o1> or <o2>?" with a
   fair option coin; answer string "### <o_R>."; input [EOT] + " " + text; **T ≤ 1,000 by item rejection** (σ-invariant: all person
   names are one token).
8. **Invariants asserted per item:** dist(R, a_k) = dist(P, a_k^P) = k for every k; o_P ∉ Reach(R); o_R ∉ Reach(P); every concept
   mentioned; T ≤ 1,000; every person name one token; **max dist(R, c) ≤ D** (REACH saturates at D).
9. **Contamination signature (S-σ2):** the sha of the name-free canonical form with the two worlds' canonical strings **sorted**
   (role-symmetric), so the redraw event is σ-invariant.
10. **Seeds:** training stream 22100, test 22200 (1,000 per D, D 3..16; **D 13–16 untouched**), pool 22300, GATE 22400, probe
    training 22500, swap pairs 22610, CF pairs 22630, PLANT directions 22550; exclusion of every test / GATE item from training / pool / probe streams
    by the role-symmetric signature, collisions logged.

**Prototype measurements** [MEASURED v1_checks/v1_sym.json: 5,000 fresh draws per D, seed 24000; leading-space check
v1_sym_lead.json: 1,000 per D at D 4–8, seed 24100]:

| D | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| invariant failures | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| max person-agnostic §59 cell (6 cues × all D + 1 levels) | 0.510 | 0.516 | 0.526 | 0.520 | 0.516 | 0.520 | 0.519 | 0.511 | 0.516 | 0.516 | 0.523 | 0.521 | 0.520 | 0.517 |
| **gated single-parent chase** max(first, random, max-outdeg) | 0.554 | 0.540 | 0.549 | 0.553 | 0.555 | 0.552 | 0.545 | 0.547 | 0.545 | 0.544 | 0.547 | 0.548 | 0.546 | 0.547 |
| LA(1) chase (avoids non-asked persons; gated) | 0.552 | 0.541 | 0.547 | 0.553 | 0.547 | 0.549 | 0.544 | 0.544 | 0.545 | 0.544 | 0.546 | 0.542 | 0.549 | 0.543 |
| LA(2) / LA(3) / LA(4) (lookahead; descriptive) | 0.75 / 1.00 / 1.00 | 0.69 / 0.80 / 1.00 | 0.74 / 0.83 / 0.79 | 0.68 / 0.77 / 0.69 | 0.68 / 0.77 / 0.63 | 0.68 / 0.76 / 0.62 | 0.68 / 0.75 / 0.61 | 0.68 / 0.75 / 0.60 | 0.68 / 0.75 / 0.61 | 0.67 / 0.75 / 0.61 | 0.68 / 0.76 / 0.60 | 0.68 / 0.75 / 0.60 | 0.68 / 0.75 / 0.61 | 0.67 / 0.75 / 0.60 |
| full backward set (H-ROOT-SOFT) / UDIST | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| uBIDIR(r), every r with 2r < D | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| share with a concept beyond D | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T GPT-2 tokens mean / p95 / max | 433 / 512 / 618 | 484 / 572 / 674 | 482 / 578 / 682 | 533 / 636 / 768 | 583 / 692 / 880 | 632 / 746 / 952 | 614 / 678 / 738 | 650 / 710 / 764 | 686 / 749 / 826 | 721 / 782 / 848 | 757 / 818 / 876 | 792 / 850 / 934 | 826 / 886 / 958 | 859 / 920 / 990 |
| concepts \|S\| mean / max | 45 / 68 | 51 / 73 | 52 / 76 | 58 / 86 | 64 / 97 | 70 / 102 | 68 / 82 | 72 / 85 | 76 / 89 | 81 / 93 | 85 / 98 | 89 / 101 | 93 / 106 | 97 / 110 |
| persons per item mean / max | 13.0 / 14 | 13.0 / 14 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 | 11.0 / 12 |
| share with a BFS layer > 4 / L-RR rollouts per chain (chunks ≤ 4) | 0.36 / 3.4 | 0.41 / 4.5 | 0.46 / 5.6 | 0.52 / 6.8 | 0.55 / 7.9 | 0.58 / 9.0 | 0.43 / 9.5 | 0.45 / 10.5 | 0.46 / 11.5 | 0.47 / 12.5 | 0.48 / 13.5 | 0.49 / 14.6 | 0.49 / 15.6 | 0.51 / 16.6 |

- 0 invariant failures in 75,000 draws. The person-agnostic panel sits at E = ½ + sampling (max over 6(D + 1) cells at n 5,000: 0.510–0.526).
- **The gated chase set is 0.540–0.555 at every D** (SE 0.007 at n 5,000), 0.045–0.06 under the 0.6 gate. At n 1,000 it reads up to
  0.5595 (v1_sym_lead.json), so **the gate is read on 5,000 draws per band** (G-L0).
- A backward chase with lookahead ≥ chain length + 1 (LA(3) at D 3–4) is exact, and a full backward or undirected search is exact by
  construction. So SYM v1 excludes person-agnostic cues and single-parent chases; it does not make the task "forward search only".
  Wording: **serial latent reachability** (§5.1).
- T (with the leading space, +1) stays ≤ 1,000 at every D through D 16, so the untouched set keeps **D 13–16**.

### 2.3 A1 `lvb_sym_s1`: LANG-SLVA on SYM v1, gold REACH, D 3–8
- **Targets:** E_t(c) = IN iff dist(R, c) ≤ t (BFS on the item's edges); E_0 = all OUT. Under v1's kernel REACH saturates at D, so
  "HOLD after τ = D" and unheld REACH are the same target (PROOFS §3 fix (b)) [PROVED §3.3]. Simulator states, in no text CoT: never C4.
- **Law:** D ~ U{3..D_max}, CurGate D_max 4 → 6 → 8 on `plain_cand_ce` < 0.35 for 2 consecutive 1k bins, cap 6k per stage; final stage
  to 24k; K_i = min(D_i + U{0..4}, 12); fresh items per draw (seed 22100 stream), per-draw names, premise shuffle, option coin.
- **Recipe:** 24,000 steps × bs 64, lr 1e-4 cosine, c_sched 1000,8000, seed 1, **RTX 3090**, micro per G-L8, off-host mirror every 2k.

### 2.4 A2 `lvb_self_s1` and A2-CTRL `lvb_ctrl_s1` (self round 1, self window D 9–12)
- **Labeller candidates** (A1's model.pt, frozen; **bf16 autocast, the same setting for GATE-1-L and the pool**):
  - **L-ROLL:** A1's own rollout to K′ 16; M_t = {c : argmax R_P(V_t)[c] = IN}.
  - **L-RR (chunked, v1):** each step re-renders the item with the asked person's direct statements replaced by "R is a f." for the
    frontier f, **at most 4 per rollout** (frontiers > 4 are split into chunks of ≤ 4, each a separate K = 2 rollout; the union is taken),
    at the original sentence slots, extra ones at random boundaries; F_{t+1} = ∪_chunks (pass-2 marks \ pass-1 marks) \ M_t.
- **Verifier SOUND-CLOSURE (v1, RT F3-2).** It parses the **original** item's rendered text (never a re-render; PROOFS §3). Given the
  labeller's marks M_t, the maximal sound sub-chain is M′_0 = ∅, M′_t = M′_{t−1} ∪ {c ∈ M_t : "<R> is a c." ∨ ∃ a ∈ M′_{t−1} with "Every a is a c."}.
  **KEEP** iff an option ∈ M′_s for some s ≤ 16 (first such s = the self stop); label = that option. Isolated false marks are dropped,
  never veto the item. [Lemma S-L′, §3.4: π_label = 1, φ = 0, s ≥ D, zero supervised noise.]
- **Selection rule (pre-stated, on GATE items only, seed 22400, 2,000 per D 9–12, gold used only here and in LABEL-AUDIT-L):** the higher
  kept yield pooled over D 11–12 wins, **except that L-RR is chosen over L-ROLL only if its yield exceeds L-ROLL's by ≥ 0.05**
  (L-RR costs 4.6–7.8× L-ROLL per drawn item, v1_pool.json; N9). GATE-1-L stores both chains' per-item KEEP, marks, stop, frontier
  sizes and gold (RT F3-6).
- **GATE-1-L pass:** π_label = 1.000 under the chosen chain (any miss = LABEL-BUG, a code fault); kept yield ≥ 0.2 at every D 9–12.
  Reported beside: omit_rate, FALSE-IN (labeller's raw marks), STOP-LAG, and **kept-vs-drawn** distributions of T, concept count,
  cross-edge count, decoy count and frontier width by D (selection bias, RT F3-2).
- **Pool:** seed 22300, D 9–12 (D never stored), built until **106,000 kept per D** or **530,000 draws per D**; one fixed interleaved
  order (D 9, 10, 11, 12 cycling by record). Record keys exactly {text_ids, slot_idx, stmts_parsed, marks_sound (M′ by pass), stop_s,
  label}, key-set assert + grep (the §215 G-S0 pattern). Fresh-once (self_reuse = 0 asserted). **POOL-SHORT** (fewer than 106,000 kept
  at the draw cap for any D) = A2 not launched (its readings fail by rule b: never reached).
- **Self targets (SOUND-ONLY R2):** IN on c ∈ M′_s from its first M′ pass; no OUT targets, no R2 terms on unmarked slots; R1 window
  [s, K_i], K_i = min(s + U{0..4}, 16).
- **A2 recipe:** FT from A1's model.pt, lr 3e-5, 12,000 × bs 64, c 0, p_self 0.5, seed 1, `--self_kmax 16`; the trainer keeps vbank.py's
  rng_s order exactly (rng_s = seed + 21800; one coin per step; one U per self record), so **seed 1 consumes exactly 5,992 self steps =
  383,488 records = 95,872 per D** [DERIVED v1_pool.py; replay asserted equal to §216's measured cursor]; pool spare 10,128 per D
  (z = 11.4 for any seed). Plain batches are drawn every step from A1's final-stage law (the §215 STREAMS identity).
- **A2-CTRL (required, L2 / RT F5-1):** A2's args with p_self 0 (identical plain stream, no self items). Same host class, launched with A2.

### 2.5 Probes, plants and swap arms (definitions; fields in §4.1)
- **PROBE(t)**, t ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12}: a fresh copy of the trained coda (2 blocks + decoder path; LM head tied, frozen)
  reads V_t + bank versions 1..t−1 of the **frozen** model and is trained on `cand_ce`: **10,000 steps × bs 64, AdamW 1e-4, on ≥ 200,000
  fresh items** (seed 22500, D 3–8 uniform), learning curve every 500 steps stored; if the last 2k steps gain > 0.02 in held-out accuracy,
  extended once to 20k (pre-stated). Read on the SYM test set. PROBE(0) = ½ exactly by construction (§3.5).
- **PLANT(t)**, t ∈ {0, 1, 2} (RT F1-1a, rewritten so the plant is independent of the answer): z ~ Bernoulli(½) per item, independent of
  y; u = a fixed unit vector (seed 22550) in the span of V_t's principal components 40–60; ε = the std of V_t along its 50th PC (2,000
  items); both option-slot rows of the stored V_t get + ε(2z − 1)u; the probe (same recipe, steps and items as PROBE(t)) is trained to
  answer "o1 if z = 1 else o2". **PLANT-XOR:** two directions, z_1 ⊕ z_2. **PROBE-VALID(t)** iff planted accuracy ≥ 0.90 for both plants;
  otherwise PROBE-INVALID(t): every reading that uses PROBE(t) is UNREADABLE (= not met). Final-state validity as drafted:
  PROBE(12) ≥ ACC(K′ = D + 2) − 0.05.
- **FUTURE-DECODE-L (FD-MIRROR, v1 N2):** per-slot probes on the single row V_t[c], **linear (logistic) and MLP (1 hidden layer, 256,
  ReLU)**, trained on fresh items (seed 22500, D 6–8, ≥ 200,000 items) to separate **c with dist(R, c) = t + k** from **c with
  dist(P, c) = t + k**; AUROC read on the SYM test set (D 6–8) for t ∈ {1, 2}, k ∈ {1, …, D − t}. Its null under "no R-specific information
  at the far slot" is ½ exactly (§3.5). **FD-VALID-L:** the same probes at (t, k) = (s, 0), s ∈ {3, 4}, AUROC ≥ 0.90, and a planted
  per-slot code (ε as PLANT, independent bit per slot) AUROC ≥ 0.90; else FD-INVALID (UNREADABLE).
- **Swap arms (ProsQA, all on the frozen model, K′ = D + 2, pairs from rng 22610, recipients = SYM test items, clean = recipient and
  donor each correct at K′ = D + 2):**
  - **PREMISE-SWAP donor** = the recipient with the two persons' names exchanged in every premise ("R is a x" ↔ "P is a x"); the
    question still asks R; the donor's answer is o_P. Token positions and slot indices are identical (asserted per pair).
  - **(ii′) BANK splice (decisive):** at pass t the recipient's stored V_1..V_t (bank entries and SLVA source) := the donor's;
    t ~ U{3..D−1}, D 4–8; t = 2 kept separately (descriptive).
  - **(i′) SLOT-LOCAL splice (decisive, co-primary; A-1):** at pass t, in every stored version 1..t, only the slots
    L_t = {c ∉ {o_R, o_P} : min(dist(R, c), dist(P, c)) ≤ t + 1} take the donor's rows; every other slot keeps the recipient's rows;
    **t ~ U{3..D−2}, D 5–8** (the options must sit ≥ 2 hops beyond t, §3.6 PS-5).
  - **(ii) newest only** (V_t := donor's V_t; descriptive; with (ii′) it reads HISTORY-READ).
  - **PERSON-SWAP** (the question's person token R → P; descriptive mechanism probe: PERSON own ∧ PREMISE follow = the ALL-PERSONS signature).
  - **RESET-t** (v1 N1, descriptive): V_1..V_t := the version-0 slot rows; read at K′ = D + 2, t ~ U{3..D−1}.
- **STALL:** at t = ⌈D/2⌉ the stored entry is V_{t−1}; recovery = pooled acc(D 4–8, K′ = D + 1, stalled) / acc(D 4–8, K′ = D, intact).
- **CONSUMPTION-FRONT (CF; decisive, co-primary for blocker 1; A-2).** An extension of Liang & Pan 2602.00449 (single-input-token
  corruption × latent-position patching in CODI; verified by Literature, addendum 85f6a055) to a looped model with a re-presented input,
  an input swap from pass t and a hop-indexed crossover curve; the prior arms are 203-6 / §207's INPUT-SWAP (the whole input swapped at
  k ∈ {1, 2}). Not new and not a replication.
  - **Pairs:** recipient x and donor x′ differ **only in hop j's fact**: R's hop-j statement "Every a_{j−1} is a a_j^R" (j = 1: "<R> is a
    a_1^R") gets the object a_j^P (a same-length two-token name). Exposed iff in x′ R reaches o_P at distance D and not o_R (y′ = o_P ≠ y),
    and x, x′ have equal token length with every concept's first-mention slot token at the same index (asserted per pair).
  - **Run:** passes 1..t on x; then passes t+1..K′ and the coda with **every read of version 0 switched to x′'s prelude** (all core-layer
    bank entries of version 0, SLVA's s = 0 key and value, and the RMS reference of newly written versions); stored versions 1..t unchanged.
    **K′ = D** (inside the readiness window); t ∈ {0..D}; H_0 = x′'s own run (y′), H_D = x's run (the coda is question-free).
  - **Crossover t\*** per pair = the least-violations change point of [answer = y] over t ∈ {0..D} (ties → the smallest t); clean = x answered
    y and x′ answered y′ at K′ = D.
  - **Decisive j ∈ {3..D−1}, D 4–8** (j ≤ 2 cannot separate precompute's t\* = 1; j = D, the last hop, is reported separately:
    Literature's late-fusion prior, §3.6 PS-6); up to 400 exposed recipients per (D, j), rng 22630.

### 2.6 DIRECT-SYM `lvb_direct_s1` (required; RT F2-2): the one-pass boundary on fresh data
- Plain GPT-2 small (12 blocks), input [EOT] + " " + text, trained on the decoder string "### o_R." (token CE) with A1's stream law
  (seed 22100 stream, CurGate on its `plain_cand_ce`, same stages / thresholds / cap), **24,000 × bs 64**, lr 1e-4 cosine, seed 1, RTX 3090
  (micro per G-L8). Exposure equals A1's.
- Eval: `cand_ce`-based accuracy at D 3..16 on the SYM test set (fp32, the Stage A eval card).
- **C1 band (fixed now, computed by the scorer from DIRECT-SYM's file):** C1 = {D ∈ 5..8 : acc_DIRECT(D′) ≤ 0.65 for every D′ ∈ D..8}.
  D*_fresh := the largest D in 3..12 with acc_DIRECT(D) > 0.65 (descriptive). **A C1 band with < 2 depths makes VB-C1-L not met**
  ("one-pass-solvable in support"; RT §5.3(d)). DIRECT-SYM and A1 launch together; no A1 reading is scored before DIRECT-SYM's file
  exists (score_s218 returns pending).
- If D*_fresh ≥ 6, a follow-up with gold support D 3–12 (RT F2-2) is a new design (its own note, Audit, cap); nothing in this one changes.

### 2.7 Stage B: B0 and B1 `lvb_gsm_s1` (launched only if A1 reads VB-C1-L; D-13)
- Items exactly CODI-20's (bench/data/gsm8k_aug/train.txt with CODI's filters, 382,859); R_ν on, R_x off (D-7, **a system comparison, not
  "one variable"**). DECOMP binary-op programs (draft §3.2) with steps k > 12 and pointers to literals > 14 masked; ambiguous pointers masked;
  the 0.31 % wrong equations' value terms masked. K_i ~ U{m_bin..16}; headline at fixed K′ 14; fixed curriculum m_bin ≤ 2 to 4k, ≤ 5 to
  12k, then all (no adaptive controller). 120,000 × bs 64, RTX 3090, seed 1, off-host mirror every 2k.
- **C4 status (L2, sent to Audit, §9):** equation results are the teacher's CoT values (distillation); **≈ 13.5 % of step value targets are
  calculator-only sub-step values that never appear in the CoT** (multi-op equations, 30 % of train items), and every operand pointer is a
  value-matched annotation. Those terms are **supervised beyond the CoT**, so **Stage B is not C4-legal as a whole**: B1's wording is
  "CoT-derived, calculator-binarised per-pass supervision (C4 qualified)". Audit's alternative: the strict variant masks the sub-step value
  terms (then the value targets are CoT-stated; pointers stay annotations).
- **Interim GO (v1, RT F1-5 / F4-2; replaces the draft's B0-5; gate B0-6):** at step 24,000 the off-host mirror's checkpoint is evaluated **locally** (the Stage B
  eval card) while the host continues: a fresh PROBE(1) (2k steps on Aug-train items, the §4.2 probe recipe) and FD-MLP on Aug-valid.
  **GO** iff acc_fixed14(valid) − PROBE(1)(valid) ≥ 0.05 ∧ FD-MLP(1, k ≥ 3) < 0.8·FD-MLP(k, k) pooled. Else the host is stopped and B1 is
  scored at 24k as **B1-24k** (every registered reading; accuracy descriptive against CODI4 0.189). No trained-coda K′ < m_bin reading
  enters any decision (D-6).
- **CONSUMPTION-FRONT-GSM (co-primary; A-2; the detector for CODI-style late-fusion compression).** x′ = the class-A donor of step j (both
  literals of step j redrawn by §200's draw_literal law **with the same GPT-2 token count**, downstream values recomputed exactly, y′ ≠ y);
  passes 1..t on x, then passes t+1..14 and the coda with version 0 (question and placeholder rows) from x′; **K′ = 14**, t ∈ {0..14}.
  Serial consumption of step j's literals gives t\* ∈ {j − 1, j}; precompute-and-carry t\* = 1; recompute / late consumption t\* = 14.
  Decisive j ∈ {3..m_bin − 1} (the last step j = m_bin descriptive: LAST-STEP-LATE; j = 2 descriptive). Recipients: the exposed steps
  × (1 + 40 R_x variants) (rng 22730; eval only). **LATE-FUSION-GSM** (descriptive) := median t\* ≥ 12 over all exposed j: the CODI-style
  "latents compress, the answer is fused late" signature (2602.00449), here possible only in the final passes because the coda is
  question-free.

### 2.8 A0-1 (registered here; $0, descriptive; worded per 191-3)
- The kept §191 models (s191_stepaux_relabel_s1, s191_textcot_relabel_s1; model.pt) re-evaluated on p8_na_test at §191's original eval
  card class, dtype and eval_bs, with per-item outputs, split by anchor proximity at depth ≥ 5 (m ≥ 2): **pro 2,630 / tie 511 / anti 78**
  [MEASURED draft anticue.py; PROOFS reproduced]; plus NN items rendered in CON38 at D ≤ 8 against p8_na at the same depth.
- Readings (all descriptive): CUE-CONTRIBUTION = acc(pro) − acc(tie) by depth; ANTI-SIGNATURE = acc(anti) ≤ 0.40 (P under no reliance
  at true acc 0.95 ≈ 0); NN-CONTRAST = acc(NN) − acc(p8_na).
- **Wording (191-3):** "A0-1 measures the anchor cue's contribution to the §191 models' accuracy at depth ≥ 5. It cannot explain §191:
  §191 A exceeds the best person-agnostic cue lookup at every depth 3–10." No registered §191 reading changes.

### 2.9 Stage flow and GO rules
1. **A0** ($0, local): A0-1 first (to Theory / Audit); G-L0 (SYM gate) before anything is built on SYM; the harness gates (§7).
2. **DIRECT-SYM ∥ A1** (two 3090 hosts; Audit's approval; admissions §8).
3. A1 and DIRECT-SYM evals (Stage A card). **A2 GO** = A1 uncensored ∧ DEEP-L ∧ NO-CLOCK-L ∧ GATE-1-L passes ∧ (the user's explicit OK if
   A1 reads MEM-TRADED-L, L6). Then labelling (§8 plan) → pool → G-L20 → **A2 ∥ A2-CTRL**, launched only together (both admitted; A2
   without its control has no SD-C2-L reading).
4. **B GO** = A1 reads **VB-C1-L** (D-13) ∧ (the user's explicit OK if any LANG cell reads MEM-TRADED-L) ∧ the family remainder covers B1's
   guard. Then B0 → B1 with the interim GO. Otherwise Stage B is not run in this design (saves ≈ $4–6).
5. Any further self round, and B2, are new designs (B2 trigger = VB-C1-GSM; any further round after MEM-TRADED-L needs the user's OK).

### 2.10 Code plan (Experiments implements; Audit's diff review; nothing written here)
| file | content | touches existing code? |
|---|---|---|
| boxes/prosqa_sym.py | SYM v1 generator (§2.2), NONCE-300 / PERSON-40 (frozen, sha), REACH, role-symmetric signature, LEVEL-PANEL + chase set + uBIDIR (G-L0), PREMISE-SWAP / PERSON-SWAP pair maker with the position assert | no |
| boxes/langvb.py | `LangSLVA(slva.SLVA)`: slot rollout, `bias_row_ls`, `coda_dec`, R_P / R_G, `lvb_loss`, `cand_ce`, the trainer (CurGate on `plain_cand_ce` only; rng_s order = vbank's; micro / self_micro); eval hooks (edit, ver_keep, splice by slot set, reset, dup_newest) | no (subclass; vbank / slva imported read-only) |
| boxes/langvb_label.py | L-ROLL, chunked L-RR, SOUND-CLOSURE (parses the original text), GATE-1-L, the pool builder (counts, caps, key set, POOL-SHORT) | no |
| boxes/langvb_eval.py | parts acc / probe / plant / fd / vswap (ii′, i′, ii, person, reset) / stall / **cf** (one-fact donor maker, version-0 switch from pass t) / hist24 / causal_deep / labeller / direct, each merged into eval_lvb.json as it finishes, per-item stores | no |
| boxes/gsm_prog.py, boxes/langvb_gsm_eval.py | DECOMP, class-A pair maker (exposure filter §3.8), GSM parts (§4.2) | no |
| theory/score_s218.py | the scorer (§4.3), fixture-checked before any output | new |

## 3. Mathematics

### 3.1 Depth lemma L1-L [PROVED; PROOFS §4 correction applied]
V_t is computed from the question by a layered network of depth ≤ P + d·t blocks **plus t SLVA attention sublayers** (P = 6, d = 4); the
answer after K′ passes is at depth ≤ P + d·K′ + 2 blocks (+ K′ sublayers). Reads of V_0 start at depth P (add nothing); the decoder
string's blocks 0–9 lie on no question path; the slot subset only removes edges. ∎ Immaterial to any capacity argument: at t = 1, V_1 is
10-block GPT-2 depth at full width, so nothing architectural stops pass-1 precompute (§3.5).

### 3.2 No pass lower bound; the one-pass boundary is measured [PROVED trivially; CAP-aff-L ASSUMED]
SYM v1 reachability (≤ 4D + 52 concepts at D ≥ 9 incl. decoys, measured max 110 at D 16) and GSM8K arithmetic (≤ 12 binary steps) are finite
functions, so no unconditional lower bound on passes holds. **CAP-aff-L** (a pass of the trained model closes ≤ c₀ + (m − 1)·c hops,
c ≈ 1) stays [ASSUMED]; its cited evidence measures no one-pass capacity (§168 DIRECT memorised a fixed set; §191's K′ 0 / 1 are
trained-decoder readings at untrained K′; PROOFS §4). v1 replaces it where it matters: the C1 band is DIRECT-SYM's measured boundary on
fresh SYM v1 data at A1's exposure (§2.6), and the in-model PROBE(1) / FD-MIRROR / PLANT read precompute directly (§3.5).

### 3.3 SYM v1 [PROVED unless tagged]
Let ω = (W_1, W_2, the cross edges C_12, C_21, extras, decoys, names, statement order π, option coin b), the question asking W_1's person.
Z = the item minus the question's person token (premises + displayed options), Y = 1[o1 is the target]. σ swaps W_1 ↔ W_2 (with their
decoys), C_12 ↔ C_21, the two persons' names, keeps each concept's name on its node and π on statement identities, and flips b.
- **(a) Exchangeability (PROOFS 2a, strong form).** Z(σω) = Z(ω), Y(σω) = 1 − Y(ω). If the law is σ-invariant, Y ⟂ Z and
  P(Y = 1 | Z) = ½, so **every person-agnostic predictor h(Z, U) has accuracy exactly ½**, however deep. σ-invariance holds because the
  two worlds are i.i.d. from G(D) including the decoy law and the per-world cap (S-σ1), the cross-edge kernel is the same in both
  directions, names are sampled jointly and uniformly, the order is uniform, the coin fair, the contamination redraw uses a role-symmetric
  signature (S-σ2) and the T ≤ 1,000 rejection is σ-invariant because every person name is one token in its leading-space form (S-σ3). ∎
- **(b) Distances.** dist(R, a_k) = k: distractor subtrees hang off path nodes; an extra edge u → a_k needs depth(u) + 1 ≥ k; decoys enter
  path nodes only from decoy persons; cross edges end at leaves of the other person's direct subtrees, which reach no path node. o_P ∉ Reach(R):
  R reaches only W_R, its direct subtrees and (by cross edges) leaves of P's direct subtrees. ∎ [MEASURED 0 / 75,000 failures]
- **(c) REACH saturates at D (fixes PROOFS §3's HOLD hole).** Every concept reachable from R lies in W_R (depth ≤ D) or in R's direct
  subtrees (depth ≤ 2) or is a cross-edge target, a leaf at depth ≤ 3 with no descendants. So max dist(R, c) ≤ max(D, 3) = D for D ≥ 3,
  R_t = R_D for t ≥ D, and "HOLD after τ = D" equals unheld REACH. ∎ [MEASURED beyond-D share 0.000 at every D 3–16]
- **(d) Undirected search.** ud(R, o_R) = D (no undirected shortcut: an extra edge u – a_k with depth(u) ≥ k − 1 never shortens; decoy
  chains and extra-person trees are dead ends). Every undirected R → o_P path passes P (worlds meet only through direct-subtree cross edges),
  so ud(R, o_P) ≥ ud(R, P) + D ≥ 3 + D. Hence **UDIST = 1 exactly** and **uBIDIR(r) = ½ exactly for every 2r < D** (neither option is
  within 2r of R). ∎ [MEASURED: 1.000 / 0.500 at every D]. Directed BIDIR is ½ / 1 by construction and is not reported (PROOFS 2b).
- **(e) Single-parent chases** have no theorem: H-ROOT (first / random / max-outdegree) and LA(1) are **gated empirically** (< 0.6 at every
  band, n 5,000) [MEASURED 0.540–0.555]. Chases with lookahead ≥ the decoy chain length + 1 can be exact (LA(3) = 1.000 at D 3).
- **(f) Concept bound.** At D ≥ 9, |S| ≤ 2(2D + 6) (capped worlds) + 2·4·4 (decoy chains ≤ 2 nodes + ≤ 2 leaves) + 2·4 (extra persons'
  trees) = 4D + 52 (116 at D 16); used only for memory; measured max 110.

### 3.4 SOUND-CLOSURE and zero supervised noise [PROVED]
**Lemma S-L′.** For any marks M_t, the maximal sound sub-chain M′ (§2.4) satisfies M′_t ⊆ R_t for all t.
*Proof.* M′_0 = ∅. If M′_{t−1} ⊆ R_{t−1}, each new c ∈ M′_t is a direct concept of R (dist 1) or has a statement parent a ∈ M′_{t−1} ⊆ R_{t−1},
so dist(R, c) ≤ t. ∎ (The proof uses only the item's own statements; it holds for any directed graph, any labeller, any dtype.)
**Corollaries.** (i) With o_P ∉ Reach(R) (§3.3b), a kept chain's label is o_R: **π_label = 1, φ = 0**, and the stop s ≥ D.
(ii) Each self R2 target is "IN at (c, t)" for t ≥ f(c), the first M′ pass of c; f(c) ≥ dist(R, c) and REACH is monotone, so the gold
target at (c, t) is IN. The self R1 window [s, K_i] lies in the gold window (s ≥ D; HOLD exact by §3.3c). **Every self target equals the
gold target of the same item, slot and pass: zero supervised label noise.** Only the changed / unchanged weighting of late marks differs. ∎
**Scope (PROOFS §3).** SOUND-CLOSURE never accepts an over-reach (a concept at dist > t marked at t) nor a wrong label; it accepts
incomplete and late chains (omission, lateness). What it distorts is coverage: (a) kept items are those the labeller partly solves
(selection: reported kept-vs-drawn, and ABSORB's denominator is all test items); (b) no OUT terms on self items (G7: FALSE-IN on GATE after
A2); (c) a late R1 start (STOP-LAG). Replacing reject-on-any-false-mark removes the (1 − f)^~600 yield coupling (RT F3-2).

### 3.5 Precompute: what each detector can and cannot see
- **PROBE(0) = ½ exactly** [PROVED]: V_0's slot rows precede the question (causal prelude, question last) and the coda is question-free,
  so PROBE(0)'s output is a function of Z (§3.3a). PROBE-GAIN-L therefore uses PROBE(1) (RT F1-4).
- **Precompute-and-reveal / -carry is zero-loss given capacity** [PROVED, draft §4.4]: R1 / R2 read only R_P / R_G and the coda.
- **PROBE(t)** is one-sided (high ⇒ present; low ⇏ absent). **PLANT** certifies the probe's power on a planted code of known size
  (1 bit, ε = the 50th-PC std, linear and XOR) in the same V_t, recipe and budget; it does not certify power on every possible hidden code.
- **FD-MIRROR null** [PROVED UNDER H0]: if the far rows V_t[c] (dist(R, c) = t + k or dist(P, c) = t + k) are functions of Z and the slot's
  identity only (no R-specific information has reached them), then σ maps the positive class onto the negative class preserving the joint
  law of (row, Z), so the class-conditional row laws are equal and **any probe's AUROC is ½**. Under a decodable precompute of R's reach
  the far rows separate (AUROC → 1). Measured class sizes ≈ 2.95 slots per item per class per (t, k) [MEASURED v1_sym.json], so ≈ 8,800
  per class per (t, k) pooled over D 6–8 test items: AUROC SE ≈ 0.004 [DERIVED v1_oc.json].
- **(i′) slot-local**: §3.6 PS-5. **STALL** detects pass-count clocks only (content-keyed serial, reveal and precompute all recover).
- **Undetectable combination** (PROOFS §4, kept): a pass-1 precompute in a code that neither the PLANT-certified probes nor the per-slot
  FD probes decode, released content-keyed and stored in slots inside L_t, passes every probe-based component. **CONSUMPTION-FRONT
  (A-2) is decode-free and fails it** (its answer is fixed by pass 1, so t\* = 1; §3.6 PS-6). Without CF-L the OC's P(pass | no serial
  use) would be conditional on PLANT-certified decodability; with it, the residual route is a model that consumes each hop's fact at its
  own pass, which is serial consumption by definition.

### 3.6 PREMISE-SWAP × (ii′) and the slot-local arm (i′): re-proof (replaces draft §4.6; L1 B2)
**Setup.** Recipient x (asked R, foil P). Donor x̃ = x with the names of R and P exchanged in every premise; the question still asks R.
x̃ = σ(x) with the question token reset to R, so x̃ is a valid SYM v1 item whose answer is o_P (dist_x̃(R, o_P) = D, o_R ∉ Reach_x̃(R)).
The concept–concept statements of x and x̃ are identical; only the person statements' subjects differ.
- **PS-0 (positions)** [PROVED; MEASURED]. With one-token person names in leading-space form, x and x̃ have equal token length and every
  concept's first mention ends at the same token, so slot indices are identical. [MEASURED: 5,000 / 5,000 pairs at D 4–8 with the
  leading-space rule; 2.3–3.0 % mismatches without it, N3.] V_0 rows differ at and after the first person token (causal prelude).
- **Model classes.** *S-NAME:* a serial rule, one hop per pass on the readable state, whose carried state is indexed by **person names**:
  either question-conditional (IN_s = IN_{s−1} ∪ children(IN_{s−1}) ∪ seed_s, seed_1 = direct(R_q), later seeds ⊆ direct(R_q)) or
  ALL-PERSONS (tags S_s(c) = {p : dist(p, c) ≤ s}, IN_s(c) = [R_q ∈ S_s(c)]); look-ahead ≤ 1 hop (the state of c at pass s may depend on R
  only if dist(·, c) ≤ s + 1). *RECOMPUTE:* for s > t, V_s depends on (the recipient's V_0, s) only. *ALL-ANCHORS:* tags by anchor concepts
  A = {objects of person statements} (the same set in x and x̃): S_s(c) = {a ∈ A : dist(a, c) ≤ s − 1}, IN_s(c) = [∃ a ∈ S_s(c): "R_q is a a"
  in V_0]. *PaR-C:* pass 1 computes the answer (or R's whole reach) into an unread subspace, carries it, reveals one hop per pass.
- **PS-1 (follow under S-NAME)** [PROVED UNDER S-NAME]. Splice (ii′) at t ∈ {3..D−1}. The donor's carried state at t marks / R-tags the W_P
  nodes at P-depth ≤ t (in x̃, R's statements are P's) and nothing of W_R. Later passes apply the recipient's concept edges, identical to
  x̃'s, so o_P is IN at pass D. o_R needs a chain re-seeded from direct(R) in the recipient's V_0 at pass ≥ t + 1, reaching o_R at pass
  ≥ t + D > D + 2 for t ≥ 3. At K′ = D + 2 only o_P is IN: **follow**. (At t = D − 1 the donor's look-ahead on o_P is spliced too.) ∎
- **PS-2 (own under RECOMPUTE)** [PROVED]: the hybrid equals the recipient after t: **own**. ∎
- **PS-3 (ALL-ANCHORS reads own; v1 N1)** [PROVED, given capacity]. ALL-ANCHORS meets R1 / R2 exactly with one hop per pass on S. S is a
  function of the concept edges and A only, identical in x and x̃, so (ii′) leaves S unchanged and IN after t is recomputed from the
  recipient's question and V_0: **own** at every t. Likewise under (i′). So "own" ⇒ RECOMPUTE ∨ ALL-ANCHORS ∨ another non-name state. ∎
- **PS-4 (PaR-C follows under (ii′))** [PROVED]: the donor's carried hidden answer is o_P; after (ii′) the recipient carries o_P and reveals
  it: **follow**. Hence (ii′) alone cannot separate PaR-C from serial use (RT F1-1). ∎
- **PS-5 (slot-local (i′))**. L_t = {c ∉ {o_R, o_P} : min(dist(R, c), dist(P, c)) ≤ t + 1}, t ∈ {3..D−2} (both options at dist ≥ t + 2).
  (a) [PROVED UNDER S-NAME] Under S-NAME every slot outside L_t carries the same state in x and x̃ at passes ≤ t (no person has reached it,
  even with one-hop look-ahead), so the (i′) hybrid equals the (ii′) hybrid on every slot that matters: **follow**. (b) [PROVED for this
  variant] PaR-C with the carried answer (or R's hidden reach) stored at the option slots or at any slot outside L_t that the reveal reads:
  those rows stay the recipient's: **own** (the nested arm (i) reads 0.996–1.000 follow, so nested D3-T / FT-SELF / R2-s1 are not open to
  this route). (c) PaR-C that replicates its hidden answer into the L_t slots is not decided by (i′); FD-MIRROR (far rows, k ≥ 3) and PLANT
  cover it as far as the probes reach (§3.5). ∎
- **RESET-t (descriptive; separates PS-2 from PS-3)** [PROVED]: V_1..V_t := the version-0 slot rows. RECOMPUTE: unchanged (recovery ≈ 1).
  S-NAME and ALL-ANCHORS: the carried state restarts, reaching an option at pass ≥ t + D > D + 2 (t ≥ 3), so neither option is IN at
  K′ = D + 2 and accuracy drops toward an undefined readout. PaR-C keyed on "no carried answer yet → compute" also recovers. ∎
- **Labels.** HISTORY-READ := (ii) own ≥ 0.5 ∧ (ii′) follow ≥ 0.80. ALL-PERSONS signature := PERSON-SWAP own ≥ 0.80 ∧ (ii′) follow ≥ 0.80.
  RECOMPUTE-L := (ii′) own > 0.3 ∧ RESET recovery ≥ 0.9. ANCHOR-TAGGED := (ii′) own > 0.5 ∧ RESET recovery ≤ 0.75. All descriptive;
  "recompute" is never worded from (ii) or PERSON-SWAP (RT §5.4).
- **PS-6 CONSUMPTION-FRONT (A-2).**
  - *Precondition (check-borrowed-technique rule)* [VERIFIED by code read]: the technique needs every pass to re-read the input. In
    vbank.rollout the prelude output V_0 is bank entry 0 of every core layer at every pass (`kv[i][0] = bank_kv(g, i, V0)` is concatenated
    into every pass's bk / bv, with the β⁰ bias) and SLVA's gather reads it at s = 0 (boxes/slva.py: `gk[:t]`, `NV[:t]` include s = 0);
    LANG-SLVA keeps both with version 0 = all T prelude rows (§2.1). So h_x is fed to every pass, and the swap is applied at version 0
    (the prelude output as every pass reads it). The coda never reads V_0 (pin 4), so CODI's late fusion at the answer position
    (2602.00449: the last input rescued only at Ans) is closed by construction: no crossover can occur after pass K′.
  - *Curves* (t\* = the pass from which the answer is y):
    - **Serial consumption** (one hop per pass; S-NAME, ALL-PERSONS **and ALL-ANCHORS** alike, since tags propagate through the current
      V_0's edges) [PROVED UNDER one-hop consumption]: for t ≤ j − 1 the frontier crosses x′'s hop-j edge into a_j^P, reaches o_P at pass D,
      and never reaches a_j^R (exposure (a)): y′. For t ≥ j, a_j^R was marked at pass j under x and x′'s later edges equal x's, so o_R is IN
      at pass D; o_P could enter only through x′'s new edge from the already-marked a_{j−1}^R, at pass ≥ t + 1 + D − j ≥ D + 1 > K′ even under
      full re-propagation: y. So **t\* = j**; with one-hop look-ahead (a_j^R's version at pass j − 1 already holds "parent IN") **t\* = j − 1**.
      A consumer lagging one pass would miss K′ = D and is not clean, so t\* = j + 1 is not a serial outcome on clean pairs.
    - **Precompute-and-carry** (answer fixed by pass-1 computation on x and carried; decode-free, so it covers §3.5's undetectable
      combination): **t\* = 1** for every j.
    - **Recompute-from-input** (V_s, s > t, a function of the current V_0): y′ for every t < D, y at t = D: **t\* = D**.
    - Partial one-pass (c₀ hops in pass 1, then serial): t\* = max(1, j − c₀ + 1). Pointer doubling: t\* ≈ ⌈j/2⌉. A D-keyed clock: t\* = f(D).
  - *Derived pass rule* **CF-L** := among exposed clean pairs with j ∈ {3..D−1}, the share with **t\* ∈ {j − 1, j}** ≥ **0.75** **∧** the
    pair-weighted mean over D 5–8 of Spearman(t\*, j) within D ≥ **0.60**. The window is the serial set derived above; Audit's proposed ±1
    also admits j + 1, which recompute reaches at j = D − 1, so it needs j ≤ D − 2. Spearman is computed within D so that a D-keyed clock
    (constant t\* within D) scores 0; precompute and recompute are constant in j and score 0.
  - *Calibration* [DERIVED v1_cf.py → v1_cf.json; Monte Carlo, 150 replicates, 100 clean pairs per (D, j) cell (the projection is ≈ 324),
    per-hybrid readout flips ε, crossover = least-violations change point]:

    | rule | serial ε 0 / 0.05 / 0.10 / 0.15 (look-ahead 0 \| 50 %) | precompute (t\* = 1) | recompute (t\* = D) | clock D − 2 / clock 4 | doubling | partial c₀ 3 |
    |---|---|---|---|---|---|---|
    | **registered**: {j − 1, j}, j 3..D−1, share ≥ 0.75, ρ ≥ 0.60 | 1.00 / 1.00 / 1.00 / 0.00 \| 1.00 / 1.00 / 1.00 / 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 |
    | {j − 1, j}, j 3..D−1, share ≥ 0.80, ρ ≥ 0.80 | 1.00 / 1.00 / 0.04 \| 1.00 / 0.70 / 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 |
    | Audit's ±1, j 3..D−2, share ≥ 0.80, ρ ≥ 0.80 | 1.00 / 1.00 / 0.00 \| 1.00 / 0.007 / 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 |

    Every non-serial class fails at every ε ≤ 0.15 under all three rules (window shares ≤ 0.57 for the clocks, ≤ 0.29 for doubling, ≤ 0.12
    for partial one-pass; ρ ≈ 0 for every j-constant class). The registered thresholds keep the power of a serial model with half its
    hops looked ahead and 10 % readout flips; Audit's literal rule loses it at ε 0.05 with look-ahead (fewer j values per D). **Noise floor:**
    CF-L needs a per-hybrid flip rate ≲ 0.12; the clean-pair filter and the K′ = D readout keep hybrids in the trained window.
  - *Last hop (Literature's prior, pre-stated):* j = D is descriptive. Serial consumption gives t\* ∈ {D − 1, D}; recompute also gives D, so
    j = D cannot separate them, and late fusion at the coda is closed here. A j = D crossover at t = D with CF-L met on j < D reads
    "LAST-HOP-LATE (consumed at the final pass)", never a failure of serial consumption at the earlier hops.
  - *Exposure (A-2):* §4.1 table, per (D, j).

### 3.7 Support identity on SYM v1 (S0-L) [PROVED]
(a) A1's targets change last at pass D ≤ 8 (REACH saturates at D, §3.3c); A2 adds self stops s ∈ [D, 16]. (b) Any t_d ≥ s_max, "expand while
t ≤ t_d", and "expand until an option is IN" both match every training target, so depth beyond the training stops is not forced by loss:
**H_shift is irreducible** (as SELFDEPTH S0(b)). STALL separates the rules in range; DEPTH-BEYOND-SELF-L, MARGIN-GROWS-L and BEATS-CTRL-L
read them out of range (D 13–16, untouched). (c) LIT-C2-L (K′ 18 vs 16 at D 15–16) is ≈ 0 under complete success because K′ 16 ≥ D already
suffices; literal C2 needs D ≥ 17, which SYM v1 does not render inside 1,024 positions: LIT-C2-L is UNINFORMATIVE here (RT F5-3).

### 3.8 GSM8K [MEASURED / DERIVED]
- **Class-A exposure (A-1)** [DERIVED v1_gsm_exposure.json, PROOFS' parser imported read-only]. A test item is exposed at step t iff it
  DECOMP-parses, 2 ≤ t ≤ m_bin − 1 ≤ 11, both operands of step t are unique, unambiguous question literals used by no other step, step t's
  result is an ancestor of the final step, and a redraw of the two literals changes the final answer. **95 of 1,319 items (94 at m ≥ 3;
  105 exposed steps)**: m 3: 39, m 4: 42, m 5: 8, m 6: 3, m 7: 1, m 8: 1 items. With the 1,319 base items alone the clean-pair count would
  be ≈ 12 at accuracy 0.27, so **recipients are the exposed items × (1 + 20 R_x variants)** (rng 22720; eval only): 1,974 candidates,
  projected clean pairs 120–553 for m ≥ 3 accuracy 0.15–0.40 and recipient–donor correctness correlation 0.3–0.5 [DERIVED v1_oc.json].
- **C4 accounting** (PROOFS §5): 1,135,784 binary step targets vs ≈ 982,500 equations in parsed train items, so ≈ 13.5 % of value targets
  are calculator-only sub-steps; pointer provenance: unique literal 48.9 %, earlier result 26.0 %, internal sub-step 6.7 %, ambiguous
  2.6 % + non-unique 2.9 % (masked), CONST 12.9 %.
- π_prog = P / (P + φ(1 − P)) (B2, deferred) [PROVED identity; φ ASSUMED].

### 3.9 Memory and time [DERIVED v1_mem.json; form ASSUMED (RT mem_cost.py), calibrated to D3-T's 0.42 s/step]
Shapes are **batch maxima** of 64-item batches from the v1 generator (padding): A1 final stage T_pad 748 (max 942), |S|_pad 83 (103); A2
self T_pad 786, |S|_pad 88. Saved memory at bs 64: A1 23–30 GB (worst 32–43), A2 self 31–44 GB (worst 35–48), DIRECT-SYM 16 GB (worst 20),
B1 3–5 GB. **Micro 4** fits 21 GiB for A1 / A2 / A2-CTRL, **micro 2** for DIRECT-SYM, **1** for B1 (G-L8 decides). s/step (incl. +4 % per extra
micro-batch [ASSUMED]): A1 2.17 (forced 2.89), A2 self 2.72, DIRECT-SYM 1.87, B1 0.56–0.70 (launch-bound).

## 4. Readings, definitions and the scorer's field contract

**General rules.** Every reading runs on the frozen model in **fp32 on the stage's eval card** (Stage A: the local **RTX 4060**; Stage B:
the RTX 4060; card recorded in every file), at the eval_bs that G-EVMEM pins per part (the "original eval_bs" for every re-check). Every
share or threshold has its per-item / per-pair outputs stored. A ProsQA band that fails G-L0 enters no reading. Each part is merged into
the eval file as it finishes. **Card fallback (pre-stated, SD2 Addendum-1 form):** if the 4060 is lost and recovers, unfinished parts
re-run on it (partials kept as `*.void_gpulost`); if not, every registered eval of the stage moves together to the RTX 3070, with the
reference re-evals (A1 / DIRECT-SYM for A2; CODI-20's B0-2 for B1) first; no partial move inside a cell's reading set.

### 4.1 ProsQA: eval_lvb.json (A1, A2, A2-CTRL) and eval_direct.json (DIRECT-SYM); SYM v1 test, 1,000 per D, D 3..16
| part | fields | definition |
|---|---|---|
| panel | `panel {D: {n_gate 5000, max_person_agnostic, chase {first, random, maxdeg, la1, la2, la3, la4}, soft, ubidir_max, udist, beyond_D, gate_pass}}` | G-L0 on generator draws per stream (copied from the gate JSON, not recomputed) |
| acc | `acc_d2 {D}` (K′ = D + 2), `acc_k {K′: {D}}` K′ 0..24, `items {D: {K′: [0/1]}}`, `cand_ce_items`, `T_band {D: {T ≤ train-max, T >}}` | candidate accuracy: Σ log p("### o_R.") > Σ log p("### o_P.") from the coda after K′ passes |
| probe | `probe {t: {acc_by_D, items, curve [(step, heldout_acc)], steps_run}}`, t ∈ {0,1,2,3,4,5,6,7,8,10,12} | §2.5 |
| plant | `plant {t: {lin: {acc, items}, xor: {acc, items}, eps, pcs}}`, t ∈ {0, 1, 2}; `probe_valid {t}` | PROBE-VALID(t) iff both ≥ 0.90 |
| fd | `fd {model: {lin, mlp}: {t: {k: {auroc, n_pos, n_neg}}}}`, t ∈ {1, 2}, k ∈ 1..D−t; `fd_valid {s: auroc}` s ∈ {3, 4}; `fd_plant {auroc}`; per-slot scores stored | FD-MIRROR (§2.5) |
| vswap | per arm a ∈ {ii_prime, i_prime, ii, person, reset}: `{n_candidates, n_exposed, n_clean, follow, own, by_D {n_exposed, n_clean, follow, own}, by_t, t2_descriptive, pairs [{rec, don, D, t, y_hyb, ok_rec, ok_don}]}` | §2.5; `n_exposed` = pairs whose position assert holds and whose arm is defined at that D, t (see the table below) |
| stall | `stall {recovery, by_D, items}` (≥ 2,000 items) | §2.5 |
| cf | `cf {D: {j: {n_candidates, n_exposed, n_clean, curve {t: P(answer = y′)}, tstar [per pair], share_window}}, spearman_within_D {D}, share_window_pooled, last_hop {D: {tstar_hist, curve}}, pairs [{rec, don, D, j, answers_by_t}]}` | CONSUMPTION-FRONT (§2.5, §3.6 PS-6) |
| hist24 | §213-21 contract (intact / bank_ablate_slva / self_ablate / dup_newest at K′ 24, D 3–8), per-item answers at K′ D + 2 and 24 (the SD2 E4 scope) | KEEPS-SELF-STAB-L as SELFDEPTH_v1 §4.3 |
| causal_deep | the §217 contract at D 9..16 with arms ii_prime / i_prime + STALL; competent band B = {D : acc_d2 ≥ 0.95} | CAUSAL-DEEP-L (§217 rule) |
| labeller (A2) | `gate1 {chain: {D: {n, kept, yield, omit_rate, false_in, stop_lag, frontier_hist}}, choice, items}`, `kept_vs_drawn {D: {T, concepts, cross, decoys, frontier}}`, `label_audit` (5,000 consumed records vs gold) | π_label must be 1.000 (S-L′) |
| direct | `acc_by_D` D 3..16, `items`, `c1_band`, `d_star_fresh` | §2.6 |

**Exposure (A-1) for the decisive swap arms** [MEASURED v1_sym_lead.json: per D, the 1,000 test-law items; MEASURED / DERIVED as marked]:

| D | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| (ii′) PREMISE-SWAP × bank: admissible t | 3 | 3–4 | 3–5 | 3–6 | 3–7 |
| (ii′) exposed items (position identity ∧ frontier differs) | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| (i′) slot-local: admissible t (t ≤ D − 2) | **none: excluded and listed** | 3 | 3–4 | 3–5 | 3–6 |
| (i′) exposed items (identity ∧ L_t frontier differs ∧ both options outside L_t) | 0 | 1,000 | 1,000 | 1,000 | 1,000 |
| mean \|L_t\| / differing slots in L_t (at the drawn t) | 21.2 / 16.3 | 25.1 / 19.2 | 28.4 / 22.1 | 31.5 / 25.1 | 34.6 / 28.1 |
| projected clean pairs at acc 0.90 / 0.95 / 0.99 (= n·a² if recipient and donor independent) | 810 / 902 / 980 per D | | | | |
| **CF exposed recipients per 1,000 items, decisive j = 3..D−1** (single-fact rewire; flip valid ∧ positions identical) [MEASURED v1_cf.json, seed 24300] | j3 596 | j3 490, j4 516 | j3 505, j4 490, j5 487 | j3 422, j4 486, j5 437, j6 482 | j3 424, j4 386, j5 471, j6 489, j7 486 |
| CF last hop j = D (descriptive) / j = 1, 2 (descriptive) | 484 / 571, 558 | 336 / 554, 539 | 356 / 564, 487 | 327 / 578, 473 | 348 / 592, 446 |
| CF pairs used: up to 400 exposed per (D, j); projected clean per cell at acc 0.90 | ≈ 324 | ≈ 324 × 2 | ≈ 324 × 3 | ≈ 324 × 4 | ≈ 324 × 5 |

- CF exposure is 39–60 % per (item, j): the rewire fails when another R-reachable path reaches a_j^R or o_R (the flip condition) or when
  the rewired sentence held a first mention (positions). The double-swap variant (both persons' hop-j objects exchanged) exposes fewer
  (14–44 %) and is not used.

- A reading is **readable** only with ≥ 100 exposed clean pairs pooled; a per-D value enters the per-D bar only with ≥ 50 exposed clean
  pairs; D values below that are **excluded and listed**, never passed. D 4 is excluded from (i′) by construction.
- Exposure is by construction 100 % at admissible (D, t) under the leading-space rule; the binding count is the clean-pair count, which
  is the model's accuracy (DEEP-L requires ≥ 0.90 at every D 4–8, so ≥ ≈ 800 clean pairs per D when VB-C1-L can pass).

**Decisive readings (A1; comparator-free).**
- **DEEP-L:** acc_d2 ≥ 0.90 at every D 4–8.
- **PROBE-GAIN-L:** pooled over the C1 band (§2.6; ≥ 2 depths), ACC(K′ = D + 2) − PROBE(1) ≥ **0.35**; requires PROBE-VALID(1) (PLANT) and
  PROBE(12) ≥ ACC − 0.05. ONESHOT-L := PROBE(2) ≥ 0.8 at D 8 (falsifies the premise).
- **SERIAL-ONSET-L:** ON(D) = the smallest t in the probe grid with PROBE(t′) ≥ 0.90 at D for every grid t′ ≥ t (never reached = failure);
  OLS slope of ON(D) on D (D 4–8) ≥ 0.6 ∧ ON(8) − ON(4) ≥ 3; requires PROBE-VALID(t) for t ≤ 2.
- **SERIAL-STEP-L (co-primary arms):** (ii′) follow ≥ **0.80** pooled over D 4–8 ∧ ≥ 0.60 at every readable D; **and** (i′) follow ≥ **0.80**
  pooled over D 5–8 ∧ ≥ 0.60 at every readable D. Each arm ≥ 100 exposed clean pairs, else UNREADABLE (= not met).
- **NO-CLOCK-L:** STALL recovery ≥ 0.9.
- **NO-PRECOMPUTE-L:** FD-VALID-L ∧ AUROC ≤ 0.65 for every (t ∈ {1, 2}, k ≥ 3) cell, linear and MLP, pooled over D 6–8 (k = 1, 2 descriptive:
  one-hop look-ahead is legitimate, k = 2 reads hidden over-reach G6).
- **CF-L (CONSUMPTION-FRONT, co-primary; A-2):** among exposed clean pairs with j ∈ {3..D−1} (D 4–8), share with t\* ∈ {j − 1, j} ≥ **0.75**
  ∧ pair-weighted mean within-D Spearman(t\*, j) over D 5–8 ≥ **0.60**; ≥ 100 exposed clean pairs pooled, else UNREADABLE; a (D, j) cell
  with < 30 exposed clean pairs is excluded and listed. The last hop (j = D) and j ≤ 2 are descriptive (LAST-HOP-LATE, §3.6 PS-6).
  Calibration (§3.6 PS-6, v1_cf.json): P(pass | serial, look-ahead 0–50 %, readout flips ≤ 0.10) = 1.00; P(pass | every non-serial class) = 0.00.
- **VB-C1-L** = DEEP-L ∧ PROBE-GAIN-L ∧ SERIAL-ONSET-L ∧ SERIAL-STEP-L ∧ **CF-L** ∧ NO-CLOCK-L ∧ NO-PRECOMPUTE-L, uncensored.

**Labels (descriptive unless stated).** ONESHOT-L; NO-ENTRY (censor, §5.2); LOSES-ONE-PASS := ¬DEEP-L ∧ PROBE-GAIN-L < 0.15; COUNT-KEYED :=
STALL < 0.9; RECOMPUTE-L, ANCHOR-TAGGED, HISTORY-READ, ALL-PERSONS signature (§3.6); PRECOMPUTE-L := NO-PRECOMPUTE-L fails with FD-VALID-L
true; RNN-IN-FUNCTION / HISTORY-USED (the §213 triple and bank ablation); EMPTY-C1-BAND; CF labels (descriptive): CF-CARRY := median
t\* = 1 over decisive cells, CF-RECOMPUTE := median t\* = D, CF-FAST := median (j − t\*) ≥ 2 (more than one hop per pass), LAST-HOP-LATE.

**Depth readings.**
- **DEPTH-L** (A1, descriptive) = mean acc_d2 over D 9–12.
- **A2:** ABSORB-L = min acc_d2 over D 9–12 ≥ 0.95; LENGTH-L = acc_d2 ≥ 0.99 at D 3–8; K′24-STABLE = min acc_k[24] over D 3–8 ≥ 0.97;
  **SD-SELF-L** = ABSORB-L ∧ VB-C1-L (re-read on A2) ∧ LENGTH-L ∧ K′24-STABLE.
  **DEPTH-BEYOND-SELF-L** = mean acc_d2 over D 13–16 ≥ 0.80. **MARGIN-GROWS-L** = mean_{D 13..16} acc_d2(A2) ≥ mean_{D 9..12} acc_d2(A1)
  + 0.10 (the +4-translated A1 curve, computed by the scorer from A1's file before any A2 output). **BEATS-CTRL-L** = mean_{D 13..16}
  acc_d2(A2) − mean_{D 13..16} acc_d2(A2-CTRL) ≥ 0.10. **SD-C2-L** = SD-SELF-L ∧ DEPTH-BEYOND-SELF-L ∧ MARGIN-GROWS-L ∧ BEATS-CTRL-L
  (A2-CTRL missing or censored → SD-C2-L not met).
- Beside every A2 positive: CAUSAL-DEEP-L (§217 rule), MEM status, FALSE-IN, STOP-LAG, MEMO-GAP-L (pool vs fresh, D 9–12), the kept-vs-drawn
  table, ABSORB-L against A2-CTRL's D 9–12 (descriptive), LIT-C2-L (UNINFORMATIVE on SYM, §3.7c).
- **MEM (every Stage A cell):** KEEPS-SELF-STAB-L := STAB_self ≥ 0.5, else **MEM-TRADED-L** (L6: every later stage needs the user's explicit OK).

### 4.2 GSM8K: eval_lvb_gsm.json (B1), all 1,319 test items, CODI's protocol (greedy, max_new 40, last number after "The answer is:", |pred − gold| < 1e-4)
| part | fields | definition |
|---|---|---|
| acc | `acc_fixed14`, `items [{ok, pred_txt}]`, `by_m`, `by_mbin`, `acc_k {K′}` K′ ≥ m_bin only in decisions (K′ < m_bin descriptive), `acc_oracle`, `acc_halt` | headline acc_fixed14 (D-6) |
| codi_pair | `{delta, se_paired, mcnemar {b, c, p_one_sided}, n, by_m, by_parse}` vs CODI-20's per-item file re-scored on the eval card (B0-2), item list asserted by sha | §6 BEATS |
| probe | `probe {t: {acc_by_m, items, curve}}`, t ∈ {0,1,2,3,4,6,8,12,14}; `probe_valid_gsm` = PROBE(14) ≥ acc_fixed14 − 0.03 (m ≥ 3); `plant_gsm {lin, xor}` on slot t's digit rows ≥ 0.90 | fresh coda copy, 2k → 10k steps on 200k train items |
| future | `future {probe ∈ {lin, mlp}: {t: {k: {acc, n}}}}` for t ∈ {0..12}, k ≥ t (**incl. t = k**), read on parsed test items with m_bin ≥ k | exact digits of q_k from concatenated slot rows of V_t |
| vswap | `{arm ∈ {ii_prime, i}: {n_candidates, n_exposed, n_clean, follow, own, neither, by_m, pairs}}`, plus `ii` and `iv` (pure-chain) descriptive | class A on the **exposed** items (§3.8) × (1 + 20 R_x variants); (ii′) = V_1..V_t; (i) = slot t only |
| stall / stepcap / gsmswap | as draft §5.2 | descriptive except NO-CLOCK-GSM |
| cf_gsm | `cf_gsm {j: {n_candidates, n_exposed, n_clean, curve {t: P(y′)}, tstar}, share_window, spearman, last_step, late_fusion_median_tstar, pairs}` | CONSUMPTION-FRONT-GSM (§2.7) on exposed steps × (1 + 40 R_x variants); exposed base steps by j: j 3: 21, j 4: 12, j 5: 1, j 6: 2, j 9: 1 (37 with j ≥ 3; j 2: 68) [DERIVED v1_gsm_exposure.json] |

**Decisive (B1; band m ≥ 3, n 879).**
- **PROBE-GAIN-GSM:** acc_fixed14 − max(PROBE(0), PROBE(1)) ≥ 0.10, requiring PROBE-VALID-GSM and PLANT-GSM.
- **SERIAL-STEP-GSM:** for each of (ii′) and (i): follow ≥ 0.6 ∧ own ≤ 0.3, ≥ 100 exposed clean pairs each (else UNREADABLE).
- **NO-CLOCK-GSM:** recovery ≥ 0.9.
- **NOT-ONE-PASS-CHAIN:** for both the linear and the MLP probe, ¬(FD(1, k) ≥ 0.8·FD(k, k) pooled over k ≥ 3), with FD(k, k) ≥ 0.3 pooled
  (else **UNREADABLE**, never a vacuous pass).
- **CF-GSM (co-primary; A-2):** share of exposed clean pairs (j ∈ {3..m_bin − 1}) with t\* ∈ {j − 1, j} ≥ 0.6 ∧ pooled Spearman(t\*, j) ≥ 0.6
  (numeric answers; RT's 0.6 GSM8K level); ≥ 100 exposed clean pairs (projected ≈ 200 at accuracy 0.27, ρ 0.3, from 37 × 41 = 1,517
  candidates), else UNREADABLE.
- **VB-C1-GSM** = the five (PROBE-GAIN-GSM, SERIAL-STEP-GSM, CF-GSM, NO-CLOCK-GSM, NOT-ONE-PASS-CHAIN), uncensored. Failure readings: PASS1-PRECOMPUTE-GSM := PROBE-GAIN-GSM < 0.05 (the trained-coda clause is
  dropped); ONE-PASS-CHAIN-L := ¬NOT-ONE-PASS-CHAIN (readable); RECOMPUTES-FROM-INPUT-L := (ii′) own ≥ 0.5; STEP-NOT-LEARNED := stepcap < 0.5.
- **Accuracy (vs CODI-20 0.3791, paired):** BEATS-CODI20 := Δ ≥ +0.03 ∧ exact one-sided McNemar p ≤ 0.05; ABOVE-NS := Δ ≥ +0.03 with
  p > 0.05 (descriptive); MATCHES := |Δ| < 0.03; BELOW := Δ ≤ −0.03.

### 4.3 Scorer theory/score_s218.py (Theory writes it; fixture-checked before any output)
- **Pending rule (215-7):** a cell reads **"pending (missing [...])"** while any registered file is missing and reads nothing else first.
  Registered files: A1 — log.jsonl, args.json, eval_lvb.json (every part), the G-L0 gate JSON, DIRECT-SYM's eval_direct.json (for the C1
  band); A2 — the same + pool_meta.json, eval_gate1.json (GATE-1-L, with the choice), label_audit, and A2-CTRL's eval_lvb.json (for SD-C2-L);
  B1 — log.jsonl, args.json, eval_lvb_gsm.json, the interim GO JSON, B0-2's CODI-20 re-score. **Pending never passes any reading.**
  `--final <cell>` (after Theory records the cell's chain as done): a file still missing fails by rule b.
- **Asserts:** card and dtype per file equal the stage's; eval_bs per part equals G-EVMEM's; items_sha per part; pair position asserts;
  exposure counts recomputed from the pairs arrays; SD-C2-L's MARGIN-GROWS bar computed from A1's file only.
- **ARGS-DIFF** vs D3-T's args.json (A1, DIRECT-SYM's own reference for the plain path) with the pre-listed new keys; A2 vs A1's args with
  {init_from, self_pool, p_self, self_kmax, self_micro, lr, steps, c_sched}; A2-CTRL vs A2's args with {p_self = 0, self_pool = none} only.
- **Fixtures (G-L17), each must give the stated label:** VB-C1-L; ONESHOT-L; LOSES-ONE-PASS; COUNT-KEYED; NO-ENTRY; dead; a G-L0-failed band;
  an empty C1 band (D*_fresh 7); (ii′) with 99 exposed clean pairs (UNREADABLE); (i′) per-D exclusion with < 50 pairs (excluded, listed,
  pooled bar still read); PROBE-INVALID (PLANT 0.85); FD-INVALID; **PaR-C** (follow_ii′ 1.0, follow_i′ 0.0 / own 1.0, FD AUROC 0.95 at
  k ≥ 3, STALL 1.0, ON slope 1.0, CF t\* = 1) → **not VB-C1-L, PRECOMPUTE-L, CF-CARRY**; the same with FD 0.5 and (i′) follow 1.0 (the
  probe-undetectable combination) → **not VB-C1-L (CF-L fails)**; **ALL-PERSONS** (PERSON own 1.0, (ii′) follow 1.0, (i′) follow 1.0, FD 0.5,
  PLANT valid, CF t\* = j) → **VB-C1-L + ALL-PERSONS signature**; **ALL-ANCHORS** ((ii′) own 1.0, (i′) own 1.0, RESET recovery 0.5, CF t\* = j) → **CF-L met,
  ¬SERIAL-STEP-L, ANCHOR-TAGGED, not RECOMPUTE-L**; recompute (own 1.0, RESET 1.0) → RECOMPUTE-L; HISTORY-READ; A2 SD-SELF / SD-C2 / MARGIN fail / CTRL
  missing (→ ¬SD-C2-L) / MEM-TRADED / CONTAMINATED / POOL-SHORT / LABEL-BUG (one wrong label); B1 VB-C1-GSM / PASS1-PRECOMPUTE-GSM /
  ONE-PASS-CHAIN-L / FD(k,k) 0.2 (UNREADABLE) / Δ 0.035 with McNemar p 0.07 (→ ABOVE-NS, not BEATS) / BELOW; not admitted (UNREAD);
  **CF counter-models (A-2):** serial (t\* = j) → CF-L; serial with half the hops looked ahead (t\* ∈ {j − 1, j}) and 10 % flips → CF-L;
  PaR-C (t\* = 1) → ¬CF-L, CF-CARRY; recompute (t\* = D) → ¬CF-L, CF-RECOMPUTE; a D-keyed clock (t\* = D − 2) → ¬CF-L (ρ 0);
  pointer doubling (t\* = ⌈j/2⌉) → ¬CF-L, CF-FAST; 99 exposed clean pairs → UNREADABLE; a last-hop crossover at t = D with CF-L met on
  j < D → CF-L ∧ LAST-HOP-LATE; GSM8K CF-GSM with late fusion (t\* = 14 for every j) → ¬CF-GSM, LATE-FUSION-GSM;
  each registered file missing → pending, and with `--final` → failure.
- **Field-path check** (memory rule): every field above is grepped against langvb_eval.py / langvb_gsm_eval.py and the scorer is run on a
  synthetic eval before any data exist.

## 5. Claims, wording, censoring, Goodhart

### 5.1 Claims and pre-stated wording
| claim | wording if met (only this) | never worded as |
|---|---|---|
| VB-C1-L (A1) | "supervised serial latent reachability in a pass-global attention memory, on natural-language premises (ProsQA-SYM v1: person-agnostic cues excluded by construction, single-parent chases gated < 0.6, uBIDIR ½ and UDIST 1 by construction), C1 band D [band] beyond DIRECT-SYM's one-pass boundary, one seed; PREMISE-SWAP and slot-local splices follow ≥ 0.80; each fact consumed at the pass of its hop (CONSUMPTION-FRONT crossover at t ≈ j); no future-distance decodable at k ≥ 3 by PLANT-certified probes" | "graph search"; "no shortcut"; C2; C4 (REACH targets are simulator states); "unsupervised" |
| DEPTH-L (A1) | descriptive: "depth past the gold support before self-training" | C2 |
| SD-SELF-L (A2) | "latent reachability depth extended from 8 to 12 by training on the model's own SOUND-CLOSURE-verified chains (labels gold-equal by construction, π = 1; no gold state beyond D 8; labeller [L-ROLL / L-RR]; in support)" | C2, C4 |
| SD-C2-L (A2) | "depth extrapolation past the self-labelled range (D 13–16, untouched SYM v1 items; margin grew vs the +4-translated A1 curve and beat the matched plain fine-tune by ≥ 0.10; C2 proxy; one lineage: an instance)", with MEM status, CAUSAL-DEEP-L and LIT-C2-L UNINFORMATIVE on the same line | literal C2 |
| VB-C1-GSM (B1) | "serial latent arithmetic in a pass-global attention memory on GSM8K (m ≥ 3), with CoT-derived, calculator-binarised per-pass supervision (C4 qualified: ≈ 13.5 % of value targets are sub-step values absent from the CoT; pointers are value-matched annotations)" | "C4-legal distillation" (unless Audit rules otherwise before registration); C2; "beyond text" |
| BEATS / MATCHES / BELOW-CODI20 | "this B1 model's accuracy differs from this CODI-20 model's by Δ on the same 1,319 items (paired; item-sampling SE [se], McNemar p [p]); one seed each, matched examples (7.66M), a system comparison (full fine-tune + dense per-step supervision vs LoRA CODI)" — **what it can claim:** these two trained models differ on these items; **what it cannot:** that the recipe beats CODI (seed variance σ ≈ 0.015 per model unmeasured, so P(BEATS \| equal recipes) ≈ 0.12) | "beats CODI" as a recipe claim; C3 |
| PASS1-PRECOMPUTE-GSM / ONE-PASS-CHAIN-L | "the §208 / §204 failure recurs in VERSION-BANK form" | — |
| A0-1 | 191-3's wording (§2.8) | an explanation of §191 |

- **Constraint check (§213):** RNN-IN-FUNCTION (SLVA one-hot on the newest version) is reported and **not credited as attention-memory use**;
  HISTORY-USED beside it.
- **Chance floors (RT F2-3, re-measured):** person-agnostic rules ½ exactly; full searches (directed or undirected) 1.0; single-parent chases
  0.54–0.56; LA(2–4) 0.60–1.00. Deep readings quote PROBE(1) against both ½ and the chase rows.
- **Plainly (L5):** the expected value of this design is mainly **mechanism information** (which of serial / precompute / recompute / one-pass
  a VERSION-BANK model uses on language). **A null is the likeliest outcome** (P(no decisive C1 positive) ≈ 0.94); its likeliest forms are
  ProsQA NO-ENTRY / LOSES-ONE-PASS / PRECOMPUTE-L / CF-CARRY / CF-RECOMPUTE, each informative only through the detectors above.

### 5.2 Censoring (rule b: each fails every reading of that cell; never-reached events are failures)
- **Dead run** (log short of the registered steps; a guard stop counts).
- **NO-ENTRY:** ProsQA `plain_cand_ce` never ≤ 0.35 in two consecutive 1k bins to the end; GSM8K Aug-valid accuracy ≤ 0.15 at every check.
- **CONTAMINATED:** a pool / test / GATE signature hit (role-symmetric), a gold key in a pool record, self_reuse > 0.
- **POOL-SHORT / NOT ADMITTED / B or A2 not GO:** the cell never runs; its readings read **UNREAD = not met** (never "dead", never "pending"), and
  the user is told (RT §5.3(c)).
- **ARGS-DIFF** beyond the pre-listed keys.
- **LABEL-BUG (A2):** any wrong label in GATE-1-L or LABEL-AUDIT-L (impossible by S-L′, so a code fault): A2 voided and re-gated.
- **UNREADABLE (= not met):** < 100 exposed clean pairs in a decisive arm (incl. CF-L / CF-GSM); < 2,000 STALL items; PROBE-INVALID (PLANT or final-state);
  FD-INVALID; FD(k, k) < 0.3 (GSM8K); a G-L0-failed band; an empty C1 band (< 2 depths); eval card / dtype / eval_bs mismatch; items_sha
  mismatch. Per-D values below 50 exposed clean pairs are excluded and listed (A-1), never passed.
- **MEM-TRADED-L** is not a censor; it routes every later stage to the user's explicit OK (L6).
- Never-reached onsets fail; never-halting acc_halt items count wrong; a never-crossed crossing reads "> 16".

### 5.3 Goodhart routes (targets and readings new in LANG-VB v1)
| route | status | read by |
|---|---|---|
| G1 pass-1 precompute (global) | open (L1-L bounds depth, not width) | PROBE(1) with PLANT, PROBE-GAIN-L, ONESHOT-L |
| G1′ precompute-and-carry / reveal | open, zero-loss [PROVED] | **CONSUMPTION-FRONT (decode-free: t\* = 1)**, (i′) co-primary, FD-MIRROR (decisive), PLANT; the probe-undetectable combination of §3.5 is caught by CF |
| G2 recompute from V_0 | open | (ii′) own + RESET (RECOMPUTE-L) |
| G2′ person-independent serial state (ALL-ANCHORS) | open, a power loss for SERIAL-STEP-L [PROVED] | ANCHOR-TAGGED (descriptive); CF-L reads it as serial |
| G3 coda solving | closed for question access (V_0 excluded; decoder path question-free) | PROBE vs ACC |
| G4 data cues | person-agnostic cues ½ exactly (§3.3a); single-parent chases gated; lookahead chases reported | G-L0 panel per band and per stream |
| G5 clock / count keying | open in principle (S0-L) | STALL; DEPTH-BEYOND |
| G6 hidden over-reach | decodable over-reach penalised by R2; hidden not | FD k = 2 (descriptive), ON slope |
| G7 over-marking on self items | open (no OUT terms) | FALSE-IN after A2 |
| G8 sabotage | no incentive [PROVED, VB §5.6, imported] | — |
| G9 history traded for depth | expected (2 / 2 nested rounds) | KEEPS-SELF-STAB-L → MEM-TRADED-L (user's OK) |
| G10 adaptive-controller contamination (214-5) | closed by design: CurGate reads `plain_cand_ce` of plain gold items only (asserted); GSM8K fixed schedule | G-L7 |
| G11 gate CE at chance (RT F5-2) | closed: two-candidate CE, init fixture 0.693 | G-L7 |
| G12 length as a depth proxy (T grows with D) | cannot pick the option (σ); could time a stop | STALL; T-band rows; DEPTH-BEYOND |
| G13 name memorisation | per-draw names from 300 / 40; fresh items | MEMO-GAP-L |
| G14 labeller cost vs quality (N9) | the tie margin favours the cheaper L-ROLL by ≤ 0.05 yield | both chains' GATE-1-L outputs stored |
| G15 exposure gaming of the slot-local arm | L_t is defined from gold distances, not from the model's marks; the scorer recomputes exposure from the pairs | §4.3 asserts |
| G16 CF donor out of support | the single-fact rewire can leave a_j^R without a parent (x′ slightly outside the generator's support); a model that fails on x′ makes the pair unclean, not a pass | clean-pair filter (x′ answered y′ at K′ = D); the share of clean pairs per (D, j) reported |

## 6. OC (one seed per cell; realised spend at registration $0) [arithmetic DERIVED v1_checks/v1_oc.py; every prior ASSUMED]

**Base (L5): RT's shaded "with fixes" column, recomputed exactly** (v1_oc.py re-runs rt_lang/oc_shade.py's mixture and asserts its JSON):
A1 P(entry) 0.55 × P(DEEP | entry) 0.60; given DEEP the mechanisms serial question-conditional 0.35 / ALL-PERSONS 0.15 / precompute-carry or
backward chase 0.30 / recompute 0.10 / one-pass at D 5–6 0.10, with P(VB-C1-L | mechanism) 0.50 / 0.45 / 0.02 / 0.02 / 0.10 → **P(VB-C1-L)
0.086, | no serial use 0.036, | serial use 0.485, false share 0.07**. GSM8K (ungated) 0.027 (| no serial use 0.008). SD-C2-L 0.007.

**L1–L4 and A-2 factors applied (each [ASSUMED], stated):**
- (a) **D-13 gate:** B1 launches only after VB-C1-L, so P(B1 launched) = P(VB-C1-L); P(VB-C1-GSM | launched) = 0.10 (RT's own figure).
- (b) **N8:** P(≥ 1 decisive C1) = P(VB-C1-L) + P(SD-C2-L ∧ ¬VB-C1-L on A1) (A2 runs on DEEP ∧ NO-CLOCK, P ≈ 0.15, RT F3-4; the second term
  ≈ 0.001). RT's 0.095 is the expected **count** of decisive positives, not P(≥ 1).
- (c) **DIRECT-SYM band:** P(C1 band < 2 depths) = P(D*_fresh ≥ 7) = 0.05.
- (d) **ALL-ANCHORS (N1):** a third of RT's ALL-PERSONS mass (0.05 of 0.15) is anchor-tagged and passes SERIAL-STEP-L with 0.02, not 0.45.
- (e) **CONSUMPTION-FRONT (A-2) in the conjunction:** P(CF-L | class) from the registered rule's Monte Carlo (v1_cf.json, 150 replicates,
  100 clean pairs per (D, j) cell, fewer than the projected ≈ 324): serial (with 0 or 50 % one-hop look-ahead, per-hybrid readout flips
  ≤ 0.10) **1.00**; precompute-and-carry, recompute, D-keyed and constant clocks, pointer doubling, partial one-pass (c₀ 3) **0.00** at every
  flip rate ≤ 0.15. Used as 0.9 (serial) / 0.02 (non-serial) / 0.2 (one-pass at D 5–6 with serial use deeper) [ASSUMED margins].
  CF is decode-free, so the no-effect bound no longer rests only on probe decodability (§3.5).

| reading | n (expected) | P(pass \| no effect) | P(pass \| effect) | headline P(pass) | "no effect" means |
|---|---|---|---|---|---|
| **VB-C1-L** | composite | **0.004** (0.036 without CF-L; CF-L is decode-free) | **0.40** (serial use; 0.44 without (e), 0.485 without (d)) | **0.063** (0.075 without CF-L; RT-exact 0.086; decoy sensitivity 0.064) | no serial use |
| **CF-L** (share t\* ∈ {j−1, j} ≥ 0.75 ∧ within-D Spearman ≥ 0.60) | ≈ 324 clean pairs per (D, j), 15 cells | **0.00** (MC; precompute t\* = 1, recompute t\* = D, clocks, doubling, partial one-pass; flips ≤ 0.15) | **1.00** (MC; serial, look-ahead 0–50 %, flips ≤ 0.10); 0.00 at flips 0.15 | — | no hop-indexed consumption |
| PROBE-GAIN-L ≥ 0.35 | ≈ 1,000 per C1 depth | ≈ 0 (gain 0 ± 0.011) | ≈ 0.8 | — | pass-1 precompute |
| SERIAL-ONSET-L | 11 probe points × 5 D | ≤ 0.05 (flat ON) | ≈ 0.7 | — | depth not tied to passes |
| SERIAL-STEP-L (ii′) and (i′) ≥ 0.80 | ≥ 800 exposed clean pairs per D if DEEP | ≈ 0 (random disruption 0.5 ± 0.016; recompute ≈ 0); PaR-C passes (ii′) but fails (i′) | ≈ 0.85–0.9 (name-tagged serial; nested follow_i 0.996–1.000); ≈ 0.02 for ALL-ANCHORS | — | no stored-state use |
| NO-CLOCK-L | ≥ 2,000 | ≤ 0.05 (count-keyed N3: 0.499) | ≈ 0.9 | — | pass-count clock |
| NO-PRECOMPUTE-L (AUROC ≤ 0.65, k ≥ 3) | ≈ 8,800 slots per class per (t, k) | fails ≈ 1 under decodable precompute (AUROC ≥ 0.9; SE 0.004) | passes ≈ 1 when no R-information is at the far slots (null ½ exactly) | — | (reversed: "effect" = no precompute) |
| PROBE-VALID / FD-VALID (PLANT) | as the probe | — (a probe-power certificate) | ≈ 0.95 | — | — |
| DEEP-L | 5,000 | — | — | ≈ 0.33 | competence |
| A2 runs (DEEP ∧ NO-CLOCK ∧ GATE-1-L) | — | — | — | ≈ 0.15 (RT F3-4: the modal A2 outcome is "no A2", a pre-stated C2-negative) | — |
| ABSORB-L | 4,000 | ≈ 0.05 vs A1's DEPTH-L; read beside A2-CTRL | ≈ 0.65 \| A2 runs | — | self items not absorbed |
| **SD-C2-L** (with MARGIN-GROWS-L + BEATS-CTRL-L) | 4,000 + A2-CTRL | **≈ 0.02 \| H_shift** (lineage σ; RT) | H_grow ≈ 0.3, H_full ≈ 0.7 | **0.005** | exact +4 translation |
| KEEPS-SELF-STAB-L | — | — | P(keeps ≥ 0.5) ≈ 0.2–0.5 (nested 0 / 2 kept after self rounds) | — | — |
| **VB-C1-GSM** (B1 launched) | 879 (m ≥ 3); ≥ 100 exposed clean pairs per arm | **≤ 0.008** (RT's value before CF-GSM; CF-GSM lowers it) | ≈ 0.35 (serial step) | **≤ 0.10 \| launched; ≈ 0.006 unconditional** (0.027 if B1 ran ungated) | one-pass / recompute / carry / late fusion |
| BEATS-CODI20 (Δ ≥ 0.03 ∧ McNemar p ≤ 0.05) | 1,319 paired | **0.016 item-only; 0.119 with seed variance** σ 0.015 per model | 0.984 / 0.881 at true Δ +0.06 | ≈ 0.04 \| launched | equal accuracy |
| A0-1 ANTI-SIGNATURE (acc(anti) ≤ 0.40, n 78) | 78 | ≈ 0 (binomial at true acc 0.95) | — | — | no cue use |
| **P(≥ 1 decisive C1 positive)** | — | — | — | **0.064** (0.076 without CF-L; RT-exact 0.087); E[count] ≈ 0.075 | — |

- **What a single-seed comparison can and cannot claim (L5):** BEATS-CODI20 is an item-level paired test of two trained models (McNemar
  adds nothing at paired SE 0.014: Δ 0.03 already gives p 0.016). It cannot say anything about recipe-level superiority: with σ_seed ≈ 0.015
  per model the false-BEATS rate under equal recipes is 0.12. A recipe claim needs ≥ 3 seeds per arm and the user's OK.
- **A positive is strong evidence** (false share ≈ 0.01 for VB-C1-L with CF-L; 0.08 without it). **A null is weak** and is the likeliest
  outcome (≈ 0.94); its value is the mechanism label it carries (CF's curve labels it decode-free: CF-CARRY / CF-RECOMPUTE / CF-FAST).
- **Rule check (OC block + censoring rule):** every registered reading carries P(pass | no effect) and P(pass | effect) at one seed; never-reached
  events count as failures (§5.2); realised $ at registration: **$0**.

## 7. Gates (all $0 and local unless marked; each writes its own gate JSON as it finishes; all green before any spend on the cell they guard; Audit reviews)

| id | gate | pass rule |
|---|---|---|
| **A0-1** | §191 anchor-cue contribution (§2.8), descriptive, run first | result to Theory / Audit; gates nothing |
| **G-L0 SYM** | boxes/prosqa_sym.py on **5,000 draws per band per stream** (training 22100, pool 22300, GATE 22400, test 22200 = its 1,000 items + 4,000 more of the same law): invariants (§2.2 item 8), names (PERSON-40 one token each, NONCE-300 distinct finals, shas), T ≤ 1,000, the person-agnostic panel (6 cues × all levels), the **gated chase set (first / random / max-outdegree / LA(1))**, uBIDIR(r) for 2r < D, UDIST, beyond-D share; the test file's own 1,000-item panel reported | 0 invariant failures; a band is usable iff **every gated cell < 0.6** in every stream (binding addition 1); uBIDIR = 0.5 and beyond-D = 0 exactly (else a code bug: stop) |
| G-L1 identity | (a) all-position slots, K = 1, bank masked == GPT-2 (fp64 ≤ 1e-9); (b) LangSLVA with S = all positions on nested inputs == D3-T's SLVA rollout bitwise (fp32) | equal |
| G-L1b init drift + binding | 2k local steps (memory-capped) on SYM v1 D 3–4 | drift below its init value; `plain_cand_ce` falls below 0.60 (descriptive preview of name binding on NONCE-300; read by no rule) |
| G-L2 gradflow | scratch/gradflow_check.py pattern on A1 and A2 batches: nonzero grad from R1, R2-changed, R2-unchanged; ∂L/∂bank K/V ≠ 0; ∂L_ans/∂V_1 ≠ 0 with the SLVA input stop-gradiented; decorators by ast | all nonzero |
| G-L3 targets | REACH re-render by an independent BFS on 100k items; S-L′ fuzz (random marks incl. injected unsound ones; M′ ⊆ R_t, label o_R, s ≥ D) on 100k | 0 diffs; 0 violations |
| G-L4 pairs | PREMISE-SWAP pair maker on 5,000 pairs per D 4–8: token length equal, slot indices equal, donor invariants (answer o_P); exposure counts per arm and D (§4.1 table) | 100 % identity; counts written to the gate JSON |
| **G-L4b CF pairs** | the CF donor maker on 1,000 items per D 4–8: flip validity by BFS on x′, token length and first-mention identity asserted per pair; exposure per (D, j) within ± 5 points of v1_cf.json; the version-0 switch hook equals a full x′ run at t = 0 and a full x run at t = D (bitwise, fp32) on 64 pairs | pass |
| G-L5 probes | the probe / PLANT / FD pipeline on the **untrained** LangSLVA (pretrained GPT-2, step-0 heads): PLANT lin and XOR ≥ 0.90 (the probe can find a planted code at ε), PROBE(0) = 0.50 ± 0.03 (the §3.5 theorem), FD-MIRROR AUROC at k ≥ 3 reported | PLANT and PROBE(0) pass |
| G-L6 plain path | D3-T constructs + 1 step; nested vbank_eval acc on 64 items bitwise = eval_vb | equal |
| G-L7 controller | unit test: self / augmented batches never reach `GATE.update`; `plain_cand_ce` at init = 0.693 ± 0.05 on 2,000 plain items | pass |
| G-L8 MEM | mem_check at the forced worst shapes per batch type (A1: 64 longest D 8 items, K 12; A2 self: 64 longest D 12 records, K 16; DIRECT-SYM; B1: K 16) → micro / self_micro = the smallest of {1, 2, 4, 8} with peak ≤ 21.0 GiB | pinned in args |
| G-EVMEM | on the eval card: the deepest chunk of each part (D 16, K′ 24; probes; hist24) → eval_bs per part, output to scratch/_tmp | peak ≤ 7.0 GiB |
| G-L9 SMOKE-AT-SCALE | the generator at full rate (timed, 100k draws), the exclusion set at full size, the trainer's loader over the full stream, the pool builder on 20k draws (timed → the §8 labelling projection) | completes; projections logged |
| G-L10 OFF-HOST + RESUME | full-state mirror every 2k for every cloud cell, sha-checked after the first copy; kill a smoke at step 300 and resume: losses 301–400 bitwise = uninterrupted (same card) | pass |
| G-L11 ARGS-DIFF | printed in each launch log vs the §4.3 references | only the listed keys differ |
| G-L12 G-TIME (admission) | per host: 20 warm-up + 300 timed steps **drawn from the registered law at the final curriculum stage** (A2: the registered self / plain coin sequence) → projected $ at the host's price | ≤ the §8 admission bound; else no launch on that host (at most 2 more measured refusals per cell; BAD hosts count toward the guard, not a try) |
| G-L13 DIFF REVIEW (Audit) | decorators by ast, return shapes, the bank cache's gradient, eval hooks (slot-set splice, reset, the CF version-0 switch), the rng_s order, CurGate exclusion, SOUND-CLOSURE parsing the original text | approved |
| G-L14 LAUNCH | RTX 3090 only (card per cell recorded); drain waiters armed; GUARD_STOP per cell and family; c logged; no module edit while any deploy is in flight; requeue watchers checked | — |
| **G-L15 GATE-1-L** (A2 only) | §2.4 on 2,000 GATE items per D 9–12, both chains, bf16 autocast (the pool's setting), the eval card | π_label = 1.000; yield ≥ 0.2 at every D (else no A2: a pre-stated C2-negative); choice by the §2.4 rule; per-item store |
| G-L16 A2 GO | §2.9 item 3 | — |
| G-L17 FIXTURES | score_s218 fixtures (§4.3), incl. the PaR-C (also probe-undetectable), ALL-PERSONS, ALL-ANCHORS and the CF counter-models | all pass before any output |
| G-L18 FIELD PATHS | grep of every scorer field in the eval modules + a synthetic-eval run | pass |
| G-L19 A2-CTRL identity | plain_hash at steps 1..200 equal between A2 and A2-CTRL (draws_log) | equal |
| **G-L20 POOL** | pool_meta: kept per D ≥ 106,000; total ≥ 1.1 × the seed's replayed demand (seed 1: 383,488); key-set assert; signature exclusion vs test ∪ GATE ∪ probe streams = ∅; the trainer's rng_s replay equals v1_pool.py's count before launch | pass (else POOL-SHORT or a code bug) |
| G-L21 RNG ORDER | the LANG trainer's coin / U draws on a 200-step smoke equal vbank.draw_step's sequence for the same seed | equal |
| **B0-1** | DECOMP over all 384,622 train lines: coverage, ambiguity, per-term mask rates, the m_bin table, m_bin > 12 and literal > 14 masks | logged; coverage < 0.95 → Theory reviews |
| **B0-2** | CODI-20 re-score on the Stage B eval card from trainable.pt | per item == gsm_codi_20ep_s1/eval_items.json (0.3791) |
| B0-3 | B-STEP smoke (local, 2k steps, m_bin ≤ 2) → descriptive learnability preview and the class-A clean-pair projection | read by no rule; projected pairs logged |
| B0-4 | the G-L1 / L2 / L6 / L8 / L10 / L11 / L12 / L13 analogues for the GSM8K path | as Stage A |
| **B0-5** | the class-A pair maker: exposure filter (§3.8) reproduces 95 exposed items (94 at m ≥ 3) and the CF step counts by j (j 3: 21, j 4: 12, j 5: 1, j 6: 2, j 9: 1); the token-length-preserving literal redraw; R_x variants keep exposure | equal |
| **B0-6 interim GO** | §2.7 at 24k, local, from the mirror | GO rule as stated |

## 8. Cost, caps, admission, guards, cut order [DERIVED v1_mem.json, v1_pool.json; price band $0.17–0.26 / h ASSUMED, $0.21 central]

| cell | where | hours law-matched / forced worst | projected $ (central) | admission (G-L12, law-matched) | in-flight guard (realised incl. refused / BAD hosts) | status |
|---|---|---|---|---|---|---|
| DIRECT-SYM `lvb_direct_s1` (24k) | 3090, micro 2 | 12.5 / 16.0 | 2.12–3.24 (2.61) | ≤ $3.6 | $3.9 | required |
| A1 `lvb_sym_s1` (24k) | 3090, micro 4 | 14.4 / 19.2 | 2.45–3.75 (3.03) | ≤ $4.3 | $4.7 | required |
| A2 `lvb_self_s1` (12k; 5,992 self steps) | 3090, micro 4 | 8.1 / 9.8 | 1.38–2.12 (1.71) | ≤ $2.5 | $2.8 | on A2 GO |
| A2-CTRL `lvb_ctrl_s1` (12k) | 3090, micro 4 | 7.2 / 9.6 | 1.23–1.88 (1.52) | ≤ $2.2 | $2.5 | required with A2 |
| B1 `lvb_gsm_s1` (120k; interim at 24k ≈ $0.6–1.2) | 3090, micro 1 | 18.7–23.3 / 25.0 | 3.18–6.06 (4.41) | ≤ $6.5 | $7.0 | on B GO (VB-C1-L) |
| pool labelling on a host (only if the local plan does not fit) | 3090 bf16 | L-RR 9–30 by yield 0.5 → 0.2; L-ROLL 1.7–4.3 | L-RR 1.9–6.3; L-ROLL 0.4–0.9 | ≤ $3.0 | $3.3 | conditional |
| A-TXT `lvb_txt_s1` (12k; optional, C3 on SYM) | 3090, micro 2 | 6.8 / 8.6 | 1.15–1.77 (1.43) | ≤ $2.0 | $2.2 | only from the remainder (below) |
| A0 / B0 gates, labelling on local cards, all evals, A0-1 | local | — | $0 | — | — | — |

- **Guards sum $24.2** (DIRECT 3.9 + A1 4.7 + A2 2.8 + A2-CTRL 2.5 + B1 7.0 + host labelling 3.3) **≤ the $25 family cap** (GUARD_STOP on the
  family); admissions sum $22.1. A-TXT runs only if realised spend + the guards of cells still eligible + $2.2 ≤ $25 at its launch.
- **Re-estimated totals:** every required cell run (DIRECT + A1 + A2 + A2-CTRL + B1): **$10.4–17.1, central $13.3** (+10 % bad hosts:
  $11.4–18.8); + host labelling ≤ $3.0 and A-TXT ≤ $1.8 → ≤ $23.6. **Nothing projects above $25**, so no cut is planned.
- **Expected realised spend ≈ $7** (Stage A base DIRECT + A1 ≈ $5.6 central, ≈ $6.2 with hosts; A2 + A2-CTRL with P ≈ 0.15; B1 with P ≈ 0.06).
  The likeliest bill is Stage A base alone. RT's realistic range ($8.5–16) is superseded: v1's padded batch shapes (T_pad 748, |S|_pad 83)
  and DIRECT-SYM at A1's exposure raise A1 / DIRECT-SYM; B1 is gated.
- **Cut order (L7)**, if a G-L12 projection or the re-estimated family total exceeds the remainder of $25: **(1) B1** (Stage B deferred to the
  user with its numbers), **(2) A-TXT**, **(3) host labelling** (→ local only; longer wall-clock, never fewer items). The required A cells
  (DIRECT-SYM, A1, A2, A2-CTRL) are never cut or shortened: if they do not fit, no launch, Audit and the user decide. Never split the design;
  never reduce registered steps. A cell no host admits is not run; its readings read UNREAD; the user is told.
- **Pool labelling plan (L2, RT F3-5 / F6-4).** After A1 lands and GATE-1-L chooses the chain, G-L9's timing on 20k drawn items × GATE-1-L's
  yield gives the projection. **Local plan:** RTX 3070 (D 9–10) ∥ RTX 4060 (D 11–12), bf16 autocast (the GATE-1-L setting), the interleaved
  order assembled after both finish. Projected wall-clock [DERIVED v1_pool.json, 14 TFLOP/s bf16 per card ASSUMED]: **L-RR 6.6–8.6 h at
  yield 0.5, 11–14 h at 0.3, 16–21 h at 0.2; L-ROLL 1.2–3.1 h.** Local iff the projected wall-clock ≤ 24 h (Experiments' queue call: it
  displaces no registered eval by > 24 h); else the 3090 host above (its $3.0 admission fits L-RR only at yield ≥ ≈ 0.45, $2.1–2.8, and L-ROLL at any yield, $0.4–0.9); if neither fits,
  A2 is not launched (UNREAD) and the user is told. GATE-1-L itself is ≈ 0.2 GPU-h.
- **Local eval time (RTX 4060, fp32) [ASSUMED from the FLOP model]:** ≈ 6–8 h per Stage A model (probes / plants / FD at 10k steps ≈ 4–5 h; acc,
  swaps, RESET, STALL, hist24, causal_deep ≈ 1–2 h; CONSUMPTION-FRONT ≈ 0.3–0.5 h: ≈ 4,900 clean pairs × (D + 1) switched rollouts);
  DIRECT-SYM ≈ 0.2 h; B1 ≈ 6–8 h + CF-GSM ≈ 0.5 h + the interim ≈ 1 h + B0-2.
- **Wall-clock:** A0 engineering + gates; DIRECT-SYM ∥ A1 ≈ 15–20 h; A1 evals ≈ 7 h; labelling 2–21 h; A2 ∥ A2-CTRL ≈ 8–10 h; their evals
  ≈ 14 h; B1 ≈ 19–25 h + 8 h of evals. Every cloud cell > 8 h: off-host full-state mirror every 2k + the tested resume (G-L10; user rule
  2026-10-15).
- **Retention:** keep model.pt of DIRECT-SYM, A1 (labeller), A2, A2-CTRL, B1 until every registered reading is read; keep evals, logs, args,
  pool_meta, gate JSONs (per-item records); delete checkpoints and mirrors at landing; delete the pool after LABEL-AUDIT-L / MEMO-GAP-L are
  scored (pool_meta + seeds + builder commit regenerate it).
- **Credit:** flag if account credit < $30 at the first launch (memory rule); Experiments re-reads it.

## 9. Open items (for Theory / Audit before registration)

1. **DECOMP's C4 status (Audit; L2 / RT F4-1).** v1 words Stage B as "C4 qualified, not C4-legal as a whole" (≈ 13.5 % of value targets are
   calculator-only sub-steps; pointers are value-matched annotations). Audit's alternative is the strict variant (mask the sub-step value
   terms; pointers stay annotations). Decide before registration; B1's args change by one flag if strict.
2. **ALL-ANCHORS (N1).** SERIAL-STEP-L has a proved power hole: an anchor-tagged serial state reads own on both decisive arms. v1 keeps the
   decisive conjunction as L1 specifies and adds RESET-t descriptively. Theory may decide whether "(ii′) own ∧ RESET drop ∧ NO-PRECOMPUTE-L"
   should become a second, pre-stated route to SERIAL-STEP-L (not adopted here: a clock-keyed precompute also drops under RESET).
3. **Labeller tie rule (N9)**: L-RR only if its yield beats L-ROLL's by ≥ 0.05 at D 11–12. Confirm, or keep the draft's "ties to L-RR".
4. **Decoy law** was tuned on the prototype (RT's law read 0.60–0.67). If boxes/prosqa_sym.py's G-L0 reads any gated cell ≥ 0.6, that band is
   unusable; a law change needs a disclosed note before any training output (never after).
5. **PLANT strength:** ε = the 50th-PC std and directions in PCs 40–60 are uncalibrated design choices; G-L5 only shows the probe finds the
   plant on the untrained model. A stricter plant (100th PC) is Theory's option before registration.
6. **Probe budget:** 10k-step probes on ≥ 200k items (RT F1-1e) cost ≈ 4–5 h per model on the 4060. Any reduction needs a note before any
   probe output (PLANT still certifies what it can).
7. **Eval cards (N10):** both stages on the RTX 4060 (fallback: the 3070, whole stage); labelling uses both cards. Experiments confirms the
   queue (§217 uses the 4060).
8. **L6 reading:** MEM-TRADED-L on **any** LANG cell (A1 included) routes every later stage to the user's explicit OK. If Theory meant A2 only,
   one line in §2.9 / §4.1 changes.
9. **DIRECT-SYM runs in parallel with A1** (the C1 band is a formula on DIRECT-SYM's file; no A1 reading is scored before it exists). If
   Theory prefers DIRECT-SYM first (so that D*_fresh ≥ 6 can re-plan A1 before spend), A1 starts ≈ 15 h later; nothing else changes.
10. L-RR re-roots point "R is a f" at concepts that have parents, which A1's direct concepts rarely do: a residual distribution shift that
    GATE-1-L measures and nothing designs for.
11. `--final` trigger: the Theory note that records each cell's chain as done (name it in the registration commit).
12. **CONSUMPTION-FRONT thresholds (A-2):** v1 registers the derived window {j − 1, j}, j 3..D−1, share ≥ 0.75, within-D ρ ≥ 0.60 (calibrated:
    serial power 1.00 up to 10 % flips with 50 % look-ahead; 0.00 for every non-serial class). Audit proposed ±1 / 0.8 / 0.8, which loses
    the look-ahead serial model at 5 % flips (P 0.007). Audit and Theory confirm before registration. CF-GSM's 0.6 / 0.6 is not separately
    calibrated (most exposed GSM8K steps have j 3–4, so its Spearman is pooled, not within m_bin).
13. Not done here: boxes / theory code; score_s218 fixtures; the arXiv IDs in LIT notes other than 2602.00449 / 2602.08783 (verified by
    Literature, addendum 85f6a055) are unchanged from the draft and not re-checked.

## 10. Verification log (this revision; all $0, CPU, OMP_NUM_THREADS=1, nice 19, `python -B`; outputs in research7/v1_checks/)

| script | output | checks |
|---|---|---|
| v1_names.py | v1_names.json | PERSON-40 (all one GPT-2 token with a leading space; 11 are two tokens without it: hence N3); NONCE-300: 300 two-token names, pairwise-distinct final tokens, finals disjoint from every non-final, person and template token, a banned-substring filter; sha256 recorded (asserts) |
| v1_sym.py (main: `--n 5000 --D 3..16 --rr 4 --rr_small 5 --pextra 0.1 --cap 2D+6`, seed 24000) | v1_sym.json, v1_sym.log | the §2.2 table; 0 / 75,000 invariant failures; heur_panel.panel_for imported read-only (the draft's §59-validated panel) |
| v1_sym.py (`--lead_space 1`, D 4–8, n 1,000, seed 24100) | v1_sym_lead.json | PREMISE-SWAP position identity 5,000 / 5,000 (vs 2.3–3.0 % mismatches without the leading space in the main run); exposure table §4.1 |
| decoy tuning (console, n 300–600 per D at D 3 / 5 / 8 / 12 / 16; outputs not kept) | — | gated-chase max per band: with the draft's extra-edge probability 0.3, six decoy laws (incl. RT's U{2..4}, k in U{2..D} / top half / top 2–3 levels) read 0.58–0.69; extra-edge 0.1 or 0.0 with 3–4 decoys on the top levels 0.57–0.65; fixed decoys per top level 0.54–0.67 (D 3 high); **round-robin 4 per world (5 at D ≤ 4), extra-edge 0.1: 0.54–0.57 → chosen** (5 everywhere: 0.53–0.56 but T + 50). Cause found by an exact random-chase DP on 200 items per D: the draft's within-world extra edges give R-leading alternative parents, so P(reach R) stayed 0.28–0.29 with 3–4 decoys |
| v1_pool.py | v1_pool.json | **replay of rng_s asserted = §216's measured 5,992 self steps / 383,488 records (seed 1)**; seeds 2–21: 5,899–6,083; 106k / D spare 10,128; exhaustion z 11.4; draft pool exhausts at step 7,500; labelling FLOPs from v1 T / \|S\| / chunks |
| v1_mem.py | v1_mem.json | batch-max shapes from 120 batches of 64 v1 items; RT's memory / FLOP form, re-calibrated (D3-T reference 0.311 s → scale 1.351); micro choice at 21 GiB; hours and $ per cell |
| v1_gsm_exposure.py | v1_gsm_exposure.json | class-A exposure on the 1,319 test items with PROOFS' gsm_check parser (imported read-only): 95 exposed items (94 at m ≥ 3), 105 exposed steps |
| v1_oc.py | v1_oc.json | **RT's oc_shade.json reproduced exactly (asserted: VB-C1-L current / fixed, VB-C1-GSM fixed, BEATS, SD-C2-L)**; L1–L4 factors (a)–(d) and A-2's (e); sampling rows; clean-pair projections |
| v1_cf.py | v1_cf.json, v1_cf.log | CF exposure per (D, j) on 1,000 items per D 4–8 (seed 24300; single-fact rewire and the double-swap variant); the pass-rule Monte Carlo for three rules × 7 model classes × flips 0–0.15 (registered rule run separately: `CF_REG=1`) |
| v1_gsm_exposure.py (re-run) | v1_gsm_exposure.json | + exposed steps by j for CF-GSM |

- **Raw files read (explicit paths only):** eval_vb.json of s213_d3t_s1, s215_ftself_s1, s216_r2_s1 (vswap follow_i / class_B, cited in L1);
  bench/data/gsm8k_aug/test.txt (via the parser); boxes/prosqa168s2.py (ENT17 / CON38 via ast); boxes/vbank.py l.520–575, 725–815.
- **THEORY.md read:** 191-2, 191-3, 214-5, 215-7, §216 registration, the §216 s1 round-2 result, 216-6/7/8/9, §217 + 217-1/217-2.
- **No repo write:** `find D:/Latent/boxes/*.py D:/Latent/theory D:/Latent/docs/research_lang_2026-10-18 D:/Latent/docs/THEORY.md -newer
  v1_checks/v1_names.py` printed nothing after all runs; no `__pycache__` anywhere under research7 or the LANG docs folder.
- **Still unbuilt (flagged):** every boxes module in §2.10, score_s218 and its fixtures, the queue scripts. All memory / time / cost figures
  are model extrapolations; G-L8 / G-L12 / G-EVMEM decide.

*v1 written 2026-10-18 13:25–14:11 EDT (Opus design-revision subagent for Theory; $0; nothing committed, registered or launched).*
