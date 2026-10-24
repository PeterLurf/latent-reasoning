# Digest of THEORY.md sections 171-228 (research history, condensed)

Scope: D:\Latent\docs\THEORY.md §171 through §228 plus their Notes (status table, result blocks, notes on archived sections). Written Fri 2026-10-23 13:28 EDT. §228 is still running (pilots). Verdict words are quoted from the text; "unclear in text" where I could not confirm. §180, §195, §197 were never registered. Goals: C1 latents do the work, C2 more thinking helps past training, C3 competitive with text CoT at matched compute, C4 trainable from scalable supervision, C5 decides how long to think, C6 stable, C7 readable.

| § | title | question | verdict | key number(s) | lesson / why | C |
|---|---|---|---|---|---|---|
| 171 | READ-SLOTS | Do 8 learned slots reading once-encoded input carry serial think passes (cheap INJECT)? | FAILED (gate closed) | 4x4 0.003, flat in K'; 60% of read mass on EOT; BLIND 0.0 | Identical-input passes get no learning signal; memory unused (matches K0/DIRECT). | C1 |
| 172 | PCCoT-PROBE | Does the GPT-2 GSM8K latent SOTA use its latents/iterations? | descriptive (closed) | release 0.4905; latents zeroed 0.170; T'=0 0.134; more iters hurt (0.451 at T'=4) | Latents and iterations are used, but step state is distributed across the block, not swappable; no C2. | C1, C2 |
| 173 | CONV-HALT | Can decoding when the answer settles give learned halting? | FAILED (C5 not shown) | on §170 K8 mean h about 4.5; acc at h 0.19 vs 0.29 at K=8 pooled | Answer-settling stops too early; delta state never settles. | C5 |
| 174 | MULTI-TASK | One INJECT model on mult + nested, no task flag? | not run (gated on K8 gates that failed; unclear in text) | none | Gate never opened. | C1 |
| 175 | STEP-AUX | Does per-pass aux decoding from text CoT make passes each do one reduction? | CLOSED, not DEEP | pooled n3-8 0.787 vs FINAL-only 0.351; K'=4 solves n<=4 | First design where passes do one reduction; late passes under-supervised. | C1 |
| 176 | RAND-K | Does random/progressive pass count rescue INJECT? | FAILED (closed) | RK and PROG 0.0 at every K' | Huginn-style randomisation alone cannot fix a memory that goes unused. | C2 |
| 177 | PASS-DISTINCT | Can pass index/fade/step-matched K make identical passes trainable? | FAILED mostly (closed in practice) | (a) 4x4 0.001, index norm ~2% of embedding; (d) gate 0.0 at 12k; (f) pooled 0.506, not HIT | Model learned to make the pass index negligible; end decoder may bind. | C1 |
| 178 | FROM-SCRATCH | Does pretraining block think-pass use? | UNINFORMATIVE (closed) | scratch DIRECT/K0 0.114/0.109 (chance 0.091) | Test exclusion removed the n=1 table; protocol cause, not model result; s178b CPU gate failed. | C1 |
| 179 | READOUT | Does the decoder's view starve the memory (SHORTCUT) or can end readout not carry products (CAPACITY)? | CAPACITY (closed) | blind scaffold 0.0 at 12k/24k; col 3 0.351 | Cols 1-2 exact from memory alone; end decoder cannot carry the product. | C1 |
| 180 | (unused) | DEC-DROP draft never registered | n/a | none | Conditional on SHORTCUT, which did not occur. | - |
| 181 | INTERLEAVE | Do passes before each answer token (memory carried) work on mult? | FAILED (closed) | k1/k2 4x4 0.0/0.001; skip drops ~0 | Failure is not input repetition; seeing decoder bypasses memory. | C1 |
| 182 | STEP-AUX-HOLD | Which of NOFADE / HOLD / WINDOW makes extra passes stable? | WINDOW STABLE 2/2; NOFADE, HOLD not STABLE | WINDOW K'16-K'8 -0.001 (s1), 0.000 (s2); NOFADE 0.786 -> 0.418 at K'16 | Sampled step-matched pass count (WINDOW) gives stability; fade is not the cause of decay. | C6 |
| 183 | INTERLEAVE-BLIND | Does a view-free per-token reader make interleave passes compute? | FAILED (NOT SUPPORTED) | cols 3-7 mean 0.182 vs 0.165; skip drops 0.88/0.31/0.36/0.75 at cols 1/2/7/8 | Passes causal only for what one pass computes; inner cols chance. | C1 |
| 184 | AUX-REWEIGHT | Does equal aux weight per pass fix late-pass collapse? | MET (HELPS; C1 conjunction 2/2 on nested) | pooled 0.787 -> 0.974 / 0.9997; SERIAL-NEST 0.85 / 0.94 | Late-pass decay was under-supervision; depth beyond n=8 is count-limited. | C1 |
| 185 | ADD-NORM | Does normalised additive read fix overthinking? | FAILED by rule (NO-FIX); RMS descriptively DEEP | DEN K'16-K'8 -0.062; RMS n9-12 depth 0.587, K'16-K'8 -0.065 | Normalising the read substitutes for reweight but does not give stability. | C1, C6 |
| 186 | INTERLEAVE-BLIND-K4 | Closing test of mult interleave line | FAILED; line closes | cols 3-7 0.210 vs 0.182; within-token cos 0.993 | Extra passes per token are idle without per-pass targets. | C1 |
| 187 | AUX-NOPASS | Does removing the pass token raise depth past training? | FAILED to replicate (2/2 fails) | s1 depth 0.623, s2 0.493 vs §184 s2 0.665 | Depth tracks basin-entry timing; cross-recipe effects confounded. | C2 |
| 188 | REWEIGHT x WINDOW | Can one recipe be DEEP and STABLE? | FAILED / confounded (NEITHER) | s1 pooled 0.715, loss stuck ~0.90, never entered | Never converged (clip); composition question unanswered. | C1, C6 |
| 189 | RMS-REP | Is RMS a substitute for reweight? | MET (SUBSTITUTE, 2 seeds) | s2 depth 0.662 vs §184 s2 0.665 | RMS read alone matches reweight at 2 seeds; later undone by s3. | C1 |
| 190 | RMS x NOPASS | Does dropping pass token on RMS free late steps? | MIXED | LATE 0.878 vs 0.648; DEPTH 0.659 vs 0.587 (< +0.10) | Early steps size-free; remaining limit is content-level (items > any training item). | C2 |
| 191 | PROSQA-RELABEL | Was name memorisation the ProsQA barrier? | MET (NAME-MEM) | A test 0.950; K'0/1 chance; TEXT-COT 0.735 -> 0.938 | Relabelling lifts even text CoT; think passes carry the computation. | C1, C4 |
| 192 | RMS-SEED3 | Is RMS seed-robust? | FAILED (NOT-DEEP s3) | s3 pooled 0.589, depth 0.138, never entered | Seed-robustness claim refuted at 3 seeds; AdamW makes clip not a 7x step cut. | C6 |
| 193 | RMS x WINDOW | C1 and C6 on the seed-robust base? | MET (C6 claim 2/2) | s1 K'8/16/24/32 0.999/0.999/0.989/0.934; s2 0.978/0.978/0.954/0.862; depth 0.688/0.647 | JOINT + LONG-STABLE both seeds; gain over WINDOW is seed robustness. | C1, C6 |
| 194 | §190-S2 | Do §190 mechanism findings replicate? | descriptive: REPLICATES | handle t3-6 n12 0.960 (nesting<=2) | Size-free early steps and AUX-DONE tracking n hold at 2 seeds; s2 entry-confounded. | C2 |
| 195 | (reserved) | SELF-DEPTH-ANS | n/a | none | Reserved; superseded by §215. | - |
| 196 | GROKFAST-S3 | Does Grokfast make a non-entering seed enter? | FAILED (NO-EFFECT) | final bin 1.394 vs control 1.031; path_indep 0.930 | Symmetric amplifier of a contractive map, not a tilt. | C6 |
| 197 | (reserved) | BUDGET-ANNEAL-S3 | n/a | none | Not registered as drafted (weak one-seed rescue). | - |
| 198 | CHEAP-THINK | Is a cheaper think pass equal in quality? | FAILED (no CHEAPER-EQUAL) | (b) depth 0.454 vs 0.570 (entry-confounded); (c) READ-SLOTS depth 0.364 | Cached family comparator held; no cheaper design matched at one seed; $6.29. | C3 |
| 199 | CONTRACTION-TRAP | Is the seed lottery a contractive pass map; can CSR fix it? | MIXED (M1 PREDICTED; CSR FAILED) | CSR final bin 1.395 vs control 0.006; floor met by expanding motion | Account untested causally; one-sided dynamics target is gamed. | C6 |
| 200 | GSM8K-AUG PILOT | Does §191 recipe reach text on GSM8K-Aug? | FAILED (latent below text) | A 0.147 vs text C 0.321; K'1 0.125 vs K'10 | Loop forms weakly but mostly one-pass; GSM-Symbolic drop > 3 points. | C1, C3 |
| 201 | OVERTHINK-EXIT | Which channel drives overthinking; can a learned halt stop it? | MET (T1 falsified additive account; T2 halt) | R_FA 0.03/0.11 in RMS; halt no-overthink on 7/7 | Overthinking is post-completion DELTA writes; SNAP-HURTS 4/4. | C5, C6 |
| 202 | WARM-START-FROM-COT | Does a CoT-trained trunk lift the latent arm? | FAILED (NEUTRAL) | W 0.146 vs A 0.147; C+ 0.336 | CoT trunk forgotten within 2k latent steps. | C4 |
| 203 | SVR | Seed-stability via value readout | CLOSED (DO NOT LAUNCH) | NOT-USED 9/9; drop(V) <= 0.0004 | Decodable but unused separator; $0 spent. | C6 |
| 204 | STEP-TARGET-FACTOR | Does a factored per-pass target help GSM8K? | FAILED (ACC MATCHES) | F-P 0.139 vs A 0.147; shortcut ratio 0.74 vs 0.85 | Values readable from M_1; one-pass chain, not more serial. | C1 |
| 205 | IIT-SERIAL | Does a whole-state interchange loss stabilise seeds? | FAILED (s3 MET-NOT-ENTERED) | MET 3/3, ENTERED 2/3; $15.41 | Compatible with convergence but does not rescue failed seed. | C6 |
| 206 | JOINT-COT-L1 | Does CODI's L1 coupling lift the warm-started arm? | FAILED (COUPLE NEUTRAL) | J1-J0 -0.006; share_mem 0.073 | Coupling never entered the memory; ~$3.24. | C4 |
| 207 | GSM-SWAP | Do GSM8K latent models carry state or recompute from input? | descriptive | INPUT-FOLLOW 0.87-0.93; CARRIED <= 0.005 | 6/6 recompute from input; memory is scaffold. | C1 |
| 208 | QUESTION-ONCE | Can memory-only passes compute GSM8K? | FAILED (PASS1-PRECOMPUTE) | ITER-GAIN -0.005; K'1 0.143 = K'10 0.138 | GSM8K latent accuracy is one forward pass. | C1 |
| 209 | CODI-AT-OUR-BUDGET | Is the GSM8K latent shortfall architecture or budget? | MIXED | CODI4 0.189 (r 0.24); 0-latent decodes text 0.325 | ARCH-BOUND excluded; BUDGET-BOUND within 1 SD. | C3, C4 |
| 210 | CODI-BUDGET-20 | Does budget close the latent gap? | MET (REACHES-PUBLISHED) | 0.3791 (>= 0.348); +0.190 vs CODI4; m>=3 0.265 | Budget, not architecture; depth is the weak axis. | C3, C4 |
| 211 | VERSION-BANK | Supervised serial latent computation in pass-global attention memory | MET (VB-C1 for N2, N1); N3 COUNT-KEYED | N2 DEEP 0.9998, PROBE-GAIN 0.868; DEPTH 0.31-0.41 | N1 computes via carry alone; C2 not met. | C1 |
| 212 | VB-2G | Fix bottlenecks inside the bank | CLOSED (G1 HELD by user; Stage 0 diagnostics) | LONG-STEP m 9 0.85, m 10 0.27, m>=11 ~0 in N2 | HEAD-LIMITED cliff motivated D3; descriptive. | C1, C2 |
| 213 | D3 (SLVA) | Does slot-local version attention remove the long-step cliff? | MIXED | D3-T STEP-GENERAL to m16; DEPTH 0.480/0.494; HISTORY-USED 0 | Cliff removed; both cells collapse to carry; every cell stops improving after pass ~10. | C1, C2 |
| 214 | D4 PAUSE passes | Do computed pause passes lift depth/stop? | MIXED (not D4-C2; KEPT) | DEPTH 0.729 vs 0.480; gain after K'10 0.079 | GATED-COUNT: limit is working passes; entry 26k vs 11k confound. | C2 |
| 215 | SELF-DEPTH-CHAIN r1 | Can self-labelled re-rooted chains extend depth? | MET in support (not C2); SELF-ATTRIBUTED, MEM-TRADED | n9-12 1.000; M(n13-16) 0.717; STAB_self 0.782 -> 0.489 | In-support absorption and growth come from self items. | C2, C4 |
| 216 | SELF-DEPTH-CHAIN r2 | Does a second self round extend further? | MIXED (s1 SD2-C2, MEM-TRADED-2; s2 FT-DIVERGED, x2 FAIL) | M2 0.8745; LIT-C2-2 0.018; STOP 21 | One lineage only; boundary still binds; memory eroding. | C2 |
| 217 | CAUSAL-ACROSS-ROUNDS | Does self-training erode causal slot use? | MET (CAUSAL-PRESERVED) | follow_ii 1.000; STALL >= 0.99; $0 | Depth bought without eroding causal use; C1 only, not C2/C4. | C1 |
| 218 | LANG-VB | Does VERSION-BANK/SLVA learn ProsQA-SYM with self-teaching? | FAILED (A1 and DIRECT-SYM at chance; CLOSED) | acc_d2 0.47-0.53 at D3-16; CurGate advanced on caps | Task x recipe failure, not architecture; need a learnability pilot first. | C1, C4 |
| 219 | TEXT-SELF | Does text-CoT self-training match latent self-depth lineage? | CLOSED (CROSSED / parity) | n50 20.161 vs 20.539; latent round-1 lead closed | Latent depth edge not established; edge is efficiency (~20-35x fewer steps). | C2, C3 |
| 220 | GSM-CACHE-SWAP | Where does CODI latent computation live; does it port? | CLOSED UNREAD at stage 0 | G-R agree 0.667 < 0.837; zen 0.411 < 0.632 | Padded official semantics batch-dependent; arms sealed. | C1 |
| 221 | GSM-CACHE-SWAP-2 | Same battery behind a fidelity gate | CLOSED (CONTEXT-BOUND) | follow 0.033 / 0.029 vs floor 0.008; NECESSARY +0.25 / +0.32 | Latents necessary but context-bound; donor-final not carried at GPT-2 scale. | C1, C7 |
| 222 | PROBE-S | Certified probe readings for §218 A1 | CLOSED UNREAD | A1 at chance; $0 | VB-C1-L-S unreachable; certification not run. | C1 |
| 223 | PORT-STATE | Portable re-rooted latent state on templated words | FAILED / CLOSED descriptively | P3a pooled d<=2 0.574 < 0.80; $0 | NESTED-WORDS d>=2 not one-pass learnable under this recipe. | C1 |
| 224 | FIXED-HOLD | Does a fixed-point penalty fix the C6 hold failure? | CLOSED: UNEXPOSED | DEPTH-WARN (Note 224-9) | RD and FH H=1.000 to K'128, NOLOSS 3/3; RD depth -0.08/-0.15/-0.26; RD SUFFICES 1/3; $17.87 | A trained far pass fixes the hold but costs depth, growing by seed; penalty adds nothing. | C6, C1 |
| 225 | MATCHED-BUDGET | C3 at matched compute on the synthetic lineage | MET in registered scoped sense | within 0.02 at n 13-20; ~20-50x fewer steps; ~0.65x FLOPs | Synthetic nested arithmetic only, one seed each; not language. | C3 |
| 226 | ORDERED-WORDS stage 0 | Is the word task one-pass learnable? | FAILED / CLOSED | P0d full-FT d=1 0.998 but d<=2 0.53; $0 | Gate conflated recipe-learnability with one-pass composition. | C1 |
| 227 | TEXT-SERIAL control | Can serial text-CoT learn d>=2 where DIRECT cannot? | MET (PASSED) | d2/3/4 1.000, d5 0.97 | First language task with an established C1 band; $0. | C1 |
| 228 | LAT-OW | C1 test of §184 recipe on English word programs | FAILED (LAT-NOT-LEARNABLE-D2, Note 228-2) | d2 0.087 vs DIRECT 0.093; pass-1 aux 0.51 | INJECT learns d=1 only; state does not carry step 1 at GPT-2 small. | C1 |
| 229 | LAT-OW-CUM | Does a cumulative re-rooted per-pass target rescue §228? | FAILED; §228 language line CLOSED (Note 229-1) | d2 0.083 vs 0.093; pass-1 aux 0.55; §228+§229 ≈ $1.19 | Recipe negative at this scale; user 10-23: synthetic C6 first, then a ~0.5–1B base. | C1 |
| 231 | LARGE-BASE | Does Qwen2.5-0.5B compute the d=2 intermediate eagerly (P1′, $0)? | FAILED; CLOSED at $0 (Note 231-3) | EAGER v_1 0.112 vs layer-0 0.116 (chance); positive control v_0 0.988 | 0.5B base does not compute the intermediate either; scale alone is not the lever. | C1 |

## What worked

- Per-pass aux targets from text CoT at equal weight give DEEP + THINK-NECESSARY + SERIAL-NEST at 2/2 on nested arithmetic (§184: 0.974/0.9997, C1 conjunction met); §175's pass-by-pass design was the precursor.
- WINDOW (sampled step-matched pass count) gives STABLE 2/2 (§182); RMS x WINDOW gives JOINT + LONG-STABLE 2/2, C6 claim holds (§193: K'24 0.989/0.954).
- Normalised (RMS) additive read matches reweight on C1 and depth at 2 seeds (§185, §189), though seed-robustness fails at s3 (§192).
- Per-step name relabelling fixed ProsQA (§191: 0.950, text CoT 0.735 -> 0.938).
- Learned halt removes overthinking within training range on 7/7 models (§201 T2); T1 localised overthinking to post-completion delta writes.
- CODI at the published budget reaches 0.379 on GSM8K (§210), so the shortfall was budget (§209 MIXED).
- VERSION-BANK supervised serial latent computation (§211 N2) and slot-local attention removing the long-step cliff (§213, STEP-GENERAL to m16).
- Self-labelled re-rooted chains extend depth in support and beyond (§215-§216: M 0.717, M2 0.875), with causal slot use preserved (§217).
- A trained far pass fixes the C6 hold to K'128 (§224 RD, H=1.000).
- Synthetic-line C3: latent matches text self-training with ~20-50x fewer sequential steps (§225).
- Serial text CoT crosses the one-pass gap on the TOPO word task (§227).

## What didn't (dead ends)

- Identical-input think passes with FINAL-only loss never train: §171, §176 (RAND-K/PROG), §177 (a)(d)(f), §173; INJECT with pass index, fade or random K all 0.0.
- READ-SLOTS as cheap scalable read: §171 and §198(c) (depth 0.364).
- Interleave per-token passes for multiplication: §181, §183, §186; the line closed.
- Pass-token removal as a depth lever: §187 (reversed at s2, entry-confounded).
- Composition REWEIGHT x WINDOW: §188 never converged.
- Mechanism-targeted seed fixes: CSR (§199, floor gamed), Grokfast (§196, no effect), SVR (§203, unused), IIT-SERIAL (§205, s3 not entered).
- Latent-vs-text on GSM8K at our budget: §200, §202 (warm start forgotten), §204, §206, §208 (one-pass dominance, §207).
- Cheaper think pass: §198 (no CHEAPER-EQUAL).
- Language learnability: §218 ProsQA-SYM at chance for A1 and DIRECT; §223/§226 word tasks not one-pass learnable.
- Pause curriculum as a depth fix: §214 (stop not moved).
- Fixed-point penalty on the hold: §224 FH adds nothing over RD and costs depth.
- GPT-2 CODI donor-final portability: §221 CONTEXT-BOUND; §220 UNREAD due to harness.

## Recurring traps / methodology lessons

- Basin-entry confound: depth tracks when loss entered the basin; report entry steps (§187, §190, §194, §198, §214).
- Seed lottery: 2-seed claims reversed or broke at the third seed (§187 s2, §192, §205 s3); single-seed comparisons only meaningful same-seed.
- Protocol/gating errors: §178 test exclusion removed the n=1 table; §226 gate conflated recipe learnability with one-pass composition; §223 order/range assumptions.
- Controls/arms that cannot be informative: §178 UNINFORMATIVE; §212 PA arms UNREADABLE; §220 harness unread.
- Decode-based readings unfaithful beyond range; use decode-free checks (§213 notes 213-24 / 213-26).
- One-sided dynamics targets get gamed (§199 CSR floor met by expanding motion).
- Wording and compute accounting: §225 corrected §219's token-cost impression (only core blocks loop); descriptive-vs-registered readings (§171 addenda, §187 wording).
- Disclose post-hoc decisions (§224 note 224-8: RD-s3 chosen after s1/s2; §215 note 215-5: scorer whitelist after FT-SELF read).
- CurGate/curriculum caps advancing on step caps rather than threshold (§214 note 214-5, §218 note 218-28).

## Open threads worth revisiting

- Depth beyond training remains count/context-limited (C2 not met): §216 boundary still binds (STOP 21, LIT-C2-2 0.018); §187 token effect, if re-tested with entry control.
- C6 vs C1 trade: RD/FH cost depth at n 9-11 (§224); needs depth re-test; seed-3 RD pending.
- Memory erosion from self items (MEM-TRADED, STAB_self 0.782 -> 0.437, §215-§216); IDLE-HORIZON/FIXED-HOLD design.
- Language C1: §227 band; §228/§229 failed at GPT-2 small; next is a ~0.5–1B base (user 10-23), after the C6 depth-cost design.
- GSM8K: CODI-style depth weakness (m>=3 0.265, §210); latents recompute from input (§207).
- C5 (learned halt) is size-free on RMS (§201) but premature at the edge; parked behind COUNT-OOD.
- C4 stays unmet (strict); self-labelled chains (§215-§219) are the nearest candidate.
