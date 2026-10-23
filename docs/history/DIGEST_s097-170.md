# Digest of THEORY.md §97-§170 (2026-10-03 to 2026-10-09; written 2026-10-23, Fri Oct 23 13:28 EDT)

Scope: one row per section, verdicts quoted from each section's own result block or Notes (results often live in later Notes / inline "result" blocks; §110 and §110b results sit at THEORY.md ~L21250-21400). Numbers are as written there. "unclear in text" = no result found in the range. Goal: C1 latents do the work, C2 more thinking helps past training, C3 competitive with text CoT, C4 trainable from scalable supervision, C5 halting, C6 stable, C7 readable. No §109 exists.

| § | title | question | verdict | key number(s) | lesson / why | C |
|---|---|---|---|---|---|---|
| 97 | General latent thinker, text step targets | Does S5 restart recipe work with no hand-built state? | MET (G4); "no hand-built state is needed" | GEN-4 0.998/1.0/1.0 @512, 1.0x3 @2048; 0.56x FLOPs; C6 5/5 | Text target via LM head + re-embedded text checkpoint replaces 5x5 head and codebook. | C3 C6 |
| 98 | Same recipe on mult4x4/5x5 vs ICoT-SI, Coconut | Does the general recipe carry multiplication? | FAILED (M1, M-SOTA, M-C1, M-C3 not met, both sizes) | GEN-M 0.163 (4x4) / 0.005 (5x5); ICoT-SI 0.998/0.996 | Hand-off wall at slot 3; fork-conflict bug (98-3) voided first runs. | C1 C3 |
| 99 | Speed up latent hand-off (k-curriculum etc.) | Which cheap change speeds hand-off? | FAILED: "No arm accelerates" | CTRL 0.163/0.143/0.184; K-CURR 0.048-0.118 | Slot 2 is recomputed from visible operands; first true hand-off is slot 3. | C1 |
| 100 | Text snap every 2 steps (k=2) on mult | Do shorter silent stretches solve mult? | MIXED: met 4x4, not met 5x5 | 4x4 GEN2 median 1.000 (ICoT-SI 0.998); 5x5 0.100/0.093/0.091 | One hand-off per chunk learnable at 4x4; 5x5 still walls; success may be in-pass recompute (100-2). | C1 C3 |
| 101 | Snap-strength dose-response on S5 | How much discrete correction is needed? | MET: smallest passing beta_min 0.01 ("a weak attractor suffices") | b=1 3/3; 0.1 2/3 (0.9995/0.0005/1.0); SELF-SNAP gate failed (purity 0.065) | Self-discovered codebook not clustered (101-2); weak attractor is enough. | C6 |
| 102 | Gated residual state path slot j-1 to j | Is the staircase an attention-retrieval problem? | FAILED ("No"): no acceleration | test 0.170/0.032/0.013 vs CTRL 0.163/0.143/0.184 | Direct state path does not help; compute per op / carrier is the issue. | C1 |
| 103 | Self-training by length (filtered) | Learn step states from final answers? | FAILED but "near miss for C4" | cum 512: 0 / 0.9615 / 0.9585 (3 seeds) | Pseudo-label precision ~0.98 capped it; s2 mean-removal 0.949 = fork bypass. | C4 |
| 104 | Donor patching / step-blocking on §92, §97 | Do slots carry state serially? | FAILED: SERIAL not met in any family; "§97's latents are bypassed" | §92 SERIAL 2/5 | Earlier C1 claims rested on zero ablation; fork recomputes from visible text. | C1 |
| 105 | §97 S5 with masked fork + donor patching | Unqualified C1 for headline? | MIXED: G4-M, MASK-DEC met; HANDOFF 0/3, unqualified C1 not met | unclear in summary | Masking decodes alone did not force carriage. | C1 |
| 106 | Bounded-state streaming multiplication (C2) | Latent chains vs written format extrapolate? | MIXED: written format met, latent failed | GEN-S-MASK 0.004-0.006 @b=4; WC-S-k2 1.000 at b 4-32 (3/3) | C2 credit belongs to written format + cut; latents add nothing; even one carry hand-off fails. | C2 C1 |
| 107 | Carrier width vs compute per op | Does more latents/pauses per op unblock mult? | FAILED: "Neither lever unblocks step 3" | LAT-2/4 test <=0.063; PAUSE-4 0.055/0.042/0.007 | Carrier held multiplier prefix (114-4), so no carrier change helps. | C1 |
| 108 | Forced carriage on S5 (Markov segments) | Serial latent carriage with visible route removed? | MIXED: ACC, C1-MK, C6 not met; "every seed that learns carries serially" | s1 1.0/0.996 @512/2048; s3 per-hop 0.99845 -> 0.59; s2 chance | Seed lottery; failing seeds on null loss curve (no-read basin, 111-1). | C1 C6 |
| 110 | Self-training v2 (independent seeds, gate, masked fork) | C4+C1 from final answers? | MIXED: SELF2, C1, C6 not met (STUDENT 3/5) | s3 0.999/0.994 with donor 0.96-0.98 (C4 qualified+C1+C2); s2,s4 0; s1 bypass | Identity-start only trains no restarts (110-1 design error); 2 seeds never acquire (start-text no-read basin). | C4 C1 |
| 110b | Self-training, random given starts | Fix §110 restart gap | FAILED: aborted after clear null | loss flat 4.86-4.92 (chance ln120); final-correct 0.0074-0.0091 | From random starts, final-only length>=3 does not get off ground (strict-C4 barrier). | C4 |
| 111 | Reliability of Markov hand-off | Can doubling budget or reader fix lottery? | MIXED: only MK-4-READ meets REL | READ 5/5 (0.9995-1.0 @512), 4/5 @2048; 2x 0/5 | Auxiliary reader from step 0 removes no-read basin. | C6 C1 |
| 112 | Short-coverage dose-response for self-training | How small can the foothold be? | MET: FOOTHOLD met at every dose; smallest f=0.1 | f=0.1: passing s21,s23 (0.996/0.9985), never-labelled k=1 1.0 certified | Shorts act as foothold; acquisition timing counts (112-5); budget limited (112-6). | C4 |
| 113 | Aux reader inside self-training | Does reader fix self-training lottery? | FAILED: REL not met | SELF2 2/5, C1 0/5 | Failure is reading the start text, not the slot latent (113-1 prediction held). | C4 |
| 114 | Looped depth per op for mult | Does serial depth unlock step 3? | FAILED: DEPTH not met | LOOP-R op-3 0.342/0.195/0.362, test 0.055/0.026/0.062, flat in r | Latent holds multiplier digits, not running sum (114-4): a shortcut. | C1 |
| 115 | No-snap Markov chains + round-trip snap trigger (eval) | Can model self-detect drift? | MIXED: NOSNAP fails, TRIGGER holds 6/6 | no-snap 0/0/0; trigger 1.000 with 61-99.9 snaps vs 127 fixed | Latent drifts but model detects own drift label-free. | C5 C6 |
| 116 | Strict C4 from operator definitions only | Learn S5 with no short labels? | FAILED: STRICT-C4 not met (NONE 0/4 at chance); FOOTHOLD-OPDEF false | OPDEF s33 learned whole table (k=1 1.0, 14,400 certified); s31,s34,s35 no | NONE+COPY void (copy_loss under no_grad); definitions minimal foothold, one seed. | C4 |
| 117 | One model C1+C4+C6 via Markov self-training + reader | Combine C1,C4,C6? | FAILED: first wave 0/4 acquire; "lottery persists" | MK s52/53, ST s62/63 all 0 @512 | Markov from step 0 does not acquire in-chunk composition from final answers. | C4 C1 C6 |
| 118 | Fully latent: anchor carried latent to text path | Remove snaps at inference? | MET (ANCHOR): 5/5 exact to 2,048 no snaps | s71 1.0/1.0/1.0; control READ no-snap 2/5 | Anchoring removes drift; contraction diag (118-3). LONG arm unclear. | C2 C6 |
| 119 | Published latent methods on our S5 protocol | Controlled C1 baselines? | unclear/uninformative: "CONTROLLED-C1 is not readable" | Coconut 0.007/0.0085/0.0115; SIM-CoT 0.007-0.014 (chance 0.0083) | No published method learned S5 (text CoT did); comparison impossible. | C1 |
| 120 | Learned discrete FSQ snap every step | Task-agnostic digital recurrence? | MIXED: pilot not met (FINAL passes flip-flop, not S5) | S5 s1 1.0 (2048 0.998); s2 code collapse (9 codes) | Seed lottery; soft-phase no-read basin; code wanders (not closed). | C6 C2 |
| 121 | Text-path anchor on mult carrier | Break the carrier shortcut? | FAILED: CARRY-3 not met | ANCHOR+READ op-3 0.231, test 0.031; READ step3 0.643/0.927 | Anchor bound to incompetent target; reader helps by recomputation. | C1 |
| 122 | TEXT-MIX keeps text path trained | Competent anchor target enough? | FAILED: CARRY-3 0/3; STOP RULE fires | op-3 0.362/0.408/0.324, test ~0.05; t1 1.0 | Text path never needed to carry the sum; mult latent line paused. | C1 |
| 123 | Dither dose-response on §120 harness | Is attractor strength causal? | CLOSED: "causal question is not reached" (CAUSAL=false) | rho_ff 0.146-0.156 at all sigma; seed 2 plateaus at all sigma | Dither moved neither contraction, accuracy nor basin; continuous 8-d bottleneck alone is exact. | C6 |
| 124 | Copy objective vs exposure control | Does reading a given state fix self-training lottery? | MIXED/descriptive: COPY uniform and earliest | COPY acquires L5 4/4, SERIAL 2/4; EXPOSE 0/4 serial; BASE 2/4 | MECH-112 false (EXPOSE acquires too); only COPY gives serial carriage. | C4 C1 |
| 125 | Standardise carried code against soft basin | Fix the entry basin? | untested: PLATEAU-FIX cannot be met | STD s2 plateau (loss 5.0 @2k); norm 35.2 vs 3.94 | Standardisation not in force at 2k; delivery checks added (127-1). | C6 |
| 126 | Raw-h carriage vs annealed bottleneck | Is bottleneck the cause of basin? | MIXED: "8-d bottleneck causes the 2k entry basin" but wide carrier doesn't learn | RAW-H s2 0.846 @512, 0.732 @2048; ANNEAL s2 0 | Entry fixed without bottleneck but raw carrier drifts/fails long. | C6 |
| 127 | Four entry fixes vs restart-on-plateau | Reliable entry without restarts? | FAILED: no entry fix adopted ("escaping 2k flag easy; learning is the failure") | STD2, LR10, VAR, SKIP; seed 4 plateaus; SKIP s4 escapes+learns | Restart-on-plateau stays; seed-2 first pull at step 650 fails (127-7). | C6 |
| 128 | One recipe, five state-tracking families | Task-agnostic recipe? | MET: TASK-AGNOSTIC = true | exact+serial on 5 families, held-out closed; S5 3/4 slots, 2 restarts | Needs restarts on S5; CLOSED certificates fp32-checked. | C6 C2 |
| 129 | Re-draw bottleneck head at 2k flag | Where does entry basin live? | descriptive: fresh head escapes, "no re-drawn seed learns" | s2 re-plateaus in both REDRAW and CLOCK-ONLY | Basin not in the head alone. | C6 |
| 130 | Early label-free plateau marker | Predict 2k plateau across GPU class? | FAILED: no PASS | dip FLAG precision 0.70 (0.35-0.93), recall 0.78 | Marker fails cross-card; keep the 2k flag. | C6 |
| 131 | Second restart point at 4k (FLAG4) | Flag "escape without learning" from loss? | MIXED: PASS but not deployable | precision 1.0 (0.63-1.0), recall 0.73; 8 flagged held-out <10 | 131-7 first prospective miss (late re-entry). | C6 |
| 132 | Rules in context (table ladder) | Carrier tracks state given per-problem tables? | MET at rung 1 (3/4); higher rungs unclear in text | rung-1 long_acc 0.9865-1.0, serial, not memorised | In-context rules learnable at small size. | C1 C2 |
| 133 | Width x correction; capacity probe | Can error correction make wide carrier exact? | MIXED: stage 1 four arms PASS, FSQ-16 FAILS | FSQ-64, NOISE-64, SOFT-16 PASS; rho 0.10-0.24; stage 2 unclear in text | Wide carriers exact once entered; entry gets worse with width. | C6 |
| 134 | COPY-ANCHOR label-free reader from step 0 | Fix entry failure / escape without learning? | MET: PASS on S5 4/4 | ext_2048 0.994-0.998 on plain-failure seeds; plain 1/4 | Guard on MOD: COPY failed closure s1 (134-6), seed-specific (2/3 close). | C6 C4 |
| 135 | Teacher-state distillation (hidden states) | Distil CoT teacher's state into carrier? | FAILED: 0/4 both arms; "worse than no distillation" | TSD 0/4 exact, BASE 0/4; main loss 3.2-4.7 | Weight-1 InfoNCE pulled context into code; host fault void b78 run. | C4 |
| 136 | GROW: start narrow, widen | Does staged growth enter wide carriers? | MET: COPY+GROW 4/4, COPY+GROW-NOISE 4/4; GROW-P 1/4, GROW-4 0/4 | ext_2048 0.992-1.0; rho 0.07-0.14 | Entering narrow then widening is safe at 64. | C6 |
| 137 | Column-by-column mult on carrier | Does carried carry solve mult? | FAILED: DROP at proof | test 0.000; col 1.0/0.326/0.124/0.103/... | Carry-dependent middle columns at chance; code carry R2 0.24 vs h 0.69. | C1 C3 |
| 138 | FULL WIDTH 768-d carrier | Does full width work? | FAILED: all four arms dropped at proof | ext_2048 0.0 all; loss ~4.8 | Learning from scratch at 768-d fails to enter; revised as §143. | C6 |
| 139 | DISTILL-2 fading warm-start on mult | CoT teacher state helps latent mult? | FAILED: "No arm learns multiplier-major 4x4" | A' test 0.000; teacher S_j not linearly decodable (0.174) | Teacher probe gate; 8-digit COPY slow. | C4 C3 |
| 140 | DIGPAIR: one digit pair at a time | Digit-pair chain fix mult? | FAILED: DROP at proof | test 0.002; first error col 3 in 685/1000 | Accumulators >=100 not even in h (R2 0.05); learning to represent bigger sums. | C1 C3 |
| 141 | Latent repeats per column + wider carrier | More depth per column help? | FAILED: all three cells dropped | test 0.000; r4s16 col2 0.495 | Same as §137. | C1 |
| 142 | MULT-READ: per-step reader, wide carrier | CoT-derived reader of running sum? | FAILED: DROP at proof | test 0.002; S1 1.0/S2 0.352/S3 0.052 | "Not acquired within budget" (3.8 epochs vs ICoT ~200); priors predicted DROP (142-2). | C1 C3 |
| 143 | FULL-GROW to 768-d | Staged growth to full width? | MET: PASS (3 decided seeds) | ext_2048 1.0, rho 0.043-0.062 | New 704 dims contribute only ~4% of carried embedding. | C6 |
| 144 | RUNNING-SUM diagnostic | Can carrier hold growing 4-digit number? | FAILED: both arms DROP | F8 no entry (4 attempts); N64 long_acc 0.0, low digits at chance | Carrier keeps coarse magnitude only. | C1 |
| 145 | MULTI-SLOT: K=4 vectors per hand-off | More vectors per hand-off? | FAILED: DROP | test 0.000; slots don't specialise (R2 ~0.13/0.3/0.1) | Same wall; CARRY1 did not run. | C1 |
| 146 | MID-PROBE: accumulator readable mid-stack | Does a middle layer hold it? | CLOSED: NO SIGNAL | layers 1-12 R2 0.85-0.88 / 0.10-0.22 (>=100) | No layer holds accumulator >=100; mid-layer hand-off not a fix. | C1 |
| 147 | DELTA-STATE linear-recurrent memory | Does additive memory break middle-column wall? | MIXED: DROP by rule (0.743) but IMPROVES | cols 3-7 0.92-0.99; accumulators >=100 0.93 | First design to break wall; loss still falling. | C1 C3 |
| 148 | LOOP-RETRO recurrent depth in GPT-2 | Depth for direct-answer mult? | FAILED: LOOP-4 0.002 = PLAIN 0.002 | accuracy flat in test r 1-16 | Loops unused; B3 stops. | C3 |
| 149 | SOTA-PROBE released GPT-2 latent reasoners | Do they carry values serially? | descriptive: none serial at release | simcot_coconut PARTIAL; simcot_codi BYPASSED | First donor/contraction scores on published models. | C1 |
| 150 | VOCAB-SLOT token-shaped working memory | Soft vocab slots as memory? | FAILED: both DROP | test 0.000; slots collapse to constant tokens | Token-shaped slots collapse. | C1 C7 |
| 151 | DELTA-LONG 48k budget | Does 2x budget reach exact? | FAILED: DROP at proof (improves) | slot1 0.837, slot2 0.889 | Helps but not exact. | C1 C3 |
| 152 | DELTA-READ CoT-derived reader | Reader of running sum then remove? | FAILED: DROP | A s1 0.331; CTRL 0.012 | Still learning row 4; 96k run worse (0.079). | C3 C4 |
| 153 | THINK-PASS item-agnostic passes | Item-agnostic latent passes? | unclear in text | PURE/REPEAT can't work with input-only writes (153-3) | Superseded by DeltaMem-R (§155). | C1 |
| 154 | SHIFT-MEM relative keys | Length generalisation? | FAILED: stopped at proof | 4x4 0.653; ext_4x8 col5 0.000 | Count schedule memorised; carry carried contrary to prediction. | C2 |
| 155 | DELTA-R write-after-read memory | Memory holds computed results? | MIXED: both arms FAIL PASS rule; exact, never SERIAL | R 4/4 exact (1.000/0.998/0.997/0.999), 0/4 SERIAL; ext_4x8 <=0.006 | Stores completed digits; no recurrent use (155-4). | C1 C3 |
| 156 | SEP-SLOT slot per answer digit | Fix carry resolution? | FAILED: "wall is not about position count" | SEP 0.002/0.004; PAUSE 0.002 | Positions don't help. | C1 |
| 157 | READ-PANEL | Is the read the bottleneck? | FAILED: "it is not the read" | PERIODIC 0.035; off-column weight 0.88 | Blend, not geometry. | C1 |
| 158 | FREE-PAIRS hybrid sequences | Force accumulation? | descriptive: SERIAL, no C1 credit | test 0.982; accumulator 0.43 vs raw digits 0.86 | Buffer + recompute; swap gap was data-induced. | C1 |
| 159 | R-5x5 SOTA leg | R at 5x5 vs text? | MET: 4/4 exact; "LATENT BEATS TEXT" | 1.000x4; 0.80x TC-M FLOPs, 0.30x passes; ICoT-SI 3.24x fewer FLOPs | Not SERIAL; ext_5x8 <=0.006. | C3 C6 |
| 160 | R-ROWS | Row segmentation with R? | FAILED: A-R 0.056 gate closed | CTRL-R 0.055; rows fail at col 5 | C3 stays with digit-pair chain. | C3 |
| 161 | SHAPE-MIX | Break 4x4 count schedule? | MET: MIX 3/3 length-general | 4x4 0.996-0.997; ext_4x8 0.964/0.917/0.912 | Varying shapes removes schedule; memory carries running sum. | C2 C1 |
| 162 | DELTA-S5 vs DIRECT boundary | Memory on S5 vs pass limit? | MIXED: LONG MIXED, SHORT BUFFER-LIKE | LONG 1.0 <=32 (DIRECT 0.007); edit propagation 0.0 | Both carry computed non-linear state with primacy/recency. | C1 C2 |
| 163 | SHAPE-MIX-2D | Both operand lengths mixed? | MET: BOUNDED 3/3 | 5x5 0.996/0.995/0.998; 6x6 ~0.97; 7x7 0.70-0.83 | Bounded generalisation. | C2 |
| 164 | MIX-BASELINES | Text CoT on same mix? | MET: LATENT BEYOND TEXT ON LENGTH | TC-MIX 4x4 1.0 but 0.0 at 4x6+ | Anchored read fixed eval. | C3 C2 |
| 165 | FIXED-CHUNK think passes | Memory finds own schedule? | FAILED: neither K arm exact | 0.0 at 4x4 and extensions | Harness schedule does the work. | C1 C5 |
| 166 | STATE-COVERAGE | Wider initial coverage fix collapse? | FAILED: COVERAGE-HELPS not shown | SP = PLAIN (L64 0.988 vs 0.991; L512 0.0) | MULT closes uninformative. | C2 C6 |
| 167 | DAS-EDIT | State causally decomposable? | MIXED: S5 3/3 DECOMPOSABLE; MULT one model (s2 t=4, IIA 0.927) | MULT s1/s3 FIT-FAILURE | Large-k fits collapse; negatives unreadable. | C1 C7 |
| 168 | DEEP-ProsQA | Graph search past one pass? | FAILED (step 2 later): D*=3; STEP-AUX not DEEP-LATENT | step 2: accuracy 0.490; train 1.000 | Memorised fixed training set; reverses "smaller task first". | C1 C3 |
| 169 | INJECT | Re-present problem every pass? | FAILED: neither K arm exact | K16 0.002; K8 0.001 | Memory unused; think_necessary vacuous. | C1 |
| 170 | NESTED expressions | Nested arithmetic past one pass? | MIXED: n*=3; K8 slot 1 not DEEP | K8 0.985/0.951/0.763...; K'=16 worse; SERIAL-NEST fails | Passes used but more thinking hurts. | C1 C2 |

## What worked

- §97/§118: S5 exact to 2,048 with text step target + text snap, and fully latent (no snaps) with an anchor.
- §101: a weak attractor (beta_min 0.01) suffices.
- §111: auxiliary reader removes no-read basin (5/5 REL).
- §112/§124: small foothold (10% short coverage) or COPY objective yields acquisition.
- §115: label-free drift trigger replaces fixed snaps (6/6).
- §128/§134/§136/§143: one task-agnostic recipe across five families; COPY-ANCHOR 4/4; staged growth to 64 then 768-d is safe.
- §147 to §163: additive/write-after-read memory breaks the multiplication middle-column wall; SHAPE-MIX gives length generalisation; R at 5x5 4/4 exact with 0.80x TC-M FLOPs (§159); latent beats text on length (§164).

## What didn't (dead ends)

- §98-§100, §107, §114, §121-§122, §137-§145, §148, §150: multiplication latent hand-off walls.
- §103/§110/§110b/§116/§117: strict C4 from final answers (strict-C4 barrier).
- §119: published methods did not learn S5.
- §125, §127, §129, §130: entry fixes and early markers.
- §135: teacher hidden-state distillation.
- §165, §169, §170, §168: think-pass harnesses leave memory unused or memorise.

## Recurring traps / methodology lessons

- Zero/mean ablation overstates C1; donor patching (§104).
- Seed lottery and GPU class change basins (§108, §125).
- Delivery bugs (§116 no_grad, §125 std not in force).
- Fork/recompute shortcuts (§98-3, §105).
- Confounds (§106-3, §158-3).
- Host faults (§135-6).
- Outcome-based triggers need absolute floors (§162-4).

## Open threads worth revisiting

- Fresh procedurally generated training (ProsQA-FRESH lead, §168).
- Strict C4 beyond operator-definition foothold (§116).
- Multiplication C1 credit (§158, §167) and large-k DAS fitter fix.
- §133 stage 2 capacity, §132 higher rungs, §153 outcome (unclear).
- Make think passes used (§165, §169, §170).
