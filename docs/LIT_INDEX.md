# Literature index (one line per LIT note; updated 2026-10-03)

Literature session. Start with the **synthesis** notes. Each per-registration check is cited by path from THEORY, so
all stay in place. Historical bibliographies: `LITERATURE.md` (to 09-27) and `LITERATURE_NOTES.md` (09-26); earlier
advisory notes are in `archive/`.

## Synthesis (read these first)
| Note | Content |
|---|---|
| LIT_FINDINGS_COMPILED_2026-10-02 | every project result → nearest prior → New / Rep / Ext / Con; honest novelty claims |
| LIT_LATENT_FROM_DISCRETE_2026-10-02 | all proposals → 7 rules; field survey; methods M1–M8 (+ M1-U, condition tags) |
| LIT_SEMI_DISCRETE_2026-10-03 | 8 bands from text to free latent; ideas: validity-triggered snap, ST-snap, snap noise; §83 caveat |
| LIT_REVIEW_2026-10-02 | top papers vs our findings; 5 open problems |
| LIT_THEORY_LEADS_2026-10-02 | 3–6 leads each: acquisition, composition, OOD confidence, paraphrase semantics, latent-vs-text length |
| LIT_S92_NOVELTY_2026-10-03 | §92 claim line: new as scoped; closest mechanism 2510.14095; no certified latent S5 in a pretrained LM |
| LIT_SOTA_TARGETS_2026-10-01 | comparator numbers: S5/A5 (DeltaProduct 2502.10297, Zhang 2607.20594), multiplication (Cho 2410.15787, Lee 2502.01612), Chomsky (MLP-LDRU 2605.26035) |
| LIT_PROJECT_AUDIT_2026-10-04 | whole-project audit: 8 headline findings → prior, novelty, overclaim, reviewer demand; contradictions; 3 missing comparisons |
| LIT_STABLE_LATENT_SYNTHESIS_2026-10-06 | stable-latent synthesis (user directive): 6 lines of prior work → predictions on §101/§123/§126/§127/§129; escape-without-learning likely a short-length shortcut, not collapse; proposed wide carrier + weak learned FSQ attractor + state-consistency + own-rollout + skip |
| LIT_GENERAL_LATENT_MAP_2026-10-07 | general latent models mapped to C1–C7 (Coconut, CODI, Huginn, Ouro, Attractor, LOTUS 2606.31779, Abstract-CoT, PLaT, SLPO 2607.19691, LTF 2606.16222): no one-recipe certified generality, no method-paper C1 tests; RL needs warm start; borrow list |
| LIT_NORTHSTAR_2026-09-29 | C1–C7 audit requirements (substitution, IIA edit, input-only control); hold firsts to 2404.18988 |

## Per-registration checks (chronological)
| Note | § | Verdict (key prior IDs) |
|---|---|---|
| LIT_PROSQA_SHORTCUT_2026-09-30 | §57 | ProsQA proximity cue undocumented: new (1906.07132 general) |
| LIT_CLR_REVIEW_2026-09-30 | §33–46 | latent loop + written checkpoint + cut is new as a combination (Delethink 2510.06557, InftyThink 2503.06692, ReSum 2509.13313) |
| LIT_SET_RESTART_2026-09-30 | §44 | set-valued restart priors (CoT2 2505.23648) |
| LIT_PILOT_M_2026-09-30 | §38 | label-free amplify-and-distil ladder = expert-iteration family (1705.08439) |
| LIT_TASKS_ARCH_2026-09-30 | PROTOCOL8b | task and architecture precedents; Shapiro 2608.11233 is the closest pretrained looped retrofit (corrected) |
| LIT_F1F5_CHECK_2026-09-30 | §52 F1–F5 | F2 is the most novel; LaTER 2605.07315 nearest for F5 |
| LIT_EXTERNAL_CHECK_2026-09-30 | ext. ideas | PENCIL 2503.14337 closest; Shapiro correction |
| LIT_F6F7_CHECK_2026-09-30 | §58, §54b | F6 = interchange-intervention training variant |
| LIT_BOUNDARY_MEMORY_CHECK_2026-10-01 | §60 | new as a combination, not as components |
| LIT_F1S_CHECK_2026-10-01 | §61 | Fan 2409.15647 is the closest design |
| LIT_NOISE_CLOCK_CHECK_2026-10-01 | §62 | label-noise priors (instance noise < ½ keeps the Bayes classifier) |
| LIT_F3_CHECK_2026-10-01 | §52 F3 | compositional self-synthesis; self-improving 2502.01612 |
| LIT_F4_DELTA_2026-10-01 | §63 | round-trip translator; edit-set precedents |
| LIT_STATE_TRACKING_PREP_2026-10-01 | §64 prep | operator-first (Zhang 2607.20594); overwritable public state |
| LIT_PERM_TRACKING_CHECK_2026-10-01 | §64–66 | new as scoped; closest Thinking States 2602.08332 |
| LIT_SNAP_VQ_CHECK_2026-10-01 | §67 | snapping priors: SR-RNN 1901.08817, QBN 1811.12530 |
| LIT_EXTERNAL_C1_CHECK_2026-10-01 | §67 | Design 1; IOHMM is the uncited closest prior |
| LIT_BENCH_VERIFY_2026-10-01 | bench | bench values verified against the papers (+ SIM-CoT addendum) |
| LIT_GSM8K_STEP2_2026-10-01 | §68 | GPT-2 GSM8K-Aug references; public D&W checkpoints 2604.04902 |
| LIT_RL_ACQUISITION_2026-10-01 | §69 | MAPO 1807.02322, RandoMer 1704.07926, spectral WFA baseline |
| LIT_ACQ_AB_2026-10-01 | §70 | second-order RNNs, ISAN 1611.09434; UEM/Lateen; Howe 2609.18966 addendum |
| LIT_S5_PROTOCOL_2026-10-01 | §71 | exact S5 protocols; Li 2503.02854 pretrained cutoff |
| LIT_FROZEN_POS0_LR_2026-10-02 | §72 | FPT 2103.05247; position-0 sink 2402.17762/2410.10781; head-lr confound |
| LIT_STABLE_LATENT_2026-10-02 | §72b+ | teacher → LM is new only as a pipeline; best-of-n reporting (Dodge 1909.03004) |
| LIT_STANDARDISE_FROZEN_2026-10-02 | §73 | known fix (Timkey 2109.04404; MAE 2111.06377); freeze statistics |
| LIT_PARAPHRASE_TRANSFER_2026-10-02 | §74 | new as scoped; lexical-exposure caveat 2212.10769 |
| LIT_SCALEUP_S4A5_2026-10-02 | §75 | coset traps 2609.29951; held-out pairs 2606.07254 |
| LIT_FREE_TABLE_2026-10-02 | §76 | tabular IO-HMM; over-parameterised EM 1810.11344; M = 2\|G\| is an oracle |
| LIT_PER_CODE_READOUT_2026-10-02 | §77 | = second-order RNN with a one-hot state; one-hot control |
| LIT_GSM8K_STEPSPLIT_2026-10-02 | §78 | same split as Hase 2401.06751 (7–70B); none at GPT-2 scale |
| LIT_PARAPHRASE_OPS_2026-10-02 | §79 | Hill 2005.09382, Lynch 2005.07648; bag-of-words control + minimal pairs |
| LIT_CHUNK_MACHINE_2026-10-02 | §80 | new as scoped; two held-out levels |
| LIT_SINKHORN_2026-10-02 | §81 | Gumbel-Sinkhorn 1802.08665; doubly stochastic ≠ no coset trap |
| LIT_ACQ_LADDER_2026-10-02 | §82 | coset-foothold prediction (**contradicted by §82**); Z59 control |
| LIT_CONFIDENCE_RESTART_2026-10-02 | §83 | FLARE 2305.06983, CALM 2207.07061 + LTT 2110.01052; OOD overconfidence |
| LIT_HALTING_DECOY_2026-10-02 | §84 | EOS decision 2010.07174; randomised PE analogue 2305.16843 |
| LIT_TYLER_2026-10-02 | read | 2606.16360: baseline artefact; budget-aware invocation reward useful for C5 |
| LIT_ANNEAL_GE3_2026-10-02 | §87 | logit-temperature ≠ DA; no prior A5 from length ≥ 3 only |
| LIT_CONTINUOUS_FADE_2026-10-02 | §89 | HRPO 2505.18454 closest; fade α not W; standardise h; keep the decoder |
| LIT_DIVERSITY_THRESHOLD_2026-10-02 | §90 | diversity priors (2306.15063, 2405.15071); confound withdrawn (corrected) |
| LIT_SEED_RELIABILITY_2026-10-03 | §91 | operator-first = Zhang 2607.20594; wd 1.0 erases pretraining → L2-SP |
| LIT_HANDOFF_COLLAPSE_2026-10-03 | §92, α → 1 | posterior-collapse analogue (1511.06349, 1506.03099); CODI projection |
| LIT_SPARSE_LABELS_2026-10-03 | §93 | no density threshold published; chunk-length footholds; 2602.14814 relevant |
| LIT_WHERE_READOUT_2026-10-03 | §94 | reversal-curse asymmetry (2309.12288, 2309.14402); reverse-direction target |
| LIT_SELF_LABELS_2026-10-03 | §96 | DNN-HMM forced-alignment precedent; pipeline C4 qualified by p |
| LIT_S97_TEXT_SNAP_2026-10-03 | §97 | new by combination (SELR 2608.13570, Thinking States); k = 1 = written-state loop |
| LIT_S98_MULT_2026-10-03 | §98 | ICoT-SI 5×5 release 0.996; no latent baselines on multiplication; 2510.00184 running sum |
| LIT_S99_ACCEL_2026-10-03 | §99 | no published speed-up figures; teacher-forced slot inputs missing candidate |
| LIT_S99_MARKOV_2026-10-03 | §99-3 | MARKOV arm: fork must be chain-masked; position re-indexing |
| LIT_S100_K2_2026-10-03 | §100 | k = 2 bypass plausible (ICoT-SI one-pass); compute claim vs text CoT only |
| LIT_DIRECTIONS_LEADS_2026-10-04 | memo | A/B/C leads: LIFT parallel state training, filtered length self-training, Howe channel / fixed points |
| LIT_S101_103_DESIGN_2026-10-04 | §101–103 | per-chunk stochastic β; T²MLR gate; same-weights vote weaker than cross-seed |
| LIT_SELF_SNAP_2026-10-04 | §101 SELF-SNAP | SR-RNN 1901.08817 closest; purity premise check |
| LIT_S104_INTERVENTIONS_2026-10-04 | §104 | IIA / patching citations; resample over zero/mean; donor- vs recipient-consistent |
| LIT_S105_106_2026-10-04 | §105–106 | MASK = decodable not hand-off; Jelassi 2306.15400, Neural GPU 1511.08228 |
| LIT_S107_MULTISLOT_2026-10-04 | §107 | capacity vs compute; pause control |
| LIT_S108_MARKOV_S5_2026-10-04 | §108 | position-0 sink fix; compounding arithmetic |
| LIT_S109_PRECHECK_2026-10-04 | §109 | latent vs pause (2609.35643); 2510.00184 = scalar MSE, 2L4H from scratch |
| LIT_S110_SELFTRAIN_2026-10-04 | §110 | agreement overestimates precision with shared weights; STaR restart |
| LIT_S110B_STARTS_2026-10-04 | §110b | random starts label the full Cayley table: restate the C4 qualifier |
| LIT_S111_RELIABILITY_2026-10-04 | §111 | budget fixes timing, not basin lotteries; DAgger 1011.0686 |
| LIT_S112_113_2026-10-04 | §112–113 | never-labelled transitions constrained by long finals; reader priors |
| LIT_LOOPED_PROPOSALS_2026-10-04 | P1–P4 | RRT 2410.20672, MoR 2507.10524, retrofits; ST-Gumbel; early triage |
| LIT_S116_STRICT_C4_2026-10-04 | §116 | the cut makes self-started shorts (leap L → L − k); NONE must start at L > k |
| LIT_S117_INTEGRATED_2026-10-04 | §117 | staged-bottleneck fallback; foothold_yield is truth-scored (log only) |
| LIT_S118_ANCHOR_2026-10-04 | §118 | anchor = professor forcing 1610.09038; LONG moves the range boundary |
| LIT_S119_BASELINES_2026-10-04 | §119 | faithful Coconut/CODI/SIM-CoT recipes (configs read); native + chunked hosts; MASK ≈ SIM-CoT with tied in-context decoder; Thinking States as 4th arm |
| LIT_TASK_AGNOSTIC_2026-10-04 | directive | no-snap stabilisers (Attractor/DEQ, nGPT, Howe, LIFT/NextLat), learned discretisation (FSQ, SR-RNN, Abstract-CoT), self-summarisation (Delethink fixed timing, LightThinker learned, PENCIL per-task traces), learned halting (MoR, CALM); gap: learned what + when in latent space |
| LIT_TASK_AGNOSTIC_PROPOSALS_2026-10-04 | P-A..P-D | FSQ-every-hand-off new as scoped (SR-RNN/QBN closest; Moore minimisation); P-B weaker than P-A; ST-Gumbel notation (1705.11192) not compositional (1706.08502); flip-flop 2306.00946 + Delétang as families; final-answer-only primary |
| LIT_S116_COPY_2026-10-05 | §116 amend | copy = self-supervised autoencoding (1511.01432, 1410.5401); name it, keep strict C4 for NONE; prefer zero-op [EOT, start] segment over marker; log stay rate |
| LIT_S121_CARRY_2026-10-05 | §121 | input-prefix carriage new as diagnosis (2510.00184 recomputes partial products; Liang&Pan late fusion); reader alone can be met by carrying (a,b≤j) — force via teacher-forced S_{j−1} text segments + ANCHOR |
| LIT_S122_TEXTMIX_2026-10-05 | §122 | competent anchor target required: TEXT-MIX = CODI's γ·L_teacher; frozen teacher as fallback; needs a TEXT-MIX-only control; not C4 |
| LIT_S123_DITHER_2026-10-05 | §123 | dither = Ballé 1611.01704 proxy / universal quantisation 2006.09952; ±0.5 ⇒ implicit quantisation, sweep σ; 8-dim continuous no-noise control; normalise ρ |
| LIT_S124_COPY_2026-10-05 | §124 | copy foothold; C4-safe exposure control TEXT-EXPOSE (LM loss on start text only); 4/5 vs historical 3/5 not distinguishable — matched rerun or descriptive + timing signature |
| LIT_S125_STD_2026-10-05 | §125 | scale/distribution-mismatch priors (Lester, prefix-tuning, HRPO, CODI, LT-Tuning, §73); RECON risks nuisance PCs → prefer next-latent prediction; EMA re-estimate at soft→hard switch (Batch Renorm 1702.03275) |
| LIT_S126_ANNEAL_2026-10-05 | §126 | wide→narrow = QBN 1811.12530 (distil), KL annealing; Burgess 1804.03599 is narrow→wide; RAW-H likely plateaus (§108) → supervision not bottleneck; ANNEAL inherits RAW-H to 2k; distil + stochastic λ |
| LIT_COLLAPSE_REMEDIES_2026-10-05 | entry basin | collapse remedies mapped (free bits/δ-VAE, aggressive encoder, word dropout, CPC/Z-forcing aux, annealing, skip-to-decoder, VQ fixes, LR tricks); top: skip-to-decoder 1807.04863, decoder-independent predictive aux, aggressive writer + high LR |
| LIT_ENTRY_MEMO_2026-10-05 | entry memo | VICReg 2105.04906 / PLDM 2502.14819 (variance across time + inverse dynamics) closest; batch variance can be met by item identity → time-wise variance; ranked label-free gradient sources; OPDEF 'sufficient' not 'minimal' |
| LIT_S127_SKIP_STABILITY_2026-10-06 | §127 SKIP | NaN at L16: LayerScale 2103.17239 / ReZero 2003.04887 small non-zero scale, normalise injected term, clip + fp32 skip; post-2k regression = skip bypass → decay skip after entry |
| LIT_S128_FAMILIES_2026-10-06 | §128 | Delétang specs (U(1,40)→U(41,500), final-output score; CYC = Cycle Navigation, MOD = Mod. Arith. Simple + leading 0); new as scoped; CYC counter shortcut → closure/minimal-automaton primary + ≫512 test; MOD ×0 resets shorten dependency; add a seeded random DFA held-out now |
| LIT_S129_HEAD_REDRAW_2026-10-06 | §129 | partial re-init priors (primacy 2205.07802, ReDo 2302.12902, S&P 1910.08475, fortuitous forgetting 2202.00155, RIFLE 2007.03349); keep moments fine; add CLOCK-ONLY control (LR restart confound); optional S&P body arm |
| LIT_S130_MARKER_2026-10-06 | §130 | zb_std_in = active units (IWAE 1509.00519); Dodge/Hyperband 1603.06560 use loss curves; add same-card held-out; ≥8 per class + exact CIs (4/4 at recall 0.8 passes 41%) |
| LIT_S131_FLAG4_2026-10-06 | §131 | learning-progress priors (Graves 1704.03003, TSCL 1707.00183); add family-null floor c·ln K + windowed means; 5/5 → CP lower bound 0.55, accumulate ≥10; out-of-sample pre-filtered by 123-3 |
| LIT_S132_INCONTEXT_TABLE_2026-10-06 | §132 | new as scoped (Akyürek 2401.12973 infers rules from examples; induction heads 2209.11895; PVR 2107.12580); row format = positional 2-hop lookup → use explicit pairs; op-history shortcut at L≤4; pilot k=4,m=2, ladder pre-registered |
| LIT_S133_WIDE_CARRIER_2026-10-06 | §133 | soft-to-hard VQ 1704.00648, Lipschitz RNN 2006.12070, Jacobian reg 1908.02729 (1802.08227 is NOT relevant); JAC global contraction conflicts with isometric carry; CERT may be infeasible at d=64; S5×4 needs register curriculum + S5×2 + FSQ-8@16-levels controls; d∈{16,64} |
| LIT_SELF_ANCHOR_2026-10-06 | Variant 2b | self-decoded anchor new as scoped (§118 gold, §112/§116 hard; back-translation 1511.06709, dual learning 1611.00179, iterated learning 2002.01365, professor forcing); text path untrained → add self-decoded-restart final-answer rollouts; diversity gate; filter by final answer; FF partial |
| LIT_COT_DISTILL_2026-10-06 | candidate 4 | CoT-teacher hidden distillation: ICoT-KD 2311.01460 original, KaVa 2510.02312 stepwise closest (robust to NL traces, supports premise); last token of step state, probe-chosen mid-late layer; distil state subspace or contrastive (2606.20075); constant weight; new as scoped; not C4 |
| LIT_S136_GROW_2026-10-06 | §136 GROW | Net2Net 1511.05641, GradMax 2201.05125, progressive GANs 1710.10196, Burgess 1804.03599, nested dropout 1402.0915, Matryoshka 2205.13147 / MatQuant 2502.06786; new as scoped; zero-in+zero-out is dead (zero only W_up); S5 alone gives no pressure → test on S5×4; nested-training alternative |
| LIT_MULT_PLAN_2026-10-07 | mult plan | no new latent mult results; final-only latent mult new as scoped; b-prefix shortcut stronger under FINAL; output-column-major + bounded carry (≈10 bits, length-general; Cho 2410.15787, Neural GPU) recommended over multiplier-major (26.6/33.2 bits, range) |
| LIT_CAND5_6_2026-10-07 | cand. 5/6 | RL over FSQ codes: Abstract-CoT closest; GRPO 2402.03300 / RLOO 2402.14740 / VIMCO; digit-fraction reward if sparse; local contraction hinge OK (local ≠ global; CAE 1104.4153, 1908.02729); ε as fraction of inter-state distance, pre-quantisation z |
| LIT_DISTILL2_2026-10-07 | DISTILL-2 | probe-decoded state distillation new as scoped (KD 1503.02531, FitNets 1412.6550, CBM 2007.04612); with a perfect teacher ≈ CE on teacher text — check probe accuracy; fade after curriculum reaches full L, log post-fade S_j readout |
| LIT_S140_DIGPAIR_2026-10-07 | §140 | new as scoped (Neural GPU, Cho 2410.15787, Shen 2311.14737, Lee 2307.03381); replace absolute (i,j) with index-free last-in-column flag for 4×8; accumulator bounded for 4×n only; n²+1 compounding |
| LIT_S137_FOLLOWUP_2026-10-07 | §137 2×2 | repeats ≈ Huginn (closest), new through discrete code; without repeat index expect fixed point; r4+index ≈ §140 with full column; FSQ rounding erases small increments; capacity not binding — nonlinear probe before 'compression loss'; predict r4 > SOFT-16 |
| LIT_S138_FULLWIDTH_2026-10-07 | §138 | bounded-vs-raw used (CODI, SoftCoT 2502.12134, HRPO, Soft Thinking 2505.15778, nGPT 2410.01131) but never isolated; FSQ at 768 no precedent (FSQ/LFQ/BSQ ≤ tens of dims); FULL-CONT vs RAW-H = bounding + COPY; exact-recovery share falls with d — pre-register ρ_ff reading; random-δ ρ is a bulk average — add centroid-direction ρ; log init code diversity |
| LIT_MULT_READ_REMOVAL_2026-10-07 | mult READ + removal | ICoT-SI 2405.14838 and Coconut remove earliest-first (smoothing + optimizer reset); no latest-first precedent — and late S_j are the large values 140-3 says need the reader, so earliest-first; closest prior 2510.00184 running-sum aux; vs GEN-M: code-only reader forces carriage, budget still 1.9 epochs; mean-per-digit CE; gated removal after n=4; log removed-S_j retention; B not strict C4 |
| LIT_S143_FULLGROW_2026-10-07 | §143 | no ~100× single-step growth precedent (Net2Net/bert2BERT/LiGO/MSG 2305.02869/FLM-101B ~2× staged); zero-init outgoing fine (ReZero 2003.04887, GradMax) but μP 2203.03466: W_up fan-in 8→768 makes Adam steps ~96× larger in output — scale new-column lr by 8/768; 760 continuous dims lack rounding correction; log new-path share, block ρ |
