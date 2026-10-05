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
