# Compiled findings vs prior work (2026-10-02)

Literature session, at the user's request via Audit. Sources: `docs/NORTH_STAR.md`, `report/findings.md`,
`docs/THEORY.md`, `docs/GSM8K_REPRO.md`, and the dated LIT_* notes. Status: **New** (no prior found), **Rep**
(replicates prior), **Ext** (extends prior to a new setting or scale), **Con** (contradicts or corrects prior).
arXiv IDs as verified in the linked notes.

| Our result (pointer) | Nearest prior | Status |
|---|---|---|
| ProsQA proximity shortcut: a person-agnostic heuristic scores 0.911; on the shortcut-free set our Coconut repro falls 0.974 → 0.738 (0.732 with no thoughts) (§57; NORTH_STAR) | Aswal 2606.12689 (thoughts barely matter on ProsQA); Dilgren 2604.04902 | **New** (the generator cue is undocumented; LIT_PROSQA_SHORTCUT). **Con** for published ProsQA claims |
| ProsQA-NA (deep, no cue): every GPT-2 latent arm at chance (0.486–0.508), text CoT 0.767 (PROTOCOL8 am. 3) | Coconut 2412.06769 (latent > CoT on ProsQA) | **Con** (latent advantage absent without the cue) |
| Toy: readable ≠ usable. Probe directions causally inert; activation patterns carry the effect (findings 1–2) | Hewitt & Liang 1909.03368; Belinkov 2102.12452 | **Rep/Ext** (interventional, on latent thoughts) |
| Toy: clocked BFS sweep; extrapolation predicted in-budget by P7 (AUC 0.993, 66 new runs) (findings 6, 9) | Zhu 2505.12514 (observational superposition); Kuo 2606.29983 (seed-unstable extrapolation) | **New** (prospective in-training predictor) |
| Range principle: every past-length failure is a quantity leaving its trained range; every fix restarts from a written state with history cut (§26, §45, §64–§66; Synthesis) | Ruiz & Gu 2507.02782 (unexplored states); Delethink 2510.06557; PENCIL 2503.14337 | **Ext** (one account across graph, S3, GPT-2 tracking; restarts as the fix) |
| §66: GPT-2 written-state chunking 1.000 at 17–32 ops vs text CoT 0.14; WC₀ (no latents) equal, so the scaffold does the work | Delethink; Markovian Transformers 2404.18988; Anil 2207.04901 (scratchpad fine-tuning fails to length-generalise) | **Ext** (state tracking, matched FLOPs). C1 not shown |
| §71: S5 written-state scaffold certified 5/5, exact to 512 and 2,048 from training at 128 (DeltaProduct protocol); comparator not reproduced on our setup | DeltaProduct 2502.10297 (1-layer, n_h = 4, to 512); Kraus 2604.25800 (transformer CoT 1.7×); Li 2503.02854 (pretrained LM cutoff ≈ train length) | **New** for a pretrained transformer at 512 / 2,048, **scaffolded**. No SOTA claim (comparator not rerun) |
| §70: matched cells, multiplicative 10/10 vs additive 0/10 (descriptive) | Goudreau 1994 (second-order > first-order); Howe 2609.18966 (additive path parasitic) | **Rep** (Howe found it first, 2026-07) |
| §69: RL-style updates don't fix acquisition (0–1/10; best-of-4 5/10); spectral WFA gets S4/A5 from ≤ 10⁶ problems | Guu 1704.07926; Allchin 2607.11953 (reward rarely teaches state); Balle spectral WFA | **Rep** |
| §72b/§73: frozen GPT-2 features 0/5 → per-coordinate standardisation 10/10 certified S3 latent from final answers; random-init 5/5 | Ethayarajh 1909.00512; Timkey 2109.04404 (rogue dimensions; standardisation); MAE 2111.06377 (affine-free BN probe) | **Ext** (known fix, first applied to latent acquisition). Pretraining not shown necessary |
| §74/§79: paraphrase transfer of the certified cell; §79 agreement ≈ 0.75–0.84 (< 0.90), negatives at chance, digit-reliant (being scored) | Hill 2005.09382; Lynch 2005.07648; Kim/Linzen/Smolensky 2212.10769 | **New** measurement (exhaustive per-entry agreement); result partial |
| §76(A): free table (exact marginal) certifies S4/A5/S5 10/10 **with** length-1/2 problems, 0/10 with lengths ≥ 3 only | IO-HMM / Baum–Welch; Xu 1810.11344 (over-parameterised EM); Wies 2204.02892 (intermediate supervision makes composites learnable) | **Ext** (scale to S5); the ge3 failure **Reps** Wies-type theory |
| §82: under ge3 every order-60 group (Z59, Z60, D30, A4×Z5, A5) merges to one block at chance; std 3/3–10/10 | Zhang & Li 2609.29951 (coset stages); Abbe staircase 2202.08658 | **Con** to *my* coset-foothold prediction (4a2aa35) and to a group-structure ladder. Supervision granularity decides |
| §76(B): GPT-2 hosts a certified S4/A5 table (20/20) but held-out rows are at chance | Wang 2405.15071 (OOD composition fails); grokking 2201.02177 | **Rep** (storage without composition) |
| §80 P0: GPT-2 small on ≤ 815 k-op words with end-state targets: unseen words 0.03–0.12 (memorises); seen words on new starts 0.80–1.00 | Lin 2503.07604 (implicit reasoning = shortcuts); Wang 2405.15071; Liu 2210.10749 | **Rep** at small data scale (note 80-2) |
| GSM8K repro: CODI 42.5 (reproduced), SIM-CoT+Coconut 44.8 (reproduced), SIM-CoT+CODI 39.4 (just outside); ablations: SIM-CoT+Coconut 0.448 vs 0.051 with 0 thoughts; CODI 0.425 vs 0.221; one-op oracle 15.5 | 2502.21074; 2509.20317; Dilgren 2604.04902 | **Rep**; the ablations show the latents carry work there (step-supervised, not C4) |
| Toy: silent pointer editable (IIA 0.36); round trip 0.947 (§34) | NORTH_STAR survey: no published behavioural round-trip / edit test on latent states | **New** (partial C7) |
| Toy: restarts fix 31/34 failures; distilled into one loop label-free 19/20 (§26, Pilot L) | Self-training / STaR 2203.14465; Delethink | **New** in mechanism (a fall-back keyed to a count, fixed by restarts) |

## What we can honestly claim as new
1. **A benchmark correction:** ProsQA's generator shortcut (0.911) and latent arms at chance without it.
2. **A unifying mechanism** (the range principle) with an interventional account of why loops fail past training
   length, and a prospective in-training predictor of extrapolation (P7).
3. **A scaffolded pretrained-transformer result:** written-state chunking certified exact on S5 to 2,048. New for
   transformers, but the scaffold, not latent computation, does the work.
4. **Acquisition boundaries:** exact-marginal tabular learners acquire S4–S5 only with directly labelled short
   products. Algebraic structure doesn't matter at order 60. LMs host but don't compose the acquired tables. These are
   clean negative and boundary results with certificates.
5. **Not new:** multiplicative over additive transitions (Howe), standardising anisotropic features (Timkey/MAE), and
   storage without composition (Wang).
