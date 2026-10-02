# S5 state tracking: published protocols, comparator code, nearest priors for §71 (WC on S5) (2026-10-01)

Literature session, for Theory; Audit copied. Committed before §71 registration. All arXiv IDs verified (Semantic
Scholar batch, 2026-10-01). Protocol details were read from the papers' HTML/PDF and, for DeltaProduct, from the code.

## 1. Exact published protocols
| | DeltaProduct 2502.10297 (NeurIPS'25) | Grazzi 2411.12537 (ICLR'25) | Merrill 2404.08819 (ICML'24) | PD-SSM 2509.22284 (NeurIPS'25) | Howe 2609.18966 (preprint, 1 author) |
|---|---|---|---|---|---|
| Inputs | **all group elements**, uniform (`random.choices` over all 120 for S5); one token each | S5 variants: all 120, swaps only (≤2 moved), ≤3 moved, or 4 tokens per permutation | all 60 elements of A5 / A4×Z5 / Z60 (**S5 not run**) | (S5, n): n generator permutations (n = 4, 8, 32) | **2 generators** (transposition + 5-cycle) |
| Train length | **exactly 128** (`strict_len=True`), no curriculum; 100k samples, 99% train | 32 (64 for the 4-token variant) | each n trained separately (plus all 3,600 length-2 pairs) | ≤ 40 | 32; 4,000 sequences |
| Supervision | **every position**: 120-way prefix product (tagging) | every position (prefix products) | every position | FSA state (Walker et al. format) | every position: **one element's position (5-way)** |
| Test | length 512 (100k samples generated) | lengths beyond training (figure) | same length n | 40–256 | 512, n = 1,000 |
| Metric | **cumulative sequence accuracy**: at x = t, the fraction of sequences with *all* positions ≤ t correct (code: `cumulative_sequence_accuracies`) | full-sequence accuracy | full-sequence accuracy; minimum depth reaching 90% | best validation accuracy, 3 seeds | per-position accuracy at 512; GEN = median > 0.9, 5 seeds |
| Model | 1 layer, d = 128, 8 heads × head_dim 32, no conv/gates, bf16; AdamW lr 1e-3, cosine, batch 2048, 100 epochs | DeltaNet[−1,1], 1 or 5 layers | 1-layer RNN / IDS4; Transformer, S4, Mamba depth sweep | 1 layer, state 128, 100k steps, batch 256 | 1 layer Householder, d = 112, Adam 3e-3, 40 epochs |
| Best S5 | **n_h = 4, 1 layer: robust to 512** (figure only, no table). n_h = 1 unsolved even at 10 layers | full S5 ≈ 0 beyond training; swaps-only learned (1 layer); 4-token variant 5 layers OK to ~200 tokens (~50 elements) | — | (S5,4) 100.0, (S5,8) 100.0, **(S5,32) 1.07** (Gated DP n_h=4: 88.7 / 57.4 / 1.23) | without additive input: **1.00 at 512 (5/5)**; with it 0.20. **GRU reference generalises everywhere** |

- **Audit's "train 128 / test 512, all-position" is right for DeltaProduct.** It is not Grazzi's protocol (train 32) or
  Merrill's (no extrapolation, no S5). "All-position" here means a stricter cumulative metric: one error anywhere
  before t fails the sequence at t.
- No paper tabulates S5 at 512. DeltaProduct's number has to be read from Figure 5 or regenerated with their code
  (recommended, §2).
- **Framing risk for "beat SOTA":** at the extrapolation end, S5 is already at ceiling for recurrent models. Merrill
  shows a 1-layer RNN solves non-solvable groups at any length. Howe reports a GRU and a Householder RNN without
  additive input at 1.00 at 16× the training length. So "beat SOTA on S5" can only mean (a) best *transformer /
  pretrained-LM* extrapolation, or (b) the same accuracy with a readable state, fewer samples or less compute. It cannot
  mean higher accuracy than the best RNN.
- **Transformer-family bests (the real bar for GPT-2 WC):**
  - Kraus et al. 2604.25800: GPT-2-architecture 25M from scratch, signpost CoT, train ≤ 50, generalises to **1.7×** on S5.
  - Zhang 2607.20594: looped, operator-first, ≥ 0.92 exact match at 2×.
  - Li, Guo & Andreas 2503.02854: pretrained Pythia/GPT-2 fine-tuned on S5, 120 tokens, all-prefix labels, train 100.
    Accuracy holds to about the training length, then hits a **steep cutoff**.
  - Nothing found at 512.

## 2. Running the comparators ourselves
- **DeltaProduct** (`automl/DeltaProduct`, **MIT**):
  - `state_tracking/` is a fork of Merrill's `jopetty/word-problem` (MIT) and bundles a flash-linear-attention snapshot
    (fla-org, MIT; the maintained `GatedDeltaProduct` lives there now).
  - Commands: `generate_data.py --group=S5 --k=128 --samples=100000`, the same with `--k=512` for test, then
    `main.py train --group=S5 --k=128 --k_test=512 --n_layers=1 --num_householder=4 --allow_neg_eigval=True --n_heads=8
    --batch_size=2048 --lr=1e-3 --epochs=100 --use_scheduler=True`.
  - The model is tiny (d = 128, one layer) and fits easily on 16 GB.
  - **It needs Triton (> 3.1), so Linux/CUDA: a Vast host or WSL, not native Windows.** Seeds: the paper's count is not
    stated; run ≥ 3.
- **Merrill word-problem** (`jopetty/word-problem`, MIT): RNN, IDS4, S4 and Mamba scripts. Use it for a 1-layer
  RNN/GRU reference.
- **Grazzi** (`automl/unlocking_state_tracking`): **no licence file** (GitHub API reports none). Use internally only;
  don't vendor it into bench. Its README covers parity and modular arithmetic; S5 runs go through the word-problem
  code anyway.
- **PD-SSM** (`IBM/expressive-sparse-state-space-model`, Apache-2.0).
- **Li et al.** (`belindal/state-tracking`, MIT): the pretrained-LM S5 fine-tuning setup. **Closest to our interface.**
- **Howe:** no public repo link found. Looped-TF (Fan 2409.15647): no licence; no S5 task.

## 3. Pretrained LM + written state per step/chunk + history cut: nearest priors
- **Pretrained LM on S5, no written state:** Li, Guo & Andreas 2503.02854 (above). No extrapolation; this is the
  direct NC-style baseline in the literature.
- **CoT/scratchpad on S5 (from scratch):**
  - Li et al. 2402.12875 (ICLR'24): partial compositions as CoT. An expressivity study at fixed lengths, no
    extrapolation.
  - Kraus 2604.25800: signposts give 1.7×.
- **Pretrained LM + scratchpad and length generalisation:** Anil et al. 2207.04901 (NeurIPS'22). Fine-tuning with a
  scratchpad fails to length-generalise at any scale; in-context scratchpad helps. Self-Notes 2305.00833 (interleaved
  state notes on toy tracking tasks). Scratchpads 2112.00114.
- **Chunked computation with a written carry-over and history cut:**
  - PENCIL 2503.14337 (reduction rules erase context); Xu et al. 2604.25166 (PENCIL scaffolding, bounded context,
    MicroPy programs, OOD to SAT/arithmetic).
  - Delethink 2510.06557, InftyThink 2503.06692, Markovian Transformers 2404.18988.
  - Latent carry-over: RMT 2207.06881, Block-Recurrent 2203.07852.
  - State passing for recurrent length generalisation: Ruiz & Gu 2507.02782 (initialising from other sequences' final
    states, the analogue of WC's random-start training).
- **Closest prior to "a short-problem solver chained over chunks with a written state":** PENCIL / Xu 2604.25166,
  which use bounded context with a written reduced state. **None evaluates S5/permutation tracking, and nobody shows
  S5 at 512 with a transformer or pretrained LM in any format.** WC at 512 would be new; the scaffold is not.

## 4. Matched-interface caveats
1. **Task format pins difficulty.**
   - DeltaProduct: 120 element tokens, 120-way state at every position.
   - Howe: 2 generators, 5-way readout.
   - PD-SSM: fails at 32 generators.
   - Use the **DeltaProduct format** (uniform over all 120 elements) as primary. Transposition-only inputs (as in §66)
     match Grazzi's *easier* swaps-only variant and must be labelled that way.
2. **Metric.** Score WC and every comparator on (i) final-state accuracy at 512 *and* (ii) DeltaProduct's cumulative
   all-position accuracy. WC writes a state only every k operations, so (ii) requires k = 1 or per-step decoding. State
   which one is used. Regenerate DeltaProduct's curve from its outputs rather than reading the figure.
3. **Error compounding.** With independent chunk errors, accuracy at 512 ≈ p_chunk^(512/k). Reaching 0.9 at k = 4
   (128 chunks) needs p_chunk ≥ 0.9992; at k = 8, ≥ 0.9984. Pre-register a per-chunk accuracy check on fresh random
   starts.
4. **The discrete written state is an advantage *and* a scaffold.** The 5-symbol arrangement carries the full S5 state
   exactly and snaps away drift. An LRNN must carry the same information in a finite-precision continuous state.
   Fixed chunk boundaries, streaming the input chunk by chunk and an external loop make WC an RNN with a GPT-2 cell
   (cf. RMT / PENCIL). Report it as scaffolded, as §66 already does.
5. **Supervision and data budget.**
   - DeltaProduct: 99k × 128 = 12.7M labelled positions per epoch × 100 epochs, always from identity.
   - WC: final-state labels on k′ ≤ k chunks with **random start states**, a state-conditioned augmentation the
     comparators lack (cf. Ruiz & Gu).
   - Match or report labelled-state counts and operations seen. Consider DeltaProduct trained with random start states
     as a fairness arm.
6. **Size and compute.** GPT-2 small (124M) vs a ~10⁵-parameter 1-layer LRNN. Report parameters and inference FLOPs at
   512. "Same accuracy at 10³× the parameters" is not a SOTA claim. The defensible claims are "first transformer /
   pretrained LM at 512" and "readable, editable state" (C4/C7).
7. **Context.** 512 S5 operations written as text far exceed GPT-2's 1,024-token window. NC/TC/CL arms cannot be scored
   at 512 without chunking, so at 512 the transformer comparators are only Li-style cutoffs or the same chunked
   scaffold.
