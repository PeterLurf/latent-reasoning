# Prepared priors: state tracking (variable reassignment, permutation composition) with an overwritable public state (2026-10-01)

Literature session, prepared ahead of Theory's novelty request (Audit's heads-up). IDs verified (arXiv API). A final
check against Theory's exact design follows when it is drafted.

## Theory of state tracking (what the task demands)
- Transformers without CoT cannot track non-solvable group state (S5) in TC⁰: CoT expressivity 2310.07923 (Merrill &
  Sabharwal); *The Illusion of State in SSMs* 2404.08819 (SSMs share the limit).
- Shortcuts: *Transformers Learn Shortcuts to Automata* 2210.10749 (log-depth composition; brittle OOD).
- Recurrent fixes: negative eigenvalues in linear RNNs 2411.12537; DeltaProduct 2502.10297 (Householder products); PaTH
  attention 2505.16381.
- **Looped / implicit:** *Implicit LMs are RNNs* 2502.07827 (iterate to a fixed point; "superior state-tracking on
  regular languages", up to 1.3B). **Zhang et al. 2607.20594**: weight-tied loops on group word problems (S4, S5, A5);
  the budget law; A5 learned, S5 deadlocks without an operator-first curriculum. **The closest looped precedent for
  permutation composition.**
- Counting needs inductive biases: Chang & Bisk 2405.20131.

## Length generalisation in looped / recurrent models
Fan 2409.15647; Kuo 2606.29983; Zhang 2607.20594; MLP-LDRU 2605.26035 (100% OOD on 18/21 regular languages); Liang
2609.33144; Labovich 2604.15259; Saunshi 2502.17416; Shapiro 2608.11233 (retrofit, ~1.5×).

## Overwritable public / discrete scratchpad state
- Looped transformers with an explicit scratchpad *region in the input* overwritten each loop: Giannou et al. 2301.13196
  (programmable computers). **The closest architectural precedent for an overwritable public state.**
- Scratchpads (append-only text): Nye et al. 2112.00114.
- Erasure and overwrite in text: PENCIL 2503.14337.
- Masked diffusion (per-position discrete states refined over passes): 2510.13117 (MDM ≡ padded looped).
- Learned external memory: NTM 1410.5401.
- Variable binding in LMs: Feng & Steinhardt 2310.17191 (binding IDs); entity tracking 2305.02363; EntNet 1612.03969.
- Our F1 / F1-S (per-entity labels).

## Arithmetic length-generalisation baselines (follow-up)
- Abacus embeddings 2405.17399.
- Position coupling 2405.20671.
- Multi-operand / length and count 2410.15787.
- Jelassi et al. 2306.15400 (relative PE: 5 → 15-digit addition).
- Zhou et al. 2402.09371 (2.5×, fragile to seed and data order).
- Self-improvement 2502.01612 (10 → 100 digits).
- RASP-L 2310.16028; NoPE 2305.19466.

## Likely novelty boundary (pending the design)
State tracking with looped models (Zhang), fixed-point models (Schöne) and overwritable scratchpad regions (Giannou)
all exist. The likely new part is an **overwritable public, readable state trained from final answers only, with C7
edit tests and path-independence / age audits**, transferred to GPT-2.
