# Prior check for §94: the latent carries the full arrangement ("all" 1.000) but "where is X" fails in 4/5 seeds (2026-10-03)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified via the arXiv API on 2026-10-03.

## 1. Query-specific readout failures from a correct state
- **The closest framing is the reversal curse and inverse search.**
  - The "all" answer reads the state in its natural order (box → letter).
  - "Where is X" needs the **inverse map** (letter → box), i.e. a search over boxes for X.
  - Berglund et al. 2309.12288: models trained on "A is B" don't generalise to "B is A".
  - Allen-Zhu & Li 2309.14402 (Physics of LMs 3.2): retrieval works, but **inverse search fails** unless trained in
    that direction (or with CoT).
  - Golovneva et al. 2403.13799: **reverse training** (data in both directions) fixes the curse.
  - Here both question types *are* in training, so this is not a strict reversal curse. But the same asymmetry
    explains why one readout is learned and the other isn't. The step-decode side loss trains only box → letter.
- **State-probing priors:**
  - Li et al. 2210.13382 (Othello): the board state is present but only nonlinearly decodable.
  - Nanda et al. 2309.00941: it becomes *linear* in the right coordinates ("mine/theirs"). The format of the stored
    state decides which readouts are easy.
  - Kim & Schuster 2305.02363 and Prakash 2402.14811: entity tracking in boxes tasks.
  - Feng & Steinhardt 2310.17191 (binding IDs): LMs answer "which box holds X" by matching a binding vector shared by
    the entity and its attribute. A latent that stores box-ordered contents without accessible letter → box binding
    vectors would support "all" but not "where".

## 2. Longer training or a query-specific target?
- **The literature's standard fix is a target in the needed direction:** reverse training (2403.13799), inverse-search
  data augmentation (2309.14402), or multi-task auxiliary heads. Our own rule 1 (per-step targets) agrees.
- **Longer training alone** is not known to fix reversal or inverse-search failures in LMs (Berglund; Allen-Zhu
  report it persists with scale and data).
- But since "where" questions *are* trained here, a 4/5 failure may also be a multi-task seed lottery. §91/§92-style
  budget fixes are plausible for that.
- So §94's 3× budget is a fair test of the lottery reading only.

## 3. Suggestions for §94
- **The linear probe is the right diagnostic. Make it decisive.** Fit letter → box probes on the final latent of
  failing seeds.
  - **Probe succeeds** (≈ 1.0): the information is there, and only the model's own "where" readout circuit failed.
    That is a readout/optimisation failure, and more budget or a query-specific target should fix it.
  - **Probe fails:** the latent stores the arrangement in a box-ordered format, with no linear letter-indexed access
    (the Othello format point).
  - Also fit box → letter probes as a positive control (should be 1.0, given "all" = 1.000).
- **Add the standard fix as a separate arm** (if budget allows): an inverse-direction step-decode target (letter → box
  per step) alongside the box → letter one, i.e. "reverse training" for the latent. Compare it with 3× budget alone.
- **Report per question type and per seed.** "Latents do the work" (mean ablation → NC floor) stands for "all"
  questions. For "where", C1 is claimed only in seeds that answer it.
