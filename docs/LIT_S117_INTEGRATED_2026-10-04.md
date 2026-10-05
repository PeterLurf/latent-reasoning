# Prior check for draft §117: the integrated C1 + C4 + C6 candidate (Markov segments + self-decoded restarts + reader) (2026-10-04)

Literature session, for Theory; Audit copied. Brief. IDs as verified earlier (LIT_INDEX).

## (a) Priors: outcome-trained latent carriers with a bottleneck, and an auxiliary reader against collapse
- **Outcome-trained bottlenecks:** Markovian Transformers 2404.18988 (RL; the answer must follow from the model's own
  CoT); RMT 2207.06881 (memory tokens as the only path, BPTT); Delethink 2510.06557. None uses a latent carrier plus a
  reader.
- **Readers or decoders against collapse:**
  - SIM-CoT 2509.20317 (per-latent decoder; but gold step targets);
  - SELR 2608.13570 (self-decoding);
  - ReLAT 2606.06252 (a reconstruction cycle);
  - MUX 2607.18264 (lossless targets "prevent latent collapse");
  - 2606.20075 (generative reconstruction preserves information);
  - and the VAE posterior-collapse fixes this mirrors: Bowman 1511.06349, free bits 1606.04934.
- **Warning prior:** Allchin 2607.11953 (reward alone rarely teaches state in permutation automata).
- **Not found:** a carrier trained from outcomes only, with a self-target reader, certified on a non-solvable group.
  **New as scoped.**

## (b) Is k = 1 K-CURR enough to replace s21's parallel-shortcut seeding?
- **Mechanism check.**
  - At k = 1 the last op restarts from the model's own decoded checkpoint, so a length-L final answer is a 1-op target
    from a self-decoded state. That is §116's foothold at full strength.
  - But **the target is consistent only when that checkpoint is correct.** Otherwise it is label noise. Early on, with
    random decodes, the consistent fraction is ≈ 1/120 plus whatever the shorts seed.
  - With f = 0.1 (or natural) shorts, the labelled transitions give correct decodes on some prefixes, so consistent
    tails grow from there. Plausible, but slower than s21's in-pass seeding, which Markov removes.
- **Fallback to pre-register:** a **staged bottleneck**. Run the first round(s) unmasked (non-Markov, so the parallel
  shortcut seeds the checkpoints), then switch to Markov for the rest. C1 is scored only on the final model.
  - Analogues: RMT's segment curriculum; capacity annealing of bottlenecks.
  - Trigger it on **final-answer accuracy at lengths > k after round 1** (final answers are allowed under C4).
    foothold_yield compares decoded checkpoints with true intermediate states, so it is truth-scored: log it, but
    don't trigger on it.
- **Log foothold_yield and acc_by_length per round**, as in §116.

## (c) Reader confirmation bias and collapse
- Early on (empty pools) the reader sees only final-slot targets, which are true final answers. That's safe.
- Later it trains on the model's own filtered decodes. Precision of about 0.98 means about 2% of errors get reinforced
  (Arazo 1908.02983).
- **Mitigations:**
  - (i) weight reader targets by the decode's agreement or confidence, or use soft labels;
  - (ii) STaR-style: re-initialise the reader head at each round, so stale errors aren't accumulated;
  - (iii) keep a fraction of reader updates on final-slot targets only.
- **Degenerate agreement** (decoder and reader settling on a shared wrong code) is blocked by the final-answer filter
  only at final slots. Monitor intermediate reader-vs-truth precision (logged, not used).

## (d) C4 wording
- **Safe:** "C4 qualified by f short-problem coverage (random given starts)", or "… by §110's natural shorts". The
  reader targets are self-decodes, teacher forcing uses only kept pseudo-labels, and the final-answer filter uses only
  final answers. All are allowed.
- **Unsafe if:**
  - any hyperparameter, seed or stopping choice uses held-out *intermediate* truth. Log only; report all 5 seeds;
  - the never-labelled k1 metric is used for selection;
  - the staged-bottleneck fallback's switch point is chosen on truth-scored intermediates. Trigger it on
    final-answer accuracy at L > k (foothold_yield is truth-scored on intermediates: log only).
- **C1:** single-slot donor SERIAL, per note 112-3, is the right bar.
