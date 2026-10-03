# Findings — what stands (condensed 2026-10-03)

Plain-language summary of verified results, newest phase first. Every line is Audit-reproduced from raw files; the
numbers and full arguments are in the cited THEORY § / PROTOCOL scorecard. The detailed Parts I–VIII write-up (findings
1–14, the revision table, training reliability, caveats) is kept unchanged in
`archive/findings_parts_I-VIII_to_2026-09-30.md`.

## Latent restarts on state tracking (GPT-2 small, 2026-10-02 →)

1. **A pretrained GPT-2 can carry an S5 state in latents, exactly, far past training length.** One latent per
   operation, trained only through a side decoder of the state (never fed back), snapped to a discrete checkpoint every
   k ops with the history cut: trained at 128 ops, exact at 512 and 2,048 in 5/5 seeds; with k = 1 certified correct on
   all 14,400 (state, op) inputs, so exact at every length. Zeroing the latents drops accuracy to ≤ 0.003. Cost: 0.29–0.45×
   the compute of writing the state as text. *§92.*
2. **It needs no hand-built state.** With the side decoder predicting the step's text and the checkpoint re-embedding
   the decoded text, the result holds (3/3 exact at 2,048; k = 1 certified; checkpoint edits followed 100 %), at 0.56× the
   written state's compute. Qualifier: under mean ablation the text decoder partly recomputes the state from the visible
   operations (0.21–0.30). *§97.*
3. **Its step labels can come from final answers, through a separate learner.** A lookup-table automaton trained on
   final answers (≤ 5 % short problems; told the number of states) learns S5 in 4/5 seeds and reproduces §92's per-step
   labels exactly. Pipeline-level C4; the language model itself learns no steps. *§96.*
4. **Per-step targets are load-bearing.** Labelling only chunk ends fails (0/6), so do single-step footholds up to 50 %
   of updates (0/9), RL, and annealing; a short-problem warm start works only as an entry into the right solution.
   *§§69, 85, 87, 93, 95.*
5. **Training budget turns seed lotteries into reliability.** Looped composition (§91, 15/15 at 4×), latent restarts
   (§86 fail → §92 pass at 3×), and the fade recipe's "where" questions (§89 1/5 → §94 5/5 at 3×).
6. **The user's fade-to-latent recipe works in range.** Train text CoT, fade its inputs into standardised hidden states,
   keep a side decoder: pure-latent answers at 1.000 on both question types in 5/5 seeds, removal to the no-CoT floor,
   +0.40 over Coconut's stages at matched compute; no length extension without restarts. Closed. *§§89, 94.*
7. **Confidence cannot trigger restarts.** The certified solver is overconfident out of range and misses 32 % of its own
   first errors. *§83.*

## Latent automata in GPT-2 (2026-10-01/02)

8. **Discrete latents learn from final answers once GPT-2 features are standardised** (0/5 → 10/10 on S3); the
   obstacle was anisotropy, not the objective. *§§72b, 73.*
9. **Without short problems the failure is optimisation, not information:** started near the true automaton the table
   stays there (5/5), from a random start it does not. *§85.*
10. **Pretraining helps tell operations apart, not transfer to new phrasings** (copy vs swap 0.40–0.52 vs 0.97 random-init;
    no held-out phrasing family transfers). *§79.*
11. **In-pass composition of known steps into unseen sequences needs per-step targets, word diversity and budget**
    (looped GPT-2: 1.000 at k = 3; 0.13 with too few words; 5/5 at 4× budget). *§§88, 90, 91.*

## Written-state restarts and GSM8K (2026-10-01/02)

12. **Restarting from a written state with the history cut fixes length on permutation tracking** (S3 0.995; S5 exact
    to 2,048, certified); the gain is the scaffold, not latent computation. *§§64–66, 71.*
13. **It does not extend GSM8K reasoning at GPT-2 small:** trained on 1–2 steps, everything is ≈ 0.02 on 5+ steps;
    long problems fail from the first step, not at a halting or depth wall. *§§78, 84.*
14. **Published GSM8K latent checkpoints do use their latents somewhat** (SIM-CoT 0.448 vs 0.051 with no thoughts; CODI
    0.425 vs 0.221). *GSM8K_REPRO.*

## Coconut GPT-2 on ProsQA (2026-09-27/30)

15. **ProsQA has a shortcut:** a proximity heuristic with no reasoning scores 0.91. Without it, published Coconut falls
    0.974 → 0.738 (0.732 with zero thoughts) and our latent arms are at chance while text CoT holds 0.77. *§57; PROTOCOL8.*
16. **Coconut's thoughts on ProsQA carry a one-pass answer, not a hop-by-hop computation,** and untrained models ignore
    written checkpoints. *PROTOCOL6 addenda D–N; PROTOCOL8b.*

## Toy graph search (Parts I–VI, 2026-09-23/29)

17. Alternatives in one thought are separately editable but weaken with width; probe directions are causally inert
    (readable ≠ usable); the budget is split per frontier edge.
18. Recurrent models run a clocked breadth-first sweep; failing loops stop at their training depth because they fall
    back on a step count. Whether a model extrapolates is predicted inside the budget (P7 ranking, AUC 0.993).
19. Restarts fix the horizon (31/34), and can be distilled into one loop without labels (19/20).

## Caveats

* Strong results are on synthetic tracking tasks with a scaffold (fixed restart points, external loop); natural tasks
  (maths, planning) are untested for the restart recipe (§98 on multiplication is running).
* C4 is met only for the pipeline, with short problems and an oracle state count; C5 is not met.
* Small models (GPT-2 small, 124M); 3–5 seeds per arm.
