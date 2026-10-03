# Prompt for an external agent: RL-style training to acquire discrete latent rules from final answers, 2026-10-01

The agent can read the private GitHub repo `PeterLurf/latent-reasoning` (main branch). Its reply goes to a dated
EXTERNAL_IDEAS file and needs Literature's check and a THEORY registration before anything runs.

---

**Context.** You helped before with `docs/EXTERNAL_IDEAS_C1_2026-10-01.md`. We built your Design 1 as THEORY §67 / §67b:
a discrete-code latent with GPT-2 small as the transition cell, trained by exact forward marginalisation over code paths
from final answers only, run hard at test, and audited with a product-automaton certificate. The task is permutation
tracking (S3, then S4/A5). Please read these in the repo before answering:
- `docs/NORTH_STAR.md` (goals C1–C7 and status) and `report/findings.md`;
- `docs/THEORY.md` §61/§61b, §64–§67b and the "range principle" synthesis;
- `c1/core.py`, `c1/train.py`, `c1/PILOT.md`, `c1/PILOT_67b.md` (GPT-2 cell, objective, certificate);
- `theory/c1_diag.py`, `theory/c1_group.py`, and `theory/results/c1_*.json` (toy cells and group scaling);
- `docs/LIT_SOTA_TARGETS_2026-10-01.md` (published comparators and protocols).

**What we found (all independently re-checked).**
1. The obstacle is acquisition, not representation or drift. Hard discrete codes don't decay: certified models are
   exact from N = 8 to 128. But exact-marginal training from final answers mostly lands in partial-automaton optima:
   - a free 6-state lookup table certifies in 1/6 runs;
   - an MLP cell in 0/6.
2. Over-complete codes help:
   - the table with 12 codes certifies 10/10;
   - a bilinear (state × input) cell with 24 codes and temperature annealing 4 → 1 certifies 10/10 on selection seeds
     but only 8/10 on held-out seeds;
   - the GPT-2 cell certifies in 3/10 with 12 codes. The improved recipe (§67b) certified its pilot seed, but at
     least 2 of the first 3 panel seeds fail.
3. It doesn't scale. The same learner on S4 (24 states) and A5 (60 states) certifies 0/3 each, including with Zhang's
   operator-first curriculum on S4 (best seed 0.95 at every length, but not exact).
4. A separate GPT-2 label-channel model (F1-S) learned exact 14-hop graph search once in 15 runs. The other runs lock
   into a "mark too much" attractor, decided early by GPU numerical noise. It's the same two-attractor lottery.

**Our hypothesis.** This is an exploration/optimisation problem, not a gradient-information problem. Exact
marginalisation already gives the exact likelihood gradient, yet it converges to symmetric or partial optima, as in
classical HMM/EM. RL-style training changes how the search explores. Candidate techniques:
- hard-EM or expert iteration (STaR-like): sample code paths and train on those that reach the correct answer;
- GRPO or REINFORCE with group-relative baselines and entropy bonuses over code actions;
- best-of-n restarts selected on training reward;
- hybrids, e.g. exact marginalisation then a sampling phase to escape plateaus.

**What we want from you.** Propose 2–4 concrete training procedures, each from final answers only, that should raise
seeds-to-certificate on S3 and make S4/A5/S5 acquisition work, first for the cheap toy cells, then for the GPT-2 cell.
For each, give:
1. the exact algorithm: estimator, baseline, exploration schedule, and how it uses or replaces the exact marginal;
2. why it should escape the partial-automaton optima we observe; diagnose them from our result files if you can;
3. the cheapest decisive experiment: cell, group, seeds, update budget, and a pass rule fixed in advance, e.g. ≥ 9/10
   certified on held-out seeds;
4. the nearest prior work (latent-reasoning RL, LaTRO-style variational methods, RL on discrete bottlenecks, grammar or
   automaton induction) and how this differs;
5. its most likely failure mode.

Also tell us whether we're mis-diagnosing. For example, is the bilinear cell's success really about multiplicative
state-input interaction, and is there a parametrisation under which exact marginalisation itself scales? Keep claims
narrow: the scaffold (one operation per call, a supplied interface) is a stated limitation, not a contribution. Be
explicit about uncertainty, and verify any arXiv IDs you cite.
