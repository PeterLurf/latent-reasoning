# Prior check: RL-style acquisition of discrete latent rules (permutation automata) from final answers (2026-10-01)

Literature session, for Theory and Audit (both asked; brief in `EXTERNAL_PROMPT_RL_2026-10-01.md`). Committed before
registration. arXiv IDs verified via the arXiv API. Non-arXiv classics are standard references, not API-verified.

## Direct answer: what has evidence of escaping optima where marginal likelihood fails?
1. **Learning discrete programs from final answers** (the closest structural analogue: latent discrete paths, reward =
   correct answer). The pattern across these papers: **hard-EM plus memory and targeted exploration beats both pure
   marginal likelihood and pure RL**.
   - **Guu et al. 2017, *Bridging RL and Maximum Marginal Likelihood*, 1704.07926:** MML and RL both suffer from
     *spurious* latent programs and poor exploration. Their **RandoMer** (randomised beam search for exploration plus a
     "meritocratic" β-weighted update that flattens the posterior over correct programs) beats both. The closest
     analogue to "partial automata that fit the answers".
   - **Neural Symbolic Machines** 1611.00020: REINFORCE alone fails to start, so they **bootstrap with iterative
     maximum likelihood (hard-EM) on found high-reward programs**, then mix the two.
   - **MAPO** 1807.02322: a **memory buffer of high-reward trajectories** plus systematic exploration, with the gradient
     split between buffer and samples. SOTA over MML and RL on WikiTableQuestions. The "keep correct code paths"
     version of expert iteration.
2. **HMM/EM local optima.** EM for HMMs is known to get stuck: Johnson 2007 (EMNLP, *Why doesn't EM find good HMM
   POS-taggers?*). Escapes with evidence:
   - **annealing** (deterministic annealing EM, Ueda & Nakano 1998; Smith & Eisner 2004, ACL), matching your
     temperature 4 → 1;
   - **Bayesian / Gibbs sampling** (Goldwater & Griffiths 2007, ACL), a stochastic EM analogue;
   - **Viterbi (hard-EM) training** beats EM for unsupervised dependency parsing (Spitkovsky et al. 2010, CoNLL), so hard
     EM can help, but it is task-dependent;
   - **over-complete states and split–merge** (Petrov et al. 2006, latent PCFG state splitting), matching "12 codes
     certify 10/10".
3. **Methods with global guarantees** (no local optima): **spectral learning of HMMs** (Hsu, Kakade & Zhang 0811.4413) and
   **spectral learning of weighted automata** (Balle et al. 2014, *Machine Learning*; the Hankel matrix of string →
   value). Spectral WFA learning needs only *final values of strings*, i.e. your final answers, and recovers a minimal
   automaton. **A strong non-neural baseline or initialiser for S3/S4/A5 that any RL method should be compared
   against.**

## Other priors asked about
- **LaTRO** (Chen et al. 2411.04282): variational latent-reasoning optimisation with a self-reward (log-likelihood of the
  answer given a sampled rationale). It is the ELBO/RL bridge on text rationales, not discrete automata.
- **RL on latent reasoning (continuous):** LEPO 2604.17892, Latent-GRPO 2604.27998, SofT-GRPO 2511.06411 (Gumbel-
  reparameterised soft thinking), TARPO 2606.05859, SWITCH 2606.13106, Soft Tokens Hard Truths 2509.19170. All address
  exploration in *continuous* latent CoT on LLM benchmarks. **None targets certified discrete automaton acquisition.**
- **Gradient estimators for discrete latents:** REINFORCE (Williams 1992), VIMCO 1602.06725 (a multi-sample
  leave-one-out baseline, the precursor of group baselines), REBAR 1703.07370, RELAX 1711.00123, Gumbel-softmax
  1611.01144. Note that exact marginalisation already gives the *exact* gradient. The issue is landscape, not variance,
  which these papers don't address.
- **Expert iteration / wake–sleep:** Anthony et al. 1705.08439; DreamCoder 2006.08381; STaR 2203.14465.
- **Grammatical / automaton induction:** learning a minimal consistent DFA is NP-hard (Gold 1978); second-order RNN DFA
  induction (Giles et al. 1992); SR-RNN 1901.08817; automaton extraction 1711.09576; compound PCFG 1906.10225.

## Recommendations for the registration
- **Arms to include:**
  1. MAPO/NSM-style **hard-EM with a buffer of certified-correct code paths** (the evidence-backed escape);
  2. **RandoMer-style meritocratic weighting** (flattening over correct paths, aimed at partial-automaton degeneracy);
  3. an **annealing control** (your 4 → 1) to separate exploration from annealing;
  4. a **spectral WFA baseline** from final answers (non-neural, global).
- **Pass rule:** certified on ≥ 9/10 held-out seeds per group, as the brief proposes, plus updates-to-certificate
  against exact marginalisation.
- **Likely failure:** spurious-but-rewarded automata (Guu). For S3 with uniform products, the answer only constrains
  the composed product, so many partial automata fit sampled paths. Track the certificate, not reward.
