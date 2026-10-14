# Prior-art scan for a seed-stability method: plateau / saddle escape, continuation, symmetry breaking, seed variance (2026-10-13)

Literature session, for Theory (the user's "deep research … a new architecture / mathematical method to combat seed stability", relayed by Audit) and Audit.
- **IDs:** all verified via arXiv abstract pages (2026-10-13).
- **Claims:** from the abstracts unless marked; I have not read the full papers. This note sets the novelty bar; I'll novelty-check Theory's proposal against it.

## Short answer: the bar
- **Generic levers are all taken.** Each already has theory:
  - escape time from a small-init saddle;
  - perturbations or noise to escape saddles;
  - continuation / graduated optimisation;
  - symmetry removal;
  - curricula and sub-task supervision;
  - restarts.
  A proposal that is only one of these isn't new.
- **The gap I found:** almost all theory gives escape *time* (or sample complexity), not the *probability that a seed enters the right basin*.
  - The closest is Zhang 2607.20594's "convergence selection" (empirical: the A5 8/22 lottery, bypassed by budget annealing).
  - I found **no theory of basin-entry probability for a multi-factor zero-initialised branch in a pretrained / looped model.** That's our situation: W_o = 0 fade-in, g_add, the code → state → read chain.
  - A method with a provable bound on success probability, or one that removes the bad invariant set by construction, would be new.
- **Two results map directly onto our failures. They're leads for Theory, not tested claims:**
  1. **Rawal & DeWeese 2605.01288:** the escape time depends on how many factors start small. With r layers at scale ε it is Θ(1/Γ) for r = 1, Θ(log(1/ε)/Γ) for r = 2, and **Θ(ε^{−(r−2)}/Γ) for r ≥ 3**, where Γ = ‖E[yx]‖.
     - Our memory path is a product of factors: the fade buffer (0 → 1 over 500 steps), W_o (0 at init), g_add (calibrated at step 0, later collapsing to about 2e-4), and the write / read projections.
     - **Count how many sit near zero at the effective start.** If three or more do, escape is polynomial in 1/ε and very sensitive to the initial values. That's a lottery mechanism.
     - Caveat: Rawal's result is for small-init layers in a feed-forward net. Applying it to our gated branch is an analogy until someone derives it for that branch.
  2. **Chen, Kunin et al. 2306.04251 (stochastic collapse):** invariant sets of SGD, such as "a unit or branch is off", can be **stochastically attractive**. Attractivity is set by curvature vs gradient noise, and **more noise strengthens it**, even at saddles.
     - "Memory unused, g_add → 2.9e-4" (§169, §171) looks like capture by such a set.
     - Ziyin et al. 2408.15495's **syre** removes symmetry-induced low-capacity states, is model-agnostic, and comes with theory.

## 1. Saddle / plateau escape-time theory
- **Saxe, McClelland, Ganguli 1312.6120:** exact deep-linear dynamics. Plateaus set by the initial scale; orthogonal init.
- **Rawal & DeWeese 2605.01288 (2026):** the escape-time law above, for deep *nonlinear* networks under any smooth activation and differentiable loss. A multi-mode teacher leaves the saddle once, not in stages.
- **Jin et al. 1703.00887 / Ge et al. 1503.02101 / Daneshmand et al. 1803.05999:** perturbed or stochastic GD escapes strict saddles in polylog(d) time.
- **Du et al. 1705.10412:** GD **can take exponential time** to escape a chain of saddles, even with natural random init. Perturbations remove this.
- **Lee et al. 1602.04915:** GD avoids strict saddles almost surely, asymptotically only.
- **Dauphin et al. 1406.2572:** saddles, not local minima, dominate in high dimensions; saddle-free Newton.
- **Hu et al. 2507.19968 (DEO):** first-order dimer curvature estimate; projects the gradient away from the minimum-curvature direction. A toy transformer only.
- **Hierarchical / leap theory:**
  - **Abbe, Boix-Adsera, Misiakiewicz 2302.11055:** time scales as d^{max(Leap, 2)}, with saddle-to-saddle learning of the support;
  - **Ben Arous, Gheissari, Jagannath 2003.10409:** information exponent; "almost all the data is used in the initial search phase";
  - **Barak et al. 2207.08799:** parities at n^{O(k)}, with hidden progress via a Fourier gap invisible to loss.
- **Saddle-to-saddle across architectures:**
  - **Zhang, Saxe, Latham 2512.20607 (ICLR 2026):** includes attention, where heads are added one at a time, via invariant manifolds. It separates data-induced from init-induced saddles;
  - also **Jacot et al. 2106.15933**, **Pesme & Flammarion 2304.00488**, **Zhang, Singh, Latham 2501.16265** (linear-attention ICL: exponentially many fixed points with separate K / Q).
- **Transformer plateaus:**
  - **Hoffmann et al. 2310.12956 (Eureka moments):** multi-step tasks stall on the intermediate step, traced to the **softmax**. The fixes raise the *probability* of learning it and robustness to hyper-parameters;
  - **Gopalani & Hu 2506.13688** (in LIT_PLATEAU_SIGNATURE);
  - **Zucchet et al. 2505.17863:** sparse-attention emergence time follows **power laws** in task / architecture / optimiser; repetition speeds it;
  - **Singh et al. 2404.07129:** induction-head formation;
  - **Boix-Adsera et al. 2306.07042:** gradual rank increase.
- **RNNs:**
  - **Eisenmann et al. 2310.17561:** loss jumps at **bifurcations** (gradients → ∞ or 0, proved for ReLU RNNs). **Generalised teacher forcing avoids** some of them;
  - **Sharma & Vu 2606.24898:** in looped LMs, per-loop CE through an RMSNorm / LayerNorm readout leaves the recurrent **state scale** uncontrolled (norms in the thousands). Relevant to §185 NORM-CAP.

## 2. Continuation / homotopy with guarantees
- **Hazan, Levy, Shalev-Shwartz 1503.03712:** graduated optimisation **provably** reaches a global ε-optimum in O(1/ε²) steps for a defined function family, stochastic setting included.
- **Mobahi 1601.04114:** RNN training by diffusion (Gaussian smoothing), recovering init, annealing and noise injection as special cases.
- **Gulcehre et al. 1608.04980 (Mollifying Networks).**
- **Pathak & Paffenroth 2507.22089:** parameter continuation as principled curriculum.
- **Curriculum theory:**
  - **Saglietti, Sarao Mannelli, Saxe 2106.08068:** in teacher–student models curricula give only **modest** speed-ups, with no generalisation gain under standard losses;
  - **Cornacchia & Mossel 2301.13833** and **Abbe et al. 2306.16921:** **provable** curriculum advantage on parities;
  - **Weinshall et al. 1802.03796.**
- **Sub-task supervision lowers the hardness:**
  - **Kim & Suzuki 2410.08633:** parity is efficient *with* CoT intermediate supervision;
  - **Wies, Levine, Shashua 2204.02892:** decomposition enables learning;
  - **Shalev-Shwartz et al. 1703.07950:** gradient-based failures.
  - These are the theory behind §175's step aux.
- **Path dependence, empirical:**
  - **Yang et al. 2609.10287 (PRICAI 2026):** an annealable attention prior. **Smooth fade-to-zero keeps the circuit (0.734); forced-zero, hard switching and post-hoc continuation fail.** Which heads carry it varies across seeds;
  - **Zhang 2607.20594:** budget annealing (warm start across loop budgets) fixes the A5 lottery 4/4.

## 3. Symmetry breaking
- **Simsek et al. 2105.12221:** symmetry-induced critical subspaces and invariances.
- **Brea et al. 1907.02911:** permutation saddles.
- **Fukumizu et al. 1906.04868:** embedding gives semi-flat minima and saddles.
- **Kunin et al. 2012.04728:** symmetry ↔ conservation laws, broken by weight decay / SGD.
- **Zhao et al. 2210.17216:** conserved quantities of gradient flow.
- **Ziyin 2309.16932:** mirror symmetries create absorbing low-rank / sparse states.
- **Ziyin et al. 2408.15495 (syre):** removes them, model-agnostic, with theory.
- **Chen et al. 2306.04251:** the stochastic-collapse condition (above).

## 4. Seed-variance reduction
- **Frankle et al. 1912.05671:** training becomes stable to SGD noise (linear mode connectivity) only after an early phase, so seed variance is decided early. That supports early diagnostics and early restart.
- **Fort et al. 2010.15110:** a chaotic early phase, then the NTK settles.
- **Summers & Dinneen 2103.04514:** a **single-bit** perturbation produces as much run-to-run variance as changing the seed. Consistent with our GPU-class and cache-rounding basin flips.
- **Jordan 2304.01910:** much test-set variance between runs is harmless or unavoidable. That's about test-set noise, not basin failure, so it doesn't apply to our lottery.
- **Restarts with theory:**
  - the Luby, Sinclair, Zuckerman (1993, IPL; not on arXiv) universal restart schedule is within a log factor of the optimal restart policy for *any* runtime distribution;
  - **Lorenz 1709.10405:** when restarts help (location / scale results);
  - **Cebrian & Cantador 0704.2725:** heavy-tailed MLP training times; restarts cut the expected time by 9–40%.
  - **For us:** if basin-entry time is heavy-tailed, a Luby-schedule restart has a guarantee. Our §126 restart-on-plateau is the in-house ad-hoc version.

## Implications for Theory's proposal
- **Counts as new:**
  - (i) a success-*probability* account of basin entry for our gated memory chain, e.g. with Rawal's r = number of small factors made explicit and then reduced by design;
  - (ii) an init or parametrisation that provably removes the "memory off" invariant set (syre-like, but specific to a zero-initialised gated branch in a pretrained LM);
  - (iii) restart theory combined with an early basin-entry detector (Frankle: decided early).
- **Doesn't count as new on its own:** noise injection, LR warm-up, curricula, gradual fade-in, generic continuation, restarts.
- **Cheap tests the literature suggests:**
  - count the near-zero factors in the memory path at step 0 (Rawal's r);
  - test the stochastic-collapse condition by varying batch size: does the memory-off rate rise with more noise?
  - check the restart-time distribution of past seeds for heavy tails.
