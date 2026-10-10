# Reading list for the user (2026-10-10)

Literature session, at the user's request. Curated from about 60 LIT notes; every arXiv ID was verified in those notes. ★ = read first (the core 10).
The order within each section is the suggested reading order. "Why for us" ties each paper to our project.

## 1. The field and its problems (start here)
- ★ **Coconut**, [2412.06769](https://arxiv.org/abs/2412.06769). The reference latent-reasoning method (raw hidden state fed back). Table 1 is
  the source of most of our baselines. Why for us: our main comparison and our ProsQA / GSM8K setting.
- ★ **SIM-CoT**, [2509.20317](https://arxiv.org/abs/2509.20317). A per-latent auxiliary step decoder. Why for us: the template for our STEP-AUX
  (§175).
- **CODI**, [2502.21074](https://arxiv.org/abs/2502.21074). Self-distillation into latents. Why for us: the second published GPT-2 baseline.
- **PCCoT**, [2506.18582](https://arxiv.org/abs/2506.18582). Parallel (Jacobi) latent thoughts. Why for us: the best GPT-2 GSM8K latent number
  (49.5), and §172 probes it.
- ★ **The Illusion of Superposition?**, [2604.06374](https://arxiv.org/abs/2604.06374). Fine-tuned pretrained latents are unused (a shortcut);
  from-scratch ones are used. Why for us: explains why our pretrained GPT-2 ignores think passes.
- **Final Checkpoints Are Not Enough**, [2607.06648](https://arxiv.org/abs/2607.06648). Causal use can *decline* during training while accuracy
  rises. Why for us: track C1 over training.
- **Interpreting Latent CoT as Dynamical Systems**, [2607.09698](https://arxiv.org/abs/2607.09698). Coconut expands; CODI is an attractor.
  Why for us: the same picture as our ρ / contraction measure.
- **Observable Patterns Are Not Explanations**, [2606.12689](https://arxiv.org/abs/2606.12689). BFS-like latent patterns also appear in
  controls. Why for us: why we insist on causal (donor) tests.
- ★ **Not All Thinking is Created Equal (ProsQA-Ext)**, [2609.35643](https://arxiv.org/abs/2609.35643). Deep, shortcut-controlled ProsQA;
  latents beat CoT on longer hops. Why for us: the external reference for deep ProsQA.

## 2. Recurrence, looping and stability
- ★ **Huginn (recurrent depth)**, [2502.05171](https://arxiv.org/abs/2502.05171). A looped LM with random loop counts and input injection.
  Why for us: the recipe behind RAND-K and INJECT.
- **Retrofitted recurrence (McLeish)**, [2511.07384](https://arxiv.org/abs/2511.07384). Converting pretrained LMs to looped models.
  Why for us: shows pretrained init helps when loops are forced from the start.
- **Retrofitting recurrent depth**, [2608.11233](https://arxiv.org/abs/2608.11233). Step-per-loop supervision on a pretrained model; about
  1.5× depth. Why for us: the closest precedent to STEP-AUX on a pretrained LM.
- ★ **Looped Transformers for Length Generalization (Fan)**, [2409.15647](https://arxiv.org/abs/2409.15647). Per-problem loop counts plus input
  injection. Why for us: the base for STEP-MATCHED and INJECT.
- **Logical Extrapolation Without Overthinking (Bansal)**, [2202.05826](https://arxiv.org/abs/2202.05826). "Overthinking", fixed by
  recall and progressive training. Why for us: our K′ > K degradation.
- ★ **Path Independent Equilibrium Models (Anil)**, [2211.09961](https://arxiv.org/abs/2211.09961). Converging regardless of the start
  predicts generalisation to harder problems. Why for us: the top lever for C6 / C2.
- **Stabilizing Extrapolation via Stochastic Stopping (Kuo)**, [2606.29983](https://arxiv.org/abs/2606.29983). A length ↔ loop-count
  correlation causes unstable OOD behaviour. Why for us: randomise K.
- **RecurTrace**, [2609.03379](https://arxiv.org/abs/2609.03379). An oracle-supervised halt head; ACT / PonderNet collapse to one loop.
  Why for us: how (not) to learn halting.
- **Loop, Think & Generalize (Kohli)**, [2604.07822](https://arxiv.org/abs/2604.07822). Threshold depth curricula and three-stage
  grokking. Why for us: a training schedule for depth extrapolation.

## 3. Memory and state tracking (DeltaMem's background)
- ★ **Unlocking State-Tracking in Linear RNNs (Grazzi)**, [2411.12537](https://arxiv.org/abs/2411.12537). Negative eigenvalues → parity,
  modular counting, regular languages. Why for us: the theory behind the delta memory.
- **Gated DeltaNet**, [2412.06464](https://arxiv.org/abs/2412.06464), and **DeltaProduct**,
  [2502.10297](https://arxiv.org/abs/2502.10297). Why for us: the layer families DeltaMem builds on.
- **Understanding and Improving Length Generalization in Recurrent Models**, [2507.02782](https://arxiv.org/abs/2507.02782). The unexplored-states
  hypothesis; state passing. Why for us: §166 STATE-COVERAGE.
- **AGCLR (persistent memory for latent reasoning)**, [2606.07720](https://arxiv.org/abs/2606.07720). GPT-2 Coconut plus a gated memory; the
  forget gate matters most. Why for us: the closest prior to our memory carrier.
- **How Linear Attention Remembers**, [2609.33093](https://arxiv.org/abs/2609.33093). Interference grows with memory load. Why for us: reading
  our memory's limits.
- **Transformers Learn Shortcuts to Automata**, [2210.10749](https://arxiv.org/abs/2210.10749). Parallel shortcuts instead of step-by-step
  state. Why for us: why serial latent use isn't automatic.

## 4. Arithmetic
- ★ **Why Can't Transformers Learn Multiplication? (Bai)**, [2510.00184](https://arxiv.org/abs/2510.00184). The middle-column failure, fixed by
  a running-sum auxiliary. Why for us: the same wall our multiplication runs hit.
- **Stepwise Internalization (ICoT-SI)**, [2405.14838](https://arxiv.org/abs/2405.14838). Removing CoT tokens gradually; GPT-2 does 9×9.
  Why for us: the multiplication SOTA and the removal schedule.
- **Separators Make Carry Propagation Learnable**, [2610.06605](https://arxiv.org/abs/2610.06605). One extra slot per digit; the carry stored
  as an angle. Why for us: §156 found it doesn't transfer to pretrained GPT-2.
- **Position Coupling**, [2405.20671](https://arxiv.org/abs/2405.20671), and **Abacus embeddings**,
  [2405.17399](https://arxiv.org/abs/2405.17399). Length generalisation via digit-significance alignment. Why for us: C2 for arithmetic.
- **FoNE (Fourier number embeddings)**, [2502.09741](https://arxiv.org/abs/2502.09741). Why for us: background for the periodic-read idea.
- **Towards Revealing the Mystery behind CoT (Feng)**, [2305.15408](https://arxiv.org/abs/2305.15408). Nested arithmetic needs CoT for bounded
  depth. Why for us: our nested-expression task.
- **CoT Empowers Transformers to Solve Inherently Serial Problems (Li)**, [2402.12875](https://arxiv.org/abs/2402.12875). Why for us: when
  serial latent use is forced (TC⁰ vs serial).

## 5. Causal-test methods and failure modes
- **Distributed Alignment Search (DAS)**, [2303.02536](https://arxiv.org/abs/2303.02536). Finding causal subspaces. Why for us: §167 DAS-EDIT.
- **Interchange Intervention Training**, [2112.00826](https://arxiv.org/abs/2112.00826). Why for us: the formal version of our donor tests;
  data augmentation doesn't induce alignment.
- **Is This the Subspace You Are Looking For?**, [2311.17030](https://arxiv.org/abs/2311.17030). The interpretability illusion in subspace
  patching. Why for us: the controls for any edit result.
- **Variational Lossy Autoencoder**, [1611.02731](https://arxiv.org/abs/1611.02731). A latent carries only what the decoder can't see.
  Why for us: the frame for §179's decoder-bypass result.
- **Pause tokens (Goyal)**, [2310.02226](https://arxiv.org/abs/2310.02226). Extra positions help only with pause *pre*training. Why for us:
  why fine-tune-only think passes go unused.

## Suggested path (about 10 papers, roughly a weekend)
Coconut → SIM-CoT → Illusion of Superposition → ProsQA-Ext → Huginn → Fan → Anil → Grazzi → Bai → (skim) Kuo + RecurTrace.
