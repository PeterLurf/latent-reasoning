# Prior check for §87 B2: deterministic annealing to acquire A5 from lengths ≥ 3 only (2026-10-02)

Literature session, for Theory; Audit copied. Short note, written before any §87 result. Classics are non-arXiv
(journal or ACL references); arXiv IDs as verified in earlier notes.

## 1. Annealing and symmetric saddles
- **Rose 1998** (Proc. IEEE; deterministic annealing for clustering and related problems).
  - At high temperature the free-energy minimum is the **fully symmetric solution**: all points in one cluster, i.e.
    your one-block state.
  - As T falls, it splits at **critical temperatures (phase transitions)**, where the symmetric fixed point loses
    stability. DA is built to break exactly this symmetry.
  - The split happens only if the symmetric point becomes *unstable* at some T, and in practice needs a small
    perturbation to leave it.
- **Ueda & Nakano 1998** (DAEM, *Neural Networks*): EM with a tempered posterior (β → 1) avoids poor local maxima in
  mixtures and HMMs. **Smith & Eisner 2004** (ACL): DA for EM in grammar induction; their 2006 "structural annealing"
  anneals a structural bias instead.
- **What differs in §87.** The project's τ is a softmax temperature on the *table logits*, i.e. a
  parameterisation. Rose and Ueda–Nakano temper the *posterior* (the E-step, or the entropy weight in the free energy).
  - Logit tempering makes transitions near-uniform and rescales gradients, closer to Gumbel-softmax temperature
    annealing (1611.01144) or a small-init / learning-rate schedule.
  - It doesn't by itself create Rose's phase-transition mechanism.
  - Call it **"logit-temperature annealing"** unless the posterior is tempered. Better still, add a posterior-tempered
    (DAEM) arm, with q ∝ p^β over code paths and β 0 → 1, as the canonical DA.
- **Group or permutation automata:** I found no DA or annealing result specific to permutation or group automata. The
  nearest annealing-in-automaton-learning work is grammar induction (Smith & Eisner) and HMM/mixture EM.

## 2. Non-solvable group automata from final states of length ≥ 3 only
- **Not found** by EM, annealing or gradient methods.
  - Neural state-tracking work (Merrill 2404.08819, DeltaProduct 2502.10297, Howe 2609.18966, PD-SSM 2509.22284) uses
    **per-position labels**, which include every prefix's state, so short products are labelled.
  - Merrill also adds all length-2 pairs.
- **Long-only final-state precedent:** Lee 2606.07254 trains a projected recurrent model on length-8 final states only
  and is exact to 2²⁰ steps. But the group is **S3×S3 (solvable)**, the projection is hand-built, and there is no
  certificate.
- **Spectral learning** (Balle et al., spectral WFA; Hsu 0811.4413) should in principle work from lengths ≥ 3 with a
  Hankel basis of prefixes ≥ 1 and suffixes ≥ 2, but I found no published A5/S5 run. Your 69-2 spectral result is
  the relevant check: state its length distribution. If it included length-1/2 strings, a ge3-only spectral run is
  the natural non-neural comparator for §87 B2.

## 3. Wording
- **Safe:** "first in this project: the exact-marginal free table acquires A5 from final answers of length ≥ 3 problems
  only, via [logit-temperature / deterministic] annealing."
- **Broader, if B2 passes:** "We found no prior report of acquiring a non-solvable group's automaton from final states
  of length ≥ 3 sequences only, by any learner. The nearest are per-position-label neural models (DeltaProduct, Howe,
  PD-SSM) and Lee 2606.07254's long-only learner on the solvable S3×S3."
- **Qualifiers:**
  - (i) The model is a tabular IO-HMM/PFA (LIT_FREE_TABLE), not neural.
  - (ii) M = 2|G| uses oracle |G| (report the M sweep).
  - (iii) A ge3-only spectral run is untested, and it might do it non-iteratively.
  - (iv) Report the temperature schedule's sensitivity (τ0 = 16 vs 64), and whether the split happens at a sharp
    "critical τ", which would be the DA signature.
