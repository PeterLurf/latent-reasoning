# Prior check for §78: GSM8K length generalisation by step count (train ≤ 3, test ≥ 6), GPT-2 small (2026-10-02)

Literature session, for Theory; Audit copied. Short note. arXiv IDs verified (Semantic Scholar batch).

## 1. Prior easy-to-hard / step-count splits on GSM8K
- **Hase et al. 2401.06751 (ACL'24):**
  - The closest prior, using the **same split idea**. GSM8K hardness is the number of reasoning steps in the
    human-annotated solution: easy = 2–3 steps, medium 4–5, hard 6–11.
  - Models: Llama-2 7B/13B/70B, Qwen-72B, Mixtral (in-context learning, QLoRA, linear probes).
  - Easy-to-hard recovers about 79% of the hard-supervision gain (supervision gap recovered, QLoRA, 70B); easy data is
    "surprisingly good". **No small-model or from-scratch fine-tuning result.**
- **Sun et al. 2403.09472:** easy-to-hard on MATH levels (1–3 → 4–5) via reward models and RL; GSM8K is not the split.
- **Dziri et al. 2305.18654 (Faith and Fate):** step and size splits on multiplication, Einstein puzzles and DP, not
  GSM8K. Fine-tuned GPT-3 fails OOD on larger compositions ("linearised subgraph matching").
- **Complexity-scaled evaluations (inference only, large models):**
  - GSM-Infinite 2502.05252: accuracy decays with operation count.
  - GSM-Symbolic 2410.05229: adding clauses (P1/P2) drops accuracy and raises variance.
  - Complexity-based prompting 2210.00720: step count as complexity.
  - GSM-Plus 2402.19255 (perturbation robustness).
  - GSM-Hard (PAL 2211.10435) uses larger numbers, not more steps.

## 2. Published numbers for "CoT trained on short, tested on long GSM8K"
- **None found at GPT-2 scale, or for any model fine-tuned only on short GSM8K problems and tested on ≥ 6-step ones.**
  Hase is the only direct number (large models, about 79% recovery). Small-model distillation (Fu 2301.12726; Deng
  2311.01460) trains and tests on the full distribution. §78 would be new as a small-model, step-split, written-state
  vs CoT comparison.

## 3. Written-state / history-cut methods evaluated by step count
- Delethink 2510.06557, InftyThink 2503.06692 and PENCIL 2503.14337 evaluate by token budget or problem family, not by
  step count. Markovian Transformers 2404.18988 evaluate on GSM8K as a whole.
- **Amos et al. 2602.08332 (Thinking States):** length generalisation on parity and variable assignment with Qwen2.5-0.5B.
  Vars trained at N = 10 → 33.8% on [10, 100]; trained at N = 40 → 97.7%. It does not use GSM8K, and does not cut
  history.
- **Not found:** a history-cut written-state arm evaluated by GSM8K step count. New as scoped.

## 4. Pitfalls in counting GSM8K-Aug equations as steps
1. **Two annotation sources.**
   - GSM8K-Aug's training traces are GPT-4-generated equation-only CoT for GPT-4-*generated* problems (Deng
     2311.01460).
   - The test references come from GSM8K's own human calculator annotations.
   - Count steps with one rule (number of `<<…>>` equations) on both, and report the step histogram per source.
   - The GSM8K test's ≥ 6-step tail is small: report n, with bootstrap CIs.
2. **Equation count ≠ dependency depth.** Some equations are independent (parallel), and some are trivial conversions.
   For the range principle, the growing quantity is the *dependency depth* and the number of live values. Report a
   secondary split by DAG depth.
3. **Leakage through near-duplicates.** Long Aug-train test items are GPT-4 variants of GSM8K-train-like problems.
   Dedupe them against the short training split by question similarity (n-gram or embedding), not just by ID.
4. **Label noise rises with length.** GPT-4-generated long traces and answers are more error-prone. Spot-check, or
   filter by recomputing the equations with a calculator. Also report the GSM8K-test-only subset (human labels;
   GSM8K-Platinum caveat in LIT_GSM8K_STEP2).
5. **The written-state arm has a range problem of its own.** "Values so far" grows with the step count. At ≥ 6 steps
   the state holds more values than any trained state (≤ 3). By our own range principle that state is OOD. Either
   bound it (keep only values still needed later, or a fixed-size state) or train with padded or longer states.
   Otherwise a written-state failure won't distinguish method from format.
6. **Position and format.** Text CoT at ≥ 6 steps runs past trained output lengths. Report the failure type per arm:
   early "####", wrong operation, or arithmetic error.
