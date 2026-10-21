# SCALE-PROBE: is "necessary but not portable" a GPT-2-scale property? The 210-4 C1 battery on released 1B latent reasoners (draft, 2026-10-20)

**Status:** DRAFT, not registered. $0, eval-only, local. Theory research step, ranked first by Theory and Audit (C1 on language is the open gap). Next: an Opus proof check / red team, then v1, then Audit (under the user's $30 rule, cap $0).

## 0. Question and decision value

- **On GPT-2 small, every latent GSM8K model fails portability.**
  - Our in-house models recompute from the question (§207: input-follow 0.87–0.93, memory carried ≤ 0.01).
  - CODI at 20 epochs (0.379, §210) has **necessary but context-bound** latents (210-4 / 210-5): K 6 − K 0 = +0.21, but DONOR-SWAP follow is 0.040, own 0.084 and neither 0.876.
  - On synthetic tasks, VERSION-BANK's latents are portable: VALUE-SWAP follow 1.000.
- **Nobody has run a donor-follow or question-swap test at ≥ 1B** (Literature 2d35f1dc).
  - Published ≥ 1B interventions are noise, mean, zero and early stop (Cui 2602.22441, Dilgren 2604.04902, Jin 2607.06648, Li 2602.08783).
  - One blog (Cywiński et al.) finds that averaging CODI-1B latents across prompts with *different* intermediate values hurts, which suggests content-bearing latents.
- **Decision this informs** (pre-stated, §4):
  - If CODI's latents are portable at 1B, the language C1 gap is a scale effect, and the language line should move to a 1B base (NORTH_STAR open item 5).
  - If they stay context-bound, the CODI-style objective does not yield a portable state at either scale, so explicit state-target supervision is the route (the VB / §218 line).

## 1. Models (pinned by HF revision sha + per-file sha256 at download; Literature verified ids / licences)

| role | HF id | licence | reported GSM8K |
|---|---|---|---|
| **primary** | zen-E/CODI-llama3.2-1b-Instruct (base meta-llama/Llama-3.2-1B-Instruct; ungated mirror alpindale/* only if the sha of the base weights matches) | MIT | 55.6 |
| contrast (descriptive) | MisakiTaro0414/MUX (mux-llama1b-gsm8k.bin): CODI + per-latent local loss | Apache-2.0 | 56.7 ± 0.5 |
| contrast (descriptive) | connordilgren/llama32-1b-gsm8k-coconut | MIT | 35.7 |
| control (descriptive) | bcywinski/codi_llama1b-answer_only (same weights, direct answer) | none stated (read-only eval) | 36.7 direct |

- Code: the official zhenyi4/codi inference path (scripts/test_llama1b.sh: 6 latents, projection 2048 + LayerNorm, LoRA r128 / α32, bf16, `--inf_latent_iterations` = K).
- Downloaded into its own directory and read before use (untrusted-code rule). Ported into a local harness by Experiments, reviewed by Audit.
- MUX / Coconut contrasts use their own released loaders.

## 2. Card, memory, items

- **Card:** one local 8 GB card for every arm. The 4060 after the 218-17 / 216-10 jobs, else the 3070, but one card per model and recorded.
- **Dtype:** bf16 as the released recipe, with an fp32 re-run of the intact arm as a descriptive check if memory allows (1B fp32 ≈ 5 GB of weights).
- **Memory plan:** Llama-3.2-1B bf16 weights ≈ 2.5 GB, plus the merged LoRA. A GSM8K prompt is ≤ ~200 tokens, plus K ≤ 10 latents and the answer (≤ 32 new tokens).
  - DONOR-SWAP needs the donor's K latent states (K × 2,048 floats per item) cached from its own intact run. That is negligible.
  - Gate G-M: memory-check the largest shape first (longest test prompt, K 10, eval_bs 16, swap arm). Peak goes in gate_mem.json. eval_bs is lowered only before any output, and recorded.
- **Items:** CODI's released GSM8K test file (1,319 items), the one test_llama1b.sh reads. Its sha256 is recorded, and the per-item gold is parsed as the official script does.

## 3. Arms (every per-item output stored; the share-gate rule)

| arm | definition (as 210-2 / 210-3 / 210-4 unless stated) |
|---|---|
| **INTACT (gate)** | the official recipe at K 6 → acc. **Gate: \|acc − 0.556\| ≤ 0.025**, else the model's battery is UNREAD (Audit's intact rule). The 0.025 band is about 1.8 SE at n 1,319 plus bf16 batch noise |
| K-SWEEP | acc at K ∈ {0, 1, 2, 3, 4, 5, 6, 8, 10}; K 0 = the answer straight after the question with the latent markers as the recipe places them, no latent iterations |
| **DONOR-SWAP (primary)** | recipients = items correct at K 6. For each, one donor (rng 21002) among correct items with a different gold. The recipient's 6 latent states are replaced by the donor's, computed on the donor's question; the answer is decoded in the recipient's question context. follow / own / neither + the coincidence floor P(recipient intact prediction == donor gold) |
| POS-SWAP k | only latent k swapped, k 1..6 (descriptive) |
| NOISE | Gaussian latents with matched per-position norm (rng 21003; descriptive) |
| **QUESTION-MASK (new, descriptive)** | latents computed normally; then the answer phase's attention to the question tokens is masked (KV dropped), so the answer sees only the latents and the answer prompt. acc_qmask beside acc(K 6) and a K 0 + qmask floor. A portable state predicts acc_qmask well above the floor; a context-bound one predicts the floor |

**Swap-equivalence lemma [PROVED by construction].** In CODI the answer is a function of (question KV, latent states). So "recipient question + donor latents" is the same operation as "donor latents moved to a new question" seen from the donor side. DONOR-SWAP therefore covers §207's QUESTION-SWAP; there is no separate arm.

## 4. Readings (primary model only; contrasts descriptive)

- **LATENTS-NECESSARY-1B:** acc(K 6) − acc(K 0) ≥ 0.10.
- **LATENTS-CAUSAL-1B (primary):** LATENTS-NECESSARY-1B ∧ DONOR-SWAP follow ≥ max(0.30, floor + 0.25). This is 210-2 / 210-3's rule unchanged, so the comparison with GPT-2 CODI20 is like for like.
- **Decision labels:**
  - **SCALE-PORTABLE** = LATENTS-CAUSAL-1B.
  - **SCALE-CONTEXT-BOUND** = LATENTS-NECESSARY-1B ∧ follow < 0.10.
  - **SCALE-UNNECESSARY** = ¬LATENTS-NECESSARY-1B.
  - **MIXED** = otherwise.
- **Wording:**
  - SCALE-PORTABLE: "the released CODI-1B's latents carry a donor's answer into a new question at ≥ 0.30 (GPT-2 CODI20: 0.04); one released model, not a recipe-level claim".
  - SCALE-CONTEXT-BOUND: "necessary but context-bound at 1B as at GPT-2 small".
  - Never C4 or C3; this is a C1 property of one released checkpoint.
- **Descriptive:** the full K-sweep (does K > 6 help? the C2 analogue), POS-SWAP profile, NOISE, QUESTION-MASK, by solution steps; MUX / Coconut / answer-only on the same arms where their loaders allow; GPT-2 CODI20's committed 210-4 numbers beside every row.

## 5. OC (priors ASSUMED)

- **Pool:** ≈ 733 recipients if the intact acc is ≈ 0.556.
- **Null (latents not carrying the answer):** follow ≈ floor (≈ 0.00–0.02). The SE at 733 pairs is ≤ 0.006, so **P(LATENTS-CAUSAL-1B | no effect) < 0.001**.
- **Effect:** at true follow 0.40, P(pass) ≈ 1.00 (SE 0.018). At true follow 0.32, P(pass) ≈ 0.86.
- **Prior P(follow ≥ 0.30) ≈ 0.20** [ASSUMED]:
  - GPT-2 CODI20 read 0.04.
  - The averaging blog suggests content but not portability.
  - Larger models can store more self-contained state in a 2,048-d latent.
- **P(SCALE-CONTEXT-BOUND) ≈ 0.6, P(SCALE-UNNECESSARY) ≈ 0.05** (Cui's noise result: 55.6 → 27.5), MIXED the rest.
- **Realised $: $0.** Every outcome is decision-relevant (§0), so the information value does not depend on the outcome.

## 6. Censoring (rule b)

- **UNREAD (= every reading not met):**
  - the INTACT gate fails;
  - the model or LoRA fails to load at the pinned shas;
  - a test-file sha mismatch;
  - card / dtype / eval_bs mismatch within the model.
- An unparseable answer counts as wrong (the official parser).
- A recipient pool < 300 makes DONOR-SWAP UNREADABLE (power), which fails LATENTS-CAUSAL-1B.
- A missing file reads pending (215-7).

## 7. Gates ($0)

- G-M memory (largest shape first).
- G-R INTACT reproduction (the gate above), read first.
- G-I known-answer swap: swapping a recipient's latents with **its own** latents reproduces INTACT bitwise. Swapping in a donor that has the **same question** (a duplicate) gives follow = own = 1.
- G-D Audit's diff review of the port, the swap hook and QUESTION-MASK (which positions are masked, checked on 3 items by hand).
- G-S the scorer (theory/score_s2xx.py) fixture-checked before any output.

## 8. Open for the red team

1. Is QUESTION-MASK well defined in CODI? The latent iterations themselves attend to the question, so only the answer phase is masked. Does masking the question break the answer prompt's format, so that the floor misleads?
2. Is follow ≥ 0.30 the right bar at 1B? A model could carry intermediate values but not the final answer. Should a "follow any donor intermediate" row be added (Cywiński-style)?
3. Should MUX-1B be co-primary, testing whether a per-latent local loss creates portability?
4. bf16 vs fp32 reproduction: is ±0.025 too loose or too tight?
