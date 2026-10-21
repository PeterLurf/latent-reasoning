# LIT first pass after the 10-18 → 10-20 outage: latent-CoT SOTA, §218 / §219 prior work, §217 / §215–216 failure modes (2026-10-20)

Written 2026-10-20 ~22:50 EDT (from `date`), by the Literature session, for Audit (claude-c5), Theory and the user.
- **Method:** three read-only search subagents, one per question. I re-checked every ID below against export.arxiv.org/api/query (title, authors and v1 date all match). I also checked the key numbers of the three lead papers in the arXiv HTML myself: Cui 2602.22441 Tables 1 and 6, MUX 2607.18264 Table 1 and Fig. 3, and Cheng & Zhang 2609.35643's OOD construction.
- **Other figures:** where a number is marked "(abs)" it comes from the abstract. Other figures come from the subagents' reading of the HTML and are marked "(sub)".

## Top 3 leads

1. **MUX 2607.18264 beats CODI on GPT-2 GSM8K-Aug, with the same data and no outside teacher.** [empirics]
   - Authors: Suleymanzade, Gozeten, Bronstein, Ceylan, Kim.
   - Date oddity: the API gives v1 as 2026-05-19, which does not fit a 2607 ID. Cite it by ID.
   - **Result:** GPT-2 + LoRA, 6 latents. **48.1 ± 0.3 over 3 seeds**, against CODI 43.7, SIM-CoT 42.6 and SFT-CoT 44.1 in the same table (checked).
   - **Ablation** (Fig. 3, checked): accuracy goes **32.7 → 48.2** as the number of latents given the local loss goes from 0 to 6.
   - **Loss** (sub): answer CE + β·KL(each latent's decoded token mixture ‖ a position-weighted mixture of its aligned CoT span's subwords) + γ·CODI L2 to a stop-grad teacher, with β = γ = 1. **Correction (23:20):** GPT-2 uses sinusoidal weighting; geometric is the LLaMA setting (LIT_MUX_FORGETTING_2026-10-20).
   - **Assumptions:**
     - LoRA r128 for 40 epochs; our §210 run was 20 epochs.
     - The CODI baseline is copied, not re-run.
     - Probes only; no causal intervention on the latents.
   - **For us:** compare by margin over CODI (+4.4), not against our 0.379. MUX is the only same-recipe add-on on our §210 harness with a seed-replicated gain. Its per-latent decodable target also gives VALUE-SWAP-style tests something to grip.
   - Second same-data result: PMPS 2609.09928 reports GPT-2 44.35 vs CODI 40.86 in its own table [empirics, seeds not reported].

2. **The ProsQA shortcut is documented: adopt Cui 2602.22441's noise / zero-latent controls for §218.** [empirics]
   - Authors: Cui, Dai, He, Shi et al.
   - **Noise injection** (σ = 100 on latents with mean ℓ2 norm 24 for GPT-2; checked, Table 1):
     - ProsQA barely moves for weakly supervised latents. Llama Coconut goes 99.4 → 99.4 and GPT-2 Coconut 97.8 → 89.0.
     - Standard CoT goes 81.0 → 0.0 on GPT-2.
     - GSM8K keeps non-trivial accuracy under noise (GPT-2 CODI 43.6 → 8.9).
   - **Fix** (Table 6, checked): cross-stage data reuse plus next-step-only supervision in the early stages.
     - Clean / noise = **93.6 / 4.4** with both changes, against 97.8 / 97.8 for vanilla Coconut.
     - Either change alone leaves noise ≥ 93.8.
   - **For §218:**
     - Report each arm under latent noise and with zero latent steps, beside clean.
     - ProsQA-SYM's role symmetry targets the person-agnostic shortcut, but Cui's result says a weakly supervised latent can be bypassed even when the answer needs reachability. VB-C1-L's swap tests cover use; noise and zero-step accuracy are the cheap field-standard companions.
   - **Prior for "already done?":**
     - Closest memory design: AGCLR 2606.07720 (Farhan & Chaudhary, ICML 2026). GPT-2 Coconut with **one** gated persistent vector: ProsQA 96.0 vs Coconut 92.0, GSM8K 34.0. One seed, no memory ablation [empirics, weak].
     - Shortcut-free reachability exists only on **synthetic** graphs: Mirtaheri 2505.21825 and Gatmiry 2609.19717. 2609.19717 uses a mirror-image distractor, so chance is 50%, and reports latent 99.9 vs written CoT 63.5 on a 15M model.
     - **§218 is NEW as designed:** versioned slot bank + reader over versions + role-symmetric natural-language ProsQA + self-teaching. No delta-rule / fast-weight latent-CoT paper was found. δ-mem 2605.12357 is long-context recall, not reasoning.
     - Theory note, Zhang 2605.30757 [theorem, logspace-reduction assumptions]: a fixed-size recurrent state cannot decide P-complete problems where CoT can. The §218 derivation should say whether the bank grows with steps (CoT-like) or is fixed (small-space).
     - Depth past training on ProsQA: Cheng & Zhang 2609.35643, from scratch, train 3–6 hops, test 7–12 hops on two vertex-disjoint paths (a symmetric design close to SYM). End-to-end bottleneck latents rank best out of distribution, and a sparse recurrent search circuit is found by ablation and transplant [empirics]. This is the nearest precedent for §218's A2 depth extension.

3. **§215–216's memory loss is at OLD depths despite gold replay, so it is forgetting by the self items, and the evidenced remedies are outcome-independent.** [empirics; reading of our own registered metric]
   - **Our facts:**
     - KEEPS-SELF-STAB is the mean over **n 3–6** (SELFDEPTH_v1 l.182).
     - Round 1 already mixed **50 % gold n ≤ 8** (l.309).
     - FT-CTRL kept 0.696 while SELF fell to 0.489.
   - **So:**
     - The loss is not a depth-capacity wall. Boesch & Wee 2609.16183 [empirics, preliminary, small synthetic] find that delta-rule recall failure is interference, fixed by an accuracy-gated curriculum.
     - Plain gold replay at the output level did not stop it, unlike in Lee 2502.01612, which keeps D_0 ∪ … ∪ D_r but never measures short-length forgetting.
   - **Evidence that outcome training does not protect causal use** [empirics, text CoT, truncation as the intervention]: Yu, Tartaglini, Hase, Guestrin, Potts 2604.22074.
     - After outcome-only RLVR on Qwen2.5 1.5–7B, CIR fell on 19/40 tasks, with no correlation to the accuracy gain (ρ 0.17, sub).
     - **Adding CIR as an auxiliary reward raised it at matched accuracy** (abs + sub).
   - **For us:** a lead for any round 3 / §218 A2 is a two-sided memory-use term on the self items. For example, a swap + V_0-mask follow loss or a KL of the read to the frozen base's read on n 3–6. Per the Goodhart memory, tie it to decodable task information.
   - **Output-level remedies are weaker for this metric:** KL replay to a frozen pre-FT model (Marek 2605.26097) and correct-set turnover tracking (Qin 2606.03087).
   - ~~Kuo 2606.29983 as a cause of §217's falling follow_iv~~ **WITHDRAWN (23:00, after reading the paper):** Kuo's outcome is the across-seed variance of OOD accuracy (from scratch, binary addition / Dyck-1 / Unique Set / Copy; K = n gives 22.1 / std 9.9, K = n ± 5 gives 30.9 / 5.9, RL-Halt gives 45.0 / 2.7) [empirics; appendix props D.2 / E.2 are derivations]. Our K = min(n + U{0..4}, 20) is already their randomised fix, and the paper says nothing about causal use. At most this suggests a $0 descriptive check: follow_iv at fixed n across eval K′.

## §219 TEXT-SELF (Q2)
- **Verdict:**
  - The text side is done: Lee 2502.01612, Huang 2511.07378, and Liu 2602.10014's finite-sample theory of easy-to-hard self-improvement [theorem].
  - **The matched latent-vs-text round-for-round comparison is still NEW.** Nothing since LIT_FAITHFULNESS_SELFIMPROVE_2026-10-18 changes that.
- **Lead:** copy Lee's label-noise probe to both arms (5 / 10 / 20 % corrupted self-labels, read the next window). Error tolerance is a sharper latent-vs-text contrast than accuracy per round.
- **Correction to a subagent's claim:** it cited Chen 2603.21676 as "per-step intermediate supervision hurts". Chen's "per-step" is final-answer CE at each step (LIT_DEPTH_RECURRENT_CHEN_HE_2026-10-11). Do not cite it as evidence about step labels.

## Other verified items (one line each)
| ID | claim | tag |
|---|---|---|
| 2607.20594 Zhang, Hu, Peng, Xie | looped ≤ 25M: frontier speed v ≈ n_tr / T_tr (R² 0.99), halting T* = n / v; S5 10 / 10 seeds with operator-first (sub) | empirics, 16 seeds / cell |
| 2605.26733 STARS (Yang et al.) | Ouro-1.4B + Jacobian spectral-radius penalty + log-normal loop sampling: GSM8K at 8 loops 74.5 vs SFT 60.1 (sub); 8 lies inside the sampled train range | empirics, no seeds |
| 2604.27998 Latent-GRPO | Llama-1B GSM8K-Aug 66.29 vs explicit GRPO 62.26; uses extra LLM-generated questions | empirics, 1 run |
| 2610.07175 CALR | Qwen3-VL-4B rendered-CoT latents 63.8; cross-problem latent swap 64.3 → 1.4 (sub) | empirics; not our scale |
| 2607.21090 Cheah et al. | faithfulness metric as RL reward raises it (sub) | empirics |
| 2503.07903 MemReasoner | ordered memory for 1–2 hop synthetic reasoning; not latent CoT | empirics |
| 2607.07386 Sparse Delta Memory | scales linear-RNN state through sparsity; option if F2 were capacity | empirics |
| 2609.11801 Looped flows | local denoising objective for looped models (ARC-AGI-1 58.8 vs TRM 44.6, sub) | empirics |
| 2602.01148 Zou, Xiong, Liu | latent CoT "exploration–execution trade-off", curriculum necessary (abs) | theorem per abs |
| **2605.05066** | **WITHDRAWN** (proof error in the main theorem): do not cite | — |
