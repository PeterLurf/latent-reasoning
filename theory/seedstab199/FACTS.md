# Seed-stability fact sheet for subagents (Theory session, 2026-10-13 21:3x EDT)

User request (relayed): "to have a novel contribution we must create something new ... deep research and create a new architecture/mathematical
method to combat seed stability. give mathematical backed reasoning." Repo: D:\Latent (READ-ONLY for subagents unless told otherwise).

## The model (boxes/deltamem.py, boxes/nested170.py; read them for exact equations)
- Trunk: pretrained GPT-2 small (12 blocks, D 768), fully fine-tuned. AdamW lr 1e-4, wd 0.01, cosine over 48k steps, warm-up 100, batch 64,
  global grad-norm clip 1.0 (binds on 65-100% of steps). bf16 autocast trunk; memory branch fp32.
- Task "nested": Feng et al. arithmetic mod 11, n operators (train n ~ 1..8), expression tree, one reduction per step. Step-t target e_{t+1} =
  the whole re-rendered expression after t reductions (string). Answer = value mod 11 (chance ~ 0.09).
- Thinking: K = 8 passes, EVERY pass sees the IDENTICAL input [EOT, expr, " ."]; the ONLY cross-pass channel is the memory state S (12 heads x 64 x 64).
  Pass t: residual h6 after blocks 0..5 (identical every pass) -> read r = fade * W_o [g_add * read_add || g_del * read_del] added at block 6 ->
  blocks 6..11 -> h_w (top residual) -> write: k, v, gate from u = (1-gamma) LN_6(h6) + gamma LN_w(h_w), sequential over the T input tokens:
    6 additive heads: S_a <- S_a + phi(k) v^T  (phi = elu + 1; no decay; an INTEGRATOR); read phi(q)^T S_a, then (RMS recipe) per-head RMS-normalised x gain.
    6 gated-delta heads: S_d <- alpha S_d (I - beta k k^T) + beta v k^T (k, q L2-normalised; beta = 2 sigmoid(.) in (0,2); alpha = sigmoid(.), bias +5 -> 0.993);
      read S_d q. At init the gate weight is 0, so beta = 1 exactly (full overwrite along k) and alpha = 0.993.
  Pass map is autonomous: S_t = F_theta(S_{t-1}; x).
- Init is DETERMINISTIC and function-preserving: memory q/k/v/LN copied from block 6, W_o = 0, gate weight 0; g_add calibrated at step 0 (0.304, identical
  across seeds). Schedules: fade 0 -> 1 over steps 0..500; gamma = 0 for steps <= 2000 (writes depend only on the input: OPEN LOOP), linear to 1 by 6000.
  => Seeds differ ONLY in the training data order (and floating-point rounding: GPU class / kernel-order changes alone flip outcomes).
- Losses: answer CE (decoder pass [EOT, expr, " ="] reading S_K) + lambda (1.0) x per-pass step aux: a read-only decoder (same trunk, no problem text)
  reading S_t must emit e_{t+1} (teacher-forced CE), for every pass t <= n.
- Clean-test property (C1) that any method must keep: identical input every pass, memory the only cross-pass channel, removal / serial tests valid.

## The seed lottery (same recipe, seeds 1/2/3, all RTX 3090)
- "RMS" recipe: s1 and s2 converge (loss 0.006 / 0.003, entry into loss < 0.5 at ~19k); s3 never (48k, final loss 1.03).
- Training loss: s3 is AHEAD of s1 at 8k (1.84 vs 1.91 answer CE; aux 0.357 vs 0.383), then s1 takes off between 8k and 20k, s3 plateaus (1.36 at 20k).
- s3 per-pass step-aux accuracy (pooled over items with n >= t): t1 0.988, t2 0.997, t3 0.961, t4 0.874, t5 0.754, t6 0.587, t7 0.409, t8 0.316.
  Its conditional per-step success FALLS with pass index (0.97, 0.92, 0.90, 0.86, 0.73, 0.72). s1/s2: >= 0.99 at every pass 1..8.
- s3 does use the memory serially: n = 4 answer 0.84 at K' 4 or 8; n = 6 0.15 at K' 4 vs 0.49 at K' 8; K' 0 (no thinking) is near chance.
- Literature: curriculum (n <= 4, K 4) made s3 enter by 10k (pilot). Grokfast EMA filter: interim no help. Gradient-error link: none found.

## Path independence: the strongest separating statistic found so far (eval field s182.path_indep, final models)
Run K' = 16 passes from M_0 = 0 vs from another item's 8-pass memory (derangement); delta_cos_final = mean cos between the two final delta-head
states; answer_agree = same answer from both starts.
| run | final-1k loss | delta_cos_final | answer_agree | acc from zero / donor |
| s182_window_s2 | 0.001 | 0.470 | 0.130 | 1.000 / 0.130 |
| s184_reweight_s2 | 0.003 | 0.436 | 0.124 | 0.994 / 0.124 |
| s185_rms_s2 | 0.003 | 0.588 | 0.140 | 0.996 / 0.140 |
| s185_rms_s1 | 0.006 | 0.625 | 0.132 | 0.954 / 0.124 |
| s190_rmsnopass_s1 | 0.002 | 0.480 | 0.100 | 0.994 / 0.100 |
| s187_nopass_s1/s2 | 0.014 / 0.005 | 0.472 / 0.474 | 0.114 / 0.124 | ~0.99 / ~0.12 |
| s184_reweight_s1 | 0.110 | 0.473 | 0.108 | 0.940 / 0.104 |
| s190_rmsnopass_s2 | 0.228 | 0.481 | 0.122 | 0.898 / 0.124 |
| s182_window_s1 | 0.239 | 0.529 | 0.140 | 0.908 / 0.144 |
| s182_nofade_s1 | 0.560 | 0.768 | 0.310 | 0.432 / 0.218 |
| s185_den_s1 | 0.642 | 0.930 | 0.748 | 0.720 / 0.642 |
| s188_rwwin_s1 | 0.901 | 0.763 | 0.302 | 0.710 / 0.296 |
| s182_hold_s1 | 1.060 | 0.898 | 0.468 | 0.694 / 0.394 |
| s185_rms_s3 | 1.031 | 0.978 | 0.820 | 0.596 / 0.576 |
=> every converged model (loss <= 0.24) is PATH-DEPENDENT (delta cos 0.44-0.63; a foreign start state ruins the answer: it is never forgotten);
   every non-converged model (loss >= 0.56) is PATH-INDEPENDENT / CONTRACTIVE (delta cos 0.76-0.98; it forgets the start). 15 models, no overlap.
   Caveat: final models only; cause vs consequence is NOT established (a model whose writes ignore the read converges trivially).
- Clean-trajectory step ratio log(||S_{t+1}-S_t|| / ||S_t-S_{t-1}||) at t = 2, 3, 4: s1 +0.005, +0.057, +0.108; s2 -0.071, -0.001, +0.060;
  s3 -0.175, -0.150, -0.095 (s3's per-pass increments SHRINK early; s1's grow).
- Relative-perturbation growth slope (1e-3 relative perturbation at t0 = 2, normalised by the state norm): s1 -0.038, s2 -0.004, s3 +0.034.
  (Absolute-gap Lyapunov is > 0 for every model by construction: the additive state carries any perturbation forward.)
- Other known structure: additive state norm grows every pass (no fixed point); delta heads are non-expansive; no Lipschitz bound on the trunk's write.
- All converged models still FAIL at length: n 9-12 at K' = n + 2 accuracy ~0.5-0.66 ("DEPTH"); and s1's K' 16 at n 8 is 0.75 vs 0.99 at K' 8
  (extra passes after completion corrupt the answer: not STABLE). Path dependence = the integrator never forgets, including its own errors.

## Prior-art bar (Literature session, docs/LIT_SEED_STABILITY_2026-10-13.md; read it)
Generic levers are NOT new: saddle escape time (Saxe 1312.6120; Rawal & DeWeese 2605.01288: escape Theta(eps^{-(r-2)}) for r >= 3 small factors),
perturbed GD, continuation / graduated optimisation (Hazan 1503.03712), symmetry removal (syre 2408.15495), curricula, restarts (Luby 1993),
stochastic collapse (Chen, Kunin et al. 2306.04251: SGD noise makes invariant "off" sets attractive). Gap: no theory of basin-ENTRY PROBABILITY for a
gated recurrent memory branch in a pretrained / looped model; a method that provably removes the bad set by construction would be new.

## Rules for subagents
- Do NOT edit anything under D:\Latent (no commits, no docs, no latent/*.py, no boxes/*.py). Write outputs ONLY to
  C:\Users\peter\AppData\Local\Temp\claude\D--\2f0034d5-4d85-51eb-880b-bab41adb0dff\scratchpad\seedstab\ .
- Never touch GPUs or cloud instances, never stop any process. CPU work: <= 3 torch threads (torch.set_num_threads(3)), run at Idle priority
  (e.g. start-process with -Priority / or `nice`), short jobs; delete any .pt / large files you create in the scratchpad when done.
- Never grep recursively inside boxes/ (run dirs hold GBs); read explicit files.
- State assumptions explicitly; separate proved statements from conjectures; give falsifiable predictions with numbers where possible.
