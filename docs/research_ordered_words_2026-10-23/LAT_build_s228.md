# §228 LAT-OW build note (Experiments, 2026-10-23): what boxes/inject_ow.py changes vs §184, and the gates

Code:
- `boxes/inject_ow.py` (training);
- `boxes/inject_ow_eval.py` (eval);
- `scratch/gate_s228.py` → `scratch/gate_s228/<part>.json`.

nested170 / deltamem / deltas5 / fsq120 are imported read-only. No existing module is edited.

§184 reference: `boxes/nested170.py` at 6f3f5c9d, run with `s184_reweight_s2/args.json` (`--arm inject --think_k 8 --aux_step 1 --aux_fade 48000,48000 --aux_reweight --fast_add`; no `--cache_lower` key, so it ran uncached).

## Registered deviations (a)–(g), with line refs (nested170 = N, inject_ow = IO)

| | §184 (nested170) | §228 (inject_ow) |
|---|---|---|
| (a) step target | aux target e_{t+1} = the whole partly reduced expression (N:204–212 `aux_steps`, N:260–272 `aux_logits`) | one step sentence `enc(" " + R_step[t]) + EOT` = "<S_t> has <v_t> <u>." (IO:51–55 `tok_ow`, IO:74–84) |
| (b) lookup | no names; the think input is the expression (N:235–236) | names, units and 0–2 distractors in English premises (`OW.make`) |
| (c) think length | `[EOT, expr, " ."]`, batch-max ≈ 29 tokens (lat_checks B) | `[EOT] + enc(" " + text)`, 36 / 72 / 127 tokens at d 1 / 4 / 8 (IO:52) |
| (d) batches | mixed n per batch: n ~ U{1..8} per item (N:157–163 `train_items`) | one d per optimiser step: `default_rng(seed + 22331).integers(1, d_max + 1)` (IO:332). Items from their own stream, so DIRECT and LAT draw identical d and items |
| (e) cache | uncached (§184 args); flag at N:644, used at N:714 | `--cache_lower` (IO:226, the same `m.lower` / `h_low` path as N:714); gated in fp64 (`cache64`) |
| (f) law | n uniform from step 0, per item | d uniform from step 0, per step (pilots U{1, 2} / U{1..4}, Stage 1 U{1..8}) |
| (g) surface | Z_11 expression tokens | English frames 0–5 (train), TOPO order, dag on, mod 11 |

## Other differences (disclosed; none changes a kernel)

1. **Answer read:**
   - §228: dec = think + `" The answer is:"`, with the logit at its last token.
   - §184: `[EOT, expr, " ="]` (N:254–258).
   - In both, the CE is over the full vocabulary at the number token, and accuracy is the argmax over the 11 number tokens (N:730 / N:768).
2. **Think input:** ends "… have?" (no `" ."`).
   - Unchanged: aux prompt `[EOT, " Step", " t", " :"]` (N:228–233), AUX_W (N:370), the per-item aux mean (N:723–731).
3. **g_add calibration:** 64 OW items, d = 1 + i mod d_max, from stream seed 22996 (§184: nested items, seed 17001; N:979–984). The rule is the same: median ‖read_del‖ / ‖read_add‖ over min(K, 4) passes.
4. **Exclusion:** TEST2 + TRANSFER + every PAIRS2 candidate (recipients and donors) + floor donors + the §226 / §227 sets. That is 160,598 signatures (`excl_n`).
5. **Eval:**
   - fp32, with the cached lower blocks on every pass (§184 evaluated bf16, uncached; N:756–769).
   - K′ sweep 0..14 from one pass sequence.
   - Not ported (nested-specific): `path_indep` (s182), `skip_pass`, `serial`, `step_recover`, `s175`. Skip-pass is uninformative here (draft §4).
6. **Logs added:**
   - 1k bins by d: steps, acc, CE, clip;
   - aux value-token acc by pass;
   - `dseq` / `ihash` (G-EXP);
   - `exposure_by_d`, `entry_d1`, `gn_preclip_mean100`.
   - `--futility` (LAT_v1 §4) is off by default.
7. **Checkpoints:** full state, adding the d stream, item stream, bins and futility state (N:1149–1156 lacks these). ckpt_every defaults to 2000 (§184 used 1000). `--stop_at` exits 9 without eval.
8. **mem_check:**
   - training: the longest 64 / micro of 256 d_max items at K;
   - eval: the longest d 12 item × eval_bs at K′ 0..14 (fp32);
   - if over budget, micro doubles first, then eval_bs halves.

## Gates (`scratch/gate_s228/`)
- **port:** nested170 vs inject_ow `--src nested`; ckpt at step 50 bitwise (model + AdamW), uncached and cached.
- **cache64:** fp64 cached vs uncached on OW d 1 / 4 / 8.
- **gradflow:** per term.
- **plain:** direct, 1 step.
- **resume:** step 20 → 40 bitwise.
- **f2:** the 4060 pilot shapes.
- **args:** `scratch/_tmp/s228/args_dir/args_diff.json`.

## Eval output layout (for score_lat)

Files are in `<run>/`. `eval.json` is a summary (parts present + train summary; no accuracies). Field paths:

- **`eval_acc.json`** (TEST2) and **`eval_transfer.json`** (frames 6–7):
  - `by_d[d].acc` (K′ = think_k) and `by_d[d].k_main` (that K′; 0 for direct);
  - `by_d[d].acc_kprime["0".."14", "d+2"]`;
  - `items[d] = {sig, ans, pred, L, s}`. `pred[i]` is a list over K′ 0..14 (inject) or an int (direct). L = think tokens, s = decoder suffix tokens.
  - `kprime0_nobias[d] = {acc, pred}`; `wo_bias_norm`.
- **`eval_mean_s.json`:**
  - `by_d[d] = {per_d, pooled}` (acc);
  - `items[d] = {per_d: [pred], pooled: [pred]}`;
  - `mean_s_at_t[d][t] = {acc, pred}` (d 1..8);
  - `mean_state_norm`.
- **`eval_swaps.json`:**
  - `items[d]` = a list of pairs, each `{rec_sig, don_sig, t (= t*), own, dfin, ca, cb, own_pred, donor_pred, state_swap {t: pred} (t 1..d−1), budget_front [pred at b = 0..K−t*], cache_swap, leak_seg, leak_rows_differ, last7_tokens_same, floor_preds [3] (−1 = no donor), floor_sigs [3]}`;
  - `by_d[d]` = descriptive summaries (follow_U, own / donor-final shares, cache_swap own / donor-final, leak follow_U, leak_changed_output, exposure shares);
  - `self_bitwise {state, cache, leak}`;
  - `pair_stats`.
- **`eval_decode.json`:** `by_d[d][t] = {exact, value_token}`; `items[d] = {sig, t: {gen, exact, value_token}}`.
- **`log.jsonl`:**
  - 100-step rows: `loss`, `answer_ce`, `aux_ce`, `gn_preclip`, `gn_preclip_mean100`, `clip_frac_100`, `gamma`, `fade`, `write_grad_norm`;
  - 1k rows (keyed by `by_d`): `bin`, `by_d[d] = {steps, acc, ce, clip}`, `exposure_by_d`, `dseq`, `ihash`, `aux_vacc_by_pass`;
  - `{entry_d1}`; `{futility}`.
- **`train_summary.json`:** `entry_d1`, `clip_rate`, `grad_norm_preclip`, `n_nonfinite`, `last_step`, `futility`.
