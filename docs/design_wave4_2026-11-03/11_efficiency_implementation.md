# 11 EFFICIENCY IMPLEMENTATION: EF1-EF5 (+ EF6) as diff-level sketches (wave 4, 2026-11-03)

## 1. Goal and backlog IDs
Backlog §13.3 "efficiency (Fable)" lists EF0 STEP-PROFILE, EF1 COMPILE-PASS, EF2 BUCKET-PAD, EF3 EVAL-BF16-SWEEP, EF4 PREFETCH-ITEMS and EF5 HOLD-SHARED-LOWER. This memo turns EF1-EF5 into code sketches against the real files. Each one gets an exactness gate, a predicted speedup and a place in the landing order. It also adds **EF6 DEC-LOWER-CACHE**, an exact eval speedup found while reading `passes()`. It is not E1 BUN-SHARED, which shares the prompt prefix across pass inputs.

Target configuration: bun (§259, `boxes/runs_fsq/bun_s1/args.json`):
- bs 64, micro 2 (two chunks of 32), K 8, d_max 4;
- cache_lower, fast_add, bf16 autocast;
- hold_inc 1.6, far pass n 16, T_far ~ U{16..40};
- distract_p 1 with mix s256/chain/givop;
- write_src answer, cue step.

Recorded timing, quoted from the files and not re-derived:
- `bun_s1_gtime/timing.json`: sec_per_step 1.306 on a 3090.
- `bun_s1/train_summary.json`: train_sec 13230 for 8000 steps.
- `bun_s1/eval.json` part secs: acc 481, halthead 249, swap_kd 91, ws_deep 48, lp5h 42.

None of this changes C1-C7. It buys more cells per dollar, which serves A3/X7 seeds (≥3-seed C2 claims) and the §264-§268 eval parts.

## 2. Where a step's time goes (analytic, to be confirmed by EF0)
Block-forward units are one GPT-2 block over one chunk (B 32, T ≈ prompt + cue 7 + SFX 4).

| Component (per step) | Code | Units, fwd (bwd ≈ 2x) |
|---|---|---|
| CE lower, d distinct inputs x 6 blocks, per chunk | `loss_reg` L370 | 6d x 2 |
| CE upper, K = 8 passes x 6 blocks | L372-386, `pass_aw` L47-56 | 48 x 2 |
| decoder (12 blocks) + aux (d x 12 blocks, short) | L396, L404 | ~12 + small, x 2 |
| hold penalty: its own lower + \|H\| = K-d-2 passes | `hold_step` L176-182 | (6 + 6\|H\|) x 2 |
| far prefix: T_far-1 no-grad passes, B 16 | L186-192 | ~6·27/2 ≈ 80 (fwd only) |
| data: `draw_items` (stream, distract, 2x `tok_aw`) | L316-346 | CPU, serial, unmeasured |

Two predictions follow (EF0 tests them):
- The far prefix is about 10-15% of the step. It is many small no-grad launches at B 16, which makes it the best CUDA-graph target.
- The penalty's private lower forward is about 2% of the step.

Eval runs in fp32 with no autocast (`run()` L1314 meta "eval_dtype fp32"; `passes()` L54-81). It calls `m.dec_logits` at every K' in kset (L69, L80), and that is a full 12-block `fwd_read` (deltas5 L91-98) each time. For `acc` (K' 0..14), decoder blocks 0-5, which do not depend on S, are recomputed 15 times.

## 3. Designs, code touch points, gates

### Common gate battery (applied per flag; names used in the criteria table)
- **G-OFF**: the edited module with the flag at 0, against the pre-edit module (`git show HEAD:boxes/inject_aw.py` into a scratch copy).
  - Setup: under `scratch/s222_det_wrap.py`, fp32, `--micro 8`, bun args, `--sched_scale 0.01`, 60 steps, one host.
  - Required: log.jsonl byte-identical after dropping the wall-clock key `sec`; identical `ihash` and `dseq`; final params `torch.equal`; ckpt rng states equal.
- **G-FP64**: CPU float64. The trunk and DeltaMem `.double()`; ac() is a no-op on CPU.
  - One step per d ∈ {1, 2, 3, 4}, bs 8, micro 2, hold on with far n 4 and T_far 40, distract on.
  - Required, flag on vs off: per-parameter max|Δgrad| / max|grad| ≤ 1e-12, and |Δloss| relative ≤ 1e-12. Each d is covered because |H| = 5..2.
  - Plus a bitwise fp32 forward wherever the flag claims "same kernels" (EF5 lower values).
  - Memory note: per `latent-exactness-fp64-not-reorder-floor`, never gate against an fp32 reorder floor.
  - Location: the local box has confirmed RAM bit flips (11-01), so the gate runs on a host CPU, or locally twice with identical results.
- **G-DET**: with the flag on, two det-wrapped GPU runs (bun args, 200 steps, fp32, micro 8) must be bitwise identical in log.jsonl (minus `sec`) and params.
- **G-MEM**: `--mem_check` with the flag on. Train peak ≤ mem_budget − 1 GiB (auto_micro headroom rule); `micro` pinned to CTRL's realised value.
- **G-TIME**: `--time_steps 60` (bf16, default kernels) on one host, interleaved off/on/off/on/off/on. The warm-up/compile time is reported separately. Timings from different hosts are censored (3090s vary ~2x).

### EF4 PREFETCH-ITEMS (bitwise on vs off: the only flag that should be)
Mechanism: generate step s+1's items on one background thread while step s computes on the GPU. Item order and ihash must stay identical.

The constraints come from the real code:
- `draw_items` reads the loop variable `step` through the closure (L323, L325, L342) and mutates the bin counters DSTAT, DTYPE and DCAP (L333-336).
- `stream.draw` advances `stream.rng` and `stream.stats` (ow_words `Stream` L147-159).
- `rng_d` is consumed once per step (L552), and `ask_items` reads `step` (L355).
- The bin record (L618-623, L634) and the ckpt (L659-663) read live counters and rng states.

Diff sketch (inject_aw.py):
```python
ap.add_argument("--prefetch", type=int, default=0, choices=[0, 1])        # EF4 (0 = off, unchanged)
def draw_items(d, step_=None, cnt=None):          # step_ None -> the closure's step; cnt None -> DSTAT/DTYPE/DCAP (flag-off path textually as now)
def gen_step(s_):                                 # worker: the only toucher of rng_d / stream / ask_stream when --prefetch 1
    d_ = int(rng_d.integers(1, a.d_max + 1)); cnt = ({}, {}, {})
    its_ = draw_items(d_, s_, cnt) + (ask_items(d_, s_) if NASK else [])
    return d_, its_, cnt, {"rng_d": rng_d.bit_generator.state, "stream_rng": stream.rng.bit_generator.state,
                           "ask_stream_rng": ask_stream.rng.bit_generator.state if ask_stream else None, "stream_stats": dict(stream.stats)}
# loop (L552-553): with --prefetch: d, its, cnt, SNAP = fut.result(); fut = ex.submit(gen_step, step + 1) if step < last else None;
#   merge cnt into DSTAT/DTYPE/DCAP here (same bin as flag-off). ckpt (L659-661) and §235 stats (L635) read SNAP, never live objects.
# try/finally: ex.shutdown(cancel_futures=True); a worker exception re-raises at fut.result() (no silent stream).
```
- Resume: start the worker only after the resume state load (L514-528).
- Futility break and `--stop_at`: the extra prefetched step is discarded. Its draws are not in the ckpt because SNAP is the consumed step's.
- GIL: the main thread releases the GIL inside CUDA syncs (`.item()`, `float(...)` at L555 and L403). If EF0 shows the thread recovers < 50% of the measured data time, EF4b moves generation to a fork-process worker that owns the stream (same SNAP contract).

Gates:
- G-OFF.
- **G-EQ4**: on vs off, det-wrapped, bitwise (log minus `sec`, params, ckpt rng states).
- **G-RES4**: kill at step 150 and resume, against uninterrupted, bitwise.
- G-TIME.
Because EF4 is bitwise, it may join `META` after passing.

Speedup: equal to the measured data-gen share. Predicted 1.03-1.10x with distract_p 1, since every distracted item is tokenised twice (L328).

### EF6 DEC-LOWER-CACHE (eval; bitwise)
In `passes()` (inject_aw_eval L67-80), build the decoder input once per batch. Compute `hdl = DM.gpt_lower(g, g.embed(xd), m.mem.layer)` once. At each K', run only `read_fixed` + blocks 6-11 + `ln_f` + `g.logits` at the last position.
- This is the same op sequence as `fwd_read` (`h = x + wpe_slice`, `_block` 0..5), so it is bitwise on deterministic forward kernels.
- Add a helper `fwd_read_low(g, hdl, mem, S)` in inject_aw_eval.py; deltas5 stays byte-identical.
- Use it only if `"dec_logits" not in vars(m)` and `not m.mem.wo._forward_hooks`. That excludes `read_mask` (L1036 instance patch) and `attn_fade` hooks, which keep the old path.
- Flag: `--dec_cache 1` (eval CLI and `EV.run(..., dec_cache=)`), default 0.

Gate **G-EQ6**: for each of {acc, lp5h, swap_kd, ws_deep, op_count, cue_clock} on bun_s1, on vs off. Every eval_*.json is byte-identical except `sec`, under det-wrap AND default kernels.

Speedup on `acc`: decoder blocks fall from 15·12 to 6 + 15·6, so the part falls ~30%, about 1.4x.

### EF5 HOLD-SHARED-LOWER (fp64-exact; recipe flag)
`hold_step` L179 recomputes `hlp = m.lower(xs[-1])`. That is bitwise the CE's `hls[-1]` from L370: same input, same ac(), same kernels. The recompute exists only because the CE graph is freed by the chunk backward (L555) before the penalty runs.

Diff:
```python
ap.add_argument("--hold_share", type=int, default=0, choices=[0, 1, 2])   # EF5: 1 share the lower; 2 + batch the H passes
# loss_reg (L370), --hold_share and stat is not None:
hls_g = [m.lower(x_) for x_ in xs]; hls = [h_.detach().requires_grad_() for h_ in hls_g]   # CE passes read leaves
stat["_low"] = (hls_g, hls)                                                # kept beside _keep
# main loop, right after (lk / a.micro).backward() (L555) and BEFORE gn_plain (L557):
for j, (hg, hl) in enumerate(zip(*st_["_low"].pop?)): hg.backward(hl.grad, retain_graph=(j == len(hls_g) - 1))
# hold_step L179: hlp = shared hls_g[-1] (graph retained) instead of m.lower(xs[-1]); its backward frees it.
# --hold_share 2: one m.write on cat_t(S_{t-1}) (n|H| rows) with xs[-1] / msk / hlp repeated |H| times; tot = per-row sums reshaped (|H|, n).
```
- gn_plain stays exact: all plain gradients, lower included, are complete before L557.
- Cost of sharing: one lower graph per chunk (~1 GiB) stays live until `hold_step`. G-MEM must run `mem_train` (L453-459) through the same path.
- The far prefix is left alone. Reusing chunk-0 rows would pad the 27 far passes to chunk-0's T. That goes to EF1b instead.

Gates: G-OFF, G-FP64 (on vs off, both sub-modes), G-DET, G-MEM, G-TIME. GPU on vs off is not expected to be bitwise, because of the AccumulateGrad order on the shared `wte`.

Speedup: 1.02-1.04x for mode 1; mode 2 adds launch savings, 1.03-1.06x in total.

### EF2 BUCKET-PAD (fp64-exact; recipe flag)
The loss is a mean of equal-size chunk means (L555 with `ch = bs // micro`). Assigning items to chunks by length is therefore exact in real arithmetic. Right-padding is invisible to real rows:
- causal SDPA;
- the position-wise read (`read_fixed`);
- the ':' gather (`pass_aw` L54);
- dec/aux gather at len-1 (inject_ow L86-102).

Diff:
```python
ap.add_argument("--bucket", type=int, default=0); ap.add_argument("--pad_mult", type=int, default=1)   # EF2 (0 / 1 = off)
NR = a.bs - NASK; order = sorted(range(NR), key=lambda i: (st_len(its[i]), i)) + list(range(NR, a.bs)) if a.bucket else list(range(a.bs))
for k_ in range(a.micro):
    ix = order[k_ * ch:(k_ + 1) * ch]; st_["_far"] = [i < a.hold_far_n for i in ix]; lk = loss_items([its[i] for i in ix], st_)
# loss_reg appends st_["_far"] into each _keep tuple; hold_step L177: w_i from the far flag when present, else the existing off-based rule (flag-off text unchanged).
# cue_inputs_m(m, items, dev, mult) in inject_aw = inject_st.cue_inputs with pad to ceil(T / mult) * mult (nested170.pad / inject_st stay byte-identical).
```
- `its` and `iseq` keep their batch order, so **ihash and dseq stay identical with the flag on**. That is a free check that the items did not change.
- The ask items stay last; `loss_items` asserts this at L438.
- EF2b (eval): sort items by length inside `passes()` and scatter back. This changes batch composition, so it falls under the EF3 flip re-check rules at fp32. It is optional.

Pre-measure **EF2-S0** ($0 CPU, no model): replay 2,000 steps of `draw_items` for seed 1 and report the padded-token fraction with and without bucketing. This sets the prediction before any GPU spend.

Gates: G-OFF, G-FP64, G-DET, G-MEM, G-TIME, plus ihash/dseq equality on vs off.

Speedup at micro 2: 1.02-1.08x (bounded by EF2-S0). Larger at micro 8 (fp32 gates) and on medium. `pad_mult 16` exists mainly to bound EF1's shape set.

### EF1 COMPILE-PASS (fp64-exact on CPU-inductor; recipe flag)
EF1a, inductor fusion on the upper pass body:
- Split `InjectAW.pass_aw` (L47-56) into an eager shell and a body. The shell keeps the data-dependent `pos` assert, the ':' gather and the T = 1 `write_seq`.
- The body `_upper(S, h_low)` is `read_fixed` + `_block` 6..11, returning (h6, h). It is compiled once: `torch.compile(..., dynamic=True, fullgraph=True)`.
- `fullgraph` forbids silent graph breaks. `dynamic=True` avoids one compile per (B, T): chunk 32, far 16, eval 250, H-batch n\|H\|.
- Required: `--cache_lower` and no `--ckpt_passes`. The body is not compiled while `mem.norms`, `alpha_rec`, `wstat` or `DM.READ_CLAMP` is set, or while forward hooks exist (eval `read_mask` and `attn_fade` stay eager).
- Log `torch._dynamo.utils.counters["stats"]["unique_graphs"]` in each 1k-bin record.
- `make_opt` / `lr_at` (fsq120 L731-743) stay eager. `lr_at` writes a Python float into `param_groups` every step (L546), and capturing `opt.step` would recompile or freeze lr.

EF1b, CUDA graphs on the no-grad far prefix only (L187-192):
- B is fixed at `hold_far_n`. T is bucketed by EF2's `pad_mult 16`, giving ≤ 8 graphs.
- One pass is captured per T bucket (`torch.cuda.CUDAGraph`, static S / h_low buffers) and replayed T_far-1 times.
- Training-graph capture (cudagraph trees under autograd) is out of scope: bun's peak is already 15.5 GiB, and the pools are per shape.

Gates:
- G-OFF.
- G-FP64: inductor CPU in fp64, compiled vs eager, ≤ 1e-12.
- G-DET: the GPU gathers sit outside the body, so no atomics are compiled. If inductor refuses under `use_deterministic_algorithms`, the gate is censored and rerun with `torch._inductor.config.fallback_random` and autotune off.
- EF1b: far-prefix states captured vs eager, bitwise in fp32 under det-wrap, since the graph replays the same kernels.
- G-MEM, G-TIME (compile warm-up reported).

Speedup:
- EF1a: 1.08-1.20x. Fusing the LN / bias / GELU / residual adds of 768-wide bf16 blocks; GEMMs stay cuBLAS.
- EF1b: 1.04-1.08x overall (≈1.5-2x on the ~12% far share).
- The backlog's 1.2-1.5x is likely optimistic for GPT-2 small at 3,200 tokens per GEMM.

### EF3 EVAL-LOWP (an eval-law change: needs an Audit ruling; descriptive parts only)
Diff (inject_aw_eval.py only):
```python
ap.add_argument("--lowp_parts", default="")        # comma list; '' = every part fp32 (unchanged)
# run() L1316: ctx = lowp(m, g) if p_ in LOWP else NULLC; with ctx: r_ = {...}[p_](); r_["eval_dtype"] = "bf16-trunk/fp32-mem/fp32-head" for those parts
@contextlib.contextmanager
def lowp(m, g):                                    # bf16 autocast on the trunk; DeltaMem already runs fp32 (read_fixed L162 / write_seq L183 disable autocast)
    g.logits = lambda h: torch.autocast("cuda", enabled=False).__enter__() or (h.float() @ g.w("wte.weight").T)   # sketch: fp32 LM head (instance patch, removed in finally)
    with torch.autocast("cuda", dtype=torch.bfloat16): yield
```
The real code uses a small function with a proper `with` block, not the one-liner. The LM head stays fp32: bf16 logits of magnitude ~16 have a resolution of ~0.06 and would manufacture argmax flips.

Margin re-check inside `passes()`:
- Under lowp, compute the top-1 minus top-2 margin over the 11 NUM_IDS at every recorded K'.
- Re-run in fp32, in the same call, every item whose minimum margin is < m*, and overwrite its rows.
- Not allowed when `edit` or `on_state` is set. Those parts are either whitelisted for full fp32 or excluded.
- m* is fixed by calibration: twice the largest margin among flipped cells in the calibration set.
- Rationale: bun trains under bf16 autocast (L273-274), so a bf16 trunk at eval is on-distribution. The question is only argmax stability over 14 passes. S stays fp32 throughout.
- TF32 is not offered: on a GA102 3090 its tensor-core peak equals the fp32 SIMT peak.

Calibration: **EF3-CAL**, on bun_s1 plus one more seed.
- Parts acc, ws_deep, lenstep, op_count, cue_clock and stab: full fp32 vs full lowp.
- Then a running audit: in 1 of every 5 lowp cells the whitelisted parts are also run in fp32.

Speedup: ~2-3x on GEMM-bound parts. The 3-5x in the backlog assumes tensor-core gains that the fp32 DeltaMem and the per-pass syncs cap. Combined with EF6, `acc` is about 3-4x.

## 4. Landing order and dependencies
1. **EF0 + EF2-S0 + EF4-S0** ($0.1 GPU; S0s are $0 CPU). torch.profiler over 20 bun steps; a wall timer around `draw_items`; a counter for far-prefix time. Re-rank 2-6 by the measured shares.
2. **EF4**: bitwise, zero recipe risk, can be META.
3. **EF6**: bitwise eval; shrinks EF3's case.
4. **EF5**, then **EF2**: fp64-exact recipe flags; EF2 gives EF1 its shape buckets.
5. **EF1a, then EF1b**.
6. **EF3**: last, behind an Audit eval-law ruling.

**Recipe-bundle rule:**
- EF5, EF2 and EF1 change rounding trajectories, and one seed can flip basins (`latent-basin-entry-mechanism`). They therefore ship as one named bundle **FASTR-1**, adopted at a registration boundary.
- Within any registration every arm and its comparator share the same FASTR setting. The flags are not in META, so `args_diff` shows them.
- §268 LINEAGE-SEED and any arm compared to a stored pre-FASTR run stay FASTR-off.
- Edits respect `no-new-args-after-freeze`: no EF flag is added to a module whose args a running registration froze until that cell's tarball has shipped (`latent-deploy-ships-working-tree`).

Predicted stack on bun:
- EF4 x EF5 x EF2 x EF1a x EF1b ≈ 1.03 x 1.04 x 1.04 x 1.12 x 1.05, about **1.3x (range 1.2-1.5x)**.
- Eval: EF6 + EF3 ≈ 3x on the descriptive parts.

## 5. Criteria (pre-registration ready)
Binding (adoption) criteria:

| ID | Criterion | Threshold | Label |
|---|---|---|---|
| B1 | G-OFF for every new flag | byte-identical log (minus `sec`), ihash/dseq, params, rng | EXACT-OFF / FAIL |
| B2 | G-FP64 for EF5, EF2, EF1a, each d 1..4 | max rel Δgrad ≤ 1e-12 and rel Δloss ≤ 1e-12; NEAR if in (1e-12, 1e-10] (re-check sum orders, no adoption) | EXACT-FP64 / NEAR / FAIL |
| B3 | G-DET for the flag-on path | bitwise across 2 det-wrapped runs | DET-OK / FAIL |
| B4 | G-EQ4 + G-RES4 (EF4), G-EQ6 (EF6) | bitwise on vs off; resume bitwise | BITWISE / FAIL |
| B5 | G-MEM | peak ≤ budget − 1 GiB at CTRL's micro | MEM-OK / FAIL |
| B6 | G-TIME, 3 interleaved pairs, same host | median speedup ≥ 1.02 and min(on) < min(off); NEAR if median in [1.00, 1.02) | FAST / NEAR / NO-GAIN (kept off) |
| B7 | EF3-CAL per part | no flips at margin ≥ m* (CP 95% upper ≤ 1e-3 on ≥ 3,000 item-K' cells); every aggregate field \|Δ\| ≤ 0.005 after the re-check, NEAR if in (0.003, 0.005]; Wilson CIs on rates | LOWP-OK / NEAR / FP32-ONLY |

Descriptive criteria:
- EF0 component shares;
- EF2-S0 padded fraction;
- the compile warm-up and graph count;
- for FASTR on vs off over 1k steps (default kernels), |Δ answer_ce| per 100-step record (expected chaotic divergence, so not binding);
- the EF3 flip census by d and K'.

**OC block (outcome priors):**
- B1 0.85 per flag on the first try. The risk is refactoring the `step` closure.
- B4 for EF4 0.75. The risk is GIL scheduling, never ordering, because the worker is the only consumer.
- EF6 0.95.
- B2 for EF2 / EF5 0.9. EF1a 0.7: inductor CPU fp64 may reassociate reductions beyond 1e-12; if so, the result is NEAR and EF1a is held.
- B3 for EF1a 0.6.
- B6 FAST: EF4 0.6, EF5 0.6, EF2 0.5, EF1a 0.65, EF1b 0.6.
- B7 LOWP-OK on at least 4 of 6 parts: 0.55.

**Censoring:**
- A gate that dies from infrastructure (OOM in a co-tenant, host loss, a det-mode NotImplemented in inductor) is censored and rerun, never scored FAIL.
- Timings across hosts are censored.
- EF3 cells where fp32 itself has margin < 1e-6 (true ties) are excluded from the flip count and reported.
- Steps after a futility break or `stop_at` are outside the G-EQ4 comparison.

Shortcut-exclusion arm: the "shortcut" an efficiency flag can take is a silent change to the computation or the data. B1 (off identity), B2 (math identity) and the ihash/dseq equality with EF2/EF4 on are the exclusion arms. Gating happens at the claimed scale: the full bun config with hold, far and distract on, each d, and T_far at its 40 maximum.

Not applicable: C2 seed counts and zero-init usage monitors. No new learnable pathway is added.

## 6. Scorer sketch: `theory/score_ef.py`
- **Inputs:**
  - `gate_ef/<flag>/gate_off.json` {log_equal, ihash_equal, dseq_equal, params_equal, rng_equal};
  - `gate_fp64.json` {d: {max_rel_grad, rel_loss}};
  - `gate_det.json` {bitwise};
  - `gate_eq.json` / `gate_res.json`;
  - `mem_check.json` (peak_gib_train, micro);
  - `gate_time.json` {host, sps_on[3], sps_off[3], warmup_sec};
  - `ef3_cal.json` {part: {n_cells, n_flip_guarded, n_flip_unguarded, m_star, fields: {name: [fp32, lowp]}}}.
- **Labels:** the B1-B7 labels above, and FASTR-1-READY only if every component flag is EXACT-OFF ∧ (EXACT-FP64 ∨ BITWISE) ∧ DET-OK ∧ MEM-OK ∧ FAST.
- **Fixtures (theory/fixtures/ef/):**
  - one passing set;
  - a 1e-11 NEAR case;
  - a log differing only in `sec` (must pass);
  - a log differing in `ihash` (must fail);
  - a cross-host timing (censored);
  - an EF3 part with one unguarded flip in 3,000 cells. It must fail: with 3,000 cells, CP 95% upper ≈ 1.6e-3 > 1e-3.
- Raw values are tested, per `latent-score-raw-values`.

## 7. Cost on an RTX 3090 (~$0.2/h)

| Item | GPU time | $ |
|---|---|---|
| EF0 profile + G-TIME baselines | 0.4 h | 0.08 |
| G-OFF / G-DET / G-EQ4 / G-RES4 (fp32 det, micro 8, ~2-3 s/step, 60-200 steps x ~10 runs) | 1.2 h | 0.24 |
| G-EQ6 (6 parts x 2 x 2 kernel modes on bun_s1) | 0.6 h | 0.12 |
| G-MEM + G-TIME for 5 flags | 0.6 h | 0.12 |
| EF3-CAL (6 parts fp32 + lowp, 2 seeds) | 1.0 h | 0.20 |
| Host setup overhead | | ~0.2 |
| **Total** | **~3.8 h** | **~$1.0** |

G-FP64 and the S0s run on CPU ($0). Payback: ~$0.2 per bun-length run, so it pays back after ~5 runs. Lanes A-E plan more than 20.

## 8. RED TEAM
(to be filled by the red-team sub-agent)

## 9. References (fetched arxiv.org/abs; titles match)
- 2205.14135: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Cited for exact attention under kernel changes.
- 2107.02027: Efficient Sequence Packing without Cross-contamination: Accelerating Large Language Models without Impacting Performance. Cited for padding waste and the length-bucketing rationale.
- 1710.03740: Mixed Precision Training.
- 1905.12322: A Study of BFLOAT16 for Deep Learning Training.
