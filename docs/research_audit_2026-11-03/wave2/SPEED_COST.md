# SPEED_COST — training throughput and cost review (Audit wave 2)

Written 2026-11-03 17:21 EST (Audit sub-review; read-only, no jobs run). Scope: the current main recipe
(boxes/inject_aw.py, "bun" family: `--think_k 8 --cache_lower --fast_add --write_src answer --cue step --d_max 4
--hold_inc 1.6 --distract_p 1 --micro 2 --bs 64`, 8k steps, in-process evals via boxes/inject_aw_eval.py), as run on
rented RTX 3090s. Numbers below are from the ledgers / guard logs in logs/ and from reading the code; anything marked
*est.* is a reading-based estimate that EF0 STEP-PROFILE must replace before money is spent on it.

## 1. Where the wall-clock and the dollars go

§1 is filled from the ledger pass (logs/s2*_admitted.log, logs/s2*_bad_cost.log, logs/s2*_guard_b*.out,
logs/gen_deploy.log) — see §1.1–§1.3.

### 1.1 Pipeline per cell (scratch/s2xx_fill.sh, scratch/gen_vast_job.sh, scratch/auto_drain_gen8_s224.sh)

offer pick -> `vast create` + ssh wait -> 1.2-1.5 GB tarball upload (code + the two init model.pt) -> pip + GPT-2 fetch +
hostcheck -> HOSTPROBE (200 steps of nested170 inject K8, reject > 0.854 s/step) -> `*_mem` (mem_check) -> `*_gtime`
(60 timed steps) -> ADMIT -> optional host gates / eval-only lines -> the train line with in-process eval parts ->
QUEUE_DONE -> drain watcher (10-min poll) tar + scp + sha -> destroy.

### 1.2 Measured phases, 3090 8k cells (12 cells with full timelines: b372-375, b393-394, b402-405, b410-411)

No 24k cells exist in §250-§270 (mix: 8k x17, 16k x11, 12k x6, 2k x5, eval-only x4). Minutes are medians [range];
$ at $0.21/h.

| phase | min | $ |
|---|---|---|
| (a) create -> STARTED | 6.0 [4.6-15] | 0.02 |
| (a) create -> HOSTPROBE result | 9.8 [7.9-20] | — |
| (a) create -> ADMIT (gtime done, training starts) | 19 [11-29] | 0.07 |
| (b) host gates / eval-only pre-lines, when present | 22-60 | 0.08-0.21 |
| (c) training, 8k steps at 1.1-1.8 s/step | 167 [149-240] | 0.58 |
| (d) in-process eval parts | 22.6 [7.7-38] | 0.08 |
| (e) QUEUE_DONE -> destroy (poll + fetch) | 16 [2.6-31] | 0.06 |
| total create -> destroy | 258 [189-311] | 0.90 (realised 0.66-1.14) |

Non-training share: median 23% of an 8k cell (20-30% without gates; 45-47% for b393/b394, which carried a 55-60 min
eval-only line). Realised $ = hours x $/h within cents, so the decomposition is complete. Extrapolated 24k cell at
1.6 s/step: 640 min training ($2.24) + 19 setup + ~25 evals + ~15 drain ~ 700 min, ~$2.45, non-training ~9%. 16k cells
(11): 266 [215-433] min, non-training 19% [15-27%], realised $0.82 median.

Eval parts by time (host logs, PART lines): `acc` mean 533 s (max 1528), `frames` 534 s, `insert_ho` 522 s, `mean_s`
256 s; every other part < 100 s. Drain: the watcher's `sleep 600` alone idles ~5 min after QUEUE_DONE; `--keep_final_ckpt`
or medium models balloon it (b396 40 min, b401 50 min, b406 41 min at $0.42-0.48/h ~ $0.35 each). The ckpt mirror pulls
1.52 GB every ~1k steps during training (bandwidth, not $).

### 1.3 Totals and waste (ledgers)

§250-§270: 44 admitted attempts, projected $37.69, realised $35.59 (mean $0.83, median ~$0.70); bad_cost $2.41 over 43
rejected deploys / 640 billed minutes (projection refusals 14 = $0.85, deploy-fail / no-start 19 = $0.81, hostprobe-slow
6 = $0.44, vanished 3 = $0.25) = 6.3% waste; plus 9 voided attempts inside realised lines = $3.01 (b376/377 OOM at micro 1
$1.23, b414 a1 guard STOP $0.50, b397 gates FAILED $0.48, b398 $0.25, b413 OOM $0.22, b412 gate crashes $0.28, b402 a1
$0.09) -> 14% including voids. §211-§270 overall: realised $59.36, bad $5.13 (8.0%); all s2* ledgers: 69 admitted
projections (median $0.96) and 76 realised lines (median $0.72, sum $62.64, max $3.55), hostprobe + deploy waste $3.56.
3090 $/h 0.147-0.244 (median 0.202); A6000 0.42-0.48. §270 (b415/416) still running, proj $1.96.

s/step: HOSTPROBE 0.51-0.72 (median 0.61); G-TIME (ledger sps) 3090 inject cells 0.44-1.48, median 1.23; in-run guard
marks (38 cells) @500 median 1.18, drift to the last mark median 0.00, max +0.32 (the bun recipe, §1.4). G-TIME
over-estimates the steady rate by ~8% except for bun, where it under-estimates by 22% (hold starts at 500).

Observations: (1) setup is 19 min / $0.07 per cell (7% of an 8k cell), of which ~6 min is boot + the 1.2-1.5 GB upload
(two init model.pt shipped every cell) and ~9 min is mem_check + G-TIME + 60-s filler polls; (2) evals are 23 min, three
parts (`acc`, `frames`, `insert_ho`) are ~9 min each; (3) drain is 16 min, 5 of them the poll interval; (4) voided attempts
(OOM at micro 1, gate crashes, missing shipped shas) cost more than all host rejections combined ($3.01 vs $2.41);
(5) no-offer waits (fill_s262 10 x 10 min) are unbilled latency, not $.

### 1.4 The step-time series inside a cell (guard CHECK lines)

The guard's `s_per_step` is the recent rate between polls (scratch/cost_guard.py:30-36), not an instantaneous value.
Three bun-recipe cells (logs/s265_guard_b410.out, logs/s270_guard_b415.out, logs/s270_guard_b416.out):

| cell | @700 | @1000-1200 | @2100 | @4200 | @6000-8000 |
|---|---|---|---|---|---|
| b410 (gb_s1) | 1.43 | 1.56 | 1.73 | 1.72 | 1.71-1.75 |
| b415 (of_s1) | 1.48 | 1.57 | 1.73 | — | — |
| b416 (of_s2) | 1.38 (@500) | 1.82 (@1200) | — | — | — |

The rate rises ~25% over the first ~2k steps and then holds at ~1.72 s/step. Two explanations, both checkable at $0:
(a) `--hold_inc` switches its penalty passes on at `--hold_start 500` (inject_aw.py:558: `inc_on = ... step >= a.hold_start`),
so steps < 500 are cheaper — this is a recipe cost, not drift; (b) a slow residual rise 1000 -> 2100 is consistent with
3090 clock/thermal settling on some hosts. EF0 should log `nvidia-smi --query-gpu=clocks.sm,temperature.gpu` alongside
`sec` per 100 steps to split (a) from (b). Projections should use the post-2k rate, which cost_guard already does (recent-rate).

## 2. Code-level speedup candidates (training loop)

### 2.1 What a bun step actually computes (reading inject_aw.py / deltamem.py / nested170.py / deltas5.py)

Per step, bs 64 = 48 regular items (`draw_items`) + 16 ask items (`--ask_p 0.25`), split into `--micro 2` chunks of 32 (24 + 8):

| component | code | work per chunk (bf16 trunk under autocast; memory branch fp32) |
|---|---|---|
| lower cache | `m.lower(x)` per distinct pass input (inject_aw.py:370; nested170.py:241 -> deltamem.gpt_lower) | blocks 0-5 on Td <= d distinct inputs x 24 items x T~80-120 tokens, with grad |
| think passes | `m.write` -> `InjectAW.pass_aw` (inject_aw.py:47-56): read_fixed at block 6, blocks 6-11, `write_seq` on the gathered ':' row (T = 1) | 8 passes x 6 blocks x 24 x T, with grad |
| aux CE | `m.aux_logits` (inject_ow.py:92) -> `deltas5.fwd_read`: a FULL 12-block read-only forward over `[EOT, Step, t, :] + target` (~6-10 tokens) | t <= d passes x 12 blocks on 24 x ~8 tokens (tiny, launch-bound) |
| decoder | `m.dec_logits` (inject_ow.py:86) -> fwd_read, 12 blocks on the full dec text | 1 x 12 blocks x 24 x T |
| ask loss | `loss_ask` (inject_aw.py:421-431): lower once, 8 pass_aw + full-vocab colon logits | 8 passes x 6 blocks x 8 x T |
| HOLD penalty | `hold_step` (inject_aw.py:159-199): recomputes `m.lower(xs[-1])` (:179), then (K - d - 2) grad passes `m.write` per chunk (:180-181) | avg 3.5 passes x 6 blocks x 24 x T, with grad, + 1 lower |
| HOLD far pass | :185-198: `T_far ~ U{16..40}` no-grad passes on 16 items, then 1 grad pass | ~27 no-grad passes x 6 blocks x 16 x T (once per step) |
| WS / stats | `.tolist()` per pass under `--perm_p` (:389), `int((..).sum())` per ask pass (:430), `.item()` per aux pass (:403), `float()` per hold ratio (:174-175, :183, :198), `assert bool((x[ar,pos]==COLON).all())` in every pass_aw (:55) | each is a host-device synchronisation |
| optimiser | `make_opt` = plain `torch.optim.AdamW` (fsq120.py:734; foreach, not fused), `clip_grad_norm_` (:566) | 124M + 2.4M params |
| data | `draw_items` serial on the main thread (:316-346); distracted items tokenised twice (:328) | ~100-150 tiktoken encodes/step |
| ckpt | full model + AdamW state (`torch.save`, :659-663) every `--ckpt_every` (1000 for bun) ~1.5 GB; model.pt 0.5 GB at the end (:673) | 8 saves x ~10-20 s |

FLOP estimate (est.): ~2.0M token-block-equivalents per step (grad passes counted x3, the far prefix x1) x ~14 MFLOP per
GPT-2-small block-token ~ 28 TFLOP/step. At 1.72 s/step that is ~16 TFLOP/s achieved, ~25% of the 3090's dense bf16 peak,
so the step is roughly half compute and half launch gaps / syncs / small-batch under-utilisation (the far pass runs 16
items; the aux passes run ~8 tokens). HOLD (penalty + far pass) is ~30-35% of the FLOPs, consistent with the ~0.3 s/step
rise after step 500 in §1.4.

Two corrections to the brief's framing. (i) In the AW recipe the DeltaMem write is a single-token `write_seq` (T = 1,
inject_aw.py:56) and the read is one vectorised einsum over the sequence (deltamem.read_fixed), so the memory branch's
per-token Python loop is NOT the bottleneck any more; it is what the hostprobe still measures (nested170 inject K8,
gen_vast_job.sh:58), which is why probe s/step (0.5-0.84) and cell s/step (1.2-1.8) correlate only loosely. (ii) The memory
branch is a few percent of the FLOPs; bf16 there would save nothing measurable and is the exactness-critical part (fp32
sums, `scratch/gate_fastadd.py`). Rejected, see C9.

### 2.2 Candidates

Gains are per training step unless stated; "bitwise" means identical outputs under scratch/s222_det_wrap.py (default SDPA
backward is not run-to-run bitwise, memory `gpu-bitwise-gates-need-det-wrap`). Every non-bitwise change is a new recipe flag
that must never be mixed between a cell and its comparator (memory `latent-exactness-fp64-not-reorder-floor`).

| # | candidate | expected gain (est.) | numerical risk | validation gate | cost to build |
|---|---|---|---|---|---|
| C1 | **Remove per-pass host syncs**: hoist the `':'`-position assert out of `pass_aw` (check once per batch on the CPU token lists, inject_aw.py:55 runs ~50x/step incl. far passes); accumulate the per-pass stats (`:382, :389, :403, :430`, hold ratios `:174-175, :183, :198`, `float(lk.item())` :555) in GPU tensors and read them once per step or per `--log_every` | 5-15% (depends on how CPU-bound the loop is; EF0's trace of `cudaStreamSynchronize` gaps decides) | none: no arithmetic changes | det-wrapped 200-step run, bitwise log.jsonl fields + final params vs HEAD | 0.5 day, $0.1 local |
| C2 | **CUDA graphs for the far-pass prefix** (:187-192: 15-39 no-grad `m.write` calls on 16 items) and optionally the aux forwards: capture one graph per (B, T) shape, replay per pass | 5-10% (the prefix is ~0.15-0.25 s/step of launch-bound work) | bitwise if shapes are not bucketed (same kernels replayed); a per-shape graph cache has ~40-80 T values per run (capture ~0.2 s each, amortised) | det-wrapped bitwise vs eager on 200 steps; graph-cache memory check in `--mem_check` | 1 day |
| C3 | **torch.compile on `deltamem._block`** (+ `read_fixed` / `write_seq` elementwise chains), `dynamic=True`; mode default, not reduce-overhead | 10-25% (fuses LN + bias + gelu + residual; SDPA is already fused) | not bitwise (fused reductions); bf16 trajectory differs -> new recipe flag `--compile`; torch 2.5.1 inductor on dynamic T recompiles | fp64 agreement of loss + grads vs eager <= 1e-10 over 20 steps (inductor supports fp64), + a 200-step bf16 run inside the reorder floor; smoke the plain path | 1-2 days; warm-up 1-3 min/host |
| C4 | **Length bucketing / sorting** (EF2): pad T to a multiple of 16 and group by length within a chunk; right-padding is exact for causal attention at the real positions and the ':' gather | 3-10% (items share d per step so padding waste is already moderate) | fp64-exact at real positions; kernel selection changes with T -> not bitwise | fp64 agreement <= 1e-12 of loss/grads on padded vs unpadded | 0.5 day |
| C5 | **Fused AdamW** (`fused=True`) and `foreach` clip | 1-2% (~20-30 ms/step on 126M params) | not bitwise (different accumulation order); new flag | fp64 params after 10 steps <= 1e-12 | 1 hour |
| C6 | **Prefetch item generation** in a thread (EF4); tokenise distracted items once (:328 builds both tok_aw variants only for an assert) | 1-3% | none (same items, same order; the RNG draws must stay on the main thread in order) | bitwise ihash per 1k bin + log.jsonl | 0.5 day |
| C7 | **HOLD shared lower** (EF5): reuse the CE chunk's `hls[-1]` for the penalty instead of `hlp = m.lower(xs[-1])` (:179) | 3-5% | needs `retain_graph=True` on the CE backward (the design keeps the graphs disjoint so gn_plain is read before the penalty): +activation memory at micro 2 on 24 GB -> OOM risk; gradient identical in exact arithmetic | fp64 gate + `--mem_check` | 0.5 day; risky on 24 GB |
| C8 | **Eval in TF32/bf16 for descriptive parts only** (EF3) | 2-3x on eval matmuls (3090 fp32 CUDA cores ~20 TFLOP/s achieved vs ~50 bf16); eval is ~N min of a cell (§1) | the eval law is fp32 (`eval_dtype: fp32`, inject_aw_eval.py run()); argmax near-ties flip (memory `latent-repro-match-batch-size`: bs alone flipped 5-7/1000) -> Audit ruling; registered thresholds stay fp32 | per-part fp32 re-check of flipped items; score only on fp32 for registered gates | ruling + 0.5 day |
| C9 | bf16 memory branch | ~0 | high (exactness-critical sums) | — | rejected |
| C10 | **Larger micro** (micro 1) | 5-10% from fewer launches and larger matmuls | needs ~22+ GiB (memory `auto-micro-headroom`: 20.95/21 at micro 1 for §251) -> only on 32-48 GB cards; batch-shape kernels change -> new numerics class | fp64 gate; same-class card for the whole family | — (hardware-bound) |
| C11 | **Checkpoint traffic**: keep `--ckpt_every 1000` x 1.5 GB saves (~1-2 min/cell, fine) but (a) ship model.pt before anything else and destroy the host only after the upload completes, (b) `--keep_final_ckpt` only when a later stage inits from it (rule exists, memory `keep-final-ckpt-only-if-needed`), (c) add `inet_up>=200` to the offer filter | drain minutes, not step time (§1.3) | none | — | 0.5 day in the drain script |
| C12 | Eval batching | eval_bs is already 250; the `run_long` K' 128 part dominates eval FLOPs (128 passes x d 1-6); CUDA graphs per (bs, T) would be bitwise for forward-only fp32 under det wrap | eval 10-20% | bitwise | det-wrapped eval_*.json equality | 1 day |

Ordering within the trunk: C1 first (free, bitwise, and it also makes EF0's trace readable); C2 next (bitwise); C3/C4/C5
together as one `--fast` recipe flag with one fp64 gate, adopted only at a family boundary; C8 only with an explicit Audit
ruling on the eval law.

What EF0 STEP-PROFILE should record ($0.1, one 300-step run on a 3090 with `torch.profiler`, the bun args): per-step wall
split into lower / passes / aux / dec / ask / hold-penalty / far-prefix / optimiser / data / syncs; count of
`cudaStreamSynchronize` and `Memcpy DtoH` events per step; GPU busy fraction; SM clock. That ordering replaces every
*est.* in this section.

## 3. Hardware options

Sources: memory notes (basin flip §123 seeds 1/2 reproduced on three 3090 hosts and did the opposite on a 4090; §258's
A6000 anchor differed from the 3090 at 2k, loss 0.364 vs 0.546 with the same flags, docs/THEORY.md; same-class 3090 hosts
differ 2.1x in s/step; same seed + recipe + card class reproduces exactly), the filler's offer filter
(scratch/fill_generic_px.sh:17: `gpu_name=RTX_3090 num_gpus=1 reliability>0.97 dph<=0.30 disk_space>=40 inet_down>=300
cuda_vers>=12.4`, driver-580 post-filter, no `verified=true`, no `inet_up`), and read-only `vastai search offers` on
2026-11-03 (on-demand and `--type=bid`; nothing created).

Ledger facts (178 admitted lines, logs/s2*_admitted.log): realised 3090 $/h min 0.147 / p25 0.199 / median 0.207 / p75
0.224; no time-of-day pattern in the deploy log's 209 labelled 3090 offers (median 0.201-0.208 in every 4-h bucket, censored
by the $0.30 cap). Hostprobe (logs/gen_deploy.log): 151 probes, 0.50 / 0.61 / 0.84 s/step min / median / max, 32
HOSTPROBE_SLOW rejects = ~21% of probed deploys paid for setup and thrown away.

| class | $/h on-demand (bid) 11-03 | VRAM | rel. step speed (est.; launch-bound loop) | rel. $/step | basin risk | note |
|---|---|---|---|---|---|---|
| RTX 3090 | 0.15-0.22 median (bid 0.08 min / 0.14 median) | 24 | 1.0 | 1.0 (bid 0.5-0.7) | none (reference class) | 11 non-580 offers under the filter, 6 at <= $0.25 |
| 3090 Ti | 0.255 (n 3) | 24 | ~1.1 | ~1.4 | low-med (same GA102) | thin supply |
| RTX 4090 | 0.35 min / 0.49 median (bid 0.20-0.27) | 24 | 1.3-1.7 | 1.0-1.5 (bid 0.6-0.9) | HIGH (documented flip) | whole experiments only |
| 4080S / 5080 | 0.18-0.25 / 0.22-0.33 | 16 | 1.2-1.5 at equal micro | 0.6-0.9 nominal | high (Ada / Blackwell) | 21 GB peak forces micro >= 4; L_max OOM risk |
| RTX 5090 | 0.50 min / 0.87 median (bid 0.24) | 32 | 1.7-2.5 (unmeasured) | 1.0-1.5 (bid ~0.5) | HIGH; needs a cu12.8+ image (ours is 2.5.1+cu12.4) | cheapest 32 GB; micro 1 fits |
| RTX A5000 | 0.23-0.27 (n 2) | 24 | 0.8-1.0 | 1.1-1.4 | medium | clock-stuck host seen (Sept) |
| RTX A6000 | 0.46-0.48 (bid 0.34); fleet paid 0.40-0.47 | 48 | 0.9-1.0 | 2.0-2.3 | medium (differs, §258) | cheapest 48 GB for larger bases |
| A100 PCIe / SXM | 0.40-0.68 / 0.43-1.0 | 40-80 | 1.0-1.2 | 2-3 | medium | dense large-base work only |
| L40S / L40 | 0.73-0.91 | 48 | 1.3-1.6 | 2.5-3 | high (Ada) | no |

The speed column is an estimate for a launch-bound loop (dlperf 3090 44 / 4090 97 / 5090 194 overstates what this code
realises); a 200-step `--time_steps` probe on one host per class (~$0.10 each) pins it.

Options, in order of value:
1. **3090 interruptible (bid) at ~$0.10-0.14/h** instead of ~$0.21: 40-50% off per step with no class change. Needs a
   bid-type create in fill_generic_px.sh, `--ckpt_every/--resume` (exist) with the ckpt mirror on, and a requeue path that
   tolerates a stopped instance (the resume rules in memory `resume-extension-new-dir` apply). Pilot on one unregistered
   cell; a pre-emption mid-eval re-runs the eval only. Note that a resumed run is bitwise the same only on the same card
   class and with the same micro.
2. **Cut probe waste** (~21% of probed deploys): add `verified=true` and a `cpu_ghz >= 3.5` pre-filter (the slow hosts are
   CPU-launch-bound), extend scratch/vast_prefer_ips.txt from the 151 HOSTPROBE lines, and add `inet_up>=200` for drains.
3. **Same-class co-tenancy for eval-only cells**: two processes on one card change nothing numerically (each process's
   kernels and shapes are unchanged); fp32 eval cells could share a 3090 with each other where VRAM allows. Training cells
   at ~21 GB cannot pair on 24 GB.
4. For the approved 24-48 GB larger-base work: A6000 (~$0.40-0.47, bid $0.34) is the cheapest 48 GB; the 5090 (bid $0.24)
   is the cheapest 32 GB but is a new numerics class and needs a new image. Run whole experiments on one class; never mix
   with 3090 cells in one comparison.
5. Do not chase 4080S / 5080 / 4070 for cost: below the 16 GB rule or forced micro >= 4, and each is a new basin class.
   A 4090 bid at $0.20-0.27 is the only faster-card option at <= 3090 $/step, only for self-contained experiments.

## 4. Ranked top-5 plan

Ranked by $ saved per $ and risk spent, for the current cadence (~$35/20 days, 8k bun cells at ~$0.90 each; training is
~65% of a cell's $, setup + evals + drain ~23%, waste 6-14%).

| rank | action | expected effect | numerics | first step |
|---|---|---|---|---|
| 1 | **EF0 STEP-PROFILE + C1 sync removal** (hoist the pass_aw assert, GPU-side stat accumulators, one read per log interval) | 5-15% of training time; makes the profile readable | bitwise (det-wrapped gate) | $0.1 local 300-step profile with the bun args; ship C1 as the next trainer blob after the gate |
| 2 | **Process fixes outside the step** (no code-numerics risk): drain poll 600 -> 120 s and model.pt-first fetch (~5-10 min/cell); skip the two init-model uploads via a per-host cache / `vast copy` (~3-5 min); `verified=true` + `cpu_ghz>=3.5` + `inet_up>=200` in the filler filter and vast_prefer_ips.txt grown from the 151 probe lines (cuts the ~21% probe rejects); mem_check pinned via args (OOM voids were $1.23) | 10-15 min/cell (~$0.05, 5-6%) + most of the 6-14% waste | none | script edits in fill_generic_px.sh / auto_drain; a G-D on the drain change |
| 3 | **3090 interruptible (bid) pilot** on one unregistered cell with the ckpt mirror on | 40-50% off $/step if pre-emptions are rare; the largest single lever | none if same class + same micro on resume (resume is bitwise only then) | bid-type create + requeue-on-stop path; measure pre-emption rate over 3 cells before registered use |
| 4 | **C2 CUDA graphs for the far-pass prefix** (and aux forwards) | 5-10% | bitwise under det wrap (per-shape graph cache) | 1 day; gate = 200-step bitwise vs eager |
| 5 | **`--fast` recipe flag = C3 torch.compile + C4 bucketing + C5 fused AdamW**, adopted only at a family boundary, with one fp64 gate | 15-35% of training time (the backlog's 30-45% is reachable only with this bundle plus C1/C2) | not bitwise -> new class of trajectories; never mixed inside a comparison family | build after EF0 confirms the trunk share; gate fp64 <= 1e-10 loss + grads, bf16 within the reorder floor |

Held back: C8 eval TF32/bf16 (needs an Audit ruling on the eval law; evals are ~9% of a cell), C7 shared lower (OOM
risk on 24 GB), C9 bf16 memory branch (rejected), C10 micro 1 (needs 32-48 GB and a new class), GPU-class moves
(every class move is a new basin; 4090 is documented to flip seeds). Larger-base work goes to A6000 (48 GB) or 5090
(32 GB, new image) as whole experiments.

Honest ceiling: with 1 + 2 + 4 (all numerics-safe) a bun cell drops from ~$0.90 to ~$0.70-0.75; adding 3 (bid) to
~$0.40-0.50; adding 5 to ~$0.30-0.40 but on a new trajectory class.
