# Training to-do: runs needed to support the Part III claims

> **Status (2026-09-28): done / superseded.** Phases 0–3 were run as PROTOCOL4 (Part IV) and PROTOCOL5 (Part V);
> Phase 4 (transfer to ProsQA) as PROTOCOL6 addenda C–N (Part VI). Kept because PROTOCOL4.md cites it. Outcomes:
> `report/findings.md`; current work: `docs/CURRENT_TASKS.md`.

Written 2026-09-24, after the Part III scorecard (`results/protocol3_confirm.json`) and a literature check
(closest work: Zhang et al. 2607.20594, Kuo et al. 2606.29983, Aswal et al. 2606.12689).
Freeze each item's prediction in a PROTOCOL4 addendum **before** the run starts, as in PROTOCOL3.

**Base recipe** (every run below unless noted; seed-0 main recipe, as in `runs/nostep_s0/config.json`):

```
python -m latent.train --mode coconut --K 6 --bs 512 --steps 30000 --warmup 300 --eval_every 1000 \
  --aux_steps 10000 --aux_anneal 5000 --children_weight 1 --curriculum gated --init gpt2 \
  --seed <s> --device <dev> --resume --out runs/<name>
```

Launch through `scratch/stage1_guard.py` (stage-1 restart rule). Gate = addendum J: overall delayed
balanced accuracy ≥ 0.85 **and** depth-matched 5-hop ≥ 0.60. Keep failed attempts on disk and in the log.

Timing from past runs: K = 4 ≈ 1.0 h, K = 8 ≈ 1.25 h (RTX 4060), K = 6 no-step ≈ 1.75 h (RTX 2070), 4-layer ≈ 2.1 h.
About half of attempts stall at stage 1, so budget ~1.7 attempts per run.

---

## Phase 0: optional, stage-1 stalls

- [ ] **Stall pilot**: 4 configs × 3 seeds, stopped at step 4 000 (≈ 10 min each):
      baseline, `--warmup 1000`, `--lr 5e-4`, `--lr 5e-4 --warmup 1000`. Record the stall rate per config.
  - Trade-off: changing the recipe breaks comparability with seeds 0–5. Use a fix only for runs that
    answer a new question (Phase 3–4), not for replications of existing cells.

## Phase 1: headline claim (step embedding carries the stop). Highest priority.

- [ ] **nostep_s1**: `--no_step_emb --seed 1`
- [ ] **nostep_s2**: `--no_step_emb --seed 2`
- [ ] **nostep_s3**: `--no_step_emb --seed 3`
  - P18 replication. nostep_s0 extrapolates only just (distance 7 at K′ = 12: 0.63, CI 0.61–0.65).
  - Prediction: each competent run reaches distance 7 ≥ 0.60 (lower bound > 0.55) at K′ = 12.
- [ ] **budget_k8_s1**: `--K 8 --seed 1`
- [ ] **budget_k8_s2**: `--K 8 --seed 2`
  - Fill in the "finishes in budget, step embedding present" cell (only budget_k8_s0 so far: E = 4, no extrapolation).
  - Prediction: sweep reaches depth ≥ 4 by step 6 and no extrapolation at K′ = 16.
- [ ] **nostep_k8_s0**: `--K 8 --no_step_emb --seed 0`
- [ ] **nostep_k8_s1**: `--K 8 --no_step_emb --seed 1`
- [ ] **nostep_k8_s2**: `--K 8 --no_step_emb --seed 2`
  - **The decisive cell.** The search finishes inside the budget, but there is no step index to learn a halt from.
  - The halt account predicts extrapolation. A "finishing early is what matters" account (Zhang et al.) predicts none.
- [ ] **randk_s0 / s1 / s2**: K sampled per batch from {4..8}. **Needs code:** a `--K_random lo:hi` flag in `train.py`.
  - Kuo-style stochastic stopping. Prediction: no step-keyed halt, so it extrapolates even with a step embedding.

## Phase 2: training budget vs extrapolation (the curve)

- [ ] **budget_k4_s3**: `--K 4 --seed 3` (third K = 4 cell; the seed-1 run failed the gate)
- [ ] **budget_k10_s1**: `--K 10 --seed 1`
- [ ] **budget_k10_s2**: `--K 10 --seed 2`
- [ ] **budget_k10_s3**: `--K 10 --seed 3`
  - budget_k10_s0 failed the gate. If K = 10 trains, add **nostep_k10_s0..s2** as a stronger version of the Phase 1 2×2.
  - Output: the fraction of competent seeds that extrapolate at K = 4 / 6 / 8 / 10, plotted against frontier depth E.

## Phase 3: mechanism (fixed attention budget, W^−β, competition)

- [x] **Check first (no training):** do the existing graphs already vary branch shape (star vs chain, internal forks)?
      **Done 2026-09-25:** yes; and on fork-mode graphs (frontier W vs 2W at fixed W) per-edge attention in s0–s3
      halves (fork/chain 0.49–0.68, depths 1–3). Amplitude follows frontier edge count. See RESEARCH_LOG.
      If they do, test "amplitude follows frontier edge count, not W" on the existing models by conditioning on
      branch shape, and skip the next item.
- [ ] ~~**fork_s0 / s1 / s2**~~ (not needed for the amplitude claim after the check above; train only for a new question): trained on graphs whose branches fork internally, so frontier size ≠ W.
      **Code done (2026-09-25):** `--shapes fork [--p_fork 0.5]`; each graph is all chains (frontier = W) or all
      two-armed forks split at depth 0/1/2 (frontier = 2W past the split); family split skipped in this mode.
      Check-first result: default graphs already vary frontier at fixed W (R² with W 0.82–0.88 at distance 2–4,
      SD at fixed W 1.1–1.5); fork graphs cut R² to 0.62–0.67 and raise the SD to 1.6–2.1. Keep 62 nodes / 60 edges fixed.
- [ ] **v3_rawfb_s1**: `--feedback pre_ln --seed 1` (P13 replication: β toward 1, competition persists)
- [ ] **v2_deep_s1**: `--layers 4 --seed 1` (second run of the most competent variant; P12 direction and the clock-freeze rescue)
- [ ] **v1_wide_s2**: `--d 512 --seed 2` (addendum H is unresolved; v1_wide_s1 failed four attempts)

## Phase 4: transfer beyond the toy model

- [ ] **coconut_gpt2_prosqa_s0 / s1 / s2**: Coconut GPT-2 on ProsQA, the setting Aswal et al. and Rizvi-Martel et al. use.
      **Needs code** (separate pipeline) and a check on 8 GB memory.
  - Tests: step-specific attention blocking; freezing position ids as the clock (Coconut has no step embedding).

---

## Rough compute

Phases 1–3 come to ~20 runs × ~1.5 h × ~1.7 attempts ≈ 50 GPU-hours, about a day of wall time on the
two remaining cards (RTX 4060, RTX 2070). Phase 4 is not included.

## Not training (evaluation only, listed so they are not forgotten)

- Fresh-graph, pre-registered re-run of delayed-format blocking and width traces on every competent model.
- Competition mediation: clamp the other branches' layer-1 attention after an edit.
- Aswal-style observational frontier read-out on pause_s0..s2, next to the causal blocking.
- Re-score all earlier models with the addendum-J gate.
