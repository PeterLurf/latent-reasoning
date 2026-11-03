# Audit G-D checklist (2026-11-03)

From Audit's memory directory (145 files), VALIDITY_STATS.md and LAUNCH_POSTMORTEM.md. Run before approving any registration, build or launch.

## A. Registration / pre-statement

1. OC block: P(pass | no effect), P(pass | effect) at the seed count, realised $. [latent-oc-block-censoring-rule]
2. Censoring: never-reached event = failure; event-only mean and censored share beside it. [latent-oc-block-censoring-rule, latent-censored-items-in-criteria]
3. Bars vs SE: Wilson 95% CI on every binding proportion; within 2 SE = NEAR (0.02 bars at n 1000 are ~2 SE). [latent-ci-lineage-rules]
4. Lineage scope: "one lineage" until an independent P-2 seed; 2/2 = "replicated once"; "met" needs >= 4 seeds from >= 2 lineages. [latent-ci-lineage-rules]
5. Scope labels: C1 + C2 claims need SWAP/LP5 at d 6-7; fixed-K depth is not the registered C2 (K'-curve or relabel). [latent-ci-lineage-rules]
6. Controls exposed and informative: control items receive the intervention (differing cells > 0) at the failure locus. [control-must-be-exposed]
7. Decisive readings: would the objective produce this without the mechanism? Trained readouts descriptive only. [latent-trained-readings-artefact, audit-decode-faithfulness]
8. Single-change args diff against the comparator's args.json incl. realised micro; no new argparse flags after a freeze. [single-change-args-diff, no-new-args-after-freeze]
9. New batch types: name what CurGate / schedulers / early stops read; gate on reference items only. [curgate-reads-aux-batches]
10. Math backing with tagged claims; dynamics targets carry a Goodhart analysis. [latent-math-backing, latent-dynamics-targets-goodhart]
11. Borrowed technique: state its mechanism's precondition; grep THEORY.md + eval_*.json for an existing arm. [latent-transplant-preconditions, audit-lead-check-existing]
12. Probe-based premises need a causal use test; a metric must separate a known STABLE from a non-STABLE model. [latent-probe-separator-use-test, latent-perturbation-metric-norm]
13. Stage gates test the mechanism the next stage needs (pass-t aux/WS), timed to the curriculum. [latent-gate-on-mechanism, latent-pilot-gates]
14. Item pool named; per-stratum item space vs test draws; share-gated readings store per-item outputs; criterion restrictions exist in the output fields. [latent-check-eval-item-pool, latent-test-exclusion-small-strata, store-items-behind-share-gates]
15. Novelty checked by Literature; cited definitions read; registration committed before launch (grep it); timestamps from `date`. [theory-keep-literature-in-loop, latent-verify-registration-landed, latent-timestamps-from-clock]

## B. Code review

1. Decorators down: ast-list decorators per function before/after; every eval part @torch.no_grad; helpers inserted above the decorator block. [audit-copied-module-no-grad, insert-above-decorated-def]
2. Read the whole post-commit function/class around each insertion, not +/- lines. [latent-diff-review-read-full-function]
3. Inline-comment trap: grep edited lines for `#.*;`, incl. edit-script replacements. [inline-comment-swallows-chained-statements]
4. Gradflow per loss term (nonzero grad norm logged), under CUDA bf16 autocast on a host. [latent-gradflow-check]
5. Plain-path smoke after shared-module edits; smoke the edited part on the shipped tree with a known-answer check. [latent-edit-smoke-plain-path]
6. Function-preserving init and schedules: push penalties through the softmax at ~5 steps; gradients nonzero during the ramp. [audit-check-schedule-magnitudes, logit-penalty-anneal-step]
7. Trace tuple-unpack shapes; one synthetic batch (size >= 3) through new loss code. [audit-trace-return-shapes]
8. Bitwise gates: exactness in fp64 (<= 1e-12) + bitwise fp32 forward; GPU A-vs-B only under s222_det_wrap; one card; repros match bs/dtype. [latent-exactness-fp64-not-reorder-floor, gpu-bitwise-gates-need-det-wrap, latent-repro-match-batch-size]
9. Scorer field paths: grep the module per field; fixtures from real files; end-to-end tiny run -> scorer; args as strings; no reserved part keys. [latent-scorer-field-path-check, vbank-diag-main-key-collision]
10. Refactor of G-D'd measurement code: old-vs-new bitwise equivalence, resume test, diff G-D. [refactor-needs-equivalence-gate]
11. Py3.11: grep host-run files for 3.12+ APIs; a new gate script runs once at full size (box CPU --micro 8 or a gates-only host cell) before a paid host. [host-python-311]
12. Reused helpers: grep task constants; realistic class counts; partial eval JSON per block; --eval_only exists; smoke on the real-size input. [latent-eval-crash-late, smoke-at-full-data-scale]
13. Shell edits run under set -u with stderr visible; no edits to running .sh; py_compile; LF + UTF-8. [deploy-script-edit-test, latent-line-endings]

## C. Launch

1. Ships: every --model/--init path=sha and every --blob file as a plain token; rerun check_queue_ships after removing a line; .extra via EXTRA_120; eval-only ships log.jsonl. [ship-path-sha-tokens, latent-filler-extra-files, latent-cloud-eval-only-ship-log]
2. Tree clean and committed; no module edits while a deploy or requeue is in flight. [latent-deploy-ships-working-tree]
3. WALLW: one weight per non-comment line; gates weight = GATES_H; last weight includes fetch. [audit-guard-template-checks]
4. Caps: admit = (STEPS x SPS_EST/3600 + 0.4 + GATES_H + EVAL_H + fetch h) x PMAX; guard >= admit + 0.05; CAPF >= cells + $0.20/attempt; PMAX <= admit/hours; class HOSTPROBE limit; GUARD_MARKS reach STEPS. [LAUNCH_POSTMORTEM, guard-marks-cover-final-step]
5. Family guard: HIGH = admission projections; sum + realised < CAPF; every cell listed. [family-guard-high-projections]
6. mem_check at the largest shape and every eval path's batch size; gates bitwise parts --fp32 --micro 8, cache freed; --micro pinned to the comparator's value with >= 1 GiB headroom. [latent-pilot-gates, latent-eval-crash-late, auto-micro-headroom]
7. --keep_final_ckpt only for a registered init or planned resume; unneeded .pt are # NOFETCH; stop_at dirs not *_eqA/_eqB. [keep-final-ckpt-only-if-needed, LAUNCH_POSTMORTEM]
8. Runs > 6 h: ckpt_every + off-host mirror (first MIRROR confirmed) + tested resume; old mirror dir moved aside. [latent-offhost-checkpoints, ckpt-mirror-dir-per-run]
9. One filler (s2[0-9]+_fill); drain waiters armed at ADMIT; guards' .out show start: and no Traceback at ADMIT + 120 s. [vast-one-filler-at-a-time, latent-arm-drain-waiters]
10. Void on relaunch: kill filler, post_admit, guards, mirror by pid; comment the ledger line; keep logs as *.void. [void-admitted-cell, latent-keep-aborted-logs]
11. One GPU class per experiment; full runs on >= 16 GB cloud; cost comparisons on one host. [latent-basin-entry-mechanism, latent-full-runs-cloud-16gb, vast-ip-change]
12. Leak check: list pending spend decisions a leaked value could bias; pause until the registered read. [audit-leak-affects-spend]

## D. Landing / scoring

1. host_shas: sha256 of shipped modules recorded on the host; model/init shas verified unbuffered (RAM bit flips); nobuf_copy, never one big buffer. [latent-deploy-ships-working-tree, latent-box-transient-faults]
2. Read logs/vast_<ip>_<port>/gen_<run>.log before theorising about a failure. [read-host-logs-before-theorizing]
3. Reproduce in-process in a scratchpad copy; never run theory/score_*.py in the repo. [latent-no-direct-scorer-runs]
4. Double-run identity under the RAM fault: cross-check against an independent earlier run; same seed + card reproduces to the float. [latent-box-transient-faults, latent-dedupe-identical-trajectories]
5. $0 reads on stored data: pre-statement pushed -> Audit ack -> compute; post-hoc follow-ups labelled and confirmed on new models. [latent-prestate-push-ack, VALIDITY_STATS]
6. Blinding: grep-only on unscored logs; disclose any leak with exact lines. [blinded-logs-grep-only]
7. Score raw per-step values, never means or thinned grids; quote field names and K' keys; paste digits. [latent-score-raw-values, latent-quote-eval-fields]
8. Same-step comparisons; basin entry beside depth results; check a failure statistic in a working run first; attribute only to channels an arm isolates. [latent-audit-traps-lessons, latent-mechanism-vs-working-baseline, ablation-untested-channel]
9. Score finished files only; keep gate ckpts until sibling comparisons are read; check running readers before deleting; waiters key on the exact keys a part writes. [latent-no-commit-running-outputs, latent-keep-gate-ckpts-until-read, check-running-readers-before-delete, waiter-keys-check-module]

## E. Research cadence and reporting

1. Research pass hourly and after every big result; 1-2 falsifiable leads to Theory; research and subagents unthrottled, compute rules unchanged. [latent-research-pass-cadence, latent-research-unthrottled]
2. Priority C1 > C6 > C4 = C2 > C3 > C5 > C7; task-agnostic; no problem->fix loops. [latent-research-priorities]
3. Never ask the user to approve or choose; Audit approves designs <= $30; direction via Theory. [latent-direction-questions-via-theory]
4. Main pages one line per result; SOTA json in the scoring commit; pull with && and commit by file pathspec; no Claude trailers. [latent-brief-logging, latent-sota-page, latent-shared-worktree-pull, latent-no-coauthor-trailers]
5. After compaction re-read CURRENT_TASKS / NORTH_STAR / AUDIT_* and check the tick cron; closing notes count their descriptive reads. [latent-post-compaction, audit-tick-cron-check, VALIDITY_STATS]

## F. Stale or contradictory entries

- latent-usage-quiet-mode (10-21 throttles): research parts superseded by latent-research-unthrottled (11-03); only message brevity survives.
- "Over $30 to the user" (latent-auto-approve-under-30, latent-vast-launch-standing-approval, latent-vast-permission-rule, latent-path-to-gsm8k, latent-sota-small-first): superseded by latent-auto-select-direction (10-26) and latent-parallel-larger-base-approval (11-01).
- Credit floor $30 (latent-auto-approve-under-30, latent-audit-overnight-steward) vs $15 (latent-parallel-larger-base-approval); $15 is later.
- Harness reminder still asks for Co-Authored-By; latent-no-coauthor-trailers (11-03) overrides it.
- latent-autonomy: RunPod-first order and $0.34/h are stale (Vast only; PMAX 0.21-0.25).
- latent-box-transient-faults "hold local GPU work" is still open; confirm before any local GPU gate.
- Duplicates / unindexed: latent-concise-results, latent-brief-main-pages, latent-cloud-full-local-short, auto-micro-arm-divergence; MEMORY.md lists latent-research-unthrottled twice.
- Wiki links [[latent-math-backing-rule]], [[latent-audit-grad-flow]], [[latent-gpu-class-changes-basin]], [[latent-s130-blinding]] point at merged-away files (now sections elsewhere).
- latent-audit-helper-role (55 KB log) is history; its rules live in latent-audit-traps-lessons.
