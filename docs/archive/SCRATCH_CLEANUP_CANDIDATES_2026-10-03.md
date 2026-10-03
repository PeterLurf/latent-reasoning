# scratch/ cleanup candidates (2026-10-03, for the user to decide; nothing deleted)

`scratch/` has 701 entries. **311** below are named nowhere outside `scratch/` (docs, PROTOCOL/THEORY, code) and are not
used by any script a live job runs. They are one-off deploy/drain scripts, queue files and `.claimed` markers of finished
protocols, doc-patch scripts, and display helpers. Kept regardless: the live set (`DO_NOT_MOVE_WHILE_RUNNING.md`),
`score_*`, `p7s_*.py` and `far_sweep*` (pre-registered scorers), and `gen_q_*` (queues of current/recent fleets).
259 of them are in git, so deleting them is reversible (`git checkout <commit> -- scratch/<file>`). **52 are untracked** (gitignored queue-claim markers and similar, marked † below); deleting those is not reversible, though they hold no results.
Suggested: `git rm` the groups below once the fleets in `docs/CURRENT_TASKS.md` are done.

* **p6** (55): `p6_after.sh`, `p6_bext2_2070.txt.claimed`†, `p6_bext_queue.txt`, `p6_bext_queue.txt.claimed`†, `p6_branch_queue.txt`, `p6_branch_queue.txt.claimed`†, `p6_branch_queue_2070.txt`, `p6_branch_queue_2070.txt.claimed`†, `p6_k8_2070.txt`, `p6_k8_2070.txt.claimed`†, `p6_k8_3070.txt`, `p6_k8_3070.txt.claimed`†, `p6_k8_4060.txt`, `p6_k8_4060.txt.claimed`†, `p6_m_2070.txt`, `p6_m_2070.txt.claimed`†, `p6_m_3070.txt`, `p6_m_3070.txt.claimed`†, `p6_m_4060.txt`, `p6_m_4060.txt.claimed`†, `p6_o_3070.txt`, `p6_o_3070.txt.claimed`†, `p6_o_4060.txt`, `p6_o_4060.txt.claimed`†, `p6_p29_3070.txt`, `p6_p29_3070.txt.claimed`†, `p6_p29b_3070.txt.claimed`†, `p6_p30_2070.txt.claimed`†, `p6_p30_3070.txt.claimed`†, `p6_p30_4060.txt`, `p6_p30_4060.txt.claimed`†, `p6_p30_hostA.txt`, `p6_p30_hostB.txt`, `p6_p30_vast1.txt`, `p6_p30_vast2.txt`, `p6_p30_vast3.txt`, `p6_prec_2070.txt`, `p6_prec_2070.txt.claimed`†, `p6_prec_3070.txt`, `p6_prec_3070.txt.claimed`†, `p6_prec_3070b.txt.claimed`†, `p6_prec_4060.txt`, `p6_prec_4060.txt.claimed`†, `p6_queue.txt.claimed`†, `p6_queue_2070.txt.claimed`†, `p6_r_3070.txt.claimed`†, `p6_r_4060.txt.claimed`†, `p6_sharp_2070.txt`, `p6_sharp_2070.txt.claimed`†, `p6_stall_3070.txt`, `p6_stall_3070.txt.claimed`†, `p6_stall_4060.txt`, `p6_stall_4060.txt.claimed`†, `p6_wait.sh`, `p6_wait2.sh`
* **c1** (29): `c1_671.sh`, `c1_672.sh`, `c1_67b_deploy.sh`, `c1_72_deploy.sh`, `c1_72b_deploy.sh`, `c1_73_deploy.sh`, `c1_75_deploy.sh`, `c1_88_chain.sh`, `c1_88fix_chain.sh`, `c1_89_chain.sh`, `c1_after88_chain.sh`, `c1_local_worker.sh`, `c1_lr2x2.sh`, `c1_next_chain.sh`, `c1_q_72b_local.txt`, `c1_q_73_local.txt`, `c1_q_74_local.txt`, `c1_q_74diag.txt`, `c1_q_75_2031.txt`, `c1_q_75_2032.txt`, `c1_q_75_2033.txt`, `c1_q_75_2034.txt`, `c1_q_75_2035.txt`, `c1_q_75_local.txt`, `c1_q_79_pilot.txt`, `c1_q_88.txt`, `c1_q_88fix.txt`, `c1_q_89.txt`, `c1_q_after88.txt`
* **patch** (24): `patch_abs2.py`, `patch_addI.py`, `patch_chips.py`, `patch_clock.py`, `patch_clock_abs.py`, `patch_clockfig.py`, `patch_competence.py`, `patch_ctrace.py`, `patch_dev.py`, `patch_eval_chunks.py`, `patch_final_text.py`, `patch_h.py`, `patch_layout.py`, `patch_log.py`, `patch_nostep.py`, `patch_p14.py`, `patch_p18.py`, `patch_pending.py`, `patch_render2.py`, `patch_rowsJK.py`, `patch_s5.py`, `patch_scoreJK.py`, `patch_stall.py`, `patch_twolimits.py`
* **bench** (17): `bench_concurrency.sh`, `bench_deploy.sh`, `bench_deploy_wide.sh`, `bench_q1.txt`, `bench_q2_0.files`, `bench_q2_1.files`, `bench_q2_2.files`, `bench_q2_3.files`, `bench_q2_4.files`, `bench_q2_5.files`, `bench_q2_6.files`, `bench_q2_7.files`, `bench_q2_8.files`, `bench_q3_local.txt`, `bench_q4_local.txt`, `bench_q5_local.txt`, `bench_worker.sh`
* **show** (17): `show_amp.py`, `show_attn.py`, `show_budget_wd.py`, `show_cache.py`, `show_cachedepth.py`, `show_cachedepth2.py`, `show_confirm.py`, `show_confirm2.py`, `show_dev.py`, `show_dev_w.py`, `show_followup.py`, `show_freeze.py`, `show_pair_gain.py`, `show_rewind_grid.py`, `show_summary.py`, `show_train.py`, `show_upcache.py`
* **p8na** (12): `p8na_q_at_s1.txt`, `p8na_q_at_s2.txt`, `p8na_q_at_s3.txt`, `p8na_q_at_s4.txt`, `p8na_q_coco_s2.txt`, `p8na_q_coco_s3.txt`, `p8na_q_coco_s4.txt`, `p8na_q_cot_s4.txt`, `p8na_q_rt_s1.txt`, `p8na_q_rt_s2.txt`, `p8na_q_rt_s3.txt`, `p8na_q_rt_s4.txt`
* **auto** (10): `auto_drain_bench.sh`, `auto_drain_bench_slow.sh`, `auto_drain_boxes.sh`, `auto_drain_c1.sh`, `auto_drain_f1sb.sh`, `auto_drain_f4.sh`, `auto_drain_gen2_test.sh`, `auto_drain_na.sh`, `auto_drain_s71.sh`, `auto_drain_tj.sh`
* **st65** (8): `st65_deploy.sh`, `st65_q_0.files`, `st65_q_1.files`, `st65_q_2.files`, `st65_q_3.files`, `st65_q_4.files`, `st65_q_5.files`, `st65_q_6.files`
* **** (6): `.hold76_cuda0`†, `.hold76_cuda1`†, `.hold76_cuda2`†, `.w76_cuda0`†, `.w76_cuda1`†, `.w76_cuda2`†
* **gen** (6): `gen_deploy.sh`, `gen_deploy_93.sh`, `gen_deploy_95.sh`, `gen_deploy_f89.sh`, `gen_memtest.sh`, `gen_proof_host.txt`†
* **run** (5): `run_f6_measures.sh`, `run_s62_measures.sh`, `run_status.py`, `run_t53.sh`, `run_t55_t56.sh`
* **s83** (5): `s83_after.sh`, `s83_after2.sh`, `s83_fetch.sh`, `s83_fetch2.sh`, `s83_host.txt`†
* **drain** (4): `drain_bench.sh`, `drain_boxes.sh`, `drain_f1sb.sh`, `drain_s71.sh`
* **p7s** (4): `p7s_t10_2070.txt.claimed`†, `p7s_t9_3070.txt`, `p7s_t9_3070.txt.claimed`†, `p7s_wait.sh`
* **t9** (4): `t9_causal_measure.sh`, `t9_causal_runs.txt`, `t9_t4_prefill.sh`, `t9_watch.sh`
* **backfill** (3): `backfill_q1.txt`, `backfill_q1_paused.txt`, `backfill_q1_paused2.txt`
* **f6** (3): `f6_q_local.txt`, `f6_q_local.txt.claimed`†, `f6_queue.sh`
* **g78** (3): `g78_capguard.sh`, `g78_deploy.sh`, `g78_smoke_host.txt`†
* **p8b** (3): `p8b_hopsweep_ctl2.sh`, `p8b_hopsweep_deep.sh`, `p8b_hopsweep_shallow.sh`
* **pod** (3): `pod_run.sh`, `pod_status.sh`, `pod_watch.sh`
* **t12** (3): `t12_v1.txt`, `t12_v2.txt`, `t12_v3.txt`
* **tj** (3): `tj_deploy.sh`, `tj_deploy_wide.sh`, `tj_local.sh`
* **bm** (2): `bm_panel.sh`, `bm_pilot.sh`
* **boxes** (2): `boxes_deploy.sh`, `boxes_deploy_wide.sh`
* **check** (2): `check_chain.py`, `check_flex_upfront.py`
* **explore** (2): `explore_pattern.py`, `explore_subspace.py`
* **f4** (2): `f4_deploy_all.sh`, `f4_deploy_some.sh`
* **far** (2): `far_prefill.sh`†, `far_prefill_runs.txt`†
* **gsm** (2): `gsm_ablate.sh`, `gsm_budget.sh`
* **local** (2): `local_after.sh`, `local_progress.sh`
* **p9x** (2): `p9x_bf16.txt.claimed`†, `p9x_fork.txt.claimed`†
* **prio** (2): `prio_fz_g_s950.sh`, `prio_fz_gat_s951.sh`
* **s62** (2): `s62_queue.txt`, `s62_worker.sh`
* **s71** (2): `s71_deploy.sh`, `s71_dpvar_rerun.sh`
* **smoke** (2): `smoke.py`, `smoke_followup.py`
* **RESEARCH** (1): `RESEARCH_LOG.backup.md`
* **after** (1): `after_p11.sh`
* **attn** (1): `attn_share.py`
* **compare** (1): `compare_curves.py`
* **dbg86** (1): `dbg86.py`
* **dbg86b** (1): `dbg86b.py`
* **dbg86c** (1): `dbg86c.py`
* **dbg86d** (1): `dbg86d.py`
* **donor** (1): `donor_check.py`
* **dp** (1): `dp_attempt1.sh`
* **dpdiag** (1): `dpdiag_s3.sh`
* **f1sb** (1): `f1sb_q_L.txt`
* **f89** (1): `f89_proof_host.txt`†
* **fast** (1): `fast_drain_s71.sh`
* **fix** (1): `fix_times.py`
* **followup** (1): `followup_confirm.before.json`
* **ftcheck** (1): `ftcheck_cpu.json`†
* **g0** (1): `g0_score.txt`
* **g1** (1): `g1_score.txt`
* **g2** (1): `g2_score.txt`
* **gate** (1): `gate_check.py`
* **gpu** (1): `gpu_stress.py`
* **gru** (1): `gru_hardstop.sh`
* **integrity** (1): `integrity_compare.py`
* **j0** (1): `j0_score.txt`
* **j1** (1): `j1_score.txt`
* **l86** (1): `l86_pilot.txt`
* **mem89** (1): `mem89.py`
* **moji** (1): `moji_preview.txt`
* **n0** (1): `n0_score.txt`
* **n1** (1): `n1_score.txt`
* **na** (1): `na_deploy_chain.sh`
* **p10** (1): `p10_worker.sh`
* **p10s** (1): `p10s_worker.sh`
* **p11** (1): `p11_worker.sh`
* **p8b1** (1): `p8b1_base_ref.sh`
* **p93** (1): `p93_proof_host.txt`†
* **pause** (1): `pause_pid.txt`
* **pipeline** (1): `pipeline.py`
* **print** (1): `print_dom.html`
* **proto** (1): `proto_smoke.json`
* **rel** (1): `rel_queue.txt.claimed`†
* **rerun** (1): `rerun_length_gen.sh`†
* **rewind** (1): `rewind_bywidth.py`
* **s65** (1): `s65_queue.txt`
* **s66** (1): `s66_kp_local.sh`
* **s82** (1): `s82_run.sh`
* **sm78** (1): `sm78_pred.json`†
* **sm94** (1): `sm94.json`†
* **speeds** (1): `speeds.py`
* **st** (1): `st_local.sh`
* **st2** (1): `st2_local.sh`
* **status** (1): `status.py`
* **test** (1): `test_krandom.py`
* **upload** (1): `upload_base_ckpt.sh`
* **variant** (1): `variant_pipeline.py`
* **vram** (1): `vram_test.py`
* **w94** (1): `w94_smoke_host.txt`†
