"""§253 LOOKUP-REBUILD Stage-1 queues (LB_v1 §2): scratch/gen_q_b382..b384 = p2p_s1 / p2p_s2 / p2p_old_s1 (scratch/s253_build.py ARMS / flags / tail; one RTX 3090 each,
12k, --keep_final_ckpt, eval_parts + test_perm; micro FIXED 1 + eval_bs 250 = aw_plaw2_s1's realised, NO --auto_micro anywhere incl. the G-TIME line (Audit 4)).
Frozen args = scratch/s253_args_frozen_<run>.json (s253_build.py args). K7 mem line, G-TIME line, the run (--args_ref the frozen args).
    python -B scratch/s253_mkq.py"""
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import s253_build as B  # noqa: E402

for n, run in ((382, "p2p_s1"), (383, "p2p_s2"), (384, "p2p_old_s1")):
    assert os.path.exists("scratch/s253_args_frozen_%s.json" % run), run
    b = " ".join(B.flags(run)); f = " ".join(B.tail(run)); init, seed = B.ARMS[run]
    txt = ("# THEORY §253 LOOKUP-REBUILD Stage 1 (LB_v1 §2; Audit guard $5): %s = aw_plaw2_s1's P-2 recipe (gen_q_b347) with init_trunk %s + --perm_p 0.5, seed %d; one non-580 "
           "RTX 3090; micro fixed 1 / eval_bs 250 (no --auto_micro).\n# Evals on host: %s. Frozen args scratch/s253_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
           "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
           "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60\n"
           "boxes/runs_fsq/%s boxes.inject_aw %s %s\n") % (run, init, seed, B.EVP, run, run, b, run, b, run, b, f)
    open("scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(txt); print("written b%d %s" % (n, run))
