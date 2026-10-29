"""§243 DEPTH-CURRIC queues (DC_v1 §2, §7): scratch/gen_q_b362..b365.txt = dc_s1 / dc_s2 / dc_ctrl_s1 / dc_ctrl_s2, one RTX 3090 each; per queue the K7 mem line,
the G-TIME line (--hold_start 1 --time_steps 60 --auto_micro) and the run (the flags of scratch/gate_s243.py full(), frozen args scratch/s243_args_frozen_<run>.json).
    python -B scratch/s243_mkq.py"""
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s243 as G  # noqa: E402

for n, run, s, dm in ((362, "dc_s1", 1, 6), (363, "dc_s2", 2, 6), (364, "dc_ctrl_s1", 1, 4), (365, "dc_ctrl_s2", 2, 4)):
    b = " ".join(G.base(s, dm)); f = " ".join(G.full(s, dm, run)[len(G.base(s, dm)):])
    txt = ("# THEORY §243 DEPTH-CURRIC (539b4b3a; DC_v1 §2-3, §7; Audit approved): %s = 8k fine-tune from lh_fh_s%d, d ~ U{1..%d}, think_k 10, --hold_inc 1.6 (hold_start 500),"
           " one non-580 RTX 3090; admit / guard per host (guard $8 over the design).\n# Evals on host: %s; eval_long_dc (run_long) + REG refs local after landing."
           " Frozen args scratch/s243_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
           "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
           "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --hold_start 1 --time_steps 60 --auto_micro\n"
           "boxes/runs_fsq/%s boxes.inject_aw %s %s\n") % (run, s, dm, G.DC_EVAL, run, run, b, run, b, run, b, f)
    open(ROOT + "/scratch/gen_q_b%d.txt" % n, "w", newline="\n").write(txt); print("written b%d %s" % (n, run))
