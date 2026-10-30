"""§250 FRAME-DIV Stage 1 queues (FD_v1 §4 + Note 250-1): scratch/gen_q_b372..b375.txt = fd_div_s1 / fd_div_s2 / fd_ctrl_s1 / fd_ctrl_s2, one RTX 3090 each; per queue the
K7 mem line, the G-TIME line (--time_steps 60 --auto_micro) and the run (gate_s250s1.full(): §240 FH recipe from lh_fh_s<n>, --hold_start 0, DIV --train_frames 0-5,8-11).
    python -B scratch/s250_mkq.py"""
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s250s1 as G  # noqa: E402

for n, (run, s, div) in zip((372, 373, 374, 375), G.RUNSPEC):
    b = " ".join(G.base(s, div)); f = " ".join(G.full(s, div, run)[len(G.base(s, div)):])
    txt = ("# THEORY §250 FRAME-DIV Stage 1 (FD_v1 §4; Note 250-1; Audit G-D): %s = 8k fine-tune from lh_fh_s%d, §240 recipe (hold_inc 1.6, hold_start 0)%s, one non-580 RTX 3090;"
           " admit / guard per host (guard $5 over the design).\n# Evals on host: %s; eval_long_k128 + REG refs (DIV) local after landing. Frozen args scratch/s250_args_frozen_%s.json.\n"
           "# HOSTPROBE\n# P=1\n"
           "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
           "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60 --auto_micro\n"
           "boxes/runs_fsq/%s boxes.inject_aw %s %s\n") % (run, s, ", --train_frames " + G.DIVF if div else ", default frames 0-5", G.EVAL, run, run, b, run, b, run, b, f)
    open(ROOT + "/scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(txt); print("written b%d %s" % (n, run))
