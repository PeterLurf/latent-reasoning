"""§244 ANSWER-ONLY queues (AO_v1 §3, §7): scratch/gen_q_b366..b368.txt = ao_s1 / ao_s2 / aux_s2, one RTX 3090 each; per queue the K7 mem line, the G-TIME line and
the run (aw_plaw2_s1's flags from gen_q_b347; AO: --aux_fade 0,0; --keep_final_ckpt; --args_ref boxes/runs_fsq/aw_plaw2_s1/args.json). b366 first runs the host replay
of aw_plaw2_s1's first 100 steps (aw_plaw2_s1_rep_gtime: its exact flags + --time_steps 100; the logged step-100 loss 3.3319097250699996), read before b367 / b368.
    python -B scratch/s244_mkq.py"""
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s244 as G  # noqa: E402

REF = "boxes/runs_fsq/aw_plaw2_s1/args.json"
for n, run, fl in ((366, "ao_s1", G.ao(G.P2)), (367, "ao_s2", G.ao(G.P2) + ["--seed", "2"]), (368, "aux_s2", G.P2 + ["--seed", "2"])):
    b = " ".join(fl)
    txt = ("# THEORY §244 ANSWER-ONLY (3c91cb55; AO_v1 §3, §7; Note 244-1; Audit approved): %s = the §234 P-2 recipe (aw_plaw2_s1's flags)%s, 12k, one non-580 RTX 3090;"
           " admit / guard per host (guard $3.5 over the design).\n# Evals on host: %s; ckpt_final.pt kept (STILL-RISING resume). Frozen args scratch/s244_args_frozen_%s.json.\n"
           "# HOSTPROBE\n# P=1\n") % (run, {"ao_s1": " + --aux_fade 0,0", "ao_s2": " + --aux_fade 0,0 + seed 2", "aux_s2": " + seed 2"}[run], G.P2_EVAL, run)
    if n == 366:
        txt += "boxes/runs_fsq/aw_plaw2_s1_rep_gtime boxes.inject_aw %s --time_steps 100 --auto_micro\n" % " ".join(G.P2)
    txt += ("boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
            "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60 --auto_micro\n"
            "boxes/runs_fsq/%s boxes.inject_aw %s --auto_micro --ckpt_every 1000 --eval_parts %s --keep_final_ckpt --args_ref %s\n") % (run, b, run, b, run, b, G.P2_EVAL, REF)
    open(ROOT + "/scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(txt); print("written b%d %s" % (n, run))
