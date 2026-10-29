"""§245 AO-REP queues (Note 245-1): scratch/gen_q_b370.txt / b371.txt = ao_s3 / ao_s4 = ao_s1's flags (§244 AO: --aux_fade 0,0) with --steps 18000 (a straight 18k
cosine) and seeds 3 / 4; one RTX 3090 each; K7 mem line, G-TIME line, the run (--keep_final_ckpt, --args_ref boxes/runs_fsq/aw_plaw2_s1/args.json).
    python -B scratch/s245_mkq.py"""
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s244 as G4  # noqa: E402
import gate_s245 as G  # noqa: E402

REF = "boxes/runs_fsq/aw_plaw2_s1/args.json"
for n, run, seed in ((370, "ao_s3", 3), (371, "ao_s4", 4)):
    b = " ".join(G.flags(seed))
    txt = ("# THEORY §245 AO-REP (Note 245-1; Audit approved): %s = the §244 AO recipe (aux_fade 0,0) straight to 18k, seed %d, one non-580 RTX 3090; admit / guard per host"
           " (guard $2.5 over the design).\n# Evals on host: %s. Frozen args scratch/s245_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
           "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
           "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60 --auto_micro\n"
           "boxes/runs_fsq/%s boxes.inject_aw %s --auto_micro --ckpt_every 1000 --eval_parts %s --keep_final_ckpt --args_ref %s\n") % (
        run, seed, G4.P2_EVAL, run, run, b, run, b, run, b, G4.P2_EVAL, REF)
    open(ROOT + "/scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(txt); print("written b%d %s" % (n, run))
