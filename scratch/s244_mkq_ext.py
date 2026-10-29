"""§244 STILL-RISING extension (AO_v1 §4; Note 244-3): scratch/gen_q_b369.txt = ao_s2_ext, the one pre-registered +6k: resume ao_s2 from its verified 12k ckpt
(boxes/runs_fsq/ao_s2_ext/ckpt_12k.pt = ao_s2/ckpt_final.pt, sha c8d52aaa) to 18k, same flags (--steps 18000; the cosine LR is recomputed over 18000 at resume:
0.25 x base at 12001 -> 0 at 18000), micro 1 (ao_s2's), the P-2 eval parts into ao_s2_ext (ao_s2's 12k evals kept). ao_s2_ext/args.json = a copy of ao_s2/args.json
(a resume writes args_resume_*.json only; the eval and the drain read args.json), shipped as --args_ref. K7 / G-TIME lines are fresh short runs (admission timing).
Frozen args for the post-admit check: scratch/s244_args_frozen_ao_s2_ext.json = that same copy (the resumed args are checked by hand from args_resume_*.json).
    python -B scratch/s244_mkq_ext.py"""
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s244 as G  # noqa: E402

base = G.ao(G.P2) + ["--seed", "2"]; ext = [x if x != "12000" else "18000" for x in base]
txt = ("# THEORY §244 STILL-RISING extension (Note 244-3): ao_s2_ext = ao_s2 resumed from its 12k ckpt to 18k (one non-580 RTX 3090; inside the $3.5 guard).\n"
       "# Evals on host: %s into ao_s2_ext. Frozen args scratch/s244_args_frozen_ao_s2_ext.json (= ao_s2/args.json; resumed args checked by hand).\n# HOSTPROBE\n# P=1\n"
       "boxes/runs_fsq/ao_s2_ext_mem boxes.inject_aw %s --mem_check\n"
       "boxes/runs_fsq/ao_s2_ext_gtime boxes.inject_aw %s --time_steps 60 --auto_micro\n"
       "boxes/runs_fsq/ao_s2_ext boxes.inject_aw %s --resume boxes/runs_fsq/ao_s2_ext/ckpt_12k.pt --micro 1 --ckpt_every 1000 --eval_parts %s --keep_final_ckpt "
       "--args_ref boxes/runs_fsq/ao_s2_ext/args.json\n") % (G.P2_EVAL, " ".join(base), " ".join(base), " ".join(ext), G.P2_EVAL)
open(ROOT + "/scratch/gen_q_b369.txt", "w", encoding="utf-8", newline="\n").write(txt)
shutil.copy(ROOT + "/boxes/runs_fsq/ao_s2/args.json", ROOT + "/scratch/s244_args_frozen_ao_s2_ext.json"); print("written b369 ao_s2_ext")
