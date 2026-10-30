"""§251 ANSWER-EXTEND queues (AE_v1 §2): scratch/gen_q_b376..b379.txt = aex_s1 / aex_s2 (rcb_plaw4_s<n> flags + --aux_d_max 2) and aex0_s1 / aex0_s2 (rcb_plaw4_s<n> flags
with --aux_fade 0,0), one RTX 3090 each, 16k; the same eval parts as rcb_plaw4 (gen_q_b354 / b355). K7 mem line, G-TIME line, the run (--args_ref the §251 frozen args).
Frozen args: scratch/s251_args_frozen_<run>.json from --args_only (CPU). AE_v1 §2: args diff vs rcb_plaw4_s<n> = {aux_d_max} (AEX) / {aux_fade} (AEX0).
    python -B scratch/s251_mkq.py"""
import json
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G  # noqa: E402
import s228_args_check as C  # noqa: E402

EVAL = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd"; diffs = {}
for n, run, s, arm in ((376, "aex_s1", 1, "aex"), (377, "aex_s2", 2, "aex"), (378, "aex0_s1", 1, "aex0"), (379, "aex0_s2", 2, "aex0")):
    fl = list(G.RCB) + (["--seed", "2"] if s == 2 else [])
    if arm == "aex":
        fl += ["--aux_d_max", "2"]
    else:
        fl[fl.index("--aux_fade") + 1] = "0,0"
    b = " ".join(fl); f = "--auto_micro --ckpt_every 1000 --eval_parts %s --args_ref scratch/s251_args_frozen_%s.json" % (EVAL, run)
    d = G8.fresh(G.TMP + "/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", EVAL, "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
    assert rc == 0, run; new = json.load(open(d + "/args.json")); ref = json.load(open("scratch/s237_args_frozen_rcb_plaw4_s%d.json" % s))
    dif = C.diff(ref, new); diffs[run] = {k: v for k, v in dif.items() if k != "pair_stats"}
    json.dump(new, open("scratch/s251_args_frozen_%s.json" % run, "w", newline="\n"), indent=1)
    txt = ("# THEORY §251 ANSWER-EXTEND (8516f651; AE_v1 §2; Audit G-D): %s = rcb_plaw4_s%d's §237 recipe (16k, REG, init aw_plaw2_s1) + %s, one non-580 RTX 3090; admit / guard per"
           " host (guard $4.5 over the design).\n# Evals on host: %s (as rcb_plaw4); eval_long local after landing. Frozen args scratch/s251_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
           "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
           "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60 --auto_micro\n"
           "boxes/runs_fsq/%s boxes.inject_aw %s %s\n") % (run, s, "--aux_d_max 2" if arm == "aex" else "--aux_fade 0,0", EVAL, run, run, b, run, b, run, b, f)
    open("scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(txt); print("written b%d %s" % (n, run))
os.makedirs("scratch/gate_s251", exist_ok=True); json.dump(diffs, open("scratch/gate_s251/args.json", "w", newline="\n"), indent=1, default=str)
print(json.dumps({k: sorted(v) for k, v in diffs.items()}))
