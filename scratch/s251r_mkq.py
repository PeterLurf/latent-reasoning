"""§251 Note 251-2 VOID-RERUN queues: scratch/gen_q_b380 / b381 = aex_s1 / aex_s2 fresh, rcb_plaw4_s<n> flags + --aux_d_max 2 + --micro 2 (CTRL's realised micro; the
first AEX launch b376 / b377 kept micro 1 under --auto_micro and OOMed on host). Frozen args overwrite scratch/s251_args_frozen_aex_s<n>.json (the micro-1 copies stay in
git history); scratch/gate_s251/args_rerun.json = the diff vs CTRL's REALISED args (boxes/runs_fsq/rcb_plaw4_s<n>/args.json) and vs §237's frozen args.
    python -B scratch/s251r_mkq.py"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G  # noqa: E402
import s228_args_check as C  # noqa: E402

EVAL = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd"; diffs = {}
for n, run, s in ((380, "aex_s1", 1), (381, "aex_s2", 2)):
    fl = list(G.RCB) + (["--seed", "2"] if s == 2 else []) + ["--aux_d_max", "2", "--micro", "2"]
    b = " ".join(fl); f = "--auto_micro --ckpt_every 1000 --eval_parts %s --args_ref scratch/s251_args_frozen_%s.json" % (EVAL, run)
    d = G8.fresh(G.TMP + "/args_r_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + ["--auto_micro", "--ckpt_every", "1000", "--eval_parts", EVAL, "--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
    assert rc == 0, run; new = json.load(open(d + "/args.json"))
    real = json.load(open("boxes/runs_fsq/rcb_plaw4_s%d/args.json" % s)); ref = json.load(open("scratch/s237_args_frozen_rcb_plaw4_s%d.json" % s))
    diffs[run] = {"vs_ctrl_realised": {k: v for k, v in C.diff(real, new).items() if k != "pair_stats"},
                  "vs_s237_frozen": {k: v for k, v in C.diff(ref, new).items() if k != "pair_stats"}}
    json.dump(new, open("scratch/s251_args_frozen_%s.json" % run, "w", newline="\n"), indent=1)
    txt = ("# THEORY §251 Note 251-2 VOID-RERUN (12fc3d5c; Audit $5.5): %s = rcb_plaw4_s%d's §237 recipe (16k, REG, init aw_plaw2_s1) + --aux_d_max 2, --micro 2 fixed (CTRL's"
           " realised micro); one non-580 RTX 3090; admit / guard per host.\n# Evals on host: %s (as rcb_plaw4); eval_long local after landing. Frozen args"
           " scratch/s251_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
           "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\n"
           "boxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60 --auto_micro\n"
           "boxes/runs_fsq/%s boxes.inject_aw %s %s\n") % (run, s, EVAL, run, run, b, run, b, run, b, f)
    open("scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(txt); print("written b%d %s" % (n, run))
json.dump(diffs, open("scratch/gate_s251/args_rerun.json", "w", newline="\n"), indent=1, default=str)
print(json.dumps({k: {kk: sorted(vv) for kk, vv in v.items()} for k, v in diffs.items()}))
