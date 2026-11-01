"""§255 AEX-REP (registered 6b7339a1): aex_s3 / aex_s4 = aex_s1's run flags (gen_q_b380: gate_s251.RCB + --aux_d_max 2 --micro 2) with --seed 3 / 4, micro 2 PINNED and NO --auto_micro
(Audit 4), --keep_final_ckpt, eval_bs 250 (aex_s1's realised), eval_parts = aex_s1's (score_ae --rep reads eval_acc, eval_swap_kd, eval_lp5h, eval_cue_only + the log); one RTX 3090
each. -> scratch/s255_args_frozen_<run>.json, scratch/gate_s255/args.json (C.diff vs boxes/runs_fsq/aex_s1/args.json), scratch/gen_q_b388 / b389.
    python -B scratch/s255_mkq.py"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402
import s228_args_check as C  # noqa: E402

EVP = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd"; ref = json.load(open("boxes/runs_fsq/aex_s1/args.json")); out = {}
assert ref["eval_parts"] == EVP and ref["micro"] == 2 and ref["aux_d_max"] == 2, "aex_s1 reference"
for n, s in ((388, 3), (389, 4)):
    run = "aex_s%d" % s; fl = list(G1.RCB) + ["--seed", str(s), "--aux_d_max", "2", "--micro", "2", "--eval_bs", "250"]
    tail = ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EVP]
    d = G8.fresh(ROOT + "/scratch/_tmp/s255/args_" + run)
    rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
    new = json.load(open(d + "/args.json")); dif = {k: v for k, v in C.diff(ref, new).items() if k != "pair_stats"}
    json.dump(new, open("scratch/s255_args_frozen_%s.json" % run, "w", newline="\n"), indent=1)
    out[run] = {"keys": sorted(dif), "diff": dif, "micro": new["micro"], "auto_micro": new["auto_micro"], "eval_bs": new["eval_bs"], "seed": new["seed"]}
    txt = ("# THEORY §255 AEX-REP (6b7339a1): %s = aex_s1's run flags (gen_q_b380) with seed %d; micro 2 pinned / eval_bs 250, no --auto_micro; --keep_final_ckpt; one non-580 "
           "RTX 3090.\n# Evals on host: %s. Frozen args scratch/s255_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
           "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\nboxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60\n"
           "boxes/runs_fsq/%s boxes.inject_aw %s %s --args_ref scratch/s255_args_frozen_%s.json\n") % (run, s, EVP, run, run, " ".join(fl), run, " ".join(fl), run, " ".join(fl),
                                                                                                       " ".join(tail), run)
    open("scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(txt)
os.makedirs("scratch/gate_s255", exist_ok=True); json.dump(out, open("scratch/gate_s255/args.json", "w", newline="\n"), indent=1, default=str)
print({r: (v["keys"], v["micro"], v["auto_micro"], v["eval_bs"], v["seed"]) for r, v in out.items()})
