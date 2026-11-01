"""§254 P2P-CHAIN (registered 53c5dfc3): p2pc_s1 = aw_plaw2_s1's P-2 recipe (gen_q_b347's flags) + --init_trunk boxes/runs_fsq/s252_askperm_s1/model.pt (+ sha, asserted by
build_aw on the host) + an explicit --perm_p 0; micro 1 pinned / eval_bs 250 (aw_plaw2_s1's realised; NO --auto_micro, incl. the G-TIME line); seed 1; 12k; --keep_final_ckpt;
eval_parts = aw_plaw2_s1's (score_pc reads eval_acc, eval_ws, log aux_vacc_by_d_pass). -> scratch/s254_args_frozen_p2pc_s1.json, scratch/gate_s254/args.json (C.diff vs
aw_plaw2_s1/args.json), scratch/gen_q_b387.txt.
    python -B scratch/s254_mkq.py"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import s228_args_check as C  # noqa: E402
import s253_build as B  # noqa: E402

RUN, N = "p2pc_s1", 387; EVP = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws"
FL = B.AW2 + B.INIT["askperm"] + ["--perm_p", "0", "--micro", "1", "--eval_bs", "250"]
TAIL = ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EVP]
d = G8.fresh(ROOT + "/scratch/_tmp/s254/args_" + RUN)
rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + FL + TAIL + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log"); assert rc == 0, rc
new = json.load(open(d + "/args.json")); ref = json.load(open("boxes/runs_fsq/aw_plaw2_s1/args.json"))
dif = {k: v for k, v in C.diff(ref, new).items() if k != "pair_stats"}; json.dump(new, open("scratch/s254_args_frozen_%s.json" % RUN, "w", newline="\n"), indent=1)
os.makedirs("scratch/gate_s254", exist_ok=True)
o = {"keys": sorted(dif), "diff": dif, "micro": new.get("micro"), "auto_micro": new.get("auto_micro"), "eval_bs": new.get("eval_bs"), "perm_p": new.get("perm_p"), "seed": new.get("seed"),
     "eval_parts_equal_ref": new.get("eval_parts") == ref.get("eval_parts")}
json.dump(o, open("scratch/gate_s254/args.json", "w", newline="\n"), indent=1, default=str)
txt = ("# THEORY §254 P2P-CHAIN (53c5dfc3): %s = aw_plaw2_s1's P-2 recipe (gen_q_b347) + init_trunk askperm (sha asserted) + --perm_p 0; one non-580 RTX 3090; micro 1 pinned / "
       "eval_bs 250 (no --auto_micro); 12k; --keep_final_ckpt.\n# Evals on host: %s. Frozen args scratch/s254_args_frozen_%s.json.\n# HOSTPROBE\n# P=1\n"
       "boxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\nboxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60\nboxes/runs_fsq/%s boxes.inject_aw %s %s --args_ref "
       "scratch/s254_args_frozen_%s.json\n") % (RUN, EVP, RUN, RUN, " ".join(FL), RUN, " ".join(FL), RUN, " ".join(FL), " ".join(TAIL), RUN)
open("scratch/gen_q_b%d.txt" % N, "w", encoding="utf-8", newline="\n").write(txt)
print(json.dumps({k: v for k, v in o.items() if k != "diff"}))
