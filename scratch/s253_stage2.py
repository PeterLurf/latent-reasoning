"""§253 LOOKUP-REBUILD Stage 2 kit (LB_v1 §2; Note 253-2 SEED 2/2 GO): lbr_s1 / lbr_s2 = rcb_plaw4_s<k>'s recipe (gate_s251.RCB = gen_q_b354's flags, seed k) with
--init_model boxes/runs_fsq/p2p_s<k>/model.pt (+ sha, asserted by the trainer) + --init_trunk boxes/runs_fsq/s252_askperm_s1/model.pt (+ sha) + --perm_p 0.5; micro FIXED 2 /
eval_bs 250 (rcb's realised; NO --auto_micro, Audit 4), 16k, --keep_final_ckpt; eval_parts = rcb's + test_perm (the scorer reads acc, test_perm (K' 2 at d 4), swap_kd, lp5h,
cue_only). -> scratch/gate_s253/stage2.json, scratch/s253_args_frozen_lbr_s<k>.json, scratch/gen_q_b385 / b386.
  load : build_aw(the lbr args) then p2p_s<k>/model.pt loaded strict (the trainer's own --init_model path) on CPU: every key matches.
  args : --args_only vs rcb_plaw4_s<k>/args.json (the realised CTRL args; C.diff).
    python -B scratch/s253_stage2.py"""
import argparse
import hashlib
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import gate_s228 as G8  # noqa: E402
import gate_s251 as G1  # noqa: E402
import s228_args_check as C  # noqa: E402

EVAL = "acc,lp5,mean_s,cue_only,cue_wrong,swap_k,swap_1,probe,ws,lp5h,swap_kd,test_perm"; TMP = ROOT + "/scratch/_tmp/s253"
sha16 = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]  # noqa: E731


def flags(k):
    fl = list(G1.RCB); i = fl.index("--init_model"); fl[i + 1] = "boxes/runs_fsq/p2p_s%d/model.pt" % k; j = fl.index("--init_model_sha"); fl[j + 1] = sha16(fl[i + 1])
    t = fl.index("--init_trunk"); fl[t + 1] = "boxes/runs_fsq/s252_askperm_s1/model.pt"; s = fl.index("--init_sha"); fl[s + 1] = "cf35fc6bcea9035e"
    return fl + (["--seed", "2"] if k == 2 else []) + ["--perm_p", "0.5", "--micro", "2", "--eval_bs", "250"]


def tail(run):
    return ["--keep_final_ckpt", "--ckpt_every", "1000", "--eval_parts", EVAL, "--args_ref", "scratch/s253_args_frozen_%s.json" % run]


def main():
    import torch
    from boxes import inject_aw as AW
    o = {}
    for k, n in ((1, 385), (2, 386)):
        run = "lbr_s%d" % k; fl = flags(k); d = G8.fresh(TMP + "/args_" + run)
        rc = G8.sh([sys.executable, "-B", "-m", "boxes.inject_aw"] + fl + tail(run)[:-2] + ["--device", "cpu", "--args_only", "--out", d], d + "/stdout.log")
        new = json.load(open(d + "/args.json")); ref = json.load(open("boxes/runs_fsq/rcb_plaw4_s%d/args.json" % k))
        dif = {kk: v for kk, v in C.diff(ref, new).items() if kk != "pair_stats"}; json.dump(new, open("scratch/s253_args_frozen_%s.json" % run, "w", newline="\n"), indent=1)
        ns = argparse.Namespace(**{**new, "init_trunk": "", "init_sha": ""}); torch.manual_seed(new["seed"]); m, g, params = AW.build_aw(ns, "cpu")
        sd = torch.load("boxes/runs_fsq/p2p_s%d/model.pt" % k, map_location="cpu", weights_only=True); res = params.load_state_dict(sd, strict=True)
        o[run] = {"rc": rc, "keys": sorted(dif), "diff": dif, "micro": new.get("micro"), "auto_micro": new.get("auto_micro"), "eval_bs": new.get("eval_bs"), "seed": new.get("seed"),
                  "init_model_sha": new.get("init_model_sha"), "strict_load_missing": list(res.missing_keys), "strict_load_unexpected": list(res.unexpected_keys)}
        b = " ".join(fl); f = " ".join(tail(run))
        txt = ("# THEORY §253 LOOKUP-REBUILD Stage 2 (LB_v1 §2; Note 253-2 SEED 2/2): %s = rcb_plaw4_s%d's §237 recipe (16k, REG) from p2p_s%d (--init_model, sha asserted) + init_trunk "
               "askperm + --perm_p 0.5; one non-580 RTX 3090; micro fixed 2 / eval_bs 250 (no --auto_micro).\n# Evals on host: %s. Frozen args scratch/s253_args_frozen_%s.json.\n"
               "# HOSTPROBE\n# P=1\nboxes/runs_fsq/%s_mem boxes.inject_aw %s --mem_check\nboxes/runs_fsq/%s_gtime boxes.inject_aw %s --time_steps 60\n"
               "boxes/runs_fsq/%s boxes.inject_aw %s %s\n") % (run, k, k, EVAL, run, run, b, run, b, run, b, f)
        open("scratch/gen_q_b%d.txt" % n, "w", encoding="utf-8", newline="\n").write(txt)
    o["pass"] = all(o[r]["rc"] == 0 and o[r]["micro"] == 2 and not o[r]["auto_micro"] and o[r]["eval_bs"] == 250 and not o[r]["strict_load_missing"]
                    and not o[r]["strict_load_unexpected"] for r in ("lbr_s1", "lbr_s2"))
    json.dump(o, open("scratch/gate_s253/stage2.json", "w", newline="\n"), indent=1, default=str)
    print({r: (o[r]["keys"], o[r]["seed"], o[r]["init_model_sha"]) for r in ("lbr_s1", "lbr_s2")}, "pass", o["pass"])


if __name__ == "__main__":
    main()
