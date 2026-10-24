"""§230 FH-FT-FIX build gates (THEORY §230 + Note 230-1; for Audit's G-D). -> scratch/gate_s230/<part>.json ("pass"). Run dirs under scratch/_tmp/s230.
  k1tree    gate_s224.k1 (far_branch gradflow: L_far and L_FP nonzero on blocks 6 / 9 / slva_wq, prefix detached, decorators) on the committed tree + this
            build (far_branch is unchanged; written here, not over scratch/gate_s224/k1.json).
  flagsoff  HEAD boxes/vbank.py (copied to boxes/_vbank_head.py for the run) vs the working tree, det wrapper, fp32, the RTX 4060: (a) FH-FT fixture
            (§224 k6's: FT from s213_d3t_s1, --far --fh --fh_dose 3,6 --far_t 13,14 --far_n 4 --bs 16, stop_at 12, ckpt 12): the dose rule sets lambda;
            (b) the plain path (FT, no --far; stop_at 6, ckpt 6): logs (minus sec) identical, ckpt model + opt torch.equal.
  fixedlam  the (a) fixture + --fh_lambda_fixed 1.6 (log_every 1): fh_lambda 1.6 from step 1, no fh_lambda_set line, the fh_lfp1_step0 line at step 1,
            fh_gnorm_fp1 > 0 finite (the L_FP gradient at fixed lambda), fh_fp_over_ans / fh_fp_over_rest finite every record; vs --fh_lambda_fixed
            1e-6: the step-12 model differs (the lambda term acts on the update).
  args      scratch/s224_args_frozen_s230_fhftfix_s1.json vs scratch/s224_args_frozen_s224_fhft_s1.json (s224_args_check CONTROL): == {fh_lambda_fixed}.
    python -B scratch/gate_s230.py k1tree flagsoff fixedlam args"""
import json
import math
import os
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
OUT = ROOT + "/scratch/gate_s230"; TMP = ROOT + "/scratch/_tmp/s230"; DEV = "cuda:0"
import gate_s228 as G8  # noqa: E402

D3T = "boxes/runs_fsq/s213_d3t_s1"
FT = ["--mode", "slva", "--init_from", D3T + "/model.pt", "--steps", "12000", "--lr", "3e-5", "--cur", "8", "--c_sched", "0,1"]
FH = FT + ["--far", "--fh", "--fh_dose", "3,6", "--far_t", "13,14", "--far_n", "4", "--bs", "16", "--micro", "1", "--stop_at", "12", "--ckpt_every", "12", "--log_every", "1"]
PL = FT + ["--bs", "16", "--micro", "1", "--stop_at", "6", "--ckpt_every", "6", "--log_every", "1"]


def run(script, fl, d):
    G8.fresh(d); rc = G8.sh([sys.executable, "-B", "-m", "scratch.s222_det_wrap", script] + fl + ["--fp32", "--device", DEV, "--out", d], d + "/stdout.log")
    lg = [json.loads(l_) for l_ in open(d + "/log.jsonl")] if os.path.exists(d + "/log.jsonl") else []
    return rc, lg


def k1tree():
    sys.argv = [sys.argv[0], "--tree", ROOT]
    import gate_s224 as G4
    return G4.k1()


def flagsoff():
    head = ROOT + "/boxes/_vbank_head.py"
    open(head, "w", newline="\n").write(subprocess.run(["git", "show", "HEAD:boxes/vbank.py"], capture_output=True, text=True, cwd=ROOT).stdout)
    o = {}
    try:
        for tag, fl in (("fh_dose", FH), ("plain", PL)):
            ra, la = run("boxes/_vbank_head.py", fl, TMP + "/off_head_" + tag); rb, lb = run("boxes/vbank.py", fl, TMP + "/off_new_" + tag)
            r = {"rc": [ra, rb], "n_log": [len(la), len(lb)]}
            strip = lambda L: [{k: v for k, v in x.items() if k != "sec"} for x in L]  # noqa: E731
            r["log_equal"] = strip(la) == strip(lb); r["lambda_set"] = [x for x in la if "fh_lambda_set" in x]
            ca, cb = TMP + "/off_head_" + tag + "/ckpt_last.pt", TMP + "/off_new_" + tag + "/ckpt_last.pt"
            if os.path.exists(ca) and os.path.exists(cb):
                r.update(G8.ck_equal(ca, cb))
            r["pass"] = ra == 0 and rb == 0 and r["log_equal"] and r.get("n_mismatch") == 0 and len(la) > 0
            o[tag] = r
    finally:
        os.remove(head)
    o["pass"] = all(v["pass"] for v in o.values()) and bool(o["fh_dose"]["lambda_set"]); return o


def fixedlam():
    ra, la = run("boxes/vbank.py", FH + ["--fh_lambda_fixed", "1.6"], TMP + "/fix16"); rb, lb = run("boxes/vbank.py", FH + ["--fh_lambda_fixed", "1e-6"], TMP + "/fix0")
    recs = [x for x in la if "fh_lambda" in x]; fin = lambda v: v is not None and math.isfinite(v)  # noqa: E731
    o = {"rc": [ra, rb], "n_rec": len(recs), "first_rec": recs[0] if recs else None, "step0_line": [x for x in la if "fh_lfp1_step0" in x],
         "lambda_set_lines": [x for x in la if "fh_lambda_set" in x], "lambdas": sorted({x["fh_lambda"] for x in recs}),
         "gnorm_fp1": [x.get("fh_gnorm_fp1") for x in recs], "fp_over_ans": [x.get("fh_fp_over_ans") for x in recs], "fp_over_rest": [x.get("fh_fp_over_rest") for x in recs]}
    ca, cb = TMP + "/fix16/ckpt_last.pt", TMP + "/fix0/ckpt_last.pt"
    if os.path.exists(ca) and os.path.exists(cb):
        o["vs_1e-6"] = G8.ck_equal(ca, cb, keys=("model",))
    o["pass"] = (ra == 0 and rb == 0 and o["n_rec"] == 12 and o["lambdas"] == [1.6] and recs[0]["step"] == 1 and len(o["step0_line"]) == 1 and o["step0_line"][0]["step"] == 1
                 and not o["lambda_set_lines"] and all(fin(v) and v > 0 for v in o["gnorm_fp1"]) and all(fin(v) for v in o["fp_over_ans"] + o["fp_over_rest"])
                 and o.get("vs_1e-6", {}).get("n_mismatch", 0) > 0)
    return o


def args():
    import s224_args_check as C
    d = C.diff(json.load(open(ROOT + "/scratch/s224_args_frozen_s224_fhft_s1.json")), json.load(open(ROOT + "/scratch/s224_args_frozen_s230_fhftfix_s1.json")))
    return {"diff": d, "pass": d == {"fh_lambda_fixed": [None, 1.6]}}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    for part in [x for x in sys.argv[1:] if x in ("k1tree", "flagsoff", "fixedlam", "args")]:
        r = {"k1tree": k1tree, "flagsoff": flagsoff, "fixedlam": fixedlam, "args": args}[part]()
        json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
