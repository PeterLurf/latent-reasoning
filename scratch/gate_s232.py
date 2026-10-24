"""§232 HOLD-C1 c1hold build gates (Note 232-1; REVIEW_HC fix 6; for Audit's G-D). -> scratch/gate_s232/<part>.json ("pass"). Everything runs on a STAGED copy
(scratch/_tmp/s232/smoke_base_s1: s213_d3t_s1's model.pt + args.json; committed run dirs are never written), the RTX 4060, fp32, --per_n 20 (small sizes:
a functional check; the memory check at the real shape runs inside part_c1hold on the 3090).
  plain    vbank_eval --parts acc,vswap (the existing path, no new flags) and vbank_diag --part d3hist (an existing part) on the staged base -> rc 0.
  c1hold   vbank_diag --part c1hold via main() on the staged base: rc 0; one top-level key "c1hold" with arms {ii_K, iv_K, ii_hold, v_K, v0off_K} all dicts;
           gate.pass (per-pair ii(K' 128 rollout, row n + 2) == ii_K: 1.0; pooled n_pairs / follow_ii / own_ii / follow_iv / own_iv == the same dir's
           eval_vb vswap exactly; v0off_K == v_K under a version-0-only keep mask, bitwise every chunk); items == n_pairs_all; known answer: arm (v) on
           the BASE follow >= 0.95 (REVIEW_HC: as newest_follow 1.000).
  halved   part_c1hold in-process with hold_bs 2 (forces the halved-hold path: the same-bs K = n + 2 reference): gate.pass, and every non-hold arm's
           per-pair answers == the c1hold run's (the hold_bs only moves the K' 128 rollout).
    python -B scratch/gate_s232.py plain c1hold halved"""
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch")
OUT = ROOT + "/scratch/gate_s232"; SM = ROOT + "/scratch/_tmp/s232/smoke_base_s1"; DEV = "cuda:0"; PN = "20"
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}


def sh(args, log):
    with open(log, "w") as f_:
        return subprocess.run([sys.executable, "-B", "-m"] + args, stdout=f_, stderr=subprocess.STDOUT, cwd=ROOT, env=ENV).returncode


def stage():
    if not os.path.exists(SM + "/model.pt"):
        os.makedirs(SM, exist_ok=True)
        for f_ in ("model.pt", "args.json"):
            shutil.copy(ROOT + "/boxes/runs_fsq/s213_d3t_s1/" + f_, SM + "/" + f_)


def plain():
    stage()
    r1 = sh(["boxes.vbank_eval", "--run", SM, "--device", DEV, "--per_n", PN, "--parts", "acc,vswap"], SM + "/plain_eval.log")
    r2 = sh(["boxes.vbank_diag", "--run", SM, "--device", DEV, "--per_n", PN, "--part", "d3hist", "--out_dir", SM], SM + "/plain_d3hist.log")
    ev = json.load(open(SM + "/eval_vb.json")) if os.path.exists(SM + "/eval_vb.json") else {}
    return {"rc": [r1, r2], "eval_vb_keys": sorted(ev), "d3hist": os.path.exists(SM + "/eval_d3hist.json"), "pass": r1 == 0 and r2 == 0 and "vswap" in ev and "acc_n2" in ev}


def c1hold():
    rc = sh(["boxes.vbank_diag", "--run", SM, "--device", DEV, "--per_n", PN, "--part", "c1hold", "--out_dir", SM], SM + "/c1hold.log")
    e = json.load(open(SM + "/eval_c1hold.json")) if os.path.exists(SM + "/eval_c1hold.json") else {}; c = e.get("c1hold", {})
    o = {"rc": rc, "top_keys": sorted(e), "arms_dicts": {k: isinstance(v, dict) for k, v in c.get("arms", {}).items()}, "gate": c.get("gate"), "hold_bs": c.get("hold_bs"),
         "mem_tried": c.get("mem_tried"), "n_pairs_all": c.get("n_pairs_all"), "n_items": len(c.get("items", [])), "n_pairs_clean": c.get("n_pairs_clean"),
         "known_answer_v_follow_ge_0.95": (c.get("arms", {}).get("v_K", {}).get("follow") or 0) >= 0.95}
    o["pass"] = (rc == 0 and "c1hold" in e and sorted(c.get("arms", {})) == ["ii_K", "ii_hold", "iv_K", "v0off_K", "v_K"] and all(o["arms_dicts"].values())
                 and bool(c.get("gate", {}).get("pass")) and o["n_items"] == o["n_pairs_all"] and o["known_answer_v_follow_ge_0.95"])
    return o


def halved():
    import numpy as np
    import torch
    import boxes.nested170 as N
    import boxes.vbank_diag as VD
    import boxes.vbank_eval as VE
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    m, _ = VE.load(SM, DEV); TE = N.test_set(int(PN)); r = VD.part_c1hold(m, TE, DEV, 32, SM, hold_bs=2)["c1hold"]
    ref = json.load(open(SM + "/eval_c1hold.json"))["c1hold"]["items"]; keys = ("ii_K", "iv_K", "v_K", "v0off_K", "clean", "y", "yd")
    same = float(np.mean([all(a[k] == b[k] for k in keys) for a, b in zip(r["items"], ref)])) if len(r["items"]) == len(ref) else None
    hold_same = float(np.mean([a["ii_hold_128"] == b["ii_hold_128"] for a, b in zip(r["items"], ref)])) if same is not None else None
    o = {"hold_bs": r["hold_bs"], "gate": r["gate"], "non_hold_arms_equal_share": same, "ii_hold_128_equal_share_vs_bs32": hold_same}
    o["pass"] = r["hold_bs"] == 2 and bool(r["gate"]["pass"]) and same == 1.0; return o


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for part in [x for x in sys.argv[1:] if x in ("plain", "c1hold", "halved")]:
        r = {"plain": plain, "c1hold": c1hold, "halved": halved}[part]()
        json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)
