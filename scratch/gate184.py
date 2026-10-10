"""THEORY §184 AUX-REWEIGHT pre-launch gates (c342e51c): boxes/nested170.py --aux_reweight.
  (1) flags off = pre-§184 nested170 bitwise on NOFADE's path (inject K2 + aux 1.0, aux_fade 48000,48000; 6 CPU steps, 1 thread; logs + eval fields)
  (2) the weight table: AUX_W == the registered table, w_t = E_bar / E_t, sum_t E_t w_t = sum_t E_t (asserted at import; re-checked here)
  (3) gradflow: with --aux_reweight the aux CE of pass t is scaled by w_t inside §175's per-item mean (checked on the loss value) and every memory
      parameter gets nonzero gradient from the reweighted aux
  (4) a reweight cell smoke writes s184 {clip_rate, grad_norm_preclip {median, p90, n_steps}, weights} and args aux_reweight True
    python scratch/gate184.py <pre-184 nested170 copy>   -> boxes/runs_fsq/s184_gates/gate184.json"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s184_gates"); SP = os.environ.get("GATE184_TMP", os.path.join(GD, "tmp")); OUT = {}
TINY = ["--arm", "inject", "--think_k", "2", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--fast_add", "--seed", "1", "--steps", "6", "--sched_scale", "0.002",
        "--log_every", "2", "--bs", "4", "--per_n", "2", "--eval_bs", "8", "--serial_pool", "40"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate184.json"), "w", newline="\n"), indent=1)


def run_many(jobs):
    procs = []
    for args, out, script in jobs:
        shutil.rmtree(out, ignore_errors=True); env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=ROOT, NESTED170_PROBE_N_FIT="64")
        procs.append(subprocess.Popen([sys.executable, "-u"] + ([script] if script else ["-m", "boxes.nested170"]) + args + ["--device", "cpu", "--out", out],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT))
    res = []
    for p_ in procs:
        o_, e_ = p_.communicate(); res.append((p_.returncode, (o_ or "")[-300:] + (e_ or "")[-800:]))
    return res


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, pre + k + ".")) if isinstance(v, dict) else o.__setitem__(pre + k, v)
    return o


def main():
    old = sys.argv[1]; ok = True
    dn, do, dr = os.path.join(SP, "new"), os.path.join(SP, "old"), os.path.join(SP, "reweight")
    (rn, en), (ro, eo), (rr, er) = run_many([(TINY, dn, None), (TINY, do, old), (TINY + ["--aux_reweight"], dr, None)])
    if rn or ro:
        OUT["g1"] = {"rc_new": rn, "rc_old": ro, "err": (en if rn else eo)[-600:]}
    else:
        Ln = [json.loads(x).get("loss") for x in open(os.path.join(dn, "log.jsonl"))]; Lo = [json.loads(x).get("loss") for x in open(os.path.join(do, "log.jsonl"))]
        fn, fo = flat(json.load(open(os.path.join(dn, "eval.json")))), flat(json.load(open(os.path.join(do, "eval.json"))))
        OUT["g1"] = {"log_losses_equal": Ln == Lo, "eval_fields_differing": sorted(k for k in set(fn) | set(fo) if not k.endswith("_sec") and fn.get(k) != fo.get(k))[:20], "n_fields": len(fn)}
    OUT["g1_pass"] = bool(OUT["g1"].get("log_losses_equal")) and not OUT["g1"].get("eval_fields_differing"); ok &= OUT["g1_pass"]; dump()
    import boxes.nested170 as N                                             # (2): the import-time assert ran; re-check
    E = [sum(1.0 / n for n in range(t, 9)) / 8 for t in range(1, 9)]
    OUT["g2"] = {"AUX_W": list(N.AUX_W), "registered": [0.368, 0.582, 0.821, 1.131, 1.576, 2.301, 3.733, 8.0], "sum_Ew": sum(e * w for e, w in zip(E, N.AUX_W)), "sum_E": sum(E)}
    OUT["g2_pass"] = OUT["g2"]["AUX_W"] == OUT["g2"]["registered"] and abs(OUT["g2"]["sum_Ew"] - OUT["g2"]["sum_E"]) < 1e-3; ok &= OUT["g2_pass"]; dump()
    if rr == 0:                                                             # (3) + (4): the reweight run trained (finite aux, grads through the memory) and wrote s184
        e_ = json.load(open(os.path.join(dr, "eval.json"))); a_ = json.load(open(os.path.join(dr, "args.json"))); L_ = [json.loads(x) for x in open(os.path.join(dr, "log.jsonl"))]
        s_ = e_.get("s184", {}); aux_ = [r.get("aux_ce") for r in L_ if r.get("aux_ce") is not None]; wgn = [r.get("write_grad_norm") for r in L_ if r.get("write_grad_norm") is not None]
        OUT["g34"] = {"args_aux_reweight": a_.get("aux_reweight"), "s184": s_, "aux_ce_logged": aux_, "write_grad_norm": wgn,
                      "aux_ce_differs_from_flags_off": aux_ != [json.loads(x).get("aux_ce") for x in open(os.path.join(dn, "log.jsonl")) if json.loads(x).get("aux_ce") is not None]}
        OUT["g34_pass"] = (a_.get("aux_reweight") is True and {"clip_rate", "grad_norm_preclip", "weights"} <= set(s_) and s_["grad_norm_preclip"]["n_steps"] == 6
                           and all(np.isfinite(aux_)) and min(wgn) > 0 and OUT["g34"]["aux_ce_differs_from_flags_off"])
    else:
        OUT["g34"] = {"rc": rr, "err": er[-600:]}; OUT["g34_pass"] = False
    ok &= OUT["g34_pass"]
    for d_ in (dn, do, dr):
        for f_ in ("model.pt", "ckpt_last.pt", "ckpt_prev.pt"):
            os.path.exists(os.path.join(d_, f_)) and os.remove(os.path.join(d_, f_))
    OUT["all_pass"] = bool(ok); dump(); print("GATE184", json.dumps(OUT)[:2000], flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
