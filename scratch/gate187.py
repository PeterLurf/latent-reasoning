"""THEORY §187 AUX-NOPASS pre-launch gates (87b7e23d): boxes/nested170.py --aux_nopass (+ halt173 load_model / T-SHIFT guard).
  (1) flags off = pre-§187 nested170 bitwise on §184's path (inject K2 + aux 1.0 + --aux_reweight, aux_fade 48000,48000; 6 CPU steps, 1 thread; log
      losses + every eval field)
  (2) the prompt: aux_prompt(t) under nopass = [EOT, " Step", " :"] for t = 1..64 and holds no pass-index token (PASS_TOKS); without nopass it is
      §175's [EOT, " Step", pass_tok(t), " :"]
  (3) gradflow: the nopass run trains (finite aux CE, differing from the flags-off run's; write_grad_norm > 0 every log); args aux_nopass True; its eval
      has s175 aux_acc_by_pass; halt173 --auxdone runs on it (nopass prompt via load_model) and --tshift refuses it
    python scratch/gate187.py <pre-187 nested170 copy>   -> boxes/runs_fsq/s187_gates/gate187.json"""
import json
import os
import shutil
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s187_gates"); SP = os.environ.get("GATE187_TMP", os.path.join(GD, "tmp")); OUT = {}
TINY = ["--arm", "inject", "--think_k", "2", "--aux_step", "1.0", "--aux_fade", "48000,48000", "--aux_reweight", "--fast_add", "--seed", "1", "--steps", "6",
        "--sched_scale", "0.002", "--log_every", "2", "--bs", "4", "--per_n", "2", "--eval_bs", "8", "--serial_pool", "40"]
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=ROOT, NESTED170_PROBE_N_FIT="64")


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate187.json"), "w", newline="\n"), indent=1)


def run_many(jobs):
    procs = []
    for args, out, script in jobs:
        shutil.rmtree(out, ignore_errors=True)
        procs.append(subprocess.Popen([sys.executable, "-u"] + ([script] if script else ["-m", "boxes.nested170"]) + args + ["--device", "cpu", "--out", out],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=ENV, cwd=ROOT))
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
    dn, do, dp = os.path.join(SP, "new"), os.path.join(SP, "old"), os.path.join(SP, "nopass")
    (rn, en), (ro, eo), (rp, ep) = run_many([(TINY, dn, None), (TINY, do, old), (TINY + ["--aux_nopass"], dp, None)])
    if rn or ro:
        OUT["g1"] = {"rc_new": rn, "rc_old": ro, "err": (en if rn else eo)[-600:]}
    else:
        Ln = [json.loads(x).get("loss") for x in open(os.path.join(dn, "log.jsonl"))]; Lo = [json.loads(x).get("loss") for x in open(os.path.join(do, "log.jsonl"))]
        fn, fo = flat(json.load(open(os.path.join(dn, "eval.json")))), flat(json.load(open(os.path.join(do, "eval.json"))))
        OUT["g1"] = {"log_losses_equal": Ln == Lo, "eval_fields_differing": sorted(k for k in set(fn) | set(fo) if not k.endswith("_sec") and fn.get(k) != fo.get(k))[:20], "n_fields": len(fn)}
    OUT["g1_pass"] = bool(OUT["g1"].get("log_losses_equal")) and not OUT["g1"].get("eval_fields_differing"); ok &= OUT["g1_pass"]; dump()
    import boxes.nested170 as N
    on, off = types.SimpleNamespace(nopass=True), types.SimpleNamespace(nopass=False)
    pr_on = [N.InjectN.aux_prompt(on, t) for t in range(1, 65)]; pr_off = [N.InjectN.aux_prompt(off, t) for t in range(1, 65)]
    OUT["g2"] = {"nopass_prompt": pr_on[0], "all_nopass_equal": all(p == [N.EOT, N.STEP_ID, N.COLON_ID] for p in pr_on),
                 "no_pass_token": not any(tk in N.PASS_TOKS for p in pr_on for tk in p),
                 "off_is_175": all(p == [N.EOT, N.STEP_ID, N.pass_tok(t), N.COLON_ID] for t, p in zip(range(1, 65), pr_off))}
    OUT["g2_pass"] = all(OUT["g2"][k] for k in ("all_nopass_equal", "no_pass_token", "off_is_175")); ok &= OUT["g2_pass"]; dump()
    if rp == 0:
        e_ = json.load(open(os.path.join(dp, "eval.json"))); a_ = json.load(open(os.path.join(dp, "args.json"))); L_ = [json.loads(x) for x in open(os.path.join(dp, "log.jsonl"))]
        aux_ = [r.get("aux_ce") for r in L_ if r.get("aux_ce") is not None]; wgn = [r.get("write_grad_norm") for r in L_ if r.get("write_grad_norm") is not None]
        aux0 = [json.loads(x).get("aux_ce") for x in open(os.path.join(dn, "log.jsonl")) if json.loads(x).get("aux_ce") is not None]
        ad = subprocess.run([sys.executable, "-m", "boxes.halt173", "--run", dp, "--auxdone", "--n", "2", "--bs", "4", "--device", "cpu", "--out", os.path.join(SP, "ad.json")],
                            capture_output=True, text=True, env=ENV, cwd=ROOT)
        ts = subprocess.run([sys.executable, "-m", "boxes.halt173", "--run", dp, "--tshift", "--n", "2", "--bs", "4", "--device", "cpu", "--out", os.path.join(SP, "ts.json")],
                            capture_output=True, text=True, env=ENV, cwd=ROOT)
        OUT["g3"] = {"args_aux_nopass": a_.get("aux_nopass"), "aux_ce": aux_, "aux_ce_flags_off": aux0, "write_grad_norm": wgn,
                     "s175_aux_acc_by_pass": (e_.get("s175") or {}).get("aux_acc_by_pass"), "auxdone_rc": ad.returncode, "auxdone_err": ad.stderr[-400:] if ad.returncode else "",
                     "tshift_refused": ts.returncode != 0 and "not §187" in ts.stderr}
        OUT["g3_pass"] = (a_.get("aux_nopass") is True and all(np.isfinite(aux_)) and aux_ != aux0 and min(wgn) > 0 and bool(OUT["g3"]["s175_aux_acc_by_pass"])
                          and ad.returncode == 0 and OUT["g3"]["tshift_refused"])
    else:
        OUT["g3"] = {"rc": rp, "err": ep[-600:]}; OUT["g3_pass"] = False
    ok &= OUT["g3_pass"]
    shutil.rmtree(SP, ignore_errors=True)
    OUT["all_pass"] = bool(ok); dump(); print("GATE187", json.dumps(OUT)[:2000], flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
