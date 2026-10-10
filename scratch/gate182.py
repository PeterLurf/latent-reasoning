"""THEORY §182 STEP-AUX-HOLD pre-launch gates (566616ef): boxes/nested170.py --aux_hold / kmatch_aux_loss (the aux on the masked K-window batch) /
--kprimes / --path_indep.
  (1) flags off = pre-§182 nested170 bitwise (CPU, 1 thread): §175's path (inject K2 + aux), §177 (f)'s path (k_window, masked), §170 K2; logs + eval fields
  (2) fixture: with hold, aux_logits' target on a pass t > n is ids(e_{n+1}) + EOT (step n's string); without hold a pass t > n is refused
  (3) gradflow: the aux CE of a HOLD pass alone gives nonzero gradient on the memory's parameters and on the state after that pass
  (4) the memory-check / training form: kmatch_aux_loss at n = 8 items, K = 12 (every item at K_max), hold on, backward runs, every aux pass counted
  (5) cell smokes (CPU, sched_scale 0.002): NOFADE / WINDOW / HOLD write eval.json with s170.acc_by_n_kprime over the --kprimes set, s175 (+ hold
      fields for HOLD), s182.path_indep, and args think_k / aux_fade / k_window / kmatch_impl / aux_hold as registered
  The 3090 memory check at n = 8, K = 12 with the aux is the timing run's own --auto_micro (mem_check's kmatch_aux_loss branch).
    python scratch/gate182.py <pre-182 nested170 copy>   -> boxes/runs_fsq/s182_gates/gate182.json (written after every gate)"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.nested170 as N  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s182_gates"); SP = os.environ.get("GATE182_TMP", os.path.join(GD, "tmp")); OUT = {}
TINY = ["--seed", "1", "--steps", "6", "--sched_scale", "0.002", "--log_every", "2", "--bs", "4", "--per_n", "2", "--eval_bs", "8", "--serial_pool", "40"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate182.json"), "w", newline="\n"), indent=1)


def run(args, out, script=None, threads="1"):
    if os.environ.get("GATE182_REUSE") and os.path.exists(os.path.join(out, "eval.json")):
        return 0, "reused"
    shutil.rmtree(out, ignore_errors=True); env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, PYTHONPATH=ROOT)
    cmd = [sys.executable, "-u"] + ([script] if script else ["-m", "boxes.nested170"]) + args + ["--device", "cpu", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT); return r.returncode, r.stdout[-300:] + r.stderr[-800:]


def run_many(jobs):
    """jobs: [(args, out, script, threads)] run CONCURRENTLY (each its own process; 1 thread each stays deterministic) -> [(rc, tail)]."""
    procs = []
    for args, out, script, threads in jobs:
        if os.environ.get("GATE182_REUSE") and os.path.exists(os.path.join(out, "eval.json")):
            procs.append(None); continue
        shutil.rmtree(out, ignore_errors=True); env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, PYTHONPATH=ROOT)
        cmd = [sys.executable, "-u"] + ([script] if script else ["-m", "boxes.nested170"]) + args + ["--device", "cpu", "--out", out]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT))
    out_ = []
    for p_ in procs:
        if p_ is None:
            out_.append((0, "reused")); continue
        o_, e_ = p_.communicate(); out_.append((p_.returncode, (o_ or "")[-300:] + (e_ or "")[-800:]))
    return out_


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, pre + k + ".")) if isinstance(v, dict) else o.__setitem__(pre + k, v)
    return o


def clean(d_):
    for f_ in ("model.pt", "ckpt_last.pt", "ckpt_prev.pt"):
        os.path.exists(os.path.join(d_, f_)) and os.remove(os.path.join(d_, f_))


def items_n(n, k, seed=3):
    r_ = np.random.default_rng(seed); out = []
    while len(out) < k:
        rt, t = N.generate(r_, n); out.append({"syms": N.render(rt), "t": t, "root": rt, "n": n})
    return out


def main():
    torch.set_num_threads(3); old = sys.argv[1]; ok = True
    # (1)
    g1 = {}
    for nm, extra in (("s175_aux", ["--arm", "inject", "--think_k", "2", "--aux_step", "1.0", "--aux_fade", "6000,8000"]),
                      ("s177f_kwin", ["--arm", "inject", "--think_k", "10", "--k_window", "0,2", "--kmatch_impl", "masked"]),
                      ("s170_k2", ["--arm", "inject", "--think_k", "2"])):
        dn, do = os.path.join(SP, "new_" + nm), os.path.join(SP, "old_" + nm); (rn, en), (ro, eo) = run_many([(extra + TINY, dn, None, "1"), (extra + TINY, do, old, "1")])
        if rn or ro:
            g1[nm] = {"rc_new": rn, "rc_old": ro, "err": (en if rn else eo)[-600:]}; continue
        Ln = [json.loads(x).get("loss") for x in open(os.path.join(dn, "log.jsonl"))]; Lo = [json.loads(x).get("loss") for x in open(os.path.join(do, "log.jsonl"))]
        fn, fo = flat(json.load(open(os.path.join(dn, "eval.json")))), flat(json.load(open(os.path.join(do, "eval.json"))))
        diff = sorted(k for k in set(fn) | set(fo) if not k.endswith("_sec") and fn.get(k) != fo.get(k))
        g1[nm] = {"log_losses_equal": Ln == Lo, "eval_fields_differing": diff[:20], "n_fields": len(fn)}; clean(dn); clean(do); dump()
    OUT["g1"] = g1; OUT["g1_pass"] = all(v.get("log_losses_equal") and not v.get("eval_fields_differing") for v in g1.values()) and len(g1) == 3; ok &= OUT["g1_pass"]; dump()
    # (2)
    torch.manual_seed(1); m = N.InjectN(s5.load_gpt2("cpu"), 6, "top"); its = items_n(2, 4); S = torch.randn(4, 12, 64, 64) * 0.05; g2 = {}
    with torch.no_grad():
        for t in (3, 4):
            _, y = m.aux_logits(S, its, t, "cpu", hold=True); yy = y.numpy()
            exp = [N.ids(N.aux_of(it)[1][0]) + [N.EOT] for it in its]
            g2["t%d" % t] = all(list(yy[r_][yy[r_] != -100]) == exp[r_] for r_ in range(len(its)))
        _, y2 = m.aux_logits(S, its, 2, "cpu", hold=True); _, y2b = m.aux_logits(S, its, 2, "cpu")
        g2["t2_hold_equals_plain"] = bool(torch.equal(y2, y2b))
        try:
            m.aux_logits(S, its, 3, "cpu"); g2["refused_without_hold"] = False
        except AssertionError:
            g2["refused_without_hold"] = True
    OUT["g2"] = g2; OUT["g2_pass"] = all(g2.values()); ok &= OUT["g2_pass"]; dump()
    # (3)
    m.train()
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5)
    x, msk = m.think_inputs(its, "cpu"); S = m.mem.zero_state(4, "cpu"); kept = []
    for t in range(1, 5):
        S = m.write(S, x, msk); S.retain_grad(); kept.append(S)
    lg_a, y_a = m.aux_logits(S, its, 4, "cpu", hold=True); Fn.cross_entropy(lg_a.transpose(1, 2), y_a, ignore_index=-100).backward()
    mz = [n_ for n_, p_ in m.mem.named_parameters() if p_.requires_grad and (p_.grad is None or float(p_.grad.norm()) == 0)]
    OUT["g3"] = {"hold_pass": 4, "n": 2, "mem_zero_grad": mz, "S_grad_by_pass": [float(k_.grad.norm()) if k_.grad is not None else 0.0 for k_ in kept]}
    OUT["g3_pass"] = not [z for z in mz if not z.startswith("s_")] and min(OUT["g3"]["S_grad_by_pass"]) > 0; ok &= OUT["g3_pass"]; dump()
    # (4)
    m.zero_grad(set_to_none=True); its8 = items_n(8, 3, seed=5); calls = []; a0 = N.InjectN.aux_logits

    def al(self, S, items, t, dev, hold=False):
        calls.append((t, len(items))); return a0(self, S, items, t, dev, hold)
    N.InjectN.aux_logits = al
    try:
        la, ax, Sf = N.kmatch_aux_loss(m, its8, [12] * 3, "cpu", lambda: torch.autocast("cpu", enabled=False), hold=True); (la + ax).backward()
    finally:
        N.InjectN.aux_logits = a0
    OUT["g4"] = {"aux_passes": [c[0] for c in calls], "items_per_pass": [c[1] for c in calls], "loss": [float(la), float(ax)],
                 "grad_wo": float(m.mem.wo.weight.grad.norm()) if m.mem.wo.weight.grad is not None else 0.0}
    OUT["g4_pass"] = OUT["g4"]["aux_passes"] == list(range(1, 13)) and OUT["g4"]["grad_wo"] > 0; ok &= OUT["g4_pass"]; dump()
    # (5)
    base = ["--arm", "inject", "--aux_step", "1.0", "--fast_add", "--kprimes", "0,1,4,8,12,16,24,32", "--path_indep"]
    cells = {"nofade": base + ["--think_k", "8", "--aux_fade", "48000,48000"],
             "window": base + ["--think_k", "12", "--aux_fade", "48000,48000", "--k_window", "1,4", "--kmatch_impl", "masked"],
             "hold": base + ["--think_k", "12", "--aux_fade", "48000,48000", "--k_window", "1,4", "--kmatch_impl", "masked", "--aux_hold"]}
    g5 = {}; rcs = dict(zip(cells, run_many([(args + TINY, os.path.join(SP, "cell_" + nm), None, "1") for nm, args in cells.items()])))
    for nm, args in cells.items():
        d_ = os.path.join(SP, "cell_" + nm); rc, err = rcs[nm]
        if rc:
            g5[nm] = {"rc": rc, "err": err[-600:]}; dump(); continue
        e_ = json.load(open(os.path.join(d_, "eval.json"))); a_ = json.load(open(os.path.join(d_, "args.json"))); s75 = e_.get("s175", {})
        g5[nm] = {"rc": 0, "kprimes": sorted(e_.get("s170", {}).get("acc_by_n_kprime", {}), key=int), "s175_keys": sorted(s75)[:14],
                  "hold_fields": {k: s75.get(k) for k in ("aux_acc_hold_by_offset", "aux_acc_hold")}, "path_indep": e_.get("s182", {}).get("path_indep"),
                  "args": {k: a_.get(k) for k in ("think_k", "aux_fade", "k_window", "kmatch_impl", "aux_hold", "kprimes", "path_indep")}}
        clean(d_); dump()
    OUT["g5"] = g5
    want = [str(x) for x in (0, 1, 4, 8, 12, 16, 24, 32)]
    OUT["g5_pass"] = all(v.get("rc") == 0 and v["kprimes"] == want and v["path_indep"] and v["path_indep"]["n_items"] > 0 and "aux_acc_by_pass" in v["s175_keys"] for v in g5.values()) \
        and len(g5) == 3 and g5["hold"]["hold_fields"]["aux_acc_hold"] is not None and g5["window"]["hold_fields"]["aux_acc_hold"] is None \
        and g5["nofade"]["args"]["k_window"] == "" and g5["hold"]["args"]["aux_hold"] is True and g5["window"]["args"]["aux_hold"] is False
    ok &= OUT["g5_pass"]; OUT["all_pass"] = bool(ok); dump(); print("GATE182", json.dumps(OUT)[:3000], flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
