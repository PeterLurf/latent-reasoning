"""THEORY note 132-4 (descriptive, Theory 2026-10-06): per-held-out-table breakdown of a §132 run from its kept model.pt.
(a) The main eval replayed exactly (rng 13200 -> held-table draws -> problems, test_n x test_len, the run's own amp): cumulative
    accuracy per held-out table; long_acc_replay must equal eval.json's long_acc (check).
(b) An extra per-table sample: n_per problems on EVERY held-out table (rng 13208), cumulative at test_len.
(c) Per table: reachable states from 0 and fsq120's CERT (holds, n_moore, status) - all held-out tables, cert_time as the run.
Writes <out>/eval_breakdown.json (+ eval.json = its summary, so the Vast queue counts it done).
    python -m scratch.table132_breakdown --run boxes/runs_fsq/s132_k4_s1 --device cuda:0 --out boxes/runs_fsq/s132bd_k4_s1 [--n_per 100]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boxes import s5  # noqa: E402
from boxes.fsq120 import certificate  # noqa: E402
from boxes.table132 import Splits, Table132, TableFam, problems, reachable, rollout, table_tokens  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
ap.add_argument("--n_per", type=int, default=100)
a = ap.parse_args(); dev = a.device; os.makedirs(a.out, exist_ok=True); t0 = time.time()
r = json.load(open(os.path.join(a.run, "args.json"))); ev = json.load(open(os.path.join(a.run, "eval.json")))
k, mm = int(r["k"]), int(r["m"]); test_n, test_len, ebs = int(r["test_n"]), int(r["test_len"]), int(r["eval_bs"])
amp = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
sp = Splits(k, mm); fam = TableFam(k, mm)
m = Table132(s5.load_gpt2(dev), fam).to(dev); m.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location=dev)); m.p_hard = 1.0; m.eval()
H = len(sp.held); out = {"run": a.run, "k": k, "m": mm, "test_n": test_n, "test_len": test_len, "eval_dtype": str(amp), "gpu": torch.cuda.get_device_name(dev),
                         "n_held": H, "n_per": a.n_per}
with torch.no_grad():
    er = np.random.default_rng(13200); idx = er.integers(0, H, test_n)    # (a): exactly table132.main's held(er, test_n) + problems
    c, X, Y = problems(er, [sp.held[i] for i in idx], test_len, mm); m.ctx = None; C = (rollout(m, c, X, dev, amp, ebs) == Y).all(1)
    out["long_acc_replay"] = float(C.mean()); out["long_acc_eval"] = ev.get("long_acc"); out["replay_matches"] = abs(out["long_acc_replay"] - ev.get("long_acc", -1)) < 1e-9
    br = np.random.default_rng(13208); tabs = []                           # (b) + (c)
    for i, T in enumerate(sp.held):
        cb, Xb, Yb = problems(br, [T] * a.n_per, test_len, mm); m.ctx = None; Cb = rollout(m, cb, Xb, dev, amp, ebs) == Yb
        ft = TableFam(k, mm, T); m.fam = ft; m.ctx = torch.tensor(table_tokens(T)[None], device=dev)
        cc = certificate(m, ft, dev, amp, int(r.get("cert_cap", 1_000_000)), time_limit=int(r.get("cert_time", 600))); m.fam = fam
        sel = idx == i
        tabs.append({"i": i, "table": T.tolist(), "reachable_states": int(reachable(T)), "eval_n": int(sel.sum()), "eval_fail": int((~C[sel]).sum()),
                     "eval_acc": float(C[sel].mean()) if sel.any() else None, "sample_acc": float(Cb.all(1).mean()),
                     "sample_first_err_median": float(np.median(np.argmin(Cb, 1)[~Cb.all(1)])) if (~Cb.all(1)).any() else None,
                     "cert_holds": cc.get("holds"), "n_moore": cc.get("n_moore"), "cert_status": cc.get("status")})
        print("TABLE", json.dumps({x_: tabs[-1][x_] for x_ in ("i", "reachable_states", "eval_fail", "sample_acc", "cert_holds", "n_moore", "cert_status")}), flush=True)
fails = sorted(tabs, key=lambda t_: t_["eval_fail"], reverse=True); tot = sum(t_["eval_fail"] for t_ in tabs)
out["tables"] = tabs
out["summary"] = {"eval_fail_total": tot, "eval_fail_top3_share": (sum(t_["eval_fail"] for t_ in fails[:3]) / tot) if tot else None,
                  "tables_with_eval_fail": sum(1 for t_ in tabs if t_["eval_fail"]), "sample_acc_mean": float(np.mean([t_["sample_acc"] for t_ in tabs])),
                  "sample_acc_min": float(min(t_["sample_acc"] for t_ in tabs)), "tables_sample_lt_0.99": sum(1 for t_ in tabs if t_["sample_acc"] < 0.99),
                  "cert_holds": sum(1 for t_ in tabs if t_["cert_holds"]), "moore_eq_reachable": sum(1 for t_ in tabs if t_["n_moore"] == t_["reachable_states"]),
                  "sample_acc_moore_eq_reachable": float(np.mean([t_["sample_acc"] for t_ in tabs if t_["n_moore"] == t_["reachable_states"]] or [np.nan])),
                  "sample_acc_open": float(np.mean([t_["sample_acc"] for t_ in tabs if t_["n_moore"] != t_["reachable_states"]] or [np.nan]))}
out["sec"] = round(time.time() - t0)
json.dump(out, open(os.path.join(a.out, "eval_breakdown.json"), "w", newline="\n"), indent=1)
json.dump({x_: out[x_] for x_ in out if x_ != "tables"}, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
print("BREAKDOWN", json.dumps(out["summary"]), flush=True)
