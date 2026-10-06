"""THEORY note 128-4 (descriptive): does the CERT's BFS cover every code the eval visits? For a §128 held-out run dir, under bf16
(the eval's dtype) and under fp32 (autocast off): the code set visited by the main eval's chains (iid items from start 0, rng 12801,
test_n x test_len as in fsq120's held-out eval) and the code set reached by fsq120's certificate() BFS from the start code;
n_uncovered = |eval codes - BFS codes|. Writes <run>/eval_cert_cover.json and <run>/eval_cert_fp32.json (the fp32 CERT).
    python scratch/cert_cover128.py <run dir> [device] [test_n] [out dir]   (out dir: default the run dir; the Vast 3090 pass writes
    to its own dir, scratch/cc_job.py, so the 4060 files stay distinguishable - Theory 2026-10-06)"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("BOXES_NB", "5")
from boxes import s5  # noqa: E402
from boxes.fsq120 import FSQ120, Family, certificate, run_test  # noqa: E402


def main(d, dev="cuda:0", test_n=None, od=None):
    d = d.rstrip("/"); od = od or d
    a = json.load(open(os.path.join(d, "args.json"))); fam = Family(a["family"]); test_n = int(test_n or a.get("test_n", 2000))
    m = FSQ120(s5.load_gpt2(dev), fam).to(dev); m.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=dev)); m.eval()
    X = np.random.default_rng(12801).integers(0, fam.n_items, (test_n, int(a.get("test_len", 512))))
    bf = torch.bfloat16 if dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8 else None
    out = {"run": d, "test_n": test_n, "test_len": int(X.shape[1]), "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}
    for name, amp in (("bf16", bf), ("fp32", None)):
        if name == "bf16" and bf is None:
            continue
        _, codes = run_test(m, fam, None, X, dev, amp, int(a.get("eval_bs", 250)))
        ev = {r.tobytes() for r in codes.reshape(-1, codes.shape[-1]).astype(np.int8)}
        bfs = []; c = certificate(m, fam, dev, amp, int(a.get("cert_cap", 1_000_000)), time_limit=int(a.get("cert_time", 2700)), codes_out=bfs)
        bs = {np.asarray(r, np.int8).tobytes() for r in bfs}
        out[name] = {"eval_distinct": len(ev), "bfs_reachable": len(bs), "n_uncovered": len(ev - bs), "cert_holds": c.get("holds"),
                     "n_moore": c.get("n_moore"), "cert_status": c.get("status")}
        if name == "fp32":
            c.update({"eval_dtype": "None", "cert_cap": int(a.get("cert_cap", 1_000_000))})
            json.dump(c, open(os.path.join(od, "eval_cert_fp32.json"), "w", newline="\n"), indent=1)
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(os.path.join(od, "eval_cert_cover.json"), "w", newline="\n"), indent=1)
    return out


if __name__ == "__main__":
    g = sys.argv[1:] + [None] * 4
    main(g[0], g[1] or "cuda:0", g[2], g[3])
