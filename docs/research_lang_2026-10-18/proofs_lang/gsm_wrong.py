"""PROOFS_LANG item 5b: share of GSM8K-Aug equations whose stated RHS is not a rounding of the exact LHS value
(|exact - stated| > 0.5 * 10^-k, k = stated decimals; also the looser truncation tolerance 10^-k), over DECOMP-parsed
equations; train (all) and test. Uses gsm_check.tree (my parser)."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_check import load, tree
out = {}
for name, p in (("test", "D:/Latent/bench/data/gsm8k_aug/test.txt"), ("train_all", "D:/Latent/bench/data/gsm8k_aug/train.txt")):
    n = wr_half = wr_trunc = nonnum = 0
    for it in load(p):
        if it is None:
            continue
        for e in it[1]:
            if "=" not in e:
                continue
            lhs, rhs = e.rsplit("=", 1); t = tree(lhs)
            if t is None or not t[1]:
                continue
            r = rhs.replace(",", "").strip()
            try:
                sv = float(r)
            except Exception:
                nonnum += 1; continue
            k = len(r.split(".")[1]) if "." in r else 0
            n += 1; d = abs(t[0] - sv)
            wr_half += d > 0.5 * 10 ** -k + 1e-9; wr_trunc += d > 10 ** -k + 1e-9
    out[name] = {"equations": n, "wrong_half_ulp": wr_half, "share_half": round(wr_half / n, 5), "wrong_trunc": wr_trunc, "share_trunc": round(wr_trunc / n, 5), "rhs_nonnumeric": nonnum}
    print(name, out[name], flush=True)
json.dump(out, open("gsm_wrong.json", "w", newline="\n"), indent=1)
