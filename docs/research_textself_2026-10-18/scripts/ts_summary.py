"""TEXT-SELF draft: one table from ts_label_check_s1.json (+ _s1_deep.json): per n keep / pi / kept-wrong (CP 95 % one-sided lower bound on pi) per
protocol, the GATE-1 rule applied to the REROOT counts as a preview (keep_n >= 0.70, pi_n >= 0.99 at 200 items: a preview only, the gate is 2,000 per n),
and the fail-stage summary. python -B ts_summary.py"""
import json
import os
import sys

sys.dont_write_bytecode = True
from math import comb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def cp_lower(k_ok, n, alpha=0.05):
    """one-sided Clopper-Pearson lower bound for a binomial proportion k_ok / n."""
    if n == 0:
        return None
    if k_ok == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        tail = sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k_ok, n + 1))   # P(X >= k_ok | p)
        if tail < alpha:
            lo = p
        else:
            hi = p
    return lo


def main():
    out = {}
    for tag in ("_s1", "_s1_deep"):
        p = os.path.join(HERE, "ts_label_check%s.json" % tag)
        if not os.path.exists(p):
            continue
        R = json.load(open(p))
        for n, rn in R["by_n"].items():
            for proto in ("full", "reroot"):
                if proto not in rn:
                    continue
                r = rn[proto]; nk = r["n_kept"]; kw = r["kept_wrong"]
                out.setdefault(n, {})[proto] = {"items": R["per_n"], "keep": r["keep"], "n_kept": nk, "pi": r["pi"], "kept_wrong": kw,
                                                "pi_lower95": cp_lower(nk - kw, nk) if nk else None, "fail_step_hist": r["fail_step_hist"]}
    json.dump(out, open(os.path.join(HERE, "ts_summary.json"), "w", newline="\n"), indent=1)
    for n in sorted(out, key=int):
        print(n, {p: (v["keep"], v["n_kept"], v["kept_wrong"], None if v["pi_lower95"] is None else round(v["pi_lower95"], 4)) for p, v in out[n].items()})


if __name__ == "__main__":
    main()
