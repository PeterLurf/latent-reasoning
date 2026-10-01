"""Exploratory (docs/PROTOCOL8B_STEP1.md, cue test; idea: Theory): is the near-target-checkpoint flip a lexical cue?

    python cue_p8b.py --ckpt <ckpt> --data p8data/p8_test.json --out <json> [--n_per_d 40] [--device cuda:0]

Per problem (probe-disjoint, answer depths 3-10), at one random statement position, the margin
log p(target answer) - log p(negative answer) after K' = 10 thoughts for four texts:
  plain; +PT "<root> is a <gold-path parent of the target>." (true, consistent);
  +PN "<root> is a <a parent of the negative option>." (makes the negative option derivable too);
  +R "<root> is a <concept that reaches neither option>." (control).
A model reasoning from premises moves toward the option the written entity leads to (PN: toward negative); a lexical
"adjacent option is wrong" cue moves away from it (PT: toward negative, PN: toward target).
"""
import argparse
import json
from collections import defaultdict, deque

import numpy as np
import torch

from reanchor_p8 import STMT, lp_of, split, with_checkpoint
from gpt2c import GPT2C, load_state


def reaches(edges, n):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
    out = []
    for s in range(n):
        seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        out.append(seen)
    return out


def probe_ids(data):
    cal = [i for i, x in enumerate(data) if x["ext"] <= 2][:100]; probe = set(cal); per = {}
    for i, x in enumerate(data):
        if i in probe or per.get(x["ext"], 0) >= 150:
            continue
        per[x["ext"]] = per.get(x["ext"], 0) + 1; probe.add(i)
    return probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_d", type=int, default=40); ap.add_argument("--Kp", type=int, default=10)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device)); model.mask_answer = True
    rng = np.random.default_rng(a.seed)
    data = json.load(open(a.data)); probe = probe_ids(data)
    seen, rows = {}, []
    for i, x in enumerate(data):
        d = len(x["steps"])
        if i in probe or not 3 <= d <= 10 or seen.get(d, 0) >= a.n_per_d:
            continue
        sym = x["idx_to_symbol"]; n = len(sym); t, ng, root = x["target"], x["neg_target"], x["root"]
        R = reaches(x["edges"], n)
        pt = STMT.match(x["steps"][-1]).group(1) or STMT.match(x["steps"][-1]).group(2)
        pn = [sym[u] for u, v in x["edges"] if v == ng and sym[u][:1].islower()]
        rc = [sym[u] for u in range(n) if sym[u][:1].islower() and u not in (t, ng) and t not in R[u] and ng not in R[u]]
        if not pn or not rc or not pt[:1].islower():
            continue
        seen[d] = seen.get(d, 0) + 1
        stmts, _ = split(x)
        pos = int(rng.integers(0, len(stmts) + 1))
        r = {"depth": d}
        for k, y in (("plain", None), ("PT", pt), ("PN", str(rng.choice(pn))), ("R", str(rng.choice(rc)))):
            lp = lp_of(model, x if y is None else with_checkpoint(x, y, pos), a.Kp)[0]
            r[k] = float(lp[0] - lp[1])
        rows.append(r)
        if len(rows) % 40 == 0:
            print(f"{len(rows)} problems", flush=True)
    json.dump({"ckpt": a.ckpt, "n": len(rows), "rows": rows}, open(a.out, "w"))
    for lo, hi in ((3, 6), (7, 10)):
        sel = [r for r in rows if lo <= r["depth"] <= hi]
        print(f"d{lo}-{hi} n {len(sel)} " + " ".join(f"{k}: acc {np.mean([r[k] > 0 for r in sel]):.2f} dm {np.mean([r[k] - r['plain'] for r in sel]):+.2f}"
                                                for k in ("plain", "PT", "PN", "R")))
    print("DONE", a.out)


if __name__ == "__main__":
    main()
