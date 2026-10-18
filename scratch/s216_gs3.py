"""§216 G-S3 STREAMS for s216_r2_s1 (SD2_v1 §7 / §2.3 D11 at e9ade2ec), from draws_log.json:
  (a) P(R | t >= 13): logged = over every logged item (self: self_n, K = min(n + U, 20); plain: n, K = n + U) sum max(0, min(n, K) - t* + 1) / sum max(0, K - t* + 1);
      precomputed = the same from the first 383,488 pool_r2.jsonl records' n and s215_ftself_s1/draws_log.json's self U (the seed-1 replay); must be EQUAL,
      ~ 1/2 (SE 0.0004), and 0 at t* >= 17 (K <= 20 but n <= 16).
  (b) plain_hash at steps 1..200 (and every logged step) == s215_ftself_s1/draws_log.json; coins and self U identical at every step (D11 replay).
  (c) consumed new-window (n 13..16) share 2/3 +- 0.001.
    python scratch/s216_gs3.py"""
import json
import os
import sys

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
R = "boxes/runs_fsq/"; A = json.load(open(R + "s216_r2_s1/draws_log.json")); B = json.load(open(R + "s215_ftself_s1/draws_log.json"))
pool = [json.loads(l)["n"] for l in open(R + "s216_pool_s1/pool_r2.jsonl", encoding="utf-8")]


def P(items, ts):
    tot = sum(max(0, K - ts + 1) for n, K in items); unf = sum(max(0, min(n, K) - ts + 1) for n, K in items)
    return unf / tot if tot else None


logged = []
for d in A:
    logged += ([(n, min(n + u, 20)) for n, u in zip(d["self_n"], d["U"])] if d["coin"] else [(n, n + u) for n, u in zip(d["n"], d["U"])])
pre, cur = [], 0
for d in B:                                                                 # FT-SELF's coins / U, with round 2's pool order
    if d["coin"]:
        ns = pool[cur:cur + len(d["U"])]; cur += len(d["U"]); pre += [(n, min(n + u, 20)) for n, u in zip(ns, d["U"])]
    else:
        pre += [(n, n + u) for n, u in zip(d["n"], d["U"])]
pa = {t: P(logged, t) for t in range(13, 21)}; pp = {t: P(pre, t) for t in range(13, 21)}
ba = {d["step"]: d for d in A}; bb = {d["step"]: d for d in B}; common = sorted(set(ba) & set(bb))
hash_eq = all(ba[s]["plain_hash"] == bb[s]["plain_hash"] for s in common); h200 = all(s in ba and s in bb and ba[s]["plain_hash"] == bb[s]["plain_hash"] for s in range(1, 201))
coin_eq = all(ba[s]["coin"] == bb[s]["coin"] for s in common); u_eq = all(ba[s]["U"] == bb[s]["U"] for s in common if ba[s]["coin"])
sn = [n for d in A if d["coin"] for n in d["self_n"]]; share = sum(13 <= n <= 16 for n in sn) / len(sn)
o = {"P_logged": pa, "P_precomputed": pp, "P_equal": pa == pp, "P13": pa[13], "steps": len(A), "common": len(common), "plain_hash_all_equal": hash_eq, "plain_hash_1_200_equal": h200,
     "coins_equal": coin_eq, "self_U_equal": u_eq, "self_items": len(sn), "new_window_share": share, "consumed_cursor": cur}
o["pass"] = (o["P_equal"] and abs(pa[13] - 0.5) <= 0.002 and all(pa[t] in (0, 0.0, None) for t in range(17, 21)) and h200 and hash_eq and coin_eq and u_eq and abs(share - 2 / 3) <= 0.001)
json.dump(o, open("scratch/gate_s216/gs3_r2_s1.json", "w", newline="\n"), indent=1)
print("G-S3", "PASS" if o["pass"] else "FAIL", {k: v for k, v in o.items() if k not in ("P_logged", "P_precomputed")}, {t: (round(v, 5) if v is not None else None) for t, v in pa.items()})
