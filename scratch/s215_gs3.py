"""§215 G-S3 STREAMS (SELFDEPTH_v1 §6 at 5ded8703; PROOFS_SELFDEPTH §2 support_check.mixture, empirical): from the two cells' draws_log.json.
  (a) P(t*) = sum over logged items of max(0, min(n, K) - t* + 1) / sum of max(0, K - t* + 1), t* 9..17 (plain items: n, K = n + U; self items:
      self_n, K = min(n + U, 16)), on FT-SELF's actual training draws; target P(9) = 0.494 +- 0.01 (trainer law, stage B), P(t*) = 0 at t* >= 13.
      Also the loss-weighted variant (weight 1 / K per pass) vs 0.490.
  (b) FT-SELF's plain_hash == FT-CTRL's at every logged step (the plain batch is drawn every step before the self coin), steps 1..200 (the registered
      window) and all common steps; self U are rng_s draws only, so (b) also shows U never touched rng / rng_k.
  (c) self share: #coin / #steps vs p_self 0.5; self n uniform over 9..12 (counts).
    python scratch/s215_gs3.py"""
import json
from collections import Counter

R = "D:/Latent/boxes/runs_fsq/"; S = json.load(open(R + "s215_ftself_s1/draws_log.json")); C = json.load(open(R + "s215_ftctrl_s1/draws_log.json"))
items = []
for d in S:
    if d["coin"]:
        items += [(n, min(n + u, 16)) for n, u in zip(d["self_n"], d["U"])]
    else:
        items += [(n, n + u) for n, u in zip(d["n"], d["U"])]
P = {}
for ts in range(9, 18):
    tot = sum(max(0, K - ts + 1) for n, K in items); unf = sum(max(0, min(n, K) - ts + 1) for n, K in items)
    totw = sum(max(0, K - ts + 1) / K for n, K in items); unfw = sum(max(0, min(n, K) - ts + 1) / K for n, K in items)
    P[ts] = {"P": unf / tot if tot else None, "P_lossw": unfw / totw if totw else None}
cs = {d["step"]: d["plain_hash"] for d in C}; ss = {d["step"]: d["plain_hash"] for d in S}; common = sorted(set(cs) & set(ss))
mis = [t for t in common if cs[t] != ss[t]]; w200 = [t for t in range(1, 201) if t in cs and t in ss]
coins = sum(d["coin"] for d in S); sn = Counter(n for d in S if d["coin"] for n in d["self_n"])
o = {"P": P, "n_items": len(items), "steps_self": len(S), "steps_ctrl": len(C), "hash_common_steps": len(common), "hash_mismatch_steps": mis[:20], "n_mismatch": len(mis),
     "hash_1_200_equal": len(w200) == 200 and not [t for t in w200 if cs[t] != ss[t]], "self_share": coins / len(S), "self_n_counts": dict(sorted(sn.items()))}
o["pass"] = (abs(P[9]["P"] - 0.494) <= 0.01 and all(P[t]["P"] in (0, 0.0, None) for t in range(13, 18)) and o["hash_1_200_equal"] and o["n_mismatch"] == 0)   # None: no pass >= t* (K <= 16)
o["def"] = __doc__
json.dump(o, open("D:/Latent/scratch/gate_s215/s3.json", "w", newline="\n"), indent=1)
print("G-S3", "PASS" if o["pass"] else "FAIL", {t: (None if v["P"] is None else round(v["P"], 4)) for t, v in P.items()}, "lossw9 %.4f" % P[9]["P_lossw"], "mismatch", o["n_mismatch"], "of", len(common),
      "self_share %.4f" % o["self_share"], o["self_n_counts"])
