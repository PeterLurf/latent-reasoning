"""THEORY §264 (b) pre-stated $0 reads on stored outputs (committed before computing; run only after Audit's ack; descriptive).
X4a FIRST-ERROR: bun_s1 / bun_s2 eval_ws_deep.json, TEST2 d 5-8 (n 1000 each): per item the first pass t <= d with ws_pred[t] != v[t - 1] (the pass-t ':' argmax vs
    v_min(t, d)); among items with an error: P(first error = t), share at t 1-2, share at t >= 5; plus per-pass conditional reliability P(ok at t | ok at 1..t-1).
    Reading (pre-stated): share(t >= 5) >= 0.6 at d 7-8 -> UNTRAINED-PASS (errors start in the never-trained passes); share(t 1-2) >= 0.4 -> EARLY (selection / length
    at the first hops); else ACCUMULATE.
M1 ECHO split: p2m_s1 (lr 3e-5) / p2m_s1b (lr 1e-4) eval_swap_k.json (d 2, S_2 := donor's S_2, decoder on the recipient's text): follow_dfin (= donor's final answer;
    S_2's v_2 content is used), follow_ca (= cf_a(1), the recipient's op 2 applied to the donor's v_1; the decoder recomputes from v_1 + text), floor (MEAN-K).
    Validity: P(dfin == ca) over pairs <= 0.5, else UNINFORMATIVE. Reading: USED-V2 if follow_dfin >= 0.5 and >= floor + 0.3; V1+TEXT if follow_ca >= 0.5 and
    follow_ca > follow_dfin; else NEITHER. Context (already read, post-hoc, by the §258 research agent): pass-2 ':' readout == v_1 on 0.837 / 1.000 of TEST2 d 2.
    python -B theory/read_s264_stored.py
"""
import json, os
R = "D:/Latent/boxes/runs_fsq"
out = {"X4a": {}, "M1": {}}
for run in ("bun_s1", "bun_s2"):
    f = json.load(open(os.path.join(R, run, "eval_ws_deep.json"))); o = {}
    for d in (5, 6, 7, 8):
        it = f["items"][str(d)]; n = len(it["sig"]); first = []
        ok_prev = [True] * n; cond = {}
        for t in range(1, d + 1):
            okt = [it["ws_pred"][str(t)][i] == it["v"][i][t - 1] for i in range(n)]
            den = sum(ok_prev); cond[str(t)] = (sum(a and b for a, b in zip(ok_prev, okt)) / den) if den else None
            for i in range(n):
                if ok_prev[i] and not okt[i]:
                    first.append(t)
            ok_prev = [a and b for a, b in zip(ok_prev, okt)]
        ne = len(first); dist = {str(t): (first.count(t) / ne if ne else 0.0) for t in range(1, d + 1)}
        s12 = sum(dist[str(t)] for t in (1, 2)); s5 = sum(dist[str(t)] for t in range(5, d + 1))
        o[str(d)] = {"n": n, "n_err": ne, "first_err_dist": dist, "share_t12": s12, "share_t5plus": s5, "cond_rel": cond}
    s5_78 = min(o["7"]["share_t5plus"], o["8"]["share_t5plus"]); s12_78 = max(o["7"]["share_t12"], o["8"]["share_t12"])
    o["reading"] = "UNTRAINED-PASS" if s5_78 >= 0.6 else "EARLY" if s12_78 >= 0.4 else "ACCUMULATE"
    out["X4a"][run] = o
for run in ("p2m_s1", "p2m_s1b"):
    f = json.load(open(os.path.join(R, run, "eval_swap_k.json"))); it = f["items"]
    eq = sum(a == b for a, b in zip(it["dfin"], it["ca"])) / len(it["dfin"])
    rec = {k: f[k] for k in ("n_pairs", "follow_dfin", "follow_ca", "follow_U", "floor", "intact_acc", "k")}; rec["p_dfin_eq_ca"] = eq
    rec["reading"] = ("UNINFORMATIVE" if eq > 0.5 else "USED-V2" if f["follow_dfin"] >= 0.5 and f["follow_dfin"] >= f["floor"] + 0.3 else
                      "V1+TEXT" if f["follow_ca"] >= 0.5 and f["follow_ca"] > f["follow_dfin"] else "NEITHER")
    out["M1"][run] = rec
os.makedirs("D:/Latent/theory/results", exist_ok=True)
json.dump(out, open("D:/Latent/theory/results/read_s264_stored.json", "w", encoding="utf-8", newline=chr(10)), indent=1)
print(json.dumps({"X4a": {r: {"reading": v["reading"], **{d: (v[d]["share_t12"], v[d]["share_t5plus"]) for d in ("5", "6", "7", "8")}} for r, v in out["X4a"].items()},
                  "M1": out["M1"]}, indent=1))
