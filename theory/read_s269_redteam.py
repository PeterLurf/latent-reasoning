"""THEORY Note 269-3 pre-stated $0 reads (Audit REDTEAM_BUN leads; committed before computing; run only after Audit's ack; descriptive).
(a) ERROR-PROPAGATION: bun_s1 / bun_s2 eval_ws_deep.json items, TEST2 d 5-8: for every item and pass t < d with WS wrong at t (pass-t ':' readout != v_t)
    and right at every pass < t, record whether pass t+1 is right (== v_t+1). recovery = P(right at t+1 | first error at t), pooled over d 5-8.
    SERIAL if recovery <= 0.10 (errors propagate through the memory; chance is ~1/11); RECOMPUTE if recovery >= 0.50 (deep passes recompute from text);
    else MIXED. Also per t.
(b) C5 PLAN-CLOCK: bun eval_stab_fresh.json items d 1-7 (pred[K'] K' 0..14): khalt by score_c5c's r = 3 rule; per d: P(khalt == d - 1), acc_halt
    (pred[khalt] == ans) and acc_dm1 (pred[d - 1] == ans). STATIC-CLOCK if P(khalt == d - 1) >= 0.95 and |acc_halt - acc_dm1| <= 0.01 at every d 2-7
    (the halt rule adds nothing to "read at d - 1 given d"); else ADAPTIVE (report where it departs). Also the batch-max khalt per d (Audit lead 5).
(c) C2 K'-PROBE (Audit lead 4): bun eval_acc.json by_d d 9 and 10 acc_kprime K' 8..12. Pass function is pass-index-blind, so with cue S_min(t, d) the
    d 10 item needs K' >= 9 to see its last cue (decoder does the last hop). MORE-THINKING if acc(K' 9..12 max) >= acc(K' 8) + 0.05 at d 10 in both seeds;
    FLAT if <= + 0.02; else MIXED. d 9 is the control (its last needed cue is at K' 8).
    python -B theory/read_s269_redteam.py
"""
import json, os, sys
sys.path.insert(0, "D:/Latent/theory")
import score_c5c as C  # noqa: E402
R = "D:/Latent/boxes/runs_fsq"
out = {}
for run in ("bun_s1", "bun_s2"):
    o = {}
    w = json.load(open(os.path.join(R, run, "eval_ws_deep.json")))
    rec = {}; tot = [0, 0]
    for d in (5, 6, 7, 8):
        it = w["items"][str(d)]; n = len(it["sig"])
        for i in range(n):
            ok = [it["ws_pred"][str(t)][i] == it["v"][i][t - 1] for t in range(1, d + 1)]
            for t in range(1, d):
                if not ok[t - 1]:
                    if all(ok[:t - 1]):
                        r_ = rec.setdefault(str(t), [0, 0]); r_[1] += 1; r_[0] += ok[t]; tot[1] += 1; tot[0] += ok[t]
                    break
    recov = tot[0] / tot[1] if tot[1] else None
    o["a"] = {"recovery": recov, "n_first_errors": tot[1], "by_t": {t: (v[0] / v[1], v[1]) for t, v in rec.items()},
              "reading": None if recov is None else "SERIAL" if recov <= 0.10 else "RECOMPUTE" if recov >= 0.50 else "MIXED"}
    f = json.load(open(os.path.join(R, run, "eval_stab_fresh.json"))); b = {}
    for d in range(1, 8):
        if str(d) not in f:
            continue
        P, Y = f[str(d)]["items"]["pred"], f[str(d)]["items"]["ans"]; ks = [C.halt_items(p, 3)[0] for p in P]; n = len(P)
        b[str(d)] = {"p_khalt_dm1": sum(k == d - 1 for k in ks) / n, "acc_halt": sum(p[k] == y for p, k, y in zip(P, ks, Y)) / n,
                     "acc_dm1": sum(p[d - 1] == y for p, y in zip(P, Y)) / n, "khalt_max": max(ks), "khalt_mean": sum(ks) / n}
    static = all(b[str(d)]["p_khalt_dm1"] >= 0.95 and abs(b[str(d)]["acc_halt"] - b[str(d)]["acc_dm1"]) <= 0.01 for d in range(2, 8) if str(d) in b)
    o["b"] = {"by_d": b, "reading": "STATIC-CLOCK" if static else "ADAPTIVE"}
    a = json.load(open(os.path.join(R, run, "eval_acc.json")))["by_d"]
    o["c"] = {d: {k: a[d]["acc_kprime"].get(k) for k in ("8", "9", "10", "11", "12")} for d in ("9", "10") if d in a}
    out[run] = o
c10 = [max(out[r]["c"]["10"][k] for k in ("9", "10", "11", "12")) - out[r]["c"]["10"]["8"] for r in out]
out["c_reading"] = "MORE-THINKING" if all(x >= 0.05 for x in c10) else "FLAT" if all(x <= 0.02 for x in c10) else "MIXED"
os.makedirs("D:/Latent/theory/results", exist_ok=True)
json.dump(out, open("D:/Latent/theory/results/read_s269_redteam.json", "w", encoding="utf-8", newline=chr(10)), indent=1)
print(json.dumps({r: {"a": out[r]["a"]["reading"], "b": out[r]["b"]["reading"]} for r in ("bun_s1", "bun_s2")}, indent=1), out["c_reading"])
