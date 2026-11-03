"""THEORY Note 264-3 (c): pre-stated $0 reads on stored bun outputs (committed before computing; run only after Audit's ack; descriptive; red-team amendment 3).
(i) KHALT-DETERMINISM: bun_s1 / bun_s2 eval_stab_fresh.json items d 1-4 (pred[K'] for K' 0..14, ans): per d, the r = 3 halt k (score_c5c.halt_items) per item;
    report mean, variance and P(khalt == mode). Reading: DETERMINISTIC if P(khalt == mode) >= 0.98 at every d 1-4 (the halt is a function of d, i.e. of the plan's
    length, by construction on in-range items; then §264 ER adds only the out-of-plan case), else VARIABLE.
(ii) ONE-HOP DECODER: eval_acc by_d['4'].acc_kprime['3'] (d 4 answered after 3 passes): if >= 0.5 the decoder completes the last hop itself (the halt at d - 1 is then the
    decoder's competence, not the loop's), else the loop must reach pass d.
    python -B theory/read_s264_pre.py
"""
import json, os, sys
sys.path.insert(0, "D:/Latent/theory")
import score_c5c as C  # noqa: E402
R = "D:/Latent/boxes/runs_fsq"
out = {}
for run in ("bun_s1", "bun_s2"):
    f = json.load(open(os.path.join(R, run, "eval_stab_fresh.json"))); a = json.load(open(os.path.join(R, run, "eval_acc.json"))); o = {}
    for d in ("1", "2", "3", "4"):
        ks = [C.halt_items(p, 3)[0] for p in f[d]["items"]["pred"]]; n = len(ks); m = sum(ks) / n
        mode = max(set(ks), key=ks.count)
        o[d] = {"n": n, "mean": m, "var": sum((k - m) ** 2 for k in ks) / n, "mode": mode, "p_mode": ks.count(mode) / n}
    o["reading_i"] = "DETERMINISTIC" if all(o[d]["p_mode"] >= 0.98 for d in ("1", "2", "3", "4")) else "VARIABLE"
    o["d4_kprime3"] = a["by_d"]["4"]["acc_kprime"]["3"]; o["reading_ii"] = "DECODER-LAST-HOP" if o["d4_kprime3"] >= 0.5 else "LOOP-TO-d"
    out[run] = o
os.makedirs("D:/Latent/theory/results", exist_ok=True)
json.dump(out, open("D:/Latent/theory/results/read_s264_pre.json", "w", encoding="utf-8", newline=chr(10)), indent=1)
print(json.dumps(out, indent=1))
