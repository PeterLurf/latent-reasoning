import sys, json, collections
sys.path.insert(0, "D:/Latent")
import boxes.ow_words as OW, boxes.inject_ow as IO
s = OW.Stream(IO.SEED_TRANSFER_DF, frames=(6, 7), **IO.OWKW, mul_s_p=2)
sets = {d: [s.draw(d) for _ in range(IO.TEST_N)] for d in IO.TEST_DS}
tab = collections.defaultdict(lambda: [0, 0])     # (frame, op) -> [n, correct]
for run in ("lh_fh_s1", "lh_fh_s2"):
    e = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_transfer.json" % run))
    for d in ("1", "2", "3", "4"):
        it = e["items"][d]; regen = sets[int(d)]
        assert [x["sig"] for x in regen] == it["sig"], ("ABORT sig mismatch", run, d)
        wp = it["ws_pred"]; w1 = wp.get("1") if isinstance(wp, dict) else None
        for i, x in enumerate(regen):
            if d == "1":
                ok = it["pred"][i] == it["ans"][i]
            else:
                ok = w1[i] == x["values"][1]
            k = (x["frames"][0], x["ops"][0]); tab[k][0] += 1; tab[k][1] += ok
print("ws_pred keys", list(wp.keys())[:10] if isinstance(wp, dict) else type(wp))
for f in (6, 7):
    errs = {op: n - c for (ff, op), (n, c) in tab.items() if ff == f}; E = sum(errs.values())
    for (ff, op), (n, c) in sorted(tab.items()):
        if ff == f:
            print("frame", f, op, "n", n, "acc %.3f" % (c / n), "err share %.2f" % (errs[op] / E if E else 0))

print("--- what the failing templates compute (step 1; pooled seeds, d 1-4) ---")
M = 11
cand = {"add": lambda a, o: (a + o) % M, "sub": lambda a, o: (a - o) % M, "rsub": lambda a, o: (o - a) % M, "mul": lambda a, o: (a * o) % M, "copyA": lambda a, o: a % M, "copyO": lambda a, o: o % M}
hit = collections.defaultdict(lambda: collections.Counter()); nn = collections.Counter()
for run in ("lh_fh_s1", "lh_fh_s2"):
    e = json.load(open("D:/Latent/boxes/runs_fsq/%s/eval_transfer.json" % run))
    for d in ("1", "2", "3", "4"):
        it = e["items"][d]; regen = sets[int(d)]
        for i, x in enumerate(regen):
            k = (x["frames"][0], x["ops"][0])
            if k not in ((6, "add_c"), (6, "sub_v"), (6, "mul_v"), (7, "add_c"), (7, "sub_c")):
                continue
            p = it["pred"][i] if d == "1" else it["ws_pred"]["1"][i]
            a, o = x["values"][0], x["operands"][0]; nn[k] += 1
            for nm, fn in cand.items():
                hit[k][nm] += (p == fn(a, o))
for k in sorted(nn):
    print(k, "n", nn[k], {nm: round(hit[k][nm] / nn[k], 3) for nm in cand})
