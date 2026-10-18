"""SD2 draft check: T-split of FT-SELF (s215_ftself_s1) items_n2 at n 9-16 (eval_selfdepth.json items_n2 + T, K' n+2). Descriptive; read-only."""
import json, collections
R = "D:/Latent/boxes/runs_fsq/"
out = {}
for cell in ["s215_ftself_s1"]:
    e = json.load(open(R + cell + "/eval_selfdepth.json"))
    I, T = e["items_n2"], e["T"]
    o = {}
    for n in sorted(I, key=int):
        acc = collections.defaultdict(lambda: [0, 0])
        for y, t in zip(I[n], T[n]):
            acc[t][0] += y; acc[t][1] += 1
        o[n] = {"mean_T": sum(T[n]) / len(T[n]), "acc": sum(I[n]) / len(I[n]), "max_T": max(T[n]), "min_T": min(T[n]),
                "by_T": {t: [round(a / b, 3), b] for t, (a, b) in sorted(acc.items())}}
    out[cell] = o
# pooled n 13-16 by T band
pool = collections.defaultdict(lambda: [0, 0])
e = json.load(open(R + "s215_ftself_s1/eval_selfdepth.json"))
for n in ["13", "14", "15", "16"]:
    for y, t in zip(e["items_n2"][n], e["T"][n]):
        pool[(n, "le47" if t <= 47 else "ge49")][0] += y; pool[(n, "le47" if t <= 47 else "ge49")][1] += 1
out["band"] = {"%s_%s" % k: [round(a / b, 3), b] for k, (a, b) in sorted(pool.items())}
json.dump(out, open("C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research5/scripts/tsplit.json", "w", newline="\n"), indent=1)
for n, v in out["s215_ftself_s1"].items():
    print(n, round(v["mean_T"], 1), v["min_T"], v["max_T"], round(v["acc"], 3), {t: x for t, x in v["by_T"].items() if x[1] >= 20})
print(out["band"])
