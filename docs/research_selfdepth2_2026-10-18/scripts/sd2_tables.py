"""SD2 per-n prediction tables from acc_n2 directly (no K' interpolation): H_shift a2(16+d) = a1(12+d); H_prop a2(16+d) = a1(12 + 0.75 d) (linear interp;
a1 beyond 16 = sd2_numbers' assumed extension); SHRINK = a0(8+d); H_full = the stated floor. SE at 1,000 items per n. Writes sd2_tables.json."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = "D:/Latent/boxes/runs_fsq/"
a0 = {int(k): v for k, v in json.load(open(R + "s213_d3t_s1/eval_vb.json"))["acc_n2"].items()}
a1 = {int(k): v for k, v in json.load(open(R + "s215_ftself_s1/eval_vb.json"))["acc_n2"].items()}
ext = {int(k): v for k, v in json.load(open(os.path.join(HERE, "sd2_numbers.json")))["a1_extension_assumed"].items()}
A1 = {**a1, **{n: v for n, v in ext.items() if n > 16}, **{n: 0.11 for n in range(21, 30)}}
A0 = {**a0, **{n: 0.11 for n in range(17, 30)}}
ip = lambda a, x: a[int(math.floor(x))] if x == int(x) else (1 - (x - math.floor(x))) * a[int(math.floor(x))] + (x - math.floor(x)) * a[int(math.floor(x)) + 1]  # noqa: E731
full = dict(zip(range(17, 25), [0.97, 0.95, 0.93, 0.90, 0.88, 0.85, 0.82, 0.80]))
T = {}
for n in range(17, 25):
    d = n - 16
    T[n] = {"H_shift": A1[12 + d], "H_prop": ip(A1, 12 + 0.75 * d), "SHRINK": A0[8 + d], "H_full_floor": full[n]}
se = lambda p: math.sqrt(p * (1 - p) / 1000)  # noqa: E731
M = {h: sum(T[n][h] for n in range(17, 21)) / 4 for h in T[17]}; Mf = {h: sum(T[n][h] for n in range(21, 25)) / 4 for h in T[17]}
seM = {h: math.sqrt(sum(se(T[n][h]) ** 2 for n in range(17, 21))) / 4 for h in T[17]}
out = {"table": {n: {h: round(v, 4) for h, v in r.items()} for n, r in T.items()}, "se": {n: {h: round(se(v), 4) for h, v in r.items()} for n, r in T.items()},
       "M2": M, "M_far": Mf, "se_M2": seM, "MG_bar_midpoint": (M["H_shift"] + M["H_prop"]) / 2,
       "gap_SE": {h: {"to_0.80": (0.80 - M[h]) / seM[h], "to_MG_0.785": (0.785 - M[h]) / seM[h]} for h in M}}
json.dump(out, open(os.path.join(HERE, "sd2_tables.json"), "w", newline="\n"), indent=1)
for n in T: print(n, {h: round(v, 3) for h, v in T[n].items()}, "SE", {h: round(se(v), 4) for h, v in T[n].items()})
print("M2", {h: round(v, 4) for h, v in M.items()}, "M_far", {h: round(v, 4) for h, v in Mf.items()}, "seM", {h: round(v, 4) for h, v in seM.items()})
print("midpoint", out["MG_bar_midpoint"], out["gap_SE"])
