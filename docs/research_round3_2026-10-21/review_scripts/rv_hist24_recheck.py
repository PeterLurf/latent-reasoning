"""REVIEW_R3 check 1 ($0, CPU, read-only on committed eval_hist24.json / eval_vb.json; writes rv_hist24_recheck.json beside this script).
(a) Recompute STAB_arm = mean_{n 3..6} [(acc_n2 - acc_k24)_arm - (acc_n2 - acc_k24)_intact] (score_s213.hist24's formula) for every arm.
(b) Masked acc_k24 (self_ablate, bank_ablate_slva) per n 3..8 and its dependence on idle passes (24 - n): OLS slope per model.
(c) eval_vb acc_k['24'] at n 1..8 and the K' keys present in eval_vb acc_k (is K' > 24 measured anywhere?).
Runs: D3-T s1, FT-CTRL, FT-SELF, R2-s1, D3-T s2, R1-s2. Usage: python -B rv_hist24_recheck.py"""
import json
import os

R = "D:/Latent/boxes/runs_fsq"
RUNS = ["s213_d3t_s1", "s215_ftctrl_s1", "s215_ftself_s1", "s216_r2_s1", "s216_d3t_s2", "s216_r1_s2"]
ARMS = ("bank_ablate_slva", "self_ablate", "other_ablate", "dup_newest", "dup_foreign")
out = {"def": __doc__, "runs": {}}


def ols(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


for r in RUNS:
    h = json.load(open(os.path.join(R, r, "eval_hist24.json")))
    v = json.load(open(os.path.join(R, r, "eval_vb.json")))
    g = lambda arm, n: h[arm]["acc_n2"][str(n)] - h[arm]["acc_k24"][str(n)]  # noqa: E731
    st = {a: round(sum(g(a, n) - g("intact", n) for n in range(3, 7)) / 4, 4) for a in ARMS if a in h}
    masked = {a: {n: h[a]["acc_k24"][str(n)] for n in range(3, 9)} for a in ("self_ablate", "bank_ablate_slva")}
    slope = {a: round(ols([24 - n for n in range(3, 9)], [masked[a][n] for n in range(3, 9)]), 4) for a in masked}
    ks = sorted(int(k) for k in v["acc_k"])
    out["runs"][r] = {"STAB": st, "masked_acc_k24": masked, "masked_mean_n3_6": {a: round(sum(masked[a][n] for n in range(3, 7)) / 4, 4) for a in masked},
                      "slope_per_idle_pass": slope, "intact_k24_n3_8": [h["intact"]["acc_k24"][str(n)] for n in range(3, 9)],
                      "vb_acc_k24_n1_8": [v["acc_k"]["24"][str(n)] for n in range(1, 9)], "vb_acc_n2_n9_16": [v["acc_n2"][str(n)] for n in range(9, 17)],
                      "vb_acc_k_keys": ks, "hist24_card": h.get("card"), "vb_card": v.get("card")}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_hist24_recheck.json"), "w", newline="\n"), indent=1)
for r, d in out["runs"].items():
    print("==", r, d["hist24_card"])
    print("  STAB", d["STAB"])
    print("  masked mean n3-6", d["masked_mean_n3_6"], " slope/idle pass", d["slope_per_idle_pass"])
    print("  vb a24 n1-8", d["vb_acc_k24_n1_8"], " acc_n2 n9-16", d["vb_acc_n2_n9_16"])
    print("  K' keys", d["vb_acc_k_keys"][:3], "...", d["vb_acc_k_keys"][-3:])
