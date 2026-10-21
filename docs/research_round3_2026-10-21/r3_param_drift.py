"""R3 draft $0 check 1 (CPU, read-only on committed model.pt files; writes only r3_param_drift.json beside this script).
Where did each self / plain fine-tune move the weights? Per parameter group, relative displacement ||theta_FT - theta_base|| / ||theta_base||
and the group's share of the total squared displacement. Also SLVA's readable addressing scalars per head (slva_beta0 = V_0 bias,
slva_gamma = age slope) and the core / coda bank's beta / beta0 / gam.
Pairs: FT-CTRL / FT-SELF vs D3-T s1; R2-s1 vs FT-SELF; R1-s2 vs D3-T s2. Usage: python -B r3_param_drift.py (torch threads 2)."""
import json
import os

import torch

torch.set_num_threads(2)
R = "D:/Latent/boxes/runs_fsq"
PAIRS = [("s215_ftctrl_s1", "s213_d3t_s1"), ("s215_ftself_s1", "s213_d3t_s1"), ("s216_r2_s1", "s215_ftself_s1"),
         ("s216_r1_s2", "s216_d3t_s2"), ("s216_r2_s1", "s213_d3t_s1"), ("s215_ftself_s1", "s215_ftctrl_s1")]
# the last pair is the self items' own contribution: same base, same plain batches / coins / seed (G-S3), only the self steps differ
CORE, CODA, PRE = range(6, 10), range(10, 12), range(0, 6)


def group(k):
    if k.startswith("slva_"):
        return "slva_addr (wq/wk/ln/beta0/gamma)"
    if k in ("beta", "beta0", "gam"):
        return "bank_bias (beta/beta0/gam)"
    if k == "gain":
        return "gain (N(V))"
    if k.startswith("rec"):
        return "rec_head"
    if k.startswith("g.p.h_"):
        L = int(k.split("_")[1]); sub = "attn" if "_attn_" in k else "mlp" if "_mlp_" in k else "ln"
        return ("core6-9_" if L in CORE else "coda10-11_" if L in CODA else "prelude0-5_") + sub
    return "embed/lnf/other"


def load(r):
    return torch.load(os.path.join(R, r, "model.pt"), map_location="cpu", weights_only=True)


cache, out = {}, {"def": __doc__, "pairs": {}, "addr": {}}
for ft, base in PAIRS:
    for r in (ft, base):
        if r not in cache:
            cache[r] = load(r)
    a, b = cache[ft], cache[base]; num, den = {}, {}
    for k in b:
        if k == "c_anneal":
            continue
        g = group(k); d = (a[k].double() - b[k].double()).pow(2).sum().item(); n0 = b[k].double().pow(2).sum().item()
        num[g] = num.get(g, 0.0) + d; den[g] = den.get(g, 0.0) + n0
    tot = sum(num.values())
    if base == "s215_ftctrl_s1":                                            # cosine of (SELF - D3-T) with (CTRL - D3-T) per group
        d0 = cache.setdefault("s213_d3t_s1", load("s213_d3t_s1")); dot, na, nb = {}, {}, {}
        for k in b:
            if k == "c_anneal":
                continue
            g = group(k); x = (a[k].double() - d0[k].double()).flatten(); y = (b[k].double() - d0[k].double()).flatten()
            dot[g] = dot.get(g, 0.0) + float(x @ y); na[g] = na.get(g, 0.0) + float(x @ x); nb[g] = nb.get(g, 0.0) + float(y @ y)
        out["cos_self_ctrl_displacement"] = {g: dot[g] / (na[g] * nb[g]) ** 0.5 for g in sorted(dot) if na[g] > 0 and nb[g] > 0}
    out["pairs"]["%s_vs_%s" % (ft, base)] = {g: {"rel": (num[g] / den[g]) ** 0.5 if den[g] else None, "share_sq": num[g] / tot} for g in sorted(num)}
for r, sd in cache.items():
    out["addr"][r] = {"slva_beta0": [round(float(v), 4) for v in sd["slva_beta0"]], "slva_gamma": [round(float(v), 5) for v in sd["slva_gamma"].flatten()],
                      "gain_mean": float(sd["gain"].mean()), "gain_std": float(sd["gain"].std())}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "r3_param_drift.json"), "w", newline="\n"), indent=1)
for p, d in out["pairs"].items():
    print("==", p)
    for g, v in sorted(d.items(), key=lambda kv: -kv[1]["share_sq"]):
        print("  %-36s rel %.3e  share %.4f" % (g, v["rel"], v["share_sq"]))
print("cos(SELF-D3T, CTRL-D3T) per group:", {g: round(v, 3) for g, v in out.get("cos_self_ctrl_displacement", {}).items()})
for r, d in out["addr"].items():
    print(r, "beta0", d["slva_beta0"][:12], "\n   gamma", d["slva_gamma"][:12], "gain", round(d["gain_mean"], 4), round(d["gain_std"], 4))
