"""Exploratory (Part VI): is end activity related to how much learning-rate mass was spent at full curriculum depth?"""
import json, math, os
import numpy as np

def rank(x):
    return np.argsort(np.argsort(x)).astype(float)
def spearmanr(x, y):
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])
def mannwhitneyu(p, n):
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n]))   # AUC

X = json.load(open("results/extrap_predictors.json"))["models"]
P = json.load(open("results/p5_endact.json"))
rows = {m["run"][5:]: m["end_activity"] for m in X}
rows.update({r[5:] if r.startswith("runs/") else r: v["end_activity"] for r, v in P.items()})
out = []
for r, ea in rows.items():
    c = json.load(open(f"runs/{r}/config.json")); L = [json.loads(l) for l in open(f"runs/{r}/log.jsonl")]
    adv = [d for d in L if "curriculum_advance_to" in d]
    full = next((d["step"] for d in adv if d["curriculum_advance_to"] is None), None)
    last = max(d["step"] for d in L if "step" in d)
    lr = lambda s: c["lr"] * min(1, (s + 1) / c["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / c["steps"])))
    integ = sum(lr(s) for s in range(full, c["steps"], 10)) * 10 if full is not None else 0.0
    lr_at_full = lr(full) if full is not None else 0.0
    out.append((r, ea, full, integ, lr_at_full, c["lr"], c.get("no_step_emb", False), c["K"]))
    print(f"{r:22s} EA {ea:6.3f} full@{str(full):>6} lr-int {integ:6.2f} lr@full {lr_at_full:.5f} lr {c['lr']}")
a = np.array([[o[1], o[3], o[4]] for o in out])
print("spearman EA vs lr-integral", spearmanr(a[:, 0], a[:, 1]))
print("spearman EA vs lr at full", spearmanr(a[:, 0], a[:, 2]))
hi = a[:, 0] >= 0.25
print("lr-int high-EA", np.median(a[hi, 1]), "low-EA", np.median(a[~hi, 1]), mannwhitneyu(a[hi, 1], a[~hi, 1]))
