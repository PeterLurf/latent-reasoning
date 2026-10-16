"""Red-team D3 (b)/(c): independent wpe analysis. CPU, 2 threads. No repo writes.
- decay factor f = prod_t (1 - lr_t * wd) from the trainer's own schedule (fsq120.lr_at, 48k, lr 1e-4, wd 0.01, AdamW decoupled: p <- p - lr*wd*p before the Adam step)
- per row: ||W - W0|| / ||W0||; best scalar fit W ~ a W0 and its residual; cross-cell identity of rows >= 33
- adjacent-row 1 - cos (raw table) per row band; seam step ||W[j+1]-W[j]|| per j
"""
import json, math, sys
import torch
torch.set_num_threads(2)
sys.path.insert(0, "D:/Latent")
try:
    import psutil; psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
from boxes.fsq120 import lr_at

f = 1.0
for s in range(1, 48001):
    f *= 1.0 - lr_at(1e-4, {}, s, 48000) * 0.01
out = {"decay_factor": f, "one_minus_f": 1 - f}
pre = torch.load("D:/Latent/prosqa/ckpt/gpt2_pretrained.bin", map_location="cpu", weights_only=True)
W0 = pre["wpe.weight"].float() if "wpe.weight" in pre else pre[[k for k in pre if k.endswith("wpe.weight")][0]].float()
cells = {"N1": "s211_vb_carry_s1", "N2": "s211_vb_pa_s1", "N3": "s211_vb_pamct_s1"}
Ws = {}
for c, d in cells.items():
    sd = torch.load("D:/Latent/boxes/runs_fsq/%s/model.pt" % d, map_location="cpu", weights_only=True)
    k = [k for k in sd if "wpe" in k]
    Ws[c] = sd[k[0]].float(); out.setdefault("keys", {})[c] = k
    if "gq_w" in sd:
        Ws[c + "_gq"] = (sd["gq_w"].float(), sd["gq_b"].float())
    del sd


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def band(x, lo, hi):
    v = x[lo:hi]; return [round(float(v.min()), 4), round(float(v.max()), 4), round(float(v.mean()), 4)]


pc = 1 - cos(W0[1:], W0[:-1])                       # pc[j] = 1 - cos(row j+1, row j)
out["pre_1mcos"] = {"j1-9": band(pc, 1, 9), "j10-19": band(pc, 10, 19), "j20-31": band(pc, 20, 31), "j33-60": band(pc, 33, 60)}
for c in cells:
    W = Ws[c]; r = {}
    rel = (W - W0).norm(dim=-1) / W0.norm(dim=-1)
    r["rel_drift"] = {"r1-9": band(rel, 1, 10), "r10-19": band(rel, 10, 20), "r20-28": band(rel, 20, 29), "r29-32": [round(float(x), 4) for x in rel[29:33]],
                      "r33-63": band(rel, 33, 64), "r64-1023": band(rel, 64, 1024), "r0": round(float(rel[0]), 4)}
    a = (W * W0).sum(-1) / (W0 * W0).sum(-1)                                   # best per-row scalar
    resid = (W - a[:, None] * W0).norm(dim=-1) / W0.norm(dim=-1)
    r["scalar_fit_rows33_1023"] = {"a_min": float(a[33:].min()), "a_max": float(a[33:].max()), "resid_max": float(resid[33:].max())}
    r["scalar_fit_rows29_32"] = {"a": [round(float(x), 5) for x in a[29:33]], "resid": [round(float(x), 5) for x in resid[29:33]]}
    r["resid_rows"] = {"r28-40": [round(float(x), 6) for x in resid[28:41]], "r41-1023_max": float(resid[41:].max()), "r41-1023_median": float(resid[41:].median())}
    r["a_rows"] = [round(float(x), 6) for x in a[28:41]]
    p = 1 - cos(W[1:], W[:-1])
    r["adj_1mcos"] = {"j1-9": band(p, 1, 9), "j10-19": band(p, 10, 19), "j20-31": band(p, 20, 32), "j32(32->33)": round(float(p[32]), 4), "j33-60": band(p, 33, 60)}
    r["adj_1mcos_minus_pre_mean"] = {k_: round(float((p - pc)[lo:hi].mean()), 4) for k_, (lo, hi) in {"j1-9": (1, 9), "j10-19": (10, 19), "j20-31": (20, 32)}.items()}
    step = (W[1:] - W[:-1]).norm(dim=-1); step0 = (W0[1:] - W0[:-1]).norm(dim=-1)
    r["step_norm"] = {"j20-31_median": round(float(step[20:32].median()), 3), "j32->33": round(float(step[32]), 3), "j33-60_median": round(float(step[33:60].median()), 3),
                      "pre_j20-31_median": round(float(step0[20:32].median()), 3)}
    r["drift_norm_rows1_28_mean"] = round(float((W - W0)[1:29].norm(dim=-1).mean()), 3)
    out[c] = r
out["rows33_identical_across_cells"] = {f"{a}-{b}": float((Ws[a][33:] - Ws[b][33:]).abs().max()) for a, b in (("N1", "N2"), ("N1", "N3"), ("N2", "N3"))}
out["ratio_drift_norm_N2_over_N1"] = round(out["N2"]["drift_norm_rows1_28_mean"] / out["N1"]["drift_norm_rows1_28_mean"], 2)
out["ratio_drift_norm_N3_over_N1"] = round(out["N3"]["drift_norm_rows1_28_mean"] / out["N1"]["drift_norm_rows1_28_mean"], 2)
for c in ("N2", "N3"):
    gw, gb = Ws[c + "_gq"]; W = Ws[c]
    q = W[:64] @ gw + gb
    cq = cos(q[1:], q[:-1])
    out[c]["gq_adj_cos"] = {"j10-19": band(cq, 10, 19), "j20-27": band(cq, 20, 28), "j28-32": [round(float(x), 3) for x in cq[28:33]], "j33-60": band(cq, 33, 60)}
    # all queries beyond the edge vs the in-range ones
    qn = torch.nn.functional.normalize(q, dim=-1)
    out[c]["gq_cos_q40_vs_q33"] = round(float(qn[40] @ qn[33]), 4); out[c]["gq_cos_q60_vs_q33"] = round(float(qn[60] @ qn[33]), 4)
    out[c]["gq_cos_q33_vs_q24"] = round(float(qn[33] @ qn[24]), 4)
    # per head
    qh = q.view(64, 12, 64); qhn = torch.nn.functional.normalize(qh, dim=-1)
    out[c]["gq_head_adj_cos_j33_60_min"] = round(float((qhn[34:61] * qhn[33:60]).sum(-1).min()), 4)
    out[c]["gq_head_adj_cos_j20_27_mean"] = round(float((qhn[21:28] * qhn[20:27]).sum(-1).mean()), 4)
json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
