"""$0 CPU: how far each wpe row moved in fine-tuning (N1 CARRY, N2 PA, N3 PA+MCT) against pretrained GPT-2; plus the gather query norm by slot (N2/N3)."""
import json, sys, torch
torch.set_num_threads(2)
pre = torch.load("D:/Latent/prosqa/ckpt/gpt2_pretrained.bin", map_location="cpu", weights_only=True)
k0 = [k for k in pre if k.endswith("wpe.weight")][0]; W0 = pre[k0].float()
out = {}
for c in ("s211_vb_carry_s1", "s211_vb_pa_s1", "s211_vb_pamct_s1"):
    sd = torch.load("D:/Latent/boxes/runs_fsq/%s/model.pt" % c, map_location="cpu", weights_only=True)
    k = [k for k in sd if k.endswith("wpe_weight") or k.endswith("wpe.weight")][0]; W = sd[k].float()
    d = (W[:64] - W0[:64]).norm(dim=-1) / W0[:64].norm(dim=-1)
    r = {"rel_drift_by_row": [round(float(x), 4) for x in d]}
    if "gq_w" in sd:
        q = W[:64] @ sd["gq_w"].float() + sd["gq_b"].float(); q0 = W0[:64] @ sd["gq_w"].float() + sd["gq_b"].float()
        r["gq_norm_by_row"] = [round(float(x), 2) for x in q.norm(dim=-1)]
        # cosine between consecutive queries (relative structure) and the query at j vs j-1 shift pattern
        cs = torch.nn.functional.cosine_similarity(q[1:], q[:-1], dim=-1); r["gq_cos_consec"] = [round(float(x), 3) for x in cs]
    out[c] = r; del sd
json.dump(out, open(sys.argv[1], "w"), indent=0)
for c, r in out.items():
    dr = r["rel_drift_by_row"]; print(c, "drift rows 1-4", dr[1:5], "10-14", dr[10:15], "24-28", dr[24:29], "29-33", dr[29:34], "34-40", dr[34:41])
    if "gq_norm_by_row" in r:
        print("  gq norm 20-28", r["gq_norm_by_row"][20:29], "29-40", r["gq_norm_by_row"][29:41]); print("  cos consec 20-28", r["gq_cos_consec"][20:28], "28-40", r["gq_cos_consec"][28:40])
