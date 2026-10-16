"""$0 CPU: structure of the wpe drift D_j = wpe_ft[j] - wpe_pre[j] (rows 1..28): share explained by the mean direction, top-k PCs; norms; and the
consecutive-row cosine of the raw tables (pretrained vs fine-tuned)."""
import json, sys, torch
torch.set_num_threads(2)
pre = torch.load("D:/Latent/prosqa/ckpt/gpt2_pretrained.bin", map_location="cpu", weights_only=True); W0 = pre[[k for k in pre if k.endswith("wpe.weight")][0]].float()
out = {"pre_cos_consec": [round(float(x), 3) for x in torch.nn.functional.cosine_similarity(W0[1:60], W0[:59], dim=-1)]}
for c in ("s211_vb_carry_s1", "s211_vb_pa_s1", "s211_vb_pamct_s1"):
    sd = torch.load("D:/Latent/boxes/runs_fsq/%s/model.pt" % c, map_location="cpu", weights_only=True); W = sd[[k for k in sd if k.endswith("wpe_weight") or k.endswith("wpe.weight")][0]].float(); del sd
    D = (W - W0)[1:29]; mu = D.mean(0); tot = float((D ** 2).sum()); mean_share = float(28 * (mu ** 2).sum() / tot)
    U, S, Vh = torch.linalg.svd(D - mu, full_matrices=False); pc = [round(float(x), 3) for x in (S ** 2 / tot)[:5]]
    cos_mu = [round(float(x), 2) for x in torch.nn.functional.cosine_similarity(D, mu[None], dim=-1)]
    out[c] = {"drift_norm_rows_1_28": [round(float(x), 2) for x in D.norm(dim=-1)], "wpe_norm_pre_rows": [round(float(x), 2) for x in W0[1:29].norm(dim=-1)],
              "mean_dir_share": round(mean_share, 3), "pc_share_after_mean": pc, "cos_with_mean": cos_mu,
              "ft_cos_consec_20_40": [round(float(x), 3) for x in torch.nn.functional.cosine_similarity(W[21:41], W[20:40], dim=-1)]}
json.dump(out, open(sys.argv[1], "w"), indent=0)
print("pre cos consec 20-40", out["pre_cos_consec"][19:40])
for c in ("s211_vb_carry_s1", "s211_vb_pa_s1", "s211_vb_pamct_s1"):
    r = out[c]; print(c, "mean share", r["mean_dir_share"], "pcs", r["pc_share_after_mean"]); print("  cos_mu", r["cos_with_mean"]); print("  ft consec 20-40", r["ft_cos_consec_20_40"])
