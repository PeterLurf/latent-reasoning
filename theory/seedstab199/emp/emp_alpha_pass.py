"""alpha_pass (Theory pre-registered rank test, 2026-10-13): per model, items n = 9..12 (NPER each, own generator seed), clean trajectory of K = 12 passes;
per item / delta head / pass: sum over real think tokens of log alpha_t, and the open-loop one-pass map P = prod_t alpha_t (I - beta_t k_t k_t^T) spectrum.
alpha_pass(model, n) = exp(mean over items of that n, the 6 delta heads and passes t = 1..n of sum_t log alpha_t). READ-ONLY on the repo; CPU fp32.
Usage: python emp_alpha_pass.py <run> [NPER]"""
import os
import sys
import json
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception as e_:  # noqa: BLE001
    print("psutil nice failed", e_)
sys.path.insert(0, r"D:\Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(3)
torch.set_grad_enabled(False)
import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1]
NPER = int(sys.argv[2]) if len(sys.argv) > 2 else 64
NS = (9, 10, 11, 12)
KMAX = 12
RD = os.path.join(r"D:\Latent\boxes\runs_fsq", RUN)
DEV = "cpu"; DH = DM.DH; BS = 128
REC = {"on": False, "data": None}
_orig_write_seq = DM.write_seq


def write_seq_rec(mem, h6, hw, S, mask=None):
    Snew = _orig_write_seq(mem, h6, hw, S, mask)
    if REC["on"]:
        H_ADD, H_DEL = mem.h_add, mem.h_del
        B, T, D = h6.shape; a6 = mem.ln(h6.float()); aw = mem.ln_w(hw.float()); gm = mem.gamma
        u = (1 - gm) * a6 + gm * aw; _, k, v = mem.qkv(u).split(D, -1); k, v = (t_.view(B, T, H_ADD + H_DEL, DH) for t_ in (k, v))
        g_ = mem.gate(u)
        REC["data"] = {"al": torch.sigmoid(g_[..., :H_DEL]), "be": 2 * torch.sigmoid(g_[..., H_DEL:]), "kd": F.normalize(k[:, :, H_ADD:], dim=-1),
                       "vd": v[:, :, H_ADD:], "mask": mask}
    return Snew


DM.write_seq = write_seq_rec


def per_item(d, Sd_prev, Sd_new):
    al, be, kd, vd, msk = (d["al"].double(), d["be"].double(), d["kd"].double(), d["vd"].double(), d["mask"].bool())
    B, T, H, _ = kd.shape; I = torch.eye(DH, dtype=torch.float64); P = I.expand(B, H, DH, DH).clone(); Sr = Sd_prev.double().clone()
    for t in range(T):
        k_ = kd[:, t]; a_ = al[:, t][..., None, None]; b_ = be[:, t][..., None, None]; m_ = msk[:, t][:, None, None, None]
        Pk = torch.einsum("bhij,bhj->bhi", P, k_); P = torch.where(m_, a_ * (P - b_ * Pk[..., :, None] * k_[..., None, :]), P)
        Sk = torch.einsum("bhvk,bhk->bhv", Sr, k_); Sr = torch.where(m_, a_ * (Sr - b_ * Sk[..., :, None] * k_[..., None, :]) + b_ * vd[:, t][..., :, None] * k_[..., None, :], Sr)
    err = float(((Sr - Sd_new.double()).flatten(1).norm(dim=1) / Sd_new.double().flatten(1).norm(dim=1).clamp_min(1e-30)).max())
    sv = torch.linalg.svdvals(P); mf = msk.double()[..., None]
    sla = (torch.log(al) * mf).sum(1)                                       # (B, H) sum_t log alpha_t over real tokens
    mlsv = ((torch.log(al) + torch.log(torch.abs(1 - be).clamp_min(1e-300)) / DH) * mf).sum(1)   # log|det P| / 64
    return {"sla": sla.numpy(), "mlsv": mlsv.numpy(), "mlsv_svd": torch.log(sv.clamp_min(1e-300)).mean(-1).numpy(), "top": sv[..., 0].numpy(),
            "nlt05": (sv < 0.5).sum(-1).double().numpy(), "alpha_tokmean": ((al * mf).sum(1) / mf.sum(1)).numpy(), "L": mf[..., 0].sum(1).numpy()}, err


def main():
    t0 = time.time(); a = json.load(open(os.path.join(RD, "args.json"))); DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0
    g = N.make_trunk({}, DEV); ML, HA, HD = N.mem_geometry(a)
    m = N.InjectN(g, ML, a.get("mem_write") or "top", HA, HD, a.get("add_norm") or None).to(DEV)
    sd = torch.load(os.path.join(RD, "model.pt"), map_location=DEV, weights_only=True); lres = str(m.load_state_dict(sd, strict=True)); del sd; m.eval()
    rng = np.random.default_rng(19615); items = []
    for n in NS:
        for _ in range(NPER):
            r, t_ = N.generate(rng, n); items.append({"n": n, "syms": N.render(r), "t": t_, "root": r})
    ns = np.array([it["n"] for it in items]); H_ADD, H_DEL = m.mem.h_add, m.mem.h_del
    acc = {k: np.zeros((len(items), KMAX, H_DEL)) for k in ("sla", "mlsv", "mlsv_svd", "top", "nlt05", "alpha_tokmean")}; L = np.zeros(len(items)); errs = []
    for i0 in range(0, len(items), BS):
        ch = items[i0:i0 + BS]; x, msk = m.think_inputs(ch, DEV); S = m.mem.zero_state(len(ch), DEV)
        for t in range(1, KMAX + 1):
            REC["on"] = True
            try:
                Sn = m.write(S, x, msk)
            finally:
                REC["on"] = False
            d = REC["data"]; REC["data"] = None; pi, e_ = per_item(d, S[:, H_ADD:H_ADD + H_DEL], Sn[:, H_ADD:H_ADD + H_DEL]); errs.append(e_)
            for k in acc:
                acc[k][i0:i0 + len(ch), t - 1] = pi[k]
            L[i0:i0 + len(ch)] = pi["L"]; S = Sn; del d
            print(RUN, "chunk", i0, "pass", t, "%.1fs" % (time.time() - t0), flush=True)
    res = {"run": RUN, "load": lres, "fade": float(m.mem.fade), "gamma": float(m.mem.gamma), "replay_rel_err_max": max(errs),
           "items": {"n": list(NS), "per_n": NPER, "seed": 19615, "generator": "boxes.nested170.generate"}, "by_n": {}, "timing_s": None,
           "defs": {"alpha_pass": "exp(mean over items of n, delta heads, passes t=1..n of sum_{real tokens} log alpha_t)",
                    "mean_log_sv_pass": "mean over items, heads, passes 1..n of log|det P_t|/64 (analytic: sum_t [log alpha_t + log|1-beta_t|/64])"}}
    for n in NS:
        s = ns == n; w = slice(0, n)
        sla = acc["sla"][s][:, w]; ml = acc["mlsv"][s][:, w]
        res["by_n"][str(n)] = {"n_items": int(s.sum()), "L_mean": float(L[s].mean()),
                               "alpha_pass": float(np.exp(sla.mean())), "alpha_pass_per_head": [float(v) for v in np.exp(sla.mean((0, 1)))],
                               "alpha_pass_min_head": float(np.exp(sla.mean((0, 1))).min()),
                               "alpha_pass_per_t": [float(v) for v in np.exp(acc["sla"][s].mean((0, 2)))],
                               "alpha_token_mean": float(acc["alpha_tokmean"][s][:, w].mean()),
                               "alpha_token_geomean": float(np.exp(sla.mean() / L[s].mean())),
                               "mean_log_sv_pass": float(ml.mean()), "mean_log_sv_pass_per_head": [float(v) for v in ml.mean((0, 1))],
                               "mean_log_sv_per_t": [float(v) for v in acc["mlsv"][s].mean((0, 2))],
                               "mean_log_sv_svd_pass": float(acc["mlsv_svd"][s][:, w].mean()),
                               "top_sv_pass": float(acc["top"][s][:, w].mean()), "n_sv_lt05_pass": float(acc["nlt05"][s][:, w].mean())}
    res["timing_s"] = time.time() - t0
    fn = os.path.join(OUT, "emp_alpha_pass_%s.json" % RUN); json.dump(res, open(fn, "w", newline="\n"), indent=1)
    print("WROTE", fn, "%.1fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
