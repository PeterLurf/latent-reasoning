"""§250 Note 250-5 $0 PROBE (pre-stated; eval-only): is v_1 linearly decodable layer by layer, and does the decodability transfer to unseen frames?
-> scratch/s250_v1probe/eval_v1probe.json {model: {layer_chosen, val_acc, test_0_5, unseen {frame: acc}, layer0 {test_0_5, unseen}, per_layer {...}}}.
  DIRECT = df_pcd4_s1 (CD-AN): its own dec (tok_aw cue all = the full problem + every cue + ' The answer is:'), the last real token's hidden at layers 0..12;
  LOOP   = lh_fh_s1: the pass-1 input (think_cues[0]: text + QUESTION(u, S_1) + ' The answer is:'), S = 0, the ':' hidden at layers 0..12 (read_fixed at mem.layer as pass_aw).
  Layer 0 = embedding + wpe; layer i = after block i - 1 (pre ln_f); blocks via DM._block (pass_aw's / gpt_lower's code). Path check: ln_f(layer 12) -> logits argmax == the
  model's own preds (IO.direct_logits / pass_aw colon_logits) on every item.
Items (d 2, DF law): frames 0-5 = OW.Stream(25100, excl = build_sets(2) excl, FRAME_TRAIN, DF) deduplicated by sig, 3,000 train / 1,000 val / 1,000 test (disjoint by sig);
unseen 6 = Stream(25006, frames (6,)) as part_frames, 12..15 = build_sets(2)['framesets'][f][2]; 1,000 each. Target v_1 = values[1] (11 classes).
Probe: multinomial logistic regression, standardised (train mean / std, std >= 1e-6), loss = mean CE + 1e-3 ||W||^2 (bias unpenalised), full-batch L-BFGS fp64 CPU (strong
Wolfe, 500 iterations), seed 25101; fit on frames 0-5 train only; the layer chosen by frames 0-5 val acc (ties -> the lower layer); test / unseen at that layer and at layer 0.
    python -B scratch/s250_v1probe.py [--device cuda:0 --bs 100]"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
from boxes import inject_aw as AW  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.ow_words as OW  # noqa: E402
from boxes.nested170 import NUM_IDS, pad  # noqa: E402

SEED_ITEMS, SEED_FIT, L2 = 25100, 25101, 1e-3


@torch.no_grad()
def layer_states(g, seqs, dev, bs, mem=None):
    """-> (H (N, L + 1, D) fp32 at each item's last real token, final logits (N, V) via ln_f). mem: the LOOP path (read_fixed(mem, h, S = 0) entering block mem.layer)."""
    D = g.w("wte.weight").shape[1]; hd = D // g.H; Hs, LG = [], []
    for i in range(0, len(seqs), bs):
        ch = seqs[i:i + bs]; ids_, _ = pad(ch); x = torch.tensor(ids_, device=dev); B, T = x.shape
        pos = torch.tensor([len(q) - 1 for q in ch], device=dev); ar = torch.arange(B, device=dev)
        h = g.embed(x) + DM.wpe_slice(g, T)[None]; st = [h[ar, pos]]; S = mem.zero_state(B, dev) if mem is not None else None
        for li in range(g.L):
            if mem is not None and li == mem.layer:
                h = DM.read_fixed(mem, h, S)
            h = DM._block(g, li, h, hd); st.append(h[ar, pos])
        Hs.append(torch.stack(st, 1).float().cpu()); hf = Fn.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias")); LG.append(g.logits(hf[ar, pos]).float().cpu())
    return torch.cat(Hs).numpy(), torch.cat(LG)


def fit(X, y):
    torch.manual_seed(SEED_FIT); Xt = torch.tensor(X, dtype=torch.float64); yt = torch.tensor(y)
    mu = Xt.mean(0); sd = Xt.std(0).clamp_min(1e-6); Z = (Xt - mu) / sd
    W = torch.zeros(Z.shape[1], 11, dtype=torch.float64, requires_grad=True); b = torch.zeros(11, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-14, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); lo = Fn.cross_entropy(Z @ W + b, yt) + L2 * (W * W).sum(); lo.backward(); return lo
    with torch.enable_grad():
        opt.step(closure)
    return mu, sd, W.detach(), b.detach()


def acc(m_, X, y):
    mu, sd, W, b = m_; Z = (torch.tensor(X, dtype=torch.float64) - mu) / sd; return float(((Z @ W + b).argmax(-1).numpy() == y).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--bs", type=int, default=100); a_ = ap.parse_args(); dev = a_.device
    s = IO.build_sets(mul_s_p=2); st = OW.Stream(SEED_ITEMS, s["excl"], frames=OW.FRAME_TRAIN, mul_s_p=2, **IO.OWKW); seen = set(); base = []
    while len(base) < 5000:
        it = st.draw(2)
        if it["sig"] not in seen:
            seen.add(it["sig"]); base.append(it)
    splits = {"train": base[:3000], "val": base[3000:4000], "test": base[4000:5000]}
    sf = OW.Stream(IO.SEED_FRAMESET + 6, frames=(6,), mul_s_p=2, **IO.OWKW); unseen = {"6": [sf.draw(2) for _ in range(IO.TEST_N)]}
    for f in (12, 13, 14, 15):
        unseen[str(f)] = s["framesets"][f][2]
    sets_ = dict(splits, **{"u" + k: v for k, v in unseen.items()}); out = {"def": __doc__.split("\n")[0], "items_sha": {k: OW.set_sha(v) for k, v in sets_.items()}, "bs": a_.bs}
    for name, run in (("DIRECT", "df_pcd4_s1"), ("LOOP", "lh_fh_s1")):
        a, m, g = E.load(os.path.join(ROOT, "boxes/runs_fsq", run), dev); H = {}; pc = {}
        for k, its in sets_.items():
            T = [AW.tok_aw(it, a.get("aux_target", "value"), a.get("cue", "none"), a.get("write_src", "all")) for it in its]
            if m is None:
                seqs = [t["dec"] for t in T]; H[k], lg = layer_states(g, seqs, dev, a_.bs); own = E.answers_by_kprime(None, g, T, 0, dev, a_.bs)[:, 0]
                pc[k] = bool((E.pred_of(lg) == own).all())
            else:
                seqs = [t["think_cues"][0] for t in T]; H[k], _ = layer_states(m.g, seqs, dev, a_.bs, mem=m.mem); num = torch.tensor(NUM_IDS, device=dev); own = []
                with torch.no_grad():
                    for i in range(0, len(seqs), a_.bs):
                        ids_, mk = pad(seqs[i:i + a_.bs]); x = torch.tensor(ids_, device=dev); msk = torch.tensor(mk, device=dev)
                        _, h, pos = m.pass_aw(m.mem.zero_state(len(ids_), dev), x, msk, m.lower(x)); own += m.colon_logits(h, pos)[:, num].argmax(-1).tolist()
                Dm = m.g.w("wte.weight").shape[1]; hf = torch.tensor(H[k][:, -1]).to(dev)
                lg = m.g.logits(Fn.layer_norm(hf, (Dm,), m.g.w("ln_f.weight"), m.g.w("ln_f.bias"))).float()
                pc[k] = bool((lg[:, NUM_IDS].argmax(-1).cpu().numpy() == np.array(own)).all())
        y = {k: np.array([int(it["values"][1]) for it in v]) for k, v in sets_.items()}; per = {}
        for li in range(H["train"].shape[1]):
            mdl = fit(H["train"][:, li], y["train"])
            per[str(li)] = {"train": acc(mdl, H["train"][:, li], y["train"]), "val": acc(mdl, H["val"][:, li], y["val"]), "test_0_5": acc(mdl, H["test"][:, li], y["test"]),
                            "unseen": {k: acc(mdl, H["u" + k][:, li], y["u" + k]) for k in unseen}}
        best = max(range(len(per)), key=lambda i: (per[str(i)]["val"], -i))
        out[name] = {"run": run, "layer_chosen": best, "val_acc": per[str(best)]["val"], "test_0_5": per[str(best)]["test_0_5"], "unseen": per[str(best)]["unseen"],
                     "layer0": {"test_0_5": per["0"]["test_0_5"], "unseen": per["0"]["unseen"]}, "per_layer": per, "path_check_all_items": pc}
        del m, g; torch.cuda.empty_cache()
    out["majority_test_0_5"] = float(np.bincount(np.array([int(it["values"][1]) for it in splits["test"]]), minlength=11).max() / 1000)
    os.makedirs(os.path.join(ROOT, "scratch/s250_v1probe"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "scratch/s250_v1probe/eval_v1probe.json"), "w", newline="\n"), indent=1)
    print("V1PROBE done; path checks", {n: all(out[n]["path_check_all_items"].values()) for n in ("DIRECT", "LOOP")})


if __name__ == "__main__":
    main()
