"""Note 199-M2 (Theory, bd749ffe; Audit's design; descriptive, $0): SAME-INIT LINEAR MODE CONNECTIVITY between two s185_rms seeds.
theta(lam) = (1 - lam) theta_A + lam theta_B for EVERY floating state_dict entry, then the buffers fade = gamma = 1; lam in {0, 0.1, .., 1}.
fp32, no autocast, one card for all points. At each lam:
  - TRAIN-DIST loss: 256 items from train_items(np.random.default_rng(19900), 256, 8, EXCL) at K 8;
  - TEST loss: the s182.path_indep items (test n 3..8, rng 18201 permutation[:500]);
    L = answer CE + 1.0 x aux CE, the training objective (nested170.loss_items at aux_lambda 1: aux = the mean over items of each item's token-mean
    aux CE over its passes t <= #steps, pass-token prompt; answer CE = the mean over items), each set's CE averaged over all its items (batched by sums);
  - the s182.path_indep fields (scratch/probe_pathindep.path_indep: n 3..8, 8-pass donor, K' 16) at every lam in PI_LAMS.
Sanity: lam 0 / 1 path_indep vs each model's eval.json s182 values (bf16 in-run -> fp32 drift).
    python scratch/lmc199.py <run A> <run B> [device, default cuda:2] [--pi_lams 0,0.2,...]   -> boxes/runs_fsq/s199_lmc/lmc_<A>_<B>.json"""
import json
import os
import sys
import time
from contextlib import nullcontext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.nested170 as N  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scratch"))
from probe_pathindep import path_indep  # noqa: E402

LAMS = [round(0.1 * i, 1) for i in range(11)]


@torch.no_grad()
def set_loss(m, items, K, dev, bs):
    """the training objective's two CEs over a fixed item set (fp32): answer CE summed over items, aux = per-item token-mean CE over its passes, item mean."""
    ans_sum, aux_items = 0.0, []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; x, msk = m.think_inputs(ch, dev); per = [[] for _ in ch]; S = m.mem.zero_state(len(ch), dev)
        for t in range(1, K + 1):
            S = m.write(S, x, msk); idx = [j for j, it in enumerate(ch) if len(N.aux_of(it)) >= t]
            if idx:
                lg_a, y_a = m.aux_logits(S[idx], [ch[j] for j in idx], t, dev)
                ce = Fn.cross_entropy(lg_a.float().transpose(1, 2), y_a, ignore_index=-100, reduction="none").sum(1) / (y_a != -100).sum(1)
                for r_, j in enumerate(idx):
                    per[j].append(float(ce[r_]))
        lg = m.dec_logits(S, ch, dev).float()
        ans_sum += float(Fn.cross_entropy(lg, torch.tensor(N.NUM_IDS[[it["t"] for it in ch]], device=dev), reduction="sum"))
        aux_items += [float(np.mean(p_)) for p_ in per if p_]
    ans, aux = ans_sum / len(items), float(np.mean(aux_items))
    return ans, aux, ans + 1.0 * aux


def main():
    ra, rb = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/"); dev = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else "cuda:2"
    pi_lams = [float(v) for v in sys.argv[sys.argv.index("--pi_lams") + 1].split(",")] if "--pi_lams" in sys.argv else LAMS
    a = json.load(open(os.path.join(ra, "args.json"))); b_ = json.load(open(os.path.join(rb, "args.json")))
    for k in ("arm", "think_k", "add_norm", "fast_add", "aux_nopass", "excl_min_n", "per_n", "mem_layer", "n_slots"):
        assert a.get(k) == b_.get(k), ("arg mismatch", k, a.get(k), b_.get(k))
    assert a["arm"] == "inject"
    DM.FAST_ADD = bool(a.get("fast_add")); K = int(a["think_k"]); bs = 64
    g = N.make_trunk(a, dev); ML, HA, HD = N.mem_geometry(a)
    m = N.InjectN(g, ML, "top", HA, HD, a.get("add_norm") or None).to(dev); m.nopass = bool(a.get("aux_nopass")); m.eval()
    sdA = torch.load(os.path.join(ra, "model.pt"), map_location="cpu", weights_only=False); sdB = torch.load(os.path.join(rb, "model.pt"), map_location="cpu", weights_only=False)
    assert set(sdA) == set(sdB) == set(m.state_dict()), "state_dict keys differ"
    TE = N.test_set(int(a.get("per_n") or 1000)); EXCL = {tuple(it["syms"]) for it in TE if it["n"] >= int(a.get("excl_min_n", 1))}
    TR = N.train_items(np.random.default_rng(19900), 256, 8, EXCL)
    pool = [it for it in TE if 3 <= it["n"] <= 8]; TS = [pool[i] for i in np.random.default_rng(18201).permutation(len(pool))[:500]]
    ac = nullcontext; out = {"A": os.path.basename(ra), "B": os.path.basename(rb), "lam": LAMS, "loss_train": [], "loss_test": [], "ans_ce_train": [], "aux_ce_train": [],
                             "ans_ce_test": [], "aux_ce_test": [], "path_indep": [], "path_indep_lams": pi_lams, "sec_per_lam": [],
                             "gpu": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "dtype": "float32 (no autocast)",
                             "def": "note 199-M2: theta(lam) = (1-lam) A + lam B on every floating state_dict entry, fade = gamma = 1; train 256 items rng 19900 (n <= 8, "
                                    "test-excluded) and test = s182.path_indep items (n 3..8 rng 18201 [:500]), K 8; L = answer CE + 1.0 aux CE (nested170.loss_items "
                                    "at aux_lambda 1); path_indep = probe_pathindep.path_indep(n 3..8, 8-pass donor, K' 16)"}
    fl = {k for k, v in sdA.items() if torch.is_floating_point(v)}; out["n_interp_tensors"] = len(fl); out["n_nonfloat_from_A"] = len(sdA) - len(fl)
    for k in sdA:
        if k not in fl:
            assert torch.equal(sdA[k], sdB[k]), ("non-float entry differs", k)
    for lam in LAMS:
        t0 = time.time(); sd = {k: ((1 - lam) * sdA[k].double() + lam * sdB[k].double()).to(sdA[k].dtype) if k in fl else sdA[k] for k in sdA}
        m.load_state_dict(sd)
        with torch.no_grad():
            m.mem.fade.fill_(1.0); m.mem.gamma.fill_(1.0)
        at, xt, lt = set_loss(m, TR, K, dev, bs); a2, x2, l2 = set_loss(m, TS, K, dev, bs)
        for k_, v_ in (("ans_ce_train", at), ("aux_ce_train", xt), ("loss_train", lt), ("ans_ce_test", a2), ("aux_ce_test", x2), ("loss_test", l2)):
            out[k_].append(v_)
        pi = path_indep(m, TE, 3, 8, 8, 16, dev, ac, bs) if any(abs(lam - p_) < 1e-9 for p_ in pi_lams) else None
        out["path_indep"].append(pi); out["sec_per_lam"].append(round(time.time() - t0, 1))
        print("LMC lam %.1f train %.4f test %.4f%s (%.0f s)" % (lam, lt, l2, "" if pi is None else " delta_cos %.3f agree %.3f" % (pi["delta_cos_final"], pi["answer_agree"]),
                                                               time.time() - t0), flush=True)
        od = os.path.join(ROOT, "boxes", "runs_fsq", "s199_lmc"); os.makedirs(od, exist_ok=True)
        json.dump(out, open(os.path.join(od, "lmc_%s_%s.json" % (out["A"], out["B"])), "w", newline="\n"), indent=1)   # dumped every lam
    san = {}
    for nm, r, i in (("A", ra, 0), ("B", rb, -1)):
        e = json.load(open(os.path.join(r, "eval.json")))["s182"]["path_indep"]; p = out["path_indep"][i]
        san[nm] = {k: {"eval_json": e[k], "lmc": p[k]} for k in ("answer_agree", "delta_cos_final", "acc_from_zero", "acc_from_donor")} if p else None
    out["sanity_endpoints_vs_eval_json"] = san
    json.dump(out, open(os.path.join(ROOT, "boxes", "runs_fsq", "s199_lmc", "lmc_%s_%s.json" % (out["A"], out["B"])), "w", newline="\n"), indent=1)
    print("DONE", json.dumps(san)[:600], flush=True)


if __name__ == "__main__":
    main()
