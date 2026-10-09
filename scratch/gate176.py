"""THEORY §176 RAND-K pre-launch gates (Theory 2a1c3be5), boxes/inject169.py --rand_k / --prog:
  (1) histograms over 10k draws (inject169.sample_k): RK "7,0.5,32" (note 176-1: mean K = 8 for equal compute with fixed K8) mean K = 8 +- 0.5 with K in 1..32; PROG "8,8" k1 ~ U{0..8}, k2 ~ U{1..8} (every
      value's share within 15 % of uniform)
  (2) K independent of n: the training loop's draw sequence (c, the operand draw, then sample_k) over 100k batches, |corr(K, n)| < 0.02 (RK and PROG's k1 + k2);
      threshold set at >= 5 SE (Audit's ruling 2026-10-09: 0.02 at 10k draws was ~2 SE, a mis-set bound; K has its own RNG stream by construction)
  (3) gradflow (fade 0.5, W_o perturbed, gamma 0.5) through inject169.think_kk, the answer CE:
      RK (0, 5): the memory after EVERY pass 1..5 gets gradient; PROG (3, 2): passes 1..3 carry no graph (requires_grad False), passes 4..5 get gradient;
      every DeltaMem parameter gets gradient in both
  (4) sampling off: 3 training steps of inject169 with no §176 flag (and with --fast_add) give the same log losses and final weights as the committed
      HEAD's inject169 (a temporary copy boxes/_inject169_head_tmp.py, removed after) - bitwise (the same code path)
    python scratch/gate176.py   -> boxes/runs_fsq/s176_gates/gate176.json"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.inject169 as I  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fixedchunk import DT, product_digits, draw, cmax  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402


def main():
    torch.set_num_threads(3); out = {}; ok = True; RK, PG = [7.0, 0.5, 32.0], [8, 8]
    r_ = np.random.default_rng(1); ks = np.array([I.sample_k(r_, RK, None, "all", 8)[0][1] for _ in range(10000)])
    r_ = np.random.default_rng(2); pg = np.array([I.sample_k(r_, None, PG, "all", 8)[0] for _ in range(10000)])
    sh1 = np.bincount(pg[:, 0], minlength=9) / 10000 * 9; sh2 = np.bincount(pg[:, 1], minlength=9)[1:] / 10000 * 8
    out["g1"] = {"rk_mean": float(ks.mean()), "rk_min": int(ks.min()), "rk_max": int(ks.max()), "rk_share_at_cap": float((ks == 32).mean()),
                 "prog_k1_rel_share_range": [float(sh1.min()), float(sh1.max())], "prog_k2_rel_share_range": [float(sh2.min()), float(sh2.max())]}
    out["g1_pass"] = bool(abs(ks.mean() - 8) <= 0.5 and ks.min() >= 1 and ks.max() <= 32 and 0.85 <= sh1.min() and sh1.max() <= 1.15 and 0.85 <= sh2.min() and sh2.max() <= 1.15)
    ok &= out["g1_pass"]
    # (2) the training loop's order: c, then the operand draw (rng), then sample_k (rng_k)
    EXCL = set(); cr = {}
    for nm, rk_, pg_ in (("rk", RK, None), ("prog", None, PG)):
        rng, rng_k = np.random.default_rng(1), np.random.default_rng(17601); nn, kk = [], []
        for step in range(1, 100001):
            rng.integers(1, cmax(step) + 1); A, B = draw(rng, 4, [2, 3, 4, 5], EXCL); k_, _ = I.sample_k(rng_k, rk_, pg_, "all", 8); nn.append(B.shape[1]); kk.append(k_[0] + k_[1])
        cr[nm] = float(np.corrcoef(nn, kk)[0, 1])
    out["g2_corr_K_n"] = cr; out["g2_pass"] = all(abs(v) < 0.02 for v in cr.values()); ok &= out["g2_pass"]
    # (3)
    torch.manual_seed(0); g = s5.load_gpt2("cpu"); m = I.Inject(g, 6, "top")
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02)
    m.mem.fade.fill_(0.5); m.mem.gamma.fill_(0.5); m.mem.s_add.data.fill_(-7.0); m.train(); A, B = gen_operands(np.random.default_rng(3), 4, 4), gen_operands(np.random.default_rng(4), 4, 3)
    ans = torch.tensor(DT[product_digits(A, B)], device="cpu"); g3 = {}; w0 = I.Inject.write
    for nm, kk in (("rk_0_5", (0, 5)), ("prog_3_2", (3, 2))):
        kept = []

        def wr(self, S, ids):
            S2 = w0(self, S, ids)
            if S2.requires_grad:
                S2.retain_grad()
            kept.append(S2); return S2
        I.Inject.write = wr; m.zero_grad(set_to_none=True)
        try:
            S = I.think_kk(m, m.mem.zero_state(4, "cpu"), A, B, kk); lg = m.dec_logits(S, A, B, ans)
            Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]), ans.reshape(-1)).backward()
        finally:
            I.Inject.write = w0
        g3[nm] = {"pass_grad": [float(k_.grad.norm()) if (k_.requires_grad and k_.grad is not None) else 0.0 for k_ in kept],
                  "pass_requires_grad": [bool(k_.requires_grad) for k_ in kept], "mem_zero": [n_ for n_, p_ in m.mem.named_parameters() if p_.grad is None or float(p_.grad.norm()) == 0]}
    out["g3"] = g3; rk, pr = g3["rk_0_5"], g3["prog_3_2"]
    out["g3_pass"] = bool(min(rk["pass_grad"]) > 0 and not rk["mem_zero"] and pr["pass_requires_grad"] == [False] * 3 + [True] * 2 and min(pr["pass_grad"][3:]) > 0
                      and max(pr["pass_grad"][:3]) == 0 and not pr["mem_zero"]); ok &= out["g3_pass"]
    # (4)
    head = subprocess.run(["git", "show", "HEAD:boxes/inject169.py"], capture_output=True, text=True, encoding="utf-8").stdout
    assert "def sample_k" not in head, "commit the §176 inject169 only after this gate"
    tmp = os.path.join(ROOT, "boxes", "_inject169_head_tmp.py"); open(tmp, "w", encoding="utf-8", newline="\n").write(head); res = {}
    sp = os.environ.get("GATE176_OUT", os.path.join(ROOT, "boxes", "runs_fsq", "s176_gates"))
    try:
        common = ["--think_k", "2", "--seed", "1", "--steps", "3", "--sched_scale", "0.002", "--log_every", "1", "--bs", "4", "--test_n", "4", "--ext_n", "4",
                  "--probe_n_fit", "32", "--eval_bs", "4", "--device", "cpu"]
        for nm, mod, extra in (("head", "boxes._inject169_head_tmp", []), ("new", "boxes.inject169", []), ("head_fast", "boxes._inject169_head_tmp", ["--fast_add"]),
                               ("new_fast", "boxes.inject169", ["--fast_add"])):
            od = os.path.join(sp, "off_" + nm); r_ = subprocess.run([sys.executable, "-m", mod] + common + extra + ["--out", od], capture_output=True, text=True,
                                                                     env=dict(os.environ, OMP_NUM_THREADS="3"))
            assert r_.returncode == 0, r_.stderr[-2000:]
            res[nm] = ([json.loads(l)["loss"] for l in open(os.path.join(od, "log.jsonl")) if '"loss"' in l], torch.load(os.path.join(od, "model.pt"), weights_only=True))
    finally:
        os.remove(tmp)
    same = {a_: bool(res[a_][0] == res[b_][0] and all(torch.equal(res[a_][1][k], res[b_][1][k]) for k in res[b_][1])) for a_, b_ in (("new", "head"), ("new_fast", "head_fast"))}
    out["g4"] = {"losses_head": res["head"][0], "bitwise_equal": same}; out["g4_pass"] = all(same.values()); ok &= out["g4_pass"]
    out["all_pass"] = bool(ok); gd = os.path.join(ROOT, "boxes", "runs_fsq", "s176_gates"); os.makedirs(gd, exist_ok=True)
    json.dump(out, open(os.path.join(gd, "gate176.json"), "w", newline="\n"), indent=1); print("GATE176", json.dumps(out), flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
