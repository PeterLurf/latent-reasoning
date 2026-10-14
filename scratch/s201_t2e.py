"""§201 T2-e (Theory plan Part V, prediction T2-e): SOFT HALT from each item's own halt t_hat (the T2 PRIMARY rule, recomputed exactly from the T1 CLEAN
dumps with scratch/s201_t2.py's fit, asserted equal to the committed T2 primary) instead of the oracle f = n + 1:
  FA-on-signal (registered): the ADDITIVE heads (S_add, and den's z) reset to their t_hat value after every later write; the delta heads keep writing.
  FD-on-signal (DESCRIPTIVE, Theory after T1 / fb1d39ed): the DELTA heads reset to their t_hat value; the additive heads keep writing.
Compared with the HARD halt a(t_hat) (Prop 7). fp32, the exact delta loop, the same items as T1 (test_set, the first 1,000 per n, n 2..12 and 16),
passes to 32, the answer read at K' 16 / 24 / 32; a censored item (never fires) gets t_hat = 32 (= CLEAN at K' 32).
-> <run>/eval_s201_t2e.json {by_n: {n: {hard, fa_signal: {K': acc}, fd_signal: {K': acc}, n_items, mean_t_hat, censored}}} and, in <run>/eval_s201_halt.json,
   "soft_halt": {"FA_on_signal" | "FD_on_signal": {"acc_by_n_kprime": {K': {n: acc}}, "status"}, "hard_halt_acc_by_n", bs, card}
    python scratch/s201_t2e.py <run> [device cuda:2]"""
import json
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch")); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_grad_enabled(False)
import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402
import s201_t2 as T2  # noqa: E402

TMAX = 32; NS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16); BS = 250; ARMS = ("FA", "FD"); KP = (16, 24, 32)


def t_hat(run):
    """the T2 primary rule's halt per test item (the same fit as s201_t2.main), in the dump's item order (NS blocks of 1,000)."""
    Dtr = np.load(os.path.join(T2.DUMP, run + "_train.npz")); Dte = np.load(os.path.join(T2.DUMP, run + "_test.npz"))
    Xtr, _, ctr, _ = T2.feats(Dtr); Xte, _, cte, _ = T2.feats(Dte); ntr, nte = Dtr["n"], Dte["n"]; T = np.arange(TMAX + 1); rows = (slice(None), slice(1, None))
    head = T2.logistic(Xtr[rows].reshape(-1, Xtr.shape[-1]), (T[None, :] >= ntr[:, None])[rows].reshape(-1))
    pi = np.zeros((len(nte), TMAX + 1)); pi[:, 1:] = head(Xte[rows].reshape(-1, Xte.shape[-1])).reshape(len(nte), TMAX)
    p_hat = 1.0 / float(ntr.mean()); pre = T[None, :] < (ntr[:, None] - 1); pre[:, 0] = False; C_FA = float(1 - ctr[pre].mean())
    post = (T[None, :-1] >= ntr[:, None]) & (T[None, :-1] >= 1) & ctr[:, :-1]; c_d = float((~ctr[:, 1:][post]).mean()) + 0.002
    A = p_hat * C_FA / (p_hat * C_FA + c_d); th, cens = T2.halt_first(pi >= A)
    return th, cens, nte, cte, A


def main():
    run = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "cuda:2"
    th, cens, nte, cte, A = t_hat(run); rd = os.path.join(ROOT, "boxes", "runs_fsq", run)
    kind, m, a = H.load_model(rd, dev); DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0; m.eval(); mem = m.mem; HA, HD = mem.h_add, mem.h_del
    den = getattr(mem, "add_norm", None) == "den"; TE = N.test_set(int(a.get("per_n", 1000))); out = {}; off = 0
    hp = os.path.join(rd, "eval_s201_halt.json"); hj = json.load(open(hp)); prim = hj["primary"]["acc_by_n"]   # the recompute must reproduce T2's primary exactly
    for n in NS:
        its = [it for it in TE if it["n"] == n][:1000]; k = len(its); thn = th[off:off + k]; assert (nte[off:off + k] == n).all(); tgt = torch.tensor([it["t"] for it in its])
        hard = cte[off:off + k][np.arange(k), thn]
        assert abs(float(hard.mean()) - prim[str(n)]) < 1e-12, ("t_hat recompute != the committed T2 primary", n, float(hard.mean()), prim[str(n)])
        ans = {arm: {kp: [] for kp in KP} for arm in ARMS}
        for arm in ARMS:
            for i in range(0, k, BS):
                ch = its[i:i + BS]; x, msk = m.think_inputs(ch, dev); S = mem.zero_state(len(ch), dev); tb = torch.tensor(thn[i:i + BS], device=dev); Sf = S.clone()
                for t in range(1, TMAX + 1):
                    S = m.write(S, x, msk)
                    hit = tb == t                                           # capture the state at each item's own t_hat
                    if bool(hit.any()):
                        Sf[hit] = S[hit]
                    frz = tb < t                                            # past t_hat: FA resets the additive heads (+ den's z), FD the delta heads
                    if bool(frz.any()):
                        S = S.clone()
                        if arm == "FA":
                            S[frz, :HA] = Sf[frz, :HA]
                            if den:
                                S[frz, HA + HD:] = Sf[frz, HA + HD:]
                        else:
                            S[frz, HA:HA + HD] = Sf[frz, HA:HA + HD]
                    if t in ans[arm]:
                        ans[arm][t].append(m.dec_logits(S, ch, dev)[:, N.NUM_IDS].argmax(-1).cpu())
        out[str(n)] = {"n_items": k, "hard": float(hard.mean()), "mean_t_hat": float(thn.mean()), "censored": float(cens[off:off + k].mean()),
                       **{arm.lower() + "_signal": {str(kp): float((torch.cat(v) == tgt).float().mean()) for kp, v in ans[arm].items()} for arm in ARMS}}
        print(run, "n", n, out[str(n)], flush=True); off += k
    card = torch.cuda.get_device_name(dev)
    res = {"by_n": out, "A": A, "bs": BS, "card": card, "dtype": "float32 (no autocast)",
           "def": "§201 T2-e: FA (registered) and FD (descriptive) from each item's T2-primary t_hat (that block reset to its t_hat value after every later write); hard = a(t_hat)"}
    json.dump(res, open(os.path.join(rd, "eval_s201_t2e.json"), "w", newline="\n"), indent=1)
    hj["soft_halt"] = {"FA_on_signal": {"acc_by_n_kprime": {str(kp): {n_: out[n_]["fa_signal"][str(kp)] for n_ in out} for kp in KP}, "status": "registered (T2-e)"},
                       "FD_on_signal": {"acc_by_n_kprime": {str(kp): {n_: out[n_]["fd_signal"][str(kp)] for n_ in out} for kp in KP}, "status": "DESCRIPTIVE (motivated after T1)"},
                       "hard_halt_acc_by_n": {n_: out[n_]["hard"] for n_ in out}, "bs": BS, "card": card, "def": res["def"]}
    json.dump(hj, open(hp, "w", newline="\n"), indent=1); print(run, "T2-e done", flush=True)


if __name__ == "__main__":
    main()
