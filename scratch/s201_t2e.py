"""§201 T2-e (Theory plan Part V, prediction T2-e; descriptive-to-registered as stated there): FA-ON-SIGNAL - freeze the ADDITIVE heads (S_add, and den's
z) from each item's own halt t_hat (the T2 PRIMARY rule, recomputed exactly from the T1 CLEAN dumps with scratch/s201_t2.py's fit) instead of the
oracle f = n + 1; the delta heads keep writing. Compared with the HARD halt a(t_hat) (Prop 7). fp32, the exact delta loop, the same items as T1
(test_set, the first 1,000 per n, n 2..12 and 16), passes to 32; a censored item (never fires) gets t_hat = 32 (= CLEAN at 32).
-> <run>/eval_s201_t2e.json {by_n: {n: {hard: acc a(t_hat), fa_signal: {"16"|"24"|"32": acc}, n_items, mean_t_hat, censored}}, bs, card, dtype}
    python scratch/s201_t2e.py <run> [device cuda:2] [--dump_dir D]"""
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

TMAX = 32; NS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16); BS = 250


def t_hat(run):
    """the T2 primary rule's halt per test item (same fit as s201_t2.main), in the dump's item order (NS blocks of 1,000)."""
    Dtr = np.load(os.path.join(T2.DUMP, run + "_train.npz")); Dte = np.load(os.path.join(T2.DUMP, run + "_test.npz"))
    Xtr, _, ctr, _ = T2.feats(Dtr); Xte, ate, cte, _ = T2.feats(Dte); ntr, nte = Dtr["n"], Dte["n"]; T = np.arange(TMAX + 1); rows = (slice(None), slice(1, None))
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
    prim = json.load(open(os.path.join(rd, "eval_s201_halt.json")))["primary"]["acc_by_n"]          # the recompute must reproduce T2's primary exactly
    for n in NS:
        its = [it for it in TE if it["n"] == n][:1000]; k = len(its); thn = th[off:off + k]; assert (nte[off:off + k] == n).all(); tgt = torch.tensor([it["t"] for it in its])
        ans = {kp: [] for kp in (16, 24, 32)}
        for i in range(0, k, BS):
            ch = its[i:i + BS]; x, msk = m.think_inputs(ch, dev); S = mem.zero_state(len(ch), dev); tb = torch.tensor(thn[i:i + BS], device=dev); Sf = S.clone()
            for t in range(1, TMAX + 1):
                S = m.write(S, x, msk)
                hit = (tb == t)                                             # capture S_add at each item's own t_hat
                if bool(hit.any()):
                    Sf[hit] = S[hit]
                frz = tb < t                                                # past t_hat: the additive heads (and den's z) reset to their t_hat value
                if bool(frz.any()):
                    S = S.clone(); S[frz, :HA] = Sf[frz, :HA]
                    if den:
                        S[frz, HA + HD:] = Sf[frz, HA + HD:]
                if t in ans:
                    ans[t].append(m.dec_logits(S, ch, dev)[:, N.NUM_IDS].argmax(-1).cpu())
        hard = cte[off:off + k][np.arange(k), thn]
        assert abs(float(hard.mean()) - prim[str(n)]) < 1e-12, ("t_hat recompute != the committed T2 primary", n, float(hard.mean()), prim[str(n)])
        out[str(n)] = {"n_items": k, "hard": float(hard.mean()), "fa_signal": {str(kp): float((torch.cat(v) == tgt).float().mean()) for kp, v in ans.items()},
                       "mean_t_hat": float(thn.mean()), "censored": float(cens[off:off + k].mean())}
        print(run, "n", n, out[str(n)], flush=True); off += k
    res = {"by_n": out, "A": A, "bs": BS, "card": torch.cuda.get_device_name(dev), "dtype": "float32 (no autocast)",
           "def": "§201 T2-e: FA from each item's T2-primary t_hat (S_add (+ den z) reset to S_add,t_hat after every later write; delta heads write); hard = a(t_hat) from the T1 CLEAN dump"}
    json.dump(res, open(os.path.join(rd, "eval_s201_t2e.json"), "w", newline="\n"), indent=1); print(run, "T2-e done", flush=True)


if __name__ == "__main__":
    main()
