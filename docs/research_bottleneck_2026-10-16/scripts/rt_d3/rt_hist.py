"""Red-team D3 Q2c / Q6: WHERE does N2 use its history? The registered BANK-ABLATE (versions 1..t-2 masked, V_0 + newest kept) masks the
core, the coda AND the PA gather together. Split it (eval-only, CPU, 3 threads, idle; nothing written to the repo):
  intact | ablate_all (registered, reproduces eval_vb) | ablate_gather (gather only) | ablate_bank (core + coda only)
  | gather_slotlocal (the PA gather restricted to keys at the query's own slot, every version: an SLVA-shaped pass input inside N2)
§170 test items (N.test_set seed), n in the args, K' = n + 2, answer at the coda after pass K'.
Usage: python rt_hist.py <run> <out.json> <per_n> <ns comma>"""
import json, sys, types
sys.path.insert(0, "D:/Latent")
try:
    import psutil; psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
import numpy as np
import torch
torch.set_num_threads(3)
import boxes.nested170 as N
import boxes.vbank as VB
import boxes.vbank_eval as VE


def keep_mid_off(t, nver):
    s = torch.arange(nver); return (s == 0) | (s >= t - 1)


@torch.no_grad()
def main():
    run, fo, per_n = sys.argv[1], sys.argv[2], int(sys.argv[3]); ns = [int(v) for v in sys.argv[4].split(",")]
    m, a = VE.load(run, "cpu"); assert m.mode == "pa"
    orig_gather = m.gather
    TE = N.test_set(1000); NUM = torch.tensor(N.NUM_IDS)
    out = {}
    for arm in ("intact", "ablate_all", "ablate_gather", "ablate_bank"):
        m.ver_keep = keep_mid_off if arm in ("ablate_all", "ablate_bank") else None
        if arm == "ablate_bank":
            m.gather = lambda t, gk, NV, pad, off_v0=False, keep=None: orig_gather(t, gk, NV, pad, off_v0, None)
        elif arm == "ablate_gather":
            m.gather = lambda t, gk, NV, pad, off_v0=False, keep=None: orig_gather(t, gk, NV, pad, off_v0, keep_mid_off(t, t))
        elif arm == "gather_slotlocal":
            def g_sl(t, gk, NV, pad, off_v0=False, keep=None):
                # same logits as the PA gather, but keys at other slots masked (-inf): U[j] = sum_s w_{j,s} N(V_s)[j]
                B, T = pad.shape; pad2 = pad.clone()
                outs = []
                for j in range(T):
                    pj = torch.zeros_like(pad); pj[:, j] = pad[:, j]
                    pj[:, j] = True
                    Uj = orig_gather(t, gk, NV, pj, off_v0, keep)[:, j]
                    outs.append(Uj)
                return torch.stack(outs, 1)
            m.gather = g_sl
        else:
            m.gather = orig_gather
        res = {}
        for n in ns:
            its = [it for it in TE if it["n"] == n][:per_n]; correct = 0
            for c0 in range(0, len(its), 32):
                part = its[c0:c0 + 32]; x, pad, lens = VB.batch_x(part, "cpu"); K = n + 2
                r_ = m.rollout(x, pad, K, coda_rows={K: torch.arange(len(part))})
                pred = m.answer_logits(r_["coda"][K][1], lens)[:, NUM].argmax(-1)
                correct += int((pred == torch.tensor([it["t"] for it in part])).sum())
            res[str(n)] = round(correct / len(its), 4); print(arm, n, res[str(n)], flush=True)
        out[arm] = res
    m.gather = orig_gather; m.ver_keep = None
    json.dump(out, open(fo, "w", newline="\n"), indent=1); print("DONE", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
