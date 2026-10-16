"""Red-team D3: is N2's history use (all of it in the PA gather, rt_hist.json) SLOT-LOCAL (SLVA-compatible) or CROSS-SLOT?
Eval-only (CPU 3 threads, idle; no repo writes). The PA gather re-implemented from VBankN.gather (fp32; validated against it at the intact arm),
with mid versions s in 1..t-2 masked (a) only at OTHER slots i != j (ablate_cross: SLVA keeps exactly this history) or (b) only at the query's own
slot i == j (ablate_self). Core / coda untouched. §170 test items, K' = n + 2.
Usage: python rt_hist2.py <run> <out.json> <per_n> <ns>"""
import json, math, sys
sys.path.insert(0, "D:/Latent")
try:
    import psutil; psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
import torch
torch.set_num_threads(3)
import boxes.nested170 as N
import boxes.vbank as VB
import boxes.vbank_eval as VE
import boxes.deltamem as DM


def make_gather(m, mode):
    def g(t, gk, NV, pad, off_v0=False, keep=None):
        assert not off_v0 and keep is None
        B, T = pad.shape; H, hd = m.H, m.hd
        q = (DM.wpe_slice(m.g, T) @ m.gq_w + m.gq_b).view(T, H, hd).transpose(0, 1)
        K_ = torch.cat(gk[:t], 2); lg = torch.einsum("htd,bhsd->bhts", q, K_) / math.sqrt(hd)
        s_idx = torch.arange(t).repeat_interleave(T); i_idx = torch.arange(T).repeat(t)
        lg = lg + m.gamma_q[None, :, None, None] * (t - 1 - s_idx).float()[None, None, None, :]
        own = (s_idx[None, :] == t - 1) & (i_idx[None, :] == torch.arange(T)[:, None])
        lg = lg - m.c_anneal * (~own).float()[None, None]
        valid = pad[:, None, :].expand(B, t, T).reshape(B, t * T)[:, None, :].expand(B, T, t * T)        # (B, Tq, tT)
        mid = ((s_idx >= 1) & (s_idx <= t - 2))[None, :]
        same = i_idx[None, :] == torch.arange(T)[:, None]                                                  # (Tq, tT)
        if mode == "cross":
            valid = valid & ~(mid & ~same)[None]
        elif mode == "self":
            valid = valid & ~(mid & same)[None]
        w = torch.softmax(lg.masked_fill(~valid[:, None], float("-inf")), -1)
        U = (w @ torch.cat(NV[:t], 1).view(B, t * T, H, hd).transpose(1, 2)).transpose(1, 2).reshape(B, T, H * hd)
        return U
    return g


@torch.no_grad()
def main():
    run, fo, per_n = sys.argv[1], sys.argv[2], int(sys.argv[3]); ns = [int(v) for v in sys.argv[4].split(",")]
    m, a = VE.load(run, "cpu"); orig = m.gather; TE = N.test_set(1000); NUM = torch.tensor(N.NUM_IDS); out = {}
    for arm in ("mine_intact", "ablate_cross", "ablate_self"):
        m.gather = make_gather(m, {"mine_intact": "none", "ablate_cross": "cross", "ablate_self": "self"}[arm]); res = {}
        for n in ns:
            its = [it for it in TE if it["n"] == n][:per_n]; c = 0
            for c0 in range(0, len(its), 32):
                part = its[c0:c0 + 32]; x, pad, lens = VB.batch_x(part, "cpu"); K = n + 2
                r_ = m.rollout(x, pad, K, coda_rows={K: torch.arange(len(part))})
                c += int((m.answer_logits(r_["coda"][K][1], lens)[:, NUM].argmax(-1) == torch.tensor([it["t"] for it in part])).sum())
                if arm == "mine_intact" and c0 == 0 and n == ns[0]:
                    m.gather = orig; r0 = m.rollout(x, pad, K); m.gather = make_gather(m, "none")
                    out["validation_maxdiff_VK"] = float((r0["V"][K] - r_["V"][K]).abs().max())
            res[str(n)] = round(c / len(its), 4); print(arm, n, res[str(n)], flush=True)
        out[arm] = res
    m.gather = orig
    json.dump(out, open(fo, "w", newline="\n"), indent=1); print("DONE", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
