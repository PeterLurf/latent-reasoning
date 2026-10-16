"""D3 $0 descriptive probe (CPU, 3 threads, idle priority; no intervention on any model output; nothing written to the repo). On a PA run (default N2):
 A. fresh single steps (pass 1, bank = {V_0}) on generator items m 6..14 (rng 31337, per_m each):
    the PURE-ANNEAL gather's pass-1 attention per query slot j (head-mean): self, offsets -4..4, sink (slot 0), ' =' slot, far (|d| > 4);
    per-slot recode errors at pass 1 against E_1, by slot j and by offset from the handle's value slot; step exactness by (m, T).
 B. in-range items (m 3..8, T <= 27; rng 31338): gather mass at passes 2..4 by (age, offset bucket), K = 4 rollout.
 C. a translation-invariant symbolic fit of the pass-1 gather (per head: M[class_j, class_i] + w[clip(i - j, +-8)] + far-left / far-right buckets +
    sink and ' =' anchors), fitted by KL on in-range items (T <= 27), then the overlap sum_i min(p, p') between N2's actual pass-1 gather and the
    fitted one, by query slot j, on the part-A items (GATHER-DRIFT: does the trained gather leave its own in-range relative pattern at large j?).
Usage: python d3_gather_probe.py <run_dir> <out.json> [per_m]"""
import json
import math
import sys
from collections import defaultdict

ROOT = "D:/Latent"
sys.path.insert(0, ROOT)
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:  # noqa: BLE001
    pass
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402
import boxes.deltamem as DM  # noqa: E402

R_OFF = 8


def handle_span(root):
    E = VB.recode_targets(root, 1); return [j for j in range(len(E[0])) if E[0][j] != E[1][j]]


def pass1_logits(m, x, pad):
    """per-head pass-1 gather logits (B, H, T, T) before the anneal term and the mask; V_0; the classes of x."""
    g = m.g; B, T = x.shape
    V0 = DM.gpt_lower(g, g.embed(x), VB.CORE[0]); gk = m.gkey(V0)
    q = (DM.wpe_slice(g, T) @ m.gq_w + m.gq_b).view(T, m.H, m.hd).transpose(0, 1)
    return torch.einsum("htd,bhsd->bhts", q, gk) / math.sqrt(m.hd)


def classes(items, T):
    C = np.zeros((len(items), T), dtype=np.int64)
    for r, it in enumerate(items):
        e = VB.recode_targets(it["root"], 0)[0]; C[r, :len(e)] = e
    return torch.tensor(C)


def gather_w(m, lg, pad, T):
    own = torch.eye(T, dtype=torch.bool)
    lg = lg - m.c_anneal * (~own).float()[None, None]
    return torch.softmax(lg.masked_fill(~pad[:, None, None, :], float("-inf")), -1)


class SymRel(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.M = torch.nn.Parameter(torch.zeros(H, VB.NCLS, VB.NCLS)); self.w = torch.nn.Parameter(torch.zeros(H, 2 * R_OFF + 3))
        self.sink = torch.nn.Parameter(torch.zeros(H)); self.eq = torch.nn.Parameter(torch.zeros(H))

    def forward(self, C, lens, T):
        B = C.shape[0]; H = self.M.shape[0]
        lg = self.M[:, C[:, :, None], C[:, None, :]].permute(1, 0, 2, 3)            # (B, H, T, T)
        d = torch.arange(T)[None, :] - torch.arange(T)[:, None]                      # i - j
        b = torch.where(d < -R_OFF, torch.zeros_like(d), torch.where(d > R_OFF, torch.full_like(d, 2 * R_OFF + 2), d + R_OFF + 1))
        lg = lg + self.w[:, b][None]
        sink = torch.zeros(T); sink[0] = 1.0
        lg = lg + self.sink[None, :, None, None] * sink[None, None, None, :]
        eqm = (torch.arange(T)[None, :] == (lens - 1)[:, None]).float()            # (B, T)
        lg = lg + self.eq[None, :, None, None] * eqm[:, None, None, :]
        return lg


@torch.no_grad()
def collect(m, items, bs=32):
    out = []
    items = sorted(items, key=lambda it: len(it["syms"]))
    for c0 in range(0, len(items), bs):
        part = items[c0:c0 + bs]; x, pad, lens = VB.batch_x(part, "cpu"); T = x.shape[1]
        out.append((part, x, pad, lens, pass1_logits(m, x, pad), classes(part, T)))
    return out


def main():
    run, fo = sys.argv[1], sys.argv[2]; per_m = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    assert DM.POS_OFFSET == 0 and not DM.POS_KEEP0
    m, a = VE.load(run, "cpu"); assert m.mode == "pa"; H = m.H
    print("c_anneal", float(m.c_anneal), flush=True)
    res = {"run": run, "c_anneal": float(m.c_anneal), "per_m": per_m}
    # ---- C (fit) first: in-range items T <= 27
    rng = np.random.default_rng(31339); fit_items = []
    while len(fit_items) < 1200:
        mo = int(rng.integers(3, 9)); r, t = N.generate(rng, mo); s = N.render(r)
        if len(s) + 2 <= 27:
            fit_items.append({"root": r, "syms": s, "m": mo})
    with torch.no_grad():
        tr = collect(m, fit_items[:900]); te = collect(m, fit_items[900:])
    f = SymRel(H); opt = torch.optim.Adam(f.parameters(), lr=0.05)

    def kl(batch, train=True):
        tot = 0.0; n = 0
        for part, x, pad, lens, lg, C in batch:
            T = x.shape[1]; msk = ~pad[:, None, None, :]
            p = torch.softmax(lg.masked_fill(msk, -1e9), -1); lq = torch.log_softmax(f(C, lens, T).masked_fill(msk, -1e9), -1)
            k_ = (p * (torch.log(p.clamp_min(1e-30)) - lq)).sum(-1)                    # (B, H, T)
            valid = pad[:, None, :].float(); tot = tot + (k_ * valid).sum(); n += float(valid.sum()) * H
        return tot / n
    for it_ in range(250):
        opt.zero_grad(); L = kl(tr); L.backward(); opt.step()
        if it_ % 50 == 0:
            print("fit", it_, float(L), flush=True)
    with torch.no_grad():
        res["fit"] = {"kl_train": float(kl(tr)), "kl_heldout": float(kl(te))}
    # ---- A: fresh long steps
    rng = np.random.default_rng(31337); items = []
    for mo in range(6, 15):
        for _ in range(per_m):
            r, t = N.generate(rng, mo); items.append({"root": r, "syms": N.render(r), "m": mo})
    G = defaultdict(lambda: defaultdict(float)); Gn = defaultdict(int)
    S = defaultdict(lambda: [0, 0]); Hh = defaultdict(lambda: [0, 0]); STEP = defaultdict(lambda: [0, 0]); FAR = defaultdict(lambda: [0, 0])
    OV = defaultdict(lambda: [0.0, 0])
    items.sort(key=lambda it: len(it["syms"])); bs = 32
    with torch.no_grad():
        for c0 in range(0, len(items), bs):
            part = items[c0:c0 + bs]; x, pad, lens = VB.batch_x(part, "cpu"); B, T = x.shape
            lg = pass1_logits(m, x, pad); W = gather_w(m, lg, pad, T); w = W.mean(1).numpy()
            pf = torch.softmax(f(classes(part, T), lens, T).masked_fill(~pad[:, None, None, :], float("-inf")), -1)
            pa = torch.softmax(lg.masked_fill(~pad[:, None, None, :], float("-inf")), -1)
            ov = torch.minimum(pf, pa).sum(-1).mean(1).numpy()                   # (B, T) head-mean overlap of the logit patterns (anneal term off)
            r_ = m.rollout(x, pad, 1); dec = m.rec_logits(r_["V"][1]).argmax(-1).numpy(); del r_
            for k_, it in enumerate(part):
                L = int(lens[k_]); E1 = VB.recode_targets(it["root"], 1)[1]; hl = handle_span(it["root"])[-1]
                okall = all(int(dec[k_, j]) == E1[j] for j in range(L)); key = "%d_%d" % (it["m"], L); STEP[key][0] += 1; STEP[key][1] += int(okall)
                for j in range(L):
                    ok = int(dec[k_, j]) == E1[j]; S[j][0] += 1; S[j][1] += int(ok)
                    d = j - hl; Hh[max(-6, min(6, d))][0] += 1; Hh[max(-6, min(6, d))][1] += int(ok)
                    if abs(d) > 6:
                        kk = "far_handle_j<29" if j < 29 else "far_handle_j>=29"; FAR[kk][0] += 1; FAR[kk][1] += int(ok)
                    wj = w[k_, j, :L]; Gn[j] += 1
                    G[j]["self"] += float(wj[j]); G[j]["sink"] += float(wj[0]); G[j]["eq"] += float(wj[L - 1])
                    for o in range(-4, 5):
                        if 0 <= j + o < L:
                            G[j]["o%+d" % o] += float(wj[j + o])
                    G[j]["far"] += float(sum(wj[i] for i in range(L) if abs(i - j) > 4 and i != 0 and i != L - 1))
                    OV[j][0] += float(ov[k_, j]); OV[j][1] += 1
            print("A", c0, T, flush=True)
    res["gather_pass1_by_j"] = {str(j): {**{k: v / Gn[j] for k, v in G[j].items()}, "n": Gn[j]} for j in sorted(G)}
    res["fit_overlap_by_j"] = {str(j): {"overlap": v[0] / v[1], "n": v[1]} for j, v in sorted(OV.items())}
    res["slot_acc_by_j"] = {str(j): {"n": v[0], "acc": v[1] / v[0]} for j, v in sorted(S.items())}
    res["slot_acc_by_handle_offset"] = {str(d): {"n": v[0], "acc": v[1] / v[0]} for d, v in sorted(Hh.items())}
    res["far_from_handle"] = {k: {"n": v[0], "acc": v[1] / v[0]} for k, v in FAR.items()}
    res["step_exact_m_T"] = {k: {"n": v[0], "exact": v[1] / v[0]} for k, v in sorted(STEP.items(), key=lambda kv: tuple(int(z) for z in kv[0].split("_")))}
    # ---- B: passes 2..4 on in-range items
    rng = np.random.default_rng(31338); its = []
    while len(its) < 256:
        mo = int(rng.integers(3, 9)); r, t = N.generate(rng, mo); s = N.render(r)
        if len(s) + 2 <= 27:
            its.append({"root": r, "syms": s, "m": mo})
    AG = defaultdict(float); AGn = defaultdict(float)
    with torch.no_grad():
        for c0 in range(0, len(its), 32):
            part = its[c0:c0 + 32]; x, pad, lens = VB.batch_x(part, "cpu"); B, T = x.shape
            r_ = m.rollout(x, pad, 4); St = r_["S"]; del r_
            gk = [m.gkey(Sv) for Sv in St]
            q = (DM.wpe_slice(m.g, T) @ m.gq_w + m.gq_b).view(T, H, m.hd).transpose(0, 1)
            for t in (2, 3, 4):
                K_ = torch.cat(gk[:t], 2); lg = torch.einsum("htd,bhsd->bhts", q, K_) / math.sqrt(m.hd)
                s_idx = torch.arange(t).repeat_interleave(T); i_idx = torch.arange(T).repeat(t)
                lg = lg + m.gamma_q[None, :, None, None] * (t - 1 - s_idx).float()[None, None, None, :]
                own = (s_idx[None, :] == t - 1) & (i_idx[None, :] == torch.arange(T)[:, None]); lg = lg - m.c_anneal * (~own).float()[None, None]
                valid = pad[:, None, :].expand(B, t, T).reshape(B, t * T)
                w = torch.softmax(lg.masked_fill(~valid[:, None, None, :], float("-inf")), -1).mean(1)   # (B, T, tT)
                for k_ in range(B):
                    L = int(lens[k_])
                    for j in range(L):
                        for sv in range(t):
                            row = w[k_, j, sv * T:sv * T + L]
                            age = t - 1 - sv
                            for o in (-2, -1, 0, 1, 2):
                                if 0 <= j + o < L:
                                    AG["t%d_age%d_o%+d" % (t, age, o)] += float(row[j + o])
                            AG["t%d_age%d_sink" % (t, age)] += float(row[0]); AG["t%d_age%d_all" % (t, age)] += float(row.sum())
                        AGn["t%d" % t] += 1
    res["gather_pass2_4_mass"] = {k: v / AGn[k.split("_")[0]] for k, v in sorted(AG.items())}
    json.dump(res, open(fo, "w", newline="\n"), indent=1); print("DONE", fo, flush=True)


if __name__ == "__main__":
    main()
