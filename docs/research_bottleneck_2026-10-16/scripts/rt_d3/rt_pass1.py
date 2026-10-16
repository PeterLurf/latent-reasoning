"""Red-team D3 (a)/(d) + $0 eval-only interventions (CPU, 3 threads, idle; nothing written to the repo).
Own pass-1 re-implementation from the module's primitives (gpt_lower, nrm, gkey, bank_kv, bias_row, block_bank, rec), validated against
m.rollout(x, pad, 1); items from my own rng (seed 90210), m 6..14, grouped by EXACT token length (no padding in any batch).
Arms (pass 1 only; PA cells):
  intact      the trained gather
  gs20/gs24   GATHER-SPLICE at J = 20 / 24 (slots j >= J take N(V_0)[j], as 212-6 but lower J)
  kmask33     the gather's keys restricted to i <= 32 (out-of-range keys masked): do OOD keys capture in-range queries?
  rel20       slots j >= 20 take a translated in-range gather: per-head mean offset profile p_h(d), d in -4..4, measured on T <= 27 items
              at j 6..18 (renormalised over the offsets that exist); ctl_rel6 applies the same splice at j >= 6 on T <= 27 items (control)
Reads per (T bucket, slot bucket): slot accuracy; changed / copy accuracy; first wrong slot; distance from the end; wrong-decode identity.
Usage: python rt_pass1.py <run_dir> <out.json> <per_m> [arms]"""
import json, math, sys
from collections import defaultdict
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
import boxes.deltamem as DM


def tb(T):
    return "T<=27" if T <= 27 else "T28-29" if T <= 29 else "T30-31" if T <= 31 else "T32-37" if T <= 37 else "T38+"


def jb(j):
    return "j00-09" if j < 10 else "j10-19" if j < 20 else "j20-24" if j < 25 else "j25-28" if j < 29 else "j29-32" if j < 33 else "j33+"


def gather_w(m, V0, pad, T):
    """my pass-1 PA gather weights (B, H, T, T): q_j = wpe[j] W_q + b_q (content-free), k_i = gkey(V_0)[i]; logits / 8 - c * 1[i != j]; t = 1 => age 0."""
    q = (DM.wpe_slice(m.g, T) @ m.gq_w + m.gq_b).view(T, m.H, m.hd).transpose(0, 1)
    k = m.gkey(V0)
    lg = torch.einsum("htd,bhsd->bhts", q, k) / math.sqrt(m.hd)
    lg = lg - m.c_anneal * (~torch.eye(T, dtype=torch.bool))[None, None].float()
    return lg


def apply_w(w, NV0, m):
    B, H, T, _ = w.shape
    return (w @ NV0.view(B, T, H, m.hd).transpose(1, 2)).transpose(1, 2).reshape(B, T, H * m.hd)


def core_pass1(m, U, V0, pad):
    g = m.g; h = U; T = U.shape[1]
    for li, i in enumerate(VB.CORE):
        bk, bv = VB.bank_kv(g, i, V0)
        bb = VB.bias_row(m.beta[li], m.beta0[li], m.gam[li], 1, 1, pad).expand(-1, -1, T, -1)
        h = VB.block_bank(g, i, h, bk, bv, bb)
    return h


@torch.no_grad()
def main():
    run, fo, per_m = sys.argv[1], sys.argv[2], int(sys.argv[3])
    arms = sys.argv[4].split(",") if len(sys.argv) > 4 else ["intact"]
    assert DM.POS_OFFSET == 0 and not DM.POS_KEEP0
    m, a = VE.load(run, "cpu"); pa = m.mode == "pa"
    rng = np.random.default_rng(90210); items = []
    for mo in range(6, 15):
        for _ in range(per_m):
            r, _t = N.generate(rng, mo); items.append({"root": r, "syms": N.render(r), "m": mo})
    byT = defaultdict(list)
    for it in items:
        byT[len(it["syms"]) + 2].append(it)
    prof = None
    if pa and any(a_.startswith("rel") or a_.startswith("ctl") for a_ in arms):          # per-head offset profile on in-range items
        rp = np.random.default_rng(90211); pit = []
        while len(pit) < 400:
            mo = int(rp.integers(4, 9)); r, _t = N.generate(rp, mo); s = N.render(r)
            if len(s) + 2 <= 27 and len(s) + 2 >= 21:
                pit.append({"root": r, "syms": s})
        acc_p = torch.zeros(m.H, 9); cnt = 0
        for c0 in range(0, len(pit), 32):
            part = pit[c0:c0 + 32]; x, pad, lens = VB.batch_x(part, "cpu"); T = x.shape[1]
            V0 = DM.gpt_lower(m.g, m.g.embed(x), VB.CORE[0]); lg = gather_w(m, V0, pad, T)
            w = torch.softmax(lg.masked_fill(~pad[:, None, None, :], float("-inf")), -1)
            for b in range(len(part)):
                for j in range(6, 19):
                    for di, d in enumerate(range(-4, 5)):
                        acc_p[:, di] += w[b, :, j, j + d]
                    cnt += 1
        prof = acc_p / cnt
    R = {}
    val_diff = 0.0
    for arm in arms:
        A = defaultdict(lambda: [0, 0]); KIND = defaultdict(lambda: [0, 0]); STEP = defaultdict(lambda: [0, 0]); FIRST = defaultdict(list); FROMEND = defaultdict(list)
        ASSOC = defaultdict(lambda: [0, 0]); LOC = None
        WRONG = defaultdict(lambda: defaultdict(int)); MASS = defaultdict(lambda: defaultdict(float)); MN = defaultdict(int)
        for T in sorted(byT):
            its = byT[T]
            if arm.startswith("ctl") and T > 27:
                continue
            for c0 in range(0, len(its), 24):
                part = its[c0:c0 + 24]; x, pad, lens = VB.batch_x(part, "cpu"); assert bool(pad.all()) and x.shape[1] == T
                V0 = DM.gpt_lower(m.g, m.g.embed(x), VB.CORE[0]); rms0 = V0.pow(2).mean(-1, keepdim=True).sqrt(); NV0 = m.nrm(V0, rms0)
                if not pa:
                    U = NV0
                else:
                    lg = gather_w(m, V0, pad, T)
                    if arm == "kmask33" and T > 33:
                        km = torch.zeros(T, dtype=torch.bool); km[33:] = True
                        lg = lg.masked_fill(km[None, None, None, :] & (torch.arange(T)[:, None] < 33)[None, None], float("-inf"))   # in-range queries only
                    w = torch.softmax(lg, -1); U = apply_w(w, NV0, m)
                    wm_ = w.mean(1); LOC = torch.stack([wm_[:, j, max(0, j - 1):j + 3].sum(-1) for j in range(T)], 1).numpy()   # (B, T) mass on j-1..j+2
                    if arm == "intact" and T >= 25:                       # gather mass by key region, per query bucket (head-mean)
                        wm = w.mean(1)
                        for b in range(len(part)):
                            for j in range(1, T - 1):
                                key = "%s|%s" % (tb(T), jb(j)); row = wm[b, j]; MN[key] += 1
                                MASS[key]["self"] += float(row[j]); MASS[key]["near(|d|<=4,!self)"] += float(row[max(0, j - 4):j + 5].sum() - row[j])
                                MASS[key]["sink"] += float(row[0]); MASS[key]["eq"] += float(row[T - 1])
                                far = torch.ones(T, dtype=torch.bool); far[max(0, j - 4):j + 5] = False; far[0] = False; far[T - 1] = False
                                MASS[key]["far_keys<25"] += float(row[:25][far[:25]].sum()); MASS[key]["far_keys25-32"] += float(row[25:33][far[25:33]].sum())
                                MASS[key]["far_keys>=33"] += float(row[33:][far[33:]].sum()) if T > 33 else 0.0
                    if arm in ("gs20", "gs24"):
                        J = int(arm[2:]); U = torch.cat([U[:, :J], NV0[:, J:]], 1)
                    if arm in ("rel20", "ctl_rel6"):
                        J = 20 if arm == "rel20" else 6
                        Uh = U.view(len(part), T, m.H, m.hd).clone(); NVh = NV0.view(len(part), T, m.H, m.hd)
                        for j in range(J, T):
                            ds = [d for d in range(-4, 5) if 0 <= j + d < T]; pw = prof[:, [d + 4 for d in ds]]; pw = pw / pw.sum(1, keepdim=True)   # (H, nd)
                            Uh[:, j] = torch.einsum("hd,bdhe->bhe", pw, NVh[:, [j + d for d in ds]])
                        U = Uh.reshape(len(part), T, m.H * m.hd)
                V1 = core_pass1(m, U, V0, pad)
                if arm == "intact" and c0 == 0 and T in (27, 35, 43):           # validation against the module's rollout
                    r_ = m.rollout(x, pad, 1); val_diff = max(val_diff, float((r_["V"][1] - V1).abs().max())); del r_
                dec = m.rec_logits(V1).argmax(-1).numpy()
                for k_, it in enumerate(part):
                    E = VB.recode_targets(it["root"], 1); E0, E1 = E[0], E[1]; bad = []
                    for j in range(T):
                        ok = int(dec[k_, j]) == E1[j]; A[(tb(T), jb(j))][0] += 1; A[(tb(T), jb(j))][1] += int(ok)
                        if LOC is not None and arm == "intact" and j >= 20 and j < T - 1:
                            lb = "loc<0.3" if LOC[k_, j] < 0.3 else "loc0.3-0.6" if LOC[k_, j] < 0.6 else "loc>=0.6"
                            ASSOC[(jb(j), lb)][0] += 1; ASSOC[(jb(j), lb)][1] += int(ok)
                        kind = "changed" if E0[j] != E1[j] else "copy"; KIND[(tb(T), kind)][0] += 1; KIND[(tb(T), kind)][1] += int(ok)
                        if not ok:
                            bad.append(j); dj = int(dec[k_, j])
                            idt = "VOID" if dj == 0 else "=E1[j-1]" if j > 0 and dj == E1[j - 1] else "=E1[j+1]" if j + 1 < T and dj == E1[j + 1] else "=E1[j+2]" if j + 2 < T and dj == E1[j + 2] else "other"
                            WRONG[tb(T)][idt] += 1
                    STEP[tb(T)][0] += 1; STEP[tb(T)][1] += int(not bad)
                    if bad:
                        FIRST[tb(T)].append(min(bad)); FROMEND[tb(T)].append(T - 1 - min(bad))
            print(arm, "T", T, flush=True)
        q = lambda v: [float(np.quantile(v, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)] + [len(v)]  # noqa: E731
        R[arm] = {"slot_acc": {"%s|%s" % k: [v[0], round(v[1] / v[0], 4)] for k, v in sorted(A.items())},
                  "by_kind": {"%s|%s" % k: [v[0], round(v[1] / v[0], 4)] for k, v in sorted(KIND.items())},
                  "step_exact": {k: [v[0], round(v[1] / v[0], 4)] for k, v in sorted(STEP.items())},
                  "first_bad_q": {k: q(v) for k, v in sorted(FIRST.items())}, "first_bad_from_end_q": {k: q(v) for k, v in sorted(FROMEND.items())},
                  "first_bad_hist": {k: np.bincount(v, minlength=60).tolist() for k, v in sorted(FIRST.items())},
                  "wrong_identity": {k: dict(v) for k, v in WRONG.items()}}
        if ASSOC:
            R[arm]["assoc_local_mass"] = {"%s|%s" % k: [v[0], round(v[1] / v[0], 4)] for k, v in sorted(ASSOC.items())}
        if MASS:
            R[arm]["gather_mass"] = {k: {kk: round(vv / MN[k], 4) for kk, vv in v.items()} | {"n": MN[k]} for k, v in sorted(MASS.items())}
    R["validation_max_abs_diff_vs_rollout"] = val_diff
    if prof is not None:
        R["profile_head_mean"] = [round(float(x), 4) for x in prof.mean(0)]
    json.dump(R, open(fo, "w", newline="\n"), indent=1); print("DONE", fo, "val", val_diff, flush=True)


if __name__ == "__main__":
    main()
