"""Red-team of note 212-13 (WPE-RESTORE): in a PA cell the table feeds TWO paths - the prelude (V_0 = blocks 0-5 on wte + wpe) and the
gather's content-free queries q_j = wpe[j] W_q + b_q. wpe_b1 changes both. Split it (pass 1, eval-only, CPU 3 threads, idle; no repo writes):
  intact | b1_both (= 212-13 wpe_b1, repro) | b1_prelude (pretrained rows >= 24 in the prelude only; gather queries on the trained table)
  | b1_gather (pretrained rows >= 24 in the gather queries only; prelude on the trained table)   [+ b2_* = every row]
Sets (212-11): exposed control m <= 8, T 25..29; target T in {31, 33}, m 8..10. Items: own rng (31415), generator draws per m.
Reads: step exact (pooled item-weighted) and per-slot copy accuracy at j >= 24.
Usage: python rt_wpesplit.py <run> <out.json> <per_m>"""
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
sys.path.insert(0, "C:/Users/peter/AppData/Local/Temp/claude/D--/2f0034d5-4d85-51eb-880b-bab41adb0dff/scratchpad/research2/rt_d3")
import rt_pass1 as P


@torch.no_grad()
def main():
    run, fo, per_m = sys.argv[1], sys.argv[2], int(sys.argv[3])
    m, a = VE.load(run, "cpu"); pa = m.mode == "pa"
    Wt = m.g.w("wpe.weight").detach().clone()
    pre = torch.load("D:/Latent/prosqa/ckpt/gpt2_pretrained.bin", map_location="cpu", weights_only=True)["wpe.weight"].float()
    Wb1 = Wt.clone(); Wb1[24:] = pre[24:]
    tabs = {"intact": (Wt, Wt), "b1_both": (Wb1, Wb1), "b1_prelude": (Wb1, Wt), "b1_gather": (Wt, Wb1), "b2_both": (pre, pre), "b2_prelude": (pre, Wt), "b2_gather": (Wt, pre)}
    if not pa:
        tabs = {k: v for k, v in tabs.items() if k in ("intact", "b1_both", "b2_both")}
    rng = np.random.default_rng(31415); cells = defaultdict(list)
    for mo in range(4, 11):
        for _ in range(per_m):
            r, _t = N.generate(rng, mo); s = N.render(r); T = len(s) + 2
            if (mo <= 8 and 25 <= T <= 29) or (8 <= mo <= 10 and T in (31, 33)):
                cells[(mo, T)].append({"root": r, "syms": s})
    out = {"cells_n": {"%d_%d" % k: len(v) for k, v in sorted(cells.items())}}
    for arm, (Wp, Wq) in tabs.items():
        S = defaultdict(lambda: [0, 0]); SL = defaultdict(lambda: [0, 0]); SR = defaultdict(lambda: [0, 0])
        for (mo, T), its in sorted(cells.items()):
            grp = "control" if (mo <= 8 and T <= 29) else "target"
            for c0 in range(0, len(its), 32):
                part = its[c0:c0 + 32]; x, pad, lens = VB.batch_x(part, "cpu"); assert bool(pad.all())
                h = m.g.embed(x) + Wp[:T][None]
                for i in range(VB.CORE[0]):
                    h = DM._block(m.g, i, h, m.hd)
                V0 = h; NV0 = m.nrm(V0, V0.pow(2).mean(-1, keepdim=True).sqrt())
                if pa:
                    q = (Wq[:T] @ m.gq_w + m.gq_b).view(T, m.H, m.hd).transpose(0, 1)
                    lg = torch.einsum("htd,bhsd->bhts", q, m.gkey(V0)) / math.sqrt(m.hd) - m.c_anneal * (~torch.eye(T, dtype=torch.bool)).float()[None, None]
                    U = P.apply_w(torch.softmax(lg, -1), NV0, m)
                else:
                    U = NV0
                dec = m.rec_logits(P.core_pass1(m, U, V0, pad)).argmax(-1).numpy()
                for k_, it in enumerate(part):
                    E0, E1 = VB.recode_targets(it["root"], 1)
                    ok = [int(dec[k_, j]) == E1[j] for j in range(T)]
                    S[grp][0] += 1; S[grp][1] += int(all(ok))
                    SR[grp][0] += 1; SR[grp][1] += int(VB.rerender(list(dec[k_, :T])) == VB.rerender(E1))   # the registered (VOID-dropped) exact
                    for j in range(24, T - 1):
                        SL[grp][0] += 1; SL[grp][1] += int(ok[j])
        out[arm] = {g_: {"exact_strict": round(v[1] / v[0], 4), "exact_rerender": round(SR[g_][1] / SR[g_][0], 4), "n": v[0], "slot_acc_j24+": round(SL[g_][1] / SL[g_][0], 4)} for g_, v in S.items()}
        print(arm, out[arm], flush=True)
    json.dump(out, open(fo, "w", newline="\n"), indent=1); print("DONE", flush=True)


if __name__ == "__main__":
    main()
