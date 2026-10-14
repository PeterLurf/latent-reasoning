"""§201 T1 ProsQA, DESCRIPTIVE (plan Part V T1: s191_stepaux_relabel_s1; oracle freeze point f = depth + 1; passes <= 24). fp32 (no autocast), one card.
Arms as the nested T1: CLEAN; FA (S_add reset to S_add,f after every write past f); FD (S_delta reset); FB (both = halting at f, Prop 7).
Scope (descriptive; the full set at every pass would be ~6 h on the 2070): the first 200 test items per depth 3..14 (p8_na_test order), candidate
accuracy (prosqa168s2.ProsMem.margins: log p(correct) > log p(negative)) at K' in {f} U {10, 12, 14, 16, 20, 24}; the think passes run every pass.
Gate: FB's margins at every evaluated K' >= f equal CLEAN's at f (bitwise).
-> <run>/eval_s201_freeze.json {arms: {ARM: {acc_by_depth_kprime: {K': {depth: acc}}, acc_at_f_by_depth}}, pools {K': {d3_10, d11_14}}, gates, scope}
    python scratch/s201_prosqa.py [run s191_stepaux_relabel_s1] [device cuda:2]"""
import json
import os
import sys
import time

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_grad_enabled(False)
import boxes.deltamem as DM  # noqa: E402
import boxes.prosqa168s2 as P  # noqa: E402
from boxes import s5  # noqa: E402

ARMS = ("CLEAN", "FA", "FD", "FB"); KEV = (10, 12, 14, 16, 20, 24); TMAX = 24; NPER = 200; BS = 16


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "s191_stepaux_relabel_s1"; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:2"; rd = os.path.join(ROOT, "boxes", "runs_fsq", run)
    a = json.load(open(os.path.join(rd, "args.json"))); DM.FAST_ADD = bool(a.get("fast_add")); DM.CHUNK_DELTA = 0
    g = s5.load_gpt2(dev); m = P.ProsMem(g).to(dev); m.load_state_dict(torch.load(os.path.join(rd, "model.pt"), map_location=dev, weights_only=False)); m.eval()
    mem = m.mem; HA, HD = mem.h_add, mem.h_del; t0 = time.time()
    Dte = P.load(os.path.join(ROOT, "prosqa", "p8data", "p8_na_test.json")); byd = {}
    for it in Dte:
        if len(byd.setdefault(it["depth"], [])) < NPER:
            byd[it["depth"]].append(it)
    acc = {A: {} for A in ARMS}; atf = {A: {} for A in ARMS}; agree = {"FA": {}, "FD": {}}; fb_ok = True; cuda = dev.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    for d in sorted(byd):
        its = byd[d]; f = d + 1; ks = sorted({f} | set(KEV)); res = {A: {k: [] for k in ks} for A in ARMS}
        for i in range(0, len(its), BS):
            ch = its[i:i + BS]; x, msk = m.think_inputs(ch, dev)
            for A in ARMS:
                S = mem.zero_state(len(ch), dev); Sf = None
                for t in range(1, TMAX + 1):
                    S = m.write(S, x, msk)
                    if A != "CLEAN" and t > f:
                        S = S.clone()
                        if A in ("FA", "FB"):
                            S[:, :HA] = Sf[:, :HA]
                        if A in ("FD", "FB"):
                            S[:, HA:HA + HD] = Sf[:, HA:HA + HD]
                    if t == f:
                        Sf = S.clone()
                    if t in res[A]:
                        res[A][t].append(m.margins(S, ch, dev))
        for A in ARMS:
            M = {k: np.concatenate(v) for k, v in res[A].items()}
            acc[A][str(d)] = {str(k): float((M[k] > 0).mean()) for k in KEV}; atf[A][str(d)] = float((M[f] > 0).mean())
            if A == "CLEAN":
                Mc = M
            if A == "FB":
                fb_ok = fb_ok and all(np.array_equal(M[k], Mc[f]) for k in ks if k >= f)
            if A in ("FA", "FD"):                                           # Theory: the share of items whose answer at K' equals CLEAN's answer at f
                agree[A][str(d)] = {str(k): float(((M[k] > 0) == (Mc[f] > 0)).mean()) for k in KEV}
        print(run, "depth", d, {A: acc[A][str(d)][str(max(KEV))] for A in ARMS}, "%.0fs" % (time.time() - t0), flush=True)
    pool = lambda A, k, ds: float(np.mean([acc[A][str(d)][str(k)] for d in ds if str(d) in acc[A]]))   # noqa: E731
    out = {"arms": {A: {"acc_by_depth_kprime": {str(k): {d: acc[A][d][str(k)] for d in acc[A]} for k in KEV}, "acc_at_f_by_depth": atf[A]} for A in ARMS},
           "pools": {str(k): {A: {"d3_10": pool(A, k, range(3, 11)), "d11_14": pool(A, k, range(11, 15))} for A in ARMS} for k in KEV},
           "agree_with_f_by_depth_kprime": {A: {str(k): {d: agree[A][d][str(k)] for d in agree[A]} for k in KEV} for A in agree},
           "gates": {"fb_identity": bool(fb_ok)}, "scope": {"n_per_depth": NPER, "kprimes": list(KEV), "plus": "f = depth + 1", "tmax": TMAX, "items": "the first 200 p8_na_test items per depth"},
           "status": "DESCRIPTIVE (§201 T1 ProsQA)", "card": torch.cuda.get_device_name(dev) if cuda else "cpu", "dtype": "float32 (no autocast)", "bs": BS,
           "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2) if cuda else None, "sec": round(time.time() - t0)}
    json.dump(out, open(os.path.join(rd, "eval_s201_freeze.json"), "w", newline="\n"), indent=1); print(run, "ProsQA T1 done", out["gates"], out["sec"], flush=True)


if __name__ == "__main__":
    main()
