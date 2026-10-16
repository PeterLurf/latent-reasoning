"""$0 CPU descriptive (3 threads, idle; no intervention): where do fresh single steps fail? Per run (N1 / N2 / N3), fresh items m 6..14 (rng 31340),
one pass, R(V_1) argmax vs E_1, slot accuracy by (T bucket, slot bucket) and by the slot's position relative to the handle; plus the recoded
class of wrong slots (is it VOID / copy / value). Usage: python d3_slot_errors.py <run> <out.json> [per_m]"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "D:/Latent")
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


def tb(T):
    return "T<=27" if T <= 27 else "T29" if T <= 29 else "T31" if T <= 31 else "T33-37" if T <= 37 else "T39+"


def jb(j):
    return "j00-09" if j < 10 else "j10-19" if j < 20 else "j20-28" if j < 29 else "j29-32" if j < 33 else "j33+"


@torch.no_grad()
def main():
    run, fo = sys.argv[1], sys.argv[2]; per_m = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    assert DM.POS_OFFSET == 0 and not DM.POS_KEEP0
    m, a = VE.load(run, "cpu")
    rng = np.random.default_rng(31340); items = []
    for mo in range(6, 15):
        for _ in range(per_m):
            r, t = N.generate(rng, mo); items.append({"root": r, "syms": N.render(r), "m": mo})
    items.sort(key=lambda it: len(it["syms"]))
    A = defaultdict(lambda: [0, 0]); HO = defaultdict(lambda: [0, 0]); KIND = defaultdict(lambda: [0, 0]); STEP = defaultdict(lambda: [0, 0])
    FIRST = defaultdict(list)
    for c0 in range(0, len(items), 32):
        part = items[c0:c0 + 32]; x, pad, lens = VB.batch_x(part, "cpu")
        r_ = m.rollout(x, pad, 1); dec = m.rec_logits(r_["V"][1]).argmax(-1).numpy(); del r_
        for k_, it in enumerate(part):
            L = int(lens[k_]); E = VB.recode_targets(it["root"], 1); E0, E1 = E[0], E[1]
            span = [j for j in range(L) if E0[j] != E1[j]]; hl = span[-1]; bad = []
            for j in range(L):
                ok = int(dec[k_, j]) == E1[j]; A[(tb(L), jb(j))][0] += 1; A[(tb(L), jb(j))][1] += int(ok)
                d = j - hl; key = (tb(L), "h%+d" % max(-6, min(6, d))); HO[key][0] += 1; HO[key][1] += int(ok)
                kind = "changed" if E0[j] != E1[j] else "copy"; KIND[(tb(L), kind)][0] += 1; KIND[(tb(L), kind)][1] += int(ok)
                if not ok:
                    bad.append(j)
            STEP[tb(L)][0] += 1; STEP[tb(L)][1] += int(not bad)
            if bad:
                FIRST[tb(L)].append(min(bad))
        print("b", c0, x.shape[1], flush=True)
    out = {"run": run, "per_m": per_m,
           "slot_acc": {"%s|%s" % k: {"n": v[0], "acc": v[1] / v[0]} for k, v in sorted(A.items())},
           "by_handle_offset": {"%s|%s" % k: {"n": v[0], "acc": v[1] / v[0]} for k, v in sorted(HO.items())},
           "by_kind": {"%s|%s" % k: {"n": v[0], "acc": v[1] / v[0]} for k, v in sorted(KIND.items())},
           "step_exact_by_T": {k: {"n": v[0], "exact": v[1] / v[0]} for k, v in sorted(STEP.items())},
           "first_bad_slot_quantiles": {k: [float(np.quantile(v, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] + [len(v)] for k, v in sorted(FIRST.items())}}
    json.dump(out, open(fo, "w", newline="\n"), indent=1); print("DONE", fo, flush=True)


if __name__ == "__main__":
    main()
