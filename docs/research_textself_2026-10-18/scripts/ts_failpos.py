"""TEXT-SELF draft check 6 (CPU, no model): where the base's FULL chains break, in steps and in absolute rows (FAIL-POS, §4.5's descriptive split).
Re-draws ts_label_check.py's items (the same stream: rng 23180, n 9..20, 200 per n, exclusion test_set(1000, 24) + SERIAL) and reads its per-item
FULL results ([keep, fail_stage, correct]). Stage k of label_full checks suffix(e_{k-1}) + ' =' + prefix(e_k) and reads v_k, so a failure at stage k
lies between the end of e_{k-1}'s value slot and e_k's value slot; its row window is reported as [start(e_{k-1} value), pos(v_k)].
Rows: the FULL sequence is [EOT] e_0 ' =' e_1 ' =' ... (P = len(e_0) + 2). Shape only (no apply). python -B ts_failpos.py [tag]"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); TAG = sys.argv[1] if len(sys.argv) > 1 else "_s1"


def items_stream():
    excl = {tuple(it["syms"]) for it in N.test_set(1000, n_max=24)}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    rng = np.random.default_rng(23180); items = {}
    for n in range(9, 21):
        items[n] = []
        while len(items[n]) < 200:
            r_, _ = N.generate(rng, n)
            if tuple(N.render(r_)) in excl:
                continue
            items[n].append(r_)
    return items


def value_rows(root):
    """absolute row of each value slot v_1..v_n in the FULL sequence, and P."""
    nd = N._copy(root); P = len(N.render(nd)) + 2; pos = P; out = []
    while nd.op is not None:
        path = N._leftmost_ready(nd); tgt = nd
        for p in path:
            tgt = getattr(tgt, p)
        tgt.op = tgt.l = tgt.r = None; tgt.v = 0; syms, vp = N._render_pos(nd, tgt); out.append(pos + vp); pos += len(syms) + 1
    return P, out


def main():
    R = json.load(open(os.path.join(HERE, "ts_label_check%s.json" % TAG))); items = items_stream(); res = {"tag": TAG, "by_n": {}}
    for key, recs in R["items"].items():
        n, proto = key.split("_"); n = int(n)
        if proto != "full":
            continue
        st, lo, hi = [], [], []
        for it, (keep, fs, corr) in zip(items[n][:len(recs)], recs):
            if keep:
                continue
            P, vr = value_rows(it); k = int(fs)
            st.append(k); lo.append(P - 1 if k == 1 else vr[k - 2]); hi.append(vr[k - 1] if k <= len(vr) else vr[-1] + 1)
        if st:
            res["by_n"][str(n)] = {"n_fail": len(st), "stage_median": float(np.median(st)), "stage_q25_q75": [float(np.quantile(st, 0.25)), float(np.quantile(st, 0.75))],
                                   "row_lo_median": float(np.median(lo)), "row_hi_median": float(np.median(hi)),
                                   "row_mid_q25_q75": [float(np.quantile((np.array(lo) + np.array(hi)) / 2, q)) for q in (0.25, 0.75)],
                                   "share_mid_gt_147": float(np.mean((np.array(lo) + np.array(hi)) / 2 > 147))}
            print(n, res["by_n"][str(n)], flush=True)
    json.dump(res, open(os.path.join(HERE, "ts_failpos%s.json" % TAG), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
