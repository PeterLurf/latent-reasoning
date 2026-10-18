"""§216 G-S15 window gates (B2; SD2_v1.md at e9ade2ec §2.1; GATE-1's scorer scratch/s215_gate1.py generalised to a window list and a labeller list).
Eval-only, separate process; gold from the syms (_parse_tree -> N.value / VB.recode_targets), never written to a pool. Local 3070 (cuda:1), fp32, bs 32.
  items: <gate dir>/gate_items.json (scratch/s216_build_pool.py --stage gates): windows {W: {seed, ns, items}}.
  per window: the round's model labels the window's items (selfdepth.label_chains); kept = KEEP and CV; rule (= GATE-1's): pi_pool >= 0.995 and
    pi_n >= 0.99 at every n and pi^E_pool >= 0.99 and keep_kv_n >= 0.70 at every n. If it fails, the pre-stated fallback (the lineage's base) labels the
    SAME items as a separate entry; window_labeller[W] = the first passing labeller (round model, then base), None if neither passes (no pool).
  eval_gate.json: by_n {n: {window, labeller, items, n_keep, n_kept, keep_kv, kept_wrong_label, kept_wrong_state, pi, pi_E, pi_ci95, pi_E_ci95, records}}
    (the primary labeller's; records = per item {i, keep, cv, label, gold, state_ok, first_bad_t}); pooled_by_window {W: {...}} (W1 = n 9..12, W2 = n 13..16); GATE_2 / GATE_1p / GATE_1 (the primary
    labeller passes the window);fallback null, or {W: {labeller, by_n, pooled, GATE}} for the failed windows; window_labeller {W: path | None}; gs2_compose (Kb 20: compose_E == recode_targets
    bitwise; differs exactly on the kept wrong-state items).
    python scratch/s216_gate2.py --lineage s1 --round 2 --device cuda:1"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "D:/Latent"); sys.path.insert(0, "D:/Latent/scratch"); os.chdir("D:/Latent")
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402
import boxes.selfdepth as SD  # noqa: E402
from s216_build_pool import CFG  # noqa: E402


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def score(m, items, dev, KB=20):
    lab = SD.label_chains(m, [it["syms"] for it in items], dev, bs=32); recs = []; gs2 = {"n_cmp": 0, "n_equal": 0, "diff_not_wrong_state": 0}
    for i, (it, o) in enumerate(zip(items, lab)):
        assert o["n"] == it["n"] == SD.n_of(it["syms"])
        tree = SD._parse_tree(it["syms"]); gold = int(N.value(tree)); kept = bool(o["keep"] and o["cv"])
        r = {"i": i, "n": it["n"], "keep": o["keep"], "cv": o["cv"], "label": o["label"], "gold": gold, "state_ok": None, "first_bad_t": None}
        if kept:
            Eg = VB.recode_targets(tree); Es = SD.compose_E(it["syms"], o["steps"], it["n"]); bad = [t for t in range(it["n"] + 1) if Es[t] != Eg[t]]
            r["state_ok"] = not bad; r["first_bad_t"] = bad[0] if bad else None
            eq = SD.compose_E(it["syms"], o["steps"], KB) == VB.recode_targets(tree, KB); gs2["n_cmp"] += 1; gs2["n_equal"] += int(eq)
            gs2["diff_not_wrong_state"] += int((not eq) and r["state_ok"])
        recs.append(r)
    return recs, gs2


def stats(recs):
    kp = [r for r in recs if r["keep"] and r["cv"]]; wl = sum(r["label"] != r["gold"] for r in kp); ws = sum(not r["state_ok"] for r in kp); nk = len(kp)
    return {"items": len(recs), "n_keep": sum(r["keep"] for r in recs), "n_kept": nk, "keep_kv": nk / len(recs) if recs else None, "kept_wrong_label": wl,
            "kept_wrong_state": ws, "pi": 1 - wl / nk if nk else None, "pi_E": 1 - ws / nk if nk else None, "pi_ci95": wilson(nk - wl, nk), "pi_E_ci95": wilson(nk - ws, nk)}


def window_read(recs, ns, lab):
    by_n = {str(n): dict(stats([r for r in recs if r["n"] == n]), labeller=lab, records=[r for r in recs if r["n"] == n]) for n in ns}
    pooled = stats(recs)
    ok = (pooled["pi"] is not None and pooled["pi"] >= 0.995 and pooled["pi_E"] >= 0.99
          and all(by_n[str(n)]["pi"] is not None and by_n[str(n)]["pi"] >= 0.99 and by_n[str(n)]["keep_kv"] >= 0.70 for n in ns))
    return by_n, pooled, bool(ok)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lineage", required=True, choices=["s1", "s2"]); ap.add_argument("--round", type=int, required=True, choices=[1, 2])
    ap.add_argument("--device", default="cuda:1"); ap.add_argument("--out_suffix", default=""); a = ap.parse_args(); dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; t0 = time.time()
    cfg = CFG[(a.lineage, a.round)]; gdir = cfg["gate_dir"] + a.out_suffix; gi = json.load(open(os.path.join(gdir, "gate_items.json")))
    res = {"by_n": {}, "pooled_by_window": {}, "fallback": None, "window_labeller": {}, "gs2_compose": {}, "lineage": a.lineage, "round": a.round}
    prim = VE.load(cfg["labeller"], dev)[0]; base = None
    for w, wd in gi["windows"].items():
        recs, gs2 = score(prim, wd["items"], dev); by_n, pooled, ok = window_read(recs, wd["ns"], cfg["labeller"])
        for n, v in by_n.items():
            res["by_n"][n] = dict(v, window=w)
        res["pooled_by_window"][w] = dict(pooled, labeller=cfg["labeller"]); res[wd["gate"]] = ok; res["gs2_compose"][w] = dict(gs2, PASS=gs2["diff_not_wrong_state"] == 0 and gs2["n_cmp"] - gs2["n_equal"] == pooled["kept_wrong_state"])
        res["window_labeller"][w] = cfg["labeller"] if ok else None
        print("WINDOW", w, "labeller", cfg["labeller"], "pass", ok, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pooled.items() if not k.endswith("ci95")}, flush=True)
        if not ok and cfg["base"]:                                          # the pre-stated fallback, same items, separate entry
            base = base or VE.load(cfg["base"], dev)[0]; recs_b, _ = score(base, wd["items"], dev); by_nb, pooled_b, ok_b = window_read(recs_b, wd["ns"], cfg["base"])
            res["fallback"] = res["fallback"] or {}; res["fallback"][w] = {"labeller": cfg["base"], "by_n": by_nb, "pooled": pooled_b, "GATE": ok_b}
            res["window_labeller"][w] = cfg["base"] if ok_b else None
            print("FALLBACK", w, "labeller", cfg["base"], "pass", ok_b, flush=True)
    if len(gi["windows"]) == 1:                                             # s2 round 1: GATE-1's round-1 layout also has a top-level pooled block
        res["pooled"] = dict(res["pooled_by_window"][next(iter(gi["windows"]))])
    res.update({"card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else dev, "dtype": "fp32", "bs": 32, "sec": round(time.time() - t0), "def": __doc__})
    json.dump(res, open(os.path.join(gdir, "eval_gate.json"), "w", newline="\n"), indent=1)
    print("GATES", {k: v for k, v in res.items() if k.startswith("GATE_")}, "window_labeller", res["window_labeller"], flush=True)


if __name__ == "__main__":
    main()
