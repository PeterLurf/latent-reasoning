"""§215 GATE-1 / G-S15 scorer (SELFDEPTH_v1 §4.3 at 5ded8703). A separate eval-only process; gold is recomputed here from the syms and is never
written to the pool. Labeller and settings as the pool builder (s213_d3t_s1/model.pt, 3070 cuda:1, fp32, bs 32); items =
boxes/runs_fsq/s215_gate1/gate_items.json (rng 21960, 2,000 per n 9..12, the §2.2 exclusion).
  kept = KEEP and CV (selfdepth.label_chains). Gold: tree = _parse_tree(syms); value = N.value(tree); E = vbank.recode_targets(tree).
  pi_n = 1 - #{kept, label != value} / #kept;  pi^E_n = #{kept, compose_E(syms, steps)(t) == E(t) for every t in 0..n} / #kept (item-weighted);
  keep_n = #kept / #items. GATE_1 = pi_pool >= 0.995 and pi_n >= 0.99 at every n and pi^E_pool >= 0.99 and keep_n >= 0.70 at every n.
  Also: Wilson 95 % CIs on pi / pi^E (LABEL-AUDIT's overlap test), the G-S2 compose part (compose_E HOLD-padded to Kb 16 == recode_targets(tree, 16)
  bitwise; differs exactly on the kept_wrong_state items), and per item: n, keep, cv, label, gold, state_ok, first_bad_t.
    python scratch/s215_gate1.py --device cuda:1"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.vbank as VB  # noqa: E402
import boxes.vbank_eval as VE  # noqa: E402
import boxes.selfdepth as SD  # noqa: E402

OUTG = "boxes/runs_fsq/s215_gate1"


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--out_suffix", default=""); a = ap.parse_args(); dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; t0 = time.time(); outg = OUTG + a.out_suffix
    gi = json.load(open(os.path.join(outg, "gate_items.json"))); items = gi["items"]
    m, _ = VE.load("boxes/runs_fsq/s213_d3t_s1", dev)
    lab = SD.label_chains(m, [it["syms"] for it in items], dev, bs=32)
    per, by_n = [], {}
    gs2 = {"n_cmp": 0, "n_equal": 0, "diff_not_wrong_state": 0}
    for it, o in zip(items, lab):
        assert o["n"] == it["n"] == SD.n_of(it["syms"])
        tree = SD._parse_tree(it["syms"]); gold = int(N.value(tree)); kept = bool(o["keep"] and o["cv"]); rec = {"n": it["n"], "keep": o["keep"], "cv": o["cv"],
                                                                                                    "label": o["label"], "gold": gold, "state_ok": None, "first_bad_t": None}
        if kept:
            Eg = VB.recode_targets(tree); Es = SD.compose_E(it["syms"], o["steps"], it["n"])
            bad = [t for t in range(it["n"] + 1) if Es[t] != Eg[t]]; rec["state_ok"] = not bad; rec["first_bad_t"] = bad[0] if bad else None
            eq16 = SD.compose_E(it["syms"], o["steps"], 16) == VB.recode_targets(tree, 16); gs2["n_cmp"] += 1; gs2["n_equal"] += int(eq16)
            gs2["diff_not_wrong_state"] += int((not eq16) and rec["state_ok"])
        per.append(rec)
    for n in sorted({r["n"] for r in per}) + ["pooled"]:
        rs = [r for r in per if n == "pooled" or r["n"] == n]; kp = [r for r in rs if r["keep"] and r["cv"]]
        wl = sum(r["label"] != r["gold"] for r in kp); ws = sum(not r["state_ok"] for r in kp); nk = len(kp)
        by_n[str(n)] = {"items": len(rs), "n_keep": sum(r["keep"] for r in rs), "n_kept": nk, "keep_kv": nk / len(rs), "kept_wrong_label": wl, "kept_wrong_state": ws,
                        "pi": 1 - wl / nk if nk else None, "pi_E": 1 - ws / nk if nk else None, "pi_ci95": wilson(nk - wl, nk), "pi_E_ci95": wilson(nk - ws, nk)}
    pooled = by_n.pop("pooled"); ns = sorted(by_n, key=int)
    terms = {"pi_pool>=0.995": pooled["pi"] >= 0.995, "pi_n>=0.99_all": all(by_n[n]["pi"] >= 0.99 for n in ns), "pi_E_pool>=0.99": pooled["pi_E"] >= 0.99,
             "keep_n>=0.70_all": all(by_n[n]["keep_kv"] >= 0.70 for n in ns)}
    gs2["PASS"] = gs2["diff_not_wrong_state"] == 0 and gs2["n_cmp"] - gs2["n_equal"] == pooled["kept_wrong_state"]
    res = {"by_n": by_n, "pooled": pooled, "terms": terms, "GATE_1": all(terms.values()), "gs2_compose": gs2, "items": per, "gate_seed": gi["seed"],
           "labeller": "boxes/runs_fsq/s213_d3t_s1/model.pt", "card": (torch.cuda.get_device_name(dev) if dev.startswith("cuda") else dev), "dtype": "fp32", "bs": 32, "sec": round(time.time() - t0),
           "def": __doc__}
    json.dump(res, open(os.path.join(outg, "eval_gate.json"), "w", newline="\n"), indent=1)
    for n in ns:
        print("GATE1 n", n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in by_n[n].items() if not k.endswith("ci95")}, flush=True)
    print("GATE1 pooled", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pooled.items() if not k.endswith("ci95")}, "terms", terms, "GATE_1", res["GATE_1"],
          "G-S2", gs2, flush=True)


if __name__ == "__main__":
    main()
