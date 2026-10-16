"""VB-2G Stage 0, note 211-15 C / S's D1 (user-approved Stage 0, $0, eval-only) on an INJECT nested170 model with §175's aux decoder:
(1) in-trajectory rows (187-3 style): test items n 3..8, 300 per n (rng 21800 permutation of each n), passes t = 1..n of the model's own
    rollout; correct = the aux decoder's teacher-forced argmax at the step-t HANDLE token (halt173.deepaux_fields' handle role), with S's
    features of step t (docs/research_bottleneck_2026-10-16/scripts/step/feat.py steps_features: t, hh, alive, held, prov, age_max, bdepth, ...);
(2) RE-ROOT rows: the same item reduced t - 1 times (cot_steps(root)[t - 1]) run as a FRESH problem; correct = pass-1 handle token; keyed by the
    original step's hh (and t).
The model's own settings: bf16 autocast on cards with capability >= 8 (as its eval), fp32 memory branch, --bs (default 250 = its eval_bs), deltamem.FAST_ADD
from args; card / bs / dtype recorded. -> <run>/eval_injectD1.json {"rows": [...], "reroot": [...], "summary": {...}}
    python scratch/inject_d1.py s185_rms_s3 --device cuda:0"""
import argparse
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.deltamem as DM  # noqa: E402
import boxes.halt173 as H  # noqa: E402
import boxes.nested170 as N  # noqa: E402
from boxes.vbank_diag import reduce_k  # noqa: E402


def feat_mod():
    sp = importlib.util.spec_from_file_location("s_feat", "D:/Latent/docs/research_bottleneck_2026-10-16/scripts/step/feat.py")
    md = importlib.util.module_from_spec(sp); sp.loader.exec_module(md); return md


def handle_rows(m, items, T, dev, ctx, bs):
    """per item: {t: 1/0} handle-token correctness at passes t = 1..min(T, #steps) (halt173.deepaux_fields' handle logic)."""
    out = [dict() for _ in items]
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; x, msk = m.think_inputs(ch, dev)
        with torch.no_grad(), ctx():
            S = m.mem.zero_state(len(ch), dev)
            for t in range(1, T + 1):
                S = m.write(S, x, msk)
                idx = [j for j, it in enumerate(ch) if len(N.aux_of(it)) >= t]
                if not idx:
                    continue
                sub = [ch[j] for j in idx]; lg_a, y_a = m.aux_logits(S[idx], sub, t, dev); pr = lg_a.argmax(-1).cpu().numpy(); yy = y_a.cpu().numpy()
                for r_, it in enumerate(sub):
                    sy, hi = N.aux_of(it)[t - 1]; out[i + idx[r_]][t] = int(pr[r_, hi] == yy[r_, hi])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--per_n", type=int, default=300); a = ap.parse_args(); dev = a.device; rd = os.path.join("D:/Latent/boxes/runs_fsq", a.run); t0 = time.time()
    kind, m, ra = H.load_model(rd, dev); assert kind == "nested" and ra.get("aux_step"), "D1 needs a nested INJECT run with §175's aux decoder"
    DM.FAST_ADD = bool(ra.get("fast_add")); DM.CHUNK_DELTA = 0; DM.READ_CLAMP = None
    amp = torch.bfloat16 if (dev.startswith("cuda") and torch.cuda.get_device_capability(dev)[0] >= 8) else None
    from contextlib import nullcontext
    ctx = (lambda: torch.autocast("cuda", dtype=amp)) if amp is not None else nullcontext
    F_ = feat_mod(); TE = N.test_set(int(ra.get("per_n") or 1000)); items = []
    for n in range(3, 9):
        pool = [it for it in TE if it["n"] == n]; items += [pool[i] for i in np.random.default_rng(21800 + n).permutation(len(pool))[:a.per_n]]
    hr = handle_rows(m, items, 8, dev, ctx, a.bs); rows = []; fresh = []; meta = []
    for k_, it in enumerate(items):
        fts = F_.steps_features(it["root"])
        for t in range(1, it["n"] + 1):
            f_ = fts[t - 1]
            rows.append({"item_n": it["n"], "item": k_, "t": t, "hh": f_["hh"], "alive": f_["alive"], "held": f_["held"], "prov": f_["prov"], "age_max": f_["age_max"],
                         "bdepth": f_["bdepth"], "age_min": f_["age_min"], "r_left": f_["r"], "op": f_["op"], "correct": hr[k_].get(t)})
            r_ = reduce_k(it["root"], t - 1); s_ = N.render(r_)
            fresh.append({"syms": s_, "root": r_, "n": it["n"] - (t - 1), "t": it["t"], "depth": N.depth(s_)}); meta.append((k_, t, f_["hh"]))
    fr = handle_rows(m, fresh, 1, dev, ctx, a.bs)
    reroot = [{"item_n": items[k_]["n"], "item": k_, "t": t, "hh": hh, "correct": fr[j].get(1)} for j, (k_, t, hh) in enumerate(meta)]

    def by_hh(rs):
        d = {}
        for r_ in rs:
            if r_["correct"] is None:
                continue
            c = d.setdefault(str(min(r_["hh"], 5)), [0, 0]); c[0] += 1; c[1] += r_["correct"]
        return {k: {"n": v[0], "acc": v[1] / v[0]} for k, v in sorted(d.items())}
    res = {"rows": rows, "reroot": reroot, "summary": {"in_trajectory_by_hh": by_hh(rows), "reroot_by_hh": by_hh(reroot), "n_items": len(items)},
           "run": a.run, "card": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "bs": a.bs, "dtype": "bf16 autocast trunk, fp32 memory branch" if amp else "fp32",
           "fast_add": DM.FAST_ADD, "sec": round(time.time() - t0),
           "def": "211-15 C / S D1: in-trajectory rows = the aux decoder's teacher-forced handle token at pass t (halt173 deepaux handle role), n 3..8, 300 per n (rng 21800 + n); "
                  "features = feat.steps_features(root)[t-1]; reroot = the item reduced t - 1 times run fresh, pass-1 handle; summary by hh class (>= 5 pooled)"}
    json.dump(res, open(os.path.join(rd, "eval_injectD1.json"), "w", newline="\n"))
    print("D1", a.run, "traj", {k: round(v["acc"], 4) for k, v in res["summary"]["in_trajectory_by_hh"].items()}, "reroot", {k: round(v["acc"], 4) for k, v in res["summary"]["reroot_by_hh"].items()}, res["sec"], flush=True)


if __name__ == "__main__":
    main()
