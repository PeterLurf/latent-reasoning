"""§219 TEXT-SELF window gates (B2; TS_v1.md at 146a25d0 §2.2, §7 G-T2 / G-T15 / G-T16; from scratch/s216_gate2.py). Local RTX 3070, fp32, tf32 off.
Two stages so that gold is scored in a separate process from the labelling:
  --stage label --labeller primary|fallback (gold patched; no gold anywhere): <gate dir>/gate_items.json -> the labeller's
     TF-GREEDY REROOT on every gate item (label batch 128; the registered labeller), TF-GREEDY FULL on every item (LABELLER-FULL, descriptive), and
     G-T16 EQUIV on the first 250 items per n (kept and rejected): per-step REROOT KV-cache generation (textself.gen_reroot) and FULL generation
     (textself.gen_full) -> <gate dir>/gate_labels_<primary|fallback>.json (per window, written as each window finishes). Round 2 needs
     s219_t1_s1/eval_text.json first (the seal, V3); --windows limits the fallback to the failed windows.
  --stage score (CPU, gold): gate_labels_* + gate_items -> <gate dir>/eval_gate.json: by_n {n: {window, labeller, items, n_kept, keep, kept_wrong_label,
     kept_wrong_E, pi, pi_E, pi_ci95, pi_E_ci95, records}}, pooled_by_window, GATE_T1 / GATE_T2 / GATE_T1p (the primary labeller), fallback {W: ...},
     window_labeller {W: run | None}, labeller_full {W: {n: keep, pi}}, kept_vs_drawn {n: {kept, drawn: mean P + C, height, brackets}}, equiv {n: ...,
     PASS (>= 0.995 of items identical in keep flag, every value and the failing step)}, equiv_full (descriptive), compose (G-T2 on the gate items:
     compose_T == the independently GENERATED tokens on every generation-kept EQUIV item; compose_T == N.cot_ids(gold) iff every value is gold, on every
     TF-kept item), card.
  Rule per window (= GATE-1's, SD2 count form): 200 * wrong_pool <= kept_pool and 100 * wrongE_pool <= kept_pool and, at every n, 100 * wrong_n <= kept_n
     and 10 * kept_n >= 7 * items_n (pi_pool >= 0.995, pi^E_pool >= 0.99, pi_n >= 0.99, keep_n >= 0.70).
    python -B scratch/s219_gate.py --round 1 --stage label --labeller primary --device cuda:1
    python -B scratch/s219_gate.py --round 1 --stage score
  Smokes: --out_root scratch/_tmp/s219/<tag>/ --labeller_override <run dir> [--per_n_cap 10]."""
import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.textself as TS  # noqa: E402
from s219_build_pool import cfg_for  # noqa: E402


def label_stage(a, cfg):
    TS.patch_gold()
    if a.round == 2 and not a.out_root:
        assert os.path.exists(os.path.join(TS.ROUNDS[1]["cell"], "eval_text.json")), "§219 V3: GATE-T2 / GATE-T1' only after s219_t1_s1/eval_text.json"
    run = a.labeller_override or (cfg["labeller"] if a.labeller == "primary" else cfg["fallback"]); assert run, "no %s labeller for round %d" % (a.labeller, a.round)
    g, sha = TS.load_model(os.path.join(run, "model.pt"), a.device); gi = json.load(open(os.path.join(cfg["gate_dir"], "gate_items.json")))
    outp = os.path.join(cfg["gate_dir"], "gate_labels_%s.json" % a.labeller)
    res = {"labeller": run, "labeller_sha256": sha, "windows": {}, "card": torch.cuda.get_device_name(a.device) if a.device.startswith("cuda") else "cpu",
           "t_start": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    wsel = [w for w in a.windows.split(",") if w] or list(gi["windows"])
    for w in wsel:
        wd = gi["windows"][w]; its = wd["items"]
        if a.per_n_cap:
            its = [it for n in wd["ns"] for it in [x for x in its if x["n"] == n][:a.per_n_cap]]
        sy = [it["syms"] for it in its]; t0 = time.time()
        tf = TS.label_reroot(g, sy, a.label_bs); t_tf = time.time() - t0; t0 = time.time()
        full = TS.label_full(g, sy, a.label_bs) if not a.skip_full else None; t_full = time.time() - t0
        eq_idx = [i for n in wd["ns"] for i in [q for q, it in enumerate(its) if it["n"] == n][:TS.EQUIV_PER_N]]; t0 = time.time()
        ge = TS.gen_reroot(g, [sy[i] for i in eq_idx]) if not a.skip_equiv else None; t_ge = time.time() - t0; t0 = time.time()
        gf = TS.gen_full(g, [sy[i] for i in eq_idx]) if not (a.skip_equiv or a.skip_full) else None; t_gf = time.time() - t0
        res["windows"][w] = {"ns": wd["ns"], "n_items": len(its), "items_used": [it["syms"] for it in its] if a.per_n_cap else None,
                             "tf": [{k: o[k] for k in ("keep", "values", "fail_step", "fail", "min_margin")} for o in tf],
                             "full": [{k: o[k] for k in ("keep", "values", "fail_step", "min_margin")} for o in full] if full else None,
                             "equiv_idx": eq_idx, "gen": [{k: o[k] for k in ("keep", "values", "fail_step", "fail_flags", "steps", "min_margin")} for o in ge] if ge else None,
                             "gen_full": [{k: o[k] for k in ("keep", "values", "fail_step", "answer", "ntok", "min_margin")} for o in gf] if gf else None,
                             "sec": {"tf": round(t_tf), "full": round(t_full), "gen": round(t_ge), "gen_full": round(t_gf)}}
        json.dump(res, open(outp + ".tmp", "w", newline="\n")); os.replace(outp + ".tmp", outp)
        print("LABELLED", w, "items", len(its), "tf keep %.4f" % np.mean([o["keep"] for o in tf]), res["windows"][w]["sec"], flush=True)
    res.update({"dtype": "fp32", "tf32": False, "label_bs": a.label_bs}); json.dump(res, open(outp + ".tmp", "w", newline="\n")); os.replace(outp + ".tmp", outp)


def _stats(recs):
    kp = [r for r in recs if r["keep"]]; wl = sum(r["label"] != r["gold"] for r in kp); we = sum(not r["values_ok"] for r in kp); nk = len(kp)
    return {"items": len(recs), "n_kept": nk, "keep": nk / len(recs) if recs else None, "kept_wrong_label": wl, "kept_wrong_E": we, "pi": 1 - wl / nk if nk else None,
            "pi_E": 1 - we / nk if nk else None, "pi_ci95": TS.wilson(nk - wl, nk), "pi_E_ci95": TS.wilson(nk - we, nk)}


def _window(recs, ns, lab):
    by_n = {str(n): dict(_stats([r for r in recs if r["n"] == n]), labeller=lab, records=[r for r in recs if r["n"] == n]) for n in ns}; po = _stats(recs)
    ok = (po["n_kept"] > 0 and 200 * po["kept_wrong_label"] <= po["n_kept"] and 100 * po["kept_wrong_E"] <= po["n_kept"]
          and all(by_n[str(n)]["n_kept"] > 0 and 100 * by_n[str(n)]["kept_wrong_label"] <= by_n[str(n)]["n_kept"] and 10 * by_n[str(n)]["n_kept"] >= 7 * by_n[str(n)]["items"] for n in ns))
    return by_n, po, bool(ok)


def _recs(its, tf, gold):
    out = []
    for i, (it, o) in enumerate(zip(its, tf)):
        gv = gold[i]; r = {"i": i, "n": it["n"], "keep": o["keep"], "label": o["values"][-1] if o["keep"] else None, "gold": gv[-1], "values_ok": (o["values"] == gv) if o["keep"] else None,
                           "first_bad_step": next((k + 1 for k, (v, g_) in enumerate(zip(o["values"], gv)) if v != g_), None) if o["keep"] else None, "fail_step": o["fail_step"],
                           "fail_region": (o.get("fail") or {}).get("region")}
        out.append(r)
    return out


def score_stage(a, cfg):
    gi = json.load(open(os.path.join(cfg["gate_dir"], "gate_items.json"))); prim = json.load(open(os.path.join(cfg["gate_dir"], "gate_labels_primary.json")))
    fbp = os.path.join(cfg["gate_dir"], "gate_labels_fallback.json"); fb = json.load(open(fbp)) if os.path.exists(fbp) else None
    res = {"round": a.round, "by_n": {}, "pooled_by_window": {}, "fallback": None, "window_labeller": {}, "labeller_full": {}, "kept_vs_drawn": {}, "equiv": {}, "equiv_full": {},
           "compose": {}, "labeller": prim["labeller"], "labeller_sha256": prim["labeller_sha256"], "card": prim["card"]}
    for w, wd in gi["windows"].items():
        if w not in prim["windows"]:
            continue
        L = prim["windows"][w]; its = [{"syms": s, "n": TS.n_of(s)} for s in L["items_used"]] if L.get("items_used") else wd["items"]
        gold = [TS.gold_values(it["syms"]) for it in its]; recs = _recs(its, L["tf"], gold); by_n, po, ok = _window(recs, wd["ns"], prim["labeller"])
        for n, v in by_n.items():
            res["by_n"][n] = dict(v, window=w)
        res["pooled_by_window"][w] = dict(po, labeller=prim["labeller"]); res[wd["gate"]] = ok; res["window_labeller"][w] = prim["labeller"] if ok else None
        if L.get("full"):                                                   # LABELLER-FULL (descriptive)
            for n in wd["ns"]:                                              # flat {n: ...} (score_s219 LAYOUTS; n is unique across windows)
                ix = [i for i, it in enumerate(its) if it["n"] == n]; kp = [i for i in ix if L["full"][i]["keep"]]
                res["labeller_full"][str(n)] = {"window": w,"keep": len(kp) / len(ix), "pi": (1 - sum(L["full"][i]["values"][-1] != gold[i][-1] for i in kp) / len(kp)) if kp else None,
                                                   "pi_E": (1 - sum(L["full"][i]["values"] != gold[i] for i in kp) / len(kp)) if kp else None, "n_kept": len(kp)}
        for n in wd["ns"]:                                                  # KEPT-VS-DRAWN
            ix = [i for i, it in enumerate(its) if it["n"] == n]; kp = [i for i in ix if L["tf"][i]["keep"]]
            def desc(sel):
                if not sel:
                    return None
                tr = [TS.parse_tree(its[i]["syms"]) for i in sel]
                return {"mean_P_plus_C": float(np.mean([len(its[i]["syms"]) + 2 + TS.c_struct(its[i]["syms"]) for i in sel])), "mean_height": float(np.mean([TS.height(t) for t in tr])),
                        "mean_brackets": float(np.mean([its[i]["syms"].count(" (") for i in sel])), "items": len(sel)}
            res["kept_vs_drawn"][str(n)] = {"kept": desc(kp), "drawn": desc(ix)}
        cmp_ = {"n_kept": 0, "gen_tokens_eq_compose": 0, "gen_kept": 0, "gold_eq_iff_values_gold": 0}
        for i, (it, o) in enumerate(zip(its, L["tf"])):                    # G-T2 on the gate items (gold side here)
            if o["keep"]:
                cmp_["n_kept"] += 1; ct = TS.compose_T(it["syms"], o["values"]); cg = N.cot_ids(TS.parse_tree(it["syms"]))
                cmp_["gold_eq_iff_values_gold"] += int((ct == cg) == (o["values"] == gold[i]))
        if L.get("gen"):
            eq = {}; dis = []
            for q, i in enumerate(L["equiv_idx"]):
                a_, b_ = L["tf"][i], L["gen"][q]; n = its[i]["n"]; e = eq.setdefault(str(n), {"items": 0, "agree_keep": 0, "agree_values": 0, "agree_fail_step": 0, "agree_all": 0})
                ak = a_["keep"] == b_["keep"]; av = a_["values"] == b_["values"]; af = a_["fail_step"] == b_["fail_step"]
                e["items"] += 1; e["agree_keep"] += int(ak); e["agree_values"] += int(av); e["agree_fail_step"] += int(af); e["agree_all"] += int(ak and av and af)
                if b_["keep"]:
                    cmp_["gen_kept"] += 1; toks = [t for s in b_["steps"] for t in s]; cmp_["gen_tokens_eq_compose"] += int(toks == TS.compose_T(its[i]["syms"], b_["values"]))
                if not (ak and av and af):
                    dis.append({"i": i, "n": n, "tf": [a_["keep"], a_["fail_step"], a_["values"]], "gen": [b_["keep"], b_["fail_step"], b_["values"]], "gen_min_margin": b_["min_margin"],
                                "tf_min_margin": a_["min_margin"]})
            for n, e in eq.items():
                e["share"] = e["agree_all"] / e["items"]; e["PASS"] = bool(e["share"] >= 0.995)
            mg = [o["min_margin"] for o in L["gen"]]
            res["equiv"][w] = {"by_n": eq, "disagreements": dis, "min_margin": float(min(mg)) if mg else None, "PASS": bool(eq and all(e["PASS"] for e in eq.values()))}
        if L.get("gen_full") and L.get("full"):
            ef = {"items": 0, "agree_keep_values": 0}
            for q, i in enumerate(L["equiv_idx"]):
                ef["items"] += 1; ef["agree_keep_values"] += int(L["full"][i]["keep"] == L["gen_full"][q]["keep"] and L["full"][i]["values"] == L["gen_full"][q]["values"])
            res["equiv_full"][w] = ef
        cmp_["PASS"] = cmp_["gold_eq_iff_values_gold"] == cmp_["n_kept"] and cmp_["gen_tokens_eq_compose"] == cmp_["gen_kept"]; res["compose"][w] = cmp_
        print("WINDOW", w, "pass", ok, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in po.items() if not k.endswith("ci95")}, flush=True)
        if not ok and fb and w in fb["windows"]:                            # the pre-stated BASE-LABELLED fallback, same items, separate entry
            Lb = fb["windows"][w]; rb = _recs(its, Lb["tf"], gold); by_nb, pob, okb = _window(rb, wd["ns"], fb["labeller"])
            res["fallback"] = res["fallback"] or {}; res["fallback"][w] = {"labeller": fb["labeller"], "labeller_sha256": fb["labeller_sha256"], "by_n": by_nb, "pooled": pob, "GATE": okb}
            res["window_labeller"][w] = fb["labeller"] if okb else None; print("FALLBACK", w, fb["labeller"], "pass", okb, flush=True)
    res.update({"dtype": "fp32", "label_bs": prim.get("label_bs"), "t_start": min([x for x in (prim.get("t_start"), (fb or {}).get("t_start")) if x] or [None]), "def": __doc__}); json.dump(res, open(os.path.join(cfg["gate_dir"], "eval_gate.json"), "w", newline="\n"), indent=1)
    print("GATES", {k: v for k, v in res.items() if k.startswith("GATE_")}, "window_labeller", res["window_labeller"], "equiv",
          {w: v["PASS"] for w, v in res["equiv"].items()}, "compose", {w: v["PASS"] for w, v in res["compose"].items()}, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--round", type=int, required=True, choices=[1, 2]); ap.add_argument("--stage", required=True, choices=["label", "score"])
    ap.add_argument("--labeller", default="primary", choices=["primary", "fallback"]); ap.add_argument("--windows", default=""); ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--label_bs", type=int, default=TS.LABEL_BS); ap.add_argument("--skip_full", action="store_true", help="smokes only"); ap.add_argument("--skip_equiv", action="store_true", help="smokes only")
    ap.add_argument("--out_root", default="", help="smokes only"); ap.add_argument("--labeller_override", default="", help="smokes only"); ap.add_argument("--per_n_cap", type=int, default=0, help="smokes only")
    a = ap.parse_args()
    if a.out_root or a.labeller_override or a.per_n_cap or a.skip_full or a.skip_equiv:
        assert a.out_root and "scratch" in a.out_root.replace("\\", "/").split("/"), "smoke options need --out_root under scratch/"
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cfg = cfg_for(a.round, a.out_root)
    (label_stage if a.stage == "label" else score_stage)(a, cfg)


if __name__ == "__main__":
    main()
