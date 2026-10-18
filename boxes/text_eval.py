"""§219 TEXT-SELF evals (TS_v1.md at 146a25d0: §2.7 E1 / E2, §2.8, G-T18; Addendum 2 item 1). Local RTX 3070, fp32, tf32 off. Gold only here.
  --part text  -> <run>/eval_text.json (E1): test_set(1000, n_max 24), n 1..24 (24,000 items, test seed 17100) in test order. Free greedy generation, KV cache,
     nested170.cot_answers' loop semantics (batches of equal prompt length in item order, cot_eval_bs 32, ASSERTED), fp32, tf32 off. Budget per batch
     min(max C_struct + 50, 1025 - P), C_struct = the shape-only chain length (textself.c_struct; cot_ids is never called); the loop breaks before feeding
     row 1,024. CTX-OVERFLOW: P + C_struct > 1025 -> not generated, a failure. Non-termination (no EOT within the budget, or a tail that is not one number)
     -> answer -1. Per-item store {idx, n, y, P, C_struct, overflow, answer, ntok, eot, form_valid, strict, first_bad_step, first_bad_pos}; aggregates by n;
     meta {run, model, model_sha256, card, dtype, tf32, cot_eval_bs, test_seed, test_n_max, n_items, items_sha256, budget_rule, commit, sec, peak_gib}.
     items_sha256 = sha256 of json.dumps([[syms, int(t)] ...], separators=(",", ":"), ensure_ascii=True) over the 24,000 items (SD2_v1 §4.2).
  --part textself -> <run>/eval_textself.json (E2; only after <run>/eval_text.json exists, V3): CPU parts (--parts cpu: label_audit, contamination; the
     round-2 condition reads these, V4) and GPU parts (--parts gpu: memo, labeller_kept; descriptive), each its own key, merged into the file.
  --part smoke -> G-T18 on fresh n-24 shapes (textself.gt18_items, rng 23195; never test items), written under --out_dir (never a cell dir).
Definitions / ambiguities resolved here (TS_v1 silent; nearest precedent followed):
  A1 overflow items are dropped BEFORE batching (they are "not generated"), so a batch never carries an overflow row; batches are otherwise
     cot_answers' (dict of len(syms) in first-seen item order, chunks of cot_eval_bs).
  A2 first_bad_pos = P + j + 1, j = the index (0-based, among the generated tokens) of the first token that leaves every CV-T continuation (the
     candidate test of textself.chain_form; a chain valid up to the budget without its final EOT is bad at j = the number generated); i.e. the 1-based
     sequence row of that token, comparable to T_train = max(P + C) (score_s219 LAYOUTS). first_bad_step is 1-based. Both null for form-valid chains and overflow items.
  A3 form_valid = the whole generation is a CV-T chain with exactly n steps ending in EOT (gold-free); strict = form_valid and answer == y.
  A4 LABEL-AUDIT-T kept-wrong record = {syms, n, step (first wrong step, 1-based), op, a, b (the handle's operands at that step; earlier steps are gold
     there), value (the labeller's), gold (apply(op, a, b)), wrong_steps (all), label, gold_label, pattern (label / values / both)}; consumed =
     pool[:cursor], cursor = the run's last logged pool_cursor (no fallback).
  A5 CONTAMINATED is read on the WHOLE pool file (TS_v1 §2.4 item 8) against textself.exclusion_sets(own pool excluded, every gate included); the consumed
     share's hits are reported beside; self_reuse = the max logged; the flag is any hit or reuse > 0.
  A6 LABELLER-KEPT-T fresh items: one stream default_rng(kept seed), n 9..24 in order, 500 per n, redrawn on a hit against the full exclusion set (every
     pool and gate incl. the round's own); labelled by TF-GREEDY REROOT at label batch 128. They are not added to later exclusion sets (TS_v1 lists only
     gates and pools); the redraw rule keeps them disjoint from every pool built before them.
  A7 MEMO-GAP-T: per window n, free generation (the eval settings) on the first 2,000 consumed pool records of n (pool order) vs the round's 2,000 gate
     items of n; gold y = the gold value of the syms; flag |gap| > 0.02.
    python -m boxes.text_eval --run boxes/runs_fsq/s219_base_s1 --part text --cot_eval_bs 32 --device cuda:1"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes.nested170 as N  # noqa: E402
import boxes.textself as TS  # noqa: E402

EOT, EQ = TS.EOT, TS.EQ
BUDGET_RULE = "per batch min(max C_struct + 50, 1025 - P); C_struct shape-only (textself.c_struct); break before feeding row 1,024; overflow (P + C_struct > 1025) not generated"


def items_sha256(items):
    return hashlib.sha256(json.dumps([[it["syms"], int(it["t"])] for it in items], separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def iso():
    """the seal-order time fields (score_s219 _time: ISO with an offset)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:                                                       # noqa: BLE001
        return None


@torch.no_grad()
def free_generate(g, syms_list, bs, times=None):
    """the eval loop (A1): -> per item {overflow, gen (tokens incl. EOT if emitted), budget}, P list, C list."""
    dev = g.w("wte.weight").device; P = [len(s) + 2 for s in syms_list]; C = [TS.c_struct(s) for s in syms_list]; out = [None] * len(syms_list); by_len = {}
    for i, s in enumerate(syms_list):
        if P[i] + C[i] > TS.CTX:
            out[i] = {"overflow": True, "gen": [], "budget": 0}; continue
        by_len.setdefault(len(s), []).append(i)
    for L, idx in by_len.items():
        for s0 in range(0, len(idx), bs):
            ii = idx[s0:s0 + bs]; P0 = L + 2; budget = min(max(C[j] for j in ii) + 50, TS.CTX - P0); t0 = time.time()
            outs, _, fed = TS.greedy_batch(g, [[EOT] + N.ids(syms_list[j]) + [EQ] for j in ii], budget)
            assert fed <= 1023, "row 1,024 fed"
            for r, j in enumerate(ii):
                out[j] = {"overflow": False, "gen": outs[r], "budget": budget}
            if times is not None:
                times.append({"P": P0, "rows": len(ii), "budget": budget, "ntok_max": max(len(o) for o in outs), "sec": time.time() - t0})
            if str(dev).startswith("cuda"):
                torch.cuda.empty_cache()
    return out, P, C


def item_record(j, it, o, P, C):
    gen = o["gen"]; eot = EOT in gen; ntok = (gen.index(EOT) + 1) if eot else len(gen); y = int(it["t"])
    if o["overflow"]:
        fv, fbs, fbp, ans = False, None, None, -1
    else:
        fv, _, fbs, fbi = TS.chain_form(it["syms"], gen); fbp = None if fbi is None else P + fbi + 1; ans = TS.cot_answer(gen)
    return {"idx": j, "n": it["n"], "y": y, "P": P, "C_struct": C, "overflow": bool(o["overflow"]), "answer": int(ans), "ntok": int(ntok), "eot": bool(eot),
            "form_valid": bool(fv), "strict": bool(fv and ans == y), "first_bad_step": fbs, "first_bad_pos": fbp}


def aggregate(recs, ns):
    q = lambda v: [float(np.quantile(v, p)) for p in (0.1, 0.25, 0.5, 0.75, 0.9)] if v else None  # noqa: E731
    A = {k: {} for k in ("acc_by_n", "acc_feasible_by_n", "form_valid_by_n", "strict_by_n", "ntok_mean_by_n", "eot_by_n", "overflow_by_n", "first_bad_pos_q_by_n",
                         "first_bad_step_q_by_n", "chance_by_n", "items_by_n")}
    for n in ns:
        rs = [r for r in recs if r["n"] == n]; fe = [r for r in rs if not r["overflow"]]; k = str(n)
        if not rs:
            continue
        A["acc_by_n"][k] = float(np.mean([r["answer"] == r["y"] for r in rs])); A["acc_feasible_by_n"][k] = float(np.mean([r["answer"] == r["y"] for r in fe])) if fe else None
        A["form_valid_by_n"][k] = float(np.mean([r["form_valid"] for r in rs])); A["strict_by_n"][k] = float(np.mean([r["strict"] for r in rs]))
        A["ntok_mean_by_n"][k] = float(np.mean([r["ntok"] for r in fe])) if fe else None; A["eot_by_n"][k] = float(np.mean([r["eot"] for r in fe])) if fe else None
        A["overflow_by_n"][k] = sum(r["overflow"] for r in rs); A["items_by_n"][k] = len(rs)
        A["first_bad_pos_q_by_n"][k] = q([r["first_bad_pos"] for r in rs if r["first_bad_pos"] is not None])
        A["first_bad_step_q_by_n"][k] = q([r["first_bad_step"] for r in rs if r["first_bad_step"] is not None])
        _, cnt = np.unique([r["y"] for r in rs], return_counts=True); A["chance_by_n"][k] = float(cnt.max() / cnt.sum())
    A["quantiles"] = [0.1, 0.25, 0.5, 0.75, 0.9]
    return A


def card(dev):
    return torch.cuda.get_device_name(dev) if str(dev).startswith("cuda") else "cpu"


def dump(obj, path):
    json.dump(obj, open(path + ".tmp", "w", newline="\n"), indent=1); os.replace(path + ".tmp", path)


def part_text(g, msha, mpath, run, dev, bs, out_path, ns, limit):
    assert bs == 32, "§219 Addendum 2 item 1: the registered eval_text runs at cot_eval_bs 32"
    t0 = time.time(); ts0 = iso(); TD = N.test_set(1000, n_max=24); full = [it for it in TD]; ish = items_sha256(full)
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
    recs = []; times = []
    for n in ns:
        js = [j for j, it in enumerate(TD) if it["n"] == n][:limit or None]; its = [TD[j] for j in js]
        out, P, C = free_generate(g, [it["syms"] for it in its], bs, times)
        recs += [item_record(j, it, o, p_, c_) for j, it, o, p_, c_ in zip(js, its, out, P, C)]
        res = aggregate(recs, ns); print("TEXT n", n, "acc %.3f" % res["acc_by_n"][str(n)], "form %.3f" % res["form_valid_by_n"][str(n)], "overflow", res["overflow_by_n"][str(n)],
                                        "%.0f s" % (time.time() - t0), flush=True)
        dump({"partial_through_n": n, **res, "items": recs}, out_path.replace(".json", ".partial.json"))
    res = aggregate(recs, ns); res["items"] = recs; res["batch_times"] = times
    res["meta"] = {"run": run, "model": mpath, "model_sha256": msha, "card": card(dev), "dtype": "fp32", "tf32": False, "cot_eval_bs": bs, "test_seed": 17100, "test_n_max": 24,
                   "ns": list(ns), "limit_per_n": limit, "n_items": len(recs), "items_sha256": ish, "items_sha256_def": "json.dumps([[syms, int(t)]] over test_set(1000, 24), "
                   "separators (',', ':'), ensure_ascii) -> sha256", "budget_rule": BUDGET_RULE, "commit": commit(), "sec": round(time.time() - t0), "t_start": ts0, "t_end": iso(),
                   "peak_gib": (torch.cuda.max_memory_allocated(dev) / 2 ** 30) if str(dev).startswith("cuda") else None}
    res["def"] = __doc__
    return res


def part_smoke(g, dev, bs, rows):
    """G-T18: (1) the `rows` longest feasible distinct rng-23195 n-24 items in eval batching; (2) the cap-bound batch = `rows` copies of the longest feasible
    item; peaks and times; the eval projection from (2)'s per-decode-step time over the registered eval's batches (budgets, shape only)."""
    t0 = time.time(); its = TS.gt18_items(); P = [len(s) + 2 for s in its]; C = [TS.c_struct(s) for s in its]
    feas = [i for i in range(len(its)) if P[i] + C[i] <= TS.CTX]; lon = sorted(feas, key=lambda i: -(P[i] + C[i]))[:rows]; res = {"items": len(its), "feasible": len(feas)}
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
    t1 = []; o1, _, _ = free_generate(g, [its[i] for i in lon], bs, t1)
    res["longest"] = {"idx": lon, "P_plus_C": [P[i] + C[i] for i in lon], "batches": t1, "ntok": [len(o["gen"]) for o in o1],
                      "peak_gib": (torch.cuda.max_memory_allocated(dev) / 2 ** 30) if str(dev).startswith("cuda") else None}
    w = lon[0]; budget = min(C[w] + 50, TS.CTX - P[w])
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
    ts = time.time(); out, _, fed = TS.greedy_batch(g, [[EOT] + N.ids(its[w]) + [EQ]] * rows, budget); sec = time.time() - ts
    ntok = max(len(o) for o in out)
    res["cap_bound"] = {"idx": w, "P": P[w], "C_struct": C[w], "budget": budget, "cap_bound": bool(C[w] + 50 > TS.CTX - P[w]), "rows": rows, "fed_max_row": fed,
                        "ntok_max": ntok, "sec": sec, "sec_per_decode_step": sec / max(1, ntok),
                        "peak_gib": (torch.cuda.max_memory_allocated(dev) / 2 ** 30) if str(dev).startswith("cuda") else None}
    TD = N.test_set(1000, n_max=24); steps_ub = steps_c = 0
    for n in range(1, 25):                                                  # the registered eval's batches (shape only)
        by = {}
        for it in TD:
            if it["n"] == n and len(it["syms"]) + 2 + TS.c_struct(it["syms"]) <= TS.CTX:
                by.setdefault(len(it["syms"]), []).append(TS.c_struct(it["syms"]))
        for L, cs in by.items():
            for s0 in range(0, len(cs), 32):
                ch = cs[s0:s0 + 32]; steps_ub += min(max(ch) + 50, TS.CTX - L - 2); steps_c += max(ch)
    spd = res["cap_bound"]["sec_per_decode_step"]
    res["projection"] = {"decode_steps_upper": steps_ub, "decode_steps_cstruct": steps_c, "hours_upper": steps_ub * spd / 3600, "hours_cstruct": steps_c * spd / 3600,
                         "def": "sequential decode steps of the registered eval (per batch: the budget = upper, max C_struct = expected for a terminating batch) x the cap-bound "
                                "batch's mean sec per decode step (context up to ~1,024: an upper per-step cost)"}
    pk = [v for v in (res["longest"]["peak_gib"], res["cap_bound"]["peak_gib"]) if v is not None]
    res["PASS_peak_le_6"] = (max(pk) <= 6.0) if pk else None; res["card"] = card(dev); res["dtype"] = "fp32"; res["cot_eval_bs"] = bs; res["sec"] = round(time.time() - t0)
    res["def"] = part_smoke.__doc__
    return res


def _last_log(run, key):
    v = None
    for ln in open(os.path.join(run, "log.jsonl")):
        try:
            r = json.loads(ln)
        except Exception:                                                   # noqa: BLE001
            continue
        if key in r and r[key] is not None:
            v = r[key] if key != "self_reuse" else max(v or 0, r[key])
    return v


def part_textself_cpu(run, cfg, pool_path, smoke_limit=None):
    """E2 CPU parts: label_audit (A4) and contamination (A5)."""
    pool, meta = TS.load_pool(pool_path); cursor = _last_log(run, "pool_cursor"); reuse = _last_log(run, "self_reuse") or 0
    assert cursor is not None and cursor > 0, ("no pool_cursor in", run)
    used = pool[:cursor]; win_of = {n: w for w, v in meta.get("windows", {}).items() for n in v["ns"]}
    sel = np.random.default_rng(cfg["audit_seed"]).choice(len(used), min(5000, len(used)), replace=False); acc = {}; kw = []
    for i in sel:
        r = used[int(i)]; gv = TS.gold_values(r["syms"]); bad = [k + 1 for k, (v, g_) in enumerate(zip(r["values"], gv)) if v != g_]; bl = r["chain_label"] != gv[-1]
        for key in ("pooled", win_of.get(r["n"], "n%d" % r["n"])):
            a_ = acc.setdefault(key, [0, 0, 0]); a_[0] += 1; a_[1] += int(bl); a_[2] += int(bool(bad))
        if bad:
            tree = TS.parse_tree(r["syms"])
            for v in r["values"][:bad[0] - 1]:
                TS.reduce_shape(tree, v)
            path = N._leftmost_ready(tree); h = tree
            for p in path:
                h = getattr(h, p)
            kw.append({"syms": r["syms"], "n": r["n"], "step": bad[0], "op": h.op, "a": h.l.v, "b": h.r.v, "v": r["values"][bad[0] - 1], "gold": gv[bad[0] - 1],
                       "wrong_steps": bad, "label": r["chain_label"], "gold_label": gv[-1], "pattern": "both" if bl else "values"})
    la = {}
    for key, (na, wl, we) in acc.items():
        blk = {"n_audit": na, "pi": 1 - wl / na, "pi_E": 1 - we / na, "pi_ci95": TS.wilson(na - wl, na), "pi_E_ci95": TS.wilson(na - we, na), "kept_wrong_label": wl, "kept_wrong_E": we}
        if key == "pooled":
            la.update(blk)
        else:
            la.setdefault("by_window", {})[key] = blk
    la.update({"kept_wrong": kw, "seed": cfg["audit_seed"], "cursor": cursor, "n_consumed": len(used), "pool_sha256": meta["sha256"]})
    S, smeta = TS.exclusion_sets(own_pool=pool_path)
    hits = {k: sum(tuple(r["syms"]) in v for r in pool) for k, v in S.items()}; hits_used = {k: sum(tuple(r["syms"]) in v for r in used) for k, v in S.items()}
    gk = 0
    with open(pool_path, encoding="utf-8") as fh:                         # raw re-read: records carrying any gold / unexpected key
        for ln in fh:
            gk += int(set(json.loads(ln)) != TS.REC_KEYS)
    con = {"pool_records": len(pool), "hits_by_set": hits, "hits_by_set_consumed": hits_used, "pool_excl_hits": sum(hits.values()), "self_reuse": reuse,
           "pool_test_hits": hits.get("test_n30", 0), "gold_key_hits": gk,
           "excl_sizes": {k: len(v) for k, v in S.items()}, "excl_meta": smeta}
    con["CONTAMINATED"] = bool(con["pool_excl_hits"] > 0 or reuse > 0 or gk > 0 or cursor > len(pool))
    print("LABEL_AUDIT", {k: v for k, v in la.items() if k not in ("kept_wrong", "by_window")}, flush=True); print("CONTAMINATION", con["pool_excl_hits"], reuse, flush=True)
    return {"label_audit": la, "contamination": con}


def part_textself_gpu(g, run, cfg, pool_path, gate_items_path, dev, bs, memo_n=2000, kept_n=500, kept_ns=range(9, 25)):
    """E2 GPU parts (descriptive): memo (A7), labeller_kept (A6)."""
    pool, meta = TS.load_pool(pool_path); cursor = _last_log(run, "pool_cursor"); used = pool[:cursor]; gi = json.load(open(gate_items_path))
    gitems = [it for w in gi["windows"].values() for it in w["items"]]; memo = {}
    for w, (_, ns, _) in cfg["windows"].items():
        for n in ns:
            pr = [r["syms"] for r in used if r["n"] == n][:memo_n]; gf = [it["syms"] for it in gitems if it["n"] == n][:memo_n]
            acc = []
            for sl in (pr, gf):
                out, _, _ = free_generate(g, sl, bs); acc.append(float(np.mean([TS.cot_answer(o["gen"]) == TS.gold_values(s)[-1] for o, s in zip(out, sl)])) if sl else None)
            gap = abs(acc[0] - acc[1]) if None not in acc else None
            memo[str(n)] = {"window": w, "acc_pool": acc[0], "acc_gate": acc[1], "items": [len(pr), len(gf)], "gap": gap, "FLAG_gt_0.02": bool(gap is not None and gap > 0.02)}
            print("MEMO n", n, memo[str(n)], flush=True)
    S, _ = TS.exclusion_sets(); allx = set().union(*S.values()); rng = np.random.default_rng(cfg["kept_seed"]); lk = {}; t_lk = iso()
    for n in kept_ns:
        its = []
        while len(its) < kept_n:
            s_ = N.render(N.generate(rng, n)[0])
            if tuple(s_) in allx:
                continue
            its.append(s_)
        lab = TS.label_reroot(g, its, TS.LABEL_BS); kept = [i for i, l_ in enumerate(lab) if l_["keep"]]
        wl = sum(lab[i]["values"] != TS.gold_values(its[i]) for i in kept); wlab = sum(lab[i]["label"] != TS.gold_values(its[i])[-1] for i in kept)
        lk[str(n)] = {"items": len(its), "n_kept": len(kept), "keep": len(kept) / len(its), "pi": (1 - wlab / len(kept)) if kept else None, "pi_E": (1 - wl / len(kept)) if kept else None,
                      "fail_step_hist": {str(s): sum(1 for l_ in lab if l_["fail_step"] == s) for s in sorted({l_["fail_step"] for l_ in lab if l_["fail_step"]})}}
        print("LABELLER_KEPT n", n, lk[str(n)], flush=True)
    return {"memo": memo, "labeller_kept": lk, "labeller_kept_meta": {"seed": cfg["kept_seed"], "per_n": kept_n, "label_bs": TS.LABEL_BS, "t_start": t_lk}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--part", required=True, choices=["text", "textself", "smoke"])
    ap.add_argument("--model", default="", help="default <run>/model.pt; s219_base_s1 -> boxes/runs_fsq/s170_cot_s1/model.pt (sha256 asserted)")
    ap.add_argument("--cot_eval_bs", type=int, default=32); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out_name", default="")
    ap.add_argument("--out_dir", default="", help="smokes only: write here (must be under scratch/) instead of the run dir")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2]); ap.add_argument("--parts", default="cpu", choices=["cpu", "gpu", "all"])
    ap.add_argument("--ns", default="", help="smokes only (with --out_dir): n subset"); ap.add_argument("--limit_per_n", type=int, default=0, help="smokes only (with --out_dir)")
    ap.add_argument("--pool", default="", help="smokes only: pool file override"); ap.add_argument("--gate_items", default="", help="smokes only: gate_items.json override")
    ap.add_argument("--smoke_rows", type=int, default=32, help="--part smoke: rows per smoke batch (registered G-T18: 32)")
    a = ap.parse_args(); dev = a.device; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    smoke = bool(a.out_dir) or a.part == "smoke"
    if a.ns or a.limit_per_n or a.pool or a.gate_items:
        assert a.out_dir, "--ns / --limit_per_n / --pool / --gate_items are smoke options: they need --out_dir under scratch/"
    if a.out_dir:
        assert os.path.normpath(a.out_dir).replace("\\", "/").split("/")[0] == "scratch" or "/scratch/" in os.path.abspath(a.out_dir).replace("\\", "/"), "--out_dir must be under scratch/"
    assert a.part != "smoke" or a.out_dir, "--part smoke writes under --out_dir only (G-T18: never a cell directory)"
    run = a.run.rstrip("/\\"); od = a.out_dir or run; os.makedirs(od, exist_ok=True)
    assert "s170_cot_s1" not in os.path.normpath(od).replace("\\", "/"), "§219: s170_cot_s1/ is never written"
    mpath = a.model or (TS.BASE_MODEL if os.path.basename(run) == "s219_base_s1" else os.path.join(run, "model.pt"))
    out_path = os.path.join(od, a.out_name or ("eval_%s.json" % ("text" if a.part == "text" else "textself" if a.part == "textself" else "text_smoke")))
    if a.part in ("text",) and os.path.exists(out_path):
        raise SystemExit("%s exists: rename it (*.void_*) before a re-run (keep-aborted-logs rule)" % out_path)
    t0 = time.time()
    if a.part == "textself":
        cfg = dict(TS.ROUNDS[a.round])
        if smoke:                                                           # smokes never draw the registered audit / LABELLER-KEPT streams
            cfg["audit_seed"] += 100_000; cfg["kept_seed"] += 100_000
        if not smoke:
            assert os.path.normpath(run) == os.path.normpath(cfg["cell"]), ("--round %d is the cell %s" % (a.round, cfg["cell"]), run)
            assert os.path.exists(os.path.join(run, "eval_text.json")), "§219 V3: eval_textself only after this model's eval_text.json"
        pool_path = a.pool or os.path.join(cfg["pool_dir"], cfg["pool_file"]); gip = a.gate_items or os.path.join(cfg["gate_dir"], "gate_items.json")
        res = json.load(open(out_path)) if os.path.exists(out_path) else {}
        if a.parts in ("cpu", "all"):
            res.update(part_textself_cpu(run, cfg, pool_path)); res["meta_cpu"] = {"commit": commit(), "sec": round(time.time() - t0), "pool": pool_path}
            dump(res, out_path)
        if a.parts in ("gpu", "all"):
            g, msha = TS.load_model(mpath, dev); t1 = time.time()
            kw = {"memo_n": a.limit_per_n or 2000, "kept_n": a.limit_per_n or 500, "kept_ns": [int(x) for x in a.ns.split(",")] if a.ns else range(9, 25)}
            res.update(part_textself_gpu(g, run, cfg, pool_path, gip, dev, a.cot_eval_bs, **kw))
            res["meta_gpu"] = {"model": mpath, "model_sha256": msha, "card": card(dev), "dtype": "fp32", "tf32": False, "cot_eval_bs": a.cot_eval_bs, "commit": commit(),
                               "sec": round(time.time() - t1)}
            dump(res, out_path)
        res["def"] = __doc__; dump(res, out_path); print("DONE", out_path, round(time.time() - t0), flush=True); return
    g, msha = TS.load_model(mpath, dev)
    if os.path.normpath(mpath).replace("\\", "/").endswith("s170_cot_s1/model.pt"):
        assert msha == TS.BASE_SHA256, ("§219 V6 base sha", msha)
    if a.part == "text":
        ns = [int(x) for x in a.ns.split(",")] if a.ns else list(range(1, 25))
        res = part_text(g, msha, mpath, run, dev, a.cot_eval_bs, out_path, ns, a.limit_per_n)
    else:
        res = part_smoke(g, dev, a.cot_eval_bs, a.smoke_rows); res["model_sha256"] = msha
    dump(res, out_path); pp = out_path.replace(".json", ".partial.json")
    if os.path.exists(pp):
        os.remove(pp)
    print("DONE", out_path, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
