"""§226 ORDERED-WORDS stage-0 build gates (registered 09368b14; Theory 10-23). Code under test: TREE/boxes/ow_words.py + TREE/boxes/codi_ps.py
(scratch/_tmp/s226_tree) until committed (then --tree D:/Latent). Each part -> scratch/gate_s226/<part>.json ("pass").
  gen_eq   dag=True == the registered prototype docs/research_ordered_words_2026-10-23/ow_words.py, item by item (every prototype key equal; only "dag",
           "wrap_final", "wrap_any" added): test_set(per_d 200, d 1..12) and 3,000 Stream draws (d 1..8), forward and topo; pair_candidates(per_d 30).
  chain    dag=False, forward, 2,000 Stream draws d 1..10 + test_set(per_d 100): no add_s / mul_s op, every src_j None, mul_s_t None, ow.solve(text) ==
           answer for every item; per-d answer max share reported.
  wrap     every gen_eq / chain item: wrap_final == (NW.apply(ops[-1], values[-2], operands[-1]) != values[-1]) recomputed here (and wrap_any likewise).
    python scratch/gate_s226.py gen_eq chain wrap [--tree DIR]"""
import importlib.util
import json
import os
import sys
from collections import Counter

TREE = "D:/Latent/scratch/_tmp/s226_tree"
for i, v in enumerate(sys.argv):
    if v == "--tree":
        TREE = sys.argv[i + 1]
OUT = "D:/Latent/scratch/gate_s226"; PROTO = "D:/Latent/docs/research_ordered_words_2026-10-23/ow_words.py"


def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


NEW = load(os.path.join(TREE, "boxes", "ow_words.py"), "ow_new"); OLD = load(PROTO, "ow_proto")
import boxes.nested_words as NW  # noqa: E402  (loaded by the modules above)
ADDED = {"dag", "wrap_final", "wrap_any"}


def same(a, b):
    """a = prototype item, b = new item -> (equal on every prototype key, added keys exactly ADDED)."""
    return all(a[k] == b[k] for k in a) and set(b) - set(a) == ADDED


def stream_items(mod, n, ds, **kw):
    s = mod.Stream(22931, **kw); out = []
    for i in range(n):
        out.append(s.draw(ds[i % len(ds)]))
    return out


def gen_eq():
    o = {}; ok = True
    for om in ("forward", "topo"):
        a = OLD.test_set(per_d=200, order_mode=om); b = NEW.test_set(per_d=200, order_mode=om)
        eq = [same(x, y) for d in a for x, y in zip(a[d], b[d])]; o["test_set_" + om] = {"n": len(eq), "equal": sum(eq)}
        a = stream_items(OLD, 3000, range(1, 9), order_mode=om); b = stream_items(NEW, 3000, range(1, 9), order_mode=om)
        eq2 = [same(x, y) for x, y in zip(a, b)]; o["stream_" + om] = {"n": len(eq2), "equal": sum(eq2)}
        ok = ok and all(eq) and all(eq2)
    a = OLD.pair_candidates(per_d=30); b = NEW.pair_candidates(per_d=30)
    eq3 = [same(x[0], y[0]) and same(x[1], y[1]) for d in a for x, y in zip(a[d], b[d])]; o["pair_candidates"] = {"n": len(eq3), "equal": sum(eq3)}
    o["pass"] = ok and all(eq3); return o


def chain():
    o = {}; its = stream_items(NEW, 2000, range(1, 11), dag=False); te = NEW.test_set(per_d=100, ds=range(1, 11), dag=False)
    allit = its + [x for d in te for x in te[d]]
    o["n"] = len(allit); o["ops_seen"] = sorted(Counter(op for it in allit for op in it["ops"]).items())
    o["bad_ops"] = sum(any(op in ("add_s", "mul_s") for op in it["ops"]) for it in allit)
    o["src_j_not_none"] = sum(any(j is not None for j in it["src_j"]) for it in allit); o["mul_s_t_set"] = sum(it["mul_s_t"] is not None for it in allit)
    sol = [NEW.solve(it["text"]) == it["answer"] for it in allit]; o["solver_agree"] = sum(sol)
    o["answer_max_share_by_d"] = {d: max(Counter(x["answer"] for x in te[d]).values()) / len(te[d]) for d in te}
    o["dag_key"] = sorted({it["dag"] for it in allit})
    o["pass"] = o["bad_ops"] == 0 and o["src_j_not_none"] == 0 and o["mul_s_t_set"] == 0 and all(sol) and o["dag_key"] == [False]; return o


def wrap():
    o = {}; bad = 0; n = 0; share = Counter()
    for kw in ({}, {"dag": False}, {"order_mode": "topo"}):
        for it in stream_items(NEW, 2000, range(1, 11), **kw):
            v, ops, b = it["values"], it["ops"], it["operands"]
            raw = [NW.apply(op, v[t], b[t]) for t, op in enumerate(ops)]
            wf = raw[-1] != v[-1]; wa = any(r != x for r, x in zip(raw, v[1:])); n += 1
            bad += (wf != it["wrap_final"]) + (wa != it["wrap_any"]); share[(str(kw), it["wrap_final"])] += 1
    o.update(n=n, mismatches=bad, wrap_final_counts={"%s %s" % k: c for k, c in share.items()}); o["pass"] = bad == 0 and n == 6000; return o


DEV = "cuda:2"
for i, v in enumerate(sys.argv):
    if v == "--device":
        DEV = sys.argv[i + 1]
TMP = "D:/Latent/scratch/_tmp/s226_gate"; P3A_ARGS = "D:/Latent/scratch/_tmp/s223_p3a/direct/args.json"
RECIPE = ["--direct", "--lr", "3e-4", "--cg_thr", "0.45", "--cg_cap", "6000", "--steps", "48000", "--stop_at", "12000", "--auto_micro", "--mem_budget", "7.0"]
CELLS = {"P0-OW": ["--task", "ow"], "P0b-TOPO": ["--task", "ow", "--order_mode", "topo"], "P0c-CHAIN": ["--task", "ow", "--dag", "off"]}


def run(tree, out, extra, det=True):
    import subprocess
    os.makedirs(out, exist_ok=True)
    for f_ in os.listdir(out):
        os.remove(os.path.join(out, f_))
    cmd = [sys.executable, "-B"] + (["D:/Latent/scratch/s222_det_wrap.py"] if det else []) + [os.path.join(tree, "boxes", "codi_ps.py"), "--out", out] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, cwd="D:/Latent", env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
    lg = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))] if os.path.exists(os.path.join(out, "log.jsonl")) else []
    return r.returncode, lg, r.stderr[-1200:]


def flags_off():
    """flags off == HEAD bitwise (det wrapper, DEV): DIRECT / M1 / M2 / PS / PS+m2_stage, eff_bs 8, 12 steps, d_stages 1,2,3 (cg_bin 3, cg_cap 4)."""
    o = {}; small = ["--device", DEV, "--eff_bs", "8", "--micro", "8", "--d_stages", "1,2,3", "--cg_bin", "3", "--cg_cap", "4", "--log_every", "1", "--steps", "24000", "--stop_at", "12"]
    cells = {"DIRECT": ["--direct"], "M1": ["--recode", "--lam", "1"], "M2": ["--bottleneck"], "PS": ["--recode", "--lam", "1", "--bottleneck"],
             "PS_m2": ["--recode", "--lam", "1", "--bottleneck", "--m2_stage", "--m2_ramp", "4"], "OW_DIRECT": ["--direct", "--task", "ow"]}
    strip = lambda lg: [{k: v for k, v in x.items() if k != "sec"} for x in lg]  # noqa: E731
    for c, fl in cells.items():
        rh = run("D:/Latent", "%s/off_%s_head" % (TMP, c), fl + small); rt = run(TREE, "%s/off_%s_tree" % (TMP, c), fl + small)
        o[c] = {"rc": [rh[0], rt[0]], "n": [len(rh[1]), len(rt[1])], "bitwise": strip(rh[1]) == strip(rt[1]), "stage_max": max((x.get("stage", 0) for x in rt[1]), default=None),
                "new_keys": sorted(set(k for x in rt[1] for k in x) - set(k for x in rh[1] for k in x)), "err": [e[-200:] for e in (rh[2], rt[2]) if "Traceback" in e]}
    o["pass"] = all(v["bitwise"] and v["n"][0] == 12 and v["rc"] == [9, 9] and v["new_keys"] == [] for k, v in o.items() if isinstance(v, dict)); return o


def ow_smoke():
    """each P0 cell's flags train 12 steps (DEV under scratch/s222_det_wrap.py: math SDP, as the 2070 needs; eff_bs 8): rc 9, the wrap counts logged, n_wrap + n_nowrap == eff_bs per step."""
    o = {}
    for c, fl in CELLS.items():
        rc, lg, err = run(TREE, "%s/smoke_%s" % (TMP, c), ["--direct", "--task"] + fl[1:] + ["--device", DEV, "--eff_bs", "8", "--micro", "8", "--log_every", "1", "--steps", "48000", "--stop_at", "12"], det=True)   # the 2070 (Turing) has no bf16 mem-efficient SDPA: math via the wrapper
        o[c] = {"rc": rc, "n": len(lg), "wrap_fields": all(all(k in x for k in ("num_ok_wrap", "n_wrap", "num_ok_nowrap", "n_nowrap")) for x in lg) and len(lg) > 0,
                "per_step_items": sorted({x.get("n_wrap", 0) + x.get("n_nowrap", 0) for x in lg}), "err": err[-300:] if "Traceback" in err else ""}
    o["pass"] = all(v["rc"] == 9 and v["n"] == 12 and v["wrap_fields"] and v["per_step_items"] == [8.0] for v in o.values() if isinstance(v, dict)); return o


def args():
    """--args_only for the three cells (P3a's recipe + the cell flags) -> the registered diffs: P0-OW vs P3a's args.json = {task, order_mode,
    stream_seed, hashes}; P0b vs P0 = {order_mode}; P0c vs P0 = {dag}. META / control keys (codi_ps META + args_only / stop_at / resume / out) ignored;
    'hashes' = the *_sha / *_sha256 keys."""
    import subprocess
    meta = {"device", "train_gpu", "train_driver", "torch_version", "transformers_version", "python_version", "cuda_version", "micro", "accumulation", "out",
            "resume_from", "args_only", "args_ref", "resume", "stop_at", "auto_micro", "mem_check", "time_steps", "mem_budget"}
    A = {}
    for c, fl in CELLS.items():
        out = "%s/args_%s" % (TMP, c); os.makedirs(out, exist_ok=True)
        r = subprocess.run([sys.executable, "-B", os.path.join(TREE, "boxes", "codi_ps.py"), "--out", out, "--args_only", "--device", "cuda:0"] + RECIPE + fl, capture_output=True, text=True, cwd="D:/Latent")
        assert r.returncode == 0, r.stderr[-800:]; A[c] = json.load(open(os.path.join(out, "args.json")))
    ref = json.load(open(P3A_ARGS)); diff = lambda x, y: {k for k in set(x) | set(y) if k not in meta and x.get(k) != y.get(k)}  # noqa: E731
    hashes = lambda ks: {k for k in ks if k.endswith(("_sha", "_sha256"))}  # noqa: E731
    d0 = diff(ref, A["P0-OW"]); d0n = d0 - hashes(d0)
    o = {"P0_vs_P3a": sorted(d0), "P0_vs_P3a_nonhash": sorted(d0n), "P0b_vs_P0": sorted(diff(A["P0-OW"], A["P0b-TOPO"])), "P0c_vs_P0": sorted(diff(A["P0-OW"], A["P0c-CHAIN"])),
         "values": {c: {k: A[c].get(k) for k in ("task", "order_mode", "dag", "stream_seed", "lr", "cg_thr", "cg_cap", "steps")} for c in A}}
    nh = lambda d: sorted(d - hashes(d))  # noqa: E731
    o["P0b_vs_P0_nonhash"] = nh(diff(A["P0-OW"], A["P0b-TOPO"])); o["P0c_vs_P0_nonhash"] = nh(diff(A["P0-OW"], A["P0c-CHAIN"]))
    o["note"] = ("hash keys (*_sha / *_sha256) differ by construction (task / layout / code); dag and order_mode are new keys, at their defaults (on / forward) in P0-OW; codi_port_sha256 differs only in the sandbox (its ROOT line is edited there; the committed codi_port is unchanged)")
    o["pass"] = (set(d0n) <= {"task", "order_mode", "stream_seed", "dag"} and {"task", "stream_seed"} <= set(d0n) and A["P0-OW"]["dag"] == "on"
                 and A["P0-OW"]["order_mode"] == "forward" and o["P0b_vs_P0_nonhash"] == ["order_mode"] and o["P0c_vs_P0_nonhash"] == ["dag"])
    return o


def main():
    os.makedirs(OUT, exist_ok=True)
    for part in [a for a in sys.argv[1:] if a in ("gen_eq", "chain", "wrap", "flags_off", "ow_smoke", "args")]:
        r = {"gen_eq": gen_eq, "chain": chain, "wrap": wrap, "flags_off": flags_off, "ow_smoke": ow_smoke, "args": args}[part](); r["tree"] = TREE
        json.dump(r, open(os.path.join(OUT, part + ".json"), "w", newline="\n"), indent=1, default=str); print(part, "pass", r["pass"], flush=True)


if __name__ == "__main__":
    main()
