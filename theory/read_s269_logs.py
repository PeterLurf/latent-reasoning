"""THEORY §269 pre-stated $0 reads on stored logs / evals (committed before computing; run only after Audit's ack; descriptive; backlog FD1 / ST2).
FD1 FORMATION-CENSUS. Population: every boxes/runs_fsq/<run>/ (top level only; no recursion) whose args.json has d_max == 2, whose log.jsonl has ws_readout rows,
    and whose name has no '.void' / '_gtime' / '_mem' suffix. Per run: the ws_readout['2']['2'] series (pass-2 ':' readout == v_2 on TEST2 d 2, logged every 1k).
    FORMED = the last logged value >= 0.80; onset t* = the first logged step with value >= 0.80 (None if never). Early markers at step 2000:
    E_ws = ws_readout['2']['2'], E_aux = aux_vacc_by_d_pass['2']['2'] (teacher-forced aux decodability), E_clip = clip_frac_100.
    Reading per marker: PREDICTIVE if some threshold on the step-2000 value classifies FORMED vs not with <= 1 error over the population (n reported); else NOT.
    Also reported: t* by seed / train_gpu / lr; the share of never-formers. Descriptive.
ST2 SEED-VARIANCE. Population: run pairs <x>_s1 / <x>_s2 (top level) both with eval_acc.json by_d d 4 and d 6 (TEST2, n per by_d['n'] if present, else 1000).
    Per pair and d: diff = acc_s1 - acc_s2. sigma_seed^2 = mean(diff^2) / 2 - mean binomial item variance p(1-p)/n (floored at 0).
    Reading at d 6: SEED-DOMINANT if sigma_seed >= 0.02; ITEM-DOMINANT if sigma_seed <= 0.01; else MIXED. Descriptive (sets the seeds-vs-items budget rule).
    python -B theory/read_s269_logs.py
"""
import json, math, os
R = "D:/Latent/boxes/runs_fsq"


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def rows(p):
    out = []
    try:
        for ln in open(p):
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def skip(name):
    return any(s in name for s in (".void", "_gtime", "_mem", ".bak"))


fd1 = {}
for name in sorted(os.listdir(R)):
    if skip(name) or not os.path.isdir(os.path.join(R, name)):
        continue
    a = jl(os.path.join(R, name, "args.json"))
    if not a or a.get("d_max") != 2:
        continue
    lg = [r for r in rows(os.path.join(R, name, "log.jsonl")) if isinstance(r.get("ws_readout"), dict) and "2" in r["ws_readout"]]
    if not lg:
        continue
    ser = [(r["step"], r["ws_readout"]["2"].get("2")) for r in lg if r["ws_readout"]["2"].get("2") is not None]
    if not ser:
        continue
    at2k = [r for r in lg if r.get("step") == 2000]
    e = at2k[0] if at2k else {}
    fd1[name] = {"formed": ser[-1][1] >= 0.80, "t_star": next((s for s, v in ser if v >= 0.80), None), "last": ser[-1],
                 "E_ws": e.get("ws_readout", {}).get("2", {}).get("2"), "E_aux": (e.get("aux_vacc_by_d_pass", {}).get("2", {}) or {}).get("2"),
                 "E_clip": e.get("clip_frac_100"), "seed": a.get("seed"), "gpu": a.get("train_gpu"), "lr": a.get("lr"), "steps": a.get("steps")}


def predictive(key):
    pts = [(v[key], v["formed"]) for v in fd1.values() if v[key] is not None]
    if len(pts) < 3 or len({f for _, f in pts}) < 2:
        return {"n": len(pts), "reading": "UNDEFINED"}
    best = len(pts)
    for thr in sorted({x for x, _ in pts}):
        for sign in (1, -1):
            err = sum((sign * (x - thr) >= 0) != f for x, f in pts)
            best = min(best, err)
    return {"n": len(pts), "min_errors": best, "reading": "PREDICTIVE" if best <= 1 else "NOT"}


fd1_sum = {"n_runs": len(fd1), "n_formed": sum(v["formed"] for v in fd1.values()), "markers": {k: predictive(k) for k in ("E_ws", "E_aux", "E_clip")}}

st2 = {}
for name in sorted(os.listdir(R)):
    if not name.endswith("_s1") or skip(name):
        continue
    sib = name[:-3] + "_s2"
    a1, a2 = jl(os.path.join(R, name, "eval_acc.json")), jl(os.path.join(R, sib, "eval_acc.json"))
    if not a1 or not a2:
        continue
    rec = {}
    for d in ("4", "6"):
        if d in a1.get("by_d", {}) and d in a2.get("by_d", {}):
            p1, p2 = a1["by_d"][d]["acc"], a2["by_d"][d]["acc"]; n = a1["by_d"][d].get("n", 1000) or 1000
            rec[d] = {"s1": p1, "s2": p2, "diff": p1 - p2, "item_var": (p1 * (1 - p1) + p2 * (1 - p2)) / (2 * n)}
    if rec:
        st2[name[:-3]] = rec
sig = {}
for d in ("4", "6"):
    xs = [v[d] for v in st2.values() if d in v]
    if xs:
        s2 = max(0.0, sum(x["diff"] ** 2 for x in xs) / (2 * len(xs)) - sum(x["item_var"] for x in xs) / len(xs))
        sig[d] = {"n_pairs": len(xs), "sigma_seed": math.sqrt(s2), "item_se_mean": math.sqrt(sum(x["item_var"] for x in xs) / len(xs))}
s6 = sig.get("6", {}).get("sigma_seed")
st2_sum = {"sigma": sig, "reading_d6": None if s6 is None else ("SEED-DOMINANT" if s6 >= 0.02 else "ITEM-DOMINANT" if s6 <= 0.01 else "MIXED")}
out = {"FD1": {"summary": fd1_sum, "runs": fd1}, "ST2": {"summary": st2_sum, "pairs": st2}}
os.makedirs("D:/Latent/theory/results", exist_ok=True)
json.dump(out, open("D:/Latent/theory/results/read_s269_logs.json", "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
print(json.dumps({"FD1": fd1_sum, "ST2": st2_sum}, indent=1, default=str))
