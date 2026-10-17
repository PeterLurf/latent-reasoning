"""D4 Stage 0 S0-1 reader (pre-stated in D4_draft.md section 5/6): EDGE-COLLAPSE / DEADLINE-GONE labels and the over-reduction share (ORS) from a cell's
eval_vbhalt.json rows (+ its eval_stepbyt.json if present). $0, CPU, read-only. Usage: python -I d4_edge_collapse.py <out.json> <run_dir> [<run_dir> ...]
  on-schedule pass-event at t: alive(E^_{t-1}) == n - t + 1 and >= 3 (a >= 2 drop is then a real over-reduction, not 'finishing'); n 10..14 pooled.
  h(t) = P(alive(E^_{t-1}) - alive(E^_t) >= 2 | on-schedule event at t).  t_fin = first t with alive(E^_t) == 0 (None = never by 24).
  EDGE-COLLAPSE iff [events(t 7..9) >= 300] and h(7..9 pooled) >= 0.30 and h(8) >= 2 h(5) and max-min of median t_fin over n 10, 11, 12 <= 1.
  DEADLINE-GONE iff [events(t 7..9) >= 300] and h(7..9 pooled) <= 0.10 and median t_fin(n 12) - median t_fin(n 10) >= 2.  (fewer events: UNREADABLE = not met)
  ORS(t) = h(t) / (1 - e(t)), e(t) = eval_stepbyt strict_exact pooled n 10..16 at t (different conditioning: strict decode vs alive count; descriptive
  except for the branch rule's ORS >= 0.5 clause, pooled over t 6..8 as sum h*events / sum (1-e)*events)."""
import json, os, sys
import numpy as np

def read(run):
    rows = json.load(open(os.path.join(run, "eval_vbhalt.json")))["rows"]
    ev = {}; tf = {}
    for r in rows:
        n, al = r["n"], r["alive"]
        if 10 <= n <= 14:
            for t in range(2, len(al)):
                if al[t - 1] == n - t + 1 and al[t - 1] >= 3:
                    e_ = ev.setdefault(t, [0, 0]); e_[0] += 1; e_[1] += int(al[t - 1] - al[t] >= 2)
        if 9 <= n <= 16:
            t_ = next((t for t in range(len(al)) if al[t] == 0), None); tf.setdefault(n, []).append(99 if t_ is None else t_)
    h = {t: v[1] / v[0] for t, v in ev.items() if v[0] >= 30}
    e79 = sum(ev.get(t, [0, 0])[0] for t in (7, 8, 9)); c79 = sum(ev.get(t, [0, 0])[1] for t in (7, 8, 9))
    h79 = c79 / e79 if e79 else None
    med = {n: float(np.median(v)) for n, v in tf.items()}
    readable = e79 >= 300
    ec = bool(readable and h79 >= 0.30 and h.get(8, 0) >= 2 * h.get(5, 1e9) and (max(med[n] for n in (10, 11, 12)) - min(med[n] for n in (10, 11, 12)) <= 1))
    dg = bool(readable and h79 <= 0.10 and med[12] - med[10] >= 2)
    out = {"events_by_t": {str(t): v[0] for t, v in sorted(ev.items())}, "h_by_t": {str(t): round(x, 4) for t, x in sorted(h.items())}, "h_7_9": h79, "events_7_9": e79,
           "median_t_fin": {str(n): v for n, v in sorted(med.items())}, "EDGE_COLLAPSE": ec, "DEADLINE_GONE": dg, "readable": readable}
    sp = os.path.join(run, "eval_stepbyt.json")
    if os.path.exists(sp):
        sb = json.load(open(sp))["step_by_t"]; ors = {}; num = den = 0.0
        for t in range(5, 10):
            E = sum(sb[str(n)].get(str(t), {}).get("events", 0) for n in range(10, 17)); X = sum(sb[str(n)].get(str(t), {}).get("strict_exact", 0) for n in range(10, 17))
            if E >= 200 and t in h:
                fail = 1 - X / E; ors[str(t)] = {"h": round(h[t], 4), "step_fail": round(fail, 4), "ORS": round(h[t] / fail, 3) if fail > 0 else None}
                if 6 <= t <= 8:
                    num += h[t] * E; den += fail * E
        out["ORS_by_t"] = ors; out["ORS_6_8"] = num / den if den else None
    return out

def fixture():
    """synthetic alive trajectories: CONTENT (one reduction per pass, 15% wrong-content errors that do not change the alive count) -> EC False, DG True;
    DEADLINE (collapse to 0 alive at t 9, hazard rising from t 5) -> EC True, DG False."""
    rng = np.random.default_rng(0); rows_c = []; rows_d = []
    for n in range(9, 17):
        for _ in range(300):
            rows_c.append({"n": n, "alive": [max(0, n - t) for t in range(25)]})
            al = [n]; a = n
            for t in range(1, 25):
                hz = {5: 0.05, 6: 0.2, 7: 0.5, 8: 0.8}.get(t, 1.0 if t >= 9 else 0.0)
                a = 0 if (t >= 9) else (max(0, a - (2 + int(rng.integers(0, 2)))) if (rng.random() < hz and a >= 3) else max(0, a - 1)); al.append(a)
            rows_d.append({"n": n, "alive": al})
    import tempfile
    res = {}
    for name, rows in (("content", rows_c), ("deadline", rows_d)):
        d = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(__file__))); json.dump({"rows": rows}, open(os.path.join(d, "eval_vbhalt.json"), "w"))
        r = read(d); res[name] = {"EC": r["EDGE_COLLAPSE"], "DG": r["DEADLINE_GONE"], "h_7_9": r["h_7_9"]}
        os.remove(os.path.join(d, "eval_vbhalt.json")); os.rmdir(d)
    res["pass"] = res["content"]["EC"] is False and res["deadline"]["EC"] is True and res["deadline"]["DG"] is False
    return res

if __name__ == "__main__":
    out = {"fixture": fixture()}
    for run in sys.argv[2:]:
        out[run] = read(run)
    json.dump(out, open(sys.argv[1], "w", newline="\n"), indent=1)
    print("fixture", out["fixture"])
    for run in sys.argv[2:]:
        r = out[run]; print("==", run, {k: r[k] for k in ("EDGE_COLLAPSE", "DEADLINE_GONE", "h_7_9", "events_7_9")}, "median t_fin", r["median_t_fin"])
        print("   h(t)", r["h_by_t"]); print("   ORS", r.get("ORS_by_t"), "ORS_6_8", r.get("ORS_6_8"))
