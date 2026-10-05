"""THEORY §111 scoring (rules fixed in §111; written before any §111 run; Theory is the only scorer).
Runs: boxes/runs_s5/s111_{mk4_2x,mk4_dag,mk4_read}_s{6..10}/eval.json (note 111-1 adds mk4_read; early criterion key early_escape = loss/null ratio < 0.9 within 1k of k2); §108 MK-4: boxes/runs_s5/s108_mk4_s{1..5}/eval.json.
Keys (as §108) plus train_handoff_err (per-hand-off error on 20k 512-op rollouts from TRAINING-split sequences) and
final_train_loss.
REL (per arm): cumulative >= 0.99 at 512 in >= 4/5 AND donor SERIAL (every in-chunk slot donor >= 0.90, recipient < 0.10)
in those runs.  SEL: in each group of 5 (§108 s1-5; each §111 arm s6-10), the run with the lowest train_handoff_err passes
ACC (>= 0.99 at 512).  Reported: AUC of each criterion (train_handoff_err, final_train_loss; lower = predicted pass) for
passing runs, pooled; best-of-n curve.
    python theory/score_s111.py
"""
import itertools
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_s5")


def load(prefix, seeds):
    out = {}
    for s in seeds:
        f = os.path.join(R, "%s_s%d" % (prefix, s), "eval.json")
        if os.path.exists(f):
            out[s] = json.load(open(f))
            b = os.path.join(R, "%s_s%d" % (prefix, s), "s111_backfill.json")  # §108 backfills (111-2); fix 111-3
            if os.path.exists(b):
                for k, v in json.load(open(b)).items():
                    if out[s].get(k) is None:
                        out[s][k] = v
            out[s]["train_handoff_err"] = the(out[s].get("train_handoff_err"))
    return out


def the(v):
    """train_handoff_err as a scalar: the in-chunk (latent) hand-off error, as 111-2 defines it (fix 111-3)."""
    return v.get("in_chunk_err") if isinstance(v, dict) else v


def passes(r):
    return r["cumulative"]["512"] >= 0.99


def serial(r):
    d = r.get("donor") or {}
    return bool(d) and all(v["donor_consistent"] >= 0.90 and v["recipient_consistent"] < 0.10 for v in d.values())


def auc(scores_pos, scores_neg):
    if not scores_pos or not scores_neg:
        return None
    wins = sum((p < n) + 0.5 * (p == n) for p, n in itertools.product(scores_pos, scores_neg))
    return wins / (len(scores_pos) * len(scores_neg))


def main():
    groups = {"§108 MK-4": load("s108_mk4", range(1, 6)), "§111 MK-4-2x": load("s111_mk4_2x", range(6, 11)),
              "§111 MK-4-DAG": load("s111_mk4_dag", range(6, 11)),
              "§111 MK-4-READ": load("s111_mk4_read", range(6, 11))}
    res = {"groups": {}}
    allruns = []
    for g, runs in groups.items():
        ok = [s for s, r in runs.items() if passes(r) and serial(r)]
        G = {"n": len(runs), "cum512": {s: r["cumulative"]["512"] for s, r in runs.items()},
             "per_handoff": {s: (r.get("per_handoff_acc") or {}).get("pooled") if isinstance(r.get("per_handoff_acc"), dict) else r.get("per_handoff_acc") for s, r in runs.items()},
             "train_handoff_err": {s: r.get("train_handoff_err") for s, r in runs.items()},
             "final_train_loss": {s: r.get("final_train_loss") for s, r in runs.items()},
             "rel_seeds": ok, "REL": len(ok) >= 4}
        crit = {s: r.get("train_handoff_err") for s, r in runs.items() if r.get("train_handoff_err") is not None}
        if crit:
            best = min(crit, key=crit.get); G["SEL_pick"] = best; G["SEL"] = passes(runs[best])
        res["groups"][g] = G
        allruns += [(passes(r), r.get("train_handoff_err"), r.get("final_train_loss")) for r in runs.values()]
        print("%-14s n %d REL %s seeds %s | cum512 %s | SEL pick %s -> %s" % (g, G["n"], G["REL"], ok,
              {s: round(v, 4) for s, v in G["cum512"].items()}, G.get("SEL_pick"), G.get("SEL")))
    allruns = [(p_, a_, b_, None) for (p_, a_, b_) in allruns]
    for g, runs in groups.items():
        pass
    for i, name in ((1, "train_handoff_err"), (2, "final_train_loss")):
        pos = [x[i] for x in allruns if x[0] and x[i] is not None]; neg = [x[i] for x in allruns if not x[0] and x[i] is not None]
        res["AUC_" + name] = auc(pos, neg)
    ee = [(passes(r), r.get("early_escape")) for runs in groups.values() for r in runs.values() if r.get("early_escape") is not None]
    if ee:
        tp = sum(1 for p_, e in ee if p_ and e); fp = sum(1 for p_, e in ee if (not p_) and e)
        res["early_escape"] = {"n": len(ee), "pass_and_escape": tp, "fail_and_escape": fp, "pass_total": sum(p_ for p_, _ in ee)}
    print("AUC train_handoff_err", res["AUC_train_handoff_err"], "| AUC final_train_loss", res["AUC_final_train_loss"], "| early_escape", res.get("early_escape"))
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s111.json"), "w", newline="\n"), indent=1, default=float)


if __name__ == "__main__":
    main()
