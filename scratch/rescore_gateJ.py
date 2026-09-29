"""Sensitivity analysis (TRAINING_TODO "not training"): every pre-addendum-J model under the addendum-J gate
(delayed >= 0.85 AND depth-matched 5-hop >= 0.60) vs the gate it was scored with (delayed >= 0.85).
Addendum J applies "from this addendum on, not retroactively", so official verdicts are unchanged; this lists
which models, and which verdicts that used them, would change.   -> results/rescore_gateJ.json"""
import json
from latent.protocol3_confirm import ALL, comp, matched
from latent.followup_confirm import REC, PAUSE

runs = {**ALL, **{t: r for t, (r, tag) in REC.items() if not tag}}
P3 = json.load(open("results/protocol3_confirm.json"))["verdicts"]
out, seen = {"models": {}, "changed": [], "affected_verdicts": {}}, set()
for t, r in runs.items():
    if r in seen:
        continue
    seen.add(r)
    c, m = comp(t), matched(t, r)
    d = c["delayed"] if c else None
    m5 = m["5"]["bal_acc"] if m and "5" in m else None
    old = d is not None and d >= 0.85
    J = old and m5 is not None and m5 >= 0.60
    out["models"][t] = {"run": r, "delayed": d, "matched5": m5, "gate_old": old, "gate_J": J}
    if old != J:
        out["changed"].append(t)
for t in out["changed"]:
    out["affected_verdicts"][t] = {k: v for k, v in P3.items() if t in json.dumps(v) and k.startswith("heldout")}
out["note"] = ("Only main_s5 changes. Under J it is not competent, so the addendum-I held-out replication would be "
               "unscorable on seed 5 instead of failed; the held-out replication then rests on seed 3 alone "
               "(seed 4 failed the old gate too). No other PROTOCOL3 verdict used s5 as a competent model.")
json.dump(out, open("results/rescore_gateJ.json", "w"), indent=1, default=float)
print("changed:", out["changed"]); print(out["note"])
