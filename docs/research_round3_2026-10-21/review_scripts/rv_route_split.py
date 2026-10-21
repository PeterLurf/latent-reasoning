"""REVIEW_R3 check 2 ($0, CPU, read-only on committed eval_hist24.json; writes rv_route_split.json beside this script).
Which holding route replaced SLVA's own-slot history? hist24 arms (vbank_diag.part_d3hist / part_hist24 docstrings; slva.py gather):
  self_ablate      = SLVA own-slot versions 1..t-2 masked; core / coda bank reads intact
  other_ablate     = core / coda bank versions 1..t-2 masked; SLVA intact
  bank_ablate_slva = versions 1..t-2 masked in SLVA, core AND coda (V_0 and V_{t-1} kept everywhere)
Per model, masked acc_k24 mean over n 3..6 (the STAB bins):
  STATE  = acc(bank_ablate_slva)                     -> holding from {V_0, V_{t-1}} alone (no history reader)
  BANK   = acc(self_ablate) - acc(bank_ablate_slva)   -> extra holding the core / coda bank history adds when SLVA's history is gone
  1 - acc(self_ablate) = what STAB_self (registered) calls history reliance (intact ~1 at n 3..6 in s1; s2 base 0.972 at n 3)
Per item (models whose eval_hist24 carries §216 E4 items): share of items with self_ablate right and bank_ablate_slva wrong (rescued by the bank route),
and with bank_ablate_slva right (state route holds), n 3..6.
Usage: python -B rv_route_split.py"""
import json
import os

R = "D:/Latent/boxes/runs_fsq"
RUNS = ["s213_d3t_s1", "s215_ftctrl_s1", "s215_ftself_s1", "s216_r2_s1", "s216_d3t_s2", "s216_r1_s2"]
NS = range(3, 7)
out = {"def": __doc__, "runs": {}}
for r in RUNS:
    h = json.load(open(os.path.join(R, r, "eval_hist24.json")))
    m = lambda arm: sum(h[arm]["acc_k24"][str(n)] for n in NS) / 4  # noqa: E731
    st, se, ot, it = m("bank_ablate_slva"), m("self_ablate"), m("other_ablate"), m("intact")
    row = {"intact": round(it, 4), "self_ablate": round(se, 4), "other_ablate": round(ot, 4), "bank_ablate_slva": round(st, 4),
           "STATE": round(st, 4), "BANK_backup": round(se - st, 4), "nonadditivity": round((it - se) + (it - ot) - (it - st), 4)}
    items = h.get("items")
    if items and "self_ablate" in items and "bank_ablate_slva" in items:
        resc = tot = state_ok = 0
        for n in NS:
            y = items["y"][str(n)]; a_s = items["self_ablate"][str(n)]["ans_k24"]; a_b = items["bank_ablate_slva"][str(n)]["ans_k24"]
            for yy, s_, b_ in zip(y, a_s, a_b):
                tot += 1; resc += int(s_ == yy and b_ != yy); state_ok += int(b_ == yy)
        row["items_rescued_by_bank"] = round(resc / tot, 4); row["items_state_ok"] = round(state_ok / tot, 4); row["n_items"] = tot
    out["runs"][r] = row
for a, b in (("s215_ftself_s1", "s215_ftctrl_s1"), ("s216_r2_s1", "s215_ftself_s1"), ("s216_r1_s2", "s216_d3t_s2"), ("s215_ftctrl_s1", "s213_d3t_s1")):
    A, B = out["runs"][a], out["runs"][b]
    d_self = A["self_ablate"] - B["self_ablate"]
    out.setdefault("deltas", {})["%s - %s" % (a, b)] = {"d_masked_self_ablate": round(d_self, 4), "d_STATE": round(A["STATE"] - B["STATE"], 4),
                                                         "d_BANK_backup": round(A["BANK_backup"] - B["BANK_backup"], 4),
                                                         "share_BANK": round((A["BANK_backup"] - B["BANK_backup"]) / d_self, 3) if d_self else None}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_route_split.json"), "w", newline="\n"), indent=1)
for r, d in out["runs"].items():
    print("%-16s %s" % (r, d))
for k, d in out["deltas"].items():
    print("%-34s %s" % (k, d))
