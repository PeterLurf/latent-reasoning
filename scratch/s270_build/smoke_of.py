"""§263 CPU smoke of the new eval parts (no readings kept -> scratch/gate_s270/smoke_of.json = layout only): a temp run dir = the CTRL frozen args (s269 pilot_nkc_s1) +
the bun_s1 init copy; REG TEST2 truncated to N items per d; part_late_premise / part_name_swap / part_shuffle_graded run in-process (cpu) and through inject_aw_eval.run's
dispatcher (parts by name). Checks: keys / n / ds as score_of reads them; every value in [0, 1]; the dispatcher writes eval_<part>.json.
    python -B scratch/s270_build/smoke_of.py [N]"""
import json
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
d = "scratch/_tmp/s270_smoke"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
shutil.copy("scratch/s269_args_frozen_pilot_nkc_s1.json", d + "/args.json"); shutil.copy("boxes/runs_fsq/bun_s1_init/model.pt", d + "/model.pt")
from boxes import inject_aw_eval as EV  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
a, m, g = EV.load(d, "cpu"); s = IO.build_sets(mul_s_p=EV.law(a)); s["test2"] = {k: v[:N] for k, v in s["test2"].items()}
EV.run(d, ["late_premise", "name_swap", "shuffle_graded"], "cpu", 4, model=(m, g), sets=s)
lp = json.load(open(d + "/eval_late_premise.json")); ns = json.load(open(d + "/eval_name_swap.json")); sg = json.load(open(d + "/eval_shuffle_graded.json"))
inr = lambda x: 0.0 <= x <= 1.0  # noqa: E731
o = {"late_premise_ds": sorted(lp["by_d"]), "late_premise_n": [lp["by_d"][k]["n"] for k in sorted(lp["by_d"])],
     "late_premise_ok": all(inr(lp["by_d"][k]["acc"]) for k in lp["by_d"]) and sorted(lp["by_d"]) == ["2", "3", "4"],
     "name_swap_ds": sorted(ns["by_d"]), "name_swap_keys": sorted(ns["by_d"]["3"]),
     "name_swap_ok": all(inr(ns["by_d"][k][f]) for k in ns["by_d"] for f in ("follow", "floor", "intact_acc")) and sorted(ns["by_d"]) == ["3", "4"],
     "shuffle_ds": sorted(sg["by_d"]), "shuffle_keys": {k: sorted(v) for k, v in sg["by_d"]["2"].items()},
     "shuffle_ok": all(inr(sg["by_d"][k][kk]["acc"]) for k in sg["by_d"] for kk in ("independent", "full")) and sorted(sg["by_d"]) == ["2", "3", "4"],
     "eval_json_parts": sorted(json.load(open(d + "/eval.json"))["parts"])}
o["pass"] = o["late_premise_ok"] and o["name_swap_ok"] and o["shuffle_ok"] and {"late_premise", "name_swap", "shuffle_graded"} <= set(o["eval_json_parts"])
os.makedirs("scratch/gate_s270", exist_ok=True); json.dump(o, open("scratch/gate_s270/smoke_of.json", "w", newline="\n"), indent=1); print("smoke_of pass", o["pass"]); shutil.rmtree(d)
