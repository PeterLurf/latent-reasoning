"""§271 N1 G0 floors (Note 271-2; N1_v1 §6.1): f(n, m) per TEST / TRANSFER / OFFSET cell on the cell's own MOVED items (so_words.floors: fit random.Random(27110),
N 200,000 per (n, m), MOVED subset, modal answer, ties -> smallest index, unseen key -> 1/(n-1)), floor_hold = P(S_m = S_0). Also checks the fit against Theory's
DATA-SIM reference (n1_oc.py: f(5, 8) ~0.326) on a fresh 100k draw of cell (5, 8). Writes <runs>/n1_floors/floors_so.json (runs = D:/Latent/boxes/runs_fsq):
  set_sha (inject_so_eval.set_shas: so_test_sha), set_shas, floors/<n>/<m>/<mode>/{f, arg, by_shortcut, n_moved, floor_hold, unseen}, fit {seed, n_fit}.
    python -B scratch/n1_build/g0_floors.py"""
import hashlib
import json
import os
import random
import sys

NEW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")).replace("\\", "/"); sys.path.insert(0, NEW); os.chdir(NEW)
import boxes.so_words as SO  # noqa: E402
import boxes.inject_so_eval as SE  # noqa: E402

OUTD = "D:/Latent/boxes/runs_fsq/n1_floors"
if __name__ == "__main__":
    sets = SO.test_sets(); shas = SE.set_shas(sets); fl = {}
    for (n, m, md), its in sorted(sets["test"].items()):
        fl.setdefault(str(n), {}).setdefault(str(m), {})[md.upper()] = SO.floors(its)
    for m, its in sorted(sets["transfer"].items()):
        fl["5"][str(m)]["TRANSFER"] = SO.floors(its)
    fl["5"]["4"]["OFFSET"] = SO.floors(sets["offset"])
    # reference check: a fresh DATA-SIM-like draw of cell (5, 8) through so_words' generator (not the test items)
    r = __import__("numpy").random.default_rng(27199); ref = [SO.make(r, 5, 8, "num") for _ in range(20000)]; fr = SO.floors(ref)
    out = {"set_sha": shas["so_test_sha"], "set_shas": shas, "fit": {"seed": SO.FIT_SEED, "n_fit": SO.N_FIT, "rule": "Note 271-2"}, "floors": fl,
           "check_5_8_fresh20k": {"f": fr["f"], "arg": fr["arg"], "theory_datasim": 0.326}}
    os.makedirs(OUTD, exist_ok=True); p = OUTD + "/floors_so.json"; open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(out, indent=1, sort_keys=True))
    print("sha256", hashlib.sha256(open(p, "rb").read()).hexdigest())
    for n in ("3", "4", "5"):
        print(n, {m: (round(fl[n][m]["NUM"]["f"], 3) if fl[n][m]["NUM"]["f"] is not None else None, fl[n][m]["NUM"]["arg"]) for m in ("4", "6", "8", "10", "12")})
    print("check (5,8) fresh", round(fr["f"], 3), fr["arg"])
