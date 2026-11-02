"""Note 259-11 C5-STABLE FRESH local checks -> scratch/gate_s264/smoke_sf.json:
(1) the FULL-size fresh draw (no model; SF_N 1000 x d 1..8, the real fit sets): no exhaustion, disjointness asserted inside, per-d set shas (compared later with the host's);
(2) part_stab_fresh on the bun_s1 init copy, CPU, SF_N 3: the field layout (pred lists of 15 ints, ans, sig) - no readings kept.
    python -B scratch/s264_build/smoke_sf.py"""
import json
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np  # noqa: E402
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
import boxes.ow_words as OW  # noqa: E402

s = IO.build_sets(mul_s_p=0); a0 = json.load(open("boxes/runs_fsq/bun_s1/args.json"))


class NoModel:                                                              # (1): the draw only - hh_dump_inc replaced by a stub returning zeros
    pass


real_dump = E.hh_dump_inc; E.hh_dump_inc = lambda m, T, dev, bs, K=14: (None, np.zeros((len(T), K + 1), dtype=np.int64), None)
full = E.part_stab_fresh(NoModel(), None, s, a0, "cpu", 8); E.hh_dump_inc = real_dump
o = {"full_set_shas": full["set_shas"], "full_n": {d: full[d]["n"] for d in full["set_shas"]}, "excl_n": full["excl_n"], "fit_sigs_n": full["fit_sigs_n"]}
d = "scratch/_tmp/s264_smoke"; os.makedirs(d, exist_ok=True)
shutil.copy("boxes/runs_fsq/bun_s1/args.json", d + "/args.json"); shutil.copy("boxes/runs_fsq/bun_s1_init/model.pt", d + "/model.pt")
a, m, g = E.load(d, "cpu"); E.SF_N = 3; sm = E.part_stab_fresh(m, g, s, a, "cpu", 8)
o["smoke_ds"] = sorted(k for k in sm if k.isdigit()); o["pred_len"] = len(sm["8"]["items"]["pred"][0]); o["json_ok"] = bool(json.dumps(sm))
o["pass"] = (all(v == 1000 for v in o["full_n"].values()) and len(o["full_set_shas"]) == 8 and o["smoke_ds"] == [str(x) for x in range(1, 9)] and o["pred_len"] == 15 and o["json_ok"])
os.makedirs("scratch/gate_s264", exist_ok=True); json.dump(o, open("scratch/gate_s264/smoke_sf.json", "w", newline="\n"), indent=1); print("smoke_sf pass", o["pass"]); shutil.rmtree(d)
