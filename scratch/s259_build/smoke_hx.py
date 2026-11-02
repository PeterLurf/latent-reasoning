"""Note 257-2 b395 CPU smoke (Audit: re-run after the hh_dump fix): inject_aw_eval.part_halthead (d 1..7, fit_law) and part_long (K' 8 / 64 / 128) on the
unbuffered-read dis_s1 init copy, CPU, the fit set shrunk (HH_N 60) and TEST2 truncated to 4 items per d - code paths and field layout only (score_hx's fields), no readings
kept -> scratch/gate_s259/smoke_hx.json.
    python -B scratch/s259_build/smoke_hx.py"""
import json
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402

d = "scratch/_tmp/s259_smoke"; os.makedirs(d, exist_ok=True)
shutil.copy("boxes/runs_fsq/dis_s1/args.json", d + "/args.json"); shutil.copy("boxes/runs_fsq/dis_s1_init/model.pt", d + "/model.pt")
a, m, g = E.load(d, "cpu"); s = IO.build_sets(mul_s_p=0); s["test2"] = {k: v[:4] for k, v in s["test2"].items()}; E.HH_N = 60
h = E.part_halthead(m, g, s, a, "cpu", 8); lo = E.part_long(m, g, s, a, "cpu", 8, [8, 64, 128], [1, 2, 3, 4, 5, 6])
o = {"halthead_ds": sorted(k for k in h if k.isdigit()), "halthead_fields_d7": sorted(h["7"]) if "7" in h else None, "head_keys": sorted(h["head"]),
     "fit_law": h["head"].get("fit_law"), "long_ds": sorted(lo["by_d"]), "long_fields": sorted(lo["by_d"]["6"]), "long_kp": sorted(lo["by_d"]["6"]["acc"], key=int),
     "json_ok": bool(json.dumps(h)) and bool(json.dumps(lo))}
need = {"n", "acc_k8", "acc_halt", "mean_khalt", "khalt_hist", "never_halt_n"}
o["pass"] = (o["halthead_ds"] == [str(x) for x in range(1, 8)] and need <= set(o["halthead_fields_d7"] or []) and "val_acc" in o["head_keys"] and o["fit_law"] == "reg"
             and o["long_ds"] == [str(x) for x in range(1, 7)] and {"acc", "s_norm"} <= set(o["long_fields"]) and o["long_kp"] == ["8", "64", "128"])
os.makedirs("scratch/gate_s259", exist_ok=True); json.dump(o, open("scratch/gate_s259/smoke_hx.json", "w", newline="\n"), indent=1); print("smoke_hx pass", o["pass"]); shutil.rmtree(d)
