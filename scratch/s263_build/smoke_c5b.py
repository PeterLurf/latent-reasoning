"""Note 259-5 C5-BUN CPU smoke: part_halt_inc + part_halthead2 on the bun_s1 init copy, CPU, fit sets shrunk (HI_N / HH_N 40) and TEST2 truncated to 3 items per d -
code paths + field layout (score_c5b's fields) only, no readings kept -> scratch/gate_s263/smoke_c5b.json.
    python -B scratch/s263_build/smoke_c5b.py"""
import json
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402

d = "scratch/_tmp/s263_smoke"; os.makedirs(d, exist_ok=True)
shutil.copy("boxes/runs_fsq/bun_s1/args.json", d + "/args.json"); shutil.copy("boxes/runs_fsq/bun_s1_init/model.pt", d + "/model.pt")
a, m, g = E.load(d, "cpu"); s = IO.build_sets(mul_s_p=0); s["test2"] = {k: v[:3] for k, v in s["test2"].items()}; E.HI_N = 40; E.HH_N = 40
hi = E.part_halt_inc(m, g, s, a, "cpu", 8); h2 = E.part_halthead2(m, g, s, a, "cpu", 8)
need_hi = {"n", "acc_k8", "acc_halt", "mean_khalt", "mean_passes_run", "never_halt_n", "khalt_hist"}; need_h2 = {"n", "acc_k8", "acc_halt", "mean_khalt", "khalt_hist", "never_halt_n"}
o = {"hi_ds": sorted(k for k in hi if k.isdigit()), "hi_fields": sorted(hi["7"]), "tau_type": type(hi["tau"]).__name__, "fit_law": hi["fit"]["fit_law"],
     "rel_inc_len": len(hi["7"]["items"]["rel_inc"][0]), "h2_ds": sorted(k for k in h2 if k.isdigit()), "h2_fields": sorted(h2["7"]), "h2_kmax": h2["head"]["kmax"],
     "h2_hist_keys": len(h2["7"]["khalt_hist"]), "json_ok": bool(json.dumps(hi)) and bool(json.dumps(h2))}
o["pass"] = (o["hi_ds"] == [str(x) for x in range(1, 8)] and need_hi <= set(o["hi_fields"]) and o["tau_type"] == "float" and o["fit_law"] == "reg" and o["rel_inc_len"] == 14
             and o["h2_ds"] == [str(x) for x in range(1, 8)] and need_h2 <= set(o["h2_fields"]) and o["h2_kmax"] == 14 and o["h2_hist_keys"] == 15 and o["json_ok"])
os.makedirs("scratch/gate_s263", exist_ok=True); json.dump(o, open("scratch/gate_s263/smoke_c5b.json", "w", newline="\n"), indent=1); print("smoke_c5b pass", o["pass"]); shutil.rmtree(d)
