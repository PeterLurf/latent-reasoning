"""§260 full stage CPU smoke -> scratch/gate_s266/smoke_gb.json: (1) scratch.s266_planfree on a temp run dir (the pilot's frozen generic-cue args + the bun_s1 copy): rc 0 and
gates_pass.json {"plan_free": true, cue identical, no entity name}; (2) part_stab with TEST2 truncated to 3 items per d: layout (d 1..7, 15 preds per item, set shas).
No readings kept.
    python -B scratch/s266_build/smoke_gb.py"""
import json
import os
import shutil
import subprocess
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
d = "scratch/_tmp/s266_smoke"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
shutil.copy("scratch/s265_args_frozen_pilot_gb_s1.json", d + "/args.json"); shutil.copy("boxes/runs_fsq/bun_s1_init/model.pt", d + "/model.pt")
rc = subprocess.call([sys.executable, "-B", "-m", "scratch.s266_planfree", "--device", "cpu", "--out", d], stdout=open("scratch/_tmp/s266_pf.log", "w"), stderr=subprocess.STDOUT)
gp = json.load(open(d + "/gates_pass.json")) if os.path.exists(d + "/gates_pass.json") else {}
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
a, m, g = E.load(d, "cpu"); s = IO.build_sets(mul_s_p=0); full_shas = {str(k): __import__("boxes.ow_words", fromlist=["set_sha"]).set_sha(s["test2"][k]) for k in range(1, 8)}
s["test2"] = {k: v[:3] for k, v in s["test2"].items()}; st = E.part_stab(m, g, s, a, "cpu", 8)
o = {"planfree_rc": rc, "gates_pass": gp, "stab_ds": sorted(k for k in st if k.isdigit()), "stab_pred_len": len(st["7"]["items"]["pred"][0]), "full_test2_shas": full_shas}
o["pass"] = rc == 0 and gp.get("plan_free") is True and gp.get("cue_has_no_entity_name") is True and o["stab_ds"] == [str(x) for x in range(1, 8)] and o["stab_pred_len"] == 15
os.makedirs("scratch/gate_s266", exist_ok=True); json.dump(o, open("scratch/gate_s266/smoke_gb.json", "w", newline="\n"), indent=1); print("smoke_gb pass", o["pass"]); shutil.rmtree(d)
