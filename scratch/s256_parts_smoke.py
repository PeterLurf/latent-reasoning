"""§256 CPU smoke of inject_aw_eval.part_insert_ho / part_ws_deep (code paths only): rcb_plaw4_s1's args + the unbuffered-read init copy, TEST2 truncated to 10 items per d,
S256_SHAS pointed at a temp json built for the truncated sets (the real assert path runs). -> scratch/gate_s256/parts_smoke.json (keys / shapes, no readings kept).
    python -B scratch/s256_parts_smoke.py"""
import json
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
import boxes.ow_words as OW  # noqa: E402
import boxes.insert_render as IR  # noqa: E402
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402

N = 10; d = "scratch/_tmp/s256_smoke"; os.makedirs(d, exist_ok=True)
shutil.copy("boxes/runs_fsq/rcb_plaw4_s1/args.json", d + "/args.json"); shutil.copy("boxes/runs_fsq/rcb_plaw4_s1_init/model.pt", d + "/model.pt")
s = IO.build_sets(mul_s_p=0); s["test2"] = {k: v[:N] for k, v in s["test2"].items()}
HO, _ = IR.held_out_sets(s["test2"][4]); tmp = d + "/shas.json"
json.dump({"held_out": IR.set_shas(HO), "ws_deep": {str(k): OW.set_sha(s["test2"][k]) for k in (5, 6, 7, 8)}}, open(tmp, "w"))
E.S256_SHAS = tmp; a, m, g = E.load(d, "cpu")
r1 = E.part_insert_ho(m, g, s, a, "cpu", 50); r2 = E.part_ws_deep(m, g, s, a, "cpu", 50)
o = {"insert_ho_fams": sorted(r1["fam"]), "insert_ho_pos": {f: sorted(v["by_pos"]) for f, v in r1["fam"].items()}, "h_op_by_op": sorted(r1["fam"]["H-OP"]["by_op"]),
     "indist": sorted(r1["indist"]), "n_item_keys": len(r1["items"]), "ws_deep_ds": sorted(r2["by_d"]), "ws_deep_passes": sorted(r2["by_d"]["6"], key=int),
     "ws_items_v_len": len(r2["items"]["6"]["v"][0]), "json_ok": bool(json.dumps(r1)) and bool(json.dumps(r2))}
o["pass"] = (o["insert_ho_fams"] == sorted(["H-OP", "H-CHAINSRC", "H-GIVPOS", "H-EXTEND", "H-LIVE", "K3"]) and o["indist"] == sorted(IR.CONDS) and o["ws_deep_ds"] == ["5", "6", "7", "8"]
             and len(o["ws_deep_passes"]) == a["think_k"] and o["ws_items_v_len"] == a["think_k"])
os.makedirs("scratch/gate_s256", exist_ok=True); json.dump(o, open("scratch/gate_s256/parts_smoke.json", "w", newline="\n"), indent=1); print("parts smoke pass", o["pass"]); shutil.rmtree(d)
