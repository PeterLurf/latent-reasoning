"""§264 / §265 CPU smoke of the new eval parts (layout only; no readings kept -> scratch/gate_s271/smoke_sc.json): a temp run dir = bun_s1's args.json + model.pt (copy);
REG TEST2 truncated to N items per d; parts cue_clock / op_count / attn_fade through inject_aw_eval.run's dispatcher. Checks score_sc's layout (items[sch][d] ans / pred [N][25]
/ rel_inc [N][23] / norm [N][24] + the add / del splits; STD d 6-8 logits [N][13][11]), op_count by_k {0, 2, 4, 6}, attn_fade conds d4k0 {2,3,4} / d4k6 {4} / d8 {4,7}
with [12][12] per item, ws_ok_d8_t7 (N entries), and the read-only assert (inside the part).
    python -B scratch/s271_build/smoke_sc.py [N]"""
import json
import os
import shutil
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
d = "scratch/_tmp/s271_smoke"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
shutil.copy("boxes/runs_fsq/bun_s1/args.json", d + "/args.json"); shutil.copy("boxes/runs_fsq/bun_s1/model.pt", d + "/model.pt")
from boxes import inject_aw_eval as EV  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
a, m, g = EV.load(d, "cpu"); s = IO.build_sets(mul_s_p=EV.law(a)); s["test2"] = {k: v[:N] for k, v in s["test2"].items()}
EV.run(d, ["cue_clock", "op_count", "attn_fade"], "cpu", 4, model=(m, g), sets=s)
cc = json.load(open(d + "/eval_cue_clock.json")); oc = json.load(open(d + "/eval_op_count.json")); af = json.load(open(d + "/eval_attn_fade.json")); o = {}
want = {"STD": ["3", "4", "5", "6", "7", "8"], "ER": ["3", "4", "5", "6"], "LF": ["3", "4", "5", "6"], "CY": ["3", "4", "5", "6"]}
o["cc_ds"] = {k: sorted(v) for k, v in cc["items"].items()}; o["cc_ds_ok"] = all(sorted(cc["items"][k]) == v for k, v in want.items())
lens = set()
for sch, dd in cc["items"].items():
    for dk, it in dd.items():
        lens.add((len(it["ans"]), len(it["pred"][0]), len(it["rel_inc"][0]), len(it["norm"][0]), len(it["rel_inc_add"][0]), len(it["norm_del"][0])))
o["cc_lens"] = sorted(lens); o["cc_lens_ok"] = lens == {(N, 25, 23, 24, 23, 24)}
o["cc_logits"] = {dk: [len(cc["items"]["STD"][dk]["logits"][0]), len(cc["items"]["STD"][dk]["logits"][0][0])] for dk in ("6", "7", "8")}
o["cc_logits_ok"] = all(v == [13, 11] for v in o["cc_logits"].values()) and all(None not in r for dk in ("6", "7", "8") for r in cc["items"]["STD"][dk]["logits"])
o["cc_no_none"] = all(None not in row for sch in cc["items"].values() for it in sch.values() for k in ("rel_inc", "norm") for row in it[k])
o["oc_k"] = sorted(oc["by_k"]); o["oc_ok"] = o["oc_k"] == ["0", "2", "4", "6"] and all(0 <= v["acc"] <= 1 for v in oc["by_k"].values())
o["af_conds"] = {c: sorted(v) for c, v in af["conds"].items()}; o["af_ok"] = o["af_conds"] == {"d4k0": ["2", "3", "4"], "d4k6": ["4"], "d8": ["4", "7"]} and all(
    len(x) == 12 and len(x[0]) == 12 for c in af["conds"].values() for v in c.values() for x in v) and len(af["ws_ok_d8_t7"]) == N
o["af_mass_range"] = [min(min(min(r) for r in x) for c in af["conds"].values() for v in c.values() for x in v), max(max(max(r) for r in x) for c in af["conds"].values() for v in c.values() for x in v)]
o["pass"] = o["cc_ds_ok"] and o["cc_lens_ok"] and o["cc_logits_ok"] and o["cc_no_none"] and o["oc_ok"] and o["af_ok"] and 0 <= o["af_mass_range"][0] <= o["af_mass_range"][1] <= 1.0 + 1e-6
os.makedirs("scratch/gate_s271", exist_ok=True); json.dump(o, open("scratch/gate_s271/smoke_sc.json", "w", newline="\n"), indent=1); print("smoke_sc pass", o["pass"]); shutil.rmtree(d)
