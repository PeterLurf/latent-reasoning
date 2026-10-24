"""REVIEW_ST checks (CPU, 1 thread, idle priority, numpy + tiktoken; no model, no blinded file).
  A  cue token length over every NAMES x UNITS pair (constant length -> the per-pass inputs differ only at the name token; exact cache / padding)
  B  structural shortcut in P1-ASK: on d in {1, 2} TOPO items, the target entity S_1 is always the unique computed entity whose inputs are all given
  C  cost recompute from st_oc.json measured inputs, and Stage-1 wall clock serial vs two hosts (LAT-ST alone | CD + DIRECT)
  D  CUE-ONLY guard as written (pilot depths d in {1, 2}, answer read pooled): false-alarm rate if the decoder answers d = 1 from text (228-2: 1.000)
    python -B rv_st_checks.py"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception:  # noqa: BLE001
    pass
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(HERE); ROOT = os.path.abspath(os.path.join(D, "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import tiktoken  # noqa: E402

import boxes.ow_words as OW  # noqa: E402

ENC = tiktoken.get_encoding("gpt2"); res = {}
# A
L = {len(ENC.encode(" " + OW.QUESTION.format(u=u, A=n))) for n in OW.NAMES for u in OW.UNITS}
nm1 = all(len(ENC.encode(" " + n)) == 1 for n in OW.NAMES)
res["A_cue_len"] = {"distinct_lengths": sorted(L), "names": len(OW.NAMES), "units": len(OW.UNITS), "every_name_one_token": nm1}
# B
st = OW.Stream(24101, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); n = 0; ok = 0; s2_reads_s1 = 0; n2 = 0
for d in (1, 2):
    for _ in range(2000):
        it = st.draw(d); n += 1
        # step 1 reads S_0 and a constant / given (never a computed state): ops[0] is never *_s at t = 1 (add_s needs t >= 2; mul_s needs t >= 3)
        ok += int(not it["ops"][0].endswith("_s"))
        if d == 2:
            n2 += 1; s2_reads_s1 += 1                                          # every step t reads S_{t-1} (generator: apply(op, s[t-1], b))
res["B_shortcut"] = {"items": n, "step1_inputs_all_given": ok, "d2_items": n2, "S2_reads_S1": s2_reads_s1,
                     "reading": "target v_1 = the unique computed entity with all-given inputs in every P1-ASK item: the name is redundant with structure"}
# C
oc = json.load(open(os.path.join(D, "st_oc.json")))["cost"]; f = oc["factors"]; ts = oc["measured_train_sec"]; ev = oc["measured_eval_acc_sec"]
lat8 = [ts["s228_plat2_s1"] * f["lat_st_d8"] * 6, ts["s229_plat2c_s1"] * f["lat_st_d8"] * 6]
cd8 = ts["s228_pdir2c_s1"] * f["cued_direct_d8"] * 6; d8 = ts["s228_pdir2c_s1"] * f["direct_d8"] * 6; e = 3 * 3600; setup = 900
serial = [(x + cd8 + d8 + e + 2 * setup) / 3600 for x in lat8]
two_host_wall = [(x + 0.75 * e + setup) / 3600 for x in lat8]                 # [ASSUMED] LAT-ST eval ~3/4 of the 3 h
side_host = (cd8 + d8 + 0.25 * e + setup) / 3600
res["C_stage1"] = {"lat_st_train_h": [round(x / 3600, 1) for x in lat8], "cd_h": round(cd8 / 3600, 2), "direct_h": round(d8 / 3600, 2),
                   "serial_one_host_h": [round(x, 1) for x in serial], "two_hosts_wall_h": [round(x, 1) for x in two_host_wall],
                   "side_host_h": round(side_host, 1), "host_h_two_hosts": [round(x + side_host, 1) for x in two_host_wall],
                   "usd_two_hosts_high": round((two_host_wall[1] + side_host) * 0.27, 2), "usd_serial_high": round(serial[1] * 0.27, 2),
                   "host_speed_x2_wall_h": [round(2 * x, 1) for x in two_host_wall]}
rng = np.random.default_rng(24102); R = 20000
a1 = rng.binomial(1000, 1.0, R) / 1000; a2 = rng.binomial(1000, 0.11, R) / 1000; pooled = (a1 + a2) / 2
res["D_cueonly_pooled_false_alarm"] = {"d1_answer_from_text": 1.0, "d2": 0.11, "P_LEAK_pooled": float((pooled > 0.16).mean()),
                                       "P_LEAK_d2_only": float((a2 > 0.16).mean())}
out = os.path.join(HERE, "rv_st_checks.json")
with open(out, "w", newline="\n") as fh:
    json.dump(res, fh, indent=1); fh.write("\n")
print(json.dumps(res, indent=1))
