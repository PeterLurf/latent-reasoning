"""Audit 10-21 (Note 218-22 branch (ii) precondition): the re-run G-L1c reference model must be the SAME model as the committed G-L1c - same card (4060),
dtype (bf16 autocast), seed and recipe. Compares scratch/gate_s218/g_l1c_rerun.json (s218_l1c_save.py) with the committed scratch/gate_s218/g_l1c.json:
items_sha, card, amp, micro, steps_run, the curve rows (step, plain_cand_ce, host_acc_256) and the final accuracy. PASS = everything bitwise equal, else
|delta final acc| <= 0.005 with the same steps_run / items / card / amp / micro (reported as NEAR). A mismatch blocks branch (ii).
-> scratch/gate_s218/g_l1c_model/same_model_gate.json"""
import json

A = json.load(open("D:/Latent/scratch/gate_s218/g_l1c.json")); B = json.load(open("D:/Latent/scratch/gate_s218/g_l1c_rerun.json"))
keys = ("items_sha", "card", "amp", "micro", "steps_run", "n_items", "bs", "cal_n")
same = {k: [A.get(k), B.get(k), A.get(k) == B.get(k)] for k in keys}
rows = [{"step": a["step"], "pcc": [a["plain_cand_ce"], b["plain_cand_ce"]], "acc": [a["host_acc_256"], b["host_acc_256"]],
         "bitwise": a["plain_cand_ce"] == b["plain_cand_ce"] and a["host_acc_256"] == b["host_acc_256"]} for a, b in zip(A["curve"], B["curve"])]
bit = all(v[2] for v in same.values()) and len(A["curve"]) == len(B["curve"]) and all(r["bitwise"] for r in rows) and A["final_host_acc_256"] == B["final_host_acc_256"]
near = all(v[2] for v in same.values()) and abs(A["final_host_acc_256"] - B["final_host_acc_256"]) <= 0.005
res = {"meta_equal": same, "curve": rows, "final_acc": [A["final_host_acc_256"], B["final_host_acc_256"]], "bitwise": bool(bit), "within_0005": bool(near),
       "verdict": "SAME (bitwise)" if bit else ("SAME (within 0.005)" if near else "MISMATCH - branch (ii) blocked"), "PASS": bool(bit or near), "def": __doc__}
json.dump(res, open("D:/Latent/scratch/gate_s218/g_l1c_model/same_model_gate.json", "w", newline="\n"), indent=1); print("SAME-MODEL", res["verdict"])
