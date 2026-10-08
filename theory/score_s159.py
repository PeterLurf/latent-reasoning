"""THEORY §159 scoring (rules fixed in §159; written before any §159 implementation or run; Theory is the only scorer).
R-5x5: §155 R's write-after-read recipe (multpair --carrier deltamem --mem_write top, write_mix 2000 / 6000) on --task 5x5, 48k; runs only if §155 R s2 >= 0.99.
EXACT: test_acc >= 0.99 (bench 5x5 test). LATENT BEATS TEXT (C3): median test_acc >= TC-M - 0.01 with fewer FLOPs and fewer sequential passes than TC-M 5x5
(1.000; 2.33e10; 87), reported beside ICoT-SI 5x5 (0.996; 5.76e9; 15) and always with the caveat "SOTA-level accuracy; serial use not shown" (note 155-4).
Dirs boxes/runs_fsq/s159_r_5x5_s{seed}.
    python theory/score_s159.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s155 as S155  # noqa: E402

R = os.path.join(ROOT, "boxes", "runs_fsq")
OFFS = (0, 100, 200, 300)
TCM5 = {"acc": 1.0, "flops": 23314827264.0, "passes": 87}; ICOT5 = {"acc": 0.996, "flops": 5764879872.0, "passes": 15}   # §98: m98_tc_5x5_s1 / icotsi_5x5
EXP = {"family": "mult_pair", "task": "5x5", "carrier": "deltamem", "mem_write": "top", "write_mix_start": 2000, "write_mix_end": 6000, "steps": 48000}
CAVEAT = "SOTA-level accuracy; serial use not shown (note 155-4)"
load = S155.load


def slot(k):
    for o in OFFS:
        d = "s159_r_5x5_s%d" % (k + o); r = load(d); a = load(d, "args.json") or {}
        if r is None:
            if a:
                return {"status": "running", "accepted": None}
            return {"status": "open", "accepted": None}
        if a.get("mem_check") or a.get("train_n") or str(a.get("device", "")).startswith("cpu"):
            continue
        bad = {kk: a.get(kk) for kk, v in EXP.items() if a.get(kk) != v}
        if bad:
            return {"status": "error", "error": bad, "accepted": None}
        c = r.get("compute") or {}
        return {"status": "accepted", "accepted": {"seed": k + o, "test_acc": r.get("test_acc"), "per_column": (r.get("acc_by_column") or {}).get("per_column"),
                                                   "serial": S155.S154.serial(r.get("donor")), "flops": c.get("flops"), "passes": c.get("sequential_passes"),
                                                   "ext_5x8": (r.get("ext_5x8") or {}).get("test_acc"), "gpu": r.get("train_gpu")}}
    return {"status": "open", "accepted": None}


def main():
    g = S155.slot("r", 2)["accepted"]
    cond = "pending (§155 R s2 not read)" if not g else ("RUN (R s2 %s >= 0.99)" % g["test_acc"] if (g["test_acc"] or 0) >= 0.99 else "NOT RUN (R s2 %s < 0.99)" % g["test_acc"])
    print("CONDITION:", cond)
    sl = {k: slot(k) for k in (1, 2, 3, 4)}; acc = [v["accepted"] for v in sl.values() if v["accepted"]]
    for k, v in sl.items():
        print("slot", k, v["status"], v.get("accepted") or v.get("error"))
    res = {"condition": cond, "slots": sl}
    if acc:
        med = lambda x: sorted(x)[len(x) // 2]  # noqa: E731
        ma = med([a["test_acc"] for a in acc]); fl = [a["flops"] for a in acc if a["flops"]]; ps = [a["passes"] for a in acc if a["passes"]]
        beats = bool(fl and ps and ma >= TCM5["acc"] - 0.01 and med(fl) < TCM5["flops"] and med(ps) < TCM5["passes"])
        res["headline"] = {"median_test_acc": ma, "exact_slots": sum(1 for a in acc if (a["test_acc"] or 0) >= 0.99), "flops_vs_tcm": fl and med(fl) / TCM5["flops"],
                           "flops_vs_icotsi": fl and med(fl) / ICOT5["flops"], "passes": ps and med(ps), "LATENT_BEATS_TEXT": beats, "caveat": CAVEAT}
        print("HEADLINE:", res["headline"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s159.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
