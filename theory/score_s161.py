"""THEORY §161 scoring (rules fixed in §161; written before any §161 implementation or run; Theory is the only scorer).
SHAPE-MIX: §155 R's recipe (DeltaMem-R, top writes, gamma 2k -> 6k, 24k) trained on 4 x n, n ~ U{2..5}; arm mixidx adds a column-index token c + r (r ~ U{0..8} per
training chain, 0 at test). 3 seeds per arm (s1-s3) unconditionally; LENGTH-GENERAL(arm): >= 2 / 3 seeds with ext_4x8 >= 0.9 and 4x4 test_acc >= 0.99. Offsets r ~ U{0..12}.
Dirs boxes/runs_fsq/s161_{mix,mixidx}_4x4_s{seed}.
    python theory/score_s161.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_s155 as S155  # noqa: E402

OFFS = (0, 100, 200, 300)
BASE = {"family": "mult_pair", "task": "4x4", "carrier": "deltamem", "mem_write": "top", "write_mix_start": 2000, "write_mix_end": 6000, "shape_mix": "2,3,4,5"}
ARMS = {"mix": {}, "mixidx": {"col_index_offset": 12}}
load = S155.load
ERRORS = []


def slot(arm, k):
    for o in OFFS:
        d = "s161_%s_4x4_s%d" % (arm, k + o); r = load(d); a = load(d, "args.json") or {}
        if r is None:
            return {"status": "running" if a else "open", "accepted": None}
        if a.get("mem_check") or a.get("train_n") or int(a.get("steps", 0) or 0) != 24000:
            continue
        exp = dict(BASE, **ARMS[arm])
        bad = {kk: a.get(kk) for kk, v in exp.items() if a.get(kk) != v}
        if arm == "mix" and a.get("col_index_offset"):
            bad["col_index_offset"] = a.get("col_index_offset")
        if bad:
            ERRORS.append((d, bad)); return {"status": "error", "accepted": None}
        ws = (load(d, "eval_what_state.json") or {}).get("what_state")
        acc = {"seed": k + o, "test_acc": r.get("test_acc"), "ext_4x8": (r.get("ext_4x8") or {}).get("test_acc"),
               "ext_4x8_per_column": ((r.get("ext_4x8") or {}).get("acc_by_column") or {}).get("per_column"),
               "per_column": (r.get("acc_by_column") or {}).get("per_column"), "serial": S155.S154.serial(r.get("donor")),
               "ext_other": {kk: (r.get(kk) or {}).get("test_acc") for kk in ("ext_4x6", "ext_4x7")}, "what_state": ws, "gpu": r.get("train_gpu")}
        acc["length_general"] = bool((acc["ext_4x8"] or 0) >= 0.9 and (acc["test_acc"] or 0) >= 0.99)
        return {"status": "accepted", "accepted": acc}
    return {"status": "open", "accepted": None}


def main():
    res = {}
    for arm in ARMS:
        sl = {k: slot(arm, k) for k in (1, 2, 3)}; acc = [v["accepted"] for v in sl.values() if v["accepted"]]   # 3 seeds, unconditional (Literature)
        n = sum(1 for a in acc if a["length_general"])
        v = "LENGTH-GENERAL" if n >= 2 else ("NOT LENGTH-GENERAL" if len(acc) - n >= 2 else "pending (%d / 3 seeds)" % len(acc))
        gate = None if v.startswith("pending") else (v == "LENGTH-GENERAL")
        res[arm] = {"slots": sl, "gate": gate, "verdict": v}
        print(arm, "|", v, "|", [a and {kk: a.get(kk) for kk in ("seed", "test_acc", "ext_4x8", "serial")} for a in acc])
    m, mi = res["mix"]["gate"], res["mixidx"]["gate"]
    if m is None or mi is None:
        rd = "pending"
    elif mi and not m:
        rd = "MIX+IDX passes, MIX does not: the schedule needed an explicit, offset-trained position signal"
    elif m:
        rd = "MIX passes: varying shapes alone remove the count schedule"
    else:
        rd = "both fail: the length barrier is not addressing (the memory's capacity or the delta recursion over 11 columns)"
    res["reading"] = rd; print("READING:", rd)
    if ERRORS:
        print("ERRORS:", ERRORS)
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "score_s161.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
