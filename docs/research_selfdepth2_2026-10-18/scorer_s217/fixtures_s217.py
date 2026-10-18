"""Fixtures for theory/score_s217.py (synthetic trees under ./_fx; never writes under D:/Latent). python -B fixtures_s217.py"""
import copy
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, "D:/Latent/theory")
import score_s217 as S  # noqa: E402

FX = os.path.join(HERE, "_fx"); RES = {}


def ev_vb(acc_9_16, f=1.0, own=0.0, st=1.0):
    a = {str(n): 1.0 for n in range(1, 9)}; a.update({str(n): v for n, v in zip(range(9, 17), acc_9_16)})
    return {"acc_n2": a, "vswap": {"follow_ii": f, "own_ii": own}, "stall": {"recovery": st}}


def ev_deep(acc_17_20):
    return {"acc_n2": {str(n): v for n, v in zip(range(17, 21), acc_17_20)}}


def cdeep(f_by_n=None, own=0.0, stall=1.0, pairs=800, items=1000, gate=(1.0, 0.0, 1.0), card="NVIDIA GeForce RTX 4060"):
    f_by_n = f_by_n or {}
    return {"meta": {"card": card, "dtype": "fp32", "eval_bs": 16, "n_lo": 9, "n_hi": 20, "test_n_max": 24},
            "gate": {"vswap": {"follow_ii": gate[0], "own_ii": gate[1]}, "stall": {"recovery": gate[2]}},
            "vswap": {"by_n": {str(n): {"n_pairs": pairs, "follow_ii": f_by_n.get(n, 1.0), "own_ii": own} for n in range(9, 21)},
                      "by_n_iv": {str(n): {"follow_iv": 0.77} for n in range(9, 21)}, "candidates_by_n": {str(n): 1000 for n in range(9, 21)},
                      "clean_share_by_n": {str(n): pairs / 1000 for n in range(9, 21)}},
            "stall": {"recovery": stall, "by_n": {str(n): {"intact_kn": 1.0, "stalled_kn1": stall, "stall_t": (n + 1) // 2, "items": items} for n in range(9, 21)}}}


BASE = ev_vb([0.923, 0.603, 0.266, 0.127, 0.14, 0.13, 0.13, 0.12]); R1 = ev_vb([1, 1, 1, 1, 0.999, 0.893, 0.633, 0.344])
R2 = ev_vb([1, 1, 1, 1, 1, 1, 0.997, 1]); R2D = ev_deep([0.998, 0.979, 0.862, 0.659])


def tree(name, files):
    root = os.path.join(FX, name); shutil.rmtree(root, ignore_errors=True)
    for (d, f), o in files.items():
        os.makedirs(os.path.join(root, d), exist_ok=True); json.dump(o, open(os.path.join(root, d, f), "w", newline="\n"))
    S.R = root
    return S.score()


def good():
    return {("s213_d3t_s1", "eval_vb.json"): BASE, ("s213_d3t_s1", "eval_causal_deep.json"): cdeep(),
            ("s215_ftself_s1", "eval_vb.json"): R1, ("s215_ftself_s1", "eval_causal_deep.json"): cdeep(),
            ("s216_r2_s1", "eval_vb.json"): R2, ("s216_r2_s1", "eval_deep.json"): R2D, ("s216_r2_s1", "eval_causal_deep.json"): cdeep()}


def fx(name):
    def deco(fn):
        try:
            RES[name] = {"pass": True, "detail": fn() or "ok"}
        except Exception as e:  # noqa: BLE001
            RES[name] = {"pass": False, "detail": "%s: %s" % (type(e).__name__, e)}
        print(("PASS " if RES[name]["pass"] else "FAIL ") + name, flush=True)
        return fn
    return deco


def chk(c, m):
    if not c:
        raise AssertionError(m)


@fx("bands_and_preserved")
def _():
    o = tree("g", good()); M = o["models"]
    chk(M["s213_d3t_s1"]["status"] == "NOT-APPLICABLE", M["s213_d3t_s1"]["label"])
    chk(M["s215_ftself_s1"]["BAND"] == [9, 10, 11, 12, 13], M["s215_ftself_s1"]["BAND"])
    chk(M["s216_r2_s1"]["BAND"] == [9, 10, 11, 12, 13, 14, 15, 16, 17, 18], M["s216_r2_s1"]["BAND"])
    chk(o["lineages"]["s1"]["status"] == "CAUSAL-PRESERVED", o["lineages"]["s1"]); chk(o["lineages"]["s2"]["status"] != "CAUSAL-PRESERVED", o["lineages"]["s2"])
    ov = o["overlap"]["s1"]; chk("s215_ftself_s1 vs s216_r2_s1" in ov and sorted(ov["s215_ftself_s1 vs s216_r2_s1"]) == [9, 10, 11, 12, 13], ov)
    return o["lineages"]["s1"]["label"]


@fx("eroded_swap_own_stall_and_min_n")
def _():
    out = []
    for tag, kw, comp in (("swap", {"f_by_n": {n: 0.93 for n in range(9, 21)}}, "swap follow"), ("own", {"own": 0.03}, "own"), ("stall", {"stall": 0.94}, "STALL"),
                          ("one n 0.89", {"f_by_n": {17: 0.89}}, "swap follow")):
        F = good(); F[("s216_r2_s1", "eval_causal_deep.json")] = cdeep(**kw); o = tree("e", F); L = o["lineages"]["s1"]
        chk(L["status"] == "ERODED-AT" and "s216_r2_s1" in L["label"] and comp in L["label"], (tag, L)); out.append(tag)
    return out


@fx("gate_fail_unreadable")
def _():
    F = good(); F[("s215_ftself_s1", "eval_causal_deep.json")] = cdeep(gate=(0.99, 0.0, 1.0)); o = tree("gt", F)
    chk(o["models"]["s215_ftself_s1"]["status"] == "UNREADABLE" and o["lineages"]["s1"]["status"] == "ERODED-AT" and "gate" in o["lineages"]["s1"]["label"], o["lineages"]["s1"])


@fx("thin_excluded_and_no_deep")
def _():
    F = good(); F[("s216_r2_s1", "eval_causal_deep.json")] = cdeep(pairs=50); o = tree("th", F); m = o["models"]["s216_r2_s1"]
    chk(m["status"] == "NOT-APPLICABLE" and all("thin" in v for v in m["BAND_EXCLUDED"].values()), m["BAND_EXCLUDED"])
    F = good(); del F[("s216_r2_s1", "eval_deep.json")]; o = tree("nd", F); m = o["models"]["s216_r2_s1"]
    chk(m["BAND"] == list(range(9, 17)) and all(n in m["BAND_EXCLUDED"] for n in range(17, 21)), (m["BAND"], m["BAND_EXCLUDED"]))


@fx("pending_and_card")
def _():
    F = good(); del F[("s216_r2_s1", "eval_causal_deep.json")]; o = tree("p", F)
    chk(o["models"]["s216_r2_s1"]["status"] == "PENDING" and o["lineages"]["s1"]["status"] == "PENDING", o["lineages"]["s1"])
    F = good(); F[("s216_r2_s1", "eval_causal_deep.json")] = cdeep(card="NVIDIA GeForce RTX 3070"); o = tree("c", F)
    chk(o["models"]["s216_r2_s1"]["status"] == "PENDING" and "card" in o["models"]["s216_r2_s1"]["label"], o["models"]["s216_r2_s1"]["label"])


@fx("absent_models_so_far")
def _():
    F = good(); F.update({("s216_d3t_s2", "eval_vb.json"): BASE, ("s216_d3t_s2", "eval_causal_deep.json"): cdeep()}); o = tree("a", F); L = o["lineages"]["s2"]
    chk(L["status"] == "NOT-APPLICABLE", L)
    F[("s216_r1_s2", "eval_vb.json")] = R1; F[("s216_r1_s2", "eval_causal_deep.json")] = cdeep()
    keep = dict(S.NEVER_RUN); S.NEVER_RUN.clear()
    try:
        o = tree("a2", F); L = o["lineages"]["s2"]
        chk(L["status"] == "CAUSAL-PRESERVED (so far)" and "s216_r2_s2" in L["label"], L)            # absent but still expected
    finally:
        S.NEVER_RUN.update(keep)
    o = tree("a3", F); L = o["lineages"]["s2"]
    chk(L["status"] == "CAUSAL-PRESERVED" and "216-8" in L["label"] and "s216_r2_s2" in L["never_run"], L)   # never-run model: final label


@fx("real_tree_no_crash")
def _():
    S.R = "D:/Latent/boxes/runs_fsq"; o = S.score()
    return {k: v["label"] for k, v in o["lineages"].items()}


n = sum(v["pass"] for v in RES.values()); print("FIXTURES %d / %d PASS" % (n, len(RES)))
json.dump(RES, open(os.path.join(HERE, "fixtures_report.json"), "w", newline="\n"), indent=1)
if n == len(RES):
    shutil.rmtree(FX, ignore_errors=True)
