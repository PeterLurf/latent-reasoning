"""G-L17 fixtures for score_s218.py (LANG_v1 §4.3 + Addendum 1; scratch, Opus subagent for Theory, 2026-10-18).
Builds synthetic registered files under ./_fx/runs_fsq (this folder only; nothing is written under D:/Latent), points score_s218.R there, and checks that every
case gives the label v1 §4.3 states. The CODI-20 per-item file is read (never written) from D:/Latent and copied into the fixture tree.
Counts, crossovers, AUROCs and McNemar values written into the synthetic files are computed here by independent code, so the scorer's recomputation is tested.
    python -B fixtures_s218.py            -> fixtures_report.json beside this file
"""
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from fractions import Fraction
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_s218 as S  # noqa: E402

FX = os.path.join(HERE, "_fx")
TREE = os.path.join(FX, "runs_fsq")
CODI_REAL = "D:/Latent/boxes/runs_fsq/gsm_codi_20ep_s1/eval_items.json"     # read only
CARD = "NVIDIA GeForce RTX 4060"
SHA_A, SHA_B = "sym_v1_test_sha_fixture", "gsm8k_test_sha_fixture"
EVBS_A = {"acc": 16, "probe": 32, "plant": 32, "fd": 64, "vswap": 16, "stall": 16, "cf": 16, "hist24": 8, "causal_deep": 8, "direct": 32}
EVBS_B = {"acc": 16, "probe": 16, "future": 16, "vswap": 16, "stall": 16, "stepcap": 16, "gsmswap": 16, "cf_gsm": 16}
PROBE_GRID = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12)
ADM = {"ii_prime": (range(4, 9), lambda D: range(3, D), True), "i_prime": (range(5, 9), lambda D: range(3, D - 1), True),
       "ii": (range(4, 9), lambda D: range(3, D), True), "person": (range(4, 9), lambda D: range(3, D), True), "reset": (range(4, 9), lambda D: range(3, D), False)}


class Text(str):
    pass


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else None


@lru_cache(None)
def bern(a, n=1000):
    k = int(round(a * n))
    return tuple([1] * k + [0] * (n - k))


def xover(ans, K):
    """independent brute-force least-violations change point (ties -> smallest c)."""
    best = None
    for c in range(K + 2):
        v = sum(1 for t in range(K + 1) if (t < c and ans[t] == 1) or (t >= c and ans[t] == 0))
        if best is None or v < best[0]:
            best = (v, c)
    return best[1]


def mcnemar(b, c):
    n = b + c
    tot = 0
    for k in range(b, n + 1):
        tot += math.comb(n, k)
    return float(Fraction(tot, 2 ** n)) if n else 1.0


# ---------------------------------------------------------------- materializer
CACHE = {}


def materialize(files):
    for rel in list(CACHE):
        if rel not in files:
            p = os.path.join(TREE, rel)
            if os.path.exists(p):
                os.remove(p)
            del CACHE[rel]
    for rel, obj in files.items():
        if CACHE.get(rel) is obj:
            continue
        p = os.path.join(TREE, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            if isinstance(obj, Text):
                fh.write(obj)
            else:
                json.dump(obj, fh, separators=(",", ":"))
        CACHE[rel] = obj


# ---------------------------------------------------------------- ProsQA builders
def acc_profile(lo, mid, hi, chance=0.5, over=None):
    def f(D, K):
        if over and (D, K) in over:
            return over[(D, K)]
        if over and D in over and K >= D:
            return over[D]
        if K < D:
            return chance
        return lo if D <= 8 else mid if D <= 12 else hi
    return f


def probe_serial(hi=0.95, lo=0.5, over=None):
    def f(t, D):
        if over and (t, D) in over:
            return over[(t, D)]
        return hi if t >= D else lo
    return f


def scores(a, n=100):
    m = int(round(a * n))
    return [1.0] * m + [-1.0] * (n - m), [0.0] * n


def t_serial(D, j, rng):
    return j


def t_serial_la(q):
    return lambda D, j, rng: j - (1 if rng.random() < q else 0)


def vs_arm(arm, follow=None, own=None, clean=1.0, per_D=120, perD=None, t2=12, d4_iprime=20):
    Ds, tr, donor = ADM[arm]
    if follow is None:
        follow = 1.0 - (own or 0.0)
    pairs = []
    Dlist = list(Ds) + ([4] if arm == "i_prime" else [])
    for D in Dlist:
        o = (perD or {}).get(D, {})
        n = o.get("n", per_D if not (arm == "i_prime" and D == 4) else d4_iprime)
        fD, cD = o.get("follow", follow), o.get("clean", clean)
        ts = list(tr(D)) if D in Ds else [3]
        nc = int(round(cD * n)); nf = int(round(fD * nc))
        for i in range(n):
            cl = i < nc; fol = i < nf
            p = {"rec": "r%d_%d" % (D, i), "D": D, "t": ts[i % len(ts)], "y_rec": "oR", "y_hyb": "oP" if fol else "oR", "ok_rec": cl or (i % 2 == 0), "pos_ok": True}
            if donor:
                p.update(don="d%d_%d" % (D, i), y_don="oP", ok_don=cl)
            else:
                p["ok_rec"] = cl
            if arm in ("ii_prime", "ii", "person"):
                p["frontier_differs"] = True
            if arm == "i_prime":
                p.update(lt_differs=True, opts_outside=True)
            pairs.append(p)
    if arm == "ii_prime":
        for i in range(t2):
            pairs.append({"rec": "t2_%d" % i, "don": "t2d_%d" % i, "D": 4 + i % 5, "t": 2, "y_rec": "oR", "y_don": "oP", "y_hyb": "oP", "ok_rec": True,
                          "ok_don": True, "pos_ok": True, "frontier_differs": True})

    def exposed(p):
        return p["D"] in Ds and p["t"] in tr(p["D"])

    def clean_(p):
        return p["ok_rec"] and (p.get("ok_don", True) if donor else True)
    ex = [p for p in pairs if exposed(p)]; cl = [p for p in ex if clean_(p)]
    out = {"n_candidates": len(pairs), "n_exposed": len(ex), "n_clean": len(cl), "follow": mean(int(p["y_hyb"] == "oP") for p in cl) if donor else 0.0,
           "own": mean(int(p["y_hyb"] == "oR") for p in cl), "by_D": {}, "by_t": {}, "t2_descriptive": {}, "pairs": pairs}
    for D in sorted(set(p["D"] for p in pairs)):
        e_ = [p for p in ex if p["D"] == D]; c_ = [p for p in cl if p["D"] == D]
        out["by_D"][str(D)] = {"n_exposed": len(e_), "n_clean": len(c_), "follow": mean(int(p["y_hyb"] == "oP") for p in c_) if donor else 0.0,
                               "own": mean(int(p["y_hyb"] == "oR") for p in c_)}
    if arm == "reset":
        out["recovery"] = out["own"]
    return out


def cf_part(tf, n=60, eps=0.0, seed=7, Ds=range(4, 9), jfun=lambda D: range(1, D + 1), ncell=None):
    rng = random.Random(seed); pairs = []
    for D in Ds:
        for j in jfun(D):
            for i in range(ncell(D, j) if ncell else n):
                t0 = tf(D, j, rng)
                ans = [1 if t >= t0 else 0 for t in range(D + 1)]
                if eps:
                    ans = [a if rng.random() >= eps else 1 - a for a in ans]
                pairs.append({"rec": "x%d_%d_%d" % (D, j, i), "don": "xp%d_%d_%d" % (D, j, i), "D": D, "j": j, "answers_by_t": ans, "ok_rec": ans[D] == 1,
                              "ok_don": ans[0] == 0, "ok_rec_d2": ans[D] == 1, "ok_don_d2": ans[0] == 0, "pos_ok": True, "flip_ok": True})
    C = {"spearman_within_D": {}, "share_window_pooled": None, "last_hop": {}, "pairs": pairs}
    cells = {}
    for p in pairs:
        cells.setdefault((p["D"], p["j"]), []).append(p)
    for (D, j), pp in cells.items():
        cl = [p for p in pp if p["ok_rec"] and p["ok_don"]]; ts = [xover(p["answers_by_t"], D) for p in cl]
        C.setdefault(str(D), {})[str(j)] = {"n_candidates": len(pp), "n_exposed": len(pp), "n_clean": len(cl),
                                            "curve": {str(t): mean(1 - p["answers_by_t"][t] for p in cl) for t in range(D + 1)} if cl else {},
                                            "tstar": sorted(ts), "share_window": mean(int(j - 1 <= x <= j) for x in ts)}
    return C


def hist_part(accf, stabs):
    H = {"items": {}}
    for arm, s in (("intact", None), ("bank_ablate_slva", stabs["bank"]), ("self_ablate", stabs["self"]), ("dup_newest", stabs["dup"])):
        H[arm] = {"acc_d2": {}, "acc_k24": {}}; H["items"][arm] = {}
        for D in range(3, 9):
            d2 = bern(accf(D, D + 2)); k24 = bern(accf(D, 24)) if s is None else bern(round(mean(d2) - s, 3))
            H["items"][arm][str(D)] = {"ok_d2": d2, "ok_k24": k24}; H[arm]["acc_d2"][str(D)] = mean(d2); H[arm]["acc_k24"][str(D)] = mean(k24)
    return H


def cdeep_part(follow=0.99, own=0.0, n=200, rec=1.0):
    by = {str(D): {"n_clean": n, "follow": follow, "own": own} for D in range(9, 17)}
    return {"ii_prime": {"by_D": by}, "i_prime": {"by_D": dict(by)}, "stall": {"by_D": {str(D): {"n": 1000, "recovery": rec} for D in range(9, 17)}}}


def spec_A(kind="a1", **over):
    acc = {"a1": acc_profile(0.97, 0.60, 0.40), "a2": acc_profile(0.995, 0.97, 0.85), "ctrl": acc_profile(0.97, 0.62, 0.40)}[kind]
    s = dict(acc=acc, probe=probe_serial(), plant={0: (0.98, 0.95), 1: (0.98, 0.95), 2: (0.98, 0.95)},
             fd=lambda m, t, k: 0.70 if k == 1 else 0.55 if k == 2 else 0.50, fd_valid=(0.97, 0.96, 0.95),
             vswap=dict(ii_prime=dict(follow=1.0), i_prime=dict(follow=1.0), ii=dict(follow=0.9), person=dict(follow=1.0), reset=dict(own=0.2)),
             stall=1.0, cf=dict(tf=t_serial), hist=dict(bank=0.2, dup=0.15, self=0.7), cdeep={}, steps_run=10000, curve=None, labeller=(kind == "a2"),
             parts=("panel", "acc", "probe", "plant", "fd", "vswap", "stall", "cf", "hist24", "causal_deep") if kind != "ctrl" else ("acc", "hist24"),
             card=CARD, dtype="fp32", evbs=EVBS_A, sha=SHA_A, n_wrong=0, card_by_part=None, acc_corrupt=False, stall_n=None, contamination=None)
    s.update(over)
    return s


CURVE_FLAT = [[st, 0.9] for st in range(500, 10001, 500)]


def ev_lvb(s):
    f = {}; accf = s["acc"]; parts = s["parts"]
    a2, ak, it = {}, {}, {}
    for D in range(3, 17):
        it[str(D)] = {}
        for K in range(25):
            lst = bern(accf(D, K)); it[str(D)][str(K)] = lst; ak.setdefault(str(K), {})[str(D)] = mean(lst)
        a2[str(D)] = ak[str(D + 2)][str(D)]
    if s["acc_corrupt"]:
        ak["10"] = dict(ak["10"]); ak["10"]["8"] = ak["10"]["8"] + 0.01; a2["8"] = ak["10"]["8"]
    f.update(acc_d2=a2, acc_k=ak, items=it, cand_ce_items={}, T_band={})
    if "panel" in parts:
        f["panel"] = {str(D): {"gate_pass": True} for D in range(3, 17)}
    if "probe" in parts:
        pf = s["probe"]
        f["probe"] = {str(t): {"acc_by_D": {str(D): mean(bern(pf(t, D))) for D in range(3, 9)}, "items": {str(D): bern(pf(t, D)) for D in range(3, 9)},
                               "curve": s["curve"] or CURVE_FLAT, "steps_run": s["steps_run"]} for t in PROBE_GRID}
    if "plant" in parts:
        f["plant"] = {str(t): {"lin": {"acc": mean(bern(l_, 200)), "items": bern(l_, 200)}, "xor": {"acc": mean(bern(x_, 200)), "items": bern(x_, 200)},
                               "eps": 0.11, "eps_rel": 0.004, "pcs": [40, 60]} for t, (l_, x_) in s["plant"].items()}
        f["probe_valid"] = {str(t): bool(mean(bern(l_, 200)) >= 0.9 and mean(bern(x_, 200)) >= 0.9) for t, (l_, x_) in s["plant"].items()}
    if "fd" in parts:
        fd = {}
        for m in ("lin", "mlp"):
            fd[m] = {}
            for t in (1, 2):
                fd[m][str(t)] = {}
                for k in range(1, 9 - t):
                    a = s["fd"](m, t, k); pos, neg = scores(a)
                    fd[m][str(t)][str(k)] = {"auroc": int(round(a * 100)) / 100, "n_pos": 100, "n_neg": 100, "pos": pos, "neg": neg}
        f["fd"] = fd; s3, s4, pl = s["fd_valid"]; f["fd_valid"] = {"3": s3, "4": s4}; f["fd_plant"] = {"auroc": pl}
    if "vswap" in parts:
        f["vswap"] = {arm: vs_arm(arm, **kw) for arm, kw in s["vswap"].items()}
    if "stall" in parts:
        items, st, inn = {}, 0, 0
        for D in range(4, 9):
            intact = bern(accf(D, D))[:s["stall_n"]] if s["stall_n"] else bern(accf(D, D)); k = sum(intact); ks = int(round(s["stall"] * k))
            stalled = tuple([1] * ks + [0] * (len(intact) - ks)); items[str(D)] = {"stalled": stalled, "intact": intact}; st += ks; inn += k
        f["stall"] = {"recovery": st / inn, "by_D": {}, "items": items}
    if "cf" in parts:
        f["cf"] = cf_part(**s["cf"])
    if "hist24" in parts:
        f["hist24"] = hist_part(accf, s["hist"])
    if "causal_deep" in parts:
        f["causal_deep"] = cdeep_part(**s["cdeep"])
    if s["labeller"]:
        f.update(gate1={"choice": "L-ROLL"}, kept_vs_drawn={str(D): {"T": [600, 610], "concepts": [70, 71]} for D in range(9, 13)},
                 label_audit={"n": 5000, "n_wrong": s["n_wrong"], "pi_label": 1 - s["n_wrong"] / 5000},
                 memo={str(D): {"acc_pool": 0.975, "acc_fresh": 0.97} for D in range(9, 13)})
    if s["contamination"] is not None:
        f["contamination"] = s["contamination"]
    pl_ = [p for p in parts if p not in ("panel",)] + (["labeller"] if s["labeller"] else [])
    f["meta"] = {"card": s["card"], "dtype": s["dtype"], "tf32": False, "eval_bs": {p: s["evbs"][p] for p in pl_ if p in s["evbs"]}, "items_sha": s["sha"]}
    if s["labeller"]:
        f["meta"]["dtype_by_part"] = {"labeller": "bf16"}
    if s["card_by_part"]:
        f["meta"]["card_by_part"] = s["card_by_part"]
    return f


def ev_direct(acc_by_D, card=CARD):
    it = {str(D): bern(a) for D, a in acc_by_D.items()}
    a = {D: mean(v) for D, v in it.items()}
    band = [D for D in range(5, 9) if all(a[str(x)] <= 0.65 for x in range(D, 9))]
    ds = max((D for D in range(3, 13) if a[str(D)] > 0.65), default=None)
    return {"acc_by_D": a, "items": it, "c1_band": band, "d_star_fresh": ds,
            "meta": {"card": card, "dtype": "fp32", "eval_bs": {"direct": EVBS_A["direct"]}, "items_sha": SHA_A}}


DIRECT_BASE = {3: 0.90, 4: 0.80, 5: 0.60, 6: 0.55, 7: 0.52, 8: 0.51, 9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.5}


def gate_l0(over=None):
    st = {}
    for name, Ds in S.L0_STREAMS.items():
        st[name] = {}
        for D in Ds:
            e = {"n_gate": 5000, "invariant_failures": 0, "max_person_agnostic": 0.52, "chase": {"first": 0.55, "random": 0.54, "maxdeg": 0.55, "la1": 0.55,
                                                                                             "la2": 0.68, "la3": 0.75, "la4": 0.60},
                 "soft": 1.0, "ubidir_max": 0.5, "udist": 1.0, "beyond_D": 0, "gate_pass": True}
            if over:
                over(name, D, e)
            st[name][str(D)] = e
    return {"streams": st, "test_items_sha": SHA_A, "meta": {"card": CARD}}


def log_text(steps, ce=None, va=None, guard=False, reuse=None, coll="auto", hits="auto"):
    """'auto': a ProsQA log (ce given) carries the trainer's stream_collisions and test_gate_sig_hits (0); None = never logged (Audit item 4 cases)."""
    ln = []; coll = (0 if ce else None) if coll == "auto" else coll; hits = (0 if ce else None) if hits == "auto" else hits
    for s_ in range(100, steps + 1, 100):
        r = {"step": s_, "loss": 1.0}
        if ce:
            r["plain_cand_ce"] = ce(s_)
        if va and s_ % 2000 == 0:
            r["valid_acc"] = va(s_)
        if reuse is not None:
            r["self_reuse"] = reuse
        if coll is not None:
            r["stream_collisions"] = coll
        if hits is not None:
            r["test_gate_sig_hits"] = hits
        ln.append(json.dumps(r))
    if guard:
        ln.append(json.dumps({"guard_stop": True}))
    return Text("\n".join(ln) + "\n")


CE_OK = lambda s_: max(0.10, 0.69 - s_ / 10000)  # noqa: E731
CE_NEVER = lambda s_: 0.50  # noqa: E731
CE_FT = lambda s_: 0.12  # noqa: E731
VA_OK = lambda s_: min(0.45, 0.05 + s_ / 100000)  # noqa: E731

D3T_ARGS = {"mode": "slva", "mct": False, "steps": 48000, "bs": 64, "micro": 1, "lr": 0.0001, "seed": 1, "cur": "3,5,8", "cur_gate": "0.3:2:8000",
            "c_sched": "1000,8000", "mu": 1.0, "rec_w": 1.0, "kmax": 12, "kextra": 4, "out": "boxes/runs_fsq/s213_d3t_s1", "train_gpu": "NVIDIA GeForce RTX 3090"}
A1_ARGS = dict(D3T_ARGS, steps=24000, cur="4,6,8", cur_gate="0.35:2:6000", task="prosqa_sym", slots="concept", cand_gate="plain_cand_ce", data_seed=22100,
               self_micro=4, out="boxes/runs_fsq/lvb_sym_s1", arch="lslva", arch_class="LangSLVA")
A2_ARGS = dict(A1_ARGS, init_from="boxes/runs_fsq/lvb_sym_s1/model.pt", self_pool="boxes/runs_fsq/lvb_pool_s1/pool.jsonl", p_self=0.5, self_kmax=16, lr=3e-5,
               steps=12000, c_sched="0,0", out="boxes/runs_fsq/lvb_self_s1")
CTRL_ARGS = dict(A2_ARGS, p_self=0.0, self_pool="none", out="boxes/runs_fsq/lvb_ctrl_s1")
DIRECT_ARGS = {"mode": "direct", "steps": 24000, "bs": 64, "lr": 0.0001, "seed": 1, "task": "prosqa_sym", "out": "boxes/runs_fsq/lvb_direct_s1", "arch": "direct",
               "arch_class": "GPT2KV"}
B1_ARGS = {"mode": "langvb_gsm", "steps": 120000, "bs": 64, "lr": 0.0001, "seed": 1, "decomp": "strict", "out": "boxes/runs_fsq/lvb_gsm_s1"}


def gate1(y_roll=0.5, y_rr=0.52, n=200, wrong=0, choice=None):
    items, chains = {}, {}
    for ch, y in (("L-ROLL", y_roll), ("L-RR", y_rr)):
        items[ch], chains[ch] = {}, {}
        for D in range(9, 13):
            k = int(round(y * n))
            lst = [{"keep": i < k, "label": "oR", "gold": "oR", "stop": D + 1} for i in range(n)]
            if ch == "L-ROLL" and D == 9:
                for i in range(min(wrong, k)):
                    lst[i] = dict(lst[i], label="oP")
            items[ch][str(D)] = lst
            chains[ch][str(D)] = {"n": n, "kept": k, "yield": k / n, "omit_rate": 0.1, "false_in": 0.01, "stop_lag": 0.5, "frontier_hist": {}}
    ch_ = choice or ("L-RR" if y_rr - y_roll >= 0.05 else "L-ROLL")
    return {"meta": {"card": CARD, "dtype": "bf16"}, "choice": ch_, "chains": chains, "items": items}


def pool_meta(kept=106000, draws=300000, sig=0, gold=0, short=None, keys=None):
    k = {str(D): kept for D in range(9, 13)}; d = {str(D): draws for D in range(9, 13)}
    for D, (kk, dd) in (short or {}).items():
        k[str(D)], d[str(D)] = kk, dd
    return {"kept_by_D": k, "draws_by_D": d, "total_kept": sum(k.values()), "demand_replay": 383488, "rng_replay_equal": True,
            "key_set": sorted(keys or S.POOL_KEYS), "sig_hits": {"test": sig, "gate": 0, "probe": 0}, "gold_key_hits": gold}


# ---------------------------------------------------------------- GSM8K builders
N = 1319
GSM_M = [(3 + i % 6) if i < 879 else (1 + i % 2) for i in range(N)]
GSM_MBIN = [m + 1 for m in GSM_M]
with open(CODI_REAL, encoding="utf-8") as _fh:
    CODI_ITEMS = json.load(_fh)
CODI_OK = [int(bool(x["ok"])) for x in CODI_ITEMS]


def b1_ok(b, c):
    right = [i for i, x in enumerate(CODI_OK) if x]; wrong = [i for i, x in enumerate(CODI_OK) if not x]
    ok = [0] * N
    for i in right[c:]:
        ok[i] = 1
    for i in wrong[:b]:
        ok[i] = 1
    return ok


def gsm_arm(pair_fn=None, bases=range(60), variants=3):
    pf = pair_fn or (lambda b, v: ("follow", True))
    pairs = []
    for b in bases:
        nv = variants(b) if callable(variants) else variants
        for v in range(nv):
            kind, cl = pf(b, v)
            yr, yd = 10.0 + b, 20.0 + b
            yh = yd if kind == "follow" else yr if kind == "own" else 99.5
            pairs.append({"rec": "g%d_%d" % (b, v), "don": "gd%d_%d" % (b, v), "base": b, "variant": v, "step": 2, "m_bin": GSM_MBIN[b], "y_rec": yr,
                          "y_don": yd, "y_hyb": yh, "ok_rec": cl, "ok_don": cl, "pos_ok": True})
    cl = [p for p in pairs if p["ok_rec"] and p["ok_don"]]
    return {"n_candidates": len(pairs), "n_exposed": len(pairs), "n_clean": len(cl), "follow": mean(int(p["y_hyb"] == p["y_don"]) for p in cl),
            "own": mean(int(p["y_hyb"] == p["y_rec"]) for p in cl), "neither": mean(int(p["y_hyb"] not in (p["y_rec"], p["y_don"])) for p in cl),
            "by_m": {}, "pairs": pairs}


def cf_gsm_part(tf, bases=range(60), variants=3, js=(3, 4)):
    pairs = []
    for b in bases:
        mb = GSM_MBIN[b]
        for j in [x for x in js if x <= mb - 1] + [mb]:
            for v in range(variants):
                t0 = tf(j, mb); ans = [1 if t >= t0 else 0 for t in range(15)]
                pairs.append({"rec": "c%d_%d_%d" % (b, j, v), "don": "cd%d_%d_%d" % (b, j, v), "base": b, "variant": v, "j": j, "m_bin": mb, "answers_by_t": ans,
                              "ok_rec": ans[14] == 1, "ok_don": ans[0] == 0, "pos_ok": True, "flip_ok": True})
    C = {"share_window": None, "spearman": None, "last_step": {}, "late_fusion_median_tstar": None, "pairs": pairs}
    for j in sorted(set(p["j"] for p in pairs)):
        pp = [p for p in pairs if p["j"] == j]; cl = [p for p in pp if p["ok_rec"] and p["ok_don"]]
        C[str(j)] = {"n_candidates": len(pp), "n_exposed": len(pp), "n_clean": len(cl), "curve": {}, "tstar": sorted(xover(p["answers_by_t"], 14) for p in cl)}
    return C


def ev_gsm(ok, probe_lo=None, future=None, arms=None, stall_rec=1.0, cf=None, card=CARD, stepcap=0.8, codi_override=None):
    items = [{"i": i, "ok": bool(ok[i]), "pred_txt": "The answer is: %d" % i, "m": GSM_M[i], "m_bin": GSM_MBIN[i]} for i in range(N)]
    by_m = {str(m): mean(ok[i] for i in range(N) if GSM_M[i] == m) for m in sorted(set(GSM_M))}
    plo = probe_lo or [ok[i] if i % 4 == 0 else 0 for i in range(N)]
    probe = {}
    for t in S.GSM_PROBE_GRID:
        it = plo if t <= 1 else ok
        probe[str(t)] = {"acc_by_m": {str(m): mean(it[i] for i in range(N) if GSM_M[i] == m) for m in sorted(set(GSM_M))}, "items": list(it), "curve": []}
    band = [i for i in range(N) if GSM_M[i] >= 3]
    pv = mean(probe["14"]["items"][i] for i in band) >= mean(ok[i] for i in band) - 0.03
    fut = future or (lambda m, t, k: 0.05 if t == 1 else 0.6 if t == k else 0.3)
    fu = {m: {str(t): {str(k): {"acc": fut(m, t, k), "n": 100} for k in range(t, 13)} for t in range(0, 13)} for m in ("lin", "mlp")}
    b = sum(1 for x, y in zip(ok, CODI_OK) if x and not y); c = sum(1 for x, y in zip(ok, CODI_OK) if y and not x)
    cp = codi_override or {"delta": (b - c) / N, "se_paired": None, "mcnemar": {"b": b, "c": c, "p_one_sided": mcnemar(b, c)}, "n": N, "by_m": {}, "by_parse": {}}
    f = {"acc_fixed14": mean(ok), "items": items, "by_m": by_m, "by_mbin": {}, "acc_k": {"14": mean(ok)}, "acc_oracle": mean(ok), "acc_halt": mean(ok),
         "codi_pair": cp, "probe": probe, "probe_valid_gsm": pv, "plant_gsm": {"lin": {"acc": 0.97}, "xor": {"acc": 0.93}}, "future": fu,
         "vswap": arms or {a: gsm_arm() for a in ("ii_prime", "i", "ii", "iv")},
         "stall": {"recovery": None, "by_m": {}, "items": None}, "stepcap": {"by_t": {}, "by_op": {}, "by_digits": {}, "pooled": stepcap}, "gsmswap": {},
         "cf_gsm": cf or cf_gsm_part(lambda j, mb: j)}
    sit = [{"i": i, "stalled": 0, "intact": ok[i]} for i in range(N) if GSM_MBIN[i] >= 3]
    cnt, tgt = 0, int(round(stall_rec * sum(x["intact"] for x in sit)))         # stalled: the first round(rec * #intact-correct) correct items stay correct
    for x in sit:
        x["stalled"] = int(x["intact"] and cnt < tgt); cnt += x["stalled"]
    f["stall"] = {"recovery": sum(x["stalled"] for x in sit) / sum(x["intact"] for x in sit), "by_m": {}, "items": sit}
    f["meta"] = {"card": card, "dtype": "fp32", "tf32": False, "eval_bs": dict(EVBS_B), "items_sha": SHA_B}
    return f


def interim(acc=0.35, p1=0.20, f1=0.10, fk=0.50, go=None, card=CARD):
    g = (acc - p1 >= 0.05) and (f1 < 0.8 * fk)
    return {"meta": {"card": card, "dtype": "fp32"}, "step": 24000, "acc_fixed14_valid": acc, "probe1_valid": p1, "fd_mlp_1k_ge3": f1, "fd_mlp_kk_ge3": fk,
            "GO": g if go is None else go}


def b02(card=CARD):
    return {"meta": {"card": card, "dtype": "fp32"}, "items_sha": SHA_B, "acc": mean(CODI_OK), "items": [dict(x) for x in CODI_ITEMS]}


# ---------------------------------------------------------------- the base tree (pure serial everywhere: every primary passes)
def base_files():
    F = {}
    F["s213_d3t_s1/args.json"] = D3T_ARGS
    for cell, a_ in (("direct", DIRECT_ARGS), ("a1", A1_ARGS), ("b1", B1_ARGS)):     # note 218-4: queued args frozen by sha256 (FROZEN_ARGS set in main)
        F["lvb_frozen/%s_args.json" % cell] = dict(a_)
    F["lvb_gates/gate_l0.json"] = gate_l0()
    F["lvb_gates/gate_evmem.json"] = {"card": CARD, "eval_bs": dict(EVBS_A)}
    F["lvb_gates/gate_evmem_gsm.json"] = {"card": CARD, "eval_bs": dict(EVBS_B)}
    F["lvb_direct_s1/log.jsonl"] = log_text(24000, CE_OK)
    F["lvb_direct_s1/args.json"] = dict(DIRECT_ARGS)
    F["lvb_direct_s1/eval_direct.json"] = ev_direct(DIRECT_BASE)
    F["lvb_sym_s1/log.jsonl"] = log_text(24000, CE_OK)
    F["lvb_sym_s1/args.json"] = A1_ARGS
    F["lvb_sym_s1/eval_lvb.json"] = ev_lvb(spec_A("a1"))
    F["lvb_gate1_s1/eval_gate1.json"] = gate1()
    F["lvb_pool_s1/pool_meta.json"] = pool_meta()
    F["lvb_self_s1/log.jsonl"] = log_text(12000, CE_FT, reuse=0)
    F["lvb_self_s1/args.json"] = A2_ARGS
    F["lvb_self_s1/eval_lvb.json"] = ev_lvb(spec_A("a2"))
    F["lvb_ctrl_s1/log.jsonl"] = log_text(12000, CE_FT)
    F["lvb_ctrl_s1/args.json"] = CTRL_ARGS
    F["lvb_ctrl_s1/eval_lvb.json"] = ev_lvb(spec_A("ctrl"))
    F["gsm_codi_20ep_s1/eval_items.json"] = CODI_ITEMS
    F["lvb_b02/codi20_rescore.json"] = b02()
    F["lvb_gsm_s1/log.jsonl"] = log_text(120000, va=VA_OK)
    F["lvb_gsm_s1/args.json"] = dict(B1_ARGS)
    F["lvb_gsm_s1/interim_go.json"] = interim()
    F["lvb_gsm_s1/eval_lvb_gsm.json"] = ev_gsm(b1_ok(120, 0))
    return F


def cf_gsm_custom(spec):
    """CF-GSM store from (base, j, n_variants, t0) rows (every candidate exposed, clean iff the endpoints are y / y')."""
    pairs = []
    for b, j, nv, t0 in spec:
        mb = GSM_MBIN[b]
        for v in range(nv):
            ans = [1 if t >= t0 else 0 for t in range(15)]
            pairs.append({"rec": "c%d_%d_%d" % (b, j, v), "don": "cd%d_%d_%d" % (b, j, v), "base": b, "variant": v, "j": j, "m_bin": mb, "answers_by_t": ans,
                          "ok_rec": ans[14] == 1, "ok_don": ans[0] == 0, "pos_ok": True, "flip_ok": True})
    C = {"share_window": None, "spearman": None, "last_step": {}, "late_fusion_median_tstar": None, "pairs": pairs}
    for j in sorted(set(p["j"] for p in pairs)):
        pp = [p for p in pairs if p["j"] == j]; cl = [p for p in pp if p["ok_rec"] and p["ok_don"]]
        C[str(j)] = {"n_candidates": len(pp), "n_exposed": len(pp), "n_clean": len(cl), "curve": {}, "tstar": sorted(xover(p["answers_by_t"], 14) for p in cl)}
    return C


def t_mix(share, other, la=0.5):
    """Addendum 1 item 2 mixtures: an item is serial (look-ahead la) with probability share, else precompute (t* 1) or recompute (t* D)."""
    def f(D, j, rng):
        if rng.random() < share:
            return j - (1 if rng.random() < la else 0)
        return 1 if other == "pre" else D
    return f


def cf_flags(C, extra=20):
    """a full CF store with the eval's per-candidate 'exposed' flags instead of pos_ok / flip_ok, plus `extra` NON-exposed candidates per cell whose answers
    read precompute (t* 1): they must never count."""
    pairs = []
    for p in C["pairs"]:
        q = {k: v for k, v in p.items() if k not in ("pos_ok", "flip_ok")}; q["exposed"] = 1; q["selected"] = 1; pairs.append(q)
    out = {k: v for k, v in C.items() if not str(k).isdigit() and k != "pairs"}
    for D, row in C.items():
        if not str(D).isdigit():
            continue
        out[D] = {}
        for j, x in row.items():
            for i in range(extra):
                ans = [0] + [1] * int(D)
                pairs.append({"rec": "nx%s_%s_%d" % (D, j, i), "D": int(D), "j": int(j), "answers_by_t": ans, "ok_rec": True, "ok_don": True,
                              "ok_rec_d2": True, "ok_don_d2": True, "exposed": 0, "selected": 0})
            out[D][j] = dict(x, n_candidates=x["n_candidates"] + extra)
    out["pairs"] = pairs
    return out


TEST_SHA_D = {str(D): "sha_D%d" % D for D in range(3, 17)}


def to_exp_eval(ev):
    """convert a LAYOUTS-format eval_lvb.json into Experiments' langvb_eval layout (2026-10-18 15:10): top-level card / dtype / eval_bs / test_sha {D: sha};
    swap pairs with 'exposed' and int y_hyb (1 = donor's answer); cf cells under cf.cells, selected exposed pairs with clean / tstar / clean_k2;
    fd per-slot scores in fd_slot_scores, fd_valid {s: {model: {auroc}}}, fd_plant {model: auroc}; plant items {D: [...]}; causal_deep stall rows."""
    f = {k: v for k, v in ev.items() if k != "meta"}; m = ev["meta"]
    f.update(card=m["card"], dtype=m["dtype"], tf32=False, eval_bs=dict(m["eval_bs"]), test_sha=dict(TEST_SHA_D))
    if "dtype_by_part" in m:
        f["dtype_by_part"] = m["dtype_by_part"]
    if "vswap" in f:
        V = {}
        for arm, A in f["vswap"].items():
            need = {"ii_prime": ("pos_ok", "frontier_differs"), "ii": ("pos_ok", "frontier_differs"), "i_prime": ("pos_ok", "lt_differs", "opts_outside"),
                    "person": ("pos_ok",), "reset": ()}[arm]
            ps, t2 = [], []
            for p in A["pairs"]:
                q = {"rec": p["rec"], "don": "person" if arm == "person" else "premise", "D": p["D"], "t": p["t"], "y_hyb": int(p["y_hyb"] == "oP"),
                     "ok_rec": int(p["ok_rec"]), "ok_don": int(p.get("ok_don", 1)), "exposed": int(all(p.get(x) for x in need))}
                q.update({x: p[x] for x in need})                            # Audit item 9: the stored exposure facts stay; 'exposed' is the cross-check
                (t2 if (arm == "ii_prime" and p["t"] == 2) else ps).append(q)
            A2_ = dict(A, pairs=ps, n_candidates=len(ps))
            if arm == "ii_prime":                                            # the module pops its t = 2 arm into a summary without pairs
                A2_["t2_descriptive"] = {"n_candidates": len(t2), "n_clean": sum(1 for q in t2 if q["ok_rec"] and q["ok_don"]), "follow": 1.0, "own": 0.0}
            if arm == "reset":
                A2_["follow"] = None if A["own"] is None else 1 - A["own"]
                A2_["by_D"] = {D: dict(x, follow=None if x["own"] is None else 1 - x["own"]) for D, x in A["by_D"].items()}
            V[arm] = A2_
        f["vswap"] = V
    if "cf" in f:
        C = f["cf"]; cells = {D: v for D, v in C.items() if str(D).isdigit()}
        ps = [{"rec": p["rec"], "D": p["D"], "j": p["j"], "answers_by_t": p["answers_by_t"], "clean": int(p["ok_rec"] and p["ok_don"]),
               "clean_k2": int(p["ok_rec_d2"] and p["ok_don_d2"]), "tstar": xover(p["answers_by_t"], p["D"])} for p in C["pairs"]]
        cells = {D: {j: dict(x, n_selected=x["n_exposed"], tstar=list(reversed(x["tstar"]))) for j, x in v.items()} for D, v in cells.items()}
        f["cf"] = {"cells": cells, "spearman_within_D": {}, "n_decisive_clean_by_D": {}, "share_window_pooled": None, "last_hop": {}, "pairs": ps}
    if "fd" in f:
        fd, sl = {}, {}
        for mo, T in f["fd"].items():
            fd[mo] = {}
            for t, Ks in T.items():
                fd[mo][t] = {}
                for k, e in Ks.items():
                    fd[mo][t][k] = {"auroc": e["auroc"], "n_pos": e["n_pos"], "n_neg": e["n_neg"], "by_D": {}}
                    sl["%s_t%s_k%s" % (mo, t, k)] = [[i, 0, 1, v] for i, v in enumerate(e["pos"])] + [[i, 1, 0, v] for i, v in enumerate(e["neg"])]
        f["fd"] = fd; f["fd_slot_scores"] = sl
        f["fd_valid"] = {s_: {mo: {"auroc": ev["fd_valid"][s_]} for mo in ("lin", "mlp")} for s_ in ("3", "4")}
        f["fd_plant"] = {mo: ev["fd_plant"]["auroc"] for mo in ("lin", "mlp")}
    if "plant" in f:
        f["plant"] = {t: {"eps": e["eps"], "pcs": e["pcs"], **{x: {"acc": e[x]["acc"], "items": {"3": list(e[x]["items"][:100]), "4": list(e[x]["items"][100:])},
                                                              "curve": [], "steps_run": 10000} for x in ("lin", "xor")}} for t, e in f["plant"].items()}
    if "causal_deep" in f:
        cd = dict(f["causal_deep"])
        cd["stall"] = {"recovery": 1.0, "by_D": {D: {"intact_kD": 1.0, "stalled_kD1": v["recovery"], "stall_t": 5, "items": v["n"]} for D, v in cd["stall"]["by_D"].items()}}
        f["causal_deep"] = cd
    return f


def to_exp_direct(ev):
    m = ev["meta"]
    return {"direct": {k: ev[k] for k in ("acc_by_D", "items", "c1_band", "d_star_fresh")}, "card": m["card"], "dtype": m["dtype"], "tf32": False,
            "eval_bs": dict(m["eval_bs"]), "test_sha": dict(TEST_SHA_D), "arch": "direct"}


def to_exp_l0(g):
    P, FR = {}, {}
    for st, rows in g["streams"].items():
        for D, e in rows.items():
            p = P.setdefault(D, {"n_gate": e["n_gate"], "invariant_failures": e["invariant_failures"], "chase_by_stream": {}, "max_person_agnostic_by_stream": {},
                                 "ubidir_by_stream": {}, "udist_by_stream": {}, "beyond_D_by_stream": {}})
            p["chase_by_stream"][st] = e["chase"]; p["max_person_agnostic_by_stream"][st] = e["max_person_agnostic"]
            p["ubidir_by_stream"][st] = {str(r): e["ubidir_max"] for r in range(1, (int(D) + 1) // 2)}; p["udist_by_stream"][st] = e["udist"]
            p["beyond_D_by_stream"][st] = float(e["beyond_D"])
            FR.setdefault(st, {})[D] = {"n": e["n_gate"], "module_invariant_fail": e["invariant_failures"], "indep_invariant_fail": 0}
    return {"spec": "fixture", "panel": P, "full_rows": FR, "usable_bands": list(range(3, 17)), "pass": True, "test_items_sha": dict(TEST_SHA_D)}


def exp_layout(files):
    F = dict(files)
    for rel in ("lvb_sym_s1/eval_lvb.json", "lvb_self_s1/eval_lvb.json", "lvb_ctrl_s1/eval_lvb.json"):
        F[rel] = to_exp_eval(F[rel])
    F["lvb_direct_s1/eval_direct.json"] = to_exp_direct(F["lvb_direct_s1/eval_direct.json"])
    F["lvb_gates/gate_l0.json"] = to_exp_l0(F["lvb_gates/gate_l0.json"])
    F["lvb_gates/gate_evmem.json"] = {"evmem": {p: {"peak_gib_by_bs": {}, "eval_bs": b, "D": 16} for p, b in EVBS_A.items()}, "card": CARD, "budget_gib": 7.0}
    return F


MOD_PART_DS = ("acc", "stall", "vswap", "cf", "hist24", "causal_deep", "probe", "fd")


def to_mod_eval(ev, extra_candidates=True):
    """a LAYOUTS-format eval_lvb.json reshaped EXACTLY as boxes/langvb_eval.py @ ec1aec8d writes it (Theory's field list, 2026-10-18): meta at the top level
    (card, dtype, tf32, per_D, c, test_sha {D}, test_items_sha, items_sha {part: full sha}, items_sha_range, card_by_part, eval_bs incl. plant); stall by_D
    rows; swap pairs {rec, don, D, t, y_hyb int, ok_rec, ok_don, exposed, pos_ok, frontier_differs | lt_differs + opts_outside}, t = 2 popped into
    ii_prime.t2_descriptive; cf cells at cf.<D>.<j> with n_selected, every candidate stored (+ non-exposed and exposed-not-selected candidates), selected
    pairs with answers_by_t / tstar / endpoints_clean; plant {lin|xor: {acc, acc_by_D, items {D}, items_flat, curve, steps_run}, eps, eps_rel, pcs}; fd cells
    with by_D and pos / neg, fd_valid {s: {model: {auroc, ...}}}, fd_plant {model: auroc}, fd_slot_scores; causal_deep stall in the stall layout; band_B."""
    f = {k: v for k, v in ev.items() if k != "meta"}; m = ev["meta"]
    parts = [q for q in MOD_PART_DS if (q in f) or (q == "acc" and "acc_d2" in f)]
    f.update(card=m["card"], dtype=m["dtype"], tf32=False, per_D=1000, c=0.0, test_sha=dict(TEST_SHA_D), test_items_sha=m["items_sha"],
             items_sha={q: m["items_sha"] for q in parts}, items_sha_range={q: "range_" + q for q in parts}, card_by_part={q: m["card"] for q in parts},
             eval_bs=dict(m["eval_bs"]), t_train_max=860)
    if "dtype_by_part" in m:
        f["dtype_by_part"] = m["dtype_by_part"]
    if "stall" in f:
        st = f["stall"]
        f["stall"] = {"recovery": st["recovery"], "items": {D: {"i": list(range(len(x["intact"]))), "intact": list(x["intact"]), "stalled": list(x["stalled"])}
                                                            for D, x in st["items"].items()},
                      "by_D": {D: {"intact_kD": mean(x["intact"]), "stalled_kD1": mean(x["stalled"]), "stall_t": (int(D) + 1) // 2, "items": len(x["intact"]),
                                   "n": len(x["intact"]), "recovery": (sum(x["stalled"]) / sum(x["intact"])) if sum(x["intact"]) else None}
                               for D, x in st["items"].items()}}
    if "vswap" in f:
        V = {}
        for arm, A in f["vswap"].items():
            need = {"ii_prime": ("frontier_differs",), "ii": ("frontier_differs",), "person": (), "i_prime": ("lt_differs", "opts_outside"),
                    "reset": ()}[arm]
            ps, t2 = [], []
            for p_ in A["pairs"]:
                q = {"rec": p_["rec"], "don": "person" if arm == "person" else "premise", "D": p_["D"], "t": p_["t"], "y_hyb": int(p_["y_hyb"] == "oP"),
                     "ok_rec": int(p_["ok_rec"]), "ok_don": None if arm == "reset" else int(p_.get("ok_don", 1)), "exposed": int(all(p_.get(x) for x in need)),
                     "pos_ok": 1}
                q.update({x: int(bool(p_[x])) for x in need})
                if arm == "person":
                    q["frontier_differs"] = int(bool(p_.get("frontier_differs")))         # kept by the writer, not its exposure
                (t2 if (arm == "ii_prime" and p_["t"] == 2) else ps).append(q)
            okd = lambda q: q["ok_don"] if q["ok_don"] is not None else 1  # noqa: E731   (RESET: ok_don null -> clean = ok_rec)
            ex = [q for q in ps if q["exposed"]]; cl = [q for q in ex if q["ok_rec"] and okd(q)]
            fol = lambda qq: (mean(q["y_hyb"] for q in qq) if qq else None)  # noqa: E731
            byD = {}
            for D in sorted({q["D"] for q in ps}):
                e_ = [q for q in ex if q["D"] == D]; c_ = [q for q in e_ if q["ok_rec"] and okd(q)]
                byD[str(D)] = {"n_candidates": sum(1 for q in ps if q["D"] == D), "n_exposed": len(e_), "n_clean": len(c_), "follow": fol(c_), "own": None if not c_ else 1 - fol(c_)}
            # the writer's exposure ignores the (D, t) window (t is drawn inside it); the scorer applies the window, so non-admissible pairs (i_prime at D 4)
            # are not written by the module: drop them here
            adm = {"i_prime": lambda q: q["D"] >= 5 and q["t"] <= q["D"] - 2}.get(arm, lambda q: True)
            ps = [q for q in ps if adm(q)]; ex = [q for q in ps if q["exposed"]]; cl = [q for q in ex if q["ok_rec"] and okd(q)]
            byD = {D: x for D, x in byD.items() if any(q["D"] == int(D) for q in ps)}
            V[arm] = {"n_candidates": len(ps), "n_exposed": len(ex), "n_clean": len(cl), "follow": fol(cl), "own": None if not cl else 1 - fol(cl), "by_D": byD,
                      "by_t": {}, "pairs": ps}
            if arm == "ii_prime":                                            # the writer: t = 2 pairs in vswap.ii_prime_t2, its summary in ii_prime.t2_descriptive
                c2 = [q for q in t2 if q["ok_rec"] and q["ok_don"]]
                V[arm]["t2_descriptive"] = {"n_candidates": len(t2), "n_exposed": len(t2), "n_clean": len(c2), "follow": fol(c2), "own": None if not c2 else 1 - fol(c2),
                                            "by_D": {}, "by_t": {}}
                V["ii_prime_t2"] = dict(V[arm]["t2_descriptive"], pairs=t2)
        f["vswap"] = V
    if "cf" in f:
        C = f["cf"]; pairs = []
        for p_ in C["pairs"]:
            D, j, ab = p_["D"], p_["j"], p_["answers_by_t"]; cl_ = bool(p_["ok_rec"] and p_["ok_don"])
            pairs.append({"rec": p_["rec"], "don": "cf_j%d" % j, "D": D, "j": j, "pos_ok": 1, "flip_ok": 1, "exposed": 1, "ok_rec": int(p_["ok_rec"]),
                          "ok_rec_d2": int(p_["ok_rec_d2"]), "ok_don": int(p_["ok_don"]), "ok_don_d2": int(p_["ok_don_d2"]), "selected": 1, "clean": int(cl_),
                          "clean_k2": int(p_["ok_rec_d2"] and p_["ok_don_d2"]), "answers_by_t": ab, "tstar": xover(ab, D),
                          "endpoints_clean": int(ab[D] == 1 and ab[0] == 0)})
        cells = {}
        for D in sorted({p_["D"] for p_ in pairs}):
            for j in sorted({p_["j"] for p_ in pairs if p_["D"] == D}):
                if extra_candidates:                                         # 3 non-exposed (flip fails; ok_don null) + 2 exposed-not-selected per cell
                    for i in range(3):
                        pairs.append({"rec": 900 + i, "don": "cf_j%d" % j, "D": D, "j": j, "pos_ok": 1, "flip_ok": 0, "exposed": 0, "ok_rec": 1, "ok_rec_d2": 1,
                                      "ok_don": None, "ok_don_d2": None, "selected": 0, "clean": 0, "clean_k2": 0})
                    for i in range(2):
                        pairs.append({"rec": 950 + i, "don": "cf_j%d" % j, "D": D, "j": j, "pos_ok": 1, "flip_ok": 1, "exposed": 1, "ok_rec": 1, "ok_rec_d2": 1,
                                      "ok_don": 1, "ok_don_d2": 1, "selected": 0, "clean": 1, "clean_k2": 1})
                ps = [q for q in pairs if q["D"] == D and q["j"] == j]; ex = [q for q in ps if q["exposed"]]; sl = [q for q in ex if q["selected"]]
                cl = [q for q in sl if q["clean"]]
                cells.setdefault(str(D), {})[str(j)] = {"n_candidates": len(ps), "n_exposed": len(ex), "n_selected": len(sl), "n_clean": len(cl),
                                                        "curve": {str(t): (mean(1 - q["answers_by_t"][t] for q in cl) if cl else None) for t in range(D + 1)},
                                                        "tstar": [q["tstar"] for q in cl], "share_window": mean(int(q["tstar"] in (j - 1, j)) for q in cl) if cl else None,
                                                        "share_clean_kd": (len(cl) / len(sl)) if sl else None,
                                                        "share_clean_kd2": (sum(q["clean_k2"] for q in sl) / len(sl)) if sl else None}
        f["cf"] = dict(cells, spearman_within_D={}, n_decisive_clean_by_D={}, share_window_pooled=None, n_decisive_clean=None, last_hop={}, pairs=pairs,
                       endpoint_mismatch=sum(1 for q in pairs if "endpoints_clean" in q and q["endpoints_clean"] != q["clean"]))
    if "plant" in f:
        f["plant"] = {t: {"eps": e["eps"], "eps_rel": e["eps_rel"], "pcs": e["pcs"],
                          **{x: {"acc": e[x]["acc"], "acc_by_D": {}, "items": {"3": list(e[x]["items"][:100]), "4": list(e[x]["items"][100:])},
                                 "items_flat": list(e[x]["items"]), "curve": [], "steps_run": 10000} for x in ("lin", "xor")}} for t, e in f["plant"].items()}
    if "fd" in f:
        sl = {}
        for mo, T in f["fd"].items():
            for t, Ks in T.items():
                for k, e in Ks.items():
                    e["by_D"] = {}
                    sl["%s_t%s_k%s" % (mo, t, k)] = [[i, 0, 1, v] for i, v in enumerate(e["pos"])] + [[i, 1, 0, v] for i, v in enumerate(e["neg"])]
        f["fd_valid"] = {s_: {mo: {"auroc": ev["fd_valid"][s_], "n_pos": 100, "n_neg": 100, "by_D": {}, "pos": [], "neg": []} for mo in ("lin", "mlp")} for s_ in ("3", "4")}
        f["fd_plant"] = {mo: ev["fd_plant"]["auroc"] for mo in ("lin", "mlp")}; f["fd_slot_scores"] = sl; f["fd_eps"] = {"1": 0.11, "2": 0.11}
    if "causal_deep" in f:
        cd = dict(f["causal_deep"])
        cd["stall"] = {"recovery": 1.0, "items": {}, "by_D": {D: {"intact_kD": 1.0, "stalled_kD1": v["recovery"], "stall_t": (int(D) + 1) // 2, "items": v["n"],
                                                                   "n": v["n"], "recovery": v["recovery"]} for D, v in cd["stall"]["by_D"].items()}}
        cd["band_B"] = [D for D in range(9, 17) if f["acc_d2"][str(D)] >= 0.95]
        f["causal_deep"] = cd
    f["contamination"] = {"test_gate_sig_hits": 0, "self_reuse": 0}
    f.update(probe_n_distinct_seen=192000, fd_n_distinct_seen=200000, fd_plant_dir={"eps": 0.11}, fd_eps={"1": 0.11, "2": 0.11})
    f["eval_bs"]["plant"] = f["eval_bs"].get("probe", 32)
    return f


def mod_layout(files):
    """the base tree with every Stage A eval in the writer of record's shape, the G-L0 in its built panel layout + test_items_sha (str) + _by_D, eval_direct
    top-level with top-level meta, G-EVMEM {card, eval_bs}; logs without stream_collisions / test_gate_sig_hits (evidence = the contamination block)."""
    F = dict(files)
    for rel in ("lvb_sym_s1/eval_lvb.json", "lvb_self_s1/eval_lvb.json", "lvb_ctrl_s1/eval_lvb.json"):
        F[rel] = to_mod_eval(F[rel])
    ed = F["lvb_direct_s1/eval_direct.json"]; m = ed["meta"]
    F["lvb_direct_s1/eval_direct.json"] = {k: ed[k] for k in ("acc_by_D", "items", "c1_band", "d_star_fresh")}
    F["lvb_direct_s1/eval_direct.json"].update(card=m["card"], dtype="fp32", tf32=False, per_D=1000, test_sha=dict(TEST_SHA_D), test_items_sha=SHA_A,
                                               items_sha={"direct": SHA_A}, card_by_part={"direct": m["card"]}, eval_bs={"direct": EVBS_A["direct"]},
                                               contamination={"test_gate_sig_hits": 0, "self_reuse": None})
    g = to_exp_l0(F["lvb_gates/gate_l0.json"]); g["test_items_sha"] = SHA_A; g["test_items_sha_by_D"] = dict(TEST_SHA_D)
    F["lvb_gates/gate_l0.json"] = g
    F["lvb_gates/gate_evmem.json"] = {"card": CARD, "eval_bs": dict(EVBS_A, plant=EVBS_A["probe"]), "peaks": {}}
    for rel, steps, ce, reuse in (("lvb_direct_s1/log.jsonl", 24000, CE_OK, None), ("lvb_sym_s1/log.jsonl", 24000, CE_OK, None),
                                  ("lvb_self_s1/log.jsonl", 12000, CE_FT, 0), ("lvb_ctrl_s1/log.jsonl", 12000, CE_FT, None)):
        F[rel] = log_text(steps, ce, reuse=reuse, coll=None, hits=None)
    return F


def tiny_eval(ev, n=2):
    """a LAYOUTS-format eval cut to n items per D (acc / probe / hist24 / stall, summaries recomputed so every consistency assert still holds) and stamped
    like a writer smoke: per_D n, card 'cpu', items_sha off-anchor, an eval_bs off G-EVMEM, probe steps_run 2 with no curve."""
    e = json.loads(json.dumps(ev))
    for D, row in e["items"].items():
        for K in row:
            row[K] = row[K][:n]; e["acc_k"][K][D] = mean(row[K])
        e["acc_d2"][D] = e["acc_k"][str(int(D) + 2)][D]
    for t, x in e["probe"].items():
        for D in x["items"]:
            x["items"][D] = x["items"][D][:n]; x["acc_by_D"][D] = mean(x["items"][D])
        x["steps_run"] = 2; x["curve"] = []
    H = e["hist24"]
    for arm, rows in H["items"].items():
        for D, r in rows.items():
            r["ok_d2"], r["ok_k24"] = r["ok_d2"][:n], r["ok_k24"][:n]; H[arm]["acc_d2"][D] = mean(r["ok_d2"]); H[arm]["acc_k24"][D] = mean(r["ok_k24"])
    st = e["stall"]
    for D, r in st["items"].items():
        r["stalled"], r["intact"] = r["stalled"][:n], r["intact"][:n]
    st["recovery"] = sum(sum(r["stalled"]) for r in st["items"].values()) / sum(sum(r["intact"]) for r in st["items"].values())
    e["meta"].update(per_D=n, card="cpu", items_sha="tiny_sha", eval_bs=dict(e["meta"]["eval_bs"], probe=99))
    return e


def configure():
    """the fixture tree: R, OUT and the G-L0 path point under _fx; FROZEN_ARGS are filled by freeze() after the base tree is written."""
    S.R = TREE; S.OUT = os.path.join(FX, "out.json"); S.PATHS["gate_l0"] = ("lvb_gates", "gate_l0.json")


def sha_file(rel):
    with open(os.path.join(TREE, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def freeze():
    """note 218-4: Theory records (path, sha256) of each queued args file before launch; here from the materialized fixture files."""
    for cell in ("direct", "a1", "b1"):
        rel = "lvb_frozen/%s_args.json" % cell
        S.FROZEN_ARGS[cell] = (os.path.join(TREE, rel), sha_file(rel))


# ---------------------------------------------------------------- cases
BASE = None
SHIPPED_FROZEN = {}
REPORT = []


def run(mods=None, drop=(), **kw):
    files = dict(BASE); files.update(mods or {})
    for d in drop:
        files.pop(d, None)
    materialize(files)
    return S.score(**kw)


def A1(**over):
    return {"lvb_sym_s1/eval_lvb.json": ev_lvb(spec_A("a1", **over))}


def A2(**over):
    return {"lvb_self_s1/eval_lvb.json": ev_lvb(spec_A("a2", **over))}


def strip_part(rel, keys, files=None):
    f = dict((files or BASE)[rel])
    for k in keys:
        f.pop(k, None)
    return {rel: f}


class Case:
    def __init__(self, name, expect):
        self.name, self.expect, self.checks, self.obs = name, expect, [], {}

    def ok(self, cond, what):
        self.checks.append((bool(cond), what))
        return bool(cond)


def case(name, expect):
    def deco(fn):
        c = Case(name, expect); t0 = time.time()
        try:
            fn(c)
        except Exception:                                                   # noqa: BLE001
            c.checks.append((False, "exception: " + traceback.format_exc()[-1500:]))
        REPORT.append({"case": name, "expect": expect, "result": "PASS" if c.checks and all(x for x, _ in c.checks) else "FAIL",
                       "failed_checks": [w for x, w in c.checks if not x], "n_checks": len(c.checks), "observed": c.obs, "sec": round(time.time() - t0, 2)})
        print("%-4s %s" % (REPORT[-1]["result"], name), flush=True)
        return fn
    return deco


def comps(cell):
    return cell.get("VB_C1_L_components") or {}


def lab(cell):
    return cell.get("LABELS") or {}


def rdg(cell, k):
    return (cell.get("READINGS") or {}).get(k) or {}


def define_cases():
    @case("VB-C1-L: pure serial (person-tracking S-NAME, question-conditional) passes every primary", "VB-C1-L; no counter-model label")
    def _(c):
        o = run(); a = o["cells"]["a1"]; c.obs = {"label": a["label"], "components": comps(a)}
        c.ok(a["VB_C1_L"] is True, "VB_C1_L"); c.ok(all(v == "MET" for v in comps(a).values()), "all seven components MET")
        for x in ("PRECOMPUTE_L", "CF_CARRY", "CF_RECOMPUTE", "CF_FAST", "ANCHOR_TAGGED", "RECOMPUTE_L", "ALL_PERSONS_SIGNATURE", "HISTORY_READ", "COUNT_KEYED"):
            c.ok(lab(a).get(x) in (False, None), "%s not set" % x)
        c.ok("C1 band D [5, 6, 7, 8]" in (a["WORDING"]["VB_C1_L"] or "") and "hop-indexed consumption" in a["WORDING"]["VB_C1_L"], "pre-stated wording (+ Addendum 1 CF clause)")
        c.ok(a["MEM"]["KEEPS_SELF_STAB_L"] is True, "KEEPS-SELF-STAB-L")
        c.ok(o["cells"]["direct"]["C1"]["band"] == [5, 6, 7, 8] and o["cells"]["direct"]["C1"]["d_star_fresh"] == 4, "C1 band / D*_fresh from DIRECT-SYM")
        cf = rdg(a, "CF_L"); c.obs["cf"] = {k: cf.get(k) for k in ("share_window", "spearman_within_D", "n_clean")}
        c.ok(cf.get("share_window") == 1.0 and abs(cf.get("spearman_within_D") - 1.0) < 1e-12, "CF share 1, rho 1")

    @case("A2 SD-SELF-L and SD-C2-L (A2 + required A2-CTRL)", "SD-SELF-L, SD-C2-L; A2 GO")
    def _(c):
        o = run(); a = o["cells"]["a2"]; c.obs = {"label": a["label"], "A2_GO": o["FLOW"]["A2_GO"]["go"]}
        c.ok(o["FLOW"]["A2_GO"]["go"] is True, "A2 GO"); c.ok(a["SD_SELF_L"] is True, "SD_SELF_L"); c.ok(a["SD_C2_L"] is True, "SD_C2_L")
        c.ok(a["WORDING"]["SD_C2_L"] and "LIT-C2-L UNINFORMATIVE" in a["WORDING"]["SD_C2_L"], "SD-C2-L wording with MEM / CAUSAL-DEEP / LIT-C2-L on the line")
        c.ok(a["BESIDE"]["CAUSAL_DEEP_L"] == "CAUSAL-DEEP-L", "CAUSAL-DEEP-L beside (B = D 9-12)")
        c.ok(abs(a["A2_READINGS"]["MARGIN_GROWS_L"]["bar_A1_9_12_plus_010"] - 0.70) < 1e-9, "MARGIN bar = A1 mean D 9-12 + 0.10 from A1's file")

    @case("B1 VB-C1-GSM (synthetic: 60 base items with CF steps j >= 3)", "VB-C1-GSM; Stage B wording 'C4 via distillation from CoT ...'")
    def _(c):
        o = run(); b = o["cells"]["b1"]; c.obs = {"label": b["label"], "B_GO": o["FLOW"]["B_GO"]["go"]}
        c.ok(o["FLOW"]["B_GO"]["go"] is True, "B GO = A1 VB-C1-L"); c.ok(b["VB_C1_GSM"] is True, "VB_C1_GSM")
        c.ok(all(x["met"] for x in b["READINGS"].values()), "five components MET")
        c.ok("C4 via distillation from CoT, with CoT-parsed pointer annotations" in (b["WORDING"]["VB_C1_GSM"] or ""), "strict DECOMP wording")
        c.ok(b["INTERIM"]["GO"] is True and b["B1_24K"] is False, "interim GO recomputed")

    @case("BEATS-CODI20: exact McNemar on a constructed pair table (b 120, c 0)", "BEATS-CODI20")
    def _(c):
        o = run(); r = o["cells"]["b1"]["ACCURACY"]; c.obs = {k: r.get(k) for k in ("LABEL", "b", "c", "delta", "p_one_sided")}
        c.ok(r["LABEL"] == "BEATS-CODI20" and r["met"], "BEATS"); c.ok(r["b"] == 120 and r["c"] == 0, "b / c recomputed vs B0-2")
        c.ok(abs(r["p_one_sided"] - 2.0 ** -120) < 1e-45, "p = 2^-120"); c.ok(abs(r["delta"] - 120 / 1319) < 1e-12, "delta")
        c.ok("cannot" in (o["cells"]["b1"]["WORDING"]["ACCURACY"] or ""), "what it can / cannot claim")

    @case("ONESHOT-L (PROBE(2) >= 0.8 at D 8)", "ONESHOT-L")
    def _(c):
        o = run(A1(probe=probe_serial(over={(2, 8): 0.85}))); a = o["cells"]["a1"]; c.obs = {"label": a["label"]}
        c.ok(lab(a)["ONESHOT_L"] is True, "ONESHOT_L")

    @case("LOSES-ONE-PASS (not DEEP-L, PROBE-GAIN < 0.15)", "LOSES-ONE-PASS; not VB-C1-L")
    def _(c):
        o = run(A1(acc=acc_profile(0.97, 0.60, 0.40, over={8: 0.85}), probe=probe_serial(over={(1, D): 0.90 for D in range(5, 9)})))
        a = o["cells"]["a1"]; c.obs = {"label": a["label"], "gain": rdg(a, "PROBE_GAIN_L").get("gain")}
        c.ok(lab(a)["LOSES_ONE_PASS"] is True, "LOSES_ONE_PASS"); c.ok(not a["VB_C1_L"], "not VB"); c.ok(rdg(a, "DEEP_L")["status"] == "NOT MET", "DEEP NOT MET")
        c.ok(o["cells"]["a2"]["status"] == "UNREAD" and o["cells"]["b1"]["status"] == "UNREAD", "A2 / B1 UNREAD (no GO)")

    @case("COUNT-KEYED (STALL recovery 0.5)", "COUNT-KEYED; NO-CLOCK-L not met")
    def _(c):
        o = run(A1(stall=0.5)); a = o["cells"]["a1"]; c.obs = {"label": a["label"], "stall": rdg(a, "NO_CLOCK_L").get("recovery")}
        c.ok(lab(a)["COUNT_KEYED"] is True, "COUNT_KEYED"); c.ok(rdg(a, "NO_CLOCK_L")["status"] == "NOT MET", "NO_CLOCK NOT MET"); c.ok(not a["VB_C1_L"], "not VB")

    @case("NO-ENTRY (plain_cand_ce never <= 0.35 in two consecutive 1k bins)", "FAILURE (NO-ENTRY; rule b); A2 / B1 UNREAD")
    def _(c):
        o = run({"lvb_sym_s1/log.jsonl": log_text(24000, CE_NEVER)}); a = o["cells"]["a1"]; c.obs = {"label": a["label"]}
        c.ok(a["status"] == "FAIL" and a["censors"]["NO_ENTRY"] and "NO-ENTRY" in a["label"], "NO-ENTRY censor"); c.ok(not a["VB_C1_L"], "not VB")
        c.ok(o["cells"]["a2"]["status"] == "UNREAD" and o["cells"]["b1"]["status"] == "UNREAD", "downstream UNREAD")

    @case("dead run (log short of 24,000; and a guard stop)", "FAILURE (dead; rule b)")
    def _(c):
        o = run({"lvb_sym_s1/log.jsonl": log_text(20000, CE_OK)}); a = o["cells"]["a1"]; c.obs = {"short": a["label"]}
        c.ok(a["status"] == "FAIL" and a["censors"]["dead"] and not a["VB_C1_L"], "short log dead")
        o = run({"lvb_sym_s1/log.jsonl": log_text(24000, CE_OK, guard=True)}); a = o["cells"]["a1"]; c.obs["guard"] = a["label"]
        c.ok(a["status"] == "FAIL" and a["censors"]["dead"], "guard stop counts as dead")

    @case("a G-L0-failed band (test stream D 6, LA(1) chase 0.61)", "DEEP-L / C1 band UNREADABLE; not VB-C1-L")
    def _(c):
        def ov(st, D, e):
            if st == "test" and D == 6:
                e["chase"] = dict(e["chase"], la1=0.61)
        o = run({"lvb_gates/gate_l0.json": gate_l0(ov)}); a = o["cells"]["a1"]; c.obs = {"label": a["label"], "failed": o["G_L0"]["failed"]}
        c.ok(o["G_L0"]["failed"] == [6], "band 6 failed"); c.ok("G-L0-failed band [6]" in rdg(a, "DEEP_L")["status"], "DEEP UNREADABLE")
        c.ok(not o["cells"]["direct"]["C1"]["readable"], "C1 band unreadable"); c.ok(not a["VB_C1_L"], "not VB")

    @case("G-L0 CODE-BUG (uBIDIR 0.51 in the pool stream at D 10)", "band 10 unusable and flagged CODE-BUG; D 4-8 readings unaffected")
    def _(c):
        def ov(st, D, e):
            if st == "pool" and D == 10:
                e["ubidir_max"] = 0.51
        o = run({"lvb_gates/gate_l0.json": gate_l0(ov)}); c.obs = {"failed": o["G_L0"]["failed"], "bug": o["G_L0"]["code_bug"]}
        c.ok(o["G_L0"]["failed"] == [10] and o["G_L0"]["code_bug"], "CODE-BUG flagged"); c.ok(o["cells"]["a1"]["VB_C1_L"] is True, "VB unaffected")

    @case("G-L0-failed band inside A2's range (test stream D 14)", "A2 DEPTH-BEYOND-SELF-L / MARGIN / BEATS UNREADABLE -> not SD-C2-L; SD-SELF-L and A1 VB-C1-L unaffected")
    def _(c):
        def ov(st, D, e):
            if st == "test" and D == 14:
                e["max_person_agnostic"] = 0.62
        o = run({"lvb_gates/gate_l0.json": gate_l0(ov)}); a = o["cells"]["a2"]; m = a["A2_READINGS"]; c.obs = {k: m[k]["status"] for k in m}
        c.ok(all("G-L0-failed band [14]" in m[k]["status"] for k in ("DEPTH_BEYOND_SELF_L", "MARGIN_GROWS_L", "BEATS_CTRL_L")), "D 13-16 readings UNREADABLE")
        c.ok(a["SD_SELF_L"] is True and a["SD_C2_L"] is False and o["cells"]["a1"]["VB_C1_L"] is True, "SD-SELF only; VB unaffected")

    @case("CF cell with 420 exposed pairs (> the registered 400)", "flagged, not voided; CF-L still read (met)")
    def _(c):
        o = run(A1(cf=dict(tf=t_serial, ncell=lambda D, j: 420 if (D, j) == (6, 4) else 60))); r = rdg(o["cells"]["a1"], "CF_L")
        c.obs = {"flags": r.get("flags"), "status": r["status"]}; c.ok(r["met"] and r.get("flags") and "420 selected" in r["flags"][0], "flag")

    @case("A2-CTRL at --final with A2 GO undecided and never launched", "UNREAD (never a pass)")
    def _(c):
        drop = ["lvb_gate1_s1/eval_gate1.json"] + [k for k in BASE if k.startswith("lvb_ctrl_s1/")]
        o = run(drop=drop, final=("ctrl",)); x = o["cells"]["ctrl"]; c.obs = {"ctrl": x["label"][:200]}
        c.ok(x["status"] == "UNREAD", "UNREAD"); o = run(drop=drop); c.ok(o["cells"]["ctrl"]["status"] == "PENDING", "pending before --final")

    @case("empty C1 band (D*_fresh 7)", "EMPTY-C1-BAND; PROBE-GAIN-L UNREADABLE; not VB-C1-L")
    def _(c):
        o = run({"lvb_direct_s1/eval_direct.json": ev_direct({**DIRECT_BASE, 7: 0.70, 8: 0.60})}); a = o["cells"]["a1"]; C1 = o["cells"]["direct"]["C1"]
        c.obs = {"band": C1["band"], "d_star": C1["d_star_fresh"], "probe_gain": rdg(a, "PROBE_GAIN_L")["status"]}
        c.ok(C1["band"] == [8] and C1["EMPTY_C1_BAND"] and C1["d_star_fresh"] == 7, "band [8], D* 7"); c.ok(lab(a)["EMPTY_C1_BAND"] is True, "EMPTY-C1-BAND label")
        c.ok("EMPTY-C1-BAND" in rdg(a, "PROBE_GAIN_L")["status"], "PROBE-GAIN UNREADABLE"); c.ok(not a["VB_C1_L"], "not VB")

    @case("(ii') with 99 exposed clean pairs", "SERIAL-STEP-L UNREADABLE")
    def _(c):
        pd = {D: {"clean": 20 / 120} for D in range(4, 8)}; pd[8] = {"clean": 19 / 120}
        vs = dict(spec_A("a1")["vswap"], ii_prime=dict(follow=1.0, perD=pd))
        o = run(A1(vswap=vs)); a = o["cells"]["a1"]; s = rdg(a, "SERIAL_STEP_L")["ii_prime"]; c.obs = {"ii_prime": s["status"], "n_clean": s.get("n_clean")}
        c.ok(s.get("n_clean") == 99 and "99 < 100" in s["status"], "99 clean -> UNREADABLE"); c.ok(not a["VB_C1_L"], "not VB")

    @case("(i') per-D exclusion (< 50 exposed clean pairs at D 5)", "D 5 excluded and listed, D 4 by construction; pooled bar still read (met)")
    def _(c):
        vs = dict(spec_A("a1")["vswap"], i_prime=dict(follow=1.0, perD={5: {"n": 40}}))
        o = run(A1(vswap=vs)); a = o["cells"]["a1"]; s = rdg(a, "SERIAL_STEP_L")["i_prime"]
        c.obs = {"status": s["status"], "excluded": s.get("excluded_listed"), "by_construction": s.get("excluded_by_construction")}
        c.ok(5 in s["excluded_listed"] and 4 in s["excluded_by_construction"], "excluded and listed"); c.ok(s["met"] and a["VB_C1_L"], "pooled bar read, VB-C1-L")

    @case("PROBE-INVALID (PLANT lin 0.85 at t 1)", "PROBE-GAIN-L and SERIAL-ONSET-L UNREADABLE; not VB-C1-L")
    def _(c):
        o = run(A1(plant={0: (0.98, 0.95), 1: (0.85, 0.95), 2: (0.98, 0.95)})); a = o["cells"]["a1"]
        c.obs = {k: rdg(a, k)["status"] for k in ("PROBE_GAIN_L", "SERIAL_ONSET_L")}
        c.ok("PROBE-INVALID(1)" in rdg(a, "PROBE_GAIN_L")["status"], "PROBE-GAIN UNREADABLE"); c.ok("PROBE-INVALID" in rdg(a, "SERIAL_ONSET_L")["status"], "ONSET UNREADABLE")
        c.ok(not a["VB_C1_L"], "not VB")

    @case("FD-INVALID (fd_valid s3 0.85)", "NO-PRECOMPUTE-L UNREADABLE; PRECOMPUTE-L not set; not VB-C1-L")
    def _(c):
        o = run(A1(fd_valid=(0.85, 0.96, 0.95))); a = o["cells"]["a1"]; c.obs = {"np": rdg(a, "NO_PRECOMPUTE_L")["status"]}
        c.ok("FD-INVALID" in rdg(a, "NO_PRECOMPUTE_L")["status"], "FD-INVALID"); c.ok(lab(a)["PRECOMPUTE_L"] is False, "no PRECOMPUTE-L"); c.ok(not a["VB_C1_L"], "not VB")

    @case("PaR-C precompute-and-carry: (ii') follow 1.0, (i') own 1.0, FD 0.95 at k >= 3, STALL 1.0, ON slope 1.0, CF t* = 1", "not VB-C1-L, PRECOMPUTE-L, CF-CARRY")
    def _(c):
        vs = dict(spec_A("a1")["vswap"], ii_prime=dict(follow=1.0), i_prime=dict(follow=0.0))
        o = run(A1(vswap=vs, fd=lambda m, t, k: 0.95 if k >= 3 else 0.7, cf=dict(tf=lambda D, j, r: 1))); a = o["cells"]["a1"]
        c.obs = {"label": a["label"], "components": comps(a)}
        c.ok(not a["VB_C1_L"], "not VB"); c.ok(lab(a)["PRECOMPUTE_L"] is True, "PRECOMPUTE-L"); c.ok(lab(a)["CF_CARRY"] is True, "CF-CARRY")
        c.ok(abs(rdg(a, "SERIAL_ONSET_L")["slope"] - 1.0) < 1e-12 and rdg(a, "NO_CLOCK_L")["met"], "ON slope 1.0, STALL 1.0 (as specified)")
        c.ok(rdg(a, "SERIAL_STEP_L")["i_prime"]["own"] == 1.0, "(i') own 1.0")

    @case("PaR-C, probe-undetectable variant: FD 0.5, (i') follow 1.0, CF t* = 1", "not VB-C1-L with CF-L the only failing component; CF-CARRY")
    def _(c):
        o = run(A1(cf=dict(tf=lambda D, j, r: 1))); a = o["cells"]["a1"]; c.obs = {"components": comps(a)}
        c.ok(not a["VB_C1_L"], "not VB"); c.ok([k for k, v in comps(a).items() if v != "MET"] == ["CF_L"], "only CF-L fails"); c.ok(lab(a)["CF_CARRY"] is True, "CF-CARRY")

    @case("ALL-PERSONS serial: PERSON own 1.0, (ii') follow 1.0, (i') follow 1.0, FD 0.5, PLANT valid, CF t* = j", "VB-C1-L + ALL-PERSONS signature")
    def _(c):
        vs = dict(spec_A("a1")["vswap"], person=dict(follow=0.0)); o = run(A1(vswap=vs)); a = o["cells"]["a1"]; c.obs = {"label": a["label"]}
        c.ok(a["VB_C1_L"] is True, "VB-C1-L"); c.ok(lab(a)["ALL_PERSONS_SIGNATURE"] is True, "ALL-PERSONS signature")

    @case("ALL-ANCHORS serial: (ii') own 1.0, (i') own 1.0, RESET recovery 0.5, CF t* = j", "CF-L met, not SERIAL-STEP-L, ANCHOR-TAGGED, not RECOMPUTE-L")
    def _(c):
        vs = dict(spec_A("a1")["vswap"], ii_prime=dict(own=1.0), i_prime=dict(own=1.0), reset=dict(own=0.5)); o = run(A1(vswap=vs)); a = o["cells"]["a1"]
        c.obs = {"label": a["label"]}
        c.ok(rdg(a, "CF_L")["met"], "CF-L met"); c.ok(rdg(a, "SERIAL_STEP_L")["status"] == "NOT MET", "not SERIAL-STEP-L")
        c.ok(lab(a)["ANCHOR_TAGGED"] is True and lab(a)["RECOMPUTE_L"] is False, "ANCHOR-TAGGED, not RECOMPUTE-L"); c.ok(not a["VB_C1_L"], "not VB")

    @case("recompute-from-V_0: (ii') own 1.0, (i') own 1.0, RESET recovery 1.0, CF t* = D", "RECOMPUTE-L, CF-RECOMPUTE, not CF-L")
    def _(c):
        vs = dict(spec_A("a1")["vswap"], ii_prime=dict(own=1.0), i_prime=dict(own=1.0), reset=dict(own=1.0))
        o = run(A1(vswap=vs, cf=dict(tf=lambda D, j, r: D))); a = o["cells"]["a1"]; c.obs = {"label": a["label"]}
        c.ok(lab(a)["RECOMPUTE_L"] is True and lab(a)["ANCHOR_TAGGED"] is False, "RECOMPUTE-L"); c.ok(lab(a)["CF_RECOMPUTE"] is True, "CF-RECOMPUTE")
        c.ok(rdg(a, "CF_L")["status"] == "NOT MET", "not CF-L")

    @case("HISTORY-READ ((ii) own 0.6, (ii') follow 1.0)", "HISTORY-READ")
    def _(c):
        vs = dict(spec_A("a1")["vswap"], ii=dict(follow=0.4)); o = run(A1(vswap=vs)); a = o["cells"]["a1"]
        c.obs = {"ii_own": lab(a)["ii_own"]}; c.ok(lab(a)["HISTORY_READ"] is True, "HISTORY-READ")

    @case("A2 MARGIN-GROWS fail (A2 D 13-16 0.82 < A1 D 9-12 0.80 + 0.10)", "SD-SELF-L, not SD-C2-L (MARGIN NOT MET)")
    def _(c):
        o = run({"lvb_sym_s1/eval_lvb.json": ev_lvb(spec_A("a1", acc=acc_profile(0.97, 0.80, 0.40))),
                 "lvb_self_s1/eval_lvb.json": ev_lvb(spec_A("a2", acc=acc_profile(0.995, 0.97, 0.82)))}); a = o["cells"]["a2"]
        m = a["A2_READINGS"]; c.obs = {k: m[k]["status"] for k in m}
        c.ok(m["MARGIN_GROWS_L"]["status"] == "NOT MET" and m["DEPTH_BEYOND_SELF_L"]["met"] and m["BEATS_CTRL_L"]["met"], "only MARGIN fails")
        c.ok(a["SD_SELF_L"] is True and a["SD_C2_L"] is False, "SD-SELF, not SD-C2")

    @case("A2-CTRL eval missing: pending, and at --final a2 -> not SD-C2-L (SD-SELF-L still read)", "pending; --final: not SD-C2-L")
    def _(c):
        o = run(drop=["lvb_ctrl_s1/eval_lvb.json"]); a = o["cells"]["a2"]; c.obs = {"pending": a["label"][:200]}
        c.ok(a["status"] == "PENDING" and "A2-CTRL" in a["label"], "A2 pending on A2-CTRL")
        o = run(drop=["lvb_ctrl_s1/eval_lvb.json"], final=("a2",)); a = o["cells"]["a2"]; c.obs["final"] = a["label"]
        c.ok(a["SD_SELF_L"] is True and a["SD_C2_L"] is False and "A2-CTRL" in a["A2_READINGS"]["BEATS_CTRL_L"]["status"], "--final: not SD-C2-L")

    @case("A2-CTRL censored (dead) -> BEATS-CTRL-L not met", "not SD-C2-L")
    def _(c):
        o = run({"lvb_ctrl_s1/log.jsonl": log_text(8000, CE_FT)}); a = o["cells"]["a2"]; c.obs = {"ctrl": o["cells"]["ctrl"]["label"], "a2": a["label"]}
        c.ok(o["cells"]["ctrl"]["status"] == "FAIL" and a["SD_C2_L"] is False and a["SD_SELF_L"] is True, "BEATS-CTRL unreadable")

    @case("MEM-TRADED-L in A1 (STAB_self 0.3): later stages need the user's explicit OK", "A2 / A2-CTRL / B1 UNREAD; with --user_ok A2,B1 they are scored")
    def _(c):
        m = A1(hist=dict(bank=0.2, dup=0.15, self=0.3)); o = run(m); c.obs = {"a1": o["cells"]["a1"]["label"], "a2": o["cells"]["a2"]["label"][:200]}
        c.ok(o["cells"]["a1"]["MEM"]["MEM_TRADED_L"] is True, "A1 MEM-TRADED-L")
        c.ok(o["cells"]["a2"]["status"] == "UNREAD" and "MEM-TRADED" in o["cells"]["a2"]["label"], "A2 UNREAD (needs OK)")
        c.ok(o["cells"]["ctrl"]["status"] == "UNREAD" and o["cells"]["b1"]["status"] == "UNREAD" and "MEM-TRADED" in o["cells"]["b1"]["label"], "CTRL / B1 UNREAD")
        c.ok(o["FLOW"]["MEM_TRADED_L"]["cells"] == ["A1"], "flow flag")
        o = run(m, user_ok=("A2", "B1")); c.obs["with_ok"] = (o["cells"]["a2"]["status"], o["cells"]["b1"]["status"])
        c.ok(o["cells"]["a2"]["status"] == "SCORED" and o["cells"]["b1"]["status"] == "SCORED", "scored with the recorded OK")

    @case("MEM-TRADED-L in A2 only: B1 needs the user's explicit OK", "B GO false without OK")
    def _(c):
        o = run(A2(hist=dict(bank=0.2, dup=0.15, self=0.3))); c.obs = {"b_go": o["FLOW"]["B_GO"]}
        c.ok(o["cells"]["a2"]["MEM"]["MEM_TRADED_L"] is True and o["FLOW"]["B_GO"]["go"] is False and o["cells"]["b1"]["status"] == "UNREAD", "B1 UNREAD")
        o = run(A2(hist=dict(bank=0.2, dup=0.15, self=0.3)), user_ok=("B1",)); c.ok(o["cells"]["b1"]["status"] == "SCORED", "OK recorded -> scored")

    @case("CONTAMINATED (pool signature hit; and self_reuse > 0)", "FAILURE (CONTAMINATED; rule b)")
    def _(c):
        o = run({"lvb_pool_s1/pool_meta.json": pool_meta(sig=1)}); a = o["cells"]["a2"]; c.obs = {"pool": a["label"]}
        c.ok(a["status"] == "FAIL" and "CONTAMINATED" in a["label"] and not a["SD_SELF_L"], "pool hit")
        o = run({"lvb_self_s1/log.jsonl": log_text(12000, CE_FT, reuse=1)}); a = o["cells"]["a2"]; c.obs["reuse"] = a["label"]
        c.ok(a["status"] == "FAIL" and a["censors"]["CONTAMINATED"], "self_reuse")

    @case("POOL-SHORT (D 12 kept 100,000 at the 530,000 draw cap); pool incomplete below the cap", "A2 and A2-CTRL UNREAD; incomplete -> pending")
    def _(c):
        o = run({"lvb_pool_s1/pool_meta.json": pool_meta(short={12: (100000, 530000)})}); c.obs = {"a2": o["cells"]["a2"]["label"]}
        c.ok(o["cells"]["a2"]["status"] == "UNREAD" and "POOL-SHORT" in o["cells"]["a2"]["label"] and o["cells"]["ctrl"]["status"] == "UNREAD", "UNREAD")
        o = run({"lvb_pool_s1/pool_meta.json": pool_meta(short={12: (100000, 400000)})}); c.obs["incomplete"] = o["cells"]["a2"]["label"][:160]
        c.ok(o["cells"]["a2"]["status"] == "PENDING", "incomplete pool -> pending")

    @case("LABEL-BUG (one wrong label in LABEL-AUDIT-L; and one in GATE-1-L with A2 launched)", "VOID (LABEL-BUG; rule b)")
    def _(c):
        o = run(A2(n_wrong=1)); a = o["cells"]["a2"]; c.obs = {"audit": a["label"]}
        c.ok(a["status"] == "FAIL" and "LABEL-BUG" in a["label"], "audit")
        o = run({"lvb_gate1_s1/eval_gate1.json": gate1(wrong=1)}); a = o["cells"]["a2"]; c.obs["gate"] = a["label"]
        c.ok(a["status"] == "FAIL" and "LABEL-BUG at GATE-1-L" in a["label"], "gate")

    @case("Audit 2a LABEL-BUG: a GATE-1-L tie (A2 launched), an unsound audited record, a tie in the pool", "VOID (LABEL-BUG) x 3")
    def _(c):
        g = gate1(); g["chains"]["L-RR"]["10"]["n_tie"] = 1; g["label_bug"] = True
        o = run({"lvb_gate1_s1/eval_gate1.json": g}); c.obs = {"tie": o["cells"]["a2"]["label"][:120]}; c.ok("LABEL-BUG" in o["cells"]["a2"]["label"], "GATE-1-L tie")
        e = dict(BASE["lvb_self_s1/eval_lvb.json"]); e["label_audit"] = dict(e["label_audit"], n_unsound=1)
        o = run({"lvb_self_s1/eval_lvb.json": e}); c.ok(o["cells"]["a2"]["status"] == "FAIL" and "LABEL-BUG" in o["cells"]["a2"]["label"], "label_audit n_unsound")
        pm = dict(pool_meta(), n_tie=2, label_bug=True); o = run({"lvb_pool_s1/pool_meta.json": pm}); c.ok("LABEL-BUG" in o["cells"]["a2"]["label"], "pool ties")
        g2 = gate1(); g2["label_bug"] = True; o = run({"lvb_gate1_s1/eval_gate1.json": g2}); c.ok(o["cells"]["a2"]["status"] in ("FAIL", "UNREAD"), "file label_bug alone")

    @case("GATE-1-L fails (kept yield 0.15 < 0.2) / choice not the N9 rule's", "A2 GO false -> A2, A2-CTRL UNREAD")
    def _(c):
        o = run({"lvb_gate1_s1/eval_gate1.json": gate1(y_roll=0.15, y_rr=0.16)}); c.obs = {"go": o["FLOW"]["A2_GO"]["why"]}
        c.ok(o["cells"]["a2"]["status"] == "UNREAD" and o["cells"]["ctrl"]["status"] == "UNREAD", "yield")
        o = run({"lvb_gate1_s1/eval_gate1.json": gate1(y_roll=0.50, y_rr=0.58, choice="L-ROLL")}); c.obs["choice"] = o["FLOW"]["A2_GO"].get("GATE_1_L_detail")
        c.ok(o["cells"]["a2"]["status"] == "UNREAD", "file choice L-ROLL vs recomputed L-RR")
        o = run({"lvb_gate1_s1/eval_gate1.json": gate1(y_roll=0.50, y_rr=0.54)})
        c.ok(o["FLOW"]["A2_GO"]["go"] is True and o["FLOW"]["A2_GO"]["GATE_1_L_detail"]["choice"] == "L-ROLL", "tie rule: +0.04 stays L-ROLL")

    @case("PASS1-PRECOMPUTE-GSM (PROBE(1) == acc)", "PASS1-PRECOMPUTE-GSM; not VB-C1-GSM; §208 / §204 wording")
    def _(c):
        ok = b1_ok(120, 0); o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(ok, probe_lo=list(ok))}); b = o["cells"]["b1"]; c.obs = {"label": b["label"]}
        c.ok(b["LABELS"]["PASS1_PRECOMPUTE_GSM"] is True and not b["VB_C1_GSM"], "PASS1"); c.ok(b["WORDING"]["PASS1_PRECOMPUTE_GSM / ONE_PASS_CHAIN_L"], "wording")

    @case("ONE-PASS-CHAIN-L (linear FD(1, k>=3) 0.55 >= 0.8 x FD(k, k) 0.6)", "ONE-PASS-CHAIN-L; NOT-ONE-PASS-CHAIN not met")
    def _(c):
        fut = lambda m, t, k: (0.55 if m == "lin" else 0.05) if t == 1 else 0.6 if t == k else 0.3  # noqa: E731
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), future=fut)}); b = o["cells"]["b1"]; c.obs = {"nop": b["READINGS"]["NOT_ONE_PASS_CHAIN"]["status"]}
        c.ok(b["LABELS"]["ONE_PASS_CHAIN_L"] is True and b["READINGS"]["NOT_ONE_PASS_CHAIN"]["status"] == "NOT MET" and not b["VB_C1_GSM"], "ONE-PASS-CHAIN-L")

    @case("FD(k, k) 0.2 (GSM8K)", "NOT-ONE-PASS-CHAIN UNREADABLE (never a vacuous pass)")
    def _(c):
        fut = lambda m, t, k: 0.05 if t == 1 else 0.2 if t == k else 0.3  # noqa: E731
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), future=fut)}); b = o["cells"]["b1"]; s = b["READINGS"]["NOT_ONE_PASS_CHAIN"]["status"]
        c.obs = {"nop": s}; c.ok(s.startswith("UNREADABLE") and "FD(k, k) < 0.3" in s and not b["VB_C1_GSM"], "UNREADABLE")

    @case("accuracy: delta 0.035 with McNemar p ~0.07", "ABOVE-NS, not BEATS-CODI20")
    def _(c):
        best = None
        for cc in range(0, 501):
            bb = cc + 46
            if bb > 819:
                break
            p = mcnemar(bb, cc)
            if best is None or abs(p - 0.07) < abs(best[2] - 0.07):
                best = (bb, cc, p)
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(best[0], best[1]))}); r = o["cells"]["b1"]["ACCURACY"]
        c.obs = {k: r.get(k) for k in ("LABEL", "b", "c", "delta", "p_one_sided")}
        c.ok(r["LABEL"] == "ABOVE-NS" and not r["met"], "ABOVE-NS"); c.ok(0.05 < r["p_one_sided"] < 0.10 and abs(r["delta"] - 0.035) < 0.001, "delta ~0.035, p ~0.07")

    @case("accuracy BELOW (delta -0.045) and MATCHES (delta +0.008)", "BELOW / MATCHES")
    def _(c):
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(0, 60))}); r = o["cells"]["b1"]["ACCURACY"]; c.obs = {"below": (r["LABEL"], r["delta"])}
        c.ok(r["LABEL"] == "BELOW" and not r["met"], "BELOW")
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(20, 10))}); r = o["cells"]["b1"]["ACCURACY"]; c.obs["matches"] = (r["LABEL"], r["delta"])
        c.ok(r["LABEL"] == "MATCHES", "MATCHES")

    @case("not admitted (A1; B1; A2-CTRL)", "UNREAD, never a pass; downstream not GO")
    def _(c):
        o = run(not_admitted=("a1",)); c.obs = {k: o["cells"][k]["status"] for k in o["cells"]}
        c.ok(all(o["cells"][k]["status"] == "UNREAD" for k in ("a1", "a2", "ctrl", "b1")), "A1 not admitted -> UNREAD chain")
        o = run(not_admitted=("b1",)); c.ok(o["cells"]["b1"]["status"] == "UNREAD" and o["cells"]["a1"]["VB_C1_L"], "B1 not admitted")
        o = run(not_admitted=("ctrl",)); c.ok(o["cells"]["a2"]["status"] == "UNREAD" and o["cells"]["ctrl"]["status"] == "UNREAD", "A2 launches only with A2-CTRL")

    @case("CF counter-model: serial t* = j", "CF-L")
    def _(c):
        o = run(); r = rdg(o["cells"]["a1"], "CF_L"); c.obs = {"status": r["status"]}; c.ok(r["met"], "CF-L")

    @case("CF counter-model: serial with half the hops looked ahead (t* in {j-1, j}) and 10 % readout flips", "CF-L")
    def _(c):
        o = run(A1(cf=dict(tf=t_serial_la(0.5), eps=0.10, n=100, seed=11))); r = rdg(o["cells"]["a1"], "CF_L")
        c.obs = {k: r.get(k) for k in ("status", "share_window", "spearman_within_D", "n_clean")}; c.ok(r["met"], "CF-L")

    @case("CF counter-model: precompute-and-carry (answer y at every t >= 1)", "not CF-L; CF-CARRY")
    def _(c):
        o = run(A1(cf=dict(tf=lambda D, j, r: 1))); a = o["cells"]["a1"]; r = rdg(a, "CF_L"); c.obs = {"share": r.get("share_window"), "rho": r.get("spearman_within_D")}
        c.ok(r["status"] == "NOT MET" and lab(a)["CF_CARRY"] is True, "not CF-L, CF-CARRY")

    @case("CF counter-model: recompute from input (y' at every t < D)", "not CF-L; CF-RECOMPUTE")
    def _(c):
        o = run(A1(cf=dict(tf=lambda D, j, r: D))); a = o["cells"]["a1"]; r = rdg(a, "CF_L"); c.obs = {"share": r.get("share_window")}
        c.ok(r["status"] == "NOT MET" and lab(a)["CF_RECOMPUTE"] is True, "not CF-L, CF-RECOMPUTE")

    @case("CF counter-model: D-keyed clock (t* = D - 2)", "not CF-L (within-D rho 0)")
    def _(c):
        o = run(A1(cf=dict(tf=lambda D, j, r: D - 2))); r = rdg(o["cells"]["a1"], "CF_L"); c.obs = {"share": r.get("share_window"), "rho": r.get("spearman_within_D")}
        c.ok(r["status"] == "NOT MET" and r["spearman_within_D"] == 0.0, "rho 0")

    @case("CF counter-model: pointer doubling (t* = ceil(j / 2))", "not CF-L; CF-FAST")
    def _(c):
        o = run(A1(cf=dict(tf=lambda D, j, r: math.ceil(j / 2)))); a = o["cells"]["a1"]; r = rdg(a, "CF_L"); c.obs = {"share": r.get("share_window")}
        c.ok(r["status"] == "NOT MET" and lab(a)["CF_FAST"] is True, "not CF-L, CF-FAST")

    @case("CF: 99 exposed clean pairs in the decisive cells (others < 30, excluded and listed)", "CF-L UNREADABLE")
    def _(c):
        keep = {(5, 3), (6, 4), (7, 5)}
        nc = lambda D, j: (33 if (D, j) in keep else 10) if 3 <= j <= D - 1 else 40  # noqa: E731
        o = run(A1(cf=dict(tf=t_serial, ncell=nc))); r = rdg(o["cells"]["a1"], "CF_L"); c.obs = {"status": r["status"], "n_excluded": len(r.get("excluded_listed") or {})}
        c.ok(r["status"].startswith("UNREADABLE") and "99 < 100" in r["status"] and len(r["excluded_listed"]) == 12, "UNREADABLE, 12 cells excluded and listed")

    @case("CF: last-hop crossover at t = D with CF-L met on j < D; and the look-ahead last hop", "CF-L and LAST-HOP-LATE; t* = D - 1 at j = D -> no LAST-HOP-LATE")
    def _(c):
        o = run(); a = o["cells"]["a1"]; c.ok(rdg(a, "CF_L")["met"] and lab(a)["LAST_HOP_LATE"] is True, "LAST-HOP-LATE")
        o = run(A1(cf=dict(tf=lambda D, j, r: D - 1 if j == D else j))); a = o["cells"]["a1"]; c.obs = {"last_hop": a["DESCRIPTIVE"]["cf"]["labels"]["last_hop"]}
        c.ok(rdg(a, "CF_L")["met"] and lab(a)["LAST_HOP_LATE"] is False, "not LAST-HOP-LATE")

    @case("CF-GSM late fusion (t* = 14 at every j)", "not CF-GSM; LATE-FUSION-GSM")
    def _(c):
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), cf=cf_gsm_part(lambda j, mb: 14))}); b = o["cells"]["b1"]
        c.obs = {"cf": b["READINGS"]["CF_GSM"]["status"], "late": b["LABELS"]["late_fusion_median_tstar"]}
        c.ok(b["READINGS"]["CF_GSM"]["status"] == "NOT MET" and b["LABELS"]["LATE_FUSION_GSM"] is True and not b["VB_C1_GSM"], "LATE-FUSION-GSM")

    @case("CF-GSM minimum (note 218-5): 35 base steps readable, 25 UNREADABLE", "CF-GSM MET at 35; UNREADABLE at 25 (< 30)")
    def _(c):
        B = [b for b in range(60) if b % 6 != 0]                            # base items with m_bin >= 5 (steps j 3 and 4 both decisive)
        s35 = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), cf=cf_gsm_custom([(b, j, 3, j) for b in B[:17] for j in (3, 4)] + [(B[17], 3, 3, 3)]))})
        s35 = s35["cells"]["b1"]["READINGS"]["CF_GSM"]
        s25 = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), cf=cf_gsm_custom([(b, j, 5, j) for b in B[:12] for j in (3, 4)] + [(B[12], 3, 5, 3)]))})
        s25 = s25["cells"]["b1"]["READINGS"]["CF_GSM"]
        c.obs = {"35": (s35["status"], s35.get("n_steps"), s35.get("n_clean")), "25": (s25["status"], s25.get("n_steps"), s25.get("n_clean"))}
        c.ok(s35["status"] == "MET" and s35["n_steps"] == 35, "35 distinct base steps -> readable and MET")
        c.ok(s25["status"].startswith("UNREADABLE") and "25 < 30" in s25["status"] and s25["n_clean"] >= 100, "25 < 30 steps (125 clean pairs) -> UNREADABLE")


    @case("GSM clustering: 21 variants of 5 base items (105 clean pairs) cannot pass", "SERIAL-STEP-GSM UNREADABLE (5 < 50 distinct base items); CF-GSM likewise")
    def _(c):
        arms = {a: gsm_arm(bases=range(5), variants=21) for a in ("ii_prime", "i", "ii", "iv")}
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), arms=arms, cf=cf_gsm_part(lambda j, mb: j, bases=range(5), variants=21))})
        b = o["cells"]["b1"]; s = b["READINGS"]["SERIAL_STEP_GSM"]["ii_prime"]; cfs = b["READINGS"]["CF_GSM"]
        c.obs = {"ii_prime": s["status"], "n_clean": s.get("n_clean_band"), "cf": cfs["status"]}
        c.ok(s.get("n_clean_band") == 105 and "5 < 50 distinct base items" in s["status"] and s.get("follow") == 1.0, "vswap >= 50 rule")
        c.ok(cfs["status"].startswith("UNREADABLE") and "< 30 distinct exposed base steps" in cfs["status"], "CF-GSM >= 30 steps rule (note 218-5)"); c.ok(not b["VB_C1_GSM"], "not VB-C1-GSM")

    @case("GSM clustering: a clustered mean differs from the pooled mean (designed)", "clustered follow 0.833 / own 0.167 (met); pair-level 0.20 / 0.80")
    def _(c):
        pf = lambda b, v: ("own", True) if b < 10 else ("follow", True)  # noqa: E731
        arm = gsm_arm(pair_fn=pf, bases=range(60), variants=lambda b: 20 if b < 10 else 1)
        arms = {"ii_prime": arm, "i": gsm_arm(), "ii": gsm_arm(), "iv": gsm_arm()}
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), arms=arms)}); s = o["cells"]["b1"]["READINGS"]["SERIAL_STEP_GSM"]["ii_prime"]
        c.obs = {"clustered": (s["follow"], s["own"]), "pair_level": s["pair_level_band"], "ci": s["follow_ci95"], "status": s["status"]}
        c.ok(abs(s["follow"] - 50 / 60) < 1e-12 and abs(s["own"] - 10 / 60) < 1e-12, "clustered"); c.ok(abs(s["pair_level_band"]["follow"] - 0.2) < 1e-12, "pooled 0.2")
        c.ok(s["met"] and s["follow_ci95"][0] < 50 / 60 < s["follow_ci95"][1], "bars read on the clustered means; bootstrap CI brackets it")

    @case("B1 not launched (A1 not VB-C1-L -> B not GO)", "B1 UNREAD (never pass), also when B1 files exist")
    def _(c):
        drop = [k for k in BASE if k.startswith("lvb_gsm_s1/")]
        o = run(A1(stall=0.5), drop=drop); b = o["cells"]["b1"]; c.obs = {"label": b["label"]}
        c.ok(b["status"] == "UNREAD" and "B not GO" in b["label"] and o["FLOW"]["B_GO"]["go"] is False, "UNREAD")
        o = run(A1(stall=0.5)); b = o["cells"]["b1"]; c.ok(b["status"] == "UNREAD" and b.get("launched_despite_no_GO") is True, "files present: still UNREAD, flagged")
        c.ok(o["PRIMARY"]["VB_C1_GSM (B1)"] == "UNREAD", "primary row UNREAD")

    @case("B1 interim no-GO (FD-MLP(1, k>=3) 0.45 >= 0.8 x 0.5): scored as B1-24k", "B1-24k: log to 24k not dead; accuracy descriptive vs CODI4; 24k log with GO -> dead")
    def _(c):
        o = run({"lvb_gsm_s1/interim_go.json": interim(f1=0.45), "lvb_gsm_s1/log.jsonl": log_text(24000, va=VA_OK)}); b = o["cells"]["b1"]
        c.obs = {"label": b["label"], "acc": b["ACCURACY"]["status"]}
        c.ok(b["status"] == "SCORED" and b["B1_24K"] and "B1-24k" in b["ACCURACY"]["status"] and b["VB_C1_GSM"], "B1-24k")
        o = run({"lvb_gsm_s1/log.jsonl": log_text(24000, va=VA_OK)}); c.ok(o["cells"]["b1"]["status"] == "FAIL" and o["cells"]["b1"]["censors"]["dead"], "GO + 24k log = dead")

    @case("per-item assert: acc_k[10][8] != mean(items)", "acc part VOID -> DEEP-L UNREADABLE; not VB-C1-L")
    def _(c):
        o = run(A1(acc_corrupt=True)); a = o["cells"]["a1"]; c.obs = {"deep": rdg(a, "DEEP_L")["status"][:160]}
        c.ok("VOID" in rdg(a, "DEEP_L")["status"] and not a["VB_C1_L"], "VOID")

    @case("probe extension rule (last-2k gain 0.03 at 10k, not extended)", "PROBE VOID -> PROBE-GAIN-L UNREADABLE")
    def _(c):
        cv = [[st, 0.85 if st <= 8000 else 0.88] for st in range(500, 10001, 500)]
        o = run(A1(curve=cv)); a = o["cells"]["a1"]; c.obs = {"pg": rdg(a, "PROBE_GAIN_L")["status"][:200]}
        c.ok("extension rule" in rdg(a, "PROBE_GAIN_L")["status"] and not a["VB_C1_L"], "VOID")

    @case("card / dtype / eval_bs / items_sha mismatches", "pending (mismatch); --final -> VOID (rule b)")
    def _(c):
        for tag, kw in (("card", dict(card="NVIDIA GeForce RTX 3070")), ("card_by_part", dict(card_by_part={"cf": "NVIDIA GeForce RTX 3070"})),
                        ("dtype", dict(dtype="bf16")), ("eval_bs", dict(evbs=dict(EVBS_A, probe=64))), ("items_sha", dict(sha="other"))):
            o = run(A1(**kw)); a = o["cells"]["a1"]
            c.ok(a["status"] == "PENDING" and "mismatch" in a["label"], "%s -> pending (mismatch)" % tag)
            o = run(A1(**kw), final=("a1",)); a = o["cells"]["a1"]; c.obs[tag] = a["label"][:140]
            c.ok(a["status"] == "FAIL" and a["label"].startswith("VOID (mismatch"), "%s --final -> VOID" % tag)
        o = run({"lvb_b02/codi20_rescore.json": b02(card="NVIDIA GeForce RTX 3090")}); c.ok(o["cells"]["b1"]["status"] == "PENDING" and "mismatch" in o["cells"]["b1"]["label"], "B0-2 card")
        o = run({"lvb_gate1_s1/eval_gate1.json": dict(gate1(), meta={"card": CARD, "dtype": "fp32"})})
        c.ok(o["cells"]["a2"]["status"] == "UNREAD", "GATE-1-L not in bf16 -> GATE-1-L fails -> no A2")

    @case("A1 waits for DIRECT-SYM; DIRECT-SYM censored -> C1 band UNREADABLE", "A1 pending while DIRECT pending; DIRECT NO-ENTRY -> not VB-C1-L")
    def _(c):
        o = run(drop=["lvb_direct_s1/log.jsonl"]); c.obs = {"a1": o["cells"]["a1"]["label"][:200]}
        c.ok(o["cells"]["a1"]["status"] == "PENDING" and "DIRECT-SYM cell pending" in o["cells"]["a1"]["label"], "pending")
        o = run({"lvb_direct_s1/log.jsonl": log_text(24000, CE_NEVER)}); a = o["cells"]["a1"]; c.obs["probe_gain"] = rdg(a, "PROBE_GAIN_L")["status"]
        c.ok("DIRECT-SYM censored" in rdg(a, "PROBE_GAIN_L")["status"] and not a["VB_C1_L"], "UNREADABLE")

    @case("ARGS-DIFF post-freeze key (note 218-14): deterministic absent / False OK, True -> ARGS_BAD", "OK / OK / FAILURE (ARGS-DIFF)")
    def _(c):
        o = run({"lvb_sym_s1/args.json": dict(A1_ARGS, deterministic=False)}); c.ok(o["cells"]["a1"]["status"] != "FAIL", "deterministic False OK")
        o = run({"lvb_sym_s1/args.json": dict(A1_ARGS)}); c.ok(o["cells"]["a1"]["status"] != "FAIL", "deterministic absent OK")
        o = run({"lvb_sym_s1/args.json": dict(A1_ARGS, deterministic=True)}); c.obs = {"true": o["cells"]["a1"]["label"][:160]}
        c.ok(o["cells"]["a1"]["status"] == "FAIL" and "ARGS-DIFF" in o["cells"]["a1"]["label"], "deterministic True -> ARGS_BAD")

    @case("ARGS-DIFF (A2-CTRL lr differs from A2's; A1 unexpected key; v1-unlisted keys -> pending)", "FAILURE (ARGS-DIFF); A2 not SD-C2-L; unlisted -> pending")
    def _(c):
        o = run({"lvb_ctrl_s1/args.json": dict(CTRL_ARGS, lr=1e-4)}); c.obs = {"ctrl": o["cells"]["ctrl"]["label"][:200]}
        c.ok(o["cells"]["ctrl"]["status"] == "FAIL" and "ARGS-DIFF" in o["cells"]["ctrl"]["label"] and o["cells"]["a2"]["SD_C2_L"] is False, "CTRL ARGS-DIFF")
        o = run({"lvb_sym_s1/args.json": dict(A1_ARGS, foo=1)}); c.ok(o["cells"]["a1"]["status"] == "FAIL" and not o["cells"]["a1"]["VB_C1_L"], "A1 unexpected key")
        o = run({"lvb_ctrl_s1/args.json": dict(CTRL_ARGS, p_self=0.5)}); c.ok(o["cells"]["ctrl"]["status"] == "FAIL", "CTRL p_self must be 0")
        keep = S.FROZEN_ARGS["a1"]; S.FROZEN_ARGS["a1"] = None
        try:
            o = run(); c.ok(o["cells"]["a1"]["status"] == "PENDING" and "no frozen args" in o["cells"]["a1"]["label"], "no frozen entry -> pending")
            o = run(final=("a1",)); c.ok(o["cells"]["a1"]["status"] == "FAIL", "no frozen entry at --final -> FAILURE")
        finally:
            S.FROZEN_ARGS["a1"] = keep

    REG = [("direct", "lvb_direct_s1/log.jsonl"), ("direct", "lvb_direct_s1/args.json"), ("direct", "lvb_direct_s1/eval_direct.json"),
           ("direct", "lvb_gates/gate_l0.json"), ("direct", "lvb_gates/gate_evmem.json"), ("direct", "lvb_frozen/direct_args.json"),
           ("a1", "lvb_sym_s1/log.jsonl"), ("a1", "lvb_sym_s1/args.json"), ("a1", "lvb_sym_s1/eval_lvb.json"), ("a1", "lvb_direct_s1/eval_direct.json"),
           ("a1", "lvb_gates/gate_l0.json"), ("a1", "lvb_gates/gate_evmem.json"), ("a1", "lvb_frozen/a1_args.json"),
           ("a2", "lvb_self_s1/log.jsonl"), ("a2", "lvb_self_s1/args.json"), ("a2", "lvb_self_s1/eval_lvb.json"), ("a2", "lvb_pool_s1/pool_meta.json"),
           ("a2", "lvb_gate1_s1/eval_gate1.json"),
           ("ctrl", "lvb_ctrl_s1/log.jsonl"), ("ctrl", "lvb_ctrl_s1/args.json"), ("ctrl", "lvb_ctrl_s1/eval_lvb.json"),
           ("b1", "lvb_gsm_s1/log.jsonl"), ("b1", "lvb_gsm_s1/args.json"), ("b1", "lvb_gsm_s1/eval_lvb_gsm.json"), ("b1", "lvb_gsm_s1/interim_go.json"),
           ("b1", "lvb_b02/codi20_rescore.json"), ("b1", "gsm_codi_20ep_s1/eval_items.json"), ("b1", "lvb_gates/gate_evmem_gsm.json"), ("b1", "lvb_frozen/b1_args.json")]

    @case("each registered file missing -> pending; with --final <cell> -> failure (rule b)", "%d (cell, file) pairs" % len(REG))
    def _(c):
        for cell, rel in REG:
            o = run(drop=[rel]); x = o["cells"][cell]
            c.ok(x["status"] == "PENDING" and x["label"].startswith("pending (") and not x.get("VB_C1_L") and not x.get("SD_C2_L") and not x.get("VB_C1_GSM"),
                 "%s missing %s -> pending (got %s)" % (cell, rel, x["status"]))
            o = run(drop=[rel], final=(cell,)); x = o["cells"][cell]
            c.ok(x["status"] == "FAIL" and ("FAILURE" in x["label"] or "VOID" in x["label"]), "%s missing %s --final -> failure (got %s: %s)" % (cell, rel, x["status"], x["label"][:120]))
        o = run({"lvb_sym_s1/log.jsonl": log_text(24000)}); c.ok(o["cells"]["a1"]["status"] == "PENDING" and "plain_cand_ce" in o["cells"]["a1"]["label"], "log without plain_cand_ce")
        o = run({"lvb_gsm_s1/log.jsonl": log_text(120000)}); c.ok(o["cells"]["b1"]["status"] == "PENDING" and "valid_acc" in o["cells"]["b1"]["label"], "log without valid_acc")
        c.obs = {"pairs_checked": len(REG)}

    @case("each registered eval part missing -> pending; --final: decisive parts fail, descriptive parts do not", "A1 / A2 / B1 parts")
    def _(c):
        rel = "lvb_sym_s1/eval_lvb.json"
        for p in S.REQ_PARTS["a1"]:
            m = strip_part(rel, S.PARTS_A[p]); o = run(m); x = o["cells"]["a1"]
            c.ok(x["status"] == "PENDING" and ("part %s" % p) in x["label"], "A1 part %s -> pending" % p)
            o = run(m, final=("a1",)); x = o["cells"]["a1"]
            if p in S.DESCRIPTIVE_A:
                c.ok(x["status"] == "SCORED" and (p != "hist24" or x["MEM"]["MEM_TRADED_L"] is True), "A1 desc part %s at --final -> read on (%s)" % (p, x["label"][:80]))
            else:
                c.ok(x["status"] == "FAIL" and "FAILURE" in x["label"], "A1 part %s at --final -> FAILURE" % p)
        m = strip_part("lvb_self_s1/eval_lvb.json", S.PARTS_A["labeller"]); o = run(m); c.ok(o["cells"]["a2"]["status"] == "PENDING", "A2 labeller part -> pending")
        o = run(m, final=("a2",)); c.ok(o["cells"]["a2"]["status"] == "FAIL", "A2 labeller part --final -> FAILURE")
        for p in S.DECISIVE_B + S.DESCRIPTIVE_B:
            m = strip_part("lvb_gsm_s1/eval_lvb_gsm.json", S.PARTS_B[p]); o = run(m); x = o["cells"]["b1"]
            c.ok(x["status"] == "PENDING", "B1 part %s -> pending" % p)
            o = run(m, final=("b1",)); x = o["cells"]["b1"]
            c.ok((x["status"] == "SCORED") if p in S.DESCRIPTIVE_B else (x["status"] == "FAIL"), "B1 part %s --final -> %s" % (p, x["status"]))
        m = {"lvb_sym_s1/eval_lvb.json": dict(BASE["lvb_sym_s1/eval_lvb.json"], cf=dict(BASE["lvb_sym_s1/eval_lvb.json"]["cf"], pairs=None))}
        o = run(m); c.ok(o["cells"]["a1"]["status"] == "PENDING" and "cf" in o["cells"]["a1"]["label"], "cf.pairs missing -> pending (field level)")

    @case("item 1: CF-GSM counts and clusters exposed base STEPS (base, j)", "20 base items x 2 steps = 40 steps MET; step share 0.5 vs pair / item share 0.75 -> NOT MET")
    def _(c):
        B = [b for b in range(60) if b % 6 != 0]
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), cf=cf_gsm_custom([(b, j, 3, j) for b in B[:20] for j in (3, 4)]))})
        r = o["cells"]["b1"]["READINGS"]["CF_GSM"]; c.obs = {"steps40": {k: r.get(k) for k in ("status", "n_steps", "n_base_items", "rho_ci95")}}
        c.ok(r["status"] == "MET" and r["n_steps"] == 40 and r["n_base_items"] == 20, "40 steps from 20 items: readable, MET")
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), cf=cf_gsm_custom([x for b in B[:40] for x in ((b, 3, 3, 3), (b, 4, 1, 6))]))})
        r = o["cells"]["b1"]["READINGS"]["CF_GSM"]; c.obs["step_vs_item"] = {k: r.get(k) for k in ("status", "share_window_clustered", "spearman_step_clusters", "pair_level")}
        c.ok(r["status"] == "NOT MET" and abs(r["share_window_clustered"] - 0.5) < 1e-12 and abs(r["pair_level"]["share_window"] - 0.75) < 1e-12
             and r["spearman_step_clusters"] > 0.99, "step-clustered share 0.5 decides (pair-level 0.75)")

    @case("item 2: frozen queued args (note 218-4)", "exact match except META; args_ref content rule; changed frozen file -> mismatch; args_frozen.json accepted (unanchored)")
    def _(c):
        o = run({"lvb_sym_s1/args.json": dict(A1_ARGS, lr=2e-4)}); a = o["cells"]["a1"]; c.obs = {"lr": a["label"][:120]}
        c.ok(a["status"] == "FAIL" and "ARGS-DIFF" in a["label"], "non-META key differs -> ARGS-DIFF")
        o = run({"lvb_sym_s1/args.json": dict(A1_ARGS, train_gpu="NVIDIA GeForce RTX 3090 (host 2)", out="boxes/runs_fsq/x", torch_version="2.5.2")})
        c.ok(o["cells"]["a1"]["VB_C1_L"] is True, "environment / META keys differ -> ok")
        for kw in (dict(arch="direct"), dict(arch_class="GPT2KV"), dict(arch="direct", arch_class="GPT2KV")):
            o = run({"lvb_sym_s1/args.json": dict(A1_ARGS, **kw)}); a = o["cells"]["a1"]
            c.ok(a["status"] == "FAIL" and "required_bad" in str(a["censors"]["ARGS_DIFF"]) and a["censors"]["ARGS_DIFF"]["required_bad"], "a1 %s -> ARGS-DIFF (arch is not META)" % kw)
        o = run({"lvb_ctrl_s1/args.json": dict(CTRL_ARGS, arch="direct")}); c.ok(o["cells"]["ctrl"]["status"] == "FAIL", "A2-CTRL with a swapped arch -> ARGS-DIFF")
        o = run({"lvb_frozen/a1_args.json": dict(A1_ARGS, lr=5e-5)}); a = o["cells"]["a1"]; c.obs["sha"] = a["label"][:160]
        c.ok(a["status"] == "PENDING" and "sha256" in a["label"], "frozen file changed -> pending (mismatch)")
        o = run({"lvb_frozen/a1_args.json": dict(A1_ARGS, lr=5e-5)}, final=("a1",)); c.ok(o["cells"]["a1"]["label"].startswith("VOID (mismatch"), "--final -> VOID")
        keep = S.FROZEN_ARGS["a1"]; S.FROZEN_ARGS["a1"] = None
        try:
            o = run({"lvb_sym_s1/args_frozen.json": dict(A1_ARGS)}); a = o["cells"]["a1"]
            c.ok(a["VB_C1_L"] is True and a["censors"]["ARGS_DIFF"]["frozen"]["anchored"] is False, "<cell>/args_frozen.json accepted, flagged unanchored")
        finally:
            S.FROZEN_ARGS["a1"] = keep
        ref = os.path.join(TREE, "lvb_frozen/a1_ref_copy.json")
        o = run({"lvb_frozen/a1_ref_copy.json": dict(A1_ARGS), "lvb_sym_s1/args.json": dict(A1_ARGS, args_ref=ref)})
        c.ok(o["cells"]["a1"]["VB_C1_L"] is True, "args_ref pointer, equal content -> ok")
        o = run({"lvb_frozen/a1_ref_copy.json": dict(A1_ARGS, lr=9e-5), "lvb_sym_s1/args.json": dict(A1_ARGS, args_ref=ref)})
        c.ok(o["cells"]["a1"]["status"] == "FAIL", "args_ref pointer, different content -> ARGS-DIFF")

    @case("item 3: CF full store with per-candidate 'exposed' flags (20 non-exposed precompute-like candidates per cell)",
          "non-exposed never counted: CF-L MET, counts recomputed; mixed store pending; flag != components VOID")
    def _(c):
        base = BASE["lvb_sym_s1/eval_lvb.json"]
        o = run({"lvb_sym_s1/eval_lvb.json": dict(base, cf=cf_flags(base["cf"]))}); a = o["cells"]["a1"]; r = rdg(a, "CF_L")
        c.obs = {"status": r["status"], "flags": r.get("flags")}
        c.ok(r["met"] and not r.get("flags") and a["VB_C1_L"], "MET with the full flag store (no selected-only flag)")
        C2 = cf_flags(base["cf"]); C2["pairs"] = [dict(C2["pairs"][0])] + [{k: v for k, v in q.items() if k != "exposed"} for q in C2["pairs"][1:]]
        o = run({"lvb_sym_s1/eval_lvb.json": dict(base, cf=C2)})
        c.ok(o["cells"]["a1"]["status"] == "PENDING" and "exposure stored for some" in o["cells"]["a1"]["label"], "mixed -> pending")
        C3 = dict(base["cf"]); C3["pairs"] = [dict(C3["pairs"][0], exposed=0)] + C3["pairs"][1:]
        o = run({"lvb_sym_s1/eval_lvb.json": dict(base, cf=C3)}); r = rdg(o["cells"]["a1"], "CF_L")
        c.ok("exposed' != pos_ok" in r["status"], "flag disagrees with components -> VOID")

    @case("item 4: CONTAMINATED needs evidence (A1, DIRECT-SYM, A2-CTRL; A2 self_reuse)", "absent -> pending, --final FAILURE; contamination block alone ok; a hit -> FAILURE")
    def _(c):
        for cell, rel, steps, ce in (("a1", "lvb_sym_s1/log.jsonl", 24000, CE_OK), ("direct", "lvb_direct_s1/log.jsonl", 24000, CE_OK),
                                     ("ctrl", "lvb_ctrl_s1/log.jsonl", 12000, CE_FT)):
            o = run({rel: log_text(steps, ce, coll=None, hits=None)}); x = o["cells"][cell]
            c.ok(x["status"] == "PENDING" and "contamination evidence" in x["label"], "%s no evidence -> pending" % cell)
            o = run({rel: log_text(steps, ce, coll=None, hits=None)}, final=(cell,)); c.ok(o["cells"][cell]["status"] == "FAIL", "%s --final -> FAILURE" % cell)
        m = A1(contamination={"test_gate_sig_hits": 0}); m["lvb_sym_s1/log.jsonl"] = log_text(24000, CE_OK, coll=None, hits=None)
        o = run(m); c.ok(o["cells"]["a1"]["VB_C1_L"] is True, "contamination block alone ok")
        o = run(A1(contamination={"test_gate_sig_hits": 2})); c.ok(o["cells"]["a1"]["status"] == "FAIL" and "CONTAMINATED" in o["cells"]["a1"]["label"], "hit -> FAILURE")
        o = run({"lvb_self_s1/log.jsonl": log_text(12000, CE_FT)}); c.obs = {"a2": o["cells"]["a2"]["label"][:160]}
        c.ok(o["cells"]["a2"]["status"] == "PENDING" and "self_reuse" in o["cells"]["a2"]["label"], "A2 without self_reuse -> pending")

    @case("item 5: items_sha anchored to G-L0's test_items_sha only", "absent -> every Stage A cell pending; per-part anchor compared per part; different forms pending")
    def _(c):
        g = dict(BASE["lvb_gates/gate_l0.json"]); g.pop("test_items_sha")
        o = run({"lvb_gates/gate_l0.json": g}); c.obs = {k: o["cells"][k]["status"] for k in ("direct", "a1", "a2", "ctrl")}
        c.ok(all(o["cells"][k]["status"] == "PENDING" for k in ("direct", "a1")) and "test_items_sha" in o["cells"]["a1"]["label"], "no anchor -> pending")
        o = run({"lvb_gates/gate_l0.json": dict(g, test_items_sha={p_: SHA_A for p_ in EVBS_A})}); c.ok(o["cells"]["a1"]["VB_C1_L"] is True, "per-part anchor ok")
        o = run({"lvb_gates/gate_l0.json": dict(g, test_items_sha={p_: SHA_A for p_ in EVBS_A if p_ != "direct"})})
        c.ok(o["cells"]["direct"]["status"] == "PENDING" and "lacks part direct" in o["cells"]["direct"]["label"], "anchor lacking a part -> pending")
        o = run({"lvb_gates/gate_l0.json": dict(g, test_items_sha=dict(TEST_SHA_D))})
        c.ok(o["cells"]["a1"]["status"] == "PENDING" and "different forms" in o["cells"]["a1"]["label"], "per-D anchor vs single sha -> pending")

    @case("item 6: every registered FD-GSM cell (t 0..12, k >= t) is required", "a missing (1, 7) or (0, 5) cell -> pending; --final -> FAILURE")
    def _(c):
        for t, k in ((1, 7), (0, 5)):
            ev = ev_gsm(b1_ok(120, 0)); fu = json.loads(json.dumps(ev["future"])); del fu["lin"][str(t)][str(k)]
            o = run({"lvb_gsm_s1/eval_lvb_gsm.json": dict(ev, future=fu)}); x = o["cells"]["b1"]
            c.ok(x["status"] == "PENDING" and "future cells" in x["label"], "(%d, %d) -> pending" % (t, k))
            o = run({"lvb_gsm_s1/eval_lvb_gsm.json": dict(ev, future=fu)}, final=("b1",)); c.ok(o["cells"]["b1"]["status"] == "FAIL", "(%d, %d) --final -> FAILURE" % (t, k))

    @case("Addendum 1 CF mixtures: serial 0.9 + precompute (flips 0.05); 0.8 + recompute (0.05); 0.9 + recompute (0.10); 0.8 + precompute (0.05)",
          "MET; NOT MET; NOT MET; NOT MET (the Addendum's Monte Carlo)")
    def _(c):
        res = {}
        for tag, sh, oth, eps, want in (("0.9 pre 0.05", 0.9, "pre", 0.05, True), ("0.8 rec 0.05", 0.8, "rec", 0.05, False),
                                        ("0.9 rec 0.10", 0.9, "rec", 0.10, False), ("0.8 pre 0.05", 0.8, "pre", 0.05, False)):
            o = run(A1(cf=dict(tf=t_mix(sh, oth), eps=eps, n=100, seed=31))); r = rdg(o["cells"]["a1"], "CF_L")
            res[tag] = (r["status"], round(r.get("share_window") or 0, 3), round(r.get("spearman_within_D") or 0, 3))
            c.ok(r["met"] is want, "%s -> %s" % (tag, "MET" if want else "NOT MET"))
        c.obs = res

    @case("SERIAL-ONSET-L failing: flat ON, slope 0.5 (span 3), never reached at D 8", "NOT MET x 3 (readable)")
    def _(c):
        res = {}
        for tag, ON in (("flat", {3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}), ("slope", {3: 3, 4: 5, 5: 5, 6: 5, 7: 4, 8: 8})):
            o = run(A1(probe=lambda t, D, ON=ON: 0.95 if t >= ON[D] else 0.5)); r = rdg(o["cells"]["a1"], "SERIAL_ONSET_L")
            res[tag] = (r["status"], r.get("slope"), r.get("span_8_4")); c.ok(r["status"] == "NOT MET" and not o["cells"]["a1"]["VB_C1_L"], "%s -> NOT MET" % tag)
        c.ok(abs(res["slope"][1] - 0.5) < 1e-12 and res["slope"][2] == 3, "slope 0.5 with span 3: the slope clause alone fails")
        o = run(A1(probe=probe_serial(over={(12, 8): 0.85}))); r = rdg(o["cells"]["a1"], "SERIAL_ONSET_L"); res["never"] = (r["status"], r.get("never_reached"))
        c.ok(r["status"] == "NOT MET" and r["never_reached"] == [8], "never reached at D 8 -> NOT MET"); c.obs = res

    @case("SERIAL-STEP-L per-D bar: (ii') pooled 0.90 with D 5 at 0.50", "NOT MET")
    def _(c):
        vs = dict(spec_A("a1")["vswap"], ii_prime=dict(follow=1.0, perD={5: {"follow": 0.5}})); o = run(A1(vswap=vs))
        s_ = rdg(o["cells"]["a1"], "SERIAL_STEP_L")["ii_prime"]; c.obs = {"follow": s_.get("follow"), "status": s_["status"]}
        c.ok(s_["status"] == "NOT MET" and s_["follow"] >= 0.8, "per-D bar fails")

    @case("PROBE(12) invalid (0.85 at every D)", "PROBE-GAIN-L and SERIAL-ONSET-L UNREADABLE (final state)")
    def _(c):
        o = run(A1(probe=probe_serial(over={(12, D): 0.85 for D in range(3, 9)}))); a = o["cells"]["a1"]
        c.obs = {k: rdg(a, k)["status"][:90] for k in ("PROBE_GAIN_L", "SERIAL_ONSET_L")}
        c.ok("final state" in rdg(a, "PROBE_GAIN_L")["status"] and "final state" in rdg(a, "SERIAL_ONSET_L")["status"], "both UNREADABLE")

    @case("A2 readings failing singly: ABSORB (D 11 0.94), LENGTH (D 5 0.985), K'24 (D 6 0.96), DEPTH-BEYOND (D 13-16 0.78)", "each NOT MET alone; SD-SELF / SD-C2 follow")
    def _(c):
        res = {}
        for tag, accf, key in (("ABSORB", acc_profile(0.995, 0.97, 0.85, over={11: 0.94}), "ABSORB_L"),
                               ("LENGTH", acc_profile(0.995, 0.97, 0.85, over={5: 0.985}), "LENGTH_L"),
                               ("K24", acc_profile(0.995, 0.97, 0.85, over={(6, 24): 0.96}), "K24_STABLE"),
                               ("DBS", acc_profile(0.995, 0.97, 0.78), "DEPTH_BEYOND_SELF_L")):
            o = run(A2(acc=accf)); a = o["cells"]["a2"]; m = a["A2_READINGS"]; res[tag] = {k: m[k]["status"] for k in m}
            others = [k for k in m if k != key and m[k]["status"] != "MET"]
            c.ok(m[key]["status"] == "NOT MET" and not others and a["SD_C2_L"] is False, "%s fails alone" % tag)
            c.ok(a["SD_SELF_L"] is (tag == "DBS"), "%s: SD-SELF %s" % (tag, tag == "DBS"))
        c.obs = res

    @case("BEATS-CTRL-L failing numerically (A2 0.85 vs readable A2-CTRL 0.80)", "NOT MET (readable); not SD-C2-L")
    def _(c):
        o = run({"lvb_ctrl_s1/eval_lvb.json": ev_lvb(spec_A("ctrl", acc=acc_profile(0.97, 0.62, 0.80)))}); a = o["cells"]["a2"]; b_ = a["A2_READINGS"]["BEATS_CTRL_L"]
        c.obs = {k: b_.get(k) for k in ("status", "delta")}
        c.ok(b_["status"] == "NOT MET" and abs(b_["delta"] - 0.05) < 1e-9 and a["SD_C2_L"] is False, "NOT MET")

    @case("SERIAL-STEP-GSM failing on own 0.35 > 0.3 (follow 0.65); NO-CLOCK-GSM failing (recovery 0.5)", "NOT MET; NOT MET")
    def _(c):
        arms = {"ii_prime": gsm_arm(pair_fn=lambda b, v: ("follow", True) if b < 39 else ("own", True)), "i": gsm_arm(), "ii": gsm_arm(), "iv": gsm_arm()}
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), arms=arms)}); s_ = o["cells"]["b1"]["READINGS"]["SERIAL_STEP_GSM"]["ii_prime"]
        c.obs = {"own": s_.get("own"), "follow": s_.get("follow"), "status": s_["status"]}
        c.ok(s_["status"] == "NOT MET" and abs(s_["own"] - 21 / 60) < 1e-12 and not o["cells"]["b1"]["VB_C1_GSM"], "own > 0.3")
        o = run({"lvb_gsm_s1/eval_lvb_gsm.json": ev_gsm(b1_ok(120, 0), stall_rec=0.5)}); r = o["cells"]["b1"]["READINGS"]["NO_CLOCK_GSM"]
        c.obs["stall"] = r["status"]; c.ok(r["status"] == "NOT MET" and not o["cells"]["b1"]["VB_C1_GSM"], "NO-CLOCK-GSM NOT MET")

    @case("STALL with 1,500 items (< 2,000)", "NO-CLOCK-L UNREADABLE")
    def _(c):
        o = run(A1(stall_n=300)); r = rdg(o["cells"]["a1"], "NO_CLOCK_L"); c.obs = {"status": r["status"]}
        c.ok(r["status"].startswith("UNREADABLE") and "1500 < 2,000" in r["status"], "UNREADABLE")

    @case("item 9: swap exposure recomputed from the stored components; the eval's flag cross-checks", "components absent -> pending; flag disagreeing -> arm VOID")
    def _(c):
        base = BASE["lvb_sym_s1/eval_lvb.json"]; V = dict(base["vswap"])
        A_ = dict(V["ii_prime"]); A_["pairs"] = [dict({k: v for k, v in q.items() if k != "frontier_differs"}, exposed=1) for q in A_["pairs"]]
        o = run({"lvb_sym_s1/eval_lvb.json": dict(base, vswap=dict(V, ii_prime=A_))}); c.obs = {"no_comp": o["cells"]["a1"]["label"][:140]}
        c.ok(o["cells"]["a1"]["status"] == "PENDING" and "exposure field frontier_differs" in o["cells"]["a1"]["label"], "components absent -> pending")
        A_ = dict(V["i_prime"]); A_["pairs"] = [dict(A_["pairs"][0], exposed=0)] + [dict(q, exposed=1) for q in A_["pairs"][1:]]
        o = run({"lvb_sym_s1/eval_lvb.json": dict(base, vswap=dict(V, i_prime=A_))}); s_ = rdg(o["cells"]["a1"], "SERIAL_STEP_L")["i_prime"]
        c.ok("'exposed' flag != the exposure recomputed" in s_["status"] and not o["cells"]["a1"]["VB_C1_L"], "disagreement -> VOID")

    @case("writer of record (boxes/langvb_eval.py @ ec1aec8d): every Stage A file shaped exactly as written (Theory's field list)",
          "identical readings: VB-C1-L, SD-C2-L; CF counts recomputed from every candidate; PaR-C still fails; a module count / flag disagreement VOIDs")
    def _(c):
        M = mod_layout(BASE); materialize(M); o = S.score(); a = o["cells"]["a1"]
        c.obs = {"a1": a["label"][:160], "a2": o["cells"]["a2"]["label"][:120], "direct": o["cells"]["direct"]["label"][:80], "cf_flags": rdg(a, "CF_L").get("flags"),
                 "t2": lab(a).get("ii_prime_t2_descriptive")}
        c.ok(a["VB_C1_L"] is True and all(v == "MET" for v in comps(a).values()), "VB-C1-L"); c.ok(o["cells"]["a2"]["SD_C2_L"] is True, "SD-C2-L")
        c.ok(o["cells"]["direct"]["C1"]["band"] == [5, 6, 7, 8], "DIRECT top-level fields")
        c.ok(not rdg(a, "CF_L").get("flags") and rdg(a, "CF_L")["n_clean"] == 900, "CF from the full candidate store (non-exposed / unselected excluded)")
        t2 = lab(a).get("ii_prime_t2_descriptive") or {}
        c.ok(t2.get("source") == "vswap.ii_prime_t2.pairs" and t2.get("n_clean", 0) > 0 and not t2.get("flag_file_summary_differs"), "t = 2 summary recomputed from ii_prime_t2.pairs")
        rs_ = rdg(a, "SERIAL_STEP_L"); c.ok(a["DESCRIPTIVE"]["arms_descriptive"]["reset"]["readable"] and abs(lab(a)["RESET_recovery"] - 0.2) < 1e-9,
                                           "RESET with ok_don null: clean = ok_rec, recovery read")
        c.ok(a["DESCRIPTIVE"]["flag_train_items_lt_200k"] == ["probe_n_distinct_seen"], "probe distinct items < 200,000 flagged (reported)")
        vs = dict(spec_A("a1")["vswap"], ii_prime=dict(follow=1.0), i_prime=dict(follow=0.0))
        pc = to_mod_eval(ev_lvb(spec_A("a1", vswap=vs, fd=lambda m, t, k: 0.95 if k >= 3 else 0.7, cf=dict(tf=lambda D, j, r: 1))))
        materialize(dict(M, **{"lvb_sym_s1/eval_lvb.json": pc})); o = S.score(); a = o["cells"]["a1"]
        c.ok(not a["VB_C1_L"] and lab(a)["PRECOMPUTE_L"] is True and lab(a)["CF_CARRY"] is True, "PaR-C in the module's shape")
        e2 = json.loads(json.dumps(M["lvb_sym_s1/eval_lvb.json"])); e2["cf"]["5"]["3"]["n_exposed"] += 1
        materialize(dict(M, **{"lvb_sym_s1/eval_lvb.json": e2})); o = S.score(); c.ok("n_exposed" in rdg(o["cells"]["a1"], "CF_L")["status"], "module CF count off by one -> VOID")
        e3 = json.loads(json.dumps(M["lvb_sym_s1/eval_lvb.json"])); q = next(x for x in e3["cf"]["pairs"] if x["selected"]); q["clean"] = 1 - q["clean"]
        materialize(dict(M, **{"lvb_sym_s1/eval_lvb.json": e3})); o = S.score(); c.ok("'clean' != exposed" in rdg(o["cells"]["a1"], "CF_L")["status"], "clean flag off -> VOID")
        e4 = json.loads(json.dumps(M["lvb_sym_s1/eval_lvb.json"])); e4["contamination"]["test_gate_sig_hits"] = None
        materialize(dict(M, **{"lvb_sym_s1/eval_lvb.json": e4})); o = S.score(); c.ok(o["cells"]["a1"]["status"] == "PENDING", "null test_gate_sig_hits -> pending")
        g = dict(M["lvb_gates/gate_l0.json"]); g.pop("test_items_sha"); materialize(dict(M, **{"lvb_gates/gate_l0.json": g})); o = S.score()
        c.ok(o["cells"]["a1"]["status"] == "PENDING", "G-L0 without test_items_sha -> pending")
        e5 = json.loads(json.dumps(M["lvb_self_s1/eval_lvb.json"])); e5["meta"] = {"card": CARD, "dtype": "fp32", "eval_bs": {"labeller": 8}, "cursor": 383488}
        materialize(dict(M, **{"lvb_self_s1/eval_lvb.json": e5})); o = S.score()
        c.ok(o["cells"]["a2"]["SD_C2_L"] is True, "labeller_part's own meta block (merged at the top level) does not shadow the file's eval_bs / items_sha")

    @case("FROZEN_ARGS from a scratch tree: the file is missing under ROOT -> read from the repo, sha-checked; distinct messages",
          "repo copy resolves (anchored); a wrong sha -> SHA; set-but-absent says 'not found at'; unset says 'unset'")
    def _(c):
        keep_root, keep_f = S.ROOT, dict(S.FROZEN_ARGS); fake = os.path.join(FX, "fake_root"); os.makedirs(fake, exist_ok=True)
        try:
            S.ROOT = fake; S.FROZEN_ARGS.update(SHIPPED_FROZEN)
            b_, fz = S.frozen_ref("a1"); c.obs = {"resolved": (fz or {}).get("path")}
            c.ok(b_ is not None and fz["anchored"] and fz["path"].replace("\\", "/").startswith("D:/Latent/"), "repo copy read (ROOT is a scratch tree)")
            S.FROZEN_ARGS["a1"] = (SHIPPED_FROZEN["a1"][0], "0" * 64); b_, fz = S.frozen_ref("a1"); c.ok(b_ is None and fz["status"] == "SHA", "repo copy still sha-checked")
            S.FROZEN_ARGS["a1"] = ("scratch/no_such_args.json", "0" * 64); b_, fz = S.frozen_ref("a1"); c.obs["absent"] = fz["why"][:160]
            c.ok(b_ is None and "not found at" in fz["why"] and "unset" not in fz["why"], "set but absent -> 'not found at <paths>'")
            S.FROZEN_ARGS["a1"] = None; b_, fz = S.frozen_ref("a1"); c.ok(b_ is None and "unset" in fz["why"], "unset -> 'unset'")
        finally:
            S.ROOT = keep_root; S.FROZEN_ARGS.clear(); S.FROZEN_ARGS.update(keep_f)

    @case("SMOKE on a tiny-size tree (per_D 2, card cpu, off-anchor items_sha, eval_bs off G-EVMEM, probe steps_run 2, args differ)",
          "normal: A1 pending; smoke: every reading runs, each relaxed check listed, labels carry 'SMOKE: not a registered reading'")
    def _(c):
        m = {"lvb_sym_s1/eval_lvb.json": tiny_eval(BASE["lvb_sym_s1/eval_lvb.json"]), "lvb_sym_s1/args.json": dict(A1_ARGS, lr=2e-4)}
        o = run(m); c.obs = {"normal": o["cells"]["a1"]["label"][:160]}; c.ok(o["cells"]["a1"]["status"] == "PENDING", "normal mode: pending (readings never run)")
        S.SMOKE = True
        try:
            o = run(m); a = o["cells"]["a1"]; rel = " ".join(a.get("smoke_relaxed") or [])
            c.obs.update(smoke=a["label"][:200], relaxed=a.get("smoke_relaxed"), components=a.get("VB_C1_L_components"))
            c.ok(a["status"] in ("SCORED", "FAIL") and a.get("READINGS") and a.get("VB_C1_L_components"), "every reading ran")
            for w in ("per_D", "card", "eval_bs", "items_sha", "steps_run", "ARGS-DIFF"):
                c.ok(w in rel, "relaxed check listed: %s" % w)
            c.ok(all(x["label"].startswith(S.SMOKE_LABEL) for x in o["cells"].values()) and o["SMOKE"] == S.SMOKE_LABEL, "every label carries the SMOKE marker")
            c.ok(a["censors"]["ARGS_DIFF"]["ok"] is False and not a["censors"]["censored"], "ARGS-DIFF computed, reported, relaxed")
        finally:
            S.SMOKE = False

    @case("SMOKE refused at --final", "ValueError")
    def _(c):
        S.SMOKE = True
        try:
            try:
                run(final=("a1",)); c.ok(False, "no exception")
            except ValueError as e:
                c.obs = {"error": str(e)[:120]}; c.ok("refused" in str(e), "raised")
        finally:
            S.SMOKE = False

    @case("SMOKE surfaces reading-stage field-path errors that normal mode hides behind pending",
          "a renamed reading-only field (probe_n_distinct_seen -> probe_n_seen) is listed absent; an exception inside a reading is caught (SMOKE ERROR)")
    def _(c):
        M = mod_layout(BASE); e = json.loads(json.dumps(M["lvb_sym_s1/eval_lvb.json"]))
        e["probe_n_seen"] = e.pop("probe_n_distinct_seen"); e["per_D"] = 2; e["card"] = "cpu"
        o = run(dict(M, **{"lvb_sym_s1/eval_lvb.json": e})); a = o["cells"]["a1"]
        c.obs = {"normal": a["label"][:140]}; c.ok(a["status"] == "PENDING" and "probe_n_distinct_seen" not in json.dumps(a), "normal: pending, the field-path error invisible")
        S.SMOKE = True; keep = S.mem_read
        try:
            o = run(dict(M, **{"lvb_sym_s1/eval_lvb.json": e})); a = o["cells"]["a1"]; c.obs["absent"] = sorted(a.get("smoke_absent_keys") or {})[:40]
            c.ok("probe_n_distinct_seen" in (a.get("smoke_absent_keys") or {}), "smoke: the reading's absent field is listed")

            def boom(P):
                raise KeyError("hist24 field renamed by the writer")
            S.mem_read = boom
            o = run(dict(M, **{"lvb_sym_s1/eval_lvb.json": e})); a = o["cells"]["a1"]; c.obs["error"] = a["label"][:160]
            c.ok(a["status"] == "SMOKE ERROR" and "KeyError" in a["label"] and a["label"].startswith(S.SMOKE_LABEL), "smoke: a reading exception is caught and reported")
        finally:
            S.SMOKE = False; S.mem_read = keep

    @case("CAUSAL-DEEP-L fail-safe: a band depth with 0 clean pairs and follow None; another with follow None at 200 pairs",
          "both excluded and listed, never raising; all depths bad -> NOT-APPLICABLE")
    def _(c):
        cd = cdeep_part(); cd["ii_prime"]["by_D"]["9"] = {"n_clean": 0, "follow": None, "own": None}
        cd["i_prime"]["by_D"] = dict(cd["i_prime"]["by_D"]); cd["i_prime"]["by_D"]["10"] = {"n_clean": 200, "follow": None, "own": 0.0}
        e = dict(BASE["lvb_self_s1/eval_lvb.json"], causal_deep=cd); o = run({"lvb_self_s1/eval_lvb.json": e}); x = o["cells"]["a2"]["BESIDE"]
        c.obs = {"a2": o["cells"]["a2"]["label"][:100], "cd": o["cells"]["a2"].get("CAUSAL_DEEP_L")}
        c.ok(o["cells"]["a2"]["status"] == "SCORED" and x["CAUSAL_DEEP_L"] == "CAUSAL-DEEP-L", "read on D 11-12")
        cd2 = cdeep_part(); cd2["ii_prime"]["by_D"] = {D: {"n_clean": 0, "follow": None, "own": None} for D in cd2["ii_prime"]["by_D"]}
        o = run({"lvb_self_s1/eval_lvb.json": dict(BASE["lvb_self_s1/eval_lvb.json"], causal_deep=cd2)})
        c.ok(o["cells"]["a2"]["status"] == "SCORED" and str(o["cells"]["a2"]["BESIDE"]["CAUSAL_DEEP_L"]).startswith("NOT-APPLICABLE"), "empty band -> NOT-APPLICABLE")
        r = S.p_causal_deep(e, {D: 0.97 for D in range(9, 13)}); c.obs["excluded"] = r.get("excluded_why")
        c.ok(r["excluded"] == [9, 10] and r["B_used"] == [11, 12], "D 9 and D 10 excluded and listed")

    @case("informational: Experiments' e2e CPU smoke (scratch/_tmp/e2e lvb_sym_s1, trained mini-run; small files copied read-only)",
          "every part it has written parses with no missing field; its log carries test_gate_sig_hits and its args carry arch / arch_class")
    def _(c):
        src = "D:/Latent/scratch/_tmp/e2e/boxes/runs_fsq/lvb_sym_s1"; dst = os.path.join(HERE, "_fx_e2e", "lvb_sym_s1")
        if not os.path.exists(os.path.join(src, "eval_lvb.json")):
            c.obs = {"absent": src}; c.ok(True, "e2e output absent (informational)"); return
        os.makedirs(dst, exist_ok=True)
        for fn in ("args.json", "log.jsonl", "eval_lvb.json"):
            with open(os.path.join(src, fn), "rb") as a_, open(os.path.join(dst, fn), "wb") as b_:
                b_.write(a_.read())
        ev = S._load(os.path.join(dst, "eval_lvb.json")); keep_pd, keep_r = S.PER_D, S.R; S.PER_D = ev.get("per_D"); S.R = os.path.dirname(dst)
        try:
            parts = [q for q in S.REQ_PARTS["a1"] if all(x in ev for x in S.PARTS_A[q])]; P = S.parse_A(ev, parts); info = S.log_info("lvb_sym_s1")
            args = S.jl("lvb_sym_s1", "args.json", plain=True) or {}
            c.obs = {"parts": parts, "not_yet": [q for q in S.REQ_PARTS["a1"] if q not in parts], "missing": {q: x.get("missing") for q, x in P.items() if x.get("missing")},
                     "void": {q: str(x.get("void"))[:160] for q, x in P.items() if x.get("void")}, "log_test_gate_sig_hits": info.get("test_gate_sig_hits"),
                     "arch": (args.get("arch"), args.get("arch_class"))}
            sm = {q: [m_ for m_ in (x.get("missing") or []) if not (q == "probe" and "curve" in m_ and all(not (v_.get("curve")) and (v_.get("steps_run") or 0) < 500
                                                                                    for v_ in (ev.get("probe") or {}).values()))] for q, x in P.items()}
            c.obs["missing_beyond_smoke_curve"] = {q: v_ for q, v_ in sm.items() if v_}
            c.ok(not any(sm.values()), "no missing fields in the written parts (empty probe curves at steps_run < 500 = smoke artefact)")
            c.ok(info.get("test_gate_sig_hits") is not None and args.get("arch") and args.get("arch_class"), "log / args carry the scorer's keys")
        finally:
            S.PER_D, S.R = keep_pd, keep_r
            for fn in ("args.json", "log.jsonl", "eval_lvb.json"):
                if os.path.exists(os.path.join(dst, fn)):
                    os.remove(os.path.join(dst, fn))

    @case("the module's own CPU smoke output (untrained LangSLVA, per_D 2; smoke_eval/eval_lvb.json)", "every part present parses with no missing field; no crash")
    def _(c):
        fp = os.path.join(HERE, "smoke_eval", "eval_lvb.json")
        if not os.path.exists(fp):
            c.ok(False, "smoke output absent"); return
        ev = S._load(fp); keep = S.PER_D; S.PER_D = ev.get("per_D")
        try:
            parts = [q for q in S.REQ_PARTS["a1"] if all(x in ev for x in S.PARTS_A[q])]
            P = S.parse_A(ev, parts); c.obs = {"parts": parts, "missing": {q: x.get("missing") for q, x in P.items() if x.get("missing")},
                                               "void": {q: (str(x.get("void"))[:200] if x.get("void") else None) for q, x in P.items()}}
            c.ok(set(parts) >= {"panel", "acc", "stall", "vswap", "cf", "hist24", "causal_deep"}, "the smoke wrote every part before probe (it segfaulted, rc 139, in the CPU probe part)")
            lg = open(os.path.join(HERE, "smoke_eval", "smoke.log"), encoding="utf-8", errors="replace").read() if os.path.exists(os.path.join(HERE, "smoke_eval", "smoke.log")) else ""
            c.obs["smoke_rc"] = lg.strip().split(chr(10))[-1] if lg else None
            c.ok(not any(x.get("missing") for q, x in P.items()), "no missing fields in any written part")
            mm, bad = S.meta_check(ev, "A", parts, {"eval_bs": ev.get("eval_bs") or {}, "card": ev.get("card")}, ev.get("test_items_sha"))
            c.obs["meta_missing"] = mm; c.obs["meta_mismatch"] = bad; c.ok(not mm, "meta fields present")
            l0 = S._load(os.path.join("D:/Latent", "scratch", "gate_s218", "g_l0.json")); c.obs["g_l0_has_test_items_sha"] = bool(l0.get("test_items_sha"))
        finally:
            S.PER_D = keep

    @case("Experiments' 15:10 draft layout (exposed flags, int y_hyb, cf.cells + selected pairs, fd_slot_scores, top-level card / test_sha, G-L0 panel layout)",
          "identical readings to the LAYOUTS form: VB-C1-L, SD-C2-L; PaR-C still not VB-C1-L / PRECOMPUTE-L / CF-CARRY")
    def _(c):
        E = exp_layout(BASE); files = dict(E); materialize(files); o = S.score(); a = o["cells"]["a1"]
        c.obs = {"a1": a["label"][:160], "a2": o["cells"]["a2"]["label"][:120], "cf_flags": rdg(a, "CF_L").get("flags")}
        c.ok(a["VB_C1_L"] is True and all(v == "MET" for v in comps(a).values()), "VB-C1-L"); c.ok(o["cells"]["a2"]["SD_C2_L"] is True, "SD-C2-L")
        c.ok(o["G_L0"]["failed"] == [] and not o["G_L0"]["missing"], "G-L0 panel layout parsed")
        c.ok(rdg(a, "CF_L").get("flags") and "read from the file" in rdg(a, "CF_L")["flags"][0], "selected-only CF store flagged")
        vs = dict(spec_A("a1")["vswap"], ii_prime=dict(follow=1.0), i_prime=dict(follow=0.0))
        pc = to_exp_eval(ev_lvb(spec_A("a1", vswap=vs, fd=lambda m, t, k: 0.95 if k >= 3 else 0.7, cf=dict(tf=lambda D, j, r: 1))))
        materialize(dict(E, **{"lvb_sym_s1/eval_lvb.json": pc})); o = S.score(); a = o["cells"]["a1"]
        c.ok(not a["VB_C1_L"] and lab(a)["PRECOMPUTE_L"] is True and lab(a)["CF_CARRY"] is True, "PaR-C in this layout")
        bad = dict(E["lvb_direct_s1/eval_direct.json"], test_sha=dict(TEST_SHA_D, **{"5": "other"}))
        materialize(dict(E, **{"lvb_direct_s1/eval_direct.json": bad})); o = S.score()
        c.ok(o["cells"]["direct"]["status"] == "PENDING" and "items_sha" in o["cells"]["direct"]["label"], "DIRECT test_sha != A1's -> mismatch")

    @case("real tree (D:/Latent/boxes/runs_fsq), read only, OUT redirected", "no crash; every cell pending or UNREAD; nothing passes; the built G-L0 file parses")
    def _(c):
        keep_r = S.R; keep = dict(S.FROZEN_ARGS); keep_p = dict(S.PATHS)
        try:
            S.R = "D:/Latent/boxes/runs_fsq"; S.FROZEN_ARGS.update(SHIPPED_FROZEN)
            for cell in ("a1", "direct"):                                      # Note 218-9's frozen files verify (sha256) and carry the architecture keys
                b_, fz = S.frozen_ref(cell); c.ok(b_ is not None and fz["anchored"] and b_.get("arch") and b_.get("arch_class"), "%s frozen args verify" % cell)
            c.ok(S.FROZEN_ARGS["b1"] is None, "b1 not frozen yet (pending)")
            S.PATHS["gate_l0"] = ("@ROOT", "scratch/gate_s218/g_l0.json")
            o = S.score(); c.obs = {k: o["cells"][k]["status"] for k in o["cells"]}
            c.ok(all(o["cells"][k]["status"] in ("PENDING", "UNREAD") for k in o["cells"]), "pending / UNREAD only")
            c.ok(not any(v is True for v in o["PRIMARY"].values()), "no primary passes")
            g = o["G_L0"]; c.obs["G_L0"] = None if g is None else {k: g[k] for k in ("failed", "missing", "code_bug", "flag_inconsistent", "file_pass")}
            c.ok(g is None or (not g["missing"] and g["failed"] == [] and not g["code_bug"]), "real G-L0 (if present): every band 3-16 usable")
        finally:
            S.R = keep_r; S.FROZEN_ARGS.update(keep); S.PATHS.clear(); S.PATHS.update(keep_p)


def layouts_case():
    @case("LAYOUTS completeness: every key the scorer read during the suite is one LAYOUTS entry", "no unlisted key")
    def _(c):
        miss = sorted(k for k in S.ACCESSED if not str(k).isdigit() and k not in S.LAYOUTS and not S.LAYOUT_PATTERN.match(str(k)))
        c.obs = {"n_accessed": len(S.ACCESSED), "unlisted": miss}; c.ok(not miss, "%d unlisted keys" % len(miss))


def main():
    global BASE
    t0 = time.time(); os.makedirs(TREE, exist_ok=True); SHIPPED_FROZEN.update(S.FROZEN_ARGS); configure(); BASE = base_files(); materialize(BASE); freeze()
    define_cases(); layouts_case()
    rep = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "scorer": os.path.join(HERE, "score_s218.py"), "registered": S.REGISTERED,
           "n_cases": len(REPORT), "n_pass": sum(r["result"] == "PASS" for r in REPORT), "n_fail": sum(r["result"] == "FAIL" for r in REPORT),
           "sec": round(time.time() - t0, 1), "cases": REPORT, "resolved_ambiguities": S.RESOLVED,
           "fixture_only_config": "FROZEN_ARGS for direct / a1 / b1 point at fixture files (sha256 of the written bytes); the shipped scorer keeps None -> pending"}
    with open(os.path.join(HERE, "fixtures_report.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rep, fh, indent=1, default=str, ensure_ascii=False)
    print("fixtures: %d / %d PASS (%.1f s)" % (rep["n_pass"], rep["n_cases"], rep["sec"]))
    return rep


if __name__ == "__main__":
    main()
