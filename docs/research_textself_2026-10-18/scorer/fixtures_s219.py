"""G-T12 fixtures for score_s219.py (THEORY §219 TEXT-SELF; TS_v1.md @ 146a25d0 + Addenda 1-2). CPU only, <= 2 threads, idle priority, python -B.
Synthetic trees under ./_fx/<case>/ (deleted after each case); the real tree D:/Latent is only read (latent constants recompute, coin stream, nested170's
parser AST, s170_cot_s1 args / eval.json, the read-only real-tree run with OUT redirected to ./_real_out*.json). Report -> ./fixtures_report.json.
    python -B fixtures_s219.py [case-substring ...]
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:                                                           # noqa: BLE001
    pass
import ast        # noqa: E402
import copy       # noqa: E402
import datetime   # noqa: E402
import hashlib    # noqa: E402
import json       # noqa: E402
import math       # noqa: E402
import shutil     # noqa: E402
import time       # noqa: E402
import traceback  # noqa: E402
try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:                                                           # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_s219 as S  # noqa: E402

REAL = "D:/Latent"; REAL_R = REAL + "/boxes/runs_fsq"
FX = os.path.join(HERE, "_fx"); REPORT = os.path.join(HERE, "fixtures_report.json")
CARD3070, CARD2070 = "NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 2070"
L = S.LIN["s1"]; RESULTS, LINES, PANEL = [], [], {}
SHA_T1, SHA_T2 = hashlib.sha256(b"t1").hexdigest(), hashlib.sha256(b"t2").hexdigest()


def rd_real(rel):
    with open(os.path.join(REAL, rel)) as fh:
        return json.load(fh)


BASE_ARGS = rd_real("boxes/runs_fsq/s170_cot_s1/args.json"); BASE_EVAL_BF16 = rd_real("boxes/runs_fsq/s170_cot_s1/eval.json")


# ------------------------------------------------------------------ curves
def ext(a, floor=0.04):
    return {n: a.get(n, floor) for n in range(1, 25)}


A0 = ext({**{n: 1.0 for n in range(1, 9)}, **{int(k): v for k, v in BASE_EVAL_BF16["acc_by_n"].items()}})   # the committed bf16 base curve, n 17-24 at 0.04


def a0x(x):
    if x <= 8:
        return 1.0
    if x >= 24:
        return A0[24]
    lo = math.floor(x); f = x - lo
    return A0[lo] if f < 1e-12 else (1 - f) * A0[lo] + f * A0[lo + 1]


def q3(a):
    out = {n: round(min(max(v, 0.0), 1.0), 3) for n, v in a.items()}
    out[24] = min(out[24], 0.971)                                          # 29 overflow items at n 24
    return out


def make_curve(x, floor=0.05):
    """a(n) = 1 below the crossing, one segment of drop 0.5 through 0.5 at x (exact up to 3-decimal rounding), floor after; x >= 24 -> never crosses."""
    if x >= 24:
        return q3({n: (1.0 if n < 24 else 0.971) for n in range(1, 25)})
    k = math.floor(x); f = x - k; a = {}
    for n in range(1, 25):
        a[n] = 1.0 if n < k else (0.5 + 0.5 * f) if n == k else (0.5 * f) if n == k + 1 else min(floor, 0.5 * f)
    return q3(a)


def sig(z):
    return 1 / (1 + math.exp(-z))


def pos_panel(kind, T_r):
    TI = S.test_info(); c = 0.0775; out = {}
    for n in range(1, 25):
        b = TI["by"][n]; ps = []
        for P_, C_ in zip(b["P"], b["C"]):
            tot = P_ + C_
            ps.append(c + (1 - c) * sig(((1.31 - tot / T_r) / 0.076) if kind == "PROP" else ((45.6 - (tot - T_r)) / 11.2)))
        out[n] = 1.0 if n <= 8 else mean(ps)
    return q3(out)


def mean(v):
    v = list(v)
    return sum(v) / len(v)


PANELS = {
    "N-SHIFT": (q3({n: a0x(n - 4) for n in range(1, 25)}), q3({n: a0x(n - 8) for n in range(1, 25)})),
    "N-PROP": (q3({n: a0x(8 + (n - 12) * 8 / 12) for n in range(1, 25)}), q3({n: a0x(8 + (n - 16) * 8 / 16) for n in range(1, 25)})),
    "BETTER": (make_curve(15.460 + 1.5), make_curve(20.539 + 3.0))}


# ------------------------------------------------------------------ file builders
_EVCACHE = {}


def eval_text_obj(curve, model_sha, card=CARD3070, bs=32, layout="list", sha=None, top_card=None, tweak=None, times=None):
    key = (tuple(sorted(curve.items())), model_sha, card, bs, layout, sha, top_card, tweak, times)
    if key in _EVCACHE:
        return _EVCACHE[key]
    TI = S.test_info(); items = {}; acc, accf, fvb, ntm, ch = {}, {}, {}, {}, {}
    for n in range(1, 25):
        b = TI["by"][n]; k = int(round(curve[n] * 1000)); feas = [q for q in range(1000) if not b["ovf"][q]]
        assert k <= len(feas), (n, k)
        cs = set(feas[:k]); cols = {f: [] for f in S.ITEM_FIELDS}
        for q in range(1000):
            y, P_, C_, ov = b["t"][q], b["P"][q], b["C"][q], b["ovf"][q]; c = q in cs
            ans = y if c else (-1 if (ov or q % 2) else (y + 1) % 11)
            if tweak == "overflow_answer" and n == 24 and ov and q == next(i for i in range(1000) if b["ovf"][i]):
                ans = y
            ovf = ov
            if tweak == "overflow_flag" and n == 24 and q == 0:
                ovf = not ov
            row = dict(idx=(n - 1) * 1000 + q, n=n, y=y, P=P_, C_struct=C_, overflow=ovf, answer=ans, ntok=0 if ov else (C_ if c else min(C_ + 50, 1025 - P_)),
                       eot=c, form_valid=c, strict=c, first_bad_step=None if (c or ov) else 1 + q % n,
                       first_bad_pos=None if (c or ov) else P_ + ((q * 37) % max(1, C_)) + 1)     # text_eval A2: P + j + 1 (1-based row)
            for f_ in S.ITEM_FIELDS:
                cols[f_].append(row[f_])
        if layout == "columnar":
            items[str(n)] = cols
        else:
            items.setdefault("_list", []).extend({f_: cols[f_][i] for f_ in S.ITEM_FIELDS} for i in range(1000))
        acc[str(n)] = k / 1000; nf = len(feas); accf[str(n)] = (min(k, nf) / nf) if nf else None; fvb[str(n)] = k / 1000
        ntm[str(n)] = mean(cols["ntok"]); yc = {}
        for y in b["t"]:
            yc[y] = yc.get(y, 0) + 1
        ch[str(n)] = max(yc.values()) / 1000
    meta = {"run": None, "model": "x", "model_sha256": model_sha, "card": card, "dtype": "fp32", "tf32": False, "cot_eval_bs": bs, "test_seed": 17100, "test_n_max": 24,
            "ns": list(range(1, 25)), "limit_per_n": 0, "n_items": 24000, "items_sha256": sha or TI["sha"], "budget_rule": "per batch min(max C_struct + 50, 1025 - P)",
            "commit": "fixture", "sec": 1, "t_start": "2026-10-20T09:00:00+00:00", "t_end": "2026-10-20T12:00:00+00:00", "peak_gib": 5.0}
    if times:
        meta.update(times)
    o = {"meta": meta, "items": items["_list"] if layout == "list" else items, "acc_by_n": acc, "acc_feasible_by_n": accf, "form_valid_by_n": fvb,
         "strict_by_n": fvb, "ntok_mean_by_n": ntm, "chance_by_n": ch}
    if top_card:
        o["card"] = top_card
    s = json.dumps(o, separators=(",", ":"))
    _EVCACHE[key] = s
    return s


def parser_defaults():
    """nested170.main's add_argument defaults by AST (read-only; v1_argkeys' method): {key: default}; store_true -> False; required -> None."""
    with open(REAL + "/boxes/nested170.py", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"); out = {}
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            name = node.args[0].value.lstrip("-"); kw = {k.arg: k.value for k in node.keywords}
            if "default" in kw:
                out[name] = ast.literal_eval(kw["default"])
            elif "action" in kw and ast.literal_eval(kw["action"]) == "store_true":
                out[name] = False
            else:
                out[name] = None
    return out


PD = parser_defaults()


def realistic_args(role, **mods):
    """vars(a) the way boxes/textself_ft.py (nested170's parser copy + FT keys) would write it for the registered launch, + info keys (META_T)."""
    a = dict(PD)
    for k, v in BASE_ARGS.items():
        if k in PD:
            a[k] = v
    a.update(task="nested", format=BASE_ARGS["format"], mem_write=None, write_mix_start=None, write_mix_end=None, think_k=0, micro=1, auto_micro=True,
             device="cuda:0", out="boxes/runs_fsq/%s" % L[role], resume=None)
    req = S.FT_REQ("s1", role); a.update(req); a.update(mem_check_steps=2, args_ref="scratch/s219_args_ref_%s.json" % role)
    a.update(train_gpu="NVIDIA GeForce RTX 3090", train_driver="595.71.05", torch_version="2.5.1+cu124", cuda_version="12.4", cudnn_version=90100,
             python_version="3.11.10", image_tag="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel", hostprobe_sps=4.1,
             excl_survive_by_n={str(n): v for n, v in zip(range(1, 9), (0.01, 0.2, 0.9, 0.99, 1.0, 1.0, 1.0, 1.0))})
    for k, v in mods.items():
        if v == "<drop>":
            a.pop(k, None)
        else:
            a[k] = v
    return a


def gate_by_n(ns, keep, pi, piE, labeller, window):
    out = {}
    for n in ns:
        kp = keep.get(n, 0.995) if isinstance(keep, dict) else keep
        out[str(n)] = {"window": window, "labeller": labeller, "items": 2000, "n_kept": int(round(2000 * kp)), "keep": kp,
                       "pi": pi.get(n, 0.9995) if isinstance(pi, dict) else pi, "pi_E": piE, "pi_ci95": [0.998, 0.9999], "pi_E_ci95": [0.998, 0.9999], "records": []}
    return out


def pooled(B, ns):
    es = [B[str(n)] for n in ns]; nk = sum(e["n_kept"] for e in es)
    return {"n_kept": nk, "pi": sum(e["pi"] * e["n_kept"] for e in es) / nk, "pi_E": sum(e["pi_E"] * e["n_kept"] for e in es) / nk,
            "pi_ci95": [0.9987, 0.9999], "pi_E_ci95": [0.9985, 0.9999]}


def rule(B, ns):
    return S.gate_rule(B, ns)[0]


class Tree:
    def __init__(self, name):
        self.root = os.path.join(FX, name); self.R = os.path.join(self.root, "boxes", "runs_fsq")
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.R)

    def p(self, d, f):
        return os.path.join(self.R, d, f)

    def w(self, d, f, obj):
        os.makedirs(os.path.join(self.R, d), exist_ok=True)
        with open(self.p(d, f), "w", newline="\n") as fh:
            fh.write(obj if isinstance(obj, str) else json.dumps(obj))

    def rm(self, d, f):
        os.remove(self.p(d, f))

    def get(self, d, f):
        with open(self.p(d, f)) as fh:
            return json.load(fh)

    def wroot(self, rel, obj):
        p = os.path.join(self.root, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="\n") as fh:
            json.dump(obj, fh)

    def done(self):
        shutil.rmtree(self.root, ignore_errors=True)


TOK_SELF, TOK_PLAIN = (64 * 300, 64 * 350, 64 * 420), (64 * 90, 64 * 100, 64 * 140)   # per step: target tokens, token-positions, padded positions


def tok_at(cum_self, step):
    ns = cum_self; npl = step - ns
    return {"plain": {"target_tokens": npl * TOK_PLAIN[0], "token_positions": npl * TOK_PLAIN[1], "padded_positions": npl * TOK_PLAIN[2]},
            "self": {"target_tokens": ns * TOK_SELF[0], "token_positions": ns * TOK_SELF[1], "padded_positions": ns * TOK_SELF[2]}}


def write_log(t, d, last=12000, reuse=0, cursor_override=None, drop_cursor=False, train_done=True, nonfinite_at=None, td_mod=None, ini_mod=None):
    """textself_ft's log.jsonl (A8 / A10) + train_done.json (A9) for a run that ended at `last`."""
    coins, _ = S.coin_stream(); cum = [0]
    for c in coins:
        cum.append(cum[-1] + int(c))
    lines = []
    for s_ in range(100, last + 1, 100):
        if nonfinite_at == s_:
            lines.append(json.dumps({"step": s_, "nonfinite": 1}))
        r = {"step": s_, "loss": 0.01, "loss_last": 0.012, "sec": s_ // 2, "plain_loss": 0.001, "self_loss": 0.02, "self_value_ce": 0.002, "self_copy_ce": 0.018,
             "self_frac": 0.5, "pool_cursor": 64 * cum[s_], "n_self_steps": cum[s_], "self_reuse": reuse if s_ == last else 0, "lr": 1e-5, "gn_preclip": 0.3,
             "clip_frac_100": 0.0, "n_nonfinite": int(nonfinite_at is not None and s_ >= nonfinite_at), "train_tokens": tok_at(cum[s_], s_)}
        if cursor_override is not None and s_ == last:
            r["pool_cursor"] = cursor_override
        if drop_cursor:
            r.pop("pool_cursor")
        lines.append(json.dumps(r))
    t.w(d, "log.jsonl", "\n".join(lines) + "\n")
    if train_done and last == 12000:
        td = {"steps": 12000, "train_sec": 9000, "n_nonfinite": 0, "n_self_steps": cum[last], "pool_cursor": cursor_override if cursor_override is not None else 64 * cum[last],
              "self_reuse": reuse, "train_tokens": tok_at(cum[last], last), "self_n_counts": {}, "model_sha256": SHA_T1 if d == L["t1"] else SHA_T2}
        td.update(td_mod or {})
        t.w(d, "train_done.json", td)
        ini = {"init_from": "boxes/runs_fsq/%s/model.pt" % ("s170_cot_s1" if d == L["t1"] else L["t1"]), "sha256": S.BASE_SHA if d == L["t1"] else SHA_T1,
               "missing": [], "unexpected": []}
        ini.update(ini_mod or {})
        for f, o in (("init_from.json", ini), ("args_diff.json", {"ref": "x", "diff": {}}),
                     ("mem_check.json", {"micro": 1}), ("mem_check_self.json", {"peak_gib": 14.0}), ("timing.json", {"sec_per_step": 0.4})):
            t.w(d, f, o)


def write_draws(t, d, rnd):
    """textself_ft A6: {step, plain_hash, coin, n (plain) | self_n + U (self)}."""
    coins, U = S.coin_stream(); ns = (9, 10, 11, 12) if rnd == 1 else (13, 14, 15, 16, 13, 14, 15, 16, 9, 10, 11, 12)
    x = []
    for i, c in enumerate(coins):
        r = {"step": i + 1, "plain_hash": hashlib.sha1(str(i + 1).encode()).hexdigest(), "coin": c}
        if c:
            r.update(self_n=[ns[(i + j) % len(ns)] for j in range(64)], U=U[i])
        else:
            r["n"] = [1 + (i + j) % 8 for j in range(64)]
        x.append(r)
    t.w(d, "draws_log.json", json.dumps(x, separators=(",", ":")))


def textself(rnd, pi=0.9996, piE=0.9994, ci=(0.9985, 0.9999), ciE=(0.9982, 0.9998), cont=None, drop=None, lk=True, lk_t_start="2026-10-21T10:00:00+00:00"):
    """text_eval E2 (part_textself_cpu + part_textself_gpu) as written at 3b0bb06f."""
    blk = lambda na: {"n_audit": na, "pi": pi, "pi_E": piE, "pi_ci95": list(ci), "pi_E_ci95": list(ciE), "kept_wrong_label": 0, "kept_wrong_E": 0}  # noqa: E731
    la = dict(blk(5000), by_window={w: blk(2500) for w in (("W1",) if rnd == 1 else ("W2", "W1"))},
              kept_wrong=[{"syms": [" 1"], "n": 9, "step": 3, "op": "*", "a": 2, "b": 3, "value": 5, "gold": 6, "wrong_steps": [3], "label": 5, "gold_label": 6,
                           "pattern": "both"}], seed=23270 if rnd == 1 else 23370, cursor=383488, n_consumed=383488, pool_sha256="0" * 64)
    con = {"pool_records": 388000, "hits_by_set": {"test_n30": 0}, "hits_by_set_consumed": {"test_n30": 0}, "pool_excl_hits": 0, "self_reuse": 0, "pool_test_hits": 0,
           "gold_key_hits": 0, "excl_sizes": {}, "excl_meta": {}}
    con.update(cont or {}); con["CONTAMINATED"] = bool(con["pool_excl_hits"] or con["self_reuse"] or con["gold_key_hits"])
    o = {"label_audit": la, "contamination": con, "meta_cpu": {"commit": "fixture", "sec": 1, "pool": "x"},
         "memo": {str(n): {"window": "W1" if n < 13 else "W2", "acc_pool": 1.0, "acc_gate": 0.995, "items": [2000, 2000], "gap": 0.005, "FLAG_gt_0.02": False}
                  for n in ((9, 10, 11, 12) if rnd == 1 else range(9, 17))}}
    if lk:
        o["labeller_kept"] = {str(n): {"items": 500, "n_kept": int(500 * max(0.0, 1.0 - 0.04 * (n - 9))), "keep": max(0.0, 1.0 - 0.04 * (n - 9)), "pi": 0.999, "pi_E": 0.999,
                                       "fail_step_hist": {}} for n in range(9, 25)}
        o["labeller_kept_meta"] = {"seed": 23280 if rnd == 1 else 23380, "per_n": 500, "label_bs": 128, "t_start": lk_t_start}
        o["meta_gpu"] = {"model": "x", "model_sha256": SHA_T1 if rnd == 1 else SHA_T2, "card": CARD3070, "dtype": "fp32", "tf32": False, "cot_eval_bs": 32}
    if drop:
        o.pop(drop)
    return o


def pool_meta(rnd, card=CARD3070, n_records=None, t_start=None, base_windows=(), wsha=None):
    """s219_build_pool's pool_meta.json."""
    if rnd == 1:
        pn = {str(n): 97000 for n in range(9, 13)}; T = {"9": 200, "10": 240, "11": 280, "12": 313}; W = {"W1": {"seed": 23260, "ns": [9, 10, 11, 12], "gate": "GATE_T1"}}
    else:
        pn = {**{str(n): 32500 for n in range(9, 13)}, **{str(n): 65000 for n in range(13, 17)}}; T = {"13": 360, "14": 420, "15": 480, "16": 531}
        W = {"W2": {"seed": 23360, "ns": [13, 14, 15, 16], "gate": "GATE_T2"}, "W1": {"seed": 23361, "ns": [9, 10, 11, 12], "gate": "GATE_T1p"}}
    for w in W:
        W[w]["labeller_sha256"] = (wsha or {}).get(w) or (S.BASE_SHA if (rnd == 1 or w in base_windows) else SHA_T1)
    return {"round": rnd, "lineage": "s1", "seeds": {"pool": 23200 if rnd == 1 else 23300, "perm": 23201 if rnd == 1 else 23301}, "per_n": pn,
            "ns": sorted(int(k) for k in pn), "n_records": n_records or sum(pn.values()), "T_max_by_n": T, "L0_max_by_n": {}, "windows": W,
            "labeller": {w: "x" for w in W}, "labeller_sha256": {w: "y" for w in W}, "sha256": "0" * 64, "builder_commit": "fixture", "card": card,
            "dtype": "fp32", "tf32": False, "label_bs": 128,
            "t_start": t_start or ("2026-10-19T08:00:00+00:00" if rnd == 1 else "2026-10-21T09:00:00+00:00")}


def equiv_obj(ns, w, share=1.0):
    bn = {str(n): {"items": 250, "agree_keep": 250, "agree_values": 250, "agree_fail_step": 250, "agree_all": int(round(250 * share)), "share": round(250 * share) / 250,
                   "PASS": round(250 * share) / 250 >= 0.995} for n in ns}
    return {"by_n": bn, "disagreements": [] if share == 1.0 else [{"i": 1, "n": ns[0], "margin": 1e-4}], "min_margin": 0.41,
            "PASS": all(v["PASS"] for v in bn.values())}


def gate1_obj(keep=0.995, pi=0.9995, piE=0.9995, flag="auto", card=CARD3070, t_start="2026-10-19T06:00:00+00:00"):
    """scratch/s219_gate.py --stage score, round 1."""
    B = gate_by_n(range(9, 13), keep, pi, piE, "boxes/runs_fsq/s170_cot_s1", "W1")
    return {"round": 1, "by_n": B, "pooled_by_window": {"W1": dict(pooled(B, range(9, 13)), labeller="boxes/runs_fsq/s170_cot_s1")}, "fallback": None,
            "window_labeller": {"W1": "boxes/runs_fsq/s170_cot_s1"}, "labeller_full": {str(n): {"window": "W1", "keep": 0.5, "pi": 0.96, "pi_E": 0.95, "n_kept": 1000}
                                                                                    for n in range(9, 13)},
            "kept_vs_drawn": {}, "equiv": {"W1": equiv_obj([9, 10, 11, 12], "W1")}, "equiv_full": {}, "compose": {"W1": {"PASS": True}}, "labeller": "boxes/runs_fsq/s170_cot_s1",
            "labeller_sha256": S.BASE_SHA, "card": card, "GATE_T1": rule(B, range(9, 13)) if flag == "auto" else flag, "dtype": "fp32", "label_bs": 128, "t_start": t_start}


def gate2_obj(keep2=0.99, keep1=0.995, fb=None, card=CARD3070, flags="auto", t_start="2026-10-21T08:00:00+00:00"):
    """scratch/s219_gate.py --stage score, round 2 (labeller T1, base fallback per failed window)."""
    lab = "boxes/runs_fsq/s219_t1_s1"
    B = {**gate_by_n(range(9, 13), keep1, 0.9995, 0.9995, lab, "W1"), **gate_by_n(range(13, 17), keep2, 0.9995, 0.9995, lab, "W2")}
    o = {"round": 2, "by_n": B, "pooled_by_window": {"W2": dict(pooled(B, range(13, 17)), labeller=lab), "W1": dict(pooled(B, range(9, 13)), labeller=lab)},
         "GATE_T2": rule(B, range(13, 17)), "GATE_T1p": rule(B, range(9, 13)), "fallback": None, "card": card, "labeller": lab, "labeller_sha256": SHA_T1,
         "labeller_full": {str(n): {"window": "W1" if n < 13 else "W2", "keep": 0.3, "pi": 0.9, "pi_E": 0.9, "n_kept": 600} for n in range(9, 17)},
         "equiv": {"W2": equiv_obj([13, 14, 15, 16], "W2"), "W1": equiv_obj([9, 10, 11, 12], "W1")}, "kept_vs_drawn": {}, "dtype": "fp32", "label_bs": 128,
         "t_start": t_start}
    if flags != "auto":
        o.update(flags)
    if fb:
        o["fallback"] = {}
        for w, kp in fb.items():
            ns = S.WIN[w]; FB = gate_by_n(ns, kp, 0.9995, 0.9995, "boxes/runs_fsq/s170_cot_s1", w)
            o["fallback"][w] = {"labeller": "boxes/runs_fsq/s170_cot_s1", "labeller_sha256": S.BASE_SHA, "by_n": FB, "pooled": pooled(FB, ns), "GATE": rule(FB, ns)}
    return o


def build(name, a0=None, a1=None, a2=None, t2=True, t1=True, g2=True, **kw):
    """the full s1 chain (base eval_text + GATE-T1 + pool_t1 + T1 + GATE-T2 + pool_t2 + T2), every registered file present and consistent."""
    t = Tree(name); a0 = a0 or A0; a1 = a1 or make_curve(15.5); a2 = a2 or make_curve(20.6)
    t.w(L["base_model"], "args.json", BASE_ARGS); t.w(L["base_model"], "eval.json", BASE_EVAL_BF16)
    t.w(L["base"], "eval_text.json", eval_text_obj(a0, S.BASE_SHA, bs=kw.get("base_bs", 32), card=kw.get("base_card", CARD3070)))
    t.w(L["gate_t1"], "eval_gate.json", kw.get("gate1", gate1_obj()))
    for r in ("t1", "t2"):
        t.wroot("scratch/s219_args_ref_%s.json" % r, BASE_ARGS)
    if t1:
        t.w(L["pool_t1"], "pool_meta.json", kw.get("pm1", pool_meta(1)))
        write_log(t, L["t1"], **kw.get("log1", {})); write_draws(t, L["t1"], 1)
        t.w(L["t1"], "args.json", kw.get("args1", realistic_args("t1")))
        t.w(L["t1"], "eval_text.json", eval_text_obj(a1, SHA_T1, card=kw.get("t1_card", CARD3070), bs=kw.get("t1_bs", 32), sha=kw.get("t1_sha"),
                                                     layout=kw.get("layout", "columnar"), top_card=kw.get("t1_top_card"), tweak=kw.get("t1_tweak"),
                                                     times=kw.get("t1_times")))
        t.w(L["t1"], "eval_textself.json", kw.get("sd1", textself(1)))
    if g2:
        t.w(L["gate_t2"], "eval_gate.json", kw.get("gate2", gate2_obj()))
        t.w(L["pool_t2"], "pool_meta.json", kw.get("pm2", pool_meta(2)))
    if t2:
        write_log(t, L["t2"], **kw.get("log2", {})); write_draws(t, L["t2"], 2)
        t.w(L["t2"], "args.json", kw.get("args2", realistic_args("t2")))
        t.w(L["t2"], "eval_text.json", eval_text_obj(a2, SHA_T2, card=kw.get("t2_card", CARD3070), tweak=kw.get("t2_tweak")))
        t.w(L["t2"], "eval_textself.json", kw.get("sd2", textself(2)))
    return t


def run(t, final=False, na=()):
    S.set_root(t.R, t.root)
    out = S.score(("s1",) if final else (), na)
    LINES.append((t.root, out["s1"]["comparison"]["line"]))
    return out["s1"]


# ------------------------------------------------------------------ cases
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def chk(cond, msg):
    if not cond:
        raise AssertionError(msg)


def outc(s):
    return s["comparison"]["outcome"]


@case
def latent_constants_recompute():
    """Addendum 2 item 4: recompute the latent n50 constants read-only from the real eval_vb (n <= 16) + eval_deep (n >= 17) and assert 3-decimal equality;
    the files agree at n 13-16; the scorer's embedded curves equal the files; the scorer itself never reads them for a decision (latent_files_ignored)."""
    det = {}
    for r, d in L["latent"].items():
        vb, de = rd_real("boxes/runs_fsq/%s/eval_vb.json" % d), rd_real("boxes/runs_fsq/%s/eval_deep.json" % d)
        a = {int(n): v for n, v in vb["acc_n2"].items() if int(n) <= 16}; a.update({int(n): v for n, v in de["acc_n2"].items() if int(n) >= 17})
        chk(sorted(a) == list(range(1, 25)), "n range %s" % d)
        chk(all(abs(vb["acc_n2"][str(n)] - de["acc_n2"][str(n)]) < 1e-12 for n in range(13, 17)), "eval_vb != eval_deep at n 13-16 (%s)" % d)
        chk(all(card in str(x.get("card")) for x, card in ((vb, "3070"), (de, "3070"))) and de["meta"]["dtype"] == "fp32" and de["meta"]["tf32"] is False, "card/dtype")
        x = None
        for n in range(1, 24):                                              # independent restatement of the rule
            if a[n] >= 0.5 and a[n + 1] < 0.5:
                x = n + (a[n] - 0.5) / (a[n] - a[n + 1]); break
        chk(x is not None and round(x, 3) == S.LATENT_N50[r], "%s n50 %s != %s" % (d, x, S.LATENT_N50[r]))
        chk(all(abs(a[n] - S.LATENT_CURVES[r][n]) < 1e-12 for n in range(1, 25)), "embedded curve != file (%s)" % d)
        det[r] = round(x, 6)
    chk(S._LSC["n50_ok"] and S._LSC["bars_ok"], "module selfcheck")
    return "n50 %s == constants %s; curves equal; vb == deep at n 13-16" % (det, S.LATENT_N50)


@case
def crossed_subcase_wording():
    """Note 219-2 (Audit's G-T12 wording item): CROSSED's sub-case is named; the label is unchanged."""
    chk(S.four_way(1.0, 0.0)[0] == "CROSSED" and S.crossed_subcase(1.0, 0.0) == "converged at round 2", "converged")
    chk(S.four_way(1.0, -1.0)[0] == "CROSSED" and S.crossed_subcase(1.0, -1.0) == "order changed", "order changed (+ / -)")
    chk(S.four_way(-1.0, 1.0)[0] == "CROSSED" and S.crossed_subcase(-1.0, 1.0) == "order changed", "order changed (- / +)")
    chk(S.four_way(0.0, 1.0)[0] == "TEXT-AHEAD" and S.four_way(0.0, -1.0)[0] == "LATENT-AHEAD", "D1 in band never reaches CROSSED")
    return "converged / order changed named; D1-in-band cases absorbed by TEXT-/LATENT-AHEAD"


@case
def latent_bars_reproduced():
    """v1 §4.3: fed the latent files, the bar formulas reproduce 0.63 / 0.40 / 0.785 / 0.60."""
    a = {}
    for r in ("r0", "r1"):
        vb = rd_real("boxes/runs_fsq/%s/eval_vb.json" % L["latent"][r]); a[r] = {int(n): v for n, v in vb["acc_n2"].items()}
    b1, b2 = S.r1_bars(a["r0"]), S.r2_bars(a["r0"], a["r1"])
    got = (b1["MARGIN_GROWS"], b1["SHRINK_below"], b2["MARGIN_GROWS_2"], b2["SHRINK_below"])
    chk(got == (0.63, 0.40, 0.785, 0.60), got)
    return "bars %s" % (got,)


@case
def coin_stream_matches_latent():
    """G-T3 basis: the scorer's rng_s replay (seed 1 + 21800, coin then 64 scalar integers(0, 5)) == s215_ftself_s1 / s216_r2_s1 draws_log coins and U."""
    coins, U = S.coin_stream(); det = {}
    for d in (L["latent"]["r1"], L["latent"]["r2"]):
        x = rd_real("boxes/runs_fsq/%s/draws_log.json" % d); x = sorted(x, key=lambda r: r["step"])
        chk([bool(r["coin"]) for r in x] == coins, "coins %s" % d)
        chk(all(r["U"] == U[i] for i, r in enumerate(x) if r["coin"]), "U %s" % d)
        det[d] = sum(coins)
    chk(sum(coins) == 5992, sum(coins))
    return "12,000 coins and every self U equal; self steps %s" % det


@case
def neutral_map_vs_parser():
    """T1 / v1_argkeys: nested170's parser (AST, read-only) has 57 keys; the 30 absent from s170_cot_s1/args.json = TEXT_NEUTRAL_NEW + keep_ckpt_steps (META);
    every TEXT_NEUTRAL_NEW value == the parser default (cot_eval_bs 64)."""
    chk(len(PD) == 57, len(PD))
    absent = {k for k in PD if k not in BASE_ARGS}
    chk(absent == set(S.TEXT_NEUTRAL_NEW) | {"keep_ckpt_steps"}, sorted(absent ^ (set(S.TEXT_NEUTRAL_NEW) | {"keep_ckpt_steps"})))
    chk("keep_ckpt_steps" in S.META_T, "keep_ckpt_steps not META")
    bad = {k: (v, PD[k]) for k, v in S.TEXT_NEUTRAL_NEW.items() if not S._eq(v, PD[k])}
    chk(not bad, bad)
    chk(S.TEXT_NEUTRAL_NEW["cot_eval_bs"] == 64, "cot_eval_bs")
    return "57 keys; 30 absent = 29 neutral + keep_ckpt_steps; defaults equal"


@case
def ctx_facts():
    """v1 §2.8 (DERIVED): only n 24 overflows (29 / 1,000), max P + C 1,085, seven items exactly at 1,025; C_struct == len(cot_ids); parse_syms inverts render."""
    TI = S.test_info(); ov = TI["ctx"]["overflow_by_n"]
    chk(ov[24] == 29 and all(ov[n] == 0 for n in range(1, 24)), ov)
    chk(TI["ctx"]["max_PC"] == 1085 and TI["ctx"]["exactly_1025_n24"] == 7, TI["ctx"])
    N = S.nested(); its = S.test_items(24); bad = 0
    for it in its[::97]:
        bad += S.chain_len(it["syms"], N) != len(N.cot_ids(it["root"]))
    chk(bad == 0, "chain_len mismatches %d" % bad)
    return "overflow %s at n 24; max P+C %d; at 1025: %d" % (ov[24], TI["ctx"]["max_PC"], TI["ctx"]["exactly_1025_n24"])


@case
def four_way_unit():
    """v1 §4.4: exhaustive and exclusive over a grid; |Delta| exactly 0.5 inside; sentinels; marginal."""
    vals = [round(-3 + 0.05 * i, 2) for i in range(121)] + [0.5, -0.5, 1.0, -1.0]
    for d1 in vals:
        for d2 in vals:
            lab, marg = S.four_way(d1, d2)
            par = abs(d1) <= 0.5 and abs(d2) <= 0.5; ta = not par and d2 > 0.5 and d1 >= -0.5; la = not par and d2 < -0.5 and d1 <= 0.5
            exp = "PARITY" if par else "TEXT-AHEAD" if ta else "LATENT-AHEAD" if la else "CROSSED"
            chk(sum((par, ta, la)) <= 1 and lab == exp, (d1, d2, lab, exp))
            chk(marg == (lab in ("TEXT-AHEAD", "LATENT-AHEAD") and abs(d2) <= 1.0), (d1, d2, marg))
    chk(S.four_way(0.5, -0.5)[0] == "PARITY" and S.four_way(0.5 + 1e-12, 0.5 - 1e-12)[0] == "PARITY", "band edge")
    chk(S.four_way(0.0, -0.8) == ("LATENT-AHEAD", True) and S.four_way(0.0, 1.2) == ("TEXT-AHEAD", False), "marginal")
    chk(abs(S.delta("> 24", 20.539) - 3.471) < 1e-9 and abs(S.delta("< 1", 15.46) - (-14.47)) < 1e-9, "sentinels")
    chk(S.round1_only(0.5) == "PARITY-1" and S.round1_only(0.51) == "TEXT-AHEAD-1" and S.round1_only(-0.51) == "LATENT-AHEAD-1", "round1_only")
    chk(S.cross({n: 0.4 for n in range(1, 25)}, 0.5) == "< 1" and S.cross({n: 0.9 for n in range(1, 25)}, 0.5) == "> 24", "cross sentinels")
    return "grid %d x %d exclusive / exhaustive; band edge inside; sentinels" % (len(vals), len(vals))


def panel_case(name, a1, a2, expect, marginal=None):
    t = build("panel_" + name, a1=a1, a2=a2)
    try:
        s = run(t); c = s["comparison"]; PANEL[name] = {"D1": c["DELTA"].get(1), "D2": c["DELTA"].get(2), "outcome": c["outcome"]}
        chk(s["t1"]["status"] == "SCORED" and s["t2"]["status"] == "SCORED", (s["t1"]["label"], s["t2"]["label"]))
        chk(c["statistic"] == "levels", c["statistic"])
        chk(outc(s).split(" (")[0] == expect, (outc(s), c["line"][:300]))
        if marginal is not None:
            chk(c["marginal"] == marginal, ("marginal", c["marginal"]))
        chk(sum(c["rows"].values()) == 1 and c["rows"][expect.replace("-", "_")], c["rows"])
        if expect == "LATENT-AHEAD":
            chk("FAIL-POS" in c["line"], "FAIL-POS missing on a LATENT-AHEAD line")
        return "Δ1 %s Δ2 %s -> %s" % (S.fmt(c["DELTA"][1]), S.fmt(c["DELTA"][2]), outc(s))
    finally:
        t.done()


@case
def panel_N_SHIFT():
    return panel_case("N-SHIFT", *PANELS["N-SHIFT"], "LATENT-AHEAD", False)


@case
def panel_N_PROP():
    return panel_case("N-PROP", *PANELS["N-PROP"], "PARITY")


@case
def panel_POS_PROP():
    return panel_case("POS-PROP", pos_panel("PROP", 313), pos_panel("PROP", 531), "PARITY")


@case
def panel_POS_ADD():
    return panel_case("POS-ADD", pos_panel("ADD", 313), pos_panel("ADD", 531), "LATENT-AHEAD", False)


@case
def panel_BETTER():
    return panel_case("BETTER", *PANELS["BETTER"], "TEXT-AHEAD", False)


@case
def crossed():
    return panel_case("CROSSED", make_curve(16.46), make_curve(19.539), "CROSSED")


@case
def delta_exactly_half_inside():
    """|Delta_1| exactly 0.5 (n50_T1 = 15.96 from a(15) 0.524, a(16) 0.499) -> inside the band -> PARITY; and -0.5 exactly."""
    a1 = dict(make_curve(15.5)); a1.update({15: 0.524, 16: 0.499, 17: 0.05})
    r = panel_case("HALF+", a1, make_curve(20.6), "PARITY")
    a1 = dict(make_curve(14.5)); a1.update({14: 0.524, 15: 0.499, 16: 0.05})
    return r + "; " + panel_case("HALF-", a1, make_curve(20.6), "PARITY")


@case
def marginal_latent():
    """Delta_2 -0.8 -> 'LATENT-AHEAD (marginal)'."""
    r = panel_case("MARG-L", make_curve(15.5), make_curve(19.739), "LATENT-AHEAD", True)
    chk(PANEL["MARG-L"]["outcome"] == "LATENT-AHEAD (marginal)", PANEL["MARG-L"])
    return r


@case
def marginal_text():
    r = panel_case("MARG-T", make_curve(15.5), make_curve(21.339), "TEXT-AHEAD", True)
    chk(PANEL["MARG-T"]["outcome"] == "TEXT-AHEAD (marginal)", PANEL["MARG-T"])
    return r


@case
def sentinel_gt24():
    """n50_T2 '> 24' -> Delta_2 = 24 - 20.539 + 0.01 (TEXT-AHEAD side)."""
    t = build("sent24", a1=make_curve(15.5), a2=make_curve(30))
    try:
        s = run(t); c = s["comparison"]
        chk(s["t2"]["n50"] == "> 24" and abs(c["DELTA"][2] - 3.471) < 1e-9 and outc(s) == "TEXT-AHEAD", (s["t2"]["n50"], c["DELTA"], outc(s)))
        return "n50 > 24 -> Δ2 %.3f -> %s" % (c["DELTA"][2], outc(s))
    finally:
        t.done()


@case
def base_parity_gain():
    """BASE-PARITY: Delta_0 0.30 -> the gain statistic. (a) a levels-PARITY curve with offset 0.3 at every round reads PARITY on gain; (b) Delta 0.30 / 0.75 /
    0.75 reads PARITY on gain while levels would read TEXT-AHEAD (both statistics reported)."""
    out = []
    for tag, x1, x2, lev in (("a", 15.76, 20.839, "PARITY"), ("b", 16.21, 21.289, "TEXT-AHEAD")):
        t = build("bp_" + tag, a0=make_curve(10.606), a1=make_curve(x1), a2=make_curve(x2))
        try:
            s = run(t); c = s["comparison"]
            chk(abs(c["DELTA"][0] - 0.30) < 0.003 and c["statistic"] == "gain" and s["base"]["BASE_PARITY"] == "gain", (c["DELTA"], c["statistic"]))
            chk(outc(s) == "PARITY" and c["BOTH_STATISTICS"]["levels"][0] == lev and c["BOTH_STATISTICS"]["gain"][0] == "PARITY", (outc(s), c["BOTH_STATISTICS"]))
            chk("statistic gain" in c["line"], "statistic not named on the line")
            out.append("%s: Δ %s levels %s, gain -> %s" % (tag, {k: S.fmt(v) for k, v in c["DELTA"].items()}, c["BOTH_STATISTICS"]["levels"][0], outc(s)))
        finally:
            t.done()
    t = build("bp_levels", a0=make_curve(10.556))                           # Delta_0 0.25 exactly-ish -> levels
    try:
        s = run(t); chk(s["comparison"]["statistic"] == "levels", s["comparison"]["DELTA"])
    finally:
        t.done()
    return "; ".join(out) + "; Δ0 0.25 -> levels"


@case
def text_not_absorbed_r1():
    """T4: T1 uncensored and not T-ABSORB -> TEXT-NOT-ABSORBED (no round-1 four-way label, every row False, never LATENT-AHEAD); T2 files present but not read."""
    a1 = dict(make_curve(15.5)); a1[12] = 0.90
    t = build("tna1", a1=a1)
    try:
        s = run(t); c = s["comparison"]
        chk(outc(s) == "TEXT-NOT-ABSORBED" and not any(c["rows"].values()) and c["ROUND_1_ONLY"] is None, (outc(s), c["rows"]))
        chk("LATENT-AHEAD" not in c["line"].split(" | ")[0] and "cannot be separated from a fault" in c["line"], c["line"][:300])
        chk(s["t1"]["status"] == "SCORED" and not s["t1"]["ROUND2_CONDITION"] and "not read" in (s["t2"].get("note") or ""), (s["t1"]["label"], s["t2"].get("note")))
        return c["line"].split(" | ")[0][:160]
    finally:
        t.done()


@case
def text_not_absorbed_r2():
    """Addendum 1 item 1: T2 fails T-ABSORB-2 -> TEXT-NOT-ABSORBED-2 (+ ROUND-1-ONLY), never LATENT-AHEAD-2."""
    a2 = dict(make_curve(20.6)); a2[16] = 0.90
    t = build("tna2", a2=a2)
    try:
        s = run(t); c = s["comparison"]
        chk(outc(s) == "TEXT-NOT-ABSORBED-2" and not any(c["rows"].values()) and c["ROUND_1_ONLY"] == "PARITY-1", (outc(s), c["rows"], c["ROUND_1_ONLY"]))
        chk("LATENT-AHEAD" not in c["line"].split(" | ")[0], c["line"][:200])
        chk(not s["t2"]["T_ABSORB_2"] and s["t2"]["MECHANISM"].startswith("UNLABELLED (not T-ABSORB-2"), s["t2"]["MECHANISM"])
        return c["line"].split(" | ")[0][:200]
    finally:
        t.done()


@case
def text_label_stall():
    """T9: GATE-T2 fails under T1 (keep_16 0.60) and under the base fallback (keep_16 0.645) -> TEXT-LABEL-STALL + PARITY-1; no T2 cell; never LATENT-AHEAD.
    Also a LATENT-AHEAD-1 stall carries FAIL-POS."""
    g2 = gate2_obj(keep2={13: 0.99, 14: 0.98, 15: 0.9, 16: 0.60}, fb={"W2": {13: 0.955, 14: 0.875, 15: 0.775, 16: 0.645}})
    t = build("stall", gate2=g2, t2=False)
    try:
        s = run(t); c = s["comparison"]
        chk(s["gate_t2"]["W2"]["status"] == "BOTH-FAIL", s["gate_t2"]["W2"]["status"])
        chk(outc(s) == "TEXT-LABEL-STALL" and c["ROUND_1_ONLY"] == "PARITY-1" and not any(c["rows"].values()), (outc(s), c["ROUND_1_ONLY"]))
        w = c["line"].split(" | ")[0]
        chk("output-format labeller limit" in w and "not a depth result" in w and "LATENT-AHEAD" not in w, w[:300])
        r = w[:120]
    finally:
        t.done()
    t = build("stall_la", a1=make_curve(14.46), gate2=g2, t2=False)
    try:
        s = run(t); c = s["comparison"]
        chk(c["ROUND_1_ONLY"] == "LATENT-AHEAD-1" and "FAIL-POS" in c["line"], (c["ROUND_1_ONLY"], c["line"][:300]))
    finally:
        t.done()
    return r + " ; LATENT-AHEAD-1 stall carries FAIL-POS"


@case
def base_labelled_window():
    """W2 fails under T1 but passes under the base fallback -> BASE-LABELLED, T2 read normally; T-SELF-2 names the labeller per window."""
    g2 = gate2_obj(keep2={16: 0.60}, fb={"W2": 0.80})
    t = build("baselab", gate2=g2, pm2=pool_meta(2, base_windows=("W2",)))
    try:
        s = run(t)
        chk(s["gate_t2"]["W2"]["status"] == "BASE-LABELLED" and s["gate_t2"]["W1"]["status"] == "SELF-LABELLED", (s["gate_t2"]["W2"]["status"],))
        chk(outc(s) == "PARITY" and "BASE-LABELLED" in (s["t2"]["WORDING"]["T_SELF_2"] or ""), (outc(s), s["t2"]["WORDING"]))
        return "W2 BASE-LABELLED; %s" % outc(s)
    finally:
        t.done()


def assert_censored(s, cell, reason, rnd):
    c = s["comparison"]; x = s[cell]
    chk(x["status"] == "FAIL" and x.get("censored"), (x["status"], x.get("label")))
    chk(reason in x["label"], (reason, x["label"][:300]))
    chk(outc(s) == "TEXT-CENSORED" and c["status"] == "CENSORED" and not any(c["rows"].values()) and c["ROUND_1_ONLY"] is None, (outc(s), c["rows"]))
    chk(("(r%d" % rnd) in c["line"], c["line"][:200])
    return x["label"][:150]


@case
def t1_dead():
    t = build("dead", log1={"last": 11900})
    try:
        s = run(t); r = assert_censored(s, "t1", "dead", 1)
        chk(not s["t1"]["T_SELF"] and not s["t1"]["T_DBS"] and s["t1"]["MECHANISM"] == "UNLABELLED (censored)", "readings must fail")
        return r
    finally:
        t.done()


@case
def t1_ft_diverged():
    a1 = dict(make_curve(15.5)); a1[5] = 0.98
    t = build("div", a1=a1)
    try:
        return assert_censored(run(t), "t1", "FT-DIVERGED-T", 1)
    finally:
        t.done()


@case
def label_noisy():
    out = []
    for tag, sd in (("pi", textself(1, pi=0.985)), ("piE", textself(1, piE=0.989)), ("ci", textself(1, pi=0.992, ci=(0.990, 0.9935)))):
        t = build("noisy_" + tag, sd1=sd)
        try:
            out.append(tag + ": " + assert_censored(run(t), "t1", "LABEL-NOISY-T", 1)[:40])
        finally:
            t.done()
    t = build("noisy_t2win", sd2=dict(textself(2), label_audit=dict(textself(2)["label_audit"], by_window={
        "W2": {"pi": 0.985, "pi_E": 0.99, "pi_ci95": [0.98, 0.99], "pi_E_ci95": [0.985, 0.995]}, "W1": textself(2)["label_audit"]["by_window"]["W1"]})))
    try:
        out.append("T2 window: " + assert_censored(run(t), "t2", "LABEL-NOISY-T", 2)[:40])
    finally:
        t.done()
    return "; ".join(out)


@case
def contaminated():
    out = []
    N = S.nested(); import numpy as np
    rng = np.random.default_rng(99999); recs = []
    for i in range(200):
        n = 9 + i % 4; r_, t_ = N.generate(rng, n); recs.append({"syms": N.render(r_), "n": n, "chain_label": t_, "values": [0] * n})
    test9 = next(it for it in S.test_items(24) if it["n"] == 9)
    for tag, kw, post in (("excl_hits", {"sd1": textself(1, cont={"pool_excl_hits": 1})}, None),
                          ("gold_key_hits", {"sd1": textself(1, cont={"gold_key_hits": 2})}, None),
                          ("self_reuse", {"log1": {"reuse": 1}}, None),
                          ("exhausted", {"pm1": pool_meta(1, n_records=383000)}, None),
                          ("recheck_test", {}, ("test", None)), ("recheck_goldkey", {}, ("gold", None)), ("recheck_dup", {}, ("dup", None))):
        t = build("cont_" + tag, **kw)
        try:
            if post:
                rr = copy.deepcopy(recs)
                if post[0] == "test":
                    rr[57] = {"syms": test9["syms"], "n": 9, "chain_label": 0, "values": [0] * 9}
                elif post[0] == "gold":
                    rr[11]["root"] = "x"
                else:
                    rr[90] = copy.deepcopy(rr[3])
                t.w(L["pool_t1"], "pool_t1.jsonl", "\n".join(json.dumps(x) for x in rr) + "\n")
            s = run(t); out.append(tag + ": " + assert_censored(s, "t1", "CONTAMINATED", 1)[:30])
        finally:
            t.done()
    t = build("cont_clean_pool")
    try:
        t.w(L["pool_t1"], "pool_t1.jsonl", "\n".join(json.dumps(x) for x in recs) + "\n")
        s = run(t); chk(s["t1"]["status"] == "SCORED" and not s["t1"]["CONTAMINATED"], s["t1"]["label"])
        chk(s["t1"]["CONTAMINATION"]["recheck"]["consumed_read"] == 200, s["t1"]["CONTAMINATION"])
        tt = s["comparison"]["descriptive"]["TRAIN_TOKENS_r1"]["text"]; chk("exact" in tt["self_method"], tt["self_method"])
    finally:
        t.done()
    return "; ".join(out) + "; clean pool re-check passes (TRAIN-TOKENS self exact)"


@case
def seal_breach():
    """T3: a trainer-written eval.json / eval_12k.json / *.void_unsealed in the T1 dir -> SEAL-BREACH (T1 void, comparison TEXT-CENSORED r1); a later-stage
    output before T1's eval_text while pending -> SEAL_WATCH; a time-field order violation -> SEAL-BREACH."""
    out = []
    for tag, f in (("eval", "eval.json"), ("eval12k", "eval_12k.json"), ("void", "eval.json.void_unsealed")):
        t = build("seal_" + tag)
        try:
            t.w(L["t1"], f, {"acc_by_n": {"13": 0.5}})
            s = run(t); out.append(tag + ": " + assert_censored(s, "t1", "SEAL-BREACH", 1)[:30])
            chk(f in s["t1"]["SEAL"]["trainer_eval_files"], s["t1"]["SEAL"])
        finally:
            t.done()
    t = build("seal_watch")
    try:
        t.rm(L["t1"], "eval_text.json"); s = run(t)
        chk(s["t1"]["status"] == "PENDING" and s["t1"]["SEAL_WATCH"] and "SEAL_WATCH" in s["t1"]["label"], s["t1"]["label"][:300])
        s = run(t, final=True); chk(s["t1"]["status"] == "FAIL" and "SEAL-BREACH" in s["t1"]["label"], s["t1"]["label"][:300])
        out.append("watch: pending + SEAL_WATCH; --final FAIL")
    finally:
        t.done()
    g2 = gate2_obj(t_start="2026-10-20T11:00:00-04:00")
    t = build("seal_order", gate2=g2, t1_times=(("t_start", "2026-10-20T09:00:00-04:00"), ("t_end", "2026-10-20T12:00:00-04:00")))
    try:
        s = run(t); out.append("order: " + assert_censored(s, "t1", "SEAL-BREACH", 1)[:30])
        chk(s["t1"]["SEAL"]["order_violations"], s["t1"]["SEAL"])
    finally:
        t.done()
    g2 = gate2_obj(t_start="2026-10-20T13:00:00-04:00")
    t = build("seal_order_ok", gate2=g2, t1_times=(("t_start", "2026-10-20T09:00:00-04:00"), ("t_end", "2026-10-20T12:00:00-04:00")))
    try:
        s = run(t); chk(s["t1"]["status"] == "SCORED" and not s["t1"]["SEAL_BREACH"], s["t1"]["label"])
    finally:
        t.done()
    return "; ".join(out) + "; order ok passes"


@case
def args_diff_rules():
    """T1 / V2 / 216-7 / Addendum 1 item 3: each mutation of the realistic args reads ARGS_BAD; the realistic (AST-built) args and an equal-content args_ref
    read ok; smoke mode frees stop_at / mem_check_self only."""
    t = build("args_unit", t2=False, g2=False)
    try:
        S.set_root(t.R, t.root)
        t.wroot("scratch/ref_bad.json", dict(BASE_ARGS, bs=32)); t.wroot("scratch/ref_meta.json", dict(BASE_ARGS, out="elsewhere", device="cuda:7"))
        ok = S.args_diff("s1", "t1", realistic_args("t1"))
        chk(ok["ok"], ok)
        chk(S.args_diff("s1", "t2", realistic_args("t2"))["ok"], "t2 realistic")
        muts = {"lr 1e-4": {"lr": 1e-4}, "warmup 1000 (neutral off default)": {"warmup": 1000}, "no_final_eval false": {"no_final_eval": False},
                "dropped base key bs": {"bs": "<drop>"}, "args_ref one changed key": {"args_ref": "scratch/ref_bad.json"},
                "args_ref missing file": {"args_ref": "scratch/nope.json"}, "trainer cot_eval_bs 32": {"cot_eval_bs": 32},
                "new unknown key": {"self_kmax": 16}, "neutral key missing": {"fast_add": "<drop>"}, "init_from wrong": {"init_from": "boxes/runs_fsq/x/model.pt"},
                "self_micro 4 in T1": {"self_micro": 4}, "stop_at 2 (registered run)": {"stop_at": 2}, "p_self 0.4": {"p_self": 0.4},
                "no_final_eval as 1": {"no_final_eval": 1}, "eval_at 12000": {"eval_at": 12000}, "probe_loss_every 100": {"probe_loss_every": 100},
                "arm inject": {"arm": "inject"}, "draws_log_steps 20000": {"draws_log_steps": 20000}, "dropped FT key steps": {"steps": "<drop>"}}
        bad = {k: S.args_diff("s1", "t1", realistic_args("t1", **m))["ok"] for k, m in muts.items()}
        chk(not any(bad.values()), {k: v for k, v in bad.items() if v})
        chk(S.args_diff("s1", "t1", realistic_args("t1", args_ref="scratch/ref_meta.json"))["ok"], "args_ref equal on non-META")
        chk(S.args_diff("s1", "t1", realistic_args("t1", args_ref=""))["ok"], "empty args_ref")
        sm = realistic_args("t1", stop_at=2, ckpt_every=1, mem_check_self=True, args_ref="")
        chk(S.args_diff("s1", "t1", sm, smoke=True)["ok"] and not S.args_diff("s1", "t1", sm)["ok"], "smoke mode")
        chk(not S.args_diff("s1", "t1", realistic_args("t1", lr=1e-4), smoke=True)["ok"], "smoke still checks lr")
        chk(S.args_diff("s1", "t1", realistic_args("t1", hostprobe_sps=None, excl_survive_by_n={}, train_driver="x"))["ok"], "META_T info keys")
    finally:
        t.done()
    t = build("args_e2e", args1=realistic_args("t1", lr=1e-4))
    try:
        r = assert_censored(run(t), "t1", "ARGS-DIFF", 1)
    finally:
        t.done()
    return "%d mutations ARGS_BAD; realistic / equal args_ref / smoke ok; e2e: %s" % (len(muts), r[:40])


@case
def eval_bs_not_32():
    """Addendum 2 item 1: eval_text at cot_eval_bs != 32 -> VOID (§5.2: meta mismatch -> every reading of that model fails), not pending; T1 and the base."""
    t = build("bs_t1", t1_bs=64)
    try:
        s = run(t); r = assert_censored(s, "t1", "cot_eval_bs", 1)
        chk(s["t1"]["label"].startswith("VOID"), s["t1"]["label"])
    finally:
        t.done()
    t = build("bs_base", base_bs=64)
    try:
        s = run(t); chk(s["base"]["status"] == "FAIL" and "cot_eval_bs" in s["base"]["label"], s["base"]["label"])
        chk(outc(s) == "TEXT-CENSORED" and "(r0" in s["comparison"]["line"], s["comparison"]["line"][:200])
        chk(s["t1"]["MECHANISM"].startswith("UNLABELLED (bars void"), s["t1"]["MECHANISM"])
    finally:
        t.done()
    return r[:60] + "; base bs 64 -> base VOID, TEXT-CENSORED (r0), T1 bars void"


@case
def items_sha_mismatch():
    t = build("sha", t1_sha="f" * 64)
    try:
        s = run(t); return assert_censored(s, "t1", "items_sha256", 1)[:80]
    finally:
        t.done()


@case
def model_sha_rules():
    """an FT eval_text whose model_sha256 is the base's -> VOID; the base's must be BASE_SHA."""
    t = build("msha")
    try:
        t.w(L["t1"], "eval_text.json", eval_text_obj(make_curve(15.5), S.BASE_SHA)); s = run(t)
        r = assert_censored(s, "t1", "model_sha256", 1)
        t.w(L["base"], "eval_text.json", eval_text_obj(A0, SHA_T1)); s = run(t)
        chk(s["base"]["status"] == "FAIL" and "model_sha256" in s["base"]["label"], s["base"]["label"])
        return r[:60] + "; base with a foreign sha -> VOID"
    finally:
        t.done()


@case
def ctx_overflow():
    """CTX-OVERFLOW counts as a failure in a_T(24), CTX-FEASIBLE beside (971 feasible items, all correct -> a 0.971, feasible 1.000); an overflow item with
    an answer or a wrong overflow flag -> VOID."""
    a2 = dict(make_curve(30))
    t = build("ctx", a2=a2)
    try:
        s = run(t); x = s["t2"]
        chk(abs(x["a"][24] - 0.971) < 1e-12 and abs(x["EVAL"]["a24_feasible"] - 1.0) < 1e-12 and abs(x["EVAL"]["ctx_feasible_share"][24] - 0.971) < 1e-12,
            (x["a"][24], x["EVAL"]["a24_feasible"]))
        chk(x["M_far_T"] < x["M_far_T_feasible"] and abs(x["M_far_T_feasible"] - 1.0) < 1e-12, (x["M_far_T"], x["M_far_T_feasible"]))
        chk("n 24: 0.971 feasible" in s["comparison"]["line"], s["comparison"]["line"][-120:])
        r = "a(24) %.3f, feasible %.3f, M_far %.4f (feasible %.3f)" % (x["a"][24], x["EVAL"]["a24_feasible"], x["M_far_T"], x["M_far_T_feasible"])
    finally:
        t.done()
    for tag in ("overflow_answer", "overflow_flag"):
        t = build("ctx_" + tag, t1_tweak=tag)
        try:
            s = run(t); chk(s["t1"]["status"] == "FAIL" and s["t1"]["label"].startswith("VOID") and "items" in s["t1"]["label"], s["t1"]["label"][:200])
        finally:
            t.done()
    return r + "; overflow answer / flag -> VOID"


T1_REG = [(L["t1"], "log.jsonl"), (L["t1"], "args.json"), (L["t1"], "draws_log.json"), (L["t1"], "eval_text.json"), (L["t1"], "eval_textself.json"),
          (L["pool_t1"], "pool_meta.json"), (L["gate_t1"], "eval_gate.json"), (L["base"], "eval_text.json")]
T2_REG = [(L["t2"], "log.jsonl"), (L["t2"], "args.json"), (L["t2"], "draws_log.json"), (L["t2"], "eval_text.json"), (L["t2"], "eval_textself.json"),
          (L["pool_t2"], "pool_meta.json"), (L["gate_t2"], "eval_gate.json"), (L["base"], "eval_text.json"), (L["t1"], "eval_text.json")]


@case
def registered_files_missing():
    """215-7: each registered file missing -> the cell reads pending naming it (and the comparison pending); --final -> failure (rule b) and the comparison
    TEXT-CENSORED. Also the CPU parts of eval_textself and the log's cursor fields."""
    tpl = build("miss_tpl"); n = 0
    try:
        for cell, reg in (("t1", T1_REG), ("t2", T2_REG)):
            for d, f in reg:
                t = Tree("miss_one"); shutil.rmtree(t.root); shutil.copytree(tpl.root, t.root)
                try:
                    t.rm(d, f); s = run(t); x = s[cell]
                    chk(x["status"] == "PENDING" and any(("%s/%s" % (d, f)) in m for m in x["missing"]), (cell, d, f, x["status"], x.get("missing")))
                    chk(s["comparison"]["status"] == "PENDING", (cell, d, f, s["comparison"]["outcome"]))
                    s = run(t, final=True); x = s[cell]
                    chk(x["status"] == "FAIL" and "missing at --final" in x["label"], (cell, d, f, x["label"][:200]))
                    chk(outc(s) == "TEXT-CENSORED" and not any(s["comparison"]["rows"].values()), (cell, d, f, outc(s)))
                    n += 1
                finally:
                    t.done()
        for tag, cell, mut in (("label_audit", "t1", lambda t: t.w(L["t1"], "eval_textself.json", textself(1, drop="label_audit"))),
                               ("contamination", "t1", lambda t: t.w(L["t1"], "eval_textself.json", textself(1, drop="contamination"))),
                               ("by_window", "t2", lambda t: t.w(L["t2"], "eval_textself.json", dict(textself(2), label_audit=textself(1)["label_audit"]))),
                               ("cursor", "t1", lambda t: write_log(t, L["t1"], drop_cursor=True)),
                               ("gate_undecided", "t1", lambda t: t.w(L["gate_t1"], "eval_gate.json", gate1_obj(flag=None))),
                               ("pool_card", "t1", lambda t: t.w(L["pool_t1"], "pool_meta.json", dict(pool_meta(1), card=None))),
                               ("unreadable", "t1", lambda t: t.w(L["t1"], "args.json", "{\"lr\": "))):
            t = Tree("miss_part"); shutil.rmtree(t.root); shutil.copytree(tpl.root, t.root)
            try:
                mut(t); s = run(t); chk(s[cell]["status"] == "PENDING", (tag, s[cell]["label"][:200]))
                s = run(t, final=True); chk(s[cell]["status"] == "FAIL" and outc(s) == "TEXT-CENSORED", (tag, s[cell]["label"][:200], outc(s)))
                n += 1
            finally:
                t.done()
    finally:
        tpl.done()
    return "%d registered files / parts: pending, then --final failure + TEXT-CENSORED" % n


@case
def degenerate_bars():
    """v1 §4.3: SB2 >= MG2 -> UNLABELLED (bars degenerate) for T2."""
    t = build("degen", a0=make_curve(12.9), a1=make_curve(13.1), a2=make_curve(19.0))
    try:
        s = run(t); b = s["t2"]["BARS"]
        chk(b["SHRINK_below"] >= b["MARGIN_GROWS_2"] and s["t2"]["MECHANISM"] == "UNLABELLED (bars degenerate)", (b, s["t2"]["MECHANISM"]))
        return "SB2 %s >= MG2 %s -> %s" % (b["SHRINK_below"], b["MARGIN_GROWS_2"], s["t2"]["MECHANISM"])
    finally:
        t.done()


@case
def t_full_needs_m_far():
    """T2's T-FULL needs M2 >= max(0.80, MG2) and M_far >= 0.60: M_far 0.55 -> T-GROW, 0.6375 -> T-FULL."""
    out = []
    for far, exp in (((0.6, 0.55, 0.5, 0.55), "T-GROW"), ((0.7, 0.65, 0.6, 0.6), "T-FULL")):
        a2 = {**{n: 1.0 for n in range(1, 21)}, **dict(zip(range(21, 25), far))}
        t = build("tfull_%s" % exp, a2=q3(a2))
        try:
            s = run(t); chk(s["t2"]["MECHANISM"] == exp, (s["t2"]["MECHANISM"], s["t2"]["M_far_T"], s["t2"]["BARS"]["FULL_M2"]))
            out.append("M_far %.4f -> %s" % (s["t2"]["M_far_T"], exp))
        finally:
            t.done()
    return "; ".join(out)


@case
def mechanism_r1_labels():
    """T1 mechanism on the bf16-base bars (MG_T 0.665 / SB_T 0.435 / FULL 0.80): M_T 0.9 / 0.7 / 0.5 / 0.3 -> T-FULL / T-GROW / T-SHIFT/MIXED / T-SHRINK."""
    out = []
    for m, exp in ((0.9, "T-FULL"), (0.7, "T-GROW"), (0.5, "T-SHIFT/MIXED"), (0.3, "T-SHRINK")):
        a1 = q3({**{n: 1.0 for n in range(1, 13)}, **{n: m for n in range(13, 17)}, **{n: 0.05 for n in range(17, 25)}})
        t = build("mech_%s" % m, a1=a1, t2=False, g2=False)
        try:
            s = run(t); b = s["t1"]["BARS"]
            chk(b["MARGIN_GROWS"] == 0.665 and b["SHRINK_below"] == 0.435 and s["t1"]["MECHANISM"] == exp, (b["MARGIN_GROWS"], b["SHRINK_below"], s["t1"]["MECHANISM"]))
            chk(s["t1"]["T_DBS"] == (m >= 0.8), s["t1"]["T_DBS"])
            out.append("%s->%s" % (m, exp))
        finally:
            t.done()
    return "; ".join(out)


@case
def t_retain_and_wording():
    """T-RETAIN fails (a(10) 0.98 in T2) -> T-SELF-2 false, comparison still four-way (T-RETAIN is not a censor) with the note; T-SELF / T-DBS wording."""
    a2 = dict(make_curve(20.6)); a2[10] = 0.98
    t = build("retain", a2=a2)
    try:
        s = run(t)
        chk(not s["t2"]["T_RETAIN"] and not s["t2"]["T_SELF_2"] and s["t2"]["T_ABSORB_2"], (s["t2"]["T_RETAIN"], s["t2"]["T_SELF_2"]))
        chk(outc(s) == "PARITY" and "T-RETAIN failed" in s["comparison"]["line"], s["comparison"]["line"][:300])
        w = s["t1"]["WORDING"]["T_SELF"]
        chk(w.startswith("text-CoT depth extended from 8 to 12") and "not attributed: no text FT-CTRL" in w and "π = 0.99950" in w, w)
        return "T-RETAIN fail noted; T-SELF wording ok"
    finally:
        t.done()


@case
def gate_t1_fail_not_run():
    """GATE-T1 fails (keep_12 0.60) -> NOT RUN (keep / π per failing n); every comparison row fails."""
    t = build("g1fail", gate1=gate1_obj(keep={12: 0.60}), t1=False, t2=False, g2=False)
    try:
        s = run(t); c = s["comparison"]
        chk(outc(s) == "NOT RUN" and not any(c["rows"].values()) and "GATE-T1" in c["line"] and "12" in c["line"].split(" | ")[0], (outc(s), c["line"][:200]))
        s = run(t, final=True); chk(outc(s) == "NOT RUN", outc(s))
        return c["line"].split(" | ")[0][:120]
    finally:
        t.done()


@case
def gate_flag_inconsistent():
    """a GATE-T1 file flag True that the by_n recompute fails -> not PASS (fail)."""
    t = build("g1incons", gate1=gate1_obj(keep={12: 0.60}, flag=True), t1=False, t2=False, g2=False)
    try:
        s = run(t); chk(s["gate_t1"]["status"] == "FAIL" and not s["gate_t1"]["consistent"] and outc(s) == "NOT RUN", (s["gate_t1"]["status"], outc(s)))
        return "flag True vs recomputed False -> FAIL / NOT RUN"
    finally:
        t.done()


@case
def not_admitted():
    t = build("na", t2=False)
    try:
        s = run(t, na=("s219_t1_s1",)); chk(s["t1"]["status"] == "UNREAD" and outc(s) == "TEXT-CENSORED" and "not admitted" in s["comparison"]["line"], outc(s))
        s = run(t, na=("s219_t2_s1",)); chk(s["t2"]["status"] == "UNREAD" and outc(s) == "TEXT-CENSORED" and "(r2, not admitted" in s["comparison"]["line"], outc(s))
        return "T1 / T2 not admitted -> UNREAD, TEXT-CENSORED (not admitted)"
    finally:
        t.done()


@case
def card_gate_path():
    """T5 / G-T21: a T1 eval_text on the 2070 without CARD-GATE -> VOID; with a passing CARD-GATE (|Δa| <= 0.005, |Δn50| <= 0.02) -> read; with a failing
    one (0.01 at n 10) -> VOID; a round's gate and pool on different cards -> censored; a two-card file -> VOID."""
    out = []
    t = build("cg_none", t1_card=CARD2070)
    try:
        s = run(t); out.append("no gate: " + assert_censored(s, "t1", "card", 1)[:20])
    finally:
        t.done()
    t = build("cg_pass", t1_card=CARD2070)
    try:
        a = dict(A0); a[20] = 0.045
        t.w(L["base"], "eval_text_2070.json", eval_text_obj(q3(a), S.BASE_SHA, card=CARD2070)); s = run(t)
        chk(s["card_gate"]["PASS"] and s["t1"]["status"] == "SCORED" and outc(s) == "PARITY", (s["card_gate"].get("label"), s["t1"]["label"][:100]))
        chk(s["card_gate"]["item_flips"].get(20), s["card_gate"])
        out.append("pass: T1 on the 2070 read (%s)" % outc(s))
        t.w(L["pool_t1"], "pool_meta.json", pool_meta(1, card=CARD2070)); s = run(t)
        out.append("gate 3070 / pool 2070: " + assert_censored(s, "t1", "one allowed card", 1)[:20])
    finally:
        t.done()
    t = build("cg_fail", t1_card=CARD2070)
    try:
        a = dict(A0); a[10] = round(A0[10] + 0.01, 3)
        t.w(L["base"], "eval_text_2070.json", eval_text_obj(q3(a), S.BASE_SHA, card=CARD2070)); s = run(t)
        chk(not s["card_gate"]["PASS"], s["card_gate"]); out.append("fail: " + assert_censored(s, "t1", "card", 1)[:20])
    finally:
        t.done()
    t = build("cg_twocard", t1_top_card=CARD2070)
    try:
        s = run(t); out.append("two cards: " + assert_censored(s, "t1", "one_card", 1)[:20])
    finally:
        t.done()
    return "; ".join(out)


@case
def card_fallback_4060():
    """§2.7 (3070 unrecoverable): CARD_CLASS 4060 without recorded 4060 latent comparators -> the comparison is pending (never a pass); with them -> read."""
    t = build("c4060", base_card="NVIDIA GeForce RTX 4060", t1_card="NVIDIA GeForce RTX 4060", t2_card="NVIDIA GeForce RTX 4060",
              gate1=gate1_obj(card="NVIDIA GeForce RTX 4060"), pm1=pool_meta(1, card="NVIDIA GeForce RTX 4060"),
              gate2=gate2_obj(card="NVIDIA GeForce RTX 4060"), pm2=pool_meta(2, card="NVIDIA GeForce RTX 4060"))
    old = S.CARD_CLASS
    try:
        S.CARD_CLASS = "4060"; s = run(t)
        chk(s["comparison"]["status"] == "PENDING" and "4060" in s["comparison"]["line"], s["comparison"]["line"][:200])
        S.LATENT_N50_BY_CARD["4060"] = dict(S.LATENT_N50); s = run(t); chk(outc(s) == "PARITY", outc(s))
        return "4060 pending until its comparators are recorded; then %s" % outc(s)
    finally:
        S.CARD_CLASS = old; S.LATENT_N50_BY_CARD.pop("4060", None); t.done()


@case
def latent_files_ignored():
    """Addendum 2 item 4: decisions use the constants only - a tree holding altered latent eval files reads the same outcome and Δ as one without them."""
    t = build("latign")
    try:
        s0 = run(t)
        for r, d in L["latent"].items():
            t.w(d, "eval_vb.json", {"acc_n2": {str(n): 1.0 for n in range(1, 17)}, "acc_k": {"24": {str(n): 1.0 for n in range(1, 17)}}, "card": CARD3070})
            t.w(d, "eval_deep.json", {"acc_n2": {str(n): 0.9 for n in range(13, 25)}, "acc_k": {"24": {str(n): 0.9 for n in range(13, 25)},
                                                                                                "26": {str(n): 0.9 for n in range(13, 25)}}, "items": {}})
        s1 = run(t)
        chk(outc(s0) == outc(s1) and s0["comparison"]["DELTA"] == s1["comparison"]["DELTA"], (outc(s0), outc(s1)))
        chk(s1["comparison"]["descriptive"]["LATENT_FIXED_K"]["r0"]["n50"] == "> 24", s1["comparison"]["descriptive"]["LATENT_FIXED_K"])
        return "outcome %s and Δ unchanged; only descriptive LATENT-FIXED-K' moved" % outc(s1)
    finally:
        t.done()


@case
def layout_list_items():
    """eval_text items in the columnar alternative read identically to the writer's list layout."""
    t = build("layout", layout="columnar")
    try:
        s = run(t); chk(s["t1"]["status"] == "SCORED" and outc(s) == "PARITY", s["t1"]["label"][:200])
        return "columnar layout: %s, n50 %s" % (outc(s), S.fmt(s["t1"]["n50"]))
    finally:
        t.done()


@case
def base_note_check():
    """G-T19: BASE_NOTE's branch is checked against the scorer's own (a mismatch is printed on the base label)."""
    t = build("note", t1=False, t2=False, g2=False)
    try:
        S.BASE_NOTE = {"note": "219-x", "branch": "gain"}; s = run(t)
        chk(s["base"]["BASE_NOTE_MATCHES"] is False and "MISMATCH" in s["base"]["label"], s["base"]["label"])
        S.BASE_NOTE = {"note": "219-x", "branch": "levels"}; s = run(t); chk(s["base"]["BASE_NOTE_MATCHES"] is True, s["base"]["label"])
        chk(s["base"]["BF16_VS_FP32"]["max_abs"] == 0.0 and "batch composition also differs" in s["base"]["BF16_VS_FP32"]["note"], s["base"]["BF16_VS_FP32"])
        return "mismatch flagged; match ok; BF16-VS-FP32 read"
    finally:
        S.BASE_NOTE = None; t.done()


@case
def final_undecided_gates():
    """--final: an undecided GATE-T1 (no T1) -> TEXT-CENSORED (not NOT RUN); an undecided GATE-T2 window -> TEXT-CENSORED (never TEXT-LABEL-STALL)."""
    t = build("undec", gate1=gate1_obj(flag=None), t1=False, t2=False, g2=False)
    try:
        s = run(t); chk(s["comparison"]["status"] == "PENDING", outc(s))
        s = run(t, final=True); chk(outc(s) == "TEXT-CENSORED", outc(s))
    finally:
        t.done()
    t = build("undec2", gate2=gate2_obj(flags={"GATE_T2": None}), t2=False)
    try:
        s = run(t); chk(s["comparison"]["status"] == "PENDING", outc(s))
        s = run(t, final=True); chk(s["gate_t2"]["W2"]["status"] == "UNDECIDED" and outc(s) == "TEXT-CENSORED" and s["comparison"]["ROUND_1_ONLY"] is None,
                                    (s["gate_t2"]["W2"]["status"], outc(s)))
    finally:
        t.done()
    return "pending, then failure at --final"


@case
def descriptive_rows():
    """DEPTH-PAIRED / TOKEN-COST / TRAIN-TOKENS / RATIO / FAIL-POS / LABELLER-REACH / LATENT-FIXED-K' with the real latent files copied read-only into the tree."""
    t = build("desc", a1=PANELS["N-SHIFT"][0], a2=PANELS["N-SHIFT"][1])
    try:
        for r, d in L["latent"].items():
            for f in ("eval_vb.json", "eval_deep.json", "eval_selfdepth.json", "eval_chainlabel_fresh.json", "draws_log.json", "args.json"):
                src = os.path.join(REAL_R, d, f)
                if os.path.exists(src):
                    os.makedirs(os.path.join(t.R, d), exist_ok=True); shutil.copyfile(src, os.path.join(t.R, d, f))
        s = run(t); D = s["comparison"]["descriptive"]
        chk(D["DEPTH_PAIRED_r1"][13]["both"] + D["DEPTH_PAIRED_r1"][13]["text_only"] + D["DEPTH_PAIRED_r1"][13]["latent_only"] + D["DEPTH_PAIRED_r1"][13]["neither"] == 1000, D["DEPTH_PAIRED_r1"])
        chk(set(D["DEPTH_PAIRED_r2"]) == {17, 18, 19, 20} and 0 <= D["DEPTH_PAIRED_r2"][20]["mcnemar_p"] <= 1, D["DEPTH_PAIRED_r2"])
        tc = D["TOKEN_COST_r2"][17]; chk(tc["latent_correct"] == 998 and tc["latent_steps_per_correct"] == 19 * 1000 / 998, tc)
        tt = D["TRAIN_TOKENS_r2"]; chk(tt["latent"]["items"] == 768000 and tt["text"]["self_items"] == 383488, (tt["latent"]["items"], tt["text"]["self_items"]))
        chk(abs(D["RATIO"]["latent"][1] - 15.46 / 12) < 1e-12 and D["FAIL_POS_r2"]["T_train"] == 531, (D["RATIO"], D["FAIL_POS_r2"]["T_train"]))
        lf = D["LATENT_FIXED_K"]; chk(all(num(lf[r]["n50"]) for r in lf), lf)
        lr = D["LABELLER_REACH"]; chk(lr["T1_vs_FTSELF"]["latent_keep"] and lr["T1_vs_FTSELF"]["text_keep"], lr)
        chk(outc(s) == "LATENT-AHEAD" and "FAIL-POS" in s["comparison"]["line"], outc(s))
        return ("LATENT-FIXED-K' n50 %s; TRAIN-TOKENS r2 text %.3g / latent %.3g targets" %
                ({r: S.fmt(lf[r]["n50"]) for r in lf}, tt["text"]["target_tokens"], tt["latent"]["target_tokens"]))
    finally:
        t.done()


@case
def writer_seal_labeller_kept_meta():
    """3b0bb06f: LABELLER-KEPT-T's start time is labeller_kept_meta.t_start; T1's n 9-24 labelling started before T1's eval_text ended -> SEAL-BREACH;
    the same start after the end -> clean."""
    t = build("lkm_bad", sd1=textself(1, lk_t_start="2026-10-20T11:30:00+00:00"))
    try:
        s = run(t); r = assert_censored(s, "t1", "SEAL-BREACH", 1)
        chk(any("labeller_kept" in v for v in s["t1"]["SEAL"]["order_violations"]), s["t1"]["SEAL"])
    finally:
        t.done()
    t = build("lkm_ok", sd1=textself(1, lk_t_start="2026-10-20T12:30:00+00:00"))
    try:
        s = run(t); chk(s["t1"]["status"] == "SCORED" and not s["t1"]["SEAL"]["order_violations"], s["t1"]["SEAL"])
    finally:
        t.done()
    return r[:40] + "; later start clean"


@case
def writer_done_markers_no_breach():
    """note 219-3 item 1: train_done.json / mem_check_self.json / timing.json / init_from.json / args_diff.json / mem_check.json and a leftover
    eval_text.partial.json / *.tmp are not SEAL-BREACH; an eval.json beside them is."""
    t = build("markers")
    try:
        t.w(L["t1"], "eval_text.partial.json", {"partial_through_n": 3}); t.w(L["t1"], "eval_textself.json.tmp", {})
        s = run(t); chk(s["t1"]["status"] == "SCORED" and not s["t1"]["SEAL_BREACH"] and outc(s) == "PARITY", (s["t1"]["label"][:150], s["t1"]["SEAL"]))
        chk(s["t1"]["TRAIN_DONE"]["consistent"], s["t1"]["TRAIN_DONE"])
        t.w(L["t1"], "eval.json", {}); s = run(t); assert_censored(s, "t1", "SEAL-BREACH", 1)
        return "markers clean; eval.json beside them -> SEAL-BREACH"
    finally:
        t.done()


@case
def writer_train_tokens_exact():
    """note 219-3 item 4 / textself_ft A8-A9: TRAIN-TOKENS' text side is the exact per-type counts of train_done.json, else the last logged train_tokens
    (equal here); the line carries them; train_done / log mismatches and a train_done model sha differing from eval_text's are caught."""
    t = build("ttok")
    try:
        s = run(t); D = s["comparison"]["descriptive"]; tt = D["TRAIN_TOKENS_r2"]["text"]; cum = s["t2"]["TRAIN_DONE"]
        exp = 5992 * TOK_SELF[0] + (12000 - 5992) * TOK_PLAIN[0]
        chk(tt["target_tokens"] == exp and tt["self_method"] == "train_done.json (exact)" and tt["self_padded_positions"] == 5992 * TOK_SELF[2], tt)
        chk("train_done.json (exact)" in s["comparison"]["line"], s["comparison"]["line"][-400:])
        t.rm(L["t1"], "train_done.json"); s = run(t); tt1 = s["comparison"]["descriptive"]["TRAIN_TOKENS_r1"]["text"]
        chk(tt1["target_tokens"] == exp and tt1["self_method"].startswith("log.jsonl step 12000"), tt1)
        chk(s["t1"]["TRAIN_DONE"]["present"] is False and s["t1"]["status"] == "SCORED", s["t1"]["TRAIN_DONE"])
    finally:
        t.done()
    t = build("ttok_mismatch", log1={"td_mod": {"n_self_steps": 5991}})
    try:
        s = run(t); chk(s["t1"]["TRAIN_DONE"]["consistent"] is False and s["t1"]["status"] == "SCORED", s["t1"]["TRAIN_DONE"])
    finally:
        t.done()
    t = build("ttok_sha", log1={"td_mod": {"model_sha256": "e" * 64}})
    try:
        s = run(t); r = assert_censored(s, "t1", "model_sha256_train_done", 1)
    finally:
        t.done()
    return "text %.4g targets from train_done (= log); mismatch flagged; foreign train_done sha -> %s" % (exp, r[:30])


@case
def writer_nonfinite_resume_records():
    """textself_ft's {step, nonfinite: 1} and {step, resumed} records are not dead and do not hide the last cursor."""
    t = build("nonfin", log1={"nonfinite_at": 6000})
    try:
        lp = t.p(L["t1"], "log.jsonl"); lines = open(lp).read().splitlines(); lines.insert(50, json.dumps({"step": 5000, "resumed": "ckpt_last.pt"}))
        t.w(L["t1"], "log.jsonl", "\n".join(lines) + "\n"); s = run(t)
        chk(s["t1"]["status"] == "SCORED" and not s["t1"]["dead"] and s["t1"]["nonfinite_records"] == 1 and s["t1"]["CONTAMINATION"]["cursor"] == 383488,
            (s["t1"]["label"][:120], s["t1"]["nonfinite_records"]))
        return "nonfinite + resumed records read; cursor 383,488"
    finally:
        t.done()


@case
def writer_gate_descriptive():
    """equiv / labeller_full in the writer's layouts are summarised (descriptive, never gating): an EQUIV window failing does not move any reading."""
    g1 = gate1_obj(); g1["equiv"]["W1"] = equiv_obj([9, 10, 11, 12], "W1", share=0.98)
    t = build("gdesc", gate1=g1)
    try:
        s = run(t); e = s["gate_t1"]["descriptive"]["EQUIV"]["W1"]; lf = s["gate_t1"]["descriptive"]["LABELLER_FULL"]
        chk(e["PASS"] is False and abs(e["min_share"] - 0.98) < 1e-12 and e["items"] == 1000 and lf["9"]["window"] == "W1" and lf["12"]["pi_E"] == 0.95, (e, lf))
        e2 = s["gate_t2"]["descriptive"]["EQUIV"]; chk(e2["W2"]["PASS"] and e2["W1"]["PASS"], e2)
        chk(outc(s) == "PARITY" and s["t1"]["status"] == "SCORED", outc(s))
        chk(s["t1"]["GATE_DESCRIPTIVE"]["EQUIV"]["W1"]["PASS"] is False, s["t1"]["GATE_DESCRIPTIVE"])
        return "EQUIV W1 share 0.98 summarised (PASS False), readings unchanged"
    finally:
        t.done()


@case
def writer_g_t4_rule():
    """note 219-3 item 2: bitwise, or every tensor bitwise except wte with max |Δ wte| <= 1e-6; the real G-T4 output (scratch/gate_s219, read-only)."""
    chk(S.g_t4_rule({})["PASS"] and S.g_t4_rule({}, single_thread_bitwise=True)["PASS"], "bitwise")
    chk(S.g_t4_rule({"p.wte_weight": 7.45e-9})["PASS"] and S.g_t4_rule({"p.wte_weight": 1e-6})["PASS"], "wte within 1e-6")
    chk(not S.g_t4_rule({"p.wte_weight": 1.2e-6})["PASS"] and not S.g_t4_rule({"p.h.0.attn.c_attn.weight": 1e-12})["PASS"], "over / other tensor")
    chk(S.g_t4_rule({"p.wte_weight": 0.0, "p.ln_f.weight": 0})["PASS"], "zero entries")
    real = rd_real("scratch/gate_s219/t4_init_resume.json"); r = S.g_t4_rule(real["ckpt3"]["model_diff"])
    chk(r["PASS"] and r["reading"] == "bitwise" and real["step3"]["bitwise"], (r, real["step3"]))
    tol = rd_real("scratch/gate_s219/t4_init_resume.tolerance_run.json"); r2 = S.g_t4_rule(tol["ckpt3"]["model_diff"])
    chk(r2["PASS"] and 0 < r2["max_wte"] <= 1e-6, r2)
    return "rule cases; real G-T4 (single-threaded) bitwise; tolerance run max |Δ wte| %.3g -> PASS" % r2["max_wte"]


@case
def writer_real_smoke_files():
    """the real writers' CPU smoke outputs (scratch/_tmp/s219, read-only, copied): GATE-T1 eval_gate (consistent, PASS), pool_meta, eval_textself (label
    audit by_window W1, contamination, labeller_kept_meta), and the G-T4 / G-T9 / G-T10 smoke args.json through G-T5 (smoke mode + the smoke pool)."""
    SM = REAL + "/scratch/_tmp/s219"; t = build("realsmoke", t2=False, g2=False)
    try:
        sp = "scratch/_tmp/s219/pipe/s219_pool_t1_s1/pool_t1.jsonl"
        for src, d, f in (("pipe/s219_gate_t1_s1/eval_gate.json", L["gate_t1"], "eval_gate.json"), ("pipe/s219_pool_t1_s1/pool_meta.json", L["pool_t1"], "pool_meta.json")):
            with open(os.path.join(SM, src)) as fh:
                t.w(d, f, json.load(fh))
        S.set_root(t.R, t.root); g = S.gate_t1("s1")
        chk(g["status"] == "PASS" and g["consistent"] and g["pooled_matches"] and g["descriptive"]["EQUIV"]["W1"]["PASS"] and g["card_class"] is None, g)
        sd = json.load(open(os.path.join(SM, "evals/eval_textself.json")))
        chk(S._sd_missing(sd, "t1") == [] and S._sd_missing(sd, "t2") == ["label_audit.by_window"], (S._sd_missing(sd, "t1"), S._sd_missing(sd, "t2")))
        la = S.label_audit_t(sd["label_audit"], {"W1": g}, "t1"); chk(la["ok"] and la["windows"]["W1"]["ci_overlap_pi"], la)
        chk(isinstance(sd["labeller_kept_meta"].get("t_start"), str) and S._time(sd["labeller_kept_meta"]["t_start"]) is not None, sd["labeller_kept_meta"])
        res = {}
        for tag in ("t4_a", "t9", "t10"):
            a = json.load(open(os.path.join(SM, tag, "args.json"))); ad = S.args_diff("s1", "t1", a, smoke=True, smoke_pool=sp)
            res[tag] = ad["ok"] if tag == "t4_a" else sorted(ad["shared_diffs"])
            if tag != "t4_a":
                chk(set(ad["shared_diffs"]) == {"bs"} and not ad["required_bad"] and not ad["unexpected_new_keys"] and not ad["neutral_bad"], (tag, ad))
            if tag == "t4_a":
                chk(ad["ok"] and not S.args_diff("s1", "t1", a)["ok"], (ad, "registered mode must fail on the smoke pool / stop_at"))
        chk(res["t4_a"], res)
        log = json.load(open(os.path.join(SM, "t10/timing.json"))); chk("sec_per_step" in log, log)
        return "real gate PASS (consistent; card 'cpu' -> no allowed card, as a smoke should); label audit ok; G-T5 smoke args: G-T4 ok, CPU G-T9 / G-T10 differ only in bs (8, CPU smokes) %s" % res
    finally:
        t.done()


@case
def round2_provenance():
    """Audit G-T13 (Theory): T2's init_from.json sha256, pool_meta windows' labeller_sha256 (self-labelled) and eval_gate.labeller_sha256 == T1's eval_text
    model_sha256; a BASE-LABELLED window's pool and fallback labeller_sha256 == BASE_SHA. Each mismatch -> CONTAMINATED (provenance), TEXT-CENSORED r2;
    a missing provenance field -> pending, failure at --final; the consistent self- and base-labelled chains read normally."""
    bad = "d" * 64; out = []
    g2fb = gate2_obj(keep2={16: 0.60}, fb={"W2": 0.80})
    g2fb_bad = gate2_obj(keep2={16: 0.60}, fb={"W2": 0.80}); g2fb_bad["fallback"]["W2"]["labeller_sha256"] = bad
    g2_badlab = gate2_obj(); g2_badlab["labeller_sha256"] = bad
    for tag, kw in (("init_from sha", {"log2": {"ini_mod": {"sha256": S.BASE_SHA}}}),
                    ("pool W2 sha", {"pm2": pool_meta(2, wsha={"W2": bad})}),
                    ("pool W1 sha = base", {"pm2": pool_meta(2, wsha={"W1": S.BASE_SHA})}),
                    ("eval_gate labeller sha", {"gate2": g2_badlab}),
                    ("base-labelled W2 pool sha = T1", {"gate2": g2fb, "pm2": pool_meta(2)}),
                    ("base-labelled W2 fallback sha", {"gate2": g2fb_bad, "pm2": pool_meta(2, base_windows=("W2",))})):
        t = build("prov_" + tag.replace(" ", "_"), **kw)
        try:
            s = run(t); r = assert_censored(s, "t2", "provenance", 2)
            chk(s["t2"]["CONTAMINATED"] and s["t2"]["CONTAMINATION"]["provenance_bad"], s["t2"]["CONTAMINATION"])
            out.append(tag)
        finally:
            t.done()
    t = build("prov_ok_base", gate2=g2fb, pm2=pool_meta(2, base_windows=("W2",)))
    try:
        s = run(t); chk(s["t2"]["status"] == "SCORED" and all(s["t2"]["CONTAMINATION"]["provenance"].values()), s["t2"]["CONTAMINATION"].get("provenance"))
    finally:
        t.done()
    t = build("prov_missing")
    try:
        t.rm(L["t2"], "init_from.json"); s = run(t); chk(s["t2"]["status"] == "PENDING" and "init_from.json" in s["t2"]["label"], s["t2"]["label"][:300])
        s = run(t, final=True); chk(s["t2"]["status"] == "FAIL" and outc(s) == "TEXT-CENSORED", s["t2"]["label"][:200])
    finally:
        t.done()
    return "%d mismatches -> CONTAMINATED (provenance); consistent base-labelled chain reads; missing init_from.json pending / --final failure" % len(out)


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@case
def real_tree_readonly():
    """the real tree (read-only; OUT redirected): absent (or in-progress: eval_text.partial.json only) cells -> pending (UNREAD with --not_admitted);
    --final -> failures; this run writes nothing under D:/Latent (the s219_* listings change only by the concurrent real writers' own eval files;
    theory/results/score_s219.json untouched)."""
    S.set_root(REAL_R, REAL)
    snap = lambda: {x: set(os.listdir(os.path.join(REAL_R, x))) for x in os.listdir(REAL_R) if x.startswith("s219_")}  # noqa: E731
    res_p = REAL + "/theory/results/score_s219.json"; res_m = os.path.getmtime(res_p) if os.path.exists(res_p) else None
    before = snap(); base_done = os.path.exists(os.path.join(REAL_R, L["base"], "eval_text.json")); outs = []
    for argv, tag in (([], "plain"), (["--final", "s1"], "final"), (["--not_admitted", "s219_t1_s1,s219_t2_s1"], "na")):
        S.OUT = os.path.join(HERE, "_real_out_%s.json" % tag); o = S.main(argv)["s1"]; outs.append(o); LINES.append(("REAL " + tag, o["comparison"]["line"]))
    p, f, na = outs
    chk(p["base"]["status"] == ("PENDING" if not base_done else p["base"]["status"]) and p["base"]["status"] in ("PENDING", "SCORED", "FAIL"), p["base"]["status"])
    chk(p["t1"]["status"] == "PENDING" and p["t2"]["status"] == "PENDING" and p["comparison"]["status"] in ("PENDING", "CENSORED"), (p["t1"]["status"], p["comparison"]["status"]))
    chk(f["t1"]["status"] == "FAIL" and f["comparison"]["outcome"] in ("TEXT-CENSORED", "NOT RUN"), f["comparison"]["outcome"])
    chk(na["t1"]["status"] == "UNREAD" and na["t2"]["status"] == "UNREAD", (na["t1"]["status"], na["t2"]["status"]))
    after = snap(); allowed = {"eval_text.json", "eval_text.partial.json", "eval_text.json.tmp", "eval_text.partial.json.tmp"}
    new = {d: sorted(after[d] - before.get(d, set())) for d in after if after[d] - before.get(d, set())}
    chk(set(after) <= set(before) | {L["base"]} and all(set(v) <= allowed for v in new.values()), ("wrote under D:/Latent?", new))
    chk((os.path.getmtime(res_p) if os.path.exists(res_p) else None) == res_m, "theory/results/score_s219.json changed")
    return "base %s (real eval %s); T1 / T2 pending; --final failures; UNREAD; nothing written under D:/Latent" % (
        p["base"]["status"], "done" if base_done else "in progress (partial only)")


@case
def train_tokens_on_every_line():
    """Addendum 2 item 2: every comparison line printed in this suite carries TRAIN-TOKENS beside TOKEN-COST, the §5.1 qualifiers and CTX-BOUND."""
    bad = [(r, ln[:120]) for r, ln in LINES if not ("TRAIN-TOKENS" in ln and "TOKEN-COST" in ln and "one lineage each" in ln and "CTX-BOUND" in ln
                                                     and "not in training tokens or FLOPs" in ln)]
    chk(LINES and not bad, bad[:3])
    return "%d comparison lines checked" % len(LINES)


def main(argv):
    os.makedirs(FX, exist_ok=True); t0 = time.time(); sel = [c for c in CASES if not argv or any(a in c.__name__ for a in argv)]
    if argv and "train_tokens_on_every_line" not in [c.__name__ for c in sel]:
        sel.append(train_tokens_on_every_line)
    for fn in sel:
        t1 = time.time()
        try:
            det = fn(); ok = True
        except Exception as e:                                              # noqa: BLE001
            ok, det = False, "%s: %s | %s" % (type(e).__name__, str(e)[:600], traceback.format_exc().splitlines()[-3:])
        S.set_root(os.path.join(REAL, "boxes", "runs_fsq"), REAL)
        RESULTS.append({"case": fn.__name__, "ok": ok, "detail": det, "sec": round(time.time() - t1, 1)})
        print(("PASS " if ok else "FAIL ") + fn.__name__ + " (%.1fs): " % (time.time() - t1) + str(det)[:400], flush=True)
    sha = hashlib.sha256(open(os.path.join(HERE, "score_s219.py"), "rb").read()).hexdigest()
    rep = {"written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "scorer": "score_s219.py", "scorer_sha256": sha,
           "registered": S.REGISTERED, "n_cases": len(RESULTS), "n_pass": sum(r["ok"] for r in RESULTS), "n_fail": sum(not r["ok"] for r in RESULTS),
           "sec": round(time.time() - t0, 1), "panel_deltas": PANEL, "cases": RESULTS}
    if not argv:
        with open(REPORT, "w", newline="\n") as fh:
            json.dump(rep, fh, indent=1, default=str)
    shutil.rmtree(FX, ignore_errors=True)
    print("%d / %d PASS in %.0fs" % (rep["n_pass"], rep["n_cases"], rep["sec"]))
    return rep


if __name__ == "__main__":
    main(sys.argv[1:])
