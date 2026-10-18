"""G-S12 fixtures for score_s216 (SD2_v1 §4.6 list, plus the reference-implementation checks). $0, CPU, idle priority, <= 2 threads, python -B.
Every fixture builds a synthetic boxes/runs_fsq tree under ./_fx/<name>/ (never under D:/Latent), points score_s216.R at it (set_root) and asserts the
outputs. Panel curves come from the real eval_vb files of s213_d3t_s1 / s215_ftself_s1, through v1_checks/v1_panels.py's own code (its module body is
executed with its json.dump / print statements removed, so it writes nothing). Per-item stores are built so that every aggregate equals its item mean
(acc values rounded to 1/1000 at 1,000 items per n). Writes ./fixtures_report.json.
    python -B fixtures_s216.py"""
import ast
import copy
import ctypes
import json
import math
import os
import shutil
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score_s216 as S  # noqa: E402

REAL = "D:/Latent/boxes/runs_fsq"; V1 = "D:/Latent/docs/research_selfdepth2_2026-10-18/v1_checks/v1_panels.py"
V1J = "D:/Latent/docs/research_selfdepth2_2026-10-18/v1_checks/v1_panels.json"
FX = os.path.join(HERE, "_fx"); REPORT = os.path.join(HERE, "fixtures_report.json")
S.OUT = os.path.join(HERE, "_fx_out.json")
assert not os.path.abspath(S.OUT).replace("\\", "/").lower().startswith("d:/latent")


def load_v1():
    """v1_panels.py executed from its source without its json.dump / print statements (it would rewrite v1_panels.json under D:/Latent)."""
    tree = ast.parse(open(V1, encoding="utf-8").read()); keep = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and (
                (isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "dump") or (isinstance(node.value.func, ast.Name) and node.value.func.id == "print")):
            continue
        if isinstance(node, ast.For) and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "print" for x in ast.walk(node)):
            continue
        keep.append(node)
    for node in keep:
        for x in ast.walk(node):
            if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "open":
                assert len(x.args) == 1 and not x.keywords, "a write-mode open() survived the filter"
            assert not (isinstance(x, ast.Attribute) and x.attr == "dump"), "json.dump survived the filter"
    ns = {"__file__": V1, "__name__": "v1_panels_exec"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), V1, "exec"), ns)
    return ns


t0 = time.time()
V = load_v1(); V1OUT = json.load(open(V1J))
rj = lambda d, f: json.load(open(os.path.join(REAL, d, f)))  # noqa: E731
EV_D3T, EV_FT = rj("s213_d3t_s1", "eval_vb.json"), rj("s215_ftself_s1", "eval_vb.json")
ARGS_D3T, ARGS_FT = rj("s213_d3t_s1", "args.json"), rj("s215_ftself_s1", "args.json")
H24_D3T, H24_FT = rj("s213_d3t_s1", "eval_hist24.json"), rj("s215_ftself_s1", "eval_hist24.json")
SP_FT, LFK_FT, SD_FT = rj("s215_ftself_s1", "eval_steppos.json"), rj("s215_ftself_s1", "eval_lfk.json"), rj("s215_ftself_s1", "eval_selfdepth.json")
LOG_D3T, LOG_FT = open(os.path.join(REAL, "s213_d3t_s1", "log.jsonl")).read(), open(os.path.join(REAL, "s215_ftself_s1", "log.jsonl")).read()
DRAWS200 = [{"step": r["step"], "plain_hash": r["plain_hash"]} for r in rj("s215_ftself_s1", "draws_log.json")[:200]]
TE16, TE24 = S.test_items(16), S.test_items(24)
BY_N24 = {n: [it for it in TE24 if it["n"] == n] for n in range(1, 25)}
J = lambda o: json.dumps(o)  # noqa: E731
FLOOR = V["FLOOR"]; A1 = V["A1"]; T1 = V["T1"]


def c3(v):
    return int(round(min(1.0, max(0.0, v)) * 1000))


def nested(c, N=1000):
    return [1] * c + [0] * (N - c)


def ans_from(c, y):
    return [y[i] if i < c else (y[i] + 1) % 11 for i in range(len(y))]


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


# ---------------------------------------------------------------------------------------------------------------- curves (v1_panels' own functions)
def _hfull():
    fl = {17: 0.97, 18: 0.95, 19: 0.93, 20: 0.90, 21: 0.88, 22: 0.85, 23: 0.82, 24: 0.80}; a2 = {**{n: 1.0 for n in range(13, 17)}, **fl}
    return a2, (lambda k, n: a2[n] if k >= n else FLOOR)


def _mixed():
    a2 = {13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 0.85, 18: 0.75, 19: 0.65, 20: 0.55, 21: 0.45, 22: 0.35, 23: 0.25, 24: 0.20}
    return a2, (lambda k, n: a2[n] if (k >= n or (n >= 17 and k >= 16)) else FLOOR)


CURVES = {h: ({n: V["PAN"][h][0](n) for n in range(13, 25)}, V["PAN"][h][1]["kn"]) for h in ("H_shift", "H_prop", "H_add_constant_growth", "SHRINK")}
CURVES["H_full"] = _hfull(); CURVES["MIXED"] = _mixed()
CURVES["Hshift_s2shift"] = ({n: A1[n - 5] for n in range(13, 25)}, lambda k, n: T1(k - 5, n - 5))   # exact H_shift on an R1 curve translated +1 in n


def ev_r2(key, base=None):
    a2, A2 = CURVES[key]; ev = copy.deepcopy(base or EV_FT)
    for n in range(13, 17):
        ev["acc_n2"][str(n)] = c3(a2[n]) / 1000
        for k in ev["acc_k"]:
            ev["acc_k"][k][str(n)] = c3(a2[n] if int(k) == n + 2 else A2(int(k), n)) / 1000
    return ev


_DEEP = {}


def deep_obj(key, T=None, meta=None, run="x"):
    a2, A2 = CURVES[key]; AK = {str(k): {} for k in range(29)}; A = {}; IT = {}
    for n in range(13, 25):
        tes = BY_N24[n]; y = [it["t"] for it in tes]; cnt = {k: c3(a2[n] if k == n + 2 else A2(k, n)) for k in range(29)}
        ans = [[y[i] if i < cnt[k] else (y[i] + 1) % 11 for k in range(29)] for i in range(len(y))]
        for k in range(29):
            AK[str(k)][str(n)] = sum(int(ans[i][k] == y[i]) for i in range(len(y))) / len(y)
        A[str(n)] = AK[str(n + 2)][str(n)]
        IT[str(n)] = {"idx": [(n - 1) * 1000 + i for i in range(len(y))], "y": y, "T": (T or {}).get(n) or [len(it["syms"]) + 2 for it in tes],
                      "height": [it["depth"] for it in tes], "ans": ans}
    m = {"run": run, "mode": "slva", "card": "NVIDIA GeForce RTX 3070", "dtype": "fp32", "tf32": False, "eval_bs": 16, "per_n": 1000, "n_lo": 13, "n_hi": 24,
         "K_grid": list(range(29)), "test_seed": 17100, "test_n_max": 24, "items_sha256": S.items_sha256(), "model_sha256": "fixture", "commit": "fixture",
         "sec": 1, "peak_gib": 4.2}
    m.update(meta or {})
    return {"meta": m, "acc_n2": A, "acc_k": AK, "items": IT, "chance": {str(n): 0.105 for n in range(13, 25)}, "def": "fixture"}


def deep_str(key):
    if key not in _DEEP:
        _DEEP[key] = J(deep_obj(key))
    return _DEEP[key]


def sd2_from(ev, la=None, contam=(0, 0)):
    o = {"cursor": 383488, "cursor_src": "fixture", "pool_sha256": "fixture",
         "memo": {str(n): {"acc_pool": 1.0, "acc_fresh": 1.0, "items": [2000, 2000]} for n in range(9, 17)},
         "label_audit": la or {"n_audit": 5000, "pi_label": 1.0, "pi_state": 0.9998, "pi_label_ci95": wilson(5000, 5000), "pi_state_ci95": wilson(4999, 5000),
                               "kept_wrong": [{"syms": [], "n": 14, "step": 3, "pattern": "state", "label": 1, "gold": 1}], "seed": 22670,
                               "by_window": {"W2": {"n_audit": 3333, "pi_label": 1.0, "pi_state": 3332 / 3333, "pi_label_ci95": wilson(3333, 3333),
                                                    "pi_state_ci95": wilson(3332, 3333)},
                                             "W1": {"n_audit": 1667, "pi_label": 1.0, "pi_state": 1.0, "pi_label_ci95": wilson(1667, 1667),
                                                    "pi_state_ci95": wilson(1667, 1667)}}},
         "items_n2": {str(n): nested(c3(ev["acc_n2"][str(n)])) for n in range(1, 17)},
         "items_k24": {str(n): nested(c3(ev["acc_k"]["24"][str(n)])) for n in range(1, 17)},
         "T": {str(n): [len(it["syms"]) + 2 for it in BY_N24[n]] for n in range(1, 17)},
         "idx": {str(n): [(n - 1) * 1000 + i for i in range(1000)] for n in range(1, 17)},
         "contamination": {"pool_test_hits": contam[0], "pool_excl_hits": contam[1]}, "def": "fixture E2", "card": "NVIDIA GeForce RTX 3070"}
    return o


def h24_items(h):
    h = copy.deepcopy(h); it = {"y": {}, "idx": {}}
    for n in range(3, 9):
        it["y"][str(n)] = [x["t"] for x in BY_N24[n]]; it["idx"][str(n)] = [(n - 1) * 1000 + i for i in range(1000)]
    for arm in S.HIST_ARMS:
        if arm in h:
            it[arm] = {str(n): {"ans_n2": ans_from(c3(h[arm]["acc_n2"][str(n)]), it["y"][str(n)]),
                                "ans_k24": ans_from(c3(h[arm]["acc_k24"][str(n)]), it["y"][str(n)])} for n in range(3, 9)}
    h["items"] = it
    return h


H24_KEEP = copy.deepcopy(H24_FT)
for _n in range(3, 7):
    H24_KEEP["self_ablate"]["acc_k24"][str(_n)] = 0.3                     # STAB_self 0.7 >= 0.5 (KEEPS-SELF-STAB-2)


def gate_entry(n, window, labeller, nk=1990, wl=0, ws=1, items=2000):
    return {"window": window, "labeller": labeller, "items": items, "n_keep": nk, "n_kept": nk, "keep_kv": nk / items, "kept_wrong_label": wl,
            "kept_wrong_state": ws, "pi": (nk - wl) / nk, "pi_E": (nk - ws) / nk, "pi_ci95": wilson(nk - wl, nk), "pi_E_ci95": wilson(nk - ws, nk)}


def pooled(es):
    nk = sum(e["n_kept"] for e in es); wl = sum(e["kept_wrong_label"] for e in es); ws = sum(e["kept_wrong_state"] for e in es)
    return {"items": sum(e["items"] for e in es), "n_kept": nk, "pi": (nk - wl) / nk, "pi_E": (nk - ws) / nk, "pi_ci95": wilson(nk - wl, nk),
            "pi_E_ci95": wilson(nk - ws, nk)}


def rule(es):                                                               # independent restatement of GATE-1's rule (fixture side)
    p = pooled(es)
    return p["pi"] >= 0.995 and p["pi_E"] >= 0.99 and all(e["pi"] >= 0.99 and e["keep_kv"] >= 0.70 for e in es)


def gate2_obj(lin, mods=None, fallback=None, flags=None):
    lab = "boxes/runs_fsq/%s/model.pt" % S.LIN[lin]["r1"]; B = {}
    for n in range(9, 17):
        B[str(n)] = gate_entry(n, "W2" if n >= 13 else "W1", lab, **((mods or {}).get(n) or {}))
    pbw = {"W1": pooled([B[str(n)] for n in range(9, 13)]), "W2": pooled([B[str(n)] for n in range(13, 17)])}
    g = {"by_n": B, "pooled_by_window": pbw, "GATE_2": rule([B[str(n)] for n in range(13, 17)]), "GATE_1p": rule([B[str(n)] for n in range(9, 13)]),
         "fallback": fallback, "labeller": lab, "card": "NVIDIA GeForce RTX 3070", "dtype": "fp32", "bs": 32}
    g.update(flags or {})
    return g


def fallback_w(lin, w, mods=None):
    lab = "boxes/runs_fsq/%s/model.pt" % S.LIN[lin]["base"]; ns = S.WIN[w]
    B = {str(n): gate_entry(n, w, lab, **((mods or {}).get(n) or {})) for n in ns}
    return {w: {"labeller": lab, "by_n": B, "pooled": pooled(list(B.values())), "GATE": rule(list(B.values()))}}


def gate1_obj(lin, mods=None, flags=None, ci=None):
    lab = "boxes/runs_fsq/%s/model.pt" % S.LIN[lin]["base"]
    B = {str(n): gate_entry(n, "W1", lab, **((mods or {}).get(n) or {})) for n in range(9, 13)}; P = pooled(list(B.values()))
    if ci:
        P["pi_ci95"], P["pi_E_ci95"] = ci, ci
    g = {"by_n": B, "pooled": P, "GATE_1": rule(list(B.values())), "labeller": lab, "card": "NVIDIA GeForce RTX 3070"}
    g.update(flags or {})
    return g


def pool_meta(lin, T16=61):
    return {"seed": 22600 if lin == "s1" else 22800, "per_n": {"9": 37000, "13": 74000}, "T_max_by_n": {"16": T16}, "labeller": S.LIN[lin]["r1"],
            "labeller_sha256": "fixture", "card": "NVIDIA GeForce RTX 3070", "dtype": "fp32", "bs": 32, "sha256": "fixture", "builder_commit": "fixture"}


def add_nonfinite(text, steps):
    """vbank.py l.805: a skipped step writes {"step": s, "nonfinite": 1} before that step's regular record; training continues."""
    out, todo = [], sorted(steps)
    for ln in text.splitlines():
        s = json.loads(ln).get("step") or 0
        while todo and todo[0] <= s:
            out.append(json.dumps({"step": todo.pop(0), "nonfinite": 1}))
        out.append(ln)
    return "\n".join(out) + "\n"


def log_r2(last=12000, reuse_at=None, nan_at=None):
    out = []
    for s in range(100, last + 1, 100):
        r = {"step": s, "loss": 0.001, "answer_ce": 0.001, "plain_answer_ce": 1e-5, "self_answer_ce": 0.002, "self_reuse": 1 if (reuse_at and s >= reuse_at) else 0,
             "pool_cursor": int(383488 * s / 12000), "n_self_steps": s // 2, "c": 0.0}
        if nan_at and s == nan_at:
            r["loss"] = float("nan")
        out.append(json.dumps(r))
    return "\n".join(out) + "\n"


def args_r2(lin, **kw):
    L = S.LIN[lin]; a = dict(ARGS_FT); a.update(init_from=S.rel(L["r1"], "model.pt"), self_pool=S.rel(L["pool_r2"], "pool_r2.jsonl"), self_kmax=20, self_micro=4,
                                                out="boxes/runs_fsq/" + L["r2"], seed=S.SEED[lin])
    a.update(kw)
    return a


def args_r1(**kw):
    L = S.LIN["s2"]; a = dict(ARGS_FT); a.update(init_from=S.rel(L["base"], "model.pt"), self_pool=S.rel(L["pool_r1"], "pool_r1.jsonl"), seed=2,
                                                 out="boxes/runs_fsq/" + L["r1"])
    a.update(kw)
    return a


def args_base(**kw):
    a = dict(ARGS_D3T); a.update(S.BASE_NEUTRAL_NEW); a.update(seed=2, out="boxes/runs_fsq/s216_d3t_s2"); a.update(kw)
    return a


def shift1(ev):
    """R1 curve translated +1 in n at n >= 9 (acc_n2 and acc_k)."""
    e = copy.deepcopy(ev)
    for n in range(9, 17):
        e["acc_n2"][str(n)] = ev["acc_n2"][str(n - 1)]
        for k in e["acc_k"]:
            kk = int(k); src = 24 if kk == 24 else max(kk - 1, 0)
            e["acc_k"][k][str(n)] = ev["acc_k"][str(src)][str(n - 1)]
    return e


_S = {}


def cs(key, obj_fn):
    if key not in _S:
        _S[key] = J(obj_fn())
    return _S[key]


def add_r2(F, lin, curve, keeps):
    L = S.LIN[lin]; d = L["r2"]; ev = ev_r2(curve)
    F[(d, "eval_vb.json")] = J(ev); F[(d, "eval_deep.json")] = deep_str(curve); F[(d, "log.jsonl")] = _S.setdefault("log_r2_text", log_r2())
    F[(d, "args.json")] = J(args_r2(lin)); F[(d, "draws_log.json")] = cs("draws", lambda: DRAWS200)
    F[(d, "eval_steppos.json")] = cs("sp", lambda: SP_FT); F[(d, "eval_lfk.json")] = cs("lfk", lambda: LFK_FT)
    F[(d, "eval_hist24.json")] = cs("h24k" if keeps else "h24", lambda: h24_items(H24_KEEP if keeps else H24_FT))
    F[(d, "eval_selfdepth.json")] = J(sd2_from(ev)); F[(L["pool_r2"], "pool_meta.json")] = J(pool_meta(lin)); F[(L["gate2"], "eval_gate.json")] = J(gate2_obj(lin))


def good(s1="H_prop", s2="H_prop", keeps=True, s2_r1="ft"):
    F = {("s213_d3t_s1", "args.json"): cs("a_d3t", lambda: ARGS_D3T), ("s213_d3t_s1", "eval_vb.json"): cs("ev_d3t", lambda: EV_D3T),
         ("s213_d3t_s1", "eval_hist24.json"): cs("h_d3t", lambda: H24_D3T), ("s213_d3t_s1", "log.jsonl"): LOG_D3T,
         ("s215_ftself_s1", "args.json"): cs("a_ft", lambda: ARGS_FT), ("s215_ftself_s1", "eval_vb.json"): cs("ev_ft", lambda: EV_FT),
         ("s215_ftself_s1", "eval_hist24.json"): cs("h_ft", lambda: H24_FT), ("s215_ftself_s1", "draws_log.json"): cs("draws", lambda: DRAWS200)}
    if s1:
        add_r2(F, "s1", s1, keeps)
    if s2:
        L = S.LIN["s2"]
        F[(L["base"], "log.jsonl")] = LOG_D3T; F[(L["base"], "args.json")] = J(args_base()); F[(L["base"], "eval_vb.json")] = cs("ev_d3t", lambda: EV_D3T)
        F[(L["base"], "eval_hist24.json")] = cs("h_d3t", lambda: H24_D3T)
        F[(L["gate1"], "eval_gate.json")] = J(gate1_obj("s2")); F[(L["pool_r1"], "pool_meta.json")] = J(pool_meta("s2"))
        r1ev = EV_FT if s2_r1 == "ft" else shift1(EV_FT)
        F[(L["r1"], "log.jsonl")] = LOG_FT; F[(L["r1"], "args.json")] = J(args_r1()); F[(L["r1"], "draws_log.json")] = cs("draws", lambda: DRAWS200)
        F[(L["r1"], "eval_vb.json")] = J(r1ev); F[(L["r1"], "eval_steppos.json")] = cs("sp", lambda: SP_FT); F[(L["r1"], "eval_lfk.json")] = cs("lfk", lambda: LFK_FT)
        F[(L["r1"], "eval_hist24.json")] = cs("h_ft", lambda: H24_FT); F[(L["r1"], "eval_selfdepth.json")] = cs("sd_ft", lambda: SD_FT)
        add_r2(F, "s2", s2, keeps)
    return F


def materialise(name, F):
    root = os.path.join(FX, name); shutil.rmtree(root, ignore_errors=True)
    for (d, f), s in F.items():
        os.makedirs(os.path.join(root, d), exist_ok=True)
        with open(os.path.join(root, d, f), "w", newline="\n") as fh:
            fh.write(s)
    return root


def run(name, F, final=(), na=()):
    root = materialise(name, F); S.set_root(root)
    try:
        return S.score(final, na)
    finally:
        S.set_root(REAL)


def mod(F, key, fn):
    """F[key] := JSON(fn(parsed F[key])) (a fresh object; the cached template strings are never mutated)."""
    o = json.loads(F[key]); fn(o); F[key] = J(o)                           # fn mutates in place; its return value is ignored


RES = {}


def fixture(name):
    def deco(fn):
        t = time.time()
        try:
            detail = fn() or "ok"; RES[name] = {"pass": True, "detail": detail}
            shutil.rmtree(os.path.join(FX, name), ignore_errors=True)
        except Exception as e:  # noqa: BLE001
            RES[name] = {"pass": False, "detail": "%s: %s" % (type(e).__name__, e), "trace": traceback.format_exc()[-1500:]}
        RES[name]["sec"] = round(time.time() - t, 1); print(("PASS " if RES[name]["pass"] else "FAIL ") + name, RES[name]["sec"], flush=True)
        return fn
    return deco


def close(x, y, tol):
    return isinstance(x, (int, float)) and abs(x - y) <= tol


def chk(c, msg):
    if not c:
        raise AssertionError(msg)


R1S, R2S = lambda o: o["s2"]["r1"], lambda o, lin="s1": o[lin]["r2"]  # noqa: E731


# ================================================================================================================== reference implementation checks
@fixture("bars_reproduce_v1_panels")
def _():
    a0, a1 = V["a0"], V["a1"]; chk(S.r1_bars(a0) == V["r1_bars"](a0) and S.r2_bars(a0, a1) == V["r2_bars"](a0, a1), "s1 instance differs")
    b1, b2 = S.r1_bars(a0), S.r2_bars(a0, a1)
    chk(b1["MARGIN_GROWS"] == 0.63 and b1["SHRINK_below"] == 0.40 and b2["MARGIN_GROWS_2"] == 0.785 and b2["SHRINK_below"] == 0.60, "registered s1 bars")
    chk(json.loads(json.dumps(b2)) == V1OUT["r2_bars_s1_instance"] and json.loads(json.dumps(b1)) == V1OUT["r1_bars_s1_instance"], "v1_panels.json s1 instance")
    import random
    rg = random.Random(216); n_ok = 0
    for _ in range(300):                                                    # random monotone curves + the shifted / degenerate lineages
        x = sorted([rg.random() for _ in range(16)], reverse=True); y = sorted([rg.random() for _ in range(16)], reverse=True)
        c0 = {n: x[n - 1] for n in range(1, 17)}; c1 = {n: y[n - 1] for n in range(1, 17)}
        chk(S.r1_bars(c0) == V["r1_bars"](c0) and S.r2_bars(c0, c1) == V["r2_bars"](c0, c1), "random curve differs"); n_ok += 1
    sh = S.curve(shift1(EV_FT)); chk(S.r2_bars(a0, sh) == V["r2_bars"](a0, sh), "shifted")
    return "s1 instance = v1_panels.json (0.63 / 0.40 / 0.785 / 0.60); %d random curves + shifted R1 identical" % n_ok


@fixture("panels_reproduce_v1_panels_per_n")
def _():
    S.set_root(REAL); p = S.panels("s1"); bad = []
    for h in ("H_shift", "H_prop", "H_add_constant_growth", "SHRINK"):
        for k in ("M2", "M_far", "n50", "n90", "WIDTH_2"):
            if abs(p[h][k] - V1OUT["panels"][h][k]) > 1e-12:
                bad.append((h, k, p[h][k], V1OUT["panels"][h][k]))
    chk(not bad, bad); chk(p["source_past_16"].startswith("assumed"), "R1 eval_deep must not be read before s216_r2_s1 launches")
    return "per-n panels equal v1_panels.json to 1e-12 (H_shift n50 %.3f, H_prop %.3f)" % (p["H_shift"]["n50"], p["H_prop"]["n50"])


@fixture("stop_loc_generalises_score_s215")
def _():
    a = {n: EV_FT["acc_n2"][str(n)] for n in range(1, 17)}; Ak = lambda k, n: EV_FT["acc_k"][str(k)][str(n)]  # noqa: E731
    G, st = S.stop_loc(a, Ak, range(9, 16), 16, 16); G2, st2 = S.S215.stop_loc(EV_FT)
    chk(st == st2 == 15 and all(abs(G[k] - G2[k]) < 1e-15 for k in G2), (st, st2))
    return "FT-SELF STOP 15 both"


@fixture("hist24_items_equals_score_s213_G-S19")
def _():
    F = good(s1="H_prop", s2=None, keeps=False); root = materialise("hist24_items_equals_score_s213_G-S19", F); S.set_root(root)
    try:
        d = "s216_r2_s1"; h = S.S213.hist24(d); hi = S.hist24_items(d)
        chk(hi["consistent"] and all(abs(hi["STAB"][a] - h["STAB"][a]) <= 1e-12 for a in h["STAB"]), (hi, h["STAB"]))
    finally:
        S.set_root(REAL)
    return "STAB from E4 items == score_s213.hist24 (STAB_self %.5f)" % h["STAB"]["self_ablate"]


@fixture("deep_read_single_source_assert")
def _():
    F = good(s1="H_prop", s2=None); root = materialise("deep_read_single_source_assert", F); S.set_root(root)
    try:
        dr = S.deep_read("s216_r2_s1"); chk(len(dr["files_opened"]) == 1 and dr["files_opened"][0].endswith("eval_deep.json"), dr["files_opened"])
        real = S._deep

        def leak(p, *a):
            open(os.path.join(os.path.dirname(p), "eval_vb.json")).close(); return real(p, *a)
        S._deep = leak
        try:
            S.deep_read("s216_r2_s1"); raised = False
        except AssertionError:
            raised = True
        finally:
            S._deep = real
        chk(raised, "a second file opened inside deep_read must raise")
    finally:
        S.set_root(REAL)
    return "opens only eval_deep.json; an extra open() raises AssertionError"


# ================================================================================================================== §4.6 G-S12 list
def panel_fixture(curve, mech, **want):   # returns the R2 cell
    o = run("panel_" + curve, good(s1=curve, s2=None)); c = R2S(o); dr = c["DEEP"]
    chk(c["status"] == "SCORED" and not c["censored"], c["label"]); chk(c["MECHANISM"] == mech, (c["MECHANISM"], c["label"]))
    for k, (v, tol) in want.items():
        got = c[k] if k in c else dr[k]
        chk(got == v if tol is None else close(got, v, tol), (k, got, v))
    return c


@fixture("H_shift_exact")
def _():
    c = panel_fixture("H_shift", "SHIFT", M2=(0.71725, 1e-9), STOP=(20, None), DBS_2=(False, None), LIT_C2_2=(0.0047, 0.002))
    chk(close(c["DEEP"]["G"][19], 0.0231, 0.002) and not c["SD2_C2"] and c["SD2_SELF"], ("G(19)", c["DEEP"]["G"][19]))
    return c["label"] + " | G(19) %.4f" % c["DEEP"]["G"][19]


@fixture("H_prop")
def _():
    return panel_fixture("H_prop", "GROW", DBS_2=(True, None), M2=(0.852, 0.0006), n50=(20.61, 0.005), WIDTH_2=(2.22, 0.005), SD2_C2=(True, None))["label"]


@fixture("constant_growth")
def _():
    return panel_fixture("H_add_constant_growth", "GROW", M2=(0.895, 0.0006), WIDTH_2=(1.57, 0.005), KB2=(0.51, 0.005))["label"]


@fixture("H_full_floor")
def _():
    return panel_fixture("H_full", "FULL", LIT_C2_2_MET=(True, None), M2=(0.9375, 1e-9), M_far=(0.8375, 1e-9), STOP=("24+", None), n50=("> 24", None))["label"]


@fixture("base_margin_SHRINK")
def _():
    return panel_fixture("SHRINK", "SHRINK", M2=(0.47975, 0.0006))["label"]


@fixture("MIXED")
def _():
    return panel_fixture("MIXED", "MIXED", M2=(0.70, 1e-9), STOP=(16, None))["label"]


@fixture("not_ABSORB-2_UNLABELLED")
def _():
    F = good(s1="H_prop", s2=None); mod(F, ("s216_r2_s1", "eval_vb.json"), lambda e: e["acc_n2"].__setitem__("16", 0.90))
    c = R2S(run("not_ABSORB-2_UNLABELLED", F)); chk(c["MECHANISM"] == "UNLABELLED (not ABSORB-2)" and not c["SD2_SELF"] and not c["SD2_C2"], c["label"])
    return c["label"]


@fixture("RETAIN_fail_ERODED")
def _():
    F = good(s1="H_prop", s2=None); mod(F, ("s216_r2_s1", "eval_vb.json"), lambda e: e["acc_n2"].__setitem__("11", 0.98))
    c = R2S(run("RETAIN_fail_ERODED", F)); chk(c["ERODED"] and not c["RETAIN"] and not c["SD2_SELF"] and "ERODED" in c["label"] and c["MECHANISM"] == "GROW", c["label"])
    return c["label"]


@fixture("dead")
def _():
    F = good(s1="H_prop", s2=None); F[("s216_r2_s1", "log.jsonl")] = log_r2(last=11900)
    c = R2S(run("dead", F)); chk(c["dead"] and c["censored"] and not c["SD2_C2"] and c["label"].startswith("FAILURE (dead"), c["label"])
    F[("s216_r2_s1", "log.jsonl")] = log_r2(last=6000, nan_at=6000); c2 = R2S(run("dead", F)); chk(c2["dead"] and c2["censored"] and c2["last_step"] == 6000, c2["label"])
    return c["label"] + " || NaN that ended training at 6000 -> incomplete log -> dead"


@fixture("nonfinite_skipped_steps_not_dead")
def _():
    F = good(); F[("s216_r2_s1", "log.jsonl")] = add_nonfinite(log_r2(), (3000, 6000, 9050))
    F[("s216_r1_s2", "log.jsonl")] = add_nonfinite(LOG_FT, (2000, 5000, 8000)); F[("s216_d3t_s2", "log.jsonl")] = add_nonfinite(LOG_D3T, (100, 24000, 47000))
    o = run("nonfinite", F); c, r1, b = R2S(o), R1S(o), o["s2"]["base"]
    chk(not c["dead"] and c["nonfinite_steps"] == 3 and c["status"] == "SCORED" and c["SD2_C2"], (c["nonfinite_steps"], c["label"]))
    chk(not r1["dead"] and r1["nonfinite_steps"] == 3 and r1["ROUND2_CONDITION"], (r1.get("nonfinite_steps"), r1["label"]))
    chk(not b["dead"] and b["nonfinite_steps"] == 3 and b["status"] == "PASS" and o["X2"]["outcome"] == "PASS", (b.get("nonfinite_steps"), b["label"], o["X2"]))
    return "complete logs with 3 nonfinite records each (R2-s1 / R1-s2 / s2 base): not dead, nonfinite_steps 3, x2 PASS"


@fixture("FT-DIVERGED")
def _():
    F = good(s1="H_prop", s2=None); mod(F, ("s216_r2_s1", "eval_vb.json"), lambda e: e["acc_k"]["24"].__setitem__("2", 0.96))
    c = R2S(run("FT-DIVERGED", F)); chk(c["FT_DIVERGED"] and c["censored"] and not c["SD2_C2"] and c["MECHANISM"] == "UNLABELLED (censored)", c["label"])
    F = good(s1="H_prop", s2=None); mod(F, ("s216_r2_s1", "eval_vb.json"), lambda e: e["acc_n2"].__setitem__("5", 0.985))
    c2 = R2S(run("FT-DIVERGED", F)); chk(c2["FT_DIVERGED"], c2["label"])
    return c["label"]


@fixture("LABEL-NOISY-2_one_window_0.985")
def _():
    out = []
    for tag, fn in (("W1 pi 0.985", lambda la: la["by_window"]["W1"].update(pi_label=0.985, pi_label_ci95=wilson(1642, 1667))),
                    ("W2 CI non-overlap", lambda la: la["by_window"]["W2"].update(pi_label=0.995, pi_label_ci95=[0.990, 0.997])),
                    ("pooled pi_E 0.985", lambda la: la.update(pi_state=0.985, pi_state_ci95=wilson(4925, 5000)))):
        F = good(s1="H_prop", s2=None); mod(F, ("s216_r2_s1", "eval_selfdepth.json"), lambda sd: fn(sd["label_audit"]))
        c = R2S(run("LABEL-NOISY-2", F)); chk(c["LABEL_NOISY_2"] and c["censored"] and not c["SD2_C2"], (tag, c["label"])); out.append(tag)
    c = R2S(run("LABEL-NOISY-2", good(s1="H_prop", s2=None))); chk(not c["LABEL_NOISY_2"], "clean audit must pass")
    return "noisy: %s; clean passes" % out


@fixture("CONTAMINATED_pool_record_equals_test_n22")
def _():
    t22 = BY_N24[22][7]["syms"]; recs = [{"syms": t22, "n": 22, "chain_label": 0, "steps": []}] + [{"syms": [" 1", " +", " +", " %d" % i], "n": 1, "chain_label": 3, "steps": []} for i in range(5)]   # not renderable: never a test item
    F = good(s1="H_prop", s2=None); F[("s216_pool_s1", "pool_r2.jsonl")] = "\n".join(json.dumps(r) for r in recs) + "\n"
    c = R2S(run("CONTAMINATED", F)); chk(c["CONTAMINATED"] and c["CONTAMINATION"]["recheck"]["test24_hits"] == 1 and c["CONTAMINATION"]["pool_test_hits"] == 0, c["CONTAMINATION"])
    F[("s216_pool_s1", "pool_r2.jsonl")] = "\n".join(json.dumps(r) for r in recs[1:]) + "\n"; c0 = R2S(run("CONTAMINATED", F))
    chk(not c0["CONTAMINATED"] and c0["CONTAMINATION"]["recheck"]["test24_hits"] == 0, "clean pool")
    F = good(s1="H_prop", s2=None); mod(F, ("s216_r2_s1", "eval_selfdepth.json"), lambda sd: sd["contamination"].update(pool_excl_hits=1))
    c1 = R2S(run("CONTAMINATED", F)); chk(c1["CONTAMINATED"] and c1["censored"], "excl hit")
    F = good(s1="H_prop", s2=None); F[("s216_r2_s1", "log.jsonl")] = log_r2(reuse_at=11000); c2 = R2S(run("CONTAMINATED", F))
    chk(c2["CONTAMINATED"] and not c2["SD2_C2"], "self_reuse")
    return "n-22 test item in pool (E2 says 0) caught by the re-check; pool_excl_hits 1; self_reuse 1"


@fixture("ARGS-DIFF_self_kmax24_and_s2_seed1")
def _():
    out = []
    for tag, lin, kw in (("s1 self_kmax 24", "s1", {"self_kmax": 24}), ("s2 seed 1", "s2", {"seed": 1}), ("s1 pause A", "s1", {"pause": "A"}),
                         ("s1 new key", "s1", {"foo": 1}), ("s1 init_from base", "s1", {"init_from": "boxes/runs_fsq/s213_d3t_s1/model.pt"}),
                         ("s1 seed 2", "s1", {"seed": 2}), ("s1 lr", "s1", {"lr": 1e-4})):
        F = good(); F[(S.LIN[lin]["r2"], "args.json")] = J(args_r2(lin, **kw)); o = run("ARGS", F); c = R2S(o, lin)
        chk(c["ARGS_BAD"] and c["censored"] and not c["SD2_C2"], (tag, c["ARGS_DIFF"])); out.append(tag)
        if lin == "s2":
            chk(o["X2"]["outcome"] == "FAIL", o["X2"])
    o = run("ARGS", good()); chk(not R2S(o)["ARGS_BAD"] and not R2S(o, "s2")["ARGS_BAD"] and not o["s2"]["base"]["ARGS_DIFF"]["unexpected_new_keys"], "clean")
    F = good(); F[("s216_d3t_s2", "args.json")] = J(args_base(p_self=0.5)); o = run("ARGS", F)
    chk(o["s2"]["base"]["status"] == "FAIL" and o["X2"]["outcome"] == "FAIL", o["s2"]["base"]["label"])
    F = good(); F[("s216_d3t_s2", "args.json")] = J(args_base(seed=1)); chk(run("ARGS", F)["s2"]["base"]["status"] == "FAIL", "base seed 1")
    F = good(); F[("s216_r1_s2", "args.json")] = J(args_r1(self_pool="boxes/runs_fsq/s215_pool/pool_r1.jsonl")); chk(R1S(run("ARGS", F))["ARGS_BAD"], "r1 pool")
    return "ARGS_BAD: %s; base p_self 0.5 / seed 1 and R1-s2 s215 pool -> fail; clean trees ok (base neutral new keys accepted)" % out


@fixture("s2_bars_by_formula")
def _():
    o = run("s2_bars_by_formula", good(s1="H_prop", s2="Hshift_s2shift", s2_r1="shift1")); c = R2S(o, "s2"); b2, b1 = o["s2"]["bars_r2"], o["s1"]["bars_r2"]
    ref = V["r2_bars"](V["a0"], S.curve(shift1(EV_FT)))
    chk(b2 == json.loads(json.dumps(ref)) or b2 == ref, (b2, ref)); chk(b2["MARGIN_GROWS_2"] != b1["MARGIN_GROWS_2"] and b2["SHRINK_below"] != b1["SHRINK_below"], "bars must move")
    chk(c["MECHANISM"] == "SHIFT", c["label"]); chk(c["DEEP"]["M2"] >= b1["MARGIN_GROWS_2"], "s1's bars would have read GROW")
    return "s2 MG %.3f / SB %.3f (s1 0.785 / 0.60); R2-s2 exact H_shift on the +1 R1: %s" % (b2["MARGIN_GROWS_2"], b2["SHRINK_below"], c["label"])


@fixture("degenerate_bars_UNLABELLED")
def _():
    F = good(s1="H_prop", s2="H_prop"); b = copy.deepcopy(EV_D3T); r = copy.deepcopy(EV_FT)
    for n in range(9, 13):
        b["acc_n2"][str(n)] = 1.0
    for n in range(13, 17):
        r["acc_n2"][str(n)] = 0.3
    F[("s216_d3t_s2", "eval_vb.json")] = J(b); F[("s216_r1_s2", "eval_vb.json")] = J(r); o = run("degenerate", F); c = R2S(o, "s2")
    chk(o["s2"]["bars_r2"]["degenerate_SH_ge_MG"] and c["MECHANISM"] == "UNLABELLED (bars degenerate)", (o["s2"]["bars_r2"], c["label"]))
    return "SB %.3f >= MG %.3f -> %s" % (o["s2"]["bars_r2"]["SHRINK_below"], o["s2"]["bars_r2"]["MARGIN_GROWS_2"], c["MECHANISM"])


@fixture("R1-s2_seed2_own_gate")
def _():
    F = good(); F[("s215_gate1", "eval_gate.json")] = J(gate1_obj("s1", ci=[0.90, 0.95], mods={10: {"wl": 30}}))   # a failing s215 gate in the tree
    o = run("R1-s2_own_gate", F); c = R1S(o)
    chk(not c["ARGS_BAD"] and c["GATE_1"]["GATE_1"] and c["LABEL_AUDIT"]["ok"] and c["status"] == "SCORED" and c["ROUND2_CONDITION"], c["label"])
    chk(o["s2"]["gate1"]["GATE_1"] and o["s2"]["row"]["status"] == "PASS", o["s2"]["row"])
    return "R1-s2 %s; GATE-1-s2 read from s216_gate1_s2 (s215_gate1 failing in the tree)" % c["label"]


@fixture("dead_s2_base_x2_FAIL")
def _():
    F = good(); F[("s216_d3t_s2", "log.jsonl")] = "".join(ln + "\n" for ln in LOG_D3T.splitlines() if json.loads(ln).get("step", 0) <= 40000)
    o = run("dead_s2_base", F); chk(o["s2"]["base"]["status"] == "FAIL" and o["s2"]["row"]["stage"] == "base" and o["X2"]["outcome"] == "FAIL", (o["s2"]["base"]["label"], o["X2"]))
    return o["X2"]["label"]


@fixture("completed_s2_base_below_floor_NOT_COMPETENT")
def _():
    F = good(); mod(F, ("s216_d3t_s2", "eval_vb.json"), lambda e: e["acc_n2"].__setitem__("6", 0.985)); o = run("not_competent", F)
    chk(o["s2"]["base"]["status"] == "NOT_COMPETENT" and o["X2"]["outcome"] == "NOT COMPETENT", o["X2"])
    F = good(); mod(F, ("s216_d3t_s2", "eval_vb.json"), lambda e: e["acc_k"]["24"].__setitem__("3", 0.965)); o2 = run("not_competent", F)
    chk(o2["X2"]["outcome"] == "NOT COMPETENT", "a_24 floor")
    return o["X2"]["label"]


@fixture("GATE-1-s2_fail_x2_FAIL")
def _():
    F = good(); F[("s216_gate1_s2", "eval_gate.json")] = J(gate1_obj("s2", mods={10: {"wl": 25}})); o = run("gate1_fail", F)
    chk(o["s2"]["row"]["stage"] == "gate1" and o["X2"]["outcome"] == "FAIL", (o["s2"]["row"], o["X2"]))
    F[("s216_gate1_s2", "eval_gate.json")] = J(gate1_obj("s2", mods={10: {"wl": 25}}, flags={"GATE_1": True})); o2 = run("gate1_fail", F)
    chk(o2["s2"]["row"]["stage"] == "gate1" and "file flag" in o2["s2"]["row"]["why"], o2["s2"]["row"])
    return o["X2"]["label"] + " | flag-mismatch file also fails"


@fixture("R1-s2_stopped_x2_FAIL")
def _():
    F = good(); mod(F, ("s216_r1_s2", "eval_vb.json"), lambda e: e["acc_n2"].__setitem__("12", 0.90)); o = run("r1_stopped", F)
    chk(not R1S(o)["ROUND2_CONDITION"] and o["s2"]["row"]["stage"] == "r1" and "stopped at round 1" in o["s2"]["row"]["why"] and o["X2"]["outcome"] == "FAIL", o["s2"]["row"])
    return o["s2"]["row"]["why"][:120]


@fixture("single_source_eval_vb_minus_0.01")
def _():
    o0 = R2S(run("single_source", good(s1="H_prop", s2=None))); F = good(s1="H_prop", s2=None)

    def shift(e):
        for n in range(13, 17):
            e["acc_n2"][str(n)] -= 0.01
            for k in e["acc_k"]:
                e["acc_k"][k][str(n)] = max(0.0, e["acc_k"][k][str(n)] - 0.01)
    mod(F, ("s216_r2_s1", "eval_vb.json"), shift); o1 = R2S(run("single_source", F))
    same = all(o0["DEEP"][k] == o1["DEEP"][k] for k in ("STOP", "n50", "n90", "WIDTH_2", "M2", "G", "LIT_C2_2"))
    chk(same and o1["BS_CHECK"]["FLAG"] and o0["DEPTH_GENERAL_2"] != o1["DEPTH_GENERAL_2"], (o0["DEEP"]["STOP"], o1["DEEP"]["STOP"]))
    return "STOP %s / n50 %.4f / WIDTH-2 %.4f unchanged; BS-CHECK flagged %.3f" % (o1["DEEP"]["STOP"], o1["DEEP"]["n50"], o1["DEEP"]["WIDTH_2"], o1["BS_CHECK"]["max_abs_delta"])


MISS_R2 = [("log.jsonl", None), ("args.json", None), ("draws_log.json", None), ("eval_vb.json", None), ("eval_vb.json", "drop vswap"), ("eval_deep.json", None),
           ("eval_steppos.json", None), ("eval_lfk.json", None), ("eval_hist24.json", None), ("eval_hist24.json", "drop E4 items"),
           ("eval_selfdepth.json", None), ("eval_selfdepth.json", "drop by_window"), ("eval_selfdepth.json", "drop pool_excl_hits"),
           ("pool_meta.json", None), ("eval_gate.json", None), ("eval_gate.json", "GATE_2 undecided"), ("bars", "R1 eval_vb missing")]


def apply_miss(F, lin, f, how):
    L = S.LIN[lin]; d = L["r2"]
    if f == "bars":
        del F[(L["r1"], "eval_vb.json")]; return
    key = (L["pool_r2"], f) if f == "pool_meta.json" else (L["gate2"], f) if f == "eval_gate.json" else (d, f)
    if how is None:
        del F[key]
    elif how == "drop vswap":
        mod(F, key, lambda e: e.pop("vswap"))
    elif how == "drop E4 items":
        mod(F, key, lambda h: h.pop("items"))
    elif how == "drop by_window":
        mod(F, key, lambda sd: sd["label_audit"].pop("by_window"))
    elif how == "drop pool_excl_hits":
        mod(F, key, lambda sd: sd["contamination"].pop("pool_excl_hits"))
    elif how == "GATE_2 undecided":
        mod(F, key, lambda g: g.update(GATE_2=None))


@fixture("each_registered_file_missing_pending_then_final_failure")
def _():
    base = R2S(run("miss_base", good(s1="H_prop", s2=None))); chk(base["SD2_C2_MEM"] and base["MARGIN_GROWS_2"], base["label"]); done = []
    for f, how in MISS_R2:
        F = good(s1="H_prop", s2=None); apply_miss(F, "s1", f, how); o = run("miss", F); c = R2S(o)
        chk(c["status"] == "PENDING" and c["label"].startswith("pending (missing") and o["s1"]["row"]["status"] == "PENDING" and "SD2_C2" not in c, (f, how, c["label"]))
        of = run("miss", F, final=("s1",)); cf = R2S(of)
        chk(cf["status"] != "PENDING" and of["s1"]["row"]["status"] in ("FAIL", "PASS"), (f, how, cf.get("label")))
        if f in S.CENSOR_FILES and f != "bars":
            chk(cf["status"] == "FAIL" and not cf["SD2_C2"] and of["s1"]["row"]["status"] == "FAIL", (f, how, cf["label"]))
        elif f == "eval_deep.json":
            chk(not cf["SD2_C2"] and not cf["DBS_2"] and cf["MECHANISM"].startswith("UNLABELLED") and cf["SD2_SELF"], cf["label"])
        elif f in ("eval_steppos.json", "eval_lfk.json"):
            chk(not cf["LENGTH_FREE"] and not cf["SD2_SELF"] and not cf["SD2_C2"], cf["label"])
        elif f == "eval_hist24.json":
            chk(not cf["KEEPS_SELF_STAB_2"] and not cf["SD2_C2_MEM"] and cf["SD2_C2"] and cf["KEEPS"]["MEM_SUBLABEL_POST_HOC"] == "unreadable (gate)", cf["label"])
        elif f == "bars":
            chk(not cf["MARGIN_GROWS_2"] and cf["MECHANISM"].startswith("UNLABELLED (bars void"), cf["label"])
        done.append("%s%s" % (f, "" if how is None else " (%s)" % how))
    # R1-s2 and s2 base registered files (x2 pending -> UNREAD; --final s2 -> the stage fails)
    L = S.LIN["s2"]
    for key, kind in ([((L["r1"], f), "r1") for f in S.REG_R1] + [((L["pool_r1"], "pool_meta.json"), "r1"), ((L["gate1"], "eval_gate.json"), "gate1")] +
                      [((L["base"], f), "base") for f in ("log.jsonl", "args.json", "eval_vb.json")]):
        F = good(); del F[key]; o = run("miss_s2", F)
        chk(o["s2"]["row"]["status"] == "PENDING" and o["X2"]["outcome"] == "UNREAD", (key, o["s2"]["row"]))
        of = run("miss_s2", F, final=("s2",))
        if key[1] in ("eval_steppos.json", "eval_lfk.json", "eval_hist24.json") and kind == "r1":   # not in the round-2 condition: the reading fails, the stage stands
            c = R1S(of); chk(c["status"] == "SCORED" and c["ROUND2_CONDITION"] and (not c["LENGTH_FREE"] if key[1] != "eval_hist24.json" else not c["KEEPS_SELF_STAB"]), (key, c["label"]))
        else:
            chk(of["s2"]["row"]["status"] == "FAIL" and of["s2"]["row"]["stage"] in (kind, "gate1") and of["X2"]["outcome"] == "FAIL", (key, of["s2"]["row"]))
        done.append("/".join(key))
    F = good(s1=None, s2=None); F[("s216_gate2_s1", "eval_gate.json")] = J(gate2_obj("s1")); F[("s216_pool_s1", "pool_meta.json")] = J(pool_meta("s1"))
    of = run("miss_dir", F, final=("s1",)); chk(R2S(of)["status"] == "FAIL" and of["s1"]["row"]["status"] == "FAIL", "absent R2 dir at --final")
    return "%d cases: pending (no reading), --final -> rule-b failure of the readings that need the file" % (len(done) + 1)


@fixture("eval_selfdepth_missing_pending_never_LABEL-AUDIT-2_pass")
def _():
    F = good(s1="H_prop", s2=None); del F[("s216_r2_s1", "eval_selfdepth.json")]; c = R2S(run("sd_missing", F))
    chk(c["status"] == "PENDING" and "LABEL_NOISY_2" not in c and "CONTAMINATED" not in c and "SD2_SELF" not in c, c)
    return c["label"][:90]


@fixture("items_sha256_mismatch_void")
def _():
    out = []
    for tag, fn in (("sha", lambda dp: dp["meta"].update(items_sha256="0" * 64)),
                    ("tf32 on", lambda dp: dp["meta"].update(tf32=True)), ("bs 32", lambda dp: dp["meta"].update(eval_bs=32)),
                    ("acc vs items", lambda dp: dp["acc_k"]["20"].__setitem__("22", dp["acc_k"]["20"]["22"] + 0.001))):
        F = good(s1="H_prop", s2=None); mod(F, ("s216_r2_s1", "eval_deep.json"), fn); c = R2S(run("sha", F))
        chk(c["DEEP"]["void"] and not c["DBS_2"] and not c["MARGIN_GROWS_2"] and not c["LIT_C2_2_MET"] and not c["SD2_C2"] and c["MECHANISM"].startswith("UNLABELLED (DEEP void"),
            (tag, c["label"]))
        out.append(tag)
    return "void on %s (SD2-SELF unaffected; every n >= 17 reading fails)" % out


@fixture("E4_items_inconsistent_hist24_gate_fails")
def _():
    F = good(s1="H_prop", s2=None, keeps=True)

    def flip(h):
        a = h["items"]["self_ablate"]["4"]["ans_k24"]; y = h["items"]["y"]["4"]; a[999] = y[999] if a[999] != y[999] else (y[999] + 1) % 11
    mod(F, ("s216_r2_s1", "eval_hist24.json"), flip); c = R2S(run("e4_bad", F))
    chk(not c["KEEPS"]["gate_with_items"] and not c["KEEPS_SELF_STAB_2"] and c["KEEPS"]["MEM_SUBLABEL_POST_HOC"] == "unreadable (gate)" and c["SD2_C2"] and not c["SD2_C2_MEM"], c["KEEPS"])
    c0 = R2S(run("e4_bad", good(s1="H_prop", s2=None, keeps=True))); chk(c0["KEEPS_SELF_STAB_2"] and c0["SD2_C2_MEM"], "consistent store keeps")
    return "one flipped item -> gate fails -> MEM-TRADED-2 (unreadable (gate)); consistent store -> SD2-C2-MEM"


@fixture("MEM-TRADED-2_sublabels_all_three")
def _():
    a = R2S(run("sub", good(s1="H_prop", s2=None, keeps=False)))
    chk(a["MEM_TRADED_2"] and a["KEEPS"]["MEM_SUBLABEL_POST_HOC"] == "STAB_self low with dup_newest destabilising", a["KEEPS"]["MEM_SUBLABEL_POST_HOC"])
    h = copy.deepcopy(H24_FT)
    for n in range(3, 9):
        h["dup_newest"]["acc_k24"][str(n)] = 0.98
    F = good(s1="H_prop", s2=None); F[("s216_r2_s1", "eval_hist24.json")] = J(h24_items(h)); b = R2S(run("sub", F))
    chk(b["KEEPS"]["hist24"]["gate"] and b["KEEPS"]["MEM_SUBLABEL_POST_HOC"] == "content channel not stabilising", b["KEEPS"]["MEM_SUBLABEL_POST_HOC"])
    h = copy.deepcopy(H24_FT); h["intact"]["acc_n2"]["4"] = 0.999
    F = good(s1="H_prop", s2=None); F[("s216_r2_s1", "eval_hist24.json")] = J(h24_items(h)); c = R2S(run("sub", F))
    chk(not c["KEEPS"]["hist24"]["gate"] and c["KEEPS"]["MEM_SUBLABEL_POST_HOC"] == "unreadable (gate)", c["KEEPS"]["MEM_SUBLABEL_POST_HOC"])
    hist = a["MEM_HISTORY"]; chk(close(hist["base"], 0.78175, 1e-9) and close(hist["r1"], 0.48925, 1e-9) and close(hist["r2"], 0.48925, 1e-9) and not hist["MEM_ERODING"], hist)
    k = R2S(run("sub", good(s1="H_prop", s2=None, keeps=True))); chk(k["MEM_HISTORY"]["r2"] > 0.5 and k["KEEPS"]["MEM_SUBLABEL_POST_HOC"] is None, "keeps")
    h = copy.deepcopy(H24_FT)
    for n in range(3, 7):
        h["self_ablate"]["acc_k24"][str(n)] = 0.6
    F = good(s1="H_prop", s2=None); F[("s216_r2_s1", "eval_hist24.json")] = J(h24_items(h)); e = R2S(run("sub", F))
    chk(e["MEM_HISTORY"]["MEM_ERODING"] is True and e["MEM_HISTORY"]["r2"] < 0.439, e["MEM_HISTORY"])
    return "(a) %s / (b) %s / (c) %s; MEM history %.3f -> %.3f -> %.3f; MEM-ERODING at %.3f" % (
        a["KEEPS"]["MEM_SUBLABEL_POST_HOC"], b["KEEPS"]["MEM_SUBLABEL_POST_HOC"], c["KEEPS"]["MEM_SUBLABEL_POST_HOC"], hist["base"], hist["r1"], hist["r2"], e["MEM_HISTORY"]["r2"])


@fixture("H_T_flag")
def _():
    T = {n: [55] * 500 + [70] * 500 for n in range(17, 23)}
    F = good(s1="H_prop", s2=None); F[("s216_r2_s1", "eval_deep.json")] = J(deep_obj("H_prop", T=T)); c = R2S(run("H_T", F))
    chk(c["DEEP"]["H_T_flag"] is True and len(c["DEEP"]["TSPLIT"]["qualifying_n"]) >= 2 and c["DEEP"]["M2_lowT"] is not None, c["DEEP"]["TSPLIT"])
    c0 = R2S(run("H_T", good(s1="H_prop", s2=None))); chk(c0["DEEP"]["H_T_flag"] is False and c["DEEP"]["M2"] == c0["DEEP"]["M2"], c0["DEEP"]["TSPLIT"]["qualifying_n"])
    return "T > T_train deficit at n %s -> H_T flag; real-T control: no flag" % c["DEEP"]["TSPLIT"]["qualifying_n"]


@fixture("BS-CHECK_flag")
def _():
    F = good(s1="H_prop", s2=None); ev = ev_r2("H_prop"); ev["acc_n2"]["14"] = round(ev["acc_n2"]["14"] - 0.01, 3)
    F[("s216_r2_s1", "eval_vb.json")] = J(ev); F[("s216_r2_s1", "eval_selfdepth.json")] = J(sd2_from(ev)); c = R2S(run("bs", F))
    chk(c["BS_CHECK"]["FLAG"] and close(c["BS_CHECK"]["max_abs_delta"], 0.01, 1e-9) and c["BS_CHECK"]["item_flips"]["items_n2"]["flips"] == 10 and "BS-CHECK FLAG" in c["label"], c["BS_CHECK"])
    c0 = R2S(run("bs", good(s1="H_prop", s2=None))); chk(not c0["BS_CHECK"]["FLAG"] and c0["BS_CHECK"]["item_flips"]["items_n2"]["flips"] == 0, c0["BS_CHECK"])
    return "max |delta| 0.010 at n 14, 10 item flips -> flagged (descriptive); identical files -> 0"


@fixture("not_admitted_UNREAD")
def _():
    o = run("na", good(), na=("s216_r2_s2",)); chk(R2S(o, "s2")["status"] == "UNREAD" and o["s1"]["row"]["status"] == "PASS" and o["X2"]["outcome"] == "UNREAD", o["X2"])
    o = run("na", good(), final=("s1", "s2"), na=("s216_r2_s1",)); chk(R2S(o)["status"] == "UNREAD" and o["X2"]["outcome"] == "UNREAD", o["X2"])
    return o["X2"]["label"]


@fixture("card_mismatch_pending_then_VOID_and_all_4060_scored")
def _():
    F = good(); mod(F, ("s216_r2_s1", "eval_lfk.json"), lambda x: x.update(card="NVIDIA GeForce RTX 4060")); o = run("card", F); c = R2S(o)
    chk(c["status"] == "PENDING" and "card mismatch" in c["label"] and "s216_r2_s1/eval_lfk.json" in c["label"] and o["X2"]["outcome"] == "UNREAD", c["label"])
    of = run("card", F, final=("s1",)); cf = R2S(of)
    chk(cf["status"] == "FAIL" and cf["label"].startswith("VOID (card mismatch") and not cf["SD2_C2"] and of["s1"]["row"]["status"] == "FAIL", cf["label"])
    for key, getter in ((("s215_ftself_s1", "eval_vb.json"), R2S), (("s216_r1_s2", "eval_hist24.json"), R1S), (("s216_d3t_s2", "eval_vb.json"), lambda o: o["s2"]["base"])):
        G = good(); mod(G, key, lambda x: x.update(card="NVIDIA GeForce RTX 4060")); g = getter(run("card", G)); gf = getter(run("card", G, final=("s1", "s2")))
        chk(g["status"] == "PENDING" and "card mismatch" in g["label"] and gf["label"].startswith("VOID (card mismatch"), (key, g["label"], gf["label"]))
    G = good(); mod(G, ("s216_r2_s1", "eval_deep.json"), lambda x: x["meta"].update(card="NVIDIA GeForce RTX 4060")); chk("card mismatch" in R2S(run("card", G))["label"], "deep meta card")
    ref = run("card", good()); H = {}
    for k, v in good().items():
        if k[1].endswith(".json"):
            x = json.loads(v)
            if isinstance(x, dict):
                if "card" in x:
                    x["card"] = "NVIDIA GeForce RTX 4060"
                if isinstance(x.get("meta"), dict) and "card" in x["meta"]:
                    x["meta"]["card"] = "NVIDIA GeForce RTX 4060"
            v = J(x)
        H[k] = v
    S.CARD_CLASS = "4060"
    try:
        o4 = run("card4060", H); o3 = run("card4060", good())
    finally:
        S.CARD_CLASS = "3070"
    chk(R2S(o4)["label"] == R2S(ref)["label"] and R2S(o4, "s2")["label"] == R2S(ref, "s2")["label"] and o4["X2"]["outcome"] == ref["X2"]["outcome"] == "PASS",
        (R2S(o4)["label"], o4["X2"]))
    chk(R2S(o3)["status"] == "PENDING" and "card mismatch" in R2S(o3)["label"], "3070 files under CARD_CLASS 4060 must not read")
    return "one 4060 file (R2 lfk / deep meta / bar input / R1-s2 / s2 base) -> pending, --final VOID; all-4060 + CARD_CLASS 4060 -> %s; x2 PASS" % R2S(o4)["label"][:40]


# ================================================================================================================== extra: gates, rows, real tree
@fixture("x2_PASS_both_lineages")
def _():
    o = run("x2_pass", good()); chk(o["X2"]["outcome"] == "PASS" and o["SD2_SELF_X2"] is True, o["X2"])
    o2 = run("x2_pass", good(s2="H_shift")); chk(o2["X2"]["outcome"] == "FAIL" and "1 of 2" in o2["X2"]["label"], o2["X2"])
    F = good(); F = {k: v for k, v in F.items() if k[0] not in ("s216_r1_s2", "s216_r2_s2", "s216_gate2_s2", "s216_pool_r2_s2")}
    o3 = run("x2_pass", F); chk(o3["X2"]["outcome"] == "UNREAD" and o3["s2"]["row"]["stage"] == "r1", o3["s2"]["row"])
    F = good(s1="H_shift"); F = {k: v for k, v in F.items() if k[0] not in ("s216_r1_s2", "s216_r2_s2", "s216_gate2_s2", "s216_pool_r2_s2")}
    o4 = run("x2_pass", F); chk(o4["X2"]["outcome"] == "FAIL" and "pending" in o4["X2"]["label"], o4["X2"])
    F = good(s1="H_shift"); F = {k: v for k, v in F.items() if not k[0].startswith("s216_") or k[0].endswith("_s1")}
    o5 = run("x2_pass", F); chk(o5["X2"]["outcome"] == "UNREAD", o5["X2"])
    return "2/2 PASS; s2 H_shift -> FAIL 1 of 2; s2 pending at R1 -> UNREAD; s1 FAIL + s2 base competent -> FAIL; s1 FAIL + s2 base pending -> UNREAD"


@fixture("GATE-2_recompute_fallback_both_fail")
def _():
    F = good(s1="H_prop", s2=None); F[("s216_gate2_s1", "eval_gate.json")] = J(gate2_obj("s1", mods={15: {"wl": 30}}, flags={"GATE_2": True}))
    o = run("g2", F); chk(R2S(o)["status"] == "PENDING" and o["s1"]["row"]["stage"] == "gate2", R2S(o)["label"])
    chk(o["s1"]["gate2"]["W2"]["primary"]["consistent"] is False, o["s1"]["gate2"]["W2"]["primary"])
    of = run("g2", F, final=("s1",)); chk(R2S(of)["label"].startswith("VOID (no round 2") and of["s1"]["row"]["status"] == "FAIL", R2S(of)["label"])
    F[("s216_gate2_s1", "eval_gate.json")] = J(gate2_obj("s1", mods={15: {"wl": 30}}, fallback=fallback_w("s1", "W2"))); o = run("g2", F); c = R2S(o)
    chk(o["s1"]["gate2"]["W2"]["status"] == "BASE-LABELLED" and c["SD2_SELF"] and "BASE-LABELLED" in c["WORDING"]["SD2_SELF"] and "for n 9–12 was itself self-trained" in c["WORDING"]["SD2_SELF"], c["WORDING"])
    F[("s216_gate2_s1", "eval_gate.json")] = J(gate2_obj("s1", mods={15: {"wl": 30}}, fallback=fallback_w("s1", "W2", mods={14: {"wl": 30}}))); o = run("g2", F)
    chk(R2S(o)["label"].startswith("VOID") and o["s1"]["row"]["stage"] == "gate2", o["s1"]["row"])
    w = R2S(run("g2", good(s1="H_prop", s2=None)))["WORDING"]["SD2_SELF"]; chk("into the loop. The labeller was itself self-trained; labels are gold-equivalent at π = 1.00000 (rule ≥ 0.995; LABEL-AUDIT-2 π = 1.0000)" in w, w)
    return "flag != recomputed -> pending, --final VOID; fallback -> BASE-LABELLED wording; both fail -> VOID / row FAIL at gate2"


@fixture("real_tree_read_only_all_UNREAD")
def _():
    S.set_root(REAL); stamp = time.time() - 1
    o = S.main([]); chk(o["X2"]["outcome"] == "UNREAD" and o["s1"]["row"]["status"] == "PENDING" and o["s2"]["row"]["status"] == "PENDING", o["X2"])
    chk(o["s1"]["r2"]["status"] == "PENDING" and o["s2"]["base"]["status"] == "PENDING" and o["BASELINES"]["s1"].startswith("not read"), "pending")
    mine = ["D:/Latent/theory/results/score_s216.json", "D:/Latent/theory/score_s216.py", V1J] + [os.path.join(dp, f) for top in (
        "D:/Latent/boxes/runs_fsq/s215_ftself_s1", "D:/Latent/boxes/runs_fsq/s213_d3t_s1", "D:/Latent/theory/__pycache__", "D:/Latent/boxes/__pycache__")
        for dp, _, fs in os.walk(top) for f in fs if ("score_s216" in f or "nested170" in f or not top.endswith("__pycache__"))]
    newer = [p for p in mine if os.path.exists(p) and os.path.getmtime(p) > stamp]
    chk(not newer, newer); chk(os.path.exists(S.OUT) and not S.OUT.replace("\\", "/").lower().startswith("d:/latent"), S.OUT)
    shutil.copy(S.OUT, os.path.join(HERE, "real_tree_score_s216.json"))
    return S.reading(o)[:300]


rep = {"fixtures": RES, "n": len(RES), "n_pass": sum(v["pass"] for v in RES.values()), "sec": round(time.time() - t0, 1),
       "scorer": os.path.join(HERE, "score_s216.py"), "v1_panels": V1, "registered": "SD2_v1.md @ e9ade2ec"}
json.dump(rep, open(REPORT, "w", newline="\n"), indent=1, ensure_ascii=False)
print("FIXTURES %d / %d PASS in %.0f s" % (rep["n_pass"], rep["n"], rep["sec"]))
if rep["n_pass"] == rep["n"]:
    shutil.rmtree(FX, ignore_errors=True)                                   # keep the synthetic trees only when something failed
