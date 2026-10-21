"""§219 TEXT-SELF CPU gates (TS_v1.md at 146a25d0 §7): every gate that runs on CPU before any GPU / cloud step. CPU only (CUDA_VISIBLE_DEVICES ""),
2 threads, Idle priority, python -B. Each gate writes scratch/gate_s219/<gate>.json as it finishes; scratch outputs under scratch/_tmp/s219/ (model-sized
files deleted after use).
  static  G-T7 (sha pin + AST: the 57 parser calls verbatim), G-T20 SEAL (AST: no generation / eval name or eval file in textself_ft.py), G-T0 grep
          (no 's219_gate'; chain_label / values sites), G-T8 decorators (ast: no decorator on any trainer / compose / loader function)
  t2      G-T2 COMPOSE-T: compose_T(syms, gold) == N.cot_ids(root) (20,000 random items n 1..24); c_struct == len(cot_ids) on all 24,000 test items; one
          changed value -> differs; check_step_T kept <=> the candidate-equality test on 100,000 perturbed steps (drop, duplicate, swap, wrong span,
          wrong bracket, non-value token, wrong terminator, none); parse_tree == vbank_diag._parse_tree on every perturbed step (A1)
  t3      G-T3 STREAMS (replay part): the trainer's coin_step vs s215_ftself_s1 / s216_r2_s1 draws_log (12,000 coins, every U); the plain stream's
          plain_hash at steps 1..200 in-process == a separate-process replay
  t0      G-T0 NO-GOLD: loader round trip on a tiny pool (+ every corrupted variant refused); self-batch cot_ids / cot_steps calls = 0 (plain > 0);
          the builder's gold patch (in a subprocess) raises on every gold function while the labeller / compose_T / chain_form still run
  t6      G-T6 LOAD on the base model.pt (strict + one synthetic forward)
  t8      G-T8 GRADFLOW: the self CE alone (base GPT-2, 4 pool-shaped records) -> nonzero grad on every trunk tensor; plain likewise; L requires grad
  t18     G-T18's CPU stub (never emits EOT): the loop stops before row 1,024 and the answer is -1 (cap-bound, mid and overflow items)
  t7      G-T7 PLAIN-EQUIV: textself_ft --p_self 0 --stop_at 3 vs nested170 --arm textcot --time_steps 3 (CPU, bs 8, log_every 1): per-step losses
          bitwise; args.json equal on every non-META key except the FT keys (at their off values)
  pipe    B1 gates -> B2 label (base, TF + FULL + EQUIV) -> B2 score -> B1 pool, all at smoke scale under scratch/_tmp/s219/pipe (G-T15 path, G-T16 /
          G-T2 on gate items, the exclusion set at full size = G-T17 part 1)
  t4      G-T4 INIT + RESUME (CPU fp32, registered args except META, the pipe pool, bs 64): ckpt_model_0 == init bitwise, INIT_FROM line, ckpt_last /
          ckpt_prev, the resumed step 3 == the uninterrupted step 3 (loss_last, draws_log, ckpt_last model + AdamW bitwise), no eval*.json
  t5      G-T5 ARGS-DIFF (a preliminary implementation of §4.6; the registered rule is theory/score_s219.py) on t4's real args.json + mutations
  t9t10   G-T9 / G-T10 code paths on CPU (--mem_check_self, --time_steps): mem_check_self.json / timing.json, args.json, no eval*.json
  t16     G-T16 EQUIV (CPU mini): TF-GREEDY REROOT vs the draft's ts_label_check.label_reroot vs per-step KV generation on the regenerated rng-23180
          items (25 per n at n 9..12, 10 at n 13 / 16) + ts_label_check_s1.json's per-item records; FULL vs FULL generation at n 10 / 11
  t17     G-T17 SMOKE-AT-SCALE: the full exclusion set; the trainer's loader on full-size synthetic 388,000 / 390,000 pools (timings)
  evals   E1 / E2 / G-T18 code paths on CPU at smoke sizes (text_eval --part text / smoke / textself) under scratch/_tmp/s219/evals
  e2e     G-T12 end to end: the real writers into scratch/_tmp/s219_e2e/boxes/runs_fsq, theory/score_s219.py imported read-only on that tree
  pre1 / pre2  pre-launch on the REAL round pool (run after B1, before the deploy): G-T1 pool ∩ exclusion = ∅, G-T3's precomputed per-n self counts,
          T_train, the exact TRAIN-TOKENS of the round (text side)
    python -B scratch/gate_s219.py static t2 t3 t0 t6 t8 t18 t7 pipe t4 t5 t9t10 t16 t17 evals e2e      (pre1 / pre2 later, on the real pools)"""
import ast
import ctypes
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time

DET = os.environ.get("S219_DET") == "1"   # Note 219-3 item 2: S219_DET=1 -> 1 thread + torch.use_deterministic_algorithms in every subprocess (sitecustomize)
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "1" if DET else "2"; os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
if DET:
    os.environ["PYTHONPATH"] = "D:/Latent/scratch/_tmp/s219_det" + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
sys.dont_write_bytecode = True; sys.path.insert(0, "D:/Latent"); os.chdir("D:/Latent")
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS (children inherit it)
except Exception:                                                           # noqa: BLE001
    pass
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1 if DET else 2)
if DET:
    torch.use_deterministic_algorithms(True)
TRAINER = os.environ.get("S219_TRAINER", "")   # pre-apply testing of a drafted trainer: a script path run in place of -m boxes.textself_ft
if TRAINER:
    os.environ["PYTHONPATH"] = "D:/Latent" + os.pathsep + os.environ.get("PYTHONPATH", "")
FTM = [TRAINER] if TRAINER else ["-m", "boxes.textself_ft"]
FTP = TRAINER or "D:/Latent/boxes/textself_ft.py"
FTSRC = TRAINER or "boxes/textself_ft.py"
import boxes.nested170 as N  # noqa: E402
import boxes.textself as TS  # noqa: E402
import boxes.textself_ft as FT  # noqa: E402

OUT = "scratch/gate_s219"; TMP = "scratch/_tmp/s219"; os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
ENV = dict(os.environ); PY = [sys.executable, "-B"]
BASE_ARGS = TS.BASE_RUN + "/args.json"; PIPE = TMP + "/pipe"; PIPE_POOL = PIPE + "/s219_pool_t1_s1/pool_t1.jsonl"


def save(name, obj):
    obj = dict(obj, gate=name, clock=time.strftime("%Y-%m-%d %H:%M:%S")); json.dump(obj, open(os.path.join(OUT, name + ".json"), "w", newline="\n"), indent=1, default=str)
    print("GATE", name, "PASS" if obj.get("PASS") else "FAIL" if obj.get("PASS") is False else "-", {k: v for k, v in obj.items() if k in ("PASS", "sec")}, flush=True)
    return obj


def run(cmd, log):
    t0 = time.time()
    with open(log, "w", newline="\n") as f:
        rc = subprocess.run(cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT, cwd="D:/Latent").returncode
    return rc, round(time.time() - t0)


def fresh(d):
    d = os.path.normpath(d).replace("\\", "/"); assert d.startswith("scratch/_tmp/s219/") and d.count("/") >= 3, d   # never a drive root / repo dir
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d); return d


def meta_t():
    """score_s213.META (read by AST, no import) + {excl_survive_by_n, hostprobe_sps} (TS_v1 §4.6)."""
    tr = ast.parse(open("theory/score_s213.py", encoding="utf-8").read())
    for nd in ast.walk(tr):
        if isinstance(nd, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "META" for t in nd.targets):
            return set(ast.literal_eval(nd.value)) | {"excl_survive_by_n", "hostprobe_sps"}
    raise AssertionError("score_s213.META not found")


def add_arg_calls(path, fn):
    tr = ast.parse(open(path, encoding="utf-8").read()); f = next(nd for nd in ast.walk(tr) if isinstance(nd, ast.FunctionDef) and nd.name == fn)
    cs = [nd for nd in ast.walk(f) if isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute) and nd.func.attr == "add_argument"]
    return sorted(cs, key=lambda c: (c.lineno, c.col_offset))


def synth_pool(d, ns, per_n, seed, values="gold"):
    """a pool-format file (+ pool_meta with its sha) of generated items: values = gold (gate-side only) or random."""
    os.makedirs(d, exist_ok=True); rng = np.random.default_rng(seed); fp = os.path.join(d, "pool.jsonl")
    with open(fp, "w", encoding="utf-8", newline="\n") as f:
        for n in ns:
            for _ in range(per_n):
                r_, _ = N.generate(rng, n); s_ = N.render(r_)
                v = TS.gold_values(s_) if values == "gold" else [int(x) for x in rng.integers(0, 11, n)]
                f.write(json.dumps(TS.record(s_, v), separators=(",", ":")) + "\n")
    json.dump({"sha256": TS.file_sha256(fp), "synthetic": True}, open(os.path.join(d, "pool_meta.json"), "w", newline="\n")); return fp


# ------------------------------------------------------------------------------------------------ static
def g_static():
    t0 = time.time(); o = {}; src = open(FTSRC, encoding="utf-8").read(); tr = ast.parse(src)
    o["nested170_sha256"] = TS.file_sha256("boxes/nested170.py"); o["sha_pinned_ok"] = o["nested170_sha256"] == TS.NESTED170_SHA256
    cn = add_arg_calls("boxes/nested170.py", "main"); ct = add_arg_calls(FTSRC, "build_parser")
    o["parser"] = {"nested170_calls": len(cn), "textself_ft_calls": len(ct), "verbatim_57": [ast.dump(c) for c in cn] == [ast.dump(c) for c in ct[:len(cn)]],
                   "ft_keys": [c.args[0].value for c in ct[len(cn):]]}
    dests = [x.dest for x in FT.build_parser()._actions if x.dest != "help"]; o["parser"]["n_dests"] = len(dests); o["parser"]["ft_keys_ok"] = sorted(o["parser"]["ft_keys"]) == sorted("--" + k for k in FT.FT_KEYS)
    bad_names = {"cot_answers", "evaluate_n", "greedy_batch", "label_reroot", "label_full", "gen_reroot", "gen_full", "free_generate", "aux_eval", "inject_answers", "direct_answers"}
    used = sorted({nd.attr for nd in ast.walk(tr) if isinstance(nd, ast.Attribute)} | {nd.id for nd in ast.walk(tr) if isinstance(nd, ast.Name)})
    o["seal"] = {"generation_names_used": sorted(set(used) & bad_names)}
    joins = [c for c in ast.walk(tr) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "join"]
    fnames = sorted({a_.value for c in joins for a_ in c.args if isinstance(a_, ast.Constant) and isinstance(a_.value, str)}
                    | {a_.left.value for c in joins for a_ in c.args if isinstance(a_, ast.BinOp) and isinstance(a_.left, ast.Constant)}
                    | {e.value for c in joins for a_ in c.args if isinstance(a_, ast.IfExp) for e in (a_.body, a_.orelse) if isinstance(e, ast.Constant)})
    o["seal"]["files_written"] = fnames; o["seal"]["eval_files"] = [f for f in fnames if "eval" in str(f)]
    o["seal"]["PASS"] = not o["seal"]["generation_names_used"] and not o["seal"]["eval_files"]
    o["no_gold"] = {"s219_gate_in_trainer": src.count("s219_gate"), "chain_label_in_trainer": src.count("chain_label"),
                    "values_sites_trainer": [i + 1 for i, ln in enumerate(src.split("\n")) if '"values"' in ln],
                    "cot_ids_sites_trainer": [i + 1 for i, ln in enumerate(src.split("\n")) if "cot_ids(" in ln and not ln.strip().startswith("#")]}
    ts_src = open("boxes/textself.py", encoding="utf-8").read()
    o["no_gold"]["chain_label_sites_textself"] = [i + 1 for i, ln in enumerate(ts_src.split("\n")) if "chain_label" in ln]
    o["no_gold"]["PASS"] = o["no_gold"]["s219_gate_in_trainer"] == 0 and o["no_gold"]["chain_label_in_trainer"] == 0 and len(o["no_gold"]["cot_ids_sites_trainer"]) == 1
    dec = {}
    for path in (FTSRC, "boxes/textself.py"):
        for nd in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
            if isinstance(nd, ast.FunctionDef):
                dec["%s:%s" % (os.path.basename(path), nd.name)] = [ast.unparse(d) for d in nd.decorator_list]
    train_fns = ["textself_ft.py:" + f for f in ("main", "make_loss", "loss_items", "draw_step", "coin_step", "mem_check", "plain_hash", "build_parser",
                                                       "deep_self", "atomic_json")] + \
                ["textself.py:" + f for f in ("compose_T", "candidate", "reduce_shape", "parse_tree", "self_item", "load_pool", "c_struct", "n_of")]
    o["decorators"] = {"train_path": {f: dec.get(f) for f in train_fns}, "no_grad_fns": sorted(k for k, v in dec.items() if any("no_grad" in x for x in v))}
    lf = next(nd for nd in ast.walk(tr) if isinstance(nd, ast.FunctionDef) and nd.name == "loss_items")
    ng = [w for w in ast.walk(lf) if isinstance(w, ast.With) and any("no_grad" in ast.unparse(i.context_expr) for i in w.items)]
    o["decorators"]["no_grad_blocks_assign_L"] = any(isinstance(t, ast.Name) and t.id == "L" for w in ng for nd in ast.walk(w) if isinstance(nd, ast.Assign) for t in nd.targets)
    o["decorators"]["PASS"] = all(v == [] for v in o["decorators"]["train_path"].values()) and not o["decorators"]["no_grad_blocks_assign_L"]
    o["PASS"] = bool(o["sha_pinned_ok"] and o["parser"]["verbatim_57"] and len(cn) == 57 and o["parser"]["ft_keys_ok"] and o["seal"]["PASS"] and o["no_gold"]["PASS"] and o["decorators"]["PASS"])
    o["sec"] = round(time.time() - t0); return save("static", o)


# ------------------------------------------------------------------------------------------------ t2
def _perturb(rng, cur, v):
    """(kind, s, term) from the correct step c(v) of `cur`."""
    pre, suf, more = TS.candidate(cur); good = pre + [TS.NUM_IDS[v]] + suf; term = TS.EQ if more else TS.EOT; e = N.ids(N.render(cur)); lo, hi = TS.handle_span(cur)
    k = ["none", "drop", "dup", "swap", "wrong_span", "wrong_bracket", "nonvalue", "wrong_term", "random_tok"][int(rng.integers(0, 9))]; s = list(good)
    if k == "drop":
        s.pop(int(rng.integers(0, len(s)))) if len(s) > 1 else None
    elif k == "dup":
        j = int(rng.integers(0, len(s))); s.insert(j, s[j])
    elif k == "swap" and len(s) > 1:
        j = int(rng.integers(0, len(s) - 1)); s[j], s[j + 1] = s[j + 1], s[j]
    elif k == "wrong_span":                                                 # reduce another span of the previous expression by one value token
        L_ = len(e); a_ = int(rng.integers(0, L_)); b_ = int(rng.integers(a_, min(L_, a_ + 7))); s = e[:a_] + [TS.NUM_IDS[int(rng.integers(0, 11))]] + e[b_ + 1:]
    elif k == "wrong_bracket":
        c = int(rng.integers(0, 3))
        if c == 0:                                                          # the handle's own brackets kept around the value
            s = e[:lo] + [N.TID[" ("], TS.NUM_IDS[v], N.TID[" )"]] + e[hi + 1:]
        elif c == 1:                                                        # a redundant bracket pair around one value token of the good step
            vs_ = [j for j, t in enumerate(s) if t in TS.NUMS]; j = vs_[int(rng.integers(0, len(vs_)))]; s = s[:j] + [N.TID[" ("], s[j], N.TID[" )"]] + s[j + 1:]
        else:                                                               # drop one bracket token
            bs_ = [j for j, t in enumerate(s) if t in (N.TID[" ("], N.TID[" )"])]
            if bs_:
                s.pop(bs_[int(rng.integers(0, len(bs_)))])
    elif k == "nonvalue":
        s[len(pre)] = [N.TID[w] for w in (" +", " -", " *", " /", " (", " )", " =", " .")][int(rng.integers(0, 8))]
    elif k == "wrong_term":
        term = TS.EOT if term == TS.EQ else TS.EQ
    elif k == "random_tok" and s:
        s[int(rng.integers(0, len(s)))] = list(TS.ID2SYM)[int(rng.integers(0, len(TS.ID2SYM)))]
    return k, s, term


def g_t2():
    t0 = time.time(); o = {}; rng = np.random.default_rng(21901)
    import boxes.vbank_diag as VD
    mism = diffchg = 0
    for _ in range(20000):
        n = int(rng.integers(1, 25)); r_, _ = N.generate(rng, n); s_ = N.render(r_); gv = TS.gold_values(s_); ct = TS.compose_T(s_, gv); cg = N.cot_ids(r_)
        mism += ct != cg; v2 = list(gv); k = int(rng.integers(0, n)); v2[k] = (v2[k] + 1 + int(rng.integers(0, 10))) % 11; diffchg += TS.compose_T(s_, v2) == cg
    o["compose_eq_cot_ids"] = {"items": 20000, "mismatch": int(mism), "changed_value_still_equal": int(diffchg)}
    TD = N.test_set(1000, n_max=24); o["c_struct_test"] = {"items": len(TD), "mismatch": int(sum(TS.c_struct(it["syms"]) != len(N.cot_ids(it["root"])) for it in TD))}
    agree = 0; tot = 0; kinds = {}; pt_bad = 0; vbad = 0
    for _ in range(100000):
        n = int(rng.integers(1, 25)); r_, _ = N.generate(rng, n); cur = TS.parse_tree(N.render(r_)); steps = int(rng.integers(0, n))
        for _q in range(steps):
            TS.reduce_shape(cur, int(rng.integers(0, 11)))
        v = int(rng.integers(0, 11)); kind, s, term = _perturb(rng, cur, v)
        f, vv, t2 = TS.check_step_T(cur, s, term); ce = TS.cand_equal(cur, s, term); kept = t2 is not None
        agree += int(kept == (ce is not None)); vbad += int(kept and vv != ce); tot += 1; d_ = kinds.setdefault(kind, [0, 0]); d_[0] += 1; d_[1] += int(kept)
        sy = [TS.ID2SYM.get(int(t)) for t in s]
        if all(x is not None for x in sy):
            a1, a2 = TS.parse_tree(sy), VD._parse_tree(sy); pt_bad += int((a1 is None) != (a2 is None) or (a1 is not None and N.render(a1) != N.render(a2)))
    o["cv_vs_candidate"] = {"steps": tot, "agree": agree, "value_mismatch": vbad, "by_kind_items_kept": kinds}; o["parse_tree_vs_vbank_diag"] = {"mismatch": pt_bad}
    o["PASS"] = bool(mism == 0 and diffchg == 0 and o["c_struct_test"]["mismatch"] == 0 and agree == tot and vbad == 0 and pt_bad == 0)
    o["sec"] = round(time.time() - t0); return save("t2_compose", o)


# ------------------------------------------------------------------------------------------------ t3
def plain_hashes(n_steps):
    TE = N.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        r__, _ = N.generate(r_sp, 8); EXCL.add(tuple(N.render(r__)))
    rng = np.random.default_rng(1); return [FT.plain_hash(N.train_items(rng, 64, 8, EXCL)) for _ in range(n_steps)]


def g_t3():
    t0 = time.time(); o = {"cells": {}}
    for cell in ("s215_ftself_s1", "s216_r2_s1"):
        dl = json.load(open(TS.R + cell + "/draws_log.json")); lc = {d["step"]: d for d in dl}; r = np.random.default_rng(1 + 21800); ac = au = ns = 0
        for s in range(1, 12001):
            c, U = FT.coin_step(r, 0.5, 64); ns += c; ac += int(lc[s]["coin"] == c); au += int(c and lc[s].get("U") == U)
        o["cells"][cell] = {"coin_agree": ac, "U_agree_self_steps": au, "self_steps": ns, "self_items": ns * 64}
    ph = plain_hashes(200)
    rc = subprocess.run(PY + ["-c", "import sys,json;sys.path.insert(0,'D:/Latent');sys.path.insert(0,'D:/Latent/scratch');import gate_s219 as G;print(json.dumps(G.plain_hashes(200)))"],
                        env=ENV, capture_output=True, text=True, cwd="D:/Latent")
    sep = json.loads(rc.stdout.strip().split("\n")[-1]); o["plain_hash_200"] = {"in_process_eq_separate_process": ph == sep, "first": ph[0]}
    o["PASS"] = bool(all(v["coin_agree"] == 12000 and v["U_agree_self_steps"] == v["self_steps"] == 5992 for v in o["cells"].values()) and ph == sep)
    o["sec"] = round(time.time() - t0); return save("t3_streams", o)


def check_draws_log(path, steps):
    """the landed / smoke draws_log vs the replays (G-T3): coins + U vs the round-1 latent cell, plain_hash vs the base stream's replay."""
    dl = {d["step"]: d for d in json.load(open(path))}; lat = {d["step"]: d for d in json.load(open(TS.R + "s215_ftself_s1/draws_log.json"))}; ph = plain_hashes(min(steps, 200))
    return {"steps": steps, "coin_agree": sum(dl[s]["coin"] == lat[s]["coin"] for s in range(1, steps + 1)),
            "U_agree": sum(dl[s].get("U") == lat[s].get("U") for s in range(1, steps + 1) if dl[s]["coin"]),
            "plain_hash_agree": sum(dl[s]["plain_hash"] == ph[s - 1] for s in range(1, min(steps, 200) + 1)), "self_steps": sum(dl[s]["coin"] for s in range(1, steps + 1))}


def precompute_self_counts(pool_ns, coins, bs=64):
    """the realised per-n self counts fixed by the pool order and the coin stream (G-T3: precomputed before launch)."""
    c = 0; out = {}
    for coin in coins:
        if coin:
            for n in pool_ns[c:c + bs]:
                out[str(n)] = out.get(str(n), 0) + 1
            c += bs
    return out, c


# ------------------------------------------------------------------------------------------------ t0
def g_t0():
    t0 = time.time(); o = {}; d = fresh(TMP + "/t0"); fp = synth_pool(d, (9, 10, 11, 12), 12, 21902); recs, meta = TS.load_pool(fp); o["roundtrip_records"] = len(recs)
    bad = {}
    def variant(name, fn, meta_fn=None):
        dv = fresh(TMP + "/t0/" + name); lines = open(fp, encoding="utf-8").read().split("\n")[:-1]; lines = fn(lines)
        open(dv + "/pool.jsonl", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        sha = TS.file_sha256(dv + "/pool.jsonl"); json.dump({"sha256": meta_fn(sha) if meta_fn else sha}, open(dv + "/pool_meta.json", "w", newline="\n"))
        try:
            TS.load_pool(dv + "/pool.jsonl"); bad[name] = "LOADED"
        except AssertionError as e:
            bad[name] = "refused: %s" % str(e)[:80]
    jl = lambda f: lambda L: [json.dumps(f(json.loads(x))) for x in L]  # noqa: E731
    variant("gold_root_key", jl(lambda r: dict(r, root=1))); variant("missing_values", jl(lambda r: {k: v for k, v in r.items() if k != "values"}))
    variant("label_not_last", jl(lambda r: dict(r, chain_label=(r["chain_label"] + 1) % 11))); variant("n_wrong", jl(lambda r: dict(r, n=r["n"] + 1)))
    variant("value_out_of_range", jl(lambda r: dict(r, values=r["values"][:-1] + [11], chain_label=11))); variant("sha_mismatch", lambda L: L, lambda s: "0" * 64)
    o["corrupted_refused"] = bad
    g = N.make_trunk({"scratch": "2,128,2", "seed": 1}, "cpu"); ac = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
    lf = FT.make_loss(g, "cpu", ac); cnt = {"cot_ids": 0, "cot_steps": 0}; orig = (N.cot_ids, N.cot_steps)
    def wrap(nm, f):
        def w(*a, **k):
            cnt[nm] += 1; return f(*a, **k)
        return w
    N.cot_ids, N.cot_steps = wrap("cot_ids", orig[0]), wrap("cot_steps", orig[1])
    try:
        self_its = [TS.self_item(r) for r in recs[:8]]; assert not any("root" in it for it in self_its); lf(self_its); c_self = dict(cnt)
        rng = np.random.default_rng(3); lf(N.train_items(rng, 4, 8, set())); c_plain = {k: cnt[k] - c_self[k] for k in cnt}
    finally:
        N.cot_ids, N.cot_steps = orig
    o["gold_calls_self_batch"] = c_self; o["gold_calls_plain_batch"] = c_plain
    code = ("import sys,json;sys.path.insert(0,'D:/Latent');import boxes.nested170 as N, boxes.textself as TS, numpy as np\n"
            "r_,t=N.generate(np.random.default_rng(4),9);s=N.render(r_);gv=TS.gold_values(s);TS.patch_gold();out={}\n"
            "for nm,f in (('apply',lambda:N.apply('+',1,2)),('value',lambda:N.value(r_)),('eval_tokens',lambda:N.eval_tokens(s)),('cot_steps',lambda:N.cot_steps(r_)),('cot_ids',lambda:N.cot_ids(r_)),('reductions',lambda:N.reductions(r_)),('gold_values',lambda:TS.gold_values(s))):\n"
            "  try:\n    f();out[nm]='NO RAISE'\n  except RuntimeError as e:\n    out[nm]='raises'\n"
            "ct=TS.compose_T(s,gv);fv=TS.chain_form(s,ct)[0];cs=TS.c_struct(s);out['gold_free_still_run']=bool(fv and cs==len(ct))\n"
            "g=N.make_trunk({'scratch':'2,128,2','seed':1},'cpu');lab=TS.label_reroot(g,[s]);gen=TS.gen_reroot(g,[s]);out['labellers_run_under_patch']=[lab[0]['keep'],gen[0]['keep']]\n"
            "print(json.dumps(out))")
    rc = subprocess.run(PY + ["-c", code], env=ENV, capture_output=True, text=True, cwd="D:/Latent"); o["gold_patch"] = json.loads(rc.stdout.strip().split("\n")[-1]) if rc.returncode == 0 else rc.stderr[-800:]
    gp = o["gold_patch"]
    o["PASS"] = bool(o["roundtrip_records"] == 48 and all(v.startswith("refused") for v in bad.values()) and c_self == {"cot_ids": 0, "cot_steps": 0} and c_plain["cot_ids"] > 0
                     and isinstance(gp, dict) and all(gp[k] == "raises" for k in ("apply", "value", "eval_tokens", "cot_steps", "cot_ids", "reductions", "gold_values")) and gp["gold_free_still_run"])
    shutil.rmtree(d); o["sec"] = round(time.time() - t0); return save("t0_nogold", o)


# ------------------------------------------------------------------------------------------------ t6 / t8 / t18
def g_t6():
    t0 = time.time(); g, sha = TS.load_model(TS.BASE_MODEL, "cpu"); x = torch.tensor(np.random.default_rng(21903).integers(0, 50257, (2, 40)))
    with torch.no_grad():
        h, _ = g(g.embed(x), None); lg = g.logits(h)
    o = {"model": TS.BASE_MODEL, "sha256": sha, "sha_ok": sha == TS.BASE_SHA256, "strict": True, "forward_shape": list(lg.shape), "finite": bool(torch.isfinite(lg).all()),
         "input": "random token ids (rng 21903), never a test item"}
    o["PASS"] = bool(o["sha_ok"] and o["finite"]); o["sec"] = round(time.time() - t0); del g; return save("t6_load", o)


def g_t8():
    t0 = time.time(); g, _ = TS.load_model(TS.BASE_MODEL, "cpu"); g.train(); ac = lambda: torch.autocast("cuda", enabled=False)  # noqa: E731
    d = fresh(TMP + "/t8"); fp = synth_pool(d, (9, 12), 2, 21904); recs, _ = TS.load_pool(fp); lf = FT.make_loss(g, "cpu", ac); o = {}
    for name, its in (("self", [TS.self_item(r) for r in recs]), ("plain", N.train_items(np.random.default_rng(5), 4, 8, set()))):
        g.zero_grad(set_to_none=True); st = {}; L = lf(its, st); rq = bool(L.requires_grad); L.backward()
        nz = {k: (v.grad is not None and float(v.grad.float().norm()) > 0) for k, v in g.p.items()}
        o[name] = {"loss": float(L), "requires_grad": rq, "tensors": len(nz), "zero_grad_tensors": [k for k, v in nz.items() if not v],
                   "value_ce": st.get("value_sum", 0) / st["n_value"] if st.get("n_value") else None, "copy_ce": st.get("copy_sum", 0) / st["n_copy"] if st.get("n_copy") else None,
                   "train_tokens": st.get("tok")}
    o["PASS"] = bool(all(o[k]["requires_grad"] and not o[k]["zero_grad_tensors"] for k in ("self", "plain")) and o["self"]["value_ce"] is not None)
    shutil.rmtree(d); o["sec"] = round(time.time() - t0); del g; return save("t8_gradflow", o)


class StubG:
    """a GPT2KV stand-in that never emits EOT (argmax = ' 3') and records the largest position row it is fed."""
    def __init__(self):
        self.Wt = torch.zeros(50257, 4); self.max_row = -1

    def w(self, k):
        return self.Wt

    def embed(self, ids):
        return torch.zeros(ids.shape[0], ids.shape[1], 4)

    def __call__(self, x, past=None):
        P0 = 0 if past is None else past; T = x.shape[1]; self.max_row = max(self.max_row, P0 + T - 1); assert P0 + T - 1 <= 1023, "row 1,024 fed"
        return torch.zeros(x.shape[0], T, 4), P0 + T

    def logits(self, h):
        lg = torch.zeros(h.shape[0], 50257); lg[:, TS.NUM_IDS[3]] = 1.0; return lg


def g_t18():
    t0 = time.time(); import boxes.text_eval as TE_; o = {"cases": {}}
    for P0, C in ((900, 200), (1000, 20), (10, 20), (1020, 4)):
        st = StubG(); budget = min(C + 50, TS.CTX - P0); out, _, fed = TS.greedy_batch(st, [[TS.EOT] * P0] * 2, budget)
        o["cases"]["P%d_C%d" % (P0, C)] = {"budget": budget, "produced": len(out[0]), "max_row_fed": st.max_row, "fed": fed, "answer": TS.cot_answer(out[0]),
                                           "ok": len(out[0]) == budget and st.max_row <= 1023 and TS.cot_answer(out[0]) == -1 and EOT_free(out[0])}
    its = TS.gt18_items(); P = [len(s) + 2 for s in its]; C = [TS.c_struct(s) for s in its]; feas = [i for i in range(len(its)) if P[i] + C[i] <= TS.CTX]
    ov = [i for i in range(len(its)) if P[i] + C[i] > TS.CTX]; lon = max(feas, key=lambda i: P[i] + C[i])
    st = StubG(); sel = [lon] + ov[:1]; out, PP, CC = TE_.free_generate(st, [its[i] for i in sel], 32)
    recs = [TE_.item_record(j, {"syms": its[i], "n": 24, "t": 0}, o_, p_, c_) for j, (i, o_, p_, c_) in enumerate(zip(sel, out, PP, CC))]
    o["eval_loop_stub"] = {"items": recs, "max_row_fed": st.max_row, "longest_P_plus_C": P[lon] + C[lon], "overflow_items_in_23195": len(ov)}
    o["PASS"] = bool(all(c["ok"] for c in o["cases"].values()) and st.max_row <= 1023 and recs[0]["answer"] == -1 and not recs[0]["overflow"] and
                     (len(recs) < 2 or (recs[1]["overflow"] and recs[1]["answer"] == -1)))
    o["sec"] = round(time.time() - t0); return save("t18_stub", o)


def EOT_free(o):
    return TS.EOT not in o


# ------------------------------------------------------------------------------------------------ t7
def g_t7():
    t0 = time.time(); o = {"nested170_sha256": TS.file_sha256("boxes/nested170.py")}; o["sha_pinned_ok"] = o["nested170_sha256"] == TS.NESTED170_SHA256
    common = ["--arm", "textcot", "--steps", "12000", "--lr", "3e-05", "--bs", "8", "--log_every", "1", "--device", "cpu"]
    dn, dt = fresh(TMP + "/t7_n170"), fresh(TMP + "/t7_ts")
    rcn = run(PY + ["-m", "boxes.nested170"] + common + ["--time_steps", "3", "--out", dn], dn + "/stdout.txt")
    rct = run(PY + FTM + common + ["--p_self", "0", "--stop_at", "3", "--no_final_eval", "--out", dt], dt + "/stdout.txt")
    o["rc"] = {"nested170": rcn, "textself_ft": rct}
    ln = {r["step"]: r["loss"] for r in map(json.loads, open(dn + "/log.jsonl")) if "loss" in r}; lt = {r["step"]: r["loss"] for r in map(json.loads, open(dt + "/log.jsonl")) if "loss" in r}
    o["losses"] = {str(s): {"nested170": ln.get(s), "textself_ft": lt.get(s), "bitwise": ln.get(s) is not None and struct.pack("<d", ln[s]) == struct.pack("<d", lt.get(s, float("nan")))}
                   for s in (1, 2, 3)}
    an, at = json.load(open(dn + "/args.json")), json.load(open(dt + "/args.json")); M = meta_t(); off = {k: FT.build_parser().get_default(k) for k in FT.FT_KEYS}
    off["no_final_eval"] = True; off["stop_at"] = 3
    diff = {k: [an.get(k), at.get(k)] for k in sorted(set(an) | set(at)) if k not in M and k not in FT.FT_KEYS and an.get(k) != at.get(k)}
    o["args"] = {"non_meta_diff_excl_ft": diff, "ft_keys": {k: at.get(k) for k in FT.FT_KEYS}, "ft_at_off_values": all(at.get(k) == off[k] for k in FT.FT_KEYS),
                 "time_steps_note": "nested170 ran with --time_steps 3 (META), textself_ft with --stop_at 3 (META / FT); both META"}
    o["eval_files_textself_ft"] = sorted(os.path.basename(p) for p in glob.glob(dt + "/eval*.json"))
    o["PASS"] = bool(o["sha_pinned_ok"] and rcn[0] == 0 and rct[0] == 0 and all(v["bitwise"] for v in o["losses"].values()) and not diff and o["args"]["ft_at_off_values"]
                     and not o["eval_files_textself_ft"])
    o["sec"] = round(time.time() - t0); return save("t7_plain_equiv", o)


# ------------------------------------------------------------------------------------------------ pipe
def g_pipe():
    t0 = time.time(); fresh(PIPE); o = {"steps": {}}
    st = [("gates", PY + ["scratch/s219_build_pool.py", "--round", "1", "--stage", "gates", "--out_root", PIPE, "--scale", "0.01", "--device", "cpu"]),
          ("label", PY + ["scratch/s219_gate.py", "--round", "1", "--stage", "label", "--labeller", "primary", "--out_root", PIPE, "--device", "cpu"]),
          ("score", PY + ["scratch/s219_gate.py", "--round", "1", "--stage", "score", "--out_root", PIPE]),
          ("pool", PY + ["scratch/s219_build_pool.py", "--round", "1", "--stage", "pool", "--out_root", PIPE, "--scale", "0.0006", "--device", "cpu"])]
    for name, cmd in st:
        rc, sec = run(cmd, "%s/%s.log" % (PIPE, name)); o["steps"][name] = {"rc": rc, "sec": sec, "tail": open("%s/%s.log" % (PIPE, name)).read()[-600:]}
        if rc != 0:
            break
    if all(v["rc"] == 0 for v in o["steps"].values()) and len(o["steps"]) == 4:
        eg = json.load(open(PIPE + "/s219_gate_t1_s1/eval_gate.json")); pm = json.load(open(PIPE + "/s219_pool_t1_s1/pool_meta.json")); recs, _ = TS.load_pool(PIPE_POOL)
        gi = json.load(open(PIPE + "/s219_gate_t1_s1/gate_items.json")); gsy = {tuple(it["syms"]) for w in gi["windows"].values() for it in w["items"]}
        o["eval_gate"] = {"GATE_T1": eg.get("GATE_T1"), "pooled": {k: v for k, v in eg["pooled_by_window"]["W1"].items() if not k.endswith("ci95")}, "equiv": eg["equiv"],
                          "equiv_full": eg["equiv_full"], "compose": eg["compose"], "labeller_full": eg["labeller_full"], "kept_vs_drawn": eg["kept_vs_drawn"]}
        o["pool"] = {"n_records": len(recs), "sha_ok": True, "keep_n": pm["keep_n"], "T_max_by_n": pm["T_max_by_n"], "L0_max_by_n": pm["L0_max_by_n"], "excl_sizes": {k: v["size"] for k, v in pm["excl"].items()},
                     "collide_n": pm["collide_n"], "pool_hits_own_gates": sum(tuple(r["syms"]) in gsy for r in recs), "precondition_counts": pm["excl_meta"]["precondition_counts"],
                     "labeller_sha256": pm["labeller_sha256"]}
        o["PASS"] = bool(eg.get("GATE_T1") is True and all(v["PASS"] for v in eg["equiv"].values()) and all(v["PASS"] for v in eg["compose"].values()) and len(recs) == 4 * 58
                         and o["pool"]["pool_hits_own_gates"] == 0 and all(v == 200 for d in pm["excl_meta"]["precondition_counts"].values() for v in d.values()))
    else:
        o["PASS"] = False
    o["sec"] = round(time.time() - t0); return save("pipe_b1_b2", o)


# ------------------------------------------------------------------------------------------------ t4 / t5 / t9t10
def t4_cmd(out, extra):
    return PY + FTM + ["--arm", "textcot", "--init_from", TS.BASE_MODEL, "--steps", "12000", "--lr", "3e-05", "--self_pool", PIPE_POOL, "--p_self", "0.5",
                 "--self_micro", "2", "--no_final_eval", "--draws_log_steps", "12000", "--auto_micro", "--args_ref", BASE_ARGS, "--device", "cpu", "--ckpt_every", "1",
                 "--keep_ckpt_steps", "0", "--out", out] + extra


def _sd_eq(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _opt_eq(a, b):
    if a["param_groups"] != b["param_groups"]:
        return False
    return all(set(a["state"][k]) == set(b["state"][k]) and all(torch.equal(a["state"][k][q], b["state"][k][q]) for q in a["state"][k]) for k in a["state"])


def g_t4():
    t0 = time.time(); o = {}; da, db = fresh(TMP + "/t4_a"), fresh(TMP + "/t4_b")
    o["rc_a"] = run(t4_cmd(da, ["--stop_at", "3"]), da + "/stdout.txt"); o["rc_b"] = run(t4_cmd(db, ["--stop_at", "2"]), db + "/stdout_1.txt")
    cl = torch.load(db + "/ckpt_last.pt", weights_only=False, map_location="cpu"); cp = torch.load(db + "/ckpt_prev.pt", weights_only=False, map_location="cpu"); o["b_ckpt_steps"] = [cp["step"], cl["step"]]; del cl, cp
    dc = fresh(TMP + "/t4_c")                                               # G-T13 item 4: a draws_log that does not cover the checkpoint must be refused
    for f in ("ckpt_last.pt", "args.json", "log.jsonl"):
        shutil.copy(db + "/" + f, dc + "/" + f)
    json.dump([d for d in json.load(open(db + "/draws_log.json")) if d["step"] < 2], open(dc + "/draws_log.json", "w", newline="\n"))
    o["rc_c_truncated_resume"] = run(t4_cmd(dc, ["--stop_at", "3", "--resume", dc + "/ckpt_last.pt"]), dc + "/stdout.txt")
    o["c_refused"] = o["rc_c_truncated_resume"][0] != 0 and "does not cover the resumed range" in open(dc + "/stdout.txt").read()
    o["no_tmp_files"] = not glob.glob(db + "/*.tmp") and not glob.glob(da + "/*.tmp") if os.path.exists(da) else None
    shutil.rmtree(dc)
    o["rc_b_resume"] = run(t4_cmd(db, ["--stop_at", "3", "--resume", db + "/ckpt_last.pt"]), db + "/stdout_2.txt")
    base = torch.load(TS.BASE_MODEL, weights_only=True, map_location="cpu"); k0 = torch.load(da + "/ckpt_model_0.pt", weights_only=True, map_location="cpu")["model"]; o["step0_eq_init"] = _sd_eq(base, k0); del base, k0
    out_a = open(da + "/stdout.txt").read(); o["init_from_line"] = [ln for ln in out_a.split("\n") if ln.startswith("INIT_FROM")]
    o["init_from_json"] = json.load(open(da + "/init_from.json"))
    la = [r for r in map(json.loads, open(da + "/log.jsonl")) if "loss" in r]; lb = [r for r in map(json.loads, open(db + "/log.jsonl")) if "loss" in r]
    ra = next(r for r in la if r["step"] == 3); rb = next(r for r in lb if r["step"] == 3)
    o["step3"] = {"a_loss_last": ra["loss_last"], "b_resumed_loss_last": rb["loss_last"], "bitwise": struct.pack("<d", ra["loss_last"]) == struct.pack("<d", rb["loss_last"]),
                  "a_cursor": ra["pool_cursor"], "b_cursor": rb["pool_cursor"], "a_self_steps": ra["n_self_steps"], "b_self_steps": rb["n_self_steps"]}
    ca = torch.load(da + "/ckpt_last.pt", weights_only=False, map_location="cpu"); cb = torch.load(db + "/ckpt_last.pt", weights_only=False, map_location="cpu")
    names = list(ca["model"].keys()); md = {k: float((ca["model"][k] - cb["model"][k]).abs().max()) for k in names if not torch.equal(ca["model"][k], cb["model"][k])}
    od = {}
    for i, st in ca["opt"]["state"].items():
        for q, v in st.items():
            if torch.is_tensor(v) and not torch.equal(v, cb["opt"]["state"][i][q]):
                od["%s.%s" % (names[int(i)], q)] = float((v.float() - cb["opt"]["state"][i][q].float()).abs().max())
    # the CPU backward of the tied wte embedding gather (index accumulate) is not bitwise repeatable (scratch/_tmp/s219/det_check.py: 1.2e-7 between two
    # identical backward passes, the other 147 tensors bitwise); so the step-3 update may differ in wte (and its AdamW moments) only, at that scale
    allowed = set() if DET else {"p.wte_weight"}                          # Note 219-3 item 2: deterministic single-thread -> every tensor bitwise
    o["deterministic"] = DET
    o["ckpt3"] = {"steps": [ca["step"], cb["step"]], "model_diff": md, "opt_diff": od, "np_rng_eq": ca["np_rng"] == cb["np_rng"], "rng_s_eq": ca["rng_s"] == cb["rng_s"],
                  "self_state_eq": ca["self_state"] == cb["self_state"], "tok_eq": ca["tok"] == cb["tok"], "opt_groups_eq": ca["opt"]["param_groups"] == cb["opt"]["param_groups"]}
    o["ckpt3"]["model_eq"] = set(md) <= allowed and all(v <= 1e-6 for v in md.values())
    o["ckpt3"]["opt_eq"] = {k.rsplit(".", 1)[0] for k in od} <= allowed and all(v <= 1e-4 for v in od.values()) and o["ckpt3"]["opt_groups_eq"]; del ca, cb
    o["draws_log_eq"] = json.load(open(da + "/draws_log.json")) == json.load(open(db + "/draws_log.json")); o["draws_vs_replay"] = check_draws_log(da + "/draws_log.json", 3)
    pool, _ = TS.load_pool(PIPE_POOL); dl = json.load(open(da + "/draws_log.json")); pc, cur = precompute_self_counts([r["n"] for r in pool], [d["coin"] for d in dl])
    real = {}
    for d in dl:
        for n in d.get("self_n", []):
            real[str(n)] = real.get(str(n), 0) + 1
    o["self_counts"] = {"precomputed": pc, "realised": real, "eq": pc == real}
    o["eval_files"] = sorted(os.path.basename(p) for d in (da, db) for p in glob.glob(d + "/eval*.json")); o["files_a"] = sorted(os.listdir(da))
    o["PASS"] = bool(o["rc_a"][0] == 0 and o["rc_b"][0] == 0 and o["rc_b_resume"][0] == 0 and o["b_ckpt_steps"] == [1, 2] and o["step0_eq_init"] and o["init_from_line"]
                     and "missing [] unexpected []" in o["init_from_line"][0] and o["init_from_json"]["sha256"] == TS.BASE_SHA256 and o["step3"]["bitwise"]
                     and all(o["ckpt3"][k] for k in ("model_eq", "opt_eq", "np_rng_eq", "rng_s_eq", "self_state_eq", "tok_eq")) and o["draws_log_eq"]
                     and o["draws_vs_replay"]["coin_agree"] == 3 and o["draws_vs_replay"]["plain_hash_agree"] == 3 and o["self_counts"]["eq"] and not o["eval_files"] and o["c_refused"] and o["no_tmp_files"])
    for d in (da, db):                                                      # model-sized files: delete (args / logs / draws_log kept for t5)
        for p in glob.glob(d + "/*.pt"):
            os.remove(p)
    o["sec"] = round(time.time() - t0); return save("t4_init_resume", o)


TEXT_NEUTRAL_NEW = {"n_slots": 8, "anneal": "", "probe_loss_every": 0, "mem_layer": 6, "cache_lower": False, "anneal_gate": "", "csr": "", "iit": "", "iit_tau": 1, "iit_nmax": 0,
                    "grokfast": "", "gf_alpha": 0.98, "gf_lamb": 2.0, "cot_eval_bs": 64, "eval_only": False, "aux_step": 0.0, "aux_fade": "36000,48000", "kmatch_impl": "masked",
                    "add_norm": "", "aux_nopass": False, "aux_reweight": False, "aux_hold": False, "kprimes": "", "path_indep": False, "excl_min_n": 1, "scratch": "",
                    "warmup": 100, "k_window": "", "fast_add": False}


def args_diff_prelim(a, b, required):
    """TS_v1 §4.6 ARGS-DIFF, preliminary (theory/score_s219.py is the registered implementation): a = the cell's args.json, b = s170_cot_s1/args.json."""
    M = meta_t(); allowed = set(required) | {"mem_check_steps", "out", "args_ref"}
    shared = {k: (b[k], a[k]) for k in a if k in b and k not in M and k not in allowed and a[k] != b[k]}
    missing = sorted(k for k in b if k not in M and k not in a)
    new = sorted(k for k in a if k not in b and k not in M and k not in allowed and not (k in TEXT_NEUTRAL_NEW and a[k] == TEXT_NEUTRAL_NEW[k]))
    req = {k: (v, a.get(k)) for k, v in required.items() if a.get(k) != v}
    va = a.get("args_ref"); ar = {"ok": va == b.get("args_ref")}
    if not ar["ok"] and va and os.path.isfile(va):
        f = json.load(open(va)); dd = sorted(k for k in set(f) | set(b) if k not in M and k != "args_ref" and f.get(k) != b.get(k)); ar = {"ok": not dd, "diff": dd[:10]}
    return {"shared_diffs": shared, "missing_base_keys": missing, "unexpected_new_keys": new, "required_bad": req, "args_ref": ar,
            "ok": not shared and not missing and not new and not req and ar["ok"]}


def g_t5():
    t0 = time.time(); a = json.load(open(TMP + "/t4_a/args.json")); b = json.load(open(BASE_ARGS))
    req = {"steps": 12000, "lr": 3e-05, "init_from": TS.BASE_MODEL, "self_pool": PIPE_POOL, "p_self": 0.5, "self_micro": 2, "no_final_eval": True, "draws_log_steps": 12000,
           "mem_check_self": False}
    o = {"real_smoke": args_diff_prelim(a, b, req), "required_for_smoke": req, "mutations": {}}
    d = fresh(TMP + "/t5"); bad_ref = dict(b, bs=32); json.dump(bad_ref, open(d + "/ref_bad.json", "w", newline="\n"))
    for name, mut in (("lr_1e-4", {"lr": 1e-4}), ("warmup_1000", {"warmup": 1000}), ("no_final_eval_false", {"no_final_eval": False}), ("drop_bs", "bs"),
                      ("args_ref_one_key_changed", {"args_ref": d + "/ref_bad.json"}), ("unknown_new_key", {"foo": 1}), ("cot_eval_bs_32", {"cot_eval_bs": 32})):
        m = {k: v for k, v in a.items() if k != mut} if isinstance(mut, str) else dict(a, **mut); o["mutations"][name] = args_diff_prelim(m, b, req)["ok"]
    o["note"] = "self_pool's required value is the smoke pool here; the registered T1 value is boxes/runs_fsq/s219_pool_t1_s1/pool_t1.jsonl"
    o["PASS"] = bool(o["real_smoke"]["ok"] and not any(o["mutations"].values())); o["sec"] = round(time.time() - t0); return save("t5_args_diff", o)


def g_t9t10():
    t0 = time.time(); o = {}; d9, d10 = fresh(TMP + "/t9"), fresh(TMP + "/t10")
    base = [x for x in t4_cmd("X", []) if True]
    c9 = [x for x in base]; c9[c9.index("X")] = d9; c9 += ["--mem_check_self", "--mem_check_steps", "2", "--bs", "8"]
    c10 = [x for x in base]; c10[c10.index("X")] = d10; c10 += ["--time_steps", "2", "--bs", "8"]
    o["rc_mem"] = run(c9, d9 + "/stdout.txt"); o["rc_time"] = run(c10, d10 + "/stdout.txt")
    o["mem_check_self"] = json.load(open(d9 + "/mem_check_self.json")) if os.path.exists(d9 + "/mem_check_self.json") else None
    o["timing"] = json.load(open(d10 + "/timing.json")) if os.path.exists(d10 + "/timing.json") else None
    o["files"] = {"t9": sorted(os.listdir(d9)), "t10": sorted(os.listdir(d10))}; o["eval_files"] = [f for v in o["files"].values() for f in v if f.startswith("eval")]
    o["PASS"] = bool(o["rc_mem"][0] == 0 and o["rc_time"][0] == 0 and o["mem_check_self"] and o["mem_check_self"]["pass"] and o["timing"] and not o["eval_files"]
                     and "args.json" in o["files"]["t9"] and "args.json" in o["files"]["t10"] and o["mem_check_self"]["n_top"] == 12
                     and o["timing"].get("deep_n") == 12 and o["timing"].get("deep_T_max") == o["mem_check_self"]["T_max"])   # Note 219-4 (3): G-T10 = the deepest shape
    for dd in (d9, d10):
        for p in glob.glob(dd + "/*.pt"):
            os.remove(p)
    o["sec"] = round(time.time() - t0); return save("t9t10_paths", o)


# ------------------------------------------------------------------------------------------------ t16
def g_t16():
    t0 = time.time(); sys.path.insert(0, "D:/Latent/docs/research_textself_2026-10-18/scripts")
    import ts_label_check as TLC                                            # read-only import (its main() is never called)
    torch.set_num_threads(2); g, _ = TS.load_model(TS.BASE_MODEL, "cpu"); pst = TS.precond_streams()[23180]
    draft = json.load(open("docs/research_textself_2026-10-18/scripts/ts_label_check_s1.json")); o = {"by_n": {}}
    plan = [(9, 25), (10, 25), (11, 25), (12, 25), (13, 10), (16, 10)]
    for n, k in plan:
        sy = pst[n][:k]; roots = [TS.parse_tree(s) for s in sy]; gv = [TS.gold_values(s) for s in sy]
        tf = TS.label_reroot(g, sy, TS.LABEL_BS); dr = TLC.label_reroot(g, roots, 50); ge = TS.gen_reroot(g, sy)
        rec = draft["items"]["%d_reroot" % n][:k]
        row = {"items": k, "keep_tf": float(np.mean([x["keep"] for x in tf])),
               "tf_eq_draft_code": sum(a_["keep"] == b_["keep"] and a_["values"] == b_["values"] and a_["fail_step"] == b_["fail_step"] for a_, b_ in zip(tf, dr)),
               "tf_eq_gen": sum(a_["keep"] == b_["keep"] and a_["values"] == b_["values"] and a_["fail_step"] == b_["fail_step"] for a_, b_ in zip(tf, ge)),
               "tf_eq_draft_json": sum([int(a_["keep"]), a_["fail_step"], (int(a_["values"] == gv[i]) if a_["keep"] else None)] == list(r_) for i, (a_, r_) in enumerate(zip(tf, rec))),
               "gen_tokens_eq_compose": sum(int([t for s_ in b_["steps"] for t in s_] == TS.compose_T(sy[i], b_["values"])) for i, b_ in enumerate(ge) if b_["keep"]),
               "gen_kept": sum(b_["keep"] for b_ in ge), "gen_min_margin": float(min(b_["min_margin"] for b_ in ge)), "tf_min_margin": float(min(a_["min_margin"] for a_ in tf))}
        if n in (10, 11):
            lf = TS.label_full(g, sy[:10], TS.LABEL_BS); gf = TS.gen_full(g, sy[:10]); dfl = TLC.label_full(g, roots[:10], 50)
            row["full"] = {"items": 10, "keep": float(np.mean([x["keep"] for x in lf])), "tf_eq_gen": sum(a_["keep"] == b_["keep"] and a_["values"] == b_["values"] for a_, b_ in zip(lf, gf)),
                           "tf_eq_draft_code": sum(a_["keep"] == b_["keep"] and a_["values"] == b_["values"] for a_, b_ in zip(lf, dfl))}
        o["by_n"][str(n)] = row; print("T16", n, row, flush=True); save("t16_equiv_partial", o)
    o["PASS"] = bool(all(r["tf_eq_draft_code"] == r["items"] and r["tf_eq_gen"] == r["items"] and r["tf_eq_draft_json"] == r["items"] and r["gen_tokens_eq_compose"] == r["gen_kept"]
                         and (("full" not in r) or (r["full"]["tf_eq_gen"] == 10 and r["full"]["tf_eq_draft_code"] == 10)) for r in o["by_n"].values()))
    o["sec"] = round(time.time() - t0); del g; return save("t16_equiv", o)


# ------------------------------------------------------------------------------------------------ t17
def g_t17():
    t0 = time.time(); o = {}; ts = time.time(); S, smeta = TS.exclusion_sets(); o["exclusion"] = {"sizes": {k: len(v) for k, v in S.items()}, "union": len(set().union(*S.values())),
                                                                                                  "sec": round(time.time() - ts), "precondition_counts": smeta["precondition_counts"]}
    del S
    for name, comp in (("T1_388k", {9: 97000, 10: 97000, 11: 97000, 12: 97000}), ("T2_390k", {**{n: 32500 for n in (9, 10, 11, 12)}, **{n: 65000 for n in (13, 14, 15, 16)}})):
        d = fresh(TMP + "/t17_" + name); fp = d + "/pool.jsonl"; rng = np.random.default_rng(21905); ts = time.time()
        with open(fp, "w", encoding="utf-8", newline="\n") as f:
            for n, k in comp.items():
                for _ in range(k):
                    s_ = N.render(N.generate(rng, n)[0]); f.write(json.dumps(TS.record(s_, [int(x) for x in rng.integers(0, 11, n)]), separators=(",", ":")) + "\n")
        json.dump({"sha256": TS.file_sha256(fp)}, open(d + "/pool_meta.json", "w", newline="\n")); t_w = time.time() - ts; ts = time.time()
        pool, _ = TS.load_pool(fp); t_l = time.time() - ts; ts = time.time()
        n_top = max(r["n"] for r in pool); deep = sorted([r for r in pool if r["n"] == n_top], key=lambda r: -len(r["syms"]))[:64]; t_d = time.time() - ts; ts = time.time()
        for c in range(0, 64 * 200, 64):
            [TS.self_item(r) for r in pool[c:c + 64]]
        t_s = time.time() - ts; ts = time.time(); [TS.compose_T(r["syms"], r["values"]) for r in deep]; t_c = time.time() - ts
        o[name] = {"records": len(pool), "bytes": os.path.getsize(fp), "write_sec": round(t_w), "load_sec": round(t_l, 1), "deep_select_sec": round(t_d, 2),
                   "slice_200_steps_sec": round(t_s, 3), "compose_64_deepest_sec": round(t_c, 3), "deep_T_max": max(len(r["syms"]) + 2 + TS.c_struct(r["syms"]) for r in deep)}
        del pool; shutil.rmtree(d); print("T17", name, o[name], flush=True)
    o["PASS"] = bool(o["T1_388k"]["load_sec"] < 600 and o["T2_390k"]["load_sec"] < 600 and o["exclusion"]["sec"] < 1800)
    o["sec"] = round(time.time() - t0); return save("t17_scale", o)


# ------------------------------------------------------------------------------------------------ evals
def g_evals():
    t0 = time.time(); d = fresh(TMP + "/evals"); o = {}
    c1 = PY + ["-m", "boxes.text_eval", "--run", TS.R + "s219_base_s1", "--part", "text", "--cot_eval_bs", "32", "--device", "cpu", "--out_dir", d, "--ns", "1,2,9", "--limit_per_n", "3"]   # never n >= 17 test items (TS_v1 §2.4 item 5)
    o["rc_text"] = run(c1, d + "/text.log")
    if o["rc_text"][0] == 0:
        e = json.load(open(d + "/eval_text.json")); TD = N.test_set(1000, n_max=24)
        o["text"] = {"meta": e["meta"], "acc_by_n": e["acc_by_n"], "form_valid_by_n": e["form_valid_by_n"], "items": e["items"][:3] + e["items"][-3:],
                     "items_sha256_ok": e["meta"]["items_sha256"] == hashlib.sha256(json.dumps([[it["syms"], int(it["t"])] for it in TD], separators=(",", ":"), ensure_ascii=True).encode()).hexdigest(),
                     "answers_eq_lenient_rule": all(r["answer"] == r["y"] or not r["strict"] for r in e["items"])}
    c2 = PY + ["-m", "boxes.text_eval", "--run", TS.R + "s219_base_s1", "--part", "smoke", "--cot_eval_bs", "32", "--device", "cpu", "--out_dir", d, "--smoke_rows", "2"]
    o["rc_smoke"] = run(c2, d + "/smoke.log")
    if o["rc_smoke"][0] == 0:
        s = json.load(open(d + "/eval_text_smoke.json")); o["smoke"] = {k: s[k] for k in ("items", "feasible", "cap_bound", "projection", "PASS_peak_le_6")}
    run_dir = TMP + "/t4_a"; gi = PIPE + "/s219_gate_t1_s1/gate_items.json"
    c3 = PY + ["-m", "boxes.text_eval", "--run", run_dir, "--part", "textself", "--round", "1", "--parts", "all", "--device", "cpu", "--out_dir", d, "--pool", PIPE_POOL,
               "--gate_items", gi, "--model", TS.BASE_MODEL, "--limit_per_n", "3", "--ns", "9,24"]
    o["rc_textself"] = run(c3, d + "/textself.log")
    if o["rc_textself"][0] == 0:
        t = json.load(open(d + "/eval_textself.json"))
        o["textself"] = {"label_audit": {k: v for k, v in t["label_audit"].items() if k != "kept_wrong"}, "contamination": {k: v for k, v in t["contamination"].items() if k != "excl_meta"},
                         "memo": t["memo"], "labeller_kept": t["labeller_kept"]}
    o["PASS"] = bool(o["rc_text"][0] == 0 and o["rc_smoke"][0] == 0 and o["rc_textself"][0] == 0 and o["text"]["items_sha256_ok"] and o["textself"]["contamination"]["CONTAMINATED"] is False)
    o["sec"] = round(time.time() - t0); return save("evals_paths", o)


class OracleG:
    """an e2e stand-in for a text model (CPU, instant): emits the gold CoT for n <= depth and EOT at once above it; the GPT2KV call surface
    greedy_batch uses (w, embed, __call__ with a KV 'past', logits)."""
    def __init__(self, depth):
        self.depth = depth; self.Wt = torch.zeros(1, 1); self.cur = None; self.gold = {}

    def w(self, k):
        return self.Wt

    def embed(self, ids):
        return ids

    def __call__(self, x, past=None):
        rows = [[int(t) for t in r] for r in x.tolist()]; past = rows if past is None else [p + r for p, r in zip(past, rows)]; self.cur = past
        return torch.zeros(len(past), x.shape[1], 1), past

    def logits(self, h):
        out = torch.zeros(len(self.cur), 50257)
        for r, seq in enumerate(self.cur):
            q = seq.index(TS.EQ); pid = tuple(seq[1:q]); gen = seq[q + 1:]
            if pid not in self.gold:
                sy = [TS.ID2SYM[t] for t in pid]; self.gold[pid] = N.cot_ids(TS.parse_tree(sy)) if TS.n_of(sy) <= self.depth else [TS.EOT]
            gl = self.gold[pid]; out[r, gl[len(gen)] if len(gen) < len(gl) else TS.EOT] = 1.0
        return out


E2E = "scratch/_tmp/s219_e2e"; E2E_RF = E2E + "/boxes/runs_fsq"


def g_e2e():
    """G-T12 end to end (coordinator 2026-10-18): the REAL writers on a CPU smoke into a tree laid out as boxes/runs_fsq under scratch/_tmp/s219_e2e/, then
    theory/score_s219.py imported read-only (set_root to the tree; OUT never written; the card class 'cpu' added in memory only) and every pending /
    void field it names reported. Writers: B1 pool + B2 eval_gate (the pipe gate's outputs, re-scored into the tree), textself_ft (a complete 3-step
    run from the tree's cwd, so args.json carries the registered relative paths), text_eval part_text (OracleG stand-ins: base depth 10, T1 depth 12;
    model_sha256 = the base pin / T1's real model.pt) and text_eval --part textself (CPU parts registered mode; GPU parts smoke mode, 3 items)."""
    t0 = time.time(); o = {"steps": {}}
    if os.path.exists(E2E):
        shutil.rmtree(E2E)
    os.makedirs(E2E_RF + "/s170_cot_s1"); EA = os.path.abspath(E2E).replace("\\", "/"); RFA = EA + "/boxes/runs_fsq"
    for f in ("args.json", "eval.json", "model.pt"):
        shutil.copy(TS.BASE_RUN + "/" + f, E2E_RF + "/s170_cot_s1/" + f)
    shutil.copytree(PIPE + "/s219_gate_t1_s1", E2E_RF + "/s219_gate_t1_s1"); shutil.copytree(PIPE + "/s219_pool_t1_s1", E2E_RF + "/s219_pool_t1_s1")
    o["steps"]["gate_score"] = run(PY + ["scratch/s219_gate.py", "--round", "1", "--stage", "score", "--out_root", E2E_RF], E2E + "/gate_score.log")
    ENV2 = dict(ENV); cmd = PY + [FTP, "--arm", "textcot", "--init_from", "boxes/runs_fsq/s170_cot_s1/model.pt", "--steps", "3", "--lr", "3e-05",
                                  "--self_pool", "boxes/runs_fsq/s219_pool_t1_s1/pool_t1.jsonl", "--p_self", "0.5", "--self_micro", "2", "--no_final_eval", "--draws_log_steps",
                                  "12000", "--auto_micro", "--args_ref", "boxes/runs_fsq/s170_cot_s1/args.json", "--device", "cpu", "--out", "boxes/runs_fsq/s219_t1_s1"]
    t1 = time.time()
    with open(E2E + "/train.log", "w", newline="\n") as fh:
        o["steps"]["train"] = [subprocess.run(cmd, env=ENV2, stdout=fh, stderr=subprocess.STDOUT, cwd=EA).returncode, round(time.time() - t1)]
    import boxes.text_eval as TE_
    t1m = TS.file_sha256(E2E_RF + "/s219_t1_s1/model.pt") if os.path.exists(E2E_RF + "/s219_t1_s1/model.pt") else None
    for cell, depth, sha in (("s219_base_s1", 10, TS.BASE_SHA256), ("s219_t1_s1", 12, t1m)):
        os.makedirs(E2E_RF + "/" + cell, exist_ok=True); t1 = time.time(); outp = E2E_RF + "/" + cell + "/eval_text.json"
        res = TE_.part_text(OracleG(depth), sha, "OracleG(depth %d) e2e stand-in" % depth, "boxes/runs_fsq/" + cell, "cpu", 32, outp, list(range(1, 25)), 0)
        TE_.dump(res, outp); pp = outp.replace(".json", ".partial.json")
        if os.path.exists(pp):
            os.remove(pp)
        o["steps"]["eval_text_" + cell] = [0, round(time.time() - t1)]
    t1 = time.time()
    with open(E2E + "/textself_cpu.log", "w", newline="\n") as fh:
        o["steps"]["textself_cpu"] = [subprocess.run(PY + ["D:/Latent/boxes/text_eval.py", "--run", "boxes/runs_fsq/s219_t1_s1", "--part", "textself", "--round", "1",
                                                            "--parts", "cpu", "--device", "cpu"], env=ENV2, stdout=fh, stderr=subprocess.STDOUT, cwd=EA).returncode, round(time.time() - t1)]
    t1 = time.time()
    with open(E2E + "/textself_gpu.log", "w", newline="\n") as fh:
        o["steps"]["textself_gpu_smoke"] = [subprocess.run(PY + ["D:/Latent/boxes/text_eval.py", "--run", "boxes/runs_fsq/s219_t1_s1", "--part", "textself", "--round", "1",
                                                                  "--parts", "gpu", "--device", "cpu", "--out_dir", RFA + "/s219_t1_s1", "--limit_per_n", "3", "--ns", "9,24",
                                                                  "--model", "boxes/runs_fsq/s219_t1_s1/model.pt"], env=ENV2, stdout=fh, stderr=subprocess.STDOUT, cwd=EA).returncode,
                                            round(time.time() - t1)]
    import importlib.util
    spec = importlib.util.spec_from_file_location("score_s219_e2e", "D:/Latent/theory/score_s219.py"); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    S.OUT = EA + "/score_s219_e2e.json"; S.set_root(RFA, EA); S.CARDS = S.CARDS + ("cpu",); S.CARD_CLASS = "cpu"; S.LATENT_N50_BY_CARD["cpu"] = S.LATENT_N50
    out = S.score(); json.dump(out, open(EA + "/score_s219_e2e.json", "w", newline="\n"), indent=1, default=str); s1 = out["s1"]
    o["reading"] = S.reading(out).encode("ascii", "backslashreplace").decode("ascii")
    o["base"] = {k: s1["base"].get(k) for k in ("status", "label", "n50", "DELTA_0", "BASE_PARITY", "missing")}; o["base_eval_checks"] = (s1["base"].get("EVAL") or {}).get("checks")
    o["gate_t1"] = {k: (s1["gate_t1"] or {}).get(k) for k in ("status", "consistent", "pooled_matches", "PASS", "problem", "card_class")}
    t1c = s1["t1"]; o["t1"] = {k: t1c.get(k) for k in ("status", "label", "missing", "censor_reasons", "VOID", "dead", "CONTAMINATED", "LABEL_NOISY_T", "SEAL_BREACH", "ARGS_BAD",
                                                    "n50", "T_train", "self_value_ce", "self_copy_ce", "G_T3", "MEMO_GAP_T")}
    o["t1_eval_checks"] = (t1c.get("EVAL") or {}).get("checks"); o["t1_args_diff"] = t1c.get("ARGS_DIFF"); o["t1_contamination"] = t1c.get("CONTAMINATION")
    o["t1_label_audit"] = t1c.get("LABEL_AUDIT_T"); o["t1_seal"] = t1c.get("SEAL"); o["t2"] = {k: s1["t2"].get(k) for k in ("status", "label", "missing")}
    o["comparison"] = {k: s1["comparison"].get(k) for k in ("status", "outcome")}
    exp_args = {"steps"}                                                    # the 3-step e2e run differs from the recipe only here (+ 'dead' by construction)
    ad = t1c.get("ARGS_DIFF") or {}
    o["unexpected"] = {"t1_pending": t1c.get("status") == "PENDING", "t1_void": t1c.get("VOID"), "base_not_scored": s1["base"].get("status") != "SCORED",
                       "gate_not_consistent": not (s1["gate_t1"] or {}).get("consistent"), "args_extra_bad": sorted(set(ad.get("required_bad", {})) - exp_args) + sorted(ad.get("shared_diffs", {}))
                       + ad.get("dropped_base_keys", []) + sorted(ad.get("neutral_bad", {})) + ad.get("unexpected_new_keys", []) + ([] if (ad.get("args_ref") or {}).get("ok") else ["args_ref"]),
                       "t1_other_censors": sorted(set(t1c.get("censor_reasons") or []) - {"dead", "ARGS-DIFF"})}
    o["PASS"] = bool(all(v[0] == 0 for v in o["steps"].values()) and not o["unexpected"]["t1_pending"] and not o["unexpected"]["t1_void"] and not o["unexpected"]["base_not_scored"]
                     and not o["unexpected"]["gate_not_consistent"] and not o["unexpected"]["args_extra_bad"] and not o["unexpected"]["t1_other_censors"])
    for p in glob.glob(E2E_RF + "/*/model.pt"):                            # model-sized files: delete (the tree's json / logs are kept)
        os.remove(p)
    o["sec"] = round(time.time() - t0); return save("t12_e2e", o)


def g_b1b2rules():
    """G-T13 items 1-3 (Audit's review of 3b0bb06f): (a) s219_build_pool.window_impl on synthetic eval_gate dicts: TF-GREEDY only on a PASSED EQUIV of the
    window's own labeller (primary equiv[W] / fallback[W].equiv), generation on a failed / missing one or --gen_windows; label_bs / dtype / tf32 / card /
    missing label_settings refused. (b) s219_gate.py --stage score on the pipe's real labels with the primary corrupted (one kept value wrong at n 9) and
    the uncorrupted labels as a BASE-LABELLED fallback: the fallback block carries its own equiv / compose, window_impl then picks TF from the fallback's
    EQUIV; a fallback on another card is refused. (c) the round-2 builder refuses a missing or sha-mismatched T1 pool and accepts the real one."""
    t0 = time.time(); o = {"impl": {}}; sys.path.insert(0, "D:/Latent/scratch"); import s219_build_pool as B1
    cfg2 = B1.cfg_for(2, ""); LS = {"card": "C", "label_bs": 128, "dtype": "fp32", "tf32": False}

    def eg_(**kw):
        e = {"labeller": "T1", "equiv": {"W2": {"PASS": True}, "W1": {"PASS": True}}, "fallback": {"W1": {"labeller": "BASE", "equiv": {"PASS": False}}},
             "label_settings": {"primary": dict(LS), "fallback": dict(LS)}}
        e.update(kw); return e
    WL = {"W2": "T1", "W1": "BASE"}

    def case(name, eg, wl=WL, gen=(), card="C", expect=None):
        try:
            g_, why = B1.window_impl(eg, cfg2, wl, set(gen), 128, card); got = {w: ("gen" if w in g_ else "tf") for w in cfg2["windows"]}; o["impl"][name] = {"got": got, "why": why}
            o["impl"][name]["ok"] = got == expect
        except AssertionError as e:
            o["impl"][name] = {"refused": str(e)[:160], "ok": expect == "refused"}
    case("fallback_equiv_failed", eg_(), expect={"W2": "tf", "W1": "gen"})
    case("fallback_equiv_pass", eg_(fallback={"W1": {"labeller": "BASE", "equiv": {"PASS": True}}}), expect={"W2": "tf", "W1": "tf"})
    case("primary_equiv_failed", eg_(equiv={"W2": {"PASS": False}, "W1": {"PASS": True}}), expect={"W2": "gen", "W1": "gen"})
    case("primary_equiv_missing", eg_(equiv={}), wl={"W2": "T1", "W1": "T1"}, expect={"W2": "gen", "W1": "gen"})
    case("gen_windows_forced", eg_(fallback={"W1": {"labeller": "BASE", "equiv": {"PASS": True}}}), gen=("W2",), expect={"W2": "gen", "W1": "tf"})
    case("label_bs_64", eg_(label_settings={"primary": dict(LS, label_bs=64), "fallback": dict(LS)}), expect="refused")
    case("card_mismatch", eg_(), card="D", expect="refused")
    case("fallback_bf16", eg_(label_settings={"primary": dict(LS), "fallback": dict(LS, dtype="bf16")}), expect="refused")
    case("no_label_settings", eg_(label_settings=None), expect="refused")
    case("labeller_not_fallback", eg_(), wl={"W2": "T1", "W1": "OTHER"}, expect="refused")
    root = fresh(TMP + "/b1b2"); gd = root + "/s219_gate_t1_s1"; os.makedirs(gd)
    shutil.copy(PIPE + "/s219_gate_t1_s1/gate_items.json", gd + "/gate_items.json"); prim = json.load(open(PIPE + "/s219_gate_t1_s1/gate_labels_primary.json"))
    fb = dict(prim, labeller="boxes/runs_fsq/s170_cot_s1_AS_FALLBACK"); bad = json.loads(json.dumps(prim)); L = bad["windows"]["W1"]
    gi = json.load(open(gd + "/gate_items.json"))["windows"]["W1"]["items"]
    i9 = next(i for i, (it, x) in enumerate(zip(gi, L["tf"])) if it["n"] == 9 and x["keep"]); L["tf"][i9]["values"][-1] = (L["tf"][i9]["values"][-1] + 1) % 11
    json.dump(bad, open(gd + "/gate_labels_primary.json", "w", newline="\n")); json.dump(fb, open(gd + "/gate_labels_fallback.json", "w", newline="\n"))
    o["score_rc"] = run(PY + ["scratch/s219_gate.py", "--round", "1", "--stage", "score", "--out_root", root], root + "/score.log")
    eg = json.load(open(gd + "/eval_gate.json")); fbw = (eg.get("fallback") or {}).get("W1") or {}
    o["score"] = {"GATE_T1": eg.get("GATE_T1"), "primary_equiv_PASS": eg["equiv"]["W1"]["PASS"], "fallback_GATE": fbw.get("GATE"),
                  "fallback_equiv_PASS": (fbw.get("equiv") or {}).get("PASS"), "fallback_compose_PASS": (fbw.get("compose") or {}).get("PASS"),
                  "window_labeller": eg["window_labeller"], "label_settings": eg.get("label_settings")}
    g_, why = B1.window_impl(eg, B1.cfg_for(1, root), eg["window_labeller"], set(), 128, "cpu"); o["score"]["impl_from_real_gate"] = why
    json.dump(dict(fb, card="NVIDIA GeForce RTX 2070"), open(gd + "/gate_labels_fallback.json", "w", newline="\n"))
    o["score_rc_card_mismatch"] = run(PY + ["scratch/s219_gate.py", "--round", "1", "--stage", "score", "--out_root", root], root + "/score_card.log")
    o["card_mismatch_refused"] = o["score_rc_card_mismatch"][0] != 0 and "ONE card" in open(root + "/score_card.log").read()
    r2 = fresh(TMP + "/b1b2r2"); c2 = PY + ["scratch/s219_build_pool.py", "--round", "2", "--stage", "gates", "--out_root", r2, "--scale", "0.002", "--device", "cpu"]
    o["r2_no_t1_pool"] = run(c2, r2 + "/no_t1.log"); o["r2_no_t1_refused"] = o["r2_no_t1_pool"][0] != 0 and "T1's pool" in open(r2 + "/no_t1.log").read()
    shutil.copytree(PIPE + "/s219_pool_t1_s1", r2 + "/s219_pool_t1_s1"); shutil.copytree(PIPE + "/s219_gate_t1_s1", r2 + "/s219_gate_t1_s1")
    with open(r2 + "/s219_pool_t1_s1/pool_t1.jsonl", "a", newline="\n") as fh:
        fh.write("\n")
    o["r2_sha_bad"] = run(c2, r2 + "/sha_bad.log"); o["r2_sha_bad_refused"] = o["r2_sha_bad"][0] != 0 and "sha" in open(r2 + "/sha_bad.log").read()
    shutil.copy(PIPE + "/s219_pool_t1_s1/pool_t1.jsonl", r2 + "/s219_pool_t1_s1/pool_t1.jsonl")
    o["r2_ok"] = run(c2, r2 + "/ok.log"); o["r2_ok_line"] = [ln for ln in open(r2 + "/ok.log").read().split("\n") if ln.startswith("T1 POOL IN EXCLUSION")]
    s = o["score"]
    o["PASS"] = bool(all(v["ok"] for v in o["impl"].values()) and o["score_rc"][0] == 0 and s["GATE_T1"] is False and s["fallback_GATE"] is True
                     and s["fallback_equiv_PASS"] is True and s["fallback_compose_PASS"] is True and s["window_labeller"]["W1"] == fb["labeller"]
                     and "fallback" in (s["label_settings"] or {}) and s["impl_from_real_gate"]["W1"].startswith("TF-GREEDY") and "fallback" in s["impl_from_real_gate"]["W1"]
                     and o["card_mismatch_refused"] and o["r2_no_t1_refused"] and o["r2_sha_bad_refused"] and o["r2_ok"][0] == 0 and o["r2_ok_line"])
    o["sec"] = round(time.time() - t0); return save("b1b2_rules", o)


def g_pre(rnd):
    """pre-launch on the REAL round pool (after B1, before the deploy): G-T1 pool ∩ exclusion = ∅ (re-built set), the loader, G-T3's precomputed per-n
    self counts from the pool order + the replayed coin stream, the consumed T_train / copy maxima, and the exact TRAIN-TOKENS of both item types
    (self from the pool order, plain from the base stream's replay on the non-self steps)."""
    t0 = time.time(); cfg = TS.ROUNDS[rnd]; fp = os.path.join(cfg["pool_dir"], cfg["pool_file"]); name = "pre%d_pool" % rnd
    if not os.path.exists(fp):
        return save(name, {"PASS": None, "skipped": "no pool at %s yet" % fp})
    pool, meta = TS.load_pool(fp); S, smeta = TS.exclusion_sets(own_pool=fp); hits = {k: sum(tuple(r["syms"]) in v for r in pool) for k, v in S.items()}
    r = np.random.default_rng(1 + 21800); coins = [FT.coin_step(r, 0.5, 64)[0] for _ in range(12000)]
    pc, cur = precompute_self_counts([r_["n"] for r_ in pool], coins); used = pool[:cur]
    st = {"target_tokens": sum(TS.c_struct(r_["syms"]) for r_ in used), "token_positions": sum(len(r_["syms"]) + 2 + TS.c_struct(r_["syms"]) for r_ in used)}
    TE = N.test_set(1000); EXCL = {tuple(it["syms"]) for it in TE}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        r__, _ = N.generate(r_sp, 8); EXCL.add(tuple(N.render(r__)))
    rng = np.random.default_rng(1); pt = {"target_tokens": 0, "token_positions": 0}
    for c in coins:
        its = N.train_items(rng, 64, 8, EXCL)
        if not c:
            for it in its:
                C_ = len(N.cot_ids(it["root"])); pt["target_tokens"] += C_; pt["token_positions"] += len(it["syms"]) + 2 + C_
    labs = {w: (v["labeller"], v["labeller_sha256"], TS.file_sha256(os.path.join(v["labeller"], "model.pt"))) for w, v in meta["windows"].items()}
    o = {"pool": fp, "records": len(pool), "sha256": meta["sha256"], "excl_hits": hits, "excl_sizes": {k: len(v) for k, v in S.items()}, "self_steps": sum(coins), "consumed": cur,
         "spare": len(pool) - cur, "self_counts_precomputed": pc, "T_train_consumed": max(len(r_["syms"]) + 2 + TS.c_struct(r_["syms"]) for r_ in used),
         "L0_max_consumed": max(len(r_["syms"]) for r_ in used), "train_tokens_expected": {"self": st, "plain": pt}, "labellers": labs,
         "precondition_counts": smeta["precondition_counts"]}
    o["PASS"] = bool(sum(hits.values()) == 0 and cur <= len(pool) and sum(coins) == 5992 and all(s1 == s2 for _, s1, s2 in labs.values()))
    o["sec"] = round(time.time() - t0); return save(name, o)


GATES = {"pre1": lambda: g_pre(1), "pre2": lambda: g_pre(2), "static": g_static, "t2": g_t2, "t3": g_t3, "t0": g_t0, "t6": g_t6, "t8": g_t8, "t18": g_t18, "t7": g_t7, "pipe": g_pipe, "t4": g_t4, "t5": g_t5, "t9t10": g_t9t10,
         "t16": g_t16, "t17": g_t17, "evals": g_evals, "e2e": g_e2e, "b1b2rules": g_b1b2rules}

if __name__ == "__main__":
    for nm in (sys.argv[1:] if sys.argv[1:] != ["all"] else list(GATES)):
        try:
            GATES[nm]()
        except Exception as e:                                              # noqa: BLE001
            import traceback
            save(nm + "_ERROR", {"PASS": False, "error": repr(e), "trace": traceback.format_exc()[-3000:]})
