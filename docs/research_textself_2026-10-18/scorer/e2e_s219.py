"""End-to-end of score_s219 on the REAL writers (boxes/textself_ft.py, boxes/textself.py, boxes/text_eval.py at 3b0bb06f), CPU, in this scratch dir only.
  T1 trainer: python -m boxes.textself_ft (real) on CPU, --steps 2 --bs 4 --log_every 1 (a smoke), out ./_e2e/boxes/runs_fsq/s219_t1_s1; its 500 MB model.pt
    is deleted once train_done.json has recorded its sha256.
  eval_text: text_eval.part_text (real batching, budgets, overflow, item_record, aggregate, meta, items_sha256) with ONLY textself.greedy_batch replaced
    by a gold-chain oracle realising a target curve (24,000 GPT-2 generations are not CPU-feasible); text_eval.card is the real "cpu" (SMOKE relaxes it).
  eval_textself T1: text_eval.part_textself_cpu (real, on the real smoke pool) + the real smoke GPU parts (scratch/_tmp/s219/evals/eval_textself.json).
  GATE-T1 / pool_t1: the real smoke pipe files. Latent comparator files: read-only copies of the real ones (descriptive rows only).
  Round 2 (stand-in): T1's smoke dir copied as T2 (trainer files, eval_textself with a W2 window block copied from W1), a stand-in GATE-T2 / pool_t2 built
    from the real round-1 gate / pool files (W2 entries = W1's relabelled n 13-16; t_start restamped after T1's eval_text), and T2 eval_text from the real
    writer at two oracle curves: (i) absorbing n 13-16 (four-way + FAIL-POS r2), (ii) a(16) 0.9 (TEXT-NOT-ABSORBED-2).
  Readings: registered mode first (expected: censored by the smoke's own settings), then --smoke (score_s219.SMOKE) for rounds 1 and 2.
Nothing is written under D:/Latent. Report -> ./e2e_report.json.
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:                                                           # noqa: BLE001
    pass
import ctypes  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:                                                           # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__)); REAL = "D:/Latent"; SM = REAL + "/scratch/_tmp/s219"
E2E = os.path.join(HERE, "_e2e"); RR = os.path.join(E2E, "boxes", "runs_fsq"); REP = {"steps": {}}
sys.path.insert(0, HERE)
import score_s219 as S  # noqa: E402
import fixtures_s219 as F  # noqa: E402   (the bf16 base curve only)
sys.path.insert(0, REAL)
import torch  # noqa: E402
torch.set_num_threads(2)
import boxes.nested170 as N  # noqa: E402
import boxes.textself as TS  # noqa: E402
import boxes.text_eval as TE  # noqa: E402

L = S.LIN["s1"]; INV = {v: k for k, v in N.TID.items()}
if os.path.exists(E2E):
    shutil.rmtree(E2E)
os.makedirs(RR)


def jload(p):
    with open(p) as fh:
        return json.load(fh)


def jdump(x, d, f):
    os.makedirs(os.path.join(RR, d), exist_ok=True)
    with open(os.path.join(RR, d, f), "w", newline="\n") as fh:
        json.dump(x, fh, indent=1)


t0 = time.time()
# ---- 1. the real trainer (CPU smoke)
out_t1 = os.path.join(RR, L["t1"])
cmd = [sys.executable, "-B", "-m", "boxes.textself_ft", "--arm", "textcot", "--init_from", "boxes/runs_fsq/s170_cot_s1/model.pt", "--steps", "2", "--bs", "4",
       "--lr", "3e-5", "--self_pool", "scratch/_tmp/s219/pipe/s219_pool_t1_s1/pool_t1.jsonl", "--p_self", "0.5", "--self_micro", "2", "--no_final_eval",
       "--draws_log_steps", "12000", "--log_every", "1", "--device", "cpu", "--out", out_t1]
r = subprocess.run(cmd, cwd=REAL, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ))
REP["steps"]["trainer"] = {"rc": r.returncode, "sec": round(time.time() - t0), "stderr_tail": r.stderr[-800:]}
print("TRAINER rc", r.returncode, round(time.time() - t0), "s", flush=True)
assert r.returncode == 0, r.stderr[-3000:]
REP["trainer_files"] = sorted(os.listdir(out_t1))
if os.path.exists(os.path.join(out_t1, "model.pt")):
    os.remove(os.path.join(out_t1, "model.pt"))
td = jload(os.path.join(out_t1, "train_done.json"))

# ---- 2. the real eval writer with a gold-chain oracle
TD = N.test_set(1000, n_max=24); POS = {}
for j, it in enumerate(TD):
    POS.setdefault(tuple(it["syms"]), (it["n"], j - (it["n"] - 1) * 1000))
CURVE = {}


def oracle_greedy_batch(g, prompts, budget, stop_at_eq=False, margins=False):
    outs = []
    for p in prompts:
        syms = [INV[t] for t in p[1:-1]]; n, q = POS[tuple(syms)]
        outs.append(N.cot_ids(TS.parse_tree(syms))[:budget] if q < round(1000 * CURVE[n]) else [TS.EOT])
    return outs, [float("inf")] * len(prompts), min(len(prompts[0]) - 2 + max(len(o) for o in outs), 1023)


class StubG:
    def w(self, name):
        return torch.zeros(1)


TS.greedy_batch = oracle_greedy_batch


def eval_text(d, curve, sha):
    CURVE.clear(); CURVE.update(curve); os.makedirs(os.path.join(RR, d), exist_ok=True); t1 = time.time(); op = os.path.join(RR, d, "eval_text.json")
    if os.path.exists(op):
        os.remove(op)
    res = TE.part_text(StubG(), sha, "e2e-oracle", "boxes/runs_fsq/" + d, "cpu", 32, op, list(range(1, 25)), 0)
    TE.dump(res, op); pp = op.replace(".json", ".partial.json")
    if os.path.exists(pp):
        os.remove(pp)
    REP["steps"]["eval_text_" + d] = {"sec": round(time.time() - t1), "acc_by_n": res["acc_by_n"], "overflow_24": res["overflow_by_n"]["24"]}
    print("EVAL_TEXT", d, round(time.time() - t1), "s", flush=True)
    return res


one = lambda hi: {n: 1.0 for n in range(1, hi + 1)}  # noqa: E731
C_T1 = {**one(12), 13: 0.9, 14: 0.7, 15: 0.45, 16: 0.3, **{n: 0.05 for n in range(17, 25)}}
C_T2 = {**one(16), 17: 0.9, 18: 0.8, 19: 0.6, 20: 0.4, **{n: 0.05 for n in range(21, 25)}}
eval_text(L["base"], F.A0, S.BASE_SHA); eval_text(L["t1"], C_T1, td["model_sha256"])

# ---- 3. E2 T1: real CPU parts + real smoke GPU parts (labeller_kept_meta restamped after eval_text: the order is the stage script's)
cfg = dict(TS.ROUNDS[1]); cfg["audit_seed"] += 100_000; cfg["kept_seed"] += 100_000; t1 = time.time()
sd = TE.part_textself_cpu(out_t1, cfg, SM + "/pipe/s219_pool_t1_s1/pool_t1.jsonl"); g_sm = jload(SM + "/evals/eval_textself.json")
for k in ("memo", "labeller_kept", "labeller_kept_meta", "meta_gpu"):
    sd[k] = g_sm[k]
sd["labeller_kept_meta"] = dict(sd["labeller_kept_meta"], t_start=TE.iso())
TE.dump(sd, os.path.join(out_t1, "eval_textself.json")); REP["steps"]["textself_cpu"] = {"sec": round(time.time() - t1)}

# ---- 4. GATE-T1 / pool_t1 (real), base files, latent comparator copies (read-only sources)
g1 = jload(SM + "/pipe/s219_gate_t1_s1/eval_gate.json"); pm1 = jload(SM + "/pipe/s219_pool_t1_s1/pool_meta.json")
jdump(g1, L["gate_t1"], "eval_gate.json"); jdump(pm1, L["pool_t1"], "pool_meta.json")
shutil.copyfile(SM + "/pipe/s219_pool_t1_s1/pool_t1.jsonl", os.path.join(RR, L["pool_t1"], "pool_t1.jsonl"))
for f in ("args.json", "eval.json"):
    jdump(jload(REAL + "/boxes/runs_fsq/s170_cot_s1/" + f), L["base_model"], f)
for r_, d in L["latent"].items():
    for f in ("eval_vb.json", "eval_deep.json", "eval_selfdepth.json", "eval_chainlabel_fresh.json", "draws_log.json", "args.json"):
        if os.path.exists(os.path.join(REAL, "boxes/runs_fsq", d, f)):
            os.makedirs(os.path.join(RR, d), exist_ok=True); shutil.copyfile(os.path.join(REAL, "boxes/runs_fsq", d, f), os.path.join(RR, d, f))

S.set_root(RR, E2E); S.OUT = os.path.join(E2E, "score_s219_e2e.json")
o_reg = S.score()["s1"]
REP["registered_mode_round1"] = {"t1": o_reg["t1"]["label"][:300], "censor_reasons": o_reg["t1"].get("censor_reasons"), "comparison": o_reg["comparison"]["outcome"]}


def need(cond, what, bad):
    if not cond:
        bad.append(what)


def check_desc(c, rounds):
    """every descriptive block populated from writer output, no None at the core fields."""
    D = c.get("descriptive") or {}; bad = []
    for r in rounds:
        rng = range(13, 17) if r == 1 else range(17, 21)
        tc = D.get("TOKEN_COST_r%d" % r) or {}
        need(all(n in tc and all(S.num(tc[n][k]) for k in ("text_steps_per_correct", "text_positions_per_correct", "latent_steps_per_correct",
                                                          "latent_positions_per_correct")) for n in rng), "token_cost r%d" % r, bad)
        tt = D.get("TRAIN_TOKENS_r%d" % r) or {}
        need(S.num((tt.get("text") or {}).get("target_tokens")) and S.num((tt.get("latent") or {}).get("target_tokens")) and "exact" in (tt.get("text") or {}).get("self_method", ""),
             "train_tokens r%d" % r, bad)
        dp = D.get("DEPTH_PAIRED_r%d" % r) or {}
        need(all(n in dp and S.num(dp[n].get("mcnemar_p")) and sum(dp[n][k] for k in ("both", "text_only", "latent_only", "neither")) == 1000 for n in rng),
             "depth_paired r%d" % r, bad)
        fp = D.get("FAIL_POS_r%d" % r) or {}
        need(S.num(fp.get("pooled_share_past_T_train")) and all(n in fp and S.num(fp[n]["median_first_bad_pos"]) and S.num(fp[n]["median_first_bad_step"]) and
                                                                S.num(fp[n]["share_past_T_train"]) for n in rng), "fail_pos r%d" % r, bad)
    if 2 in rounds:
        lr = D.get("LABELLER_REACH") or {}
        need(all((lr.get(k) or {}).get("text_keep") and (lr.get(k) or {}).get("latent_keep") for k in ("T1_vs_FTSELF", "T2_vs_R2s1")) and lr["base_vs_D3T"]["text_GATE_T1_keep"],
             "labeller_reach", bad)
    lf = D.get("LATENT_FIXED_K") or {}
    need(all(S.num((lf.get(r) or {}).get("n50")) for r in ("r0", "r1", "r2")), "latent_fixed_k", bad)
    need(S.num((D.get("ROUND_0") or {}).get("delta_0_fp32")) and S.num((D.get("ROUND_0") or {}).get("delta_0_bf16")), "round_0", bad)
    return bad


S.SMOKE = True
o1 = S.score()["s1"]; c1 = o1["comparison"]
REP["smoke_round1_only"] = {"t1_status": o1["t1"]["status"], "t1_smoke_relaxed": o1["t1"].get("smoke_relaxed"), "outcome": c1["outcome"], "line": c1["line"][:700],
                            "missing_blocks": check_desc(c1, (1,)), "t2_status": o1["t2"]["status"]}
print("SMOKE R1", c1["outcome"], REP["smoke_round1_only"]["missing_blocks"], flush=True)

# ---- 5. round 2 stand-in
S.SMOKE = False
out_t2 = os.path.join(RR, L["t2"]); shutil.copytree(out_t1, out_t2)
for f in ("eval_text.json",):
    os.remove(os.path.join(out_t2, f))
sd2 = jload(os.path.join(out_t2, "eval_textself.json")); bw = sd2["label_audit"]["by_window"]; bw["W2"] = dict(bw["W1"])
TE.dump(sd2, os.path.join(out_t2, "eval_textself.json"))
g2 = json.loads(json.dumps(g1)); now = TE.iso()
for n in range(13, 17):
    g2["by_n"][str(n)] = dict(g1["by_n"][str(n - 4)], window="W2")
g2["pooled_by_window"]["W2"] = dict(g1["pooled_by_window"]["W1"]); g2["GATE_T2"] = True; g2["GATE_T1p"] = True; g2.pop("GATE_T1", None)
g2["equiv"]["W2"] = g1["equiv"]["W1"]; g2["round"] = 2; g2["t_start"] = now
jdump(g2, L["gate_t2"], "eval_gate.json")
pm2 = json.loads(json.dumps(pm1)); pm2["windows"]["W2"] = dict(pm1["windows"]["W1"], ns=[13, 14, 15, 16], gate="GATE_T2"); pm2["t_start"] = now; pm2["round"] = 2
jdump(pm2, L["pool_t2"], "pool_meta.json"); shutil.copyfile(SM + "/pipe/s219_pool_t1_s1/pool_t1.jsonl", os.path.join(RR, L["pool_t2"], "pool_t2.jsonl"))
REP["round2_variants"] = {}
for tag, curve in (("absorbing", C_T2), ("not_absorbed_2", {**C_T2, 16: 0.9})):
    S.SMOKE = False; eval_text(L["t2"], curve, td["model_sha256"])
    sd2["labeller_kept_meta"]["t_start"] = TE.iso(); TE.dump(sd2, os.path.join(out_t2, "eval_textself.json"))   # E2 GPU parts run after the model's eval_text
    S.SMOKE = True; o = S.score()["s1"]; c = o["comparison"]
    REP["round2_variants"][tag] = {"t2_status": o["t2"]["status"], "t2_label": o["t2"]["label"][:300], "t2_smoke_relaxed": o["t2"].get("smoke_relaxed"),
                                   "outcome": c["outcome"], "rows": c["rows"], "ROUND_1_ONLY": c["ROUND_1_ONLY"], "DELTA": c.get("DELTA"), "line": c["line"][:900],
                                   "missing_blocks": check_desc(c, (1, 2)), "line_has_latent_formula": "documented formula" in c["line"],
                                   "fail_pos_r2_on_line": "FAIL-POS" in c["line"]}
    print("SMOKE R2", tag, c["outcome"], REP["round2_variants"][tag]["missing_blocks"], flush=True)
S.SMOKE = False
REP["PASS"] = bool(not REP["smoke_round1_only"]["missing_blocks"] and all(not v["missing_blocks"] and v["outcome"] not in ("PENDING",) for v in REP["round2_variants"].values())
                   and REP["round2_variants"]["not_absorbed_2"]["outcome"] == "TEXT-NOT-ABSORBED-2")
REP["sec"] = round(time.time() - t0)
with open(os.path.join(HERE, "e2e_report.json"), "w", newline="\n") as fh:
    json.dump(REP, fh, indent=1, default=str)
print("E2E PASS", REP["PASS"], REP["sec"], "s", flush=True)
shutil.rmtree(E2E, ignore_errors=True)
