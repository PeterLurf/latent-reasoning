"""End-to-end of score_s219 on the REAL writers (boxes/textself_ft.py, boxes/textself.py, boxes/text_eval.py at 3b0bb06f), CPU, in this scratch dir only.
  T1 trainer: python -m boxes.textself_ft (real) on CPU, --steps 2 --bs 4 (a smoke: dead + ARGS-DIFF by design), out = ./_e2e/boxes/runs_fsq/s219_t1_s1;
    model.pt (500 MB) is deleted after its sha256 is recorded by the trainer's train_done.json.
  eval_text (base + T1): text_eval.part_text (real: batching, budgets, overflow, item_record, aggregate, meta, items_sha256) with ONLY textself.greedy_batch
    replaced by a gold-chain oracle that realises a target accuracy curve (no GPT-2 forward: 24,000 generations are not CPU-feasible) and text_eval.card
    replaced by an RTX 3070 label (the card check is exercised by the fixtures, not here).
  eval_textself (T1): text_eval.part_textself_cpu (real label audit + contamination on the real smoke pool) + the real smoke GPU parts copied from
    scratch/_tmp/s219/evals/eval_textself.json (labeller_kept_meta.t_start restamped after eval_text in the second scoring; the first keeps the original
    stamp, which precedes this eval_text, to show the seal order check fire on writer output).
  GATE-T1 eval_gate.json / pool_meta.json / pool_t1.jsonl: the real smoke pipe files (card 'cpu' relabelled RTX 3070, as for eval_text).
Nothing is written under D:/Latent (the trainer's --out and every dump point into ./_e2e). Report -> ./e2e_report.json.
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
import fixtures_s219 as F  # noqa: E402   (curves only)
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


def cp(src, d, f, mod=None):
    os.makedirs(os.path.join(RR, d), exist_ok=True); x = json.load(open(src))
    if mod:
        mod(x)
    json.dump(x, open(os.path.join(RR, d, f), "w", newline="\n"), indent=1)


t0 = time.time()
# ---- 1. the real trainer (CPU smoke), out in scratch
out_t1 = os.path.join(RR, L["t1"])
cmd = [sys.executable, "-B", "-m", "boxes.textself_ft", "--arm", "textcot", "--init_from", "boxes/runs_fsq/s170_cot_s1/model.pt", "--steps", "2", "--bs", "4",
       "--lr", "3e-5", "--self_pool", "scratch/_tmp/s219/pipe/s219_pool_t1_s1/pool_t1.jsonl", "--p_self", "0.5", "--self_micro", "2", "--no_final_eval",
       "--draws_log_steps", "12000", "--log_every", "1", "--device", "cpu", "--out", out_t1]
r = subprocess.run(cmd, cwd=REAL, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ))
REP["steps"]["trainer"] = {"rc": r.returncode, "sec": round(time.time() - t0), "tail": r.stdout[-600:] + r.stderr[-1500:]}
print("TRAINER rc", r.returncode, round(time.time() - t0), "s", flush=True)
assert r.returncode == 0, r.stderr[-3000:]
REP["trainer_files"] = sorted(os.listdir(out_t1))
mp = os.path.join(out_t1, "model.pt")
if os.path.exists(mp):
    os.remove(mp)                                                           # scratchpad .pt cleanup (sha kept in train_done.json)
td = json.load(open(os.path.join(out_t1, "train_done.json")))

# ---- 2. the real eval writer with a gold-chain oracle for the model forward
TD = N.test_set(1000, n_max=24); POS = {}
for j, it in enumerate(TD):
    POS.setdefault(tuple(it["syms"]), (it["n"], j - (it["n"] - 1) * 1000))
CURVE = {}


def oracle_greedy_batch(g, prompts, budget, stop_at_eq=False, margins=False):
    outs = []
    for p in prompts:
        syms = [INV[t] for t in p[1:-1]]; n, q = POS[tuple(syms)]
        chain = N.cot_ids(TS.parse_tree(syms))
        outs.append(chain[:budget] if q < round(1000 * CURVE[n]) else [TS.EOT])
    fed = len(prompts[0]) - 1 + max(len(o) for o in outs) - 1
    return outs, [float("inf")] * len(prompts), min(fed, 1023)


class StubG:
    def w(self, name):
        return torch.zeros(1)


TS.greedy_batch = oracle_greedy_batch; TE.card = lambda dev: "NVIDIA GeForce RTX 3070 (e2e: real text_eval writer, oracle forward on cpu)"
for d, curve, sha in ((L["base"], F.A0, S.BASE_SHA), (L["t1"], F.make_curve(15.5), td["model_sha256"])):
    CURVE.clear(); CURVE.update(curve); os.makedirs(os.path.join(RR, d), exist_ok=True); t1 = time.time(); op = os.path.join(RR, d, "eval_text.json")
    res = TE.part_text(StubG(), sha, "e2e-oracle", "boxes/runs_fsq/" + d, "cpu", 32, op, list(range(1, 25)), 0)
    TE.dump(res, op); pp = op.replace(".json", ".partial.json")
    if os.path.exists(pp):
        os.remove(pp)
    REP["steps"]["eval_text_" + d] = {"sec": round(time.time() - t1), "acc_10": res["acc_by_n"]["10"], "overflow_24": res["overflow_by_n"]["24"]}
    print("EVAL_TEXT", d, round(time.time() - t1), "s", flush=True)

# ---- 3. the real E2 CPU parts on the smoke pool + the real smoke GPU parts
cfg = dict(TS.ROUNDS[1]); cfg["audit_seed"] += 100_000; cfg["kept_seed"] += 100_000; t1 = time.time()
sd = TE.part_textself_cpu(out_t1, cfg, SM + "/pipe/s219_pool_t1_s1/pool_t1.jsonl")
g_sm = json.load(open(SM + "/evals/eval_textself.json"))
for k in ("memo", "labeller_kept", "labeller_kept_meta", "meta_gpu"):
    sd[k] = g_sm[k]
TE.dump(sd, os.path.join(out_t1, "eval_textself.json")); REP["steps"]["textself_cpu"] = {"sec": round(time.time() - t1)}
print("TEXTSELF_CPU", round(time.time() - t1), "s", flush=True)

# ---- 4. gate / pool (real smoke files), base args / bf16 eval
relabel = lambda x: x.update(card="NVIDIA GeForce RTX 3070 (e2e relabel of a cpu smoke)")  # noqa: E731
cp(SM + "/pipe/s219_gate_t1_s1/eval_gate.json", L["gate_t1"], "eval_gate.json", relabel)
cp(SM + "/pipe/s219_pool_t1_s1/pool_meta.json", L["pool_t1"], "pool_meta.json", relabel)
shutil.copyfile(SM + "/pipe/s219_pool_t1_s1/pool_t1.jsonl", os.path.join(RR, L["pool_t1"], "pool_t1.jsonl"))
cp(REAL + "/boxes/runs_fsq/s170_cot_s1/args.json", L["base_model"], "args.json"); cp(REAL + "/boxes/runs_fsq/s170_cot_s1/eval.json", L["base_model"], "eval.json")

# ---- 5. score (registered mode), twice: original GPU-part stamp, then restamped after eval_text
S.set_root(RR, E2E); S.OUT = os.path.join(E2E, "score_s219_e2e.json")
o1 = S.score()["s1"]
x = json.load(open(os.path.join(out_t1, "eval_textself.json"))); x["labeller_kept_meta"]["t_start"] = TE.iso(); TE.dump(x, os.path.join(out_t1, "eval_textself.json"))
o2 = S.score()["s1"]; t1c = o2["t1"]
tt = S.train_tokens("s1", "t1")
adm = S.args_diff("s1", "t1", smoke=True, smoke_pool="scratch/_tmp/s219/pipe/s219_pool_t1_s1/pool_t1.jsonl")
chk = {
    "base_scored": o2["base"]["status"] == "SCORED", "base_n50": o2["base"]["n50"], "base_delta0": o2["base"]["DELTA_0"],
    "t1_not_pending": t1c["status"] != "PENDING", "t1_status": t1c["status"], "t1_censor_reasons": t1c.get("censor_reasons"),
    "t1_eval_text_void": (t1c.get("EVAL") or {}).get("void"), "t1_n50": t1c.get("n50"), "t1_T_ABSORB_raw": t1c.get("T_ABSORB_raw"),
    "seal_first_scoring": o1["t1"].get("SEAL"), "seal_second_scoring": t1c.get("SEAL"),
    "gate_t1": {k: o2["gate_t1"][k] for k in ("status", "consistent", "pooled_matches")}, "gate_equiv": o2["gate_t1"]["descriptive"]["EQUIV"],
    "label_audit_ok": t1c.get("LABEL_AUDIT_T", {}).get("ok"), "contamination": {k: (t1c.get("CONTAMINATION") or {}).get(k) for k in ("pool_excl_hits", "CONTAMINATED", "recheck", "cursor")},
    "train_done": t1c.get("TRAIN_DONE"), "train_tokens_text": tt.get("text"), "train_done_tokens": td["train_tokens"],
    "args_diff_registered": {k: (t1c.get("ARGS_DIFF") or {}).get(k) for k in ("shared_diffs", "required_bad", "unexpected_new_keys", "neutral_bad", "dropped_base_keys")},
    "args_diff_smoke_mode": {k: adm.get(k) for k in ("shared_diffs", "required_bad", "unexpected_new_keys", "neutral_bad", "dropped_base_keys", "ok")},
    "comparison": o2["comparison"]["outcome"], "line_has_train_tokens": "TRAIN-TOKENS" in o2["comparison"]["line"]}
REP["checks"] = chk; REP["reading"] = S.reading({"s1": o2}); REP["sec"] = round(time.time() - t0)
json.dump(REP, open(os.path.join(HERE, "e2e_report.json"), "w", newline="\n"), indent=1, default=str)
print(json.dumps(chk, default=str)[:4000])
shutil.rmtree(E2E, ignore_errors=True)
