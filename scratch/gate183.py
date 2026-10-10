"""THEORY §183 INTERLEAVE-BLIND pre-launch gates (d5c160a7): boxes/inject169.py --interleave k --blind_decoder.
  (1) flags off (no --blind_decoder) = the pre-§183 interleave path bitwise (6 CPU steps, 1 thread, --interleave 1: logs + every eval field) + smoke_plain
  (2) the blind per-token decoder's context is [EOT, " =", ans_<j] - no problem token (instrumented il_read, train + greedy) - and its logits are
      bitwise equal across two problem sets with the same memory and answers (the seeing decoder's differ)
  (3) gradflow (blind, 4x2, k = 1; W_o perturbed, fade 1, gamma 0.5): each token's loss alone gives nonzero gradient on its S_j, the carried S_(j-1)
      and the memory's write-path parameters
  (4) a blind cell smoke (CPU, sched_scale 0.002) writes eval.json with every s181 field, args blind_decoder True / interleave 1
    python scratch/gate183.py <pre-183 inject169 copy>   -> boxes/runs_fsq/s183_gates/gate183.json (written after every gate)"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import boxes.inject169 as I  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.fixedchunk import DT, EQ, product_digits  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402
from boxes.fsq120 import EOT  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s183_gates"); SP = os.environ.get("GATE183_TMP", os.path.join(GD, "tmp")); OUT = {}
BENCH = os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")
TINY = ["--seed", "1", "--steps", "6", "--sched_scale", "0.002", "--log_every", "2", "--bs", "4", "--test_n", "4", "--ext_n", "4", "--probe_n_fit", "32",
        "--eval_bs", "4", "--ckpt_every", "1000", "--bench", BENCH, "--think_k", "0", "--interleave", "1", "--pass_index", "sin_gain", "--fast_add"]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate183.json"), "w", newline="\n"), indent=1)


def run_many(jobs):
    procs = []
    for args, out, script in jobs:
        shutil.rmtree(out, ignore_errors=True); env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=ROOT)
        procs.append(subprocess.Popen([sys.executable, "-u"] + ([script] if script else ["-m", "boxes.inject169"]) + args + ["--device", "cpu", "--out", out],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT))
    res = []
    for p_ in procs:
        o_, e_ = p_.communicate(); res.append((p_.returncode, (o_ or "")[-300:] + (e_ or "")[-800:]))
    return res


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, pre + k + ".")) if isinstance(v, dict) else o.__setitem__(pre + k, v)
    return o


def build(blind):
    torch.manual_seed(1); m = I.Inject(s5.load_gpt2("cpu"), 6, "top"); m.enable_pass_index(); m.blind = blind
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5); m.pgain.normal_(0, 0.01)
    return m


def main():
    torch.set_num_threads(3); old = sys.argv[1]; ok = True
    # (1) + (4) concurrently: flags-off new / old, and the blind cell smoke
    dn, do, db = os.path.join(SP, "new_il1"), os.path.join(SP, "old_il1"), os.path.join(SP, "cell_blind")
    (rn, en), (ro, eo), (rb, eb) = run_many([(TINY, dn, None), (TINY, do, old), (TINY + ["--blind_decoder"], db, None)])
    if rn or ro:
        OUT["g1"] = {"rc_new": rn, "rc_old": ro, "err": (en if rn else eo)[-600:]}
    else:
        Ln = [json.loads(x).get("loss") for x in open(os.path.join(dn, "log.jsonl"))]; Lo = [json.loads(x).get("loss") for x in open(os.path.join(do, "log.jsonl"))]
        fn, fo = flat(json.load(open(os.path.join(dn, "eval.json")))), flat(json.load(open(os.path.join(do, "eval.json"))))
        OUT["g1"] = {"log_losses_equal": Ln == Lo, "eval_fields_differing": sorted(k for k in set(fn) | set(fo) if not k.endswith("_sec") and fn.get(k) != fo.get(k))[:20], "n_fields": len(fn)}
    sp = subprocess.run([sys.executable, "scratch/smoke_plain.py"], capture_output=True, text=True, cwd=ROOT); OUT["g1"]["smoke_plain"] = sp.stdout.strip().splitlines()[-1:]
    OUT["g1_pass"] = bool(OUT["g1"].get("log_losses_equal")) and not OUT["g1"].get("eval_fields_differing") and sp.returncode == 0; ok &= OUT["g1_pass"]
    if rb == 0:
        e_ = json.load(open(os.path.join(db, "eval.json"))); a_ = json.load(open(os.path.join(db, "args.json"))); s_ = e_.get("s181", {})
        need = {"k", "acc_4x4", "acc_by_column", "ext_4x8", "acc_by_kprime", "skip_matrix", "mem_norm_by_token", "delta_cos", "train_loss_tail", "compute"}
        OUT["g4"] = {"rc": 0, "missing": sorted(need - set(s_)), "blind_decoder": a_.get("blind_decoder"), "interleave": a_.get("interleave"), "res_blind": e_.get("blind_decoder"),
                     "decoder_input": a_.get("decoder_input")}
        OUT["g4_pass"] = not OUT["g4"]["missing"] and OUT["g4"]["blind_decoder"] is True and OUT["g4"]["interleave"] == 1
    else:
        OUT["g4"] = {"rc": rb, "err": eb[-600:]}; OUT["g4_pass"] = False
    ok &= OUT["g4_pass"]
    for d_ in (dn, do, db):
        for f_ in ("model.pt", "ckpt_last.pt", "ckpt_prev.pt"):
            os.path.exists(os.path.join(d_, f_)) and os.remove(os.path.join(d_, f_))
    dump()
    # (2)
    reads = []; r0 = I.il_read

    def rd(m, S, ctx):
        reads.append(ctx.detach().clone()); return r0(m, S, ctx)
    I.il_read = rd; g2 = {}
    try:
        m = build(True); r_ = np.random.default_rng(3); A, B = gen_operands(r_, 3, 4), gen_operands(r_, 3, 3); ans = torch.tensor(DT[product_digits(A, B)]); L = ans.shape[1]
        with torch.no_grad():
            I.il_loss(m, A, B, ans, 1, L)
        g2["train_ctx_ok"] = all(torch.equal(reads[j - 1], torch.cat([torch.full((3, 1), EOT), torch.full((3, 1), EQ), ans[:, :j - 1]], 1)) for j in range(1, L + 1))
        reads.clear(); pr = I.predict_il(m, A, B, 1, "cpu", None, 8)
        g2["greedy_ctx_no_problem"] = all(r.shape[1] == 2 + j and bool((r[:, 0] == EOT).all()) and bool((r[:, 1] == EQ).all()) for j, r in enumerate(reads))
    finally:
        I.il_read = r0
    for blind in (True, False):
        m = build(blind); S = torch.randn(3, 12, 64, 64) * 0.05; r_ = np.random.default_rng(5); A1, B1 = gen_operands(r_, 3, 4), gen_operands(r_, 3, 3)
        A2, B2 = gen_operands(r_, 3, 4), gen_operands(r_, 3, 3); toks = torch.tensor(DT[product_digits(A1, B1)])[:, :3]
        with torch.no_grad():
            l1 = I.il_read(m, S, I.il_dec_ctx(m, torch.tensor(I.dec_prefix(A1, B1)), toks)); l2 = I.il_read(m, S, I.il_dec_ctx(m, torch.tensor(I.dec_prefix(A2, B2)), toks))
        g2["blind" if blind else "seeing"] = {"logits_equal_across_problems": bool(torch.equal(l1, l2))}
    OUT["g2"] = g2; OUT["g2_pass"] = g2["train_ctx_ok"] and g2["greedy_ctx_no_problem"] and g2["blind"]["logits_equal_across_problems"] and not g2["seeing"]["logits_equal_across_problems"]
    ok &= OUT["g2_pass"]; dump()
    # (3)
    m = build(True); m.train(); r_ = np.random.default_rng(7); A, B = gen_operands(r_, 4, 4), gen_operands(r_, 4, 2); ans = torch.tensor(DT[product_digits(A, B)]); L = ans.shape[1]
    pre = torch.tensor(I.dec_prefix(A, B)); S = m.mem.zero_state(4, "cpu"); Ss, ls = [], []
    for j in range(1, L + 1):
        S = I.il_token(m, S, m.ids(torch.cat([pre, ans[:, :j - 1]], 1)), j - 1, 1); S.retain_grad(); Ss.append(S)
        ls.append(Fn.cross_entropy(I.il_read(m, S, I.il_dec_ctx(m, pre, ans[:, :j - 1])), ans[:, j - 1]))
    g3 = {}; wp = ("mem.qkv.weight", "mem.gate.weight", "mem.wo.weight")
    for j in range(1, L + 1):
        m.zero_grad(set_to_none=True)
        for s_ in Ss:
            s_.grad = None
        ls[j - 1].backward(retain_graph=True); P_ = dict(m.named_parameters())
        g3[str(j)] = {"S_j": float(Ss[j - 1].grad.norm()), "S_prev": (float(Ss[j - 2].grad.norm()) if Ss[j - 2].grad is not None else 0.0) if j >= 2 else None,
                      "write": {n_: float(P_[n_].grad.norm()) if P_[n_].grad is not None else 0.0 for n_ in wp}}
    OUT["g3"] = g3; OUT["g3_pass"] = all(v["S_j"] > 0 and (v["S_prev"] is None or v["S_prev"] > 0) and min(v["write"].values()) > 0 for v in g3.values())
    ok &= OUT["g3_pass"]; OUT["all_pass"] = bool(ok); dump(); print("GATE183", json.dumps(OUT)[:2500], flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
