"""THEORY §181 INTERLEAVE pre-launch gates (ba4cf16e): boxes/inject169.py --interleave k (+ --time_steps).
  (1) flags off = pre-§181 inject169 bitwise (6 CPU steps, 1 thread: the fade + index path and the K2 path; logs + every eval field) + smoke_plain
  (2) token j's first pass starts from token j - 1's last state (zeros before token 1), pass index 1..k*L in order - training (il_loss) and greedy
  (3) teacher-forced contexts are [EOT, prefix, gold ans_<j]; greedy contexts are [EOT, prefix, the model's own argmax tokens_<j]
  (4) gradflow (4x2, k = 2; W_o perturbed, fade 1, gamma 0.5): each token's loss alone gives nonzero gradient on its own write output S_j, on the
      carried state S_(j-1) (j >= 2), and on the memory's write-path parameters
  (5) smoke of the cell flags (CPU, sched_scale 0.002): --interleave 1 and 2 write eval.json with every s181 field; --time_steps writes timing.json
  The memory check at 4x5 (9 tokens x k) is the run's own --auto_micro check on the 3070 preview and on the 3090 timing host.
    python scratch/gate181.py <pre-181 inject169 copy>   -> boxes/runs_fsq/s181_gates/gate181.json (written after every gate)"""
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
from boxes.fixedchunk import DT, product_digits  # noqa: E402
from boxes.multcol import gen_operands  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s181_gates"); SP = os.environ.get("GATE181_TMP", os.path.join(GD, "tmp")); OUT = {}
BENCH = os.path.join(ROOT, "bench", "data", "mult_4x4_test.txt")
TINY = ["--seed", "1", "--steps", "6", "--sched_scale", "0.002", "--log_every", "2", "--bs", "4", "--test_n", "4", "--ext_n", "4", "--probe_n_fit", "32",
        "--eval_bs", "4", "--ckpt_every", "1000", "--bench", BENCH]


def dump():
    os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate181.json"), "w", newline="\n"), indent=1)


def run(args, out, script=None, threads="1"):
    if os.environ.get("GATE181_REUSE") and os.path.exists(os.path.join(out, "eval.json")):
        return 0, "reused"
    shutil.rmtree(out, ignore_errors=True); env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, PYTHONPATH=ROOT)
    cmd = [sys.executable, "-u"] + ([script] if script else ["-m", "boxes.inject169"]) + args + ["--device", "cpu", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT); return r.returncode, r.stdout[-300:] + r.stderr[-600:]


def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, pre + k + ".")) if isinstance(v, dict) else o.__setitem__(pre + k, v)
    return o


def build(k_seed=1):
    torch.manual_seed(k_seed); m = I.Inject(s5.load_gpt2("cpu"), 6, "top"); m.enable_pass_index()
    with torch.no_grad():
        m.mem.wo.weight.normal_(0, 0.02); m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5); m.pgain.normal_(0, 0.01)
    return m


def main():
    torch.set_num_threads(3); old = sys.argv[1]; ok = True
    # (1)
    g1 = {}
    for nm, extra in (("fade_index", ["--think_k", "4n", "--fade", "12000,36000", "--pass_index", "sin_gain"]), ("k2", ["--think_k", "2"])):
        dn, do = os.path.join(SP, "new_" + nm), os.path.join(SP, "old_" + nm); rn, en = run(extra + TINY, dn); ro, eo = run(extra + TINY, do, old)
        if rn or ro:
            g1[nm] = {"rc_new": rn, "rc_old": ro, "err": (en if rn else eo)[-500:]}; continue
        Ln = [json.loads(x).get("loss") for x in open(os.path.join(dn, "log.jsonl"))]; Lo = [json.loads(x).get("loss") for x in open(os.path.join(do, "log.jsonl"))]
        fn, fo = flat(json.load(open(os.path.join(dn, "eval.json")))), flat(json.load(open(os.path.join(do, "eval.json"))))
        diff = sorted(k for k in set(fn) | set(fo) if not k.endswith("_sec") and fn.get(k) != fo.get(k))
        g1[nm] = {"log_losses_equal": Ln == Lo, "eval_fields_differing": diff[:20], "n_fields": len(fn)}
        for d_ in (dn, do):
            for f_ in ("model.pt", "ckpt_last.pt", "ckpt_prev.pt"):
                os.path.exists(os.path.join(d_, f_)) and os.remove(os.path.join(d_, f_))
    sp = subprocess.run([sys.executable, "scratch/smoke_plain.py"], capture_output=True, text=True, cwd=ROOT); g1["smoke_plain"] = sp.stdout.strip().splitlines()[-1:] if sp.stdout else sp.stderr[-300:]
    OUT["g1"] = g1; OUT["g1_pass"] = all(g1[n].get("log_losses_equal") and not g1[n].get("eval_fields_differing") for n in ("fade_index", "k2")) and sp.returncode == 0
    ok &= OUT["g1_pass"]; dump()
    # (2) + (3): instrument write / il_read
    calls = []; w0 = I.Inject.write; reads = []; r0 = I.il_read

    def w(self, S, ids, t=None):
        S2 = w0(self, S, ids, t); calls.append({"S_in": S.detach().clone(), "S_out": S2.detach().clone(), "ctx": ids.detach().clone(), "t": t}); return S2

    def rd(m, S, ctx):
        lg = r0(m, S, ctx); reads.append({"ctx": ctx.detach().clone(), "argmax": lg.argmax(-1).detach().clone()}); return lg
    I.Inject.write = w; I.il_read = rd; g2 = {}; g3 = {}
    try:
        for k in (1, 2):
            m = build(); r_ = np.random.default_rng(3); A, B = gen_operands(r_, 3, 4), gen_operands(r_, 3, 3); P = product_digits(A, B); ans = torch.tensor(DT[P]); L = P.shape[1]
            pre = torch.tensor(I.dec_prefix(A, B))
            for mode in ("train", "greedy"):
                calls.clear(); reads.clear()
                if mode == "train":
                    with torch.no_grad():
                        I.il_loss(m, A, B, ans, k, L)
                else:
                    I.predict_il(m, A, B, k, "cpu", None, 8)
                chain = all(torch.equal(calls[q]["S_in"], calls[q - 1]["S_out"]) for q in range(1, len(calls)))
                first0 = bool((calls[0]["S_in"] == 0).all()); ts = [c["t"] for c in calls]
                g2["k%d_%s" % (k, mode)] = {"n_passes": len(calls), "expected": k * L, "state_chain_unbroken": chain, "token1_from_zero": first0,
                                            "pass_index_1_to_kL": ts == list(range(1, k * L + 1))}
                if mode == "train":
                    gold = all(torch.equal(calls[(j - 1) * k]["ctx"], m.ids(torch.cat([pre, ans[:, :j - 1]], 1))) for j in range(1, L + 1))
                    same = all(torch.equal(reads[j - 1]["ctx"], calls[(j - 1) * k]["ctx"]) for j in range(1, L + 1))
                    g3["k%d_train" % k] = {"pass_ctx_gold": gold, "read_ctx_equals_pass_ctx": same}
                else:
                    own = all(torch.equal(calls[(j - 1) * k]["ctx"][:, -(j - 1):], torch.stack([reads[q]["argmax"] for q in range(j - 1)], 1)) for j in range(2, L + 1))
                    differs = any(not torch.equal(calls[(j - 1) * k]["ctx"], m.ids(torch.cat([pre, ans[:, :j - 1]], 1))) for j in range(2, L + 1))
                    g3["k%d_greedy" % k] = {"pass_ctx_own_tokens": own, "differs_from_gold_somewhere": differs}
    finally:
        I.Inject.write = w0; I.il_read = r0
    OUT["g2"] = g2; OUT["g2_pass"] = all(v["n_passes"] == v["expected"] and v["state_chain_unbroken"] and v["token1_from_zero"] and v["pass_index_1_to_kL"] for v in g2.values())
    OUT["g3"] = g3; OUT["g3_pass"] = all(g3["k%d_train" % k]["pass_ctx_gold"] and g3["k%d_train" % k]["read_ctx_equals_pass_ctx"] and g3["k%d_greedy" % k]["pass_ctx_own_tokens"] for k in (1, 2))
    ok &= OUT["g2_pass"] and OUT["g3_pass"]; dump()
    # (4)
    m = build(); m.train(); k = 2; r_ = np.random.default_rng(5); A, B = gen_operands(r_, 4, 4), gen_operands(r_, 4, 2); P = product_digits(A, B); ans = torch.tensor(DT[P]); L = P.shape[1]
    pre = torch.tensor(I.dec_prefix(A, B)); S = m.mem.zero_state(4, "cpu"); Ss = []; losses = []
    for j in range(1, L + 1):
        ctx = m.ids(torch.cat([pre, ans[:, :j - 1]], 1)); S = I.il_token(m, S, ctx, (j - 1) * k, k); S.retain_grad(); Ss.append(S)
        losses.append(Fn.cross_entropy(I.il_read(m, S, ctx), ans[:, j - 1]))
    wp = ("mem.qkv.weight", "mem.gate.weight", "mem.wo.weight"); g4 = {}
    for j in range(1, L + 1):
        m.zero_grad(set_to_none=True)
        for s_ in Ss:
            s_.grad = None
        losses[j - 1].backward(retain_graph=True)
        pg = {n_: float(dict(m.named_parameters())[n_].grad.norm()) if dict(m.named_parameters())[n_].grad is not None else 0.0 for n_ in wp}
        g4[str(j)] = {"S_j_grad": float(Ss[j - 1].grad.norm()), "S_prev_grad": float(Ss[j - 2].grad.norm()) if j >= 2 and Ss[j - 2].grad is not None else (None if j == 1 else 0.0),
                      "write_param_grads": pg, "pgain_grad": float(m.pgain.grad.norm()) if m.pgain.grad is not None else 0.0}
    OUT["g4"] = g4; OUT["g4_pass"] = all(v["S_j_grad"] > 0 and (v["S_prev_grad"] is None or v["S_prev_grad"] > 0) and min(v["write_param_grads"].values()) > 0 for v in g4.values())
    ok &= OUT["g4_pass"]; dump()
    # (5)
    g5 = {}
    for kk in (1, 2):
        d_ = os.path.join(SP, "cell_k%d" % kk); rc, err = run(["--think_k", "0", "--interleave", str(kk), "--pass_index", "sin_gain", "--fast_add"] + TINY, d_, threads="3")
        e_ = json.load(open(os.path.join(d_, "eval.json"))) if rc == 0 else {}; s_ = e_.get("s181", {})
        need = {"k", "acc_4x4", "acc_by_column", "ext_4x8", "acc_by_kprime", "skip_matrix", "mem_norm_by_token", "delta_cos", "train_loss_tail", "compute"}
        g5["k%d" % kk] = {"rc": rc, "err": err[-400:] if rc else "", "missing": sorted(need - set(s_)), "k": s_.get("k"), "kprimes": sorted(s_.get("acc_by_kprime", {})),
                          "skip_rows": len(s_.get("skip_matrix", {})), "norm_tokens": len(s_.get("mem_norm_by_token", {})), "delta_cos": {x: s_.get("delta_cos", {}).get(x) for x in ("within_token", "across_tokens")},
                          "compute_passes_4x4": s_.get("compute", {}).get("4x4", {}).get("think_passes"), "args_interleave": json.load(open(os.path.join(d_, "args.json"))).get("interleave") if rc == 0 else None}
        for f_ in ("model.pt", "ckpt_last.pt", "ckpt_prev.pt"):
            os.path.exists(os.path.join(d_, f_)) and os.remove(os.path.join(d_, f_))
    d_ = os.path.join(SP, "timing_k1"); rc, err = run(["--think_k", "0", "--interleave", "1", "--pass_index", "sin_gain", "--fast_add", "--time_steps", "3"] + TINY, d_, threads="3")
    g5["time_steps"] = {"rc": rc, "timing": json.load(open(os.path.join(d_, "timing.json"))) if rc == 0 and os.path.exists(os.path.join(d_, "timing.json")) else err[-300:]}
    OUT["g5"] = g5; OUT["g5_pass"] = all(g5["k%d" % kk]["rc"] == 0 and not g5["k%d" % kk]["missing"] and g5["k%d" % kk]["k"] == kk and g5["k%d" % kk]["skip_rows"] == 8
                                         and g5["k%d" % kk]["norm_tokens"] == 12 and g5["k%d" % kk]["compute_passes_4x4"] == 8 * kk for kk in (1, 2)) and g5["time_steps"]["rc"] == 0
    ok &= OUT["g5_pass"]; OUT["all_pass"] = bool(ok); dump(); print("GATE181", json.dumps(OUT)[:3000], flush=True); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
