"""§191 PROSQA-RELABEL gates (prosqa168s2 --relabel / --test500_every; Theory's sec191 draft + Audit's checks) -> boxes/runs_fsq/s168s2_gates/gate191.json
  g1 round-trip on 1,000 train items (rng 19110): the name map is injective, entity -> ENT17 and concept -> CON38 only; inverse-substituting the relabelled
     question gives the original premise multiset and the original (or, when swapped, option-swapped) question tail; answer and every step invert exactly;
     the relabelled premises parse to exactly the item's edges under the relabelled idx_to_symbol; encode(): a1 / a2 / step_ids (the aux targets) / node /
     opt_true / opt_false decode to the relabelled texts, and the steps form the root -> target path in the relabelled premises; swap rate and premise-order change logged
  g2 test never relabelled: relabel() refuses a test / ctl raw item; every relabel( call site in the module draws from Rtr (training items)
  g3 flags-off bitwise vs the pre-change module (argv[1]), 1 CPU thread, 3 steps: stepaux and textcot (log, model.pt, eval.json)
  g4 --relabel runs (stepaux, textcot; 3 steps, --test500_every 2): rc 0; log has relabel_swap_rate and test500_acc at steps 2 and 3; eval.json has
     train_fit_relabel (stepaux K' 0 / 1 / 2 / 10); --test500_every alone (flags-off otherwise) leaves the training log and model.pt bitwise unchanged
  g5 dedupe: no p8_na_train graph equals a p8_na_test graph by (edges, root, target, neg_target)
  g6 resume: 4 steps --relabel in one go == 2 steps + --resume ckpt_last (model.pt bitwise; relabel rng and counts restored)
  g7 gradflow on relabelled items (stepaux, fade 1, gamma 0.5): answer CE alone and aux CE alone each give a nonzero grad on mem.qkv, mem.wo, s_add-path and the trunk
  g8 memory: worst relabel (longest-token names to the most frequent symbols) vs every item and 20,000 random relabels of the 200 longest; peak memory at
     chunk 4 on argv[2] (default cuda:0) flags-off vs --relabel (the worst relabel of the 200 longest), projected onto the 3090 checks (micro 4 = chunk 16)
    python scratch/gate191.py <pre-change prosqa168s2.py> [cuda device]"""
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
import boxes.prosqa168s2 as P  # noqa: E402
from boxes import s5  # noqa: E402

GD = os.path.join(ROOT, "boxes", "runs_fsq", "s168s2_gates"); OUT = {}; SP = os.environ.get("GATE_TMP", os.path.join(ROOT, "scratch", "_gate191_tmp"))
PR = os.path.join(ROOT, "prosqa", "p8data")
COMMON = ["--device", "cpu", "--log_every", "1", "--bs", "2", "--train_n", "40", "--test_n", "4", "--extras_n", "2", "--probe_n_fit", "8", "--eval_bs", "4",
          "--sched_scale", "0.0001", "--no_normcap", "--ckpt_every", "0", "--max_new", "24", "--cot_eval_bs", "4"]
ARMS = {"stepaux": ["--arm", "stepaux", "--think_k", "2", "--fast_add"], "textcot": ["--arm", "textcot"]}
PREM_RE = re.compile(r"(?:Every (\w+)|(\w+)) is a (\w+)\.")


def run(mod, out, extra=(), steps=3, keep_ckpt=False, clean=True):
    if clean:
        shutil.rmtree(out, ignore_errors=True)
    code = ("import torch,sys,os,runpy;torch.set_num_threads(1);sys.argv=['x']+sys.argv[1:];" + ("os.remove=lambda p: None;" if keep_ckpt else "") +
            "runpy.run_module('%s',run_name='__main__')" % mod)
    args = list(extra)
    r = subprocess.run([sys.executable, "-c", code] + COMMON + ["--steps", str(steps)] + args + ["--out", out], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": ROOT, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    assert r.returncode == 0, (mod, extra, r.stderr[-2500:])
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))], json.load(open(os.path.join(out, "eval.json"))), torch.load(os.path.join(out, "model.pt"), weights_only=True)


def edges_of(prem, sym):
    E = []
    for p_ in prem:
        mm = PREM_RE.fullmatch(p_); assert mm, p_
        E.append((sym.index(mm.group(1) or mm.group(2)), sym.index(mm.group(3))))
    return E


def g1():
    R = P.load_train_raw(os.path.join(PR, "p8_na_train.json")); rng = np.random.default_rng(19110); pick = np.random.default_rng(19111).permutation(len(R))[:1000]
    c = Counter(); bad = []
    for i in pick:
        x = R[i]; y, sw = P.relabel(x, rng); sym, ysym = P._lit(x["idx_to_symbol"]), y["idx_to_symbol"]; mp = dict(zip(sym, ysym)); inv = {v: k for k, v in mp.items()}
        ok = {}
        ok["injective"] = len(set(ysym)) == len(ysym) == len(set(sym))
        ok["in_vocab"] = all((w in P.ENT17) if w[0].isupper() else (w in P.CON38) for w in ysym) and all(w[0].isupper() == mp[w][0].isupper() for w in sym)
        pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, inv), key=len, reverse=True)) + r")\b"); un = lambda t_: pat.sub(lambda m_: inv[m_.group(1)], t_)   # noqa: E731
        mx, my = P.Q_RE.fullmatch(x["question"]), P.Q_RE.fullmatch(y["question"])
        px = [p_.strip() for p_ in re.findall(r"[^.]+\.", mx.group(1))]; py = [p_.strip() for p_ in re.findall(r"[^.]+\.", my.group(1))]
        ok["premises_multiset"] = Counter(un(p_) for p_ in py) == Counter(px)
        tail_x = "Is %s a %s or %s?" % ((mx.group(2), mx.group(4), mx.group(3)) if sw else (mx.group(2), mx.group(3), mx.group(4)))
        ok["tail"] = un("Is %s a %s or %s?" % my.group(2, 3, 4)) == tail_x
        ok["answer"] = un(y["answer"]) == x["answer"]; ok["steps"] = [un(s) for s in y["steps"]] == P._lit(x["steps"])
        Ey = edges_of(py, ysym); ok["premises_are_edges"] = sorted(Ey) == sorted(map(tuple, P._lit(x["edges"]))) and len(Ey) == len(P._lit(x["edges"]))
        it = P.encode(y); tg, ng = ysym[int(x["target"])], ysym[int(x["neg_target"])]
        ok["a1"] = P.ENC.decode(it["a1"]) == "### " + y["answer"] and y["answer"].endswith(" %s." % tg) and y["answer"].startswith(ysym[int(x["root"])] + " ")
        ok["a2"] = P.ENC.decode(it["a2"]) == "### " + y["answer"][:-len(tg) - 1] + ng + "."
        ok["aux_targets"] = [P.ENC.decode(s) for s in it["step_ids"]] == [" " + s for s in y["steps"]] and it["depth"] == len(P._lit(x["steps"]))
        ok["node"] = it["node"] == [mp[n] for n in P.encode(x)["node"]] and it["opt_true"] == tg and it["opt_false"] == ng
        ok["options_in_question"] = {my.group(3), my.group(4)} == {tg, ng} and my.group(2) == ysym[int(x["root"])]
        se = set(Ey); path = edges_of(y["steps"], ysym)                       # the steps: a root -> target chain of premise edges
        ok["steps_path"] = all(e in se for e in path) and path[0][0] == int(x["root"]) and path[-1][1] == int(x["target"]) and all(a_[1] == b_[0] for a_, b_ in zip(path, path[1:]))
        c["swapped"] += sw; c["renamed_any"] += ysym != sym; c["premise_order_changed"] += [un(p_) for p_ in py] != px
        c["n"] += 1
        for k, v in ok.items():
            c["ok_" + k] += v
            if not v and len(bad) < 5:
                bad.append((int(i), k))
    OUT["g1"] = {**dict(c), "swap_rate": c["swapped"] / c["n"], "first_failures": bad}
    OUT["g1_pass"] = all(c["ok_" + k] == c["n"] for k in ("injective", "in_vocab", "premises_multiset", "tail", "answer", "steps", "premises_are_edges", "a1", "a2",
                                                           "aux_targets", "node", "options_in_question", "steps_path")) and 0.45 < OUT["g1"]["swap_rate"] < 0.55 and c["renamed_any"] == c["n"]


def g2():
    ref = []
    for f in ("p8_na_test.json", "p8_ctl.json"):
        x = json.load(open(os.path.join(PR, f)))[0]
        try:
            P.relabel(x, np.random.default_rng(0)); ref.append(False)
        except AssertionError:
            ref.append(True)
    import ast
    tree = ast.parse(open(os.path.join(ROOT, "boxes", "prosqa168s2.py"), encoding="utf-8").read())
    calls = [ast.unparse(n.args[0]) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "relabel"]
    OUT["g2"] = {"refuses_test_ctl": ref, "call_site_first_args": calls}
    OUT["g2_pass"] = all(ref) and len(calls) > 0 and all(a_.startswith("Rtr[") for a_ in calls)


def strip_e(e):
    return {k: v for k, v in e.items() if k not in ("eval_sec", "train_sec")}


def nosec(L):
    return [{k: v for k, v in r.items() if k not in ("sec", "test500_sec")} for r in L]


def g3(pre_mod):
    res = {}
    for arm, ax in ARMS.items():
        l_new, e_new, m_new = run("boxes.prosqa168s2", os.path.join(SP, "new_" + arm), ax)
        l_pre, e_pre, m_pre = run(pre_mod, os.path.join(SP, "pre_" + arm), ax)
        res[arm] = {"log_equal": nosec(l_new) == nosec(l_pre), "model_equal": set(m_new) == set(m_pre) and all(torch.equal(m_new[k], m_pre[k]) for k in m_new),
                    "eval_equal": strip_e(e_new) == strip_e(e_pre)}
        res[arm + "_n_log"] = len(l_new)
    OUT["g3"] = res; OUT["g3_pass"] = all(all(v.values()) for k, v in res.items() if isinstance(v, dict))


def g4():
    res = {}
    for arm, ax in ARMS.items():
        L, E, M = run("boxes.prosqa168s2", os.path.join(SP, "rl_" + arm), ax + ["--relabel", "--test500_every", "2"])
        t5 = [r for r in L if "test500_acc" in r]; sr = [r["relabel_swap_rate"] for r in L if "relabel_swap_rate" in r]
        l_t5, _, m_t5 = run("boxes.prosqa168s2", os.path.join(SP, "t5_" + arm), ax + ["--test500_every", "1"])
        l_new = json.loads(json.dumps([r for r in l_t5 if "test500_acc" not in r]))
        l_off = [json.loads(l) for l in open(os.path.join(SP, "new_" + arm, "log.jsonl"))]; m_off = torch.load(os.path.join(SP, "new_" + arm, "model.pt"), weights_only=True)
        res[arm] = {"test500_steps": [r["step"] for r in t5], "test500_acc": [r["test500_acc"] for r in t5], "swap_rate_log": sr, "relabel_n_last": [r.get("relabel_n") for r in L if "relabel_n" in r][-1:],
                    "train_fit_relabel": E.get("train_fit_relabel"), "acc": E.get("acc"),
                    "test500_only_log_equal": nosec(l_new) == nosec(l_off), "test500_only_model_equal": all(torch.equal(m_t5[k], m_off[k]) for k in m_off),
                    "test500_only_n_records": sum("test500_acc" in r for r in l_t5)}
        tf = (E.get("train_fit_relabel") or {}); key = "acc_by_kprime" if arm == "stepaux" else "acc_cot"
        res[arm + "_pass"] = (res[arm]["test500_steps"] == [2, 3] and len(sr) == 3 and tf.get("n") == 40 and set(tf.get(key, {})) == ({"0", "1", "2", "10"} if arm == "stepaux" else {"acc"})
                              and res[arm]["test500_only_log_equal"] and res[arm]["test500_only_model_equal"] and res[arm]["test500_only_n_records"] == 3)
    OUT["g4"] = res; OUT["g4_pass"] = all(res[a_ + "_pass"] for a_ in ARMS)


def g5():
    sig = lambda x: (frozenset(map(tuple, P._lit(x["edges"]))), int(x["root"]), int(x["target"]), int(x["neg_target"]))   # noqa: E731
    tr = {sig(x) for x in json.load(open(os.path.join(PR, "p8_na_train.json")))}; te = [sig(x) for x in json.load(open(os.path.join(PR, "p8_na_test.json")))]
    OUT["g5"] = {"n_train_sigs": len(tr), "n_test": len(te), "test_in_train": sum(s in tr for s in te)}; OUT["g5_pass"] = OUT["g5"]["test_in_train"] == 0


def g6():
    ax = ARMS["stepaux"] + ["--relabel", "--ckpt_every", "2"]; ax = [v for i, v in enumerate(ax)]
    _, _, m_full = run("boxes.prosqa168s2", os.path.join(SP, "res_full"), ax, steps=4)
    half = os.path.join(SP, "res_half"); run("boxes.prosqa168s2", half, ax, steps=4, keep_ckpt=True)   # ckpt_last.pt = step 2 (os.remove disabled)
    ck = torch.load(os.path.join(half, "ckpt_last.pt"), weights_only=False); cp = os.path.join(SP, "res_ck2.pt"); shutil.copy(os.path.join(half, "ckpt_last.pt"), cp)
    res_dir = os.path.join(SP, "res_resumed"); shutil.rmtree(res_dir, ignore_errors=True); os.makedirs(res_dir)
    _, _, m_res = run("boxes.prosqa168s2", res_dir, ax + ["--resume", cp], steps=4, clean=False)
    OUT["g6"] = {"ckpt_step": ck["step"], "ckpt_rl_n": ck.get("rl_n"), "has_rl_rng": "rl_rng" in ck, "model_equal": all(torch.equal(m_full[k], m_res[k]) for k in m_full)}
    OUT["g6_pass"] = ck["step"] == 2 and ck.get("rl_n", [0])[0] == 4 and OUT["g6"]["has_rl_rng"] and OUT["g6"]["model_equal"]


def g7():
    R = P.load_train_raw(os.path.join(PR, "p8_na_train.json")); rng = np.random.default_rng(19112)
    its = [P.encode(P.relabel(R[i], rng)[0]) for i in range(4)]; P.DM.FAST_ADD = True; torch.manual_seed(1); g = s5.load_gpt2("cpu"); m = P.ProsMem(g)
    with torch.no_grad():
        m.mem.fade.fill_(1.0); m.mem.gamma.fill_(0.5); m.mem.wo.weight.normal_(0, 0.02)   # W_o is zero at init (no grad reaches qkv until it moves; as gate_rmsnopass)
    gr = {}
    for term in ("answer", "aux"):
        m.zero_grad(set_to_none=True); x, msk = m.think_inputs(its, "cpu"); S = m.mem.zero_state(len(its), "cpu"); aux = []
        for t in range(1, 3):
            S = m.write(S, x, msk)
            if term == "aux":
                lg, y = m.aux_logits(S, its, t, "cpu"); aux.append(Fn.cross_entropy(lg.transpose(1, 2), y, ignore_index=-100))
        loss = m.dec_loss(S, its, "cpu") if term == "answer" else torch.stack(aux).mean(); loss.backward()
        gs = lambda ps: float(sum(p_.grad.abs().sum() for p_ in ps if p_.grad is not None))   # noqa: E731
        gr[term] = {"mem_qkv": gs([m.mem.qkv.weight]), "mem_wo": gs([m.mem.wo.weight]), "mem_all": gs(m.mem.parameters()), "trunk": gs(g.parameters()), "loss": float(loss)}
    OUT["g7"] = gr; OUT["g7_pass"] = all(v > 0 for t_ in gr.values() for k_, v in t_.items() if k_ != "loss")


def g8(dev):
    R = P.load_train_raw(os.path.join(PR, "p8_na_train.json")); D = [P.encode(x) for x in R]; L = lambda it: (len(it["q"]), len(it["q"]) + sum(len(s) for s in it["step_ids"]))   # noqa: E731
    orig = [L(it) for it in D]; worst = [L(P.encode(P.relabel(x, None, worst=True)[0])) for x in R]
    top = sorted(range(len(D)), key=lambda i: -orig[i][1])[:200]; rng = np.random.default_rng(19113); rnd = [L(P.encode(P.relabel(R[top[j % 200]], rng)[0])) for j in range(20000)]
    w200 = max(worst[i][1] for i in top); wall = max(w[1] for w in worst)
    res = {"orig_max_q": max(o[0] for o in orig), "orig_max_qsteps": max(o[1] for o in orig), "worst_max_q_all": max(w[0] for w in worst), "worst_max_qsteps_all": wall,
           "worst_max_qsteps_top200": w200, "random20k_max_q": max(r[0] for r in rnd), "random20k_max_qsteps": max(r[1] for r in rnd),
           "worst_ge_random": max(r[1] for r in rnd) <= w200, "top200_worst_is_global": w200 == wall}
    pk = {}
    for tag, ex in (("off", []), ("relabel", ["--relabel"])):
        out = os.path.join(SP, "mem_" + tag)
        cmd = [sys.executable, "-m", "boxes.prosqa168s2", "--arm", "stepaux", "--think_k", "10", "--aux_step", "1.0", "--aux_fade", "12000,12000", "--fast_add", "--chunk_delta", "64", "--ckpt_passes", "--cache_lower", "--bs", "4", "--micro", "1", "--mem_check",
               "--device", dev, "--out", out] + ex
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
        pk[tag] = json.load(open(os.path.join(out, "mem_check.json"))) if r.returncode == 0 else {"err": r.stderr[-800:]}
    a3090 = json.load(open("boxes/runs_fsq/s168s2_stepaux_s1/args.json")); res["s168s2_stepaux_args"] = {k: a3090.get(k) for k in ("cache_lower", "ckpt_passes", "chunk_delta", "micro")}
    ratio = pk["relabel"]["peak_gib"] / pk["off"]["peak_gib"] if "peak_gib" in pk.get("off", {}) and "peak_gib" in pk.get("relabel", {}) else None
    res.update({"chunk4_peak": pk, "chunk4_ratio_relabel_over_off": ratio, "gpu": torch.cuda.get_device_name(dev),
                "proj_3090_stepaux_micro4_gib": 17.71 * ratio if ratio else None, "proj_3090_textcot_micro4_gib_upper": 14.74 * ratio if ratio else None,
                "note": "the 3090 cells run --auto_micro, whose mem_check uses the worst relabel of the 200 longest items (it_long); projection = s168s2's chunk-16 peak x the chunk-4 ratio"})
    OUT["g8"] = res; OUT["g8_pass"] = bool(ratio) and res["worst_ge_random"] and res["top200_worst_is_global"] and res["proj_3090_stepaux_micro4_gib"] < 21.0


def main():
    os.makedirs(SP, exist_ok=True); pre = os.path.join(ROOT, "boxes", "_pre191_gate.py"); shutil.copy(sys.argv[1], pre); dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
    try:
        for f in (g1, g2, g5, g7):
            f(); print(f.__name__, OUT.get(f.__name__ + "_pass"), flush=True)
        g3("boxes._pre191_gate"); print("g3", OUT["g3_pass"], flush=True)
        g4(); print("g4", OUT["g4_pass"], flush=True)
        g6(); print("g6", OUT["g6_pass"], flush=True)
        g8(dev); print("g8", OUT["g8_pass"], flush=True)
    finally:
        os.remove(pre); shutil.rmtree(SP, ignore_errors=True)
        OUT["all_pass"] = bool(all(OUT.get("g%d_pass" % i) for i in range(1, 9)))
        os.makedirs(GD, exist_ok=True); json.dump(OUT, open(os.path.join(GD, "gate191.json"), "w", newline="\n"), indent=1, default=str); print(json.dumps(OUT, default=str)[:3500], flush=True)


if __name__ == "__main__":
    main()
