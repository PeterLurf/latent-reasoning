"""Note 252-4 ASKNAME-PERM ($0, local RTX 4060): the s233_askname_s1 recipe exactly (docs/research_steptold_2026-10-24/st_ask_name_local.py, ST_v1 P1-ASK+NAME four classes,
boxes/inject_ow.py --arm direct; d_max 2, 8k, auto_micro / mem_budget 7.0, eval_bs 125, seed 1) with ONE change: every tokenized OW item with d >= 2 is re-rendered with its op
sentences in a dependency-violating order (boxes.perm_render.permuted; >= 1 inversion; givens / distractors keep their slots) with probability 0.5, BEFORE the ask class is
applied (class (iv) texts are built from the permuted text). Per item k (the k-th call of the tokenizer in this process): coin default_rng((1, 25163, k)), permutation
default_rng((1, 25162, k)) - a family disjoint from TEST-PERM (25150 + d) and from --perm_p (25160 / 25161); OW.solve == answer asserted; every tokenized training text
asserted absent from the REG TEST-PERM d 1 / d 2 texts (Note 252-1 condition 2 / Note 252-4). The trainer's own eval is skipped (--eval_parts '': its tokd would route the
test sets through the same hook); evaluation = `eval` below on unpermuted inputs.
  train : python -B scratch/s252_askperm.py train          -> boxes/runs_fsq/s252_askperm_s1 (+ perm_stats.json)
  eval  : python -B scratch/s252_askperm.py eval           -> scratch/s252_askperm/probe.json for s252_askperm_s1 AND s233_askname_s1 (same call, 4060, fp32):
          (a) the a01e3d24 probe (scratch/s252_scale0.py): CHAIN / PERM / D1, the same items, askname-style zero-shot (the pass-1 input; argmax over ' 0'..' 10');
          (b) the full TEST2 d 1 (REG, 1,000 items, every wrap) on the same input;
          (c) ST_v1's four class accuracies (st_ask_name_local.ask_of on the law-1 TEST2 d 1 / d 2 it was defined on, unpermuted; per class x d + pooled d 1-2)."""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scratch"); os.chdir(ROOT)
import boxes.inject_ow as IO  # noqa: E402
import boxes.perm_render as PR  # noqa: E402

SEED_ASKP, SEED_ASKP_COIN, P = 25162, 25163, 0.5
OUT_RUN = "boxes/runs_fsq/s252_askperm_s1"; OUT_EV = ROOT + "/scratch/s252_askperm"
_spec = importlib.util.spec_from_file_location("st_ask_name_local", ROOT + "/docs/research_steptold_2026-10-24/st_ask_name_local.py")
SA = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SA)  # the original wrapper, read-only (its tok_askname / ask_of)
STATS = {"calls": 0, "d2plus": 0, "permuted": 0, "by_class_perm": {}}


def tp_texts():
    s = IO.build_sets(mul_s_p=0); P_, _ = PR.test_perm_sets({d: s["test2"][d] for d in (1, 2)}, (1, 2)); return {it["text"] for v in P_.values() for it in v}


TPT = None


def tok_askperm(it, aux_target="value"):
    k = STATS["calls"]; STATS["calls"] += 1; pm = 0
    if it["d"] >= 2:
        STATS["d2plus"] += 1
        if np.random.default_rng((1, SEED_ASKP_COIN, k)).random() < P:
            pit, _, inv = PR.permuted(it, np.random.default_rng((1, SEED_ASKP, k))); PR.check(it, pit); assert inv >= 1
            pit["sig"] = it["sig"]; it = pit; STATS["permuted"] += 1; pm = 1
    assert it["text"] not in TPT, "ASKNAME-PERM: a training text equals a TEST-PERM text"
    o = SA.tok_askname(it, aux_target)
    b = STATS["by_class_perm"].setdefault("d%d_%s" % (it["d"], o["ask_class"]), [0, 0]); b[0] += pm; b[1] += 1   # [permuted, n] per d x class
    return o


def train():
    global TPT
    TPT = tp_texts(); a = json.load(open("boxes/runs_fsq/s233_askname_s1/args.json"))
    argv = ["--d_max", str(a["d_max"]), "--steps", str(a["steps"]), "--auto_micro", "--mem_budget", str(a["mem_budget"]), "--eval_bs", str(a["eval_bs"]),
            "--eval_parts", "", "--device", "cuda:0", "--out", OUT_RUN]
    os.makedirs(OUT_RUN, exist_ok=True)
    meta = {"check": "Note 252-4 ASKNAME-PERM (s233_askname_s1 recipe + permuted training texts p 0.5)", "wrapper": "scratch/s252_askperm.py",
            "base_wrapper_sha256": a.get("module_sha256"), "seeds": {"perm": SEED_ASKP, "coin": SEED_ASKP_COIN}, "p": P}
    json.dump(meta, open(OUT_RUN + "/ask_meta.json", "w", newline="\n"), indent=1)
    IO.tok_ow = tok_askperm                                                 # inject_ow.main looks tok_ow up at call time (as st_ask_name_local does)
    rc = IO.main(["--arm", "direct"] + argv)
    STATS["perm_share_d2plus"] = STATS["permuted"] / max(1, STATS["d2plus"])
    json.dump(STATS, open(OUT_RUN + "/perm_stats.json", "w", newline="\n"), indent=1); print("TRAIN rc", rc, {k: v for k, v in STATS.items() if k != "by_class_perm"}, flush=True); return rc


def evaluate():
    import torch
    import s252_scale0 as SC
    dev = "cuda:0"; S, fs, info = SC.sets_(); s1 = IO.build_sets(); s0 = IO.build_sets(mul_s_p=0); res = {"probe_n": {k: len(v) for k, v in S.items()}, "models": {}}
    num = torch.tensor(IO.NUM_IDS if hasattr(IO, "NUM_IDS") else __import__("boxes.nested170", fromlist=["NUM_IDS"]).NUM_IDS)
    for name in ("s252_askperm_s1", "s233_askname_s1"):
        g = SC.load_gpt2(dev, "boxes/runs_fsq/%s/model.pt" % name); cid = SC.cands("gpt2"); r = {}
        with torch.no_grad():
            for k, its in S.items():                                        # (a) the probe
                scs = np.array(SC.score_trunk(g, SC.prompts("askname", its, fs), cid, dev)); pred = scs.argmax(1); y = np.array([it["values"][1] for it in its])
                r[k] = {"n": len(its), "acc": float((pred == y).mean()), "items": {"sig": [it["sig"] for it in its], "v1": y.tolist(), "pred": pred.tolist()}}
            d1 = s0["test2"][1]                                             # (b) the full REG TEST2 d 1
            scs = np.array(SC.score_trunk(g, SC.prompts("askname", d1, fs), cid, dev)); pred = scs.argmax(1); y = np.array([it["values"][1] for it in d1])
            r["TEST2_D1_ALL"] = {"n": len(d1), "acc": float((pred == y).mean()), "items": {"sig": [it["sig"] for it in d1], "pred": pred.tolist()}}
            cls = {}                                                        # (c) ST_v1's four classes (law-1 TEST2 d 1 / d 2, unpermuted, the original ask_of)
            for d in (1, 2):
                its = s1["test2"][d]; T = [SA.tok_askname(it) for it in its]
                scs = np.array(SC.score_trunk(g, [t["dec"] for t in T], cid, dev)); pred = scs.argmax(1)
                for t, p_ in zip(T, pred):
                    b = cls.setdefault(str(d), {}).setdefault(t["ask_class"], [0, 0]); b[0] += int(p_ == t["ans"]); b[1] += 1
            pooled = {}
            for d, cv in cls.items():
                for c, (ok, n) in cv.items():
                    pb = pooled.setdefault(c, [0, 0]); pb[0] += ok; pb[1] += n
            r["ST_CLASSES"] = {"by_d": {d: {c: {"acc": v[0] / v[1], "n": v[1]} for c, v in sorted(cv.items())} for d, cv in cls.items()},
                               "pooled_d1_d2": {c: {"acc": v[0] / v[1], "n": v[1]} for c, v in sorted(pooled.items())}}
        res["models"][name] = r; del g; torch.cuda.empty_cache()
    res["def"] = ("Note 252-4 eval: (a) a01e3d24's probe sets (REG no-wrap; CHAIN / PERM / D1), askname-style zero-shot = [EOT] + enc(' ' + text + ' ' + QUESTION(S_1)) + SFX, "
                  "argmax over ' 0'..' 10'; (b) the full REG TEST2 d 1; (c) ST_v1 P1-ASK+NAME classes on law-1 TEST2 d 1 / d 2 (st_ask_name_local.tok_askname, unpermuted); fp32, 4060")
    os.makedirs(OUT_EV, exist_ok=True); json.dump(res, open(OUT_EV + "/probe.json", "w", newline="\n"), indent=1); print("EVAL written", flush=True)


if __name__ == "__main__":
    sys.exit({"train": train, "eval": lambda: evaluate() or 0}[sys.argv[1]]())
