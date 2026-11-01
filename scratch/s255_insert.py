"""Note 255-3 INSERT ($0, eval only; scorer theory/score_insert.py) on the 4060 (cuda:0), fp32, eval_bs --bs -> scratch/s255_insert/{gate,insert}.json
Items: REG TEST2 d 4 (1,000). Each condition inserts k in {1, 2} disconnected PAIRS of statements about NEW names (not in the item's spine / givens / distractors): a given
"Y has v u." (NW.GIVEN) followed by an op "X has c more u than Y." rendered with FRAMES["add_c"][f] (B = X, A = Y), f drawn from OW.FRAME_TRAIN (the training templates);
v in 1..9, c in 1..(10 - v) (no mod-11 wrap); per-item rng default_rng((25190, cond index, item index)). Positions in the statement sequence (it["order"]):
before = right before S1's sentence; inside = right after S2's sentence (before S3's); after = right after S4's sentence. The question stays last. Answers unchanged:
OW.solve(new text) == answer asserted per item; the rebuilt clean text (NW statements joined as _finish does) == it["text"] asserted per item.
Conditions: clean, before_k1, before_k2, inside_k1, inside_k2, after_k1, after_k2 (named cues as trained: tok_aw on the new text).
Models: rcb_plaw4_s1, rcb_plaw4_s2, aex_s1 (if its model.pt exists). Per (model, cond): hop {k: share of items with answer at K' = k == v_{k+1}} for k = 1..3, acc8 = acc at
K' 8, n; per-item answers at K' 1, 2, 3, 8 stored under items.
    python -B scratch/s255_insert.py gate eval [--bs 100 --device cuda:0]"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
import boxes.nested_words as NW  # noqa: E402
import boxes.ow_words as OW  # noqa: E402

OUT = ROOT + "/scratch/s255_insert"; CONDS = ["clean", "before_k1", "before_k2", "inside_k1", "inside_k2", "after_k1", "after_k2"]; SEED = 25190
MODELS = ("rcb_plaw4_s1", "rcb_plaw4_s2", "aex_s1")


def statements(it):
    nm = it["names"]; st, _ = NW._statements(it["ops"], it["operands"], it["src_j"], it["values"], nm["spine"], nm["givens"], nm["distractors"], it["frames"],
                                             it["distractor_values"])
    byid = {x["id"]: x for x in st}; u = it["unit"]

    def render(x):
        if x["kind"] in ("given", "distractor"):
            return NW.GIVEN.format(A=x["name"], v=x["v"], u=u)
        return NW.FRAMES[x["op"]][x["frame"]].format(B=x["B"], A=x["A"], C=x["C"], c=x["c"], u=u)
    return [render(byid[i]) for i in it["order"]], NW.QUESTION.format(u=u, A=nm["spine"][-1])


def insert(it, cond, idx):
    sents, q = statements(it); assert " ".join(sents + [q]) == it["text"], "clean rebuild"
    if cond == "clean":
        return dict(it), []
    pos_name, k = cond.split("_k"); k = int(k); order = list(it["order"])
    at = {"before": order.index("S1"), "inside": order.index("S2") + 1, "after": order.index("S4") + 1}[pos_name]
    rng = np.random.default_rng((SEED, CONDS.index(cond), idx)); used = set(it["names"]["spine"]) | set(it["names"]["givens"]) | set(it["names"]["distractors"])
    free = [n for n in OW.NAMES if n not in used]; pick = rng.choice(len(free), size=2 * k, replace=False); new = []; meta = []
    for j in range(k):
        Y, X = free[pick[2 * j]], free[pick[2 * j + 1]]; v = int(rng.integers(1, 10)); c = int(rng.integers(1, 11 - v)); f = int(OW.FRAME_TRAIN[int(rng.integers(0, len(OW.FRAME_TRAIN)))])
        new += [NW.GIVEN.format(A=Y, v=v, u=it["unit"]), NW.FRAMES["add_c"][f].format(B=X, A=Y, C=None, c=c, u=it["unit"])]; meta.append({"Y": Y, "X": X, "v": v, "c": c, "frame": f})
    text = " ".join(sents[:at] + new + sents[at:] + [q]); o = dict(it); o["text"] = text; o["sig"] = it["sig"] + ":" + cond
    assert OW.solve(text) == it["answer"], ("INSERT: solve != answer", cond, it["sig"]); return o, meta


def build(raw):
    sets = {}; metas = {}
    for c in CONDS:
        its = []; ms = []
        for i, it in enumerate(raw):
            o, m = insert(it, c, i); its.append(o); ms.append(m)
        sets[c] = its; metas[c] = ms
    return sets, metas


def gate(dev, bs):
    a = json.load(open("boxes/runs_fsq/rcb_plaw4_s1/args.json")); raw = IO.build_sets(mul_s_p=0)["test2"][4]; sets, metas = build(raw)   # asserts: rebuild + solve
    T = {c: E.tokd({"test2": {4: v}}, a, (4,))[4] for c, v in sets.items()}
    ok_cue = all(T[c][i]["think_cues"][t][-len(T["clean"][i]["think_cues"][t]) + len([E.EOT]) + len(IO.ENC.encode(" " + raw[i]["text"])):] ==
                 T["clean"][i]["think_cues"][t][len([E.EOT]) + len(IO.ENC.encode(" " + raw[i]["text"])):] for c in CONDS for i in range(len(raw)) for t in range(4))
    o = {"n": len(raw), "rebuild_and_solve_asserted": True, "cue_tail_unchanged": ok_cue,
         "inserted_sentences_per_cond": {c: (0 if c == "clean" else 2 * int(c[-1])) for c in CONDS},
         "examples": {c: sets[c][0]["text"] for c in CONDS}, "example_meta": {c: metas[c][0] for c in CONDS},
         "frames_used": sorted({m["frame"] for c in CONDS for ms in metas[c] for m in ms})}
    o["pass"] = bool(ok_cue); return o


def evaluate(dev, bs):
    raw = IO.build_sets(mul_s_p=0)["test2"][4]; sets, metas = build(raw); res = {"meta": {"conds": CONDS, "bs": bs, "dtype": "fp32", "gpu": torch.cuda.get_device_name(0)}}
    for name in MODELS:
        if not os.path.exists("boxes/runs_fsq/%s/model.pt" % name):
            res[name] = {"skipped": "no model.pt"}; continue
        a, m, g = E.load(ROOT + "/boxes/runs_fsq/" + name, dev); r = {}
        with torch.no_grad():
            for c in CONDS:
                T = E.tokd({"test2": {4: sets[c]}}, a, (4,))[4]; A, _ = E.passes(m, T, dev, bs, {1, 2, 3, 8})
                v = np.array([it["values"] for it in sets[c]]); y = np.array([it["answer"] for it in sets[c]])
                r[c] = {"hop": {str(k): float((A[k] == v[:, k + 1]).mean()) for k in (1, 2, 3)}, "acc8": float((A[8] == y).mean()), "n": len(T),
                        "items": {"sig": [it["sig"] for it in sets[c]], "ans": {str(k): A[k].tolist() for k in (1, 2, 3, 8)}, "insert": metas[c]}}
        res[name] = r; del m, g; torch.cuda.empty_cache()
    res["def"] = ("Note 255-3 INSERT: REG TEST2 d 4; k disconnected given + add_c op pairs on new names (train frames) before S1 / after S2 / after S4; answers unchanged (solve "
                  "asserted); hop[k] = share with answer at K' = k == v_{k+1}, acc8 = acc at K' 8; named cues as trained")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("parts", nargs="+"); ap.add_argument("--bs", type=int, default=100); ap.add_argument("--device", default="cuda:0"); o_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); fns = {"gate": gate, "eval": evaluate}
    for p_ in o_.parts:
        r = fns[p_](o_.device, o_.bs); json.dump(r, open(os.path.join(OUT, ("insert" if p_ == "eval" else p_) + ".json"), "w", newline="\n"), indent=1, default=str)
        print(p_, ("pass " + str(r["pass"])) if p_ == "gate" else "written", flush=True)
