"""STEP-TOLD $0 check ZS (CPU, 2 threads, idle priority): does a trunk that never computes the unasked intermediate v_1 compute it when ASKED?

Derivation A1 (ST_draft sec. 3) says one pass computes the value of the entity its question names, when that entity is one op away from values it can
read. §228's failure (REVIEW_LB check T; Notes 231-2 / 231-3) is that v_1 is never asked. This check asks for it, zero-shot, in the two closed §228 models:
  DIRECT  boxes/runs_fsq/s228_pdir2c_s1 (one pass; d = 1 acc 0.623): answer argmax over the 11 number tokens after " The answer is:".
  LAT     boxes/runs_fsq/s228_plat2_s1 (INJECT): one think pass (S_0 = 0) on a variant input, then the model's own teacher-forced pass-1 aux
          value-token acc (R_step[1] = "<S_1> has <v_1> <u>.") - the LP2 quantity, read by depth.
Variants (d = 2 items unless stated; the question is the text's last sentence, asserted):
  D1     d = 1 items, unchanged (target v_1 = the asked answer)
  ORIG   d = 2, unchanged (targets v_2 and v_1)
  REPL   d = 2, the question re-targeted to S_1: text[:-len(question)] + QUESTION(S_1)   (target v_1)
  APP    d = 2, the original text + " " + QUESTION(S_1) appended (target v_1)        (the ST_draft cue form)
  ASK0   d = 2, the question re-targeted to S_0 (a GIVEN value: lookup positive control; target v_0) - DIRECT only
  CHAIN  LAT descriptive: passes t = 1..8 with think = text + QUESTION(S_min(t, 2)), decoder unchanged; answer acc at K' 2 and 8 (target v_2).
Items: TOPO, frames 0-5, dag on, stream seed 23241 (unused in boxes/ theory/ docs scripts, grep 10-24). Pre-stated predictions and rule: ST_draft.md
stub (11:48 EDT), written before this script ran.
    python -B st_told_zs.py [--n 1000] [--bs 50]
"""
import argparse
import contextlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
try:
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
except Exception:  # noqa: BLE001
    pass
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
import boxes.ow_words as OW  # noqa: E402
import boxes.inject_ow as IO  # noqa: E402
import boxes.inject_ow_eval as IOE  # noqa: E402
from boxes.nested170 import NUM_IDS, EOT  # noqa: E402

RUNS = os.path.join(ROOT, "boxes", "runs_fsq")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "st_told_zs.json")


def variant(it, kind, t_ask=1):
    """-> the item text under a variant (the question is the last sentence)."""
    q = it["question"]; txt = it["text"]; assert txt.endswith(q), "question not last"
    sp, u = it["names"]["spine"], it["unit"]
    if kind in ("D1", "ORIG"):
        return txt
    if kind == "REPL":
        return txt[:-len(q)] + OW.QUESTION.format(u=u, A=sp[t_ask])
    if kind == "ASK0":
        return txt[:-len(q)] + OW.QUESTION.format(u=u, A=sp[0])
    if kind == "APP":
        return txt + " " + OW.QUESTION.format(u=u, A=sp[t_ask])
    raise ValueError(kind)


def tok(it, text):
    o = IO.tok_ow(it, "value"); th = [EOT] + IO.ENC.encode(" " + text); o["think"] = th; o["dec"] = th + IO.SFX; return o


def majority(y):
    y = np.asarray(y); return float(np.bincount(y, minlength=11).max() / len(y))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--bs", type=int, default=50); a = ap.parse_args(argv)
    t0 = time.time(); st = OW.Stream(23241, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True)
    I1 = [st.draw(1) for _ in range(a.n)]; I2 = [st.draw(2) for _ in range(a.n)]
    assert all(OW.QUESTION.format(u=it["unit"], A=it["names"]["spine"][-1]) == it["question"] for it in I1 + I2)
    v = lambda its, k: [int(it["values"][k]) for it in its]  # noqa: E731
    res = {"seed": 23241, "n_per_d": a.n, "majority": {"d1_v1": majority(v(I1, 1)), "d2_v0": majority(v(I2, 0)), "d2_v1": majority(v(I2, 1)),
                                                      "d2_v2": majority(v(I2, 2))}, "direct": {}, "lat": {}}
    nc = contextlib.nullcontext
    # ---------------- DIRECT
    _, m0, g0 = IOE.load(os.path.join(RUNS, "s228_pdir2c_s1"), "cpu"); assert m0 is None

    def dpred(its, kind):
        out = []
        with torch.no_grad():
            for i in range(0, len(its), a.bs):
                ch = [tok(it, variant(it, kind)) for it in its[i:i + a.bs]]; out += list(IOE.pred_of(IO.direct_logits(g0, ch, "cpu", nc)))
        return np.array(out)
    for kind, its, tgts in (("D1", I1, {"v1": 1}), ("ORIG", I2, {"v2": 2, "v1": 1}), ("REPL", I2, {"v1": 1, "v2": 2}), ("APP", I2, {"v1": 1, "v2": 2}),
                            ("ASK0", I2, {"v0": 0})):
        p = dpred(its, kind); res["direct"][kind] = {k: float(np.mean(p == np.array(v(its, j)))) for k, j in tgts.items()}
        res["direct"][kind]["pred_share_max"] = majority(p); print("DIRECT", kind, res["direct"][kind], flush=True)
    del g0
    # ---------------- LAT
    _, m, g = IOE.load(os.path.join(RUNS, "s228_plat2_s1"), "cpu"); mem = m.mem

    def aux1(its, kind):
        ok = []
        with torch.no_grad():
            for i in range(0, len(its), a.bs):
                ch = [tok(it, variant(it, kind)) for it in its[i:i + a.bs]]; x, msk = m.think_inputs(ch, "cpu")
                S1 = m.write(mem.zero_state(len(ch), "cpu"), x, msk, m.lower(x)); lg, ya = m.aux_logits(S1, ch, 1, "cpu"); pr = lg.argmax(-1)
                ok += [int(pr[r, it["vidx"][0]] == ya[r, it["vidx"][0]]) for r, it in enumerate(ch)]
        return float(np.mean(ok))
    for kind, its in (("D1", I1), ("ORIG", I2), ("REPL", I2), ("APP", I2)):
        res["lat"]["aux_pass1_" + kind] = aux1(its, kind); print("LAT aux1", kind, res["lat"]["aux_pass1_" + kind], flush=True)

    def chain(its):
        """descriptive: cue S_min(t, 2) appended at pass t; the decoder reads the plain question (dec unchanged); answers at K' = 2 and 8."""
        p2, p8 = [], []
        with torch.no_grad():
            for i in range(0, len(its), a.bs):
                raw = its[i:i + a.bs]; ch = [IO.tok_ow(it, "value") for it in raw]; S = mem.zero_state(len(ch), "cpu")
                for t in range(1, 9):
                    tt = [tok(it, variant(it, "APP", min(t, 2))) for it in raw]
                    x, msk = m.think_inputs(tt, "cpu"); S = m.write(S, x, msk, m.lower(x))
                    if t in (2, 8):
                        (p2 if t == 2 else p8).extend(list(IOE.pred_of(m.dec_logits(S, ch, "cpu"))))
        y = np.array(v(its, 2)); return {"acc_K2": float(np.mean(np.array(p2) == y)), "acc_K8": float(np.mean(np.array(p8) == y))}
    res["lat"]["chain_d2"] = chain(I2); print("LAT chain", res["lat"]["chain_d2"], flush=True)
    res["lat"]["gamma"] = float(mem.gamma); res["sec"] = round(time.time() - t0, 1)
    d, l_ = res["direct"], res["lat"]
    sup = (max(d["REPL"]["v1"], d["APP"]["v1"]) >= 0.30) or (max(l_["aux_pass1_REPL"], l_["aux_pass1_APP"]) >= 0.45)
    mj = res["majority"]["d2_v1"]
    null = max(d["REPL"]["v1"], d["APP"]["v1"]) <= mj + 0.10 and max(l_["aux_pass1_REPL"], l_["aux_pass1_APP"]) <= mj + 0.10
    res["reading"] = "TOLD-ZS-SUPPORT" if sup else ("TOLD-ZS-NULL" if null else "WEAK")
    res["format"] = "REPL" if (d["REPL"]["v1"] - d["APP"]["v1"] >= 0.15 and l_["aux_pass1_REPL"] - l_["aux_pass1_APP"] >= 0.15) else "APP"
    with open(OUT, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print("READING", res["reading"], "format", res["format"], "sec", res["sec"], flush=True)


if __name__ == "__main__":
    main()
