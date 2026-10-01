"""THEORY §67: train one arm, then evaluate.
    python -m c1.train --arm {d1,tab,reset,fp_ex,fp_cmp} --seed 1 --updates 2000 --bs 512 --lr 1e-4 --out c1/runs/<name>
Training: final answers only. Curriculum: the updates are split in 4 equal stages; stage s draws each batch's length
uniformly from {1, 2, 4, 8}[: s + 1] (all rows of a batch share the length). d1/tab/reset: exact forward
marginalisation (core.marginal_loss) over the 42-call tables (d1/reset) or the learned tables (tab). fp_ex: full-prompt
GPT-2 on the same batches (example-matched). fp_cmp: full-prompt GPT-2 with rows per update cut so its GPT-2 token
positions per update equal d1's 120 (36 x 3 + 6 x 2): compute-matched.
Evaluation (eval.json): hard-execution accuracy at N in {8, 16, 32, 64, 128} (test generator seed 6700, n_test per
depth); for d1 additionally the nulls (10,000 per depth; Wilson 95 % upper bounds), the mechanism tests and the
36-transition audit (tab/reset: audit descriptive).
Null definitions (fixed here, before any run): T0 = decode the initial code; T1 = one update with the last op; T2 = two
updates with the last two ops; first_op = one update with the first op; last_op = T1; two_op = updates with the first
and the last op. All from the initial code.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boxes.core import GPT2KV  # noqa: E402
from c1.core import COMP, FP, Cell, Tab, audit, hard_rollout, marginal_loss, sample, truth, wilson_upper  # noqa: E402

DEPTHS = (8, 16, 32, 64, 128)
TEST_SEED = 6700
D1_TOKENS = 36 * 3 + 6 * 2


def gpt2(dev):
    sd = torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
    sd = {k[len("transformer."):] if k.startswith("transformer.") else k: v for k, v in sd.items()}
    return GPT2KV(sd).to(dev)


def build(arm, dev):
    if arm == "tab":
        return Tab().to(dev)
    if arm in ("fp_ex", "fp_cmp"):
        return FP(gpt2(dev)).to(dev)
    return Cell(gpt2(dev)).to(dev)


def predict(model, arm, ops):
    if arm in ("fp_ex", "fp_cmp"):
        return model(ops).argmax(-1)
    return hard_rollout(model, ops, reset=arm == "reset")[1]


@torch.no_grad()
def evaluate(model, arm, dev, n_test, n_null, out):
    model.eval(); res = {"acc": {}, "n_test": n_test}
    for N in DEPTHS:
        rng = np.random.default_rng(TEST_SEED + N); ops, y = sample(rng, n_test, N)
        ops_t = torch.tensor(ops, device=dev); pred = []
        for i in range(0, n_test, 2000 if arm.startswith("fp") and N >= 64 else 5000):
            pred.append(predict(model, arm, ops_t[i:i + (2000 if arm.startswith("fp") and N >= 64 else 5000)]).cpu())
        res["acc"][str(N)] = float((torch.cat(pred).numpy() == y).mean())
    if arm in ("d1", "tab", "reset"):
        res["audit"] = audit(model)
    if arm == "d1":
        res["nulls"] = nulls(model, dev, n_null); res["mechanism"] = mechanism(model, dev)
    json.dump(res, open(os.path.join(out, "eval.json"), "w", newline="\n"), indent=1)
    return res


@torch.no_grad()
def nulls(model, dev, n):
    out = {}
    for N in DEPTHS:
        rng = np.random.default_rng(TEST_SEED + 1000 + N); ops, y = sample(rng, n, N); o = torch.tensor(ops, device=dev)
        sel = {"T0": o[:, :0], "T1": o[:, -1:], "T2": o[:, -2:], "first_op": o[:, :1], "last_op": o[:, -1:],
               "two_op": torch.stack([o[:, 0], o[:, -1]], 1)}
        out[str(N)] = {}
        for k, oo in sel.items():
            _, ans, _ = hard_rollout(model, oo)
            c = int((ans.cpu().numpy() == y).sum())
            out[str(N)][k] = {"acc": c / n, "upper95": wilson_upper(c, n)}
    return out


@torch.no_grad()
def mechanism(model, dev, n=2000, N=32, k=16):
    """Donor substitution at step k; irrelevant substitution (another code with the recipient's decoded state);
    same-semantic donors taken at different depths. Expected answers from the decoded donor state + recipient suffix."""
    dec = model.dec_logits(torch.arange(6, device=dev)).argmax(-1).cpu().numpy()
    rng = np.random.default_rng(TEST_SEED + 77)
    r_ops, r_y = sample(rng, n, N); d_ops, _ = sample(rng, n, N)
    r, d = torch.tensor(r_ops, device=dev), torch.tensor(d_ops, device=dev)
    _, r_ans, r_tr = hard_rollout(model, r)
    _, _, d_tr = hard_rollout(model, d)
    cd = d_tr[k]                                                     # donor code after k ops
    _, ans, _ = hard_rollout(model, r, start=cd, from_step=k)
    want = []
    for i in range(n):
        s = int(dec[int(cd[i])])
        for a in r_ops[i, k:]:
            s = COMP[a, s]
        want.append(s)
    res = {"donor_acc": float((ans.cpu().numpy() == np.array(want)).mean())}
    rc = r_tr[k].cpu().numpy()
    alt = [[c for c in range(6) if c != rc[i] and dec[c] == dec[rc[i]]] for i in range(n)]
    have = [i for i in range(n) if alt[i]]
    if have:
        st = torch.tensor([alt[i][0] for i in have], device=dev)
        _, a2, _ = hard_rollout(model, r[have], start=st, from_step=k)
        res["irrelevant_change_rate"] = float((a2.cpu().numpy() != r_ans.cpu().numpy()[have]).mean())
    else:
        res["irrelevant_change_rate"] = None
    res["irrelevant_n"] = len(have)
    res["irrelevant_note"] = "vacuous: every decoded state has exactly one code" if not have else ""
    # same-semantic donors from a different depth: a donor code reached at step k2 != k with the recipient's decoded state
    pool = {}
    for k2 in (3, 7, 25, 31):
        for i, c in enumerate(d_tr[k2].cpu().numpy()):
            pool.setdefault(int(dec[c]), []).append(int(c))
    st = torch.tensor([pool.get(int(dec[rc[i]]), [int(rc[i])])[i % len(pool.get(int(dec[rc[i]]), [0]))] for i in range(n)],
                      device=dev)
    _, a3, _ = hard_rollout(model, r, start=st, from_step=k)
    res["same_semantic_agreement"] = float((a3.cpu().numpy() == r_ans.cpu().numpy()).mean())
    res["same_semantic_distinct_codes"] = int(sum(st.cpu().numpy() != rc))
    return res


def train(a):
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    assert a.seed < TEST_SEED
    model = build(a.arm, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; seen = 0; toks = 0
    lens = (1, 2, 4, 8)
    for u in range(1, a.updates + 1):
        stage = min(3, (u - 1) * 4 // a.updates)
        L = int(rng.choice(lens[:stage + 1]))
        B = a.bs if a.arm != "fp_cmp" else max(1, D1_TOKENS // (L + 1))
        ops, y = sample(rng, B, L)
        o, yy = torch.tensor(ops, device=dev), torch.tensor(y, device=dev)
        model.train()
        if a.arm.startswith("fp"):
            loss = torch.nn.functional.cross_entropy(model(o), yy); toks += B * (L + 1)
        else:
            logT, logD = model.tables(); loss = marginal_loss(logT, logD, o, yy, reset=a.arm == "reset")
            toks += D1_TOKENS if a.arm != "tab" else 0
        seen += B
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append(loss.item())
        if u % 100 == 0 or u == a.updates:
            rec = {"update": u, "L": L, "loss": float(np.mean(hist)), "examples": seen, "gpt2_tokens": toks, "sec": round(time.time() - t0)}
            hist = []
            if u % 500 == 0 or u == a.updates:
                model.eval(); vr = np.random.default_rng(10_000 + u)
                for N in (8, 32):
                    vo, vy = sample(vr, 500, N)
                    with torch.no_grad():
                        rec[f"acc{N}"] = float((predict(model, a.arm, torch.tensor(vo, device=dev)).cpu().numpy() == vy).mean())
                if a.arm in ("d1", "tab", "reset"):
                    au = audit(model); rec["certified"] = au["certified"]; rec["trans_err"] = au["transition_errors"]
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            if time.time() - t0 > a.max_hours * 3600:
                print("TIMEOUT", flush=True); json.dump({"timeout": True}, open(os.path.join(a.out, "timeout.json"), "w")); break
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    res = evaluate(model, a.arm, dev, a.n_test, a.n_null, a.out)
    res["train"] = {"updates": u, "examples": seen, "gpt2_tokens": toks, "sec": round(time.time() - t0)}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("acc",) if k in res}), res.get("audit", {}).get("certified"), flush=True)
    print("DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["d1", "tab", "reset", "fp_ex", "fp_cmp"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_test", type=int, default=2000); ap.add_argument("--n_null", type=int, default=10_000)
    ap.add_argument("--max_hours", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
