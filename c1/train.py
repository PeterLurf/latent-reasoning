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
from c1.core import COMP, FP, Cell, layer_code_variance, MLPCell, Tab, TinyTF, gpt2_random_sd, audit, certificate, counter_check, hard_rollout, marginal_loss, sample, truth, wilson_upper  # noqa: E402

DEPTHS = (8, 16, 32, 64, 128)
TEST_SEED = 6700
def d1_tokens(M):
    """GPT-2 token positions per d1 update: 6M transition calls x 3 positions + M decoder calls x 2 = 20M (amendment
    67b-impl-1; the original constant 120 was right only for M = 6)."""
    return 20 * M


def gpt2(dev):
    sd = torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True)
    sd = {k[len("transformer."):] if k.startswith("transformer.") else k: v for k, v in sd.items()}
    return GPT2KV(sd).to(dev)


GPT2_INIT = "pretrained"


def build(arm, dev, seed=0):
    if arm == "tab":
        return Tab().to(dev)
    if arm == "mlp":                                                  # amendment 67-1
        return MLPCell().to(dev)
    if arm == "tf_small":
        return TinyTF().to(dev)
    if arm == "gpt2_rand":
        return Cell(GPT2KV(gpt2_random_sd(seed))).to(dev)
    if arm == "d1" and GPT2_INIT == "random":                         # §72 RF: random-init GPT-2 body in the d1 cell
        return Cell(GPT2KV(gpt2_random_sd(seed))).to(dev)
    if arm in ("fp_ex", "fp_cmp"):
        return FP(gpt2("cpu")).to(dev)
    return Cell(gpt2("cpu")).to(dev)


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
        ch = 250 if arm.startswith("fp") else 5000                   # memory only
        for i in range(0, n_test, ch):
            pred.append(predict(model, arm, ops_t[i:i + ch]).cpu())
        res["acc"][str(N)] = float((torch.cat(pred).numpy() == y).mean())
    if not arm.startswith("fp"):
        res["audit"] = audit(model); res["certificate"] = certificate(model)
        if res["certificate"]["certified"]:
            res["counter_check"] = counter_check(model)
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
    import c1.core as cc
    dec = model.dec_logits(torch.arange(cc.M, device=dev)).argmax(-1).cpu().numpy()
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
    alt = [[c for c in range(cc.M) if c != rc[i] and dec[c] == dec[rc[i]]] for i in range(n)]
    have = [i for i in range(n) if alt[i]]
    if have:
        st = torch.tensor([alt[i][0] for i in have], device=dev)
        _, a2, _ = hard_rollout(model, r[have], start=st, from_step=k)
        res["irrelevant_change_rate"] = float((a2.cpu().numpy() != r_ans.cpu().numpy()[have]).mean())
    else:
        res["irrelevant_change_rate"] = None
    res["irrelevant_n"] = len(have)
    # §67b clarification (fixed before any panel result was read): with M > 6 the substitute must be a code the machine
    # actually reaches (the certificate's reachable set); unreachable codes are outside the certified machine. The
    # unrestricted rate above is kept as descriptive.
    # Audit's condition: substitutes are codes VISITED in the donors' real hard rollouts (after >= 1 op), fixed per model
    # before the test, and drawn from those rollouts (not by enumerating codes).
    visited = sorted({int(c) for t in range(1, len(d_tr)) for c in d_tr[t].cpu().numpy()})
    altr = [[c for c in visited if c != rc[i] and dec[c] == dec[rc[i]]] for i in range(n)]
    haver = [i for i in range(n) if altr[i]]
    if haver:
        st = torch.tensor([altr[i][0] for i in haver], device=dev)
        _, a4, _ = hard_rollout(model, r[haver], start=st, from_step=k)
        res["irrelevant_change_rate_reachable"] = float((a4.cpu().numpy() != r_ans.cpu().numpy()[haver]).mean())
    else:
        res["irrelevant_change_rate_reachable"] = None
    res["irrelevant_n_reachable"] = len(haver); res["visited_codes"] = visited
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
    import c1.core as cc
    cc.M = a.M                                                       # number of codes (§67: 6; Theory's M = 12 option)
    dev = a.device; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    assert a.seed < TEST_SEED
    global GPT2_INIT
    GPT2_INIT = a.gpt2_init
    model = build(a.arm, dev, a.seed)
    if a.head_init == "zero" and hasattr(model, "zero_heads"):
        model.zero_heads()
    if a.head_init == "small" and hasattr(model, "small_heads"):
        model.small_heads()
    if a.code_init == "distinct" and hasattr(model, "distinct_codes"):
        model.distinct_codes()
    if a.head_norm == "ln" and hasattr(model, "ln_heads"):
        model.ln_heads()
    if a.readout == "bil" and hasattr(model, "bil_readout"):
        model.bil_readout()
    if a.feat_norm == "tablestd" and hasattr(model, "feat_norm"):
        model.feat_norm = "tablestd"                                 # §72 PF-std
    if a.freeze_gpt2 and hasattr(model, "g"):
        model.g.requires_grad_(False)                                # §72 PF / RF: every GPT-2 weight frozen (incl. its LNs)
    dist0 = model.distinctness() if getattr(model, "readout", "") == "bil" else None
    new = [p for n, p in model.named_parameters() if not n.startswith("g.") and p.requires_grad]   # codes, heads, tables
    old = [p for n, p in model.named_parameters() if n.startswith("g.") and p.requires_grad]       # GPT-2 (none if frozen)
    groups = [{"params": new, "lr": a.lr_new or a.lr}] + ([{"params": old, "lr": a.lr}] if old else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t0 = time.time(); hist = []; seen = 0; toks = 0
    lens = (1, 2, 4, 8)
    for u in range(1, a.updates + 1):
        stage = min(3, (u - 1) * 4 // a.updates)
        L = int(rng.choice(lens[:stage + 1]))
        B = a.bs if a.arm != "fp_cmp" else max(1, d1_tokens(a.M) // (L + 1))   # fp_cmp: matched to d1 at --M
        ops, y = sample(rng, B, L)
        o, yy = torch.tensor(ops, device=dev), torch.tensor(y, device=dev)
        model.train()
        if a.arm.startswith("fp"):
            loss = torch.nn.functional.cross_entropy(model(o), yy); toks += B * (L + 1)
        else:
            tau = 1.0 + (a.tau0 - 1.0) * max(0.0, 1.0 - (u - 1) / (0.5 * a.updates))   # §67b: tau0 -> 1 by halfway
            logT, logD = model.tables(tau); loss = marginal_loss(logT, logD, o, yy, reset=a.arm == "reset")
            toks += d1_tokens(a.M) if a.arm != "tab" else 0
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
                if not a.arm.startswith("fp"):
                    au = audit(model); rec["certified"] = au["certified"]; rec["trans_err"] = au["transition_errors"]; rec["cert"] = certificate(model)["certified"]
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            if time.time() - t0 > a.max_hours * 3600:
                print("TIMEOUT", flush=True); json.dump({"timeout": True}, open(os.path.join(a.out, "timeout.json"), "w")); break
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    model.eval()
    if hasattr(model, "_stats"):
        model._stats = None                                           # tablestd: final full-table statistics at test
    res = evaluate(model, a.arm, dev, a.n_test, a.n_null, a.out)
    if dist0 is not None:
        res["distinctness"] = {"init": dist0, "end": model.distinctness()}
        if not a.freeze_gpt2:
            res["layer_code_variance"] = layer_code_variance(model)
    res["train"] = {"updates": u, "examples": seen, "gpt2_tokens": toks, "sec": round(time.time() - t0)}
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("EVAL", json.dumps({k: res[k] for k in ("acc",) if k in res}), res.get("audit", {}).get("certified"), flush=True)
    print("DONE", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["d1", "tab", "reset", "fp_ex", "fp_cmp", "mlp", "tf_small", "gpt2_rand"])
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head_init", default="default", choices=["default", "zero", "small"])
    ap.add_argument("--M", type=int, default=6, help="number of codes (core.M)")
    ap.add_argument("--freeze_gpt2", action="store_true", help="§72: freeze every GPT-2 weight")
    ap.add_argument("--feat_norm", default="ln", choices=["ln", "tablestd"], help="§72: head-input normalisation")
    ap.add_argument("--gpt2_init", default="pretrained", choices=["pretrained", "random"], help="§72 RF")
    ap.add_argument("--readout", default="linear", choices=["linear", "bil"])
    ap.add_argument("--tau0", type=float, default=1.0, help="§67b: temperature annealed tau0 -> 1 over the first half")
    ap.add_argument("--head_norm", default="none", choices=["none", "ln"])
    ap.add_argument("--code_init", default="default", choices=["default", "distinct"])
    ap.add_argument("--lr_new", type=float, default=0.0, help="lr for codes / heads / tables (0 = --lr); pilot-fixed")
    ap.add_argument("--n_test", type=int, default=2000); ap.add_argument("--n_null", type=int, default=10_000)
    ap.add_argument("--max_hours", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
