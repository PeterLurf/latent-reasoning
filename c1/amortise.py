"""THEORY §76(B) (+ amendment 76-2 B-std): amortise a certified free-table teacher into the stable GPT-2 cell.
    C1_GROUP=S4 python -m c1.amortise --teacher theory/results/c1_tab_local/S4_ge3_s2051.json --student pfstd \
        --device cuda:0 --out c1/runs/p76_ge3_pfstd_s2051
Teacher file: {"<seed>": {"certified", "table": [M][A] argmax next code, "dec": [M] element index, ...}}; element / op
indices follow theory/c1_group.comp_table(G), which equals c1.core's PERMS order with COMP_theory[s, a] = COMP_c1[a, s]
(checked at start). Start code 0, start state the identity (index 0).
Students: §73 PF-std (frozen pretrained GPT-2, bilinear ReLU readout on table-standardised features, stats frozen after
training) and PT-std (GPT-2 fine-tuned at 1e-4); single-token op symbols (c1.core.OP_TOK), M = teacher M, distinct
code init. Supervision: cross-entropy on the teacher's hard transitions over the teacher's reachable codes R (BFS from
code 0), with ~10 % of those (code, op) rows held out by seed, plus the decoder rows R -> dec. The student never sees a
final answer. lr 1e-4, up to 4,000 updates, stopping when every training row (transitions and decoder) is right.
Scoring (eval.json):
  * certificate on the student's OWN hard table (product automaton from (code 0, identity) against the true group; every
    reached pair after >= 1 op must decode right; codes outside R count when wrong) -- the main certificate;
  * the same with the held-out rows patched to the teacher's entries ("without held-out", descriptive);
  * held-out-row accuracy (student next code == teacher next code) -- composition vs copying;
  * training-row and decoder-row accuracy; hard accuracy at N 8-128 (test seed 6700 + N, n_test each);
  * teacher-alone certificate (recomputed here with the same function, plus the file's flag).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "theory"))
import c1.core as cc  # noqa: E402
from c1.train import build, DEPTHS, TEST_SEED  # noqa: E402


def reach(T):
    seen, fr = {0}, [0]
    while fr:
        c = fr.pop()
        for n in T[c]:
            if n not in seen:
                seen.add(n); fr.append(n)
    return sorted(seen)


def cert(T, D):
    """Product automaton over (code, true element) from (0, identity); bad = reached pairs (>= 1 op) decoding wrong."""
    seen, fr, bad = set(), [(0, 0)], []
    while fr:
        c, s = fr.pop()
        for a in range(cc.NG):
            nx = (int(T[c][a]), int(cc.COMP[a, s]))
            if nx not in seen:
                seen.add(nx); fr.append(nx)
                if D[nx[0]] != nx[1]:
                    bad.append(nx)
    return {"certified": not bad, "bad_pairs": len(bad), "bad_list": [list(b) for b in bad[:50]],
            "reachable_codes": len({c for c, _ in seen}), "reachable_pairs": len(seen)}


def acc_N(T, D, n):
    T = np.asarray(T); D = np.asarray(D); out = {}
    for N in DEPTHS:
        rng = np.random.default_rng(TEST_SEED + N); ops, y = cc.sample(rng, n, N)
        c = np.zeros(n, int)
        for t in range(N):
            c = T[c, ops[:, t]]
        out[str(N)] = float((D[c] == y).mean())
    return out


@torch.no_grad()
def student_tables(model):
    model.eval(); model._stats = None
    dev = model.code.device; M = cc.M
    codes = torch.arange(M, device=dev)
    D = model.dec_logits(codes).argmax(-1).tolist()
    T = model.step_logits(codes.repeat_interleave(cc.NG), torch.arange(cc.NG, device=dev).repeat(M)).argmax(-1).view(M, cc.NG).tolist()
    return T, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--student", required=True, choices=["pfstd", "ptstd"])
    ap.add_argument("--updates", type=int, default=4000); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.10); ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--allow_uncertified", action="store_true", help="plumbing smoke only (non-evidential)")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from c1_group import comp_table
    Ct, e, A = comp_table(cc.GROUP)
    assert A == cc.NG and e == 0 and np.array_equal(Ct, cc.COMP.T), "group order / convention mismatch"
    tf = json.load(open(a.teacher)); seed_s, t = next(iter(tf.items())); seed = int(seed_s)
    T0, D0 = t["table"], t["dec"]; M = len(T0)
    assert len(T0[0]) == A
    if not t["certified"] and not a.allow_uncertified:
        sys.exit(f"teacher {a.teacher} not certified: a system failure under §76; nothing to amortise")
    cc.M = M
    R = reach(T0)
    rows = [(c, op) for c in R for op in range(A)]
    rng = np.random.default_rng(seed)
    k = int(round(a.holdout * len(rows))); ho_idx = set(rng.choice(len(rows), size=k, replace=False).tolist())
    tr = [r for i, r in enumerate(rows) if i not in ho_idx]; ho = [r for i, r in enumerate(rows) if i in ho_idx]
    dev = a.device; torch.manual_seed(seed)
    model = build("d1", dev, seed)
    model.distinct_codes(); model.ln_heads(); model.bil_readout(); model.feat_norm = "tablestd"
    if a.student == "pfstd":
        model.g.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    trc = torch.tensor([c for c, _ in tr], device=dev); tro = torch.tensor([o for _, o in tr], device=dev)
    trt = torch.tensor([T0[c][o] for c, o in tr], device=dev)
    Rt = torch.tensor(R, device=dev); Rd = torch.tensor([D0[c] for c in R], device=dev)
    os.makedirs(a.out, exist_ok=True)
    meta = dict(vars(a), seed=seed, group=cc.GROUP, M=M, teacher_certified_file=bool(t["certified"]), R=len(R),
                rows=len(rows), train_rows=len(tr), heldout_rows=len(ho), evidential=bool(t["certified"]))
    json.dump(meta, open(os.path.join(a.out, "args.json"), "w", newline="\n"), indent=1)
    log = open(os.path.join(a.out, "log.jsonl"), "w", newline="\n"); t_start = time.time(); u = 0; done = False
    for u in range(1, a.updates + 1):
        model.train()
        logT, logD = model.tables(1.0)                                 # full table, full-table stats (§73 recipe)
        loss = F.nll_loss(logT[tro, trc], trt) + F.nll_loss(logD[Rt], Rd)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if u % 25 == 0 or u == a.updates:
            with torch.no_grad():
                ta = float((logT[tro, trc].argmax(-1) == trt).float().mean()); da = float((logD[Rt].argmax(-1) == Rd).float().mean())
            rec = {"update": u, "loss": loss.item(), "train_row_acc": ta, "dec_row_acc": da, "sec": round(time.time() - t_start)}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
            if ta == 1.0 and da == 1.0:
                Ts, Ds = student_tables(model)                         # confirm with the frozen-stats hard table
                if all(Ts[c][o] == T0[c][o] for c, o in tr) and all(Ds[c] == D0[c] for c in R):
                    done = True; break
    torch.save({k_: v for k_, v in model.state_dict().items() if not (a.student == "pfstd" and k_.startswith("g."))},
               os.path.join(a.out, "trainable.pt"))
    Ts, Ds = student_tables(model)                                     # stats frozen after training
    Tp = [row[:] for row in Ts]
    for c, o in ho:
        Tp[c][o] = T0[c][o]
    res = {"evidential": meta["evidential"], "stopped_at_100pct": done, "updates": u, "sec": round(time.time() - t_start),
           "train_row_acc": float(np.mean([Ts[c][o] == T0[c][o] for c, o in tr])),
           "dec_row_acc": float(np.mean([Ds[c] == D0[c] for c in R])),
           "heldout_row_acc": float(np.mean([Ts[c][o] == T0[c][o] for c, o in ho])) if ho else None,
           "heldout_rows": len(ho),
           "certificate": cert(Ts, Ds), "certificate_heldout_patched": cert(Tp, Ds),
           "acc": acc_N(Ts, Ds, a.n_test),
           "teacher": {"certified_file": bool(t["certified"]), "certificate": cert(T0, D0), "acc": acc_N(T0, D0, a.n_test),
                       "R": len(R)},
           "student_codes_reached": len(reach(Ts)), "outside_R": len(set(reach(Ts)) - set(R))}
    if not res["evidential"]:
        res["note"] = "PLUMBING SMOKE on an uncertified teacher: non-evidential, not a §76 result"
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w", newline="\n"), indent=1)
    print("DONE", a.out, {k_: res[k_] for k_ in ("evidential", "updates", "train_row_acc", "heldout_row_acc")}, flush=True)


if __name__ == "__main__":
    main()
