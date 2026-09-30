"""THEORY §34 (T-C7c): edit and round-trip tests on silent latent thoughts, toy models, CPU, no training of the model.

The "translation" of thought h_i is the node the model's own sweep head reads next: v_i = source of the edge with the
largest layer-1 sweep-head attention at step i + 1 (the pointer of §1 / ptr_traj, per graph). On the long-chain probe the
reachable sequence is A0..A5, B0..B5 and its look-alike unreachable counterpart is D0..D5, C0..C5 (same index).

Encoders (text -> thought), fitted per model on separate probe graphs (seed 1, never the test graphs):
  * lin : h_i ~ mu_i + U id_emb[v_i]   (U: d_id -> d, ridge least squares over in-range thoughts i = 1..K-1; mu_i the step mean)
  * emb : the model's own node-token embedding (slot 2 of the id table), scaled by the model's own coefficient
          a = mean <h_i - mu_i, e_v> / |e_v|^2 over the same fitting thoughts (§33 T-C7c as first written)
Interventions at thought h_i (fed into step i + 1), on test graphs whose unedited v_i is on the reachable sequence at
index j <= 8:
  * edit  : h_i + Enc(u) - Enc(v_i), u = the unreachable counterpart at index j
  * rand  : h_i + r, r a random direction with |r| = |Enc(u) - Enc(v_i)|   (control)
  * lin edit / rand also at x2 and x4 (dose; the ridge fit shrinks U)
  * rtrip : lin only, full substitution mu_i + U id_emb[v_i] (residual discarded)
Measures (per model; test graphs = rollout.probe(512), seed 4247):
  M0 read     : share of eligible graphs whose step-(i+1) top edge has source u (the edit is read as written)
  M1 follow   : share whose step-(i+2) top edge has source = the counterpart at index j + 1 (u's successor)
  ins / del    : change (edited - unedited) in P("reachable") on the counterparts / the reachable nodes at index j+2, j+3,
                 i.e. beyond the one-pass horizon (nodes at j, j+1 are decided by the prefix pass; smoke model ph_c_s102)
  M3 collat    : share of graphs where any answer flips on the reachable nodes 0..j+1 or the counterparts 0..j-1
  IIA          : share of graphs whose answers equal BFS from the edited pointer at j+2, j+3 on both sequences
                 (counterparts reachable, reachable nodes not) with no collateral flip; base_ok: the unedited run's
                 answers at those four nodes are correct
  rtrip_agree : per-query answer agreement (all 24 queries) between full substitution and the unedited run;
                rtrip_read: share whose step-(i+1) top edge still has source v_i
    python theory/exp15_c7_edit.py --out theory/results/exp15_c7_edit.json [--runs runs/p30_s102 ...] [--steps 2 3]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("THEORY_THREADS", "2")))
from latent.graphs import NPB  # noqa: E402
from rollout import load, probe, rollout  # noqa: E402

T = lambda a: torch.as_tensor(a)


def seqs(g, br):
    bi = np.arange(len(g))
    reach = np.concatenate([g.nodes[bi, br["A"]], g.nodes[bi, br["B"]]], 1)       # (n, 12) index j -> depth j + 1
    decoy = np.concatenate([g.nodes[bi, br["D"]], g.nodes[bi, br["C"]]], 1)
    return reach, decoy


def top_src(o, g, k, head):
    """Source node of the sweep head's most-attended edge at step k (1-based)."""
    a = o["attn1"][k - 1][:, head, :o["P"] - 1]
    return g.src[np.arange(len(g)), a.argmax(1).numpy()]


def index_in(nodes, seq):
    """(n,) node ids -> index in each row of seq (n, 12), -1 if absent."""
    hit = seq == nodes[:, None]
    return np.where(hit.any(1), hit.argmax(1), -1)


@torch.no_grad()
def fit_encoders(m, K, head, n=512):
    g, br, _ = probe(n, seed=1)
    reach, _ = seqs(g, br)
    o = rollout(m, K, T(g.src), T(g.dst), T(g.start), K)
    E = m.id_emb.weight
    H, Y, S = [], [], []
    for i in range(1, K):
        v = top_src(o, g, i + 1, head)
        ok = index_in(v, reach) >= 0
        H.append(o["thoughts"][ok, i]); Y.append(T(v[ok])); S.append(torch.full((int(ok.sum()),), i))
    H, Y, S = torch.cat(H), torch.cat(Y), torch.cat(S)
    mu = {i: H[S == i].mean(0) for i in range(1, K)}
    R = H - torch.stack([mu[int(s)] for s in S])
    X = E[Y]                                                                  # (N, d_id)
    lam = 1e-2 * X.shape[0]
    U = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T @ R).T  # (d, d_id)
    ev = m.node_emb(Y)                                                        # (N, d) slot-2 embeddings
    a = float(((R * ev).sum(1) / (ev * ev).sum(1)).mean())
    r2 = 1 - float(((R - X @ U.T) ** 2).sum() / (R ** 2).sum())
    return {"mu": mu, "lin": lambda ids: E[T(ids)] @ U.T, "emb": lambda ids: a * m.node_emb(T(ids)),
            "a_emb": a, "r2_lin": r2, "n_fit": int(X.shape[0])}


@torch.no_grad()
def run_model(m, K, head, steps, g, br, seed=0):
    reach, decoy = seqs(g, br)
    n = len(g); bi = np.arange(n)
    q = np.concatenate([reach, decoy], 1)                                     # 24 queries: reach 0..11, decoy 0..11
    Q = T(q)
    enc = fit_encoders(m, K, head)
    base = rollout(m, K, T(g.src), T(g.dst), T(g.start), K, qnodes=Q)
    ans0 = (base["qlogit"] > 0).numpy()
    out = {"K": K, "head": head, "a_emb": enc["a_emb"], "r2_lin": enc["r2_lin"], "n_fit": enc["n_fit"]}
    gen = torch.Generator().manual_seed(seed)
    for i in steps:
        if i + 2 > K:
            continue
        v = top_src(base, g, i + 1, head)
        j = index_in(v, reach)
        el = (j >= 0) & (j <= 8)
        jj = np.where(el, j, 0)
        u = decoy[bi, jj]
        succ = decoy[bi, np.minimum(jj + 1, 11)]
        row = {"eligible": int(el.sum()), "base_M1": float((top_src(base, g, i + 2, head) == succ)[el].mean()),
               "base_read_u": float((v == u)[el].mean())}
        hi = base["thoughts"][:, i]
        for kind in ("lin", "emb"):
            delta = enc[kind](u) - enc[kind](v)
            rdir = torch.randn(delta.shape, generator=gen)
            rdir = rdir / rdir.norm(dim=1, keepdim=True) * delta.norm(dim=1, keepdim=True)
            conds = {"edit": delta, "rand": rdir}
            if kind == "lin":                                   # dose: ridge shrinks U, so x2 / x4 are reported too
                conds.update({"edit_x2": 2 * delta, "rand_x2": 2 * rdir, "edit_x4": 4 * delta, "rand_x4": 4 * rdir,
                              "rtrip": enc["mu"][i] + enc["lin"](v) - hi})
            for cname, dd in conds.items():
                feed = lambda k, h, hist, dd=dd: h + dd if k == i + 1 else h
                o = rollout(m, K, T(g.src), T(g.dst), T(g.start), K, qnodes=Q, feed=feed)
                ans = (o["qlogit"] > 0).numpy()
                s1, s2 = top_src(o, g, i + 1, head), top_src(o, g, i + 2, head)
                R_ = lambda a, off: a[bi, jj + off]                    # reachable sequence, index j + off (j <= 8)
                D_ = lambda a, off: a[bi, 12 + jj + off]               # unreachable counterpart, index j + off
                ins = D_(ans, 2) & D_(ans, 3)                          # counterparts j+2, j+3 answered reachable
                dele = ~R_(ans, 2) & ~R_(ans, 3)                       # reachable j+2, j+3 answered unreachable
                colmask = np.zeros((n, 24), bool)
                for b in range(n):
                    colmask[b, :jj[b] + 2] = True                      # reachable 0..j+1
                    colmask[b, 12:12 + jj[b]] = True                   # counterparts 0..j-1
                collat = ((ans != ans0) & colmask).any(1)
                chg = lambda f: float(np.mean([(f(ans, o).astype(float) - f(ans0, o))[el].mean() for o in (2, 3)]))
                r = {"M0_read": float((s1 == u)[el].mean()), "M1_follow": float((s2 == succ)[el].mean()),
                     "ins": chg(D_), "del": chg(R_),
                     "M3_collat": float(collat[el].mean()), "IIA": float((ins & dele & ~collat)[el].mean()),
                     "base_ok": float((R_(ans0, 2) & R_(ans0, 3) & ~D_(ans0, 2) & ~D_(ans0, 3))[el].mean()),
                     "agree_all": float((ans == ans0)[el].mean()), "read_v": float((s1 == v)[el].mean()),
                     "delta_norm_over_h": float((dd.norm(dim=1) / hi.norm(dim=1))[T(el)].mean())}
                row[f"{kind}_{cname}"] = r
        out[f"i{i}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="theory/results/exp15_c7_edit.json")
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--steps", nargs="*", type=int, default=[2, 3])
    a = ap.parse_args()
    R = json.load(open("theory/results/exp12_reanchor.json"))
    names = a.runs or ["runs/" + r for r in sorted(R, key=lambda r: (R[r]["extrap"], r))]
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    g, br, _ = probe(512)
    for path in names:
        name = os.path.basename(path.rstrip("/\\"))
        if name in res:
            continue
        m, c = load(path, "cpu")
        if name in R:
            head = R[name]["head"]
        else:
            from analyse_pilotA import ptr_traj
            gg, bb, EE = probe(512, seed=1)
            with torch.no_grad():
                _, _, head = ptr_traj(m, c["K"], gg, bb, EE, {})
        o = run_model(m, c["K"], head, a.steps, g, br)
        o["extrap"] = R[name]["extrap"] if name in R else None
        res[name] = o
        json.dump(res, open(a.out, "w", newline="\n"), indent=1)
        for i in a.steps:
            r = o.get(f"i{i}")
            if not r:
                continue
            f = lambda k: "%.2f/%.2f/%+.2f/%+.2f/%.2f" % tuple(r[k][x] for x in ("M0_read", "M1_follow", "ins", "del", "IIA"))
            print("%-10s %s i%d el %3d r2 %.2f | lin edit %s rand %s | emb edit %s rand %s | rtrip agree %.3f read_v %.2f" % (
                name, "EXT" if o["extrap"] else "   ", i, r["eligible"], o["r2_lin"], f("lin_edit"), f("lin_rand"),
                f("emb_edit"), f("emb_rand"), r["lin_rtrip"]["agree_all"], r["lin_rtrip"]["read_v"]), flush=True)


if __name__ == "__main__":
    main()
