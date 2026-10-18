"""TEXT-SELF draft check 1 (CPU, no model): token lengths of the TEXT-COT chain per n, from STRUCTURE only.
For every item the form-valid chain's length is a function of the input's tree shape (each value is one GPT-2 token, leaves never carry brackets),
so the context need is gold-free. Here: struct_len (no apply) is asserted equal to len(N.cot_ids(root)) on the test items (a check only).
Outputs (JSON beside): per n 1..24 of test_set(1000, n_max 24) (generator statistics, no model output; disclosed): prompt P = len(syms) + 2,
chain C = len(cot_ids) (incl. EOT), need = P + C - 1 = the largest position index + 1 the generation feeds (GPT-2 wpe: 1024 rows), feasible iff
P + C <= 1025; quantiles. Also the training shapes for fresh draws (rng 23190, own stream; not a pool): max (P + C) over 64-record batches of the
round mixtures T1 (n 9-12 uniform) and T2 (2/3 n 13-16, 1/3 n 9-12), and of the plain stream (n 1-8 uniform).
python -B ts_tokens.py  (OMP/MKL threads 2)"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); sys.dont_write_bytecode = True
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402

import boxes.nested170 as N  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ts_tokens.json")


def struct_chain(root):
    """the chain's expressions e_1..e_n rendered with placeholder values (0) at each new leaf: lengths only (no apply)."""
    nd = N._copy(root); out = []
    while nd.op is not None:
        path = N._leftmost_ready(nd); tgt = nd
        for p in path:
            tgt = getattr(tgt, p)
        tgt.op = tgt.l = tgt.r = None; tgt.v = 0; out.append(len(N.render(nd)))
    return out


def struct_len(root):
    L = struct_chain(root); return sum(L) + (len(L) - 1) + 1          # e_k tokens + ' =' between + EOT


def q(a, ps=(0.5, 0.9, 0.99, 1.0)):
    a = np.asarray(a); return {str(p): float(np.quantile(a, p)) for p in ps}


def main():
    TE = N.test_set(1000, n_max=24); res = {"per_n": {}, "def": "P = len(syms)+2 (EOT, expr, ' ='); C = chain tokens incl. EOT; feasible iff P + C <= 1025 (wpe 1024 rows)"}
    for n in range(1, 25):
        its = [it for it in TE if it["n"] == n]; P = np.array([len(it["syms"]) + 2 for it in its]); C = np.array([struct_len(it["root"]) for it in its])
        if n <= 24:
            gold = np.array([len(N.cot_ids(it["root"])) for it in its]); assert (gold == C).all(), ("struct len != cot_ids len", n)
        tot = P + C; feas = tot <= 1025
        res["per_n"][str(n)] = {"P_mean": float(P.mean()), "P_max": int(P.max()), "C_mean": float(C.mean()), "C_max": int(C.max()), "tot_mean": float(tot.mean()),
                                "tot_max": int(tot.max()), "tot_q": q(tot), "feasible_share": float(feas.mean()), "n_infeasible": int((~feas).sum()),
                                "share_le_512": float((tot <= 512).mean()), "share_le_768": float((tot <= 768).mean())}
        print(n, res["per_n"][str(n)]["tot_mean"], res["per_n"][str(n)]["tot_max"], res["per_n"][str(n)]["feasible_share"], flush=True)
    # training shapes (fresh draws, own stream 23190; NOT a pool, never used again)
    rng = np.random.default_rng(23190); sh = {}

    def draw(ns, w, k):
        out = []
        for _ in range(k):
            n = int(rng.choice(ns, p=w)); r, _ = N.generate(rng, n); out.append(len(N.render(r)) + 2 + struct_len(r))
        return np.array(out)
    for name, ns, w in (("plain_n1_8", list(range(1, 9)), [1 / 8] * 8), ("T1_n9_12", [9, 10, 11, 12], [0.25] * 4),
                        ("T2_mix", [9, 10, 11, 12, 13, 14, 15, 16], [1 / 12] * 4 + [1 / 6] * 4), ("n16", [16], [1.0]), ("n12", [12], [1.0])):
        L = draw(ns, w, 64 * 200); B = L.reshape(200, 64)
        sh[name] = {"tot_mean": float(L.mean()), "tot_max": int(L.max()), "batch64_max_mean": float(B.max(1).mean()), "batch64_max_max": int(B.max(1).max()),
                    "batch16_max_mean": float(L.reshape(800, 16).max(1).mean()), "pad_eff64": float(L.mean() / B.max(1).mean())}
        print(name, sh[name], flush=True)
    res["train_shapes"] = sh
    json.dump(res, open(OUT, "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
