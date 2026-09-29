"""PROTOCOL7 T11: attention of each continuous thought onto ProsQA statements, for the ProsQA analogue of P7.

On make_long.py problems with extension m (default 4, so every problem has a statement at hop 7), saves per problem the
attention (summed over each statement's tokens, heads averaged, every layer) of thought positions 1..K' onto every
statement, plus labels: BFS hop of each statement from the root, and the matched extension chains — the correct
option's chain (reachable; hops orig+1 .. orig+m) and the wrong option's chain (unreachable look-alike, same length,
hung the same way). The analogue of the toy's B-vs-C excess is correct-chain minus wrong-chain attention at the same
chain position; analysis in the T11 scorer (floor fixed after the pilot, before confirmatory seeds).
    python p7_prosqa.py --ckpt <ckpt> --data data/prosqa_long_test.json --out res/p7p.npz [--no_mask] [--Kp 12] [--m 4]
"""
import argparse
import json

import numpy as np
import torch

from analyze import parse
from gpt2c import GPT2C, load_state


def chains(x, P):
    """Statement indices of the correct and wrong extension chains, in chain order."""
    m = x["ext"]
    new = [tuple(e) for e in x["edges"][-2 * m:]]
    stmt_edges = []
    sym = x["idx_to_symbol"]; ix = {s: i for i, s in enumerate(sym)}
    import re
    parts = re.split(r"(?<=\.) ", x["question"].rsplit(" Start: ", 1)[0] if x.get("query") else x["question"])
    for s in parts[:len(P["spans"])]:
        mm = re.match(r"^(?:Every (\w+)|(\w+)) is an? (\w+)\.$", s)
        stmt_edges.append((ix[mm.group(1) or mm.group(2)], ix[mm.group(3)]))
    pos = {e: i for i, e in enumerate(stmt_edges)}
    return [pos[e] for e in new[:m]], [pos[e] for e in new[m:]]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--Kp", type=int, default=12); ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT2C(load_state(a.ckpt, a.device)); model.mask_answer = not a.no_mask
    data = [x for x in json.load(open(a.data)) if x["ext"] == a.m][: a.n]
    rec = {"att": [], "hop": [], "orig_len": [], "chain_ok": [], "chain_bad": [], "margin": []}
    for x in data:
        P = parse(x)
        lp, qpos, att = model.rollout(P["ids"], a.Kp, P["answers"], trace=True)
        A = att[:, 0].mean(1).cpu().numpy()                                  # (L, Kp, T)
        per = np.stack([A[:, :, s:e].sum(-1) for s, e in P["spans"]], -1)   # (L, Kp, n_stmt)
        ok, bad = chains(x, P)
        rec["att"].append(per.astype(np.float16)); rec["hop"].append(np.array([h if h is not None else -1 for h in P["hop"]]))
        rec["orig_len"].append(x["orig_len"]); rec["chain_ok"].append(np.array(ok)); rec["chain_bad"].append(np.array(bad))
        rec["margin"].append(float(lp[0, 0] - lp[0, 1]))
    def obj(v):
        arr = np.empty(len(v), dtype=object)
        for i, e in enumerate(v):
            arr[i] = e
        return arr
    np.savez_compressed(a.out, **{k: obj(v) for k, v in rec.items()}, m=a.m, Kp=a.Kp)
    print("DONE p7_prosqa", a.out, len(data), flush=True)


if __name__ == "__main__":
    main()
