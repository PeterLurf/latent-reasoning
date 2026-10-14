"""WARM-COT probe, part 3 (CPU, 3 threads, Idle; D:\\Latent read-only). Does C's trunk select e_1's first operand at its OWN operand position?
Input [EOT] + question + "\\n" + "<<" (C's prefix up to the first operand; "<<" = C's first CoT token), last position:
  op1_lm : C's (or raw's) LM-head argmax there == the first token of e_1's first operand (C's own next-token selection);
  probes : SLOT / OP as probe_plan (same items), on h_top and h_6 at that position.
Also op1_lm without the EOT prefix (C's training format). -> probe_op1.json"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import probe_plan as PP  # noqa: E402
from probe_plan import DM, ENC, EOT, G, pad, s5  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

LL = ENC.encode("<<")[0]
assert ENC.encode("<<4-2=2>>\n")[0] == LL


def gold_tok(r):
    ids = ENC.encode("<<%s>>\n" % r["eqs"][0]); return ids[1] if ids[0] == LL else None


@torch.no_grad()
def run(g, items, eot=True, bs=48):
    hd = 768 // g.H; F = {"op1_top": [], "op1_h6": []}; hits = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; seqs = [([EOT] if eot else []) + ENC.encode(r["q"] + "\n") + [LL] for r in ch]
        ids_, msk = pad(seqs); x = torch.tensor(ids_); msk = torch.tensor(msk).float(); h = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]; h6 = None
        for li in range(g.L):
            if li == 6:
                h6 = h
            h = DM._block(g, li, h, hd)
        last = (msk.sum(1) - 1).long(); ar = torch.arange(len(ch))
        hl = Fn.layer_norm(h, (768,), g.w("ln_f.weight"), g.w("ln_f.bias"))[ar, last]; nt = g.logits(hl).float().argmax(-1).tolist()
        hits += [int(t == gold_tok(r)) if gold_tok(r) is not None else 0 for t, r in zip(nt, ch)]
        F["op1_top"].append(Fn.layer_norm(h.float(), (768,))[ar, last]); F["op1_h6"].append(Fn.layer_norm(h6.float(), (768,))[ar, last])
    return {k: torch.cat(v).numpy() for k, v in F.items()}, float(np.mean(hits))


def main():
    T0 = time.time(); Rtr, _ = G.load_aug("train"); Rva, _ = G.load_aug("valid", filter_bad=False); rng = np.random.default_rng(31415)
    pool = [Rtr[i] for i in rng.permutation(len(Rtr))[:4 * (PP.N_TR + PP.N_HO)]]; tr_items, ho_items = [], []
    for r in pool:
        t = PP.targets(r)
        if t is None or t["op"] is None or t["slot"] is None:
            continue
        (tr_items if len(tr_items) < PP.N_TR else ho_items).append((r, t))
        if len(ho_items) >= PP.N_HO:
            break
    va_items = [(r, PP.targets(r)) for r in Rva]; va_items = [(r, t) for r, t in va_items if t and t["op"] is not None and t["slot"] is not None]
    res = {"n": [len(tr_items), len(ho_items), len(va_items)]}
    for mk, path in (("C", os.path.join(PP.ROOT, "boxes", "runs_fsq", "gsm_txt_aug_s1", "model.pt")), ("raw", None)):
        g = s5.load_gpt2("cpu")
        if path:
            g.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        g.eval(); res[mk] = {}
        F_tr, a_tr = run(g, [r for r, _ in tr_items]); F_ho, a_ho = run(g, [r for r, _ in ho_items]); F_va, a_va = run(g, [r for r, _ in va_items])
        res[mk]["op1_lm_eot"] = {"train": a_tr, "heldout": a_ho, "valid": a_va}
        _, a_va2 = run(g, [r for r, _ in va_items], eot=False); res[mk]["op1_lm_noeot_valid"] = a_va2
        print(mk, "op1_lm", json.dumps(res[mk]), round(time.time() - T0), flush=True)
        for nm, ncls in (("op", 4), ("slot", 5)):
            res[mk]["probe_" + nm] = PP.probe(F_tr, np.array([t[nm] for _, t in tr_items]), F_ho, np.array([t[nm] for _, t in ho_items]),
                                              F_va, np.array([t[nm] for _, t in va_items]), ncls)
            print(mk, nm, json.dumps(res[mk]["probe_" + nm]), round(time.time() - T0), flush=True)
        del g
    res["sec"] = round(time.time() - T0); json.dump(res, open(os.path.join(HERE, "probe_op1.json"), "w", newline="\n"), indent=1); print("DONE", flush=True)


if __name__ == "__main__":
    main()
