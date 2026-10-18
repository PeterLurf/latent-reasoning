"""TEXT-SELF draft check 2 (CPU fp32, 2 threads, idle): the TEXT-COT base model's own chain on FRESH items, two labeller protocols.
Precondition data for the design (the text analogue of 213-33...213-39); never a registered reading. Items: fresh nested170.generate draws from
default_rng(23180), n in --ns, --per_n each, redrawn on a hit against test_set(1000, n_max 24) + SERIAL (rng 17300); these syms must join every
later TEXT-SELF exclusion set. Model: boxes/runs_fsq/<run>/model.pt (read-only).
TF-GREEDY (the labeller form proposed in TS_draft §2.1): a form-valid step is UNIQUE up to its value token (the leftmost-ready handle replaced by one
leaf; values are single tokens, leaves carry no brackets), so greedy decoding emits a form-valid chain iff, teacher-forced on its own accepted
prefix, the argmax equals the structural candidate at every fixed token and is a value token at every value slot. Values are always the model's
argmax; the candidate is built from the tree SHAPE only (no N.apply / N.value in the labeller path; gold is scored afterwards, separately).
  FULL  : prompt [EOT] e0 ' =' then e1 ' =' e2 ... ' =' e_n EOT, one sequence (Lee-style: the model's own whole chain).
  REROOT: per step a fresh prompt [EOT] e_{k-1} ' =' then e_k + terminator (' =' if e_k has an operator, else EOT); e_k becomes the next root
          (the latent labeller's protocol, SELFDEPTH_v1 §2.1, with the text model's single step).
kept = every check passes (KEEP and CV-T); label = v_n. pi = share of kept chains whose every step equals the gold chain (N.cot_steps, scored after).
Also --gen_check: plain KV-cache greedy generation (as nested170.cot_answers, batch 1) on a few items, compared with TF-GREEDY (exactness of the form).
python -B ts_label_check.py --ns 9,10,11,12,13,14,15,16 --per_n 100 --protocols full,reroot"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); sys.dont_write_bytecode = True
sys.path.insert(0, "D:/Latent")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import boxes.nested170 as N  # noqa: E402

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
EOT = N.EOT; EQ = N.TID[" ="]; NUMS = set(int(x) for x in N.NUM_IDS); NUM2V = {int(t): v for v, t in enumerate(N.NUM_IDS)}


def load(run):
    from boxes import s5
    g = s5.load_gpt2("cpu"); sd = torch.load("D:/Latent/boxes/runs_fsq/%s/model.pt" % run, map_location="cpu", weights_only=True)
    g.load_state_dict(sd); g.eval(); return g


@torch.no_grad()
def fwd(g, ids, pos, past, kvalid):
    """batched GPT-2 forward with per-row positions and key validity: ids / pos (B, T) long; past per layer (k, v) (B, H, P, hd) or None;
    kvalid (B, P + T) bool (False = a pad key nobody may attend to; a query always sees itself). -> argmax ids (B, T), new past."""
    B, T = ids.shape; P = 0 if past is None else past[0][0].shape[2]; D = g.D; H = g.H; hd = D // H
    h = g.w("wte.weight")[ids] + g.w("wpe.weight")[pos]
    causal = torch.ones(T, P + T, dtype=torch.bool).tril(P)
    selfm = torch.zeros(T, P + T, dtype=torch.bool); selfm[torch.arange(T), P + torch.arange(T)] = True
    mask = (causal[None] & (kvalid[:, None, :] | selfm[None]))[:, None]     # (B, 1, T, P + T)
    new = []
    for i in range(g.L):
        pre = f"h.{i}."
        a = F.layer_norm(h, (D,), g.w(pre + "ln_1.weight"), g.w(pre + "ln_1.bias"))
        q, k, v = (a @ g.w(pre + "attn.c_attn.weight") + g.w(pre + "attn.c_attn.bias")).split(D, -1)
        q, k, v = (t.view(B, T, H, hd).transpose(1, 2) for t in (q, k, v))
        if past is not None:
            k = torch.cat([past[i][0], k], 2); v = torch.cat([past[i][1], v], 2)
        new.append((k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ g.w(pre + "attn.c_proj.weight") + g.w(pre + "attn.c_proj.bias")
        a = F.layer_norm(h, (D,), g.w(pre + "ln_2.weight"), g.w(pre + "ln_2.bias"))
        h = h + F.gelu(a @ g.w(pre + "mlp.c_fc.weight") + g.w(pre + "mlp.c_fc.bias"), approximate="tanh") @ g.w(pre + "mlp.c_proj.weight") + g.w(pre + "mlp.c_proj.bias")
    h = F.layer_norm(h, (D,), g.w("ln_f.weight"), g.w("ln_f.bias"))
    return (h @ g.w("wte.weight").T).argmax(-1), new


def reduce_shape(tree, v):
    """tree with its leftmost-ready handle replaced by the leaf v (in place; SHAPE operation, no apply). -> the new leaf node."""
    path = N._leftmost_ready(tree); tgt = tree
    for p in path:
        tgt = getattr(tgt, p)
    tgt.op = tgt.l = tgt.r = None; tgt.v = v; return tgt


def candidate(tree):
    """(prefix ids, suffix ids, has_op_after) of the next form-valid step from `tree` (copied), value slot open."""
    t2 = N._copy(tree); leaf = reduce_shape(t2, 0); syms, pos = N._render_pos(t2, leaf); ids = N.ids(syms)
    return ids[:pos], ids[pos + 1:], t2.op is not None


def run_chunks(g, chunks, lens0, past, kvalid):
    """feed one chunk per row (right-padded). lens0: tokens already in each row. -> argmax per row (list of arrays over the chunk), past, kvalid."""
    B = len(chunks); T = max(len(c) for c in chunks); ids = torch.full((B, T), EOT, dtype=torch.long); pos = torch.zeros((B, T), dtype=torch.long)
    kv_new = torch.zeros((B, T), dtype=torch.bool)
    for r, c in enumerate(chunks):
        ids[r, :len(c)] = torch.tensor(c); pos[r, :len(c)] = torch.arange(lens0[r], lens0[r] + len(c)); kv_new[r, :len(c)] = True
        pos[r, len(c):] = min(1023, lens0[r] + len(c))
    kvalid = kv_new if kvalid is None else torch.cat([kvalid, kv_new], 1)
    am, past = fwd(g, ids, pos, past, kvalid)
    return [am[r, :len(c)].numpy() for r, c in enumerate(chunks)], past, kvalid


def label_full(g, roots, bs):
    """FULL protocol, TF-GREEDY. -> per item {keep, fail_step, label, values}."""
    out = []
    for b0 in range(0, len(roots), bs):
        rs = roots[b0:b0 + bs]; B = len(rs); trees = [N._copy(r) for r in rs]; n0 = [sum(1 for s in N.render(r) if s.strip() in N.PREC) for r in rs]
        ok = [True] * B; vals = [[] for _ in range(B)]; fail = [None] * B; lens = [0] * B; past = None; kvalid = None; pend = [None] * B
        # stage 1 chunk: prompt + prefix(e1); later: [v_{k-1}] + suffix(e_{k-1}) + ' =' + prefix(e_k); last: [v_n] -> EOT
        for k in range(1, max(n0) + 2):
            chunks, checks = [], []
            for r in range(B):
                if k == 1:
                    pre, suf, more = candidate(trees[r]); pr = [EOT] + N.ids(N.render(trees[r])) + [EQ]; c = pr + pre
                    chk = [(len(pr) - 1 + j, pre[j]) for j in range(len(pre))]; pend[r] = (suf, more)
                elif k <= n0[r] + 1 and ok[r]:
                    suf, more = pend[r]; v = vals[r][-1]; mid = suf + ([EQ] if more else [EOT])
                    if more:
                        pre, suf2, more2 = candidate(trees[r]); pend[r] = (suf2, more2)
                    else:
                        pre = []
                    c = [N.NUM_IDS[v]] + mid + pre; chk = [(j, c[j + 1]) for j in range(len(c) - 1)]
                    if not more:
                        chk = [(j, c[j + 1]) for j in range(len(c) - 1)] + [(len(c) - 1, None)]   # after EOT nothing is read
                else:
                    c = [EOT]; chk = []
                chunks.append(c); checks.append(chk)
            am, past, kvalid = run_chunks(g, chunks, lens, past, kvalid)
            for r in range(B):
                c = chunks[r]
                if ok[r] and (k == 1 or k <= n0[r] + 1):
                    for (j, tok) in checks[r]:
                        if tok is None:
                            continue
                        if int(am[r][j]) != int(tok):
                            ok[r] = False; fail[r] = k; break
                    done_now = (k == n0[r] + 1)
                    if ok[r] and not done_now:
                        v_ = int(am[r][len(c) - 1])
                        if v_ not in NUMS:
                            ok[r] = False; fail[r] = k
                        else:
                            vals[r].append(NUM2V[v_]); reduce_shape(trees[r], NUM2V[v_])
                lens[r] += len(c)
        for r in range(B):
            out.append({"n": n0[r], "keep": bool(ok[r]), "fail_step": fail[r], "values": vals[r] if ok[r] else None, "label": vals[r][-1] if ok[r] else None})
    return out


def label_reroot(g, roots, bs):
    """REROOT protocol, TF-GREEDY: per step a fresh prompt; batched over items at the same step index."""
    S = len(roots); trees = [N._copy(r) for r in roots]; n0 = [sum(1 for s in N.render(r) if s.strip() in N.PREC) for r in roots]
    ok = [True] * S; vals = [[] for _ in range(S)]; fail = [None] * S
    for k in range(1, max(n0) + 1):
        act = [i for i in range(S) if ok[i] and k <= n0[i]]; act.sort(key=lambda i: len(N.render(trees[i])))
        for b0 in range(0, len(act), bs):
            part = act[b0:b0 + bs]; ca, cb, pm = [], [], []
            for i in part:
                pre, suf, more = candidate(trees[i]); pr = [EOT] + N.ids(N.render(trees[i])) + [EQ]; ca.append(pr + pre); pm.append((len(pr), pre, suf, more))
            am, past, kvalid = run_chunks(g, ca, [0] * len(part), None, None)
            vb = {}
            for r, i in enumerate(part):
                P, pre, suf, more = pm[r]
                bad = any(int(am[r][P - 1 + j]) != pre[j] for j in range(len(pre)))
                v_ = int(am[r][len(ca[r]) - 1])
                if bad or v_ not in NUMS:
                    ok[i] = False; fail[i] = k
                else:
                    vb[r] = NUM2V[v_]
            cb = [([N.NUM_IDS[vb[r]]] + pm[r][2]) if r in vb else [EOT] for r in range(len(part))]
            am2, _, _ = run_chunks(g, cb, [len(c) for c in ca], past, kvalid)
            for r, i in enumerate(part):
                if r not in vb:
                    continue
                P, pre, suf, more = pm[r]; tgt = suf + [EQ if more else EOT]
                if any(int(am2[r][j]) != tgt[j] for j in range(len(tgt))):
                    ok[i] = False; fail[i] = k
                else:
                    vals[i].append(vb[r]); reduce_shape(trees[i], vb[r])
    return [{"n": n0[i], "keep": bool(ok[i]), "fail_step": fail[i], "values": vals[i] if ok[i] else None, "label": vals[i][-1] if ok[i] else None} for i in range(S)]


def gold_values(root):
    """GOLD (scoring only, separate from the labeller): the value written at each step of N.cot_steps."""
    nd = N._copy(root); out = []
    while nd.op is not None:
        path = N._leftmost_ready(nd); tgt = nd
        for p in path:
            tgt = getattr(tgt, p)
        tgt.v = N.apply(tgt.op, tgt.l.v, tgt.r.v); tgt.op = tgt.l = tgt.r = None; out.append(tgt.v)
    return out


@torch.no_grad()
def gen_greedy(g, root, budget):
    """nested170.cot_answers' loop for one item (KV cache, GPT2KV forward), CPU fp32: the generated ids up to EOT (incl.)."""
    x = torch.tensor([[EOT] + N.ids(N.render(root)) + [EQ]]); h, past = g(g.embed(x), None); gen = []
    for _ in range(budget):
        nt = g.logits(h[:, -1]).argmax(-1); gen.append(int(nt))
        if int(nt) == EOT or x.shape[1] + len(gen) >= 1024:
            break
        h, past = g(g.embed(nt[:, None]), past)
    return gen


def chain_ids(root, values):
    t2 = N._copy(root); out = []
    for k, v in enumerate(values):
        reduce_shape(t2, v); out += N.ids(N.render(t2)) + ([EQ] if t2.op is not None else [EOT])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="s170_cot_s1"); ap.add_argument("--ns", default="9,10,11,12,13,14,15,16")
    ap.add_argument("--per_n", type=int, default=100); ap.add_argument("--protocols", default="full,reroot"); ap.add_argument("--bs", type=int, default=50)
    ap.add_argument("--gen_check", type=int, default=0, help="items per n for the KV-cache greedy comparison (first ns only)"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); ns = [int(x) for x in a.ns.split(",")]; t0 = time.time()
    excl = {tuple(it["syms"]) for it in N.test_set(1000, n_max=24)}; r_sp = np.random.default_rng(17300)
    for _ in range(2200):
        excl.add(tuple(N.render(N.generate(r_sp, 8)[0])))
    rng = np.random.default_rng(23180); items = {}; hits = {}
    for n in range(9, 21):                                                  # ONE stream: 200 per n, n 9..20 in order; each run uses the first per_n
        items[n] = []; hits[n] = 0
        while len(items[n]) < 200:
            r_, _ = N.generate(rng, n)
            if tuple(N.render(r_)) in excl:
                hits[n] += 1; continue
            items[n].append(r_)
        items[n] = items[n][:a.per_n]
    g = load(a.run); res = {"run": a.run, "per_n": a.per_n, "seed": 23180, "excl_hits": hits, "dtype": "fp32 cpu", "by_n": {}, "items": {}}
    print("loaded %.0f s" % (time.time() - t0), flush=True)
    for n in ns:
        roots = items[n]; gv = [gold_values(r) for r in roots]; rn = {}
        for proto in a.protocols.split(","):
            t1 = time.time(); lab = (label_full if proto == "full" else label_reroot)(g, roots, a.bs)
            kept = [i for i, l_ in enumerate(lab) if l_["keep"]]; corr = [i for i in kept if lab[i]["values"] == gv[i]]
            fs = [l_["fail_step"] for l_ in lab if not l_["keep"]]
            rn[proto] = {"keep": len(kept) / len(roots), "n_kept": len(kept), "pi": (len(corr) / len(kept)) if kept else None, "kept_wrong": len(kept) - len(corr),
                         "kept_and_correct_share": len(corr) / len(roots), "fail_step_hist": {str(s): fs.count(s) for s in sorted(set(fs))}, "sec": time.time() - t1}
            res["items"]["%d_%s" % (n, proto)] = [[int(l_["keep"]), l_["fail_step"], int(l_["values"] == gv[i]) if l_["keep"] else None] for i, l_ in enumerate(lab)]
            print(n, proto, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rn[proto].items()}, flush=True)
            if proto == "full" and a.gen_check:
                agree = 0; recs = []
                for i in range(min(a.gen_check, len(roots))):
                    gen = gen_greedy(g, roots[i], N.cot_ids(roots[i]).__len__() + 50)
                    tf = chain_ids(roots[i], lab[i]["values"]) if lab[i]["keep"] else None
                    same = (gen == tf) if tf is not None else True
                    if tf is None:                                          # TF says not form-valid: the generated chain must not be any form-valid chain
                        tt = N._copy(roots[i]); pos = 0; valid = True; vv = []
                        while tt.op is not None and valid:
                            pre, suf, more = candidate(tt); L_ = len(pre) + 1 + len(suf) + 1; seg = gen[pos:pos + L_]
                            valid = len(seg) == L_ and seg[:len(pre)] == pre and seg[len(pre)] in NUMS and seg[len(pre) + 1:] == suf + [EQ if more else EOT]
                            if valid:
                                reduce_shape(tt, NUM2V[seg[len(pre)]]); pos += L_
                        same = not (valid and pos == len(gen))
                    agree += same; recs.append(int(same))
                rn["gen_check"] = {"items": len(recs), "agree": agree}; print("gen_check", n, agree, "/", len(recs), flush=True)
        res["by_n"][str(n)] = rn
        json.dump(res, open(os.path.join(HERE, "ts_label_check%s.json" % a.tag), "w", newline="\n"), indent=1)
    res["sec"] = time.time() - t0; json.dump(res, open(os.path.join(HERE, "ts_label_check%s.json" % a.tag), "w", newline="\n"), indent=1)


if __name__ == "__main__":
    main()
