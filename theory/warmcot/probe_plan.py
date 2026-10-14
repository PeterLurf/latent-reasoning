"""WARM-COT precondition probes (CPU, <= 3 threads, idle priority; reads D:\\Latent read-only, writes only next to this script).

P-PLAN: is e_1's plan linearly readable from the think-pass features the memory writes, and does C's CoT training add to it?
  think input = [EOT] + question (ProsMem.think_inputs, no "\\n"); features at the LAST think token and mean over think tokens of
  (a) h6 = the residual entering block 6 (the open-loop write source, gamma = 0) and (b) h_top = the residual after block 11 (the write
  source once gamma = 1); both layer-normalised without affine (the probe standardises anyway).
  targets: OP = e_1's first binary operator (+ - * /); SLOT = index of e_1's first operand among the question's distinct numbers (<= 4).
  models: raw GPT-2 (prosqa/ckpt/gpt2_pretrained.bin) and §200 C (boxes/runs_fsq/gsm_txt_aug_s1/model.pt).
  probe: multinomial logistic regression (L2 picked on an inner split), trained on Aug-train items, scored on held-out Aug-train items
  and on Aug valid (the test-like split).
K0-STEP0: the WARM cell's K'0 at step 0 = C's trunk answering directly after [EOT] + question + "\\n" + "###" (W_o = 0 and fade = 0, so the
  read adds exactly 0: deltamem.read_fixed l.176). Also C's own format without the EOT prefix, and raw GPT-2.
E1-INCONTEXT: C's greedy first CoT line vs "<<e_1>>" (C's format, and with the EOT prefix), Aug valid.
"""
import ctypes
import json
import os
import re
import sys
import time

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True; os.environ["BOXES_NB"] = "5"
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)   # IDLE_PRIORITY_CLASS
except Exception:
    pass
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
ROOT = r"D:\Latent"; sys.path.insert(0, ROOT)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, pad  # noqa: E402
from bench.tasks import norm_number  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__)); NUM = re.compile(r"\d*\.?\d+"); T0 = time.time()
N_TR, N_HO = int(os.environ.get("N_TR", 2400)), int(os.environ.get("N_HO", 800))


def qnums(q):
    vals = []
    for s in NUM.findall(re.sub(r"(?<=\d),(?=\d{3})", "", q)):
        v = float(s)
        if not any(abs(v - u) < 1e-9 for u in vals):
            vals.append(v)
    return vals


def targets(r):
    if not r["eqs"]:
        return None
    lhs = r["eqs"][0].split("=")[0].strip(); m_ = NUM.search(lhs)
    if not m_:
        return None
    rest = lhs[m_.end():]; op = next((c for c in rest if c in "+-*/"), None)
    v = float(m_.group()); qs = qnums(r["q"]); slot = next((i for i, u in enumerate(qs) if abs(u - v) < 1e-9), None)
    return {"op": "+-*/".index(op) if op else None, "slot": slot if (slot is not None and slot <= 4) else None}


@torch.no_grad()
def feats(g, items, bs=48):
    hd = 768 // g.H; out = {k: [] for k in ("h6_last", "h6_mean", "top_last", "top_mean")}
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]; ids_, msk = pad([[EOT] + ENC.encode(r["q"]) for r in ch]); x = torch.tensor(ids_); msk = torch.tensor(msk).float()
        h = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]; h6 = None
        for li in range(g.L):
            if li == 6:
                h6 = h
            h = DM._block(g, li, h, hd)
        last = (msk.sum(1) - 1).long(); ar = torch.arange(len(ch))
        for nm, hh in (("h6", h6), ("top", h)):
            z = Fn.layer_norm(hh.float(), (768,))
            out[nm + "_last"].append(z[ar, last]); out[nm + "_mean"].append((z * msk[..., None]).sum(1) / msk.sum(1, keepdim=True))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def logreg(Xtr, ytr, Xte_list, ncls, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; X = torch.tensor((Xtr - mu) / sd, dtype=torch.float64); y = torch.tensor(ytr)
    W = torch.zeros(X.shape[1], ncls, dtype=torch.float64, requires_grad=True); b = torch.zeros(ncls, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

    def clo():
        opt.zero_grad(); L = Fn.cross_entropy(X @ W + b, y) + lam * (W ** 2).sum(); L.backward(); return L
    opt.step(clo)
    with torch.no_grad():
        return [float(((torch.tensor((Xe - mu) / sd, dtype=torch.float64) @ W + b).argmax(1) == torch.tensor(ye)).double().mean()) for Xe, ye in Xte_list]


def probe(F_tr, y_tr, F_ho, y_ho, F_va, y_va, ncls):
    rs = {}
    for fk in F_tr:
        n_in = int(0.8 * len(y_tr)); best = None
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            a_in = logreg(F_tr[fk][:n_in], y_tr[:n_in], [(F_tr[fk][n_in:], y_tr[n_in:])], ncls, lam)[0]
            if best is None or a_in > best[1]:
                best = (lam, a_in)
        a_ho, a_va = logreg(F_tr[fk], y_tr, [(F_ho[fk], y_ho), (F_va[fk], y_va)], ncls, best[0]); rs[fk] = {"lam": best[0], "inner": best[1], "heldout": a_ho, "valid": a_va}
    return rs


@torch.no_grad()
def greedy(g, prefixes, maxn, stop_nl=True, bs=50):
    NL = {t for t in range(ENC.n_vocab) if "\n" in ENC.decode([t])}; out = []
    for i in range(0, len(prefixes), bs):
        cur = [list(p) for p in prefixes[i:i + bs]]; gen = [[] for _ in cur]; done = [False] * len(cur)
        for _ in range(maxn):
            ids_, _ = pad(cur); x = torch.tensor(ids_); last = torch.tensor([len(c) - 1 for c in cur])
            h = g(g.embed(x), None)[0]; nt = g.logits(h[torch.arange(len(cur)), last]).float().argmax(-1).tolist()
            for k, t in enumerate(nt):
                if done[k]:
                    continue
                if t == EOS or (stop_nl and t in NL) or t == ANS:
                    done[k] = True
                else:
                    gen[k].append(t); cur[k].append(t)
            if all(done):
                break
        out += [ENC.decode(x_) for x_ in gen]
    return out


def main():
    Rtr, _ = G.load_aug("train"); Rva, _ = G.load_aug("valid", filter_bad=False); rng = np.random.default_rng(31415)
    Rva = Rva[:int(os.environ.get("N_VA", len(Rva)))]
    pool = [Rtr[i] for i in rng.permutation(len(Rtr))[:4 * (N_TR + N_HO)]]
    sel = {"op": {"tr": [], "ho": []}, "slot": {"tr": [], "ho": []}}; tr_items, ho_items = [], []
    for r in pool:
        t = targets(r)
        if t is None or t["op"] is None or t["slot"] is None:
            continue
        (tr_items if len(tr_items) < N_TR else ho_items).append((r, t))
        if len(ho_items) >= N_HO:
            break
    va_items = [(r, targets(r)) for r in Rva]; va_items = [(r, t) for r, t in va_items if t and t["op"] is not None and t["slot"] is not None]
    res = {"n_train": len(tr_items), "n_heldout": len(ho_items), "n_valid": len(va_items)}
    for nm in ("op", "slot"):
        ys = np.array([t[nm] for _, t in tr_items]); vals, cnt = np.unique(ys, return_counts=True)
        res["majority_" + nm] = {"train_share": float(cnt.max() / cnt.sum()), "heldout": float(np.mean([t[nm] == vals[cnt.argmax()] for _, t in ho_items])),
                                 "valid": float(np.mean([t[nm] == vals[cnt.argmax()] for _, t in va_items]))}
    models = {"raw": None, "C": os.path.join(ROOT, "boxes", "runs_fsq", "gsm_txt_aug_s1", "model.pt")}
    Rva_all = [{**G.encode(r, ENC, EOS)} for r in Rva]
    for mk, path in models.items():
        g = s5.load_gpt2("cpu")
        if path:
            g.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        g.eval(); print(mk, "loaded", round(time.time() - T0), flush=True)
        F_tr = feats(g, [r for r, _ in tr_items]); F_ho = feats(g, [r for r, _ in ho_items]); F_va = feats(g, [r for r, _ in va_items])
        print(mk, "feats", round(time.time() - T0), flush=True); res[mk] = {}
        for nm, ncls in (("op", 4), ("slot", 5)):
            res[mk]["probe_" + nm] = probe(F_tr, np.array([t[nm] for _, t in tr_items]), F_ho, np.array([t[nm] for _, t in ho_items]),
                                           F_va, np.array([t[nm] for _, t in va_items]), ncls)
            print(mk, nm, json.dumps(res[mk]["probe_" + nm]), round(time.time() - T0), flush=True)
        # K'0 at warm step 0 (latent answer path with an exactly-zero read): [EOT] + pre + ###
        p_eot = greedy(g, [[EOT] + it["pre"] + [ANS] for it in Rva_all], 12)
        res[mk]["direct_eot"] = float(np.mean([norm_number(p or "") == norm_number(it["ans"]) for p, it in zip(p_eot, Rva_all)]))
        p_c = greedy(g, [it["pre"] + [ANS] for it in Rva_all], 12)
        res[mk]["direct_noeot"] = float(np.mean([norm_number(p or "") == norm_number(it["ans"]) for p, it in zip(p_c, Rva_all)]))
        print(mk, "direct", res[mk]["direct_eot"], res[mk]["direct_noeot"], round(time.time() - T0), flush=True)
        if mk == "C":
            for tag, pre_ in (("noeot", lambda it: it["pre"]), ("eot", lambda it: [EOT] + it["pre"])):
                e1 = greedy(g, [pre_(it) for it in Rva_all], 24)
                ok = [bool(r["eqs"]) and e.strip() == "<<%s>>" % r["eqs"][0] for e, r in zip(e1, Rva)]
                okv = []
                for e, r in zip(e1, Rva):                                   # value right given the model's own expression
                    m_ = re.match(r"\s*<<([^=]+)=([^>]+)>>", e)
                    okv.append(bool(m_) and G.check_eq(m_.group(1).strip() + "=" + m_.group(2).strip()) in ("exact", "round"))
                res[mk]["e1_exact_" + tag] = float(np.mean(ok)); res[mk]["e1_selfconsistent_value_" + tag] = float(np.mean(okv))
                print(mk, "e1", tag, res[mk]["e1_exact_" + tag], res[mk]["e1_selfconsistent_value_" + tag], round(time.time() - T0), flush=True)
        del g
    res["sec"] = round(time.time() - T0)
    json.dump(res, open(os.path.join(OUT, os.environ.get("OUTF", "probe_plan.json")), "w", newline="\n"), indent=1); print("DONE", json.dumps(res)[:3000], flush=True)


if __name__ == "__main__":
    main()
