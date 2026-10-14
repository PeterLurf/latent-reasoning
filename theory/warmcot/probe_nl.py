"""WARM-COT probe, part 2 (CPU, 3 threads, Idle priority; reads D:\\Latent read-only). The same items and targets as probe_plan.py, an extended L2 grid
(probe_plan's best lambda sat at its grid edge 0.1), and two input formats:
  think : [EOT] + question            (the latent think input; features at the last token and the token mean of h_top and h6)
  nl    : [EOT] + question + "\\n"     (C's planning position: the token after which C emits <<e_1>>; last-token h_top)
-> probe_nl.json."""
import ctypes
import json
import os
import sys
import time

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True; os.environ["BOXES_NB"] = "5"
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
except Exception:
    pass
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import probe_plan as PP  # noqa: E402  (its module body only imports; main() is not run)
from probe_plan import DM, ENC, EOT, G, pad, s5  # noqa: E402

LAMS = (0.03, 0.1, 0.3, 1.0, 3.0)


TOK = []


def num_positions(q):
    """token index (in ENC.encode(q)) of the last BPE token of the first occurrence of each distinct question number, in probe_plan.qnums order;
    None if the per-token decode does not reproduce q or the numbers cannot be located."""
    ids = ENC.encode(q); pieces = [ENC.decode([t]) for t in ids]
    if "".join(pieces) != q:
        return None
    ends = np.cumsum([len(x) for x in pieces]); qq = q; vals, pos = [], []
    for m_ in PP.NUM.finditer(qq):
        s0, e0 = m_.span(); txt = m_.group()
        # comma-grouped numbers ("1,000"): extend the match like qnums does
        j = e0
        while j + 4 <= len(qq) and qq[j] == "," and qq[j + 1:j + 4].isdigit() and (j + 4 == len(qq) or not qq[j + 4].isdigit()):
            txt += qq[j + 1:j + 4]; j += 4
        if s0 > 0 and qq[s0 - 1] == "," and s0 >= 2 and qq[s0 - 2].isdigit():
            continue                                                        # a comma group already absorbed
        v = float(txt)
        if any(abs(v - u) < 1e-9 for u in vals):
            continue
        vals.append(v); pos.append(int(np.searchsorted(ends, j, side="left")))
    ref = PP.qnums(q)
    if len(ref) != len(vals) or any(abs(a - b) > 1e-9 for a, b in zip(ref, vals)):
        return None
    return pos


def tok_probe(tr, ho, va, which, lams=LAMS):
    """binary 'is the first operand' on per-number features; slot accuracy = argmax over the item's numbers (items with >= 2 numbers)."""
    def flat(S):
        X, y = [], []
        for feats_, slot in S:
            for k, f in enumerate(feats_):
                X.append(f[which]); y.append(int(k == slot))
        return np.array(X), np.array(y)

    def slot_acc(S, W, b, mu, sd):
        ok = []
        for feats_, slot in S:
            Z = (np.array([f[which] for f in feats_]) - mu) / sd; sc = Z @ W[:, 1] - Z @ W[:, 0] + (b[1] - b[0]); ok.append(int(np.argmax(sc)) == slot)
        return float(np.mean(ok))

    def fit(S, lam):
        X, y = flat(S); mu, sd = X.mean(0), X.std(0) + 1e-6; Xt = torch.tensor((X - mu) / sd, dtype=torch.float64); yt = torch.tensor(y)
        W = torch.zeros(X.shape[1], 2, dtype=torch.float64, requires_grad=True); b = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

        def clo():
            opt.zero_grad(); L = Fn.cross_entropy(Xt @ W + b, yt) + lam * (W ** 2).sum(); L.backward(); return L
        opt.step(clo); return W.detach().numpy(), b.detach().numpy(), mu, sd
    n_in = int(0.8 * len(tr)); best = None
    for lam in lams:
        a_ = slot_acc(tr[n_in:], *fit(tr[:n_in], lam))
        if best is None or a_ > best[1]:
            best = (lam, a_)
    P = fit(tr, best[0]); return {"lam": best[0], "inner": best[1], "heldout": slot_acc(ho, *P), "valid": slot_acc(va, *P),
                                  "n": [len(tr), len(ho), len(va)], "chance_heldout": float(np.mean([1 / len(f) for f, _ in ho])),
                                  "first_number_heldout": float(np.mean([s_ == 0 for _, s_ in ho])), "first_number_valid": float(np.mean([s_ == 0 for _, s_ in va]))}


@torch.no_grad()
def feats2(g, items, bs=48):
    hd = 768 // g.H; out = {k: [] for k in ("h6_mean", "top_last", "top_mean", "nl_top_last", "nl_h6_last")}
    for fmt in ("think", "nl"):
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]; seqs = [[EOT] + ENC.encode(r["q"] + ("\n" if fmt == "nl" else "")) for r in ch]
            ids_, msk = pad(seqs); x = torch.tensor(ids_); msk = torch.tensor(msk).float(); h = g.embed(x) + DM.wpe_slice(g, x.shape[1])[None]; h6 = None
            for li in range(g.L):
                if li == 6:
                    h6 = h
                h = DM._block(g, li, h, hd)
            last = (msk.sum(1) - 1).long(); ar = torch.arange(len(ch)); zt = Fn.layer_norm(h.float(), (768,)); z6 = Fn.layer_norm(h6.float(), (768,))
            if fmt == "think":
                out["top_last"].append(zt[ar, last]); out["top_mean"].append((zt * msk[..., None]).sum(1) / msk.sum(1, keepdim=True))
                for r_, r in enumerate(ch):                                 # per distinct question number: h_top / h_11-in at its last BPE token
                    pos = num_positions(r["q"])
                    TOK.append([(zt[r_, 1 + p_].numpy(), z6[r_, 1 + p_].numpy()) for p_ in pos] if pos is not None else None)
                out["h6_mean"].append((z6 * msk[..., None]).sum(1) / msk.sum(1, keepdim=True))
            else:
                out["nl_top_last"].append(zt[ar, last]); out["nl_h6_last"].append(z6[ar, last])
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def probe2(F_tr, y_tr, F_ho, y_ho, F_va, y_va, ncls):
    rs = {}
    for fk in F_tr:
        n_in = int(0.8 * len(y_tr)); best = None
        for lam in LAMS:
            a_in = PP.logreg(F_tr[fk][:n_in], y_tr[:n_in], [(F_tr[fk][n_in:], y_tr[n_in:])], ncls, lam)[0]
            if best is None or a_in > best[1]:
                best = (lam, a_in)
        a_ho, a_va = PP.logreg(F_tr[fk], y_tr, [(F_ho[fk], y_ho), (F_va[fk], y_va)], ncls, best[0])
        rs[fk] = {"lam": best[0], "inner": best[1], "heldout": a_ho, "valid": a_va}
    return rs


def main():
    T0 = time.time(); Rtr, _ = G.load_aug("train"); Rva, _ = G.load_aug("valid", filter_bad=False); rng = np.random.default_rng(31415)
    Rva = Rva[:int(os.environ.get("N_VA", len(Rva)))]
    pool = [Rtr[i] for i in rng.permutation(len(Rtr))[:4 * (PP.N_TR + PP.N_HO)]]; tr_items, ho_items = [], []
    for r in pool:                                                          # probe_plan.main's selection, verbatim
        t = PP.targets(r)
        if t is None or t["op"] is None or t["slot"] is None:
            continue
        (tr_items if len(tr_items) < PP.N_TR else ho_items).append((r, t))
        if len(ho_items) >= PP.N_HO:
            break
    va_items = [(r, PP.targets(r)) for r in Rva]; va_items = [(r, t) for r, t in va_items if t and t["op"] is not None and t["slot"] is not None]
    res = {"n_train": len(tr_items), "n_heldout": len(ho_items), "n_valid": len(va_items), "lams": LAMS}
    for nm in ("op", "slot"):                                               # probe_plan.main's majority baselines (lost in its crash)
        ys = np.array([t[nm] for _, t in tr_items]); vals, cnt = np.unique(ys, return_counts=True); mj = vals[cnt.argmax()]
        res["majority_" + nm] = {"class": int(mj), "train_share": float(cnt.max() / cnt.sum()), "heldout": float(np.mean([t[nm] == mj for _, t in ho_items])),
                                 "valid": float(np.mean([t[nm] == mj for _, t in va_items]))}
    print("majority", json.dumps({k: v for k, v in res.items() if k.startswith("majority")}), flush=True)
    for mk, path in (("raw", None), ("C", os.path.join(PP.ROOT, "boxes", "runs_fsq", "gsm_txt_aug_s1", "model.pt"))):
        g = s5.load_gpt2("cpu")
        if path:
            g.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        g.eval(); TOK.clear(); F_tr = feats2(g, [r for r, _ in tr_items]); T_tr = list(TOK); TOK.clear(); F_ho = feats2(g, [r for r, _ in ho_items]); T_ho = list(TOK)
        TOK.clear(); F_va = feats2(g, [r for r, _ in va_items]); T_va = list(TOK); TOK.clear()
        mk_ = lambda T, its: [(f, t["slot"]) for f, (_, t) in zip(T, its) if f is not None and len(f) >= 2 and t["slot"] < len(f)]  # noqa: E731
        S_tr, S_ho, S_va = mk_(T_tr, tr_items), mk_(T_ho, ho_items), mk_(T_va, va_items)
        res.setdefault(mk, {})
        for wi, wn in ((0, "tok_top"), (1, "tok_h6")):
            res[mk]["slot_" + wn] = tok_probe(S_tr, S_ho, S_va, wi); print(mk, wn, json.dumps(res[mk]["slot_" + wn]), round(time.time() - T0), flush=True)
        print(mk, "feats", round(time.time() - T0), flush=True)
        for nm, ncls in (("op", 4), ("slot", 5)):
            res[mk]["probe_" + nm] = probe2(F_tr, np.array([t[nm] for _, t in tr_items]), F_ho, np.array([t[nm] for _, t in ho_items]),
                                            F_va, np.array([t[nm] for _, t in va_items]), ncls)
            print(mk, nm, json.dumps(res[mk]["probe_" + nm]), round(time.time() - T0), flush=True)
        if mk == "C":                                                       # C's in-context first CoT line (KV-cached; cut at the first newline)
            from boxes.prosqa168s2 import generate_cot_batched
            Dva = [G.encode(r, ENC, PP.EOS) for r in Rva]
            for tag, pre_ in (("noeot", lambda it: it["pre"]), ("eot", lambda it: [EOT] + it["pre"])):
                pres = [pre_(it) for it in Dva]; outs = generate_cot_batched(g, pres, 24, "cpu", None, 50)
                lines = [ENC.decode(o[len(p_):]).split(chr(10))[0].strip() for o, p_ in zip(outs, pres)]
                ok = [bool(r["eqs"]) and ln == "<<%s>>" % r["eqs"][0] for ln, r in zip(lines, Rva)]
                okv = []
                for ln in lines:
                    m_ = PP.re.match(r"<<([^=]+)=([^>]+)>>", ln)
                    okv.append(bool(m_) and G.check_eq(m_.group(1).strip() + "=" + m_.group(2).strip()) in ("exact", "round"))
                res[mk]["e1_exact_" + tag] = float(np.mean(ok)); res[mk]["e1_value_selfconsistent_" + tag] = float(np.mean(okv))
                print(mk, "e1", tag, res[mk]["e1_exact_" + tag], res[mk]["e1_value_selfconsistent_" + tag], round(time.time() - T0), flush=True)
        del g
    res["sec"] = round(time.time() - T0); json.dump(res, open(os.path.join(HERE, "probe_nl.json"), "w", newline="\n"), indent=1); print("DONE", flush=True)


if __name__ == "__main__":
    main()
