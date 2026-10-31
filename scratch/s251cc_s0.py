"""Note 251-12 CACHED-CUE stage S0 ($0, eval only; REVIEW_CC #2) on the 4060 (cuda:0), fp32, eval_bs --bs (one card / bs / dtype for every mode) -> scratch/s251cc_s0/<part>.json
  gate : the G-D of inject_aw_eval.read_mask on rcb_plaw4_s1, REG TEST2 d 1-4 first 200 items each:
         (a) keep_all (every row kept) == no mask BITWISE: S_8 (grab_states) and the decoder logits on S_8, the answers at K' {d - 1, 8}, the WS preds per pass;
         (b) the masked rows are exactly the prompt rows: on one pass-1 batch per d, the hooked W_o contribution is exactly 0 at rows < P and == the unmasked contribution at
             rows >= P (P = len([EOT] + enc(' ' + text)), real rows); the same on the decoder (P = len([EOT] + enc(' ' + text minus its final question)));
         (c) coverage: every read armed (the hook raises otherwise), calls_pass / calls_dec > 0.
  s0   : rcb_plaw4_s1 / s2, REG TEST2 d 1-4 (1,000 each): acc at K' = d - 1 and 8 (d 1: K' 0 and 8), WS per pass t 1-8 (AW.ws_readout), masked and unmasked in the same call.
  has  : descriptive (REVIEW_CC #3), REG TEST2 d 1 (1,000): the single-step argmax over the 11 number tokens at ' <S_1> has' ([EOT] + enc(' ' + text + ' ' + S_1 + ' has'))
         vs at ':' (pass 1's own input) vs v_1, for s233_askname_s1 (the trunk alone) and rcb_plaw4_s1 (pass 1 from S_0: read_fixed on the zero state at mem.layer); also
         split by whether ' <S_1> has' occurs literally in the prompt text (copy risk).
    python -B scratch/s251cc_s0.py gate s0 has [--bs 100 --device cuda:0]"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
from boxes import inject_aw_eval as E  # noqa: E402
from boxes import inject_aw as AW  # noqa: E402
from boxes import inject_ow as IO  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
from boxes.nested170 import NUM_IDS, EOT, pad  # noqa: E402

OUT = ROOT + "/scratch/s251cc_s0"; K = 8; DS = (1, 2, 3, 4)


def sets_for(a):
    return E.tokd(IO.build_sets(mul_s_p=E.law(a)), a, DS)


def run_eval(m, T, dev, bs, n=None):
    """acc at K' {d - 1, 8} and WS per pass, per d."""
    out = {}
    for d in DS:
        its = T[d][:n] if n else T[d]; y = np.array([it["ans"] for it in its]); ks = sorted({d - 1, K})
        A, _ = E.passes(m, its, dev, bs, set(ks)); r_, pv = AW.ws_readout(m, {d: its}, dev, K, bs, per_item=True)
        out[str(d)] = {"n": len(its), "acc_kprime": {str(k): float((A[k] == y).mean()) for k in ks}, "ws": r_[str(d)], "_ans": {str(k): A[k].tolist() for k in ks}, "_ws": pv[str(d)]}
    return out


def gate(dev, bs):
    a, m, g = E.load(ROOT + "/boxes/runs_fsq/rcb_plaw4_s1", dev); T = sets_for(a); n = 200; o = {"run": "rcb_plaw4_s1", "n_per_d": n}
    with torch.no_grad():
        base = run_eval(m, T, dev, bs, n); allits = [it for d in DS for it in T[d][:n]]
        with E.read_mask(m, allits, keep_all=True) as st:
            ka = run_eval(m, T, dev, bs, n); st_ka = dict(st)
        o["a_answers_ws_equal"] = all(base[k]["_ans"] == ka[k]["_ans"] and base[k]["_ws"] == ka[k]["_ws"] for k in base)
        eq = []
        for d in DS:
            ch = T[d][:bs]; S0 = E.grab_states(m, ch, dev, bs, (K,))[K]; L0 = m.dec_logits(S0.to(dev), ch, dev)
            with E.read_mask(m, ch, keep_all=True):
                S1 = E.grab_states(m, ch, dev, bs, (K,))[K]; L1 = m.dec_logits(S1.to(dev), ch, dev)
            eq.append({"d": d, "S8_equal": bool(torch.equal(S0, S1)), "dec_logits_equal": bool(torch.equal(L0, L1))})
        o["a_bitwise"] = eq; o["a_stats"] = st_ka
        rows = []
        for d in DS:
            ch = T[d][:bs]; cap = []
            def cap_hook(mod, inp, out):
                cap.append(out.detach().clone())
            hc = m.mem.wo.register_forward_hook(cap_hook)
            try:
                xs, msk = AW.cue_inputs(m, ch, dev); S = m.mem.zero_state(len(ch), dev); S1_ = m.write(S, xs[0], msk, None)
                L = [len(it["dec"]) for it in ch]; m.dec_logits(S1_, ch, dev); un = list(cap); cap.clear()
                hc.remove(); hc = None
                with E.read_mask(m, ch):
                    hc = m.mem.wo.register_forward_hook(cap_hook)   # registered after the mask hook -> sees the masked output
                    m.write(S, xs[0], msk, None); m.dec_logits(S1_, ch, dev)
                mk = list(cap)
            finally:
                if hc is not None:
                    hc.remove()
            Lt = msk.long().sum(1).tolist(); Pt = [len([EOT] + IO.ENC.encode(" " + it["raw"]["text"])) for it in ch]
            Pd = [len([EOT] + IO.ENC.encode(" " + it["raw"]["text"][:-len(it["raw"]["question"])].rstrip())) for it in ch]
            def chk(u, k_, P, Ls):
                z = all(bool((k_[b, :P[b]] == 0).all()) for b in range(len(P))); s = all(bool(torch.equal(k_[b, P[b]:Ls[b]], u[b, P[b]:Ls[b]])) for b in range(len(P)))
                nz = all(bool((u[b, :P[b]] != 0).any()) for b in range(len(P))); return z, s, nz
            zt, st_, nzt = chk(un[0], mk[0], Pt, Lt); zd, sd, nzd = chk(un[1], mk[1], Pd, L)
            rows.append({"d": d, "think_prompt_rows_zero": zt, "think_cue_rows_unchanged": st_, "think_unmasked_prompt_rows_nonzero": nzt,
                         "dec_prompt_rows_zero": zd, "dec_cue_rows_unchanged": sd, "dec_unmasked_prompt_rows_nonzero": nzd,
                         "n_reads_captured": [len(un), len(mk)], "cue_rows_think_mean": float(np.mean([l_ - p_ for l_, p_ in zip(Lt, Pt)])),
                         "cue_rows_dec_mean": float(np.mean([l_ - p_ for l_, p_ in zip(L, Pd)]))})
        o["b_rows"] = rows
        with E.read_mask(m, T[2][:bs]) as st:
            E.passes(m, T[2][:bs], dev, bs, {1, K}); o["c_stats"] = dict(st)
    o["pass"] = (o["a_answers_ws_equal"] and all(x["S8_equal"] and x["dec_logits_equal"] for x in eq)
                 and all(r[k] for r in rows for k in ("think_prompt_rows_zero", "think_cue_rows_unchanged", "dec_prompt_rows_zero", "dec_cue_rows_unchanged"))
                 and all(r["n_reads_captured"] == [2, 2] for r in rows) and o["c_stats"]["calls_pass"] > 0 and o["c_stats"]["calls_dec"] > 0)
    return o


def s0(dev, bs):
    o = {}
    for run in ("rcb_plaw4_s1", "rcb_plaw4_s2"):
        a, m, g = E.load(ROOT + "/boxes/runs_fsq/" + run, dev); T = sets_for(a); allits = [it for d in DS for it in T[d]]
        with torch.no_grad():
            un = run_eval(m, T, dev, bs)
            with E.read_mask(m, allits) as st:
                mk = run_eval(m, T, dev, bs); stats = dict(st)
        o[run] = {"unmasked": un, "masked": mk, "mask_stats": stats, "bs": bs, "dtype": "fp32", "gpu": torch.cuda.get_device_name(0)}
        del m, g; torch.cuda.empty_cache()
    o["def"] = ("Note 251-12 S0: REG TEST2 d 1-4 x 1,000; acc at K' = d - 1 and 8; WS per pass 1-8; masked = DeltaMem reads kept only at the cue rows (inject_aw_eval.read_mask), "
                "unmasked = the same call, same card / bs / fp32; _ans / _ws = per-item answers / WS preds")
    return o


def name_s1(it):
    q = E.ST.cue_str(it["raw"], 1); mt = re.search(r"does (.+?) have\?$", q); assert mt, q; return mt.group(1)


def has(dev, bs):
    num = torch.tensor(NUM_IDS, device=dev); o = {}
    a, m, g = E.load(ROOT + "/boxes/runs_fsq/rcb_plaw4_s1", dev); T = sets_for(a)[1]
    gt = AW.load_trunk(dev, {"scratch": ""}); gt.load_state_dict(torch.load(ROOT + "/boxes/runs_fsq/s233_askname_s1/model.pt", map_location=dev, weights_only=True)); gt.eval()
    y = np.array([int(it["raw"]["values"][1]) for it in T]); nm = [name_s1(it) for it in T]
    lit = np.array([(" " + n_ + " has") in (" " + it["raw"]["text"]) for n_, it in zip(nm, T)])
    has_in = [[EOT] + IO.ENC.encode(" " + it["raw"]["text"] + " " + n_ + " has") for n_, it in zip(nm, T)]; col_in = [it["think_cues"][0] for it in T]

    def last_logits(gg, seqs, mem=None):
        P_ = []
        for i in range(0, len(seqs), bs):
            ch = seqs[i:i + bs]; ids_, _ = pad(ch); x = torch.tensor(ids_, device=dev); B, Tn = x.shape; D = gg.w("wte.weight").shape[1]; hd = D // gg.H
            h = gg.embed(x) + DM.wpe_slice(gg, Tn)[None]; S = mem.zero_state(B, dev) if mem is not None else None
            for li in range(gg.L):
                if mem is not None and li == mem.layer:
                    h = DM.read_fixed(mem, h, S)
                h = DM._block(gg, li, h, hd)
            pos = torch.tensor([len(q) - 1 for q in ch], device=dev); hf = Fn.layer_norm(h, (D,), gg.w("ln_f.weight"), gg.w("ln_f.bias"))
            P_.append(gg.logits(hf[torch.arange(B, device=dev), pos])[:, num].argmax(-1).cpu().numpy())
        return np.concatenate(P_)
    with torch.no_grad():
        for name, gg, mem in (("s233_askname_s1", gt, None), ("rcb_plaw4_s1", m.g, m.mem)):
            ph = last_logits(gg, has_in, mem); pc = last_logits(gg, col_in, mem)
            if mem is not None:                                             # path check: pass_aw's own ':' readout from S_0 == pc
                own = []
                for i in range(0, len(T), bs):
                    ch = T[i:i + bs]; xs, msk = AW.cue_inputs(m, ch, dev); _, h_, pos_ = m.pass_aw(m.mem.zero_state(len(ch), dev), xs[0], msk, None)
                    own.append(m.colon_logits(h_, pos_)[:, num].argmax(-1).cpu().numpy())
                path = bool((np.concatenate(own) == pc).all())
            else:
                path = None
            o[name] = {"n": len(y), "acc_has": float((ph == y).mean()), "acc_colon": float((pc == y).mean()), "agree": float((ph == pc).mean()),
                       "literal_share": float(lit.mean()), "acc_has_literal": float((ph == y)[lit].mean()) if lit.any() else None,
                       "acc_has_not_literal": float((ph == y)[~lit].mean()) if (~lit).any() else None, "colon_path_check": path,
                       "items": {"sig": [it["sig"] for it in T], "name": nm, "v1": y.tolist(), "pred_has": ph.tolist(), "pred_colon": pc.tolist(), "literal": lit.astype(int).tolist()}}
    o["def"] = ("REVIEW_CC #3 (descriptive): REG TEST2 d 1; argmax over the 11 number tokens at the last token of [EOT] + enc(' ' + text + ' ' + S_1 + ' has') vs at ':' of pass 1's "
                "own input, vs v_1; rcb with read_fixed on S_0 at mem.layer; literal = ' <S_1> has' occurs in the prompt text")
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("parts", nargs="+"); ap.add_argument("--bs", type=int, default=100); ap.add_argument("--device", default="cuda:0"); o_ = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); fns = {"gate": gate, "s0": s0, "has": has}
    for p_ in o_.parts:
        r = fns[p_](o_.device, o_.bs); json.dump(r, open(os.path.join(OUT, p_ + ".json"), "w", newline="\n"), indent=1, default=str)
        print(p_, "pass" if p_ != "gate" else "gate pass", r.get("pass", "written"), flush=True)
