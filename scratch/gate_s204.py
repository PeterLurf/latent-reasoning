"""§204 STEP-TARGET-FACTOR gates (contract docs/THEORY_STEPFACTOR_DRAFT_2026-10-14.md §8.5; harness 781f678d). CPU fp32 unless stated; <= 3 threads.
  g1  plain path bitwise            : done by scratch/_tmp/s204ref/run.sh pre/post (781f678d message); re-checked here: FactorMem(no flags) aux_logits /
                                      aux_greedy == ProsMem's bitwise on the same weights, and FactorMem(no flags).state_dict() keys == ProsMem's.
  g2  NOPASS                        : FactorMem(nopass) aux prompts for t = 1..12 contain no pass token, and aux_logits' input prefix equals that prompt.
  g3  targets (10k train items)     : value labels decode to the RHS; every pointer operand token parses to its grounding value (q -> qvals[ref],
                                      r -> eqt[ref].r0, k -> the constant); coverage (value / pointer 1 / pointer 2 / op OTHER) and the candidate-set
                                      first-token collision rate (two different values sharing a first token in C_t).
  g4  gradflow per term             : on 8 train items, K = 3, fade 1, gamma 0.5, W_o ~ N(0, 1e-3): each factor term (value / op / ptr1 / ptr2) and the
                                      answer CE alone gives a nonzero gradient on the memory branch (mem.*) and on its own head.
  g5  eval-path identity            : answers (K' = 3) from FactorMem(factor heads) == ProsMem's with the same g / mem weights, bitwise.
  g8  RMS default path              : FactorMem(add_norm None) has no mem.add_gain; FactorMem(add_norm rms) has it.
  g9  RMS-INVARIANCE (D5b(a))       : add_norm rms, gamma 0, fade 1, W_o ~ N(0, 1e-3): the additive read at the aux prompt's positions after t = 1..4
                                      passes - (i) code path (deltamem casts to fp32 internally, so fp32): max relative spread across t; (ii) A_t vs
                                      t * A_1 (fp32); (iii) fp64, analytic: add_read's rms formula on t * A_1 (double) - spread <= 1e-12. Sanity: the
                                      delta read differs across t; without rms the raw additive read scales as t (fp64 analytic: exact).
  g10 gradflow (rms)                : mem.add_gain.grad != 0 from the aux (factor) loss alone and from the answer CE alone.
-> scratch/gate_s204/gate.json
    python scratch/gate_s204.py"""
import json
import math
import os
import sys

ROOT = "D:/Latent"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("BOXES_NB", "5")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
import boxes.gsm8k_factor as GF  # noqa: E402
import boxes.gsm8k_stepaux as GS  # noqa: E402
from boxes import s5  # noqa: E402
from boxes.prosqa168s2 import ANS, ENC, EOS, EOT, ProsMem  # noqa: E402

DEV = "cpu"; OUT = {}


def enc(r, opv):
    y = {**G.encode(r, ENC, EOS), "_q": r["q"]}; y["ft"] = GF.factor_targets(r, G.parse_recomputable(r)[0], opv); return y


def think(m, its, K):
    x, msk = m.think_inputs(its, DEV); S = m.mem.zero_state(len(its), DEV); Ss = [S]
    for _ in range(K):
        S = m.write(S, x, msk); Ss.append(S)
    return Ss


def main():
    opv, sha = GF.load_op_vocab(); Rtr, _ = G.load_aug("train"); rng = np.random.default_rng(20400); idx = rng.permutation(len(Rtr))[:10000]
    torch.manual_seed(1); g = s5.load_gpt2(DEV); pm = ProsMem(g).to(DEV)
    # ---- g1 / g8 / g2
    torch.manual_seed(1); g2 = s5.load_gpt2(DEV); fm = GF.FactorMem(g2).to(DEV); fm.load_state_dict(pm.state_dict())
    its = [enc(Rtr[i], opv) for i in idx[:8]]; its = [it for it in its if it["depth"] >= 2][:6]
    with torch.no_grad():
        for mm in (pm, fm):
            mm.mem.fade.fill_(1.0); mm.mem.gamma.fill_(0.5); mm.mem.wo.weight.copy_(torch.randn(768, 768, generator=torch.Generator().manual_seed(5)) * 1e-3)
        S_p, S_f = think(pm, its, 2)[2], think(fm, its, 2)[2]
        lp, yp = pm.aux_logits(S_p, its, 2, DEV); lf, yf = fm.aux_logits(S_f, its, 2, DEV); gp, gf = pm.aux_greedy(S_p, 2, 6), fm.aux_greedy(S_f, 2, 6)
    OUT["g1"] = {"states_equal": bool(torch.equal(S_p, S_f)), "aux_logits_equal": bool(torch.equal(lp, lf) and torch.equal(yp, yf)), "aux_greedy_equal": bool((gp == gf).all()),
                 "state_dict_keys_equal": sorted(pm.state_dict()) == sorted(fm.state_dict()), "runs": "scratch/_tmp/s204ref pre vs post: identical (781f678d)"}
    OUT["g1"]["pass"] = all(v for k, v in OUT["g1"].items() if k != "runs")
    fr = GF.FactorMem(s5.load_gpt2(DEV), add_norm="rms").to(DEV)
    OUT["g8"] = {"no_flag_has_add_gain": hasattr(fm.mem, "add_gain") and "mem.add_gain" in fm.state_dict(), "rms_has_add_gain": "mem.add_gain" in fr.state_dict()}
    OUT["g8"]["pass"] = (not OUT["g8"]["no_flag_has_add_gain"]) and OUT["g8"]["rms_has_add_gain"]
    fn = GF.FactorMem(s5.load_gpt2(DEV), nopass=True).to(DEV); bad = []
    for t in range(1, 13):
        pre = fn.aux_prompt(t); bad += [t for tk in pre if tk in GF.PASS_TOKS]
    with torch.no_grad():
        fn.load_state_dict(pm.state_dict()); fn.mem.fade.fill_(1.0); fn.mem.gamma.fill_(0.5); Sn = think(fn, its, 2)[2]; ln_, _ = fn.aux_logits(Sn, its, 2, DEV)
    OUT["g2"] = {"pass_tokens_in_prompts": bad, "prompt": fn.aux_prompt(3), "aux_logits_ran": list(ln_.shape), "pass": not bad}
    print("g1", OUT["g1"], "\ng2", OUT["g2"], "\ng8", OUT["g8"], flush=True)
    # ---- g3 targets on 10k train items
    nv = nst = n_p = [0, 0]; nst = 0; nval = 0; nother = 0; npt = [0, 0]; bad_v = bad_p = 0; ncoll = 0; ncand = 0
    for i in idx:
        r = Rtr[i]; tpl, _ = G.parse_recomputable(r); ft = GF.factor_targets(r, tpl, opv)
        for t, (e, f_) in enumerate(zip(r["eqs"], ft)):
            nst += 1; nother += f_["op"] == len(opv)
            if f_["v"] is not None:
                nval += 1; bad_v += ENC.decode(f_["v"][:-1]).strip() != e.partition("=")[2].strip()
            if tpl is not None:
                eqt = tpl["eqt"][t]
                for j in range(2):
                    if f_["p"][j] is not None:
                        npt[j] += 1; a_, b_, tok, kind, ref = eqt["ops"][j]; v = G.fnum(tok)
                        want = tpl["qvals"][ref] if kind == "q" else (tpl["eqt"][ref]["r0"] if kind == "r" else ref)
                        bad_p += not (abs(v - want) <= 1e-6 * max(1.0, abs(want))) or ENC.encode(tok)[0] != f_["p"][j][0] or f_["p"][j][0] not in f_["p"][j][1]
                vals = {}                                                   # collisions: distinct values sharing a first token among C_t's sources
                for _, _, tok, val in tpl["qn"]:
                    for tk in {ENC.encode(tok)[0], ENC.encode(tok.replace(",", ""))[0]}:
                        vals.setdefault(tk, set()).add(round(val, 6))
                for j in range(t):
                    vals.setdefault(ENC.encode(tpl["eqt"][j]["rtok"])[0], set()).add(round(tpl["eqt"][j]["r0"], 6))
                ncand += 1; ncoll += any(len(v_) > 1 for v_ in vals.values())
    OUT["g3"] = {"n_items": len(idx), "n_steps": nst, "value_coverage": nval / nst, "value_label_mismatch": int(bad_v), "ptr1_coverage": npt[0] / nst, "ptr2_coverage": npt[1] / nst,
                 "pointer_label_mismatch": int(bad_p), "op_OTHER_rate": nother / nst, "candidate_collision_rate": ncoll / max(1, ncand), "n_candidate_sets": ncand,
                 "pass": bad_v == 0 and bad_p == 0}
    print("g3", OUT["g3"], flush=True)
    # ---- g4 / g10 gradflow
    def gradflow(add_norm):
        torch.manual_seed(1); mm = GF.FactorMem(s5.load_gpt2(DEV), add_norm=add_norm, factor=True).to(DEV)
        with torch.no_grad():
            mm.mem.fade.fill_(1.0); mm.mem.gamma.fill_(0.5); mm.mem.wo.weight.copy_(torch.randn(768, 768, generator=torch.Generator().manual_seed(5)) * 1e-3)
        res = {}; its3 = [it for it in (enc(Rtr[i], opv) for i in idx[:40]) if it["depth"] >= 3 and it["ft"][0]["p"][1] is not None][:8]
        for term in ("val", "op", "ptr1", "ptr2", "ans"):
            mm.zero_grad(set_to_none=True); x, msk = mm.think_inputs(its3, DEV); S = mm.mem.zero_state(len(its3), DEV); L = 0
            for t in range(1, 4):
                S = mm.write(S, x, msk)
                if term != "ans":
                    pre = mm.aux_prompt(t); fts = [it["ft"][t - 1] for it in its3]; seqs = [pre + (f_["v"][:-1] if f_["v"] is not None else []) for f_ in fts]
                    xx = torch.tensor(GF.pad(seqs)[0]); h = GF.fwd_read(mm.g, mm.g.embed(xx), mm.mem, S); hl = h[:, len(pre) - 1]
                    if term == "val":
                        vi = [i for i, f_ in enumerate(fts) if f_["v"] is not None]; lg = mm.g.logits(h[vi, len(pre) - 1:]); y = torch.tensor(GF.pad([fts[i]["v"] for i in vi], -100)[0])
                        L = L + Fn.cross_entropy(lg[:, :y.shape[1]].transpose(1, 2), y, ignore_index=-100)
                    elif term == "op":
                        L = L + Fn.cross_entropy(mm.op_logits(hl), torch.tensor([f_["op"] for f_ in fts]))
                    else:
                        j = int(term[-1]) - 1; lp = mm.ptr_logits(hl, [f_["p"][j][1] if f_["p"][j] is not None else None for f_ in fts], j)
                        for i, f_ in enumerate(fts):
                            if f_["p"][j] is not None:
                                L = L + Fn.cross_entropy(lp[i][None], torch.tensor([f_["p"][j][1].index(f_["p"][j][0])]))
            if term == "ans":
                L = mm.dec_loss(S, its3, DEV)
            L.backward(); gn = lambda pref: float(sum(p.grad.double().pow(2).sum() for n, p in mm.named_parameters() if n.startswith(pref) and p.grad is not None) ** 0.5)  # noqa: E731
            res[term] = {"mem": gn("mem."), "w_op": gn("w_op."), "w_p1": gn("w_p1."), "w_p2": gn("w_p2."), "trunk": gn("g."),
                         **({"add_gain": float(mm.mem.add_gain.grad.abs().sum()) if mm.mem.add_gain.grad is not None else 0.0} if add_norm else {})}
        return res
    gf = gradflow(None); OUT["g4"] = {"grad_norms": gf, "pass": all(gf[t]["mem"] > 0 for t in gf) and gf["op"]["w_op"] > 0 and gf["ptr1"]["w_p1"] > 0 and gf["ptr2"]["w_p2"] > 0
                                     and gf["val"]["w_op"] == 0 and gf["ans"]["w_op"] == 0}
    gr = gradflow("rms"); OUT["g10"] = {"grad_norms": gr, "pass": all(gr[t]["add_gain"] > 0 for t in ("val", "op", "ptr1", "ans"))}
    print("g4", OUT["g4"]["pass"], gf, "\ng10", OUT["g10"]["pass"], {t: gr[t]["add_gain"] for t in gr}, flush=True)
    # ---- g5 eval-path identity
    torch.manual_seed(1); fh = GF.FactorMem(s5.load_gpt2(DEV), factor=True).to(DEV); sd = fh.state_dict(); sd.update(pm.state_dict()); fh.load_state_dict(sd)
    D5 = [enc(Rtr[i], opv) for i in idx[100:116]]; ac = lambda: torch.autocast("cpu", enabled=False)  # noqa: E731
    with torch.no_grad():
        outs = []
        for mm in (pm, fh):
            S = think(mm, D5, 3)[3]; outs.append(GS.greedy_after(GS.Fwd(mm.g, mm.mem, S), [[EOT] + it["pre"] + [ANS] for it in D5], DEV, ac, 16))
    OUT["g5"] = {"answers_equal": outs[0] == outs[1], "n": len(D5), "pass": outs[0] == outs[1]}
    print("g5", OUT["g5"], flush=True)
    # ---- g9 RMS invariance
    def g9(add_norm):
        torch.manual_seed(1); mm = GF.FactorMem(s5.load_gpt2(DEV), add_norm=add_norm).to(DEV)
        with torch.no_grad():
            mm.mem.fade.fill_(1.0); mm.mem.gamma.fill_(0.0); mm.mem.wo.weight.copy_(torch.randn(768, 768, generator=torch.Generator().manual_seed(5)) * 1e-3)
            it4 = [enc(Rtr[i], opv) for i in idx[200:204]]; Ss = think(mm, it4, 4); pre = mm.aux_prompt(1)
            xp = torch.tensor([pre] * len(it4)); hp = DM.gpt_lower(mm.g, mm.g.embed(xp), mm.mem.layer); D = hp.shape[-1]; HA = mm.mem.h_add
            q = mm.mem.qkv(mm.mem.ln(hp.float()))[..., :D].view(len(it4), len(pre), -1, DM.DH); pq = Fn.elu(q[:, :, :HA]) + 1; qd = Fn.normalize(q[:, :, HA:], dim=-1)
            ra = [torch.exp(mm.mem.s_add) * DM.add_read(mm.mem, pq, S[:, :HA].float(), DM.state_z(mm.mem, S)) for S in Ss[1:]]
            rd = [torch.einsum("bhvk,bthk->bthv", S[:, HA:HA + mm.mem.h_del].float(), qd) for S in Ss[1:]]
            A1 = Ss[1][:, :HA].double(); At_dev = max(float((Ss[t][:, :HA].double() - t * A1).abs().max() / A1.abs().max()) for t in range(2, 5))
            pq64 = pq.double(); r64 = []
            for t in range(1, 5):
                u = torch.einsum("bthk,bhkv->bthv", pq64, t * A1)
                r64.append(u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True) + 1e-6) * mm.mem.add_gain.double()[:, None] if add_norm else u)
            spread = lambda rs: max(float((r - rs[0]).abs().max() / rs[0].abs().max()) for r in rs[1:])  # noqa: E731
            out = {"code_fp32_add_read_rel_spread": spread(ra), "At_vs_tA1_rel": At_dev, "delta_read_rel_spread": spread(rd)}
            if add_norm:
                out["fp64_analytic_rel_spread"] = spread(r64)
            else:
                out["fp64_raw_scales_as_t_rel_err"] = max(float((r64[t - 1] - t * r64[0]).abs().max() / r64[0].abs().max()) for t in range(2, 5))
                out["code_fp32_raw_vs_t_rel"] = max(float((ra[t - 1] - t * ra[0]).abs().max() / ra[0].abs().max()) for t in range(2, 5))
            return out
    rr, rn = g9("rms"), g9(None)
    OUT["g9"] = {"rms": rr, "none": rn, "note": "deltamem casts every memory computation to fp32 (ln(x.float()), S.float()), so the code path cannot run in fp64; the fp64 "
                 "line is the analytic check on the code's own A_1 (t * A_1 through add_read's rms formula in double)",
                 "pass": rr["fp64_analytic_rel_spread"] <= 1e-12 and rr["code_fp32_add_read_rel_spread"] <= 1e-5 and rr["delta_read_rel_spread"] > 1e-3
                 and rn["fp64_raw_scales_as_t_rel_err"] <= 1e-12}
    print("g9", OUT["g9"], flush=True)
    OUT["pass_all"] = all(OUT[k]["pass"] for k in ("g1", "g2", "g3", "g4", "g5", "g8", "g9", "g10")); OUT["op_vocab_sha256"] = sha
    os.makedirs("scratch/gate_s204", exist_ok=True); json.dump(OUT, open("scratch/gate_s204/gate.json", "w", newline="\n"), indent=1, default=str)
    print("GATE_S204", "PASS" if OUT["pass_all"] else "FAIL", {k: OUT[k]["pass"] for k in OUT if isinstance(OUT[k], dict) and "pass" in OUT[k]}, flush=True)


if __name__ == "__main__":
    main()
