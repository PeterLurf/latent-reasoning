"""STEP-TARGET-FACTOR diagnostics on §200 A's FINAL model (boxes/runs_fsq/gsm_lat_aug_s1/model.pt), CPU fp32 (A trained / evaluated in bf16 on a
3090: small drift expected). Read-only on the repo; output diag_A.json in this folder. POST-HOC (A has landed and was scored).
  fidelity : greedy aux exact by pass from M_t on eval.json's 500 test extras (rng 16802 perm[:500]) -> compare with eval.json aux_exact_by_pass.
  valid    : Aug valid (500; design decisions use valid, never test). Per pass t <= min(m, 8), the teacher-forced (TF) aux CE / top-1 by token category
             (format, operator, operand q / r / k / unparsed, value) and greedy exact, reading three memories with the SAME prompt [EOT, ' Step', ' t', ' :']:
               M_t (the trained path), M_1 (PLAN-LEAK: is step t's content already in pass 1's memory?), M_0 = 0 (the memory-free floor).
             At the first token of each grounded operand (q / r): full-vocab top-1 vs candidate-restricted top-1 (candidates = the item's question-literal
             first tokens + previous teacher results' first tokens + the target), i.e. identity PRECISION vs operand SELECTION."""
import json
import os
import sys
import time

import numpy as np
import psutil

try:
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
except Exception:
    pass
os.environ["BOXES_NB"] = "5"
sys.path.insert(0, "D:/Latent")
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

torch.set_num_threads(3)
from boxes import s5  # noqa: E402
import boxes.deltamem as DM  # noqa: E402
import boxes.gsm8k_data as G  # noqa: E402
from boxes.prosqa168s2 import ENC, EOS, EOT, ProsMem  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from step_stats import char_cats, tok_cats  # noqa: E402

RUN = "D:/Latent/boxes/runs_fsq/gsm_lat_aug_s1"; DEV = "cpu"; K = 10; TMAX = 8; BS = 25
DM.FAST_ADD = True; DM.CHUNK_DELTA = 64                                   # A's flags (fast_add, chunk_delta 64)


def build():
    g = s5.load_gpt2(DEV); m = ProsMem(g).to(DEV); sd = torch.load(os.path.join(RUN, "model.pt"), map_location=DEV, weights_only=False)
    m.load_state_dict(sd); m.eval(); return m


def enc(r):
    y = G.encode(r, ENC, EOS); tpl, _ = G.parse_recomputable(r); y["tpl"] = tpl; cats = []
    for t, e in enumerate(r["eqs"]):
        s, cc = char_cats(e, tpl["eqt"][t] if tpl is not None else None); ids, tc = tok_cats(s, cc); assert ids == y["step_ids"][t]; cats.append(tc + ["eot"])
    y["cats"] = cats
    if tpl is not None:                                                     # candidate first tokens per pass: question literals + previous results
        ql = set()
        for _, _, tok, _ in tpl["qn"]:
            ql.add(ENC.encode(tok)[0]); ql.add(ENC.encode(tok.replace(",", ""))[0])
        y["cand"] = [sorted(ql | {ENC.encode(tpl["eqt"][j]["rtok"])[0] for j in range(t)}) for t in range(len(r["eqs"]))]
        fo = []                                                             # (token index in step_ids, kind) for each grounded operand's first token
        for t, (eqt, e) in enumerate(zip(tpl["eqt"], r["eqs"])):
            ids = y["step_ids"][t]; offs = np.cumsum([0] + [len(ENC.decode_single_token_bytes(i)) for i in ids]); f_ = []
            for (a, b, tok, kind, ref) in eqt["ops"]:
                ci = 3 + a; j = int(np.searchsorted(offs, ci, side="right") - 1)
                if offs[j] == ci and kind in ("q", "r"):
                    f_.append((j, kind))
            fo.append(f_)
        y["fo"] = fo
    return y


@torch.no_grad()
def states(m, ch):
    x, msk = m.think_inputs(ch, DEV); hl = m.lower(x); S = m.mem.zero_state(len(ch), DEV); out = [S]
    for _ in range(TMAX):
        S = m.write(S, x, msk, hl); out.append(S)
    return out                                                              # out[t] = M_t, t = 0..TMAX


def add(d, k, v):
    d.setdefault(k, []).append(v)


@torch.no_grad()
def run_valid(m, D):
    R = {}
    for i in range(0, len(D), BS):
        ch = D[i:i + BS]; Ss = states(m, ch)
        for t in range(1, TMAX + 1):
            ix = [j for j, it in enumerate(ch) if t <= it["m"]]
            if not ix:
                continue
            its = [ch[j] for j in ix]
            for src, S in (("Mt", Ss[t]), ("M1", Ss[1]), ("M0", Ss[0])):
                Si = S[ix]; lg, y = m.aux_logits(Si, its, t, DEV); lp = lg.log_softmax(-1); top = lg.argmax(-1)
                for r_, it in enumerate(its):
                    cats = it["cats"][t - 1]; L = len(cats)
                    for p_ in range(L):
                        tg = int(y[r_, p_]); add(R, (src, t, "ce", cats[p_]), float(-lp[r_, p_, tg])); add(R, (src, t, "top1", cats[p_]), int(top[r_, p_]) == tg)
                    if it.get("fo") is not None:
                        cand = torch.tensor(it["cand"][t - 1])
                        for (j, kind) in it["fo"][t - 1]:
                            tg = int(y[r_, j]); cs = cand if tg in it["cand"][t - 1] else torch.cat([cand, torch.tensor([tg])])
                            add(R, (src, t, "opd_full_top1", kind), int(top[r_, j]) == tg)
                            add(R, (src, t, "opd_cand_top1", kind), int(cs[lg[r_, j, cs].argmax()]) == tg); add(R, (src, t, "opd_ncand", kind), len(cs))
                gen = m.aux_greedy(Si, t, max(len(it["step_ids"][t - 1]) + 1 for it in its))
                for r_, it in enumerate(its):
                    tg_ = it["step_ids"][t - 1] + [EOT]; add(R, (src, t, "exact", "all"), bool(list(gen[r_, :len(tg_)]) == tg_))
        print("valid", i + len(ch), "/", len(D), round(time.time() - T0), "s", flush=True)
    return R


@torch.no_grad()
def run_fidelity(m, D):
    hop = {}
    for i in range(0, len(D), BS):
        ch = D[i:i + BS]; x, msk = m.think_inputs(ch, DEV); S = m.mem.zero_state(len(ch), DEV)
        for t in range(1, TMAX + 1):
            S = m.write(S, x, msk); ix = [j for j, it in enumerate(ch) if t <= it["depth"]]
            if ix:
                gen = m.aux_greedy(S[ix], t, max(len(ch[j]["step_ids"][t - 1]) + 1 for j in ix))
                for r_, j in enumerate(ix):
                    tg_ = ch[j]["step_ids"][t - 1] + [EOT]; hop.setdefault(t, []).append(bool(list(gen[r_, :len(tg_)]) == tg_))
        print("fid", i + len(ch), round(time.time() - T0), "s", flush=True)
    return {str(t): [float(np.mean(v)), len(v)] for t, v in sorted(hop.items())}


def summarise(R):
    out = {}
    for (src, t, kind, cat), v in sorted(R.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
        out.setdefault(src, {}).setdefault(str(t), {}).setdefault(kind, {})[cat] = [round(float(np.mean(v)), 4), len(v)]
    pooled = {}                                                             # pooled over t (and t >= 2), per source
    for (src, t, kind, cat), v in R.items():
        for tag in (["all"] + (["t>=2"] if t >= 2 else [])):
            pooled.setdefault(src, {}).setdefault(tag, {}).setdefault(kind, {}).setdefault(cat, []).extend(v)
    pooled = {s: {tg: {k: {c: [round(float(np.mean(v)), 4), len(v)] for c, v in d2.items()} for k, d2 in d1.items()} for tg, d1 in dd.items()} for s, dd in pooled.items()}
    ce_share = {}                                                           # share of the TF aux CE (sum over tokens) by category, M_t, per t
    for (src, t, kind, cat), v in R.items():
        if kind == "ce":
            ce_share.setdefault(src, {}).setdefault(str(t), {})[cat] = float(np.sum(v))
    ce_share = {s: {t: {c: round(x / sum(d.values()), 4) for c, x in d.items()} for t, d in dd.items()} for s, dd in ce_share.items()}
    return {"by_pass": out, "pooled": pooled, "ce_share_by_pass": ce_share}


if __name__ == "__main__":
    T0 = time.time(); m = build(); print("built", round(time.time() - T0), flush=True)
    Rte, _ = G.load_aug("test", filter_bad=False); Rva, _ = G.load_aug("valid", filter_bad=False)
    ex_i = np.random.default_rng(16802).permutation(len(Rte))[:500]; Dte = [G.encode(Rte[i], ENC, EOS) for i in ex_i]
    fid = run_fidelity(m, Dte); print("FIDELITY", fid, flush=True)
    Dva = [enc(r) for r in Rva]; R = run_valid(m, Dva)
    res = {"model": RUN + "/model.pt", "dtype": "cpu fp32", "fidelity_test500_greedy_exact_by_pass_Mt": fid,
           "eval_json_aux_exact_by_pass": json.load(open(os.path.join(RUN, "eval.json")))["aux_exact_by_pass"], "valid": summarise(R),
           "valid_n_items": len(Dva), "valid_n_recomputable": int(sum(it["tpl"] is not None for it in Dva)), "sec": round(time.time() - T0)}
    json.dump(res, open(os.path.join(HERE, "diag_A.json"), "w", newline="\n"), indent=1); print("DONE", res["sec"], flush=True)
