"""THEORY §149 SOTA-PROBE (note 149-1): our donor SERIAL, ablation and contraction on the released GPT-2 GSM8K-Aug latent reasoners, at release
(eval only). Models (bench harness: bench/models.py CODI / CoconutGSM recipes):
    coconut         prosqa/ckpt_ext/hf/dw_gsm/coconut_checkpoint_33, K 6 (raw last hidden fed back)        units = thought pairs (2t-1, 2t)
    simcot_coconut  prosqa/ckpt_ext/hf/simcot_coconut/checkpoint_28, K 10                                 units = thought pairs (2t-1, 2t)
    simcot_codi     prosqa/ckpt_ext/hf/simcot_codi, 6 latents via prj                                      unit t = latent index t-1 (0 = prj(h_bot))
    codi            prosqa/ckpt_ext/hf/codi, 6 latents via prj                                             unit t = the latent whose held-out ridge
                                                                                                           probe best reads step t's value
Test: the 1,319 GSM8k-Aug items (question text as the bench harness; equation chains <<expr=val>> from bench/data/gsm8k_aug/test.txt).
eval.json: test_acc, by_steps; ablation {unit / "all": {zero, mean}: {acc, by_steps}}; donor {t: {same_template / all: {n, donor_consistent,
recipient_consistent, chance_overlap, excluded_share, candidates}}} (t = 1..3); contraction {rho_k, dist_med_k, jacobian_top}; alignment.
    python scratch/s149_probe.py <model> [device] [n_items (smoke)]   -> boxes/runs_fsq/s149_<model>/eval.json"""
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "theory")); sys.path.insert(0, os.path.join(ROOT, "prosqa")); sys.path.insert(0, ROOT)
os.environ.setdefault("CODI_SP", os.path.join(ROOT, "prosqa", "ckpt_ext")); os.environ.setdefault("THEORY_THREADS", "3")
import codi_probe as cp  # noqa: E402

MODELS = {"coconut": ("coconut", "prosqa/ckpt_ext/hf/dw_gsm/coconut_checkpoint_33", 6), "simcot_coconut": ("coconut", "prosqa/ckpt_ext/hf/simcot_coconut/checkpoint_28", 10),
          "simcot_codi": ("codi", "simcot_codi", 6), "codi": ("codi", "codi", 6)}
NUM = re.compile(r"\d+\.\d*|\.\d+|\d+")                                  # also ".25" / "25." (a leading-dot operand broke substitution)


# ---------------------------------------------------------------- equation chains
def parse_chain(cot):
    """'<<16-3-4=9>> <<9*2=18>> ...' -> [(expr, value)]; None if any step fails to parse / evaluate to its stated value."""
    out = []
    for m_ in re.finditer(r"<<([^<>=]+)=([^<>]+)>>", cot):
        expr, val = m_.group(1).replace(",", "").strip(), m_.group(2).replace(",", "").strip()
        try:
            v = float(val); e = safe_eval(expr)
        except (ValueError, ZeroDivisionError, SyntaxError, TypeError):
            return None
        if e is None or abs(e - v) > 1e-6 * max(1.0, abs(v)):
            return None
        out.append((expr, v))
    return out or None


def safe_eval(expr):
    if not re.fullmatch(r"[0-9.+\-*/() ]+", expr):
        return None
    return float(eval(expr, {"__builtins__": {}}, {}))


def operands(expr):
    return [(m_.start(), m_.end(), float(m_.group())) for m_ in NUM.finditer(expr)]


def typed(chain):
    """Per step: the expression with each number typed as S<k> (= an earlier step's value, the latest match) or Q (question / constant)."""
    res = []
    for i, (expr, _) in enumerate(chain):
        refs = []; tpl = ""; last = 0
        for a_, b_, v in operands(expr):
            k = next((j for j in range(i - 1, -1, -1) if abs(chain[j][1] - v) < 1e-9), None)
            tpl += expr[last:a_] + ("Q" if k is None else "S%d" % k); last = b_; refs.append(k)
        tpl += expr[last:]; res.append((tpl.replace(" ", ""), refs))
    return res


def template(chain):
    return tuple(t for t, _ in typed(chain))


def substitute(chain, t, v_new):
    """Re-evaluate the chain with step t (1-based) := v_new; later steps' references to earlier steps use the new values. -> final value."""
    ty = typed(chain); vals = []
    for i, (expr, v) in enumerate(chain):
        if i == t - 1:
            vals.append(v_new); continue
        ops = operands(expr); e2 = ""; last = 0
        for (a_, b_, x), k in zip(ops, ty[i][1]):
            e2 += expr[last:a_] + ("(%r)" % vals[k] if k is not None else repr(x)); last = b_
        e2 += expr[last:]
        try:
            vals.append(safe_eval(e2))
        except (ZeroDivisionError, SyntaxError, TypeError, ValueError):
            return None
    return vals[-1]


def referenced_downstream(chain, t):
    return any(t - 1 in [k for k in refs if k is not None] for _, refs in typed(chain)[t:])


def qnumbers(question):
    """Numbers in the question text (commas removed)."""
    return {float(x) for x in NUM.findall(question.replace(",", ""))}


def ambiguous_ref(chain, t, qn):
    """Audit (§149 review): a later step's operand typed S<t-1> whose value also occurs in the question text cannot be attributed to step t
    (it may be the question's number); pairs whose step-t reference is ambiguous are dropped."""
    v = chain[t - 1][1]
    return v in qn and any(t - 1 in [k for k in refs if k is not None] for _, refs in typed(chain)[t:])


def unit_tests():
    c = parse_chain("<<16-3-4=9>> <<9*2=18>>"); assert c == [("16-3-4", 9.0), ("9*2", 18.0)]
    assert template(c) == ("Q-Q-Q", "S0*Q") and referenced_downstream(c, 1) and not referenced_downstream(c, 2)
    assert substitute(c, 1, 5.0) == 10.0 and substitute(c, 2, 7.0) == 7.0
    c = parse_chain("<<80000+50000=130000>> <<80000*1.5=120000>> <<120000+80000=200000>>")
    assert template(c) == ("Q+Q", "Q*Q", "S1+Q"), template(c)                  # 80000 in step 3 is a question number, 120000 = step 2
    assert substitute(c, 2, 100.0) == 80100.0 and not referenced_downstream(c, 1) and referenced_downstream(c, 2)
    c = parse_chain("<<2/2=1>> <<2+1=3>>"); assert template(c) == ("Q/Q", "Q+S0") and substitute(c, 1, 4.0) == 6.0
    c = parse_chain("<<3*4=12>> <<12/2=6>> <<12-6=6>>"); assert template(c) == ("Q*Q", "S0/Q", "S0-S1") and substitute(c, 1, 20.0) == 10.0
    c = parse_chain("<<3*1=3>> <<3+5=8>>"); assert ambiguous_ref(c, 1, {3.0, 5.0}) and not ambiguous_ref(c, 1, {1.0, 5.0})
    assert parse_chain("<<2+2=5>>") is None
    c = parse_chain("<<4*.25=1>> <<1+3=4>>"); assert template(c) == ("Q*Q", "S0+Q") and substitute(c, 1, 2.0) == 5.0
    return "ok"


# ---------------------------------------------------------------- model runners (thought overrides / perturbations / records)
class CodiRun:
    def __init__(self, ckpt, dev):
        self.m, self.prj, self.tok = cp.load(ckpt); self.m, self.prj = self.m.to(dev), self.prj.to(dev); self.dev = dev; self.K = 6

    @torch.no_grad()
    def __call__(self, qs, override=None, perturb=None, decode=True, bs=32, max_new=40, keep_k=None):
        """override: {j: (N, 768)} fed latent j (0-based) replaced; perturb: {j: (N, 768)} added. Returns preds (N,), thoughts (N, K, 768)."""
        m, prj, tok, dev = self.m, self.prj, self.tok, self.dev; V = m.config.vocab_size; bot, eot = V - 2, V - 1; wte = m.transformer.wte
        N = len(qs); preds = np.full(N, np.inf); T = np.zeros((N, self.K, 768), np.float32); K = self.K if keep_k is None else keep_k
        for i in range(0, N, bs):
            sl = slice(i, i + bs); ch = qs[sl]; B = len(ch); tok.padding_side = "left"
            enc = {k: v.to(dev) for k, v in tok(list(ch), return_tensors="pt", padding=True).items()}
            ids = torch.cat([enc["input_ids"], torch.full((B, 1), bot, device=dev)], 1); mask = torch.cat([enc["attention_mask"], torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
            pos = (mask.cumsum(1) - 1).clamp(min=0); out = m.transformer(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
            pkv = out.past_key_values; h = prj(out.last_hidden_state[:, -1:, :]); lat_mask = mask.clone(); nextpos = pos[:, -1:] + 1   # memory: no prompt logits
            for j in range(K):
                if override is not None and j in override:
                    h = torch.tensor(override[j][sl], device=dev, dtype=h.dtype)[:, None]
                if perturb is not None and j in perturb:
                    h = h + torch.tensor(perturb[j][sl], device=dev, dtype=h.dtype)[:, None]
                T[sl, j] = h[:, 0].float().cpu().numpy()
                cur = torch.ones(B, 1, dtype=torch.long, device=dev); am = torch.cat([lat_mask, cur], 1)
                o = m(inputs_embeds=h, attention_mask=am, position_ids=nextpos, past_key_values=pkv, use_cache=True, output_hidden_states=True)
                pkv = o.past_key_values; lat_mask = am; nextpos = nextpos + 1; h = prj(o.hidden_states[-1][:, -1:, :])
            if not decode:
                continue
            x = wte(torch.full((B, 1), eot, device=dev)); am = lat_mask; gen = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=dev)
            for _ in range(max_new):
                am = torch.cat([am, torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
                o = m(inputs_embeds=x, attention_mask=am, position_ids=nextpos, past_key_values=pkv, use_cache=True); pkv = o.past_key_values
                nxt = o.logits[:, -1, :V - 1].argmax(-1)
                for b in range(B):
                    if not done[b]:
                        gen[b].append(int(nxt[b]))
                done |= nxt == tok.eos_token_id
                if done.all():
                    break
                x = wte(nxt[:, None]); nextpos = nextpos + 1
            for b in range(B):
                preds[i + b] = cp.extract(tok.decode([t for t in gen[b] if t != tok.eos_token_id]))
        return preds, T


@torch.no_grad()
def coco_hidden(m, x):
    """gpt2c.GPT2C.forward (extra=None, attn_q=None) without the logits: final hidden after ln_f, op for op (Audit 20:29: memory)."""
    import torch.nn.functional as F_
    sd, B, T, D = m.sd, x.shape[0], x.shape[1], x.shape[2]
    h = x + m.wpe[:T][None]
    mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)[None]
    hd = D // m.H
    for i in range(m.L):
        p = f"h.{i}."
        a = F_.layer_norm(h, (D,), sd[p + "ln_1.weight"], sd[p + "ln_1.bias"])
        qkv = a @ sd[p + "attn.c_attn.weight"] + sd[p + "attn.c_attn.bias"]
        q, k, v = qkv.split(D, -1)
        q, k, v = (t.view(B, T, m.H, hd).transpose(1, 2) for t in (q, k, v))
        pr = ((q @ k.transpose(-1, -2)) / math.sqrt(hd) + mask[:, None]).softmax(-1)
        o = (pr @ v).transpose(1, 2).reshape(B, T, D)
        h = h + o @ sd[p + "attn.c_proj.weight"] + sd[p + "attn.c_proj.bias"]
        a = F_.layer_norm(h, (D,), sd[p + "ln_2.weight"], sd[p + "ln_2.bias"])
        h = h + F_.gelu(a @ sd[p + "mlp.c_fc.weight"] + sd[p + "mlp.c_fc.bias"], approximate="tanh") @ sd[p + "mlp.c_proj.weight"] + sd[p + "mlp.c_proj.bias"]
    return F_.layer_norm(h, (D,), sd["ln_f.weight"], sd["ln_f.bias"])


class CocoRun:
    def __init__(self, ckpt, K, dev):
        import tiktoken
        from gpt2c import GPT2C, load_state
        sd = {k: v for k, v in load_state(os.path.join(ROOT, ckpt), dev).items() if not k.startswith(("expainable_llm.", "embedding."))}
        self.m = GPT2C(sd); self.K = K; self.enc = tiktoken.get_encoding("gpt2"); self.dev = dev

    @torch.no_grad()
    def __call__(self, qs, override=None, perturb=None, decode=True, max_new=128, keep_k=None, bmax=64):
        """Exact batching: only questions of identical token length share a batch (gpt2c uses absolute positions 0..T-1, no padding)."""
        from gpt2c import END, EOS, START
        m, dev = self.m, self.dev; N = len(qs); preds = np.full(N, np.inf); T = np.zeros((N, self.K, 768), np.float32); K = self.K if keep_k is None else keep_k
        toks = [self.enc.encode(q + "\n") + [START] for q in qs]; groups = {}
        for i, t in enumerate(toks):
            groups.setdefault(len(t), []).append(i)
        for _, idx_all in sorted(groups.items()):
            for s_ in range(0, len(idx_all), bmax):
                idx = idx_all[s_:s_ + bmax]; B = len(idx); x = m.embed(torch.tensor([toks[i] for i in idx], device=dev))
                for j in range(K):
                    h = coco_hidden(m, x); th = h[:, -1:, :]
                    if override is not None and j in override:
                        th = torch.tensor(override[j][idx], device=dev, dtype=th.dtype)[:, None]
                    if perturb is not None and j in perturb:
                        th = th + torch.tensor(perturb[j][idx], device=dev, dtype=th.dtype)[:, None]
                    T[idx, j] = th[:, 0].float().cpu().numpy(); x = torch.cat([x, th], 1)
                if not decode:
                    continue
                x = torch.cat([x, m.embed(torch.full((B, 1), END, device=dev))], 1); gen = [[] for _ in range(B)]; done = np.zeros(B, bool)
                for _ in range(max_new):
                    nt = (coco_hidden(m, x)[:, -1] @ m.wte.T).argmax(-1)
                    for b in range(B):
                        if not done[b]:
                            if int(nt[b]) == EOS:
                                done[b] = True
                            else:
                                gen[b].append(int(nt[b]))
                    if done.all():
                        break
                    x = torch.cat([x, m.embed(nt[:, None])], 1)
                for b, i in enumerate(idx):
                    txt = self.enc.decode([t for t in gen[b] if t < 50257]).split("#")[-1].replace(",", "").strip()
                    try:
                        preds[i] = float(re.findall(r"-?\d+\.?\d*", txt)[0]) if re.findall(r"-?\d+\.?\d*", txt) else np.inf
                    except ValueError:
                        preds[i] = np.inf
        return preds, T


# ---------------------------------------------------------------- measurements
def by_steps(ok, ns):
    return {k: (float(ok[msk].mean()) if msk.any() else None) for k, msk in (("2", ns == 2), ("3", ns == 3), ("4+", ns >= 4))}


def ridge_r2(Xf, yf, Xs, ys, lam=1e-2):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6; Z = np.c_[(Xf - mu) / sd, np.ones(len(Xf))]; Zs = np.c_[(Xs - mu) / sd, np.ones(len(Xs))]
    w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ yf); r = ys - Zs @ w
    return float(1 - (r ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-12))


def main():
    name = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"; nlim = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    from boxes import memcap
    memcap.apply(dev)                                                       # Audit 20:29: hard cap (no-op when LATENT_MEM_CAP_GB is unset)
    kind, ckpt, K = MODELS[name]; out_d = os.path.join(ROOT, "boxes", "runs_fsq", "s149_%s" % name); os.makedirs(out_d, exist_ok=True); t0 = time.time()
    import hashlib
    res = {"model": name, "ckpt": ckpt, "K": K, "unit_tests": unit_tests(), "probe_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16], "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu"}
    aug = cp.gsm8k(10_000); lines = open(os.path.join(ROOT, "bench", "data", "gsm8k_aug", "test.txt"), encoding="utf-8").read().splitlines()
    assert len(aug) == len(lines) == 1319
    N = nlim or len(aug); qs = np.array([q for q, _, _ in aug[:N]], dtype=object); gold = np.array([a for _, a, _ in aug[:N]])
    qn = [qnumbers(l.split("||")[0]) for l in lines[:N]]
    chains = [parse_chain(l.split("||")[1]) if "||" in l else None for l in lines[:N]]
    ns = np.array([len(c) if c else (aug[i][2] or 0) for i, c in enumerate(chains)])
    valid = np.array([c is not None and abs(c[-1][1] - gold[i]) < 1e-4 for i, c in enumerate(chains)]); res["chains"] = {"n": int(N), "valid_final_eq_answer": int(valid.sum())}
    R = CodiRun(ckpt, dev) if kind == "codi" else CocoRun(ckpt, K, dev)
    pred, T = R(qs); ok = np.abs(pred - gold) < 1e-4; res.update({"test_acc": float(ok.mean()), "by_steps": by_steps(ok, ns), "n": int(N)})
    print("SANITY", name, res["test_acc"], "%.0fs" % (time.time() - t0), flush=True)
    # units: thought indices per aligned unit
    if kind == "coconut":
        units = {str(u + 1): [2 * u, 2 * u + 1] for u in range(K // 2)}
    else:
        units = {str(u + 1): [u] for u in range(K)}
    if name == "codi":                                                      # probe-defined alignment: latent best reading step t's value
        h = N // 2; grid = {}; best = {}
        for t in (1, 2, 3):
            ii = np.array([i for i in range(N) if chains[i] and len(chains[i]) >= t])
            y = np.array([math.copysign(math.log1p(abs(chains[i][t - 1][1])), chains[i][t - 1][1]) for i in ii]); hf = len(ii) // 2
            grid[str(t)] = [ridge_r2(T[ii[:hf], j], y[:hf], T[ii[hf:], j], y[hf:]) for j in range(K)]; best[str(t)] = int(np.argmax(grid[str(t)]))
        align = {t: [best[t]] for t in best}; res["alignment"] = {"kind": "probe-defined (ridge R^2 of step t's signed log value from each latent)", "r2_grid": grid, "chosen": best}
    elif kind == "codi":
        align = {str(t): [t - 1] for t in (1, 2, 3)}; res["alignment"] = {"kind": "latent t-1 (0 = prj(h_bot))", "chosen": {k: v[0] for k, v in align.items()}}
    else:
        align = {str(t): [2 * t - 2, 2 * t - 1] for t in (1, 2, 3)}; res["alignment"] = {"kind": "thought pair (2t-1, 2t)", "chosen": align}
    # ablation
    mean_T = T.mean(0); abl = {}
    for u, js in list(units.items()) + [("all", list(range(K)))]:
        r_ = {}
        for mode in ("zero", "mean"):
            ov = {j: (np.zeros((N, 768), np.float32) if mode == "zero" else np.tile(mean_T[j], (N, 1))) for j in js}
            p_, _ = R(qs, override=ov); o_ = np.abs(p_ - gold) < 1e-4; r_[mode] = {"acc": float(o_.mean()), "by_steps": by_steps(o_, ns)}
        abl[u] = r_
    res["ablation"] = abl; print("ABLATION done %.0fs" % (time.time() - t0), flush=True)
    # donor SERIAL
    rng = np.random.default_rng(14901); donor = {}
    for t in (1, 2, 3):
        js = align[str(t)]; cell = {}
        for mode in ("same_template", "all"):
            f0 = [i for i in range(N) if valid[i]]; f1 = [i for i in f0 if ok[i]]; f2 = [i for i in f1 if len(chains[i]) > t]
            f3 = [i for i in f2 if referenced_downstream(chains[i], t)]; rec = [i for i in f3 if not ambiguous_ref(chains[i], t, qn[i])]
            cands, pairs = 0, []; ex_same_val = ex_same_ans = no_donor = 0
            for i in rec:
                tp = template(chains[i])
                pool = [j for j in range(N) if j != i and valid[j] and len(chains[j]) >= t and (mode == "all" or template(chains[j]) == tp)]
                rng.shuffle(pool)
                if not pool:
                    no_donor += 1
                for j in pool[:20]:
                    cands += 1; vd = chains[j][t - 1][1]
                    if abs(vd - chains[i][t - 1][1]) < 1e-9:
                        ex_same_val += 1; continue
                    e = substitute(chains[i], t, vd)
                    if e is None or abs(e - gold[i]) < 1e-9:
                        ex_same_ans += 1; continue
                    pairs.append((i, j, e)); break
            pairs = pairs[:300]; n = len(pairs)
            if n:
                ri = np.array([p[0] for p in pairs]); di = np.array([p[1] for p in pairs]); ex = np.array([p[2] for p in pairs])
                ov = {jj: T[di, jj] for jj in js}; p_, _ = R(qs[ri], override=ov)
                cell[mode] = {"n": n, "donor_consistent": float((np.abs(p_ - ex) < 1e-4).mean()), "recipient_consistent": float((np.abs(p_ - gold[ri]) < 1e-4).mean()),
                              "chance_overlap": 0.0, "excluded_share": float(1 - n / max(1, len(rec))), "recipients_eligible": len(rec), "candidates_seen": cands, "thoughts_swapped": js,
                              "funnel": {"valid_chain": len(f0), "correct_unpatched": len(f1), "has_step_after_t": len(f2), "step_t_referenced_downstream": len(f3), "dropped_ambiguous_reference": len(f3) - len(rec),
                                         "recipients": len(rec), "no_donor_in_pool": no_donor, "donor_excluded_same_step_value": ex_same_val, "donor_excluded_same_or_invalid_answer": ex_same_ans, "pairs": n}}
            else:
                cell[mode] = {"n": 0, "recipients_eligible": len(rec), "funnel": {"valid_chain": len(f0), "correct_unpatched": len(f1), "has_step_after_t": len(f2),
                                                                                 "step_t_referenced_downstream": len(f3), "dropped_ambiguous_reference": len(f3) - len(rec),
                                                                                 "recipients": len(rec), "no_donor_in_pool": no_donor, "pairs": 0}}
        donor[str(t)] = cell
    res["donor"] = donor; print("DONOR done %.0fs" % (time.time() - t0), flush=True)
    # contraction: perturb the first thought by 0.1 x its norm, follow the later fed thoughts (median ratios); finite-difference power iteration
    ci = rng.permutation(N)[:min(300, N)]; base = T[ci]; U = rng.standard_normal(base[:, 0].shape).astype(np.float32); U /= np.linalg.norm(U, axis=1, keepdims=True)
    delta = 0.1 * np.linalg.norm(base[:, 0], axis=1, keepdims=True) * U
    _, Tp = R(qs[ci], perturb={0: delta}, decode=False); D = np.linalg.norm(Tp - base, axis=2)          # (n, K)
    rk = np.median(D[:, 1:] / np.maximum(D[:, :-1], 1e-9), axis=0); dk = np.median(D, axis=0)
    jn = ci[:64]; v = rng.standard_normal((len(jn), 768)).astype(np.float32); v /= np.linalg.norm(v, axis=1, keepdims=True); sig = None
    for _ in range(8):                                                      # J = d thought_1 / d thought_0 (fed space), central differences
        fd = 1e-2 * np.median(np.linalg.norm(T[jn, 0], axis=1))
        _, Ta = R(qs[jn], perturb={0: fd * v}, decode=False, keep_k=2); _, Tb = R(qs[jn], perturb={0: -fd * v}, decode=False, keep_k=2)
        Jv = (Ta[:, 1] - Tb[:, 1]) / (2 * fd); sig = np.linalg.norm(Jv, axis=1); v = Jv / np.maximum(sig[:, None], 1e-12)
    res["contraction"] = {"rho_k": rk.round(4).tolist(), "dist_med_k": dk.round(5).tolist(), "jacobian_top_eig": float(np.median(sig)),
                          "jacobian_method": "power iteration on J by central differences: the top |EIGENVALUE| of thought_0 -> thought_1 (fed space), not the spectral norm (Audit)", "n": int(len(ci)),
                          "eps": "0.1 x ||thought_0||, random direction", "space": "fed-back raw hidden" if kind == "coconut" else "after prj (fed latent)"}
    res["sec"] = round(time.time() - t0)
    if dev.startswith("cuda"):
        res["torch_peak_gib"] = {"allocated": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 3), "reserved": round(torch.cuda.max_memory_reserved(dev) / 2 ** 30, 3)}
    json.dump(res, open(os.path.join(out_d, "eval.json"), "w", newline="\n"), indent=1, default=float)
    print("DONE", name, json.dumps({"test_acc": res["test_acc"], "donor_t1": donor.get("1", {}).get("same_template")}), flush=True)


if __name__ == "__main__":
    main()
