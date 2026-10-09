"""THEORY §172 PCCoT-PROBE (registration 2026-10-09 01:54 EDT; scorer theory/score_s172.py): §149's battery on whynlp/pccot-gpt2 at release (eval only).
Model: GPT-2 small (prosqa/ckpt/gpt2_pretrained.bin, fp32) + the released adapter (prosqa/ckpt_ext/hf/pccot_gpt2, from huggingface.co/whynlp/pccot-gpt2):
LoRA r 128 / alpha 32 on every c_attn / c_proj / c_fc, merged as W (in, out) += (alpha / r) (B A)^T (fan_in_fan_out); peft's trainable tokens
(50257 bot, 50258 eot, 50259 latent, 50260 pad, 50256 eos) REPLACE those wte rows (peft 0.17.1 index_copy; the LM head is tied, its delta is identical).
use_projection false (prj = identity). The student path re-implemented on boxes.core.GPT2KV from github.com/whyNLP/PCCoT models/modeling_gpt2.py (MIT):
    pass 0     [question, bot, latent x c] -> z_0 = the final hidden (after ln_f) at positions bot .. latent_{c-1} (c vectors); the question KV = [question, bot]
    iteration  run z after the question KV (positions qb .. qb + c - 1) -> h; z <- [z[0], h[0 .. c-2]]      (T times; z[0] = h_bot is never updated)
    answer     [eot, "The answer is:"] after the KV of the LAST latent run (T = 0: pass 0's KV, i.e. the latent placeholder tokens = pause tokens), greedy,
               10 new tokens, stop at eos; exact string match against the bench answer (their metric; numeric match logged beside it).
Items are batched by question length (no padding; the release eval left-pads, so tiny numeric differences are possible - the sanity gate decides).
Test: bench/data/gsm8k_aug/test.txt (1,319; the question as written before "||").
eval.json: acc_release (T = 3, c = 24), acc_by_T {"0".."8"}, acc_latents_zeroed, acc_c0, jacobi_ratio_by_T {"1".."8"}, rho_perturb_by_iter, serial
{t: {donor, recipient, n, latent}}, plus numeric-match twins, by_steps, alignment r2 grid, the donor funnel.
    acc_latents_zeroed: after T = 3 iterations, the final latent run is fed z = 0 (every latent input zeroed; positions and the question KV kept), so the
        answer reads the KV of an input-free latent block (as §149's zero ablation of the fed latents).
    acc_c0: c = 0 (no latent tokens): [question, bot] then [eot, "The answer is:"] (out of distribution, as GSM8K_REPRO's caveat).
    jacobi_ratio_by_T: r_T = ||z_{T+1} - z_T|| / ||z_T - z_{T-1}|| over the latent block (z_T = the block after T iterations), the mean over items, T = 1..8.
    rho_perturb_by_iter: §149's finite perturbation: z_1[1:] (the iterated latents) + eps (0.1 x ||z_1[1:]||, a random direction), iterations 2..9 run on
        both; D_t = ||z'_t - z_t||, the median over items of D_t / D_{t-1} for t = 2..9 (key t). z[0] is not iterated and is left unperturbed.
    serial: per equation step t <= 3, the latent j whose held-out ridge probe best reads step t's signed log value from z_{T-1}[j] (the block fed to the
        final latent run, whose KV the answer reads); §149's donor funnel (valid chain, recipient correct, a later step references step t, unambiguous,
        same template, donor step value differs, substituted answer differs); the donor's z_{T-1}[j] replaces the recipient's in the final latent run.
        donor = share predicting the substituted answer, recipient = share predicting the original answer.
    python scratch/s172_probe.py [device] [n_items (smoke)]   -> boxes/runs_fsq/s172_pccot/eval.json"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch"))
from boxes import core as bc  # noqa: E402
import s149_probe as P149  # noqa: E402

AD = os.path.join(ROOT, "prosqa", "ckpt_ext", "hf", "pccot_gpt2")
BOT, EOTL, LAT, EOS = 50257, 50258, 50259, 50256
C_LAT, T_REL, MAX_NEW = 24, 3, 10


def load_model(dev):
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(AD, "adapter_config.json"))); assert cfg["fan_in_fan_out"] and set(cfg["target_modules"]) == {"c_proj", "c_attn", "c_fc"}
    sc = cfg["lora_alpha"] / cfg["r"]; f = safe_open(os.path.join(AD, "adapter_model.safetensors"), "pt"); g = lambda k: f.get_tensor(k).float()  # noqa: E731
    sd = {k: v.float() for k, v in torch.load(os.path.join(ROOT, "prosqa", "ckpt", "gpt2_pretrained.bin"), map_location="cpu", weights_only=True).items()}
    pre = "base_model.model."; bw = g(pre + "transformer.wte.token_adapter.base_layer.weight"); w = torch.cat([sd["wte.weight"], bw[sd["wte.weight"].shape[0]:]], 0)
    idx = cfg["trainable_token_indices"]["wte"]; dw = g(pre + "transformer.wte.token_adapter.trainable_tokens_delta")
    assert torch.equal(dw, g(pre + "lm_head.token_adapter.trainable_tokens_delta")) and w.shape[0] == 50261
    sd["wte.weight"] = w.index_copy(0, torch.tensor(idx), dw); n = 0
    for i in range(12):
        for mod in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            A, B = g(pre + "transformer.h.%d.%s.lora_A.weight" % (i, mod)), g(pre + "transformer.h.%d.%s.lora_B.weight" % (i, mod))
            k = "h.%d.%s.weight" % (i, mod); assert sd[k].shape == (A.shape[1], B.shape[0]); sd[k] = sd[k] + sc * (B @ A).T; n += 1
    assert n == 48 and sum(1 for k in f.keys() if "lora_" in k) == 96
    return bc.GPT2KV(sd).to(dev).eval()


class PCCoT:
    def __init__(self, G, dev):
        self.G, self.dev = G, dev; self.prompt = bc.ENC.encode("The answer is:")

    @torch.no_grad()
    def run(self, Q, T=T_REL, c=C_LAT, zero_final=False, keep_z=False, perturb=None, override=None, decode=True):
        """Q: list of token lists, ALL the same length. -> (answer strings or None, zs (T + 1, b, c, D) if keep_z).
        perturb: (t, delta (b, c - 1, D)) added to z_t[1:] after iteration t; override: {j: (b, D)} replaces z_{T-1}[j] before the final latent run."""
        G, dev = self.G, self.dev; b = len(Q); q = torch.tensor(Q, device=dev); qb = q.shape[1] + 1
        T = 0 if c == 0 else T                                              # no latents: nothing to iterate
        ids0 = torch.cat([q, torch.full((b, 1), BOT, device=dev), torch.full((b, c), LAT, device=dev)], 1)
        h, kv = G(G.embed(ids0)); qkv = [(k[:, :, :qb], v[:, :, :qb]) for k, v in kv]; z = h[:, qb - 1:qb - 1 + c]; zs = [z] if keep_z else None; past = kv
        for it in range(1, T + 1):
            if it == T and (zero_final or override):
                z = z.clone()
                if zero_final:
                    z.zero_()
                for j, vec in (override or {}).items():
                    z[:, j] = torch.as_tensor(vec, device=dev, dtype=z.dtype)
            hl, past = G(z, past=qkv); z = torch.cat([z[:, :1], hl[:, :-1]], 1)
            if perturb is not None and perturb[0] == it:
                z = torch.cat([z[:, :1], z[:, 1:] + torch.as_tensor(perturb[1], device=dev, dtype=z.dtype)], 1)
            if keep_z:
                zs.append(z)
        if c == 0:
            past = [(k[:, :, :qb], v[:, :, :qb]) for k, v in kv]
        out = None
        if decode:
            x = torch.tensor([[EOTL] + self.prompt] * b, device=dev); h, past = G(G.embed(x), past=past); toks = []
            for _ in range(MAX_NEW):
                nt = G.logits(h[:, -1]).argmax(-1); toks.append(nt); h, past = G(G.embed(nt[:, None]), past=past)
            tk = torch.stack(toks, 1).cpu().numpy(); out = []
            for row in tk:
                r_ = []
                for t_ in row:
                    if t_ == EOS:
                        break
                    r_.append(int(t_))
                out.append(bc.ENC.decode([t_ for t_ in r_ if t_ < EOS]))
        return out, (torch.stack(zs).float().cpu().numpy() if keep_z else None)


def grouped(Q, idx, bs):
    """batches of item indices with equal question length."""
    by = {}
    for i in idx:
        by.setdefault(len(Q[i]), []).append(i)
    for L_ in sorted(by):
        v = by[L_]
        for s in range(0, len(v), bs):
            yield v[s:s + bs]


def answers(M, Q, idx, bs, **kw):
    """-> (strings dict i -> str, zs dict i -> array) over idx; per-item kw arrays (override / perturb deltas) are indexed by position in idx."""
    pos = {i: p for p, i in enumerate(idx)}; strs, zs = {}, {}
    for bt in grouped(Q, idx, bs):
        k2 = dict(kw); sel = [pos[i] for i in bt]
        if "override" in kw and kw["override"]:
            k2["override"] = {j: v[sel] for j, v in kw["override"].items()}
        if "perturb" in kw and kw["perturb"] is not None:
            k2["perturb"] = (kw["perturb"][0], kw["perturb"][1][sel])
        s_, z_ = M.run([Q[i] for i in bt], **k2)
        for p, i in enumerate(bt):
            if s_ is not None:
                strs[i] = s_[p]
            if z_ is not None:
                zs[i] = z_[:, p]
    return strs, zs


def num(s):
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return float("nan")


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"; nlim = int(sys.argv[2]) if len(sys.argv) > 2 else 0; t0 = time.time(); torch.set_num_threads(3)
    try:
        from boxes import memcap
        memcap.apply(dev)
    except ImportError:
        pass
    out_d = os.path.join(ROOT, "boxes", "runs_fsq", "s172_pccot" if not nlim else "s172_pccot_smoke"); os.makedirs(out_d, exist_ok=True)
    lines = open(os.path.join(ROOT, "bench", "data", "gsm8k_aug", "test.txt"), encoding="utf-8").read().splitlines(); assert len(lines) == 1319
    N = nlim or len(lines); qtext = [l.split("||")[0] for l in lines[:N]]; gold_s = [l.split("####")[-1].strip() for l in lines[:N]]; gold = np.array([num(s) for s in gold_s])
    chains = [P149.parse_chain(l.split("||")[1].split("####")[0]) if "||" in l else None for l in lines[:N]]; qn = [P149.qnumbers(q) for q in qtext]
    ns = np.array([len(c) if c else 0 for c in chains]); valid = np.array([c is not None and abs(c[-1][1] - gold[i]) < 1e-4 for i, c in enumerate(chains)])
    Q = [bc.ENC.encode(q) for q in qtext]; M = PCCoT(load_model(dev), dev); bs = 64; ALL = list(range(N)); import hashlib
    res = {"model": "whynlp/pccot-gpt2", "adapter_sha256_16": hashlib.sha256(open(os.path.join(AD, "adapter_model.safetensors"), "rb").read()).hexdigest()[:16],
           "probe_sha256_16": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16], "n": N, "c": C_LAT, "T_release": T_REL, "max_new_tokens": MAX_NEW,
           "device": torch.cuda.get_device_name(dev) if dev.startswith("cuda") else "cpu", "chains": {"valid_final_eq_answer": int(valid.sum())}}

    def score(strs):
        ex = np.array([strs[i].strip() == gold_s[i] for i in ALL]); nm = np.array([abs(num(strs[i]) - gold[i]) < 1e-4 for i in ALL]); return ex, nm
    res["acc_by_T"], res["acc_by_T_numeric"] = {}, {}
    for T in (0, 1, 2, 3, 4, 6, 8):
        s_, zz = answers(M, Q, ALL, bs, T=T, keep_z=(T == T_REL)); ex, nm = score(s_); res["acc_by_T"][str(T)] = float(ex.mean()); res["acc_by_T_numeric"][str(T)] = float(nm.mean())
        if T == T_REL:
            ok = ex; Z = np.stack([zz[i] for i in ALL]); res["acc_release"] = float(ex.mean()); res["acc_release_numeric"] = float(nm.mean())
            res["by_steps"] = P149.by_steps(ex, ns); res["preds_release"] = [s_[i] for i in ALL]
            print("SANITY", res["acc_release"], "(numeric %.4f)" % nm.mean(), "%.0fs" % (time.time() - t0), flush=True)
        print("T", T, res["acc_by_T"][str(T)], "%.0fs" % (time.time() - t0), flush=True)
    res["gate_T3_reproduces_release"] = res["acc_by_T"][str(T_REL)] == res["acc_release"]
    s_, _ = answers(M, Q, ALL, bs, zero_final=True); ex, nm = score(s_); res["acc_latents_zeroed"] = float(ex.mean()); res["acc_latents_zeroed_numeric"] = float(nm.mean())
    s_, _ = answers(M, Q, ALL, bs, c=0); ex, nm = score(s_); res["acc_c0"] = float(ex.mean()); res["acc_c0_numeric"] = float(nm.mean())
    print("ABLATIONS", res["acc_latents_zeroed"], res["acc_c0"], "%.0fs" % (time.time() - t0), flush=True)
    # Jacobi step ratios (z_0 .. z_9) and the finite perturbation
    _, zz = answers(M, Q, ALL, bs, T=9, keep_z=True, decode=False); Zt = np.stack([zz[i] for i in ALL])                     # (N, 10, c, D)
    d = np.linalg.norm((Zt[:, 1:] - Zt[:, :-1]).reshape(N, 9, -1), axis=2)                                                  # d[:, T-1] = ||z_T - z_{T-1}||
    res["jacobi_ratio_by_T"] = {str(T): float(np.mean(d[:, T] / np.maximum(d[:, T - 1], 1e-12))) for T in range(1, 9)}
    res["jacobi_step_norm_by_T"] = {str(T): float(np.median(d[:, T - 1])) for T in range(1, 10)}
    rng = np.random.default_rng(17201); ci = sorted(rng.permutation(N)[:min(300, N)].tolist()); U = rng.standard_normal((len(ci), C_LAT - 1, 768)).astype(np.float32)
    U /= np.linalg.norm(U.reshape(len(ci), -1), axis=1)[:, None, None]; base = Zt[ci]; delta = 0.1 * np.linalg.norm(base[:, 1, 1:].reshape(len(ci), -1), axis=1)[:, None, None] * U
    _, zp = answers(M, Q, ci, bs, T=9, keep_z=True, decode=False, perturb=(1, delta)); Zp = np.stack([zp[i] for i in ci])
    D = np.linalg.norm((Zp - base).reshape(len(ci), 10, -1), axis=2)                                                         # D[:, t] = ||z'_t - z_t||
    res["rho_perturb_by_iter"] = {str(t): float(np.median(D[:, t] / np.maximum(D[:, t - 1], 1e-12))) for t in range(2, 10)}
    res["perturb"] = {"n": len(ci), "eps": "0.1 x ||z_1[1:]||, random direction", "D_median_by_iter": {str(t): float(np.median(D[:, t])) for t in range(1, 10)}}
    print("RHO", res["jacobi_ratio_by_T"], "%.0fs" % (time.time() - t0), flush=True)
    # SERIAL: probe-defined alignment on z_{T-1} (the final run's input block), §149's donor funnel, same template
    Zf = Z[:, T_REL - 1]; grid, best = {}, {}
    for t in (1, 2, 3):
        ii = np.array([i for i in ALL if chains[i] and len(chains[i]) >= t]); y = np.array([math.copysign(math.log1p(abs(chains[i][t - 1][1])), chains[i][t - 1][1]) for i in ii]); hf = len(ii) // 2
        grid[str(t)] = [P149.ridge_r2(Zf[ii[:hf], j], y[:hf], Zf[ii[hf:], j], y[hf:]) for j in range(C_LAT)]; best[str(t)] = int(np.argmax(grid[str(t)]))
    res["alignment"] = {"kind": "probe-defined: ridge R^2 of step t's signed log value from z_{T-1}[j] (held-out half)", "r2_grid": grid, "chosen": best}
    serial = {}; rng = np.random.default_rng(14901)
    for t in (1, 2, 3):
        j = best[str(t)]; f0 = [i for i in ALL if valid[i]]; f1 = [i for i in f0 if ok[i]]; f2 = [i for i in f1 if len(chains[i]) > t]
        f3 = [i for i in f2 if P149.referenced_downstream(chains[i], t)]; rec = [i for i in f3 if not P149.ambiguous_ref(chains[i], t, qn[i])]; pairs = []; no_donor = ex_v = ex_a = 0
        for i in rec:
            tp = P149.template(chains[i]); pool = [k for k in ALL if k != i and valid[k] and len(chains[k]) >= t and P149.template(chains[k]) == tp]; rng.shuffle(pool)
            if not pool:
                no_donor += 1
            for k in pool[:20]:
                vd = chains[k][t - 1][1]
                if abs(vd - chains[i][t - 1][1]) < 1e-9:
                    ex_v += 1; continue
                e = P149.substitute(chains[i], t, vd)
                if e is None or abs(e - gold[i]) < 1e-9:
                    ex_a += 1; continue
                pairs.append((i, k, e)); break
        pairs = pairs[:300]; n = len(pairs); cell = {"n": n, "latent": j, "funnel": {"valid_chain": len(f0), "correct_unpatched": len(f1), "has_step_after_t": len(f2),
                                                                               "step_t_referenced_downstream": len(f3), "dropped_ambiguous_reference": len(f3) - len(rec),
                                                                               "recipients": len(rec), "no_donor_in_pool": no_donor, "donor_excluded_same_step_value": ex_v,
                                                                               "donor_excluded_same_or_invalid_answer": ex_a, "pairs": n}}
        if n:
            ri = [p[0] for p in pairs]; di = np.array([p[1] for p in pairs]); exv = np.array([p[2] for p in pairs])
            s_, _ = answers(M, Q, ri, bs, override={j: Zf[di, j]}); pv = np.array([num(s_[i]) for i in ri])
            cell.update({"donor": float((np.abs(pv - exv) < 1e-4).mean()), "recipient": float((np.abs(pv - gold[ri]) < 1e-4).mean())})
        serial[str(t)] = cell
    res["serial"] = serial; res["sec"] = round(time.time() - t0)
    if dev.startswith("cuda"):
        res["torch_peak_gib"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 3)
    json.dump(res, open(os.path.join(out_d, "eval.json"), "w", newline="\n"), indent=1, default=float)
    print("DONE", json.dumps({k: res[k] for k in ("acc_release", "acc_latents_zeroed", "acc_c0")}), flush=True)


if __name__ == "__main__":
    main()
