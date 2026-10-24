"""REVIEW_LB check Q ($0, CPU, 2 threads, forward only): base-side facts the LB draft's L3 / L6 / L7 / D5 rely on, Qwen2.5-0.5B vs GPT-2 small.
  A  weight RMS (lb_weight_rms recompute) and the lr rules (relative-update 1.6e-5; muP-width 1e-4 x 768 / 896)
  B  memory-layer geometry at init on 64 TOPO think inputs: residual norm entering the memory layer (Qwen block 12 / GPT-2 block 6),
     key-bias share ||b_k|| / ||W_k a + b_k||, mean pairwise cosine of keys across tokens (delta heads: L2-normalised k; additive: phi(k) = elu + 1),
     and the self-read ||o|| / ||h|| at the last think token (additive + delta reads from the pass's own writes, g_add calibrated as note 147-2)
  C  GQA expansion (L6): repeat-interleave 7x -> number of distinct (k, v) head parameter sets per head type at init
  D  tokenizer: answer strings, aux prompt, and the TOPO answer distribution (chance for the 11-way read)
Usage: python -B rv_lb_qwen_checks.py --qwen <dir with config + model.safetensors + tokenizer> --gpt2 <HF gpt2 dir> [--out rv_lb_qwen_checks.json]
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(2)
import boxes.ow_words as OW  # noqa: E402

QKEYS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight")
GKEYS = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def rms_median(sd, keys):
    v = [float(t.float().pow(2).mean().sqrt()) for k, t in sd.items() if any(k.endswith(s) for s in keys)]
    return float(np.median(v)), len(v)


def pair_cos(x):
    """x (T, H, dh) -> mean off-diagonal cosine over token pairs, per head, then the mean over heads."""
    xn = F.normalize(x, dim=-1).transpose(0, 1); c = xn @ xn.transpose(1, 2); T = x.shape[0]
    off = (c.sum((1, 2)) - c.diagonal(dim1=1, dim2=2).sum(1)) / (T * (T - 1)); return float(off.mean())


def mem_geometry(h, a, q, k, v, bk, n_add, dh=64):
    """h (T, D) residual entering the memory layer, a = its norm-layer output; q, k, v (T, H, dh) in the memory's head split (n_add additive heads
    first). Key stats and the self-read at the last token (alpha = sigmoid 5, beta = 1: the DeltaMem init)."""
    T, H, _ = k.shape; ka, kd = k[:, :n_add], k[:, n_add:]
    pk = F.elu(ka) + 1; Sa = torch.einsum("thk,thv->hkv", pk, v[:, :n_add]); oa = torch.einsum("hk,hkv->hv", F.elu(q[-1, :n_add]) + 1, Sa)
    Sd = torch.zeros(H - n_add, dh, dh); al = torch.sigmoid(torch.tensor(5.0)); knd = F.normalize(kd, dim=-1); vd = v[:, n_add:]
    for t in range(T):
        Skk = torch.einsum("hvk,hk->hv", Sd, knd[t]); Sd = al * (Sd - Skk[..., None] * knd[t][:, None, :]) + vd[t][..., None] * knd[t][:, None, :]
    od = torch.einsum("hvk,hk->hv", Sd, F.normalize(q[-1, n_add:], dim=-1))
    kb_share = float((bk.norm() / k.reshape(T, -1).norm(dim=-1)).median())
    return {"h_norm_med": float(h[1:].norm(dim=-1).median()), "a_norm_med": float(a[1:].norm(dim=-1).median()),
            "key_bias_share_med": kb_share, "cos_k_delta": pair_cos(knd[1:]), "cos_phi_k_add": pair_cos(pk[1:]),
            "read_add": float(oa.norm()), "read_del": float(od.norm()), "h_last": float(h[-1].norm())}


@torch.no_grad()
def run_qwen(qdir, texts):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(qdir); m = AutoModelForCausalLM.from_pretrained(qdir, dtype=torch.float32).eval()
    sd = m.state_dict(); L = m.config.num_hidden_layers; Lm = L // 2; pre = "model.layers.%d." % Lm
    H, KV, dh = m.config.num_attention_heads, m.config.num_key_value_heads, 64; rep = H // KV; eps = m.config.rms_norm_eps
    Wq, bq = sd[pre + "self_attn.q_proj.weight"], sd[pre + "self_attn.q_proj.bias"]; Wk, bk = sd[pre + "self_attn.k_proj.weight"], sd[pre + "self_attn.k_proj.bias"]
    Wv, bv = sd[pre + "self_attn.v_proj.weight"], sd[pre + "self_attn.v_proj.bias"]; nw = sd[pre + "input_layernorm.weight"]
    Wk_e = Wk.view(KV, dh, -1).repeat_interleave(rep, 0).reshape(H * dh, -1); bk_e = bk.view(KV, dh).repeat_interleave(rep, 0).reshape(-1)
    Wv_e = Wv.view(KV, dh, -1).repeat_interleave(rep, 0).reshape(H * dh, -1); bv_e = bv.view(KV, dh).repeat_interleave(rep, 0).reshape(-1)
    Wk_h, Wv_h = Wk_e.view(H, dh, -1), Wv_e.view(H, dh, -1)
    distinct = {nm: len({(Wk_h[i].sum().item(), Wv_h[i].sum().item(), Wk_h[i][0, :4].tolist().__repr__()) for i in rng}) for nm, rng in
                (("additive_heads_0_6", range(0, 7)), ("delta_heads_7_13", range(7, 14)))}
    eos = tok.convert_tokens_to_ids("<|endoftext|>"); rows = []
    for tx in texts:
        ids = torch.tensor([[eos] + tok(" " + tx, add_special_tokens=False)["input_ids"]])
        h = m(input_ids=ids, output_hidden_states=True).hidden_states[Lm][0]
        a = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * nw
        q = (a @ Wq.T + bq).view(-1, H, dh); k = (a @ Wk_e.T + bk_e).view(-1, H, dh); v = (a @ Wv_e.T + bv_e).view(-1, H, dh)
        rows.append(mem_geometry(h, a, q, k, v, bk_e, 7))
    tokd = {s: [tok.convert_ids_to_tokens(i) for i in tok(s, add_special_tokens=False)["input_ids"]] for s in
            (" 10", " 1", " 3", " The answer is: 10", " The answer is: 7", " Step 1 :", " Kim has 10 books.", " Kim has 7 books.")}
    return sd, rows, distinct, tokd


@torch.no_grad()
def run_gpt2(gdir, texts):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(gdir); m = AutoModelForCausalLM.from_pretrained(gdir, dtype=torch.float32).eval(); sd = m.state_dict()
    Lm = 6; pre = "transformer.h.%d." % Lm; W, b = sd[pre + "attn.c_attn.weight"], sd[pre + "attn.c_attn.bias"]; D = 768
    lw, lb = sd[pre + "ln_1.weight"], sd[pre + "ln_1.bias"]; rows = []
    for tx in texts:
        ids = torch.tensor([[50256] + tok(" " + tx)["input_ids"]]); h = m(input_ids=ids, output_hidden_states=True).hidden_states[Lm][0]
        a = F.layer_norm(h, (D,), lw, lb); q, k, v = (a @ W + b).split(D, -1)
        rows.append(mem_geometry(h, a, q.reshape(-1, 12, 64), k.reshape(-1, 12, 64), v.reshape(-1, 12, 64), b[D:2 * D], 6))
    return sd, rows


def agg(rows):
    out = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    g = float(np.median([r["read_del"] / r["read_add"] for r in rows])); out["g_add"] = g     # note 147-2: g_add = median ||read_del|| / ||read_add||
    out["self_read_over_h"] = float(np.median([np.hypot(g * r["read_add"], r["read_del"]) / r["h_last"] for r in rows])); return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--qwen", required=True); ap.add_argument("--gpt2", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rv_lb_qwen_checks.json")); a = ap.parse_args(argv)
    st = OW.Stream(23131, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); texts = [st.draw(1 + (i % 4))["text"] for i in range(64)]
    sq, rq, dkv, tokd = run_qwen(a.qwen, texts); sg, rg = run_gpt2(a.gpt2, texts)
    mq, nq = rms_median({k: v for k, v in sq.items() if "layers." in k}, QKEYS); mg, ng = rms_median({k: v for k, v in sg.items() if ".h." in k}, GKEYS)
    res = {"A_rms": {"qwen_median": mq, "qwen_n": nq, "gpt2_median": mg, "gpt2_n": ng, "ratio": mq / mg, "lr_relupdate": 1e-4 * mq / mg,
                     "lr_mup_width": 1e-4 * 768 / 896, "rel_update_qwen_at_1e-4_over_gpt2": mg / mq},
           "B_geometry": {"qwen_L12": agg(rq), "gpt2_L6": agg(rg)}, "C_gqa_distinct_kv_param_sets": dkv, "D_tokens": tokd}
    st2 = OW.Stream(23141, frames=OW.FRAME_TRAIN, order_mode="topo", dag=True); dist = {}
    for d in (1, 2, 4):
        c = np.bincount([int(st2.draw(d)["answer"]) for _ in range(4000)], minlength=11); dist[str(d)] = {"counts": c.tolist(), "max_share": float(c.max() / c.sum())}
    res["D_answer_dist"] = dist
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
