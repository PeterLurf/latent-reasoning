"""LB draft check C ($0, CPU, 2 threads): the trunk learning rate by relative-update matching (draft §4 deviation L7).

Rule [DERIVED in LB_draft.md §4]: AdamW's per-element step is ~lr (|m / sqrt(v)| <= ~1), so one step moves a matrix W by ~lr / RMS(W) relative to
its size. §228 trained GPT-2 small at lr 1e-4 (§170's full-FT recipe). Holding the per-step relative update of the hidden matrices fixed across
bases gives lr_base = 1e-4 x median RMS(hidden matrices of the base) / median RMS(hidden matrices of GPT-2 small).
Hidden matrices = attention q / k / v / o and MLP weights of every block (embeddings, norms and biases excluded).
Reads safetensors files only (no code). Usage:
  python -B lb_weight_rms.py --gpt2 <gpt2 model.safetensors> --qwen <Qwen2.5-0.5B model.safetensors> [--out lb_weight_rms.json]
"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
LR_REF = 1e-4


def hidden_rms(path, kind):
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if kind == "gpt2":
                ok = any(s in k for s in ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"))
            else:
                ok = k.endswith(".weight") and any(s in k for s in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                                                       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
            if ok:
                t = f.get_tensor(k).float(); out[k] = float(t.pow(2).mean().sqrt())
    return out


def summary(r):
    v = np.array(list(r.values())); return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()), "p10": float(np.quantile(v, 0.1)),
                                            "p90": float(np.quantile(v, 0.9))}


def by_type(r, kind):
    keys = (("c_attn", "c_proj.weight\x00attn", "c_fc", "mlp.c_proj") if kind == "gpt2" else ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    out = {}
    for k in keys:
        if kind == "gpt2" and k.startswith("c_proj"):
            sel = [v for n, v in r.items() if "attn.c_proj" in n]; k = "attn.c_proj"
        else:
            sel = [v for n, v in r.items() if k in n]
        out[k] = float(np.median(sel)) if sel else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--gpt2", required=True); ap.add_argument("--qwen", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "lb_weight_rms.json")); a = ap.parse_args(argv)
    g = hidden_rms(a.gpt2, "gpt2"); q = hidden_rms(a.qwen, "qwen"); sg, sq = summary(g), summary(q)
    ratio = sq["median"] / sg["median"]
    res = {"gpt2": {"summary": sg, "by_type": by_type(g, "gpt2")}, "qwen25_05b": {"summary": sq, "by_type": by_type(q, "qwen")},
           "rms_ratio_median": ratio, "lr_ref_gpt2": LR_REF, "lr_qwen_matched": LR_REF * ratio,
           "rule": "lr_base = 1e-4 x median RMS(hidden matrices, base) / median RMS(hidden matrices, GPT-2 small)"}
    with open(a.out, "w", newline="\n") as f:
        json.dump(res, f, indent=1); f.write("\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
