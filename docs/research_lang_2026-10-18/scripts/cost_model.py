"""LANG-VB draft $0: per-item work and attention memory of the slot-subset VB, relative to nested D3-T (measured
0.42 s/step: $1.12 / 48k steps at ~$0.20/h). Token-layer counts: prelude 6 x T_q; core 4 x |S| x K; coda 2 x (|S| + L_dec) x
(#readiness passes, <= 5). Attention entries (saved probs): core sum_t |S| x (T0 + (t-1)|S| + |S|) x 12 heads x 4 layers;
prelude 6 x 12 x T_q^2 / 2. ASSUMED: step time = max(launch floor 0.42 s, 0.42 x work ratio x 0.6) (nested was ~60%
launch-bound at T 34), bs 64, $0.22/h 3090."""
import json
def item(Tq, S, K, Ldec, T0=None, nwin=5):
    T0 = Tq if T0 is None else T0
    tl = 6 * Tq + 4 * S * K + 2 * (S + Ldec) * nwin
    att = 12 * 4 * sum(S * (T0 + (t - 1) * S + S) for t in range(1, K + 1)) + 6 * 12 * Tq * Tq / 2
    return tl, att
ref_tl, ref_att = item(34, 34, 12, 0, T0=34)
rows = {}
for name, (Tq, S, K, Ldec) in {"nested_D3T_ref": (34, 34, 12, 0), "prosqa_sym_D8": (450, 48, 12, 12), "prosqa_sym_D12_self": (600, 64, 16, 12),
                               "prosqa_sym_D16_eval": (760, 82, 20, 12), "prosqa_mention_slots_D8": (450, 100, 12, 12), "gsm_q12": (60, 12, 14, 12), "gsm_q12_p90": (90, 12, 14, 12)}.items():
    tl, att = item(Tq, S, K, Ldec, T0=(34 if name.startswith("nested") else Tq))
    r = tl / ref_tl; s = max(0.42, 0.42 * (0.4 + 0.6 * r))
    rows[name] = {"token_layers": tl, "work_ratio": round(r, 2), "attn_entries_M": round(att / 1e6, 1), "attn_GB_fp32_bs64": round(att * 4 * 64 / 1e9, 1),
                  "est_s_per_step": round(s, 2), "usd_per_24k": round(24000 * s / 3600 * 0.22, 2), "usd_per_120k": round(120000 * s / 3600 * 0.22, 2)}
    print(name, rows[name])
json.dump(rows, open("cost_model.json", "w", newline="\n"), indent=1)
