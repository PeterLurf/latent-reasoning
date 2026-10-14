"""Descriptive (post hoc, final models only): how far each fine-tuned model's GPT-2 trunk moved from the pretrained weights,
per block, and the memory branch's norms. Converged vs failed seeds (FACTS.md table). CPU, 3 threads, idle priority."""
import json, sys, torch, psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS); torch.set_num_threads(3)
from transformers import GPT2LMHeadModel

RUNS = "D:/Latent/boxes/runs_fsq/"
CONV = ["s182_window_s2", "s184_reweight_s2", "s185_rms_s2", "s185_rms_s1", "s190_rmsnopass_s1", "s187_nopass_s1", "s187_nopass_s2",
        "s184_reweight_s1", "s190_rmsnopass_s2", "s182_window_s1"]
FAIL = ["s182_nofade_s1", "s185_den_s1", "s188_rwwin_s1", "s182_hold_s1", "s185_rms_s3"]
ref = {k: v.detach().float() for k, v in GPT2LMHeadModel.from_pretrained("openai-community/gpt2", local_files_only=True).transformer.state_dict().items()}


def refkey(k):  # g.p.h_6_attn_c_attn_weight -> h.6.attn.c_attn.weight
    s = k[len("g.p."):]
    if s.startswith("h_"):
        _, l, rest = s.split("_", 2)
        part, *tail = rest.split("_")
        name = "_".join(tail)  # e.g. c_attn_weight / 1_weight / c_fc_bias
        sub, leaf = name.rsplit("_", 1)
        if part == "ln":
            return f"h.{l}.ln_{sub}.{leaf}"
        return f"h.{l}.{part}.{sub}.{leaf}"
    a, leaf = s.rsplit("_", 1)
    return f"{a}.{leaf}"


out = {}
for run in CONV + FAIL:
    sd = torch.load(RUNS + run + "/model.pt", map_location="cpu", weights_only=False)
    blk_d = [0.0] * 12; blk_n = [0.0] * 12; tot_d = tot_n = 0.0; other = {}
    for k, v in sd.items():
        if not k.startswith("g.p."):
            continue
        r = ref[refkey(k)]; v = v.float()
        if v.shape != r.shape:
            v = v.t()
        d2 = float((v - r).pow(2).sum()); n2 = float(r.pow(2).sum())
        tot_d += d2; tot_n += n2
        if k.startswith("g.p.h_"):
            l = int(k.split("_")[1]); blk_d[l] += d2; blk_n[l] += n2
        else:
            other[k[4:]] = (d2 / n2) ** 0.5
    c6 = ref["h.6.attn.c_attn.weight"].t(); q = sd["mem.qkv.weight"].float()
    rec = {"trunk_rel": (tot_d / tot_n) ** 0.5, "blk_rel": [round((a / b) ** 0.5, 5) for a, b in zip(blk_d, blk_n)], "other_rel": other,
           "mem_qkv_rel_vs_blk6": float((q - c6).norm() / c6.norm()), "wo_norm": float(sd["mem.wo.weight"].float().norm()),
           "gate_w_norm": float(sd["mem.gate.weight"].float().norm()), "gate_b": [round(float(x), 3) for x in sd["mem.gate.bias"]],
           "ln_w_w_norm": float(sd["mem.ln_w.weight"].float().norm()), "add_gain": [round(float(x), 4) for x in sd["mem.add_gain"].flatten()]
           if "mem.add_gain" in sd else None, "group": "conv" if run in CONV else "fail"}
    out[run] = rec
    print(run, rec["group"], "trunk %.5f" % rec["trunk_rel"], "blk", rec["blk_rel"], "qkv %.4f wo %.2f gate %.2f lnw %.2f" % (
        rec["mem_qkv_rel_vs_blk6"], rec["wo_norm"], rec["gate_w_norm"], rec["ln_w_w_norm"]), flush=True)
    del sd
json.dump(out, open(sys.argv[1], "w"), indent=1)
