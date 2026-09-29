import json

items = [("s0", "runs/r3_pc_gated_gpt2", "discovery_cache2"), ("s1", "runs/main_s1", "discovery_cache2"),
         ("s2", "runs/main_s2", "discovery_cache2"), ("p0 k0", "runs/pause_s0", "discovery_k0"),
         ("p0 k1", "runs/pause_s0", "discovery_k1"), ("p0 k2", "runs/pause_s0", "discovery_k2")]
order = ["T", "L", "TL", "clamp", "Gq", "Gl", "G", "TLGq", "TLGl", "LGq", "L_wrong"]
for name, run, tag in items:
    S = json.load(open(f"{run}/followup/{tag}/summary.json"))
    print(f"== {name} k={S['k']}")
    for W, R in S["results"].items():
        c = R["cache"]
        cl = c["clean_gap"][0]
        row = " ".join(f"{k}:{c['conds'][k]['targeted'][0] / cl:.2f}" for k in order if k in c["conds"])
        print(f"  W{W} clean {cl:.2f} | fraction of clean: {row} | L_wrong col {c['conds'].get('L_wrong', {}).get('collateral_mean', [float('nan')])[0]:.3f}")
        if "attn" in R:
            e = R["attn"]["doses"]["1.0"]
            print(f"      attn b=1: self {e['att_boost_self'][0]:+.3f} others {e['att_boost_other_attached'][0]:+.3f} decoy-boost->att {e['dec_boost_attached'][0]:+.3f} all->att {e['all_boost_attached'][0]:+.3f}")
