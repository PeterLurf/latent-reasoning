import json

for s, run in [("s0", "runs/r3_pc_gated_gpt2"), ("s1", "runs/main_s1"), ("s2", "runs/main_s2")]:
    S = json.load(open(run + "/followup/fresh_cachedepth/summary.json"))["results"]
    ref = json.load(open(run + "/followup/fresh/summary.json"))["results"]
    for W in ("2", "4", "8"):
        bd = S[W]["cache"]["by_depth"]
        same = abs(S[W]["cache"]["conds"]["TL"]["targeted"][0] - ref[W]["cache"]["conds"]["TL"]["targeted"][0]) < 1e-9
        print(s, "W" + W, "same graphs:", same, " | ".join(
            "d%s gap %.2f TL %.2f Gq %.2f" % (d, v["clean_gap"], v["shares"]["TL"], v["shares"]["Gq"])
            for d, v in bd.items() if v["clean_gap"] > 0.05))
