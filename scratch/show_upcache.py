import json

for s, run in [("s0", "runs/r3_pc_gated_gpt2"), ("s1", "runs/main_s1"), ("s2", "runs/main_s2")]:
    S = json.load(open(f"{run}/followup/discovery_upcache/summary.json"))["results"]
    ref = json.load(open(f"{run}/followup/fresh/summary.json"))["results"]
    for W, R in S.items():
        u = R["upcache"]
        dl = ref[W]["cache"]
        dshare = lambda c: dl["conds"][c]["targeted"][0] / dl["clean_gap"][0]
        print(s, "W" + W, f"clean {u['clean_gap'][0]:.2f} | upfront shares: " + " ".join(f"{c}:{v:.2f}" for c, v in u["shares"].items()),
              f"| delayed (fresh) TL {dshare('TL'):.2f} Gq {dshare('Gq'):.2f}")
        print("     by target distance:", {d: {k: round(v, 2) for k, v in x.items()} for d, x in u["by_dist"].items()})
