import json
import sys

split = sys.argv[1] if len(sys.argv) > 1 else "discovery"
for s, run in [("s0", "runs/r3_pc_gated_gpt2"), ("s1", "runs/main_s1"), ("s2", "runs/main_s2")]:
    try:
        S = json.load(open(f"{run}/followup/{split}_amp/summary.json"))["results"]
    except FileNotFoundError:
        continue
    for W, R in S.items():
        a = R["amp"]["lams"]
        print(s, "W" + W, " | ".join(
            f"lam {lam}: own_deep {e['own_deep'][0]:+.3f} [{e['own_deep'][1]:+.3f},{e['own_deep'][2]:+.3f}] others {e['other_attached'][0]:+.3f} "
            f"decoy_self {e['decoy_self'][0]:+.3f} rnd_abs {e['random_abs_attached'][0]:.3f}" for lam, e in a.items()))
