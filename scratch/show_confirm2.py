import json

C = json.load(open("results/followup_confirm.json"))
for name, o in C["recurrent"].items():
    print(f"\n==== {name} (k={o['k']})")
    print(" clean gap", {W: round(v[0], 3) for W, v in o["clean_gap"].items()})
    print(" route shares:", {W: {k: round(v, 2) for k, v in s.items()} for W, s in o["route_shares"].items()})
    print(" recovery ratio (dose 1):", {W: {k: round(v, 3) for k, v in r.items()} for W, r in o["recovery_ratio"].items()})
    for W, dk in o["dose"].items():
        print(f"  dose W{W} full:", {lam: round(v[0][0], 3) for lam, v in dk["full"].items()}, "pattern:", {lam: round(v[0][0], 3) for lam, v in dk["pattern"].items()})
    print(" P3", o["P3"])
    for W, s in o["swap"].items():
        print(f"  swap W{W}: swap col/signed {s['swap'][0]:.3f}/{s['swap'][1]:+.3f} remove {s['remove'][0]:.3f}/{s['remove'][1]:+.3f} insert {s['insert'][0]:.3f}/{s['insert'][1]:+.3f}"
              f" | A_drop swap {s['swap'][2]:.3f} vs remove {s['remove'][2]:.3f}; B_rise swap {s['swap'][3]:.3f} vs insert {s['insert'][2]:.3f} | clean col swap {s['clean_swap_col']:.3f} remove {s['clean_remove_col']:.3f} same {s['same_status_col']:.3f}")
    for key, c in o["continuation"].items():
        print(f"  cont {key}: follow true {c['follow_true'][0]:+.3f} [{c['follow_true'][1]:+.3f},{c['follow_true'][2]:+.3f}] wrong {c['follow_wrong'][0]:+.3f} pattern {c['follow_pattern'][0]:+.3f}"
              f" | C_rec edit {c['edit_C_rec']:+.3f} natural {c['natural_C_rec']:+.3f} C_don edit {c['edit_C_don']:+.3f} | comp {c['competence_Nt']:.2f} n={c['n']}")
    for W, e in o["attention"].items():
        print(f"  attn W{W} b=1: self {e['att_boost_self'][0]:+.3f} others {e['att_boost_other_attached'][0]:+.3f} [{e['att_boost_other_attached'][1]:+.3f},{e['att_boost_other_attached'][2]:+.3f}] "
              f"decoy->att {e['dec_boost_attached'][0]:+.3f} all->att {e['all_boost_attached'][0]:+.3f} decoy->self {e['dec_boost_self_decoy'][0]:+.3f}")
    print(" gain", o["gain"])
    if "controllability" in o:
        print(" controllability", json.dumps(o["controllability"]))
for name, o in C["pause"].items():
    print(f"\n==== {name}", {W: {k: round(v, 2) for k, v in s.items()} for W, s in o["route_shares"].items()},
          {W: round(e["att_boost_other_attached"][0], 3) for W, e in o["attention"].items()})
