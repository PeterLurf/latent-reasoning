import json
import sys

r = json.load(open(sys.argv[1]))
E = r["effects"]
f = lambda t: f"{t[0]:+.3f}[{t[1]:+.3f},{t[2]:+.3f}]"
print("== signed collateral (remove / insert), full & pattern")
for W in range(1, 9):
    row = []
    for kind in ("full", "pattern"):
        for d in ("remove", "insert"):
            e = E.get(f"W{W}|k1|{kind}|{d}|hub")
            if e and e["collateral_signed"][0] == e["collateral_signed"][0]:
                row.append(f"{kind[:4]}-{d[:3]} {f(e['collateral_signed'])}")
    print(f"W{W}: " + "  ".join(row))
print("== targeted by width (full / pattern remove)")
for W in range(1, 9):
    a, b = E[f"W{W}|k1|full|remove|hub"], E[f"W{W}|k1|pattern|remove|hub"]
    print(f"W{W}: full tgt {f(a['targeted'])} col {a['collateral_attached'][0]:.3f} | pattern tgt {f(b['targeted'])} col {b['collateral_attached'][0]:.3f}")
print("== depth-resolved targeted (full remove, k1): within-branch depth j -> distance j+1")
for W in (2, 4, 8):
    ex = E[f"W{W}|k1|full|remove|hub"]["targeted_exact_bdepth"]
    print(f"W{W}: " + " ".join(f"d{int(j)+1}:{v[0]:+.3f}" for j, v in ex.items()))
print("== step profile (full / pattern remove), targeted/collateral")
for W in (2, 4, 8):
    s = []
    for k in range(5):
        a, b = E[f"W{W}|k{k}|full|remove|hub"], E[f"W{W}|k{k}|pattern|remove|hub"]
        s.append(f"k{k} {a['targeted'][0]:+.3f}/{a['collateral_attached'][0]:.3f} p{b['targeted'][0]:+.3f}/{b['collateral_attached'][0]:.3f}")
    print(f"W{W}: " + " | ".join(s))
print("== depth-resolved targeted at k0 (full)")
for W in (2, 4, 8):
    ex = E[f"W{W}|k0|full|remove|hub"]["targeted_exact_bdepth"]
    print(f"W{W}: " + " ".join(f"d{int(j)+1}:{v[0]:+.3f}" for j, v in ex.items()))
print("== controls")
for k, v in E.items():
    if "control" in k:
        print(k, f(v["mean_abs_dP"]), f"diffnorm {v['diff_norm']:.2f}")
print("== upfront")
for k, v in E.items():
    if "upfront" in k:
        print(k, "tgt", f(v["targeted"]), "col", f(v["collateral"]) if v["collateral"] else None, "base_acc", round(v["base_acc"], 3))
print("== decodability (matched AUC by step; branch AUC by width at k1)")
D = r["decodability"]
print({k: round(v["matched_auc"], 3) for k, v in D.items()})
print({w: round(a, 3) for w, a in D["1"]["branch_auc_by_width"].items()})
print("== downstream margins (edited-branch minus decoy probe logit)")
for key, v in r["downstream"].items():
    print(key, " ".join(f"k{kp}: rec {m['recipient_margin'][0]:+.2f} pat {m['patched_margin'][0]:+.2f}" for kp, m in v.items()))
