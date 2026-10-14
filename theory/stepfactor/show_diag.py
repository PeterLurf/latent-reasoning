"""Print the key diag_A.json numbers for the draft's §1.4 table."""
import json
import os

d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_A.json")))
print("fidelity (test500, M_t greedy):", d["fidelity_test500_greedy_exact_by_pass_Mt"]); print("eval.json:", d["eval_json_aux_exact_by_pass"])
v = d["valid"]; P = v["pooled"]
CATS = ["fmt", "eq", "eot", "op", "opd_q", "opd_r", "opd_k", "opd_u", "val"]
for src in ("Mt", "M1", "M0"):
    for tag in ("all", "t>=2"):
        if tag not in P[src]:
            continue
        p = P[src][tag]
        print(src, tag, "exact", p.get("exact", {}).get("all"), "| top1", {c: p["top1"].get(c) for c in CATS if c in p["top1"]})
        print("      ce", {c: p["ce"].get(c) for c in CATS if c in p["ce"]})
        print("      opd full / cand top1", p.get("opd_full_top1"), p.get("opd_cand_top1"), "ncand", p.get("opd_ncand"))
for src in ("Mt", "M1", "M0"):
    print(src, "exact by pass", {t: x["exact"]["all"] for t, x in v["by_pass"][src].items()})
    print(src, "val top1 by pass", {t: x["top1"].get("val") for t, x in v["by_pass"][src].items()})
    print(src, "opd_r top1 by pass", {t: x["top1"].get("opd_r") for t, x in v["by_pass"][src].items()})
    print(src, "opd_q top1 by pass", {t: x["top1"].get("opd_q") for t, x in v["by_pass"][src].items()})
    print(src, "op top1 by pass", {t: x["top1"].get("op") for t, x in v["by_pass"][src].items()})
cs = v["ce_share_by_pass"]["Mt"]
for t in sorted(cs, key=int)[:4]:
    print("CE share Mt pass", t, {c: cs[t].get(c) for c in CATS})
for src in ("Mt", "M0"):                                                    # CE share pooled over passes (mean x n)
    ce = P[src]["all"]["ce"]; S_ = {c: m_ * n_ for c, (m_, n_) in ce.items()}; Z = sum(S_.values())
    grp = {"format": sum(S_.get(c, 0) for c in ("fmt", "eq", "eot")), "operator": S_.get("op", 0), "operand": sum(S_.get(c, 0) for c in ("opd_q", "opd_r", "opd_k", "opd_u")),
           "value": S_.get("val", 0), "mixed": sum(v_ for c, v_ in S_.items() if c.startswith("mixed") or c == "oth")}
    print("pooled CE share", src, {k: round(x / Z, 3) for k, x in grp.items()}, "mean CE per token", round(Z / sum(n_ for _, n_ in ce.values()), 3))
print("n valid", d["valid_n_items"], "recomputable", d["valid_n_recomputable"], "sec", d["sec"])
