"""REVIEW_HD: PAR null-calibration and mechanism, from hd_check_cpu.<run>.json (read-only). CPU, no torch.
    python -B review_scripts/rv_par.py [dir_with_jsons ...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(HERE)
dirs = [D, HERE] + sys.argv[1:]
def load(rn):
    for d in dirs:
        p = os.path.join(d, "hd_check_cpu.%s.json" % rn)
        if os.path.exists(p):
            return json.load(open(p))["deep"]
    return None
def items(o):
    return [(int(n), r) for n in sorted(o, key=int) for r in o[n]]
def right(n, r): return r["ans"][n + 2] == r["y"]
def ahead(n, r): return r["t_ans"] is not None and r["t_ans"] <= n - 1
out = {}
def cell_desc(rn):
    o = load(rn)
    if o is None: return None
    R = items(o); d = {}
    d["acc"] = np.mean([right(n, r) for n, r in R]); d["ahead"] = np.mean([ahead(n, r) for n, r in R])
    A = [(n, r) for n, r in R if ahead(n, r)]
    # how is the base 'ahead': state on schedule at n-1 (coda does last op) vs state ahead (dec == E_n at n-1)
    d["ahead_state_exact_nm1"] = np.mean([r["exact"][n - 1] for n, r in A]) if A else None
    d["ahead_state_ahead_nm1"] = np.mean([r["ahead"][n - 1] for n, r in A]) if A else None
    d["exact_n_all"] = np.mean([r["exact"][n] for n, r in R]); d["exact_nm1_all"] = np.mean([r["exact"][n - 1] for n, r in R])
    # first pass where the state is off-schedule (neither exact nor ahead)
    d["by_n"] = {}
    for n in (9, 10, 11):
        Rn = [r for m, r in R if m == n]
        d["by_n"][n] = {"acc": np.mean([right(n, r) for r in Rn]), "ahead": np.mean([ahead(n, r) for r in Rn]),
                        "acc_at_nm1": np.mean([r["ans"][n - 1] == r["y"] for r in Rn]), "acc_at_nm2": np.mean([r["ans"][n - 2] == r["y"] for r in Rn]),
                        "exact_nm1": np.mean([r["exact"][n - 1] for r in Rn]), "exact_n": np.mean([r["exact"][n] for r in Rn]),
                        "height_mean": np.mean([r["height"] for r in Rn])}
    # ahead vs height slack
    hs = {}
    for n, r in R:
        k = n - r["height"]; hs.setdefault(k, []).append(ahead(n, r))
    d["ahead_by_slack"] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(hs.items())}
    return d
def pair(bn, an):
    ob, oa = load(bn), load(an)
    if ob is None or oa is None: return None
    B, A = items(ob), items(oa); assert all(rb["y"] == ra["y"] for (_, rb), (_, ra) in zip(B, A))
    rows = [(n, rb, ra) for (n, rb), (_, ra) in zip(B, A)]
    br = [x for x in rows if right(x[0], x[1])]
    bra = [x for x in br if ahead(x[0], x[1])]; brn = [x for x in br if not ahead(x[0], x[1])]
    p_keep_a = np.mean([right(n, ra) for n, _, ra in bra]); p_keep_n = np.mean([right(n, ra) for n, _, ra in brn])
    lost = [x for x in br if not right(x[0], x[2])]
    par = np.mean([ahead(n, rb) for n, rb, _ in lost]) if lost else None
    null_par = len(bra) / len(br)
    # Fisher-style: hypergeometric two-sided via scipy if present
    try:
        from scipy.stats import fisher_exact
        tab = [[sum(right(n, ra) for n, _, ra in bra), sum(not right(n, ra) for n, _, ra in bra)], [sum(right(n, ra) for n, _, ra in brn), sum(not right(n, ra) for n, _, ra in brn)]]
        pval = fisher_exact(tab)[1]
    except Exception:
        tab, pval = None, None
    # null d_ahead if the arm lost base-right items at random and gained none: d_ahead_null = -(lost share) * null_par ; observed
    both = [x for x in br if right(x[0], x[2])]
    ahead_both_base = np.mean([ahead(n, rb) for n, rb, _ in both]); ahead_both_arm = np.mean([ahead(n, ra) for n, _, ra in both])
    gained = [x for x in rows if not right(x[0], x[1]) and right(x[0], x[2])]
    # lost ahead items: arm state at n-1 / n / n+2
    la = [x for x in lost if ahead(x[0], x[1])]
    st = {"arm_exact_nm1": np.mean([ra["exact"][n - 1] for n, _, ra in la]) if la else None,
          "arm_exact_n": np.mean([ra["exact"][n] for n, _, ra in la]) if la else None,
          "arm_finished": np.mean([ra["finished"] for n, _, ra in la]) if la else None,
          "base_exact_nm1": np.mean([rb["exact"][n - 1] for n, rb, _ in la]) if la else None,
          "base_exact_n": np.mean([rb["exact"][n] for n, rb, _ in la]) if la else None}
    return {"n_base_right": len(br), "P_arm_right|base_ahead": p_keep_a, "P_arm_right|base_right_not_ahead": p_keep_n, "fisher_tab": tab, "fisher_p": pval,
            "PAR_share_obs": par, "PAR_share_null(=P(ahead|base right))": null_par, "n_lost": len(lost), "n_gained": len(gained),
            "ahead_among_both_right: base / arm": (ahead_both_base, ahead_both_arm), "lost_ahead_items_state": st}
cells = ["s213_d3t_s1", "s216_d3t_s2", "s224_rd_s1", "s224_rd_s2", "s224_rd_s3", "s224_fh_s1", "s224_fh_s2", "s224_rdft_s1", "s224_rdft_s2", "s224_ftctrl_s2", "s224_fhft_s2"]
for c in cells:
    d = cell_desc(c)
    if d: out["cell:" + c] = d
for b, a in [("s213_d3t_s1", "s224_rd_s1"), ("s216_d3t_s2", "s224_rd_s2"), ("s213_d3t_s1", "s224_fh_s1"), ("s216_d3t_s2", "s224_fh_s2"),
             ("s213_d3t_s1", "s224_rdft_s1"), ("s216_d3t_s2", "s224_rdft_s2"), ("s216_d3t_s2", "s224_ftctrl_s2"), ("s216_d3t_s2", "s224_fhft_s2"), ("s224_ftctrl_s2", "s224_rdft_s2")]:
    p = pair(b, a)
    if p: out["pair:%s_vs_%s" % (a, b)] = p
def cv(x):
    if isinstance(x, (np.floating, float)): return round(float(x), 4)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, dict): return {str(k): cv(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [cv(v) for v in x]
    return x
out = cv(out)
json.dump(out, open(os.path.join(HERE, "rv_par.out.json"), "w", newline="\n"), indent=1)
print(json.dumps(out, indent=1))
