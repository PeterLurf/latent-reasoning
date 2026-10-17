"""D4 draft, derivation (iii) + (v) [PAUSE-B: D ~ U{0..floor(P/3)} (scattered-heavy)]: pass-schedule statistics of the training distribution (no model, pure simulation; CPU, seconds).
Schedules: BASE = D3-T (n ~ U{1..8}, K = min(n + U{0..4}, 12)); PAUSE-A / PAUSE-B (D4 spec, see D4_draft.md section 2).
Per pass t = 1..K_i (every pass carries an R2 term): F_t = input state finished (rho_{t-1} = n), M_t = pause marker, r_t = operators alive at the input.
Outputs: I(t; F), I(t; M) in bits; the loss mass of every deadline rule 't >= t* => finished'; coverage of the test diagonals (n 9..12 on schedule:
computing pass t with r = n - t + 1 operators alive) by training computing passes; batch-max K (Kb) and mean passes (cost)."""
import json, math, sys
import numpy as np

def sched(rng, spec):
    """-> list of pass types for one item: 'P' pause, 'R' reduction, 'I' idle (finished input), and n."""
    n = int(rng.integers(1, 9))
    if spec["name"] == "BASE" or rng.random() >= spec["p_aug"]:
        K = min(n + int(rng.integers(0, 5)), 12); seq = ["R"] * n + ["I"] * (K - n); return seq, n, False
    P = int(rng.integers(1, spec["P_max"] + 1)); D = int(rng.integers(0, (P // spec.get("D_div", 1)) + 1)); S = P - D
    mid = ["R"] * n + ["P"] * S; rng.shuffle(mid)
    K = min(n + P + int(rng.integers(0, 5)) + (int(rng.integers(0, spec.get("tail_max", 0) + 1)) if rng.random() < spec.get("p_tail", 0) else 0), spec["KCAP"])
    seq = ["P"] * D + mid; assert len(seq) <= K, (n, P, K)
    return seq + ["I"] * (K - len(seq)), n, True

def mi(joint):
    """joint: dict (a, b) -> mass -> I(a; b) bits."""
    tot = sum(joint.values()); pa = {}; pb = {}
    for (a, b), v in joint.items():
        pa[a] = pa.get(a, 0) + v; pb[b] = pb.get(b, 0) + v
    return sum(v / tot * math.log2((v / tot) / (pa[a] / tot * pb[b] / tot)) for (a, b), v in joint.items() if v > 0)

def run(spec, n_items=200_000, seed=0, bs=64):
    rng = np.random.default_rng(seed); jF = {}; jM = {}; jR = {}; cov = {}; ded = {}; Ks = []; tr_max = 0; per_batch_aug = spec.get("per_batch", False)
    batch_aug = None
    for i in range(n_items):
        if per_batch_aug and i % bs == 0:
            batch_aug = rng.random() < spec.get("p_aug", 0)
        sp = dict(spec)
        if per_batch_aug:
            sp["p_aug"] = 1.0 if batch_aug else 0.0
        seq, n, aug = sched(rng, sp if spec["name"] != "BASE" else spec); Ks.append(len(seq)); rho = 0
        for t, typ in enumerate(seq, 1):
            F = int(rho >= n); M = int(typ == "P"); r = n - rho
            jF[(t, F)] = jF.get((t, F), 0) + 1; jM[(t, M)] = jM.get((t, M), 0) + 1; jR[(t, rho)] = jR.get((t, rho), 0) + 1
            if typ == "R":
                tr_max = max(tr_max, t + r)
            if typ == "R":
                cov[(t, r)] = cov.get((t, r), 0) + 1
            if typ == "R":
                rho += 1
    Ks = np.array(Ks); nb = len(Ks) // bs; Kb = Ks[:nb * bs].reshape(nb, bs).max(1)
    tot_items = n_items
    out = {"I_t_F_bits": mi(jF), "I_t_M_bits": mi(jM) if any(k[1] for k in jM) else 0.0,
           "I_t_rho_bits": mi(jR), "max_t_plus_r_computing": tr_max, "p_unfinished_by_t": {str(t): jF.get((t, 0), 0) / max(1, jF.get((t, 0), 0) + jF.get((t, 1), 0)) for t in range(1, 25)}, "H_t_bits": -sum((v / sum(jF.values())) * math.log2(v / sum(jF.values())) for v in [sum(jF.get((t, f), 0) for f in (0, 1)) for t in range(1, 30)] if v),
           "mean_passes_per_item": float(Ks.mean()), "Kb_mean": float(Kb.mean()), "Kb_p90": float(np.percentile(Kb, 90)), "Kb_max": int(Kb.max())}
    # deadline-rule loss: P(input unfinished | t >= t*) (pass-weighted) and per-item probability that the rule errs at least once
    for ts in range(6, 21):
        un = sum(v for (t, F), v in jF.items() if t >= ts and F == 0); al = sum(v for (t, F), v in jF.items() if t >= ts)
        ded[str(ts)] = {"p_unfinished_given_t_ge": un / al if al else None, "unfinished_passes_per_item": un / tot_items}
    out["deadline_rule_loss"] = ded
    # coverage of test diagonals (expected training computing passes per 1M items at (t, r)); r <= 8 always in training
    dg = {}
    for n in range(9, 13):
        row = {}
        for t in range(1, n + 1):
            r = n - t + 1; row[str(t)] = {"r": r, "per_1M_items": cov.get((t, r), 0) / tot_items * 1e6}
        dg[str(n)] = row
    out["test_diagonal_coverage"] = dg
    return out

SPECS = [{"name": "BASE"},
         {"name": "PAUSE-A", "p_aug": 0.5, "P_max": 8, "KCAP": 20, "per_batch": True},
         {"name": "PAUSE-A-item", "p_aug": 0.5, "P_max": 8, "KCAP": 20, "per_batch": False},
         {"name": "PAUSE-B", "p_aug": 0.5, "P_max": 12, "KCAP": 24, "per_batch": True, "p_tail": 0.25, "tail_max": 8, "D_div": 3}]
res = {s["name"]: run(s) for s in SPECS}
json.dump(res, open(sys.argv[1], "w", newline="\n"), indent=1)
for k, v in res.items():
    print("==", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if not isinstance(vv, dict)})
    print("   deadline loss P(unfinished | t>=t*):", {t: round(x["p_unfinished_given_t_ge"], 4) for t, x in v["deadline_rule_loss"].items() if x["p_unfinished_given_t_ge"] is not None and int(t) in (8, 9, 10, 12, 14, 16)})
    for n, row in v["test_diagonal_coverage"].items():
        print("   n%s diag (t:r per1M):" % n, " ".join("%s:%d/%.0f" % (t, x["r"], x["per_1M_items"]) for t, x in row.items() if int(t) >= 4))
