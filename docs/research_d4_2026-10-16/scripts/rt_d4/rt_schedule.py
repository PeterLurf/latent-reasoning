"""RT-5: re-check d4_schedule_stats (PAUSE-A per-batch) and probe what it does not report.
(i) trailing pauses: share of augmented items whose shuffled 'mid' ends in a pause (a pause on FINISHED input; contradicts 'no pauses after completion');
    the completion pass vs n + P (R1 window start tau = n + P per draft).
(ii) the test stratum: passes with t >= 9, unfinished input, and NO pause before t (H = 0). D0b says 0; report P(unfinished | t >= 9, H) for H = 0 / 1.
(iii) the marker-gated clock u_t = #unmarked passes before t: max u on unfinished passes (zero-loss threshold).
(iv) a constrained variant (last mid element forced to be a reduction): P(unfinished | t >= t*) and the trailing share = 0.
(v) the shifted-deadline loss: expected unfinished passes per item at t >= t_d (gradient pressure on an ungated deadline at t_d)."""
import json, sys
import numpy as np
def sched(rng, constrained):
    n = int(rng.integers(1, 9)); P = int(rng.integers(1, 9)); D = int(rng.integers(0, P + 1)); S = P - D
    mid = ["R"] * n + ["P"] * S; rng.shuffle(mid)
    if constrained and mid[-1] == "P":
        k = max(i for i, x in enumerate(mid) if x == "R"); mid[k], mid[-1] = mid[-1], mid[k]
    K = min(n + P + int(rng.integers(0, 5)), 20); seq = ["P"] * D + mid
    return seq + ["I"] * (K - len(seq)), n
def plain(rng):
    n = int(rng.integers(1, 9)); K = min(n + int(rng.integers(0, 5)), 12); return ["R"] * n + ["I"] * (K - n), n
res = {}
for constrained in (False, True):
    rng = np.random.default_rng(0); trailing = 0; aug = 0; comp_gap = []; cnt = {}; strat = {0: [0, 0], 1: [0, 0]}; umax = 0; ded = {}
    for b in range(200000 // 64):
        is_aug = rng.random() < 0.5
        for _ in range(64):
            if is_aug:
                seq, n = sched(rng, constrained); aug += 1
                mids = [x for x in seq if x != "I"]; trailing += mids[-1] == "P"
                last_R = max(i for i, x in enumerate(seq) if x == "R") + 1; comp_gap.append((sum(1 for x in seq if x != "I")) - last_R)
            else:
                seq, n = plain(rng)
            rho = 0; H = 0; u = 0
            for t, typ in enumerate(seq, 1):
                F = rho >= n
                c = cnt.setdefault(t, [0, 0]); c[0] += 1; c[1] += (not F)
                if t >= 9:
                    strat[H][0] += 1; strat[H][1] += (not F)
                if not F:
                    umax = max(umax, u)
                if typ == "P":
                    H = 1
                else:
                    u += 1
                if typ == "R":
                    rho += 1
    tot_items = 200000 // 64 * 64
    p_unf = {}
    for ts in (9, 12, 14, 16, 17):
        a = sum(v[0] for t, v in cnt.items() if t >= ts); u_ = sum(v[1] for t, v in cnt.items() if t >= ts); p_unf[str(ts)] = round(u_ / a, 4)
        ded[str(ts)] = round(u_ / tot_items, 4)
    res["constrained" if constrained else "as_simulated"] = {"trailing_pause_share_of_aug": round(trailing / aug, 3), "mean_n_plus_P_minus_completion": round(float(np.mean(comp_gap)), 3),
        "share_completion_before_n_plus_P": round(float(np.mean([g > 0 for g in comp_gap])), 3), "P_unf_given_t_ge": p_unf, "unfinished_passes_per_item_t_ge": ded,
        "test_stratum_t_ge_9_no_pause_unfinished": strat[0][1], "P_unf_t_ge_9_given_H0": round(strat[0][1] / max(1, strat[0][0]), 4),
        "P_unf_t_ge_9_given_H1": round(strat[1][1] / max(1, strat[1][0]), 4), "max_unmarked_count_on_unfinished": umax}
json.dump(res, open(sys.argv[1], "w", newline="\n"), indent=1)
for k, v in res.items():
    print(k, v)
