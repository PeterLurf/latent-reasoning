"""Candidate 3 prototype reading (THEORY_STABLE_LATENT, LOCAL_BACKLOG 4060 #4; descriptive, $0; written before any rampproto run is read;
revised before any late run exists, after the Audit helper's review: MOVES / VANISHES / STAYS, R', per card).
Paired runs per seed on one card: plain (p_hard 0 -> 1 over 2k-4k, §120; stop 8k) and late (over 6k-8k; stop 8.9k, before L_max 16
at 9k); L_max schedule unchanged (4 from 2k, 8 from 5k). Dirs boxes/runs_fsq/rampproto_{plain,late}_s{S}_{card}, S 601-610.
Before 6k the late carrier is unrounded (p_hard 0), and continuous carriers pass L_max 4 easily (§123 / §126), so few late A-flags is
expected whatever the cause; the informative windows are the late arm's ramp windows.
Windows (loss at the nearest logged step; FLAG4 form: ratio >= 0.9 and end loss >= 0.25 loss(100)):
  A  = loss(4k) / loss(2.6k): plain = ramp + L_max 4 step; late = L_max 4 step only (unrounded).
  R  = loss(8k) / loss(6.6k): late = ramp only (L_max 8 held since 5k); plain = nothing changes.
  R' = loss(8.9k) / loss(6.6k), late only: a lasting (not transient) rise after the late ramp.
  B  = loss(7k) / loss(5.6k): the L_max 8 step, reported only.
Seeds flagged at 2k by 123-3 in the plain run (an entry plateau, F1; both arms are identical to 2k) are excluded.
Reading, fixed now (pooled; needs >= 2 plain A-flags among informative seeds, else UNINFORMATIVE; per card printed, descriptive).
P = seeds plain flags in A:
  STAYS:     late A-flags >= ceil(0.75 |P|) (the failure stays with the L_max 4 step; rounding not needed for it).
  MOVES:     late R'-flags cover >= ceil(|P| / 2) of P (the failure follows the ramp: rounding fails the code whenever applied;
             delaying it does not help).
  VANISHES:  late A-flags <= floor(|P| / 2), late R'-flags <= 1 (among informative seeds), and late loss(8.9k) < plain loss(8k) on a
             majority of P (delaying the ramp let the code separate first: candidate 3's claim; the 900 extra late steps are disclosed).
  MIXED otherwise. Precedence STAYS > MOVES > VANISHES; every condition is printed.
    python theory/proto_ramp.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
SEEDS = [(s, "4060") for s in range(601, 606)] + [(s, "3070") for s in range(606, 611)]
# Revised before any informative pair is read (the Audit helper): 601 / 602 flag at 2k (uninformative by construction). The rest are
# drawn from seeds whose local 2k replay on the SAME card was NOT flagged (§130 replay130 / replay_x; selection uses only pre-2k data,
# where both arms are identical; seeds with a known 24k outcome, the s130chk seeds, are excluded): 4060 3 / 4 / 8, 3070 14 / 16 / 27 / 28 / 31.
# A plain run that flags at 2k may stop there and its late twin is not run (both arms are identical to 2k).
SEEDS = [(601, "4060"), (602, "4060"), (603, "4060"), (3, "4060"), (4, "4060"), (8, "4060")] + [(s, "3070") for s in (14, 16, 27, 28, 31)]


def losses(d):
    p = os.path.join(R, d, "log.jsonl")
    if not os.path.exists(p):
        return None
    L = {x["step"]: x["loss"] for x in map(json.loads, open(p)) if "loss" in x and "step" in x}
    return L or None


def at(L, t):
    return L[min(L, key=lambda s: abs(s - t))]


def flag(L, t0, t1):
    if max(L) < t1 - 50:
        return None, None
    r = at(L, t1) / max(at(L, t0), 1e-9)
    return r, bool(r >= 0.9 and at(L, t1) >= 0.25 * at(L, 100))


def reading(rows):
    inf = [(s, r) for s, r in rows if not r["flag2k"]]
    P = [s for s, r in inf if r["plain_A"][1]]; la = [s for s, r in inf if r["late_A"][1]]
    lr = [s for s, r in inf if r["late_Rp"][1]]; rd = dict(inf)
    if len(P) < 2:
        return {"reading": "UNINFORMATIVE (%d plain A-flags among %d informative seeds)" % (len(P), len(inf)), "P": P, "late_A": la, "late_Rp": lr}
    stays = len(la) >= math.ceil(0.75 * len(P))
    moved = sorted(set(P) & set(lr)); moves = len(moved) >= math.ceil(len(P) / 2)
    lower = [s for s in P if rd[s]["late_8.9k"] < rd[s]["plain_8k"]]
    vanishes = len(la) <= len(P) // 2 and len(lr) <= 1 and len(lower) > len(P) / 2
    v = "STAYS" if stays else ("MOVES" if moves else ("VANISHES" if vanishes else "MIXED"))
    return {"reading": v, "P": P, "late_A": la, "late_Rp": lr, "moved": moved, "late_lower_on_P": lower,
            "conditions": {"STAYS": stays, "MOVES": moves, "VANISHES": vanishes}, "n_informative": len(inf)}


def main():
    rows, out = [], {}
    for s, card in SEEDS:
        lp, ll = losses("rampproto_plain_s%d_%s" % (s, card)), losses("rampproto_late_s%d_%s" % (s, card))
        if lp is not None and max(lp) >= 2000 and at(lp, 2000) >= 0.9 * at(lp, 1000) and at(lp, 2000) >= 0.25 * at(lp, 100):
            print("s%d %s: flagged at 2k (uninformative by construction; excluded)" % (s, card)); continue
        if lp is None or ll is None or max(lp) < 7950 or max(ll) < 8850:
            print("s%d %s: missing or running (plain to %s, late to %s)" % (s, card, lp and max(lp), ll and max(ll))); continue
        f2k = bool(at(lp, 2000) >= 0.9 * at(lp, 1000) and at(lp, 2000) >= 0.25 * at(lp, 100))
        r = {"card": card, "flag2k": f2k, "par2k": round(abs(at(lp, 2000) - at(ll, 2000)), 4)}
        for arm, L in (("plain", lp), ("late", ll)):
            for w, (t0, t1) in (("A", (2600, 4000)), ("B", (5600, 7000)), ("R", (6600, 8000))):
                r["%s_%s" % (arm, w)] = flag(L, t0, t1)
            r["%s_8k" % arm] = round(at(L, 8000), 3)
        r["late_Rp"] = flag(ll, 6600, 8900); r["late_8.9k"] = round(at(ll, 8900), 3)
        # Descriptive, added after reading the s14 pair (no label changes): the onset of the late arm's rise, i.e. the first logged step
        # >= 5k where the loss exceeds its running minimum since 5k by >= 1.0. p_hard is 0 up to 6,000, so an onset <= 6,000 is a failure
        # of the UNROUNDED carrier, not caused by the late ramp.
        mn, onset = None, None
        for st in sorted(x for x in ll if x >= 5000):
            mn = ll[st] if mn is None else min(mn, ll[st])
            if ll[st] >= mn + 1.0:
                onset = st; break
        r["late_rise_onset"] = onset; r["late_rise_before_ramp"] = None if onset is None else onset <= 6000
        out[s] = r; rows.append((s, r))
        print("s%d %s 2k-flag %s par %.4f | A plain %.3f %s late %.3f %s | R plain %.3f late %.3f %s | R' late %.3f %s | B plain %.3f late %.3f"
              " | loss plain 8k %.3f late 8k %.3f 8.9k %.3f | late rise onset %s (before ramp %s)" % (
                  s, card, f2k, r["par2k"], r["plain_A"][0], r["plain_A"][1], r["late_A"][0], r["late_A"][1], r["plain_R"][0], r["late_R"][0],
                  r["late_R"][1], r["late_Rp"][0], r["late_Rp"][1], r["plain_B"][0], r["late_B"][0], r["plain_8k"], r["late_8k"], r["late_8.9k"],
                  r["late_rise_onset"], r["late_rise_before_ramp"]))
    res = {"runs": out, "pooled": reading(rows), "per_card": {c: reading([x for x in rows if x[1]["card"] == c]) for c in ("4060", "3070")}}
    for k in ("4060", "3070"):
        print("card %s (descriptive): %s" % (k, res["per_card"][k]))
    print("POOLED:", res["pooled"])
    json.dump(res, open(os.path.join(ROOT, "theory", "results", "proto_ramp.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
