"""Candidate 3 prototype reading (THEORY_STABLE_LATENT, LOCAL_BACKLOG 4060 #4; descriptive, $0; written before any rampproto run).
Paired runs per seed on one card: plain (p_hard 0 -> 1 over 2k-4k, §120) and late (over 6k-8k); L_max schedule unchanged (4 from 2k,
8 from 5k, 16 from 9k); stopped at 8k. Dirs boxes/runs_fsq/rampproto_{plain,late}_s{S}_{card}, S 601-610.
Windows (loss at the nearest logged step; FLAG4 form: ratio >= 0.9 and end loss >= 0.25 loss(100)):
  A = loss(4k) / loss(2.6k): plain = ramp + L_max 4 step; late = L_max 4 step only.
  R = loss(8k) / loss(6.6k): late = ramp only (L_max 8 held since 5k); plain = nothing changes (already hard).
  B = loss(7k) / loss(5.6k): the L_max 8 step (both arms; late is mid-ramp from 6k), reported only.
Seeds flagged at 2k by 123-3 in the plain run (an entry plateau, F1; both arms are identical to 2k) are excluded.
Reading, fixed now (needs >= 2 plain A-flags among informative seeds, else UNINFORMATIVE):
  ROUNDING: late A-flags <= floor(plain A-flags / 2) (the failure window empties when the ramp moves).
  L_MAX:    late A-flags >= ceil(0.75 * plain A-flags) (the window stays with the L_max step).
  MIXED otherwise. Also printed: late R-flags and whether they fall on the seeds that plain flagged in A ("moved").
    python theory/proto_ramp.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
SEEDS = [(s, "4060") for s in range(601, 606)] + [(s, "3070") for s in range(606, 611)]


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


def main():
    rows, out = [], {}
    for s, card in SEEDS:
        lp, ll = losses("rampproto_plain_s%d_%s" % (s, card)), losses("rampproto_late_s%d_%s" % (s, card))
        if lp is None or ll is None or max(lp) < 7950 or max(ll) < 7950:
            print("s%d %s: missing or running (plain to %s, late to %s)" % (s, card, lp and max(lp), ll and max(ll))); continue
        f2k = bool(at(lp, 2000) >= 0.9 * at(lp, 1000) and at(lp, 2000) >= 0.25 * at(lp, 100))
        r = {"card": card, "flag2k": f2k, "par2k": round(abs(at(lp, 2000) - at(ll, 2000)), 4)}
        for arm, L in (("plain", lp), ("late", ll)):
            for w, (t0, t1) in (("A", (2600, 4000)), ("B", (5600, 7000)), ("R", (6600, 8000))):
                r["%s_%s" % (arm, w)] = flag(L, t0, t1)
            r["%s_8k" % arm] = round(at(L, 8000), 3)
        out[s] = r; rows.append((s, r))
        print("s%d %s 2k-flag %s par %.4f | A plain %.3f %s late %.3f %s | R plain %.3f %s late %.3f %s | B plain %.3f late %.3f | 8k %.3f / %.3f" % (
            s, card, f2k, r["par2k"], r["plain_A"][0], r["plain_A"][1], r["late_A"][0], r["late_A"][1], r["plain_R"][0], r["plain_R"][1],
            r["late_R"][0], r["late_R"][1], r["plain_B"][0], r["late_B"][0], r["plain_8k"], r["late_8k"]))
    inf = [(s, r) for s, r in rows if not r["flag2k"]]
    pa = [s for s, r in inf if r["plain_A"][1]]; la = [s for s, r in inf if r["late_A"][1]]; lr = [s for s, r in inf if r["late_R"][1]]
    if len(pa) < 2:
        v = "UNINFORMATIVE (%d plain A-flags among %d informative seeds)" % (len(pa), len(inf))
    elif len(la) <= len(pa) // 2:
        v = "ROUNDING"
    elif len(la) >= math.ceil(0.75 * len(pa)):
        v = "L_MAX"
    else:
        v = "MIXED"
    moved = sorted(set(pa) & set(lr))
    print("informative %d | plain A-flags %s | late A-flags %s | late R-flags %s (moved: %s) | READING %s" % (len(inf), pa, la, lr, moved, v))
    json.dump({"runs": out, "plain_A": pa, "late_A": la, "late_R": lr, "moved": moved, "reading": v},
              open(os.path.join(ROOT, "theory", "results", "proto_ramp.json"), "w", newline=chr(10)), indent=1, default=float)


if __name__ == "__main__":
    main()
