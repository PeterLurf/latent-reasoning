# Monte Carlo of wave-based sequential seed designs with an optional conservative 8k detector.
# Cost model (assumption): $0.05 per 1k steps on a cloud 3090 (48k ~ $2.4) + $0.10 per seed overhead; 0.74 s/step.
import random, sys
sys.stdout.reconfigure(encoding="utf-8")
C_STEP, OVH, T_END, T_DET, SPS = 0.05, 0.10, 30, 8, 0.885
def seed_outcome(rng, p, det):
    ok = rng.random() < p
    if det is None: return ok, False
    sens, f = det
    flag = (rng.random() < f) if ok else (rng.random() < sens)
    return ok, flag
def run(design, p, det, rng, wave=3):
    cost = 0.0; hours = 0.0; succ = fail = 0; n = 0
    while True:
        if design in ("A", "A8", "C3"):   # all-of-N, stop at first failure
            N = {"A": 6, "A8": 8, "C3": 3}[design]
            if fail >= 1: return False, cost, hours, n
            if succ >= N: return True, cost, hours, n
            k = min(wave, N - n)
        else:               # Simon n1=6 r1=4 (<=1 fail in first 6), n=13 r=10 (<=2 fails in 13)
            if (n <= 6 and fail >= 2) or fail >= 3: return False, cost, hours, n
            if succ >= 11: return True, cost, hours, n
            k = min(wave, (6 if n < 6 else 13) - n)
        outs = [seed_outcome(rng, p, det) for _ in range(k)]; n += k
        flags = sum(fl for _, fl in outs)
        allowed = (0 if design in ("A", "A8", "C3") else (1 - fail if n <= 6 else 2 - fail))
        if det is not None and flags > allowed:      # the claim is already lost at 8k: stop the whole wave
            cost += k * (T_DET * C_STEP + OVH); hours += T_DET * 1000 * SPS / 3600
            fail += flags; succ += 0; return False, cost, hours, n
        for ok, fl in outs:
            if fl: cost += T_DET * C_STEP + OVH; fail += 1
            else:
                cost += (48 if design == "C3" else T_END) * C_STEP + OVH
                if ok: succ += 1
                else: fail += 1
        hours += (48 if design == "C3" else T_END) * 1000 * SPS / 3600
R = 100000
for design, dets in (("C3", (None,)), ("A", (None, (0.8, 0.05), (0.5, 0.0))), ("A8", (None, (0.5, 0.0))), ("B", (None, (0.5, 0.0)))):
    for det in dets:
        line = []
        for p in (0.6, 0.667, 0.9, 0.95):
            rng = random.Random(1234)
            res = [run(design, p, det, rng) for _ in range(R)]
            pr = sum(r[0] for r in res) / R; ec = sum(r[1] for r in res) / R; eh = sum(r[2] for r in res) / R; en = sum(r[3] for r in res) / R
            mx = max(r[1] for r in res)
            line.append("p=%.3f: P(claim) %.3f, E[$] %.2f (max %.1f), E[seeds] %.1f, E[h] %.1f" % (p, pr, ec, mx, en, eh))
        print("design %s, detector %s" % (design, det)); [print("   " + l) for l in line]
