"""§205 seed-1 admission at HOSTPROBE (Audit 2026-10-15 04:5x): refuse a slow s1 host before it trains, instead of the rate guard's stop at ~4k.
s1 projection = (G0 x probe + (48000 - G0) x probe x R) / 3600 + 0.35 h, times the host's $/h, with R = the largest post-gamma / probe ratio seen on the
s2 / s3 hosts (b298 1.527 / 0.6591 = 2.32, b297 1.815 / 0.8012 = 2.27; argv[3] overrides). s2 + s3 are projected live exactly as s205_rate_guard
(post-gamma rate from their logs). Family = s1 + s2 + s3 + BAD-HOST total. Prints ADMIT / REFUSE with the numbers; exit 0 = admit, 1 = refuse.
    python scratch/s205_s1_admit.py <probe s/step> <dph> [R]"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("rg", "D:/Latent/scratch/s205_rate_guard.py"); RG = importlib.util.module_from_spec(spec); spec.loader.exec_module(RG)
probe, dph = float(sys.argv[1]), float(sys.argv[2]); R = float(sys.argv[3]) if len(sys.argv) > 3 else 2.32
H = RG.hosts(); P = RG.prices() or {}; proj = {}
for n in (297, 298):
    iid, h, p = H[n]; rec = RG.records(h, p, RG.CELLS[n]); d_ = P.get(iid)
    post = [r for r in rec if r[0] >= RG.G0] if rec else []
    if d_ is None or len(post) < 2:
        print("REFUSE (cannot read b%d's log or price - no admission without the live s2 / s3 projection)" % n); sys.exit(1)
    rate = (post[-1][1] - post[0][1]) / (post[-1][0] - post[0][0]); st, sec = rec[-1]
    proj[n] = ((sec + (RG.STEPS - st) * rate) / 3600 + RG.EVAL_SETUP_H) * d_
s1 = ((RG.G0 * probe + (RG.STEPS - RG.G0) * probe * R) / 3600 + RG.EVAL_SETUP_H) * dph; bad = RG.bad_total(); fam = s1 + sum(proj.values()) + bad
msg = "s1 probe %.4f s/step x R %.2f at $%.4f/h -> $%.2f; s2 $%.2f + s3 $%.2f + BAD-HOST $%.2f -> family $%.2f vs cap $%.2f" % (probe, R, dph, s1, proj[297], proj[298], bad, fam, RG.CAP)
print(("ADMIT " if fam <= RG.CAP else "REFUSE ") + msg); sys.exit(0 if fam <= RG.CAP else 1)
