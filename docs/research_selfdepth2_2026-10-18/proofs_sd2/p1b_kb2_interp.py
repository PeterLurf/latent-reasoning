"""PROOFS_SD2 check 1b ($0): sensitivity of KB2 (and STOP-LOC-2) under the fractional maps (H_prop, H_add) to the interpolation axis.
p1_panels interpolates round 1's a_k(n) bilinearly in (K', n); accuracy jumps across the diagonal K' ~ n, so the alternative is bilinear in (lag j = K' - n, n).
H_prop maps (k, n) -> (12 + .75 (k - 16), 12 + .75 (n - 16)), i.e. lag j -> .75 j; H_add (rigid) keeps the lag."""
import json, math, os, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p1", os.path.join(HERE, "p1_panels.py"))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    p1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p1)
T1, A1, delta = p1.T1, p1.A1, p1.delta


def lagf(j, n):                                                             # bilinear in (lag, n) on round 1's table
    n0 = math.floor(n); j0 = math.floor(j); fn, fj = n - n0, j - j0
    g = lambda jj, nn: T1(nn + jj, nn)  # noqa: E731
    return (1 - fn) * ((1 - fj) * g(j0, n0) + fj * g(j0 + 1, n0)) + fn * ((1 - fj) * g(j0, n0 + 1) + fj * g(j0 + 1, n0 + 1))


maps = {"H_prop": lambda k, n: lagf(0.75 * (k - n), 12 + 0.75 * (n - 16)), "H_add_rigid_delta": lambda k, n: lagf(k - n, n - 4 - delta)}
out = {}
for h, f in maps.items():
    a2 = {n: p1.H[h][1](n) for n in p1.NS}                                  # per-n a2 from acc_n2 (as sd2_tables)
    r = p1.readings(a2, f); out[h] = {"KB2_lag_interp": r["KB2"], "STOP_lag_interp": r["STOP"], "G": r["G"], "KB2_kn_bilinear": p1.pan[h]["per_n"]["KB2"],
                                      "terms": {n: [round(a2[n], 4), round(f(18, n), 4)] for n in (19, 20)}}
json.dump(out, open(os.path.join(HERE, "p1b_kb2_interp.json"), "w", newline="\n"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
