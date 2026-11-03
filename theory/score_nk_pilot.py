"""THEORY §262 NAME-KEYED pilot gate (NK_v1 §4; Theory is the only scorer; committed before output).
Cells (from bun_s1, --steps 8000 --stop_at 2000; both arms share perm-then-insert text (perm_p 0.5) and askperm-style one-hop co-training items):
  pilot_nk_s1 (+ NK heads, packed into S, tied q/k), pilot_nkc_s1 (no heads).
Fields (host evals on the 2k ckpt, eval_nk_pilot.json per cell):
  cond_ws_perm[d][t] for d in 3, 4 and t >= 2 = P(pass-t WS correct | pass t-1 correct) on permuted TEST2-style items (TEST-PERM renderer);
  ws1_perm[d] = pass-1 WS on permuted items (reported, not gated: pass-1 lookup is not what NK acts on);
  NK cell only: cond_ws_perm_ablated[d][t] = the same with the NK read zeroed (W_on r -> 0).
Gate: pooled cond = mean over d 3-4, t >= 2 of cond_ws_perm.
  GO if NK pooled >= 0.60 AND NK pooled >= CTRL pooled + 0.20 AND NK pooled - NK ablated pooled >= 0.20 (causal use of the binding);
  PARK if NK pooled < 0.30 or the ablation drop < 0.05 (binding unused); else NOTE.
    python -B theory/score_nk_pilot.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_nk_pilot.json")
NK, CT = "pilot_nk_s1", "pilot_nkc_s1"


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def pooled(c):
    v = [c[d][t] for d in ("3", "4") for t in c.get(d, {}) if int(t) >= 2]
    return sum(v) / len(v) if v else None


def score(runs=R):
    a = jl(os.path.join(runs, NK, "eval_nk_pilot.json")); b = jl(os.path.join(runs, CT, "eval_nk_pilot.json"))
    if a is None or b is None:
        return {"verdict": "pending"}
    nk, ct, ab = pooled(a["cond_ws_perm"]), pooled(b["cond_ws_perm"]), pooled(a["cond_ws_perm_ablated"])
    drop = nk - ab
    go = nk >= 0.60 and nk >= ct + 0.20 and drop >= 0.20
    park = nk < 0.30 or drop < 0.05
    return {"verdict": "GO" if go else "PARK" if park else "NOTE", "nk_pooled": nk, "ctrl_pooled": ct, "nk_ablated_pooled": ab, "ablation_drop": drop,
            "ws1_perm": {"nk": a.get("ws1_perm"), "ctrl": b.get("ws1_perm")}}


def fixtures():
    res = []

    def world(nk, ct, ab):
        d = tempfile.mkdtemp()
        c = lambda v: {dd: {str(t): v for t in range(2, int(dd) + 1)} for dd in ("3", "4")}  # noqa: E731
        for run, v, extra in ((NK, nk, {"cond_ws_perm_ablated": c(ab)}), (CT, ct, {})):
            os.makedirs(os.path.join(d, run)); json.dump({"cond_ws_perm": c(v), "ws1_perm": {"3": 0.3}, **extra}, open(os.path.join(d, run, "eval_nk_pilot.json"), "w"))
        return d
    for name, args, want in (("go", (0.8, 0.4, 0.3), "GO"), ("no gain over ctrl", (0.7, 0.6, 0.2), "NOTE"), ("binding unused", (0.7, 0.3, 0.68), "PARK"),
                             ("low", (0.2, 0.1, 0.1), "PARK"), ("weak ablation", (0.7, 0.3, 0.6), "NOTE")):
        o = score(world(*args)); res.append((name, o["verdict"] == want, o["verdict"]))
    o = score(tempfile.mkdtemp()); res.append(("pending", o["verdict"] == "pending", o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1)
