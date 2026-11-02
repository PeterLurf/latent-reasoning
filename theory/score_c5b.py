"""THEORY Note 259-5 C5-BUN scorer (eval-only, frozen bun_s1 / bun_s2; Theory is the only scorer; committed before output).
Two halting arms, each tested on TEST2 d 1..7, fit / calibrated on d <= 4 items only:
  H-INC (training-free): run pass t; halt when rel_inc_t = ||S_t - S_{t-1}|| / ||S_t|| < tau (tau = the largest value with acc_halt >= acc_k8 - 0.02 at every d 1-4 on the fit set);
    the answer is read at K' = t - 1, so khalt = t - 1 and passes_run = t. File eval_halt_inc.json per d {acc_k8, acc_halt, mean_khalt, mean_passes_run, never_halt_n, n}, plus "tau".
  H-HEAD2 (Audit's relabel): the §248 head on bun with y = [pred(K') == pred(K' 14)] over K' 0..14 (fit d <= 4); halt at the first K' with p >= 0.90 else 14.
    File eval_halthead2.json in eval_halthead's layout (per d {acc_k8, acc_halt, mean_khalt, never_halt_n, n}, head.val_acc).
Rule per arm per seed (censoring as Note 257-2: never-halt counts at K'max):
  IN-RANGE d 1-4: acc_halt >= acc_k8 - 0.02; mean_khalt increasing; mean_khalt <= d + 1.
  EXTRAP d 5-7: acc_halt >= acc_k8 - 0.02 each d; mean_khalt <= d each d; mean_khalt strictly increasing over d 1-7.
  H-INC also reports mean_passes_run (= khalt + 1; the price of detecting the fixed point), descriptive.
  Labels C5-EXTRAP / C5-IN-RANGE / NO-SIGNAL (VOID if H-HEAD2 val_acc < 0.80).
    python -B theory/score_c5b.py [--fixtures]
"""
import argparse, json, os, sys, tempfile
_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HR, "theory"))
import score_hx as HX  # noqa: E402

R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_c5b.json")
RUNS = ("bun_s1", "bun_s2")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def score(runs=R):
    o = {"inc": {}, "head2": {}}
    for r in RUNS:
        e = jl(os.path.join(runs, r, "eval_halt_inc.json"))
        o["inc"][r] = HX.halt(e) if e is not None else {"label": "pending"}
        if e is not None:
            o["inc"][r]["tau"] = e.get("tau"); o["inc"][r]["passes_run"] = {d: e[str(d)].get("mean_passes_run") for d in range(1, 8)}
        o["head2"][r] = HX.halt(jl(os.path.join(runs, r, "eval_halthead2.json")))
    o["verdict"] = "H-INC %s | H-HEAD2 %s" % (" / ".join(o["inc"][r]["label"] for r in RUNS), " / ".join(o["head2"][r]["label"] for r in RUNS)); return o


def fixtures():
    res = []

    def world(inc_drop=0.0, head_late=False):
        d = tempfile.mkdtemp()
        for r in RUNS:
            p = os.path.join(d, r); os.makedirs(p)
            e = {"tau": 0.05}; h = {"head": {"val_acc": 0.95}}
            for x in range(1, 8):
                e[str(x)] = {"acc_k8": 1.0, "acc_halt": 1.0 - (inc_drop if x >= 5 else 0), "mean_khalt": x - 0.9, "mean_passes_run": x + 0.1, "n": 1000, "never_halt_n": 0}
                h[str(x)] = {"acc_k8": 1.0, "acc_halt": 1.0, "mean_khalt": (x - 0.9) if not (head_late and x >= 5) else 9.0, "n": 1000, "never_halt_n": 0}
            json.dump(e, open(os.path.join(p, "eval_halt_inc.json"), "w")); json.dump(h, open(os.path.join(p, "eval_halthead2.json"), "w"))
        return d
    for name, kw, want in (("both extrap", {}, "H-INC C5-EXTRAP / C5-EXTRAP | H-HEAD2 C5-EXTRAP / C5-EXTRAP"), ("inc lossy", {"inc_drop": 0.1}, "H-INC C5-IN-RANGE / C5-IN-RANGE"),
                           ("head late", {"head_late": True}, "H-HEAD2 C5-IN-RANGE / C5-IN-RANGE")):
        o = score(world(**kw)); res.append((name, want in o["verdict"], o["verdict"]))
    o = score(tempfile.mkdtemp()); res.append(("pending", "pending" in o["verdict"], o["verdict"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    k = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (k, len(res))); return k == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", encoding="utf-8", newline=chr(10)), indent=1, default=str)
