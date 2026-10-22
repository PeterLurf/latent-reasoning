"""THEORY §224 FIXED-HOLD scorer, Stage 1 (registered text docs/research_fixedhold_2026-10-21/FH_v1.md at 232e84ab; Theory is the only scorer).
Stage 2 readings are added (with fixtures) before any Stage-2 output, only if TRIGGER passes.

Cells under boxes/runs_fsq (Theory's names, given to Experiments): FH-FT s224_fhft_s{1,2}, RD-FT s224_rdft_s{1,2}, FT-CTRL s224_ftctrl_s2 (new) and
s224_ftctrl_s1_re (3090 re-eval of s215_ftctrl_s1/model.pt; eval files only).
Files per cell: eval_khorizon.json (acc.intact[K'][n], K' 24..128), eval_vb.json (acc_n2[n] n 1..16, stall.recovery), card (both files); trained cells
also log.jsonl (step; fh_r_far) and args.json (steps). One 3090 class across every Stage-1 eval file (else VOID).
H(K') = mean over n 1..8 of acc.intact[K'][n]. FT-NOLOSS(A, s): acc_n2 n 1..8 >= 0.99, stall.recovery >= ctrl - 0.02, DEPTH (mean acc_n2 n 9..11)
>= ctrl - 0.10. TRIGGER: H_FH(128) - H_CTRL(128) >= 0.30 in both seeds and FT-NOLOSS(FH) in both; one seed only -> SPLIT (no Stage 2).
Censors (rule b): a dead trained cell (log last step < args steps), a missing file / K' 128 at --final, a non-3090 or mixed card -> that seed fails.
Descriptive: RD-FT gain, FH - RD at 128, r ratio fh_r_far(FH) / fh_r_far(RD) (mean of the last 10 logged values), the lambda line.
    python -B theory/score_s224.py [--final] [--fixtures]
"""
import argparse, json, os, sys, tempfile
import numpy as np

_HR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(_HR, "boxes", "runs_fsq")
OUT = os.path.join(_HR, "theory", "results", "score_s224.json")
S1 = {s: {"fh": "s224_fhft_%s" % s, "rd": "s224_rdft_%s" % s, "ctrl": ("s224_ftctrl_s1_re" if s == "s1" else "s224_ftctrl_s2")} for s in ("s1", "s2")}
KS = [24, 32, 48, 64, 96, 128]
CARD = "3090"


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def rows(p):
    try:
        return [json.loads(l) for l in open(p) if l.strip()]
    except OSError:
        return None


def cell(runs, name, trained):
    d = os.path.join(runs, name); kh, vb = jl(os.path.join(d, "eval_khorizon.json")), jl(os.path.join(d, "eval_vb.json"))
    miss = [f for f, v in (("eval_khorizon.json", kh), ("eval_vb.json", vb)) if v is None]
    o = {"name": name, "missing": miss}
    if trained:
        lg, ar = rows(os.path.join(d, "log.jsonl")), jl(os.path.join(d, "args.json"))
        if lg is None or ar is None:
            o["missing"] += [f for f, v in (("log.jsonl", lg), ("args.json", ar)) if v is None]
        else:
            last = max((r.get("step", 0) for r in lg), default=0); o["dead"] = last < int(ar.get("steps", 0)); o["last_step"] = last
            rr = [r["fh_r_far"] for r in lg if r.get("fh_r_far") is not None]; o["r_far_end"] = float(np.mean(rr[-10:])) if rr else None
            lam = [r for r in lg if "fh_lambda_set" in r]; o["lambda_line"] = lam[-1] if lam else None
    if kh is not None:
        ks = [k for k in KS if str(k) in (kh.get("acc") or {}).get("intact", {})]
        o["H"] = {k: float(np.mean([kh["acc"]["intact"][str(k)][str(n)] for n in range(1, 9)])) for k in ks}
        o["cards"] = [kh.get("card")]
    if vb is not None:
        o["acc18"] = min(vb["acc_n2"][str(n)] for n in range(1, 9)); o["depth"] = float(np.mean([vb["acc_n2"][str(n)] for n in (9, 10, 11)]))
        o["stall"] = (vb.get("stall") or {}).get("recovery"); o.setdefault("cards", []).append(vb.get("card"))
    return o


def score(runs=R, final=False):
    out = {"section": 224, "stage": 1, "registered": "FH_v1.md @ 232e84ab", "seeds": {}}
    cards = []
    for s, C in S1.items():
        c = {k: cell(runs, v, not v.endswith("_re")) for k, v in C.items()}; cards += [x for v in c.values() for x in v.get("cards", [])]
        gate = [c["fh"], c["ctrl"]]                                         # TRIGGER reads FH and FT-CTRL; RD is descriptive (its own censor only)
        miss = [(v["name"], m) for v in gate for m in v["missing"]] + [(v["name"], "K'128") for v in gate if "H" in v and 128 not in v["H"]]
        r = {"cells": c}
        if miss:
            r["status"] = ("FAIL (rule b: missing %s)" if final else "pending (missing %s)") % miss; out["seeds"][s] = r; continue
        dead = [v["name"] for v in gate if v.get("dead")]
        if dead:
            r["status"] = "FAIL (dead: %s)" % dead; out["seeds"][s] = r; continue
        fh, rd, ct = c["fh"], c["rd"], c["ctrl"]
        rd_ok = not rd["missing"] and not rd.get("dead") and 128 in rd.get("H", {})
        nol = fh["acc18"] >= 0.99 and fh["stall"] is not None and ct["stall"] is not None and fh["stall"] >= ct["stall"] - 0.02 and fh["depth"] >= ct["depth"] - 0.10
        gain = fh["H"][128] - ct["H"][128]
        r.update({"gain_fh": gain, "gain_rd": (rd["H"][128] - ct["H"][128]) if rd_ok else "RD censored / missing", "fh_minus_rd": (fh["H"][128] - rd["H"][128]) if rd_ok else None,
                  "ft_noloss": bool(nol), "r_ratio": (fh["r_far_end"] / rd["r_far_end"]) if rd_ok and fh.get("r_far_end") and rd.get("r_far_end") else None, "seed_pass": bool(gain >= 0.30 and nol), "status": "SCORED"})
        out["seeds"][s] = r
    cls = sorted({str(x) for x in cards if x is not None})
    out["card_classes"] = cls
    if cls and (len(cls) > 1 or CARD not in cls[0]):
        out["verdict"] = "VOID (card: %s; one 3090 class required)" % cls; return out
    st = [out["seeds"][s]["status"] for s in S1]
    if any(x.startswith("pending") for x in st):
        out["verdict"] = "pending"; return out
    passes = [out["seeds"][s].get("seed_pass", False) for s in S1]
    out["verdict"] = "TRIGGER (Stage 2 runs)" if all(passes) else ("SPLIT (no Stage 2)" if any(passes) else "NO TRIGGER (Stage 2 not run)")
    return out


# ------------------------------------------------------------------ fixtures
def _mk(d, name, H128, acc18=1.0, depth=0.6, stall=1.0, card="NVIDIA GeForce RTX 3090", trained=True, steps=12000, last=12000, r=1e-3, k128=True):
    p = os.path.join(d, name); os.makedirs(p, exist_ok=True)
    ks = KS if k128 else KS[:-1]
    acc = {"intact": {str(k): {str(n): (1.0 if k <= 32 else H128) for n in range(1, 9)} for k in ks}}
    json.dump({"acc": acc, "card": card}, open(os.path.join(p, "eval_khorizon.json"), "w"))
    an2 = {str(n): (acc18 if n <= 8 else depth) for n in range(1, 17)}
    json.dump({"acc_n2": an2, "stall": {"recovery": stall}, "card": card}, open(os.path.join(p, "eval_vb.json"), "w"))
    if trained:
        json.dump({"steps": steps}, open(os.path.join(p, "args.json"), "w"))
        with open(os.path.join(p, "log.jsonl"), "w") as fh:
            for s in range(100, last + 1, 100):
                fh.write(json.dumps({"step": s, "fh_r_far": r}) + "\n")


def _world(over=None):
    over = over or {}
    d = tempfile.mkdtemp()
    spec = {"s1": dict(fh=0.95, rd=0.4, ctrl=0.3), "s2": dict(fh=0.95, rd=0.4, ctrl=0.25)}
    for s, C in S1.items():
        for k, name in C.items():
            kw = dict(over.get((s, k), {})); h = kw.pop("H128", spec[s][k])
            _mk(d, name, h, trained=not name.endswith("_re"), **kw)
    return d


def fixtures():
    res = []
    o = score(_world()); res.append(("both seeds gain 0.65 / 0.70 -> TRIGGER", o["verdict"].startswith("TRIGGER"), o["verdict"]))
    o = score(_world({("s2", "fh"): {"H128": 0.5}})); res.append(("s2 gain 0.25 -> SPLIT", o["verdict"].startswith("SPLIT"), o["verdict"]))
    o = score(_world({("s1", "fh"): {"H128": 0.5}, ("s2", "fh"): {"H128": 0.4}})); res.append(("no gain -> NO TRIGGER", o["verdict"].startswith("NO TRIGGER"), o["verdict"]))
    o = score(_world({("s1", "fh"): {"depth": 0.45}})); res.append(("FH depth -0.15 vs ctrl -> s1 fails -> SPLIT", o["verdict"].startswith("SPLIT"), o["verdict"]))
    o = score(_world({("s1", "fh"): {"stall": 0.95}})); res.append(("FH stall -0.05 -> SPLIT", o["verdict"].startswith("SPLIT"), o["verdict"]))
    o = score(_world({("s2", "fh"): {"acc18": 0.98}})); res.append(("FH acc_n2 0.98 -> SPLIT", o["verdict"].startswith("SPLIT"), o["verdict"]))
    o = score(_world({("s1", "rd"): {"last": 9000}})); res.append(("dead RD -> descriptive only, TRIGGER stands", o["verdict"].startswith("TRIGGER") and o["seeds"]["s1"]["fh_minus_rd"] is None, o["verdict"]))
    o = score(_world({("s2", "ctrl"): {"last": 9000}})); res.append(("dead FT-CTRL s2 -> seed FAIL -> SPLIT", o["seeds"]["s2"]["status"].startswith("FAIL (dead") and o["verdict"].startswith("SPLIT"), o["verdict"]))
    o = score(_world({("s1", "fh"): {"last": 9000}})); res.append(("dead FH s1 -> SPLIT", o["verdict"].startswith("SPLIT"), o["verdict"]))
    d = _world(); os.remove(os.path.join(d, "s224_fhft_s2", "eval_khorizon.json"))
    o = score(d); res.append(("missing file -> pending", o["verdict"] == "pending", o["verdict"]))
    o = score(d, final=True); res.append(("missing at --final -> FAIL -> SPLIT", o["seeds"]["s2"]["status"].startswith("FAIL (rule b") and o["verdict"].startswith("SPLIT"), o["verdict"]))
    o = score(_world({("s1", "fh"): {"k128": False}})); res.append(("no K'128 -> pending", o["verdict"] == "pending", o["verdict"]))
    o = score(_world({("s2", "ctrl"): {"card": "NVIDIA GeForce RTX 3070"}})); res.append(("mixed card -> VOID", o["verdict"].startswith("VOID"), o["verdict"]))
    o = score(_world({("s1", "fh"): {"r": 1e-4}})); res.append(("r ratio descriptive 0.1", abs(o["seeds"]["s1"]["r_ratio"] - 0.1) < 1e-9, o["seeds"]["s1"]["r_ratio"]))
    for r in res:
        print("FIXTURE", "PASS" if r[1] else "FAIL", r[0], r[2])
    ok = sum(bool(r[1]) for r in res); print("fixtures %d/%d" % (ok, len(res))); return ok == len(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--final", action="store_true"); ap.add_argument("--fixtures", action="store_true"); a = ap.parse_args()
    if a.fixtures:
        sys.exit(0 if fixtures() else 1)
    o = score(final=a.final); print(o["verdict"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(o, open(OUT, "w", newline=chr(10)), indent=1, default=str)
