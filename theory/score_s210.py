"""THEORY §210 scoring (CODI-BUDGET-20; registered in docs/THEORY.md §210 at de919de5; Theory is the only scorer).
Cell boxes/runs_fsq/gsm_codi_20ep_s1: eval.json (acc with 6 latents, epochs, k_latent_ablation, valid_hist, by_steps, train_items, effective_bs),
eval_latent_texts.json ({"0": {"share_txt_with_<<"}}), eval_items.json. Comparator gsm_codi_4ep_s1 (CODI4 0.189, 3090).
Primary: REACHES-PUBLISHED acc >= 0.348 / STAYS-LOW acc <= 0.217 / PARTIAL. Secondary (TXT20 = 0-latent acc, read only if >= 50% of its
outputs contain "<<"): GAP-CLOSES acc >= TXT20 - T / GAP-PERSISTS acc <= TXT20 - 0.068 / GAP-NARROWS. Censoring (rule b): epochs < 20 -> UNREAD.
    python theory/score_s210.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "boxes", "runs_fsq")
CELL, REF = "gsm_codi_20ep_s1", "gsm_codi_4ep_s1"
T, EPS, HI, LO, PERSIST, EPOCHS, SIG = 0.028, 1e-9, 0.348, 0.217, 0.068, 20.0, 0.0242


def jl(d, f="eval.json"):
    p = os.path.join(R, d, f)
    return json.load(open(p)) if os.path.exists(p) else None


def depth(by, lo, hi):
    ks = [k for k in by if lo <= int(k) <= hi]; n = sum(by[k][1] for k in ks)
    return (sum(by[k][0] * by[k][1] for k in ks) / n, n) if n else (None, 0)


def main():
    e = jl(CELL); out = {}
    if e is None:
        out["reading"] = "pending (not landed)"
    elif (e.get("epochs") or 0) < EPOCHS - 1e-6:
        out["reading"] = "UNREAD (run ended at %.2f epochs < 20; rule b, counts as NOT REACHES-PUBLISHED in any tally)" % (e.get("epochs") or 0)
        out["valid_last"] = (e.get("valid_hist") or [[None, None]])[-1]
    else:
        x = e["acc"]
        lab = "REACHES-PUBLISHED" if x >= HI - EPS else ("STAYS-LOW" if x <= LO + EPS else "PARTIAL")
        zone = [b for b in (HI, LO) if abs(x - b) <= SIG + EPS]
        out.update(acc=x, label=lab, misread_zone=("within 1 sigma (%.4f) of %s" % (SIG, zone)) if zone else None)
        tx = jl(CELL, "eval_latent_texts.json") or {}; share = (tx.get("0") or {}).get("share_txt_with_<<")
        t20 = (e.get("k_latent_ablation") or {}).get("0")
        if t20 is None or share is None or share < 0.5 - EPS:
            gap_lab, gap = "UNREAD (0-latent arm not text CoT: share %s)" % share, None
        else:
            gap = x - t20; gap_lab = "GAP-CLOSES" if gap >= -T - EPS else ("GAP-PERSISTS" if gap <= -PERSIST + EPS else "GAP-NARROWS")
        out.update(TXT20=t20, txt_share=share, gap=gap, gap_label=gap_lab)
        vh = e.get("valid_hist") or []; per = (e.get("effective_bs") or 128) / (e.get("train_items") or 382859)
        if vh:
            fin = vh[-1][1]; first = next((s for s, v in vh if v >= 0.9 * fin - EPS), None)
            out.update(valid_final=fin, epoch_valid_0p9_final=None if first is None else first * per, valid_n=len(vh))
        by = e.get("by_steps") or {}; ref = jl(REF)
        out["by_depth"] = {"m<=2": depth(by, 0, 2), "m>=3": depth(by, 3, 99)}
        a, b = jl(CELL, "eval_items.json"), jl(REF, "eval_items.json")
        if a and b and len(a) == len(b):
            d = [int(u["ok"]) - int(v["ok"]) for u, v in zip(a, b)]; m = sum(d) / len(d)
            sd = math.sqrt(sum((v - m) ** 2 for v in d) / (len(d) - 1)); out.update(minus_codi4=m, paired_se=sd / math.sqrt(len(d)))
        g20, g4 = (jl(CELL, "args.json") or {}).get("train_gpu"), (jl(REF, "args.json") or {}).get("train_gpu")
        out["train_gpu_note"] = "CODI20 on %s vs CODI4 on %s (GPU-class caveat)" % (g20, g4)
        out["reading"] = "%s (CODI20 %.3f%s) | %s (TXT20 %s, gap %s) | vs CODI4 %s (SE %s)" % (
            lab, x, ("; " + out["misread_zone"]) if zone else "", gap_lab, t20, None if gap is None else round(gap, 3),
            None if "minus_codi4" not in out else round(out["minus_codi4"], 3), None if "paired_se" not in out else round(out["paired_se"], 4))
    print("READING:", out["reading"])
    json.dump(out, open(os.path.join(ROOT, "theory", "results", "score_s210.json"), "w", newline=chr(10)), indent=1, default=str)


if __name__ == "__main__":
    main()
