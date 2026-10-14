"""GSM-TEMPLATE draft, measurement 5 (POST HOC, descriptive): accuracy of the public GPT-2 GSM8K checkpoints (bench/results/gsm8k__*.json, per-item
'correct') on GSM8K test items whose functional fingerprint (extra_stats.fingerprint) IS vs IS NOT seen in GSM8K-Aug train, by equation count m,
with a Mantel-Haenszel-style m-stratified difference (seen - unseen, weights n_seen*n_unseen/(n_seen+n_unseen)).   python -I seen_split.py"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import audit_aug as AA  # noqa: E402
import extra_stats as ES  # noqa: E402
import random  # noqa: E402

R = "D:/Latent/bench/results/"
TAGS = ["dw_nocot_ck25", "dw_cot_ck25", "dw_coconut_ck33_K6", "codi_L0", "codi_L6", "simcot_codi_L6", "simcot_coconut_K0", "simcot_coconut_K10"]


def main():
    rng = random.Random(7); V1 = [rng.choice([3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]) * 1.0 + rng.random() for _ in range(40)]
    V2 = [rng.uniform(2, 50) for _ in range(40)]                                    # identical draws to extra_stats.main
    def fp(it):
        a, b = ES.fingerprint(it["skel"], V1), ES.fingerprint(it["skel"], V2)
        return None if a is None or b is None else (a, b, len(it["st"]))
    Q, E, A = AA.load_aug("train"); FA = set(fp(AA.analyse(q, e, a)) for q, e, a in zip(Q, E, A)); FA.discard(None)
    Qt, Et, At = AA.load_openai("gsm8k_test"); ft = [fp(AA.analyse(q, e, a)) for q, e, a in zip(Qt, Et, At)]; mt = [len(e) for e in Et]
    grp = ["nokey" if f is None else ("seen" if f in FA else "unseen") for f in ft]
    out = {"n": len(grp), "groups": dict(collections.Counter(grp)), "models": {}}
    for tag in TAGS:
        p = R + "gsm8k__" + tag + ".json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); it = d["items"]; assert len(it) == len(grp)
        c = [float(x["correct"]) for x in it]
        acc = {g: round(sum(ci for ci, gi in zip(c, grp) if gi == g) / max(1, grp.count(g)), 4) for g in ("seen", "unseen", "nokey")}
        by_m = {}; num = den = 0.0
        for m in sorted(set(mt)):
            s = [ci for ci, gi, mi in zip(c, grp, mt) if mi == m and gi == "seen"]; u = [ci for ci, gi, mi in zip(c, grp, mt) if mi == m and gi == "unseen"]
            by_m[m] = {"seen": [round(sum(s) / len(s), 3) if s else None, len(s)], "unseen": [round(sum(u) / len(u), 3) if u else None, len(u)]}
            if s and u:
                w = len(s) * len(u) / (len(s) + len(u)); num += w * (sum(s) / len(s) - sum(u) / len(u)); den += w
        out["models"][tag] = {"acc_all": d["accuracy"], "acc_by_group": acc, "mh_seen_minus_unseen": round(num / den, 4) if den else None, "by_m": by_m}
    json.dump(out, open(os.path.join(HERE, "seen_split.json"), "w", newline="\n"), indent=1)
    print(json.dumps({k: (v if k != "models" else {t: {kk: vv for kk, vv in x.items() if kk != "by_m"} for t, x in v.items()}) for k, v in out.items()}, indent=1))
    for t in ("dw_cot_ck25", "simcot_coconut_K10", "codi_L6"):
        if t in out["models"]:
            print(t, {m: (x["seen"], x["unseen"]) for m, x in out["models"][t]["by_m"].items()})


if __name__ == "__main__":
    main()
